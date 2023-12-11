import logging

# Math and Data
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer

# Utility
from functools import reduce
from tqdm import tqdm
import os.path
import collections
from typing import List, Optional, Tuple, Dict

# Own modules
from common import io
from info import IcdInfo
from data.merging_rules import mimic_merging, mimic_merging_simple


LoadedData = collections.namedtuple(
    "LoadedData",
    ["hadm_id", "static", "exams", "drugs"]
)

static_cols_prefix = 'static_info'


def positional_encoding_calculation(times_unnormalized: np.ndarray, pos_enc_dims: int) -> np.ndarray:
    positional_encoding = np.zeros(shape=(len(times_unnormalized), pos_enc_dims))
    funcs = {
        0: np.sin,
        1: np.cos
    }
    max_time = times_unnormalized[-1] / (0.75 * np.pi)  # Divide by 0.75 * pi so that the last dimension makes 37.5%
    # of a period in the training data time range
    for dim in range(pos_enc_dims):
        # Each dimension has its own frequency factor (here, we calculate the reciprocal of it)
        k = (dim - (dim % 2)) // 2
        omega = np.power(max_time, 2 * k / (pos_enc_dims - 2))  # (-2 because of the way I generate the k)

        # Use sine and cosine alternately (starting with sine)
        f = funcs[dim % 2]

        for t_idx in range(len(times_unnormalized)):
            positional_encoding[t_idx, dim] = f(times_unnormalized[t_idx] / omega)

    return positional_encoding


class Preprocessor:
    """
    Converts categorical values into one-hot, etc.
    """

    def __init__(self, iom, extractor_exams, extractor_drugs, encounter_ids, filtering_min_age=None,
                 include_drugs=False, min_exams_support=0.5, min_drugs_support=0.005, scaling_mode='quantile',
                 positional_encoding_dims=64, ignore_static_data=False, time_column='charttime'):
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)

        # IO: Creating and naming folders and files
        self.iom = iom

        # Trainer: This field is only later set by the trainer
        self.trainer = None

        # Extractors: The extractors download data from databases. There are two main sources of data: the exams
        # performed on the patient (measurement like heart rate but also lab results such as creatinine) and drugs
        # (i.e. medication administered to the patient)
        self.extractor_exams = extractor_exams
        self.extractor_drugs = extractor_drugs

        # Settings
        self.min_exams_support = float(min_exams_support)  # minimal "support" of exams data attributes, i.e.
        # the fraction of encounters we require to have an attribute for that attribute to be included in the data
        self.min_drugs_support = float(min_drugs_support)  # minimal support for drugs attributes (same role and
        # behavior as for exams)
        self.time_column = time_column  # column that denotes the time of the row
        self.meta_columns = {self.time_column}  # columns that don't really give us information but rather concern
        # _other_ information
        self._filter_min_age = filtering_min_age  # If not None, filter out admissions below the minimal age
        self.include_drugs = io.str_to_bool(include_drugs)  # allow including drugs (otherwise, they will not even be
        # extracted)
        self.ignore_static_data = io.str_to_bool(ignore_static_data)

        # Scaling mode: Allow scaling of data using different operations
        assert scaling_mode in ['minmax', 'robust', 'standard', 'power', 'quantile'],\
            f"Scaling mode '{scaling_mode}' unknown!"
        self._scaler_kwargs = {}
        if scaling_mode == 'minmax':
            self._scaler_class = MinMaxScaler
        elif scaling_mode == 'robust':
            self._scaler_class = RobustScaler
        elif scaling_mode == 'standard':
            self._scaler_class = StandardScaler
        elif scaling_mode == 'power':
            self._scaler_class = PowerTransformer
        elif scaling_mode == 'quantile':
            self._scaler_class = QuantileTransformer
            self._scaler_kwargs['output_distribution'] = 'normal'
        self.scaling_mode_name = scaling_mode.capitalize()

        # Baseline scaler (it also respects the scaling mode)
        self.baseline_scaler = None

        # Positional encoding: Either use the raw, normalized time or use a positional encoding inspired by
        # Vaswani et al. ("Attention is all you need")
        self.positional_encoding_dims = int(positional_encoding_dims)
        self.use_positional_encoding = self.positional_encoding_dims != 1  # (1 means using regular normalized time)
        if self.use_positional_encoding:
            assert self.positional_encoding_dims >= 4 and self.positional_encoding_dims % 2 == 0,\
                "Number of dimensions for positional encoding must be even and positive!"

        # List of encounter IDs (e.g. admission IDs for MIMIC)
        self.encounter_ids = encounter_ids
        self.encounter_ids_extracted = []  # List of encounter IDs we actually managed to extract

        # Cache for data: Used to dynamically compute and serve data
        self.max_dyn_val_observed = 0
        self.masking_value = None
        self._loaded_data = None
        self._static_data_arr = None
        self._static_data_categorical = None
        self._dyn_data_list = None
        self._dyn_charts_masks = []  # mask is True where NaN entries were in original data (before imputation)
        self._exams_high_sup = None
        self._drugs_high_sup = None
        self._static_data_categ_max_idxs = None
        self._static_data_categ_values = None
        self._static_data_categ_value_counts = None
        self._static_data_attrs_listlike = None
        self._static_data_attrs_non_listlike = None
        self._static_data_numerical_names = None
        self._columns_exams = None
        self._columns_drugs = None

        # Dynamic data columns used for all dynamic data
        self.dyn_data_columns = None
        self.dyn_col_medians_train = {}

        # Scaling information - can be used to reverse scaling
        self._scaler_by_col_name = {}

        # Relative support by column name
        self._rel_exams_support_stats = {}
        self._rel_drugs_support_stats = {}

        # Assign pretty labels to fused columns
        self._fused_col_labels = {}

        # Indices of training and validation data - set by training when available
        self.train_data_idxs = None

    def get_hadm_id(self, adm_idx):
        return self.get_raw_loaded_data()[adm_idx].hadm_id

    def get_raw_loaded_data(self) -> List[LoadedData]:
        if self._loaded_data is None:
            self._loaded_data = self._load_from_disk()

            # Save the list of actually extracted encounter IDs
            self.encounter_ids_extracted = [ld.hadm_id for ld in self._loaded_data]

        return self._loaded_data

    def label_for_any_item_id(self, item_id) -> str:
        if item_id in self._fused_col_labels:
            return self._fused_col_labels[item_id]
        elif item_id in self.meta_columns:
            return f"meta_{item_id}"
        else:
            return self.extractor_exams.label_for_item_id(item_id)

    def get_rel_support(self, item_id: str) -> float:
        # Find out if item ID belongs to exams or to drugs
        if item_id in self._columns_exams:
            return self._rel_exams_support_stats[item_id]
        elif item_id in self._columns_drugs:
            return self._rel_drugs_support_stats[item_id]
        else:
            assert False, f"Item ID '{item_id}' neither in exams nor in drugs!"

    def _load_from_disk(self):
        loaded = []
        for hadm_id in tqdm(self.encounter_ids, desc="Loading data from disk..."):
            # Load exams file
            exams = self.extractor_exams.load_dynamic_data(hadm_id)
            data_kinds_loaded = [exams]  # keep track of data we loaded

            # Load drugs file
            if self.include_drugs:
                drugs = self.extractor_drugs.load_dynamic_data(hadm_id)
                data_kinds_loaded.append(drugs)
            else:
                drugs = None  # Note: If excluding drugs, it is allowed to be None

            # Load static file
            stat_path = self.extractor_exams.path_for_static_data(hadm_id)
            if stat_path.exists():
                stat = pd.read_pickle(stat_path)
            else:
                stat = None
            if not self.ignore_static_data:
                data_kinds_loaded.append(stat)

            # Check if all could be loaded: If not, go to next admission
            if any([dk is None for dk in data_kinds_loaded]):
                continue

            # Apply cohort filtering
            if self._filter_min_age is not None:
                if stat['age_years'] < self._filter_min_age:
                    continue

            # Compose data
            data_loaded = LoadedData(hadm_id=hadm_id, static=stat, exams=exams, drugs=drugs)
            loaded.append(data_loaded)

        if len(loaded) < len(self.encounter_ids):
            logging.info(f"Out of {len(self.encounter_ids)} encounter IDs, only {len(loaded)} were loaded. "
                         f"The others had missing dynamic or static data or didn't fit the applied filters.")
        else:
            logging.info(f"All {len(loaded)} requested encounter IDs loaded.")

        return loaded

    def _analyze_possible_static_values(self):
        possible_vals = {}
        possible_vals_counts = {}
        for ld in tqdm(self.get_raw_loaded_data(), desc="Analyzing static data possible values..."):
            for idx in ld.static.keys():

                if idx not in possible_vals:
                    possible_vals[idx] = set()
                    possible_vals_counts[idx] = {}

                val = ld.static[idx]

                # Register that the value occurred
                if type(val) in [str, bool]:
                    possible_vals[idx].add(val)
                if type(val) == list:
                    possible_vals[idx] = possible_vals[idx].union(val)

                # Count the occurrences of the value
                if type(val) != list:
                    vals = [val]
                else:
                    vals = val
                for val in vals:
                    if val not in possible_vals_counts[idx]:
                        possible_vals_counts[idx][val] = 0
                    possible_vals_counts[idx][val] += 1

        # Convert sets to lists
        cleaned_possible_vals = {}
        for val_name in possible_vals.keys():
            possible_val_set = possible_vals[val_name]
            if len(possible_val_set) > 0:
                cleaned_possible_vals[val_name] = list(possible_val_set)
            # Discard empty sets

        return cleaned_possible_vals, possible_vals_counts

    def get_static_data_categ_values(self):
        if self._static_data_categ_values is None:
            self._static_data_categ_values, self._static_data_categ_value_counts = \
                self._analyze_possible_static_values()
        return self._static_data_categ_values

    def get_scaled_static_data_array(self):
        if self._static_data_arr is None or self._static_data_categorical is None:
            self._static_data_arr, self._static_data_categorical = self._compute_scaled_static_arr()
        return self._static_data_arr, self._static_data_categorical

    def _get_fresh_scaler(self):
        return self._scaler_class(**self._scaler_kwargs)

    def _compute_scaled_static_arr(self):
        # Get loaded data
        loaded = self.get_raw_loaded_data()

        # Get possible values for each attribute in the static data
        possible_vals = self.get_static_data_categ_values()

        # Save which categorical static attributes are list-like and which are not
        static_categ_listlike = []
        static_categ_non_listlike = []

        # Preprocess static data
        static_preprocessed = []
        static_categorical = []
        static_num_attr_names = []
        for ld in tqdm(loaded, desc="Pre-processing static data ..."):
            stat_numerical_pre = []
            stat_categorical_pre = {}
            for key in sorted(ld.static.keys()):

                # Don't handle future information - we constrain ourselves to information we would know when the
                # patient is admitted.
                if key.startswith("FUTURE"):
                    continue

                # Leave numbers unchanged and transform categorical data into a list of indices (standing for the
                # categorical values)
                if key in possible_vals:
                    # Categorical
                    present_val = ld.static[key]

                    # Get all indices of the present values (it will be a single index when the present value
                    # is not a list)
                    if type(present_val) is not list:
                        present_val = [present_val]

                        # Remember that this attribute is not list-like
                        if key not in static_categ_non_listlike:
                            static_categ_non_listlike.append(key)
                    else:
                        # Remember that this attribute is list-like
                        if key not in static_categ_listlike:
                            static_categ_listlike.append(key)

                    val_indices = []
                    for single_val in present_val:
                        if single_val is None:
                            continue  # Count a None element as not being there at all
                        val_index = possible_vals[key].index(single_val)
                        val_indices.append(val_index)

                    # Save indices
                    stat_categorical_pre[key] = val_indices
                else:
                    # Numerical
                    val = np.array([ld.static[key]])

                    # Remember order of numerical attributes
                    if key not in static_num_attr_names:
                        static_num_attr_names.append(key)

                    # Add value to static data list
                    stat_numerical_pre.append(val)

            # Concatenate all this admission's numerical static data
            stat_numerical = np.concatenate(stat_numerical_pre)
            static_preprocessed.append(stat_numerical)

            # Save this admission's categorical static data
            static_categorical.append(stat_categorical_pre)

        # Save the list-like and non-list-like attributes. Remove the categories that appear as both list-like and
        # non-list-like from the list of non-list-like categories.
        static_categ_non_listlike = [categ for categ in static_categ_non_listlike if categ not in static_categ_listlike]
        self._static_data_attrs_non_listlike = static_categ_non_listlike
        self._static_data_attrs_listlike = static_categ_listlike

        # Stack admissions to produce numerical static data array
        static_numerical_arr = np.stack(static_preprocessed, axis=0)
        # shape (num_admissions, len(static_num_attr_names))

        # Save names of static numerical attributes
        self._static_data_numerical_names = static_num_attr_names

        # Fit the scaler only on training data - Make sure we know what indices belong to training data
        assert self.train_data_idxs is not None, "self.train_data_idxs must be known (not None) to fit scaler!"

        logging.info(f"Scaling static data ({self.scaling_mode_name})...")
        static_scaler = self._get_fresh_scaler()
        static_scaler.fit(static_numerical_arr[self.train_data_idxs])  # fit on train
        static_numerical_arr_norm = static_scaler.transform(static_numerical_arr)  # transform all
        logging.info(f"Scaling static data ({self.scaling_mode_name}) done.")

        return static_numerical_arr_norm, static_categorical

    def get_dyn_charts(self):
        """
        Dyn data with scaling, which is fitted on train dynamic data and then applied to both validation and
        train data. This function uses as input the dynamic data with unsupported columns
         removed (self.get_supported_dyn_data_list).
        :return:
        """
        if self._dyn_data_list is None:
            self._dyn_data_list = self._compute_scaled_dyn_list()
        return self._dyn_data_list

    def get_supported_dyn_data_list(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Dyn data without scaling but with columns with little support removed. This step does not require a
        train/val split.
        :return: exams, drugs
        """

        # Check if already extracted
        extracted_already = self._exams_high_sup is not None
        if self.include_drugs:
            extracted_already = extracted_already and self._drugs_high_sup is not None

        if not extracted_already:
            self._exams_high_sup, self._drugs_high_sup = self._filter_out_low_support_from_dyn()
        return self._exams_high_sup, self._drugs_high_sup

    def _merge_columns(self, df, columns_fused, label_fused, label_split_idx=None, label_split_count=None):
        # Function for checking if all elements of a list are equal to one another (for arrays)
        def all_equal(lis):
            if len(lis) < 2:
                return True
            else:
                it_first = lis[0]
                equalities = [len(it) == len(it_first) and np.equal(it, it_first) for it in lis]
                for e_idx, e in enumerate(equalities):
                    if type(e) != bool:
                        e = all(e)  # Convert array comparison to bool
                    equalities[e_idx] = e
                return all(equalities)

        # Don't fuse when none of the columns occur in the dataframe
        present_cols = [c for c in columns_fused if c in df]
        if len(present_cols) == 0:
            return df

        if len(present_cols) == 1:
            # If only a single column is present, "fuse" by essentially renaming
            new_col_content = df[present_cols[0]]
        else:
            # Check if measurements line up in time
            nan_masks = [df[p_col].isna() for p_col in present_cols]
            nan_equal = all_equal(nan_masks)

            # Determine if the columns have roughly the same number of values
            measurements = [df[p_col].dropna() for p_col in present_cols]
            measured_vals = [dfc.values for dfc in measurements]
            num_measurements = [len(m) for m in measured_vals]

            # Check if measurements overlap at all - for some attributes (like heart rate), it can be the case that
            # the first temporal half of the data resides in one column and the second half in the other column.
            val_indices = [mask[~mask].index for mask in nan_masks]
            val_index_bounds = [np.arange(idx.min(), idx.max() + 1) for idx in val_indices]
            intersection_bounds = reduce(np.intersect1d, val_index_bounds)  # intersection of *bounds*

            ranges_small_overlap = len(intersection_bounds) < 0.1 * max(num_measurements)

            # If measurements all have the same times, fuse by taking mean at each time
            if nan_equal or ranges_small_overlap:
                new_col_content = df[present_cols].mean(axis=1)  # create new column
            else:
                # Find out which column delivers data earlier
                mean_indices = [m.index.to_numpy().mean() for m in measurements]
                earlier_col = present_cols[np.argmin(mean_indices)]

                if all_equal(measured_vals):
                    # Measurements are equal, but times are different -> Take the column that delivers measurements
                    # earlier
                    chosen_col = earlier_col

                    # Save values of chosen column under new name
                    new_col_content = df[chosen_col]
                else:
                    # Count how many measurements there are per column
                    more_measure_col = present_cols[np.argmax(num_measurements)]

                    # Use the column with the most measurements - it must be the better-quality data source
                    # (Note: This might not be the "earlier" column)
                    chosen_col = more_measure_col

                    # Save values of chosen column under new name
                    new_col_content = df[chosen_col]

        # Store new column content
        assert new_col_content is not None, f"Problem fusing {label_fused}!"
        new_col_name = "_".join([label_fused] + columns_fused)
        if label_split_idx is not None:
            new_col_name += f"_{label_split_idx}"
        df[new_col_name] = new_col_content

        # Remember pretty label for plotting and evaluation
        pretty_label = label_fused
        if label_split_idx is not None:
            pretty_label += f" ({label_split_idx + 1}/{label_split_count})"
        self._fused_col_labels[new_col_name] = pretty_label

        # Drop original columns - they are no longer needed
        df.drop(present_cols, axis=1, inplace=True)

        return df

    def _filter_dataframe_cols_by_support(self, dfs: List[pd.DataFrame], min_col_support: float, logging_name: str) \
            -> Tuple[List[pd.DataFrame], Dict[str, float]]:
        """
        Removes columns with low support from a list of DataFrames
        :param dfs:
        :param min_col_support: minimal support. Columns removed if LOWER than minimal support.
        :param logging_name: Name to give this list of DataFrames in logging
        :return: Tuple[processed DataFrames, Dict from col names to relative support]
        """

        # Abort early if we have no data
        if len(dfs) == 0:
            # This happens when no raw dynamic data is available for the selected admissions
            logging.warning("No dynamic data is available!")
            return [], {}

        # Count support
        def rel_sup(dataframes):
            column_support = {}
            num_non_empty_adms = 0
            for df_sup in dataframes:

                # Don't count empty dataframes toward support - their columns are not structured like those of
                # non-empty dataframes
                if len(df_sup) == 0:
                    continue
                num_non_empty_adms += 1

                for col_name in df_sup.columns:

                    # Count the occurrence of this column towards its support
                    if col_name not in column_support:
                        column_support[col_name] = 0
                    column_support[col_name] += 1

            # Convert to relative support
            for col_name in column_support.keys():
                column_support[col_name] = float(column_support[col_name]) / num_non_empty_adms

            return column_support

        # Fuse columns with the same label
        cols_by_labels = collections.defaultdict(list)
        for label, col in [(self.label_for_any_item_id(k), k) for k in rel_sup(dfs).keys()]:
            cols_by_labels[label].append(col)
        cols_by_labels = {lab: cols for (lab, cols) in cols_by_labels.items() if len(cols) > 1}
        for collision_idx, (label, cols) in enumerate(cols_by_labels.items()):

            # Fuse the columns
            origins = {c: self.extractor_exams.item_meta_info[c] for c in cols}
            # Columns can have duplicate labels for a variety of reasons, which necessitates a variety of fusing
            # strategies. In turn, we try to apply each of the strategies if they are applicable.

            # Look up fusion rules - some labels require special handling
            merge_opts = {}
            strategy_used = 'untested'
            if label in mimic_merging:
                merge_opts = mimic_merging[label]['opts']
                strategy_used = mimic_merging[label]['strategy']
            elif label in mimic_merging_simple:
                strategy_used = 'simple'
            else:
                assert False, f"No merging strategy for label {label} with columns {cols} defined!"

            logging.info(f"[{logging_name}] Fusing collision {collision_idx + 1} of {len(cols_by_labels)}:"
                         f" {len(origins)} columns for label '{label}' (strategy: {strategy_used}) ... ")

            # Fuse
            if 'split' in merge_opts:
                split = merge_opts['split']

                # If split is empty, we split every collided column into its own attribute
                if len(split) == 0:
                    split = [[c] for c in cols]

                assert set(cols).issubset(set(sum(split, []))), \
                    f"Split for {label} incomplete/incorrect! ({split} vs. {cols})"
                for split_idx, col_subset in enumerate(split):
                    dfs = [self._merge_columns(df, col_subset, label, label_split_idx=split_idx,
                                               label_split_count=len(split))
                           for df in dfs]
            else:
                dfs = [self._merge_columns(df, cols, label) for df in dfs]

        # Remove attributes (columns) with little support, i.e. those that are present for few of the patients
        col_support = rel_sup(dfs)
        all_columns = []
        removed_attributes = []
        logging.info(f"[{logging_name}] Identifying attributes with too little support...")
        for col, rel_sup in col_support.items():

            # Skip if already removed
            if col in removed_attributes:
                continue

            if rel_sup < min_col_support:
                removed_attributes.append(col)
            else:
                # Add this column: Its support is high enough
                all_columns.append(col)
        logging.info(f"[{logging_name}] Removed {len(removed_attributes)} of {len(col_support)} attributes "
                     f"({len(removed_attributes) / len(col_support) * 100:0.0f}%) because their support was "
                     f"lower than the minimal support of {min_col_support * 100:0.4f}%. "
                     f"Keeping {len(all_columns)} {logging_name} attributes.")

        # Write complete support statistics to disk for later analysis
        sup_complete = [(self.label_for_any_item_id(c), s) if c != self.time_column else (self.time_column, s)
                        for (c, s) in sorted(
                [(c, s) for (c, s) in col_support.items()],
                key=lambda t: t[1],
                reverse=True
            )]
        io.write_json(sup_complete, os.path.join(self.iom.get_plots_dir(), "input_data_col_sup.json"), pretty=True)

        # Save support
        rel_support_by_col_name = {col_name: col_support[col_name] for col_name in all_columns}

        # Detect meaningless columns: A column can be said to be meaningless if it has the same value for every
        # patient, since in this case, it can never add any discriminating information.
        meaningless_columns = []
        for col in all_columns:
            meaningful = False
            different_observed_values = []  # keep a running list of values we have seen

            dyns_with_col = [d for d in dfs if col in d]
            for df in dyns_with_col:

                # Abort if column is already proven to be meaningful
                if meaningful:
                    break

                # Only regard values that are not NaN
                vals = df[col][df[col].notnull()].values

                for val in vals:
                    if val not in different_observed_values:
                        different_observed_values.append(val)

                    # Check if the column can be said to be meaningless
                    if len(different_observed_values) > 1:
                        # This column is NOT meaningless!
                        meaningful = True
                        break

            # Continue with next column if this one is meaningful
            if meaningful:
                continue

            # The column has been proven to be meaningless.
            meaningless_columns.append(col)

        logging.info(f"[{logging_name}] Removed {len(meaningless_columns)} meaningless columns that have the same "
                     f"value for all admissions. Keeping {len(all_columns) - len(meaningless_columns)} columns.")

        # Drop removed columns from all tables
        for df in tqdm(dfs, desc=f"[{logging_name}] Dropping removed attributes from tables..."):
            df.drop(
                removed_attributes + meaningless_columns,
                axis=1,
                errors='ignore',
                inplace=True
            )

        return dfs, rel_support_by_col_name

    def _filter_out_low_support_from_dyn(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        # Handle exams data (examinations done for patients, e.g. heart rate measurement)
        exams_high_sup, self._rel_exams_support_stats = self._filter_dataframe_cols_by_support(
            dfs=[ld.exams for ld in self.get_raw_loaded_data()],
            min_col_support=self.min_exams_support,
            logging_name="Exams"
        )

        # Handle drugs data (medications administered to patients, e.g. pain medication)
        if self.include_drugs:
            drugs_high_sup, self._rel_drugs_support_stats = self._filter_dataframe_cols_by_support(
                dfs=[ld.drugs for ld in self.get_raw_loaded_data()],
                min_col_support=self.min_drugs_support,
                logging_name="Drugs"
            )
        else:
            drugs_high_sup = None

        # Remember which of the columns are exams and which are drugs
        self._columns_exams = self._unique_nonempty_columns(exams_high_sup)
        if self.include_drugs:
            self._columns_drugs = self._unique_nonempty_columns(drugs_high_sup)

        return exams_high_sup, drugs_high_sup

    @staticmethod
    def _unique_nonempty_columns(dfs: List[pd.DataFrame]) -> List:
        return sorted(np.unique(np.concatenate([df.columns.values for df in dfs if len(df) > 0])))

    def _fuse_and_scale_dataframes(self, exams_dfs: List[pd.DataFrame],
                                   drugs_dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:

        def dataframes_to_array(dfs: List[pd.DataFrame], col_name: str) -> np.ndarray:
            """
            Concatenates rows from multiple dataframes into a single numpy array
            :param dfs:
            :param col_name:
            :return:
            """
            col_data_arrays = []
            for data_frame in dfs:
                values = data_frame[col_name].values
                col_data_arrays.append(values)
            columns_data = np.concatenate(col_data_arrays).astype(dtype=float).transpose()
            return columns_data

        def combine_tables(u: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
            """
            Combines two DataFrames into one using time column as the index
            (assumes that both tables have the same columns)

            :param u:
            :param v:
            :return:
            """

            # If one of the dataframes is empty, return the other one
            if len(u) == 0:
                return v
            if len(v) == 0:
                return u

            # Index both using time column
            u = u.set_index(self.time_column)
            v = v.set_index(self.time_column)

            # Fuse tables
            f = u.combine_first(v)
            f = f.reset_index()  # Remove time column from index again and go back to regular int index

            # Sort columns
            f = f[sorted(f.columns)]

            return f

        # Make sure that both types of facts have the same number of dataframes
        if self.include_drugs:
            assert len(exams_dfs) == len(drugs_dfs), f"There must be the same number of " \
                                                     f"exams dataframes ({len(exams_dfs)}) as " \
                                                     f"drugs dataframes ({len(drugs_dfs)})!"
        else:
            # If drugs are excluded, drugs_dfs is None. Replace this with an empty list since it is easier to work with
            drugs_dfs = []

        # List all columns that exist in the dynamic data
        dyn_data_cols_implied_by_data = self._unique_nonempty_columns(exams_dfs + drugs_dfs)
        loaded_dyn_data_info = io.read_json(self.iom.get_dyn_data_cols_path())
        if loaded_dyn_data_info is None:
            self.dyn_data_columns = dyn_data_cols_implied_by_data
        else:
            self.dyn_data_columns = loaded_dyn_data_info['cols']

        # Create an artificial DataFrame that has all the columns (but no data) and concatenate each table with it.
        # That way, the tables will all have the same columns
        all_col_df = pd.DataFrame([], columns=self.dyn_data_columns)
        total_length = 0
        logging.info(f"Instating same columns for all dynamic data tables...")
        for facts in [exams_dfs, drugs_dfs]:
            for dyn_idx, df in enumerate(facts):

                # Remove all columns if dataframe empty: Empty dataframes have additional columns like 'itemid' that
                # would get in the way otherwise
                if len(df) == 0:
                    df = df.drop(columns=df.columns)

                # Drop all columns but the ones that are shared between all tables
                df = df.filter(self.dyn_data_columns)

                df_full = pd.concat([all_col_df, df], sort=False)
                facts[dyn_idx] = df_full
                total_length += len(df)

        logging.info("Done!")

        # Abort early if we have no data
        if total_length == 0:
            # This happens when there is no data to concatenate. This may be the case when the selected admissions all
            # have no data.
            logging.warning(f"No dynamic data is available!")
            return []

        # Fuse pairs of tables from exams and drugs into single tables each (by adding the drugs information to the
        # exams table)
        if self.include_drugs:
            fused = [combine_tables(e_df, d_df) for e_df, d_df in zip(exams_dfs, drugs_dfs)]
        else:
            fused = list(exams_dfs)
        del exams_dfs  # Free up memory
        del drugs_dfs

        # For fitting, we need to know which dynamic data belongs to train patients and which to val patients
        assert self.train_data_idxs is not None, "self.train_data_idxs must be known (not None)" \
                                                 " to fit dynamic data (for scaling)!"

        # Add static data columns
        static_columns = set()
        for static_data, df in tqdm(
                zip([ld.static for ld in self._loaded_data], fused),
                total=len(fused),
                desc="Adding static data to dyn charts..."
        ):
            # Skip if ignoring static data
            if self.ignore_static_data:
                continue

            for static_name, static_val in static_data.items():

                # Don't add future information
                if 'future' in static_name.lower() or static_name in self._static_data_attrs_listlike:
                    continue

                # Handle categorical values by using the index of the value
                if static_name in self._static_data_categ_values:
                    possible_values = self._static_data_categ_values[static_name]
                    static_val = possible_values.index(static_val)

                # Save static value in data frame
                static_col_name = f"{static_cols_prefix}_{static_name}"
                df[static_col_name] = [static_val] * len(df)

                # Add this column to the meta columns, so it will be ignored by plotting code built for dynamic data
                self.meta_columns.add(static_col_name)
                static_columns.add(static_col_name)

        # Add static columns to dynamic data column list (if they are not already present)
        for stat_col in static_columns:
            if stat_col not in self.dyn_data_columns:
                self.dyn_data_columns.append(stat_col)

        # Temporally impute dynamic data charts
        for df_idx in range(len(fused)):
            # Save a mask of original, raw values - this is later used in evaluation, where imputed entries are then
            # ignored
            raw_values_mask = fused[df_idx].isna()  # True where NaNs are, False for original data
            self._dyn_charts_masks.append(raw_values_mask)

            # Interpolate linearly between known values
            df = fused[df_idx].set_index(self.time_column, drop=False)  # keep time as column as well
            df = df.interpolate(method='index')
            df = df.reset_index(drop=True)  # remove time column from index again

            # Save chart back to list of charts
            fused[df_idx] = df

        # Analyze each column (i.e. attribute type) independently
        dyn_col_medians = {}
        dyns_train = [fused[idx] for idx in self.train_data_idxs]
        for col_idx, column_name in enumerate(self.dyn_data_columns):
            logging.info(f"Dyn Scaling ({self.scaling_mode_name}): Fitting column {col_idx + 1} "
                         f"of {len(self.dyn_data_columns)} ('{column_name}') on training data\n"
                         f"(-> '{self.label_for_any_item_id(column_name)}')")

            # Concatenate all the training data of this column
            col_data = dataframes_to_array(
                dfs=dyns_train,
                col_name=column_name
            )

            # Scale the data
            scaler = self._get_fresh_scaler()
            scaler.fit(col_data.reshape(-1, 1))

            # Remember scaler (this makes reversal of scaling possible)
            assert column_name not in self._scaler_by_col_name, \
                f"Col name {column_name} is already saved in self._scaler_by_col_name!"
            self._scaler_by_col_name[column_name] = scaler

            # Remember median of data
            col_median = np.nanmedian(col_data)
            assert not np.isnan(col_median), "Median for dynamic data column is NaN! (This indicates a problem with" \
                                             " the input data)"
            dyn_col_medians[column_name] = col_median

        # Fill remaining NaNs with median of dynamic attribute (median measured in train split)
        for df in fused:
            for column_name in self.dyn_data_columns:
                df[column_name] = df[column_name].fillna(value=dyn_col_medians[column_name])

        # Save train medians
        self.dyn_col_medians_train = dyn_col_medians

        # Apply scaling to both train and validation dynamic data
        for column_name in tqdm(self.dyn_data_columns, desc=f"Applying dyn scaling ({self.scaling_mode_name})..."):
            # Concatenate data for this column over all dynamic data
            # (this makes scaling faster since scaling itself can happen in one big operation)
            col_data = dataframes_to_array(
                dfs=fused,
                col_name=column_name
            )

            # Apply scaling
            col_data = self.perform_scaling(col_data, column_name)

            # Make sure data does not contain illegal values: Abort if there are invalid values (like Inf or NaN)
            if not np.all(np.isfinite(col_data)):
                illegal_indices = np.where(~np.isfinite(col_data))[0]
                col_label = self.label_for_any_item_id(column_name)
                assert False, f"Column {column_name} ({col_label}) contains invalid entries after normalization!" \
                              f"({len(illegal_indices)} entries\n" \
                              f"at {illegal_indices}:\n" \
                              f"{col_data[illegal_indices]})"

            # Find maximum value
            self.max_dyn_val_observed = max(self.max_dyn_val_observed, np.nanmax(col_data))

            # Put the data back into the original tables
            col_data_offset = 0
            for dyn_df in fused:
                dyn_len = len(dyn_df)
                dyn_df[column_name] = col_data[col_data_offset:col_data_offset + dyn_len]
                col_data_offset += dyn_len

        # Put time column at the end - this fact is used in the model's architecture
        cols = list(fused[0].columns)
        cols.remove(self.time_column)
        cols.append(self.time_column)
        fused = [df[cols] for df in fused]
        self.dyn_data_columns = cols

        # Set masking value used for batch generation during training
        self.masking_value = np.round(self.max_dyn_val_observed + 1)

        logging.info(f"Dyn Scaling ({self.scaling_mode_name}) done!")

        return fused

    def _compute_scaled_dyn_list(self) -> List[pd.DataFrame]:
        # Get dynamic data (without columns that don't have strong support)
        exams, drugs = self.get_supported_dyn_data_list()

        # Scale exams and drugs together
        dyn_dfs = self._fuse_and_scale_dataframes(
            exams_dfs=exams,
            drugs_dfs=drugs
        )

        # Compute positional encoding
        if self.use_positional_encoding:
            self._precompute_positional_encoding(dyn_dfs)

            # Add positional encoding to each of the dynamic data charts
            for dyn_chart in tqdm(dyn_dfs, desc="Updating charts with positional encoding ..."):
                self._append_positional_encoding(dyn_chart)

        return dyn_dfs

    def _append_positional_encoding(self, dyn_chart: pd.DataFrame):
        # Gather positional encoding for every time point
        times = dyn_chart[self.time_column].values
        if len(times) == 0:
            return
        pos_enc = np.stack([self._pos_encoding_matrix[self._unique_times_normalized.index(t)] for t in times])
        # shape: (steps, self.positional_encoding_dims)

        # Add each dimension to the data frame
        for dim_idx in range(self.positional_encoding_dims):
            dim_name = f"pos_enc_{dim_idx}"
            dyn_chart[dim_name] = pos_enc[:, dim_idx]
            self.meta_columns.add(dim_name)

        return

    def _precompute_positional_encoding(self, dyn_dfs: List[pd.DataFrame]) -> None:
        """
        Compute the positional encodings for every observed time
        :param dyn_dfs:
        :return:
        """

        # This function should never be called if positional encodings are not in use (it then serves no purpose)
        if not self.use_positional_encoding:
            return

        logging.info(f"Computing positional encoding with {self.positional_encoding_dims} dimensions ...")

        # Get a list of unique times observed in the data
        times_normalized = np.array([])
        for df in dyn_dfs:
            times_normalized = np.unique(np.concatenate([df[self.time_column], times_normalized]))
        times_normalized.sort()  # Sort times in ascending order

        # Reverse normalization of the time values
        times = self.reverse_scaling(times_normalized, column_name=self.time_column)

        # For each time, calculate the positional encoding (this closely follows the formula given by Vaswani et al.)
        positional_encoding = positional_encoding_calculation(
            times_unnormalized=times,
            pos_enc_dims=self.positional_encoding_dims
        )

        # Save the encoding in array form along with the sorted normalized times. Retrieving the positional encoding
        # then becomes a task of finding the normalized time index in the sorted list and getting the corresponding row
        # of the matrix.
        self._pos_encoding_matrix = positional_encoding
        self._unique_times_normalized = list(times_normalized)

        logging.info(f"Pre-computing positional encoding done!")

    def perform_scaling(self, unscaled_value, column_name) -> np.ndarray:
        if type(unscaled_value) == np.ndarray:
            assert len(unscaled_value.shape) == 1
            unscaled_value = unscaled_value.reshape(-1, 1)  # reshape into (n_samples, n_features)
        else:
            assert False, f"unscaled_value has type {type(unscaled_value)}, which is not supported!"

        # Scale
        scaler = self._scaler_by_col_name[column_name]
        scaled_val = scaler.transform(unscaled_value)

        # Squeeze back into 1d shape
        scaled_val = np.squeeze(scaled_val)

        return scaled_val

    def reverse_scaling(self, scaled_value, column_name) -> np.ndarray:
        # If there is no data, we are done
        if len(scaled_value) == 0:
            return np.array([])

        # Convert to array
        if type(scaled_value) in [list, tuple]:
            scaled_value = np.array(scaled_value)
        original_shape = scaled_value.shape

        # Reshape
        if type(scaled_value) == np.ndarray:
            assert len(scaled_value.shape) == 1
            scaled_value = scaled_value.reshape(-1, 1)  # reshape into (n_samples, n_features)
        else:
            assert False, f"scaled_value has type {type(scaled_value)}, which is not supported!"

        # Unscale
        scaler = self._scaler_by_col_name[column_name]
        original_val = scaler.inverse_transform(scaled_value)

        # Reshape back into original shape
        original_val = np.reshape(original_val, original_shape)

        return original_val

    def get_static_attrs_listlike(self):
        if self._static_data_attrs_listlike is None:
            self._compute_static_categ_max_idxs()
        return self._static_data_attrs_listlike

    def get_static_attrs_non_listlike(self):
        if self._static_data_attrs_non_listlike is None:
            self._compute_static_categ_max_idxs()
        return self._static_data_attrs_non_listlike

    def get_static_categ_max_idxs(self):
        if self._static_data_categ_max_idxs is None:
            self._static_data_categ_max_idxs = self._compute_static_categ_max_idxs()
        return self._static_data_categ_max_idxs

    def _compute_static_categ_max_idxs(self):
        # Get static data
        static_arr, static_categorical = self.get_scaled_static_data_array()

        # Find out the maximum index for each categorical static attribute
        logging.info("Finding out the maximal index for each static categorical data attribute...")
        max_categ_idxs = {}
        for static_categ_info in static_categorical:
            for categ_name, indices in static_categ_info.items():

                if categ_name not in max_categ_idxs:
                    max_categ_idxs[categ_name] = 0

                if len(indices) == 0:
                    continue

                max_categ_idxs[categ_name] = max(
                    max_categ_idxs[categ_name],
                    max(indices)
                )
        logging.info("Done.")

        return max_categ_idxs

    def dyn_medical_data_fit_lines(self, adm_indices):
        """
        Medical dynamic data, but expressed as a broad trend: angle of a line fitted through the time series'
        :param adm_indices:
        :return:
        """

        # Define function for line fitting: This is an extreme smoothing of the time series. As opposed to the step-wise
        # deltas, this function captures the general direction of movement in a time series
        def fit_line(values: np.ndarray, time_points: np.ndarray) -> List:
            # If there are less than two values, we can not fit a line
            if len(values) < 2:
                return []

            # Fit line
            try:
                if len(np.unique(values)) == 1:
                    # Line fitting has a bug where a line that should be perfectly straight (if all y values are the
                    # same) instead has an angle of 45 degrees.
                    angle = 0
                else:
                    fitted_poly = np.polynomial.polynomial.Polynomial.fit(x=time_points, y=values, deg=1)
                    line_slope = fitted_poly.convert().coef[-1]

                    # Convert slope to degrees
                    # tan(alpha) = opposing cathetus / adjacent cathetus
                    angle = float(np.degrees(np.arctan(line_slope / 1)))

                return [angle]  # since this angle describes a time series, return single value as list so other
                # functions expecting lists (like statistical tests or plotting functions) work
            except Exception as e:
                # In any kind of error (e.g. if line fit does not converge), don't assume any angle at all for this data
                return []

        # Process each of the dynamic chart columns
        charts_deimputed = {}  # cache de-imputed charts
        dyn_trend_info = {}
        for item_id in self.dyn_data_columns:

            # Skip over meta columns like time
            if item_id in self.meta_columns:
                continue

            # Create a new entry for this column
            item_key = self.key_for_item_id(item_id)
            if item_key not in dyn_trend_info:
                dyn_trend_info[item_key] = []

            # Process every admission
            for adm_idx in adm_indices:
                if adm_idx not in charts_deimputed:
                    charts_deimputed[adm_idx] = self.get_deimputed_dyn_chart(adm_idx)
                chart = charts_deimputed[adm_idx]
                vals = chart[item_id]

                # Skip this admission if all values are NaN
                if vals.isnull().all():
                    dyn_trend_info[item_key].append([])
                    continue

                # Remove NaN values
                vals_idcs_valid = vals.loc[vals.notnull()].index
                vals = vals[vals_idcs_valid].values
                times = chart[self.time_column][vals_idcs_valid].values

                # Reverse scaling for values and times
                vals = self.reverse_scaling(vals, column_name=item_id)
                times = self.reverse_scaling(times, column_name=self.time_column)

                # Fit line
                dyn_trend_info[item_key].append(fit_line(values=vals, time_points=times))

        return dyn_trend_info

    @staticmethod
    def dyn_medical_data_temporal_delta(dyn_info, dyn_times):
        """
        Retrieves medical dynamic data, but expressed in terms of deltas from one point in time to the next
        :param dyn_info:
        :param dyn_times:
        :return:
        """

        # Define function for getting deltas from t to t+1
        def calculate_stepwise_deltas(vals: List, times: List) -> List:
            # If there are less than two values, there can be no deltas between the values
            if len(vals) < 2:
                return []

            # Calculate difference between each pair of values: Positive difference means the value increased from
            # t to t+1
            diffs = []
            for v_1, v_2, t_1, t_2 in zip(vals[:-1], vals[1:], times[:-1], times[1:]):
                d = (v_2 - v_1) / (t_2 - t_1)
                diffs.append(d)

            return diffs

        # Derive temporal deltas (between adjacent time points)
        dyn_temporal_info = {attr_name: [calculate_stepwise_deltas(adm_vals, adm_times)
                                         for adm_vals, adm_times in zip(dyn_info[attr_name], dyn_times[attr_name])]
                             for attr_name in dyn_info.keys()}

        return dyn_temporal_info

    def get_deimputed_dyn_chart(self, adm_idx):
        # Copy chart
        chart = self.get_dyn_charts()[adm_idx].copy(deep=True)

        # Set imputed entries to NaN
        nan_mask = self._dyn_charts_masks[adm_idx]
        assert len(chart) == len(nan_mask), f"NaN mask for admission {adm_idx} has shape {chart.shape} but chart " \
                                            f"itself has shape {nan_mask.shape}!"
        chart[nan_mask] = pd.NA

        return chart

    def get_dyn_medical_data(self, adm_indices, with_times=False):
        """
        Retrieve dynamic medical information for the selected admission indices.
        Values are-scaled.

        :param adm_indices:
        :param with_times: if True, also return times
        :return: dict with key = column name (item id), value = list of reduced lists (i.e. list of values)
        """
        dyn_info = {}
        dyn_times = {}
        for adm_idx in adm_indices:
            # Load scaled dynamic data: Set imputed entries to NaN
            chart = self.get_deimputed_dyn_chart(adm_idx)

            # If requested, get times and un-scale them
            if with_times:
                adm_dyn_times = chart[self.time_column].dropna().values
                adm_dyn_times = self.reverse_scaling(adm_dyn_times, column_name=self.time_column)

            # Process each of the dynamic data attributes
            for col_name in self.dyn_data_columns:

                # Ignore columns that are also ignored by the model
                if col_name in self.meta_columns:
                    continue

                # Make a new dict entry for this column
                dyn_key = self.key_for_item_id(item_id=col_name)
                if dyn_key not in dyn_info:
                    dyn_info[dyn_key] = []
                    dyn_times[dyn_key] = []

                # Get values
                dyn_vals = chart[col_name].values

                # If requested, match values to times and save times
                if with_times:
                    attr_times = []
                    dyn_vals_non_nan = []
                    for t, v in zip(adm_dyn_times, dyn_vals):
                        if not np.isnan(v):
                            attr_times.append(t)
                            dyn_vals_non_nan.append(v)
                    dyn_vals = dyn_vals_non_nan

                    dyn_times[dyn_key].append(attr_times)
                else:
                    dyn_vals = dyn_vals[~np.isnan(dyn_vals)]

                # Un-scale values and add them to the list
                dyn_vals = self.reverse_scaling(dyn_vals, column_name=col_name)
                dyn_info[dyn_key].append(dyn_vals)

        if not with_times:
            return dyn_info
        else:
            return dyn_times, dyn_info

    def key_for_item_id(self, item_id):
        label = self.label_for_any_item_id(item_id)
        return f"{label}_{item_id}"

    @staticmethod
    def split_days_in_care_by_survival(static_info: Dict[str, List]) -> None:
        """
        Split the attribute for days in care into two attributes based on the survival of the admissions. These two new
        attributes have None values since according to the structure of static_info, all admissions need to be present,
        but e.g. the "days_in_care_survivors" attribute does not have information for the deceased.
        :param static_info:
        :return:
        """
        # Retrieve data we need
        survival = static_info['FUTURE_survival']

        for dic_attr_name in ['days_in_care', 'days_in_care_hospital']:
            days_in_care = static_info[f'FUTURE_{dic_attr_name}']
            assert len(days_in_care) == len(survival)

            # Prepare new attributes
            days_survivors = []
            days_deceased = []
            for days_spent, has_survived in zip(days_in_care, survival):
                if has_survived:
                    days_survivors.append(days_spent)
                    days_deceased.append(None)
                else:
                    days_survivors.append(None)
                    days_deceased.append(days_spent)

            # Save attributes
            static_info[f'FUTURE_{dic_attr_name}_survivors'] = days_survivors
            static_info[f'FUTURE_{dic_attr_name}_deceased'] = days_deceased

    def extract_static_medical_data(self, adm_indices) -> Dict[str, List]:
        """
        Extract static data
        :param adm_indices:
        :return: dict with key = static value name, value = list of values
        """
        static_info = {}  # key: static data name, value: list of encountered values
        static_list_types = []  # contains names of static attributes that are lists (like ICD codes)
        for adm_idx in adm_indices:
            # Load data
            stat = self.get_raw_loaded_data()[adm_idx].static

            # Process static data
            for stat_name, stat_val in stat.items():

                # Skip this entry if the value is a list: We can't plot a situation where each admission has a list
                # of values for an attribute
                if type(stat_val) == list or stat_name in static_list_types:
                    static_list_types.append(stat_name)
                    continue

                # Make a new entry in the static info list
                if stat_name not in static_info:
                    static_info[stat_name] = []

                # Save this entry
                static_info[stat_name].append(stat_val)

        # Remove list-typed static info from the dictionary
        static_info = {stat_name: values for (stat_name, values) in static_info.items()
                       if stat_name not in static_list_types}
        return static_info

    def prepare_input_data_for_matrix(self, static_info: Dict[str, List], dyn_info: Dict[str, List[List]],
                                      dyn_trend_info: Optional[Dict[str, List[List]]], split: str,
                                      aggregate_icds=True,
                                      agg_funcs: Optional[dict] = None,
                                      overwrite_agg_funcs=False,
                                      time_series_partitions=1,
                                      with_future_info=False):
        """
        Prepares input data in columns (and names) to be used in matrix-like methods such as decision trees or
        the baseline method.
        
        :param static_info: 
        :param dyn_info: 
        :param dyn_trend_info: 
        :param split:
        :param aggregate_icds: If True, aggregate individual ICD codes to the top level in ICD hierarchy (categories)
        :param agg_funcs: If not None, can be a dictionary like {"STD": np.std} of aggregation functions to use 
        for dynamic data aggregation
        :param overwrite_agg_funcs: If True, overwrite aggregation functions with those supplied, otherwise, add the
        supplied ones to the standard aggregation functions
        :param time_series_partitions: Number of partitions to split time series into. Default is 1, which means that
        aggregation will be performed over the whole time series. For e.g. 3, time series will be split into 3
        equally sized parts and aggregation performed over those parts individually
        :param with_future_info: If True, include "future" information such as the patient's survival
        :return: 
        """

        # Dynamic data: Take min, max and mean of the time series of individual patients
        aggregation_funcs = {
            "MAX": np.max,
            "MIN": np.min,
            "MEAN": np.mean
        }
        if agg_funcs is not None:
            if not overwrite_agg_funcs:
                aggregation_funcs.update(agg_funcs)
            else:
                aggregation_funcs = agg_funcs
        dyn_stats_cols = []
        dyn_stats_names = []
        for dyn_attr_name, dyn_data in dyn_info.items():
            for func_name, func in aggregation_funcs.items():

                # Split time series into multiple partitions
                for part_idx in range(time_series_partitions):
                    dyn_stats_names.append(f"{func_name}({dyn_attr_name})({part_idx + 1}/{time_series_partitions})")

                    feat_col = []
                    for time_series_whole in dyn_data:

                        # Split time series into parts
                        if time_series_partitions > 1:
                            time_series = np.array_split(time_series_whole, time_series_partitions)[part_idx]
                        else:
                            time_series = time_series_whole

                        if len(time_series) < 1:
                            aggregate = np.nan
                        else:
                            aggregate = func(time_series)
                        feat_col.append(aggregate)
                    dyn_stats_cols.append(feat_col)

        # Dynamic data trend lines (which are trend lines fitted on dynamic data)
        dyn_derived_cols = []
        dyn_derived_names = []
        if dyn_trend_info is not None:
            for dyn_trend_attr_name, dyn_trend_data in dyn_trend_info.items():
                dyn_derived_names.append(f"TREND({dyn_trend_attr_name})")

                trend_col = [vals[0] if len(vals) > 0 else np.nan for vals in dyn_trend_data]
                dyn_derived_cols.append(trend_col)

        # Static data: Leave numerical values as they are and transform categorical data to integer indices
        static_cols = []
        static_names = []
        static_categs_values = self.get_static_data_categ_values()
        for static_attr_name, static_data in static_info.items():

            if not with_future_info and "future" in static_attr_name.lower():
                continue

            static_names.append(static_attr_name)
            if static_attr_name not in static_categs_values:
                # Numerical attribute
                static_cols.append(static_data)
            else:
                # Categorical attribute
                static_categ_indices = [static_categs_values[static_attr_name].index(val) for val in static_data]
                static_cols.append(static_categ_indices)

        # ICD codes (diagnoses and procedures)
        # Since there are so many ICD codes, include them as a one-hot encoding of top-level ICD categories.
        # For each patient the tree will then have access to the presence or non-presence of those categories.

        # Extract ICD attributes and data
        icd_diagnosis_names = []
        icd_diagnosis_cols = []
        icd_procedure_names = []
        icd_procedure_cols = []
        if with_future_info:
            _, static_categorical = self.get_scaled_static_data_array()
            icd_attrs = [attr for attr in self.get_static_attrs_listlike() if "icd" in attr]

            for icd_attr_name in icd_attrs:
                # Find out which ICD codes are possible in the population (this list is also the interpretation
                # for the present_icd_codes_indices)
                possible_icd_codes = self.get_static_data_categ_values()[icd_attr_name]

                # For each code (which is a string), either find out the top-level category or the code node itself
                if aggregate_icds:
                    icd_idx_lookup = [IcdInfo.icd_categ_level_1(icd_attr_name, code) for code in possible_icd_codes]
                else:
                    icd_idx_lookup = [IcdInfo.name_for_code(icd_attr_name, code) for code in possible_icd_codes]

                # Find out ICD codes for each of the admissions
                present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
                present_icd_codes_indices = [present_icd_codes_indices[idx]  # bring into split order
                                             for idx in self.trainer.get_split_indices(split)]

                # Set one-hot columns for each ICD category
                icd_columns = {}
                for adm_internal_idx, adm_icd_indices in enumerate(present_icd_codes_indices):
                    # Find out what ICD codes/categories are present for this admission
                    present_icd_props = np.unique([icd_idx_lookup[icd_idx] for icd_idx in adm_icd_indices])

                    # Make an entry for each of the present categories
                    for icd_entity in present_icd_props:
                        if icd_entity not in icd_columns:
                            # Init one-hot with all zeros (for each admission)
                            icd_columns[icd_entity] = np.zeros(shape=(len(present_icd_codes_indices),))

                        # Enter a one for this admission
                        col = icd_columns[icd_entity]
                        col[adm_internal_idx] = 1.

                # Add ICD columns to feature columns (and names)
                icd_categ_names = list(icd_columns.keys())
                icd_categ_cols = [icd_columns[icd_categ] for icd_categ in icd_categ_names]
                if icd_attr_name == 'icd9_code_diagnoses':
                    icd_diagnosis_names = icd_categ_names
                    icd_diagnosis_cols = icd_categ_cols
                elif icd_attr_name == 'icd9_code_procedures':
                    icd_procedure_names = icd_categ_names
                    icd_procedure_cols = icd_categ_cols
                else:
                    assert False, f"ICD attribute '{icd_attr_name}' unknown!"

        # Assemble return data
        return (
            (dyn_stats_names, dyn_stats_cols),
            (dyn_derived_names, dyn_derived_cols),
            (static_names, static_cols),
            (icd_diagnosis_names, icd_diagnosis_cols),
            (icd_procedure_names, icd_procedure_cols)
        )

    def generate_baseline_data_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Generate matrix for baseline method to use as input. The matrix contains the medical input data in such a way
        that the baseline method (which is a PCA) can work with it

        :return: np.ndarray with shape (n_samples, n_features), data column names
        """

        logging.info("Gathering baseline model data ...")

        # Prepare input data
        static_info = self.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(io.split_name_all)
        )

        dyn_times, dyn_info = self.get_dyn_medical_data(
            adm_indices=self.trainer.get_split_indices(io.split_name_all),
            with_times=True
        )

        # Put the data in columns (first step in assembling it in matrix form)
        (
            (dyn_stats_names, dyn_stats_cols),
            (_, _),
            (static_names, static_cols),
            (_, _),
            (_, _)
        ) = self.prepare_input_data_for_matrix(
            static_info=static_info,
            dyn_info=dyn_info,
            dyn_trend_info=None,
            split=io.split_name_all,
            aggregate_icds=False,
            agg_funcs={
                "MEAN": np.mean
            },
            overwrite_agg_funcs=True,
            time_series_partitions=1,  # don't split time series into multiple segments - use only a single segment
            with_future_info=False  # Don't let baseline see "future" information such as the patient's survival
        )

        # Compose into one large array
        data_columns = dyn_stats_cols + static_cols
        data_matrix = np.stack(data_columns).T
        # (shape (n_samples, n_features))

        # Also concatenate column names
        col_names = dyn_stats_names + static_names

        # Impute missing data in the matrix
        data_matrix = self._impute_baseline_matrix(
            data_matrix=data_matrix
        )

        # Normalize columns: Fit on train rows, then apply to all rows
        pca_scaler = self._get_fresh_scaler()
        matrix_train_indices = self.trainer.find_train_indices_in_all()  # matrix was created using 'all' split, not
        # original admission order!
        matrix_train = data_matrix[matrix_train_indices]

        pca_scaler.fit(matrix_train)
        data_matrix = pca_scaler.transform(data_matrix)
        self.baseline_scaler = pca_scaler

        logging.info("Gathering baseline model data done!")

        return data_matrix, col_names

    def _impute_baseline_matrix(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Impute missing data: Fit on train indices, then apply to whole matrix
        :param data_matrix:
        :return:
        """

        logging.info("Imputing baseline model data ...")

        # Set up imputer
        imputer = SimpleImputer(strategy='median')  # median strategy is the safest and most robust

        # Extract train admissions
        # Special care must be taken since the admissions in data_matrix are not in the original admission order,
        # but the order given by split_indices_all
        matrix_train_indices = self.trainer.find_train_indices_in_all()
        matrix_train = data_matrix[matrix_train_indices]

        # Fit imputer and apply it to full matrix
        imputer.fit(matrix_train)
        data_matrix = imputer.transform(data_matrix)

        logging.info("Imputing baseline model data done!")

        return data_matrix

