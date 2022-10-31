# Logging
import logging

# Math and data processing
import re
import numpy as np
import pandas as pd

# Utility and files
from typing import Dict
from tqdm import tqdm
from pathlib import Path
import os
import sqlalchemy

# Multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool, cpu_count


logging.getLogger().setLevel(logging.INFO)


"""
Credit: Richard Polzin, Konstantin Sharafutdinov, Kilian Merkelbach, Jayesh Bhat 
"""


class BaseExtractor:
    def __init__(self, database_name, extraction_path, extraction_kind, raw_data_tables, raw_data_table_preferred=None,
                 static_extraction_details=None, disable_multiprocessing=False):
        # Name of the specific database
        self.database_name = database_name

        # Prepare path for extraction
        self.extraction_path_proc = Path(extraction_path, f"{extraction_kind}_processed")
        self.extraction_path_proc.mkdir(exist_ok=True, parents=True)
        self.extraction_path_raw = Path(extraction_path, f"{extraction_kind}_raw")
        self.extraction_path_raw.mkdir(exist_ok=True, parents=True)
        self.extraction_path_buffered = Path(extraction_path, "buffered_tables")
        self.extraction_path_buffered.mkdir(exist_ok=True, parents=True)
        self.extraction_path_static = Path(extraction_path, "static")
        self.extraction_path_static.mkdir(exist_ok=True, parents=True)

        # Save raw data names: These are the tables of temporally-changing attributes ("events") we will extract
        # from the database or the names of ItemIds to extract.
        # For some classes, this field will contain tuples of values that are then unpacked by query_data.
        self.raw_data_tables = raw_data_tables
        self.raw_data_table_trusted = raw_data_table_preferred

        # Define what static data to extract
        self.static_extraction_details = static_extraction_details
        if self.static_extraction_details is None:
            logging.info("Not extracting static data.")
            self.tables = {}
        else:
            logging.info("Loading static tables into memory...")
            self.tables = self.load_static_tables()
            logging.info("Done loading static tables!")

        # Multiprocessing
        if not disable_multiprocessing:
            self.num_cpus = cpu_count()
        else:
            self.num_cpus = 1

        # Admission times (for each encounter ID)
        self.admissions_times = self.retrieve_full_table("admissions")[["hadm_id", "admittime"]]\
            .rename(columns={"hadm_id": "encounterid", "admittime": "admission_time"})\
            .set_index("encounterid")
        assert self.admissions_times.index.name == "encounterid", "The admission times table must be indexed with" \
                                                                  "the encounter ID!"

        # Define error for non-implemented (abstract) methods
        self._not_implemented_error = NotImplementedError("Please implement this method in the subclass.")

    @staticmethod
    def get_eng():
        try:
            from itcdb import sqlengine
            return sqlengine(dbname='mimic')
        except ImportError:
            # Find out user-specified MIMIC database URL
            db_var_name = "MIMIC_URL"
            db_url = os.getenv(db_var_name, None)
            assert db_url is not None, f"Please set the environment variable '{db_var_name}' to the URL of a " \
                                       f"MIMIC-III database"

            # Establish connection with database
            return sqlalchemy.create_engine(db_url)

    def load_static_tables(self):
        """
        Loads (or downloads, if necessary) the requested static data tables into memory
        :return: dictionary with table names as keys, tables as values
        """
        # Retrieve tables specified in the static data details
        tables = {}
        for table_name in self.static_extraction_details.keys():
            if table_name == "chartevents":
                continue  # Don't download chartevents table
            tables[table_name] = self.retrieve_full_table(table_name=table_name, buffer_on_disk=True, silent=False)
        # Table might already exist, but that is not a problem.

        return tables

    def extract_dynamic_raw_data(self, encounter_id):
        """
        Extracts and normalizes the events in multiple tables for a single encounter ID
        :param encounter_id: ID of encounter to extract
        :return: df (DataFrame), encounter_id
        """

        # Retrieve events for each raw data name listed for events
        results = {table_name: self.query_data(encounter_id, table_name) for table_name in self.raw_data_tables}
        results = {table_name: tab.dropna(subset=['valuenum']) for (table_name, tab) in results.items()}
        data_amounts = [len(df) for df in results.values()]

        # It is possible for a table to be designated as preferred, i.e. it is to be trusted more than the other data
        # sources (e.g. for MIMIC, labevents should be trusted if there are conflicts between chartevents and labevents)
        if self.raw_data_table_trusted is None or min(data_amounts) == 0:
            df = pd.concat(results.values())
        else:
            # Split into trusted table and others
            table_trust = results[self.raw_data_table_trusted]
            del results[self.raw_data_table_trusted]
            table_normal = pd.concat(results.values())

            # Set item id and time as index for the tables
            table_trust = table_trust.set_index(['itemid', 'charttime'])
            table_normal = table_normal.set_index(['itemid', 'charttime'])

            # Group by the index (which eliminates entries with duplicated index)
            table_trust = table_trust.groupby(level=table_trust.index.names).mean()
            table_normal = table_normal.groupby(level=table_normal.index.names).mean()
            # (note: if one or more of the entries being grouped is NaN, pandas first uses any existing non-NaN
            # entries, i.e. the "mean" of [NaN, 12] would be 12.)

            # "Update" (overwrite) conflicting entries in normal table with trusted table
            table_normal.update(table_trust)

            # Add rows of trusted table that do *not* appear in normal table
            trust_new_entries = table_trust.loc[[i for i in table_trust.index if i not in table_normal.index], :]
            table_normal = table_normal.append(trust_new_entries)

            # Reset index - it was only used to merge the rows of the data sources
            df = table_normal.reset_index()

        # Pre-process the data structurally: Bring it into a format that is unified among datasets
        self.preprocess_raw_events(df)

        # Drop any intermediate indices on the DataFrame that we might have created
        df.reset_index(drop=True, inplace=True)

        return df, encounter_id

    def extract_dyn_data_as_static(self, table_name: str, encounter_info: Dict, key_field: Dict, fields_to_extract: Dict) -> Dict:
        """
        Extract dynamic data and treats it as static. Used for weight, height, etc.

        :param table_name:
        :param encounter_info:
        :param key_field:
        :param fields_to_extract:
        :return:
        """

        # Retrieve data from table
        query_tokens = [
            "SELECT itemid, charttime, valuenum",
            f"FROM {table_name}",
            f"WHERE {key_field['name']} = {encounter_info[key_field['value']]}",
            "AND valuenum IS NOT NULL",
            "AND valuenum <> 0",
            f"AND ( itemid IN ({', '.join(sum(list(fields_to_extract.values()), []))}) )"
        ]
        query = "\n".join(query_tokens)
        df = self.execute_query(query)

        # Prepare output dictionary
        out_dict = {
            'weight_kg': np.nan,
            'height_cm': np.nan
        }

        if len(df) == 0:
            return out_dict

        # Pivot table
        df = df.pivot_table(index='charttime', columns='itemid', values='valuenum', aggfunc='mean')

        # Remove temporal dimension by median-aggregating
        df = df.median(skipna=True)

        # Aggregate values
        agg = {}
        for field_name, item_ids in fields_to_extract.items():
            field_vals = [df[int(i_id)] for i_id in item_ids if int(i_id) in df]
            if len(field_vals) == 0:
                val = None
            else:
                val = np.median(field_vals)
            agg[field_name] = val

        # Convert all weight into kg
        all_weights_kg = [
            agg['weight_kg'] if agg['weight_kg'] is not None else np.nan,
            agg['weight_oz'] / 35.274 if agg['weight_oz'] is not None else np.nan,
            agg['weight_lb'] / 2.2046 if agg['weight_lb'] is not None else np.nan
        ]
        weight_kg = np.nanmedian(all_weights_kg)

        # Convert all heights into cm
        all_heights_cm = [
            agg['height_cm'] if agg['height_cm'] is not None else np.nan,
            agg['height_in'] * 2.54 if agg['height_in'] is not None else np.nan
        ]
        height_cm = np.nanmedian(all_heights_cm)

        # Return new values as a dictionary
        out_dict['weight_kg'] = weight_kg
        out_dict['height_cm'] = height_cm
        return out_dict

    def extract_static_raw_data(self, encounter_id):
        """
        Extracts static data. Static data is data that does not change over time and is often known about a patient
        when they are admitted into the hospital.

        :param encounter_id: ID of encounter
        :return: encounter_info (dict), encounter_id
        """

        # Dict in which we store everything we know about this encounter. Each table we extract from is going to give
        # us additional information about the encounter. All attributes in encounter_info can also serve as a key for
        # tables.
        encounter_info = {
            "encounter_id": encounter_id
        }

        # List of tables out of which all requested information has been extracted
        extracted_tables = []

        # Extract from the tables
        all_extracted = False
        while not all_extracted:
            for table_name, extraction_details in self.static_extraction_details.items():

                # Skip this table if it has been extracted already
                if table_name in extracted_tables:
                    continue

                # Check if we can extract from this table already - for this, we need to have the key value already
                # extracted (e.g. 'subject_id' for MIMIC's patient table)
                if extraction_details['key_field']['value'] not in encounter_info:
                    # We need to handle this table later - we don't know the key value yet
                    continue

                # Some fields (weight, height) need to be extract from dynamic data tables but are still treated
                # as static data
                if type(extraction_details['fields_to_extract']) == dict:
                    dyn_as_static_extracted = self.extract_dyn_data_as_static(
                        table_name=table_name,
                        encounter_info=encounter_info,
                        key_field=extraction_details['key_field'],
                        fields_to_extract=extraction_details['fields_to_extract']
                    )
                    encounter_info.update(dyn_as_static_extracted)

                    # Add this table to the list of tables we have extracted from
                    extracted_tables.append(table_name)

                    continue

                # Get table
                t = self.tables[table_name]

                # Filter this table to rows conforming to the key
                key_value = encounter_info[extraction_details['key_field']['value']]
                key_field_name = extraction_details['key_field']['name']
                encounter_rows = t[t[key_field_name] == key_value]

                # Extract the specified fields to encounter_info
                for field_name in extraction_details['fields_to_extract']:
                    field_values = encounter_rows[field_name]

                    # Rename the field if requested
                    if 'field_renamings' in extraction_details and field_name in extraction_details['field_renamings']:
                        dest_field_name = extraction_details['field_renamings'][field_name]
                    else:
                        dest_field_name = field_name

                    if len(field_values) > 1:
                        # More than one possible value for this field
                        encounter_info[dest_field_name] = list(field_values)
                    elif len(field_values) == 1:
                        # Get first field value and add it to encounter_info
                        field_value = encounter_rows[field_name].iloc[0]
                        encounter_info[dest_field_name] = field_value
                    else:
                        # There is no value stored
                        encounter_info[dest_field_name] = []

                # Add this table to the list of tables we have extracted from
                extracted_tables.append(table_name)

            # Check if everything was extracted
            all_extracted = extracted_tables == list(self.static_extraction_details.keys())

        encounter_info = self.postprocess_raw_static_data(encounter_info)

        return encounter_info, encounter_id

    def postprocess_raw_static_data(self, encounter_info):
        """
        Post-processes encounter_info dictionary. This can be things like calculating the patient's age from their date
        of birth and admission date or other, more elaborate procedures.
        Implementation depends on specific data, so is performed in subclasses.

        :param encounter_info:
        :return: encounter_info with new information added
        """

        return encounter_info

    def preprocess_raw_events(self, df):
        """
        Pre-processes downloaded raw events structurally. This means that no normalization or other change dependent on
        the actual content of the data is performed. Only the structure of the data (e.g. the data type of specific
        fields or the structure of a table) is changed.
        The implementation of this is specific to each dataset.

        :param df: DataFrame to restructure
        :return: restructured DataFrame
        """

        return df

    def query_data(self, encounter_id, data_name_info):
        """
        Constructs the query for the database and retrieves it.
        Implementation is specific to dataset.

        :param encounter_id: ID of encounter to extract
        :param data_name_info: Name of raw data item or some other object like a tuple of values
        :return:
        """

        raise self._not_implemented_error

    def extract(self, encounter_ids, override):

        def raw_path_for_id(encounter_id_0):
            return Path(str(self.extraction_path_raw),  str(encounter_id_0) + ".feather")

        logging.info("Starting raw dynamic data extraction...")
        if not override:
            encounter_ids_raw_to_get = set(encounter_ids) - \
                                       set(BaseExtractor.get_existing_ids(self.extraction_path_raw))
            logging.info(f"{len(encounter_ids) - len(encounter_ids_raw_to_get)} ids already found.")
        else:
            encounter_ids_raw_to_get = set(encounter_ids)
            logging.info("Extracting raw dynamic data for all requested encounter IDs since override was requested.")

        # Extract raw dynamic data from database
        if len(encounter_ids_raw_to_get) > 0:
            logging.info(f"Extracting raw dynamic data for {len(encounter_ids_raw_to_get)} encounters...")
            raw = []
            for enc_id in tqdm(encounter_ids_raw_to_get, total=len(encounter_ids_raw_to_get),
                               desc="Extracting raw dynamic data"):
                df, encounter_id = self.extract_dynamic_raw_data(enc_id)

                # Save raw dynamic data to file
                df.to_feather(raw_path_for_id(encounter_id))

                raw.append((df, encounter_id))
            logging.info("Extraction of raw dynamic data finished.")

        # Determine which of the encounters are still only raw (and have never been processed)
        unprocessed_encounters = self.check_if_dyn_data_present(
            encounter_ids=encounter_ids
        )

        # Load saved data (for encounters missing from HDF file)
        if len(unprocessed_encounters) > 0:
            logging.info(f"Loading existing raw dynamic data ({len(unprocessed_encounters)} encounters)")
            raw = [(pd.read_feather(raw_path_for_id(encounter_id)), encounter_id)
                   for encounter_id in unprocessed_encounters]
            logging.info("Loading existing raw dynamic data done!")

            # Convert raw dynamic data into a proper time-series
            logging.info("Converting raw dynamic data to time series...")
            self.raw_to_timeseries(raw)
            logging.info("Converting done!")

    def extract_static(self, encounter_ids):
        """
        Extract static data
        :param encounter_ids:
        :return:
        """

        logging.info(f"Starting static data extraction of {len(encounter_ids)} encounters")

        # Check which of the files already exists
        remaining_encounter_ids = []
        completed_encounter_count = 0
        for encounter_id in encounter_ids:
            path = self.path_for_static_data(encounter_id)
            if not path.exists():
                remaining_encounter_ids.append(encounter_id)
            else:
                completed_encounter_count += 1
        if completed_encounter_count > 0:
            logging.info(f"{completed_encounter_count} encounter ids were already done. Not extracting them again.")

        # Extract the remaining encounters
        logging.info(f"Extracting and saving {len(remaining_encounter_ids)} remaining encounters...")
        for enc_id in tqdm(remaining_encounter_ids, desc="Extracting and saving..."):
            enc_info, _ = self.extract_static_raw_data(enc_id)

            # Convert to pandas Series
            enc_info = pd.Series(enc_info)

            # Write to disk
            enc_info.to_pickle(self.path_for_static_data(enc_id).as_posix(), protocol=4)

        logging.info("Done extracting and saving.")

    def path_for_static_data(self, encounter_id):
        return Path(self.extraction_path_static, f"{encounter_id}.pkl")

    def path_for_dyn_proc(self, encounter_id):
        return Path(self.extraction_path_proc, f"dynproc_{encounter_id}.feather")

    def load_dynamic_data(self, encounter_id):
        dyn_path = self.path_for_dyn_proc(encounter_id)
        if not dyn_path.exists():
            return None  # We could not load the file

        # Load the file
        loaded_dynamic = pd.read_feather(dyn_path)
        return loaded_dynamic

    def retrieve_all_encounter_ids(self):
        """
        Retrieves a lists of all encounter IDs. Implementation is specific to subclass.
        :return: list of all encounter IDs
        """

        raise self._not_implemented_error

    def raw_to_timeseries(self, raw):
        logging.info("Converting raw data to timeseries.")
        with Pool(ncpus=self.num_cpus) as pool:
            list(tqdm(pool.imap(self.df_to_ts, raw), total=len(raw)))

    def check_if_dyn_data_present(self, encounter_ids):
        """
        Checks which of the given encounter ids are present in storage of processed dynamic data
        :param encounter_ids:
        :return: list of encounter IDs NOT present
        """

        missing_encounters = []
        for encounter_id in encounter_ids:
            dyn_proc_path = self.path_for_dyn_proc(encounter_id)
            if not dyn_proc_path.exists():
                missing_encounters.append(encounter_id)

        return missing_encounters

    def df_to_ts(self, raw):
        # Unpack raw tuple into distinct variables
        df, encounter_id, columns, index, values = raw

        # Try loading existing data from HDF file
        if self.load_dynamic_data(encounter_id) is not None:
            return  # The processed dynamic data is already present in our storage

        # Process raw dynamic data into time series
        try:
            # Further processing of data can only continue when we actually have data for this admission
            if not df.empty:
                # Sanity check: Count the number of unique item ids
                num_item_ids = len(np.unique(df.itemid))

                df = df.pivot_table(index=index, columns='itemid', values=values, aggfunc='mean')

                # Get the admission time - after this step, timestamps are no longer expressed in absolute terms but
                # rather in them number of minutes since the beginning of the admission.
                df.index = (df.index - self.admissions_times.loc[encounter_id, 'admission_time']).total_seconds() // 60

                # Sanity check: Number of columns should be equal to the number of item ids we counted earlier
                assert num_item_ids == len(df.columns)

                df = df.reset_index()

            # Writing of admission to file happens also for EMPTY DataFrames.
            # Even though there may not be data, we still want to save a reference to this admission in our data store,
            # since otherwise, we will try to retrieve this data from the database again and again

            # Write this admission's data to file
            df.columns = df.columns.astype(str)  # Convert column names to strings (feather requires this)
            df.to_feather(self.path_for_dyn_proc(encounter_id))

        except pd.core.base.DataError:  # happens when all indices are empty
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                logging.error("Encountered critical error during timeseries conversion for the following dataset:")
                logging.error(df)
                logging.error("Skipping and continuing ...")
        except KeyError:
            logging.error(f"Could not extract data for encounter id {encounter_id} since it is not present in "
                          f"self.admission_times!")
            logging.error("Continuing with other files.")

    def execute_query(self, query):
        """Runs a Query"""
        query = query.replace('[', '(').replace(']', ')')
        eng = self.get_eng()
        result = pd.read_sql(query, con=eng)
        return result

    def retrieve_full_table(self, table_name, buffer_on_disk=True, silent=True):
        """
        Download the full table using the database connection. Warning: May take a long time.
        :param table_name: name of the table to be downloaded
        :param buffer_on_disk: If True, save downloaded table to disk (for faster loading next time), otherwise do not.
        :param silent: If True, don't print what is happening
        :return: table as DataFrame
        """

        # Determine path (if buffered)
        if buffer_on_disk:
            table_path = Path(self.extraction_path_buffered, table_name)

            if table_path.exists():
                table = pd.read_feather(table_path)
                file_found = True
            else:
                file_found = False

        # Download the table if necessary
        if not buffer_on_disk or (buffer_on_disk and not file_found):
            if not silent:
                logging.info(f"Downloading table '{table_name}'. This may take a while...")
            take_all_query = f"select * from {table_name}"
            table = BaseExtractor.execute_query(self, take_all_query)

        # Buffer the table if requested
        if buffer_on_disk and not file_found:
            table.to_feather(table_path)

        return table

    @staticmethod
    def get_existing_ids(path):
        """Checks output path and returns a set of all existing encounter_ids files there."""
        logging.info("Looking for previously created files")
        all_files = [f for f in Path(path).glob("*.feather")]
        if len(all_files) > 0:
            results = [BaseExtractor.id_from_filename(f.name) for f in all_files]
            logging.info(f"Found {len(results)} existing ids.")
            return results
        else:
            return []

    @staticmethod
    def id_from_filename(f):
        return int(re.search('[0-9]+', str(f)).group(0))
