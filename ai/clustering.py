# Logging
import logging

# Randomness
import random
import string
from collections import namedtuple
from itertools import product

# Types
from typing import List, Dict, Union, Tuple

# Util
import os

# Math and data
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.api as sms
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, adjusted_rand_score, davies_bouldin_score,\
    calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform
from collections import Counter

from common import io


# Info on clustering
class ClusteringInfo:
    def __init__(self, algorithm: str, options: Dict[str, Union[int, float, str]], labels: np.ndarray):
        self.algorithm = algorithm
        self.options = options
        self.labels = labels

        # Give random ID to clustering
        self.random_id = self._gen_random_id()

        # Technicals: technical quality indicators for clustering
        self.technicals = []

        # Robustness: Describes robustness of clustering w.r.t. resampling of data points
        self.robustness_measurements = []
        self.robustness = {}  # type: Dict[str, RobustnessInfo]
        self.is_robust = None

        # Interest: Cluster is "interesting" if it is a top performer in one of the robustness measures
        self.is_interesting = False

        # Actual and "effective" number of clusters
        self.num_clus = None
        self.num_clus_effective = None

        # "Parentage" of clustering (w.r.t. coarser clusterings)
        self.parent_labels = {}  # key: parent clustering ID, label
        self.parent_clusterings = []

        # For plotting and evaluation: A descriptive path name for this clustering
        self._path_name = None

    @staticmethod
    def _gen_random_id():
        return "".join(
            [random.choice(string.ascii_uppercase) for _ in range(2)]
            + ["_"]
            + [random.choice(string.digits) for _ in range(6)]
        )

    def __repr__(self):
        return f"ClusteringInfo({self.algorithm}, {self.options}, {self.random_id})"

    def get_path_name(self):
        if self._path_name is None:
            avg_technical_quality = np.mean([tech.score for tech in self.technicals])
            self._path_name = f"clus{self.num_clus:03d}_quali{avg_technical_quality:0.2f}_{self.random_id}"
        return self._path_name


ClusteringCollection = namedtuple("ClusteringCollection", ["clusterings", "sampled_indices"])


RobustnessInfo = namedtuple("RobustnessInfo", ["rob_name", "rob_threshold", "rob_actual", "is_robust"])


TechnicalQuality = namedtuple("TechnicalQuality", ["name", "score"])


def analyze_clusterings_lineage(clusterings: List[ClusteringInfo]) -> None:
    """
    Determine the parentage of each clustering, i.e. where admissions for each cluster come from in other, coarser
    clusterings
    :param clusterings:
    :return:
    """
    # Sort the clusterings by their number of clusters
    clusterings.sort(key=lambda c: c.num_clus)

    # Determine parentage for each cluster (except the coarsest clustering, which does not have parent clustering)
    for clus_child_idx in range(1, len(clusterings)):
        clus_child = clusterings[clus_child_idx]

        # Go up the tree over all ancestors (from most immediate to the root)
        for clus_parent_idx in range(clus_child_idx)[::-1]:
            clus_parent = clusterings[clus_parent_idx]

            # Save parentage of the child clustering to the parent clustering
            clus_child.parent_clusterings.append(clus_parent.random_id)

            # Remember the parent label distribution for each label of the child clustering
            clus_child.parent_labels[clus_parent.random_id] = {}
            for label in np.unique(clus_child.labels):
                label_idcs = np.where(clus_child.labels == label)[0]
                child_size = len(label_idcs)

                # Find out the labels within the parent clustering
                parent_labels, parent_counts = np.unique(clus_parent.labels[label_idcs], return_counts=True)

                # Save parentage
                clus_child.parent_labels[clus_parent.random_id][label] = {
                    int(p_label): float(p_count / child_size)
                    for (p_label, p_count) in zip(parent_labels, parent_counts)
                }


def mutual_information(clus_1_labels, clus_2_labels):
    mutual_info = adjusted_mutual_info_score(
        labels_true=clus_1_labels,
        labels_pred=clus_2_labels,
        average_method='arithmetic'
    )
    mutual_info = float(mutual_info)
    # 0. -> random, 1. -> same labeling
    return mutual_info


def rand_score(clus_1_labels, clus_2_labels):
    score = adjusted_rand_score(
        labels_true=clus_1_labels,
        labels_pred=clus_2_labels
    )  # -1. -> no similarity, 0. -> random, 1. -> same labeling
    score = float(score)
    return score


def swap(arr, num_1, num_2):
    intermediate_value = max(arr) + 1
    arr[arr == num_1] = intermediate_value
    arr[arr == num_2] = num_1
    arr[arr == intermediate_value] = num_2
    return arr


def jaccard_score_label_agnostic(clus_1_labels, clus_2_labels):
    """
    Jaccard score is usually not agnostic to the literal labels. What this means is that the labelings [0,0,1,1]
    and [0,0,1,1] would get a perfect score but the labelings [0,0,1,1] and [1,1,0,0] would get the worst score
    possible, even though the labelings in the second case are just as similar if one disregards the literal
    name of the labels and instead focuses on which subsets of points are labeled the same.

    :param clus_1_labels:
    :param clus_2_labels:
    :return:
    """

    def jaccard(labels_1, labels_2):
        # Handle labels with more support first: Swapping for a less represented label is only possible if the swap
        # is not forbidden (or was executed in the other direction) by a label with more support.
        labels_uniq, counts = np.unique(labels_1, return_counts=True)
        labels_uniq, counts = zip(*sorted(zip(labels_uniq, counts), key=lambda tup: tup[1], reverse=True))
        num_points = len(labels_1)
        jaccard_index = 0
        for label in labels_uniq:
            # Get clustering 2 labels at the positions of the current label from clustering 1
            corresponding_labels = labels_2[np.where(labels_1 == label)]
            most_common_label = scipy.stats.mode(corresponding_labels).mode[0]

            # Jaccard score measures intersection over union. The largest possible intersection with label of
            # clustering 1 exists in most_common_label in clustering 2.
            intersection_size = len(corresponding_labels[corresponding_labels == most_common_label])

            # Find out size of union: the points occupied by label in clustering 1 or by most_common_label in
            # clustering 2
            clus_1_pos = np.where(labels_1 == label)[0]
            clus_2_pos = np.where(labels_2 == most_common_label)[0]
            union_size = len(np.union1d(clus_1_pos, clus_2_pos))

            # Find out IoU
            intersection_over_union = intersection_size / union_size

            # To correct for label imbalances, weight the IoU by the number of points it pertains to
            weight = union_size / num_points
            jaccard_index += weight * intersection_over_union

        jaccard_index = float(jaccard_index)
        return jaccard_index

    # To attain symmetry, run Jaccard index in both "directions" and choose the lower one
    return min(
        jaccard(clus_1_labels, clus_2_labels),
        jaccard(clus_2_labels, clus_1_labels)
    )


# List robustness functions explicitly: Key is name as string and value is the function for computing the robustness
robustness_functions = {
    'mutual_info': mutual_information,
    'rand_score': rand_score,
    'jaccard_score': jaccard_score_label_agnostic
}


def split_by_cluster(labeling, values):
    labels = sorted(np.unique(labeling))
    vals_by_cluster_sorted = []
    for lab in labels:
        lab_idcs = np.where(labeling == lab)[0]
        lab_vals = [val for (idx, val) in enumerate(values) if idx in lab_idcs]
        vals_by_cluster_sorted.append(lab_vals)
    return vals_by_cluster_sorted


class Clustering:
    """
    Partition sets of points based on similarity in location.
    """

    def __init__(self, trainer, iom, preprocessor, split, num_bootstraps: int = 10):
        self.trainer = trainer
        self.iom = iom
        self.preprocessor = preprocessor
        self.split = split

        # Clusters
        self.clusterings = []  # type: List[ClusteringInfo]
        self.clusterings_robust = []  # type: List[ClusteringInfo]
        self.clusterings_all = []  # type: List[ClusteringInfo]

        # Clustering information table: One table for each cluster (of those chosen for eval)
        # The table contains information about each cluster (like average age and other medical facts)
        self._clustering_summary_tables = {}  # type: Dict[str, pd.DataFrame]
        # (key: clustering_info.random_id)

        # Create directory for storing (or loading) clusterings
        self.clusterings_dir = os.path.join(self.iom.get_clusterings_dir(), self.split)
        io.makedirs(self.clusterings_dir)
        self._clustering_file_path = os.path.join(self.clusterings_dir, f"{io.sanitize(self.split)}.pkl")

        # Settings
        self.num_bootstraps = num_bootstraps
        self.frac_clus_picked_per_measure = 0.5  # 0.5 -> best 50%
        self.num_clusterings_picked_total = 6

    def _determine_robustness_threshold(self, robustness_func, clustering_info: ClusteringInfo) -> float:
        """
        Determines the thresholds for a robustness measure. Clusterings that meet the threshold can
        be considered robust
        :return:
        """

        labeling = clustering_info.labels

        # Shuffle multiple times to make procedure more stable
        thresholds = []
        for _ in range(self.num_bootstraps):
            # Shuffle the original labeling
            labeling_shuffled = labeling.copy()
            random.shuffle(labeling_shuffled)

            # Get robustness between original and shuffled labeling
            rob = robustness_func(labeling, labeling_shuffled)
            thresholds.append(rob)
            # (this provides a meaningful lower bound for the robustness we want in a clustering since the original and
            #  shuffled labelings are not truly similar)

        # Final threshold (lower bound) is pretty strict: We take the 90%-percentile of all trials. Taking the
        # 90%-percentile instead of the mean makes it *harder* for a clustering to be considered robust (since it has
        # to surpass a higher threshold)
        lower_bound = float(np.percentile(thresholds, 90))

        return lower_bound

    @staticmethod
    def _is_pseudo_trivial(clustering_info: ClusteringInfo) -> bool:
        """
        Clustering can be considered "pseudo-trivial" if it has one extremely large cluster that contains nearly all
        points
        :param clustering_info:
        :return:
        """

        # Get sizes of clusters
        labels, counts = np.unique(clustering_info.labels, return_counts=True)
        top_label_idx = np.argmax(counts)

        # Compare size of maximal cluster with rest of points
        top_count = counts[top_label_idx]
        total_count = np.sum(counts)
        top_ratio = top_count / total_count

        # Decide if this clustering is pseudo-trivial
        pseudo_trivial = top_ratio > 0.90
        return pseudo_trivial

    @staticmethod
    def _is_trivial(clustering_info: ClusteringInfo) -> bool:
        return len(np.unique(clustering_info.labels)) < 2

    def _bootstrap(self, features_full: np.ndarray, size_ratio: float = 0.7) -> List[ClusteringCollection]:
        bootstrap_size = int(size_ratio * len(features_full))
        bootstrappings = []
        for boot_idx in range(self.num_bootstraps):
            # Sample indices of feature points to use for this bootstrap
            logging.info(f"Bootstrapping {boot_idx + 1} of {self.num_bootstraps}")
            features_indices = np.random.choice(np.arange(0, len(features_full)), bootstrap_size, replace=False)
            features_bootstrapped = features_full[features_indices]

            # Save bootstrapped point indices and clusterings
            bootstrappings.append(
                ClusteringCollection(
                    clusterings=self._perform_all_clusterings(features=features_bootstrapped),
                    sampled_indices=features_indices
                )
            )

        return bootstrappings

    def cluster_admissions(self):
        """
        Clusters validation admissions based on their features
        :return:
        """

        # Get admission features
        features = self.trainer.compute_features(self.split)

        # Try to load clusters from disk
        self.load_clusterings_from_disk()

        # Cluster admissions based on the features (if loading didn't work)
        if len(self.clusterings) == 0:
            clusterings, clusterings_robust, clusterings_all = self._cluster_complete(features=features)
            self.clusterings = clusterings.clusterings
            self.clusterings_robust = clusterings_robust.clusterings
            self.clusterings_all = clusterings_all.clusterings

            # Analyze lineage of clusterings, i.e. where admissions end up on multiple clusterings
            analyze_clusterings_lineage(clusterings=self.clusterings)

            # Save clusterings
            self.save_clusterings_to_disk()

        logging.info("Clustering done!")

    def get_summary_table(self, clustering_info: ClusteringInfo) -> pd.DataFrame:
        """
        Either retrieves stored table or creates one
        :param clustering_info:
        :return:
        """
        key = clustering_info.random_id
        if key not in self._clustering_summary_tables:
            self._clustering_summary_tables[key] = self._create_summary_table(clustering_info)
        return self._clustering_summary_tables[key]

    def _create_summary_table(self, clustering_info: ClusteringInfo) -> pd.DataFrame:
        """
        Table shows information about clustering at a glance
        :param clustering_info:
        :return: summary table
        """
        # Create the table. Medical information will be added later.
        label_str = "Label"
        size_str = "Size"
        size_num_str = "Size (only number)"
        num_adm_total = len(clustering_info.labels)
        data_rows = []

        def size_to_str(size):
            size_rel = size / num_adm_total
            return f"{size} ({100 * size_rel:0.2f}%)"

        for cluster_label in sorted(np.unique(clustering_info.labels)):
            cluster_size = np.count_nonzero(clustering_info.labels == cluster_label)
            data_rows.append({
                label_str: io.label_for_cluster_label(cluster_label),
                size_str: size_to_str(cluster_size),
                size_num_str: cluster_size
            })

        # Add entry for full population (as a comparison for clusters)
        data_rows.insert(
            0,  # Population always at index 0
            {
                label_str: "Population",
                size_str: size_to_str(num_adm_total),
                size_num_str: num_adm_total
            }
        )

        # Make a DataFrame and populate it with some basic info
        table = pd.DataFrame(data=data_rows).set_index(label_str)
        self._populate_summary_table(clustering_info, table)

        return table

    def _populate_summary_table(self, clustering_info: ClusteringInfo, table: pd.DataFrame):
        # Load data required for the table (admission's age, sex, survival, etc.)
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        self.preprocessor.split_days_in_care_by_survival(static_info)

        # Extract age
        def age_stats(ages):
            if len(ages) == 0:
                return "/"
            return f"{np.median(ages):0.0f} years ({np.percentile(ages, 25):0.0f}, {np.percentile(ages, 75):0.0f})"

        # Prepare an additional column that consists only of numerical values. This makes it easier to sort by
        # that attribute when viewing the table
        def age_stats_numbers_only(ages):
            if len(ages) == 0:
                return -1
            else:
                return np.median(ages)

        age_vals = static_info['age_years']

        age_col = [
            age_stats(age_vals)  # first row: population
        ]
        age_nums_col = [
            age_stats_numbers_only(age_vals)
        ]

        for cluster_ages in split_by_cluster(clustering_info.labels, age_vals):
            age_col.append(age_stats(cluster_ages))
            age_nums_col.append(age_stats_numbers_only(cluster_ages))

        table['Age'] = age_col
        table['Age Median (years)'] = age_nums_col

        # BMI, height and weight
        def bmi_stats(bmis, only_mean=False):
            if len(bmis) == 0:
                return "/"
            mean = np.nanmean(bmis)
            if only_mean:
                return mean
            if np.isnan(mean):
                return "/"
            return f"{mean:0.1f} kg/m^2 ({np.nanpercentile(bmis, 25):0.1f}, {np.nanpercentile(bmis, 75):0.1f})"

        bmi_vals = static_info['FUTURE_bmi']
        bmi_upper_limit = int(np.ceil(min(np.nanpercentile(bmi_vals, 95), 75)))
        bmi_vals = [v if np.isfinite(v) and v <= bmi_upper_limit else np.nan for v in bmi_vals]
        bmi_col = [
            bmi_stats(bmi_vals)  # population
        ]
        bmi_nums_col = [
            bmi_stats(bmi_vals, only_mean=True)  # population
        ]
        for cluster_bmis in split_by_cluster(clustering_info.labels, bmi_vals):
            bmi_col.append(bmi_stats(cluster_bmis))
            bmi_nums_col.append(bmi_stats(cluster_bmis, only_mean=True))

        table[f'BMI (mean and percentiles, over {bmi_upper_limit:0.0f} excluded)'] = bmi_col
        table[f'BMI (mean only, over {bmi_upper_limit:0.0f} excluded)'] = bmi_nums_col

        # Extract survival (but express as mortality)
        def mortality_stats(survivals, numerical_output=False):
            if len(survivals) == 0:
                return "/"

            deaths = [not s for s in survivals]
            num_dead = np.sum(deaths)
            percent_dead = 100 * num_dead / len(deaths)

            if numerical_output:
                return percent_dead

            mortality_str = f"{percent_dead:0.1f}%"
            if 0 < percent_dead < 100:
                dead_conf_lower, dead_conf_upper = sms.DescrStatsW(deaths).tconfint_mean()
                mortality_str += f" ({100 * dead_conf_lower:0.1f}%, {100 * dead_conf_upper:0.1f}%)"
            return mortality_str

        # Handle three different definitions of survival: 28-day survival (which means patients survive at least 28 days
        # after being admitted), in-hospital survival (which means patients survive at least until they are discharged
        # from the hospital), and final survival (which means that patients survived beyond the recorded data).
        for mortality_name, mortality_data_source in [('28-day Mortality', 'FUTURE_survival'),
                                                      ('Hospital Mortality', 'FUTURE_survival_to_discharge'),
                                                      ('Final Mortality', 'FUTURE_survival_all_time')]:
            survival_vals = static_info[mortality_data_source]

            mortality_col = [
                mortality_stats(survival_vals)  # first row: population
            ]
            mortality_nums_col = [
                mortality_stats(survival_vals, numerical_output=True)
            ]

            for cluster_survivals in split_by_cluster(clustering_info.labels, survival_vals):
                mortality_col.append(mortality_stats(cluster_survivals))
                mortality_nums_col.append(mortality_stats(cluster_survivals, numerical_output=True))

            table[mortality_name] = mortality_col
            table[f'{mortality_name} (%)'] = mortality_nums_col

        # Extract sex (male/female)
        def sex_stats(sexes, numerical_output=False):
            if len(sexes) == 0:
                return "/"

            num_female = len([sex for sex in sexes if sex == "F"])
            percent_female = num_female / len(sexes)

            if numerical_output:
                return percent_female
            else:
                return f"{100 * percent_female:0.1f}%"

        sex_vals = static_info['gender']
        sex_col = [
            sex_stats(sex_vals)  # first row: population
        ]
        sex_nums_col = [
            sex_stats(sex_vals, numerical_output=True)  # first row: population
        ]

        for cluster_sex in split_by_cluster(clustering_info.labels, sex_vals):
            sex_col.append(sex_stats(cluster_sex))
            sex_nums_col.append(sex_stats(cluster_sex, numerical_output=True))

        table['Sex (% female)'] = sex_col
        table['Sex'] = sex_nums_col

        # Extract days in care
        def days_in_care_stats(days, numerical_output=False):
            days = [d for d in days if d is not None]

            if len(days) == 0:
                if numerical_output:
                    return np.nan
                else:
                    return "/"

            if numerical_output:
                return np.median(days)
            else:
                return f"{np.median(days):0.0f} days ({np.percentile(days, 25):0.0f}, {np.percentile(days, 75):0.0f})"

        # Days in care is expressed in three attributes: One for all admissions, one for survivors and one for the
        # deceased
        for dic_attr_kind, dic_kind_label in [('FUTURE_days_in_care', "ICU"),
                                              ('FUTURE_days_in_care_hospital', "Hospital")]:
            for dic_suffix, dic_suffix_label in [('', "All"),
                                                 ('survivors', "Survivors"),
                                                 ('deceased', "Deceased")]:

                # Produce full attribute name that is stored in static_info
                dic_attr_name = dic_attr_kind
                if len(dic_suffix) > 0:
                    dic_attr_name += f"_{dic_suffix}"

                # Produce user-facing label for the table
                dic_label = f"Days in Care ({dic_kind_label}, {dic_suffix_label})"

                # Extract the data
                days_vals = static_info[dic_attr_name]
                days_care_col = [
                    days_in_care_stats(days_vals)  # first row: population
                ]
                days_care_nums_col = [
                    days_in_care_stats(days_vals, numerical_output=True)  # first row: population
                ]

                for cluster_days in split_by_cluster(clustering_info.labels, days_vals):
                    days_care_col.append(days_in_care_stats(cluster_days))
                    days_care_nums_col.append(days_in_care_stats(cluster_days, numerical_output=True))

                table[dic_label] = days_care_col
                table[dic_label + " (Numerical)"] = days_care_nums_col

        # Extract total number of ICU visits
        def icu_visits_num_stats(icu_visits, numerical_output=False):
            if len(icu_visits) == 0:
                if numerical_output:
                    return np.nan
                else:
                    return "/"

            if numerical_output:
                return np.median(icu_visits)
            else:
                return f"{np.median(icu_visits):0.0f} times ({np.percentile(icu_visits, 25):0.0f}," \
                       f" {np.percentile(icu_visits, 75):0.0f})"

        icu_visits_vals = static_info['FUTURE_icu_visits']
        icu_visits_col = [
            icu_visits_num_stats(icu_visits_vals)  # first row: population
        ]
        icu_visits_nums_col = [
            icu_visits_num_stats(icu_visits_vals, numerical_output=True)  # first row: population
        ]

        for cluster_icu_visits in split_by_cluster(clustering_info.labels, icu_visits_vals):
            icu_visits_col.append(icu_visits_num_stats(cluster_icu_visits))
            icu_visits_nums_col.append(icu_visits_num_stats(cluster_icu_visits, numerical_output=True))

        table['ICU Visits'] = icu_visits_col
        table['ICU Visits Num'] = icu_visits_nums_col

        # ICU Station
        def icu_stations_stats(icu_stations):
            if len(icu_stations) == 0:
                return "/"

            # Unpack all into lists of stations
            icu_stations = [[s] if "__" not in s else s.split("__") for s in icu_stations]

            # Count stations
            c = Counter()
            for stations in icu_stations:
                c.update(stations)

            # Generate a text describing the distribution
            counted_items = sorted(c.items())
            counted_items = [(name, count) if len(name) > 0 else ("Unknown", count) for (name, count) in counted_items]
            total_count = sum(c.values())  # count of stations, not necessarily the admission count!
            text_lines = [f"{name}: {count} ({100 * count / total_count:0.1f}%)" for (name, count) in counted_items]
            text = "\n".join(text_lines)
            return text

        icu_stations_vals = static_info['FUTURE_icu_stations']
        icu_stations_col = [
            icu_stations_stats(icu_stations_vals)  # first row: population
        ]

        for cluster_icu_stations in split_by_cluster(clustering_info.labels, icu_stations_vals):
            icu_stations_col.append(icu_stations_stats(cluster_icu_stations))

        table['ICU Stations'] = icu_stations_col

        # Extract admission diagnosis: List top diagnoses for each cluster.
        # Note: Admissions diagnoses are free-form text fields that clinicians fill out upon a patient's admission,
        # so often, they show the reason for the admission.
        def top_admission_diagnoses(texts, num_entries=5):
            # Sort texts by number of occurrences
            texts, counts = np.unique(texts, return_counts=True)
            sorting = np.argsort(counts)[::-1]
            texts = texts[sorting]
            counts = counts[sorting]

            # Print list of top strings
            diag_lines = []
            for txt, count in zip(texts[:num_entries], counts[:num_entries]):
                diag_lines.append(f"({count}) {txt}")

            return ",\n".join(diag_lines)

        def canonize_admission_diag_text(text):
            return text.strip().replace("  ", " ").replace("//", "/").replace("\\\\", "\\")

        diagnosis_texts = [canonize_admission_diag_text(txt) for txt in static_info['FUTURE_admission_diagnosis']]
        admission_diagnosis_col = [
            top_admission_diagnoses(diagnosis_texts)  # first row: population
        ]

        for cluster_diag_texts in split_by_cluster(clustering_info.labels, diagnosis_texts):
            admission_diagnosis_col.append(top_admission_diagnoses(cluster_diag_texts))

        table['Admission Diagnosis'] = admission_diagnosis_col

        # Parentage of clusters (only available if the clustering is not the most coarse clustering)
        for parent_idx, parent_clustering_id in enumerate(clustering_info.parent_clusterings):
            parent_labels_dist = clustering_info.parent_labels[parent_clustering_id]  # keys: our labels,
            # values: parent clustering label distribution

            # List the distributions in textual form
            parent_clus_col = [
                "/"  # population has no parentage information
            ]
            for label in np.unique(clustering_info.labels):
                parent_fracs = sorted([(p_label, frac) for (p_label, frac) in parent_labels_dist[label].items()],
                                      key=lambda x: x[-1], reverse=True)  # Sort by size of fraction, the largest first
                parent_fracs = ", ".join([f"{io.label_for_cluster_label(p_label)}: {100 * frac:0.0f}%"
                                          for (p_label, frac) in parent_fracs])  # e.g. "Cluster 5: 80%, Cluster 3: 20%"
                parent_clus_col.append(parent_fracs)
            table[f'Parentage (level {parent_idx + 1}, {parent_clustering_id})'] = parent_clus_col

        # (Since table is changed in-place, there is no need to return it)

    def load_clusterings_from_disk(self):
        # Check if clusterings file exists
        if not os.path.isfile(self._clustering_file_path):
            return

        # Load the saved clusterings
        clusterings = io.read_pickle(self._clustering_file_path)
        self.clusterings = clusterings['clusterings']
        self.clusterings_robust = clusterings['clusterings_robust']
        self.clusterings_all = clusterings['clusterings_all']

        logging.info(f"Loaded clusterings from {self._clustering_file_path}")

    def save_clusterings_to_disk(self):
        """
        Save clusterings to disk
        :return:
        """

        # Save all clusterings using a dictionary
        clusterings = {
            'clusterings': self.clusterings,
            'clusterings_robust': self.clusterings_robust,
            'clusterings_all': self.clusterings_all
        }
        io.write_pickle(clusterings, self._clustering_file_path)

        logging.info(f"Saved clusterings to {self._clustering_file_path}")

    def _clean_clusters(self, clusterings_full: ClusteringCollection, bootstrappings: List[ClusteringCollection]):
        clusterings_full_cleaned = []
        clusterings_matched = []  # type: List[Dict[int, ClusteringInfo]]
        for clus_idx, clus in enumerate(clusterings_full.clusterings):

            # Find matching clustering in each bootstrapping
            boot_clusterings_matched = {}  # type: Dict[int, ClusteringInfo]
            for boot_idx, boot in enumerate(bootstrappings):
                for boot_clus in boot.clusterings:
                    if boot_clus.options == clus.options and boot_clus.algorithm == clus.algorithm:
                        boot_clusterings_matched[boot_idx] = boot_clus

            # Skip this clustering if match was not found for every bootstrapping
            if len(boot_clusterings_matched) < len(bootstrappings):
                continue

            # Skip this clustering if it or any of its matches are trivial
            all_clus_versions = [clus] + list(boot_clusterings_matched.values())
            any_clusterings_trivial = any([self._is_trivial(c) for c in all_clus_versions])
            if any_clusterings_trivial:
                continue

            # Skip this clustering if it or any of its matches are pseudo-trivial
            any_clusterings_trivial = any([self._is_pseudo_trivial(c) for c in all_clus_versions])
            if any_clusterings_trivial:
                continue

            # Mark this clustering "cleaned"
            clusterings_full_cleaned.append(clus)

            # Remember matched clusterings
            clusterings_matched.append(boot_clusterings_matched)

        # Set cleaned clusterings as only clusterings
        clusterings_cleaned = ClusteringCollection(
            clusterings=clusterings_full_cleaned,
            sampled_indices=clusterings_full.sampled_indices
        )

        return clusterings_cleaned

    def _filter_to_robust_clusterings(self, clusterings: ClusteringCollection) -> ClusteringCollection:
        # Find out robustness for every clustering
        for clustering_info in clusterings.clusterings:
            for robustness_name, robustness_func in robustness_functions.items():

                # Abort further checks if clustering not robust w.r.t. a different robustness measure
                if not all([rob.is_robust for rob in clustering_info.robustness.values()]):
                    continue

                # Check if the measurements of robustness for this clustering meet the minimum threshold
                min_robustness = self._determine_robustness_threshold(robustness_func, clustering_info)
                lower_bound_measured_robustness = np.percentile(
                    [r[robustness_name] for r in clustering_info.robustness_measurements],
                    10
                )

                # Write into ClusteringInfo
                robustness_info = RobustnessInfo(
                    rob_name=robustness_name,
                    rob_threshold=min_robustness,
                    rob_actual=lower_bound_measured_robustness,
                    is_robust=lower_bound_measured_robustness > min_robustness
                )
                clustering_info.robustness[robustness_name] = robustness_info

        # Filter out non-robust clusterings
        robust_clusterings = [clustering_info for clustering_info in clusterings.clusterings
                              if all([rob.is_robust for rob in clustering_info.robustness.values()])]

        # Mark as robust
        for clustering_info in robust_clusterings:
            clustering_info.is_robust = True

        return ClusteringCollection(
            clusterings=robust_clusterings,
            sampled_indices=clusterings.sampled_indices
        )

    def _cluster_complete(self, features) -> Tuple[ClusteringCollection, ClusteringCollection, ClusteringCollection]:
        # The first step is to cluster on the full data once
        clusterings_full = ClusteringCollection(
            clusterings=self._perform_all_clusterings(features=features),
            sampled_indices=np.arange(len(features))  # indices always w.r.t. features array
        )

        # Bootstrap from features on cluster on the sampled subsets
        bootstrappings = self._bootstrap(features_full=features)

        # Match up full-data-clusterings with their bootstrapped counterparts
        # (essentially, this step removes clusterings for which no match can be found and ensures the order of the
        # clusterings within each ClusteringCollection is the same)
        clusterings_full = self._clean_clusters(
            clusterings_full=clusterings_full,
            bootstrappings=bootstrappings
        )

        # Determine number of clusters and effective number of clusters for each clustering
        for clusterings_info in clusterings_full.clusterings:
            clusterings_info.num_clus = len(np.unique(clusterings_info.labels))
            clusterings_info.num_clus_effective = self._num_clus_effective(clusterings_info)

        # Check pairwise clustering similarity between each clustering on the full data and its bootstrapped
        # counterparts
        for boot_idx, boot in enumerate(bootstrappings):
            logging.info(f"Checking clustering similarity for bootstrapping {boot_idx + 1} of {len(bootstrappings)}")
            self._measure_pairwise_clustering_robustness(clusterings_full, boot)

        # Filter out non-robust clusterings
        clusterings_robust = self._filter_to_robust_clusterings(clusterings=clusterings_full)
        logging.info(f"{len(clusterings_robust.clusterings)} of {len(clusterings_full.clusterings)} clusterings "
                     f"were deemed robust.")

        # Pick interesting clusters (which are different from one another)
        clusterings_interesting = self._pick_interesting_clusterings(
            robust_clusterings=clusterings_robust,
            features=features
        )

        return clusterings_interesting, clusterings_robust, clusterings_full

    def _pick_interesting_clusterings(self, robust_clusterings: ClusteringCollection,
                                      features: np.ndarray) -> ClusteringCollection:
        # Abort if there are no robust clusterings to be picked
        if len(robust_clusterings.clusterings) == 0:
            return robust_clusterings

        # Assess technical quality of the clusterings
        for clustering_info in robust_clusterings.clusterings:
            clustering_info.technicals = self._assess_technical_quality(
                features=features,
                clustering_info=clustering_info
            )

        # Sort by different cluster measures and select clusterings that seem interesting based on those
        num_picked_by_quality = int(np.ceil(self.frac_clus_picked_per_measure * len(robust_clusterings.clusterings)))
        for rob_name in robustness_functions.keys():

            # Sort clusters by their performance in the robustness measure
            robust_clusterings.clusterings.sort(
                key=lambda ci: ci.robustness[rob_name].rob_actual,
                reverse=True
            )

            # Mark top performers as interesting
            for clustering_info in robust_clusterings.clusterings[:num_picked_by_quality]:
                clustering_info.is_interesting = True

        # Pick clusterings based on technical quality criteria (sort by average technical quality score)
        num_technicals = len(robust_clusterings.clusterings[0].technicals)
        for technical_idx in range(num_technicals):
            robust_clusterings.clusterings.sort(
                key=lambda ci: ci.technicals[technical_idx].score,
                reverse=True
            )

            # Mark top performers as interesting
            for clustering_info in robust_clusterings.clusterings[:num_picked_by_quality]:
                clustering_info.is_interesting = True

        # Mark clusterings with the smallest cluster count as interesting
        min_cluster_count = min([ci.num_clus for ci in robust_clusterings.clusterings])
        for clustering_info in robust_clusterings.clusterings:
            if clustering_info.num_clus == min_cluster_count:
                clustering_info.is_interesting = True

        # Remove clusterings that are duplicates w.r.t. the number of clusters
        interesting_clusterings = [ci for ci in robust_clusterings.clusterings if ci.is_interesting]
        interesting_clusterings.sort(key=lambda ci: abs(ci.num_clus - ci.num_clus_effective))  # sort by how "honest"
        # the clustering is, i.e. how small the difference between cluster count and effective cluster count is
        _, uniq_idcs = np.unique([ci.num_clus for ci in interesting_clusterings], return_index=True)
        interesting_clusterings = [interesting_clusterings[idx] for idx in uniq_idcs]

        # Keep only a limited number of clusterings (this is necessary to speed up statistical testing, which takes a
        # lot of time for each clustering)
        if len(interesting_clusterings) > self.num_clusterings_picked_total:
            interesting_clusterings.sort(key=lambda ci: ci.num_clus)
            chosen_indices = self._pick_interesting_clusters_by_cluster_count(
                counts=[ci.num_clus for ci in interesting_clusterings]
            )
            interesting_clusterings = [interesting_clusterings[idx] for idx in chosen_indices]

        # Return interesting clusterings
        logging.info(f"{len(interesting_clusterings)} of {len(robust_clusterings.clusterings)} robust clusterings"
                     f" were deemed interesting")
        return ClusteringCollection(
            clusterings=interesting_clusterings,
            sampled_indices=robust_clusterings.sampled_indices
        )

    def _pick_interesting_clusters_by_cluster_count(self, counts) -> List[int]:
        """
        Tries to pick out clusterings with a variety of cluster counts, assuming the cluster counts are in roughly
         equal intervals from min to max (no sorting assumed)

        :param counts: List of cluster counts for different clusterings
        :return: List of indices of picked clusterings
        """

        # Determine indices that are log-evenly distributed
        indices = np.unique(
            np.round(
                np.logspace(0, np.log10(len(counts) - 1), self.num_clusterings_picked_total)[1:-1]
            ).astype(np.int)
        )

        # Add first and last index (smallest and largest cluster count)
        indices = sorted(set(list(indices) + [0, len(counts) - 1]))

        # Indices are w.r.t. a sorted count list: Convert those back to the original order of the count list
        sorting = np.argsort(counts)
        sorting_reverse = {dst_index: src_index for (dst_index, src_index) in enumerate(sorting)}
        indices = [sorting_reverse[idx] for idx in indices]

        return indices

    @staticmethod
    def _assess_technical_quality(features, clustering_info: ClusteringInfo) -> List[TechnicalQuality]:
        # Silhouette coefficient
        try:
            silhouette = silhouette_score(X=features, labels=clustering_info.labels)
        except ValueError:
            logging.error(f"Silhouette score could not be calculated: "
                          f"num_labels = {len(np.unique(clustering_info.labels))},"
                          f" n_samples = {len(clustering_info.labels)}")
            silhouette = -1  # Worst possible silhouette score
        silhouette_q = TechnicalQuality("Silhouette Coefficient", silhouette)

        # Calinski Harabasz: within-cluster dispersion vs. between-cluster dispersion
        calinski = calinski_harabasz_score(X=features, labels=clustering_info.labels)  # higher is better
        calinski_score = TechnicalQuality("Calinski Harabasz Score", calinski)

        # Davies Bouldin score: Average similarity of clusters
        davies = davies_bouldin_score(X=features, labels=clustering_info.labels)  # closer to zero is better
        davies_score = TechnicalQuality("Davies-Bouldin Similarity", -davies)  # negative score so higher = better

        return [
            silhouette_q,
            calinski_score,
            davies_score
        ]

    @staticmethod
    def index_multi(containing_arr: np.ndarray, values_to_be_indexed: np.ndarray) -> List:
        indices = []
        for v in values_to_be_indexed:
            v_idx = np.where(containing_arr == v)[0].item()
            indices.append(v_idx)
        return indices

    def _num_clus_effective(self, clustering_info: ClusteringInfo) -> float:
        return float(np.power(np.e, self._label_entropy(clustering_info.labels)))

    @staticmethod
    def _label_entropy(labels: np.ndarray) -> float:
        _, labels_counts = np.unique(labels, return_counts=True)
        distribution = [count / len(labels) for count in labels_counts]
        entropy = float(scipy.stats.entropy(pk=distribution))
        return entropy

    @staticmethod
    def _measure_pairwise_clustering_robustness(full: ClusteringCollection, bootstrapped: ClusteringCollection):
        """
        Measures clustering robustness between each clustering of the full data clusterings and clusterings for one
        bootstrapping

        :param full:
        :param bootstrapped:
        :return:
        """

        # Get intersection of sampled points
        _, idcs_full, idcs_boot = np.intersect1d(
            full.sampled_indices,
            bootstrapped.sampled_indices,
            assume_unique=True,
            return_indices=True
        )

        # Calculate similarity for each of the clusterings
        for clustering_idx, (clus_full, clus_boot) in enumerate(zip(full.clusterings, bootstrapped.clusterings)):
            # Restrict labels to points belonging to the intersection between the bootstrappings
            labels_full = clus_full.labels[idcs_full]
            labels_boot = clus_boot.labels[idcs_boot]

            # Compute scores for each of the robustness metrics
            robustness = {}
            for robustness_fun_name, robustness_fun in robustness_functions.items():
                robustness[robustness_fun_name] = robustness_fun(
                    clus_1_labels=labels_full,
                    clus_2_labels=labels_boot
                )

            # Save the robustness info we have just measured for the full clustering
            clus_full.robustness_measurements.append(robustness)

    def _perform_all_clusterings(self, features: np.ndarray) -> List[ClusteringInfo]:
        clusterings = []

        # Determine number of clusters for methods that need it as input
        max_k = min(40, len(features) // 2)
        k_range = range(2, max_k + 1)

        # Cluster based on k-Medoids
        for k in k_range:
            clusterings.append(self._cluster_kmedoids(
                features=features,
                k=k
            ))

        # Cluster hierarchically
        clusterings += self._cluster_hierarchically(features=features, n_clusters=list(k_range))

        # Cluster based on k-means
        # for k in k_range:
        #     clusterings.append(self._cluster_kmeans(
        #         features=features,
        #         k=k
        #     ))

        # Cluster based on DBSCAN
        # clusterings += self._perform_all_dbscan(features=features)

        return clusterings

    def _perform_all_dbscan(self, features: np.ndarray) -> List[ClusteringInfo]:
        clusterings = []

        pairwise_distances = squareform(pdist(features))
        eps_space = np.linspace(
            np.percentile(pairwise_distances, 10),
            np.percentile(pairwise_distances, 90),
            100
        )
        eps_space = eps_space[eps_space > 0]
        for eps, min_samples in product(
                eps_space,
                np.unique(np.round(np.linspace(
                    1,
                    min(200, np.ceil(len(features) / 10)),
                    50
                )))
        ):
            clusterings.append(self._cluster_dbscan(
                features=features,
                eps=eps,
                min_samples=min_samples
            ))

        return clusterings

    @staticmethod
    def _cluster_dbscan(features: np.ndarray, eps: float, min_samples: float) -> ClusteringInfo:
        # Cluster
        logging.info(f"Clustering {features.shape[0]} features with DBSCAN (eps={eps}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(features)

        # Generate clustering
        return ClusteringInfo(
            algorithm='dbscan',
            options={
                'eps': eps,
                'min_samples': min_samples
            },
            labels=dbscan.labels_
        )

    @staticmethod
    def _cluster_kmeans(features: np.ndarray, k: int) -> ClusteringInfo:
        # Cluster
        logging.info(f"Clustering {features.shape[0]} features with k-Means (k={k})...")
        kmeans = KMeans(n_clusters=k).fit(features)

        # Generate clustering
        return ClusteringInfo(
            algorithm='kmeans',
            options={
                'k': k
            },
            labels=1 + kmeans.labels_
        )

    @staticmethod
    def _cluster_kmedoids(features: np.ndarray, k: int) -> ClusteringInfo:
        # Cluster
        logging.info(f"Clustering {features.shape[0]} features with k-Medoids (k={k})...")
        kmedoids = KMedoids(n_clusters=k, init='k-medoids++').fit(features)

        # Generate clustering
        return ClusteringInfo(
            algorithm='kmedoids',
            options={
                'k': k
            },
            labels=1 + kmedoids.labels_
        )

    @staticmethod
    def _cluster_hierarchically(features, n_clusters: List[int], linkage_type='centroid') -> List[ClusteringInfo]:
        logging.info(f"Clustering {features.shape[0]} features with hierarchical clustering ({linkage_type})...")

        # Measure all pairwise distances
        dists = pdist(features, 'euclidean')

        # Determine point linkage
        linkage = hierarchy.linkage(dists, linkage_type)

        # Cut linkage tree at the requested positions to quantize the hierarchical clustering
        cut_tree = hierarchy.cut_tree(linkage, n_clusters=n_clusters)

        # Save the clusterings
        clustering_algo = f"hierarchical_{linkage_type}"
        clusterings = []
        num_clusters_observed = []
        for col_idx in range(cut_tree.shape[1]):
            cluster_labels = cut_tree[:, col_idx]

            # Skip clustering if its number of clusters is too small or already observed
            labels_uniq = np.unique(cluster_labels)
            num_clus = len(labels_uniq)
            if num_clus < 2 or num_clus in num_clusters_observed:
                continue
            num_clusters_observed.append(num_clus)

            # Normalize labels to be represented by the smallest possible integers
            cluster_labels_copy = np.copy(cluster_labels)
            labels_norm = np.arange(len(labels_uniq))
            for lab_orig, lab_norm in zip(labels_uniq, labels_norm):
                cluster_labels_copy[cluster_labels == lab_orig] = lab_norm
            cluster_labels = cluster_labels_copy

            clusterings.append(
                ClusteringInfo(
                    algorithm=clustering_algo,
                    options={
                        'k': num_clus
                    },
                    labels=1 + cluster_labels
                )
            )

        return clusterings
