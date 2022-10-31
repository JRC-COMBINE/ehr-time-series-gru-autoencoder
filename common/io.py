import os
import pickle
import json
import csv
import logging
from pathlib import Path

# Multiprocessing
from multiprocessing import cpu_count


# Names of folders
clusterings_name = "clusterings"
model_snapshots_name = "model_snapshots"


# Names of data splits
split_name_all = "all"
split_name_val = "validation"
split_name_train = "training"

split_name_unit_tests = "test_split"


# Static functions
def write_txt(txt, path, verbose=False):
    if verbose:
        logging.info(f"Writing text file to '{path}'")
    with open(path, "w") as file:
        file.write(txt)


def write_csv(table, path, verbose=False):
    if verbose:
        logging.info(f"Writing CSV file to '{path}'")
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(table)


def write_pickle(obj, path, verbose=False):
    if verbose:
        logging.info(f"Writing pickle to '{path}'")
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def read_pickle(path, verbose=False):
    if verbose:
        logging.info(f"Reading pickle from '{path}'")
    if not os.path.isfile(path):
        if verbose:
            logging.info("Path does not exist!")
        return None
    with open(path, "rb") as file:
        return pickle.load(file)


def makedirs(path):
    """
    Creates the specified directories, including parent directories.

    :param path:
    :return:
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def write_json(obj, path, verbose=False, pretty=False):
    if verbose:
        logging.info(f"Writing JSON to '{path}'")
    with open(path, "w") as file:
        kwargs = {}
        if pretty:
            kwargs = dict(sort_keys=True, indent=4, separators=(',', ': '))
        json.dump(obj, file, **kwargs)


def read_json(path, verbose=False):
    if verbose:
        logging.info(f"Reading JSON from '{path}'")
    if not os.path.isfile(path):
        return None
    with open(path, "r") as file:
        return json.load(file)


def sanitize(st):
    st = st.replace(":", "_")
    st = st.replace("/", "_")
    st = st.replace(" ", "_")
    return st


def prettify(underscored_str: str):
    # Replace _ with spaces
    p = underscored_str.replace("_", " ")

    # Capitalize every new word
    p_tokens = p.split(" ")
    p_tokens_cap = [pt.capitalize() for pt in p_tokens]

    # Put string back together
    p = " ".join(p_tokens_cap)

    return p


def label_for_cluster_label(cluster_label):
    if cluster_label != -1:
        return f"Cluster {cluster_label}"
    else:
        return "Noise"


def str_to_bool(str_var):
    if type(str_var) == bool:
        return str_var
    else:
        return str_var.lower() == "true"


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class IOFunctions:
    """
    Provides IO functionality like file saving and reading.
    """

    def __init__(self, dataset_key, training_id="anon", treat_as_search_result=True):
        # Define base storage path
        self.base_storage = Path(os.getenv("HOME"), "datasets")

        # Dataset key: allows identifying the dataset
        self.dataset_key = dataset_key
        self._dataset_dir = Path(self.base_storage, dataset_key)
        makedirs(self._dataset_dir)

        # Results directory - depends on training ID
        self._results_dir = Path(self._dataset_dir)
        if treat_as_search_result:
            self._results_dir = Path(self._results_dir, "search_results", f"{training_id}")
        makedirs(self._results_dir)

        # Plots directory
        self._plots_dir = Path(self._results_dir, "plots")
        makedirs(self._plots_dir)

        # Eval directory
        self._eval_dir = Path(self._results_dir, "eval")
        makedirs(self._eval_dir)

        # Model checkpoints directory
        self._models_dir = Path(self._results_dir, model_snapshots_name)
        makedirs(self._models_dir)

        # Clusterings directory
        self._clusterings_dir = Path(self._results_dir, clusterings_name)
        makedirs(self._clusterings_dir)

        # Multiprocessing
        slurm_cpu_restriction = int(os.getenv("SLURM_CPUS_PER_TASK", 8))
        self._num_cpus = min(cpu_count(), slurm_cpu_restriction)

    def get_cpu_count(self):
        return self._num_cpus

    def get_named_dir(self, name, in_results=False):
        """
        Creates (if not already existing) and returns path to named directory
        :param name:
        :param in_results:
        :return:
        """
        if in_results:
            base_path = self.get_results_dir()
        else:
            base_path = self._dataset_dir
        named_dir = Path(base_path, name)
        makedirs(named_dir)
        return named_dir

    def get_results_dir(self):
        """
        :return: results directory path (depends on training ID)
        """
        return self._results_dir

    def get_dataset_dir(self):
        """
        :return: dataset directory path
        """
        return self._dataset_dir

    def get_eval_dir(self):
        """
        :return: eval directory path
        """
        return self._eval_dir

    def get_clusterings_dir(self):
        """
        :return: clusterings directory path
        """
        return self._clusterings_dir

    def get_plots_dir(self):
        """
        :return: directory where plots are stored
        """
        return self._plots_dir

    def get_models_dir(self):
        """
        :return: directory where model snapshots are stored
        """
        return self._models_dir

    def get_dyn_data_cols_path(self):
        return os.path.join(self.get_models_dir(), "dyn_data_cols.json")
