"""
DLCLUS Demo (for foreign data)

"""

import logging
import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path

from ai.training import Training
from common import io
from data.preprocessor import Preprocessor


# Show time along with log messages
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Get data (CSV files)
csv_dir = "/set/this/path/to/where_your_files_are"  # TODO: Change this path
paths = glob.glob(os.path.join(csv_dir, "*.csv"))
print(f"Found {len(paths)} CSV files at {csv_dir}")

# For testing, limit number of files
num_files = 50  # TODO: After you are done with testing, set this to False to include all files
np.random.seed(1024)  # fix seed to make sampling repeatable for testing
if num_files:
    paths = [paths[idx] for idx in np.random.choice(np.arange(len(paths)), num_files, replace=False)]
    print(f"Randomly sampled {len(paths)} files for testing")

# Read CSV files
dfs = [pd.read_csv(f) for f in paths]
print(f"Loaded {sum([len(df) for df in dfs])} rows of data from {len(dfs)} files.")

# Get IO manager
training_id = "your_training_id"  # TODO: Change to identify your dataset
iom = io.IOFunctions(
    dataset_key=f"{training_id}_dataset",
    training_id=training_id
)

# IDs for admissions are going to be their indices
encounter_ids = list(range(len(dfs)))


class PandasExtractor:
    def __init__(self, dataframes):
        self._dfs = dataframes

    def load_dynamic_data(self, enc_id):
        return self._dfs[enc_id]

    def path_for_static_data(self, enc_id):
        return Path("DummyPathDoesntExist")

    def label_for_item_id(self, item_id):
        return item_id


# Init preprocessor for the data
prep = Preprocessor(
    iom=iom,
    extractor_exams=PandasExtractor(dataframes=dfs),
    extractor_drugs=None,
    encounter_ids=encounter_ids,
    ignore_static_data=True,
    time_column='change_this_to_the_name_of_the_column_with_time'  # strictly needed to interpret the data: Column
    # that tells the time for each row
    # TODO: Change the `time_column` value to the actual name of the column in your data that contains the time as a
    #  floating point number (e.g. in minutes, individually for each sample)
)

# Train a model
trainer = Training(
    iom=iom,
    prep=prep,
    additional_model_args={'bottleneck_size': 46}  # this sets the number of feature dimensions (for MIMIC, 46 seems
    # to be good). Odd numbers of feature dimensions are rounded up to even numbers, e.g. 3 results in 4.
    # TODO: Change `bottleneck_size` if desired
)
trainer.pipeline_args = {'id': training_id}
trainer.train_or_load()

# Calculate features
features = trainer.compute_features("all")

features_n_samples, features_n_dims = features.shape
print(f"features.shape = ({features_n_samples}, {features_n_dims}) = (n_samples, n_feature_dims)")

# Save features as a numpy file
features_path = os.path.join(iom.get_results_dir(), "features.npy")
np.save(
    features_path,
    features,
    allow_pickle=False
)
print(f"Saved features at {features_path}")

# To load features in another script, simply use np.load
print(f"Load saved features file in another script using `np.load(\"{features_path}\")`")

print("Done!")
