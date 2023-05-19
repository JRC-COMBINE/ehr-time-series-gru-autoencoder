# Novel architecture for gated recurrent unit autoencoder trained on time series from electronic health records enables detection of ICU patient subgroups

_Read our paper at https://www.nature.com/articles/s41598-023-30986-1_

## Installation
1. Clone this repository:
```bash
git clone git@github.com:JRC-COMBINE/ehr-time-series-gru-autoencoder.git
```
2. Enter the repository's root directory.
3. Using the package manager [pip](https://pip.pypa.io/en/stable/), install the requirements:
```bash
pip install -r requirements.txt
```
(this has been tested with Python 3.7.16)

4. Get access to [MIMIC-III](https://mimic.mit.edu/docs/gettingstarted/).
5. Set the environment variable `MIMIC_URL` to the URL of a MIMIC-III database (local or remote).
6. (Optional) If you want to use the
[HCUP Chronic Condition Indicator](https://www.hcup-us.ahrq.gov/toolssoftware/chronic/chronic.jsp), download the CSV
file into the correct directory:
```bash
cd info
wget https://www.hcup-us.ahrq.gov/toolssoftware/chronic/cci2015.csv
cd ..
```
and change `use_cci = False` to `use_cci = True` in `info/IcdInfo.py`.

7. Done. Try training your model!


## Usage
All commands should be run in the project's root directory unless otherwise specified.

Display the command line help using:
```bash
python full_pipeline_mimic.py --help
```

### Demo on Generic Files

A full example of using the software for feature extraction on generic CSV files with time series is available in `generic_dataframe_demo.py`.

### Full Pipeline (Training, Clustering, Evaluation) on MIMIC

Using default settings:
```bash
python full_pipeline_mimic.py
```

The script allows modifying the settings using command line arguments, e.g. for running with 20000 admissions and
training for at most 25 epochs:
```bash
python full_pipeline_mimic.py --admissions 20000 --max_epochs 25
```

Preprocessing, training or model hyperparameters can be set directly when executing on the command
line using the `prep`, `training`, and `model` prefixes followed by the name of the hyperparameter, e.g.:
```bash
python full_pipeline_mimic.py --prep_scaling_mode standard --training_batch_size 12 --model_rnn_size 150
```

Using the script `run_full_published.sh`, you can run the pipeline with the same settings as used in the publication.

### Random Architecture Search
Search can be run either locally or on a [SLURM](https://slurm.schedmd.com/documentation.html)-based compute cluster.
The search script needs to be run from its own directory:
```bash
cd search
```

Using arguments, one can define a common prefix for search runs, the number of runs, and the number of admissions
to train and evaluate on:
```bash
python cluster_search.py --bout_id my_random_search --admissions 10000 --runs 50
```

For submitting the resulting SLURM job under a different account name, set the environment variable
`SLURM_ACCOUNT_DL` to the desired account name.

The search can also be run sequentially on a single computer by setting the `--interactive` switch.


## Citation
You can access the paper for this repository at https://www.nature.com/articles/s41598-023-30986-1.
```
Merkelbach, K., Schaper, S., Diedrich, C. et al. Novel architecture for gated recurrent unit autoencoder trained on time series from electronic health records enables detection of ICU patient subgroups. Sci Rep 13, 4053 (2023). https://doi.org/10.1038/s41598-023-30986-1
```
**DOI** https://doi.org/10.1038/s41598-023-30986-1


## Credit
- **Paper**: Kilian Merkelbach, Steffen Schaper, Christian Diedrich, Sebastian Johannes Fritsch, Andreas Schuppert
- **Training, Model, Clustering, Evaluation, Search**: Kilian Merkelbach
- **Data Extraction Pipeline**: Richard Polzin, Konstantin Sharafutdinov, Jayesh Bhat, Kilian Merkelbach
