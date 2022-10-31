# Logging
import argparse
import itertools
import json
import logging
import os

# Utility
import random
import sys
import tempfile
import hashlib
from collections import OrderedDict, Counter, defaultdict
from glob import glob
import copy
from datetime import datetime
from tqdm import tqdm

# Math and data
import numpy as np
import pandas as pd

# NN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from tensorflow.keras import backend as keras_backend

# Search
import skopt

# Text
from pyfiglet import Figlet

# My own modules
sys.path.append("..")
from evaluation import plot
from common import io


# Show time along with log messages
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Settings
results_dir = os.path.join(
    os.getenv("HOME"),
    "datasets",
    "mimic",
    "search_results"
)
observations_path = os.path.join(results_dir, "observations.json")
search_name_prefix = "srch_"
no_data_loss = 10e6
arg_not_used_str = 'NotSet'


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster search")
    parser.add_argument("--use_gpu", action='store_true',
                        help="if this flag is set, execute on GPU nodes rather than regular CPU nodes")
    parser.add_argument("--runs", type=int, help="number of runs in the search",
                        default=50)
    parser.add_argument("--tasks_concurrent", type=int, help="number of tasks to run at the same time",
                        default=25)
    parser.add_argument("--admissions", type=int, help="number of admissions",
                        default=1000)
    parser.add_argument("--starting_offset_hours", type=int, help="hours to wait before executing the jobs",
                        default=0)
    parser.add_argument("--early_stopping_patience", type=int, help="early stopping patience",
                        default=8)
    parser.add_argument("--experiment", type=str, nargs='+', help="list of search parameters to vary in an experiment."
                                                                  " All other parameters will be kept fixed at their"
                                                                  " default value.")
    parser.add_argument("--no_shuffle", action='store_true',
                        help="do not randomly shuffle admissions before processing")
    parser.add_argument("--no_plot", action='store_true',
                        help="do not plot")
    parser.add_argument("--no_clustering", action='store_true',
                        help="do not cluster (and thus, do not evaluate clusterings)")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--analyze", action='store_true',
                            help="if this flag is set, don't run search but instead analyze existing results")
    mode_group.add_argument("--bout_id", type=str, help="ID of this bout of search")
    parser.add_argument("--index", type=int,
                        help="ignore number of runs entirely and start single search run with the given index. "
                             "Useful for restarting specific runs of a larger search.")
    parser.add_argument("--include_drugs", action='store_true',
                        help="enable inclusion of drugs into dynamic data that the model receives (default is "
                             "EXCLUDING drugs)")
    parser.add_argument("--include_neonates", action='store_true',
                        help="enable inclusion of neonates (default is EXCLUDING them)")
    parser.add_argument("--interactive", action='store_true',
                        help="If set, execute commands directly (without starting jobs for them)")
    parser.add_argument("--no_training", action='store_true',
                        help="If set, don't train models but instead only evaluate them on validation data (useful "
                             "for finding model architectures prone to perform well)")
    parser.add_argument("--random", action='store_true',
                        help="only propose fully random options (and do not train options model)")
    args = parser.parse_args()
    return args


def main_func():
    # Parse arguments
    args = parse_args()

    if not args.analyze:
        # Run mode: Start new search jobs on the cluster
        run_new_search(
            num_runs=args.runs,
            num_tasks_concurrent=args.tasks_concurrent,
            num_admissions=args.admissions,
            starting_offset_hours=args.starting_offset_hours,
            experiment=args.experiment,
            random_only=args.random,
            early_stopping_patience=args.early_stopping_patience,
            no_shuffle=args.no_shuffle,
            bout_id=args.bout_id,
            include_drugs=args.include_drugs,
            include_neonates=args.include_neonates,
            no_plot=args.no_plot,
            no_clustering=args.no_clustering,
            use_gpus=args.use_gpu,
            index=args.index,
            interactive=args.interactive,
            no_training=args.no_training
        )
    else:
        # Analysis mode: Analyze existing results
        analyze_results()


def print_large(text):
    figlet = Figlet(font='slant', width=120)
    text = " ".join([c for c in text])
    print(figlet.renderText(text))


def run_new_search(num_runs, num_tasks_concurrent, num_admissions, starting_offset_hours, experiment, random_only,
                   early_stopping_patience, no_shuffle, bout_id, include_drugs, include_neonates, no_plot=False,
                   no_clustering=False, use_gpus=False, index=None, interactive=False, no_training=False):
    # Greet user
    print_large("Search")

    logging.info(f"Starting new search: Array job with {num_runs} runs.")

    # Set seed using search bout id
    new_random_seed = random.randint(1, 2 ** 14)
    bout_id_seed = int.from_bytes(bout_id.encode(), 'little') % 10 ** 6
    random.seed(bout_id_seed)
    np.random.seed(bout_id_seed)

    # Define search space
    search_space = {
        # Model parameters
        'model_num_dec_dense_layers': skopt.space.Integer(1, 2),
        'model_rnn_size': [158],
        'model_rnn_layers': [1],
        'model_rnn_type': ['gru'],
        'model_bottleneck_size': [46],
        'model_dropout_rate': skopt.space.Real(0.025, 0.3),
        'model_activation': ['relu', 'elu'],
        'model_temporal_pool_mode': ['average'],
        'model_bidirectional_merge_mode': ['sum'],  # 'mul' not good
        'model_normalization_type': ['layer'],  # disabling normalization layers almost never leads to
        # convergence
        'model_input_noise_sigma': skopt.space.Real(0.0, 0.0625),
        'model_post_bottleneck_noise_sigma': skopt.space.Real(0.0, 0.0625),
        'model_reconstruct_times': [True],  # Reconstructing times has shown to improve reconstruction performance A LOT
        # in experiment
        'model_times_to_encoder': [True],  # Supplying times to encoder improves reconstruction performance
        # (shown in experiment)
        'model_l2_regularization': [0.0],

        # Training parameters
        'training_batch_size': skopt.space.Integer(2, 8, prior='log-uniform'),
        'training_max_batch_len': [0],  # 0 means no cutting takes place - so far, 0 seems like the best value
        'training_loss': ['huber_loss'],  # 'mae' and 'mean_absolute_percentage_error' not good;
        # 'huber_loss' and 'mse' promising
        'training_optimizer': ['Adam'],
        'training_learning_rate': [0.00075],
        'training_clip_grad_norm': [True, False],
        'training_shuffle_admissions': [False],

        # Preprocessing parameters
        'prep_scaling_mode': ['quantile'],  # no 'minmax'; it seems to lead to bad convergence ('power'
        # is also less stable than 'standard')
        'prep_positional_encoding_dims': [64]  # larger sizes might work even better
    }

    # Notify user of drugs policy
    special_messages = []
    if include_drugs:
        special_messages.append("INCLUDING DRUGS dosages into dynamic data!")

    # Notify user about neonate policy
    if include_neonates:
        special_messages.append("INCLUDING NEONATES!")

    # Notify user of training policy
    if no_training:
        search_space['training_no_training'] = [True]
        special_messages.append("NOT TRAINING, only testing untrained models on validation data")

    # Experiment: Check if experiment is requested and which parameters to vary (and which to keep fixed)
    experiment_mode = experiment is not None
    if experiment_mode:
        print_large("Lab")

        # Check if parameters requested for experiment are actually in the search space
        for varied_param in experiment:
            assert varied_param in search_space, f"Experiment parameter '{varied_param}' is not in search space!"

        # Inform user about experiment
        logging.info(f"Experiment: Varying {len(experiment)} parameters.")
        for param_name in search_space.keys():
            if param_name in experiment:
                logging.info(f"Experiment: Varying parameter '{param_name}'")
            else:
                logging.info(f"Experiment: FIXING parameter  '{param_name}'")

        # Restrict search space to varied parameters
        search_space = {param_name: search_space[param_name] for param_name in experiment}

    # Sample from search space (using randomness and a model to find promising options)
    experiment_mode = experiment_mode or random_only
    opts = explore_search_space(search_space, num_runs, experiment_mode=experiment_mode)

    # Restore a *random* seed
    np.random.seed(new_random_seed)

    # Create file in which commands to be called are saved (and later read by the array jobs)
    command_file_path = write_commands_to_file(
        all_options=opts,
        bout_id=bout_id,
        num_admissions=num_admissions,
        early_stopping_patience=early_stopping_patience,
        no_shuffle=no_shuffle,
        no_plot=no_plot,
        no_clustering=no_clustering,
        include_drugs=include_drugs,
        include_neonates=include_neonates,
        interactive=interactive
    )

    # In interactive mode, we are done, now
    if interactive:
        return

    # Calculate memory requirement
    total_mem = mem_gb_by_num_admissions(k=num_admissions)
    num_cpus = 16  # Note: look at cluster machine specs using `sinfo -o "%15N %10c %10m  %25f %10G"`
    if num_admissions < 5000:
        num_cpus = 8
    mem_per_cpu = total_mem / num_cpus
    mem_per_cpu = int(mem_per_cpu)

    # Create file which starts array job
    job_file_path = create_new_job_file(
        bout_id=bout_id,
        num_tasks=num_runs,
        num_cpus=num_cpus,
        num_tasks_concurrent=num_tasks_concurrent,
        command_file_path=command_file_path,
        mem_per_cpu=mem_per_cpu,
        runtime_days=calc_runtime_days(num_adms=num_admissions),
        use_gpus=use_gpus,
        index=index
    )

    # Submit job
    submit_job_to_slurm(
        job_file_path=job_file_path,
        starting_offset_hours=starting_offset_hours,
        gpus=use_gpus
    )

    # Print special messages
    if len(special_messages) > 0:
        logging.info("SPECIAL MESSAGES FOR THIS SEARCH")
        for msg in special_messages:
            logging.info(f"\t -> {msg}")
    else:
        logging.info("There are no special messages for this search.")


def explore_search_space(search_space, num_runs, min_training_observations=200, experiment_mode=False):
    # Generate random options if requested
    if experiment_mode:
        logging.info(f"Reverting to purely random search ...")
        return {job_idx: sample_from_search_space(search_space) for job_idx in range(num_runs)}

    # Try to open file of previous observations
    observations = io.read_json(observations_path)
    if observations is None:
        observations = {}

    # Remove meta entry (which contains statistics about the runs)
    meta_entry_name = 'meta'
    if meta_entry_name in observations:
        del observations[meta_entry_name]

    # Gather information about existing search runs
    results, arg_results = find_search_results()

    # Link arguments to runs
    for run_idx, run_info in enumerate(results.values()):
        run_info['args'] = {}
        for arg_name, arg_vals in arg_results.items():
            arg_val, obj_val = arg_vals[run_idx]
            assert obj_val == run_info['objective']
            run_info['args'][arg_name] = arg_val

    # Find out hash of model file (it is used to confirm identity of run) and if run is still in progress
    runs = {}
    for run_name, run_info in results.items():
        # Check if we want to include run in observations
        run_info['name'] = run_name
        models_dir = os.path.join(run_info['dir'], io.model_snapshots_name)
        training_info = io.read_json(os.path.join(models_dir, "training.json"))
        include_run = training_info is not None and training_info['done']
        if not include_run:
            continue

        # Get hash of model file - in addition to the run's name, this serves to uniquely identify the run (even vs.
        # other runs with the same name)
        model_path = os.path.join(models_dir, "dyn_model.h5")
        with open(model_path, "rb") as model_file:
            model_hash = hashlib.sha256(model_file.read()).hexdigest()
            run_info['model_hash'] = model_hash
            runs[f"{run_info['name']}_{model_hash}"] = run_info

    # Add new runs to json file
    for run_key, run_info in runs.items():
        if run_key in observations:
            logging.info(f"Run already stored ; skipping {run_key})")
            continue
        logging.info(f"Found new run {run_key}!")
        run_info['date_added_utc'] = datetime.utcnow().isoformat()
        observations[run_key] = run_info

    # Save observations back to json (and update meta entry)
    logging.info(f"Total observations stored: {len(observations)}")
    observations[meta_entry_name] = {
        'count': len(observations)
    }
    io.write_json(observations, observations_path, verbose=True, pretty=True)
    del observations[meta_entry_name]

    # Generate random options if we don't have enough observations for training a model to predict the objective
    # function
    if len(observations) < min_training_observations:
        logging.info(f"{min_training_observations} observations are required for smart search; reverting to "
                     f"purely random search ...")
        return {job_idx: sample_from_search_space(search_space) for job_idx in range(num_runs)}

    # Train model to suggest promising options
    model, obs_cols, str_possible_vals = train_opt_suggestion_model(observations)

    # Try out random options and have the model judge their likely performance
    options_tested = []
    for _ in tqdm(range(10 * num_runs), desc="Simulating options using model..."):

        # Sample a fresh set of options
        opts = sample_from_search_space(search_space)
        opts_orig = copy.deepcopy(opts)

        # Remove the prefixes of the option names
        prefixes = ["model", "training", "prep"]
        opts_filtered = {}
        skip_these_opts = False
        for arg_name, arg_val in opts.items():
            prefix_found = False
            for pref in prefixes:
                if prefix_found:
                    break
                if arg_name.startswith(pref):
                    arg_name = arg_name.replace(pref, "arg")
                    prefix_found = True

            # Skip arguments that are not known to the model
            if arg_name not in obs_cols:
                continue

            # Convert strings to integer indices
            if type(arg_val) == np.str_:
                if arg_val in str_possible_vals[arg_name]:
                    arg_val = str_possible_vals[arg_name].index(arg_val)
                else:
                    # The model does not know this possible value of a string option -> Discard this set of options
                    skip_these_opts = True

            if skip_these_opts:
                break

            opts_filtered[arg_name] = arg_val
        opts = opts_filtered

        # Go to next set of options if these are skipped due to incompatibility with the model
        if skip_these_opts:
            continue

        # Add a random early stopping patience (we have to add it manually because it is not supplied by the search
        # space)
        opts['arg_early_stopping_patience'] = skopt.space.Integer(2, 8, prior='log-uniform').rvs(1)[0]

        # Build dataframe for consulting model about options
        opts_df = pd.DataFrame(data=[opts])
        opts_df = opts_df[obs_cols]  # same order as was used for training model
        opts_df = model_one_hot_encoding(opts_df, str_possible_vals)
        opts_arr = opts_df.to_numpy()

        # Ask the model about the expected performance of the options
        expected_error = float(model(opts_arr)[0, 0])

        # Store the error along with the options
        options_tested.append((expected_error, opts_orig))

    # Let some percentage of the runs be suggested by the model and some be random
    options_tested.sort(reverse=True, key=lambda tup: tup[0])  # best runs are at the back
    num_suggested_runs = np.ceil(0.5 * num_runs).astype(int)
    job_opts = {}
    for job_idx in range(num_runs):
        if job_idx < num_suggested_runs:
            job_opts[job_idx] = options_tested.pop()[1]
        else:
            job_opts[job_idx] = sample_from_search_space(search_space)

    return job_opts


def train_opt_suggestion_model(observations):
    # Collect all information in a dataframe
    rows = []
    unused_args = ['admission_filter', 'admissions', 'baseline', 'evaluated_fraction',
                   'id', 'include_drugs', 'include_neonates', 'max_epochs', 'no_clustering', 'no_plot', 'no_shuffle',
                   'num_bootstraps', 'plot_architecture']
    unused_run_info_keys = ['date_added_utc', 'dir', 'marker', 'model_hash', 'name', 'args', 'log']
    for run_info in observations.values():
        for arg_name, arg_val in run_info['args'].items():
            if arg_name not in unused_args:
                run_info[f'arg_{arg_name}'] = arg_val

        for unused_run_info_key in unused_run_info_keys:
            if unused_run_info_key in run_info:
                del run_info[unused_run_info_key]

        rows.append(run_info)

    obs_df = pd.DataFrame(data=rows)

    # Remove columns that have the same value for all observations
    num_unique = obs_df.nunique()
    cols_to_drop = num_unique[num_unique == 1].index
    obs_df = obs_df.drop(cols_to_drop, axis=1)

    # Limit size of objective value
    obj_median = np.median(obs_df['objective'])
    obs_df['objective'] = obs_df['objective'].clip(upper=3 * obj_median)

    # Split into input (i.e. the arguments) and target (the objective value)
    target_arr = obs_df['objective'].to_numpy()
    obs_df = obs_df.drop(['objective'], axis=1)

    # Convert each column's data into a type suitable for a NN model

    def isnan(n):
        return type(n) in [float, np.float, np.float64] and np.isnan(n)

    # Store string index mapping - we will later need it for handling concrete runs
    str_possible_vals = {}

    for col_name in obs_df.columns:
        # Find out most common non-NaN value
        vals = [v for v in obs_df[col_name].values if not isnan(v)]
        most_common = Counter(vals).most_common(1)[0][0]

        # Set all NaN values to the most common value
        vals = [v if not isnan(v) else most_common for v in obs_df[col_name].values]

        # Convert strings to integer indices
        val_types = defaultdict(int)
        for v in vals:
            val_types[type(v)] += 1

        if len(val_types) == 1:
            if str in val_types:
                possible_vals = list(np.unique(vals))
                str_possible_vals[col_name] = possible_vals
                vals = [possible_vals.index(v) for v in vals]

        # Store updated values in dataframe
        obs_df[col_name] = vals

    # Convert string index columns to one-hot encoding
    obs_df_cols_normal = list(obs_df.columns)
    obs_df = model_one_hot_encoding(obs_df, str_possible_vals)

    # Set up model
    input_arr = obs_df.to_numpy()
    model = build_opt_suggestion_model(input_arr)

    # If we have enough data, use a validation set
    if len(obs_df) > 1000:
        monitored = 'val_loss'
        fit_kwargs = {'validation_split': 0.1}
    else:
        monitored = 'loss'
        fit_kwargs = {}

    # Compile and train model

    # Create early stopping callback
    patience = 128
    early_stopping = EarlyStopping(
        monitor=monitored,
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks = [early_stopping]

    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # Fit the model
    model.summary()
    logging.info(f"Fitting model on {len(input_arr)} examples ...")
    model.fit(
        x=input_arr,
        y=target_arr,
        epochs=500 * patience,
        callbacks=callbacks,
        **fit_kwargs
    )

    return model, obs_df_cols_normal, str_possible_vals


def model_one_hot_encoding(observations_df, str_possible_vals):
    for str_col, possible_vals in str_possible_vals.items():
        for v_idx, v in enumerate(possible_vals):

            # Make a new column with a binary value: Either the value is present, or not
            has_v = (observations_df[str_col] == v_idx).astype(int)
            new_col_name = f"{str_col}_{v}"
            observations_df[new_col_name] = has_v

        #  Remove original integer index column
        observations_df = observations_df.drop([str_col], axis=1)

    return observations_df


def build_opt_suggestion_model(input_arr, num_dense=3, dense_units_factor=1.5, activation='relu'):
    num_samples, num_features = input_arr.shape
    opt_input = Input(shape=(num_features,))
    x = opt_input

    # Dense layers for processing
    dense_units = np.ceil(dense_units_factor * num_features).astype(int)
    for _ in range(num_dense - 1):
        x = Dense(
            units=dense_units,
            activation=activation
        )(x)

    # Final dense layer uses no activation
    output = Dense(units=1, activation=None)(x)

    # Construct model
    model = Model(
        inputs=opt_input,
        outputs=output
    )

    return model


def sample_from_search_space(search_space):
    # Generate option values from the search space
    options = {}
    for opt_name, search_range in search_space.items():
        if type(search_range) == list:
            # Sample from a list of options
            opt_val = np.random.choice(search_range, 1)[0]
        else:
            # Sample from a skopt space
            opt_val = search_range.rvs(1)[0]
        options[opt_name] = opt_val
    return options


def calc_runtime_days(num_adms, epochs=500):
    """
    Calculates runtime (in days) of the jobs
    :param num_adms:
    :param epochs:
    :return:
    """

    # Data Loading
    loading_hours = 0.5 + 5/100. * num_adms / 3600. + 1/1000. * num_adms / 3600. + 1/5. * num_adms / 3600.
    loading_hours /= 5

    # Training
    hours_per_epoch = num_adms * 0.045 / 3600
    training_hours = epochs * hours_per_epoch

    # Plotting
    plotting_hours = 50 * loading_hours

    # Conversion to days
    runtime_hours = loading_hours + training_hours + plotting_hours
    runtime_hours *= 1.3  # Tolerance for longer runtime
    runtime_days = int(np.ceil(runtime_hours / 24.))

    # Do not return more days than the cluster allows
    runtime_days = min(runtime_days, 7)

    return runtime_days


def mem_gb_by_num_admissions(k):
    """
    Calculates memory needed with respect to the number of loaded admissions
    :param k:
    :return:
    """
    init_mem = 0.5 * k
    data_loading_mem = 0.9 * k
    data_proc_mem = 1.0 * k
    training_mem = 16000 + 0.45 * k
    plotting_mem = 2000 + 0.15 * k
    mem_mb = init_mem + data_loading_mem + data_proc_mem + training_mem + plotting_mem
    mem_gb = int(np.ceil(float(mem_mb) / 1000))
    mem_gb = min(mem_gb, 168)  # Maximum reasonable amount of memory
    return mem_gb


def submit_job_to_slurm(job_file_path, starting_offset_hours=0, gpus=True):
    # Print job file
    logging.info("JOB FILE")
    with open(job_file_path, "r") as job_file:
        for line in job_file.readlines():
            logging.info(line.strip())

    # Find out SLURM account
    acc_var_name = "SLURM_ACCOUNT_DL"  # for using an account different from the default one, set this environment
    # variable
    acc_name = os.getenv(acc_var_name, None)

    # Compose command
    cmd_tokens = ["sbatch"]
    if acc_name is not None:
        cmd_tokens.append("--account=jrc_combine")
    if gpus:
        cmd_tokens.append("--gres=gpu:volta:2")
    if starting_offset_hours != 0:
        cmd_tokens.append(f"--begin=now+{starting_offset_hours}hours")  # Used for starting the job later
    cmd_tokens.append(job_file_path)

    # Generate the finished command
    cmd = " ".join(cmd_tokens)

    # Execute the command
    logging.info(f"Executing submission command '{cmd}'...")
    os.system(cmd)
    logging.info(f"Submission done.")


def write_commands_to_file(all_options, bout_id, num_admissions=10, early_stopping_patience=8,
                           no_shuffle=False, no_plot=False, no_clustering=False, include_drugs=False,
                           include_neonates=False, interactive=False):
    """
    Writes a file where each line contains a command to be called by a job
    :return: path of file
    """

    # Create temporary file
    if not interactive:
        command_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False,
            dir=os.getenv("WORK")  # Command file needs to be accessible for all nodes on the cluster
        )
        command_file_name = command_file.name

    # Add command for every set of options
    for opts_idx, opts in all_options.items():
        # Name of the run
        run_name = f"{search_name_prefix}{bout_id}_{opts_idx + 1}"

        # Prepare training script call
        training_command_tokens = [
            "python full_pipeline_mimic.py",
            f"--id {run_name}",
            f"--admissions {num_admissions}",
            f"--early_stopping_patience {early_stopping_patience}"
        ]
        if no_shuffle:
            training_command_tokens.append("--no_shuffle")
        if no_plot:
            training_command_tokens.append("--no_plot")
        if no_clustering:
            training_command_tokens.append("--no_clustering")
        if include_drugs:
            training_command_tokens.append("--include_drugs")
        if include_neonates:
            training_command_tokens.append("--include_neonates")

        # Add options arguments
        for opt_name, opt_val in opts.items():
            training_command_tokens.append(
                f"--{opt_name} {opt_val}"
            )

        # Compose command
        training_command = " ".join(training_command_tokens)

        # Execute command directly
        if interactive:
            logging.info(f"Executing run command interactively: '{training_command}'...")
            old_dir = os.path.realpath(os.curdir)
            tgt_dir = os.path.realpath(os.path.join(os.path.split(__file__)[0], ".."))
            os.system(f"cd {tgt_dir} ; {training_command} ; cd {old_dir}")
            logging.info(f"Run {opts_idx} done.")

        # Write command to file
        if not interactive:
            command_file.write(f"{training_command}\n")

    if not interactive:
        # Close file
        command_file.write("\n")
        command_file.close()

        return command_file_name
    else:
        return None


def create_new_job_file(bout_id, num_tasks, command_file_path, mem_per_cpu=2, num_cpus=32, runtime_days=4,
                        num_tasks_concurrent=50, use_gpus=False, index=None):
    """
    Creates a new job file with the specified parameters.
    :returns: path to file
    """

    # Create temporary file
    job_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    job_file_name = job_file.name

    # Write shebang
    job_file.write("#!/usr/local_rwth/bin/zsh\n")

    # Declare this as an array job
    if index is None:
        job_file.write(f"#SBATCH --array=1-{num_tasks}%{num_tasks_concurrent}\n")
    else:
        job_file.write(f"#SBATCH --array={index}\n")

    # SLURM memory comment
    job_file.write(f"#SBATCH --mem-per-cpu={mem_per_cpu}G\n")

    # SLURM cpu count
    job_file.write(f"#SBATCH --cpus-per-task={num_cpus}\n")

    # SLURM job name
    job_file.write(f"#SBATCH --job-name={bout_id}\n")

    # SLURM log path
    # {search_name_prefix}{bout_id}_{opts_idx + 1}
    job_file.write(f"#SBATCH --output={results_dir}/log_{search_name_prefix}{bout_id}_%a.txt\n")

    # SLURM maximum runtime
    job_file.write(f"#SBATCH --time={runtime_days}-0:00:00\n")

    # Exclude GPU nodes if not using GPUs
    if not use_gpus:
        job_file.write(f"#SBATCH --exclude=nihg[01-08]\n")

    # Change into code directory
    job_file.write("cd ~/src/projects/dl-clus\n")

    # Source conda environment
    job_file.write("source ~/.zshrc\n")

    # Get line from command file and execute it
    job_file.write("$(sed -n ${SLURM_ARRAY_TASK_ID}p " + command_file_path + ")\n")

    # Write newline at the end of script and close the file
    job_file.write("\n")
    job_file.close()

    return job_file_name


def calculate_objective_value(eval_results, training_results, allow_untrained_loss=False, obj_val_suffix=None):
    """
    Calculates (or extracts) objective value from evaluation results
    :param eval_results:
    :param training_results:
    :param allow_untrained_loss:
    :param obj_val_suffix:
    :return:
    """

    if 'val_loss' in eval_results:
        # Objective value is the median reconstruction error of all evaluated reconstructions. The 'weighted' means that
        # attributes with more measurements also have a higher influence on the median.
        obj_val_key = 'rec_error_median_overall_weighted'
        if obj_val_suffix is not None:
            obj_val_key += obj_val_suffix
        obj_val = float(eval_results[obj_val_key])
        mode = 'eval'
    else:
        if training_results is None or not allow_untrained_loss:
            obj_val = 5000
            mode = 'eval'
        else:
            obj_val = float(training_results['val_loss_before_training'])
            mode = 'val_loss_pre'

    return mode, obj_val


def find_search_results(obj_val_suffix=None):
    # Get a list of all search result folders
    search_result_dirs = glob(
        os.path.join(results_dir, f"{search_name_prefix}*")
    )
    logging.info(f"Found {len(search_result_dirs)} search runs.")

    # Create entries for the results
    results = OrderedDict()
    for search_result_dir in search_result_dirs:
        run_name = os.path.split(search_result_dir)[1]
        run_name = run_name.replace(search_name_prefix, "")
        results[run_name] = {
            'dir': search_result_dir,
            'name': run_name
        }

    # Open logs for all search runs
    for run_name, run_info in results.items():
        # Open log
        log_glob_path = os.path.join(run_info['dir'], "..", f"*{run_info['name']}.txt")
        log_paths = glob(log_glob_path)
        log_file_found = len(log_paths) == 1
        if log_file_found:
            log_path = os.path.normpath(log_paths[0])
            with open(log_path, "r") as log_file:
                log_txt = log_file.readlines()
            run_info['log'] = log_txt[-1]  # Only save last line: It tells us about the exit state

    # Check if log indicates that run did not run to completion but ended in an error
    success_marker = "success"
    error_marker = "error"
    oom_marker = "oom-kill event"
    ongoing_marker = "ETA"
    default_marker = "other_non_success"
    final_statuses = {
        success_marker: {},
        oom_marker: {},
        ongoing_marker: {},
        error_marker: {},
        default_marker: {}
    }
    for run_name, run_info in results.items():
        if 'log' not in run_info:
            run_info['marker'] = default_marker
            continue

        last_line = run_info['log']

        entry_assigned_to_status = False
        for marker in [success_marker, oom_marker, ongoing_marker, error_marker]:

            # Check if we find the marker
            if marker.lower() in last_line.lower() and not entry_assigned_to_status:

                # List this run together with the last line in the status dictionary
                entry_assigned_to_status = True
                if last_line not in final_statuses[marker]:
                    final_statuses[marker][last_line] = []
                final_statuses[marker][last_line].append(run_name)

                # Save the marker in the run
                run_info['marker'] = marker

        # Put this entry in the list of statuses that are neither error nor success (maybe an incomplete search that is
        # still running)
        if not entry_assigned_to_status:
            if last_line not in final_statuses[default_marker]:
                final_statuses[default_marker][last_line] = []
            final_statuses[default_marker][last_line].append(run_name)

            # Save the marker in the run
            run_info['marker'] = default_marker

    # Report error and 'other' states
    for marker in [oom_marker, error_marker, default_marker]:
        states = final_statuses[marker]
        logging.info(f"{marker} state: There are {len(states)} different final states:")
        for state_line, state_runs in states.items():
            logging.info(f"{marker} state:\n"
                         f"{state_line.strip()}\n"
                         f"there are {len(state_runs)} runs: {state_runs}")

    # Get evaluation for all runs that have an evaluation
    succeeding_runs = sum(final_statuses[success_marker].values(), [])
    logging.info(f"{len(succeeding_runs)} runs succeeded.")
    arg_results = {}
    mode_all = None
    skipped_runs = []
    for run_idx, (run_name, run_info) in enumerate(results.items()):
        # Open eval
        logging.info(f"Extracting eval info from run {run_idx + 1} of {len(results)} "
                     f"({os.path.split(run_info['dir'])[-1]}) ...")
        eval_paths = glob(os.path.join(run_info['dir'], "eval", io.split_name_all, "*.json"))
        eval_found = len(eval_paths) == 1

        if not eval_found:
            run_info['objective'] = no_data_loss
        else:
            eval_path = eval_paths[0]
            with open(eval_path, "r") as eval_file:
                eval_info = json.load(eval_file)

            # Open training file
            training_info = io.read_json(path=os.path.join(run_info['dir'], io.model_snapshots_name, "training.json"))

            # Extract information from eval file
            # Get representative value that allows us to judge the final performance of the model
            mode, objective_value = calculate_objective_value(
                eval_info,
                training_info,
                obj_val_suffix=obj_val_suffix
            )
            if mode_all is None:
                mode_all = mode
            else:
                assert mode == mode_all, "Can not mix objective value modes!"

            # Save objective value to run info for later use
            saved_objective_value = objective_value
            if saved_objective_value is None:
                saved_objective_value = no_data_loss  # Make it comparable to other losses but really high
            run_info['objective'] = saved_objective_value

        # Abort if this run is ongoing
        if run_info['marker'] == ongoing_marker:
            skipped_runs.append((run_idx, run_name))
            continue

        # Handle each of the options
        for arg_name, arg_value in eval_info['args'].items():

            # Save the performance reached with this value of the argument
            if arg_name not in arg_results:
                arg_results[arg_name] = []

            # Convert None values into a string that says so
            if arg_value is None:
                arg_value = arg_not_used_str

            # Convert the value of the argument to a stricter type: all of them are given as strings, but numbers should
            # be typed as numbers
            try:
                arg_value = int(arg_value)
            except ValueError:
                try:
                    arg_value = float(arg_value)
                except ValueError:
                    pass

            arg_results[arg_name].append(
                (arg_value, objective_value)
            )

    # Remove skipped runs from results
    for run_idx, run_name in skipped_runs:
        logging.info(f"Removing run {run_idx + 1}: {run_name} (it is still ongoing)")
        del results[run_name]

    return results, arg_results


def analyze_results():
    """
    Analyzes search results and gives insight into optimal parameters for the model or problems with the search itself
    (such as jobs being shut down on the cluster)
    :return:
    """

    # Greet user
    print_large("Analysis")

    # Perform for MSE error and for MAPE error
    print_buffer = []
    for err_name, err_suffix in [("MSE", None), ("MAPE", "_MAPE")]:

        # Find results
        logging.info(f"Running analysis for error type '{err_name}'")
        results, arg_results = find_search_results(obj_val_suffix=err_suffix)
        if len(arg_results) == 0:
            arg_results = defaultdict(list)

        # Load observations file (it contains runs from previous searches) and add those runs to the argument analysis
        observations = io.read_json(observations_path)
        if observations is None:
            observations = {}
        for obs_info in observations.values():
            # Skip meta entry
            if 'args' not in obs_info:
                continue

            obs_args = obs_info['args']
            obs_objective = obs_info['objective']

            for arg_name, arg_val in obs_args.items():
                arg_results[arg_name].append((arg_val, obs_objective))

        # Make sure arguments are unified across all included runs
        num_args_set = {k: len(v) for (k, v) in arg_results.items()}
        unique_num_arg_res = np.unique(list(num_args_set.values()))
        assert len(unique_num_arg_res) == 1, f"Not all arguments were set for all runs! ({num_args_set})"

        # Remove extreme outliers at the high side: They distort the analysis
        arg_results_orig = arg_results
        for kept_fraction in [10, 50, 90, 99]:
            arg_results = copy.deepcopy(arg_results_orig)
            obj_values = [v for v in list(list(zip(*arg_results[list(arg_results.keys())[0]]))[1]) if v is not None]
            if kept_fraction < 100:
                threshold = np.nanpercentile(obj_values, kept_fraction)
                for arg_name in arg_results.keys():
                    arg_results[arg_name] = [(opt_name, obj_v) if obj_v is not None else (opt_name, obj_v)
                                             for (opt_name, obj_v) in arg_results[arg_name]
                                             if obj_v is None or obj_v <= threshold]

            # Draw scatter plot for each argument
            iom = io.IOFunctions(
                dataset_key="mimic",
                training_id=None,
                treat_as_search_result=False
            )
            plotter = plot.Plotting(
                iom=iom,
                preprocessor=None,
                trainer=None,
                evaluator=None,
                clustering=None,
                split=f"srch_analyze_top{kept_fraction}_{err_name}"
            )
            for arg_name, arg_val_results in arg_results.items():

                # Only plot if the values for the argument are actually varied
                arg_inputs = [x for (x, fx) in arg_val_results]
                if len(np.unique(arg_inputs)) < 2:
                    continue

                # Don't plot ID
                if arg_name == "id":
                    continue

                plotter.plot_function_output_scatter(
                    func_evals=arg_val_results,
                    function_name=f"search_arg_{arg_name}_{err_name.lower()}"
                )

            # Draw heat map of option performance for each pair of arguments
            for arg_1, arg_2 in itertools.combinations(arg_results.keys(), 2):

                # Only plot if the values for both of the arguments are actually varied
                arg_1_inputs = [x for (x, fx) in arg_results[arg_1]]
                arg_2_inputs = [y for (y, fy) in arg_results[arg_2]]
                if len(np.unique(arg_1_inputs)) < 2 or len(np.unique(arg_2_inputs)) < 2:
                    continue

                # Don't plot ID
                if "id" in [arg_1, arg_2]:
                    continue

                logging.info(f"Plotting heat map for performance between {arg_1} and {arg_2} ...")
                plotter.plot_heat_map(
                    x_name=arg_1,
                    x_inputs=arg_1_inputs,
                    y_name=arg_2,
                    y_inputs=arg_2_inputs,
                    func_vals=[fx for (x, fx) in arg_results[arg_1]],  # Would be the same for `arg_2`
                    heat_center=0  # since we are plotting errors, the center should be 0 (best error = 0)
                )
                logging.info(f"Plotting heat map done!")

        # Print out list of best runs
        all_runs = list(results.values())
        obj_values = [r['objective'] for r in all_runs]
        run_ordering = np.argsort(obj_values)
        all_runs = [all_runs[idx] for idx in run_ordering]

        # Also order arg results by objective values
        arg_results_ordered = {arg_name: [vals[idx] for idx in run_ordering]
                               for (arg_name, vals) in arg_results_orig.items()}

        # Remove arg results entries which are the same for all runs
        removed_args = []
        for arg_name, arg_val_results in arg_results_ordered.items():
            arg_inputs = [x for (x, fx) in arg_val_results]
            if len(np.unique(arg_inputs)) < 2:
                removed_args.append(arg_name)
        for arg_name in removed_args:
            del arg_results_ordered[arg_name]

        show_best_x = min(15, len(all_runs))
        print_buffer.append(f"Best {show_best_x} runs ({err_name}):")
        for idx in range(show_best_x):
            opts = {k: v[idx][0] for (k, v) in arg_results_ordered.items()}
            obj = all_runs[idx]['objective']
            run_name = all_runs[idx]['name']
            assert opts['id'].endswith(run_name)
            del opts['id']
            opts_str = ",\n".join([f"{arg_name}: {v}" for (arg_name, v) in opts.items()])
            print_buffer.append(f"{idx + 1}. best run with objective value {obj} ({err_name}): {run_name}\n"
                                f"{opts_str}")

        print_buffer.append(f"Worst {show_best_x} runs:")
        worst_shown = 0
        for idx in reversed(range(len(all_runs))):
            objective_value = all_runs[idx]['objective']
            if objective_value == no_data_loss:
                continue
            print_buffer.append(f"{worst_shown + 1}. worst run with objective value {objective_value} ({err_name}):"
                                f" {all_runs[idx]['name']}")
            worst_shown += 1
            if worst_shown == show_best_x:
                break

    for line in print_buffer:
        logging.info(line)


if __name__ == "__main__":
    main_func()
