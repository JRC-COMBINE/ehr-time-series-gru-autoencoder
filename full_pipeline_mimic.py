# Logging
import logging

# Math and data
import numpy as np

# Utility
import random
import os
import argparse

# Import my own modules
from data.MimicExtractor import MimicExtractor
from data.MimicMedicationExtractor import MimicMedicationExtractor
from common import io
from evaluation import plot, eval
from data.preprocessor import Preprocessor
from ai.training import Training
from ai.clustering import Clustering


# Show time along with log messages
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline for clustering patients")
    parser.add_argument("--no_shuffle", action='store_true',
                        help="do not randomly shuffle admissions before processing")
    parser.add_argument("--id", type=str, help="ID of this training (used for writing results into distinct "
                                               "directory based on ID)",
                        default="anon")
    parser.add_argument("--no_plot", action='store_true',
                        help="do not plot")
    parser.add_argument("--no_clustering", action='store_true',
                        help="do not cluster (and thus, do not evaluate clusterings)")
    parser.add_argument("--admissions", type=int, help="number of admissions to train (and evaluate) on",
                        default=60000)
    parser.add_argument("--max_epochs", type=int, help="maximum number of epochs to train (usually, training will "
                                                       "stop earlier due to early stopping)",
                        default=20000)
    parser.add_argument("--early_stopping_patience", type=int, help="early stopping patience",
                        default=8)
    parser.add_argument("--num_bootstraps", type=int, help="number of bootstrappings",
                        default=10)
    parser.add_argument("--evaluated_fraction", type=float, help="Fraction of admissions used for model evaluation",
                        default=0.2)
    parser.add_argument("--plot_architecture", action='store_true',
                        help="plot the model architecture")
    parser.add_argument("--baseline", action='store_true',
                        help="run the baseline model (PCA) instead of the NN model")
    parser.add_argument("--admission_filter", type=str, help="after training, filter the admissions using predefined "
                                                             "criteria and run the clustering only on those admissions")
    parser.add_argument("--include_neonates", action='store_true',
                        help="If set, include neonates (default is to exclude them)")
    parser.add_argument("--include_drugs", action='store_true',
                        help="If set, include drugs from data")

    args, generic_args = parser.parse_known_args()
    return args, generic_args


def get_prep_and_trainer_for_testing():
    # Change into directory required for using data extractors
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    # Parse arguments to get default arguments but modify them to be more suitable for testing
    args, generic_args = parse_args()
    args.no_shuffle = True

    # Load data
    model_args, training_args, pipeline_args, split_order, iom, prep = prepare_data(
        args=args,
        generic_args=generic_args
    )

    trainer = prepare_model(
        args=args,
        pipeline_args=pipeline_args,
        training_args=training_args,
        model_args=model_args,
        iom=iom,
        prep=prep
    )

    return prep, trainer


def main_func():
    # Parse arguments
    args, generic_args = parse_args()

    # Load data
    model_args, training_args, pipeline_args, split_order, iom, prep = prepare_data(
        args=args,
        generic_args=generic_args
    )

    # Load data, train or load a model
    trainer = prepare_model(
        args=args,
        pipeline_args=pipeline_args,
        training_args=training_args,
        model_args=model_args,
        iom=iom,
        prep=prep
    )

    # Exit early if in no-training mode (i.e. if no model is being trained or loaded)
    if trainer.no_training:
        logging.info("Done with trying out model architecture, exiting...")
        return

    # Handle clustering and plotting on both the validation and the test set
    for split in split_order:
        logging.info(f"Performing pipeline for {split} split")

        # Cluster and evaluate model and clusters
        clustering, evaluator = cluster_and_eval(
            args=args,
            pipeline_args=pipeline_args,
            iom=iom,
            prep=prep,
            trainer=trainer,
            split=split,
            perform_clusterings=not args.no_clustering
        )

        # Plot
        if not args.no_plot:
            plot_all_plots(
                args=args,
                iom=iom,
                prep=prep,
                trainer=trainer,
                evaluator=evaluator,
                clustering=clustering,
                split=split,
                clustering_performed=not args.no_clustering
            )

        logging.info(f"Pipeline done for {split} split")

    # Print that run completed successfully
    logging.info("Run completed successfully.")


def prepare_data(args, generic_args):
    # Process generic arguments
    training_args = {}
    model_args = {}
    prep_args = {}
    gen_arg_names = generic_args[::2]  # every other, starting with index 0
    gen_arg_vals = generic_args[1::2]  # every other, starting with index 1
    for gen_arg_name, gen_arg_val in zip(gen_arg_names, gen_arg_vals):
        # Remove -- from the argument names
        gen_arg_name = gen_arg_name.replace('--', '')

        if gen_arg_name.startswith('training'):
            gen_arg_name = gen_arg_name.replace('training_', '')
            training_args[gen_arg_name] = gen_arg_val
        elif gen_arg_name.startswith('model'):
            gen_arg_name = gen_arg_name.replace('model_', '')
            model_args[gen_arg_name] = gen_arg_val
        elif gen_arg_name.startswith('prep'):
            gen_arg_name = gen_arg_name.replace('prep_', '')
            prep_args[gen_arg_name] = gen_arg_val
        else:
            assert False, f"Generic argument name '{gen_arg_name}' not recognized!"

    # Set ID of training - it determines where results will be saved once training completes
    training_id = args.id

    # Set order in which splits are processed
    if "admission_filter" in args:
        adm_filter = args.admission_filter
    else:
        adm_filter = None
    if adm_filter is None:
        split_order = [io.split_name_all, io.split_name_train, io.split_name_val]
    else:
        split_order = [adm_filter]

    # Create IO helper
    iom = io.IOFunctions(
        dataset_key="mimic",
        training_id=training_id
    )

    # Save arguments used for training this model
    pipeline_args = vars(args)  # Convert from Namespace to dict
    pipeline_args.update(training_args)
    pipeline_args.update(model_args)
    pipeline_args.update(prep_args)
    evaluated_fraction = args.evaluated_fraction if "evaluated_fraction" in args else 0.2
    evaluator_pre_training = eval.Evaluation(
        iom=iom,
        preprocessor=None,
        trainer=None,
        clustering=None,
        split=split_order[0],
        evaluated_fraction=evaluated_fraction
    )
    evaluator_pre_training.eval_model(
        pipeline_args=pipeline_args,
        after_training=False
    )

    # Init extractor
    ext = MimicExtractor(
        extraction_path=iom.get_dataset_dir(),
        disable_multiprocessing=True,
        textual_diagnosis_selection=None
    )

    # Get all admission IDs
    all_hadms = ext.retrieve_all_encounter_ids()

    # Limit (and optionally shuffle) admission IDs
    if not args.no_shuffle:
        logging.info("Shuffling data randomly before retrieval.")

        # Shuffle based on ID of run
        new_random_seed = random.randint(1, 2**14)
        shuffling_seed = int.from_bytes(args.id.encode(), 'little') % 10**6
        random.seed(shuffling_seed)
        random.shuffle(all_hadms)

        # Restore a *random* seed
        random.seed(new_random_seed)
    else:
        logging.info("NOT shuffling data randomly before retrieval.")
    hadm_id_selection = all_hadms[:args.admissions]

    # Sort hadm ids. This is important so that when all the admissions are selected, they are always loaded in the
    # same order. This improves repeatability of tests and trainings and makes comparisons of clusterings between
    # different runs possible.
    hadm_id_selection.sort()

    if args.include_drugs:
        logging.info("INCLUDING drugs in dynamic data.")

        # Extract medication data
        ext_med = MimicMedicationExtractor(
            extraction_path=iom.get_dataset_dir(),
            disable_multiprocessing=True
        )
        ext_med.extract(encounter_ids=hadm_id_selection, override=False)

    else:
        logging.info("Excluding drugs from dynamic data.")
        ext_med = None

    # Extract dynamic data
    ext.extract(encounter_ids=hadm_id_selection, override=False)

    # Extract static data
    ext.extract_static(hadm_id_selection)

    # Pre-process data
    if args.include_neonates:
        logging.info("INCLUDING neonates.")
        min_age = None
    else:
        min_age = 22  # remove neonates from consideration: Their patterns of disease are vastly different from older
        # patients (starting from their 22nd birthday, the FDA classifies people as adults)
        logging.info(f"Excluding non-adults. Minimum age for inclusion set to {min_age}.")

    prep = Preprocessor(
        iom=iom,
        extractor_exams=ext,
        extractor_drugs=ext_med,
        encounter_ids=hadm_id_selection,
        filtering_min_age=min_age,
        include_drugs=args.include_drugs,
        **prep_args  # Preprocessor arguments as keyword arguments
    )

    return model_args, training_args, pipeline_args, split_order, iom, prep


def prepare_model(args, pipeline_args, training_args, model_args, iom, prep):
    # Init model training: It will later be used to either load an existing model a train a new one
    if "early_stopping_patience" in args and args.early_stopping_patience is not None:
        if 'early_stopping_patience' not in training_args:
            training_args['early_stopping_patience'] = args.early_stopping_patience
    if "max_epochs" in args and args.max_epochs is not None:
        training_args['max_epochs'] = args.max_epochs
    if "baseline" in args and args.baseline is not None:
        training_args['baseline'] = args.baseline
    trainer = Training(
        iom=iom,
        prep=prep,
        **training_args,  # Supply args as keyword arguments
        additional_model_args=model_args
    )

    # Save pipeline args for wandb
    trainer.pipeline_args = pipeline_args

    # Make sure a fully trained model is available
    training_iter = 0
    max_training_iter = 5
    while not trainer.training_state['done']:
        training_iter += 1
        logging.info(f"Starting training iteration {training_iter}")
        trainer.train_or_load()

        if trainer.no_training:
            break
        if training_iter >= max_training_iter:
            logging.info(f"Stopping training after {training_iter} iterations")
            trainer.training_state['done'] = True
    else:
        logging.info(f"Training done! (after {training_iter} iterations)")

    # Plot descriptive plots about data: This has nothing to do with the trained model and only takes the
    # input data into account
    plot.Plotting(
        iom=iom,
        preprocessor=prep,
        # Plotter usually requires more modules (evaluator, clustering, etc.) to work, but these are not
        # necessary for plotting the plots about the input data
        trainer=trainer,
        evaluator=None,
        clustering=None,
        split=None
    ).plot_input_data_descriptive()

    return trainer


def cluster_and_eval(args, pipeline_args, iom, prep, trainer, split, perform_clusterings=True):
    # Only perform clustering if on the split with all data
    perform_clusterings = perform_clusterings and split not in [io.split_name_train, io.split_name_val]

    # Cluster points using different algorithms
    clustering = Clustering(
        iom=iom,
        preprocessor=prep,
        trainer=trainer,
        split=split,
        num_bootstraps=args.num_bootstraps
    )
    if perform_clusterings:
        clustering.cluster_admissions()

    # Start evaluator (it is also used in plotting)
    evaluator = eval.Evaluation(
        iom=iom,
        preprocessor=prep,
        trainer=trainer,
        clustering=clustering,
        split=split,
        evaluated_fraction=args.evaluated_fraction
    )

    # Eval the model
    evaluator.eval_model(
        pipeline_args=pipeline_args
    )

    # Evaluate clusters
    if perform_clusterings:
        evaluator.eval_clusters()

    return clustering, evaluator


def plot_reconstructions(trainer, plotter, split):
    adms_plotted_for_full_split = 100
    full_split_size = 50000
    split_size = len(trainer.get_split_indices(split))
    exponent = np.log(adms_plotted_for_full_split) / np.log(full_split_size)

    plotted_num = np.ceil(np.power(split_size, exponent)).astype(int)

    plotted_adms_reconstructions = trainer.sample_from_split(
        split,
        sample_num=plotted_num
    )
    logging.info(f"Plotting reconstruction for {plotted_num} admissions of {split} split")
    for adm_idx in plotted_adms_reconstructions:
        plotter.plot_time_series_reconstruction(admission_idcs=[adm_idx])
    plotter.plot_time_series_reconstruction(admission_idcs=plotted_adms_reconstructions)  # Plot all together


def plot_all_plots(args, iom, prep, trainer, evaluator, clustering, split, clustering_performed=True):
    plotter = plot.Plotting(
        iom=iom,
        preprocessor=prep,
        trainer=trainer,
        evaluator=evaluator,
        clustering=clustering,
        split=split
    )
    if args.plot_architecture:
        plotter.plot_model_architecture()

    # # # # # # # # # # # #
    # High priority general plots
    # # # # # # # # # # # #

    # Plot the reconstructions for a random sample of the admissions
    plot_reconstructions(
        trainer=trainer,
        plotter=plotter,
        split=split
    )

    # Abort rest of plotting if not on full data split
    if split in [io.split_name_train, io.split_name_val] or not clustering_performed:
        return

    # Plot behavior of mortality with respect to reconstruction error
    plotter.plot_rec_err_vs_survival_bars()

    # Plot distribution of p values in actual clusterings vs. random permutation
    plotter.plot_p_value_distribution()

    # Plot alluvial plot (Sankey diagram) of flow of admissions between competing clusterings
    plotter.plot_alluvial_flow()

    # Plot quality of reconstruction for all dynamic data attributes
    plotter.plot_dyn_attr_reconstruction_quality()

    # # # # # # # # # # # #
    # Clustering plots
    # # # # # # # # # # # #

    # Cluster similarity matrix
    plotter.plot_cluster_similarity_matrix()

    # Bars plot that explains how ICD codes "cover" patients in different clusters
    plotter.plot_icd_cumulative_covering_bars()

    # Plot tables where each row shows the mortality of a diagnosis or a procedure (w.r.t. each cluster and the
    # population)
    plotter.plot_icd_mortality_tables()

    # Plot tables where each row shows the distribution (over clusters) of a diagnosis or a procedure
    plotter.plot_icd_distribution_tables()

    # Plot bar plots of ICD distribution
    plotter.plot_icd_distribution_bar_plots()

    # Plot clusters
    plotter.plot_clusters()

    # Plot Kaplan-Meier plots (admission survival)
    plotter.plot_kaplan_meier_survival()

    # Plot age vs. survival curve
    plotter.plot_age_vs_survival()

    # Plot time series aggregated together by cluster, colored by cluster of the admission
    max_adms_per_cluster = 50
    for clustering_info in clustering.clusterings:
        # Group admission indices by cluster label
        clus_labels = clustering_info.labels
        adm_by_clus = {lab: [] for lab in np.unique(clus_labels)}
        for adm_idx, lab in zip(trainer.get_split_indices(split), clus_labels):
            adm_by_clus[lab].append(adm_idx)

        # Sample from all clusters in equal amounts
        sampled_adm_indices = []
        for clus_adms in adm_by_clus.values():
            if len(clus_adms) <= max_adms_per_cluster:
                sampled_adm_indices += clus_adms
            else:
                sampled_adm_indices += list(np.random.choice(clus_adms, size=max_adms_per_cluster, replace=False))

        plotter.plot_time_series_reconstruction(
            admission_idcs=sampled_adm_indices,
            color_by_cluster=clustering_info
        )

    # Plot results of decision tree analysis
    plotter.plot_all_decision_tree_analyses()

    # Plot bar chart for static categorical values (in clusters)
    plotter.plot_static_categ_bars()

    # Plot violin plots of numerical data within each cluster
    plotter.plot_numerical_data_clusters_violins()

    # Plot ICD tornado plots
    plotter.plot_icd_grouped_tornado()

    # Plot ICD tornado plots (for the second level of the ICD hierarchy)
    plotter.plot_icd_grouped_tornado(icd_level=2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Low priority general plots
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Plot comparison of reconstruction and ground truth at different points in time
    plotter.plot_time_series_comparisons()

    # Plot dendrogram
    plotter.plot_dendrograms()

    # Plot reconstruction error vs. the amount of data we have for each dynamic data attribute
    plotter.plot_error_by_data_amount()

    # Plot reconstructed vs. ground truth (as a scatter plot)
    plotter.plot_recon_vs_gt_scatter()

    # Plot clustering meta plots (plots about clustering process)
    plotter.plot_clustering_robustness()

    # Plot clustering robustness meta test plots
    plotter.plot_clustering_robustness_test(shuffle=False)
    plotter.plot_clustering_robustness_test(shuffle=True)


if __name__ == "__main__":
    main_func()
