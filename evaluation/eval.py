# Logging
import logging

# Utility
from datetime import datetime
import copy
import csv
import os
import sys
import humanfriendly as hf
from collections import namedtuple, defaultdict
from typing import Dict, List, Tuple, Set
from itertools import count
from tqdm import tqdm

# Math and data
import scipy.stats
import numpy as np
from scipy.stats import ranksums, fisher_exact
from sklearn import tree
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpmax
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from tabulate import tabulate

# NN
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ICD information
from info import IcdInfo

from ai.clustering import ClusteringInfo
from common import io

TreeAnalysisResult = namedtuple("TreeAnalysisResult", ["tree", "score", "feature_names", "class_counts",
                                                       "nan_fill_value"])

FrequentItemsetResults = namedtuple("FrequentItemsetResults", ["pop_itemsets", "clusterings_results", "icd_depth",
                                                               "significance_results"])

SignificanceResults = namedtuple("SignificanceResults", ["alpha", "num_itemsets", "num_clusters", "num_tests_total",
                                                         "significance_level"])

# Column names for frequent itemset mining tables
cn_itemsets = 'itemsets'  # cn stands for column name
cn_length = 'length'
cn_support = 'support'
cn_complement_support = 'complement_support'
cn_factor = 'factor'
cn_p_value = 'p_value'
cn_significance = 'significance'

aggregation_functions = {
    'median': np.median,
    'min': np.min,
    'max': np.max
}
no_aggregation_key = "no_aggregation"


class Evaluation:
    """
    Evaluation (both in the application-focused sense and in the technical sense) of the model and the finished
    clusterings
    """

    def __init__(self, trainer, iom, preprocessor, clustering, split, evaluated_fraction: float = 1.0):
        self.trainer = trainer
        self.iom = iom
        self.preprocessor = preprocessor
        self.clustering = clustering
        self.split = split

        # Eval settings
        self.evaluated_fraction = evaluated_fraction

        # Cached results
        self._eval_result = None

        # Eval report generation settings
        self.eval_dir = os.path.join(self.iom.get_eval_dir(), split)
        io.makedirs(self.eval_dir)
        report_name = "evaluation.json"
        self.report = None
        self.eval_report_path = os.path.join(self.eval_dir, report_name)
        self._cluster_eval_path = os.path.join(self.eval_dir, "cluster_eval.pkl")

    def get_reconstruction_loss(self):
        """
        Checks the quality of reconstruction using the trained autoencoder.
        :return: dictionary with losses and admission lengths
        """

        # Only recompute if necessary
        if self._eval_result is None:
            # Sample some data
            adm_indices = self.trainer.get_split_indices(self.split)
            adm_indices = np.random.choice(
                adm_indices,
                size=int(np.ceil(self.evaluated_fraction * len(adm_indices)))
            )
            logging.info(f"Validating on {len(adm_indices)} {self.split} examples.")

            # Save variance of each time series - we later use this to calculate statistics of variance for each
            # dynamic data attribute
            dyn_variances = defaultdict(list)

            # For each sample of val data, get reconstruction loss
            losses = []
            rec_errors = []
            adms_evaluated = []
            lengths = []
            error_mse_by_label = defaultdict(list)
            error_mape_by_label = defaultdict(list)
            for num_processed, adm_idx in zip(count(start=1), adm_indices):
                logging.info(f"[{num_processed} / {len(adm_indices)}] Evaluating reconstruction ...")
                recon_arr, recon_loss = self.trainer.reconstruct_time_series(
                    adm_idx=adm_idx,
                    evaluate=True
                )
                # recon_arr.shape = (steps, features)
                losses.append(recon_loss)

                # Store the ground truth along with the reconstruction: Use the de-imputed dyn chart since
                # reconstruction quality should only be measured on raw, original data (not imputed data points)
                dyn_chart_gt = self.preprocessor.get_deimputed_dyn_chart(adm_idx)
                gt_and_reco = {}  # type: Dict[str, List[Tuple[float, float]]]
                adm_rec_errors = []  # type: List[Tuple[int, float]]
                for col_idx, col_name in enumerate(dyn_chart_gt.columns):
                    # Don't evaluate meta columns (time and static values)
                    if col_name in self.preprocessor.meta_columns:
                        continue

                    # Extract ground truth values but filter out NaNs
                    gt = dyn_chart_gt[col_name]
                    gt_valid_idcs = gt.loc[gt.notnull()].index
                    gt = gt[gt_valid_idcs].to_numpy()

                    # Get reconstruction (also only for entries which are not NaN in original data)
                    rec = recon_arr[:, col_idx]
                    rec = rec[gt_valid_idcs]

                    gt_and_reco[col_name] = (gt, rec)

                # Calculate error for each attribute
                for item_id, attr_gt_and_reco in gt_and_reco.items():
                    # Unpack ground truth and reconstruction
                    attr_gt, attr_reco = attr_gt_and_reco

                    # No error can be measured for zero-length time series
                    if len(attr_gt) == 0:
                        continue

                    # Calculate MSE between ground truth and reconstruction
                    error_mse = np.mean(
                        np.power(
                            attr_gt - attr_reco,
                            2
                        )
                    )

                    # Save MSE (this lets us calculate the admission's mean error)
                    adm_rec_errors.append((len(attr_gt), error_mse))

                    # Save reconstruction error result for this admission (and this attribute)
                    item_label = self.preprocessor.label_for_any_item_id(item_id)
                    error_mse_by_label[item_label].append(error_mse)

                    # Also calculate and save MAPE
                    error_mape = self._error_mape(
                        attr_gt=attr_gt,
                        attr_reco=attr_reco,
                        item_id=item_id
                    )
                    if error_mape is not None:
                        error_mape_by_label[item_label].append(error_mape)

                    # Save variance of this time series
                    dyn_variances[item_label].append(np.var(attr_gt))

                # Save reconstruction error for this admission (mean over all of its time steps)
                adm_obs_count = 0
                adm_obs_err = 0
                for obs_len, ts_err in adm_rec_errors:
                    adm_obs_count += obs_len  # total number of observations
                    adm_obs_err += ts_err * obs_len  # total error (not mean)
                if adm_obs_count > 0:
                    adm_err = adm_obs_err / adm_obs_count  # a mean over all observations of this admission
                    rec_errors.append(float(adm_err))
                    adms_evaluated.append(int(adm_idx))
                    lengths.append(adm_obs_count)  # total number of observations for this admission

            # Save variance statistics for all dynamic data attributes
            dyn_variance_stats = {k:
                                      {
                                          'mean': np.mean(v),
                                          'median': np.median(v),
                                          'min': np.min(v),
                                          'max': np.max(v),
                                          'p25': np.percentile(v, 25),
                                          'p75': np.percentile(v, 75)
                                      }
                for (k, v) in dyn_variances.items()}

            # Cache the result of the evaluation
            self._eval_result = {
                'dyn_variance_stats': dyn_variance_stats,
                'losses': losses,
                'adm_rec_errors': rec_errors,
                'adms_evaluated': adms_evaluated,
                'lengths': lengths,
                'rec_error_scores': error_mse_by_label,
                'rec_error_scores_MAPE': error_mape_by_label
            }

        return self._eval_result

    def _error_mape(self, attr_gt, attr_reco, item_id):
        # Reverse normalization for ground truth and reconstruction
        attr_gt = self.preprocessor.reverse_scaling(attr_gt, column_name=item_id)
        attr_reco = self.preprocessor.reverse_scaling(attr_reco, column_name=item_id)
        # When using MAPE error, it is extremely important to evaluate the error in un-normalized space.
        # Normalized space has a mean of 0 or close to 0, so small deviations from that can cause huge
        # errors w.r.t. MAPE.

        # Calculate the error as MAPE (mean absolute percentage error).
        # The error is undefined for any ground truth value equal to 0 (since we divide by the ground truth
        # in the error computation).
        # Remove all time points with attr_gt == 0.
        nonzero_idcs = np.intersect1d(np.nonzero(attr_gt), np.nonzero(attr_reco))
        attr_gt = attr_gt[nonzero_idcs]
        attr_reco = attr_reco[nonzero_idcs]
        valid_idcs = (~np.isnan(attr_gt)) & (~np.isnan(attr_reco))
        attr_gt = attr_gt[valid_idcs]
        attr_reco = attr_reco[valid_idcs]

        # Abort calculation if there are no data points left
        if len(attr_gt) == 0 or any(np.isinf(attr_reco)):
            return None

        try:
            # Error (MAPE) is the mean of the relative size of the error w.r.t. the ground truth
            return np.mean(np.abs((attr_gt - attr_reco) / attr_gt))

        except ValueError:
            logging.error(f"MAPE ERROR: attr_gt = {attr_gt}")
            logging.error(f"MAPE ERROR: attr_reco = {attr_reco}")
            return None

    def analyze_bottleneck(self):
        """
        Analyzes the bottleneck
        :return: dictionary with human-readable information about bottleneck
        """

        logging.info("Analyzing bottleneck ...")
        bottleneck_info = {}

        # Feature space dimensionality (for reference)
        features = self.trainer.compute_features(self.split)
        dimensionality = features.shape[1]
        bottleneck_info['dimensionality'] = dimensionality

        # Sparsity (i.e. how many dimensions are unused)
        sparsity = 1 - np.count_nonzero(features) / np.size(features)
        bottleneck_info['sparsity'] = sparsity

        # How information-dense is bottleneck? To find this out, train PCA and examine how many principal components
        # (PCs) are required to explain 99% of the variance
        variance_explained = 0.99
        pca = PCA(
            n_components=variance_explained,
            svd_solver='full'
        )
        pca.fit(features)
        bottleneck_info['pca_features_required'] = int(pca.n_components_)
        bottleneck_info['pca_variance_explained'] = variance_explained

        logging.info("Testing prediction power of feature space ...")

        # Train classifier to predict mortality from feature space

        # Find out ages and survival of admissions and convert to one-hot
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        admission_survived = np.array(static_info['FUTURE_survival'])  # True if alive, False if dead

        # Create training data to fit model for predicting mortality from features
        survival_target = to_categorical(admission_survived.astype(int))  # shape: (num_adm, 2)

        bottleneck_info['mortality_prediction_model'] = self._train_feature_space_classifier(
            features=features,
            target_arr=survival_target,
            class_labels={
                0: 'deceased',
                1: 'survived'
            },
            multi_label=False
        )

        # Train classifier for diagnosis category prediction
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        for icd_attr_name in icd_attrs:
            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[idx]
                                         for idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]

            # Find out the ICD categories for each admission
            present_icd_categs = [[IcdInfo.icd_categ_level_1(icd_kind=icd_attr_name, icd_code=possible_icd_codes[idx])
                                   for idx in adm_icd_indices]
                                  for adm_icd_indices in present_icd_codes_indices]

            # Convert ICD categories for each admission into a multi-hot vector (multi-hot instead of one-hot since
            # every admission can have ICD codes from more than one class)
            icd_categs_uniq = sorted(set(sum(present_icd_categs, [])))
            icd_categs_multi_hot = np.zeros(
                shape=(len(present_icd_categs), len(icd_categs_uniq))
            )
            for idx, adm_categs in enumerate(present_icd_categs):
                adm_categ_indices = [icd_categs_uniq.index(c) for c in adm_categs]
                icd_categs_multi_hot[idx, adm_categ_indices] = 1

            # Train classifier
            bottleneck_info[f'{icd_attr_name}_prediction_model'] = self._train_feature_space_classifier(
                features=features,
                target_arr=icd_categs_multi_hot,
                class_labels={class_idx: categ for (class_idx, categ) in enumerate(icd_categs_uniq)},
                multi_label=True
            )

        return bottleneck_info

    def _train_feature_space_classifier(self, features, target_arr, class_labels, multi_label) -> Dict:
        # Count classes (they might be imbalanced)
        class_counts = np.sum(target_arr, axis=0).astype(int)
        num_classes = len(class_counts)
        num_adm, dimensionality = features.shape

        weight_per_class = num_adm / num_classes
        class_weights = {class_idx: float((1 / class_counts[class_idx]) * weight_per_class)
                         for class_idx in range(num_classes)}

        # Loss function and final layer activation depends on multilabel
        if multi_label:
            # (multilabel means that a single example may have more than one class)
            final_layer_activation = 'sigmoid'
            loss = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            final_layer_activation = 'softmax'
            loss = 'categorical_crossentropy'

        # Create simple model
        num_hidden_layers = 1
        hidden_layer_size = min(
            max(dimensionality // 2, num_classes),
            3 * num_classes
        )
        x = features_input = Input(shape=(dimensionality,))
        for _ in range(num_hidden_layers):
            x = Dense(units=hidden_layer_size, activation='relu')(x)
        mort_pred_output = Dense(num_classes, activation=final_layer_activation)(x)  # output to classification task
        classifier_model = Model(
            inputs=features_input,
            outputs=mort_pred_output
        )

        # Compile the model
        classifier_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        # Split into training and validation data (according to the split we also use for the main model)
        if self.split != io.split_name_all:
            return {
                "result": None,
                "explanation": "This can only be perform the 'all' split since it requires validation and training"
                               " data and uses the same split as the main model."
            }

        # Get official split indices
        all_split = self.trainer.get_split_indices(io.split_name_all)
        val_split = self.trainer.get_split_indices(io.split_name_val)
        train_split = self.trainer.get_split_indices(io.split_name_train)

        # The features we have are ordered with respect to the all_split split. Split them into two arrays according
        # to the validation and training splits WITHIN the all_split split.
        val_indices = self.clustering.index_multi(all_split, val_split)
        train_indices = self.clustering.index_multi(all_split, train_split)

        # Split features
        features_train = features[train_indices, :]
        features_val = features[val_indices, :]
        del features

        # Split targets
        targets_train = target_arr[train_indices, :]
        targets_val = target_arr[val_indices, :]
        del target_arr

        # Fit model to data
        val_split = 0.2
        model_hist = classifier_model.fit(
            x=features_train,
            y=targets_train,
            validation_data=(features_val, targets_val),
            epochs=500000,
            callbacks=[
                EarlyStopping(patience=500, restore_best_weights=True)
            ],
            class_weight=class_weights
        )

        # Generate accuracy report about model (on validation data)
        if not multi_label:
            y_pred_raw = classifier_model.predict(features_val)

            # Generate accuracy report
            y_pred = np.argmax(y_pred_raw, axis=-1)  # take argmax of prediction
            y_ground_truth = np.argmax(targets_val, axis=-1)
            acc_report = classification_report(
                y_ground_truth,
                y_pred,
                output_dict=True
            )

            # Calculate ROC AUC
            y_pred_score = y_pred_raw[:, 1]  # probability of predicting a 1
            try:
                acc_report['roc_auc'] = roc_auc_score(y_ground_truth, y_pred_score)
            except ValueError:
                pass  # this happens if by chance, the target values all belong to one class

        else:
            y_pred = np.round(classifier_model.predict(features_val))  # round prediction
            y_ground_truth = targets_val

            acc_report = {}
            for class_idx in range(num_classes):
                rep = acc_report[class_idx] = {}
                rep['label'] = class_labels[class_idx]
                rep['support'] = int(class_counts[class_idx])

                class_gt = y_ground_truth[:, class_idx]
                class_pred = y_pred[:, class_idx]

                # Recall (how many of the positives did we find?)
                gt_pos_indices = np.where(class_gt == 1)
                num_gt_pos = len(gt_pos_indices[0])
                if num_gt_pos > 0:
                    rep['recall'] = float(sum(class_pred[gt_pos_indices]) / num_gt_pos)
                else:
                    rep['recall'] = float('nan')

                # Precision (how many of those we found were positive?)
                pred_pos_indices = np.where(class_pred == 1)
                num_pred_pos = len(pred_pos_indices[0])
                if num_pred_pos > 0:
                    rep['precision'] = float(sum(class_gt[pred_pos_indices]) / num_pred_pos)
                else:
                    rep['precision'] = float('nan')

                # F1-Score
                if not any(np.isnan([rep['recall'], rep['precision']])) and min([rep['recall'], rep['precision']]) > 0:
                    rep['f1_score'] = 2 / (1 / rep['recall'] + 1 / rep['precision'])
                else:
                    rep['f1_score'] = float('nan')

        # Compile all information

        res_info = {
            'multi_label': multi_label,
            'val_split': val_split,
            'num_epochs': model_hist.epoch[-1],
            **{k: v[-1] for (k, v) in model_hist.history.items()}
        }
        res_info.update(
            {'classes':
                {
                    class_idx: {
                        'label': class_labels[class_idx],
                        'count': int(class_counts[class_idx]),
                        'fraction': float(class_counts[class_idx] / num_adm),
                        'weight': float(class_weights[class_idx])
                    } for class_idx in range(num_classes)
                }
            }
        )
        res_info['accuracy_overall'] = (1 - val_split) * res_info['accuracy'] + val_split * res_info['val_accuracy']
        res_info['accuracy_report'] = acc_report

        return res_info

    def eval_model(self, pipeline_args, after_training=True):
        """
        Evaluates the model and generates a report

        :param pipeline_args:
        :param after_training:
        :return:
        """

        # Load report; abort evaluation if it's already done
        self._load_report()
        eval_already_done = 'after_training' in self.report and self.report['after_training']
        if eval_already_done:
            logging.info("Evaluation of model already done, skipping ...")
            return

        # Validation dictionary will hold all info pertaining to model performance
        logging.info("Starting evaluation of model")
        val_results = {
            'after_training': after_training,
            'args': pipeline_args,  # Add the arguments given to the training pipeline
            'cmd': " ".join(sys.argv)  # Add the training command (this lets me easily replicate search runs)
        }

        if after_training:
            # Get val loss over some samples
            model_eval = self.get_reconstruction_loss()
            val_losses = model_eval['losses']
            loss_stats = {
                'loss_mean': np.mean(val_losses),
                'loss_median': np.median(val_losses),
                'loss_min': np.min(val_losses),
                'loss_max': np.max(val_losses),
            }
            val_results['val_loss'] = loss_stats
            val_results['_eval_result_raw'] = self._eval_result

            # Reconstruction error
            for error_suffix in ["", "_MAPE"]:
                error_key = f'rec_error_scores{error_suffix}'

                for agg_func_name, agg_func in [('median', np.median), ('iqr', scipy.stats.iqr),
                                                ('mean', np.mean), ('std', np.std)]:

                    rec_error_results = {}
                    for item_label, error_scores in model_eval[error_key].items():
                        # Get median of reconstruction error scores for this attribute
                        rec_error_results[item_label] = agg_func(error_scores)
                    val_results[f'rec_error_{agg_func_name}{error_suffix}'] = rec_error_results
                    val_results[f'rec_error_{agg_func_name}_median_same{error_suffix}'] = np.median(list(rec_error_results.values()))
                    val_results[f'rec_error_{agg_func_name}_overall_weighted{error_suffix}'] = agg_func(
                        sum(model_eval[error_key].values(), [])
                    )

            # Analyze the nature of the feature space (bottleneck)
            val_results['bottleneck_info'] = self.analyze_bottleneck()

            if self.trainer.baseline_mode:
                # Save information about the PCA fit
                val_results['baseline_model'] = self.trainer.baseline_eval

            # Save admission mean reconstruction error
            val_results['adm_rec_errors'] = self._eval_result['adm_rec_errors']
            val_results['adms_evaluated'] = self._eval_result['adms_evaluated']

        # Write json file describing the model performance
        self.update_report(new_report_content=val_results)

    def update_report(self, new_report_content, write_to_disk=True):
        # Load report
        self._load_report()

        # Update old report with new content
        self.report.update(new_report_content)

        if write_to_disk:
            io.write_json(self.report, self.eval_report_path, verbose=True, pretty=True)

    def _load_report(self):
        # Open an existing report (if one exists)
        report = io.read_json(self.eval_report_path)
        if report is None:
            report = {}

        # Save the report (for access by other modules)
        self.report = report

        # Save model eval result
        if '_eval_result_raw' in self.report:
            self._eval_result = self.report['_eval_result_raw']

    @staticmethod
    def _statistical_test_distributions(medical_values, inside_cluster_indices):
        """
        Performs a statistical test as a comparison of value distribution in clusters vs.
        outside clusters
        :param medical_values:
        :param inside_cluster_indices:
        :return: {cluster_label: test results for cluster_label in clustering_info}
        """

        test_results = {}
        for cluster_label, cluster_indices in inside_cluster_indices.items():
            # Filter to values of _this_ cluster
            inside_cluster_values = [v for (idx, v) in enumerate(medical_values) if idx in cluster_indices]
            outside_cluster_values = [v for (idx, v) in enumerate(medical_values) if idx not in cluster_indices]
            # (subsets I test against one another must be disjoint)

            # Remove None entries (these are introduced by aggregation when the admission has no entries)
            inside_cluster_values = [v for v in inside_cluster_values if v is not None]
            outside_cluster_values = [v for v in outside_cluster_values if v is not None]

            # Only perform test if cluster is not all-encompassing (or we have no data of inside the cluster)
            if len(inside_cluster_values) == 0 or len(outside_cluster_values) == 0:
                p_value = 1.0
            else:
                # Perform test
                _, p_value = ranksums(
                    inside_cluster_values,
                    outside_cluster_values
                )
                # Note: Test is symmetric w.r.t. input values, i.e. if there are only two clusters, both will have
                # the same p value

            # Record results
            test_results[int(cluster_label)] = p_value  # If this is high, it means that the probability of cluster and
            # full population being the same is high (more or less)

        return test_results

    def _fishers_exact_test(self, cluster_report):
        logging.info("Performing Fisher's exact test for static categorical data...")
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        static_categorical = [static_categorical[idx] for idx in self.trainer.get_split_indices(self.split)]
        # (this is especially nice for list-like attributes like ICD codes, which are not part of the KS test)

        # Copy the data for evaluation so that other parts of the pipeline are not affected by data transformations
        # and changes
        logging.info(" - Copying static categorical data")
        static_categorical = [copy.deepcopy(stat) for stat in static_categorical]

        # Ensure every entry is actually a list (this step is necessary since sometimes, single-entry lists of
        # e.g. str will be stored as str instead of list of str)
        for stat in static_categorical:
            for stat_name, stat_vals in stat.items():
                if type(stat_vals) != list:
                    stat_vals = [stat_vals]
                stat[stat_name] = stat_vals

        # Replace value indices by actual values (e.g. for gender, 'M' and 'F' are coded as 0 and 1)
        logging.info(" - Replacing value indices by actual values")
        static_categ_val_labels = self.preprocessor.get_static_data_categ_values()
        for stat in static_categorical:
            for stat_name, stat_vals in stat.items():
                stat[stat_name] = [static_categ_val_labels[stat_name][v_idx] for v_idx in stat_vals]

        # static_categorical is a list of dicts. We instead want a single dict which has lists as values
        logging.info(" - Restructuring static categorical data")
        static_categ_data = {}
        for stat in static_categorical:
            for stat_name, stat_vals in stat.items():
                if stat_name not in static_categ_data:
                    static_categ_data[stat_name] = []
                static_categ_data[stat_name].append(stat_vals)

        # Add aggregated attribute types for ICD diagnoses and  procedures
        logging.info(" - Adding aggregated attribute types for ICD diagnoses and procedures")
        for icd_categ_name in ['icd9_code_diagnoses', 'icd9_code_procedures']:
            # Get original ICD codes
            icd_codes_lists = static_categ_data[icd_categ_name]

            # Retrieve ICD group for each of the entries
            icd_codes_lists_grouped = [[IcdInfo.icd_categ_level_1(icd_categ_name, c) for c in cl]
                                       for cl in icd_codes_lists]

            # Remove duplicate entries in each list that were introduced by the grouping
            icd_codes_lists_grouped = [list(set(cl)) for cl in icd_codes_lists_grouped]

            # Save aggregated list under a new attribute name
            static_categ_data[f'{icd_categ_name}_grouped'] = icd_codes_lists_grouped

        # Now we are ready for Fisher's exact test
        logging.info(" - Performing Fisher's test")
        fisher_test_results = {}
        for static_categ_name, static_categ_vals in static_categ_data.items():
            fisher_test_results[static_categ_name] = {}

            # Do Fisher's test for each of the observed values of this categorical attribute
            static_categ_vals_observed = list(np.unique(
                sum(static_categ_vals, [])
            ))

            for static_categ_val_obs in static_categ_vals_observed:
                obs_results = fisher_test_results[static_categ_name][static_categ_val_obs] = {}

                for clustering_info in self.clustering.clusterings:
                    obs_results[clustering_info.random_id] = self._fishers_exact_test_internal(
                        static_categ_vals=static_categ_vals,
                        static_categ_target_val=static_categ_val_obs,
                        clustering_info=clustering_info
                    )

        # Add Fisher's test to report
        cluster_report['static_categorical'] = {
            'fisher': fisher_test_results
        }
        logging.info("Fisher's exact test for static categorical data done!")

    @staticmethod
    def _fishers_exact_test_internal(static_categ_vals, static_categ_target_val, clustering_info):
        """
        Perform Fisher's exact test to determine if differences in distributions of list-like attributes between
        clusters are random or significant.
        :param static_categ_vals: list of lists; categorical values for each of the admissions
        :param static_categ_target_val: categorical value to test for
        :param clustering_info:
        :return:
        """

        # Do test for each cluster
        test_results = {}
        for cluster_label in np.unique(clustering_info.labels):
            # Index groups of admissions
            indices_inside_cluster = np.where(clustering_info.labels == cluster_label)[0]
            indices_outside_cluster = np.where(clustering_info.labels != cluster_label)[0]
            indices_has_target = [idx for idx in range(len(static_categ_vals))
                                  if static_categ_target_val in static_categ_vals[idx]]
            indices_has_target_not = [idx for idx in range(len(static_categ_vals))
                                      if static_categ_target_val not in static_categ_vals[idx]]

            # Count occurrences of target value inside and outside cluster
            count_has_target_inside_cluster = len([idx for idx in range(len(static_categ_vals))
                                                   if idx in indices_has_target and idx in indices_inside_cluster])
            count_has_target_outside_cluster = len([idx for idx in range(len(static_categ_vals))
                                                    if idx in indices_has_target and idx in indices_outside_cluster])
            count_has_not_target_inside_cluster = len([idx for idx in range(len(static_categ_vals))
                                                       if idx in indices_has_target_not
                                                       and idx in indices_inside_cluster])
            count_has_not_target_outside_cluster = len([idx for idx in range(len(static_categ_vals))
                                                        if idx in indices_has_target_not
                                                        and idx in indices_outside_cluster])

            # Construct contingency table
            contingency_table = np.array(
                [[count_has_target_inside_cluster, count_has_target_outside_cluster],
                 [count_has_not_target_inside_cluster, count_has_not_target_outside_cluster]]
            )

            # Perform Fisher's test
            _, p_value = fisher_exact(contingency_table)

            # Record results
            test_results[int(cluster_label)] = p_value  # If this is low, it means categorical value might be
            # distributed differently between inside and outside cluster

        return test_results

    @staticmethod
    def _mine_frequent_itemsets(itemsets: List[List], min_sup: float = 0.05, remove_length_1_itemsets=False) \
            -> pd.DataFrame:
        """
        Mines frequent itemsets
        :param itemsets:
        :param min_sup:
        :param remove_length_1_itemsets:
        :return: frequent_itemsets
        """
        # Convert list of items into one-hot encoded dataframe
        try:
            te = TransactionEncoder()
            te_ary = te.fit(itemsets).transform(itemsets, sparse=True)
            df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
        except ValueError:
            return pd.DataFrame([], columns=[cn_support, cn_itemsets])

        # Mine frequent itemsets
        frequent_itemsets = fpmax(
            df,
            min_support=min_sup,
            use_colnames=True
        )

        if remove_length_1_itemsets:
            # Remove trivial (length 1) itemsets
            frequent_itemsets[cn_length] = frequent_itemsets[cn_itemsets].str.len()
            frequent_itemsets = frequent_itemsets[frequent_itemsets[cn_length] > 1]
            frequent_itemsets = frequent_itemsets.drop(columns=[cn_length])

        return frequent_itemsets

    @staticmethod
    def _train_decision_trees(target_labels: np.ndarray, features: np.ndarray) \
            -> List[Tuple[tree.DecisionTreeClassifier, float, Dict[int, int]]]:
        """
        Trains a decision tree and returns it
        :param target_labels:
        :param features: must be of shape (n_samples, n_features)
        :return: (trained tree, fit score)
        """

        # Train trees in different configurations
        logging.info(f"Training decision trees on features ({features.shape[0]} samples, {features.shape[1]} features)"
                     f" ...")

        trees = []
        for max_depth in [6, 10]:
            for min_impurity_decrease in 1 / 2 ** np.arange(4, 8 + 1):
                clf = tree.DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_impurity_decrease=min_impurity_decrease,
                    min_samples_split=50  # Only split if at least this number of admissions in node
                )
                clf = clf.fit(features, target_labels)

                # Discard tree if it only consists of the root
                if clf.get_depth() < 1:
                    continue

                # Test if tree has same distributions of classes as any previous tree
                # (if it has, we don't need it)
                tree_seen_before = False
                for prev_tree, _, _ in trees:
                    tree_seen_before = tree_seen_before or np.all(prev_tree.tree_.value == clf.tree_.value)
                    if tree_seen_before:
                        break
                if tree_seen_before:
                    continue

                # Test how well the tree fits the data
                score = clf.score(features, target_labels)

                # Determine the counts of different target labels (in the dataset used for the tree - this is used for
                # explaining the meaning of class labels in the plot)
                class_counts = {c: np.count_nonzero(target_labels == c) for c in clf.classes_}

                trees.append((clf, score, class_counts))

        logging.info("Training decision trees done!")

        return trees

    def decision_tree_analysis(self, static_info: Dict[str, List], dyn_info: Dict[str, List[List]],
                               dyn_trend_info: Dict[str, List[List]]) -> Dict[str, Dict[str, List[TreeAnalysisResult]]]:
        """
        Trains a decision tree for each clustering that tries to predict the clustering labels based on medical
        data of the corresponding admissions. The purpose is to find reasonable explanations for why clusters separate.

        :param static_info:
        :param dyn_info:
        :param dyn_trend_info:
        :return:
        """

        # Prepare data for decision trees
        (
            (dyn_stats_names, dyn_stats_cols),
            (dyn_derived_names, dyn_derived_cols),
            (static_names, static_cols),
            (icd_diagnosis_names, icd_diagnosis_cols),
            (icd_procedure_names, icd_procedure_cols)
        ) = self.preprocessor.prepare_input_data_for_matrix(
            static_info=static_info,
            dyn_info=dyn_info,
            dyn_trend_info=dyn_trend_info,
            split=self.split,
            with_future_info=True  # Also include "future" information such as the patient's survival
        )

        analysis_results = {}  # type: Dict[str, Dict[str, List[TreeAnalysisResult]]]
        for data_selection in ['all', 'icd', 'icd_diagnoses', 'icd_procedures', 'static',
                               'dyn_stats', 'dyn_derived', 'dyn']:

            # Compile the data
            if data_selection == 'all':
                feature_names = dyn_stats_names + dyn_derived_names + static_names + icd_diagnosis_names + \
                                icd_procedure_names
                feature_columns = dyn_stats_cols + dyn_derived_cols + static_cols + icd_diagnosis_cols + \
                                  icd_procedure_cols
            elif data_selection == 'icd':
                feature_names = icd_diagnosis_names + icd_procedure_names
                feature_columns = icd_diagnosis_cols + icd_procedure_cols
            elif data_selection == 'icd_diagnoses':
                feature_names = icd_diagnosis_names
                feature_columns = icd_diagnosis_cols
            elif data_selection == 'icd_procedures':
                feature_names = icd_procedure_names
                feature_columns = icd_procedure_cols
            elif data_selection == 'static':
                feature_names = static_names
                feature_columns = static_cols
            elif data_selection == 'dyn_stats':
                feature_names = dyn_stats_names
                feature_columns = dyn_stats_cols
            elif data_selection == 'dyn_derived':
                feature_names = dyn_derived_names
                feature_columns = dyn_derived_cols
            elif data_selection == 'dyn':
                feature_names = dyn_stats_names + dyn_derived_names
                feature_columns = dyn_stats_cols + dyn_derived_cols
            else:
                assert False, f"Tree Analysis Data Selection '{data_selection}' unknown!"

            # Make sure all data columns have the same lengths
            col_lengths = [len(f) for f in feature_columns]
            if len(np.unique(col_lengths)) != 1:
                for f_len, f_name in zip(col_lengths, feature_names):
                    logging.info(f"Length {f_len} for feature column {f_name}")
                assert False, "Not all feature columns have the same length!"

            # Clean data: The decision tree is not able to deal with NaN entries. Thus, fill up NaN entries with a
            # value that doesn't otherwise occur in the data.
            feature_columns, nan_fill_value = self._nan_filling(feature_columns)

            # Compose into one large array
            feature_array = np.stack(feature_columns).T
            # (shape (n_samples, n_features))

            # Run analysis for every clustering
            selection_analysis_results = {}  # type: Dict[str, List[TreeAnalysisResult]]
            for clustering_info in self.clustering.clusterings:
                # Train trees
                tree_training_results = self._train_decision_trees(
                    target_labels=clustering_info.labels,
                    features=feature_array
                )

                # Save trees
                selection_analysis_results[clustering_info.random_id] = [
                    TreeAnalysisResult(tree=tree_trained, score=tree_score, feature_names=feature_names,
                                       class_counts=tree_class_counts, nan_fill_value=nan_fill_value)
                    for (tree_trained, tree_score, tree_class_counts) in tree_training_results]
            analysis_results[data_selection] = selection_analysis_results

        return analysis_results

    @staticmethod
    def _nan_filling(feature_columns: List[List]) -> Tuple[List[List], np.float]:
        """
        Fills NaN values up with a different float value that is not present in the data. For usage with decision trees,
        which can not deal with NaN values.

        :param feature_columns:
        :return:
        """

        # Find global minimum of data
        m = np.nanmin(feature_columns)

        # Go a bit smaller still
        m = np.floor(m - 1)

        # Replace all NaNs with the new filler value
        for col in feature_columns:
            for row_idx, val in enumerate(col):
                if np.isnan(val):
                    col[row_idx] = m

        return feature_columns, m

    def frequent_itemset_analysis(self, clusterings: List[ClusteringInfo], num_clusters_per_clustering: int = 500,
                                  icd_path_depth: int = 3) -> Dict[str, FrequentItemsetResults]:
        """
        Mine frequent itemsets in both the full population and each of the clusters in every clustering
        :param clusterings: Clusterings to perform the analysis on
        :param num_clusters_per_clustering:
        :param icd_path_depth: Depth of the ICD tree traversal for each individual code, e.g. depth 1 is the categories.
         Usually, the deepest depth is depth 6 (the tree has different depths at different regions).
        :return: Dictionary of results with the ICD attribute name as the key and FrequentItemsetResults as value
        """

        # Extract ICD attributes and data
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr]

        itemsets_results = {}
        for icd_attr_name in icd_attrs:
            logging.info(f"Starting ICD ({icd_attr_name.upper()}) itemset mining, depth {icd_path_depth} ...")

            # Find out ICD codes (or rather, their indices) for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[idx]
                                         for idx in self.trainer.get_split_indices(self.split)]

            # Find actual codes (as nodes in the ICD tree)
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            pop_icd_codes_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                    for idx in adm_idcs]
                                   for adm_idcs in present_icd_codes_indices]

            # Find out the higher-level ICD codes (according to icd_path_depth argument)
            pop_icd_codes_nodes = [[IcdInfo.name_for_node(n.path[min(icd_path_depth, len(n.path) - 1)])
                                    for n in adm_nodes]
                                   for adm_nodes in pop_icd_codes_nodes]

            # Mine frequent itemsets in the full population
            pop_itemsets = self._mine_frequent_itemsets(
                itemsets=pop_icd_codes_nodes
            )

            clusterings_results = {}
            significance_results = {}
            num_clusterings = len(clusterings)
            for cluster_idx, clustering_info in enumerate(clusterings):

                logging.info(f"Mining clustering {cluster_idx + 1} of {num_clusterings}"
                             f" ({clustering_info.random_id}) ...")

                # Sort the cluster labels by their count (we can only afford to analyze the biggest clusters
                # for each clustering)
                cluster_labels, cluster_counts = np.unique(clustering_info.labels, return_counts=True)
                cluster_labels = cluster_labels[np.argsort(cluster_counts)[::-1]]
                cluster_labels = cluster_labels[:num_clusters_per_clustering]

                c_id = clustering_info.random_id
                clusterings_results[c_id] = {}
                significance_results[c_id] = {}
                raw_items = {}
                for cluster_label in cluster_labels:
                    # Constrain ICD sets to admissions inside each of the clusters
                    cluster_indices = np.where(clustering_info.labels == cluster_label)[0]
                    cluster_icd_codes_nodes = [pop_icd_codes_nodes[idx]
                                               for idx in cluster_indices]
                    raw_items[cluster_label] = cluster_icd_codes_nodes

                    # Mine frequent itemsets in cluster
                    cluster_itemsets = self._mine_frequent_itemsets(
                        itemsets=cluster_icd_codes_nodes
                    )
                    clusterings_results[clustering_info.random_id][cluster_label] = cluster_itemsets

                # Different itemsets are frequent in each of the clusters. In order to extract useful information, we
                # want to know the frequency of all of these itemsets in all the clusters (and even inside the
                # population). We will count the support of itemsets in the different clusters (if that itemset is
                # not frequent in the cluster already).

                # Determine itemsets that occur in at least one of the clusters inside this clustering
                clustering_itemsets_union = set().union(*[set(df[cn_itemsets].values)
                                                          for df in clusterings_results[c_id].values()])

                # Also add the population frequent itemsets
                clustering_itemsets_union = clustering_itemsets_union.union(set(pop_itemsets[cn_itemsets].values))

                # Also add each individual item as a trivial itemset - this allows the analysis of diseases in
                # separation
                all_observed_items = set().union(*[itemset for itemset in clustering_itemsets_union])
                singleton_itemsets = [frozenset({item}) for item in all_observed_items]
                clustering_itemsets_union = clustering_itemsets_union.union(singleton_itemsets)

                # For each cluster itemset table (and for the population table), go through each of the union itemsets
                # and add its support (by counting) if it is not already present
                for cluster_key in clusterings_results[c_id].keys():
                    cluster_itemsets = clusterings_results[c_id][cluster_key]
                    cluster_itemsets = self._count_all_itemsets(
                        cluster_itemsets,
                        raw_items[cluster_key],
                        clustering_itemsets_union
                    )
                    clusterings_results[c_id][cluster_key] = cluster_itemsets
                pop_itemsets = self._count_all_itemsets(pop_itemsets, pop_icd_codes_nodes, clustering_itemsets_union)

                # Perform statistical tests for each itemset and cluster - calculate the total number of tests here in
                # order to perform Bonferroni correction of the results
                num_itemsets = len(clustering_itemsets_union)
                num_clusters = len(cluster_labels)
                num_tests_total = num_itemsets * num_clusters
                significance_alpha = 0.01
                significance_level = significance_alpha / num_tests_total
                significance_results[c_id] = SignificanceResults(
                    alpha=significance_alpha,
                    num_itemsets=num_itemsets,
                    num_clusters=num_clusters,
                    num_tests_total=num_tests_total,
                    significance_level=significance_level
                )

                # For each cluster, analyze if the itemset is more or less frequent in the cluster vs. the complement
                for cluster_label in cluster_labels:
                    # Constrain ICD sets to admissions OUTSIDE the cluster
                    complement_indices = np.where(clustering_info.labels != cluster_label)[0]
                    complement_icd_codes_nodes = [pop_icd_codes_nodes[idx]
                                                  for idx in complement_indices]

                    # Count itemset support in complement
                    cluster_itemsets = clusterings_results[c_id][cluster_label]
                    complement_itemsets = pd.DataFrame(columns=[cn_complement_support, cn_itemsets])  # empty at first
                    complement_itemsets = self._count_all_itemsets(
                        complement_itemsets,
                        complement_icd_codes_nodes,
                        set(cluster_itemsets[cn_itemsets].values)
                    )

                    # Save the complement support in the cluster's DataFrame
                    cluster_itemsets = cluster_itemsets \
                        .set_index(cn_itemsets) \
                        .join(complement_itemsets.set_index(cn_itemsets)) \
                        .reset_index()
                    clusterings_results[c_id][cluster_label] = cluster_itemsets

                    # Calculate the factor between the frequency in the cluster and the frequency in the complement
                    cluster_itemsets[cn_factor] = cluster_itemsets[cn_support] / cluster_itemsets[cn_complement_support]

                    # For each itemset perform a statistical test to see if it is *significantly* more
                    # or less present in the cluster than outside the cluster

                    # Find out cluster size (we need it for calculating the contingency table)
                    cluster_size = np.count_nonzero(clustering_info.labels == cluster_label)
                    complement_size = len(clustering_info.labels) - cluster_size

                    # Perform test for every itemset
                    p_values = []
                    for _, itemsets_row in cluster_itemsets.iterrows():
                        # Build contingency table
                        fraction_with_code_in_cluster = itemsets_row[cn_support]
                        num_with_code_in_cluster = np.round(fraction_with_code_in_cluster * cluster_size)
                        num_without_code_in_cluster = cluster_size - num_with_code_in_cluster

                        fraction_with_code_outside_cluster = itemsets_row[cn_complement_support]
                        num_with_code_outside_cluster = np.round(
                            fraction_with_code_outside_cluster * complement_size)
                        num_without_code_outside_cluster = complement_size - num_with_code_outside_cluster

                        contingency_table = np.array([[num_with_code_in_cluster, num_with_code_outside_cluster],
                                                      [num_without_code_in_cluster,
                                                       num_without_code_outside_cluster]])

                        # Execute Fisher's exact test
                        _, p_value = fisher_exact(contingency_table)
                        # (p value = probability of observing this distribution by chance even if cluster has the same
                        # distribution same as the complement. In other words, a low p value tells us that the cluster
                        # is likely different from its complement.)
                        p_values.append(p_value)

                    # Determine if the p value is significant for each p value
                    significances = [p_val <= significance_level for p_val in p_values]

                    # Save both columns in the table
                    cluster_itemsets[cn_p_value] = p_values
                    cluster_itemsets[cn_significance] = significances

            # Save results
            itemsets_results[icd_attr_name] = FrequentItemsetResults(
                pop_itemsets=pop_itemsets,
                clusterings_results=clusterings_results,
                significance_results=significance_results,
                icd_depth=icd_path_depth
            )

        return itemsets_results

    @staticmethod
    def _count_all_itemsets(frequent_items_df: pd.DataFrame, itemsets: List[List], itemsets_to_count: Set) \
            -> pd.DataFrame:
        # Collect counted support for new (previously non-frequent) itemsets in a list
        counted_itemsets = []
        num_adm = len(itemsets)
        for it_set in itemsets_to_count:
            if it_set not in frequent_items_df[cn_itemsets].values:
                # Count admissions that have the itemset it_set
                supporting_admissions = [adm_set for adm_set in itemsets if all(it in adm_set for it in it_set)]

                # Support is the fraction of itemsets that contain the itemset in question
                support = len(supporting_admissions) / num_adm
                counted_itemsets.append([support, it_set])

        # Concatenate the new counted itemsets onto the existing frequent itemset table
        counted_itemsets_df = pd.DataFrame(data=counted_itemsets, columns=frequent_items_df.columns)
        frequent_items_df = frequent_items_df.append(counted_itemsets_df)

        return frequent_items_df

    def write_frequent_itemsets(self, icd_attribute_name: str, frequent_itemsets_results: FrequentItemsetResults,
                                clustering_info: ClusteringInfo):
        """
        Writes frequent itemset mining results about a single clustering to a text file

        :param icd_attribute_name: Name of the ICD attribute
        :param frequent_itemsets_results:
        :param clustering_info:
        :return:
        """

        # Get the mining results for the clustering to be plotted
        cluster_mining_orig = frequent_itemsets_results.clusterings_results[clustering_info.random_id]
        significance_results = frequent_itemsets_results.significance_results[clustering_info.random_id]

        # Index all tables using the frequent itemsets and rename their support columns to be unique
        cluster_keys = {}
        complement_keys = {}
        factor_keys = {}
        p_value_keys = {}
        significance_keys = {}
        cluster_mining = {}
        for cluster_label, cluster_df in cluster_mining_orig.items():
            # Index via itemset
            cluster_df = cluster_df.set_index(cn_itemsets)

            # Rename cluster-agnostic column names so they contain the cluster value
            cluster_key = f"{io.label_for_cluster_label(cluster_label)} Sup"
            complement_key = f"{io.label_for_cluster_label(cluster_label)} Sup Complement"
            factor_key = f"{io.label_for_cluster_label(cluster_label)} Factor"
            p_value_key = f"{io.label_for_cluster_label(cluster_label)} P-Value"
            significance_key = f"{io.label_for_cluster_label(cluster_label)} Significance " \
                               f"(level = {significance_results.alpha:0.2f} / {significance_results.num_tests_total})"
            cluster_df = cluster_df.rename(
                columns={
                    cn_support: cluster_key,
                    cn_complement_support: complement_key,
                    cn_factor: factor_key,
                    cn_p_value: p_value_key,
                    cn_significance: significance_key
                }
            )

            cluster_keys[cluster_label] = cluster_key
            complement_keys[cluster_label] = complement_key
            factor_keys[cluster_label] = factor_key
            p_value_keys[cluster_label] = p_value_key
            significance_keys[cluster_label] = significance_key
            cluster_mining[cluster_label] = cluster_df

        # Join tables on frequent itemset index
        sup_df = None
        for cluster_df in cluster_mining.values():
            if sup_df is None:
                sup_df = cluster_df
            else:
                if len(cluster_df) > 0:
                    sup_df = sup_df.join(cluster_df)

        # Join population support onto cluster table
        pop_df = frequent_itemsets_results.pop_itemsets
        pop_support_key = "Population Sup"
        pop_df = pop_df.set_index(cn_itemsets)
        pop_df = pop_df.rename(columns={cn_support: pop_support_key})
        sup_df = sup_df.join(pop_df)  # this join leaves out the rows of pop_df without corresponding index in sup_df

        # Sort rows by population support
        sup_df = sup_df.sort_values(pop_support_key, ascending=False)

        # Sort table columns (first the population, then the clusters)
        cols = [col for col in sup_df.columns.values if col != pop_support_key]
        cols.sort(key=lambda col: (int(col.split(" ")[1]), 12))
        cols = [pop_support_key] + cols
        sup_df = sup_df[cols]

        # Convert frozenset itemsets to string representations
        def sets_to_str(itemset):
            # Remove semicolons in itemset names (this is necessary to preserve integrity when writing to a CSV file)
            its = [it.replace(";", "") for it in itemset]

            # Join into a string with newlines separating the items
            if len(its) == 1:
                is_str = its[0]
            else:
                is_str = f"({len(itemset)}) [" + ", \n".join(it for it in its) + "]"

            return is_str

        sup_df.index = sup_df.index.map(sets_to_str)

        # Export CSV file
        itemsets_results_dir = os.path.join(
            self.eval_dir,
            clustering_info.get_path_name(),
            "icd_mining",
            f"depth_{frequent_itemsets_results.icd_depth}"
        )
        io.makedirs(itemsets_results_dir)
        plot_path = os.path.join(
            itemsets_results_dir,
            f"freq_itemsets_{icd_attribute_name}_depth{frequent_itemsets_results.icd_depth}"
        )
        plot_path_base, _ = os.path.splitext(plot_path)  # Remove default png file extension
        csv_path = plot_path_base + ".csv"
        sup_df.to_csv(csv_path, quoting=csv.QUOTE_NONNUMERIC)

        # Save table as text file
        table_str = tabulate(sup_df, tablefmt="fancy_grid")
        plot_path = plot_path_base + ".txt"
        logging.info(f"Saving frequent itemset mining results to {plot_path} ... ")
        io.write_txt(table_str, plot_path)

        # Save a special (filtered) file for each cluster
        clusters_analysis_dir = os.path.join(os.path.split(plot_path_base)[0], "clusters")
        io.makedirs(clusters_analysis_dir)

        cluster_dfs_by_cluster = {}
        cluster_dfs_by_cluster_full = {}
        for cluster_label, cluster_key in cluster_keys.items():
            complement_key = complement_keys[cluster_label]
            factor_key = factor_keys[cluster_label]
            p_val_key = p_value_keys[cluster_label]
            significance_key = significance_keys[cluster_label]

            # Restrict columns to population column and the columns for this cluster
            cluster_df = sup_df.filter(items=[pop_support_key, cluster_key, complement_key, factor_key,
                                              p_val_key, significance_key])

            # Sort by support
            cluster_df = cluster_df.sort_values(cluster_key, ascending=False)

            # Filter to significantly enriched itemsets
            cluster_df_full = cluster_df
            cluster_df = cluster_df[cluster_df[significance_key]]

            # Save to file
            cluster_analysis_path = os.path.join(
                clusters_analysis_dir,
                f"itemsets_filtered_{icd_attribute_name}_d{frequent_itemsets_results.icd_depth}"
                f"_cluster{cluster_label}"
            )
            if len(cluster_df) > 0:
                cluster_df.to_csv(cluster_analysis_path + ".csv", quoting=csv.QUOTE_NONNUMERIC)

            # Collect the information to be stored in the cluster summary table
            cluster_dfs_by_cluster[cluster_label] = cluster_df

            # Also store the unfiltered version of the table - parts of it might be shown in the cluster summary table
            # if there are no frequent itemsets
            cluster_dfs_by_cluster_full[cluster_label] = cluster_df_full

        # Store ICD information in the cluster summary table
        self._store_icd_itemsets_in_summary_table(
            icd_dfs_by_cluster=cluster_dfs_by_cluster,
            icd_dfs_by_cluster_full=cluster_dfs_by_cluster_full,
            sup_df=sup_df,
            summary_table=self.clustering.get_summary_table(clustering_info=clustering_info),
            icd_attr_name=io.prettify(icd_attribute_name),
            icd_depth=frequent_itemsets_results.icd_depth
        )

    @staticmethod
    def _store_icd_itemsets_in_summary_table(icd_dfs_by_cluster: Dict[int, pd.DataFrame],
                                             icd_dfs_by_cluster_full: Dict[int, pd.DataFrame],
                                             sup_df: pd.DataFrame,
                                             summary_table: pd.DataFrame,
                                             icd_attr_name: str,
                                             icd_depth: int):
        # Define a function for translating frequent itemset tables into strings to be inserted into the cluster
        # summary table
        def freq_itemset_tables_to_str(itemsets_df, num_entries=7, max_entries=20, population=False):
            # Prepare the output lines list
            lines = []

            # If we are printing the population itemsets, there is no need to check for significance
            if not population:
                # Find out the number of enriched itemsets
                significance_cols = [col_name for col_name in itemsets_df.columns
                                     if cn_significance.capitalize() in col_name]
                significance_col_name = significance_cols[0]
                num_significant_itemsets = len(itemsets_df[itemsets_df[significance_col_name]])

                if num_significant_itemsets == 0:
                    # Display a disclaimer if there are no enriched itemsets
                    lines.append("(not *frequent*, but prevalent)")
                else:
                    # Extend the number of printed itemsets such that it prints at least until the last enriched itemset
                    last_enriched_row_index = itemsets_df.index[itemsets_df[significance_col_name]][-1]
                    last_enriched_idx = itemsets_df.index.get_loc(last_enriched_row_index)
                    num_entries = min(max(num_entries, last_enriched_idx + 1), max_entries)

            # Find out name of support column
            sup_cols = [col_name for col_name in itemsets_df.columns if col_name.endswith("Sup")]
            if population:
                sup_col_name = sup_cols[0]  # population support
            else:
                sup_col_name = sup_cols[-1]  # cluster support

            # Find out name of factor column
            factor_col_name = sup_col_name.replace("Sup", cn_factor.capitalize())

            # Sort by support column
            itemsets_df = itemsets_df.sort_values(sup_col_name, ascending=False)

            # Limit the number of entries
            itemsets_df = itemsets_df[:num_entries]

            # Generate all lines of the description one at a time
            for itemset_str, itemset_info in itemsets_df.iterrows():

                # Show enrichment direction
                if population:
                    enrichment_dir = ""
                    enrichment_factor_str = ""
                else:
                    # Mark the itemset as positively or negatively enriched,
                    # e.g. "++ DISEASES OF THE CIRCULATORY SYSTEM (390-459)"
                    is_significant = itemset_info[significance_col_name]
                    enrichment_factor = itemset_info[factor_col_name]
                    if is_significant:
                        enrichment_dir = "-- " if enrichment_factor < 1 else "++ "
                    else:
                        enrichment_dir = "~ "

                    # Show the factor between cluster and complement support
                    enrichment_factor_str = f"{100 * (enrichment_factor - 1):0.1f}%"
                    enrichment_factor_str = ("+" if enrichment_factor > 1 else "") + enrichment_factor_str
                    # (e.g. 0.9 -> -10% or 1.3427 -> +34.3%)
                    enrichment_factor_str = f", {enrichment_factor_str}"

                sup = itemset_info[sup_col_name]
                lines.append(f"{enrichment_dir}{itemset_str} ({100 * sup:0.1f}%{enrichment_factor_str})")

            text = ",\n".join(lines)
            return text

        # Collect statistics population and for each of the clusters
        icd_col = [
            freq_itemset_tables_to_str(sup_df, population=True)
        ]
        for cluster_label in sorted(icd_dfs_by_cluster.keys()):
            cluster_df_full = icd_dfs_by_cluster_full[cluster_label]
            icd_col.append(freq_itemset_tables_to_str(cluster_df_full))

        # Save ICD information in summary table
        table_col_name = f"{icd_attr_name} (Depth {icd_depth})"
        summary_table[table_col_name] = icd_col

    def write_all_frequent_itemset_results(self, clusterings: List[ClusteringInfo],
                                           frequent_itemset_results_all_depths):
        for max_icd_depth, freq_res in frequent_itemset_results_all_depths.items():

            for icd_attr_name, frequent_itemset_results in freq_res.items():

                # Write tables for each of the clusterings
                for clustering_info in clusterings:
                    self.write_frequent_itemsets(
                        icd_attribute_name=icd_attr_name,
                        frequent_itemsets_results=frequent_itemset_results,
                        clustering_info=clustering_info
                    )

        # Now that they are complete, write cluster summary tables to disk
        self.write_clustering_tables(clusterings=clusterings)

    def write_clustering_tables(self, clusterings: List[ClusteringInfo]):
        for clustering_info in clusterings:
            summary_table_path = os.path.join(
                self.clustering.clusterings_dir,
                f"summary_{clustering_info.get_path_name()}.csv"
            )
            self.clustering.get_summary_table(clustering_info=clustering_info) \
                .to_csv(summary_table_path, quoting=csv.QUOTE_NONNUMERIC)

    @staticmethod
    def apply_agg_func(values_admissions, f):
        """
        Applies an aggregation functions to a list of lists of values.
        :param values_admissions:
        :param f: Function which aggregates data, i.e. returns a single value for a list of values
        :return: aggregated value for each list, None for each empty list
        """
        return [f(lis) if len(lis) > 0 else None for lis in values_admissions]

    def frequent_itemsets_analysis_and_tables(self, clusterings: List[ClusteringInfo], write_tables: bool):
        """
        Perform frequent itemset analysis for different depths of the ICD tree. Afterwards, write tables of itemset
        information to disk.
        :param clusterings:
        :param write_tables:
        :return:
        """

        freq_itemset_mining = {}
        for icd_tree_max_depth in range(1, 4 + 1):
            freq_itemset_mining[icd_tree_max_depth] = self.frequent_itemset_analysis(
                clusterings=clusterings,
                icd_path_depth=icd_tree_max_depth
            )

        # Write the tables with results to disk
        if write_tables:
            self.write_all_frequent_itemset_results(
                clusterings=clusterings,
                frequent_itemset_results_all_depths=freq_itemset_mining
            )

        return freq_itemset_mining

    @staticmethod
    def extract_p_values_from_freq_itemset_results(freq_itemset_res_all_depths):
        p_values = []
        for depth_res in freq_itemset_res_all_depths.values():
            for icd_kind_res in depth_res.values():
                for clustering_res in icd_kind_res.clusterings_results.values():
                    for cluster_df in clustering_res.values():
                        p_values += list(cluster_df[cn_p_value])
        return p_values

    def perform_random_sampling_p_value_analysis(self, max_num_iterations: int = 1000, max_hours_runtime=24) \
            -> List[List[float]]:
        """
        Analyze frequent itemset p values in shuffled versions of the real clusterings. This step in the evaluation
        pipeline serves to confirm that the patterns observed in the real clusterings are not due to chance.

        :param max_num_iterations:
        :param max_hours_runtime: Maximum runtime in hours. The number of iterations will be limited such that this
        time is not exceeded (unless a single iterations takes more than the allotted time).

        :return: List of lists of p values; one list for each iteration
        """

        # Permutations will always use the real clusterings as the basis
        original_clusterings = self.clustering.clusterings

        # Calculate time budget in seconds (at least one iteration is performed, regardless of budget)
        total_budget_seconds = int(np.floor(max_hours_runtime * 60 * 60))

        # Function that shuffles the labelings of clusterings. This will not alter the sizes of clusters, but totally
        # remove any connection between cluster assignment and feature.
        def shuffle_clusterings(ci: ClusteringInfo):
            fake_name_prefix = f"FAKE_i{p_val_stat_idx}_"
            labels = np.copy(ci.labels)
            np.random.shuffle(labels)
            fake_ci = ClusteringInfo(
                algorithm=fake_name_prefix + ci.algorithm,
                options=copy.deepcopy(ci.options),
                labels=labels
            )
            fake_ci.random_id = fake_name_prefix + ci.random_id
            return fake_ci

        # Repeat the sampling step as often as possible within the time budget
        date_start = datetime.now()
        shuffled_clusterings_p_vals = []  # type: List[List[float]]  # list of lists of p values
        for p_val_stat_idx in range(max_num_iterations):

            logging.info(f"[Random Sampling, i{p_val_stat_idx + 1}] Creating shuffled clusterings and analyzing p "
                         f"values...")

            # Create shuffled versions of the original clusterings
            fake_clusterings = [shuffle_clusterings(ci) for ci in original_clusterings]

            # Perform frequent itemset mining for the fake clustering - any patterns captured by the clustering
            # are broken now, and the only patterns that will remain are those inherent in the data.
            fake_freq_itemset_res_all_depths = self.frequent_itemsets_analysis_and_tables(
                clusterings=fake_clusterings,
                write_tables=False  # Do not write result tables for fake clusterings
            )

            # Save the p values that were observed for this fake clustering
            fake_p_values = self.extract_p_values_from_freq_itemset_results(
                freq_itemset_res_all_depths=fake_freq_itemset_res_all_depths
            )
            shuffled_clusterings_p_vals.append(fake_p_values)

            # Check if we can afford to run another iteration of random sampling
            seconds_elapsed = (datetime.now() - date_start).total_seconds()
            seconds_per_iteration = seconds_elapsed / (p_val_stat_idx + 1)
            logging.info(f"[Random Sampling, i{p_val_stat_idx + 1}] {p_val_stat_idx + 1} iterations of random sampling"
                         f" took {hf.format_timespan(seconds_elapsed)} ({hf.format_timespan(seconds_per_iteration)}"
                         f" per iteration).")
            budget_remaining = total_budget_seconds - seconds_elapsed
            if budget_remaining - seconds_per_iteration <= 0:
                logging.info(f"[Random Sampling, i{p_val_stat_idx + 1}] Another iteration would take "
                             f"~{hf.format_timespan(seconds_per_iteration, max_units=1)}, but only "
                             f"{hf.format_timespan(budget_remaining)} is left in computation "
                             f"budget. Stopping after {p_val_stat_idx + 1} iterations.")
                break
            else:
                logging.info(f"[Random Sampling, i{p_val_stat_idx + 1}] {hf.format_timespan(budget_remaining)} is left"
                             f" in computation budget; starting another iteration.")

        return shuffled_clusterings_p_vals

    def eval_clusters(self, fishers_test=False, with_dyn_derived=False, with_icd_p_val_dist_testing=False):
        """
        Evaluates clusters and adds results of evaluation to report

        :param fishers_test:
        :param with_dyn_derived:
        :param with_icd_p_val_dist_testing: Testing is very expensive, so it is disabled by default

        :return:
        """

        logging.info("Starting cluster evaluation ...")

        # Try to load cluster evaluation results from disk
        cluster_report = io.read_pickle(self._cluster_eval_path)

        # Only eval if we could not load evaluation from disk
        if cluster_report is not None:
            logging.info(f"Loaded cluster evaluation from file ({self._cluster_eval_path}).")
        else:
            cluster_report = {}

            # Retrieve data
            static_info = self.preprocessor.extract_static_medical_data(
                adm_indices=self.trainer.get_split_indices(self.split)
            )

            dyn_times, dyn_info = self.preprocessor.get_dyn_medical_data(
                adm_indices=self.trainer.get_split_indices(self.split),
                with_times=True
            )

            # Compare distributions of broad trends in time series
            dyn_trend_info = self.preprocessor.dyn_medical_data_fit_lines(
                adm_indices=self.trainer.get_split_indices(self.split)
            )

            # Run frequent itemset analysis
            logging.info("Performing frequent itemset analysis...")
            original_clusterings = self.clustering.clusterings
            cluster_report['frequent_itemset_analysis'] = self.frequent_itemsets_analysis_and_tables(
                clusterings=original_clusterings,
                write_tables=True
            )
            logging.info("Frequent itemset analysis done!")

            # Run frequent itemset analysis again (and again, ...) for shuffled variations of the original clusterings.
            # We do this to make sure that the patterns we see in the original clusterings can not also be found by
            # chance.
            if with_icd_p_val_dist_testing:
                cluster_report['random_sampling_p_values'] = self.perform_random_sampling_p_value_analysis()

            # Run decision tree analysis
            logging.info("Performing decision tree analysis...")
            decision_tree_results = self.decision_tree_analysis(
                static_info=static_info,
                dyn_info=dyn_info,
                dyn_trend_info=dyn_trend_info
            )
            cluster_report['decision_tree_analysis'] = decision_tree_results
            logging.info("Decision tree analysis done!")

            # Compare distributions of medical and biological facts in clusters vs. base population
            # (if distributions are different, this could mean clusters are meaningful)
            logging.info("Performing Wilcoxon test...")
            dist_comparison_static = {}
            dist_comparison_dynamic = {}
            dist_comparison_dynamic_delta = {}
            dist_comparison_dynamic_trend = {}
            stat_test_info = [
                (static_info, dist_comparison_static, "static", False),
                (dyn_info, dist_comparison_dynamic, "dynamic", True),
            ]
            if with_dyn_derived:
                dyn_delta_info = self.preprocessor.dyn_medical_data_temporal_delta(
                    dyn_info=dyn_info,
                    dyn_times=dyn_times
                )
                stat_test_info.append((dyn_delta_info, dist_comparison_dynamic_delta, "dynamic delta", False))
                stat_test_info.append((dyn_trend_info, dist_comparison_dynamic_trend, "dynamic trend", False))

            # Aggregate each dynamic (list-like) attribute using different aggregation functions
            # Note: The purpose of this is to better compare and contrast lists of values between clusters.
            # By aggregating before the comparison, the individual admission's time series' do not get mixed up.
            for data_idx in range(len(stat_test_info)):

                # Unpack attributes
                medical_info, test_results, data_kind_name, do_aggregate = stat_test_info[data_idx]

                # Aggregate only dynamic attributes. Other attributes are saved in the same data structure as aggregated
                # ones
                data_proc = {}
                if not do_aggregate:
                    data_proc[no_aggregation_key] = medical_info
                else:
                    for agg_func_name, agg_func in aggregation_functions.items():
                        # Aggregate each of the attributes
                        medical_info_agg = {dyn_name: self.apply_agg_func(dyn_vals, agg_func)
                                            for (dyn_name, dyn_vals) in medical_info.items()}

                        # Save aggregated data
                        data_proc[agg_func_name] = medical_info_agg

                # Save tuple back into stat_test_info
                stat_test_info[data_idx] = (data_proc, test_results, data_kind_name, do_aggregate)

            for clustering_info in tqdm(self.clustering.clusterings, desc="Stat test"):

                # Figure out indices if in-cluster and outside of cluster for each cluster label
                clusters_uniq = np.unique(clustering_info.labels)
                inside_cluster_indices = {label: np.where(clustering_info.labels == label)[0]
                                          for label in clusters_uniq}

                for data_agg_forms, test_results, data_kind_name, _ in tqdm(
                        stat_test_info,
                        desc=f"Stat test (clustering {clustering_info.random_id})"
                ):

                    for agg_func_name, medical_info in data_agg_forms.items():

                        for medical_info_name, medical_values in medical_info.items():

                            if medical_info_name not in test_results:
                                test_results[medical_info_name] = {}
                            if clustering_info.random_id not in test_results[medical_info_name]:
                                test_results[medical_info_name][clustering_info.random_id] = {}

                            # Perform test
                            test_res = self._statistical_test_distributions(
                                medical_values=medical_values,
                                inside_cluster_indices=inside_cluster_indices
                            )

                            # Save results in test results dictionary
                            test_results[medical_info_name][clustering_info.random_id][agg_func_name] = test_res

            # Add comparison to report
            cluster_report['distribution_comparison'] = {
                'static': dist_comparison_static,
                'dynamic': dist_comparison_dynamic  # dict: dyn_name, cluster id, agg func name
            }
            if with_dyn_derived:
                cluster_report['distribution_comparison']['dynamic_delta'] = dist_comparison_dynamic_delta
                cluster_report['distribution_comparison']['dynamic_trend'] = dist_comparison_dynamic_trend

            logging.info("Wilcoxon test done!")

            # Eval how categorical attributes (e.g. ICD codes) are distributed between clusters using Fisher's exact
            # test
            if fishers_test:
                self._fishers_exact_test(cluster_report)

            # Write full cluster eval report to disk
            io.write_pickle(cluster_report, self._cluster_eval_path, verbose=True)

        # Update report in memory (but don't write the additions to disk)
        self.update_report(new_report_content=cluster_report, write_to_disk=False)

        logging.info("Finished cluster evaluation!")
