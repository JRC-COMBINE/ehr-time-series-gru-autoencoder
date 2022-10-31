# Logging
import csv
import itertools
import json
import logging
import os
import random
from datetime import timedelta
import matplotlib

# Math and data
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats
from numpy.polynomial import Polynomial
from scipy.spatial.distance import pdist, jensenshannon
from scipy.cluster import hierarchy
from scipy.special import softmax
from sklearn import tree
from collections import OrderedDict, Counter

# Plotting
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import graphviz
from lifelines import KaplanMeierFitter
import textwrap

# Utility
from collections import defaultdict
from umap import UMAP
import gc
import string
from typing import List, Dict, Tuple, Optional
import sys

matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
from matplotlib import gridspec
import matplotlib.patheffects as pe
import seaborn as sns
from info import IcdInfo

from ai.clustering import mutual_information, rand_score, jaccard_score_label_agnostic, robustness_functions, \
    ClusteringInfo, RobustnessInfo, split_by_cluster
from evaluation.eval import TreeAnalysisResult, aggregation_functions, no_aggregation_key
from common import io


class Plotting:
    """
    Plotting of clusters and of evaluation results.
    """

    def __init__(self, iom, preprocessor, trainer, evaluator, clustering, split):
        self.iom = iom
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.evaluator = evaluator
        self.clustering = clustering
        self.split = split

        # Define plot parameters
        if split is not None:
            # Plots belong to the results of the training and the clustering
            self.plots_dir = os.path.join(self.iom.get_plots_dir(), self.split)
        else:
            # Plots belong to the input data or method itself, not to any specific trained model or a clustering
            adm_count = len(self.preprocessor.get_dyn_charts())
            plots_dir = self.iom.get_named_dir("plots_meta", in_results=False)
            self.plots_dir = os.path.join(plots_dir, f"admissions_{adm_count}")
        io.makedirs(self.plots_dir)
        self.dpi = 150
        self._file_ending = "png"

        # Save stats about data that are reused in multiple plots
        self._dyn_data_stats = {}

        # Coloring: Use same colormap whenever a color indicates a cluster membership
        self._colormap_names = ['plasma', 'viridis', 'turbo']
        self.clustering_cm_index = 2

    def _save_plot(self, name, seaborn_ax=None, dpi=None, create_dirs=False):
        if seaborn_ax is None:
            plt_obj = plt
        else:
            plt_obj = seaborn_ax.get_figure()

        # Get path
        plot_path = self._get_plot_path(name, rel_dirs=create_dirs)

        # Plot quality
        if dpi is None:
            dpi = self.dpi

        plt_obj.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        logging.info(f"Saved plot '{name}' at {plot_path}")

        # Reset matplotlib
        self._reset_matplotlib(ax=seaborn_ax)

        # Perform garbage collection (helps to stop memory leak when plotting many figures)
        gc.collect()

    @staticmethod
    def _reset_matplotlib(ax=None):
        if ax is not None:
            plt.close(ax.get_figure())
        plt.clf()
        plt.close("all")

    def _get_plot_path(self, name, rel_dirs=False, create_dirs=True):
        # Give the name a file ending if it doesn't have one
        name_no_ending, file_ending = os.path.splitext(name)
        if file_ending not in ["jpg", "png", self._file_ending]:
            file_ending += "." + self._file_ending
        name = name_no_ending + file_ending

        # Determine if `name` is a path or not
        if os.path.isabs(name):
            dir_path, filename = os.path.split(name)
            filename = io.sanitize(filename)
            if rel_dirs:
                os.makedirs(dir_path, exist_ok=True)
            return os.path.join(dir_path, filename)

        # Create subdirectories based on plot name (only if requested)
        if rel_dirs:
            dir_path, filename = os.path.split(name)
            plots_dir = os.path.join(self.plots_dir, dir_path)
            if create_dirs:
                os.makedirs(plots_dir, exist_ok=True)
            name = filename
        else:
            plots_dir = self.plots_dir

        # Sanitize the name
        name = io.sanitize(name)

        return os.path.join(plots_dir, name)

    def plot_input_data_descriptive(self):
        """
        Plots descriptive plots about the input data. Depending on the type of input data available, different plots
        will be used.

        :return:
        """

        # Plot different kinds of data in their own functions
        self._plot_input_data_static()
        self._plot_input_data_static_listlike()
        self._plot_input_data_dyn()

    def _plot_input_data_dyn(self):
        # Define scope (all admissions)
        all_adm_indices = self.trainer.get_split_indices(io.split_name_all)

        # Plot the distribution of number of time steps per admission
        dyn_plots_dir = "dyn_data"
        dyn_counts_path = os.path.join(dyn_plots_dir, "distribution_time_steps_per_admission")
        dyn_charts = None
        if not self._plot_path_exists(dyn_counts_path):

            # Load dyn charts
            if dyn_charts is None:
                dyn_charts = self.preprocessor.get_dyn_charts()

            num_steps_by_patient = [len(chart) for chart in dyn_charts]
            self.plot_distribution(
                numbers=num_steps_by_patient,
                title=f"Number of Dynamic Time Steps per Admission",
                path=dyn_counts_path
            )

        # Plot dynamic attributes ranked by their total time steps
        dyn_attrs_ranked_path = os.path.join(dyn_plots_dir, "ranking_time_steps_per_dyn_attr")
        dyn_info = None
        if not self._plot_path_exists(dyn_attrs_ranked_path):

            # Load dyn info
            if dyn_info is None:
                dyn_info = self.preprocessor.get_dyn_medical_data(adm_indices=all_adm_indices)

            # Count number of measurements for each attribute
            dyn_steps_count = {dyn_attr_name: sum([len(time_series) for time_series in dyn_info[dyn_attr_name]])
                               for dyn_attr_name in dyn_info.keys()}
            dyn_attr_names, dyn_attr_count = zip(*dyn_steps_count.items())
            self.plot_named_bars(
                bar_magnitudes=dyn_attr_count,
                bar_labels=dyn_attr_names,
                title=f"Number of time steps by dynamic attribute",
                path=dyn_attrs_ranked_path
            )

        # Plot dynamic attributes ranked by their median time steps per admission
        dyn_attrs_ranked_adm_path = os.path.join(dyn_plots_dir, "ranking_time_steps_per_dyn_attr_per_adm")
        if not self._plot_path_exists(dyn_attrs_ranked_adm_path):

            # Load dyn info
            if dyn_info is None:
                dyn_info = self.preprocessor.get_dyn_medical_data(adm_indices=all_adm_indices)

            # Count number of steps for each attribute
            dyn_attr_names = sorted(dyn_info.keys())
            dyn_attr_lengths = [[len(time_series) for time_series in dyn_info[dyn_attr_name]]
                                for dyn_attr_name in dyn_attr_names]
            dyn_attr_steps_medians, error_lower, error_upper = self._medians_and_errors(dyn_attr_lengths)
            self.plot_named_bars(
                bar_magnitudes=dyn_attr_steps_medians,
                bar_labels=dyn_attr_names,
                title=f"Number of time steps by dynamic attribute (per admission)",
                path=dyn_attrs_ranked_adm_path,
                bar_errors=[error_lower, error_upper]
            )

    @staticmethod
    def _medians_and_errors(value_lists: List[List]) -> Tuple[List, List, List]:
        medians = [np.median(vals) for vals in value_lists]
        percentile_25 = [np.percentile(vals, 25) for vals in value_lists]
        percentile_75 = [np.percentile(vals, 75) for vals in value_lists]
        error_lower = np.abs(np.array(medians) - np.array(percentile_25))
        error_upper = np.array(percentile_75) - np.array(medians)

        return medians, error_lower, error_upper

    def _plot_input_data_static(self):
        # Create folder for static attributes
        static_plots_dir = os.path.join(self.plots_dir, "static")

        # Get static data (only non-list-like)
        all_adm_indices = self.trainer.get_split_indices(io.split_name_all)
        static_non_list = self.preprocessor.extract_static_medical_data(adm_indices=all_adm_indices)

        # Find out information about static categorical data
        categ_vals = self.preprocessor.get_static_data_categ_values()

        # Plot each of the static data in an appropriate way
        for static_attr_name, observed_values in static_non_list.items():

            # Plot in a different way depending on nature of data
            is_categorical = static_attr_name in categ_vals

            if is_categorical:

                # Only plot if plot does not already exist
                plot_path = os.path.join(static_plots_dir, f"bars_{static_attr_name}")
                if not self._plot_path_exists(plot_path):
                    # Show distribution of values as bar plot if the static attribute is categorical
                    try:
                        categs, counts = np.unique(observed_values, return_counts=True)
                    except TypeError as e:
                        logging.error("Exception when plotting static input data!")
                        logging.getLogger().exception(e)
                        logging.error("Additional information:")
                        logging.error(f"static_attr_name = {static_attr_name}")
                        logging.error(f"len(observed_values) = {len(observed_values)}")
                        logging.error(f"observed_values[:10] = {observed_values[:10]}")
                        none_idx = observed_values.index(None)
                        logging.error(f"observed_values.index(None) = {none_idx}")
                        logging.error(f"hadm_idx of none index: {self.preprocessor.get_hadm_id(none_idx)}")
                        assert False, "Quitting after error"

                    self.plot_named_bars(
                        bar_magnitudes=counts,
                        bar_labels=categs,
                        title=f"Distribution of {io.prettify(static_attr_name)}",
                        path=plot_path,
                        random_colors=True,
                        vertical_bars=True
                    )
            else:

                # Only plot if plot does not already exist
                plot_path = os.path.join(static_plots_dir, f"distribution_{static_attr_name}")

                if not self._plot_path_exists(plot_path):
                    # Show numerical distribution of values
                    self.plot_distribution(
                        numbers=observed_values,
                        title=f"Distribution of {io.prettify(static_attr_name)}",
                        path=plot_path
                    )

    def _plot_input_data_static_listlike(self):
        # Create folder for list-like attributes
        listlike_plots_dir = os.path.join(self.plots_dir, "listlike_static")

        # Get list-like static data (e.g. ICD codes are list-like since a patient can have more than one)
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        for listlike_attr in self.preprocessor.get_static_attrs_listlike():
            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[listlike_attr] for stat in static_categorical]

            # Find out which ICD codes are possible in the population (this list is also the interpretation
            # for the present_icd_codes_indices)
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[listlike_attr]

            # Plot the distribution of the number of ICD codes
            num_distr_path = os.path.join(listlike_plots_dir, f"distribution_num_item_per_admission_{listlike_attr}")
            if not self._plot_path_exists(num_distr_path):
                num_codes_by_admission = [len(codes) for codes in present_icd_codes_indices]
                self.plot_distribution(
                    numbers=num_codes_by_admission,
                    title=f"Number of {io.prettify(listlike_attr)} per Admission",
                    path=num_distr_path
                )

            # Plot ranked codes: most popular code first
            code_ranked_plot_path = os.path.join(listlike_plots_dir, f"ranked_codes_{listlike_attr}")
            if not self._plot_path_exists(code_ranked_plot_path):

                # Count the number of times an index appears
                code_idx_counts = {}
                for adm_idcs in present_icd_codes_indices:
                    for idx in adm_idcs:
                        # Count this code index
                        if idx not in code_idx_counts:
                            code_idx_counts[idx] = 0
                        code_idx_counts[idx] += 1

                # Rank the indices based on the count
                indices_ranked = sorted(code_idx_counts.keys(), key=lambda c_idx: code_idx_counts[c_idx])  # small to
                # large
                codes_ranked = [possible_icd_codes[idx] for idx in indices_ranked]
                counts_ranked = [code_idx_counts[idx] for idx in indices_ranked]
                icd_names_ranked = [IcdInfo.name_for_code(icd_kind=listlike_attr, icd_code=c) for c in codes_ranked]
                self.plot_named_bars(
                    bar_magnitudes=counts_ranked,
                    bar_labels=icd_names_ranked,
                    title=f"Counts of {io.prettify(listlike_attr)}",
                    path=code_ranked_plot_path
                )

    def plot_named_bars(self, bar_magnitudes, bar_labels, title, path, max_labeled_ticks=50, random_colors=False,
                        vertical_bars=False, bar_errors=None):
        # Sort by magnitude
        sorted_indices = np.argsort(bar_magnitudes)
        bar_magnitudes = [bar_magnitudes[idx] for idx in sorted_indices]
        bar_labels = [bar_labels[idx] for idx in sorted_indices]
        if bar_errors is not None:
            errors_lower, errors_upper = bar_errors
            errors_lower = [errors_lower[idx] for idx in sorted_indices]
            errors_upper = [errors_upper[idx] for idx in sorted_indices]
            bar_errors = errors_lower, errors_upper

        # Prepare plot
        fig_size = [5, 16]
        if vertical_bars:
            fig_size = list(reversed(fig_size))
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title(title)

        # Plot bars from top to bottom
        bar_height = 1.1
        bar_indices = np.arange(0, len(bar_magnitudes))

        # Coloring
        if not random_colors:
            colors = ["xkcd:sea blue"] * len(bar_indices)
        else:
            colors = [np.random.uniform(0.2, 0.7, 3) for _ in range(len(bar_indices))]

        # Bars
        if not vertical_bars:
            bar_func = plt.barh
            axis_tick_func = ax.set_yticks
            axis_label_func = ax.set_yticklabels
            bar_height_arg_name = 'height'
            bar_error_arg_name = 'xerr'
        else:
            bar_func = plt.bar
            axis_tick_func = ax.set_xticks
            axis_label_func = ax.set_xticklabels
            bar_height_arg_name = 'width'
            bar_error_arg_name = 'yerr'

        plot_info = {
            'color': colors,
            bar_height_arg_name: bar_height
        }
        if bar_errors is not None:
            plot_info[bar_error_arg_name] = bar_errors
            plot_info['ecolor'] = "purple"  # gray error lines

        bar_func(
            bar_indices,
            bar_magnitudes,
            **plot_info
        )

        # Set y-axis tick labels: Only label some ticks
        labeled_ticks = np.unique(np.linspace(0, len(bar_labels) - 1, max_labeled_ticks).astype(int))
        tick_labels = [bar_labels[idx] if idx in labeled_ticks else "" for idx in bar_indices]
        axis_tick_func(bar_indices)
        axis_label_func(tick_labels)

        self._save_plot(
            path,
            create_dirs=True,
            dpi=self.dpi
        )

    def _plot_path_exists(self, plot_path):
        plot_dir, plot_filename = os.path.split(plot_path)
        plot_filename = io.sanitize(plot_filename)
        plot_path = os.path.join(plot_dir, plot_filename)
        plot_path = self._get_plot_path(plot_path, rel_dirs=True, create_dirs=False)
        return os.path.exists(plot_path)

    def plot_distribution(self, numbers, title, path):
        # Mention some stats in title
        mean = np.mean(numbers)
        median = np.median(numbers)
        mi, ma = np.min(numbers), np.max(numbers)
        q1, q3 = np.percentile(numbers, 25), np.percentile(numbers, 75)
        title += f"\n" \
                 f"Min: {mi}, Max: {ma}, Median: {median}, Mean: {mean}\n" \
                 f"Q1: {q1}, Q3: {q3}"

        # Plot distribution
        fig, ax = plt.subplots(figsize=(12, 4))
        numbers_series = pd.Series(numbers, name=title)
        ax = sns.distplot(numbers_series, kde=True, kde_kws={"cut": 0}, ax=ax)

        # Plot mean and median
        plt.axvline(mean, color="xkcd:bright blue", label="Mean")
        plt.axvline(median, color="xkcd:aquamarine", label="Median")
        plt.legend()

        self._save_plot(
            path,
            create_dirs=True,
            dpi=self.dpi,
            seaborn_ax=ax
        )

    def plot_p_value_distribution(self):
        # Extract p values of original clusterings
        freq_itemsets_all_depths = self.evaluator.report['frequent_itemset_analysis']
        original_p_values = self.evaluator.extract_p_values_from_freq_itemset_results(freq_itemsets_all_depths)

        # Extract p values of random permutations
        if 'random_sampling_p_values' not in self.evaluator.report:
            return  # Don't plot if the analysis was not performed
        samplings_p_values_list = self.evaluator.report['random_sampling_p_values']

        # Plot
        self._plot_p_value_distribution(
            original_p_vals=original_p_values,
            sampling_p_vals_list=samplings_p_values_list
        )

    def _plot_p_value_distribution(self, original_p_vals: List[float], sampling_p_vals_list: List[List[float]]):
        # Prepare data
        data = []
        source_str = "Source"
        p_val_str = "P-Value"
        for p_val in original_p_vals:
            data.append({
                p_val_str: p_val,
                source_str: "Actual Clustering"
            })
        for sampling_p_vals in sampling_p_vals_list:
            for p_val in sampling_p_vals:
                data.append({
                    p_val_str: p_val,
                    source_str: "Random Permutation"
                })
        dist_df = pd.DataFrame(data=data)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        num_permutation_p_vals_total = sum([len(vals) for vals in sampling_p_vals_list])
        sns_ax = sns.displot(
            data=dist_df,
            x=p_val_str,
            hue=source_str,
            kde=True,
            kde_kws={"cut": 0},  # truncate estimated curve at data limits
            element="step",  # draw as step function instead of histogram bars
            stat="density",  # don't display count but density
            common_norm=False,  # normalize different curves to the same density (which is good because we have more
            # p values from random permutations than original p values)
            log_scale=True,  # Set x axis to log scale
            ax=ax
        )

        # Also set y axis to log scale
        sns_ax.ax.set_yscale('log')

        # Perform Welch's unequal variances t-test to find out if distributions are dissimilar by chance
        p_vals_original_arr = np.array(original_p_vals)
        p_vals_permutated_arr = np.concatenate([np.array(vals) for vals in sampling_p_vals_list])
        p_val_distributions_ttest = scipy.stats.ttest_ind(
            p_vals_original_arr,
            p_vals_permutated_arr,
            equal_var=False
        ).pvalue

        # Set descriptive plot title
        sns_ax.ax.set_title(f"Distribution of p-values in actual clusterings and {len(sampling_p_vals_list)}"
                            f" random permutations\n"
                            f"({len(original_p_vals)} p-values for original cluster; {num_permutation_p_vals_total}"
                            f" for random permutations)\n"
                            f"Welch's t-test: {p_val_distributions_ttest:.4e}")

        # Save plot
        self._save_plot(
            "p_values_distribution",
            create_dirs=True,
            dpi=self.dpi,
            seaborn_ax=sns_ax.ax
        )

    def plot_age_vs_survival(self):
        # Find out ages and survival of admissions
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        admission_survived = np.array(static_info['FUTURE_survival'])  # True if alive, False if dead
        admission_ages = np.array(static_info['age_years'])

        for clustering_info in self.clustering.clusterings:
            plot_dir = os.path.join(self._get_dir_for_clustering(clustering_info), "age_vs_survival")

            self._plot_age_vs_survival_curve(
                labels=clustering_info.labels,
                survivals=admission_survived,
                ages=admission_ages,
                plot_dir=plot_dir
            )

    def plot_rec_err_vs_survival_bars(self):
        # Retrieve reconstruction error from evaluator
        adm_err_key = 'adm_rec_errors'
        if adm_err_key not in self.evaluator.report:
            logging.info("Can not plot reconstruction error vs. survival: Data not available in eval.")
            return
        rec_errors = self.evaluator.report[adm_err_key]

        # Retrieve survival of the admissions
        evaluated_adm_indices = self.evaluator.report['adms_evaluated']
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=evaluated_adm_indices
        )
        admission_survived = static_info['FUTURE_survival']  # True if alive, False if dead

        # Plot with multiple different binning settings
        for num_bins, percentile in itertools.product(np.linspace(2, 50, 25).astype(int), [90, 95, 99, 100]):
            self._plot_rec_err_vs_survival_bars(
                rec_errors=rec_errors,
                survivals=admission_survived,
                num_bins=num_bins,
                percentile=percentile
            )

    def _plot_rec_err_vs_survival_bars(self, rec_errors: List[float], survivals: List[bool], num_bins: int = 10,
                                       percentile=95):
        # Sort both errors and survivals
        rec_errors = np.array(rec_errors)
        sorting = np.argsort(rec_errors)
        rec_errors = rec_errors[sorting]
        survivals = np.array(survivals)[sorting]

        # Remove outliers
        max_rec_err = np.percentile(rec_errors, percentile)
        rec_err_filter = rec_errors <= max_rec_err
        rec_errors = rec_errors[rec_err_filter]
        survivals = survivals[rec_err_filter]

        # Don't plot if there are too few points
        if len(rec_errors) < 100:
            logging.info("Not plotting rec. error vs. survival: Too few points")
            return

        # Determine bin edges for histogram-like plot (x-axis is reconstruction error)
        bin_edges = np.histogram_bin_edges(rec_errors, bins=num_bins)

        # Assign survivals to bins
        survivals_by_bins = []
        current_idx = 0
        for bin_idx, (left_border, right_border) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            illegal_idx = None
            for idx in range(current_idx, len(rec_errors)):
                if rec_errors[idx] >= right_border:
                    illegal_idx = idx
                    break

            # If this is the last bin, the right border should be inclusive as well
            if bin_idx == num_bins - 1:
                illegal_idx += 1

            # Copy all survivals into bin
            survivals_by_bins.append(survivals[current_idx:illegal_idx])
            current_idx = illegal_idx

        # Sanity check: Make sure every bin is accounted for
        assert len(survivals_by_bins) == num_bins

        # Count the number of admissions for each bar
        counts = [len(sbin) for sbin in survivals_by_bins]

        # Count rates of survival within bins
        survival_rates = [100 * np.mean([int(s) for s in sbin]) for sbin in survivals_by_bins]

        # Convert "survival" into "mortality"
        mortality_rates = [100 - s for s in survival_rates]

        # Plot bars
        fig, ax = plt.subplots(figsize=(14 + 7 * (num_bins // 20), 6))
        bars_x_left = bin_edges[:-1]
        bars_widths = bin_edges[1:] - bars_x_left

        bars = ax.bar(
            bars_x_left,
            mortality_rates,  # bar height is equivalent to mortality rate within bar
            width=bars_widths,
            align='edge',  # align on left edge
            color='xkcd:sea blue',
            edgecolor='black'
        )

        # Label bars
        label_texts = ax.bar_label(
            bars,
            labels=[f"{sr:0.0f}%\n({c})" for sr, c in zip(mortality_rates, counts)],
            label_type='edge',
            padding=10,
            color="black"
        )

        # "Zoom out" a bit on y-axis
        ax.set_ylim(ymin=0, ymax=115)

        # Label axes
        ax.set_xlabel(f"Reconstruction Error (MSE)")
        ax.set_ylabel("Mortality (%)")

        ax.set_title(f"Mortality vs. Reconstruction Error (bottom {percentile:0.1f}%)")

        # Write plot
        plot_path = os.path.join(
            self.plots_dir,
            "rec_error_mortality",
            f"rec_error_mortality_bars_perc{percentile}_{num_bins}bins",
        )
        self._save_plot(plot_path, create_dirs=True, dpi=self.dpi * 2)

    def _plot_age_vs_survival_curve(self, labels, survivals, ages, plot_dir):
        # Sort all values by age
        sorting = np.argsort(ages)
        ages = ages[sorting]
        labels = labels[sorting]
        survivals = survivals[sorting]

        # Set colors
        clus_uniq = list(np.unique(labels))
        cm = self._colormap(num_colors=len(clus_uniq), index=self.clustering_cm_index)
        colors = {lab: cm[clus_uniq.index(lab)] for lab in clus_uniq}

        # Fix scaling based on all ages
        ages_ptp = np.ptp(ages)
        min_x = min(ages) - 0.05 * ages_ptp
        max_x = max(ages) + 0.05 * ages_ptp

        # Plot for population
        fig, ax_pop = plt.subplots(figsize=(15, 7))
        ax_pop.set_title("Age vs. Survival - Population")
        self._plot_age_vs_survival_curve_inner(
            survivals=survivals,
            ages=ages,
            ax=ax_pop,
            min_x=min_x,
            max_x=max_x
        )
        self._save_plot(
            name=os.path.join(plot_dir, "age_vs_survival_pop"),
            seaborn_ax=ax_pop,
            create_dirs=True
        )

        # Plot for each of the clusters in its own axis
        for cluster_label in clus_uniq:
            # Limit values to those found in cluster
            label_idcs = np.where(labels == cluster_label)[0]
            cluster_ages = ages[label_idcs]
            cluster_survivals = survivals[label_idcs]

            # Plot
            fig, ax_cluster = plt.subplots(figsize=(15, 7))
            ax_cluster.set_title(f"Age vs. Survival - {io.label_for_cluster_label(cluster_label)}")
            self._plot_age_vs_survival_curve_inner(
                survivals=cluster_survivals,
                ages=cluster_ages,
                ax=ax_cluster,
                min_x=min_x,
                max_x=max_x,
                color=colors[cluster_label]
            )
            self._save_plot(
                name=os.path.join(plot_dir, f"age_vs_survival_clus_{cluster_label}"),
                seaborn_ax=ax_cluster,
                create_dirs=True
            )

        # Plot all clusters in one plot
        fig, ax_cluster = plt.subplots(figsize=(15, 7))
        ax_cluster.set_title(f"Age vs. Survival - Clusters")
        for cluster_label in clus_uniq:
            # Limit values to those found in cluster
            label_idcs = np.where(labels == cluster_label)[0]
            cluster_ages = ages[label_idcs]
            cluster_survivals = survivals[label_idcs]

            # Plot
            self._plot_age_vs_survival_curve_inner(
                survivals=cluster_survivals,
                ages=cluster_ages,
                ax=ax_cluster,
                min_x=min_x,
                max_x=max_x,
                color=colors[cluster_label],
                legend_label=io.label_for_cluster_label(cluster_label),
                add_ax_title=False
            )
        self._save_plot(
            name=os.path.join(plot_dir, f"age_vs_survival_clus_all"),
            seaborn_ax=ax_cluster,
            create_dirs=True
        )

    @staticmethod
    def _plot_age_vs_survival_curve_inner(survivals, ages, ax, min_x, max_x, min_y=-0.05, max_y=1.05,
                                          color=None, legend_label=None, max_num_points_plotted=250, add_ax_title=True):
        # Convert survivals to integer
        survivals = survivals.astype(int)

        # Set color
        plot_kwargs = {}
        if color is not None:
            plot_kwargs['color'] = color

        # Sample from points if there are too many for scatter plot
        if len(survivals) > max_num_points_plotted:
            indices = range(len(survivals))
            indices_used = np.random.choice(indices, size=max_num_points_plotted, replace=False)
            survivals_sampled = survivals[indices_used]
            ages_sampled = ages[indices_used]
        else:
            survivals_sampled = survivals
            ages_sampled = ages

        # Plot sampled ages along with their survival as a scatter plot
        ax.scatter(
            ages_sampled,  # x
            survivals_sampled,  # y
            linewidths=0,
            alpha=0.5,
            **plot_kwargs
        )

        # Label line (if set)
        if legend_label is not None:
            plot_kwargs['label'] = legend_label

        # Plot line that captures "average" behavior for different ages by fitting a curve to the data
        deg = 3  # e.g. 1 -> line, 2 -> cubic, 3 -> quadratic, etc.
        if len(ages) >= deg:
            try:
                poly = Polynomial.fit(
                    x=ages,  # perform using full, non-sampled data
                    y=survivals,
                    deg=deg
                )
                x_sampled = np.linspace(min(ages), max(ages), 400, endpoint=True)
                ax.plot(
                    x_sampled,
                    poly(x_sampled),
                    **plot_kwargs
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                logging.warning("Could not fit line")

        # Label axes and set limits
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("P(Survival)")
        ax.set_ylim(ymin=min_y, ymax=max_y)
        ax.set_xlim(xmin=min_x, xmax=max_x)
        if add_ax_title:
            sampled_explanation_str = "" if len(ages_sampled) == len(ages) else f" shown of {len(ages)}"
            ax.set_title(ax.get_title() + f" ({len(ages_sampled)} adm.{sampled_explanation_str})")

        # Legend (if set)
        if legend_label is not None:
            ax.legend()

    def plot_kaplan_meier_survival(self):
        # Find out survival information
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        length_of_stay = np.array(static_info['FUTURE_days_in_care'])  # How many days in hospital until discharged
        stay_ends_in_death = np.array([not s for s in static_info['FUTURE_survival_to_discharge']])  # True if dead,
        # False if alive when leaving hospital

        for clustering_info in self.clustering.clusterings:
            kaplan_meier_dir = os.path.join(self._get_dir_for_clustering(clustering_info), "kaplan_meier")

            self._plot_cluster_kaplan_meier_survival(
                labels=clustering_info.labels,
                length_of_stay=length_of_stay,
                stay_ends_in_death=stay_ends_in_death,
                plot_dir=kaplan_meier_dir
            )

    def _plot_cluster_kaplan_meier_survival(self, labels, length_of_stay, stay_ends_in_death, plot_dir):
        # Prepare plotting axes in order to plot all cluster's survival curves into the same plot (but also each
        # into its own plot)
        def get_axis():
            _, new_ax = plt.subplots(figsize=(18, 6))
            return new_ax

        labs_uniq = list(np.unique(labels))

        # Function for saving plot
        def save_axis(ax, plot_name):
            # Decorate the axis
            ax.grid(True)
            ax.set_title("Survival in Clusters")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("P(Survival)")

            # Set limits
            ax.set_ylim(ymin=0, ymax=1.)

            # Save the plot
            self._save_plot(
                os.path.join(plot_dir, f"kaplan_meier_survival_{plot_name}"),
                create_dirs=True,
                seaborn_ax=ax
            )

        # Handle each of the clusters
        colormap = self._colormap(num_colors=len(labs_uniq), index=self.clustering_cm_index)
        kmf_cache = {}
        single_axes = {}
        for lab in labs_uniq:
            # Filter the survival information
            clus_indices = np.where(labels == lab)[0]
            clus_len_stay = length_of_stay[clus_indices]
            clus_dead = stay_ends_in_death[clus_indices]

            # Fit the Kaplan-Meier curve
            kmf = KaplanMeierFitter()
            kmf.fit(clus_len_stay, clus_dead, label=io.label_for_cluster_label(lab))

            # Plot the curve onto its own axis
            kwargs = {
                'color': colormap[labs_uniq.index(lab)],
                'show_censors': True
            }
            single_axes[lab] = get_axis()
            kmf.plot_survival_function(ax=single_axes[lab], **kwargs)
            save_axis(single_axes[lab], f"{lab:03d}")

            # Save the information required to plot this curve, so we can later use it to plot a plot containing all
            # clusters' curves
            kmf_cache[lab] = (kmf, kwargs)

        # Finally, save plot with all curves in it
        ax_common = get_axis()
        for lab in labs_uniq:
            kmf, kwargs = kmf_cache[lab]
            kmf.plot_survival_function(ax=ax_common, **kwargs)
        save_axis(ax_common, "all")

    def plot_clusters(self, max_points_plotted=2500, use_umap=False):
        """
        Plots clusters (which are based on autoencoder features)
        :param max_points_plotted:
        :param use_umap:
        :return:
        """

        # Plot all possible combinations of
        # - 2d projection method (t-SNE, PCA, etc.)
        # - medical information
        # - clusterings

        # Prepare medical information
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )

        # Compute features for all admissions in the split
        features = self.trainer.compute_features(self.split)

        # Sample from the admissions if there are more than can be plotted
        plotted_indices = np.arange(len(features))
        if len(features) > max_points_plotted:
            # Randomly sample admissions to plot
            plotted_indices = np.random.choice(plotted_indices, max_points_plotted, replace=False)

        # Prepare all 2d projections
        # (in order to plot the high-dimensional features, we need to project them to two dimensions)
        projections = {}

        # Project using t-SNE
        for perplexity in np.linspace(1, 100, 4):
            perplexity = int(perplexity)
            projections[f"tsne_p{perplexity}"] = self.projection_tsne(
                features=features,
                perplexity=perplexity
            )

        # Project using UMAP
        if use_umap:
            for n_neighbors, min_dist in itertools.product(
                    np.linspace(2, features.shape[0] // 2, 2),
                    np.linspace(0, 1, 2)
            ):
                n_neighbors = int(n_neighbors)
                projections[f"umap_nN{n_neighbors}_minD{min_dist}"] = self.projection_umap(
                    features=features,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist
                )

        # Also project into three dimensions (so that I can plot in 3d if I want to)
        if features.shape[1] > 2:
            proj_3d = self.projection_pca(
                features=features,
                n_components=3  # Project into 3d
            )
            proj_3d_path = os.path.join(self.plots_dir, "projection_3d.json")
            with open(proj_3d_path, "w") as proj_3d_file:

                # Static info labels are only saved once
                static_info_labels = list(static_info.keys())

                # Save each point as its own entry in a list
                points_3d = []
                for point_idx in range(len(proj_3d)):
                    # 3d coordinates
                    point = {
                        "coord_x": float(proj_3d[point_idx][0]),
                        "coord_y": float(proj_3d[point_idx][1]),
                        "coord_z": float(proj_3d[point_idx][2]),
                        "static_info_values": [static_info[label][point_idx] for label in static_info_labels]
                    }
                    points_3d.append(point)

                points_info = {
                    "points": points_3d,
                    "static_info_labels": static_info_labels
                }

                json.dump(points_info, proj_3d_file, sort_keys=True, indent=4, separators=(",", ": "))

        # Project using PCA
        projections["pca"] = self.projection_pca(
            features=features
        )

        # Plot all projections
        for proj_name, proj in projections.items():

            # Plot all clusters
            for clustering_info in self.clustering.clusterings:
                self._plot_clustering(proj_name, proj, plotted_indices, None, None, clustering_info,
                                      plot_clustering=True, features_dimensionality=features.shape[1])

            # Plot all medical data views
            for static_info_name, static_values in static_info.items():
                self._plot_clustering(proj_name, proj, plotted_indices, static_info_name, static_values, None,
                                      plot_medical=True)

    def _plot_clustering(self, proj_name, proj, plotted_indices, static_info_name, static_values, clustering_info,
                         plot_medical=False, plot_clustering=False,
                         size_points=10, max_bins_static=4, font_size_textbox=4, font_size_legend=8,
                         max_legend_entries=25, marker_types=("o", "^", "s", "*", "d"),
                         point_border_width=0.25, features_dimensionality=None):

        # Make sure that only a single property is selected to be plotted
        assert sum([1 if mode else 0 for mode in [plot_medical, plot_clustering]]) == 1, \
            "Exactly one plotting mode has to be selected!"

        # Init plt plot
        fig, ax = plt.subplots()

        # Prepare legend labels for clusters
        if plot_clustering:
            cluster_legend_labels = {clus_label: io.label_for_cluster_label(clus_label)
                                     for clus_label in np.unique(clustering_info.labels)}

        # Prepare legend labels for medical information
        if plot_medical:
            static_values_plotted = np.array(static_values)[plotted_indices]
            static_uniq = np.unique(static_values_plotted)
            if len(static_uniq) > max_bins_static and type(static_values[0]) != str:
                # Numerical data attribute: Legend contains only colors for extremal data points.
                # All other points are colored in between those colors.

                val_min = np.nanmin(static_values_plotted)
                val_max = np.nanmax(static_values_plotted)
                medical_legend_labels = {
                    val_min: f"{val_min:0.1f}",
                    val_max: f"{val_max:0.1f}"
                }
                color_by_scale = True  # Coloring given through a scale between two extremal colors

            else:
                # No binning necessary since number of unique values is low enough
                medical_legend_labels = {med_val: med_val for med_val in static_uniq}
                color_by_scale = False  # Fixed colors
        else:
            # Clusters are never colored by scale
            color_by_scale = False

        # Define different legend plotting locations
        legend_loc_right_top = {
            'bbox_to_anchor': (1.04, 1),
            'loc': "upper left",
            'borderaxespad': 0,
        }
        legend_loc_right_bottom = {
            'bbox_to_anchor': (1.04, 0),
            'loc': "lower left",
            'borderaxespad': 0
        }

        if plot_clustering:
            chosen_plot_element = [
                clustering_info.labels,
                cluster_legend_labels,
                legend_loc_right_top,
                "Clusters"
            ]
        else:
            chosen_plot_element = [
                np.array(static_values),
                medical_legend_labels,
                legend_loc_right_bottom,
                io.prettify(static_info_name)
            ]

        # Unpack chosen plot elements into components
        values, legend_labels, legend_loc, title = chosen_plot_element

        # Generate colormap for all values
        uniq_vals = list(np.unique(values))
        if plot_clustering:
            cm_index = self.clustering_cm_index
        else:
            cm_index = 1
        colormap = self._colormap(num_colors=len(uniq_vals), index=cm_index)

        def color_for_value(v):
            if not color_by_scale:
                # Fixed colors for clusters and categorical medical data
                return colormap[uniq_vals.index(v)]
            else:
                # Color by scale for numerical medical data

                # Find out where the value is positioned on the colormap
                val_relative = (v - val_min) / (val_max - val_min)  # from 0 to 1
                if np.isnan(val_relative):
                    val_relative = 0
                cm_pos = val_relative * (len(colormap) - 1)  # from 0 to the highest index in the colormap, e.g. 47.38
                cm_lower_idx = int(np.floor(cm_pos))  # e.g. 47
                cm_upper_idx = int(np.ceil(cm_pos))  # e.g. 48
                cm_indices_interp = cm_pos - cm_lower_idx  # e.g. 0.38
                return (1 - cm_indices_interp) * colormap[cm_lower_idx] + cm_indices_interp * colormap[cm_upper_idx]

        # Generate marker types for all values
        marker_map = list(marker_types)

        def marker_for_value(v):
            if not color_by_scale:
                marker_idx = uniq_vals.index(v)
            else:
                marker_idx = 0  # Choose first marker if not coloring by clustering
            return marker_map[marker_idx % len(marker_map)]

        # Plot points in a random order
        values_indices = np.copy(plotted_indices)
        np.random.shuffle(values_indices)
        for value_idx in values_indices:

            # Find out value to plot
            value = values[value_idx]

            # Set color and marker
            color = color_for_value(value)
            marker = marker_for_value(value)

            # Plot this point
            plot_kwargs = {}
            if not color_by_scale:
                plot_kwargs['label'] = legend_labels[value]  # Label, which corresponds to legend
            scatter = ax.scatter(
                [proj[value_idx, 0]],  # x
                [proj[value_idx, 1]],  # y
                s=size_points,  # Point size
                c=[color],  # Color
                edgecolors='grey',  # gray but very thin point edges
                linewidths=point_border_width,
                alpha=1,
                marker=marker,
                **plot_kwargs
            )

        # Show a colorbar if coloring was performed by scale
        if color_by_scale:
            fig.colorbar(
                mappable=matplotlib.cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(
                        vmin=val_min,
                        vmax=val_max
                    ),
                    cmap=self._colormap_names[cm_index]
                ),
                ax=ax
            )

        # Legend
        if not color_by_scale:
            def leg_handler_for_value(v):
                return plt.Line2D([], [], color=color_for_value(v), ls="", marker=marker_for_value(v))

            # Restrict max legend entries
            uniq_vals_legend = list(legend_labels.keys())[:max_legend_entries]

            legend = ax.legend(
                handles=[leg_handler_for_value(value) for value in uniq_vals_legend],
                labels=[legend_labels[value] for value in uniq_vals_legend],
                title=title,
                prop={'size': font_size_legend},
                **legend_loc
            )

        # Text box with meta information about the clusters and the plot
        if plot_clustering:
            text_box_lines = [
                f"{len(proj)} Points ({len(plotted_indices)} shown)"
            ]

            # Include the number of points in each cluster in the text box
            cluster_labels, cluster_point_counts = np.unique(clustering_info.labels, return_counts=True)
            num_clus_text = 25
            cluster_counts = list(zip(cluster_point_counts, cluster_labels))
            if len(cluster_counts) > num_clus_text:
                # (only show largest clusters if there are too many)
                cluster_counts = sorted(cluster_counts, reverse=True)[:num_clus_text]
            text_box_lines.append(f"Top {num_clus_text} clusters:")
            for num_points, cluster_label in cluster_counts:
                text_box_lines.append(
                    f"{io.label_for_cluster_label(cluster_label)}: {num_points} points"
                )

            # Other info
            text_box_lines.append(f"Projection: {proj_name}")
            if features_dimensionality is not None:
                text_box_lines.append(f"Features: {features_dimensionality} dim.")
            text_box_lines += [
                f"Clustering algo: \n"
                f" -> {clustering_info.algorithm}"
            ]
            text_box_lines += [f" -> {opt_name}: {opt_val}" for (opt_name, opt_val) in clustering_info.options.items()]

            for tech in clustering_info.technicals:
                text_box_lines.append(f"{tech.name}: {tech.score:0.4f}")

            # Robustness
            for rob_name in robustness_functions.keys():
                rob = clustering_info.robustness[rob_name]  # type: RobustnessInfo
                text_box_lines.append(f"{io.prettify(rob_name)}: {rob.is_robust} "
                                      f"(actual: {rob.rob_actual:0.2f}, threshold: {rob.rob_threshold:0.2f})")
            text_box_lines.append(f"Robust (All): {clustering_info.is_robust}")

            text_box_lines.append(f"Clustering {clustering_info.random_id}")
            text_box_text = "\n".join(text_box_lines)

            ax.text(
                -0.04,  # x pos (0 to 1 -> left to right)
                0.5,  # y pos (0 to 1 -> bottom to top)
                text_box_text,
                transform=ax.transAxes,  # use axis coordinates, not data coordinates for box placement
                fontsize=font_size_textbox,
                verticalalignment='center',  # bottom == box is ABOVE coordinate
                horizontalalignment='right',  # left == box is to the right of coordinate
                bbox=dict(
                    edgecolor='xkcd:dark',
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.33
                )
            )

        # Plot title
        ax.set_title(
            f"{title} ({proj_name})"
        )

        # Disable axes (they are meaningless for embedded cluster points)
        plt.axis('off')

        # Determine filename
        if plot_clustering:
            plot_path = os.path.join(
                self._get_dir_for_clustering(clustering_info),
                f"cluster_{proj_name}"
                f"_clustering"
                f"_{clustering_info.algorithm}_{clustering_info.random_id}"
            )
        else:
            plot_path = os.path.join(
                "feature_space",
                "medical_views",
                f"{static_info_name}_medical_{proj_name}"
            )

        # Write plot to disk
        self._save_plot(
            plot_path,
            create_dirs=True,
            dpi=self.dpi * 2
        )

    @staticmethod
    def _get_dir_for_clustering(clustering_info: ClusteringInfo):
        return os.path.join("feature_space", clustering_info.get_path_name())

    def _colormap(self, num_colors, index=0):
        cm_name = self._colormap_names[index]
        cm = matplotlib.cm.get_cmap(
            name=cm_name,
            lut=num_colors
        )
        colormap = cm.colors
        return colormap

    def plot_density(self, list_of_values, name, value_label, cumulative=False, hist_name=None):
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot the cumulative histogram
        if hist_name is None:
            hist_name = "Density"
        num_bins = int(np.sqrt(len(list_of_values)))
        if cumulative:
            num_bins *= 10
        n, bins, patches = ax.hist(list_of_values, bins=num_bins, density=True, histtype='stepfilled',
                                   cumulative=cumulative, label=hist_name)

        # Mark mean and median position
        val_range = max(list_of_values) - min(list_of_values)
        mean = np.mean(list_of_values)
        median = np.median(list_of_values)
        if val_range > 20:
            mean = int(mean)
            median = int(median)
        ax.axvline(mean, color="xkcd:lilac", label="Mean")
        ax.text(x=mean, y=ax.dataLim.y1, s=f"{mean}")
        ax.axvline(median, color="blue", label="Median")
        ax.text(x=median, y=ax.dataLim.y1 * 0.9, s=f"{median}")

        # Tidy up the figure
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title(io.prettify(name))
        ax.set_xlabel(value_label)
        ax.set_ylabel('Density')

        # Save
        self._save_plot(name=name, seaborn_ax=ax)

    def plot_scatter(self, x, line_heights, line_colors, line_labels, x_axis_label, y_axis_label, title, path,
                     plot_lines=False, y_log_scale=False):
        """
        Plot points on a common x axis

        :param x:
        :param line_heights:
        :param line_colors:
        :param line_labels:
        :param x_axis_label:
        :param y_axis_label:
        :param title:
        :param path:
        :param plot_lines:
        :param y_log_scale:
        :return:
        """
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 4))

        # Decide on which function to plot with
        if not plot_lines:
            plot_function = ax.scatter
        else:
            plot_function = ax.plot

        # Register points that need to be plotted
        for y_vals, color, label in zip(line_heights, line_colors, line_labels):
            plot_function(x, y_vals, color=color, label=label, marker='.', alpha=0.75)

        if y_log_scale:
            plt.yscale('log')

        # Set up meta info
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.legend()

        # Save
        self._save_plot(name=path, seaborn_ax=ax, create_dirs=True, dpi=2 * self.dpi)

    def plot_multi_y_function(self, x_list, x_label, y_lists, y_labels, name, plot_dir):
        """
        Stacked plots with common x-axis and multiple y axes stacked on top of each other (each y-axis gets their own
        plot)

        :param x_list:
        :param x_label:
        :param y_lists: list of list of lists
        :param y_labels:
        :param name:
        :param plot_dir:
        :return:
        """
        # Set up the figure
        num_y_axes = len(y_lists)
        fig, y_axes = plt.subplots(num_y_axes, sharex=True, figsize=(12, 4))

        # Plot lines for each axis
        for y_idx, (y_values, y_label, ax) in enumerate(zip(y_lists, y_labels, y_axes)):
            # y_values is a list of lists of floats. The top-level list corresponds to the x values while the
            # second-level lists contain multiple y values for a single x value.

            # Plot each of the "inner" y values (of which there are the same number for each x value) in a
            # different color
            colors = self._colormap(num_colors=len(y_values[0]))
            for y_inner_val_idx in range(min([len(y_inner) for y_inner in y_values])):
                # Get all y values corresponding to this index
                y_inner_values = [inner_vals[y_inner_val_idx] for inner_vals in y_values]
                ax.scatter(x=x_list, y=y_inner_values, c=[colors[y_inner_val_idx]], alpha=0.7, s=20.)

            ax.set_ylabel(y_label, rotation='horizontal', horizontalalignment='right')

            # Only plot x-axis label on last axis
            if y_idx == num_y_axes - 1:
                ax.set_xlabel(x_label)

        # Set up meta info
        fig.suptitle(io.prettify(name))

        # Save
        self._save_plot(
            name=os.path.join(plot_dir, name),
            create_dirs=True
        )

    def _alluvial_flow(self, clusterings: List[ClusteringInfo], title: str):
        """
        Alluvial plot shows individual admissions are clustered in different, competing clusterings. Thus, it
        is possible to e.g. see how a cluster "splits" into multiple clusters in a different clustering.

        :param clusterings:
        :param title:
        :return:
        """

        logging.info(f"Plotting alluvial flow for {len(clusterings)} clusterings ('{title}') ...")

        # 1) Extract flow data from clusterings

        # Store number of clusters for each clustering
        num_clus_by_random_id = {c.random_id: c.num_clus for c in clusterings}
        labels_uniq_by_random_id = {c.random_id: list(np.unique(c.labels)) for c in clusterings}

        # Formatting functions
        def name_for_clustering(clustering_info):
            return f"{clustering_info.random_id} ({clustering_info.num_clus})"

        node_name_sep = ", "

        def name_for_cluster(clustering_info, clus_label):
            return name_for_clustering(clustering_info) + node_name_sep + str(clus_label)

        # Every cluster in every clustering will be a 'node' in the plot and flow is shown between each pair of
        # adjacent nodes
        flow = []
        for clus_left, clus_right in zip(clusterings[:-1], clusterings[1:]):

            # Get outgoing flow for each label of the left clustering
            cluster_labels = np.unique(clus_left.labels)
            for lab in cluster_labels:
                flow_source_indices = np.where(clus_left.labels == lab)[0]
                flow_destinations = clus_right.labels[flow_source_indices]
                flow_destination_stats = np.unique(flow_destinations, return_counts=True)

                for flow_dest_label, flow_amount in zip(*flow_destination_stats):
                    flow.append(
                        {
                            'source': name_for_cluster(clus_left, lab),
                            'source_clustering': clus_left,
                            'amount': flow_amount,
                            'destination': name_for_cluster(clus_right, flow_dest_label),
                            'destination_clustering': clus_right
                        }
                    )

        # Get node names from flow
        node_names = list(np.unique(sum([[f['source'], f['destination']] for f in flow], [])))

        # (every source and every destination)

        # Find out color for each of the nodes

        def color_to_rgba_string(color, alpha=None):
            r, g, b, a = color
            if alpha is not None:
                a = alpha
            return f"rgba({r}, {g}, {b}, {a})"

        node_colors = []
        for name in node_names:
            # Get back original k means option and the cluster label for this node
            clus_node_name, label = name.split(node_name_sep)
            clus_random_id = clus_node_name.split(" ")[0]
            clus_num_clus = num_clus_by_random_id[clus_random_id]
            label = int(label)

            # Get color based on color map
            cm = self._colormap(num_colors=clus_num_clus, index=self.clustering_cm_index)
            clus_labels_uniq = labels_uniq_by_random_id[clus_random_id]
            node_colors.append(cm[clus_labels_uniq.index(label)])

        # 2) Plot the flow
        flow_sources = [node_names.index(f['source']) for f in flow]
        flow_targets = [node_names.index(f['destination']) for f in flow]

        # Rename node names to be more pleasing to the eye (remove random cluster ID)
        nn_id_and_counts = list(np.unique([" ".join(n.split(" ")[:2]) for n in node_names]))
        observed_cluster_counts = []  # make sure that clustering name is unique w.r.t. cluster count
        node_renamings = {}
        for node_info in nn_id_and_counts:
            clus_random_id, clus_count_info = node_info.split(" ")
            clus_count = "".join(c for c in clus_count_info if c.isdigit())

            if clus_count not in observed_cluster_counts:
                observed_cluster_counts.append(clus_count)
                node_renamings[node_info] = f"[{clus_count}],"
        for node_name_old, node_name_new in node_renamings.items():
            node_names = [n.replace(node_name_old, node_name_new) for n in node_names]

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=node_names,
                color=[color_to_rgba_string(col) for col in node_colors]
            ),
            link=dict(
                source=flow_sources,
                target=flow_targets,
                value=[f['amount'] for f in flow],
                color=[color_to_rgba_string(node_colors[fs], alpha=0.5) for fs in flow_sources]
            ))])

        fig.update_layout(
            title={
                'text': f"Alluvial Flow Between {len(clusterings)} Clusterings, "
                        f"{len(clusterings[0].labels)} Admissions ({title})",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            font_size=12
        )

        # Pick a nice filename
        plot_base_path = os.path.join("alluvial_flow")
        if len(clusterings) <= 3:
            filename_appendix = "_to_".join([str(c.num_clus) for c in clusterings])
            plot_base_path = os.path.join(plot_base_path, "pairs" if len(clusterings) == 2 else "tuples")
            write_html = False
        else:
            filename_appendix = "all"
            write_html = True

        # Save plot
        pixel_width = 2000
        pixel_height = 1000
        for scale in np.linspace(0.75, 2, 5):
            plot_path = self._get_plot_path(
                os.path.join(plot_base_path, f"alluvial_flow_{filename_appendix}_scale_{scale:0.2f}"),
                rel_dirs=True,
                create_dirs=True
            )
            fig.write_image(plot_path, width=pixel_width / scale, height=pixel_height / scale, scale=scale)

        # Also write as interactive HTML
        if write_html:
            fig.write_html(
                os.path.join(os.path.split(plot_path)[0], "alluvial_flow.html")
            )

    def plot_alluvial_flow(self, clusterings=None, max_num_clusterings=20):
        # Only plot the interesting clusterings
        if clusterings is None:
            clusterings = self.clustering.clusterings

        if len(clusterings) == 0:
            return

        # Sort the clusterings by their number of clusters
        clusterings.sort(key=lambda c: c.num_clus)

        # Filter the clusterings to a manageable number
        if len(clusterings) > max_num_clusterings:
            plotted_clusterings_indices = np.unique(
                np.linspace(0, len(clusterings) - 1, max_num_clusterings).astype(int)
            )
            clusterings = [clustering for (clustering_idx, clustering) in enumerate(clusterings)
                           if clustering_idx in plotted_clusterings_indices]

        # Plot for all (interesting) clusterings
        self._alluvial_flow(clusterings=clusterings, title='All')

        # Plot the alluvial flow plot in multiple different constellations of clusterings
        for clustering_tuple in itertools.combinations(clusterings, 2):
            self._alluvial_flow(clusterings=list(clustering_tuple), title='Pair')
        for clustering_tuple in itertools.combinations(clusterings, 3):
            self._alluvial_flow(clusterings=list(clustering_tuple), title='Triple')

    def plot_clustering_robustness(self):
        """
        Plot for each algorithm how robustness of clustering changed with changing options
        :return:
        """
        # Get robustness info about all clusterings, then plot it
        all_clusterings = self.clustering.clusterings_all  # type: List[ClusteringInfo]
        self._plot_clustering_robustness(all_clusterings=all_clusterings)

    def _plot_clustering_robustness(self, all_clusterings: List[ClusteringInfo]):
        # Plot for each algorithm: Split robustness info into lists based on algorithm
        robustness_by_algo = {}
        for info in all_clusterings:
            if info.algorithm not in robustness_by_algo:
                robustness_by_algo[info.algorithm] = []
            robustness_by_algo[info.algorithm].append(info)

        for algorithm, robustness in robustness_by_algo.items():

            # Gather information about what option values lead to which robustness
            option_robustness = {}
            for info in robustness:
                options = info.options

                for opt_name, opt_val in options.items():

                    # New entry for the option name (e.g. 'k' for k-means)
                    if opt_name not in option_robustness:
                        option_robustness[opt_name] = {}

                    # Save the robustness resulting from this option value
                    option_robustness[opt_name][opt_val] = info

            # Plot for each of the options
            for opt_name, info_by_val in option_robustness.items():
                # Sort robustness info by option value
                vals_sorted = sorted(info_by_val.keys())
                info_sorted = [info_by_val[val] for val in vals_sorted]

                # Plot how the robustness measures change with a changing value for this option
                rob_names = list(robustness_functions.keys())
                rob_vals = [[[r[rm] for r in info.robustness_measurements] for info in info_sorted] for rm in rob_names]
                rob_labels = [io.prettify(rm) for rm in rob_names]

                # Also plot effective number of clusters (which may be lower than the actual number of clusters if
                # the points are clumped in few clusters)
                num_clus_eff = [[ci.num_clus_effective] for ci in info_sorted]

                # Compose plotted data
                y_labels = rob_labels + ['Num Cluster (effective)']
                y_values = rob_vals + [num_clus_eff]

                self.plot_multi_y_function(
                    x_list=vals_sorted,
                    x_label=opt_name,
                    y_lists=y_values,
                    y_labels=y_labels,
                    plot_dir="clustering_robustness",
                    name=f"robustness_{algorithm}_{opt_name}"
                )

    def plot_clustering_robustness_test(self, max_label: int = 5, num_labels: int = 1500, num_resamples: int = 10,
                                        shuffle: bool = False):
        """
        Plots how clustering robustness metrics respond to different percentages of randomized values. The resulting
        plot allows judging how the metrics respond to changes in labelings.

        :param max_label:
        :param num_labels:
        :param num_resamples:
        :param shuffle: Shuffle the "randomly changed" labels instead of re-sampling them

        :return:
        """

        # Generate the test labeling
        labels_1 = np.random.randint(low=0, high=max_label, size=num_labels)

        # List of robustness measures
        robustness_measures = {
            'mutual_info': [],
            'rand_score': [],
            'jaccard_score': []
        }

        # Iterate over percentages
        fractions = np.linspace(0, 1, 250)
        for randomly_changed_fraction in fractions:

            # Add lists for measures
            fraction_mutual_info = []
            fraction_rand_scores = []
            fraction_jaccard_scores = []

            # Re-sample a few times
            for _ in range(num_resamples):

                # Randomly change a fraction of the labels and check how similar the two labelings are after the change
                labels_2 = np.copy(labels_1)
                if randomly_changed_fraction != 0:
                    changed_indices = np.random.choice(
                        np.arange(num_labels),
                        size=int(np.ceil(randomly_changed_fraction * num_labels)),
                        replace=False
                    )

                    # Create labels_2 as identical to labels_1 except for the changed indices
                    if not shuffle:
                        # Re-sample from label distribution so that the labels at the changed_indices are replaced with
                        # completely now labels
                        labels_2[changed_indices] = np.random.randint(low=0, high=max_label, size=len(changed_indices))
                    else:
                        # Merely shuffle the labels at the changed_indices
                        shuffled_labels = labels_2[changed_indices]
                        np.random.shuffle(shuffled_labels)
                        labels_2[changed_indices] = shuffled_labels

                # Check each of the robustness measures
                fraction_mutual_info.append(
                    mutual_information(
                        clus_1_labels=labels_1,
                        clus_2_labels=labels_2
                    )
                )

                # Compute adjusted rand score
                fraction_rand_scores.append(
                    rand_score(
                        clus_1_labels=labels_1,
                        clus_2_labels=labels_2
                    )
                )

                # Jaccard index
                fraction_jaccard_scores.append(
                    jaccard_score_label_agnostic(
                        clus_1_labels=labels_1,
                        clus_2_labels=labels_2
                    )
                )

            # Add this fraction's measures to list
            robustness_measures['mutual_info'].append(fraction_mutual_info)
            robustness_measures['rand_score'].append(fraction_rand_scores)
            robustness_measures['jaccard_score'].append(fraction_jaccard_scores)

        # Plot
        if shuffle:
            method = 'shuffling'
        else:
            method = 'resampling'
        measures_sorted = sorted(robustness_measures.keys())
        self.plot_multi_y_function(
            x_list=fractions,
            x_label="Fraction of Random Labels",
            y_lists=[robustness_measures[rm] for rm in measures_sorted],
            y_labels=[io.prettify(rm) for rm in measures_sorted],
            plot_dir="clustering_robustness",
            name=f"robustness_measures_test_{method}"
        )

    def plot_numerical_data_clusters_violins(self):
        """
        Plots violin plots
            - for each numerical data attribute (static and dynamic)
            - for each clustering
            - comparing distributions between clusters
        :return:
        """

        # Prepare static data: Make sure it's numerical
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        self.preprocessor.split_days_in_care_by_survival(static_info)
        static_info = {stat_name: stat_vals for (stat_name, stat_vals) in static_info.items()
                       if type(stat_vals[0]) not in [str, bool]}  # Remove non-numerical attributes

        # Get dynamic data (grouped by dyn attribute)
        dyn_times, dyn_info = self.preprocessor.get_dyn_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split),
            with_times=True
        )

        # Derive temporal deltas (between adjacent time points)
        dyn_delta_info = self.preprocessor.dyn_medical_data_temporal_delta(
            dyn_info=dyn_info,
            dyn_times=dyn_times
        )

        # Determine trend line (fitted line onto time series) for dynamic attributes
        dyn_trend_info = self.preprocessor.dyn_medical_data_fit_lines(
            adm_indices=self.trainer.get_split_indices(self.split)
        )

        # Count dynamic time steps
        dyn_time_steps = [[len(v) for v in adm_vals] for adm_vals in dyn_info.values()]
        dyn_time_steps = np.array(dyn_time_steps)  # shape: (num_dyn_attr, num_admissions)
        dyn_time_steps = np.sum(dyn_time_steps, axis=0)
        dyn_info['num_time_steps'] = list(dyn_time_steps)

        # Cache aggregated dynamic data
        aggregated = {}

        # Plot for each clustering
        for clustering_info in self.clustering.clusterings:

            # Create a new directory for the plots: There will be a lot of them
            violin_plot_dir = os.path.join(self._get_dir_for_clustering(clustering_info), "violins_numerical")

            # Plot for each of the numerical data attributes
            for data_kind_name, numerical_data in [("static", static_info),
                                                   ("dynamic", dyn_info),
                                                   ("dynamic_delta", dyn_delta_info),
                                                   ("dynamic_trend", dyn_trend_info)]:

                for num_data_name, num_data_vals in numerical_data.items():

                    # Aggregate data
                    data_to_be_plotted = []
                    if data_kind_name == "dynamic" and not num_data_name == "num_time_steps":

                        # Only aggregate if this has not been done
                        if data_kind_name not in aggregated:
                            aggregated[data_kind_name] = {}
                        if num_data_name not in aggregated[data_kind_name]:
                            aggregated[data_kind_name][num_data_name] = {}

                        for agg_func_name, agg_func in aggregation_functions.items():

                            if agg_func_name not in aggregated[data_kind_name][num_data_name]:
                                aggregated[data_kind_name][num_data_name][agg_func_name] = \
                                    self.evaluator.apply_agg_func(num_data_vals, agg_func)

                            data_to_be_plotted.append(
                                {
                                    'data': aggregated[data_kind_name][num_data_name][agg_func_name],
                                    'agg_func_name': agg_func_name
                                }
                            )
                    else:
                        # No need to aggregate static data, it can directly be added to the list of data to be plotted
                        data_to_be_plotted.append(
                            {
                                'data': num_data_vals,
                                'agg_func_name': None  # Signals that no aggregation took place
                            }
                        )

                    for violin_plot_info in data_to_be_plotted:
                        self._plot_violins_numerical_data(
                            data_kind_name=data_kind_name,
                            numerical_attr_name=num_data_name,
                            numerical_attr_vals=violin_plot_info['data'],
                            agg_func_name=violin_plot_info['agg_func_name'],
                            clustering_info=clustering_info,
                            plot_dir=violin_plot_dir
                        )

    def _plot_violins_numerical_data(self, data_kind_name, numerical_attr_name, numerical_attr_vals, agg_func_name,
                                     clustering_info, plot_dir, outlier_percentage=1,
                                     clusters_per_page: int = 10):
        """
        Plots violin plots of numerical data for each cluster next to one another.

        :param data_kind_name:
        :param numerical_attr_name:
        :param numerical_attr_vals:
        :param agg_func_name:
        :param clustering_info:
        :param plot_dir:
        :param outlier_percentage:
        :param clusters_per_page:
        :return:
        """

        # Aggregation: Either aggregation of data was performed or not
        agg_key = agg_func_name if agg_func_name is not None else no_aggregation_key
        # (agg_key is used to retrieve statistical test results for the data being plotted)

        # For each cluster, prepare a flat list of observed values
        cluster_labels_uniq = sorted(np.unique(clustering_info.labels))
        p_value_unknown_token = "?"
        clusters_to_plot = []
        for cluster_label in cluster_labels_uniq:
            # Get list of values observed within cluster
            inside_cluster_vals = [numerical_attr_vals[idx]
                                   for idx in np.where(clustering_info.labels == cluster_label)[0]]

            # Flatten list if necessary (e.g. for dynamic_delta)
            if type(inside_cluster_vals[0]) == list:
                inside_cluster_vals = sum([list(sl) for sl in inside_cluster_vals], [])

            # Retrieve KS test results
            test_result_known = False
            if data_kind_name in self.evaluator.report['distribution_comparison']:
                data_kind_attr_report = self.evaluator.report['distribution_comparison'][data_kind_name]
                if numerical_attr_name in data_kind_attr_report:
                    attr_report = data_kind_attr_report[numerical_attr_name]
                    stat_test = attr_report[clustering_info.random_id][agg_key]

                    if cluster_label in stat_test:
                        cluster_stat_test_p_value = f"{stat_test[cluster_label]:0.4f}"
                        test_result_known = True
            if not test_result_known:
                cluster_stat_test_p_value = p_value_unknown_token  # If cluster is not part of evaluation, don't
                # show p value.
                # It may happen that a cluster is not part of the evaluation for a specific dynamic attribute
                # if all points in the cluster lack valid dynamic data for the attribute.

            # Remove None entries (which are caused by aggregation when the list of values for an admission is empty
            # or for the days_in_care attributes split by survival)
            inside_cluster_vals = [v for v in inside_cluster_vals if v is not None and not np.isnan(v)]

            # Don't plot this cluster if it only has a single point
            if len(inside_cluster_vals) < 2:
                continue

            # Generate label that includes the p value of the cluster
            label_pretty = io.label_for_cluster_label(cluster_label)
            if cluster_stat_test_p_value is not p_value_unknown_token:
                label_pretty += f" (p={cluster_stat_test_p_value})"
            label_pretty += f"\nn = {len(inside_cluster_vals)},\n" \
                            f"median = {np.median(inside_cluster_vals):0.4f},\n" \
                            f"iqr = {scipy.stats.iqr(inside_cluster_vals, nan_policy='omit'):0.4f}"

            # Save information about this cluster
            cluster_plot_info = {
                'label': cluster_label,
                'label_pretty': label_pretty,
                'vals': inside_cluster_vals,
                'num_data': len(inside_cluster_vals),
                'test_p_value_str': cluster_stat_test_p_value
            }
            clusters_to_plot.append(cluster_plot_info)

        # Set attribute name (and with it, y-axis label)
        attr_name = f"{io.prettify(numerical_attr_name)}"
        if "_" in data_kind_name:
            # Append data kind if it is a complex type
            attr_name += f" ({io.prettify(data_kind_name)})"

        # Generate colormap (according to regular cluster colors)
        cm = self._colormap(num_colors=len(cluster_labels_uniq), index=self.clustering_cm_index)

        # Plot in multiple pages if there are too many clusters for a single plot
        num_pages = int(np.ceil(len(clusters_to_plot) / clusters_per_page))
        for page_idx, page_plotted_clusters in enumerate(io.chunk_list(clusters_to_plot, clusters_per_page)):

            # Build DataFrame for plotting with Seaborn
            cluster_vals_data = []
            for clus_info in page_plotted_clusters:
                for val in clus_info['vals']:
                    cluster_vals_data.append([val, clus_info['label_pretty']])
            cluster_vals_df = pd.DataFrame(cluster_vals_data, columns=[attr_name, "Cluster"])

            # Remove data outliers
            all_values = cluster_vals_df[attr_name].values
            if len(all_values) > 0:
                bottom_limit = np.percentile(all_values, outlier_percentage / 2)
                top_limit = np.percentile(all_values, 100 - outlier_percentage / 2)
                non_outlier_data = (cluster_vals_df[attr_name] <= top_limit) & \
                                   (cluster_vals_df[attr_name] >= bottom_limit)
                cluster_vals_df = cluster_vals_df[non_outlier_data]

            # Plotting only makes sense when there are points
            if len(cluster_vals_df) == 0:
                logging.info("Not plotting cluster violin plot since there are no points.")
                return

            # Color each violin according to the cluster
            violin_coloring = {c['label_pretty']: cm[cluster_labels_uniq.index(c['label'])]
                               for c in page_plotted_clusters}

            # Create violin plot
            fig_h = 5
            fig_w = 2 + 2 * len(page_plotted_clusters)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            try:
                violin = sns.violinplot(
                    x="Cluster",
                    y=attr_name,
                    data=cluster_vals_df,
                    order=[c['label_pretty'] for c in page_plotted_clusters],
                    palette=violin_coloring,
                    ax=ax,
                    cut=0,  # Don't let density estimate extend past extreme values
                    scale='width'
                )
            except ValueError:
                logging.error(f"Could not plot violin plot! Cluster DataFrame: {cluster_vals_df}")

            # Title
            title = f"Distribution of {io.prettify(data_kind_name)} attribute " \
                    f"{io.prettify(numerical_attr_name)}\n" \
                    f"({100 - outlier_percentage}% of data: outliers removed in plot"
            if agg_key != no_aggregation_key:
                title += f"; Per-Admission-Aggregation: {agg_key}"
            title += ")"
            if num_pages > 1:
                title += f"\n{page_idx + 1}/{num_pages}"
            ax.set_title(title)

            # Write plot
            plot_path = os.path.join(
                plot_dir,
                data_kind_name,
                f"violin_dist_{data_kind_name}_{io.sanitize(numerical_attr_name)}_{clustering_info.random_id}"
            )
            if agg_key != no_aggregation_key:
                plot_path += f"_{agg_key}"
            if num_pages > 1:
                plot_path += f"_page_{page_idx + 1:02d}_of_{num_pages:02d}"

            self._save_plot(
                plot_path,
                create_dirs=True,
                dpi=2 * self.dpi,
                seaborn_ax=ax
            )

    def _plot_dendrogram(self, linkage_type='single'):
        # Get features
        features = self.trainer.compute_features(self.split)
        # shape: (m, n) where
        # m = num samples
        # and n = num features

        # Measure all pairwise distances
        dists = pdist(features, 'euclidean')

        # Determine point linkage
        linkage = hierarchy.linkage(dists, linkage_type, optimal_ordering=True)

        # Prepare plotting of dendrogram
        fig, ax = plt.subplots()

        # Increase system recursion limit: Otherwise, scipy plotting code can fail if a large number of admissions is
        # plotted
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000 * features.shape[0])

        # Plot dendrogram
        dn = hierarchy.dendrogram(linkage)

        # Title
        ax.set_title(f"Dendrogram (Features {features.shape[0]}x{features.shape[1]}, linkage: '{linkage_type}')")

        # Save plot
        plot_path = os.path.join(
            "feature_space",
            "dendrograms",
            f"dendrogram_{linkage_type}"
        )
        self._save_plot(
            plot_path,
            create_dirs=True
        )

        # Restore original recursion limit
        sys.setrecursionlimit(original_recursion_limit)

    def plot_dendrograms(self):
        # Plot dendrograms with different linkage types
        for linkage_type in ['single']:  # , 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']:
            self._plot_dendrogram(linkage_type=linkage_type)

    def plot_icd_grouped_tornado(self, icd_level: int = 1):
        """
        Plots stacked bar chart (tornado plot) for ICD diagnoses and procedures
        :param icd_level: Level of the ICD hierarchy (top level == 1 is the default)
        :return:
        """

        logging.info("Plotting ICD tornado plots ...")
        if icd_level != 1:
            logging.info(f"ICD Level: {icd_level}")

        # Retrieve static categorical data (only validation points)
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        static_categorical = [static_categorical[idx] for idx in self.trainer.get_split_indices(self.split)]

        # Plot for each of the two ICD kinds: diagnoses and procedures
        for icd_kind in ['icd9_code_diagnoses', 'icd9_code_procedures']:
            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_kind] for stat in static_categorical]

            # Find out which ICD codes are possible in the population (this list is also the interpretation
            # for the present_icd_codes_indices)
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_kind]

            # Plot for each clustering
            for clustering_info in self.clustering.clusterings:
                self._plot_icd_tornado(
                    clustering_info=clustering_info,
                    icd_kind=icd_kind,
                    possible_codes=possible_icd_codes,
                    present_codes_indices=present_icd_codes_indices,
                    icd_level=icd_level
                )

        logging.info("Plotting ICD tornado plots done!")

    @staticmethod
    def _top_n_cluster_labels(labels, n):
        cluster_labels_uniq, cluster_labels_counts = np.unique(labels, return_counts=True)
        cluster_labels_uniq = [t[1] for t in sorted(zip(cluster_labels_counts, cluster_labels_uniq), reverse=True)]
        cluster_labels_uniq = cluster_labels_uniq[:n]
        cluster_labels_uniq.sort()
        return cluster_labels_uniq

    @staticmethod
    def _icd_categ_occurrence_prevalence(icd_kind: str, possible_codes: List[str],
                                         present_codes_indices: List[List[int]],
                                         icd_level: int = 1) -> Dict[str, object]:
        """

        Determine prevalence in three steps:
        1) Finding: For each possible ICD code, identify the admissions with the code. Add these admissions
                    (or their indices) to a set.
        2) Counting: Count the size of the set of admissions for each of the categories
        3) Relation: Divide the absolute counts for each broader category by the number of admissions

        ICD LEVELS:
        For ICD level 1 (top level), a dictionary of type Dict[str, float] will be returned, i.e. for each
        category (str), the prevalence (float) is returned.

        For ICD level 2, a dictionary of type Dict[str, Dict[str, float]] will be returned. Here, each value of the
        dictionary is a dictionary listing the prevalence of each level-2 category (for that level1 category).

        :param icd_kind:
        :param possible_codes:
        :param present_codes_indices:
        :param icd_level: level of the ICD hierarchy
        :return:
        """

        # Find admissions with occurrences for each of the possible codes
        admission_indices_by_categ = {}
        for code_index, code in enumerate(possible_codes):

            # Find out ICD "path", i.e. path in the ICD tree from root to the respective code
            node = IcdInfo.icd_tree_search(
                icd_kind=icd_kind,
                icd_code=code
            )
            node_path = node.path[:icd_level + 1]  # Only go until requested ICD level
            node_path = node_path[1:]  # Skip root node
            node_path = [IcdInfo.name_for_node(n) for n in node_path]

            # Create a dictionary entry for each node in the ICD path
            counter = admission_indices_by_categ
            for path_idx, path_node_name in enumerate(node_path):
                if path_node_name not in counter:
                    if path_idx + 1 < len(node_path):
                        counter[path_node_name] = {}
                    else:
                        counter[path_node_name] = set()
                counter = counter[path_node_name]

            # Find admissions that have the code
            adms_with_code = [adm_idx for (adm_idx, adm_codes) in enumerate(present_codes_indices)
                              if code_index in adm_codes]
            counter.update(adms_with_code)

        # Divide the total counts that each category received by the total number of admissions
        def calc_freq(dic, total_count):
            for k, v in dic.items():
                if type(v) == dict:
                    v = calc_freq(v, total_count)
                else:
                    v = len(v) / total_count
                dic[k] = v
            return dic

        num_adms = len(present_codes_indices)
        freq_by_categ = calc_freq(admission_indices_by_categ, num_adms)

        return freq_by_categ

    def _plot_icd_tornado_inner(self, plot_context, pop_categ_prevalence, plotted_cluster_labels, clustering_info,
                                icd_kind, present_codes_indices, possible_codes, icd_level, cm, clusters_orig_uniq,
                                bars_plotted_per_page):
        # Gather prevalence data for each of the clusters (in each of the ICD categories)
        cluster_categ_diff_freq = {}  # type: Dict[str, Dict[int, float]]
        cluster_sizes = {}  # type: Dict[int, int]
        for cluster_label in plotted_cluster_labels:

            # Constrain admissions to those inside of this cluster
            cluster_adm_indices = [idx for (idx, label) in enumerate(clustering_info.labels) if label == cluster_label]
            cluster_adms = [adm_codes for (adm_idx, adm_codes) in enumerate(present_codes_indices)
                            if adm_idx in cluster_adm_indices]

            # Remember cluster size
            cluster_sizes[cluster_label] = len(cluster_adm_indices)

            # Determine prevalence of every category in this cluster
            freq_inside_cluster = self._icd_categ_occurrence_prevalence(
                icd_kind=icd_kind,
                possible_codes=possible_codes,
                present_codes_indices=cluster_adms,
                icd_level=icd_level
            )

            if plot_context in freq_inside_cluster:
                # This is an ICD-level-2 plot
                cluster_categ_prevalence = freq_inside_cluster[plot_context]
            else:
                # This is an ICD-level-1 plot
                cluster_categ_prevalence = freq_inside_cluster

            # Save the change in frequency for each category (and this cluster)
            for labeled_categ in pop_categ_prevalence.keys():

                # New entry for this category
                if labeled_categ not in cluster_categ_diff_freq:
                    cluster_categ_diff_freq[labeled_categ] = {}

                # Find out prevalence of this category for the full population
                prevalence_pop = pop_categ_prevalence[labeled_categ]

                # Find out prevalence of this category for the cluster
                if labeled_categ in cluster_categ_prevalence:
                    prevalence_cluster = cluster_categ_prevalence[labeled_categ]
                else:
                    prevalence_cluster = 0.  # If this labeled category does not occur in the cluster, it means that
                    # there is not a single patient with a code for the category. Thus, the prevalence is 0.

                # Calculate difference in frequency
                diff_freq = prevalence_cluster - prevalence_pop

                # Save relative prevalence
                cluster_categ_diff_freq[labeled_categ][cluster_label] = diff_freq

        # Determine order in which ICD categories should be plotted
        # (sort by maximum difference in frequency to the population)
        icd_categs_plot_order_total = sorted(
            pop_categ_prevalence.keys(),
            key=lambda categ: max(np.abs(list(cluster_categ_diff_freq[categ].values()))),
            reverse=False
        )

        # Set x-axis bounds of plot: It should be as "zoomed in" as possible while still showing all bars
        max_bar_width = max(
            [max(100 * np.abs(list(cluster_categ_diff_freq[categ].values()))) for categ in icd_categs_plot_order_total]
        )
        x_limit = max_bar_width * 1.075

        # Pagination: Split the plot into multiple pages with few categories per page
        bars_per_category = len(plotted_cluster_labels)
        max_categs_per_page = bars_plotted_per_page // bars_per_category
        num_pages = int(np.ceil(len(icd_categs_plot_order_total) / max_categs_per_page))
        avg_categs_per_page = int(np.ceil(len(icd_categs_plot_order_total) / num_pages))
        pages_categs = [icd_categs_plot_order_total[avg_categs_per_page * seg:avg_categs_per_page * (seg + 1)]
                        for seg in range(num_pages)]

        # Plot each of the pages
        for page_idx, icd_categs_plot_order in enumerate(reversed(pages_categs)):

            # Plot this clustering's distribution
            bar_height = 0.8
            bar_distance = 0.2
            icd_categ_distance = 3
            num_clus = len(plotted_cluster_labels)
            num_categ = len(icd_categs_plot_order)
            space_per_categ = ((num_clus - 1) * bar_distance + num_clus * bar_height)
            fig_w = 5
            fig_h = int(5 + num_categ * num_clus / 15)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))

            # Calculate offsets of bar positions
            space_between_bars = bar_height + bar_distance
            extremal_bar_pos = (num_clus // 2) * space_between_bars
            y_offsets = np.linspace(extremal_bar_pos, -extremal_bar_pos, num_clus)

            # y position for each ICD category
            ys = np.arange(1, num_categ + 1) * (space_per_categ + icd_categ_distance + extremal_bar_pos)

            # Plot the bars
            label_done = []  # Save for each cluster (which is to say, for each color of the bars) if already labeled
            for y, labeled_categ in zip(ys, icd_categs_plot_order):

                # Get prevalence data for this category
                prev_by_cluster = cluster_categ_diff_freq[labeled_categ]

                # Plot bars for each of the clusters one by one
                for bar_y, (cluster_label, diff_freq) in zip(
                        y + y_offsets,
                        prev_by_cluster.items()
                ):
                    if cluster_label not in label_done:
                        num_pnts_total = len(present_codes_indices)
                        cluster_size_percent = cluster_sizes[cluster_label] / num_pnts_total
                        cluster_size_label = f" ({cluster_sizes[cluster_label]} a., {100 * cluster_size_percent:0.1f}%)"
                        label_kwargs = {
                            'label': io.label_for_cluster_label(cluster_label) + cluster_size_label
                        }
                    else:
                        label_kwargs = {}

                    # Multiply diff_freq by 100 - this causes the plot to read as in percent
                    diff_freq *= 100

                    # Set color
                    color = cm[clusters_orig_uniq.index(cluster_label)]

                    plt.barh(
                        [bar_y],
                        [diff_freq],
                        height=bar_height,
                        color=[color],
                        **label_kwargs
                    )

                    # Save that labeling is done
                    label_done.append(cluster_label)

                    # Display the value as text. It should be positioned in the center of
                    # the 'high' bar, except if there isn't any room there, then it should be
                    # next to bar instead.
                    text_offset = (max_bar_width + 0.05) / 15
                    text_pos = diff_freq + np.sign(diff_freq) * text_offset
                    if text_pos == 0:
                        text_pos += text_offset
                    rel_change_sign = {0: "", 1: "+", -1: "-"}[np.sign(diff_freq)]
                    text_align = "left" if text_pos >= 0 else "right"
                    plt.text(text_pos, bar_y, f"{rel_change_sign}{abs(diff_freq):0.1f} (C{cluster_label})", va='center',
                             ha=text_align, fontdict={'size': 5})

            # Draw a vertical line down the middle
            plt.axvline(0, color='black')

            # Hide all the other spines (=axis lines) but the bottom x-axis
            axes = plt.gca()  # (gca = get current axes)
            axes.spines['left'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['top'].set_visible(False)
            axes.xaxis.set_ticks_position('bottom')

            # Make the y-axis display the ICD categories
            icd_tick_labels = [f"{io.prettify(icd_categ)} ({100 * pop_categ_prevalence[icd_categ]:0.2f}%)"
                               for icd_categ in icd_categs_plot_order]
            plt.yticks(ys, icd_tick_labels)

            # Set the portion of the x-axis to show (we are showing a symmetric view with 0 in the middle)
            plt.xlim(-x_limit, x_limit)

            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Title
            ax.set_ylabel("ICD Category")
            ax.set_xlabel(r"$freq_{cluster} - freq_{pop}$")
            title = f"{io.prettify(icd_kind)}"
            if icd_level == 2:
                title += f"\n(Category: {io.prettify(plot_context)})"
            if len(clusters_orig_uniq) > len(plotted_cluster_labels):
                title += f" (top {len(plotted_cluster_labels)} of {len(clusters_orig_uniq)} clusters)"
            if num_pages > 1:
                title += f"\n({page_idx + 1} of {num_pages})"
            ax.set_title(title)

            # Write plot
            plot_filename = f"ICD_tornado_{icd_kind.split('_')[-1]}_level_{icd_level}"
            if icd_level == 2:
                plot_filename += f"_{io.prettify(plot_context)}"
            if num_pages > 1:
                plot_filename += f"_page_{page_idx + 1:02d}_of_{num_pages:02d}"
            plot_path = os.path.join(
                self._get_dir_for_clustering(clustering_info),
                f"ICD_tornado_level_{icd_level}",
                plot_filename
            )
            self._save_plot(
                plot_path,
                create_dirs=True,
                dpi=2 * self.dpi
            )

    def _plot_icd_tornado(self, clustering_info: ClusteringInfo, icd_kind: str, possible_codes: List[str],
                          present_codes_indices: List[List[int]], icd_level: int = 1, max_num_clusters: int = 45):

        # Determine prevalence of each of the ICD categories within the population
        freq_pop = self._icd_categ_occurrence_prevalence(
            icd_kind=icd_kind,
            possible_codes=possible_codes,
            present_codes_indices=present_codes_indices,
            icd_level=icd_level
        )

        # Constrain to top few clusters
        clusters_orig_uniq = list(np.unique(clustering_info.labels))
        plotted_cluster_labels = self._top_n_cluster_labels(
            labels=clustering_info.labels,
            n=max_num_clusters
        )

        # Colormap for clusters (create colors for *all clusters*. Of course, only the plotted clusters
        # will actually be shown. This is important for color addressing, especially with respect to other plots of
        # the same clustering.)
        cm = self._colormap(num_colors=len(clusters_orig_uniq), index=self.clustering_cm_index)

        # The number of plots resulting from this depends on the ICD level: For icd_level == 1, only a single plot will
        # be produced. The plot will contain all level-1 categories. If icd_level == 2, the number of plots produced is
        # equal to the number of level-1 categories.
        if icd_level == 1:
            plots_produced = {icd_kind: freq_pop}
        elif icd_level == 2:
            plots_produced = freq_pop
        else:
            assert False, f"icd_level > 2 not supported! (chosen icd_level = {icd_level} for {icd_kind})"

        for plot_context, pop_categ_prevalence in plots_produced.items():
            self._plot_icd_tornado_inner(
                plot_context=plot_context,
                pop_categ_prevalence=pop_categ_prevalence,
                plotted_cluster_labels=plotted_cluster_labels,
                clustering_info=clustering_info,
                icd_kind=icd_kind,
                present_codes_indices=present_codes_indices,
                possible_codes=possible_codes,
                icd_level=icd_level,
                cm=cm,
                clusters_orig_uniq=clusters_orig_uniq,
                bars_plotted_per_page=max_num_clusters
            )

    def plot_dyn_attr_reconstruction_quality(self):
        """
        Plots a bar chart that displays the quality of reconstruction reached for different dynamic data attributes
        :return:
        """

        # Abort if in baseline mode
        if self.trainer.baseline_mode:
            return

        # Find out errors of reconstruction
        model_rec_loss = self.evaluator.get_reconstruction_loss()
        errors = model_rec_loss['rec_error_scores']

        # Plot bars
        dyn_attrs = list(errors.keys())
        dyn_errors = [errors[attr_name] for attr_name in dyn_attrs]
        error_medians, error_lower, error_upper = self._medians_and_errors(dyn_errors)
        error_weighted = self.evaluator.report['rec_error_median_overall_weighted']
        self.plot_named_bars(
            bar_magnitudes=error_medians,
            bar_labels=dyn_attrs,
            title="Dynamic Data Reconstruction Errors\n"
                  f"(Weighted: {error_weighted:0.4e})",
            path="dyn_reconstruction_error_bars",
            bar_errors=(error_lower, error_upper)
        )

    def plot_cluster_similarity_matrix(self):
        self._plot_cluster_similarity_matrix(clusterings=self.clustering.clusterings)

    def _plot_cluster_similarity_matrix(self, clusterings: List[ClusteringInfo]):
        # Extract ICD attributes
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        # Extract ICD nodes (for codes of diagnoses and procedures)
        nodes_by_attr = {}
        for icd_attr_name in icd_attrs:
            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[idx]
                                         for idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            present_icd_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                  for idx in adm_icd_indices]
                                 for adm_icd_indices in present_icd_codes_indices]

            # Save nodes
            nodes_by_attr[icd_attr_name] = present_icd_nodes

        # Create matrix plot for each ICD depth and clustering
        for icd_tree_max_depth in range(1, 4 + 1):

            # Merge ICD nodes for ICD attribute kinds (diagnoses and procedures)
            icd_nodes = [tup[0] + tup[1] for tup in zip(*nodes_by_attr.values())]

            # Get node names at requested level of ICD tree
            icd_code_strings = [
                [IcdInfo.name_for_node(n.path[min(icd_tree_max_depth, len(n.path) - 1)]) for n in adm_nodes]
                for adm_nodes in icd_nodes
            ]

            # Draw plot for each clustering
            for clustering_info in clusterings:
                for apply_softmax_to_rows in [True, False]:
                    self._plot_cluster_similarity_matrix_for_clustering(
                        clustering_info=clustering_info,
                        icd_strings=icd_code_strings,
                        icd_depth=icd_tree_max_depth,
                        softmax_rows=apply_softmax_to_rows
                    )

    def _plot_cluster_similarity_matrix_for_clustering(self, clustering_info: ClusteringInfo,
                                                       icd_strings: List[List[str]], icd_depth: int,
                                                       softmax_rows: bool):

        # Split ICD strings between clusters
        icd_str_split = split_by_cluster(clustering_info.labels, icd_strings)
        labels = sorted(np.unique(clustering_info.labels))

        # Create distribution of ICD codes by counting code occurrences
        counters = []
        for clus_icd_strings in icd_str_split:

            # Count occurrences of all codes for this cluster
            counter = Counter()
            for adm_strings in clus_icd_strings:
                counter.update(adm_strings)
            counters.append(counter)

        # Fill in missing elements in counters
        observed_strings = sorted(np.unique(sum([list(c.keys()) for c in counters], [])))
        counts_arr = []
        for counter in counters:
            dist = [counter[obs_str] for obs_str in observed_strings]
            dist = np.array(dist)
            counts_arr.append(dist)

        # Create matrix by comparing all cluster's distributions to each other
        n = len(counts_arr)
        matrix = np.zeros(shape=(n, n))
        for row_idx in range(n):

            # Get counts array for cluster corresponding to current row
            counts_row = counts_arr[row_idx]

            # Take the softmax - this accentuates the ICD codes that are already strong within the cluster.
            # Then, when comparing the row distribution with the (non-softmaxed) column distribution, the similarity
            # will be greater if the distributions agree about the most-frequent ICD codes. In other words, the
            # comparison will focus on the popular codes and less on the fringe codes.
            if softmax_rows:
                dist_row = softmax(counts_row)
            else:
                dist_row = counts_row / np.sum(counts_row)

            for col_idx in range(n):
                # Normalize column count vector
                dist_col = counts_arr[col_idx] / np.sum(counts_arr[col_idx])

                jensen_shannon_distance = jensenshannon(dist_row, dist_col, base=2)
                # 0 -> same distribution; 1 -> totally different distribution
                matrix[row_idx, col_idx] = jensen_shannon_distance

        # Find out cluster names - we need them for plotting
        labels_pretty = [f"C{l}" for l in labels]

        # Draw matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            matrix,
            xticklabels=labels_pretty,
            yticklabels=labels_pretty,
            square=True,
            annot=True,
            fmt="0.2f",
            linewidth=0.5,
            ax=ax
        )
        ax.set_title(f"ICD code dissimilarity between clusters (ICD depth {icd_depth})")

        # Put x-axis labels at the top
        ax.xaxis.tick_top()

        # Save plot
        softmax_str = "using_softmax" if softmax_rows else "symmetric"
        plot_dir = os.path.join(self._get_dir_for_clustering(clustering_info), "similarity_matrix")
        self._save_plot(
            name=os.path.join(plot_dir, f"matrix_icd_depth_{icd_depth}_{softmax_str}"),
            seaborn_ax=ax,
            create_dirs=True
        )

    def plot_static_categ_bars(self):
        """
        Plots bar chart for each of the static categorical data attributes
        :return:
        """

        logging.info("Plotting static categorical attributes bar chart...")

        # Retrieve static categorical data
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        static_categorical = [static_categorical[idx] for idx in self.trainer.get_split_indices(self.split)]

        # Retrieve static medical data (includes also future values like survival - those are not part of the training)
        static_categorical_also_future = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )

        # Plot for each attribute of static *categorical* data
        for categ_name, static_categ_labels in self.preprocessor.get_static_data_categ_values().items():

            # Get categorical values for this attribute
            if categ_name not in static_categorical[0]:
                # Get static data indices out of non-training data
                static_vals_raw = static_categorical_also_future[categ_name]
                static_categ_values = [[static_categ_labels.index(v)] for v in static_vals_raw]
            else:
                # Get static indices out of training data
                static_categ_values = [stat[categ_name] for stat in static_categorical]

            # Plot for each clustering
            for clustering_info in self.clustering.clusterings:
                self._plot_static_categ_bar_chart_for_clustering(
                    clustering_info=clustering_info,
                    categ_name=categ_name,
                    static_categ_values=static_categ_values,
                    static_categ_labels=static_categ_labels
                )

        logging.info("Plotting static categorical attributes bar chart done!")

    def _plot_static_categ_bar_chart_for_clustering(self, clustering_info, categ_name, static_categ_values,
                                                    static_categ_labels, max_num_clusters=50):

        # Gather information about the distribution of category values within the clusters of the clustering
        categ_freqs = {}
        categ_vals_uniq = list(np.unique(
            sum([val_list for val_list in static_categ_values], [])
        ))

        # Handle few-valued attributes (e.g. boolean attributes like survival or attributes like sex) differently:
        # Instead of plotting the difference between the clusters and population, just show the in-cluster frequencies
        # of values directly.
        simple_mode = len(categ_vals_uniq) < 3

        # Constrain to top few labels (i.e. clusters)
        cluster_labels_uniq = self._top_n_cluster_labels(
            labels=clustering_info.labels,
            n=max_num_clusters
        )

        pop_freq = {}
        for categ_val in categ_vals_uniq:

            # For this categorical value, we need to know the fraction (w.r.t. the number of admissions in total) of
            # admissions residing in a specific cluster. Thus, we want to know how people with categorical value
            # `categ_val` are distributed over the clusters.

            # Check in each of the clusters
            freqs = []
            num_admissions_total = len(static_categ_values)
            num_categ_adms = 0  # Accumulate cluster admissions with the categ value in this accumulator
            for cluster_label in cluster_labels_uniq:
                # Get number of admissions
                # - exhibiting the categorical value we are looking for
                # - inside the cluster for are in
                clus_idcs = np.where(clustering_info.labels == cluster_label)[0]
                static_categ_values_cluster = [static_categ_values[idx] for idx in clus_idcs]
                static_categ_values_cluster = [val_list for val_list in static_categ_values_cluster
                                               if categ_val in val_list]  # Filter to current categ val
                num_target_admissions = len(static_categ_values_cluster)
                num_categ_adms += num_target_admissions

                # Divide by number of admissions in cluster
                num_adms_cluster = len(clus_idcs)
                cluster_categ_fraction = num_target_admissions / num_adms_cluster
                freqs.append(cluster_categ_fraction)

            # Save the distribution for this categorical value
            categ_freqs[categ_val] = freqs

            # Save population frequency
            pop_freq[categ_val] = 100 * (num_categ_adms / num_admissions_total)

        categ_freqs_plotted = {categ_val: 100 * np.array(distribution)
                               for (categ_val, distribution) in categ_freqs.items()}
        if not simple_mode:
            # Subtract population frequency from in-cluster frequencies
            categ_freqs_plotted = {categ_val: distribution - pop_freq[categ_val]
                                   for (categ_val, distribution) in categ_freqs_plotted.items()}

        # Sort the bars by color (similarly-colored bars go next to one another) and by total mass
        bars_colors = []
        bars_color_broader_categ = []
        bars_labels = []
        bars_total_mass = []
        cm = self._colormap(num_colors=len(categ_vals_uniq))
        for categ_val in categ_vals_uniq:

            # Get label
            categ_val_label = static_categ_labels[categ_val]
            legend_label = f"{categ_val_label} ({pop_freq[categ_val]:0.1f}%)"
            bars_labels.append(legend_label)

            # Color: either color using colormap or color by overarching grouping of categorical values
            broader_categ = IcdInfo.icd_categ_level_1(icd_kind=categ_name, icd_code=categ_val_label)
            color = IcdInfo.icd_color_for_code(icd_kind=categ_name, icd_code=categ_val_label)

            # Color using colormap (if this is not an ICD category after all)
            if broader_categ is None or color is None:
                color = cm[categ_vals_uniq.index(categ_val)]
                broader_categ = 0

            bars_colors.append(color)

            # Store the broader color category
            bars_color_broader_categ.append(broader_categ)

            # Compute "total mass" of the bars for this categorical value: the sum over all clusters
            freq_diff = categ_freqs_plotted[categ_val]
            bars_total_mass.append(sum(freq_diff))

        # Sort
        sorting_vals = list(zip(bars_colors, bars_labels, bars_total_mass, bars_color_broader_categ, categ_vals_uniq))
        sorting_vals.sort(key=lambda tup: tup[2])  # Sort by total mass
        sorting_vals.sort(key=lambda tup: tup[3])  # Sort by broader category
        # Unpack lists after sorting
        bars_colors, bars_labels, bars_total_mass, bars_color_broader_categ, categ_vals_uniq = zip(*sorting_vals)

        # Remove one of the values if there are only two different values (i.e. we are in simple mode)
        if simple_mode:
            categ_vals_uniq = categ_vals_uniq[-1:]  # Decrease unique values to a single value (the frequency of which
            # is enough to represent a distribution of boolean values)
            bars_colors = bars_colors[-1:]
            bars_labels = bars_labels[-1:]

        # Plot this clustering's distribution
        fig_w = 4 + 2 * len(cluster_labels_uniq)
        fig_h = 5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # If in simple mode, plot mean value of attribute
        if simple_mode:
            ax.axhline(y=pop_freq[categ_vals_uniq[0]], color="grey")

        # Set up horizontal bar plot
        x_pos_leftmost = np.arange(len(cluster_labels_uniq))
        bar_width = 0.8 / len(categ_vals_uniq)
        ax.set_xticks(x_pos_leftmost + 0.5 * bar_width * (len(categ_vals_uniq) - 1))
        ax.set_xticklabels([f"{io.label_for_cluster_label(lab)}"
                            f" ({len(np.where(clustering_info.labels == lab)[0])} adm.)"
                            for lab in cluster_labels_uniq])

        # Plot bars for each of the clusters
        bar_x_offset = 0
        for categ_val, color, label in zip(categ_vals_uniq, bars_colors, bars_labels):
            # Get difference in frequency (w.r.t. pop freq) over clusters for this category
            freq_diff = categ_freqs_plotted[categ_val]

            ax.bar(
                x_pos_leftmost + bar_x_offset,
                freq_diff,
                bar_width,
                label=label,
                color=color
            )

            # Set next bar off to the left
            bar_x_offset += bar_width

        # Plot horizontal line at y=0
        ax.axhline(y=0, color="grey")

        # Make y-axis symmetric (same distance from 0 for positive and negative)
        max_categ_freq_diff = np.max(np.abs(np.array(sum([list(arr) for arr in categ_freqs_plotted.values()], []))))
        upper_y_lim = 1.1 * max_categ_freq_diff  # Some padding
        if not simple_mode:
            lower_y_lim = -upper_y_lim
        else:
            lower_y_lim = 0
        plt.ylim(lower_y_lim, upper_y_lim)

        # Title
        if not simple_mode:
            ax.set_ylabel(r"$freq_{cluster} - freq_{pop}$")
        else:
            ax.set_ylabel(f"% {io.prettify(categ_name)} = {static_categ_labels[categ_vals_uniq[0]]} in Clusters")
        ax.set_xlabel("Cluster")
        ax.set_title(f"{io.prettify(categ_name)} Distribution over Clusters")

        # Legend - only show it if number of unique categorical values is manageable
        if len(categ_vals_uniq) <= 15:
            ax.legend()

        # Write plot
        plot_path = os.path.join(
            self._get_dir_for_clustering(clustering_info),
            "static_categ_bars",
            f"static_categ_bars_{io.sanitize(categ_name)}"
        )
        self._save_plot(
            plot_path,
            create_dirs=True,
            dpi=2 * self.dpi
        )

        # Write the information to disk as a CSV file (for easier manual analysis)

        # Don't perform this step for ICD codes: it is too unwieldy to analyze by hand
        if "icd" in categ_name:
            return

        all_categ_freqs_saved = 100 * np.array([categ_freqs[cv] for cv in categ_vals_uniq]).T
        # shape: (num_clus, len(static_categ_labels))
        saved_table_data = [["Population"] + [pop_freq[cv] for cv in categ_vals_uniq]]
        for cluster_label, categ_freqs_saved in zip(cluster_labels_uniq, all_categ_freqs_saved):
            saved_table_data.append([io.label_for_cluster_label(cluster_label)] + list(categ_freqs_saved))
        cluster_name_col = "Cluster Name"
        freq_table = pd.DataFrame(saved_table_data, columns=[cluster_name_col] + list(bars_labels))
        freq_table = freq_table.set_index(cluster_name_col)
        table_path = os.path.splitext(self._get_plot_path(plot_path, rel_dirs=True))[0] + ".csv"
        freq_table.to_csv(table_path)

    def plot_embedding(self, features, labels, weights, name, coloring=None, color_group_colors_by_label=None,
                       min_font_size=10, max_font_size=18,
                       min_alpha=0.6, max_alpha=0.8, full_path=None):
        """
        Plots an embedding
        :param features: np array of shape (num_points, embedding_dim)
        :param labels: list of label strings of length num_points
        :param weights: list of "importance" of points from 0 to 1, length: num_points
        :param name:
        :param coloring: If None, use random colors. Otherwise, it must be a list of colors of length num_points
        :param color_group_colors_by_label: If None, don't plot "legend" of colors for each group of colors.
            Otherwise, plot each color group label in its color outside the main plotting area.
            dict with color label as key and color as value.
        :param min_font_size:
        :param max_font_size:
        :param min_alpha:
        :param max_alpha:
        :param full_path:
        :return:
        """

        # Transform feature space to 2d using t-sne
        tsne_features = self.projection_tsne(
            features=features
        )

        # Plot
        fig, ax = plt.subplots()
        for point_idx in range(tsne_features.shape[0]):
            ax.scatter(
                tsne_features[point_idx, 0], tsne_features[point_idx, 1],
                alpha=0,  # points are invisible, but without plotting them,
                # matplotlib does not want to plot labels
                s=0
            )

        # Adjust weights so that they sum up to 1
        weights = np.array(weights)
        weights /= np.sum(weights)

        # Plot label for each point
        if coloring is None:
            coloring = np.random.rand(tsne_features.shape[0], 3)  # random colors if user didn't specify colors
        for point_idx, label in enumerate(labels):
            weight = weights[point_idx]
            ax.annotate(
                label,
                (tsne_features[point_idx, 0], tsne_features[point_idx, 1]),
                alpha=np.interp([weight], [0, 1], [min_alpha, max_alpha])[0],
                fontsize=np.interp([weight], [0, 1], [min_font_size, max_font_size]),
                c=coloring[point_idx],
                horizontalalignment='center',
                verticalalignment='center'
            )

        # Disable axes (they are meaningless for an embedding)
        plt.axis('off')

        # If requested, plot each color group label in its color next to the main plot
        if color_group_colors_by_label is not None:
            color_label_y_positions = np.linspace(0.2, 0.8, len(color_group_colors_by_label))
            color_group_labels_sorted = sorted(color_group_colors_by_label.keys())
            for label_idx, color_group_label in enumerate(color_group_labels_sorted):
                ax.text(
                    1.04,  # x pos (0 to 1 -> left to right)
                    color_label_y_positions[label_idx],  # y pos (0 to 1 -> bottom to top)
                    color_group_label,
                    transform=ax.transAxes,  # use axis coordinates, not data coordinates for box placement
                    fontsize=9,
                    verticalalignment='center',
                    horizontalalignment='left',
                    color=color_group_colors_by_label[color_group_label]
                )

        plt.title(f"{io.prettify(name)} Embedding")

        # Determine path
        if full_path is None:
            path = f"{name}_embedding"
        else:
            path = full_path
        self._save_plot(path, dpi=300)

    def plot_icd_mortality_tables(self):
        self._plot_icd_mortality_tables(clusterings=self.clustering.clusterings)

    def _plot_icd_mortality_tables(self, clusterings: List[ClusteringInfo]):
        # Extract ICD attributes
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        # Extract survival
        static_info = self.preprocessor.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(self.split)
        )
        survival = static_info['FUTURE_survival']

        # Handle table creation for each ICD category (diagnoses and procedures)
        for icd_attr_name in icd_attrs:

            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[idx]
                                         for idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            present_icd_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                  for idx in adm_icd_indices]
                                 for adm_icd_indices in present_icd_codes_indices]

            # Create table for each ICD depth and clustering
            for icd_tree_max_depth in range(1, 4 + 1):

                # Get ICD nodes for each admission (at the requested level)
                icd_node_names_at_level = [[IcdInfo.name_for_node(n.path[min(icd_tree_max_depth, len(n.path) - 1)])
                                            for n in adm_nodes]
                                           for adm_nodes in present_icd_nodes]

                for clustering_info in clusterings:
                    # Create table
                    table = self._create_icd_mortality_table(
                        clustering_info=clustering_info,
                        icd_code_names=icd_node_names_at_level,
                        survival=survival
                    )

                    # Write table to disk
                    table_path = self._get_plot_path(
                        os.path.join(
                            self._get_dir_for_clustering(clustering_info),
                            "icd_mortality_tables",
                            f"mortality_icd_d{icd_tree_max_depth}_table_{icd_attr_name}_{clustering_info.random_id}"
                        ),
                        rel_dirs=True
                    )
                    table_path, _ = os.path.splitext(table_path)  # Remove default png file extension
                    table_path = table_path + ".csv"
                    table.to_csv(table_path, quoting=csv.QUOTE_NONNUMERIC)

    @staticmethod
    def _create_icd_mortality_table(clustering_info: ClusteringInfo, icd_code_names: List[List[str]],
                                    survival: List[bool]):
        # Find out all possible cluster labels
        labs = sorted(np.unique(clustering_info.labels))

        # Find out all observed code names
        observed_codes = sorted(np.unique(sum(icd_code_names, [])))

        # Prepare table
        code_str = "ICD Code"
        total_count_str = "Total Count"
        pop_mortality_str = "Mortality (Population), %"
        distribution_surprise_str = "Surprise (Diff. Squared)"
        cluster_mort_strings = {lab: f"Mortality ({io.label_for_cluster_label(lab)}), %" for lab in labs}
        cluster_count_strings = {lab: f"Count ({io.label_for_cluster_label(lab)})" for lab in labs}
        data_rows = []

        # Each code will become a line in the final table
        cols = None
        for code in observed_codes:

            # Initialize accumulator for counting the number of admissions in each of the clusters that have the code
            # and are either survivors or deceased
            code_survival = {lab: [] for lab in labs}

            # Go through all admissions to check if they have the current code
            for adm_codes, adm_cluster_label, adm_survived in zip(icd_code_names, clustering_info.labels, survival):
                if code in adm_codes:
                    code_survival[adm_cluster_label].append(int(adm_survived))

            # Find out mortality for the population
            pop_code_survival = sum(code_survival.values(), [])
            code_total_count = len(pop_code_survival)
            pop_mortality = 1 - (sum(pop_code_survival) / code_total_count)

            # Find out mortality for each of the clusters
            clus_mortality = {}
            clus_code_counts = {}
            for lab, cluster_survival in code_survival.items():
                clus_code_counts[lab] = len(cluster_survival)
                if len(cluster_survival) == 0:
                    continue
                clus_survived_fraction = sum(cluster_survival) / len(cluster_survival)
                clus_mortality[lab] = 1 - clus_survived_fraction

            # Find out if the distribution of mortality over clusters is different from a uniform distribution.
            mortality_distribution = [clus_mortality[lab] if lab in clus_mortality else pop_mortality for lab in labs]
            mortality_distribution_surprise = sum(
                [np.square(mort_clus - pop_mortality) for mort_clus in mortality_distribution]
            )
            # higher value -> less uniform distribution

            # Create line for table
            data_line = OrderedDict()
            data_line[code_str] = code
            data_line[total_count_str] = code_total_count
            data_line[distribution_surprise_str] = mortality_distribution_surprise
            data_line[pop_mortality_str] = f"{100 * pop_mortality:0.1f}"
            for lab in labs:
                if lab in clus_mortality:
                    data_line[cluster_mort_strings[lab]] = f"{100 * clus_mortality[lab]:0.1f}"
                data_line[cluster_count_strings[lab]] = f"{clus_code_counts[lab]}"

            data_rows.append(data_line)

            # Remember columns
            if cols is None or len(data_line.keys()) > len(cols):
                cols = data_line.keys()

        # Create DataFrame out of rows of data
        table = pd.DataFrame(data=data_rows, columns=cols).set_index(code_str)
        table = table.sort_values(total_count_str, ascending=False)  # highest counts first
        return table

    def plot_icd_cumulative_covering_bars(self):
        self._plot_icd_cumulative_covering_bars(clusterings=self.clustering.clusterings)

    def _plot_icd_cumulative_covering_bars(self, clusterings: List[ClusteringInfo]):
        # Extract ICD attributes and data
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        # Handle table creation for each ICD category (diagnoses and procedures)
        for icd_attr_name in icd_attrs:

            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[adm_idx]
                                         for adm_idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            present_icd_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                  for idx in adm_icd_indices]
                                 for adm_icd_indices in present_icd_codes_indices]

            # Get ICD nodes for each admission (at the requested level)
            for icd_tree_max_depth in range(1, 4 + 1):
                icd_node_names_at_level = [[IcdInfo.name_for_node(n.path[min(icd_tree_max_depth, len(n.path) - 1)])
                                            for n in adm_nodes]
                                           for adm_nodes in present_icd_nodes]

                for clustering_info in clusterings:
                    # Write bars plot
                    self._plot_icd_cumulative_covering_bars_inner(
                        clustering_info=clustering_info,
                        icd_code_names=icd_node_names_at_level,
                        icd_attr_name=icd_attr_name,
                        icd_depth=icd_tree_max_depth
                    )

    def _plot_icd_cumulative_covering_bars_inner(self, clustering_info: ClusteringInfo, icd_code_names: List[List[str]],
                                                 icd_attr_name: str, icd_depth: int, num_bars=20):
        # Split codes between clusters
        icd_str_split = split_by_cluster(clustering_info.labels, icd_code_names)
        labels = sorted(np.unique(clustering_info.labels))

        # Plot for each cluster separately
        for lab, icd_codes in zip(labels, icd_str_split):

            # Count codes to find out the most frequent
            counter = Counter()
            for adm_codes in icd_codes:
                counter.update(adm_codes)
            most_freq_codes = [code for (code, _) in counter.most_common(n=num_bars)]
            num_bars = min(len(most_freq_codes), num_bars)

            # Step by step (code by code), find out how many admissions in the cluster exhibit a code in a cumulative
            # fashion
            num_adm = len(icd_codes)
            has_code_mask = [False] * num_adm
            code_cum_counts = []
            code_abs_counts = {}
            for code in most_freq_codes:

                # Go over admissions that are not yet flagged and flag them if they have the code
                code_count = 0
                for idx in range(num_adm):
                    if code in icd_codes[idx]:
                        code_count += 1
                        if not has_code_mask[idx]:
                            has_code_mask[idx] = True

                # Store absolute count for the code
                code_abs_counts[code] = code_count

                # Count flagged admissions for this code (and all previous codes)
                code_cum_counts.append(
                    sum([int(flag) for flag in has_code_mask])  # True -> 1, False -> 0
                )

            # Express cumulative counts in percent
            code_cum_percent = [100 * cnt / num_adm for cnt in code_cum_counts]

            # Stop once 100% is reached - any codes beyond that don't add any new information
            bars_cutoff = None
            for code_idx, code_percent in enumerate(code_cum_percent):
                if code_percent == 100:
                    bars_cutoff = code_idx
                    break
            if bars_cutoff is not None:
                num_bars = bars_cutoff + 1
                code_cum_percent = code_cum_percent[:num_bars]
                most_freq_codes = most_freq_codes[:num_bars]

            # Plot bar plot
            fig, ax = plt.subplots(figsize=(2 * len(code_cum_percent), 6))
            x_pos = np.arange(len(most_freq_codes))

            ax.set_xticks(x_pos)
            wrapped_codes = ["\n".join(textwrap.wrap(c, width=30, break_long_words=True, break_on_hyphens=True))
                             for c in most_freq_codes]
            ax.set_xticklabels(wrapped_codes, rotation=45, ha='right')

            # Turn off y-axis
            ax.get_yaxis().set_visible(False)

            # Turn off plot borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Bars
            bars = ax.bar(
                x_pos,
                code_cum_percent,
                width=1.0,
                color='xkcd:spring green',
                edgecolor='black'
            )
            bar_labels = ax.bar_label(
                bars,
                labels=[f"{p:0.1f}%" for p in code_cum_percent],
                label_type='edge',
                padding=5
            )

            # Also add labels that mention the prevalence of the code
            code_freq_labels = [f"{100 * code_abs_counts[c] / num_adm:0.1f}%" for c in most_freq_codes]
            bar_labels_freq = ax.bar_label(
                bars,
                labels=code_freq_labels,
                label_type='center',
                color="xkcd:dark grey"
            )

            # Give bars a little more vertical headroom
            x_min, x_max, y_min, y_max = ax.axis()

            # Restore previous y limits and x limits
            ax.set_ylim((y_min, 1.07 * y_max))

            icd_type = icd_attr_name.split("_")[-1].lower()
            ax.set_title(f"Cumulative covering of admissions in {io.label_for_cluster_label(lab)} using top {num_bars}"
                         f" {icd_type} codes (depth {icd_depth})")

            # Write plot to disk
            plot_path = os.path.join(
                self._get_dir_for_clustering(clustering_info),
                "icd_covering_bars",
                f"clus_{lab}_icd_{icd_attr_name}_d{icd_depth}_{clustering_info.random_id}"
            )
            self._save_plot(plot_path, create_dirs=True)

    def plot_icd_distribution_bar_plots(self):
        self._plot_icd_distribution_bar_plots(clusterings=self.clustering.clusterings)

    def _plot_icd_distribution_bar_plots(self, clusterings: List[ClusteringInfo]):
        # Extract ICD attributes and data
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        # Handle table creation for each ICD category (diagnoses and procedures)
        for icd_attr_name in icd_attrs:

            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[adm_idx]
                                         for adm_idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            present_icd_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                  for idx in adm_icd_indices]
                                 for adm_icd_indices in present_icd_codes_indices]

            # Get ICD nodes for each admission (at the requested level)
            icd_tree_max_depth = 1  # plot only works for first level
            icd_node_names_at_level = [[IcdInfo.name_for_node(n.path[min(icd_tree_max_depth, len(n.path) - 1)])
                                        for n in adm_nodes]
                                       for adm_nodes in present_icd_nodes]

            for clustering_info in clusterings:
                # Write plot
                self._plot_icd_distribution_bar_plot_inner(
                    clustering_info=clustering_info,
                    icd_code_names=icd_node_names_at_level,
                    icd_attr_name=icd_attr_name
                )

    def _plot_icd_distribution_bar_plot_inner(self, clustering_info: ClusteringInfo, icd_code_names: List[List[str]],
                                              icd_attr_name: str):
        # Count occurrences of ICD codes within clusters - we want to find out what the cluster is made of
        labels = list(np.unique(clustering_info.labels))

        clus_dicts = []
        for lab in labels:
            clus_counts = defaultdict(int)

            for idx in np.where(clustering_info.labels == lab)[0]:
                icd_names_list = icd_code_names[idx]
                for icd_name in icd_names_list:
                    clus_counts[icd_name] += 1

            clus_dicts.append(clus_counts)

        # (Note that the total sum of counts might be different between clusters since each patient is counted for
        # *all* their codes)

        # Draw bars for each observed ICD code - first the often-occurring bars, then the rarer ones
        total_count = defaultdict(int)
        for clus_counts in clus_dicts:
            for icd_name, count in clus_counts.items():
                total_count[icd_name] += count
        icd_draw_order = sorted([(count, icd_name) for (icd_name, count) in total_count.items()], reverse=True)
        icd_draw_order = [icd_name for (count, icd_name) in icd_draw_order]

        # Bring bars for all clusters to the same width (even though total counts might be different)
        count_totals = [sum(counts.values()) for counts in clus_dicts]
        if 0 in count_totals:
            logging.info("Can not draw ICD distribution plot: There are 0-width bars.")
            return

        # Init figure
        fig, ax = plt.subplots(figsize=(18, 2 * len(labels)))
        labels_pretty = [io.label_for_cluster_label(lab) for lab in labels]
        y_pos = np.arange(len(labels_pretty))

        ax.set_yticks(y_pos, labels=labels_pretty)
        ax.invert_yaxis()  # labels read top-to-bottom

        # Turn off x-axis
        ax.get_xaxis().set_visible(False)

        # Turn off plot borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Draw each set of bars
        target_width = 200
        prev_bar_right_side = np.zeros(len(labels))
        for icd_name_idx, icd_name in enumerate(icd_draw_order):
            # Bar length corresponds to count of ICD name within clusters
            bar_lengths = [counts[icd_name] / total * target_width for (counts, total) in zip(clus_dicts, count_totals)]
            bar_lengths = np.array(bar_lengths)

            # Remove zero-length bars
            valid_bar_indices = [idx for (idx, l) in enumerate(bar_lengths) if l > 0]

            # Draw bars for this ICD name
            bar_color = IcdInfo.icd_color_for_code(icd_kind=icd_attr_name, icd_code=icd_name)
            bars = ax.barh(y_pos[valid_bar_indices], bar_lengths[valid_bar_indices], align='center', color=bar_color,
                           alpha=1.0, edgecolor='black', left=prev_bar_right_side[valid_bar_indices])

            # Label bars using the ICD name
            short_icd_name = IcdInfo.icd_categ_level_1(icd_kind=icd_attr_name, icd_code=icd_name, short_name=True)
            short_icd_name = short_icd_name.capitalize()
            label_texts = ax.bar_label(bars, labels=[short_icd_name] * len(valid_bar_indices), label_type='center',
                                       color="white", path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
                                       rotation="vertical")

            # Reduce font size of labels for narrow bars
            for text, bar_len in zip(label_texts, bar_lengths[valid_bar_indices]):
                fontsize = text.get_fontsize()
                if bar_len < 0.02 * target_width:
                    text.set_fontsize(fontsize * 0.3)
                elif bar_len < 0.04 * target_width:
                    text.set_fontsize(fontsize * 0.6)

            # Remember bar lengths for next set of bars, which will be drawn to the right of the current bars
            prev_bar_right_side += bar_lengths

        ax.set_title(f"Distribution of {icd_attr_name.split('_')[-1].capitalize()}")

        # Write plot
        plot_path = os.path.join(
            self._get_dir_for_clustering(clustering_info),
            "icd_dist_bars",
            f"icd_{icd_attr_name}_{clustering_info.random_id}"
        )
        self._save_plot(plot_path, create_dirs=True, dpi=self.dpi * 2)  # double DPI to make reading small text easier

    def plot_icd_distribution_tables(self):
        self._plot_icd_distribution_tables(clusterings=self.clustering.clusterings)

    def _plot_icd_distribution_tables(self, clusterings: List[ClusteringInfo]):
        # Extract ICD attributes and data
        _, static_categorical = self.preprocessor.get_scaled_static_data_array()
        icd_attrs = [attr for attr in self.preprocessor.get_static_attrs_listlike() if "icd" in attr.lower()]

        # Handle table creation for each ICD category (diagnoses and procedures)
        for icd_attr_name in icd_attrs:

            # Find out ICD codes for each of the admissions
            present_icd_codes_indices = [stat[icd_attr_name] for stat in static_categorical]
            present_icd_codes_indices = [present_icd_codes_indices[idx]
                                         for idx in self.trainer.get_split_indices(self.split)]
            possible_icd_codes = self.preprocessor.get_static_data_categ_values()[icd_attr_name]
            present_icd_nodes = [[IcdInfo.icd_tree_search(icd_attr_name, possible_icd_codes[idx])
                                  for idx in adm_icd_indices]
                                 for adm_icd_indices in present_icd_codes_indices]

            # Create table for each ICD depth and clustering
            for icd_tree_max_depth in range(1, 4 + 1):

                # Get ICD nodes for each admission (at the requested level)
                icd_node_names_at_level = [[IcdInfo.name_for_node(n.path[min(icd_tree_max_depth, len(n.path) - 1)])
                                            for n in adm_nodes]
                                           for adm_nodes in present_icd_nodes]

                for clustering_info in clusterings:
                    # Create table
                    table = self._create_icd_distribution_table(
                        clustering_info=clustering_info,
                        icd_code_names=icd_node_names_at_level
                    )

                    # Write table to disk
                    table_path = self._get_plot_path(
                        os.path.join(
                            self._get_dir_for_clustering(clustering_info),
                            "icd_distribution_tables",
                            f"icd_d{icd_tree_max_depth}_table_{icd_attr_name}_{clustering_info.random_id}"
                        ),
                        rel_dirs=True
                    )
                    table_path, _ = os.path.splitext(table_path)  # Remove default png file extension
                    table_path = table_path + ".csv"
                    table.to_csv(table_path, quoting=csv.QUOTE_NONNUMERIC)

    @staticmethod
    def _create_icd_distribution_table(clustering_info: ClusteringInfo, icd_code_names: List[List[str]]):
        # Find out all possible cluster labels
        cluster_sizes_by_label = {l: c for (l, c) in zip(*np.unique(clustering_info.labels, return_counts=True))}
        labs = sorted(cluster_sizes_by_label.keys())
        cluster_sizes = [cluster_sizes_by_label[label] for label in labs]

        # Find out all observed code names
        observed_codes = sorted(np.unique(sum(icd_code_names, [])))

        # Prepare table
        code_str = "ICD Code"
        total_count_str = "Total Count"
        code_surprise_str = "Jensen-Shannon Distance"
        data_rows = []
        cluster_label_names = {lab: io.label_for_cluster_label(lab) for lab in labs}

        # Each code will become a line in the final table
        for code in observed_codes:

            # Initialize accumulator for counting the number of admissions in each of the clusters that have the code
            cluster_dist = {label: 0 for label in labs}

            # Go through all admissions to check if they have the current code
            for adm_codes, adm_cluster_label in zip(icd_code_names, clustering_info.labels):
                if code in adm_codes:
                    cluster_dist[adm_cluster_label] += 1

            # Find out if the distribution of codes onto clusters is different from the size distribution of the
            # clusters.
            # We do this because in a larger cluster, we also expect to find more of the codes. But it might be that
            # we actually find more of the code in a rather small cluster. We will be able to tell with this test.
            code_distribution = [cluster_dist[label] for label in labs]
            jensen_shannon_distance = jensenshannon(code_distribution, cluster_sizes, base=2)
            # 0 -> same distribution; 1 -> totally different distribution

            # Create line for table
            code_total_count = sum(cluster_dist.values())
            data_line = {
                code_str: code,
                total_count_str: code_total_count,
                code_surprise_str: jensen_shannon_distance
            }
            for label, count in cluster_dist.items():
                data_line[cluster_label_names[label]] = f"{100 * (count / code_total_count):0.2f}"
            data_rows.append(data_line)

        # Create DataFrame out of rows of data
        table = pd.DataFrame(data=data_rows).set_index(code_str)
        table = table.sort_values(code_surprise_str, ascending=False)  # largest distance first
        return table

    def plot_all_decision_tree_analyses(self):
        # Load decision tree results
        tree_data_selections = self.evaluator.report['decision_tree_analysis']

        for data_selection_name, data_sel_results in tree_data_selections.items():
            for clustering_info in self.clustering.clusterings:
                # Retrieve trees
                trees = data_sel_results[clustering_info.random_id]  # type: List[TreeAnalysisResult]

                # Plot results (for each tree)
                for tree_analysis in trees:
                    self.plot_decision_tree_analysis(
                        trained_tree=tree_analysis.tree,
                        tree_score=tree_analysis.score,
                        tree_class_counts=tree_analysis.class_counts,
                        nan_fill_value=tree_analysis.nan_fill_value,
                        clustering_info=clustering_info,
                        feature_labels=tree_analysis.feature_names,
                        data_selection_name=data_selection_name
                    )

    def plot_decision_tree_analysis(self, trained_tree: tree.DecisionTreeClassifier, tree_score: float,
                                    tree_class_counts: Dict[int, int], nan_fill_value: np.float,
                                    clustering_info: ClusteringInfo, feature_labels: List[str],
                                    data_selection_name: str, total_cluster_width=5):
        """
        Plots the result of the decision tree analysis of a clustering (i.e. trying to understand the clustering as a
        classification decision using decision trees)

        :param trained_tree:
        :param tree_score:
        :param tree_class_counts:
        :param nan_fill_value:
        :param clustering_info:
        :param feature_labels:
        :param data_selection_name:
        :param total_cluster_width: Total width of colored blocks inside tree nodes used t=to represent cluster
         fractions inside of node
        :return:
        """

        # Export tree to graphviz format
        tree_exported = tree.export_graphviz(
            trained_tree,
            feature_names=feature_labels,
            proportion=True,
            filled=True,
            out_file=None
        )

        # Add a title
        class_counts = np.array(list(tree_class_counts.values()))
        cluster_labels = list(tree_class_counts.keys())
        num_samples = np.sum(class_counts)
        class_fractions = class_counts / num_samples
        random_class_chance = np.max(class_fractions)  # when taking majority class as prediction
        title = f"Decision Tree '{io.prettify(data_selection_name)}' (Score {100 * tree_score:0.1f}%," \
                f" chance would be {100 * random_class_chance:0.1f}% for classes {cluster_labels};" \
                f" NaN = {nan_fill_value};  n = {num_samples}, left=True, right=False)"
        title_str = "\n" + \
                    "labelloc=\"t\";" + "\n" + \
                    f"label=\"{title}\";" + "\n" + \
                    "}"
        tree_exported = tree_exported[:-1] + title_str

        # Define colormap for clustering
        cluster_labels_uniq = list(np.unique(clustering_info.labels))
        colormap = self._colormap(num_colors=len(cluster_labels_uniq), index=self.clustering_cm_index)

        # Modify the nodes in the graph to better represent the clusters present in the node
        t_lines = tree_exported.split("\n")
        inner_nodes_for_cluster = {}
        for line_idx, line in enumerate(t_lines):
            # Skip over non-node lines
            if "[label=" not in line:
                continue

            # Extract info from line
            node_name = line.split(" ")[0]
            node_label = line[line.find("[label=") + len("[label=") + 1:line.find(", f") - 1]

            # Remove value part from node label
            cluster_fractions = eval(node_label[node_label.find("value = ") + len("value = "):].replace('\\n', ", "))
            node_label = node_label[:node_label.find("value = ") - 2]

            # Generate subgraph entry
            subgraph_str = f"subgraph cluster{node_name}\n" \
                           "{\n" \
                           f'label="{node_label}"\n'

            # Add nodes representing cluster fractions to the subgraph
            for cluster_label, cluster_fraction in reversed(list(zip(cluster_labels, cluster_fractions))):
                # Determine width of the colored block representing a cluster - if it's zero, we do not include the
                # block
                cluster_width = cluster_fraction * total_cluster_width
                if cluster_width == 0:
                    continue

                # Add colored block for this cluster
                cluster_color = colormap[cluster_labels_uniq.index(cluster_label)]
                cluster_color_hex = matplotlib.colors.to_hex(cluster_color)

                cluster_node_label = f"{cluster_label} ({100 * cluster_fraction:0.0f}%)"

                # Adjust text color based on background color
                color_inverted = np.ones(shape=cluster_color.shape) * \
                                 (1 - np.round(cluster_color[:-1].sum() / len(cluster_color[:-1])))
                color_inverted_hex = matplotlib.colors.to_hex(color_inverted)

                inner_node_name = f'node{node_name}_{cluster_label}'
                subgraph_str += inner_node_name + f' [color="{cluster_color_hex}", fontcolor="{color_inverted_hex}",' \
                                                  f' width={cluster_width},' \
                                                  f' label="{cluster_node_label}"];\n'

                # Remember one of the inner nodes of this cluster (i.e. of the tree node)
                if node_name not in inner_nodes_for_cluster:
                    inner_nodes_for_cluster[node_name] = inner_node_name

            # Conclude subgraph entry
            subgraph_str += "}\n\n"

            t_lines[line_idx] = subgraph_str

        # Rework connections: They should now go between subgraphs instead of nodes
        for line_idx, line in enumerate(t_lines):
            # Skip over lines that are not connections
            if " -> " not in line:
                continue

            # Extract info about node connection
            line_splits = line.split(" ")
            node_begin = line_splits[0]
            node_end = line_splits[2]

            # Generate new connection
            inner_node_begin = inner_nodes_for_cluster[node_begin]
            inner_node_end = inner_nodes_for_cluster[node_end]
            cluster_conn = f"{inner_node_begin} -> {inner_node_end} [ltail=cluster{node_begin}," \
                           f" lhead=cluster{node_end}];"
            t_lines[line_idx] = cluster_conn

        # Mark the graph as "compound", which means that connections between nodes in different subgraphs are displayed
        # as links between the subgraphs
        t_lines.insert(1, "graph [compound=true, nodesep=0, ranksep=2.1];")
        # Note: This also defines the horizontal distance between nodes (nodesep) and the vertical distance between
        # ranks, i.e. levels of the tree (ranksep)

        # Read the source text with graphviz
        tree_exported = "\n".join(t_lines)
        graph = graphviz.Source(tree_exported)

        # Save plot
        plot_path = self._get_plot_path(
            os.path.join(
                self._get_dir_for_clustering(clustering_info),
                "decision_tree",
                io.sanitize(data_selection_name),
                f"tree_{io.sanitize(data_selection_name)}_h{trained_tree.get_depth()}"
                f"_score{100 * tree_score:0.2f}"
                f"_minImpDecPow2{int(np.log2(trained_tree.min_impurity_decrease))}"
            ),
            rel_dirs=True
        )
        plot_path, _ = os.path.splitext(plot_path)
        logging.info(f"Plotting analysis decision tree to {plot_path} ... ")
        graph.render(plot_path)  # pdf
        graph.render(plot_path, format='png')

        # Remove graph file which gets automatically created when rendering
        os.remove(plot_path)

    def plot_recon_vs_gt_scatter(self):
        """
        Plots scatterplot of reconstructed value vs. true value on a selection of validation admissions
        :return:
        """
        # Collect data: Iterate over validation admissions
        logging.info("Plotting scatter plot of reconstructed values vs. ground-truth values...")
        vals_by_col_gt, vals_by_col_rec, times_by_col = self._aggregate_gt_and_rec(
            admission_indices=self.trainer.get_split_indices(self.split)
        )

        # Plot for each of the attributes
        all_attributes = list(vals_by_col_gt.keys())
        for plot_order_idx, col_name in enumerate(all_attributes):

            attr_gt = vals_by_col_gt[col_name]
            attr_recon = vals_by_col_rec[col_name]

            # Find out attribute name
            item_label = self.preprocessor.label_for_any_item_id(col_name)

            # Plot the points
            plt.scatter(x=attr_gt, y=attr_recon)

            # Plot a line standing for the optimal mapping of ground-truth to reconstruction
            x_range = [np.min(attr_gt), np.max(attr_gt)]
            plt.plot(x_range, x_range, color='green')

            # Plot setup
            plt.xlabel("ground-truth")
            plt.ylabel("reconstruction")
            errors = self.evaluator.get_reconstruction_loss()['rec_error_scores']
            if item_label in errors:
                median_error = np.median(errors[item_label])
                median_error_str = f" (median error: {median_error:0.4f})"
            else:
                median_error_str = ""
            plt.title(f"GT vs Rec. of attr. {item_label}{median_error_str}")
            logging.info(f"Wrote plot for {item_label} ({plot_order_idx + 1} of {len(all_attributes)})")
            self._save_plot(os.path.join("scatter", f"scatter_gt_vs_recon_{item_label}"),
                            create_dirs=True)

    def _aggregate_gt_and_rec(self, admission_indices):
        # Get original values and perform reconstruction for the admissions
        vals_by_col_gt = defaultdict(list)  # key: column name, value: list of ground truth values
        vals_by_col_rec = defaultdict(list)  # key: column name, value: list of reconstructed values
        times_by_col = defaultdict(list)  # key: column name, value: list of times
        for admission_idx in admission_indices:
            # Reconstruct
            chart_rec = self.trainer.reconstruct_time_series(adm_idx=admission_idx)
            chart_gt = self.preprocessor.get_deimputed_dyn_chart(adm_idx=admission_idx)

            # Extract times and de-normalize them
            times = self.preprocessor.reverse_scaling(
                chart_gt[self.preprocessor.time_column].values, self.preprocessor.time_column
            )

            # Gather data on ground truth and reconstruction for each column
            for col_idx, col_name in enumerate(self.preprocessor.dyn_data_columns):
                # Note that we are not skipping meta columns like we do for most plotting tasks

                # Continue if requested column in not present in reconstruction (this happens for baseline
                # reconstruction)
                if col_idx >= chart_rec.shape[-1]:
                    continue

                # Extract ground truth and reconstruction for this column
                col_gt = chart_gt[col_name].dropna()  # drop NaN entries
                col_rec = chart_rec[:, col_idx]

                if len(col_gt) == 0:
                    continue

                # Add reconstruction and ground truth values
                vals_by_col_gt[col_name] += list(col_gt.values)
                vals_by_col_rec[col_name] += list(col_rec[col_gt.index])
                times_by_col[col_name] += list(times[col_gt.index])

        # De-normalize ground truth and reconstructions
        for col_name in vals_by_col_gt.keys():
            vals_by_col_gt[col_name] = self.preprocessor.reverse_scaling(vals_by_col_gt[col_name],
                                                                         column_name=col_name)
            vals_by_col_rec[col_name] = self.preprocessor.reverse_scaling(vals_by_col_rec[col_name],
                                                                          column_name=col_name)

        return vals_by_col_gt, vals_by_col_rec, times_by_col

    def plot_time_series_comparisons(self, max_admissions=1000):
        """
        Plots for each dynamic data attribute:
            - violin plots of reconstruction and ground truth at different time points
        Purpose is the comparison of the temporal behavior between reconstruction and ground truth.
        :return:
        """

        # Sample a manageable number of admission indices from the split
        indices = self.trainer.get_split_indices(self.split)
        if len(indices) > max_admissions:
            indices = np.random.choice(indices, size=max_admissions, replace=False)
        vals_by_col_gt, vals_by_col_rec, times_by_col = self._aggregate_gt_and_rec(
            admission_indices=indices
        )

        # Plot for each of the attributes
        for col_name in vals_by_col_gt.keys():
            self._plot_time_series_comparison_for_attr(
                col_name=col_name,
                gts=vals_by_col_gt[col_name],
                recs=vals_by_col_rec[col_name],
                times=times_by_col[col_name]
            )

    def _plot_time_series_comparison_for_attr(self, col_name, gts, recs, times, temporal_base=2,
                                              min_points_total=100, min_points_per_time=10):
        # Abort if too few points in time available
        if len(gts) < min_points_total:
            return

        # Convert to larger temporal units
        max_time = max(times)
        unit = "minutes"
        factor = 1
        if max_time > 300:
            unit = "hours"
            factor *= 60

            if max_time / factor > 48:
                unit = "days"
                factor *= 24
        times = [t / factor for t in times]

        # Determine extent of logarithmic temporal range
        max_time = max(times)
        max_exponent = np.ceil(np.log(max_time) / np.log(temporal_base))
        exp_range = np.power(2, np.arange(max_exponent + 1))
        # (time points are exponentially growing)

        # Ensure that enough data is available for each of the time points
        exp_range = [None, 0] + list(exp_range)
        exp_range_filtered = []
        for step_idx, (lower, upper) in enumerate(zip(exp_range, exp_range[1:])):
            time_points = [t for t in times if t <= upper]
            if lower is not None:
                time_points = [t for t in time_points if lower < t]
            num_data_points = len(time_points)
            if num_data_points >= min_points_per_time or step_idx == (len(exp_range) - 2):
                exp_range_filtered.append((lower, upper))

        # Replace first lower bound by None
        _, upper_first = exp_range_filtered[0]
        exp_range_filtered[0] = (None, upper_first)

        # Construct combination of gt values, reconstructed values and times
        joint = list(zip(gts, recs, times))

        # For each time point, compare ground truth and reconstruction
        violin_plot_data = []
        for lower, upper in exp_range_filtered:
            # Collect ground truth and reconstructed values that fit the temporal bounds from all admissions
            vals = [(gt, reco, time) for (gt, reco, time) in joint if time <= upper]
            if lower is not None:
                vals = [(gt, reco, time) for (gt, reco, time) in vals if lower < time]

            # Get gt values and reconstructed values
            gt_vals, reco_vals, _ = zip(*vals)

            # Set labels for each point in time
            if upper == 0:
                label_time = "$\\leq 0$"
            else:
                exponent = np.log(upper) / np.log(temporal_base)
                exponent = int(exponent)
                label_time = f"$\\leq {temporal_base}^{{{exponent}}}$"
            label_time += f"\n({len(gt_vals)} points)"

            # Save values along with labels (this will later be used to draw violin plots)
            for kind, k_vals in [("GT", gt_vals), ("REC", reco_vals)]:
                for v in k_vals:
                    violin_plot_data.append([
                        v,
                        label_time,
                        kind
                    ])

        # Build DataFrame for plotting with Seaborn
        x_axis_label = f"Times ({unit})"
        item_label = self.preprocessor.label_for_any_item_id(col_name)
        vals_df = pd.DataFrame(violin_plot_data, columns=[item_label, x_axis_label, "Kind"])

        # Plotting only makes sense when there are points
        if len(vals_df) == 0:
            logging.info("Not plotting reconstruction comparison violin plot since there are no points.")
            return

        # Create violin plot
        fig_h = 5
        fig_w = 2 + 2.5 * len(exp_range_filtered)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        try:
            violin = sns.violinplot(
                x=x_axis_label,
                y=item_label,
                hue="Kind",
                data=vals_df,
                ax=ax,
                cut=0,  # Don't let density estimate extend past extreme values
                scale='width'
            )
        except ValueError:
            logging.error(f"Could not plot violin plot! Cluster DataFrame: {vals_df}")

        # Title
        ax.set_title(f"{item_label} (ID {col_name})\n"
                     f"({len(times)} points)")

        # Write plot
        self._save_plot(
            os.path.join(
                "reconstructions",
                "variability",
                f"admission_variability_{len(times)}_{io.sanitize(item_label)}"
            ),
            create_dirs=True,
            dpi=self.dpi,
            seaborn_ax=ax
        )

    def plot_time_series_reconstruction(self, admission_idcs: List[int] = None,
                                        min_time_steps: int = 3,
                                        color_by_cluster: Optional[ClusteringInfo] = None):
        """
        Plots a random dynamic attribute of a (random or specified) patient vs. the reconstruction for that attribute.

        :param admission_idcs:
        :param min_time_steps:
        :param color_by_cluster: If supplied, enables clustering coloring mode: This means that reconstruction will not
         be plotted and colors of ground truth will be given by the color of the cluster
        :return:
        """

        # Randomly choose an admission from the validation data
        if admission_idcs is None:
            admission_idcs = [random.choice(self.trainer.get_split_indices(self.split))]  # plot single random admission

        # Get de-imputed charts
        charts = [self.preprocessor.get_deimputed_dyn_chart(adm_idx) for adm_idx in admission_idcs]

        # Reconstruct: We perform the reconstruction using the natural order inherent in the data
        # (Performing this step early and with non-attr-grouped data is important for "natural" reconstruction)
        if color_by_cluster is None:
            reconstructions = [self.trainer.reconstruct_time_series(adm_idx=adm_idx) for adm_idx in admission_idcs]
        else:
            # If coloring by cluster, reconstructions are not needed. Use dummy reconstructions in this case.
            reconstructions = [None for c in charts]

        # Plot each of the charted columns
        for col_idx, col_name in enumerate(self.preprocessor.dyn_data_columns):

            # Don't plot meta columns like time
            if col_name in self.preprocessor.meta_columns:
                continue

            # Filter out admissions that have too few or no data for this attributes
            col_adm_indices = []  # admissions that have enough data for this attribute
            col_gt = []
            col_rec = []
            col_times = []
            for adm_idx, dyn_chart, reco in zip(admission_idcs, charts, reconstructions):
                gt = dyn_chart[col_name]
                valid_indices = gt.loc[gt.notnull()].index

                if len(valid_indices) < min_time_steps:
                    continue

                # Extract ground truth, reconstruction and times for the valid data points
                col_adm_indices.append(adm_idx)
                col_gt.append(gt[valid_indices].to_numpy())
                if reco is not None:
                    col_rec.append(reco[valid_indices, col_idx])
                col_times.append(dyn_chart[self.preprocessor.time_column][valid_indices].to_numpy())

            # Don't plot this attribute if no admission has it
            if len(col_adm_indices) < 1:
                continue

            # Reverse scaling for times, ground truth and reconstruction
            col_gt = [self.preprocessor.reverse_scaling(gt, column_name=col_name) for gt in col_gt]
            col_rec = [self.preprocessor.reverse_scaling(reco, column_name=col_name) for reco in col_rec]
            col_times = [self.preprocessor.reverse_scaling(tim, column_name=self.preprocessor.time_column)
                         for tim in col_times]

            # Plot
            if color_by_cluster is None:
                self._plot_time_series_reconstruction_for_attr(
                    col_name=col_name,
                    adm_indices=col_adm_indices,
                    ground_truths=col_gt,
                    reconstructions=col_rec,
                    times=col_times
                )
            else:
                self._plot_time_series_agg_by_clusters(
                    col_name=col_name,
                    adm_indices=col_adm_indices,
                    ground_truths=col_gt,
                    times=col_times,
                    color_by_cluster=color_by_cluster
                )

    def _plot_time_series_agg_by_clusters(self, col_name, adm_indices, ground_truths, times,
                                          color_by_cluster: ClusteringInfo):
        # Convert time to days and let times start at 0
        times = [[t / 60 / 24 for t in tr] for tr in times]
        times = [list(np.array(t_range) - min(t_range)) for t_range in times]

        # Find out if plotting single admission or more than one
        num_adms = len(adm_indices)

        # Find out attribute name
        item_label = self.preprocessor.label_for_any_item_id(col_name)

        # Plotting with coloring by cluster only makes sense if more than a single admission is plotted.
        # Abort the plotting if there is only one admission
        if num_adms < 2:
            logging.info(f"Not plotting multi time series colored by clusters: Only one admission"
                         f" available for attribute {item_label}.")
            return

        # Find the admission indices within the split's admission index list (we need it for finding out the
        # cluster label, which is stored w.r.t. the split indices)
        split_indices = self.trainer.get_split_indices(self.split)
        adm_indices_split_internal = self.clustering.index_multi(
            containing_arr=split_indices,
            values_to_be_indexed=adm_indices
        )

        # Make sure that all the admission indices were found - if they were not, this means that the clustering
        # originates from a different split than this plotting run
        if len(adm_indices_split_internal) != len(adm_indices):
            logging.info(f"Not plotting multi time series colored by clusters: Clustering likely comes from "
                         f"different split than current split of plotting ({self.split}).")
            return  # Can't plot if the split is different

        # Find out cluster labels of each of the admissions
        adm_cluster_labels = [color_by_cluster.labels[idx_internal] for idx_internal in adm_indices_split_internal]

        # Group the time series by cluster label
        vals_and_times_by_cluster = {}
        for lab, gt, ts in zip(adm_cluster_labels, ground_truths, times):
            if lab not in vals_and_times_by_cluster:
                vals_and_times_by_cluster[lab] = []
            vals_and_times_by_cluster[lab].append((gt, times))

        # Get colors for each of the corresponding clusters
        cluster_labels_uniq = list(np.unique(color_by_cluster.labels))
        colormap = self._colormap(num_colors=len(cluster_labels_uniq), index=self.clustering_cm_index)
        adm_colors_gt = [colormap[cluster_labels_uniq.index(cluster_label)] for cluster_label in adm_cluster_labels]

        # Legend label is given by cluster label and the number of admissions used for the plot
        adm_legend_labels = [f"{io.label_for_cluster_label(lab)} ({len(vals_and_times_by_cluster[lab])} adm. shown)"
                             for lab in adm_cluster_labels]

        # Opacity/Alpha: Make it easier to see trends and commonalities
        adm_alphas = [max(min(3 / len(vals_and_times_by_cluster[label]), 1.0), 0.35) for label in adm_cluster_labels]
        marker = None

        # Plot each cluster on its own axis
        clusters_with_data = list(vals_and_times_by_cluster.keys())
        if len(clusters_with_data) == 0:
            return
        del vals_and_times_by_cluster

        # Generate random identifier for this attribute - it could theoretically happen that two attributes have the
        # same name in MIMIC
        attr_rnd_id = self._random_identifier()

        # Construct the plot
        num_subplots = len(clusters_with_data)
        plot_grid_size = min(int(np.ceil(np.sqrt(num_subplots))), 4)  # max is a 4 by 4 grid of subplots
        max_subplots_per_page = plot_grid_size ** 2
        num_plot_pages = int(np.ceil(num_subplots / max_subplots_per_page))

        # Plot each of the pages as its own file
        for page_idx, page_clusters in enumerate(io.chunk_list(clusters_with_data, max_subplots_per_page)):

            # Set up figure for this page
            fig = plt.figure(figsize=(20, 20))  # (width, height)
            gs = gridspec.GridSpec(plot_grid_size, plot_grid_size)
            axes_by_cluster = {cluster_label: plt.subplot(gs[cluster_label_idx])
                               for (cluster_label_idx, cluster_label) in enumerate(page_clusters)}

            # Plot
            gt_labeling_done = []
            for adm_plot_idx, (t_range, gt, cluster_label) in enumerate(
                    zip(times, ground_truths, adm_cluster_labels)
            ):
                # Only plot admission if it belongs to the current page
                if cluster_label not in axes_by_cluster:
                    continue

                # Label only one of the plotted lines for the ground truth (for each of the possibly different labels)
                legend_label = adm_legend_labels[adm_plot_idx]
                kwargs = {}
                if legend_label not in gt_labeling_done:
                    kwargs['label'] = legend_label
                    gt_labeling_done.append(legend_label)

                # Get color
                color = adm_colors_gt[adm_plot_idx]

                # Get alpha
                alpha = adm_alphas[adm_plot_idx]  # only different and less than 1. if plotting more than a single
                # admission

                # Get axis
                plot_axis = axes_by_cluster[cluster_label]

                # Plot line
                plot_axis.plot(t_range, gt, c=color, linewidth=2, marker=marker, alpha=alpha, **kwargs)

                # Label the axis and give it a legend
                plot_axis.set_ylabel(f"Value of {item_label}")
                plot_axis.set_xlabel("Time (Days)")
                plot_axis.legend()

            # Adjust axis limits such that the scaling for each of the plot axes is the same
            ylims_lower, ylims_upper = list(zip(*[ax.get_ylim() for ax in axes_by_cluster.values()]))
            y_lim_min = min(ylims_lower)
            y_lim_max = max(ylims_upper)
            for ax in axes_by_cluster.values():
                ax.set_ylim(ymin=y_lim_min, ymax=y_lim_max)

            xlims_lower, xlims_upper = list(zip(*[ax.get_xlim() for ax in axes_by_cluster.values()]))
            x_lim_min = min(xlims_lower)
            x_lim_max = max(xlims_upper)
            for ax in axes_by_cluster.values():
                ax.set_xlim(xmin=x_lim_min, xmax=x_lim_max)

            # Give the figure a title
            title = f"{item_label} (ID {col_name})\n"
            title += f"({num_adms} admissions)"
            if num_plot_pages > 1:
                title += f" {page_idx + 1}/{num_plot_pages}"
            fig.suptitle(title)

            # Save in directory of the clustering this plot is colored by
            plot_dir = os.path.join(self._get_dir_for_clustering(color_by_cluster), "time_series_by_cluster")
            plot_name = f"time_series_gt_attr_{io.sanitize(item_label)}_rnd_id_{attr_rnd_id}"

            # Mention page in plot filename
            if num_plot_pages > 1:
                plot_name += f"_page_{page_idx + 1:02d}_of_{num_plot_pages:02d}"

            self._save_plot(os.path.join(plot_dir, plot_name),
                            create_dirs=True)

    def _plot_time_series_reconstruction_for_attr(self, col_name, adm_indices, ground_truths, reconstructions, times,
                                                  analysis_mode=False, color_gt='xkcd:aquamarine',
                                                  color_rec='xkcd:rust'):

        # Find out if plotting single admission or more than one
        num_adms = len(adm_indices)
        single_admission_mode = num_adms == 1

        # If plotting multiple admissions, have all times start with 0
        if not single_admission_mode:
            times = [np.array(t_range) - min(t_range) for t_range in times]

        # Set up plot
        fig_width, fig_height = 16, 7
        if single_admission_mode and analysis_mode:
            fig_height = int(fig_height * 1.3)
        fig_size = (fig_width, fig_height)
        logging.info(f"Plotting time series reconstruction ({len(adm_indices)} adms. ({adm_indices}),"
                     f" column {col_name}, figure: {fig_size}) ...")
        fig = plt.figure(figsize=fig_size)

        if single_admission_mode and analysis_mode:
            axes_needed = 2
            grid_kwargs = {"height_ratios": [3, 1]}
        else:
            axes_needed = 1
            grid_kwargs = {}
        gs = gridspec.GridSpec(axes_needed, 1, **grid_kwargs)

        # Axis for reconstruction and ground truth
        ax_reco = plt.subplot(gs[0])

        # Plot residuals only if we have just a single admission
        if single_admission_mode and analysis_mode:
            ax_resi = plt.subplot(gs[1])

        # X-axis: Convert time to reasonable unit
        minutes_per_hour = 60
        minutes_per_day = 24 * minutes_per_hour
        minutes_per_week = 7 * minutes_per_day

        num_minutes = max([tim[-1] for tim in times])  # in minutes
        num_hours = num_minutes / minutes_per_hour
        num_days = num_minutes / minutes_per_day
        num_weeks = num_minutes / minutes_per_week

        if num_weeks >= 2.5:
            time_unit = "Weeks"
            time_factor = minutes_per_week
        elif num_days >= 2.5:
            time_unit = "Days"
            time_factor = minutes_per_day
        elif num_hours >= 2.5:
            time_unit = "Hours"
            time_factor = minutes_per_hour
        else:
            time_unit = "Minutes"
            time_factor = 1
        times = [[t / time_factor for t in tims] for tims in times]

        # Enable the grid
        ax_reco.grid()

        time_axis_labeling = f"Time ({time_unit})"
        if single_admission_mode and analysis_mode:
            ax_resi.set_xlabel(time_axis_labeling)
        else:
            ax_reco.set_xlabel(time_axis_labeling)

        # Find out attribute name
        item_label = self.preprocessor.label_for_any_item_id(col_name)

        # Color all admission's ground truth the same
        adm_colors_gt = len(adm_indices) * [color_gt]

        # Legend label is the same for all ground-truth curves
        adm_legend_labels = len(adm_indices) * ["Ground Truth"]

        # Opacity/Alpha: Make it easier to see trends and commonalities when plotting more than a single admission
        if not single_admission_mode:
            adm_alphas = num_adms * [max(min(3 / num_adms, 1.0), 0.50)]
            marker = None
        else:
            adm_alphas = [1.0]  # only a single admission
            marker = 'o'

        # Plot ground-truth
        gt_labeling_done = []
        for adm_plot_idx, (t_range, gt_values) in enumerate(zip(times, ground_truths)):
            # Label only one of the plotted lines for the ground truth (for each of the possibly different labels)
            legend_label = adm_legend_labels[adm_plot_idx]
            kwargs = {}
            if legend_label not in gt_labeling_done:
                kwargs['label'] = legend_label
                gt_labeling_done.append(legend_label)

            # Get color
            color = adm_colors_gt[adm_plot_idx]  # only different between admissions if coloring by cluster

            # Get alpha
            alpha = adm_alphas[adm_plot_idx]  # only different and less than 1. if plotting more than a single admission

            ax_reco.plot(t_range, gt_values, c=color, linewidth=2, marker=marker, alpha=alpha, **kwargs)

            # Plot mean of the GT only if we are plotting a single admission
            if single_admission_mode and analysis_mode:
                ax_reco.axhline(y=np.mean(gt_values), color=color_gt, alpha=0.5)  # mean of this time series

        # Plot global mean of GT
        if analysis_mode:
            ax_reco.axhline(
                y=self.preprocessor.dyn_col_medians_train[col_name],
                color=color_gt,
                linestyle='dashed',
                alpha=0.5,
                label='gt: train median'
            )

        # Plot reconstructed values
        reco_labeling_done = False
        for t_range, reco in zip(times, reconstructions):

            # Label only one of the plotted lines for the reconstruction
            kwargs = {}
            if not reco_labeling_done:
                kwargs['label'] = 'Reconstruction'
                reco_labeling_done = True

            ax_reco.plot(t_range, reco, c=color_rec, linewidth=2, marker='o', **kwargs)
            if analysis_mode:
                ax_reco.axhline(y=np.mean(reco), color=color_rec, alpha=0.5)

        # Label the axes
        ax_reco.set_ylabel(f"Value of {item_label}")

        # Give the top axis a title
        title = f"{item_label}"
        if analysis_mode:
            title += " (ID {col_name})\n"
        if single_admission_mode:
            if analysis_mode:
                title += f"Admission Index {adm_indices[0]}"
        else:
            title += f"({num_adms} admissions)"
        ax_reco.set_title(title)

        # Plot residuals
        if single_admission_mode and analysis_mode:
            res_pos = []
            res_neg = []
            residuals = np.array(reconstructions[0]) - np.array(ground_truths[0])
            for res_time, res_val in zip(times[0], residuals):
                if res_val >= 0:
                    res_pos.append((res_time, res_val))
                else:
                    res_neg.append((res_time, res_val))
            ax_resi.bar([r[0] for r in res_pos], [r[1] for r in res_pos],
                        color='xkcd:sky blue', edgecolor='xkcd:sky blue', linewidth=2)
            ax_resi.bar([r[0] for r in res_neg], [r[1] for r in res_neg],
                        color='xkcd:dark pink', edgecolor='xkcd:dark pink', linewidth=2)
            eval_errors = self.evaluator.get_reconstruction_loss()['rec_error_scores']
            if item_label in eval_errors:
                median_error = np.median(eval_errors[item_label])
                median_error_str = f" (median error of {item_label}: {median_error:0.4f})"
            else:
                median_error_str = ""
            ax_resi.set_title(f"Residuals" + median_error_str)

        # Legend
        self._sort_legend_by_label(ax_reco)

        # Save in directory for general time series vs. reconstruction plots
        plot_dir = "reconstructions"
        plot_name = f"time_series_gt_vs_recon_attr_{io.sanitize(item_label)}"
        if single_admission_mode:
            plot_name += f"_adm_{adm_indices[0]}_{len(ground_truths[0])}"
        else:
            plot_dir += "_multi"
            plot_name += f"_multi_{num_adms}_admissions"

        # Prepend variance of reconstruction to plot name - it helps when looking for interesting-looking
        # reconstructions
        if single_admission_mode:
            # Calculate reconstruction MSE in scaled space
            rec = self.preprocessor.perform_scaling(reconstructions[0].astype(np.float64),
                                                    column_name=col_name)
            gt = self.preprocessor.perform_scaling(ground_truths[0].astype(np.float64),
                                                   column_name=col_name)
            mse = np.mean(np.power(rec - gt, 2))
            plot_name = f"mse{mse:013.06f}_{plot_name}"

        # Put into folders depending on number of time steps
        if single_admission_mode:
            steps_log = np.log2(len(times[0]))
            floored = int(np.floor(steps_log))
            ceiled = max(int(np.ceil(steps_log)), floored + 1)
            step_folder = f"steps_{2 ** floored}_{2 ** ceiled}"
            plot_dir = os.path.join(plot_dir, step_folder)

        self._save_plot(os.path.join(plot_dir, plot_name),
                        create_dirs=True)

    @staticmethod
    def _sort_legend_by_label(ax):
        leg_handles, leg_labels = ax.get_legend_handles_labels()
        leg_handles, leg_labels = zip(*sorted(zip(leg_handles, leg_labels), key=lambda t: t[1]))
        ax.legend(leg_handles, leg_labels)

    @staticmethod
    def _random_identifier(length=4):
        characters = string.digits + string.ascii_lowercase
        return "".join(random.choice(characters) for _ in range(length))

    def plot_error_by_data_amount(self):
        """
        Plot reconstruction quality by data density
        :return:
        """

        # Abort if in baseline mode
        if self.trainer.baseline_mode:
            return

        logging.info(f"Plotting errors by dynamic data amount ...")

        # Get errors
        errors_by_attr = self.evaluator.get_reconstruction_loss()['rec_error_scores']

        # Get median of errors for each dynamic data attribute
        amounts = []
        errors = []
        for dyn_label, attr_errors in errors_by_attr.items():
            amounts.append(len(attr_errors))
            errors.append(np.median(attr_errors))

        # Plot
        plt.scatter(x=amounts, y=errors)

        # Plot line at error = 0
        plt.axhline(y=0, color="black")

        plt.xlabel("Amount Per Dynamic Data Attribute")
        plt.ylabel("Error")

        # Use logarithmic scales since the magnitude of values can differ by a lot
        plt.xscale('log')

        plt.title(f"Dynamic Data Amount vs. Mean Absolute Scaled Error")
        self._save_plot("errors_by_data_amount")

    def plot_function_output_scatter(self, func_evals, function_name='no_name', mean_color='red', color_by_value=False):
        """
        Plot the output of an approximated function as a scatter plot

        :param func_evals: list of (x, f(x)) pairs of function evaluations. f(x) may be None if f failed.
        :param function_name: Function name that appears in plot title
        :color_by_value:
        :param mean_color: color in which mean values are shown
        :return:
        """

        logging.info(f"Plotting {len(func_evals)} function evaluations of function '{function_name}'...")

        # Set up figure
        fig, ax = plt.subplots(figsize=(20, 9))

        # Remove entries where f failed - these are plotted separately
        x_vals = []
        fx_vals = []
        x_vals_f_failed = []
        for x, fx in func_evals:
            if fx is not None:
                x_vals.append(x)
                fx_vals.append(fx)
            else:
                x_vals_f_failed.append(x)

        # Color the point based on f(x)
        colors = []

        if len(fx_vals) > 0:
            f_min = min(fx_vals)
            f_max = max(fx_vals)
            f_std = np.std(fx_vals)
            f_range = f_max - f_min
            for fx in fx_vals:
                rel_fx = (fx - f_min) / f_range
                colors.append(rel_fx)
        else:
            f_min = 0
            f_std = 1

        # Plot failed evals
        if len(x_vals_f_failed) > 0:
            ax.scatter(x=x_vals_f_failed, y=[f_min - 0.5 * f_std] * len(x_vals_f_failed), c='grey',
                       s=250, alpha=min(1., 10 / len(x_vals_f_failed)), linewidth=0)

        # Plot non-failed evals
        if not color_by_value:
            colors = [0.5] * len(colors)
        ax.scatter(x=x_vals, y=fx_vals, c=colors)

        if len(x_vals) > 0 and type(x_vals[0]) == str or len(np.unique(x_vals)) <= 3:
            # Plot mean (in f(x)) of every possible value of x
            x_uniq = np.unique(x_vals)
            for x_v in x_uniq:
                mean_fx = np.mean([fx for (x, fx) in zip(x_vals, fx_vals) if x == x_v])
                ax.scatter(x=[x_v], y=[mean_fx], c=mean_color)
        else:
            # Plot smoothed approximation of the function
            poly_fit = np.polynomial.polynomial.Polynomial.fit(
                x=x_vals,
                y=fx_vals,
                deg=4
            )
            x_range = np.linspace(
                start=min(x_vals),
                stop=max(x_vals),
                num=100
            )

            # Remember y limits before plotting, so we can restore them after plotting the curve
            x_min, x_max, y_min, y_max = ax.axis()

            # Plot curve
            ax.plot(x_range, poly_fit(x_range), c=mean_color)

            # Restore previous y limits and x limits
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))

        ax.set_xlabel("x")
        ax.set_ylabel(f"{function_name}(x) (noisy function)")
        ax.set_title(f"Function evaluations for function {function_name}")
        self._save_plot(f"func_eval_scatter_{function_name}")

    def plot_heat_map(self, x_name, x_inputs, y_name, y_inputs, func_vals, max_labeled_cells=6, max_bins=4,
                      heat_center=None):
        """
        Plots 2D heat map of a 2D function

        :param x_name: name of x-axis
        :param x_inputs: list of inputs in x
        :param y_name: name of y-axis
        :param y_inputs: list of inputs in y
        :param func_vals: list of function outputs
        :param max_labeled_cells: maximum number of cells that may be individually labeled until axis gets binned
        :param max_bins: maximum number of bins if axis is binned
        :param heat_center: Center of the data (e.g. when plotting residuals, center should be 0)
        :return:
        """

        # Filter out evaluations of the functions that resulted in None
        none_indices = [idx for (idx, val) in enumerate(func_vals) if val is None]
        x_inputs = [x for (idx, x) in enumerate(x_inputs) if idx not in none_indices]
        y_inputs = [y for (idx, y) in enumerate(y_inputs) if idx not in none_indices]
        func_vals = [fxy for (idx, fxy) in enumerate(func_vals) if idx not in none_indices]
        if len(func_vals) == 0:
            logging.warning("All runs failed (are None)!")
            return

        # Determine axis construction (binned or labeled)
        def axis_construction(ax_inputs):
            vals_uniq = np.unique(ax_inputs)
            vals_num = len(vals_uniq)
            if vals_num <= max_labeled_cells:
                # Axis will be labeled
                return {
                    'labels': vals_uniq,
                    'binned': False
                }
            else:
                # Axis will be binned
                bins_num = min(max_bins, int(vals_num / 3))
                _, bin_edges = np.histogram(ax_inputs, bins=bins_num)
                return {
                    'labels': list(zip(bin_edges[:-1], bin_edges[1:])),
                    'binned': True
                }

        # Create heat map for given values
        x_axis_construction = axis_construction(x_inputs)
        y_axis_construction = axis_construction(y_inputs)
        heat_map = np.zeros(shape=(len(y_axis_construction['labels']), len(x_axis_construction['labels'])))
        for y_idx, x_idx in itertools.product(range(len(y_axis_construction['labels'])),
                                              range(len(x_axis_construction['labels']))):

            # Get labels
            y_label = y_axis_construction['labels'][y_idx]
            x_label = x_axis_construction['labels'][x_idx]

            # Filter the function evaluations to the input ranges belonging to this cell
            cell_func_vals = []
            for y_in, x_in, f_out in zip(y_inputs, x_inputs, func_vals):
                # Filter based on Y axis
                if y_axis_construction['binned']:
                    bin_lower, bin_upper = y_label
                    if not bin_lower <= y_in < bin_upper:
                        continue
                else:
                    if not y_in == y_label:
                        continue

                # Filter based on X axis
                if x_axis_construction['binned']:
                    bin_lower, bin_upper = x_label
                    if not bin_lower <= x_in < bin_upper:
                        continue
                else:
                    if not x_in == x_label:
                        continue

                # Add the function value to this cell (it belongs here)
                cell_func_vals.append(f_out)

            # Enter the data in the heat map
            if len(cell_func_vals) > 0:
                # Average all values within the cell
                cell_value = np.mean(cell_func_vals)
                heat_map[y_idx, x_idx] = cell_value

            else:
                heat_map[y_idx, x_idx] = np.NaN

        def index_for_axis(axis_constr, name):
            name = io.prettify(name)

            if not axis_constr['binned']:
                return pd.CategoricalIndex(
                    data=axis_constr['labels'],
                    categories=axis_constr['labels'],
                    name=name
                )
            else:
                # Depending on magnitude of values, cast to int
                labels = [label[0] for label in axis_constr['labels']]  # select lower bound of each bin
                if np.mean(labels) >= 8:
                    labels = [int(lab) for lab in labels]
                    return pd.Int64Index(
                        data=labels,
                        name=name
                    )
                else:
                    labels = [f"{lab:0.1f}" for lab in labels]
                    return pd.CategoricalIndex(
                        data=labels,
                        name=name
                    )

        # Wrap heat map in a pandas DataFrame so seaborn will plot axis labels correctly
        heat_map_df = pd.DataFrame(
            data=heat_map,
            index=index_for_axis(y_axis_construction, y_name),
            columns=index_for_axis(x_axis_construction, x_name)
        )

        # Plot heat map
        extra_hm_args = {}
        if heat_center is not None:
            extra_hm_args['center'] = heat_center
        sns_heat = sns.heatmap(
            data=heat_map_df,
            cmap=sns.cubehelix_palette(
                rot=random.random(),
                reverse=True,
                as_cmap=True
            ),
            annot=True,  # show numbers in each cell
            cbar=False,  # don't plot color bar
            **extra_hm_args
        )

        # Title
        title = f"Score influence of {io.prettify(x_name)} vs. {io.prettify(y_name)}\n" \
                r"(darker $\rightarrow$ better, white $\rightarrow$ missing)"
        plt.title(title)

        self._save_plot(f"joint_score_heatmap_{x_name}___vs___{y_name}", seaborn_ax=sns_heat)

    def plot_fit_progress(self):
        """
        Plot development of train and val loss over time (while training)
        :return:
        """

        # Plot with and without batch-wise losses
        self._plot_fit_progress_inner(plot_batch_wise_loss=False)
        if self.trainer.eval_after_every_batch:
            self._plot_fit_progress_inner(plot_batch_wise_loss=True)

    def _plot_fit_progress_inner(self, plot_batch_wise_loss=False, batch_dot_size=10, batch_dot_alpha=0.5,
                                 epoch_loss_line_width=3):
        # Get training history
        if len(self.trainer.losses_train) == 0:
            logging.info("Can not plot fitting progress: Trained model was loaded from disk")
            return
        epoch_loss_train = self.trainer.losses_train
        epoch_loss_val = self.trainer.losses_val

        # Construct plot
        fig = plt.figure(figsize=(18, 6))
        title = "Loss while Training"
        plt.yscale('log')  # Use logarithmic scale for losses (since they can get really large sometimes)

        # Show after-each-batch losses if available
        epochs = np.array(self.trainer.trained_epochs) - 1
        if self.trainer.eval_after_every_batch and plot_batch_wise_loss:
            # Extract losses that were captured after each batch
            batch_loss_train = self.trainer.eval_after_batch_callback.losses_training
            batch_loss_eval_mode = self.trainer.eval_after_batch_callback.losses_eval_mode

            # Determine the appropriate x coordinate to plot each after-batch loss
            batches_per_epoch = len(self.trainer.eval_after_batch_callback.training_data)
            batches_per_epoch_inv = 1 / batches_per_epoch
            batches_x = batches_per_epoch_inv + np.arange(min(epochs), max(epochs) + 1, step=batches_per_epoch_inv)

            # Plot
            scatter_args = {
                's': batch_dot_size,
                'alpha': batch_dot_alpha
            }
            plt.scatter(batches_x, batch_loss_train, label="train (after batch)", **scatter_args)
            plt.scatter(batches_x, batch_loss_eval_mode, label="train (after batch, eval mode)", **scatter_args)

            title += f", {batches_per_epoch} batches per epoch"

        # Plot training and validation losses after each epoch
        line_args = {
            'linewidth': epoch_loss_line_width
        }
        plt.plot(1 + epochs, epoch_loss_train, label="train (after epoch)", color="xkcd:bright blue", **line_args)
        plt.plot(1 + epochs, epoch_loss_val, label="val (after epoch)", color="xkcd:light red", **line_args)

        plt.title(title)
        plt.ylabel(f"loss ({self.trainer.loss})")
        plt.xlabel("epoch")
        plt.legend(loc='upper right')

        # Save in model checkpoint dir
        num_epochs = len(epochs)
        filename = f"fitting_progress_e{num_epochs:06d}_at{epochs[-1]}"
        if plot_batch_wise_loss:
            filename += "_batchwise"
        self._save_plot(os.path.join(self.iom.get_models_dir(), filename))

    @staticmethod
    def projection_tsne(features, perplexity=30, n_components=2):
        clus_s = features.shape
        logging.info(f"t-SNE... ({clus_s[0]} points, {clus_s[1]} source dimensions, {n_components} target dimensions)")
        tsne = TSNE(
            n_components=n_components,  # n_components should be set to 2 for 2d plotting
            perplexity=perplexity
        )
        features_embedded = tsne.fit_transform(features)
        logging.info("Done fitting.")

        return features_embedded

    @staticmethod
    def projection_pca(features, n_components=2):
        clus_s = features.shape
        logging.info(f"PCA... ({clus_s[0]} points, {clus_s[1]} source dimensions, {n_components} target dimensions)")

        pca = PCA(n_components=n_components)  # 2 dimensions for plotting
        features_embedded = pca.fit_transform(features)
        logging.info("Done fitting.")

        return features_embedded

    @staticmethod
    def projection_umap(features, n_neighbors, min_dist, n_components=2):
        clus_s = features.shape
        logging.info(f"UMAP... ({clus_s[0]} points, {clus_s[1]} source dimensions, {n_components} target dimensions)")

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components
        )
        features_embedded = reducer.fit_transform(features)
        logging.info("Done fitting.")

        return features_embedded
