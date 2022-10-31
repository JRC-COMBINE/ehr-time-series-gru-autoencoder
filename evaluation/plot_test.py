import unittest
import unittest.mock
import sys

import numpy as np
import random
from scipy.ndimage import gaussian_filter1d

from evaluation.plot import Plotting
from common import io
from ai import clustering
from eval import Evaluation
from full_pipeline_mimic import get_prep_and_trainer_for_testing


class PlotTest(unittest.TestCase):
    def setUp(self) -> None:
        # Init preprocessor and trainer
        args = ['--max_epochs', '1',
                '--early_stopping_patience', '0',
                '--admissions', '100']
        with unittest.mock.patch('sys.argv', sys.argv + args):
            prep, trainer = get_prep_and_trainer_for_testing()

        # Fake admissions
        num_admissions = len(trainer.get_split_indices('all'))
        admissions = np.arange(num_admissions)

        # Fake clusterings
        self.clusterings_fake = []
        for k in range(2, 12+1):
            labeling = np.random.randint(0, k, size=admissions.shape)
            c = clustering.ClusteringInfo(
                algorithm="kmeans",
                options={'k': k},
                labels=labeling
            )
            c.num_clus = k
            self.clusterings_fake.append(c)

        # Evaluator
        iom = io.IOFunctions(
            dataset_key="mimic",
            training_id="test"
        )
        evaluator = Evaluation(
            iom=iom,
            preprocessor=prep,
            trainer=trainer,
            clustering=None,
            split=io.split_name_unit_tests,
            evaluated_fraction=0.2
        )

        # Plotter
        self.plotter = Plotting(
            iom=iom,
            preprocessor=prep,
            trainer=trainer,
            evaluator=evaluator,
            clustering=None,
            split=io.split_name_unit_tests
        )

    def test_plot_single_reconstruction(self):
        def plot_rec():
            # Generate random "ground truth"
            num_steps = np.random.randint(5, 60)
            gt = [np.random.uniform(15, 150)]
            times = [np.random.uniform(-200, 200)]
            for _ in range(num_steps - 1):
                time_delta = np.random.uniform(1, 1000)
                val_delta = time_delta * np.random.uniform(-5, 5)
                new_val = gt[-1] + val_delta
                gt.append(new_val)
                new_time = times[-1] + time_delta
                times.append(new_time)

            # Generate reconstruction that tries to track the ground truth
            rec = []
            val_magnitude = np.abs(np.mean(gt))
            for idx in range(num_steps):
                error = val_magnitude * np.random.normal()
                rec_val = gt[idx] + error
                rec.append(rec_val)
            rec = np.array(rec)
            rec = gaussian_filter1d(rec, np.ceil(num_steps / 10).astype(int))

            self.plotter._plot_time_series_reconstruction_for_attr(
                col_name='50861',
                adm_indices=[45],
                ground_truths=[np.array(gt)],
                reconstructions=[rec],
                times=[np.array(times)]
            )
            self.assertTrue(True)

        for _ in range(20):
            plot_rec()

    def test_icd_cumulative_covering_bars(self):
        self.plotter._plot_icd_cumulative_covering_bars(
            clusterings=self.clusterings_fake
        )
        self.assertTrue(True)

    def test_icd_distribution_bar_plot(self):
        self.plotter._plot_icd_distribution_bar_plots(
            clusterings=self.clusterings_fake
        )
        self.assertTrue(True)

    def test_icd_distribution_table(self):
        self.plotter._plot_icd_distribution_tables(
            clusterings=self.clusterings_fake
        )
        self.assertTrue(True)

    def test_icd_mortality_table(self):
        self.plotter._plot_icd_mortality_tables(
            clusterings=self.clusterings_fake
        )
        self.assertTrue(True)

    def test_cluster_similarity_matrix(self):
        self.plotter._plot_cluster_similarity_matrix(
            clusterings=self.clusterings_fake
        )
        self.assertTrue(True)

    def test_p_value_distribution(self, num_p_vals=5000, num_sampling_iterations=2):
        # Generate list of p values from actual clusterings
        original_p_vals = np.concatenate([np.random.random(num_p_vals), np.random.random(num_p_vals) / 2])
        original_p_vals = np.random.choice(original_p_vals, num_p_vals, replace=False)
        # (distribution is closer to 0 than to 1)

        # Generate list of p values from random sampling iterations
        sampling_p_vals_list = [np.random.random(num_p_vals) for _ in range(num_sampling_iterations)]

        # Plot
        self.plotter._plot_p_value_distribution(
            original_p_vals=original_p_vals,
            sampling_p_vals_list=sampling_p_vals_list
        )
        self.assertTrue(True)

    def test_alluvial(self):
        # Plot the fake clusterings using alluvial flow
        self.plotter.plot_alluvial_flow(
            clusterings=self.clusterings_fake
        )

        # If plotting does not crash, the test is passed
        self.assertTrue(True)

    def test_age_vs_survival(self):
        # Generate random ages and survivals
        max_age = 89
        ages = np.random.uniform(14, max_age, 100)
        survivals = np.array([a / max_age <= random.random() for a in ages])  # higher age -> worse survival chances

        self.plotter._plot_age_vs_survival_curve(
            labels=self.clusterings_fake[0].labels,
            survivals=survivals,
            ages=ages,
            plot_dir=self.plotter.plots_dir
        )

        self.assertTrue(True)

    def test_rec_performance_vs_survival(self, mean_rec_error=0.25, num_adms=2000):
        # Generate random reconstruction performance
        rec_errors = np.random.normal(mean_rec_error, 0.1, size=num_adms)
        rec_errors -= np.min(rec_errors)
        rec_errors += mean_rec_error * np.random.random()
        rec_errors.sort()

        # Generate random survival: Make survival less likely if patient has high reconstruction error
        rec_err_z_scores = (rec_errors - np.mean(rec_errors)) / np.std(rec_errors)
        survivals = [0.5 > np.random.normal(score, 2) for score in rec_err_z_scores]

        # Plot
        for n_bins in np.linspace(2, 30, 10).astype(int):
            self.plotter._plot_rec_err_vs_survival_bars(
                rec_errors=list(rec_errors),
                survivals=survivals,
                num_bins=n_bins
            )

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
