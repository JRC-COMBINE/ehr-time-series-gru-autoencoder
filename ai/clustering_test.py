import unittest
import sys

import numpy as np

from ai.clustering import mutual_information, rand_score, jaccard_score_label_agnostic, swap, Clustering,\
    ClusteringInfo, analyze_clusterings_lineage
from common import io
from full_pipeline_mimic import get_prep_and_trainer_for_testing


class ClusteringTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Init preprocessor and trainer
        args = ['--max_epochs', '1',
                '--early_stopping_patience', '0',
                '--admissions', '100']
        with unittest.mock.patch('sys.argv', sys.argv + args):
            prep, trainer = get_prep_and_trainer_for_testing()

        # Init Clustering
        self.clustering = Clustering(
            trainer=trainer,
            iom=io.IOFunctions(
                dataset_key="mimic",
                training_id="test"
            ),
            preprocessor=prep,
            split=io.split_name_all
        )

        # Set settings so that testing is faster
        self.clustering.num_bootstraps = 3

        # Generate features to cluster
        num_adms_per_cluster = 20
        self.num_clusters = 5
        num_features = 6
        cluster_points = []
        for _ in range(self.num_clusters):
            cluster_features = np.random.randn(num_adms_per_cluster, num_features)
            cluster_features += np.random.random()  # random offset in feature space
            cluster_features *= np.random.random(num_features)  # random skew/scaling in feature space
            cluster_points.append(cluster_features)
        features = np.concatenate(cluster_points)

        # Limit to the data actually present
        features = features[:len(prep.encounter_ids_extracted)]

        self.features = features

    def test_cluster_complete(self):
        clusterings, robust_clusterings, all_clusterings = self.clustering._cluster_complete(features=self.features)
        self.assertTrue(True)  # If this is reached without error, test is successful

    def test_cluster_count_picking(self, min_count=2, max_count=100):
        cluster_counts = list(np.random.choice(
            range(min_count, max_count + 1),
            size=(max_count - min_count) // 2,
            replace=False
        ))

        # Pick some clusterings to analyze
        chosen_indices = self.clustering._pick_interesting_clusters_by_cluster_count(cluster_counts)

        # Test if extremal counts were picked
        chosen_cluster_counts = [cluster_counts[idx] for idx in chosen_indices]
        self.assertIn(min(cluster_counts), chosen_cluster_counts)
        self.assertIn(max(cluster_counts), chosen_cluster_counts)

        # Test if total count is correct
        self.assertLessEqual(len(chosen_indices), self.clustering.num_clusterings_picked_total)

    def test_clustering_summary_table_creation(self):
        # Get a clustering_info to test with
        clusterings, robust_clusterings, all_clusterings = self.clustering._cluster_complete(features=self.features)
        analyze_clusterings_lineage(clusterings=clusterings.clusterings)
        test_info = clusterings.clusterings[-1]  # type: ClusteringInfo

        # Create a summary table for it
        table = self.clustering._create_summary_table(clustering_info=test_info)

        # Check if the table has the right amount of entries (one for population and one for each cluster)
        self.assertEqual(len(table), 1 + len(np.unique(test_info.labels)))

        # Check if population is at index 0
        self.assertEqual(table.iloc[0].name, "Population")

    def test_clustering_lineage_analysis(self):
        # Cluster
        clusterings, robust_clusterings, all_clusterings = self.clustering._cluster_complete(features=self.features)
        clusterings = clusterings.clusterings

        # Analyze lineage for robust clusterings
        analyze_clusterings_lineage(clusterings=clusterings)

        # Assert that all clusterings have parents set
        clusterings.sort(key=lambda ci: ci.num_clus)
        for c in clusterings[1:]:  # Root does not have parents
            self.assertGreater(len(c.parent_clusterings), 0)


class SimilarityMetricsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Create identical sets of labels
        self.data_size = 50000
        self.min_label = 0
        self.max_label = 10
        labeling = self.get_random_labeling()
        self.identical_labelings = [labeling, labeling]

        # Create random labelings
        self.random_labelings = [labeling, self.get_random_labeling()]

        # Create permuted labelings
        labeling_perm = np.copy(labeling)
        for _ in range(10):
            num_1, num_2 = np.random.choice(self.max_label, size=2, replace=False)
            labeling_perm = swap(labeling_perm, num_1=num_1, num_2=num_2)
        self.permuted_labelings = [labeling, labeling_perm]

    def get_random_labeling(self):
        return np.random.randint(low=self.min_label, high=self.max_label, size=self.data_size)

    def test_swap(self):
        # Test normal swap
        arr = np.array([0, 0, 1, 1])
        arr_swapped = np.array([1, 1, 0, 0])
        arrays_equal = arr_swapped == swap(arr=arr, num_1=0, num_2=1)
        self.assertTrue(
            arrays_equal.all()
        )

        # Swap with unaffected numbers
        arr = np.array([0, 0, 1, 1, 2, 3, 10])
        arr_swapped = np.array([1, 1, 0, 0, 2, 3, 10])
        arrays_equal = arr_swapped == swap(arr=arr, num_1=0, num_2=1)
        self.assertTrue(
            arrays_equal.all()
        )

        # Swapping the same number (does nothing)
        arr = np.array([0, 0, 1, 1])
        arrays_equal = arr == swap(arr=arr, num_1=0, num_2=0)
        self.assertTrue(
            arrays_equal.all()
        )

    def test_identical_mutual_information(self):
        self.assertAlmostEqual(
            1.,
            mutual_information(
                clus_1_labels=self.identical_labelings[0],
                clus_2_labels=self.identical_labelings[1]
            ),
            places=3
        )

    def test_permuted_mutual_information(self):
        self.assertAlmostEqual(
            1.,
            mutual_information(
                clus_1_labels=self.permuted_labelings[0],
                clus_2_labels=self.permuted_labelings[1]
            ),
            places=3
        )

    def test_random_mutual_information(self):
        self.assertAlmostEqual(
            0.,
            mutual_information(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            places=1
        )

    def test_symmetric_mutual_information(self):
        self.assertAlmostEqual(
            mutual_information(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            mutual_information(
                clus_1_labels=self.random_labelings[1],
                clus_2_labels=self.random_labelings[0]
            ),
            6
        )

    def test_identical_rand_score(self):
        self.assertAlmostEqual(
            1.,
            rand_score(
                clus_1_labels=self.identical_labelings[0],
                clus_2_labels=self.identical_labelings[1]
            ),
            places=3
        )

    def test_permuted_rand_score(self):
        self.assertAlmostEqual(
            1.,
            rand_score(
                clus_1_labels=self.permuted_labelings[0],
                clus_2_labels=self.permuted_labelings[1]
            ),
            places=3
        )

    def test_random_rand_score(self):
        self.assertAlmostEqual(
            0.,
            rand_score(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            places=1
        )

    def test_symmetric_rand_score(self):
        self.assertAlmostEqual(
            rand_score(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            rand_score(
                clus_1_labels=self.random_labelings[1],
                clus_2_labels=self.random_labelings[0]
            ),
            6
        )

    def test_identical_jaccard_index(self):
        self.assertAlmostEqual(
            1.,
            jaccard_score_label_agnostic(
                clus_1_labels=self.identical_labelings[0],
                clus_2_labels=self.identical_labelings[1]
            ),
            places=3
        )

    def test_permuted_jaccard_index(self):
        self.assertAlmostEqual(
            1.,
            jaccard_score_label_agnostic(
                clus_1_labels=self.permuted_labelings[0],
                clus_2_labels=self.permuted_labelings[1]
            ),
            places=3
        )

    def test_random_jaccard_index(self):
        self.assertLess(
            jaccard_score_label_agnostic(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            0.25  # Jaccard index of two completely random labelings is often in the neighborhood of 0.25
        )

    def test_symmetric_jaccard_index(self):
        self.assertAlmostEqual(
            jaccard_score_label_agnostic(
                clus_1_labels=self.random_labelings[0],
                clus_2_labels=self.random_labelings[1]
            ),
            jaccard_score_label_agnostic(
                clus_1_labels=self.random_labelings[1],
                clus_2_labels=self.random_labelings[0]
            ),
            6
        )


if __name__ == '__main__':
    unittest.main()
