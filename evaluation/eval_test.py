import unittest

import numpy as np

import random

from ai.clustering import Clustering, ClusteringInfo
from evaluation.eval import Evaluation
from evaluation.plot import Plotting
from common import io

from sklearn import tree


class DecisionTreeEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        # Init Clustering
        split = io.split_name_unit_tests
        iom = io.IOFunctions(dataset_key="mimic", training_id="test")
        self.clustering = Clustering(
            trainer=None,
            iom=iom,
            preprocessor=None,
            split=split
        )

        # Fake clustering
        self.num_adms = 100
        num_labels = 5
        num_clusterings = 2
        min_label = 1
        self.clustering.clusterings = [
            ClusteringInfo(
                algorithm="fake_alg",
                options={},
                labels=np.random.randint(low=min_label, high=min_label + num_labels, size=self.num_adms)
            ) for _ in range(num_clusterings)]

        # Init eval
        self.eval = Evaluation(
            trainer=None,
            iom=iom,
            preprocessor=None,
            clustering=self.clustering,
            split=split
        )

        # Init plotting
        self.plotter = Plotting(
            iom=iom,
            preprocessor=None,
            trainer=None,
            evaluator=self.eval,
            clustering=self.clustering,
            split=split
        )

    def test_frequent_itemset_mining(self, num_items: int = 25, num_itemsets: int = 1000, max_items_per_set: int = 10):
        # Generate random itemsets
        items = np.arange(num_items)
        probs = np.random.random(len(items))
        probs /= np.sum(probs)
        itemsets = [np.random.choice(items, size=np.random.randint(1, max_items_per_set + 1), replace=False, p=probs)
                    for _ in range(num_itemsets)]

        # Determine frequent itemsets
        freq_itemsets = self.eval._mine_frequent_itemsets(
            itemsets=itemsets
        )

        # Test if we found any frequent itemsets
        self.assertGreater(len(freq_itemsets), 0)

    def test_decision_tree_train(self, num_features=5):

        # Create fictional feature data
        tree_features = np.random.random((self.num_adms, num_features))

        # Run test for each clustering
        for clustering_info in self.clustering.clusterings:
            all_tree_training_results = self.eval._train_decision_trees(
                target_labels=clustering_info.labels,
                features=tree_features
            )
            trained_tree, _, _ = all_tree_training_results[0]

            self.assertIsInstance(trained_tree, tree.DecisionTreeClassifier)

    def test_decision_tree_nan_filling(self, num_features=10, nan_ratio=0.3):
        # Generate feature columns
        feature_cols = []
        for _ in range(num_features):
            col = []
            for adm_idx in range(self.num_adms):
                r = random.random()
                if r < nan_ratio:
                    val = np.nan
                else:
                    val = r
                col.append(val)
            feature_cols.append(col)

        # Fill up NaN values with something else (that also does not occur in the data)
        feature_cols, _ = self.eval._nan_filling(feature_columns=feature_cols)

        # Assert that no NaN is present anymore
        self.assertFalse(np.any(np.isnan(feature_cols)))

    def test_decision_tree_plotting(self, num_features=5):

        # Create fictional feature data
        tree_features = np.random.random((self.num_adms, num_features))

        # Train tree for first clustering
        clus_info = self.clustering.clusterings[0]
        all_tree_training_results = self.eval._train_decision_trees(
            target_labels=clus_info.labels,
            features=tree_features
        )

        # Plot all trees
        for trained_tree, tree_score, tree_class_counts in all_tree_training_results:
            self.plotter.plot_decision_tree_analysis(
                trained_tree=trained_tree,
                tree_score=tree_score,
                tree_class_counts=tree_class_counts,
                nan_fill_value=-10000,
                clustering_info=clus_info,
                feature_labels=[f"Test Feature Lab {i + 1}" for i in range(num_features)],
                data_selection_name='test_selection'
            )


if __name__ == '__main__':
    unittest.main()
