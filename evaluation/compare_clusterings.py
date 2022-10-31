# Logging
import argparse
import logging
import os

# Utility
import sys
import itertools

# Math and data
import numpy as np

# My own modules
sys.path.append("..")
from common import io
from ai.clustering import robustness_functions


class CompareClusterings:

    def __init__(self):
        # Parse arguments
        self.args = self.parse_args()

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Compare clusterings between runs")
        parser.add_argument("run_1_path", type=str, help="First run of the pair to be compared")
        parser.add_argument("run_2_path", type=str, help="Second run of the pair to be compared")
        parser.add_argument("--split", type=str, help="Split to execute the comparison on. Split has to exist for "
                                                      "both of the runs.",
                            default=io.split_name_all)
        args = parser.parse_args()
        return args

    def start(self):
        # Open serialized clustering files for both of the runs
        res_1_path = os.path.realpath(self.args.run_1_path)
        res_1_name = os.path.split(res_1_path)[-1]
        res_1 = self.open_saved_clusterings(res_1_path)
        if not res_1:
            logging.error(f"Run 1 could not be opened! ({res_1_path})")

        res_2_path = os.path.realpath(self.args.run_2_path)
        res_2_name = os.path.split(res_2_path)[-1]
        res_2 = self.open_saved_clusterings(res_2_path)
        if not res_2:
            logging.error(f"Run 2 could not be opened! ({res_2_path})")

        logging.info(f"Comparing two runs {res_1_name} and {res_2_name} ... ")

        # Handle only the picked clusterings of each run (not e.g. all robust clusterings)
        res_1 = res_1['clusterings']
        res_2 = res_2['clusterings']

        # Compare each pair of clusterings w.r.t. similarity metrics
        # WARNING: This only works correctly when the admission order for each of the runs is the same! To guarantee
        # this, admissions are sorted upon loading, before the clustering is performed.
        comparisons = self.compare_all_clusterings(res_1, res_2)

        # Create a copy of the comparisons that is sorted by similarity
        comparisons_sorted = sorted(
            comparisons,
            key=lambda comp: comp['sim_score'],
            reverse=True  # largest similarity first
        )

        # Determine overall similarity of the two runs being compared by summing over the clustering similarity
        similarities = [comp['sim_score'] for comp in comparisons]

        # Compose results
        runs_similarity = {
            'res_1_path': res_1_path,
            'res_2_path': res_2_path,
            'sim_score_max': float(np.max(similarities)),
            'sim_score_min': float(np.min(similarities)),
            'sim_score_mean': float(np.mean(similarities)),
            'comparisons': comparisons_sorted,
            'unsorted_comparisons': comparisons
        }

        logging.info(f"Overall similarity (max): {runs_similarity['sim_score_max']}")
        # Note: "Overall" similarity between two runs is given by the maximum similarity achievable between clusterings
        # of those two runs. This makes more sense than using the mean since if a single pair of clusterings between
        # the runs matches well, it doesn't matter how many pairs don't match well.

        # Save results as JSON
        filename = f"similarity_{res_1_name}_vs_{res_2_name}.json"
        results_path = os.path.realpath(os.path.join(os.curdir, filename))
        io.write_json(runs_similarity, results_path, verbose=True, pretty=True)

    def compare_all_clusterings(self, clusterings_1, clusterings_2):
        # Save results of each of the comparisons in a list
        comparison_results = []

        # Compare each pair of clusterings
        for clus_1, clus_2 in itertools.product(clusterings_1, clusterings_2):
            comparison_results.append(self.compare_clusterings(clus_1, clus_2))

        return comparison_results

    @staticmethod
    def compare_clusterings(clus_1, clus_2):
        # Save result of comparison in a dictionary
        similarities = {}
        comparison_result = {
            'comparison_name': f"({clus_1.num_clus})_{clus_1.random_id}___vs___({clus_2.num_clus})_{clus_2.random_id}",
            'similarities': similarities
        }

        # Sanity check: Abort comparison if the number of admissions is not the same for both clusterings
        if len(clus_1.labels) != len(clus_2.labels):
            comparison_result['error'] = "Number of admissions is not equal!"
        else:

            # Compare w.r.t. each of the similarity metrics available
            for func_name, func in robustness_functions.items():
                similarities[func_name] = func(
                    clus_1_labels=clus_1.labels,
                    clus_2_labels=clus_2.labels
                )

        # Heuristic for overall similarity - this makes sense as an overall score since all the metrics get
        # larger when the clusterings are more similar
        comparison_result['sim_score'] = float(np.mean(list(similarities.values())))

        return comparison_result

    def open_saved_clusterings(self, run_abs_path):
        # Clusterings dir contains multiple folders with clusterings results for all different splits that the
        # clustering was performed for. Navigate into the clustering dir and select the chosen split.
        clus_dir = os.path.join(run_abs_path, io.clusterings_name, self.args.split)

        # Abort if path doesn't exist
        if not os.path.exists(clus_dir):
            return False

        # Find clustering results pickle file
        res_path = os.path.join(clus_dir, f"{self.args.split}.pkl")
        if not os.path.isfile(res_path):
            return False

        # Open file
        res = io.read_pickle(res_path)
        return res


if __name__ == "__main__":
    c = CompareClusterings()
    c.start()
