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


class EvalReportGen:

    def __init__(self):
        # Parse arguments
        self.args = self.parse_args()

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="Generate evaluation report using eval file")
        parser.add_argument("--eval_file_paths", type=str, nargs='+')
        args = parser.parse_args()
        return args

    def start(self):
        paths = [os.path.realpath(p) for p in self.args.eval_file_paths]

        logging.info(f"Generating report for {len(paths)} eval files ... ")

        reports = {}
        for p in paths:
            # Open file
            eval_raw = io.read_json(p)

            # Get array that contains error values for all time series
            all_err = np.array(sum(eval_raw['_eval_result_raw']['rec_error_scores'].values(), []))

            # Get array of errors for admissions - each admission contributes the mean error over all of its time steps
            adm_errors = np.array(eval_raw['adm_rec_errors'])

            # Helper functions

            def leave_worst_out(arr, percent_left_out):
                perc = np.percentile(arr, 100 - percent_left_out)
                arr_without_worst = arr[arr <= perc]
                return {
                    'mean': float(np.mean(arr_without_worst)),
                    'median': float(np.median(arr_without_worst)),
                    'threshold': float(perc)
                }

            def percentiles(arr):
                return {float(perc): float(np.percentile(arr, perc)) for perc in np.linspace(0, 100, 11)}

            def full_rep(arr):
                return {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'percentiles': percentiles(arr),
                    'worst_left_out': {perc: leave_worst_out(arr, percent_left_out=perc) for perc in [0.1, 1, 5, 10]},
                    'len': len(arr)
                }

            # Compose into report
            reports[p] = {
                'aggregation_over_time_series': full_rep(all_err),
                'aggregation_over_admissions': full_rep(adm_errors)
            }

        # Have user examine report interactively
        pass  # (set a breakpoint here!)


if __name__ == "__main__":
    e = EvalReportGen()
    e.start()
