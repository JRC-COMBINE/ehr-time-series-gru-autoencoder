# Logging
import argparse
import logging
import os

# Utility
import sys
import itertools

# Math and data
import numpy as np
import humanfriendly as hf

# My own modules
sys.path.append("..")
from common import io


# Show time along with log messages
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


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

            # If file is not valid, skip it
            if eval_raw is None:
                logging.info(f"File {p} could not be loaded or path doesn't exist.")
                continue
            if not eval_raw['after_training']:
                logging.info(f"File {p}: Training or evaluation is not completed!")
                continue

            # Initialize report
            reports[p] = {
                'full_path': p
            }

            # Get array that contains error values for all time series
            all_err = np.array(sum(eval_raw['_eval_result_raw']['rec_error_scores'].values(), []))

            # Get array of errors for admissions - each admission contributes the mean error over all of its time steps
            adm_errors = np.array(eval_raw['_eval_result_raw']['adm_rec_errors'])

            # Collect the observation-wise errors
            # (I call these observations because an admission may have multiple (e.g. two) measurements for a given
            # time step. The results would be two observations and one time step.
            adm_total_observations = np.array(eval_raw['_eval_result_raw']['lengths'])
            reports[p]['num_observations'] = {
                'num_adm': len(adm_total_observations),
                'min': int(min(adm_total_observations)),
                'max': int(max(adm_total_observations)),
                'mean': float(np.mean(adm_total_observations)),
                'p25': float(np.percentile(adm_total_observations, 25)),
                'p75': float(np.percentile(adm_total_observations, 75)),
                'median': np.median(adm_total_observations),
                'sum': int(np.sum(adm_total_observations)),
                'note': "Observations are not the same as time steps. A single time step may have more than one "
                        "observation. The number of observations for an admission is the sum of the length of all its "
                        "time series."
            }
            reports[p]['num_observations']['sum_humanreadable'] = \
                hf.format_number(reports[p]['num_observations']['sum'])

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
                return {float(perc): float(np.percentile(arr, perc)) for perc in np.linspace(0, 100, 201)}

            def full_rep(arr):
                return {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'percentiles': percentiles(arr),
                    'worst_left_out': {perc: leave_worst_out(arr, percent_left_out=perc) for perc in [0.1, 1, 5, 10]},
                    'len': len(arr)
                }

            # Compose into report
            reports[p].update({
                'aggregation_over_time_series': full_rep(all_err),
                'aggregation_over_admissions': full_rep(adm_errors)
            })

            # Add some other values into the report
            reports[p]['rec_error_median_overall_weighted_MAPE'] = eval_raw['rec_error_median_overall_weighted_MAPE']

        # Print simplified version of report
        logging.info("\n" + "\n".join([f"{k}: {v['aggregation_over_time_series']['median']:0.6f}"
                                       for (k, v) in reports.items()]))

        # Have user examine report interactively
        pass  # (set a breakpoint here!)


if __name__ == "__main__":
    e = EvalReportGen()
    e.start()
