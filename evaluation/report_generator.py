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
from tabulate import tabulate
from scipy.stats import percentileofscore
from scipy.stats import pearsonr

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
        all_err_by_rep = {}
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
            all_err_by_rep[p] = all_err

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

            # Add error aggregates for dynamic data classes
            dyn_attr_errors = {k: np.array(v) for (k, v) in eval_raw['_eval_result_raw']['rec_error_scores'].items()}
            dyn_attr_var_stats = eval_raw['_eval_result_raw']['dyn_variance_stats']
            dyn_attr_keys = set(dyn_attr_errors.keys())
            assert dyn_attr_keys == set(dyn_attr_var_stats.keys()),\
                "rec_error_scores keys are different from dyn_variance_stats keys!"

            dyn_err_report = {}
            for dyn_attr_name in dyn_attr_keys:
                ts_errors = dyn_attr_errors[dyn_attr_name]
                var_stats = dyn_attr_var_stats[dyn_attr_name]
                dyn_err_report[dyn_attr_name] = {
                    'err_median': float(np.median(ts_errors)),
                    'var_median': float(var_stats['median']),
                    'var_mean': float(var_stats['mean'])
                }
            reports[p]['agg_dyn_attrs'] = dyn_err_report

            # Add some other values into the report
            reports[p]['rec_error_median_overall_weighted_MAPE'] = eval_raw['rec_error_median_overall_weighted_MAPE']

        # Print simplified version of report
        logging.info("\n" + "\n".join([f"{k}: {v['aggregation_over_time_series']['median']:0.6f}"
                                       for (k, v) in reports.items()]))

        # Have user examine report interactively
        pass  # (set a breakpoint here to do that!)

        # Generate a latex table that is ready to be put into a publication
        with_percentiles = True
        for rep_key, rep_val in reports.items():
            print(f"Dynamic class error table for {rep_key}:")

            # Convert from dictionary of dicts to list of dicts
            err_rep = rep_val['agg_dyn_attrs']
            dyn_stats = [dict(v, name=k) for (k, v) in err_rep.items()]

            # Generate table rows
            table_rows = []
            all_err_retrieved = all_err_by_rep[rep_key]
            var_and_err = []
            for d in dyn_stats:

                median_err = d['err_median']
                variance = d['var_median']
                row = [
                    d['name'],
                    median_err,
                    d['var_median']
                ]

                # Add error divided by variance
                if variance != 0:
                    row.append(median_err / variance)
                else:
                    row.append(float('Inf'))

                # Save variance and error in order to calculate the Pearson correlation coefficient later
                var_and_err.append((variance, median_err))

                # Add percentile
                if with_percentiles:
                    row.append(percentileofscore(all_err_retrieved, median_err))

                table_rows.append(row)

            # Sort rows
            table_rows.sort(key=lambda r: r[1])  # sort by mse

            # Make table
            dummy_label = "DummyLabelErrDivVar"
            headers = ["Class Name", "MSE", "Variance", dummy_label]
            if with_percentiles:
                headers.append("Percentile")
            table = tabulate(
                table_rows,
                headers=headers,
                floatfmt=".4f",
                tablefmt="latex_longtable"
            )
            table_str = str(table)

            # Put in proper latex column label
            table_str = table_str.replace(dummy_label, "$\\frac{\\text{MSE}}{\\text{Variance}}$")

            # Replace inf by latex explanation
            table_str = table_str.replace(" inf ", " \\textit{undefined} ")

            print()
            print(table_str)
            print()

            # Determine Pearson correlation coefficient between variance and error
            var_and_err = np.array(var_and_err)
            variances = var_and_err[:, 0]
            errors = var_and_err[:, 1]
            pearson_r, pearson_p = pearsonr(variances, errors)
            print(f"Variances and errors are correlated with Pearson correlation coefficient of {pearson_r:0.4f}"
                  f" (p = {pearson_p})")

if __name__ == "__main__":
    e = EvalReportGen()
    e.start()
