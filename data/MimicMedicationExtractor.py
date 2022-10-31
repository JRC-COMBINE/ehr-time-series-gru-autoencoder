import logging

import pandas as pd
from data.BaseExtractor import BaseExtractor

logging.getLogger().setLevel(logging.INFO)


"""
Credit: Richard Polzin, Konstantin Sharafutdinov, Jayesh Bhat, Kilian Merkelbach 
"""


class MimicMedicationExtractor(BaseExtractor):
    def __init__(self, extraction_path, disable_multiprocessing=True):
        BaseExtractor.__init__(
            self,
            database_name="mimic",
            extraction_path=extraction_path,
            extraction_kind="drugs",
            raw_data_tables=[None],  # We need a list of length 1 so that medication query is performed only a single
            # time
            disable_multiprocessing=disable_multiprocessing
        )

    def df_to_ts(self, raw):
        df, encounter_id = raw
        BaseExtractor.df_to_ts(self, (df, encounter_id, ["itemid"], 'charttime', 'valuenum'))

    def query_data(self, encounter_id, itemids):
        """
        Query for medication data
        :param encounter_id: admission ID to query for
        :param itemids: IGNORED by this function since filtering for item ids is not performed during extraction
        but during preprocessing
        :return:
        """

        query_tokens = [
            "("
            "SELECT itemid, charttime, rate as valuenum, rateuom",
            "FROM inputevents_cv",
            f"WHERE hadm_id={encounter_id}",
            "AND originalroute in ('IV Drip', 'Intravenous','Drip')",
            "AND rate is not null"
            ")",
            "union",
            "("
            "SELECT itemid, starttime as charttime, rate as valuenum, rateuom",
            "FROM inputevents_mv",
            f"WHERE hadm_id={encounter_id}",
            "AND rate is not null"
            ")"
        ]
        query = "\n".join(query_tokens)
        return BaseExtractor.execute_query(self, query)
