import logging

# Math and data processing
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from typing import Dict

# Parent class
from data.BaseExtractor import BaseExtractor


"""
Credit: Richard Polzin, Kilian Merkelbach, Konstantin Sharafutdinov, Jayesh Bhat 
"""


class MimicExtractor(BaseExtractor):
    def __init__(self, extraction_path, disable_multiprocessing=False, textual_diagnosis_selection=None):
        # Decide what static data to extract from MIMIC
        static_extraction_details = {
            "admissions": {
                "key_field": {  # Field which is used as a key for the table
                    "name": "hadm_id",  # name of the key field in THIS table
                    "value": "encounter_id"  # name of the attribute we should use as the value for the key
                },
                # List of fields to be extracted
                "fields_to_extract": ["admittime", "dischtime", "subject_id", "ethnicity", "diagnosis"],
                "field_renamings": {
                    "diagnosis": "admission_diagnosis"
                }
            },
            "patients": {
                "key_field": {
                    "name": "subject_id",
                    "value": "subject_id"
                    # Explanation: For the MIMIC patients table, we take the patient's "subject_id" attribute (which we
                    # already know from the admissions table) and use it as a key in the patient table's "subject_id"
                    # field.
                },
                # List of fields to be extracted
                "fields_to_extract": ["gender", "dob", "dod"]
            },
            "diagnoses_icd": {
                "key_field": {
                    "name": "hadm_id",
                    "value": "encounter_id"
                },
                "fields_to_extract": ["icd9_code"],
                "field_renamings": {
                    "icd9_code": "icd9_code_diagnoses"
                }
            },
            "procedures_icd": {
                "key_field": {
                    "name": "hadm_id",
                    "value": "encounter_id"
                },
                "fields_to_extract": ["icd9_code"],
                "field_renamings": {
                    "icd9_code": "icd9_code_procedures"
                }
            },
            "icustays": {
                "key_field": {
                    "name": "hadm_id",
                    "value": "encounter_id"  # encounter_id is our internal name for the hadm_id
                },
                "fields_to_extract": ["intime", "outtime", "first_careunit", "last_careunit"],
                "field_renamings": {
                    "intime": "icu_intime",
                    "outtime": "icu_outtime",
                    "first_careunit": "icu_first_careunit",
                    "last_careunit": "icu_last_careunit"
                }
            },
            "chartevents": {  # extract "static" data like height and weight from chartevents, which also
                # contains dynamic data
                "key_field": {
                    "name": "hadm_id",
                    "value": "encounter_id"  # encounter_id is our internal name for the hadm_id
                },
                "fields_to_extract": {
                    "weight_kg": ['762', '763', '3723', '3580'],
                    "weight_lb": ['3581'],
                    "weight_oz": ['3582'],
                    "height_in": ['920', '1394', '4187', '3486'],
                    "height_cm": ['3485', '4188']
                }
            },
        }

        # Save diagnosis filter
        self.textual_diagnosis_selection = textual_diagnosis_selection  # e.g. "myocardial infarction"

        # Cache item labels (for each item id)
        self._item_label_cache = {}  # type: Dict[str, str]
        self.item_meta_info = {}  # type: Dict[str, Dict]

        BaseExtractor.__init__(
            self,
            database_name="mimic",
            extraction_path=extraction_path,
            extraction_kind="exams",
            raw_data_tables=["chartevents", "labevents"],
            raw_data_table_preferred="labevents",
            static_extraction_details=static_extraction_details,
            disable_multiprocessing=disable_multiprocessing
        )

    def retrieve_all_encounter_ids(self):
        """
        Retrieve a list of all admission IDs in MIMIC
        (applying any given textual diagnosis descriptions as a filter)
        :return:
        """

        # Download admission table
        admissions_table = self.retrieve_full_table("admissions")

        # Filter based on textual diagnosis description
        if self.textual_diagnosis_selection is not None:

            # Get list of all diagnoses (from admissions table)
            diagnoses = list(enumerate(admissions_table.diagnosis))
            diagnoses = [(idx, label) for (idx, label) in diagnoses if label is not None]  # Filter out None's

            # Filter based on each of the supplied terms
            for diagnosis_term in self.textual_diagnosis_selection:

                # Convert the search term to lower case
                diagnosis_term = diagnosis_term.lower()

                # Filter based on the term
                logging.info(f"Filtering to admission with diagnosis containing '{diagnosis_term}'...")
                diagnoses = [(idx, label) for (idx, label) in diagnoses if diagnosis_term in label.lower()]
                logging.info(f"{len(diagnoses)} admission left after filtering")

            # Select admission indices
            selected_adm_idx = [idx for (idx, label) in diagnoses]
            selected_adms = admissions_table.iloc[selected_adm_idx]
        else:
            selected_adms = admissions_table

        return list(selected_adms.hadm_id)

    def retrieve_item_id_descriptions(self):
        # Download d_items table from MIMIC
        d_items_table = self.retrieve_full_table("d_items")
        return d_items_table

    def label_for_item_id(self, item_id) -> str:
        if item_id not in self._item_label_cache:
            # Convert item id to integer (otherwise it will not be found in the tables, where it is stored as a number)
            item_id_int = int(item_id)

            # Get label for item (e.g. a charted item)
            item_label_table = self.retrieve_item_id_descriptions()
            label_row = item_label_table.loc[item_label_table['itemid'] == item_id_int]
            label = label_row['label']

            # Get label of labitem (any lab measurement)
            labitem_label_table = self.retrieve_labitem_id_descriptions()
            label_row_lab = labitem_label_table.loc[labitem_label_table['itemid'] == item_id_int]
            label_lab = label_row_lab['label']

            # Assign the correct label to this attribute type
            extra_info = {}
            if len(label) > 0:
                item_label = label.iloc[0]
                extra_info['origin'] = "d_items"
                extra_info.update(label_row[['dbsource', 'category']].iloc[0].to_dict())
            elif len(label_lab) > 0:
                item_label = label_lab.iloc[0]
                extra_info['origin'] = "d_labitems"
                extra_info.update(label_row_lab[['fluid', 'category']].iloc[0].to_dict())
            else:
                item_label = f"unknown_item_label_for_id_{item_id}"  # should never actually happen

            # Save meta info
            self.item_meta_info[item_id] = extra_info

            # Save label for item id in cache
            self._item_label_cache[item_id] = item_label

        return self._item_label_cache[item_id]

    def retrieve_labitem_id_descriptions(self):
        # Download d_labitems table from MIMIC
        d_labitems_table = self.retrieve_full_table("d_labitems")
        return d_labitems_table
    
    def postprocess_raw_static_data(self, encounter_info):
        """
        Post-processes encounter_info dictionary by calculating age from date of birth and current date. Also,
        determine if the patient has died.

        :param encounter_info:
        :return: encounter_info with new information added
        """

        # Add entry to encounter_info for information that is NOT known at the time of admission (time of death,
        # duration of stay, etc.)
        encounter_info['future_info'] = {}

        # Calculate age
        age = encounter_info['admittime'].to_pydatetime() - encounter_info['dob'].to_pydatetime()
        age_years = age.total_seconds() / 60. / 60. / 24. / 365.25  # a float, e.g. 39.01622556
        if age_years > 89:
            # This patient's age was obscured to comply with HIPAA
            age_years = 89
        encounter_info['age_years'] = age_years
        del encounter_info['dob']

        # Calculate BMI (body-mass index)
        # (the formula is kg / m**2)
        weight_kg = encounter_info['weight_kg']
        height_cm = encounter_info['height_cm']
        if not any(np.isnan([weight_kg, height_cm])):
            height_meters = height_cm / 100
            bmi = weight_kg / (height_meters ** 2)
        else:
            bmi = np.nan
        encounter_info['future_info']['bmi'] = bmi
        encounter_info['future_info']['weight_kg'] = weight_kg
        encounter_info['future_info']['height_cm'] = height_cm
        del encounter_info['weight_kg']
        del encounter_info['height_cm']

        # Find out which ICU units the patient was on
        first_unit = encounter_info['icu_first_careunit']
        if type(first_unit) != list:
            first_unit = [first_unit]
        last_unit = encounter_info['icu_first_careunit']
        if type(last_unit) != list:
            last_unit = [last_unit]
        icu_units = "__".join(sorted(set(first_unit + last_unit)))
        encounter_info['future_info']['icu_stations'] = icu_units
        del encounter_info['icu_first_careunit']
        del encounter_info['icu_last_careunit']

        # Determine how many days patients spent in the ICU
        # There can be multiple ICU stays for a single stay in the hospital - if this is the case, the `icu_intime`
        # and `icu_outtime` fields will be lists. If there is only a single ICU stay for the hospital stay, we also
        # want to treat the dates of the ICU stay as a list for simplicity.
        icu_intimes = encounter_info['icu_intime']
        icu_outtimes = encounter_info['icu_outtime']
        if type(icu_intimes) != list:
            icu_intimes = [icu_intimes]
            icu_outtimes = [icu_outtimes]

        def get_days(in_time: datetime, out_time: datetime):
            duration = out_time - in_time
            days = duration.total_seconds() / 3600 / 24
            return days

        # Calculate sum of ICU days
        days_in_care_icu = 0
        icu_visit_count = 0
        for icu_in, icu_out in zip(icu_intimes, icu_outtimes):
            icu_in = icu_in.to_pydatetime()
            icu_out = icu_out.to_pydatetime()

            # In any case where one of the ICU entry/exit dates is not known, we can't assume anything and don't count
            # the ICU visit in the days statistics
            if icu_in is pd.NaT or icu_out is pd.NaT:
                continue

            days_in_care_icu += get_days(
                icu_in,
                icu_out
            )
            icu_visit_count += 1

        encounter_info['future_info']['days_in_care'] = days_in_care_icu
        encounter_info['future_info']['icu_visits'] = icu_visit_count
        del encounter_info['icu_intime']
        del encounter_info['icu_outtime']

        # Determine how many days patients spent in the hospital (even non-ICU)
        days_in_care_hospital = get_days(
            encounter_info['admittime'].to_pydatetime(),
            encounter_info['dischtime'].to_pydatetime()
        )
        encounter_info['future_info']['days_in_care_hospital'] = days_in_care_hospital

        # Did the patient die in the hospital?
        date_death = encounter_info['dod'].to_pydatetime()
        death_is_known = date_death is not pd.NaT
        # Note: If the patient died in the hospital, the `deathtime` attribute in the `admissions` table offers more
        # precision than the `dod` attribute (the `dod` attribute does not have hours and minutes, only the day of
        # death). However, for this distinction of mortality, the precision offered by `dod` suffices.
        if not death_is_known:
            death_in_hospital = False
            death_within_28days_of_adm = False
            death_after_28days_after_adm = False
        else:
            death_in_hospital = date_death <= encounter_info['dischtime'].to_pydatetime()
            four_weeks_after_admission = encounter_info['admittime'].to_pydatetime() + timedelta(days=28)
            death_within_28days_of_adm = date_death <= four_weeks_after_admission
            death_after_28days_after_adm = date_death > four_weeks_after_admission

        encounter_info['future_info']['survival'] = not death_within_28days_of_adm
        encounter_info['future_info']['survival_to_discharge'] = not death_in_hospital
        encounter_info['future_info']['survival_all_time'] = not death_after_28days_after_adm
        del encounter_info['admittime']
        del encounter_info['dischtime']
        del encounter_info['dod']

        # Save ethnicity as future info
        ethnicity = encounter_info['ethnicity'].capitalize()
        if "/" in ethnicity:
            ethnicity = ethnicity.split("/")[0]
        if " - " in ethnicity:
            ethnicity = ethnicity.split(" - ")[0]
        ethnicity_merge = {
            'Other': 'Other',
            'Multi race ethnicity': 'Other'
        }
        if ethnicity in ethnicity_merge:
            ethnicity = ethnicity_merge[ethnicity]
        encounter_info['future_info']['ethnicity'] = ethnicity
        del encounter_info['ethnicity']

        # "Diagnosis" upon admission (often the reason for admission). This free-form text field is only used for
        # cluster evaluation.
        admission_diagnosis = encounter_info['admission_diagnosis']
        if admission_diagnosis is None:
            admission_diagnosis = "No admission diagnosis"
        encounter_info['future_info']['admission_diagnosis'] = admission_diagnosis
        del encounter_info['admission_diagnosis']

        # Remove information that only served as an intermediate key (e.g. subject id)
        del encounter_info['subject_id']
        del encounter_info['encounter_id']

        # Convert future info entries to normal entries with future prefix
        for entry_name, entry_val in encounter_info['future_info'].items():
            encounter_info[f'FUTURE_{entry_name}'] = entry_val
        del encounter_info['future_info']

        return encounter_info

    def preprocess_raw_events(self, df):
        df["itemid"] = df["itemid"].astype(int)

        # Drop rows that have NaN as the numerical value
        df.dropna(subset=["valuenum"], inplace=True)

        return df

    def df_to_ts(self, raw):
        df, encounter_id = raw
        BaseExtractor.df_to_ts(self, (df, encounter_id, ["itemid"], 'charttime', 'valuenum'))

    def query_data(self, encounter_id, raw_data_name):
        query_tokens = [
            "SELECT itemid, charttime, valuenum, valueuom",
            f"FROM {raw_data_name}",
            f"WHERE hadm_id = {encounter_id}"
        ]
        query = "\n".join(query_tokens)

        return BaseExtractor.execute_query(self, query)
