import unittest

import numpy as np
import random

from data.preprocessor import Preprocessor
from common import io
from full_pipeline_mimic import get_prep_and_trainer_for_testing


class PreprocessorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # Init preprocessor and trainer
        prep, self.trainer = get_prep_and_trainer_for_testing()
        self.prep = prep  # type: Preprocessor

    def test_not_none(self):
        # Extract static data
        static_info = self.prep.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(io.split_name_all)
        )

        # Check if every single value is not None
        for static_attr, static_vals in static_info.items():
            for val_idx, val in enumerate(static_vals):
                self.assertIsNotNone(val, msg=f"Static values for attribute {static_attr} contains a None "
                                              f"value at index {val_idx}!")

    def test_days_in_care_survival(self):
        # Extract static data
        static_info = self.prep.extract_static_medical_data(
            adm_indices=self.trainer.get_split_indices(io.split_name_all)
        )

        # Split days in care into two attributes: one for survivors and one for deceased
        self.prep.split_days_in_care_by_survival(static_info)

        # Check if attributes present and contain all admissions
        new_attributes = ['FUTURE_days_in_care_survivors', 'FUTURE_days_in_care_deceased']
        random_old_attribute = random.choice(list(static_info.keys()))
        for attr in new_attributes:
            self.assertIn(attr, static_info)
            self.assertEqual(len(static_info[attr]), len(static_info[random_old_attribute]))

    def test_scaling_and_unscaling(self):
        # Generate random data
        orig = np.random.random(100)

        # Scale it using a selection of scalers from prep
        col_selection = np.random.choice(list(self.prep._scaler_by_col_name.keys()), 10)
        for col_name in col_selection:

            # Scale
            scaled = self.prep.perform_scaling(
                unscaled_value=orig,
                column_name=col_name
            )

            # Unscale and check if it is the same as the original data
            unscaled = self.prep.reverse_scaling(
                scaled_value=scaled,
                column_name=col_name
            )
            self.assertEqual(orig.shape, unscaled.shape)
            self.assertTrue(np.allclose(orig, unscaled))


if __name__ == '__main__':
    unittest.main()
