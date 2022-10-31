import unittest
from tqdm import tqdm
from info import IcdInfo


class TestIcdInfo(unittest.TestCase):
    def test_diagnoses_lookup(self):
        # Real ICD codes from MIMIC
        diagnosis_codes = [
            '76524',
            '7678',
            '5111',
            '135',
            '99685',
            'V1047',
            '03849',
            '70724',
            '03849',
            'E9331',
            '585'
        ]

        # Get ICD tree path (from root to node) for each of the codes
        nodes = []
        for code in diagnosis_codes:
            node = IcdInfo.icd_tree_search(
                icd_kind='icd9_code_diagnoses',
                icd_code=code
            )
            nodes.append(node)

        # Check if all the codes were found
        self.assertNotIn(None, nodes)

    def test_diagnoses_naming(self):
        # Real ICD codes from MIMIC
        diagnosis_codes = [
            '76524',
            '7678',
            '5111',
            '135',
            '99685',
            'V1047',
            '03849',
            '70724',
            '03849',
            'E9331',
            '585'
        ]

        # Get ICD tree path (from root to node) for each of the codes
        nodes = []
        for code in diagnosis_codes:
            node = IcdInfo.icd_tree_search(
                icd_kind='icd9_code_diagnoses',
                icd_code=code
            )
            nodes.append(node)

        # Check if all the codes were found
        self.assertNotIn(None, nodes)

        # Find out name for all code - this should use the already identified nodes for each code
        for code in diagnosis_codes:
            code_name = IcdInfo.name_for_code(
                icd_kind='icd9_code_diagnoses',
                icd_code=code
            )

            # Make sure the name is not the default name given to codes for which the name can not be found
            self.assertNotIn('icd9_code_diagnoses', code_name)

    def test_procedures_lookup(self):
        # Real ICD codes from MIMIC
        procedure_codes = [
            '7865',
            '3142',
            '0124',
            '0045',
            '9955',
            '311',
            '9390',
            '3892',
            '3932',
            '3733'
        ]

        # Get ICD tree path (from root to node) for each of the codes
        nodes = []
        for code in procedure_codes:
            node = IcdInfo.icd_tree_search(
                icd_kind='icd9_code_procedures',
                icd_code=code
            )
            nodes.append(node)

        # Check if all the codes were found
        self.assertNotIn(None, nodes)

    def test_node_naming_all(self):
        # Test for both diagnoses and for procedures
        for icd_kind in IcdInfo.icd_kind_keys:

            # Load ICD tree
            IcdInfo._load_from_disk(icd_kind)

            # Check for each node if it receives a valid name
            for icd_node in tqdm(IcdInfo._cached_icd_trees[icd_kind].descendants,
                                 desc=f"Testing naming of {icd_kind} nodes"):
                name = IcdInfo.name_for_node(icd_node)
                self.assertIsNotNone(name, msg=f"ICD node {icd_node} ({icd_kind}) has None name!")

    def test_node_naming_and_tagging(self):
        # Test if chronic diagnosis is marked as such
        for chronic_code in ["135", "585"]:
            chronic_node = IcdInfo.icd_tree_search(
                icd_kind='icd9_code_diagnoses',
                icd_code=chronic_code
            )
            chronic_name = IcdInfo.name_for_node(
                icd_node=chronic_node
            )

            # If "chronic" is already in the disease's name, we don't need to mark it again
            if "chronic" not in chronic_name.lower():
                self.assertIn("(C)", chronic_name)
            else:
                self.assertNotIn("(C)", chronic_name)

        # Test if diagnosis with acute in the name can still be chronic
        acute_hf_code = "42831"
        acute_hf_node = IcdInfo.icd_tree_search(
            icd_kind='icd9_code_diagnoses',
            icd_code=acute_hf_code
        )
        acute_hf_name = IcdInfo.name_for_node(
            icd_node=acute_hf_node
        )
        self.assertIn("(C)", acute_hf_name)


if __name__ == '__main__':
    unittest.main()
