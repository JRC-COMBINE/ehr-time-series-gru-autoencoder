import logging
import os
import string

import pickle
import pandas as pd


"""
Information about ICD 9 codes used in MIMIC
"""

"""
Diagnoses: Info from https://en.wikipedia.org/wiki/List_of_ICD-9_codes

ICD-9 codes 001–139: infectious and parasitic diseases
ICD-9 codes 140–239: neoplasms
ICD-9 codes 240–279: endocrine, nutritional and metabolic diseases, and immunity disorders
ICD-9 codes 280–289: diseases of the blood and blood-forming organs
ICD-9 codes 290–319: mental disorders
ICD-9 codes 320–389: diseases of the nervous system and sense organs
ICD-9 codes 390–459: diseases of the circulatory system
ICD-9 codes 460–519: diseases of the respiratory system
ICD-9 codes 520–579: diseases of the digestive system
ICD-9 codes 580–629: diseases of the genitourinary system
ICD-9 codes 630–679: complications of pregnancy, childbirth, and the puerperium
ICD-9 codes 680–709: diseases of the skin and subcutaneous tissue
ICD-9 codes 710–739: diseases of the musculoskeletal system and connective tissue
ICD-9 codes 740–759: congenital anomalies
ICD-9 codes 760–779: certain conditions originating in the perinatal period
ICD-9 codes 780–799: symptoms, signs, and ill-defined conditions
ICD-9 codes 800–999: injury and poisoning
ICD-9 codes E and V codes: external causes of injury and supplemental classification

"""
icd9_diagnoses_colors = {
    'INFECTIOUS AND PARASITIC DISEASES (001-139)': "xkcd:bright pink",
    'NEOPLASMS (140-239)': "xkcd:dark blue",
    'ENDOCRINE, NUTRITIONAL AND METABOLIC DISEASES, AND IMMUNITY DISORDERS (240-279)': "xkcd:salmon",
    'DISEASES OF THE BLOOD AND BLOOD-FORMING ORGANS (280-289)': "xkcd:brick red",
    'MENTAL, BEHAVIORAL AND NEURODEVELOPMENTAL DISORDERS (290-319)': "xkcd:indigo",
    'DISEASES OF THE NERVOUS SYSTEM AND SENSE ORGANS (320-389)': "xkcd:electric blue",
    'DISEASES OF THE CIRCULATORY SYSTEM (390-459)': "xkcd:seafoam",
    'DISEASES OF THE RESPIRATORY SYSTEM (460-519)': "xkcd:grey blue",
    'DISEASES OF THE DIGESTIVE SYSTEM (520-579)': "xkcd:sienna",
    'DISEASES OF THE GENITOURINARY SYSTEM (580-629)': "xkcd:golden yellow",
    'COMPLICATIONS OF PREGNANCY, CHILDBIRTH, AND THE PUERPERIUM (630-679)': "xkcd:purplish",
    'DISEASES OF THE SKIN AND SUBCUTANEOUS TISSUE (680-709)': "xkcd:sage green",
    'DISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE (710-739)': "xkcd:avocado",
    'CONGENITAL ANOMALIES (740-759)': "xkcd:charcoal",
    'CERTAIN CONDITIONS ORIGINATING IN THE PERINATAL PERIOD (760-779)': "xkcd:yellowish",
    'SYMPTOMS, SIGNS, AND ILL-DEFINED CONDITIONS (780-799)': "xkcd:hot purple",
    'INJURY AND POISONING (800-999)': "xkcd:dark lilac",
    'SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS & CONTACT WITH HEALTH SERVICES (V01-V91)':
        "xkcd:bright teal",
    'SUPPLEMENTARY CLASSIFICATION OF EXTERNAL CAUSES OF INJURY AND POISONING (E000-E999)': "xkcd:dark pink"
}

icd9_diagnoses_short_names = {
    'INFECTIOUS AND PARASITIC DISEASES (001-139)': "infectious",
    'NEOPLASMS (140-239)': "neoplasms",
    'ENDOCRINE, NUTRITIONAL AND METABOLIC DISEASES, AND IMMUNITY DISORDERS (240-279)': "metabolic",
    'DISEASES OF THE BLOOD AND BLOOD-FORMING ORGANS (280-289)': "blood",
    'MENTAL, BEHAVIORAL AND NEURODEVELOPMENTAL DISORDERS (290-319)': "mental",
    'DISEASES OF THE NERVOUS SYSTEM AND SENSE ORGANS (320-389)': "nervous",
    'DISEASES OF THE CIRCULATORY SYSTEM (390-459)': "heart",
    'DISEASES OF THE RESPIRATORY SYSTEM (460-519)': "lung",
    'DISEASES OF THE DIGESTIVE SYSTEM (520-579)': "digestive",
    'DISEASES OF THE GENITOURINARY SYSTEM (580-629)': "urinary",
    'COMPLICATIONS OF PREGNANCY, CHILDBIRTH, AND THE PUERPERIUM (630-679)': "pregnancy",
    'DISEASES OF THE SKIN AND SUBCUTANEOUS TISSUE (680-709)': "skin",
    'DISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE (710-739)': "muscles",
    'CONGENITAL ANOMALIES (740-759)': "congenital",
    'CERTAIN CONDITIONS ORIGINATING IN THE PERINATAL PERIOD (760-779)': "perinatal",
    'SYMPTOMS, SIGNS, AND ILL-DEFINED CONDITIONS (780-799)': "ill-defined",
    'INJURY AND POISONING (800-999)': "injury",
    'SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS & CONTACT WITH HEALTH SERVICES (V01-V91)':
        "add. factors",
    'SUPPLEMENTARY CLASSIFICATION OF EXTERNAL CAUSES OF INJURY AND POISONING (E000-E999)': "add. injury"
}


"""
Procedures: Info from https://en.wikipedia.org/wiki/ICD-9-CM_Volume_3

(00) Procedures and interventions, not elsewhere classified
(01–05) Operations on the nervous system
(06–07) Operations on the endocrine system
(08–16) Operations on the eye
(18–20) Operations on the ear
(21-29) Operations on the nose, mouth and pharynx
(30–34) Operations on the respiratory system
(35–39) Operations on the cardiovascular system
(40–41) Operations on the hemic and lymphatic system
(42–54) Operations on the digestive system
(55–59) Operations on the urinary system
(60–64) Operations on the male genital organs
(65–71) Operations on the female genital organs
(72–75) Obstetrical procedures
(76–84) Operations on the musculoskeletal system
(85–86) Operations on the integumentary system
(87–99) Miscellaneous diagnostic and therapeutic procedures

"""
icd9_procedures_colors = {
    'PROCEDURES AND INTERVENTIONS , NOT ELSEWHERE CLASSIFIED (00)': "xkcd:hot purple",
    'OPERATIONS ON THE NERVOUS SYSTEM (01-05)': "xkcd:electric blue",
    'OPERATIONS ON THE ENDOCRINE SYSTEM (06-07)': "xkcd:salmon",
    'OPERATIONS ON THE EYE (08-16)': "xkcd:kelly green",
    'OTHER MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES (17)': "xkcd:grey blue",
    'OPERATIONS ON THE EAR (18-20)': "xkcd:aquamarine",
    'OPERATIONS ON THE NOSE, MOUTH, AND PHARYNX (21-29)': "xkcd:grey blue",
    'OPERATIONS ON THE RESPIRATORY SYSTEM (30-34)': "xkcd:brick red",
    'OPERATIONS ON THE CARDIOVASCULAR SYSTEM (35-39)': "xkcd:pale purple",
    'OPERATIONS ON THE HEMIC AND LYMPHATIC SYSTEM (40-41)': "xkcd:sienna",
    'OPERATIONS ON THE DIGESTIVE SYSTEM (42-54)': "xkcd:golden yellow",
    'OPERATIONS ON THE URINARY SYSTEM (55-59)': "xkcd:purplish",
    'OPERATIONS ON THE MALE GENITAL ORGANS (60-64)': "xkcd:baby blue",
    'OPERATIONS ON THE FEMALE GENITAL ORGANS (65-71)': "xkcd:light pink",
    'OBSTETRICAL PROCEDURES (72-75)': "xkcd:avocado",
    'OPERATIONS ON THE MUSCULOSKELETAL SYSTEM (76-84)': "xkcd:sage green",
    'OPERATIONS ON THE INTEGUMENTARY SYSTEM (85-86)': "xkcd:cyan",
    'MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES (87-99)': "xkcd:dusty rose"
}

icd9_procedures_short_names = {
    'PROCEDURES AND INTERVENTIONS , NOT ELSEWHERE CLASSIFIED (00)': "procedure",
    'OPERATIONS ON THE NERVOUS SYSTEM (01-05)': "nervous",
    'OPERATIONS ON THE ENDOCRINE SYSTEM (06-07)': "endocrine",
    'OPERATIONS ON THE EYE (08-16)': "eye",
    'OTHER MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES (17)': "misc.",
    'OPERATIONS ON THE EAR (18-20)': "ear",
    'OPERATIONS ON THE NOSE, MOUTH, AND PHARYNX (21-29)': "mouth",
    'OPERATIONS ON THE RESPIRATORY SYSTEM (30-34)': "lung",
    'OPERATIONS ON THE CARDIOVASCULAR SYSTEM (35-39)': "heart",
    'OPERATIONS ON THE HEMIC AND LYMPHATIC SYSTEM (40-41)': "lymphatic",
    'OPERATIONS ON THE DIGESTIVE SYSTEM (42-54)': "digestive",
    'OPERATIONS ON THE URINARY SYSTEM (55-59)': "urinary",
    'OPERATIONS ON THE MALE GENITAL ORGANS (60-64)': "male genitals",
    'OPERATIONS ON THE FEMALE GENITAL ORGANS (65-71)': "female genitals",
    'OBSTETRICAL PROCEDURES (72-75)': "obstetrics",
    'OPERATIONS ON THE MUSCULOSKELETAL SYSTEM (76-84)': "muscles",
    'OPERATIONS ON THE INTEGUMENTARY SYSTEM (85-86)': "skin",
    'MISCELLANEOUS DIAGNOSTIC AND THERAPEUTIC PROCEDURES (87-99)': "diagnostics"
}


# Cache ICD trees for diagnoses and procedures
_icd_files_path = os.path.realpath(os.path.join(os.path.split(__file__)[0], "icd_codes"))
_cached_nodes = {}
_cached_icd_trees = {}

# Cache string names given to ICD nodes (codes)
_cci_path = os.path.realpath(os.path.join(os.path.split(__file__)[0], "cci2015.csv"))
_cached_cci_table = None  # DataFrame containing chronic condition indicator (CCI) information
_cached_node_names = {}

# Two ICD kinds: diagnoses and procedures
icd_kind_keys = ['icd9_code_diagnoses', 'icd9_code_procedures']


def icd_tree_search(icd_kind, icd_code):
    """

    :param icd_kind:
    :param icd_code:
    :return:
    """

    # Exit if ICD kind unknown
    global icd_kind_keys
    if icd_kind not in icd_kind_keys:
        return None

    # Try to load code from a cached search result
    global _cached_nodes

    if icd_kind not in _cached_nodes:
        _cached_nodes[icd_kind] = {}

    # Only search for the code if the node is not cached
    if icd_code not in _cached_nodes[icd_kind]:
        _cached_nodes[icd_kind][icd_code] = _icd_tree_search_inner(icd_kind=icd_kind, icd_code=icd_code)

    return _cached_nodes[icd_kind][icd_code]


def _find_out_code(tree_node):
    # Return code if it is already known for node
    if "code" not in dir(tree_node):

        # If the node has a concrete code (instead of a range of codes), its name will be the code, then a space,
        # then a textual description.
        name_tokens = tree_node.name.split(" ")

        if name_tokens[0][0] in (string.digits + "V" + "E") and name_tokens[0][1] in string.digits:
            new_code = name_tokens[0]
        else:
            new_code = None  # this is a range of codes
        tree_node.code = new_code

    if tree_node.code is None:
        tree_node.code = tree_node.name

    return tree_node.code


def _load_from_disk(icd_kind):
    global _cached_icd_trees
    _cached_icd_trees[icd_kind] = _load_icd_file(
        os.path.join(_icd_files_path, f"{icd_kind.split('_')[-1]}.pkl")
    )


def _load_icd_file(path):
    # Load file
    with open(path, "rb") as f:
        icd_file = pickle.load(f)

    logging.info(f"Loaded {len(icd_file['root'].leaves)} codes ('{icd_file['root'].name}')"
                 f" from {path}")
    return icd_file['root']


def _icd_tree_search_inner(icd_kind, icd_code):
    # Load ICD tree from disk if not already done
    global _cached_icd_trees
    if icd_kind not in _cached_icd_trees:
        _load_from_disk(icd_kind)
    all_nodes = _cached_icd_trees[icd_kind].descendants

    # Search the tree for the code
    node = None
    found = False
    while not found:

        # Search nodes for code
        for node in all_nodes:

            # Find out code of the node (if it is not just a range of codes)
            code = _find_out_code(node)
            if code is None:
                continue

            # If we found the
            if code.replace(".", "") == icd_code:
                found = True
                break

        # Exit loop if found
        if found:
            break

        # If node was not found, maybe it belongs to a deleted ICD code no longer available in the current ICD tree.
        # -> Delete the last digit of the code to maybe find an ancestor of the code
        icd_code = icd_code[:-1]
        if len(icd_code) <= 2:
            return None

    # Return node
    return node


def icd_categ_level_1(icd_kind, icd_code, short_name=False):
    node = icd_tree_search(icd_kind, icd_code)
    if node is None:
        return None
    level_1_node = node.path[1]
    level_1_categ = name_for_node(level_1_node)

    if short_name:
        if icd_kind == "icd9_code_diagnoses":
            short_dict = icd9_diagnoses_short_names
        else:
            short_dict = icd9_procedures_short_names
        level_1_categ = short_dict[level_1_categ]

    return level_1_categ


def icd_color_for_code(icd_kind, icd_code):
    # Color is based solely on level 1 category
    categ = icd_categ_level_1(icd_kind, icd_code)
    if categ is None:
        return None

    # Choose correct color dictionary for ICD kind
    if icd_kind == "icd9_code_diagnoses":
        colors = icd9_diagnoses_colors
    elif icd_kind == "icd9_code_procedures":
        colors = icd9_procedures_colors
    else:
        assert False, f"ICD kind {icd_kind} does not have colors defined!"

    if categ not in colors:
        assert False, f"ICD categ {categ} of kind {icd_kind} does not have a color!"

    return colors[categ]


def name_for_node(icd_node) -> str:
    # Load node name from cache, if possible
    global _cached_node_names
    code_name = icd_node.name
    if code_name in _cached_node_names:
        return _cached_node_names[code_name]

    # Determine chronic or acute marker w.r.t. CCI (Chronic Condition Indicator)
    # (only if the node is a diagnosis code)
    use_cci = False  # if set to True, file must be available
    if use_cci:

        # Find out ICD code for node
        icd_code = _find_out_code(icd_node)

        diagnosis_categ = None
        if icd_code is not None and "diagnostic" in icd_node.root.name.lower():

            # Declare use of global variables
            global _cached_cci_table
            global _cci_path

            # Load CCI CSV file if not already done
            if _cached_cci_table is None:
                if os.path.isfile(_cci_path):
                    _cached_cci_table = pd.read_csv(_cci_path, skiprows=1)
                else:
                    assert False,\
                        f"Chronic Indicator File (cci2015.csv) not available at {_cci_path}. " \
                        f"Download the file from https://www.hcup-us.ahrq.gov/toolssoftware/chronic/chronic.jsp"
            cci = _cached_cci_table

            # Find row of node in CCI table to find out if it is a chronic or an acute diagnosis
            cci_code_str = "'" + icd_code.replace(".", "").ljust(5) + "'"
            code_col = "'ICD-9-CM CODE'"
            code_row = cci.loc[cci[code_col] == cci_code_str]

            # Translate to chronic or acute diagnoses
            if len(code_row) > 0:
                is_chronic = bool(code_row["'CATEGORY DESCRIPTION'"].item().replace("'", ""))
                # (0 - non-chronic condition or 1 - chronic condition)

                diagnosis_categ = {
                    False: "acute",
                    True: "chronic"
                }[is_chronic]

        # Add diagnosis category (chronic or acute) to name if it is not already part of the name
        if diagnosis_categ is not None and diagnosis_categ not in code_name.lower():
            code_name += f" ({diagnosis_categ[0].upper()})"

    # Store code name in cache
    _cached_node_names[code_name] = code_name

    return code_name


def name_for_code(icd_kind, icd_code):
    node = icd_tree_search(icd_kind=icd_kind, icd_code=icd_code)
    if node is None:
        return icd_kind.upper() + "__" + icd_code
    else:
        return name_for_node(icd_node=node)
