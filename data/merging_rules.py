mimic_merging = {
    'Glucose': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['51478'],  # urine
                ['50809',
                 '50931',
                 '1529']  # blood
            ]
        }
    },
    'O2 Flow': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['50815'],  # blood
                ['223834']
            ]
        }
    },
    'pH': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['50820'],  # blood
                ['51491',  # urine
                 '51094'],
                ['50831']  # other body fluid
            ]
        }
    },
    'RBC': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['51493'],  # urine
                ['833']  # blood
            ]
        }
    },
    'WBC': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['51516'],  # urine
                ['220546',  # blood
                 '1542']
            ]
        }
    },
    'Atypical Lymphocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Bands': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['51144',  # blood
                 '3738'],
                ['51344'],  # cerebrospinal fluid
                ['51111'],  # ascites
                ['51441'],  # pleural
                ['51386'],  # other body fluid
                ['51366']  # joint fluid
            ]
        }
    },
    'Basophils': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
            # (1/6 is blood)
        }
    },
    'Blasts': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Eosinophils': {
        'strategy': 'split',
        'opts': {
            'split': [
                ['51200',  # blood
                 '3754'],
                ['51347'],  # cerebrospinal fluid
                ['51114'],  # ascites
                ['51444'],  # pleural
                ['51419'],  # other body fluid
                ['51368']  # joint fluid
            ]
        }
    },
    'Lymphocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
            # (1/5 is blood)
        }
    },
    'Macrophage': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Mesothelial cells': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Mesothelial Cells': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Metamyelocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Monocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
            # (1/4 is blood)
        }
    },
    'Monos': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Myelocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Other': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Plasma Cells': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Polys': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Promyelocytes': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'NRBC': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'QTc': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns exhibit vastly different distributions
        }
    },
    'Hypersegmented Neutrophils': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'CD4/CD8 Ratio': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'CD19': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Plasma': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Young': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    },
    'Young Cells': {
        'strategy': 'split',
        'opts': {
            'split': []  # empty list means all columns with this label are slit into separate attributes
            # reason: columns are all from different fluids
        }
    }
}

mimic_merging_simple = [
    'Albumin',
    'Alkaline Phosphatase',
    'ALT',
    'Ammonia',
    'Amylase',
    'Arterial Base Excess',
    'AST',
    'Base Excess',
    'Bladder Pressure',
    'BUN',
    'Chloride',
    'Cholesterol',
    'Cortisol',
    'Creatinine',
    'Current Goal',
    'Cyclosporin',
    'Daily Weight',
    'Differential-Bands',
    'Differential-Basos',
    'Differential-Eos',
    'Differential-Lymphs',
    'Differential-Monos',
    'Digoxin',
    'Fibrinogen',
    'FK506',
    'Heart Rate',
    'Hematocrit',
    'Hemoglobin',
    'INR',
    'Ionized Calcium',
    'Lactic Acid',
    'LDH',
    'Lymphs',
    'Magnesium',
    'Mean Airway Pressure',
    'Mixed Venous O2% Sat',
    'pCO2',
    'Peak Insp. Pressure',
    'PEEP',
    'Phosphorous',
    'Plateau Pressure',
    'Platelet Count',
    'pO2',
    'Potassium',
    'PT',
    'PTT',
    'Respiratory Rate',
    'Sed Rate',
    'Serum Osmolality',
    'Sodium',
    'SvO2',
    'Theophylline',
    'Thrombin',
    'Total Protein',
    'Triglyceride',
    'Uric Acid',
    'Phenobarbital',
    'Ethanol',
    'D-Dimer',
    'Procan',
    'Procan Napa',
    'Rapamycin',
    'Salicylate',
    'Tidal Volume',
    'ART Lumen Volume',
    'Assisted Systole',
    'Augmented Diastole',
    'BAEDP',
    'Baseline Current/mA',
    'CVP Alarm [High]',
    'CVP Alarm [Low]',
    'HR Alarm [High]',
    'HR Alarm [Low]',
    'IABP Mean',
    'Inspiratory Time',
    'Minute Volume',
    'PCWP',
    'Total PEEP Level',
    'VEN Lumen Volume',
    'Lipase',
    'Orthostatic HR lying',
    'Bleeding Time',
    'Cuff Pressure',
    'Doppler BP',
    'ECMO',
    'Haptoglobin',
    'Apnea Interval',
    'Nitric Oxide',
    'Sodium, Body Fluid'
]
