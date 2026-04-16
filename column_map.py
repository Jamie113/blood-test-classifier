# column_map.py
# Maps CSV column headers (from real blood test exports) to classifier test names.
#
# "test"  — must match a key in thresholds.THRESHOLDS
# "scale" — multiply raw CSV value by this before classifying
#            (e.g. Haematocrit is stored as a fraction 0–1, threshold expects %)

COLUMN_MAP = {
    "Blood Test Info Haemoglobin Levels":  {"test": "Haemoglobin",       "scale": 1},
    "Blood Test Info HBA1C Levels":        {"test": "HbA1C",             "scale": 1},
    "Blood Test Info Haematocrit Levels":  {"test": "Haematocrit (HCT)", "scale": 100},
}

ID_COLUMN = "Blood Test Info Blood Test ID"
