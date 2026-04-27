# column_map.py
# Maps CSV column headers (from real blood test exports) to classifier test names.
#
# "test"  — must match a key in thresholds.THRESHOLDS
# "scale" — multiply raw CSV value by this before classifying
#            (e.g. Haematocrit is stored as a fraction 0–1, threshold expects %)
#
# Unit auto-detection (e.g. nmol/L vs ng/dL) is handled separately by
# unit_conversions.to_canonical(), which is applied after scaling.

COLUMN_MAP = {
    # ── Liver / Metabolic ─────────────────────────────────────────────────────
    "Blood Test Info Albumin Levels":    {"test": "Albumin",       "scale": 1},
    "Blood Test Info ALP Levels":        {"test": "ALP",           "scale": 1},
    "Blood Test Info ALT Levels":        {"test": "ALT",           "scale": 1},
    "Blood Test Info GGT Levels":        {"test": "GGT",           "scale": 1},

    # ── Kidney ────────────────────────────────────────────────────────────────
    "Blood Test Info eGFR Rate":         {"test": "eGFR",          "scale": 1},

    # ── Iron / Blood ──────────────────────────────────────────────────────────
    "Blood Test Info Ferritin Levels":   {"test": "Ferritin",      "scale": 1},
    "Blood Test Info Haemoglobin Levels":{"test": "Haemoglobin",   "scale": 1},
    "Blood Test Info Haematocrit Levels":{"test": "Haematocrit (HCT)", "scale": 100},
    # Haematocrit ADJ is an altitude-adjusted variant — mapped to same thresholds
    "Blood Test Info Haematocrit ADJ Levels": {"test": "Haematocrit (HCT)", "scale": 100},
    "Blood Test Info Platelet Count":    {"test": "Platelet Count","scale": 1},

    # ── Glucose / Diabetes ────────────────────────────────────────────────────
    "Blood Test Info HBA1C Levels":      {"test": "HbA1C",         "scale": 1},

    # ── Lipids ────────────────────────────────────────────────────────────────
    "Blood Test Info Total Cholesterol Levels":       {"test": "Total Cholesterol",        "scale": 1},
    "Blood Test Info LDL Cholesterol Levels":         {"test": "LDL Cholesterol",          "scale": 1},
    "Blood Test Info HDL Cholesterol Levels":         {"test": "HDL Cholesterol",          "scale": 1},
    "Blood Test Info Total Cholesterol : HDL Ratio":  {"test": "Total Cholesterol:HDL Ratio", "scale": 1},

    # ── Thyroid ───────────────────────────────────────────────────────────────
    "Blood Test Info TSH Levels":        {"test": "TSH",           "scale": 1},
    "Blood Test Info Free T4 Levels":    {"test": "Free T4",       "scale": 1},

    # ── Hormones ──────────────────────────────────────────────────────────────
    "Blood Test Info Testosterone Levels":            {"test": "Testosterone",      "scale": 1},
    "Blood Test Info Free Testosterone Levels":       {"test": "Free Testosterone", "scale": 1},
    "Blood Test Info SHBG Levels":                    {"test": "SHBG",              "scale": 1},
    "Blood Test Info Oestradiol Levels":              {"test": "Oestradiol",        "scale": 1},
    "Blood Test Info Prolactin Levels":               {"test": "Prolactin",         "scale": 1},
    "Blood Test Info Folicle Stimulating Hormone Levels": {"test": "FSH",           "scale": 1},
    "Blood Test Info Lutenising Hormone Levels":      {"test": "LH",                "scale": 1},
    "Blood Test Info PSA Levels":                     {"test": "PSA",               "scale": 1},

    # ── Blood Cells ───────────────────────────────────────────────────────────
    "Blood Test Info White Blood Cell Count":         {"test": "White Blood Cell Count", "scale": 1},
    "Blood Test Info Neutrophil Count":               {"test": "Neutrophil Count",       "scale": 1},
}

ID_COLUMN  = "Blood Test Info Blood Test ID"
AGE_COLUMN = "Current Age"
