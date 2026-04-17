# thresholds.py
# Male reference ranges for blood test classification.
# All values are in canonical units (see unit_conversions.py for auto-detection).
#
# Format: test_name → {"normal": (lo, hi), "borderline": (lo, hi), "unit": str}
# Values outside both ranges are classified as "Abnormal".
#
# For tests where "lower is worse" (e.g. eGFR, HDL), the normal range has
# a high upper bound to capture all values above the threshold.
#
# Two-sided borderline tests (Ferritin, Platelet, Prolactin):
#   These have borderline zones on both low and high ends.
#   The structure captures the high-end borderline; values below the normal
#   lower bound fall through as Abnormal.

THRESHOLDS = {
    # ── Liver / Metabolic ─────────────────────────────────────────────────────
    "Albumin": {
        "normal":     (35.0, 50.0),
        "borderline": (30.0, 35.0),
        "unit":       "g/L",
    },
    "ALP": {
        "normal":     (30.0, 130.0),
        "borderline": (130.0, 160.0),
        "unit":       "U/L",
    },
    "ALT": {
        "normal":     (7.0, 56.0),
        "borderline": (56.0, 70.0),
        "unit":       "U/L",
    },
    "GGT": {
        "normal":     (8.0, 61.0),
        "borderline": (61.0, 80.0),
        "unit":       "U/L",
    },

    # ── Kidney ────────────────────────────────────────────────────────────────
    "eGFR": {
        "normal":     (60.0, 200.0),   # higher is better; 200 is a safe upper bound
        "borderline": (45.0, 60.0),
        "unit":       "mL/min/1.73m²",
    },

    # ── Iron / Blood ──────────────────────────────────────────────────────────
    "Ferritin": {
        "normal":     (30.0, 400.0),
        "borderline": (400.0, 500.0),
        "unit":       "µg/L",
    },
    "Haemoglobin": {
        "normal":     (130.0, 170.0),
        "borderline": (170.0, 185.0),
        "unit":       "g/L",
    },
    "Haematocrit (HCT)": {
        "normal":     (38.5, 50.0),
        "borderline": (50.0, 54.0),
        "unit":       "%",
    },
    "Platelet Count": {
        "normal":     (150.0, 400.0),
        "borderline": (400.0, 450.0),
        "unit":       "10⁹/L",
    },

    # ── Glucose / Diabetes ────────────────────────────────────────────────────
    "HbA1C": {
        "normal":     (20.0, 42.0),
        "borderline": (42.0, 47.0),
        "unit":       "mmol/mol",
    },

    # ── Lipids ────────────────────────────────────────────────────────────────
    "Total Cholesterol": {
        "normal":     (0.0, 5.0),
        "borderline": (5.0, 6.2),
        "unit":       "mmol/L",
    },
    "LDL Cholesterol": {
        "normal":     (0.0, 3.0),
        "borderline": (3.0, 4.0),
        "unit":       "mmol/L",
    },
    "HDL Cholesterol": {
        "normal":     (1.0, 10.0),   # higher is better; 10 is a safe upper bound
        "borderline": (0.9, 1.0),
        "unit":       "mmol/L",
    },
    "Total Cholesterol:HDL Ratio": {
        "normal":     (0.0, 5.0),
        "borderline": (5.0, 6.0),
        "unit":       "ratio",
    },

    # ── Thyroid ───────────────────────────────────────────────────────────────
    "TSH": {
        "normal":     (0.4, 4.0),
        "borderline": (4.0, 10.0),
        "unit":       "mIU/L",
    },
    "Free T4": {
        "normal":     (12.0, 22.0),
        "borderline": (22.0, 25.0),
        "unit":       "pmol/L",
    },

    # ── Hormones ──────────────────────────────────────────────────────────────
    "Testosterone": {
        "normal":     (10.0, 35.0),
        "borderline": (8.0, 10.0),
        "unit":       "nmol/L",
    },
    "Free Testosterone": {
        "normal":     (0.174, 0.729),
        "borderline": (0.100, 0.174),
        "unit":       "nmol/L",
    },
    "SHBG": {
        "normal":     (17.0, 55.0),
        "borderline": (55.0, 70.0),
        "unit":       "nmol/L",
    },
    "Oestradiol": {
        "normal":     (10.0, 40.0),
        "borderline": (40.0, 60.0),
        "unit":       "pg/mL",
    },
    "Prolactin": {
        "normal":     (86.0, 324.0),
        "borderline": (324.0, 530.0),
        "unit":       "mIU/L",
    },
    "FSH": {
        "normal":     (1.5, 12.4),
        "borderline": (12.4, 20.0),
        "unit":       "IU/L",
    },
    "LH": {
        "normal":     (1.7, 8.6),
        "borderline": (8.6, 12.0),
        "unit":       "IU/L",
    },
    "PSA": {
        "normal":     (0.0, 4.0),
        "borderline": (4.0, 10.0),
        "unit":       "ng/mL",
    },

    # ── Blood Cells ───────────────────────────────────────────────────────────
    "White Blood Cell Count": {
        "normal":     (4.0, 11.0),
        "borderline": (11.0, 14.0),
        "unit":       "10⁹/L",
    },
    "Neutrophil Count": {
        "normal":     (1.8, 7.5),
        "borderline": (7.5, 9.0),
        "unit":       "10⁹/L",
    },
    "Basophil Count": {
        "normal":     (0.01, 0.20),
        "borderline": (0.20, 0.30),
        "unit":       "10⁹/L",
    },
    "Eosinophil Count": {
        "normal":     (0.02, 0.40),
        "borderline": (0.40, 0.50),
        "unit":       "10⁹/L",
    },
}


def classify_test(test_name: str, value: float) -> str:
    """Return 'Normal', 'Borderline', or 'Abnormal' using THRESHOLDS."""
    if test_name not in THRESHOLDS:
        raise ValueError(f"Unknown test: '{test_name}'. Valid tests: {list(THRESHOLDS.keys())}")
    rules = THRESHOLDS[test_name]
    lo_n, hi_n = rules["normal"]
    lo_b, hi_b = rules["borderline"]
    if lo_n <= value <= hi_n:
        return "Normal"
    if lo_b <= value <= hi_b:
        return "Borderline"
    return "Abnormal"
