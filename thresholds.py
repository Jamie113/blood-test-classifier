# thresholds.py
# Medical classification thresholds — edit here to update all downstream logic.
# Format: test_name → {"normal": (lo, hi), "borderline": (lo, hi), "unit": str}
# Values outside both ranges are classified as "Abnormal".

THRESHOLDS = {
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
    "HbA1C": {
        "normal":     (20, 38),
        "borderline": (38, 45),
        "unit":       "mmol/mol",
    },
    "Haematocrit (HCT)": {
        "normal":     (36, 47),
        "borderline": (47, 50),
        "unit":       "%",
    },
    "Haemoglobin": {
        "normal":     (130, 160),
        "borderline": (160, 170),
        "unit":       "g/L",
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
