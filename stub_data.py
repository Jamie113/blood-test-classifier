# stub_data.py
# Generates realistic synthetic blood test data for demo purposes.
# Two patient subgroups with different hormonal/metabolic profiles to
# produce meaningful clusters in both per-marker and population views.
#
# Returns a long-format DataFrame matching the output of parse_upload().

import numpy as np
import pandas as pd
from thresholds import THRESHOLDS

# ── Per-marker Gaussian parameters ────────────────────────────────────────────
# Each marker has two subgroups (A and B) with different means.
# (mean_A, std_A, mean_B, std_B) — all in canonical units.
# Group A (55 patients): typical metabolic profile
# Group B (25 patients): elevated metabolic / lower hormonal profile

_MARKER_PARAMS = {
    # Liver / Metabolic
    "Albumin":                   (44.0, 2.5,   40.0, 2.5),
    "ALP":                       (62.0, 14.0,  95.0, 18.0),
    "ALT":                       (25.0, 8.0,   48.0, 10.0),
    "GGT":                       (22.0, 7.0,   54.0, 10.0),
    # Kidney
    "eGFR":                      (88.0, 10.0,  68.0, 9.0),
    # Iron / Blood
    "Ferritin":                  (120.0, 40.0, 280.0, 60.0),
    "Haemoglobin":               (152.0, 7.0,  142.0, 7.0),
    "Haematocrit (HCT)":         (45.0, 2.0,   42.0, 2.0),
    "Platelet Count":             (240.0, 40.0, 310.0, 45.0),
    # Glucose / Diabetes
    "HbA1C":                     (32.0, 3.5,   41.0, 3.0),
    # Lipids
    "Total Cholesterol":          (4.2, 0.5,    5.5, 0.5),
    "LDL Cholesterol":            (2.3, 0.4,    3.4, 0.4),
    "HDL Cholesterol":            (1.6, 0.2,    1.1, 0.2),
    "Total Cholesterol:HDL Ratio":(2.8, 0.4,    5.2, 0.5),
    # Thyroid
    "TSH":                        (1.8, 0.6,    3.2, 0.8),
    "Free T4":                    (16.0, 1.5,   14.5, 1.5),
    # Hormones
    "Testosterone":               (20.0, 4.0,   12.0, 3.0),
    "Free Testosterone":          (0.45, 0.07,  0.22, 0.05),
    "SHBG":                       (28.0, 6.0,   48.0, 7.0),
    "Oestradiol":                 (22.0, 5.0,   34.0, 6.0),
    "Prolactin":                  (180.0, 35.0, 290.0, 50.0),
    "FSH":                        (4.5, 1.5,    8.5, 2.0),
    "LH":                         (4.0, 1.2,    6.5, 1.5),
    "PSA":                        (0.8, 0.4,    2.2, 0.8),
    # Blood Cells
    "White Blood Cell Count":     (6.2, 1.0,    8.5, 1.2),
    "Neutrophil Count":           (3.8, 0.7,    5.5, 0.8),
    "Basophil Count":             (0.06, 0.02,  0.14, 0.03),
    "Eosinophil Count":           (0.15, 0.05,  0.28, 0.07),
}

_N_A   = 55   # typical group
_N_B   = 25   # elevated metabolic / lower hormonal group
_SEED  = 42


def generate_stub_data() -> pd.DataFrame:
    """
    Return a long-format DataFrame (patient_id, test_name, value, unit)
    with 80 synthetic patients across two realistic subgroups.
    Values are clipped to physiologically plausible minimums.
    """
    rng = np.random.default_rng(_SEED)
    rows = []

    for group_idx, (n, suffix, param_offset) in enumerate(
        [(_N_A, "A", 0), (_N_B, "B", 2)]
    ):
        for patient_num in range(1, n + 1):
            patient_id = f"DEMO-{suffix}{patient_num:03d}"
            for test_name, params in _MARKER_PARAMS.items():
                mean = params[param_offset]
                std  = params[param_offset + 1]
                value = float(rng.normal(mean, std))
                # Clip to a small positive floor to avoid nonsensical negatives
                value = max(value, mean * 0.1)
                rows.append({
                    "patient_id": patient_id,
                    "test_name":  test_name,
                    "value":      round(value, 4),
                    "unit":       THRESHOLDS[test_name]["unit"],
                })

    return pd.DataFrame(rows)
