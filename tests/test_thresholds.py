import pytest
from thresholds import classify_test, THRESHOLDS


# ── Every marker in THRESHOLDS should have required keys ──────────────────────

def test_all_thresholds_have_required_keys():
    for name, t in THRESHOLDS.items():
        assert "normal" in t,     f"{name} missing 'normal'"
        assert "borderline" in t, f"{name} missing 'borderline'"
        assert "unit" in t,       f"{name} missing 'unit'"
        assert len(t["normal"]) == 2,     f"{name} normal range must be (lo, hi)"
        assert len(t["borderline"]) == 2, f"{name} borderline range must be (lo, hi)"


def test_all_normal_ranges_are_valid():
    for name, t in THRESHOLDS.items():
        lo, hi = t["normal"]
        assert lo < hi, f"{name}: normal lo ({lo}) must be < hi ({hi})"


def test_all_borderline_ranges_are_valid():
    for name, t in THRESHOLDS.items():
        lo, hi = t["borderline"]
        assert lo < hi, f"{name}: borderline lo ({lo}) must be < hi ({hi})"


def test_unknown_test_raises_value_error():
    with pytest.raises(ValueError, match="Unknown test"):
        classify_test("Not A Real Test", 1.0)


# ── Spot-checks across all 27 markers ─────────────────────────────────────────
# Each tuple: (test_name, value, expected_category)

@pytest.mark.parametrize("test_name,value,expected", [
    # Albumin
    ("Albumin",             42.0,  "Normal"),
    ("Albumin",             32.0,  "Borderline"),
    ("Albumin",             25.0,  "Abnormal"),
    # ALP
    ("ALP",                 80.0,  "Normal"),
    ("ALP",                145.0,  "Borderline"),
    ("ALP",                200.0,  "Abnormal"),
    # ALT
    ("ALT",                 30.0,  "Normal"),
    ("ALT",                 60.0,  "Borderline"),
    ("ALT",                 80.0,  "Abnormal"),
    # GGT
    ("GGT",                 30.0,  "Normal"),
    ("GGT",                 70.0,  "Borderline"),
    ("GGT",                100.0,  "Abnormal"),
    # eGFR (higher is better — normal ≥60)
    ("eGFR",                90.0,  "Normal"),
    ("eGFR",                52.0,  "Borderline"),
    ("eGFR",                30.0,  "Abnormal"),
    # Ferritin
    ("Ferritin",           150.0,  "Normal"),
    ("Ferritin",           430.0,  "Borderline"),
    ("Ferritin",           600.0,  "Abnormal"),
    ("Ferritin",            10.0,  "Abnormal"),   # too low
    # Haemoglobin (updated range: normal 130–170)
    ("Haemoglobin",        145.0,  "Normal"),
    ("Haemoglobin",        130.0,  "Normal"),
    ("Haemoglobin",        170.0,  "Normal"),
    ("Haemoglobin",        175.0,  "Borderline"),
    ("Haemoglobin",        185.0,  "Borderline"),
    ("Haemoglobin",        190.0,  "Abnormal"),
    ("Haemoglobin",        100.0,  "Abnormal"),   # too low
    # Haematocrit (updated range: normal 38.5–50, fraction input scaled ×100 before calling)
    ("Haematocrit (HCT)",   44.0,  "Normal"),
    ("Haematocrit (HCT)",   38.5,  "Normal"),
    ("Haematocrit (HCT)",   50.0,  "Normal"),
    ("Haematocrit (HCT)",   52.0,  "Borderline"),
    ("Haematocrit (HCT)",   35.0,  "Abnormal"),   # too low
    ("Haematocrit (HCT)",   56.0,  "Abnormal"),   # too high
    # Platelet Count
    ("Platelet Count",     250.0,  "Normal"),
    ("Platelet Count",     420.0,  "Borderline"),
    ("Platelet Count",     500.0,  "Abnormal"),
    ("Platelet Count",     100.0,  "Abnormal"),   # too low
    # HbA1C (updated range: normal 20–42, borderline 42–47)
    ("HbA1C",               30.0,  "Normal"),
    ("HbA1C",               42.0,  "Normal"),
    ("HbA1C",               45.0,  "Borderline"),
    ("HbA1C",               55.0,  "Abnormal"),
    ("HbA1C",               10.0,  "Abnormal"),   # too low
    # Total Cholesterol
    ("Total Cholesterol",    4.5,  "Normal"),
    ("Total Cholesterol",    5.5,  "Borderline"),
    ("Total Cholesterol",    7.0,  "Abnormal"),
    # LDL Cholesterol
    ("LDL Cholesterol",      2.5,  "Normal"),
    ("LDL Cholesterol",      3.5,  "Borderline"),
    ("LDL Cholesterol",      5.0,  "Abnormal"),
    # HDL Cholesterol (higher is better — normal ≥1.0)
    ("HDL Cholesterol",      1.5,  "Normal"),
    ("HDL Cholesterol",      0.95, "Borderline"),
    ("HDL Cholesterol",      0.7,  "Abnormal"),
    # Total Cholesterol:HDL Ratio
    ("Total Cholesterol:HDL Ratio", 4.0, "Normal"),
    ("Total Cholesterol:HDL Ratio", 5.5, "Borderline"),
    ("Total Cholesterol:HDL Ratio", 7.0, "Abnormal"),
    # TSH
    ("TSH",                  2.0,  "Normal"),
    ("TSH",                  6.0,  "Borderline"),
    ("TSH",                 12.0,  "Abnormal"),
    ("TSH",                  0.1,  "Abnormal"),   # suppressed
    # Free T4
    ("Free T4",             16.0,  "Normal"),
    ("Free T4",             23.0,  "Borderline"),
    ("Free T4",             28.0,  "Abnormal"),
    ("Free T4",             10.0,  "Abnormal"),   # too low
    # Testosterone
    ("Testosterone",        18.0,  "Normal"),
    ("Testosterone",         9.0,  "Borderline"),
    ("Testosterone",         5.0,  "Abnormal"),
    ("Testosterone",        40.0,  "Abnormal"),   # too high
    # Free Testosterone
    ("Free Testosterone",   0.35,  "Normal"),
    ("Free Testosterone",   0.13,  "Borderline"),
    ("Free Testosterone",   0.05,  "Abnormal"),
    # SHBG
    ("SHBG",                35.0,  "Normal"),
    ("SHBG",                60.0,  "Borderline"),
    ("SHBG",                80.0,  "Abnormal"),
    ("SHBG",                10.0,  "Abnormal"),   # too low
    # Oestradiol
    ("Oestradiol",          25.0,  "Normal"),
    ("Oestradiol",          50.0,  "Borderline"),
    ("Oestradiol",          80.0,  "Abnormal"),
    # Prolactin
    ("Prolactin",          200.0,  "Normal"),
    ("Prolactin",          400.0,  "Borderline"),
    ("Prolactin",          600.0,  "Abnormal"),
    ("Prolactin",           50.0,  "Abnormal"),   # too low
    # FSH
    ("FSH",                  5.0,  "Normal"),
    ("FSH",                 15.0,  "Borderline"),
    ("FSH",                 25.0,  "Abnormal"),
    # LH
    ("LH",                   4.0,  "Normal"),
    ("LH",                  10.0,  "Borderline"),
    ("LH",                  15.0,  "Abnormal"),
    ("LH",                   0.5,  "Abnormal"),   # too low
    # PSA
    ("PSA",                  1.0,  "Normal"),
    ("PSA",                  6.0,  "Borderline"),
    ("PSA",                 12.0,  "Abnormal"),
    # White Blood Cell Count
    ("White Blood Cell Count", 6.0, "Normal"),
    ("White Blood Cell Count",12.0, "Borderline"),
    ("White Blood Cell Count",16.0, "Abnormal"),
    ("White Blood Cell Count", 2.0, "Abnormal"),  # too low
    # Neutrophil Count
    ("Neutrophil Count",     4.0,  "Normal"),
    ("Neutrophil Count",     8.0,  "Borderline"),
    ("Neutrophil Count",    10.0,  "Abnormal"),
    # Basophil Count
    ("Basophil Count",      0.10,  "Normal"),
    ("Basophil Count",      0.25,  "Borderline"),
    ("Basophil Count",      0.40,  "Abnormal"),
    # Eosinophil Count
    ("Eosinophil Count",    0.20,  "Normal"),
    ("Eosinophil Count",    0.45,  "Borderline"),
    ("Eosinophil Count",    0.65,  "Abnormal"),
])
def test_classify(test_name, value, expected):
    assert classify_test(test_name, value) == expected, (
        f"{test_name}={value}: expected {expected}, got {classify_test(test_name, value)}"
    )
