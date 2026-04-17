import pytest
from column_map import COLUMN_MAP, ID_COLUMN
from thresholds import THRESHOLDS


def test_all_mapped_tests_exist_in_thresholds():
    for col, mapping in COLUMN_MAP.items():
        assert mapping["test"] in THRESHOLDS, (
            f"Column '{col}' maps to '{mapping['test']}' which is not in THRESHOLDS"
        )


def test_all_scales_are_positive_numbers():
    for col, mapping in COLUMN_MAP.items():
        assert isinstance(mapping["scale"], (int, float)), f"Scale for '{col}' must be numeric"
        assert mapping["scale"] > 0, f"Scale for '{col}' must be positive"


def test_id_column_is_defined():
    assert ID_COLUMN is not None
    assert isinstance(ID_COLUMN, str)
    assert len(ID_COLUMN) > 0


def test_haematocrit_levels_scale_is_100():
    mapping = COLUMN_MAP.get("Blood Test Info Haematocrit Levels")
    assert mapping is not None, "Haematocrit Levels column missing from map"
    assert mapping["scale"] == 100


def test_haematocrit_adj_scale_is_100():
    mapping = COLUMN_MAP.get("Blood Test Info Haematocrit ADJ Levels")
    assert mapping is not None, "Haematocrit ADJ Levels column missing from map"
    assert mapping["scale"] == 100


def test_all_27_export_columns_are_mapped():
    """Verify every column from the real export is accounted for."""
    expected_columns = [
        "Blood Test Info Albumin Levels",
        "Blood Test Info ALP Levels",
        "Blood Test Info ALT Levels",
        "Blood Test Info eGFR Rate",
        "Blood Test Info Ferritin Levels",
        "Blood Test Info Folicle Stimulating Hormone Levels",
        "Blood Test Info Free T4 Levels",
        "Blood Test Info Free Testosterone Levels",
        "Blood Test Info GGT Levels",
        "Blood Test Info Haematocrit ADJ Levels",
        "Blood Test Info Haematocrit Levels",
        "Blood Test Info Haemoglobin Levels",
        "Blood Test Info HBA1C Levels",
        "Blood Test Info HDL Cholesterol Levels",
        "Blood Test Info Oestradiol Levels",
        "Blood Test Info Neutrophil Count",
        "Blood Test Info Lutenising Hormone Levels",
        "Blood Test Info LDL Cholesterol Levels",
        "Blood Test Info Platelet Count",
        "Blood Test Info Prolactin Levels",
        "Blood Test Info PSA Levels",
        "Blood Test Info SHBG Levels",
        "Blood Test Info Testosterone Levels",
        "Blood Test Info Total Cholesterol : HDL Ratio",
        "Blood Test Info Total Cholesterol Levels",
        "Blood Test Info White Blood Cell Count",
        "Blood Test Info TSH Levels",
    ]
    for col in expected_columns:
        assert col in COLUMN_MAP, f"Export column '{col}' is not in COLUMN_MAP"


def test_no_duplicate_test_mappings_except_haematocrit():
    """Only Haematocrit ADJ should share a test name with another column."""
    from collections import Counter
    test_counts = Counter(m["test"] for m in COLUMN_MAP.values())
    for test_name, count in test_counts.items():
        if test_name == "Haematocrit (HCT)":
            assert count == 2  # Levels + ADJ
        else:
            assert count == 1, f"Test '{test_name}' appears {count} times in COLUMN_MAP"
