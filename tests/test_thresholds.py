import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thresholds import classify_test


class TestBasophilCount:
    def test_normal_interior(self):
        assert classify_test("Basophil Count", 0.10) == "Normal"

    def test_normal_lower_bound(self):
        assert classify_test("Basophil Count", 0.01) == "Normal"

    def test_normal_upper_bound(self):
        assert classify_test("Basophil Count", 0.20) == "Normal"

    def test_borderline_interior(self):
        assert classify_test("Basophil Count", 0.25) == "Borderline"

    def test_borderline_upper_bound(self):
        assert classify_test("Basophil Count", 0.30) == "Borderline"

    def test_abnormal_above(self):
        assert classify_test("Basophil Count", 0.40) == "Abnormal"

    def test_abnormal_below(self):
        assert classify_test("Basophil Count", 0.005) == "Abnormal"


class TestEosinophilCount:
    def test_normal_interior(self):
        assert classify_test("Eosinophil Count", 0.20) == "Normal"

    def test_normal_lower_bound(self):
        assert classify_test("Eosinophil Count", 0.02) == "Normal"

    def test_normal_upper_bound(self):
        assert classify_test("Eosinophil Count", 0.40) == "Normal"

    def test_borderline_interior(self):
        assert classify_test("Eosinophil Count", 0.45) == "Borderline"

    def test_borderline_upper_bound(self):
        assert classify_test("Eosinophil Count", 0.50) == "Borderline"

    def test_abnormal_above(self):
        assert classify_test("Eosinophil Count", 0.65) == "Abnormal"

    def test_abnormal_below(self):
        assert classify_test("Eosinophil Count", 0.01) == "Abnormal"


class TestHbA1C:
    def test_normal_interior(self):
        assert classify_test("HbA1C", 30) == "Normal"

    def test_normal_lower_bound(self):
        assert classify_test("HbA1C", 20) == "Normal"

    def test_normal_upper_bound(self):
        assert classify_test("HbA1C", 38) == "Normal"

    def test_borderline_interior(self):
        assert classify_test("HbA1C", 41) == "Borderline"

    def test_borderline_upper_bound(self):
        assert classify_test("HbA1C", 45) == "Borderline"

    def test_abnormal_high(self):
        assert classify_test("HbA1C", 52) == "Abnormal"

    def test_abnormal_low(self):
        assert classify_test("HbA1C", 10) == "Abnormal"


class TestHaematocritHCT:
    def test_normal_interior(self):
        assert classify_test("Haematocrit (HCT)", 42) == "Normal"

    def test_normal_lower_bound(self):
        assert classify_test("Haematocrit (HCT)", 36) == "Normal"

    def test_normal_upper_bound(self):
        assert classify_test("Haematocrit (HCT)", 47) == "Normal"

    def test_borderline_interior(self):
        assert classify_test("Haematocrit (HCT)", 48.5) == "Borderline"

    def test_borderline_upper_bound(self):
        assert classify_test("Haematocrit (HCT)", 50) == "Borderline"

    def test_abnormal_above(self):
        assert classify_test("Haematocrit (HCT)", 54) == "Abnormal"

    def test_abnormal_below(self):
        assert classify_test("Haematocrit (HCT)", 30) == "Abnormal"


class TestHaemoglobin:
    def test_normal_interior(self):
        assert classify_test("Haemoglobin", 145) == "Normal"

    def test_normal_lower_bound(self):
        assert classify_test("Haemoglobin", 130) == "Normal"

    def test_normal_upper_bound(self):
        assert classify_test("Haemoglobin", 160) == "Normal"

    def test_borderline_interior(self):
        assert classify_test("Haemoglobin", 165) == "Borderline"

    def test_borderline_upper_bound(self):
        assert classify_test("Haemoglobin", 170) == "Borderline"

    def test_abnormal_above(self):
        assert classify_test("Haemoglobin", 185) == "Abnormal"

    def test_abnormal_below(self):
        assert classify_test("Haemoglobin", 100) == "Abnormal"


class TestUnknownTest:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown test"):
            classify_test("Platelet Count", 200)

    def test_error_message_contains_test_name(self):
        with pytest.raises(ValueError, match="Platelet Count"):
            classify_test("Platelet Count", 200)
