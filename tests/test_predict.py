# Note: these tests require the full pipeline to have been run first:
#   python generate_mock_data.py → python database_setup.py → python train_classifier.py
# The model (blood_test_classifier.pkl) must exist before running these tests.
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predict import predict_blood_test


class TestHbA1C:
    def test_normal(self):
        assert predict_blood_test("HbA1C", 29) == "Normal"

    def test_borderline(self):
        assert predict_blood_test("HbA1C", 41) == "Borderline"

    def test_abnormal_high(self):
        assert predict_blood_test("HbA1C", 52) == "Abnormal"


class TestBasophilCount:
    def test_normal(self):
        assert predict_blood_test("Basophil Count", 0.10) == "Normal"

    def test_borderline(self):
        assert predict_blood_test("Basophil Count", 0.25) == "Borderline"

    def test_abnormal(self):
        assert predict_blood_test("Basophil Count", 0.40) == "Abnormal"


class TestEosinophilCount:
    def test_normal(self):
        assert predict_blood_test("Eosinophil Count", 0.20) == "Normal"

    def test_borderline(self):
        assert predict_blood_test("Eosinophil Count", 0.45) == "Borderline"

    def test_abnormal(self):
        assert predict_blood_test("Eosinophil Count", 0.65) == "Abnormal"


class TestHaematocritHCT:
    def test_normal(self):
        assert predict_blood_test("Haematocrit (HCT)", 42) == "Normal"

    def test_borderline(self):
        assert predict_blood_test("Haematocrit (HCT)", 48.5) == "Borderline"

    def test_abnormal(self):
        assert predict_blood_test("Haematocrit (HCT)", 54) == "Abnormal"


class TestHaemoglobin:
    def test_normal(self):
        assert predict_blood_test("Haemoglobin", 145) == "Normal"

    def test_borderline(self):
        assert predict_blood_test("Haemoglobin", 165) == "Borderline"

    def test_abnormal(self):
        assert predict_blood_test("Haemoglobin", 185) == "Abnormal"


class TestInvalidInput:
    def test_unknown_test_name(self):
        with pytest.raises(ValueError, match="Unknown test"):
            predict_blood_test("Platelet Count", 200)

    def test_non_numeric_value(self):
        with pytest.raises(TypeError):
            predict_blood_test("HbA1C", "high")

    def test_string_number_raises(self):
        with pytest.raises(TypeError):
            predict_blood_test("HbA1C", "42")
