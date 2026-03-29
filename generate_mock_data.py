import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(seed=42)

# Per-test population parameters: Normal (70%) / Borderline (20%) / Abnormal (10%)
# Each category is a Gaussian centred on that population's typical value.
# Abnormal values are placed clearly outside the borderline range.
TEST_POPULATIONS = {
    "Basophil Count": {
        "unit": "10⁹/L",
        "normal":     {"centre": 0.105, "std": 0.025, "weight": 0.70},
        "borderline": {"centre": 0.25,  "std": 0.015, "weight": 0.20},
        "abnormal":   {"centre": 0.40,  "std": 0.030, "weight": 0.10},
    },
    "Eosinophil Count": {
        "unit": "10⁹/L",
        "normal":     {"centre": 0.21,  "std": 0.05,  "weight": 0.70},
        "borderline": {"centre": 0.45,  "std": 0.015, "weight": 0.20},
        "abnormal":   {"centre": 0.65,  "std": 0.05,  "weight": 0.10},
    },
    "HbA1C": {
        "unit": "mmol/mol",
        "normal":     {"centre": 29,    "std": 2.5,   "weight": 0.70},
        "borderline": {"centre": 41.5,  "std": 1.2,   "weight": 0.20},
        "abnormal":   {"centre": 52,    "std": 2.5,   "weight": 0.10},
    },
    "Haematocrit (HCT)": {
        "unit": "%",
        "normal":     {"centre": 41.5,  "std": 2.0,   "weight": 0.70},
        "borderline": {"centre": 48.5,  "std": 0.5,   "weight": 0.20},
        "abnormal":   {"centre": 54,    "std": 1.5,   "weight": 0.10},
    },
    "Haemoglobin": {
        "unit": "g/L",
        "normal":     {"centre": 145,   "std": 5,     "weight": 0.70},
        "borderline": {"centre": 165,   "std": 1.5,   "weight": 0.20},
        "abnormal":   {"centre": 185,   "std": 5,     "weight": 0.10},
    },
}

N_PATIENTS = 500
categories = ["normal", "borderline", "abnormal"]

data = []
for _ in range(N_PATIENTS):
    patient_id = fake.uuid4()[:8]
    for test_name, cfg in TEST_POPULATIONS.items():
        weights = [cfg[c]["weight"] for c in categories]
        category = rng.choice(categories, p=weights)
        value = round(float(rng.normal(cfg[category]["centre"], cfg[category]["std"])), 2)
        data.append({
            "Patient ID": patient_id,
            "Test Name": test_name,
            "Value": value,
            "Units": cfg["unit"],
        })

df = pd.DataFrame(data)
df.to_csv("mock_blood_tests.csv", index=False)
print(f"Mock blood test data generated: 'mock_blood_tests.csv' ({len(df)} rows)")
