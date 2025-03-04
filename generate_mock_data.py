import pandas as pd
import random
from faker import Faker

fake = Faker()

# Possible test types and normal ranges (example values) need to add full B2 results 
blood_tests = [
    {"name": "Basophil Count", "min": 0.01, "max": 0.3, "unit": "10⁹/L"},
    {"name": "Eosinophil Count", "min": 0.02, "max": 0.5, "unit": "10⁹/L"},
    {"name": "HbA1C", "min": 20, "max": 45, "unit": "mmol/mol"},
    {"name": "Haematocrit (HCT)", "min": 35, "max": 50, "unit": "%"},
    {"name": "Haemoglobin", "min": 120, "max": 170, "unit": "g/L"},
]

# Generate mock records
data = []
for _ in range(500):
    patient_id = fake.uuid4()[:8]  # Unique patient ID
    for test in blood_tests:
        value = round(random.uniform(test["min"], test["max"]), 2)
        data.append({"Patient ID": patient_id, "Test Name": test["name"], "Value": value, "Units": test["unit"]})

# Convert and save as CSV
df = pd.DataFrame(data)
df.to_csv("mock_blood_tests.csv", index=False)

print("Mock blood test data generated: 'mock_blood_tests.csv'")
