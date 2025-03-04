import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

DATABASE_URL = "sqlite:///blood_tests.db"
engine = create_engine(DATABASE_URL)

# Load from SQLite database
df = pd.read_sql("SELECT * FROM blood_tests", con=engine)

# Define predfined classifications
classification_rules = {
    "Basophil Count": {"normal": (0.01, 0.2), "borderline": (0.2, 0.3)},
    "Eosinophil Count": {"normal": (0.02, 0.4), "borderline": (0.4, 0.5)},
    "HbA1C": {"normal": (20, 38), "borderline": (38, 45)},
    "Haematocrit (HCT)": {"normal": (36, 47), "borderline": (47, 50)},
    "Haemoglobin": {"normal": (130, 160), "borderline": (160, 170)},
}

def classify_test(test_name, value):
    if test_name in classification_rules:
        rules = classification_rules[test_name]
        if rules["normal"][0] <= value <= rules["normal"][1]:
            return "Normal"
        elif rules["borderline"][0] <= value <= rules["borderline"][1]:
            return "Borderline"
        else:
            return "Abnormal"
    return "Unknown"

# Apply classification
df["Category"] = df.apply(lambda row: classify_test(row["test_name"], row["value"]), axis=1)

# Encode test names into numerical labels
encoder = LabelEncoder()
df["Test Name Encoded"] = encoder.fit_transform(df["test_name"])

# Encode classification categories
category_encoder = LabelEncoder()
df["Category Encoded"] = category_encoder.fit_transform(df["Category"])

# Prepare dataset
X = df[["Test Name Encoded", "value"]]
y = df["Category Encoded"] 

# Train decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
clf.fit(X_train, y_train)

# Save model and encoders
with open("blood_test_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

print("✅ Decision tree trained to classify blood tests into Normal, Borderline, or Abnormal.")

