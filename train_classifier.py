import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from thresholds import classify_test

DATABASE_URL = "sqlite:///blood_tests.db"
engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM blood_tests", con=engine)

# Apply rule-based labels from thresholds.py
df["Category"] = df.apply(lambda row: classify_test(row["test_name"], row["value"]), axis=1)

# Encode test names
encoder = LabelEncoder()
df["Test Name Encoded"] = encoder.fit_transform(df["test_name"])

# Encode categories
category_encoder = LabelEncoder()
df["Category Encoded"] = category_encoder.fit_transform(df["Category"])

X = df[["Test Name Encoded", "value"]]
y = df["Category Encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=5)
clf.fit(X_train, y_train)

# Metrics
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=category_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and encoders
with open("blood_test_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

print("\nDecision tree trained and saved.")
