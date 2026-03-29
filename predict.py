import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from thresholds import THRESHOLDS

# Load trained model and encoders
with open("blood_test_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

test_names = list(THRESHOLDS.keys())
test_name_encoder = LabelEncoder()
test_name_encoder.fit(test_names)


def predict_blood_test(test_name: str, value: float) -> str:
    if test_name not in test_names:
        raise ValueError(f"Unknown test: '{test_name}'. Valid tests: {test_names}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"Value must be numeric, got {type(value).__name__}")

    test_encoded = test_name_encoder.transform([test_name])[0]
    input_df = pd.DataFrame([[test_encoded, value]], columns=["Test Name Encoded", "value"])
    prediction = clf.predict(input_df)
    return category_encoder.inverse_transform(prediction)[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a blood test result")
    parser.add_argument("--test", required=True, choices=test_names, help="Test name")
    parser.add_argument("--value", required=True, type=float, help="Numeric test value")
    args = parser.parse_args()

    result = predict_blood_test(args.test, args.value)
    print(f"{args.test}: {args.value} → {result}")
