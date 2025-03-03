import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load trained model and encoders
with open("blood_test_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("category_encoder.pkl", "rb") as f:
    category_encoder = pickle.load(f)

# Load label encoder for test names
test_names = ["Basophil Count", "Eosinophil Count", "HbA1C", "Haematocrit (HCT)", "Haemoglobin"]
test_name_encoder = LabelEncoder()
test_name_encoder.fit(test_names)

# Function to predict category of a blood test
def predict_blood_test(test_name, value):
    test_encoded = test_name_encoder.transform([test_name])[0]
    
    # Convert to DataFrame with proper feature names
    input_df = pd.DataFrame([[test_encoded, value]], columns=["Test Name Encoded", "Value"])
    
    prediction = clf.predict(input_df)
    return category_encoder.inverse_transf
