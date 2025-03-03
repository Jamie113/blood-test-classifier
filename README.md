# 🧪 Blood test classifier

This project is a **ML based decision tree classifier** that analyses blood test results and classifies them as **Normal, Borderline, or Abnormal** based on predefined medical thresholds. Goal is over time to evolve it so it creates its own nodes and thresholds. 

## 📌 Features
- Stores blood test results in an SQLite database
- Uses decision tree classification to categorize test results
- Provides explainability for how the classifier makes predictions
- Supports continuous learning (updating with new data)
- Goal is to expanded with a Ruby API for real-time classification

---

## 🚀 **Setup Instructions**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR_USERNAME/blood-test-classifier.git
cd blood-test-classifier
pip install pandas scikit-learn sqlalchemy faker

## Generate fake test data
 python3 generate_mock_data.py

## Train the classifier 
python3 train_classifier.py

## Create predicitons 
python3 predict.py

## Explain predictions 
python3 explain_model.py
