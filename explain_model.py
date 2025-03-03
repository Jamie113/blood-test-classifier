import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

with open("blood_test_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=["Test Name", "Value"], filled=True)
plt.savefig("decision_tree.png")
plt.show()
