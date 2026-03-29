import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from thresholds import classify_test, THRESHOLDS

DATABASE_URL = "sqlite:///blood_tests.db"


def predict_gmm(test_name: str, value: float, gmm_result: dict) -> str:
    boundaries = gmm_result["boundaries"]
    label_map = gmm_result["label_map"]
    if value <= boundaries[0]:
        component = 0
    elif value <= boundaries[1]:
        component = 1
    else:
        component = 2
    return label_map.get(component, "Unknown")


def agreement_pct(a: pd.Series, b: pd.Series) -> float:
    return (a == b).mean() * 100


def plot_histogram(test_name: str, values: np.ndarray, gmm_result: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(values, bins=50, density=True, alpha=0.4, color="steelblue", label="Data")

    x = np.linspace(values.min() - values.std(), values.max() + values.std(), 500)
    means, stds, weights = gmm_result["means"], gmm_result["stds"], gmm_result["weights"]

    colours = {"Normal": "green", "Borderline": "orange", "Abnormal": "red"}
    for i, (m, s, w) in enumerate(zip(means, stds, weights)):
        label = gmm_result["label_map"].get(i, f"Component {i}")
        colour = colours.get(label, "grey")
        ax.plot(x, w * norm.pdf(x, m, s), color=colour, linewidth=1.5, label=f"GMM: {label}")

    total = sum(w * norm.pdf(x, m, s) for m, s, w in zip(means, stds, weights))
    ax.plot(x, total, "k--", linewidth=1, label="GMM total")

    for boundary in gmm_result["boundaries"]:
        ax.axvline(boundary, color="orange", linestyle="--", linewidth=1.5,
                   label=f"GMM boundary: {boundary:.3f}")

    rules = THRESHOLDS[test_name]
    ax.axvline(rules["normal"][1], color="green", linestyle=":", linewidth=1.5,
               label=f"Rules N→B: {rules['normal'][1]}")
    ax.axvline(rules["borderline"][1], color="red", linestyle=":", linewidth=1.5,
               label=f"Rules B→A: {rules['borderline'][1]}")

    ax.set_title(f"{test_name} — GMM vs Rules")
    ax.set_xlabel(f"Value ({THRESHOLDS[test_name]['unit']})")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM blood_tests", con=engine)

    with open("blood_test_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("category_encoder.pkl", "rb") as f:
        category_encoder = pickle.load(f)
    with open("gmm_results.pkl", "rb") as f:
        gmm_results = pickle.load(f)

    test_names = list(THRESHOLDS.keys())
    test_name_encoder = LabelEncoder()
    test_name_encoder.fit(test_names)

    # Rules predictions
    df["pred_rules"] = df.apply(lambda r: classify_test(r["test_name"], r["value"]), axis=1)

    # Decision tree predictions
    encoded = test_name_encoder.transform(df["test_name"])
    X = pd.DataFrame({"Test Name Encoded": encoded, "value": df["value"].values})
    df["pred_tree"] = category_encoder.inverse_transform(clf.predict(X))

    # GMM predictions
    df["pred_gmm"] = df.apply(
        lambda r: predict_gmm(r["test_name"], r["value"], gmm_results[r["test_name"]]), axis=1
    )

    print("=== Agreement Percentages ===")
    print(f"Rules vs Tree:  {agreement_pct(df['pred_rules'], df['pred_tree']):.1f}%")
    print(f"Rules vs GMM:   {agreement_pct(df['pred_rules'], df['pred_gmm']):.1f}%")
    print(f"Tree  vs GMM:   {agreement_pct(df['pred_tree'],  df['pred_gmm']):.1f}%")

    print("\n=== GMM Boundary Comparison ===")
    header = f"{'Test':<25} {'Known N→B':>10} {'GMM N→B':>10} {'Diff%':>7}  {'Known B→A':>10} {'GMM B→A':>10} {'Diff%':>7}"
    print(header)
    print("-" * len(header))
    for test_name in test_names:
        rules = THRESHOLDS[test_name]
        known_nb = rules["normal"][1]
        known_ba = rules["borderline"][1]
        b0, b1 = gmm_results[test_name]["boundaries"]
        diff_nb = abs(b0 - known_nb) / known_nb * 100
        diff_ba = abs(b1 - known_ba) / known_ba * 100
        print(f"{test_name:<25} {known_nb:>10.3f} {b0:>10.3f} {diff_nb:>6.1f}%  {known_ba:>10.3f} {b1:>10.3f} {diff_ba:>6.1f}%")

    print("\n=== BIC Analysis (lower = better) ===")
    for test_name in test_names:
        bics = gmm_results[test_name]["bic"]
        best = min(bics, key=bics.get)
        print(f"{test_name}: BIC(2)={int(bics[2])}, BIC(3)={int(bics[3])}, BIC(4)={int(bics[4])} → best: {best}")

    print("\n=== Saving Histogram Plots ===")
    for test_name in test_names:
        values = df[df["test_name"] == test_name]["value"].values
        safe_name = test_name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_histogram(test_name, values, gmm_results[test_name], f"plot_{safe_name}.png")
