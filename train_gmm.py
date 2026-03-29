import numpy as np
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from scipy.stats import norm
from thresholds import THRESHOLDS

DATABASE_URL = "sqlite:///blood_tests.db"


def fit_gmm(values: np.ndarray, n_components: int = 3) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=5)
    gmm.fit(values.reshape(-1, 1))
    return gmm


def sort_components(gmm: GaussianMixture):
    """Return (means, stds, weights) sorted ascending by mean."""
    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order]
    stds = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    return means, stds, weights


def pdf_intersection(means, stds, weights, i: int, j: int) -> float:
    """Find the x where weighted PDF of component i equals component j, between their means."""
    lo, hi = means[i], means[j]

    def diff(x):
        pi = weights[i] * norm.pdf(x, means[i], stds[i])
        pj = weights[j] * norm.pdf(x, means[j], stds[j])
        return pi - pj

    return brentq(diff, lo, hi)


def map_components_to_labels(means: np.ndarray, test_name: str) -> dict:
    """Map sorted component indices to Normal/Borderline/Abnormal using threshold midpoints."""
    rules = THRESHOLDS[test_name]
    normal_mid = (rules["normal"][0] + rules["normal"][1]) / 2
    borderline_mid = (rules["borderline"][0] + rules["borderline"][1]) / 2
    normal_width = rules["normal"][1] - rules["normal"][0]
    abnormal_mid = rules["borderline"][1] + normal_width / 2

    anchors = {"Normal": normal_mid, "Borderline": borderline_mid, "Abnormal": abnormal_mid}
    label_map = {}
    for label, anchor in anchors.items():
        closest = int(np.argmin(np.abs(means - anchor)))
        label_map[closest] = label
    return label_map


def compute_bics(values: np.ndarray, component_range=(2, 3, 4)) -> dict:
    return {
        n: GaussianMixture(n_components=n, random_state=42, n_init=5)
           .fit(values.reshape(-1, 1))
           .bic(values.reshape(-1, 1))
        for n in component_range
    }


def train_all_gmms(engine) -> dict:
    df = pd.read_sql("SELECT * FROM blood_tests", con=engine)
    results = {}

    for test_name in THRESHOLDS:
        values = df[df["test_name"] == test_name]["value"].values
        gmm = fit_gmm(values)
        means, stds, weights = sort_components(gmm)

        boundary_0_1 = pdf_intersection(means, stds, weights, 0, 1)
        boundary_1_2 = pdf_intersection(means, stds, weights, 1, 2)

        label_map = map_components_to_labels(means, test_name)
        bics = compute_bics(values)

        results[test_name] = {
            "gmm": gmm,
            "means": means,
            "stds": stds,
            "weights": weights,
            "boundaries": [boundary_0_1, boundary_1_2],
            "label_map": label_map,
            "bic": bics,
        }

        print(f"\n{test_name}")
        print(f"  Component means:  {np.round(means, 4)}")
        print(f"  Boundaries:       {np.round([boundary_0_1, boundary_1_2], 4)}")
        print(f"  Label map:        {label_map}")
        best_n = min(bics, key=bics.get)
        print(f"  BIC (2/3/4):      {int(bics[2])}/{int(bics[3])}/{int(bics[4])} → best: {best_n}")

    with open("gmm_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\nGMM results saved to gmm_results.pkl")
    return results


if __name__ == "__main__":
    from sqlalchemy import create_engine as _ce
    engine = _ce(DATABASE_URL)
    train_all_gmms(engine)
