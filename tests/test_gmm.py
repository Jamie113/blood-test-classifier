# Note: requires gmm_results.pkl to exist.
# Run: python train_gmm.py before running these tests.
import pytest
import pickle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thresholds import THRESHOLDS

TOLERANCE = 0.15  # 15% tolerance for GMM boundary vs known threshold


@pytest.fixture(scope="module")
def gmm_results():
    with open(os.path.join(os.path.dirname(__file__), "..", "gmm_results.pkl"), "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("test_name", list(THRESHOLDS.keys()))
def test_normal_borderline_boundary_within_tolerance(test_name, gmm_results):
    known = THRESHOLDS[test_name]["normal"][1]
    discovered = gmm_results[test_name]["boundaries"][0]
    pct_diff = abs(discovered - known) / known
    assert pct_diff <= TOLERANCE, (
        f"{test_name}: GMM N→B boundary {discovered:.3f} is {pct_diff*100:.1f}% from known {known:.3f}"
    )


@pytest.mark.parametrize("test_name", list(THRESHOLDS.keys()))
def test_borderline_abnormal_boundary_within_tolerance(test_name, gmm_results):
    known = THRESHOLDS[test_name]["borderline"][1]
    discovered = gmm_results[test_name]["boundaries"][1]
    pct_diff = abs(discovered - known) / known
    assert pct_diff <= TOLERANCE, (
        f"{test_name}: GMM B→A boundary {discovered:.3f} is {pct_diff*100:.1f}% from known {known:.3f}"
    )


@pytest.mark.parametrize("test_name", list(THRESHOLDS.keys()))
def test_bic_favours_3_components(test_name, gmm_results):
    bics = gmm_results[test_name]["bic"]
    best_n = min(bics, key=bics.get)
    assert best_n == 3, (
        f"{test_name}: BIC favours {best_n} components, not 3. BICs: {bics}"
    )


@pytest.mark.parametrize("test_name", list(THRESHOLDS.keys()))
def test_three_components_fitted(test_name, gmm_results):
    assert gmm_results[test_name]["gmm"].n_components == 3


@pytest.mark.parametrize("test_name", list(THRESHOLDS.keys()))
def test_label_map_covers_all_categories(test_name, gmm_results):
    labels = set(gmm_results[test_name]["label_map"].values())
    assert labels == {"Normal", "Borderline", "Abnormal"}, (
        f"{test_name}: label_map missing categories — got {labels}"
    )
