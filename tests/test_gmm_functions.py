import numpy as np
import pytest
from sklearn.mixture import GaussianMixture
from gmm import fit_optimal_gmm, sort_gmm, get_boundaries, assign_clusters


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_bimodal(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(10, 1, n // 2), rng.normal(20, 1, n // 2)])


def make_trimodal(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(10, 1, n // 3),
        rng.normal(20, 1, n // 3),
        rng.normal(30, 1, n // 3),
    ])


# ── fit_optimal_gmm ───────────────────────────────────────────────────────────

def test_returns_gmm_object():
    gmm, n, bics = fit_optimal_gmm(make_bimodal())
    assert isinstance(gmm, GaussianMixture)


def test_returns_n_components():
    _, n, _ = fit_optimal_gmm(make_bimodal())
    assert isinstance(n, int)
    assert 2 <= n <= 4


def test_returns_bic_scores_dict():
    _, _, bics = fit_optimal_gmm(make_bimodal())
    assert isinstance(bics, dict)
    assert all(isinstance(k, int) for k in bics)
    assert all(isinstance(v, float) for v in bics.values())


def test_bimodal_data_selects_2_components():
    _, n, _ = fit_optimal_gmm(make_bimodal(n=200))
    assert n == 2


def test_trimodal_data_selects_3_components():
    _, n, _ = fit_optimal_gmm(make_trimodal(n=300))
    assert n == 3


def test_small_dataset_caps_at_2_components():
    # 9 points → max_n = min(4, max(2, 9//5)) = min(4, max(2, 1)) = 2
    rng = np.random.default_rng(0)
    values = rng.normal(10, 1, 9)
    _, n, _ = fit_optimal_gmm(values)
    assert n == 2


def test_bic_scores_cover_tried_components():
    values = make_trimodal(n=300)
    _, _, bics = fit_optimal_gmm(values)
    assert 2 in bics
    assert 3 in bics


# ── sort_gmm ──────────────────────────────────────────────────────────────────

def test_means_sorted_ascending():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    means, _, _ = sort_gmm(gmm)
    assert list(means) == sorted(means)


def test_sort_returns_three_arrays():
    gmm, _, _ = fit_optimal_gmm(make_bimodal())
    result = sort_gmm(gmm)
    assert len(result) == 3


def test_weights_sum_to_one():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    _, _, weights = sort_gmm(gmm)
    assert abs(weights.sum() - 1.0) < 1e-6


# ── get_boundaries ────────────────────────────────────────────────────────────

def test_boundary_count_is_n_minus_1():
    gmm, n, _ = fit_optimal_gmm(make_trimodal())
    means, stds, weights = sort_gmm(gmm)
    boundaries = get_boundaries(means, stds, weights)
    assert len(boundaries) == n - 1


def test_boundary_lies_between_adjacent_means():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    means, stds, weights = sort_gmm(gmm)
    boundaries = get_boundaries(means, stds, weights)
    for i, b in enumerate(boundaries):
        assert means[i] < b < means[i + 1], (
            f"Boundary {b:.3f} not between means {means[i]:.3f} and {means[i+1]:.3f}"
        )


def test_boundary_close_to_midpoint_for_equal_gaussians():
    # Two equal Gaussians centred at 0 and 10 → boundary should be near 5
    means   = np.array([0.0, 10.0])
    stds    = np.array([1.0,  1.0])
    weights = np.array([0.5,  0.5])
    boundaries = get_boundaries(means, stds, weights)
    assert abs(boundaries[0] - 5.0) < 0.1


# ── assign_clusters ───────────────────────────────────────────────────────────

def test_all_labels_valid():
    values = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
    boundaries = [7.5, 12.5]
    labels = assign_clusters(values, boundaries)
    assert set(labels).issubset({0, 1, 2})


def test_values_below_first_boundary_are_cluster_0():
    values = np.array([1.0, 2.0, 3.0])
    labels = assign_clusters(values, [10.0, 20.0])
    assert all(labels == 0)


def test_values_above_last_boundary_are_highest_cluster():
    values = np.array([25.0, 30.0])
    labels = assign_clusters(values, [10.0, 20.0])
    assert all(labels == 2)


def test_cluster_assignment_is_monotone():
    values = np.array([1.0, 5.0, 11.0, 21.0])
    boundaries = [7.5, 15.0]
    labels = assign_clusters(values, boundaries)
    assert list(labels) == [0, 0, 1, 2]


def test_single_boundary():
    values = np.array([3.0, 8.0])
    labels = assign_clusters(values, [5.0])
    assert list(labels) == [0, 1]
