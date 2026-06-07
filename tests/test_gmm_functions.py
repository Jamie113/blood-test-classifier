import numpy as np
from sklearn.mixture import GaussianMixture

from gmm import assign_clusters, fit_optimal_gmm, get_boundaries, sort_gmm

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


def test_small_sample_reports_one_group_even_if_separated():
    """Evidence floor (#58): below ~10 points, K=2 isn't even tried — a handful
    of points can't be reported as "two groups", however clean the separation."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(10, 0.4, 4), rng.normal(30, 0.4, 4)])  # n=8
    _, n, bics = fit_optimal_gmm(values)
    assert n == 1
    assert max(bics) == 1   # K=2 not attempted below the floor


def test_evidence_floor_allows_k2_at_n10():
    """At n=10 (5 per component) K=2 becomes eligible and wins on clear data."""
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(10, 0.4, 5), rng.normal(30, 0.4, 5)])  # n=10
    _, n, bics = fit_optimal_gmm(values)
    assert 2 in bics
    assert n == 2


def test_uniform_data_selects_k1():
    """A genuinely uniform sample should not be forced into K≥2."""
    rng = np.random.default_rng(0)
    values = rng.normal(10, 1, 200)
    _, n, _ = fit_optimal_gmm(values)
    assert n == 1


def test_bic_scores_always_include_k1():
    """The K=1 fit must always be present so callers can apply the ΔBIC margin."""
    _, _, bics = fit_optimal_gmm(make_bimodal())
    assert 1 in bics


def test_bic_scores_cover_tried_components():
    values = make_trimodal(n=300)
    _, _, bics = fit_optimal_gmm(values)
    assert 2 in bics
    assert 3 in bics


# ── sort_gmm ──────────────────────────────────────────────────────────────────

def test_means_sorted_ascending():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    means, _, _, _ = sort_gmm(gmm)
    assert list(means) == sorted(means)


def test_sort_returns_four_arrays():
    gmm, _, _ = fit_optimal_gmm(make_bimodal())
    result = sort_gmm(gmm)
    assert len(result) == 4


def test_weights_sum_to_one():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    _, _, weights, _ = sort_gmm(gmm)
    assert abs(weights.sum() - 1.0) < 1e-6


def test_sort_order_is_a_permutation():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    _, _, _, order = sort_gmm(gmm)
    assert sorted(order) == list(range(gmm.n_components))


def test_sort_order_remaps_predict_correctly():
    """Inverse-order remapping must match a posterior-sorted prediction."""
    import numpy as np
    values = make_trimodal()
    gmm, _, _ = fit_optimal_gmm(values)
    means, _, _, order = sort_gmm(gmm)
    inverse = np.argsort(order)
    raw = gmm.predict(values.reshape(-1, 1))
    sorted_labels = inverse[raw]
    # Every label index must point to a real sorted component
    assert sorted_labels.min() >= 0
    assert sorted_labels.max() < gmm.n_components
    # Within each sorted-label group, the mean of values should be close to
    # the corresponding sorted mean (sanity check that the remap is correct)
    for k in range(gmm.n_components):
        mask = sorted_labels == k
        if mask.sum() > 0:
            assert abs(values[mask].mean() - means[k]) < 2.0  # generous: clusters overlap a bit


# ── get_boundaries ────────────────────────────────────────────────────────────

def test_boundary_count_is_n_minus_1():
    gmm, n, _ = fit_optimal_gmm(make_trimodal())
    means, stds, weights, _ = sort_gmm(gmm)
    boundaries = get_boundaries(means, stds, weights)
    assert len(boundaries) == n - 1


def test_boundary_lies_between_adjacent_means():
    gmm, _, _ = fit_optimal_gmm(make_trimodal())
    means, stds, weights, _ = sort_gmm(gmm)
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
