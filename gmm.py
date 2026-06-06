# gmm.py
# Core GMM fitting functions used by analysis.py and bake_demo.py.

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as scipy_norm
from sklearn.mixture import GaussianMixture


def fit_optimal_gmm(values: np.ndarray, delta_bic_threshold: float = 6.0) -> tuple:
    """
    Fit a GMM with BIC-optimal number of components.

    Tries K = 1, 2, …, up to `min(4, max(2, n // 5))`. The best K (lowest BIC)
    only "wins" over K=1 if its BIC improvement exceeds `delta_bic_threshold`
    (default 6, the conventional "strong evidence" cutoff for BIC). Otherwise
    we fall back to K=1 — i.e. report a single Gaussian rather than overstate
    spurious sub-groups in genuinely uniform markers.

    Returns (gmm, n_components, bic_scores_dict). bic_scores_dict always
    contains an entry for K=1 so callers can show the comparison.
    """
    max_n = min(4, max(2, len(values) // 5))
    bic_scores: dict = {}
    fits: dict = {}
    for n in range(1, max_n + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
        gmm.fit(values.reshape(-1, 1))
        bic_scores[n] = float(gmm.bic(values.reshape(-1, 1)))
        fits[n] = gmm

    # Pick lowest BIC overall, then enforce the ΔBIC margin against K=1.
    best_n = min(bic_scores, key=bic_scores.get)
    if best_n > 1 and (bic_scores[1] - bic_scores[best_n]) < delta_bic_threshold:
        best_n = 1
    return fits[best_n], best_n, bic_scores


def sort_gmm(gmm: GaussianMixture) -> tuple:
    """Return (means, stds, weights, order) sorted ascending by component mean.

    `order` is the permutation that sorts the original GMM components by mean —
    i.e. `means_sorted[i] == gmm.means_.ravel()[order[i]]`. Callers that need
    posterior-correct hard labels should compute `np.argsort(order)` and use
    that to remap `gmm.predict(...)` output into the sorted indexing.
    """
    order   = np.argsort(gmm.means_.ravel())
    means   = gmm.means_.ravel()[order]
    stds    = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    return means, stds, weights, order


def _weighted_pdf_gap(x: float, w0: float, m0: float, s0: float,
                      w1: float, m1: float, s1: float) -> float:
    """Difference of two weighted Gaussian PDFs — zero at their crossing point."""
    return w0 * scipy_norm.pdf(x, m0, s0) - w1 * scipy_norm.pdf(x, m1, s1)


def get_boundaries(means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> list:
    """
    Find the x-value where adjacent sorted components have equal weighted PDF.
    Falls back to the midpoint between means if brentq finds no crossing.
    """
    boundaries = []
    for i in range(len(means) - 1):
        params = (weights[i],   means[i],   stds[i],
                  weights[i+1], means[i+1], stds[i+1])
        try:
            b = brentq(_weighted_pdf_gap, means[i], means[i+1], args=params)
        except ValueError:
            b = (means[i] + means[i+1]) / 2
        boundaries.append(b)
    return boundaries


def assign_clusters(values: np.ndarray, boundaries: list) -> np.ndarray:
    """
    Assign a 0-indexed cluster label to each value.
    Cluster 0 = lowest values; labels increment at each boundary.
    """
    labels = np.zeros(len(values), dtype=int)
    for b in boundaries:
        labels[values > b] += 1
    return labels
