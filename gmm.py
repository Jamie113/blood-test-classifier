# gmm.py
# Core GMM fitting functions shared between app.py and any offline scripts.

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from scipy.stats import norm as scipy_norm


def fit_optimal_gmm(values: np.ndarray) -> tuple:
    """
    Fit a GMM with BIC-optimal number of components (2–4).
    Upper bound scales with sample size: max_n = min(4, max(2, n // 5)).
    Returns (gmm, n_components, bic_scores_dict).
    """
    max_n = min(4, max(2, len(values) // 5))
    best_n, best_bic, best_gmm, bic_scores = 2, np.inf, None, {}
    for n in range(2, max_n + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
        gmm.fit(values.reshape(-1, 1))
        bic = gmm.bic(values.reshape(-1, 1))
        bic_scores[n] = bic
        if bic < best_bic:
            best_bic, best_n, best_gmm = bic, n, gmm
    return best_gmm, best_n, bic_scores


def sort_gmm(gmm: GaussianMixture) -> tuple:
    """Return (means, stds, weights) sorted ascending by component mean."""
    order   = np.argsort(gmm.means_.ravel())
    means   = gmm.means_.ravel()[order]
    stds    = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    return means, stds, weights


def get_boundaries(means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> list:
    """
    Find the x-value where adjacent sorted components have equal weighted PDF.
    Falls back to the midpoint between means if brentq finds no crossing.
    """
    boundaries = []
    for i in range(len(means) - 1):
        try:
            b = brentq(
                lambda x: (weights[i]   * scipy_norm.pdf(x, means[i],   stds[i]) -
                           weights[i+1] * scipy_norm.pdf(x, means[i+1], stds[i+1])),
                means[i], means[i+1],
            )
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
