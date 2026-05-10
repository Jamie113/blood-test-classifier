# analysis.py
# Pure analysis functions — framework-agnostic.
# Imported by web/main.py at request time and bake_demo.py at build time.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from gmm import fit_optimal_gmm, sort_gmm, get_boundaries, assign_clusters


def analyse_upload(df_long: pd.DataFrame) -> dict:
    """Fit GMM per marker. Returns per-marker analysis dict."""
    results = {}
    for test_name in df_long["test_name"].unique():
        values = df_long[df_long["test_name"] == test_name]["value"].dropna().values

        if len(values) < 4:
            results[test_name] = {"error": f"Only {len(values)} data points", "values": values}
            continue

        gmm, n, bic_scores = fit_optimal_gmm(values)
        means, stds, weights, order = sort_gmm(gmm)
        boundaries = get_boundaries(means, stds, weights)
        # Use the GMM's posterior assignments for hard labels — these are
        # consistent with predict_proba and don't depend on brentq finding the
        # PDF crossing inside [mean_i, mean_{i+1}]. Boundaries remain only as
        # chart annotations for the user.
        order_inverse = np.argsort(order)
        raw_labels = gmm.predict(values.reshape(-1, 1))
        labels = order_inverse[raw_labels]

        cluster_stats = []
        for i in range(n):
            mask = labels == i
            cv = values[mask]
            cluster_stats.append({
                "Group":      f"Group {i + 1}",
                "Average":    round(float(means[i]), 3),
                "Spread (±)": round(float(stds[i]), 3),
                "Min":        round(float(cv.min()), 3) if len(cv) else "—",
                "Max":        round(float(cv.max()), 3) if len(cv) else "—",
                "Patients":   int(mask.sum()),
                "% of Total": f"{mask.mean() * 100:.1f}%",
            })

        results[test_name] = {
            "n_components":  n,
            "bic_scores":    bic_scores,
            "means":         means,
            "stds":          stds,
            "weights":       weights,
            "boundaries":    boundaries,
            "labels":        labels,
            "values":        values,
            "cluster_stats": cluster_stats,
            "small_sample":  len(values) < 30,
            "gmm":           gmm,
            "order_inverse": order_inverse,
        }

    return results


def build_labelled_df(df_long: pd.DataFrame, gmm_results: dict) -> pd.DataFrame:
    """Attach group labels to every row, using the GMM's posterior assignments
    so the labels stay consistent with the per-marker `labels` array."""
    df = df_long.copy()
    df["Group"] = "—"
    for test_name, res in gmm_results.items():
        if "error" in res:
            continue
        mask = df["test_name"] == test_name
        values = df.loc[mask, "value"].values
        raw = res["gmm"].predict(values.reshape(-1, 1))
        sorted_labels = res["order_inverse"][raw]
        df.loc[mask, "Group"] = [f"Group {l + 1}" for l in sorted_labels]
    return df


def analyse_population(df_long: pd.DataFrame) -> dict:
    """
    Multivariate patient clustering across all markers.
    Pipeline: wide pivot → median impute → StandardScaler → PCA → GMM.
    """
    df_wide = df_long.pivot_table(index="patient_id", columns="test_name", values="value")
    df_wide = df_wide.dropna(thresh=int(len(df_wide) * 0.5), axis=1)
    df_wide = df_wide.fillna(df_wide.median())

    n_patients, n_markers = df_wide.shape
    if n_patients < 4:
        return {"error": f"Only {n_patients} patients — need at least 4 for population clustering."}
    if n_markers < 2:
        return {"error": "Not enough markers with sufficient coverage for population clustering."}

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_wide.values)

    max_components = min(n_patients - 1, n_markers)
    pca   = PCA(n_components=max_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    cumvar        = np.cumsum(pca.explained_variance_ratio_)
    n_for_80      = int(np.searchsorted(cumvar, 0.80)) + 1
    n_cluster_dims = max(2, min(n_for_80, max_components))
    X_cluster     = X_pca[:, :n_cluster_dims]

    # Sample-size-aware K cap (~25 patients per cluster minimum). Caps are
    # max(2, min(5, n // 25)) so n=80 demo can reach at most K=3, n=125+ can
    # reach K=5. Always include K=1 in the search so we can apply a ΔBIC=6
    # threshold against the no-cluster null hypothesis.
    max_k_by_size = max(2, min(5, n_patients // 25))
    max_n = min(max_k_by_size, n_patients - 1)
    # Diagonal covariance: PCA already decorrelates the global data, so
    # within-cluster off-diagonal terms tend to be small. Diagonal cuts each
    # component's parameter count by ~5x in 10-D space (10 vs 55 free params),
    # so BIC stops over-penalising K>1 on small cohorts. Empirically: with
    # full covariance the demo's two ground-truth subgroups can't beat K=1
    # on BIC; with diagonal they win by 100+ BIC units.
    bic_scores: dict = {}
    fits: dict = {}
    for n in range(1, max_n + 1):
        gmm = GaussianMixture(
            n_components=n, covariance_type="diag",
            random_state=42, n_init=5,
        )
        gmm.fit(X_cluster)
        bic_scores[n] = float(gmm.bic(X_cluster))
        fits[n] = gmm

    best_n = min(bic_scores, key=bic_scores.get)
    if best_n > 1 and (bic_scores[1] - bic_scores[best_n]) < 6.0:
        best_n = 1
    best_gmm = fits[best_n]

    labels       = best_gmm.predict(X_cluster)
    posteriors   = best_gmm.predict_proba(X_cluster)
    log_likelihood = best_gmm.score_samples(X_cluster)

    # Per-test squared Mahalanobis distance to the assigned cluster's mean.
    # With diagonal covariance, covariances_[k] is a 1-D variance vector, so
    # Mahalanobis² = Σ (x - μ)² / σ². Under the model this follows χ² with
    # df = n_cluster_dims, enabling a principled outlier threshold (vs the
    # sample-quantile rule that always flagged some tests).
    mahalanobis_sq = np.zeros(n_patients)
    for i, label in enumerate(labels):
        diff = X_cluster[i] - best_gmm.means_[label]
        var = best_gmm.covariances_[label]
        mahalanobis_sq[i] = float(np.sum(diff ** 2 / var))

    z_scores     = pd.DataFrame(X_scaled, index=df_wide.index, columns=df_wide.columns)
    fingerprint  = z_scores.copy()
    fingerprint["Group"] = [f"Group {l + 1}" for l in labels]
    fingerprint  = fingerprint.groupby("Group").mean().T

    return {
        "patient_ids":     list(df_wide.index),
        "labels":          labels,
        "posteriors":      posteriors,
        "log_likelihood":  log_likelihood,
        "mahalanobis_sq":  mahalanobis_sq,
        "n_clusters":      best_n,
        "bic_scores":      bic_scores,
        "X_pca_2d":        X_pca[:, :2],
        "pca_var":         pca.explained_variance_ratio_,
        "n_cluster_dims":  n_cluster_dims,
        "fingerprint":     fingerprint,
        "z_scores":        z_scores,
        "df_wide":         df_wide,
        "small_sample":    n_patients < 30,
    }


def filter_long(
    df_long: pd.DataFrame,
    age_range: tuple | None = None,
    marker_ranges: dict | None = None,
) -> pd.DataFrame:
    """
    Return a subset of df_long restricted to patients matching the given filters.

    age_range: (lo, hi) inclusive, in years. Skipped if df_long has no age column or all NaN.
    marker_ranges: {marker_name: (lo, hi)} inclusive, in canonical units. Each filter
                   removes patients whose value for that marker is outside [lo, hi];
                   patients without a value for the marker are kept.
    """
    if df_long.empty:
        return df_long

    keep = pd.Series(True, index=df_long["patient_id"].unique(), name="patient_id")

    if age_range is not None and "age" in df_long.columns:
        ages = df_long.drop_duplicates("patient_id").set_index("patient_id")["age"]
        ages = pd.to_numeric(ages, errors="coerce")
        in_range = ages.between(age_range[0], age_range[1])
        keep &= in_range.reindex(keep.index, fill_value=False)

    if marker_ranges:
        for marker, (lo, hi) in marker_ranges.items():
            sub = df_long[df_long["test_name"] == marker]
            if sub.empty:
                continue
            patient_vals = sub.groupby("patient_id")["value"].first()
            in_range = patient_vals.between(lo, hi)
            in_range = in_range.reindex(keep.index, fill_value=True)
            keep &= in_range

    kept_ids = keep[keep].index
    return df_long[df_long["patient_id"].isin(kept_ids)].reset_index(drop=True)


def most_separated_marker(gmm_results: dict) -> tuple | None:
    """
    Return (marker_name, separation_score) for the marker with the
    most pronounced cluster separation. Score is the spread of cluster means
    measured in pooled-std units (Cohen's-d-style). Returns None if no marker
    has at least two components.
    """
    best_name, best_score = None, -1.0
    for name, res in gmm_results.items():
        if "error" in res or res.get("n_components", 0) < 2:
            continue
        means = np.asarray(res["means"], dtype=float)
        stds  = np.asarray(res["stds"],  dtype=float)
        pooled = float(np.mean(stds))
        if pooled <= 0:
            continue
        score = float(means.max() - means.min()) / pooled
        if score > best_score:
            best_name, best_score = name, score
    if best_name is None:
        return None
    return best_name, best_score


def strongest_marker_pair(df_long: pd.DataFrame, min_overlap: int = 10) -> tuple | None:
    """
    Return (marker_a, marker_b, r) for the pair of markers with the largest
    |Pearson r| in df_long, requiring at least `min_overlap` patients with both
    measured. Returns None if no qualifying pair exists.
    """
    if df_long.empty:
        return None
    wide = df_long.pivot_table(index="patient_id", columns="test_name", values="value")
    if wide.shape[1] < 2:
        return None

    corr = wide.corr(min_periods=min_overlap)
    if corr.isna().all().all():
        return None

    mask = ~np.eye(len(corr), dtype=bool)
    corr_no_diag = corr.where(mask)
    flat = corr_no_diag.unstack().dropna()
    if flat.empty:
        return None
    flat_abs = flat.abs().sort_values(ascending=False)
    a, b = flat_abs.index[0]
    return str(a), str(b), float(corr.loc[a, b])
