# analysis.py
# Pure analysis functions — no Streamlit imports.
# Used by app.py at runtime and bake_demo.py at build time.

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
        means, stds, weights = sort_gmm(gmm)
        boundaries = get_boundaries(means, stds, weights)
        labels = assign_clusters(values, boundaries)

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
        }

    return results


def build_labelled_df(df_long: pd.DataFrame, gmm_results: dict) -> pd.DataFrame:
    """Attach group labels to every row."""
    df = df_long.copy()
    df["Group"] = "—"
    for test_name, res in gmm_results.items():
        if "error" in res:
            continue
        mask = df["test_name"] == test_name
        lbs  = assign_clusters(df.loc[mask, "value"].values, res["boundaries"])
        df.loc[mask, "Group"] = [f"Group {l + 1}" for l in lbs]
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

    max_n = min(5, n_patients - 1)
    best_n, best_bic, best_gmm, bic_scores = 2, np.inf, None, {}
    for n in range(2, max_n + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(X_cluster)
        bic = gmm.bic(X_cluster)
        bic_scores[n] = float(bic)
        if bic < best_bic:
            best_bic, best_n, best_gmm = bic, n, gmm

    labels      = best_gmm.predict(X_cluster)
    z_scores    = pd.DataFrame(X_scaled, index=df_wide.index, columns=df_wide.columns)
    fingerprint = z_scores.copy()
    fingerprint["Group"] = [f"Group {l + 1}" for l in labels]
    fingerprint = fingerprint.groupby("Group").mean().T

    return {
        "patient_ids":    list(df_wide.index),
        "labels":         labels,
        "n_clusters":     best_n,
        "bic_scores":     bic_scores,
        "X_pca_2d":       X_pca[:, :2],
        "pca_var":        pca.explained_variance_ratio_,
        "n_cluster_dims": n_cluster_dims,
        "fingerprint":    fingerprint,
        "df_wide":        df_wide,
        "small_sample":   n_patients < 30,
    }
