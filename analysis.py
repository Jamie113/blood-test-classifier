# analysis.py
# Pure analysis functions — framework-agnostic.
# Imported by web/main.py at request time and bake_demo.py at build time.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from gmm import fit_optimal_gmm, get_boundaries, sort_gmm


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

        results[test_name] = {
            "n_components":  n,
            "bic_scores":    bic_scores,
            "means":         means,
            "stds":          stds,
            "weights":       weights,
            "boundaries":    boundaries,
            "labels":        labels,
            "values":        values,
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
        df.loc[mask, "Group"] = [f"Group {lbl + 1}" for lbl in sorted_labels]
    return df


def _winsorize(values: np.ndarray, limit: float = 0.01) -> np.ndarray:
    """Clip each column to its [limit, 1-limit] percentiles.

    Caps the leverage any single extreme/erroneous value can exert on the PCA
    axes (and hence on cluster geometry) without discarding the row — the record
    is still placed and still flagged by the downstream Mahalanobis outlier test.
    """
    lo = np.percentile(values, limit * 100, axis=0)
    hi = np.percentile(values, 100 - limit * 100, axis=0)
    return np.clip(values, lo, hi)


def _cluster_fingerprint(z_scores: pd.DataFrame, labels, reliable) -> pd.DataFrame:
    """Mean z-score per marker per cluster (index=markers, columns="Group N").

    Each cluster is characterised from its RELIABLY-measured members only — a
    mostly median-filled row is ~0 on its imputed markers and would drag the
    signature toward the cohort mean. Cluster membership/counts still include
    everyone (a patient is genuinely assigned); only this descriptive fingerprint
    is gated. A cluster with no reliable members falls back to all its members,
    so no column is dropped (consumers index every cluster).
    """
    group_of = np.array([f"Group {lbl + 1}" for lbl in labels])
    reliable = np.asarray(reliable, dtype=bool)
    cols = {}
    for g in sorted(set(group_of)):
        in_group = group_of == g
        use = in_group & reliable
        cols[g] = z_scores[use if use.any() else in_group].mean()
    return pd.DataFrame(cols)


def analyse_population(df_long: pd.DataFrame) -> dict:
    """
    Multivariate patient clustering across all markers.
    Pipeline: wide pivot → median impute → winsorise → RobustScaler → PCA → GMM,
    selecting K only among solutions whose every cluster is well populated.
    """
    df_wide = df_long.pivot_table(index="patient_id", columns="test_name", values="value")
    # Drop algebraically-derived markers (e.g. the TC:HDL ratio): a near-collinear
    # combination of other columns would let PCA double-weight that axis.
    df_wide = df_wide.drop(columns=[m for m in DERIVED_MARKERS if m in df_wide.columns])
    df_wide = df_wide.dropna(thresh=int(len(df_wide) * 0.5), axis=1)
    # Per-patient imputation fraction over the markers actually used, measured
    # BEFORE filling — patients pulled toward the median by lots of missing data
    # have unreliable cluster placement (see _investigate_context).
    imputed_frac = (
        df_wide.isna().mean(axis=1).to_numpy()
        if df_wide.shape[1] else np.zeros(len(df_wide))
    )
    df_wide = df_wide.fillna(df_wide.median())

    n_patients, n_markers = df_wide.shape
    if n_patients < 4:
        return {"error": f"Only {n_patients} patients — need at least 4 for population clustering."}
    if n_markers < 2:
        return {"error": "Not enough markers with sufficient coverage for population clustering."}

    # Three views of the data, deliberately separate:
    #  • X_scaled (standard) feeds ONLY the descriptive fingerprint, so the
    #    per-cluster "what defines it" copy stays in familiar standard-deviation
    #    units (the strength thresholds in web/contexts.py key on |z|).
    #  • X_robust (winsorise → median/IQR) feeds PCA + clustering, so a few
    #    extreme or erroneous values can't dominate the PCA axes. On raw
    #    StandardScaler one bad record steered PC1 and produced unstable splits
    #    and one-person "clusters" on real uploads — and the extreme tails
    #    themselves inflated the apparent separation (ΔBIC 166→29 once capped).
    #  • X_full projects the UN-winsorised data through the same axes, used only
    #    for the Mahalanobis outlier distance below, so a capped record is still
    #    flagged as an outlier rather than hidden inside a cluster.
    X_scaled = StandardScaler().fit_transform(df_wide.values)
    robust   = RobustScaler().fit(_winsorize(df_wide.values))
    X_robust = robust.transform(_winsorize(df_wide.values))
    X_full   = robust.transform(df_wide.values)

    max_components = min(n_patients - 1, n_markers)
    pca        = PCA(n_components=max_components, random_state=42)
    X_pca      = pca.fit_transform(X_robust)
    X_pca_full = pca.transform(X_full)

    cumvar        = np.cumsum(pca.explained_variance_ratio_)
    n_for_80      = int(np.searchsorted(cumvar, 0.80)) + 1
    # Cap the cluster-space dimensionality by cohort size (≈10 patients per
    # dimension): a high-D diagonal Gaussian estimated from few points is noisy
    # and inflates the χ² df used for outlier flagging. Floor of 2. Caveat: the
    # ≈10/dim rule of thumb is framed for a single Gaussian; under a K-component
    # mixture each cluster sees only ~n/K points, so effective coverage is lower
    # as K grows (the cap is computed before K is chosen, so it can't scale by K).
    n_cluster_dims = max(2, min(n_for_80, max_components, n_patients // 10))
    X_cluster      = X_pca[:, :n_cluster_dims]       # winsorised — fit + labels
    X_cluster_full = X_pca_full[:, :n_cluster_dims]  # uncapped — outlier distance

    # Sample-size-aware K cap (~25 patients per cluster minimum). Evidence floor:
    # K>1 needs n>=50 (two clusters of ~25), n>=75 for K=3, etc. Below n=50 only
    # K=1 — too few patients to claim distinct multivariate clusters. n=80 demo
    # reaches K=3, n=125+ reaches K=5. K=1 is always in the search for the ΔBIC=6
    # test against the no-cluster null.
    max_k_by_size = max(1, min(5, n_patients // 25))
    max_n = min(max_k_by_size, n_patients - 1)
    # Diagonal covariance: PCA already decorrelates the global data, so
    # within-cluster off-diagonal terms tend to be small. Diagonal cuts each
    # component's parameter count by ~5x in 10-D space (10 vs 55 free params),
    # so BIC stops over-penalising K>1 on small cohorts. Empirically: with
    # full covariance the demo's two ground-truth subgroups can't beat K=1
    # on BIC; with diagonal they win by 100+ BIC units.
    bic_scores: dict = {}
    fits: dict = {}
    well_populated: dict = {}
    for n in range(1, max_n + 1):
        gmm = GaussianMixture(
            n_components=n, covariance_type="diag",
            random_state=42, n_init=5,
        )
        gmm.fit(X_cluster)
        bic_scores[n] = float(gmm.bic(X_cluster))
        fits[n] = gmm
        counts = np.bincount(gmm.predict(X_cluster), minlength=n)
        well_populated[n] = int(counts.min()) >= MIN_CLUSTER_SIZE

    # Only consider K whose every component is well populated: a "cluster" of one
    # or two is an outlier, not a sub-population. Disqualifying such K lets the
    # extreme record fall into a real cluster, where the Mahalanobis χ² test then
    # flags it on the Outliers tab. K=1 is always eligible (its single component
    # holds everyone), so the ΔBIC≥6 floor still runs against the no-cluster null.
    eligible = [n for n in bic_scores if well_populated[n]]
    best_n = min(eligible, key=bic_scores.get)
    if best_n > 1 and (bic_scores[1] - bic_scores[best_n]) < 6.0:
        best_n = 1
    best_gmm = fits[best_n]

    labels       = best_gmm.predict(X_cluster)
    posteriors   = best_gmm.predict_proba(X_cluster)

    # Per-test squared Mahalanobis distance to the assigned cluster's mean.
    # With diagonal covariance, covariances_[k] is a 1-D variance vector, so
    # Mahalanobis² = Σ (x - μ)² / σ². Under the model this follows χ² with
    # df = n_cluster_dims, enabling a principled outlier threshold (vs the
    # sample-quantile rule that always flagged some tests). Caveat: distance is
    # to the test's OWN assigned cluster mean (predict-then-measure), which
    # biases it DOWNWARD vs a true χ² draw — yet the realised flag rate can still
    # sit above the nominal 1% (the demo flags ~2.5%) when a cluster's PCA
    # distribution has heavier-than-Gaussian tails. So the χ²₀.₉₉ cut is an
    # approximate, model-based threshold, not a calibrated guarantee. Distance is
    # measured on the UN-winsorised projection (X_cluster_full) so a record whose
    # extreme value was capped for the fit is still flagged here.
    mahalanobis_sq = np.zeros(n_patients)
    for i, label in enumerate(labels):
        diff = X_cluster_full[i] - best_gmm.means_[label]
        var = best_gmm.covariances_[label]
        mahalanobis_sq[i] = float(np.sum(diff ** 2 / var))

    z_scores = pd.DataFrame(X_scaled, index=df_wide.index, columns=df_wide.columns)
    reliable = imputed_frac <= HEAVY_IMPUTE_FRAC
    fingerprint = _cluster_fingerprint(z_scores, labels, reliable)

    return {
        "patient_ids":     list(df_wide.index),
        "labels":          labels,
        "posteriors":      posteriors,
        "mahalanobis_sq":  mahalanobis_sq,
        "imputed_frac":    imputed_frac,
        "n_clusters":      best_n,
        "bic_scores":      bic_scores,
        "X_pca_2d":        X_pca[:, :2],
        "pca_var":         pca.explained_variance_ratio_,
        "n_cluster_dims":  n_cluster_dims,
        "fingerprint":     fingerprint,
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


# Markers that are algebraically derived from other markers in the panel, so
# they correlate / co-vary with their components by construction. Excluded from
# the "strongest"/"clearest" auto-pick scans (they'd surface tautological
# findings); still selectable manually in the dropdowns.
DERIVED_MARKERS = frozenset({"Total Cholesterol:HDL Ratio"})

# A patient with more than this fraction of markers median-imputed sits near the
# cohort centroid by construction, so their cluster placement and the cluster's
# z-score fingerprint can't be trusted. Used by the population fingerprint here
# and by the Outliers-tab set-aside in web/contexts.py (one source of truth).
HEAVY_IMPUTE_FRAC = 0.5

# Minimum members for a multivariate cluster to count as a sub-population rather
# than an outlier. A K whose smallest component falls below this is disqualified
# during model selection (see analyse_population), so a lone extreme record can
# never be reported as its own "cluster".
MIN_CLUSTER_SIZE = 3

# Markers bound by a structural identity, so a pair WITHIN a group correlates by
# construction (Total Cholesterol ≈ HDL + LDL + ~0.45·Triglycerides; LDL is
# often Friedewald-derived from the others). Such pairs are skipped by the
# strongest-pair auto-pick — they'd be a less-obvious version of the same
# tautology DERIVED_MARKERS guards against. (Only purely-algebraic cases; this
# is a deliberately small, panel-specific list, not a general collinearity test.)
_CONSTRAINT_GROUPS = (
    frozenset({"Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol"}),
)


def _same_constraint_group(a: str, b: str) -> bool:
    return any({a, b} <= group for group in _CONSTRAINT_GROUPS)


def n_comparable_pairs(markers) -> int:
    """Count of candidate marker pairs the strongest-pair auto-pick considers:
    unordered pairs of non-derived markers, excluding within-constraint-group
    pairs. Upper bound on the search (before the min-overlap filter)."""
    ms = [m for m in markers if m not in DERIVED_MARKERS]
    return sum(
        1
        for i in range(len(ms))
        for j in range(i + 1, len(ms))
        if not _same_constraint_group(ms[i], ms[j])
    )


def most_separated_marker(gmm_results: dict) -> tuple | None:
    """
    Return (marker_name, separation_score) for the marker with the
    most pronounced cluster separation. Score is the spread of cluster means
    measured in pooled-std units (Cohen's-d-style). Returns None if no marker
    has at least two components. Derived markers are excluded; iteration is
    sorted so ties resolve deterministically.
    """
    best_name, best_score = None, -1.0
    for name, res in sorted(gmm_results.items()):
        if name in DERIVED_MARKERS or "error" in res or res.get("n_components", 0) < 2:
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
    measured. Returns None if no qualifying pair exists. Derived markers are
    excluded (they correlate with their components by construction); ties break
    deterministically by marker name.
    """
    if df_long.empty:
        return None
    wide = df_long.pivot_table(index="patient_id", columns="test_name", values="value")
    wide = wide.drop(columns=[m for m in DERIVED_MARKERS if m in wide.columns])
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
    # Sort by |r| desc, then by the (a, b) marker-name key so ties are stable
    # regardless of the corr matrix's column order. Skip within-constraint-group
    # pairs (structurally correlated → tautological headline).
    ranked = sorted(flat.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    for (a, b), _ in ranked:
        if not _same_constraint_group(str(a), str(b)):
            return str(a), str(b), float(corr.loc[a, b])
    return None
