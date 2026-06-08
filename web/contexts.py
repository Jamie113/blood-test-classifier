"""Per-tab Jinja2 context builders.

Each `_*_context` function consumes a `data` dict (from the analysis layer)
and returns the variables a template needs to render. Charts are pre-rendered
to HTML here; templates just embed the strings. `_common` assembles the
cross-tab context (filters, units, banner state), and `_build_tab_ctx`
dispatches on the active tab.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from web.charts import (
    CLUSTER_COLOURS,
    _heatmap_html,
    _marker_chart_html,
    _pair_chart_html,
    _population_scatter_html,
)
from web.filters import FilterSpec
from web.state import (
    _cohort_count,
    _display_unit,
    _filter_ui_context,
    _units_ui_context,
    state,
)

from analysis import (
    HEAVY_IMPUTE_FRAC,
    most_separated_marker,
    n_comparable_pairs,
    strongest_marker_pair,
)
from unit_conversions import transform_for_display

# How decisively a K>1 split beat the K=1 null. ΔBIC ≥ 6 is "strong evidence";
# a margin only just past that floor is reported as tentative, not asserted.
_DELTA_BIC_TENTATIVE = 10.0

# Correlation robustness (M3): |Pearson − Spearman| above this signals an
# outlier / non-linear shape; a "strong" claim needs at least this many paired
# points (r from ~10 points has a very wide confidence interval).
_CORR_DIVERGENCE = 0.2
_STRONG_MIN_OVERLAP = 25


def _delta_bic(bic_scores: dict, n_chosen: int) -> float | None:
    """ΔBIC of the chosen K vs the K=1 null. None when K=1 was chosen."""
    if n_chosen <= 1 or 1 not in bic_scores or n_chosen not in bic_scores:
        return None
    return bic_scores[1] - bic_scores[n_chosen]


# ── Marker explorer (Groups tab) ─────────────────────────────────────────────

def _marker_context(data: dict, marker: str) -> dict | None:
    available = sorted([t for t in data["gmm_results"] if "error" not in data["gmm_results"][t]])
    if not available:
        return None
    if marker not in available:
        marker = available[0]

    res = data["gmm_results"][marker]
    disp_unit = _display_unit(marker)
    values_d, means_d, stds_d, _ = transform_for_display(
        marker, res["values"], res["means"], res["stds"], res["boundaries"], disp_unit,
    )
    weights = res["weights"]
    n_comp  = res["n_components"]

    groups_sorted = sorted(
        [
            {
                "name":    f"Group {i + 1}",
                "mean":    float(means_d[i]),
                "std":     float(stds_d[i]),
                "weight":  float(weights[i]),
                "count":   int((res["labels"] == i).sum()),
                "percent": float((res["labels"] == i).mean() * 100),
                "colour":  CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)],
            }
            for i in range(n_comp)
        ],
        key=lambda g: -g["weight"],
    )

    return {
        "marker":       marker,
        "n_components": n_comp,
        "n_patients":   data["df_long"]["patient_id"].nunique(),
        "n_markers":    data["df_long"]["test_name"].nunique(),
        "disp_unit":    disp_unit,
        "groups":       groups_sorted,
        "available":    available,
        "chart_html":   _marker_chart_html(data, marker),
        "bic_pairs":    sorted(res["bic_scores"].items()),
        "delta_bic":    _delta_bic(res["bic_scores"], n_comp),
        "tentative":    (d := _delta_bic(res["bic_scores"], n_comp)) is not None
                        and d < _DELTA_BIC_TENTATIVE,
        "small_sample": res["small_sample"],
    }


# ── Population (Clusters tab) ────────────────────────────────────────────────

def _population_context(data: dict, colour_by: str = "type") -> dict:
    pop = data["pop_results"]
    if "error" in pop:
        return {"error": pop["error"]}

    n_patients = len(pop["patient_ids"])
    n_clusters = pop["n_clusters"]
    labels     = pop["labels"]
    patient_ids = pop["patient_ids"]
    df = data["df_long"]

    age_lookup = (
        df.drop_duplicates("patient_id").set_index("patient_id")["age"].to_dict()
        if "age" in df.columns else {}
    )
    has_age = bool(age_lookup) and any(
        age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid))
        for pid in patient_ids
    )

    fp = pop["fingerprint"].round(2)

    types: list[dict] = []
    sub_summaries: list[str] = []
    for g in range(n_clusters):
        col = fp[f"Group {g + 1}"]
        col_sorted = col.abs().sort_values(ascending=False)
        bullets = []
        for marker in col_sorted.index:
            z = float(col[marker])
            if abs(z) < 0.3:
                break
            direction = "higher" if z > 0 else "lower"
            strength = (
                "notably" if abs(z) > 1.0 else
                "clearly" if abs(z) > 0.6 else
                "slightly"
            )
            bullets.append({
                "marker": marker, "direction": direction,
                "strength": strength, "z": z,
            })
            if len(bullets) >= 5:
                break

        members = [pid for pid, lbl in zip(patient_ids, labels, strict=True) if lbl == g]
        ages_g = [
            age_lookup.get(pid) for pid in members
            if age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid))
        ]
        median_age = int(np.median(ages_g)) if ages_g else None
        age_range = (int(min(ages_g)), int(max(ages_g))) if ages_g else None

        types.append({
            "name":       f"Cluster {g + 1}",
            "colour":     CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)],
            "count":      len(members),
            "median_age": median_age,
            "age_range":  age_range,
            "bullets":    bullets,
        })

        top_marker = col_sorted.index[0] if len(col_sorted) else None
        if top_marker is not None:
            top_z = float(col[top_marker])
            top_dir = "higher" if top_z > 0 else "lower"
            age_bit = f", median age {median_age}" if median_age is not None else ""
            sub_summaries.append(
                f"<strong>Cluster {g + 1}</strong> ({len(members)} tests{age_bit}) "
                f"— most distinct in <strong>{top_marker}</strong> ({top_dir} than average)"
            )

    return {
        "n_patients":     n_patients,
        "n_clusters":     n_clusters,
        "n_markers":      pop["df_wide"].shape[1],
        "type_word":      "cluster" if n_clusters == 1 else "distinct clusters",
        "types":          types,
        "sub_summaries":  sub_summaries,
        "scatter_html":   _population_scatter_html(data, colour_by),
        "heatmap_html":   _heatmap_html(data),
        "colour_by":      colour_by,
        "has_age":        has_age,
        "small_sample":   pop["small_sample"],
        "bic_pairs":      sorted(pop["bic_scores"].items()),
        "delta_bic":      _delta_bic(pop["bic_scores"], n_clusters),
        "tentative":      (d := _delta_bic(pop["bic_scores"], n_clusters)) is not None
                          and d < _DELTA_BIC_TENTATIVE,
        "var_pct":        int(pop["pca_var"][:pop["n_cluster_dims"]].sum() * 100),
        "n_cluster_dims": pop["n_cluster_dims"],
    }


# ── Investigate (Outliers tab) ───────────────────────────────────────────────

def _investigate_context(data: dict) -> dict:
    """Identify boundary tests (max posterior < 0.7) and multivariate outliers
    (Mahalanobis² > χ²₀.₉₉ in the cluster-dim PCA space). Returned dict has
    either {error: ...} or the rows."""
    pop = data["pop_results"]
    if "error" in pop:
        return {"error": pop["error"]}

    df = data["df_long"]
    patient_ids = pop["patient_ids"]
    labels      = pop["labels"]
    posts       = pop["posteriors"]
    mahal_sq    = pop["mahalanobis_sq"]
    n_patients  = len(patient_ids)
    n_clusters  = pop["n_clusters"]
    n_cluster_dims = pop["n_cluster_dims"]
    # Patients whose profile is mostly imputed sit near the cohort median by
    # construction, so their cluster placement and Mahalanobis distance are
    # unreliable — we can't honestly flag (or clear) them. Set them aside and
    # report the count instead of shipping a misleading flag.
    imputed_frac = pop.get("imputed_frac")
    if imputed_frac is None:
        imputed_frac = np.zeros(n_patients)

    age_lookup = (
        df.drop_duplicates("patient_id").set_index("patient_id")["age"].to_dict()
        if "age" in df.columns else {}
    )

    max_post = posts.max(axis=1)
    # χ² critical value at p = 0.01 with df = n_cluster_dims. Tests whose
    # squared Mahalanobis distance to their assigned cluster's mean exceeds
    # this threshold are flagged as multivariate outliers. This is principled
    # (an absolute threshold from the assumed Gaussian model) rather than the
    # quantile-based 5%-of-cohort rule, which always flagged some tests.
    outlier_thresh = float(chi2.ppf(0.99, df=n_cluster_dims))

    # Mostly median-filled → the row is mostly noise and its placement is
    # unreliable, so we don't flag (or clear) it. Shared threshold with the
    # cluster fingerprint (analysis.HEAVY_IMPUTE_FRAC); partially-imputed rows
    # below it are annotated rather than gated.
    n_low_data = int((imputed_frac > HEAVY_IMPUTE_FRAC).sum())
    n_assessed = n_patients - n_low_data

    rows: list[dict] = []
    for i, pid in enumerate(patient_ids):
        if imputed_frac[i] > HEAVY_IMPUTE_FRAC:
            continue  # too little real data to flag or clear; counted in n_low_data
        reasons: list[str] = []
        tags: list[str] = []
        is_boundary = max_post[i] < 0.7 and posts.shape[1] >= 2
        is_outlier  = mahal_sq[i] > outlier_thresh
        if is_boundary:
            a, b = np.argsort(posts[i])[::-1][:2]
            reasons.append(
                f"{posts[i, a] * 100:.0f}% Cluster {a + 1} · "
                f"{posts[i, b] * 100:.0f}% Cluster {b + 1}"
            )
            tags.append("borderline")
        if is_outlier:
            reasons.append("unusual overall profile across all markers")
            tags.append("outlier")
        if reasons:
            age = age_lookup.get(pid)
            rows.append({
                "patient":     pid,
                "age":         int(age) if age is not None and pd.notna(age) else "—",
                "cluster":     f"Cluster {labels[i] + 1}",
                "type_colour": CLUSTER_COLOURS[labels[i] % len(CLUSTER_COLOURS)],
                "confidence":  int(round(max_post[i] * 100)),
                "why":         " · ".join(reasons),
                "tags":        tags,
                "is_boundary": is_boundary,
                "is_outlier":  is_outlier,
                "imputed_pct": int(round(imputed_frac[i] * 100)),
            })
    rows.sort(key=lambda r: r["confidence"])

    n_boundary = sum(1 for r in rows if r["is_boundary"])
    n_outlier  = sum(1 for r in rows if r["is_outlier"])

    return {
        "rows":       rows,
        "n_flagged":  len(rows),
        "n_boundary": n_boundary,
        "n_outlier":  n_outlier,
        "n_patients": n_patients,
        "n_assessed": n_assessed,
        "n_clusters": n_clusters,
        "n_low_data": n_low_data,
    }


# ── Pair (Correlations tab) ──────────────────────────────────────────────────

def _pair_context(data: dict, x: str | None, y: str | None) -> dict:
    df = data["df_long"]
    markers_in_data = sorted(df["test_name"].unique())
    if not markers_in_data:
        return {"error": "No markers in cohort."}

    auto_pair = strongest_marker_pair(df)
    if x is None or x not in markers_in_data:
        x = auto_pair[0] if auto_pair else markers_in_data[0]
    if y is None or y not in markers_in_data or y == x:
        y = (
            auto_pair[1] if auto_pair and auto_pair[1] != x else
            next((m for m in markers_in_data if m != x), markers_in_data[0])
        )

    wide = (
        df[df["test_name"].isin([x, y])]
        .pivot_table(index="patient_id", columns="test_name", values="value")
        .dropna()
    )
    if len(wide) >= 4 and x in wide.columns and y in wide.columns:
        r = float(wide[x].corr(wide[y]))
        rho = float(wide[x].corr(wide[y], method="spearman"))   # rank, outlier-robust
    else:
        r = rho = None

    showing_auto = auto_pair is not None and {x, y} == {auto_pair[0], auto_pair[1]}
    n_overlap = len(wide)
    # Robustness signals (M3): a large Pearson↔Spearman gap means a single
    # bivariate outlier or a non-linear shape is inflating/deflating Pearson;
    # and a "strong" claim needs more than the 10-point scan floor to be trusted.
    diverges = r is not None and rho is not None and abs(r - rho) > _CORR_DIVERGENCE
    low_overlap = r is not None and n_overlap < _STRONG_MIN_OVERLAP
    # Size of the search the auto-pick won, so the UI can frame it as the
    # strongest of many candidate pairs rather than a singular discovery. This
    # is the candidate count (before the min-overlap filter), labelled as such.
    n_pairs_searched = n_comparable_pairs(markers_in_data)

    if r is not None:
        strength_word = (
            "strong" if abs(r) > 0.6 else
            "moderate" if abs(r) > 0.3 else
            "weak"
        )
        dir_word = "positive" if r > 0 else "negative"
        track = "track each other" if abs(r) > 0.3 else "show little relationship"
    else:
        strength_word = dir_word = track = None

    return {
        "markers":       markers_in_data,
        "x":             x,
        "y":             y,
        "y_options":     [m for m in markers_in_data if m != x],
        "r":             r,
        "rho":           rho,
        "diverges":      diverges,
        "low_overlap":   low_overlap,
        "showing_auto":  showing_auto,
        "n_pairs_searched": n_pairs_searched,
        "strength_word": strength_word,
        "dir_word":      dir_word,
        "track":         track,
        "n_overlap":     n_overlap,
        "chart_html":    _pair_chart_html(data, x, y),
    }


# ── Cross-tab context (left rail + page chrome) ──────────────────────────────

def _cohort_context(spec: FilterSpec) -> dict:
    """Context for the cohort popover alone — the chips, count, and add control,
    WITHOUT a GMM refit. Used by the fast-path add route so revealing a new
    marker's range editor is instant (a full-range filter changes nothing)."""
    return {
        "filters":    spec,
        "filter_qs":  spec.to_query_string(),
        "filtered":   spec.is_active(),
        "n_full":     state.df_long_full["patient_id"].nunique(),
        "n_active":   _cohort_count(spec),
        **_filter_ui_context(spec),
    }


def _common(spec: FilterSpec, data: dict) -> dict:
    n_full = state.df_long_full["patient_id"].nunique()
    df = data["df_long"]
    n_active = df["patient_id"].nunique() if not df.empty else 0
    return {
        "is_demo":         state.is_demo,
        "upload_filename": state.upload_filename,
        "upload_unit_report": state.upload_unit_report,
        "filters":         spec,
        "filter_qs":       spec.to_query_string(),
        "filtered":        spec.is_active(),
        "n_full":          n_full,
        "n_active":        n_active,
        "n_patients":      n_active,
        "n_markers":       df["test_name"].nunique() if not df.empty else 0,
        "data_error":      data.get("error"),
        **_filter_ui_context(spec),
        **_units_ui_context(),
    }


def _build_tab_ctx(data: dict, tab: str) -> dict:
    if data.get("error"):
        return {}
    if tab == "explorer":
        pick = most_separated_marker(data["gmm_results"])
        available = sorted(
            t for t in data["gmm_results"] if "error" not in data["gmm_results"][t]
        )
        initial = pick[0] if pick else (available[0] if available else None)
        if initial is None:
            return {}
        ctx = _marker_context(data, initial)
        return {"explorer": ctx} if ctx else {}
    if tab == "population":
        return {"pop": _population_context(data)}
    if tab == "investigate":
        return {"inv": _investigate_context(data)}
    if tab == "pairs":
        return {"pair": _pair_context(data, None, None)}
    return {}
