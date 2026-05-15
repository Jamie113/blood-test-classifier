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

from analysis import most_separated_marker, strongest_marker_pair
from unit_conversions import transform_for_display

from web.charts import (
    CLUSTER_COLOURS,
    _heatmap_html,
    _marker_chart_html,
    _pair_chart_html,
    _population_scatter_html,
)
from web.filters import FilterSpec
from web.state import (
    _display_unit,
    _filter_ui_context,
    _units_ui_context,
    state,
)


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

    pick = most_separated_marker(data["gmm_results"])
    auto_choice = pick[0] if pick else available[0]

    return {
        "marker":       marker,
        "auto_choice":  auto_choice,
        "is_auto":      marker == auto_choice,
        "n_components": n_comp,
        "n_patients":   data["df_long"]["patient_id"].nunique(),
        "n_markers":    data["df_long"]["test_name"].nunique(),
        "disp_unit":    disp_unit,
        "groups":       groups_sorted,
        "available":    available,
        "chart_html":   _marker_chart_html(data, marker),
        "bic_pairs":    sorted(res["bic_scores"].items()),
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

        members = [pid for pid, lbl in zip(patient_ids, labels) if lbl == g]
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

    rows: list[dict] = []
    for i, pid in enumerate(patient_ids):
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
        "n_clusters": n_clusters,
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
    else:
        r = None

    showing_auto = auto_pair is not None and {x, y} == {auto_pair[0], auto_pair[1]}
    n_overlap = len(wide)

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
        "showing_auto":  showing_auto,
        "strength_word": strength_word,
        "dir_word":      dir_word,
        "track":         track,
        "n_overlap":     n_overlap,
        "chart_html":    _pair_chart_html(data, x, y),
    }


# ── Cross-tab context (left rail + page chrome) ──────────────────────────────

def _common(spec: FilterSpec, data: dict) -> dict:
    n_full = state.df_long_full["patient_id"].nunique()
    df = data["df_long"]
    n_active = df["patient_id"].nunique() if not df.empty else 0
    return {
        "is_demo":         state.is_demo,
        "upload_filename": state.upload_filename,
        "upload_error":    state.last_upload_error,
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
        available = sorted([t for t in data["gmm_results"] if "error" not in data["gmm_results"][t]])
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
