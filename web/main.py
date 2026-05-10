"""FastAPI front-end for the blood-test classifier.

Renders the same insights as the Streamlit app but with full control over
typography, layout, and interactions. Analysis logic is unchanged — imports
from the project's existing modules.
"""
from __future__ import annotations

import functools
import io
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.io import to_html as plotly_to_html
from scipy.stats import chi2, norm as scipy_norm

# Project analysis layer — imported, not duplicated.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import (  # noqa: E402
    analyse_population,
    analyse_upload,
    build_labelled_df,
    filter_long,
    most_separated_marker,
    strongest_marker_pair,
)
from parsing import parse_csv  # noqa: E402
from thresholds import THRESHOLDS  # noqa: E402
from unit_conversions import (  # noqa: E402
    available_units,
    from_canonical,
    transform_for_display,
)

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
CHART_FONT = ("-apple-system, BlinkMacSystemFont, 'Inter', system-ui, "
              "'Helvetica Neue', sans-serif")

VALID_TABS = {"explorer", "population", "investigate", "pairs"}


# ── Filter spec ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarkerFilter:
    marker: str
    lo: float
    hi: float


@dataclass(frozen=True)
class FilterSpec:
    age_min: int | None = None
    age_max: int | None = None
    markers: tuple = ()  # tuple[MarkerFilter, ...]

    def is_active(self) -> bool:
        return (
            self.age_min is not None
            or self.age_max is not None
            or len(self.markers) > 0
        )

    def cache_key(self) -> tuple:
        return (self.age_min, self.age_max, self.markers)

    @classmethod
    def from_request(
        cls,
        age_min: int | None,
        age_max: int | None,
        m: list[str] | None,
    ) -> "FilterSpec":
        markers: list[MarkerFilter] = []
        for s in (m or []):
            try:
                name, lo, hi = s.split(":", 2)
                markers.append(MarkerFilter(name.strip(), float(lo), float(hi)))
            except (ValueError, IndexError):
                continue
        markers.sort(key=lambda mf: mf.marker)
        return cls(age_min=age_min, age_max=age_max, markers=tuple(markers))

    def to_query_params(self) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        if self.age_min is not None:
            params.append(("age_min", str(self.age_min)))
        if self.age_max is not None:
            params.append(("age_max", str(self.age_max)))
        for mf in self.markers:
            params.append(("m", f"{mf.marker}:{mf.lo}:{mf.hi}"))
        return params

    def to_query_string(self) -> str:
        return urlencode(self.to_query_params())

    def without_marker(self, marker: str) -> "FilterSpec":
        return FilterSpec(
            age_min=self.age_min,
            age_max=self.age_max,
            markers=tuple(mf for mf in self.markers if mf.marker != marker),
        )


# ── State ────────────────────────────────────────────────────────────────────

class AppState:
    df_long_full: Any = None
    gmm_results_full: dict = {}
    pop_results_full: dict = {}
    df_labelled_full: Any = None
    is_demo: bool = True
    upload_filename: str | None = None
    unit_prefs: dict[str, str] = {}  # marker → display unit
    last_upload_error: str | None = None


state = AppState()


# Markers with more than one supported display unit.
MULTI_UNIT_MARKERS = sorted(m for m in THRESHOLDS if len(available_units(m)) > 1)


def _load_demo() -> None:
    cache_path = PROJECT_ROOT / "demo_cache.pkl"
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    state.df_long_full     = cached["df_long"]
    state.gmm_results_full = cached["gmm_results"]
    state.pop_results_full = cached["pop_results"]
    state.df_labelled_full = cached["df_labelled"]
    state.is_demo          = True
    _filtered_data_cached.cache_clear()


def _full_data() -> dict:
    return {
        "df_long":     state.df_long_full,
        "gmm_results": state.gmm_results_full,
        "pop_results": state.pop_results_full,
        "df_labelled": state.df_labelled_full,
    }


@functools.lru_cache(maxsize=32)
def _filtered_data_cached(spec: FilterSpec) -> dict:
    """Compute and cache the filtered analysis. Bounded LRU keeps memory in
    check on the 512 MB Render free tier — each entry is ~400 KB. Callers
    must NOT mutate the returned dict; cache hits return the same object."""
    age_range: tuple | None = None
    if spec.age_min is not None or spec.age_max is not None:
        full_range = _age_full_range()
        lo = spec.age_min if spec.age_min is not None else (full_range[0] if full_range else 0)
        hi = spec.age_max if spec.age_max is not None else (full_range[1] if full_range else 200)
        age_range = (lo, hi)

    marker_ranges = {mf.marker: (mf.lo, mf.hi) for mf in spec.markers}

    df_filtered = filter_long(state.df_long_full, age_range=age_range, marker_ranges=marker_ranges)
    n = df_filtered["patient_id"].nunique() if not df_filtered.empty else 0

    if n < 4:
        msg = f"Only {n} blood test{'s' if n != 1 else ''} match the current filters — need at least 4 to run analysis."
        return {
            "error":       msg,
            "df_long":     df_filtered,
            "gmm_results": {},
            "pop_results": {"error": msg},
            "df_labelled": df_filtered,
        }

    gmm = analyse_upload(df_filtered)
    pop = analyse_population(df_filtered)
    return {
        "df_long":     df_filtered,
        "gmm_results": gmm,
        "pop_results": pop,
        "df_labelled": build_labelled_df(df_filtered, gmm),
    }


def _filtered_data(spec: FilterSpec) -> dict:
    """Return a {df_long, gmm_results, pop_results, df_labelled} dict for the
    cohort defined by `spec`."""
    if not spec.is_active():
        return _full_data()
    return _filtered_data_cached(spec)


_load_demo()


def _age_full_range() -> tuple[int, int] | None:
    df = state.df_long_full
    if "age" not in df.columns:
        return None
    ages = pd.to_numeric(df.drop_duplicates("patient_id")["age"], errors="coerce").dropna()
    if ages.empty:
        return None
    return (int(ages.min()), int(ages.max()))


def _marker_value_range(marker: str) -> tuple[float, float] | None:
    df = state.df_long_full
    sub = df[df["test_name"] == marker]
    if sub.empty:
        return None
    return float(sub["value"].min()), float(sub["value"].max())


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Blood Test Classifier")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


# ── Chart styling ────────────────────────────────────────────────────────────

def _base_layout(**overrides: Any) -> dict:
    layout = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family=CHART_FONT, size=12, color="#1f1f1f"),
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        xaxis=dict(gridcolor="#f0f0f0", zerolinecolor="#e0e0e0"),
        yaxis=dict(gridcolor="#f0f0f0", zerolinecolor="#e0e0e0"),
    )
    layout.update(overrides)
    return layout


def _embed(fig: go.Figure) -> str:
    return plotly_to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"displayModeBar": False},
    )


# ── Marker explorer chart + context ─────────────────────────────────────────

def _display_unit(marker: str) -> str:
    """The unit to render `marker` in. Falls back to canonical if the user's
    preference isn't one of this marker's supported units."""
    canonical = THRESHOLDS[marker]["unit"]
    pref = state.unit_prefs.get(marker)
    if pref and pref in available_units(marker):
        return pref
    return canonical


def _marker_chart_html(data: dict, marker: str) -> str:
    res = data["gmm_results"][marker]
    disp_unit = _display_unit(marker)

    values_d, means_d, stds_d, boundaries_d = transform_for_display(
        marker, res["values"], res["means"], res["stds"], res["boundaries"], disp_unit,
    )
    weights = res["weights"]
    rules   = THRESHOLDS[marker]
    ref_nb  = from_canonical(marker, rules["normal"][1],     disp_unit)
    ref_ba  = from_canonical(marker, rules["borderline"][1], disp_unit)

    x = np.linspace(
        values_d.min() - 2 * values_d.std(),
        values_d.max() + 2 * values_d.std(),
        500,
    )

    fig = go.Figure()

    if len(values_d) >= 30:
        fig.add_trace(go.Histogram(
            x=values_d, histnorm="probability density",
            nbinsx=20, opacity=0.18,
            marker_color="#4C72B0", name="Your tests",
            marker_line_width=0,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=values_d, y=np.zeros(len(values_d)),
            mode="markers",
            marker=dict(symbol="line-ns", size=16, line=dict(width=2, color="#1f1f1f")),
            name="Test values",
        ))

    for i, (m, s, w) in enumerate(zip(means_d, stds_d, weights)):
        colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
        fig.add_trace(go.Scatter(
            x=x, y=w * scipy_norm.pdf(x, m, s),
            mode="lines", line=dict(color=colour, width=2.5),
            name=f"Group {i + 1} · avg {m:.2f}",
        ))

    total = sum(w * scipy_norm.pdf(x, m, s) for m, s, w in zip(means_d, stds_d, weights))
    fig.add_trace(go.Scatter(
        x=x, y=total,
        mode="lines", line=dict(color="#1f1f1f", width=1, dash="dash"),
        opacity=0.35, name="Combined",
    ))

    for b in boundaries_d:
        fig.add_vline(
            x=b, line=dict(color="#9a9a9a", width=1, dash="dash"),
            annotation_text=f"boundary {b:.2f}",
            annotation_position="top",
            annotation_font=dict(size=10, color="#6b6b6b"),
        )
    fig.add_vline(
        x=ref_nb, line=dict(color="#7aa37a", width=1, dash="dot"),
        annotation_text=f"ref {ref_nb:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#5a8a5a"),
    )
    fig.add_vline(
        x=ref_ba, line=dict(color="#c98a8a", width=1, dash="dot"),
        annotation_text=f"upper ref {ref_ba:.2f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#a06060"),
    )

    fig.update_layout(**_base_layout(
        xaxis_title=f"{marker} ({disp_unit})",
        yaxis_title="Density",
        yaxis=dict(rangemode="tozero", gridcolor="#f0f0f0"),
        height=420,
    ))
    return _embed(fig)


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


# ── Population chart + context ──────────────────────────────────────────────

def _population_scatter_html(data: dict, colour_by: str = "type") -> str:
    pop = data["pop_results"]
    df  = data["df_long"]
    X2          = pop["X_pca_2d"]
    labels      = pop["labels"]
    patient_ids = pop["patient_ids"]
    n_clusters  = pop["n_clusters"]
    var1        = pop["pca_var"][0] * 100
    var2        = pop["pca_var"][1] * 100

    age_lookup = (
        df.drop_duplicates("patient_id").set_index("patient_id")["age"].to_dict()
        if "age" in df.columns else {}
    )

    fig = go.Figure()

    if colour_by == "age" and age_lookup:
        ages = [age_lookup.get(pid) for pid in patient_ids]
        fig.add_trace(go.Scatter(
            x=X2[:, 0], y=X2[:, 1], mode="markers",
            marker=dict(size=11, color=ages, colorscale="RdYlBu_r",
                        showscale=True, colorbar=dict(title="Age", thickness=10)),
            text=patient_ids,
            customdata=[
                f"{pid}" + (f" · Age {age_lookup[pid]}"
                            if age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid))
                            else "")
                for pid in patient_ids
            ],
            hovertemplate="<b>%{customdata}</b><extra></extra>",
            showlegend=False,
        ))
    else:
        for g in range(n_clusters):
            mask = labels == g
            fig.add_trace(go.Scatter(
                x=X2[mask, 0], y=X2[mask, 1], mode="markers",
                marker=dict(size=11, color=CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]),
                name=f"Cluster {g + 1}",
                text=[pid for pid, m in zip(patient_ids, mask) if m],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    fig.update_layout(**_base_layout(
        xaxis_title=f"Similarity axis 1 ({var1:.1f}% of variation)",
        yaxis_title=f"Similarity axis 2 ({var2:.1f}% of variation)",
        height=480,
    ))
    return _embed(fig)


def _heatmap_html(data: dict) -> str:
    fp = data["pop_results"]["fingerprint"].round(2)
    fp = fp.rename(columns={c: c.replace("Group ", "Cluster ") for c in fp.columns})
    fig = go.Figure(go.Heatmap(
        z=fp.values, x=fp.columns.tolist(), y=fp.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-2, zmax=2,
        text=fp.values.round(2), texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="vs avg", thickness=10),
    ))
    fig.update_layout(**_base_layout(
        height=max(320, len(fp.index) * 22),
        margin=dict(l=160, r=40, t=20, b=40),
        xaxis=dict(side="top", gridcolor="#f0f0f0"),
        yaxis=dict(autorange="reversed", gridcolor="#f0f0f0"),
    ))
    return _embed(fig)


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


# ── Investigate (boundary + outlier) ─────────────────────────────────────────

def _investigate_context(data: dict) -> dict:
    """Identify boundary patients (low max posterior) and multivariate outliers
    (low log-likelihood). Returned dict has either {error: ...} or the rows."""
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


# ── Pair chart + context ────────────────────────────────────────────────────

def _pair_chart_html(data: dict, x_marker: str, y_marker: str, colour_by: str = "type") -> str:
    df = data["df_long"]
    wide = (
        df[df["test_name"].isin([x_marker, y_marker])]
        .pivot_table(index="patient_id", columns="test_name", values="value")
        .dropna()
    )
    disp_x = _display_unit(x_marker)
    disp_y = _display_unit(y_marker)
    xs = wide[x_marker].apply(lambda v: from_canonical(x_marker, v, disp_x))
    ys = wide[y_marker].apply(lambda v: from_canonical(y_marker, v, disp_y))

    pop = data["pop_results"]
    pop_label_lookup = (
        dict(zip(pop["patient_ids"], pop["labels"]))
        if "labels" in pop else {}
    )

    fig = go.Figure()
    if colour_by == "type" and pop_label_lookup:
        n_clusters_p = pop["n_clusters"]
        for g in range(n_clusters_p):
            mask = [pop_label_lookup.get(pid) == g for pid in wide.index]
            if not any(mask):
                continue
            fig.add_trace(go.Scatter(
                x=xs[mask], y=ys[mask], mode="markers",
                marker=dict(size=10, color=CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]),
                name=f"Cluster {g + 1}",
                text=[pid for pid, m in zip(wide.index, mask) if m],
                hovertemplate=(
                    f"<b>%{{text}}</b><br>{x_marker}: %{{x:.2f}} {disp_x}"
                    f"<br>{y_marker}: %{{y:.2f}} {disp_y}<extra></extra>"
                ),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=10, color=CLUSTER_COLOURS[0]),
            text=list(wide.index),
            hovertemplate=(
                f"<b>%{{text}}</b><br>{x_marker}: %{{x:.2f}} {disp_x}"
                f"<br>{y_marker}: %{{y:.2f}} {disp_y}<extra></extra>"
            ),
            showlegend=False,
        ))

    ref_x = from_canonical(x_marker, THRESHOLDS[x_marker]["normal"][1], disp_x)
    ref_y = from_canonical(y_marker, THRESHOLDS[y_marker]["normal"][1], disp_y)
    fig.add_vline(x=ref_x, line=dict(color="#7aa37a", width=1, dash="dot"),
                  annotation_text=f"{x_marker} ref",
                  annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#5a8a5a"))
    fig.add_hline(y=ref_y, line=dict(color="#7aa37a", width=1, dash="dot"),
                  annotation_text=f"{y_marker} ref",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#5a8a5a"))

    fig.update_layout(**_base_layout(
        xaxis_title=f"{x_marker} ({disp_x})",
        yaxis_title=f"{y_marker} ({disp_y})",
        height=500,
    ))
    return _embed(fig)


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


# ── Filter UI context ────────────────────────────────────────────────────────

def _filter_ui_context(spec: FilterSpec) -> dict:
    age_full = _age_full_range()
    active_marker_names = {mf.marker for mf in spec.markers}
    addable_markers = [
        m for m in sorted(state.df_long_full["test_name"].unique())
        if m not in active_marker_names
    ]

    filter_chips = []
    for mf in spec.markers:
        full = _marker_value_range(mf.marker)
        unit = _display_unit(mf.marker) if mf.marker in THRESHOLDS else ""
        without = spec.without_marker(mf.marker)
        filter_chips.append({
            "marker":  mf.marker,
            "lo":      mf.lo,
            "hi":      mf.hi,
            "full_lo": full[0] if full else mf.lo,
            "full_hi": full[1] if full else mf.hi,
            "unit":    unit,
            "remove_qs": without.to_query_string(),
        })

    return {
        "age_full":        age_full,
        "addable_markers": addable_markers,
        "filter_chips":    filter_chips,
    }


def _units_ui_context() -> dict:
    """Per-marker display unit options. Only markers with > 1 supported unit
    AND a value in the current cohort are shown."""
    present = set(state.df_long_full["test_name"].unique()) if state.df_long_full is not None else set()
    rows: list[dict] = []
    for marker in MULTI_UNIT_MARKERS:
        if marker not in present:
            continue
        rows.append({
            "marker":  marker,
            "options": available_units(marker),
            "current": _display_unit(marker),
        })
    return {
        "unit_options":      rows,
        "any_unit_override": bool(state.unit_prefs),
        "n_unit_overrides":  len(state.unit_prefs),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


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


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    if tab not in VALID_TABS:
        tab = "explorer"
    spec = FilterSpec.from_request(age_min, age_max, m)
    data = _filtered_data(spec)
    ctx = {"active": tab, **_common(spec, data), **_build_tab_ctx(data, tab)}
    return templates.TemplateResponse(request, "index.html", ctx)


@app.get("/tab/{name}", response_class=HTMLResponse)
def tab_partial(
    request: Request,
    name: str,
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    if name not in VALID_TABS:
        name = "explorer"
    spec = FilterSpec.from_request(age_min, age_max, m)
    data = _filtered_data(spec)
    ctx = {"active": name, **_common(spec, data), **_build_tab_ctx(data, name)}
    return templates.TemplateResponse(request, "partials/page_body.html", ctx)


@app.get("/marker", response_class=HTMLResponse)
def marker_partial(
    request: Request,
    name: str,
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    spec = FilterSpec.from_request(age_min, age_max, m)
    data = _filtered_data(spec)
    explorer = _marker_context(data, name) if not data.get("error") else None
    ctx: dict = {"active": "explorer", **_common(spec, data)}
    if explorer:
        ctx["explorer"] = explorer
    return templates.TemplateResponse(request, "partials/page_body.html", ctx)


@app.get("/population/scatter", response_class=HTMLResponse)
def population_scatter_partial(
    request: Request,
    colour_by: str = "type",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    if colour_by not in {"type", "age"}:
        colour_by = "type"
    spec = FilterSpec.from_request(age_min, age_max, m)
    data = _filtered_data(spec)
    if data.get("error"):
        return HTMLResponse("")
    ctx = {"active": "population", **_common(spec, data),
           "pop": _population_context(data, colour_by)}
    return templates.TemplateResponse(request, "partials/_population_scatter.html", ctx)


@app.get("/pair", response_class=HTMLResponse)
def pair_partial(
    request: Request,
    x: str | None = None,
    y: str | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    spec = FilterSpec.from_request(age_min, age_max, m)
    data = _filtered_data(spec)
    pair = _pair_context(data, x, y) if not data.get("error") else None
    ctx: dict = {"active": "pairs", **_common(spec, data)}
    if pair:
        ctx["pair"] = pair
    return templates.TemplateResponse(request, "partials/page_body.html", ctx)


def _normalise_age(age_min: int | None, age_max: int | None) -> tuple[int | None, int | None]:
    """Treat (full_min, full_max) as 'no age filter' — the form's default state."""
    full = _age_full_range()
    if full is None:
        return None, None
    if age_min is not None and age_max is not None \
            and age_min <= full[0] and age_max >= full[1]:
        return None, None
    return age_min, age_max


def _resolve_tab(tab: str) -> str:
    return tab if tab in VALID_TABS else "explorer"


def _filter_response(
    request: Request, spec: FilterSpec, tab: str, full: bool,
) -> HTMLResponse:
    """Build the response for a filter mutation. `full=True` means OOB swap the
    rail filter section as well as page-body; otherwise just page-body."""
    data = _filtered_data(spec)
    tab = _resolve_tab(tab)
    ctx = {"active": tab, **_common(spec, data), **_build_tab_ctx(data, tab)}
    template = "partials/full_render.html" if full else "partials/page_body.html"
    return templates.TemplateResponse(request, template, ctx)


@app.get("/filters/set", response_class=HTMLResponse)
def set_filter_partial(
    request: Request,
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    age_min, age_max = _normalise_age(age_min, age_max)
    spec = FilterSpec.from_request(age_min, age_max, m)
    return _filter_response(request, spec, tab, full=False)


@app.get("/filters/add", response_class=HTMLResponse)
def add_filter_partial(
    request: Request,
    marker: str = "",
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    age_min, age_max = _normalise_age(age_min, age_max)
    new_m = list(m)
    if marker and marker in state.df_long_full["test_name"].unique():
        rng = _marker_value_range(marker)
        if rng is not None:
            new_m.append(f"{marker}:{rng[0]}:{rng[1]}")
    spec = FilterSpec.from_request(age_min, age_max, new_m)
    return _filter_response(request, spec, tab, full=True)


@app.get("/filters/remove", response_class=HTMLResponse)
def remove_filter_partial(
    request: Request,
    marker: str,
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    age_min, age_max = _normalise_age(age_min, age_max)
    spec = FilterSpec.from_request(age_min, age_max, m).without_marker(marker)
    return _filter_response(request, spec, tab, full=True)


@app.get("/filters/reset", response_class=HTMLResponse)
def reset_filters_partial(request: Request, tab: str = "explorer") -> HTMLResponse:
    return _filter_response(request, FilterSpec(), tab, full=True)


# ── Upload ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> Response:
    """Parse the uploaded CSV, run the full analysis, and replace the demo state.

    Returns an HX-Redirect so HTMX reloads the page (clears all in-page state
    cleanly). On parse failure, returns an HTML fragment for inline display.
    """
    contents = await file.read()
    try:
        df_long, recognised, unrecognised = parse_csv(io.BytesIO(contents))
    except Exception as exc:  # noqa: BLE001
        state.last_upload_error = f"Could not read CSV: {exc}"
        return Response(
            content=f'<div class="cohort-error">Could not read CSV: {exc}</div>',
            status_code=400, media_type="text/html",
        )

    if df_long.empty:
        state.last_upload_error = "No recognised columns in the upload."
        return Response(
            content='<div class="cohort-error">No recognised columns in the upload.</div>',
            status_code=400, media_type="text/html",
        )

    n_patients = df_long["patient_id"].nunique()
    if n_patients < 4:
        state.last_upload_error = (
            f"Only {n_patients} blood tests in the upload — need at least 4 to run analysis."
        )
        return Response(
            content=f'<div class="cohort-error">{state.last_upload_error}</div>',
            status_code=400, media_type="text/html",
        )

    gmm = analyse_upload(df_long)
    pop = analyse_population(df_long)
    df_labelled = build_labelled_df(df_long, gmm)

    state.df_long_full     = df_long
    state.gmm_results_full = gmm
    state.pop_results_full = pop
    state.df_labelled_full = df_labelled
    state.is_demo          = False
    state.upload_filename  = file.filename
    state.unit_prefs       = {}
    state.last_upload_error = None
    _filtered_data_cached.cache_clear()

    return Response(headers={"HX-Redirect": "/"})


@app.post("/upload/reset")
def upload_reset() -> Response:
    """Discard the uploaded data and return to demo mode."""
    _load_demo()
    state.unit_prefs = {}
    state.upload_filename = None
    state.last_upload_error = None
    return Response(headers={"HX-Redirect": "/"})


# ── Display units ────────────────────────────────────────────────────────────

@app.get("/units/set", response_class=HTMLResponse)
def set_units(
    request: Request,
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
    u: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    """Update display unit preferences. Each `u` item is `marker:unit`."""
    new_prefs = dict(state.unit_prefs)
    for s in u:
        try:
            marker, unit = s.split(":", 1)
        except ValueError:
            continue
        marker = marker.strip()
        unit = unit.strip()
        if marker in MULTI_UNIT_MARKERS and unit in available_units(marker):
            if unit == THRESHOLDS[marker]["unit"]:
                new_prefs.pop(marker, None)  # canonical = no override
            else:
                new_prefs[marker] = unit
    state.unit_prefs = new_prefs

    age_min, age_max = _normalise_age(age_min, age_max)
    spec = FilterSpec.from_request(age_min, age_max, m)
    return _filter_response(request, spec, tab, full=True)
