"""Plotly chart helpers used by the four tabs.

Each function takes a `data` dict (produced by the analysis layer) and
returns an HTML fragment ready to embed in a template. Display-unit
resolution defers to `web.state._display_unit` so user unit preferences
flow through.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html as plotly_to_html
from scipy.stats import norm as scipy_norm

from thresholds import THRESHOLDS
from unit_conversions import from_canonical, transform_for_display
from web.state import _display_unit

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
CHART_FONT = ("-apple-system, BlinkMacSystemFont, 'Inter', system-ui, "
              "'Helvetica Neue', sans-serif")


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
