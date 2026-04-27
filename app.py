import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm as scipy_norm
import plotly.graph_objects as go

from thresholds import THRESHOLDS
from column_map import COLUMN_MAP, ID_COLUMN, AGE_COLUMN
from unit_conversions import (
    to_canonical,
    available_units, from_canonical, transform_for_display,
)
from gmm import sort_gmm, get_boundaries, assign_clusters
from stub_data import generate_stub_data
from analysis import analyse_upload, analyse_population, build_labelled_df

@st.cache_data
def load_stub_data():
    return generate_stub_data()

st.set_page_config(page_title="Classifier", layout="wide")

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── Plain-English helpers ─────────────────────────────────────────────────────

def sample_size_label(n: int) -> str:
    if n < 30:
        return "🔴 Small sample — treat findings as a starting point"
    if n < 100:
        return "🟡 Moderate sample — per-marker groups are forming"
    if n < 200:
        return "🟢 Good sample — per-marker groups are reliable"
    return "🟢 Large sample — all findings are reliable"


def marker_plain_english(
    test_name: str, n_comp: int, means_d: np.ndarray,
    weights: np.ndarray, disp_unit: str, ref_nb: float, ref_ba: float,
) -> str:
    """One-paragraph plain English summary of a marker's group structure."""
    groups = sorted(
        [(f"Group {i + 1}", float(means_d[i]), float(weights[i]))
         for i in range(n_comp)],
        key=lambda x: x[2], reverse=True,
    )
    group_parts = ", ".join(
        f"**{name}** ({w * 100:.0f}% of patients, average {m:.2f} {disp_unit})"
        for name, m, w in groups
    )
    summary = (
        f"**{test_name}** shows **{n_comp} natural {'group' if n_comp == 1 else 'groups'}** "
        f"in your population: {group_parts}."
    )
    # Add reference range context if the highest group exceeds normal
    highest_mean = float(means_d[-1])
    if highest_mean > ref_nb:
        summary += (
            f" The higher group sits above the standard reference threshold "
            f"({ref_nb:.2f} {disp_unit}) — worth investigating what distinguishes those patients."
        )
    elif highest_mean < ref_ba and n_comp > 1:
        summary += " All groups fall within the standard reference range."
    return summary


def group_plain_english(fp: pd.DataFrame, group_col: str, top_n: int = 4) -> list[str]:
    """Ranked plain-English bullets describing what defines a population group."""
    col = fp[group_col]
    col_sorted = col.abs().sort_values(ascending=False)
    lines = []
    for marker in col_sorted.index:
        z = col[marker]
        if abs(z) < 0.3:
            break
        direction = "higher than average" if z > 0 else "lower than average"
        strength  = "notably" if abs(z) > 1.0 else "slightly"
        lines.append(f"- **{marker}**: {strength} {direction}")
        if len(lines) >= top_n:
            break
    return lines


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_upload(uploaded_file) -> tuple:
    """Wide-format CSV → long-format DataFrame with canonical units."""
    df_raw = pd.read_csv(uploaded_file)
    df_raw = df_raw[
        df_raw[ID_COLUMN].notna() &
        (df_raw[ID_COLUMN].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    has_age = AGE_COLUMN in df_raw.columns

    # Build long-format column-by-column (much faster than iterrows)
    frames = []
    seen_tests: set = set()
    for col, mapping in COLUMN_MAP.items():
        if col not in df_raw.columns:
            continue
        test_name = mapping["test"]
        if test_name in seen_tests:
            continue  # skip duplicate column mappings (e.g. Haematocrit ADJ)
        seen_tests.add(test_name)

        sub = df_raw[[ID_COLUMN, col]].copy()
        if has_age:
            sub[AGE_COLUMN] = df_raw[AGE_COLUMN]
        sub = sub.dropna(subset=[col]).reset_index(drop=True)
        if sub.empty:
            continue

        sub["patient_id"] = sub[ID_COLUMN].astype(str)
        sub["age"] = (
            pd.to_numeric(sub[AGE_COLUMN], errors="coerce").astype("Int64")
            if has_age else None
        )
        sub["value"] = sub[col].astype(float) * mapping["scale"]
        sub["value"] = sub["value"].apply(lambda v: to_canonical(test_name, v))
        sub["test_name"] = test_name
        sub["unit"] = THRESHOLDS[test_name]["unit"]
        frames.append(sub[["patient_id", "age", "test_name", "value", "unit"]])

    if frames:
        df_long = pd.concat(frames, ignore_index=True)
    else:
        df_long = pd.DataFrame(columns=["patient_id", "age", "test_name", "value", "unit"])

    recognised   = [c for c in COLUMN_MAP if c in df_raw.columns]
    unrecognised = [c for c in df_raw.columns if c not in COLUMN_MAP and c != ID_COLUMN]
    return df_long, recognised, unrecognised


# ── Header ────────────────────────────────────────────────────────────────────

header_left, header_right = st.columns([2, 1])

with header_left:
    st.title("Classifier")
    st.caption(
        "Discovers natural patterns in blood test results using unsupervised machine learning."
    )

with header_right:
    st.write("")  # nudge upward to align with title
    uploaded = st.file_uploader(
        "Upload blood test CSV", type="csv", key="upload",
        label_visibility="collapsed",
        help="Wide-format export — one row per patient.",
    )
    if uploaded:
        st.caption(f"📄 {uploaded.name}")

# ── Data loading (runs once per file, above tabs) ─────────────────────────────

MULTI_UNIT_MARKERS = [m for m in THRESHOLDS if len(available_units(m)) > 1]

if uploaded:
    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Reading your data and finding groups…"):
            df_long, recognised, unrecognised = parse_upload(uploaded)
            gmm_results = analyse_upload(df_long)
            pop_results = analyse_population(df_long)
            df_labelled = build_labelled_df(df_long, gmm_results)
        st.session_state.update({
            "df_long":      df_long,
            "gmm_results":  gmm_results,
            "pop_results":  pop_results,
            "df_labelled":  df_labelled,
            "is_demo":      False,
            "file_id":      file_id,
            "recognised":   recognised,
            "unrecognised": unrecognised,
        })
    with st.expander(
        f"Column mapping — {len(st.session_state['recognised'])} recognised, "
        f"{len(st.session_state['unrecognised'])} skipped",
        expanded=False,
    ):
        st.write(f"**Recognised:** {', '.join(COLUMN_MAP[c]['test'] for c in st.session_state['recognised'])}")
        if st.session_state["unrecognised"]:
            st.write(f"**Skipped:** {', '.join(st.session_state['unrecognised'])}")
else:
    if "df_long" not in st.session_state:
        import pickle
        from pathlib import Path
        _cache_path = Path(__file__).parent / "demo_cache.pkl"
        if _cache_path.exists():
            with open(_cache_path, "rb") as _f:
                _cached = pickle.load(_f)
        else:
            with st.spinner("Loading demo data…"):
                _df_long     = load_stub_data()
                _gmm_results = analyse_upload(_df_long)
                _pop_results = analyse_population(_df_long)
                _df_labelled = build_labelled_df(_df_long, _gmm_results)
            _cached = {
                "df_long":     _df_long,
                "gmm_results": _gmm_results,
                "pop_results": _pop_results,
                "df_labelled": _df_labelled,
            }
        st.session_state.update({**_cached, "is_demo": True, "file_id": None})

if st.session_state.get("is_demo"):
    st.info(
        "**Demo mode** — uses 80 synthetic patients across two subgroups."
    )

with st.expander("Unit preferences", expanded=False):
    st.caption(
        "Some markers can be reported in different unit systems. "
        "Select the units your export uses to update results"
    )
    unit_prefs = {}
    pref_cols  = st.columns(3)
    for i, marker in enumerate(MULTI_UNIT_MARKERS):
        units = available_units(marker)
        unit_prefs[marker] = pref_cols[i % 3].selectbox(
            marker, units, key=f"unit_{marker}"
        )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs([
    "How does my population look?",
    "What types of patient exist?",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — How does my population look?
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if "df_long" in st.session_state:
        df_long     = st.session_state["df_long"]
        gmm_results = st.session_state["gmm_results"]

        n_patients = df_long["patient_id"].nunique()
        n_markers  = df_long["test_name"].nunique()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Patients", n_patients)
        col_b.metric("Markers",  n_markers)
        col_c.metric("Sample quality", "")
        col_c.caption(sample_size_label(n_patients))

        st.divider()

        # ── Marker explorer ──────────────────────────────────────────────────
        available = [t for t in gmm_results if "error" not in gmm_results[t]]
        errored   = [t for t in gmm_results if "error" in gmm_results[t]]

        if errored:
            st.caption(f"Skipped (too few data points): {', '.join(errored)}")

        selected_marker = st.selectbox("Select a marker to explore", available)

        if selected_marker:
            res       = gmm_results[selected_marker]
            disp_unit = unit_prefs.get(selected_marker, THRESHOLDS[selected_marker]["unit"])

            values_d, means_d, stds_d, boundaries_d = transform_for_display(
                selected_marker,
                res["values"], res["means"], res["stds"], res["boundaries"],
                disp_unit,
            )

            rules  = THRESHOLDS[selected_marker]
            ref_nb = from_canonical(selected_marker, rules["normal"][1],     disp_unit)
            ref_ba = from_canonical(selected_marker, rules["borderline"][1], disp_unit)

            # Plain English summary — leads the section
            st.markdown(marker_plain_english(
                selected_marker, res["n_components"],
                means_d, res["weights"], disp_unit, ref_nb, ref_ba,
            ))

            if res["small_sample"]:
                st.warning(
                    f"Small sample ({len(res['values'])} patients) — "
                    "treat these groups as a starting point. Results will stabilise around 100+ patients."
                )

            # Stats table — use pre-computed labels from analyse_upload
            n_comp = res["n_components"]
            labels = res["labels"]
            display_stats = []
            for i in range(n_comp):
                mask = labels == i
                cv   = values_d[mask]
                display_stats.append({
                    "Group":      f"Group {i + 1}",
                    "Average":    round(float(means_d[i]), 3),
                    "Spread (±)": round(float(stds_d[i]), 3),
                    "Min":        round(float(cv.min()), 3) if len(cv) else "—",
                    "Max":        round(float(cv.max()), 3) if len(cv) else "—",
                    "Patients":   int(mask.sum()),
                    "% of Total": f"{mask.mean() * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(display_stats), use_container_width=True, hide_index=True)

            # Chart
            weights = res["weights"]
            x = np.linspace(
                values_d.min() - 2 * values_d.std(),
                values_d.max() + 2 * values_d.std(),
                500,
            )

            fig = go.Figure()

            if len(values_d) >= 30:
                fig.add_trace(go.Histogram(
                    x=values_d, histnorm="probability density",
                    nbinsx=20, opacity=0.25,
                    marker_color="steelblue", name="Your patients",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=values_d, y=np.zeros(len(values_d)),
                    mode="markers",
                    marker=dict(symbol="line-ns", size=16, line=dict(width=2, color="#333")),
                    name="Patient values",
                ))

            for i, (m, s, w) in enumerate(zip(means_d, stds_d, weights)):
                colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
                fig.add_trace(go.Scatter(
                    x=x, y=w * scipy_norm.pdf(x, m, s),
                    mode="lines", line=dict(color=colour, width=2),
                    name=f"Group {i + 1} (avg {m:.2f})",
                ))

            total = sum(w * scipy_norm.pdf(x, m, s) for m, s, w in zip(means_d, stds_d, weights))
            fig.add_trace(go.Scatter(
                x=x, y=total,
                mode="lines", line=dict(color="black", width=1, dash="dash"),
                opacity=0.4, name="Combined",
            ))

            for b in boundaries_d:
                fig.add_vline(
                    x=b, line=dict(color="grey", width=1.5, dash="dash"),
                    annotation_text=f"Group boundary: {b:.2f}",
                    annotation_position="top",
                )

            fig.add_vline(
                x=ref_nb, line=dict(color="green", width=1.5, dash="dot"),
                annotation_text=f"Standard ref: {round(ref_nb, 2)}",
                annotation_position="top right",
            )
            fig.add_vline(
                x=ref_ba, line=dict(color="red", width=1.5, dash="dot"),
                annotation_text=f"Upper ref: {round(ref_ba, 2)}",
                annotation_position="top right",
            )

            fig.update_layout(
                title=f"{selected_marker} — {res['n_components']} groups found",
                xaxis_title=f"{selected_marker} ({disp_unit})",
                yaxis_title="How many patients",
                yaxis=dict(rangemode="tozero"),
                legend=dict(orientation="v", font=dict(size=11)),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Technical details", expanded=False):
                bic_str = " · ".join(
                    f"{n} groups: {int(bic)}" for n, bic in sorted(res["bic_scores"].items())
                )
                st.caption(
                    f"Group count selected by BIC scoring (lower = better fit): {bic_str}. "
                    f"{res['n_components']} groups selected."
                )

        st.divider()

        # ── Full results table ───────────────────────────────────────────────
        with st.expander("View full results table", expanded=False):
            df_display = st.session_state["df_labelled"].copy()

            present = set(df_display["test_name"].unique())
            for marker, disp_unit in unit_prefs.items():
                if marker not in present:
                    continue
                mask = df_display["test_name"] == marker
                df_display.loc[mask, "value"] = df_display.loc[mask, "value"].apply(
                    lambda v, du=disp_unit, mn=marker: from_canonical(mn, v, du)
                )
                df_display.loc[mask, "unit"] = disp_unit

            df_display = df_display.rename(columns={
                "patient_id": "Patient", "age": "Age",
                "test_name": "Test", "value": "Value", "unit": "Unit",
            })
            df_display = df_display[["Patient", "Age", "Test", "Value", "Unit", "Group"]]
            df_display["Value"] = df_display["Value"].round(3)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=df_display.to_csv(index=False),
                file_name="clustered_results.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — What types of patient exist?
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.write(
        "Understand patterns in blood markers across the population."
    )

    if "pop_results" not in st.session_state:
        st.info("Upload a CSV in the first tab to get started.")
    else:
        pop = st.session_state["pop_results"]

        if "error" in pop:
            st.warning(pop["error"])
        else:
            n_patients  = len(pop["patient_ids"])
            n_clusters  = pop["n_clusters"]
            labels      = pop["labels"]
            patient_ids = pop["patient_ids"]

            # Age lookup (None if column not in upload)
            df_long    = st.session_state["df_long"]
            age_lookup = (
                df_long.drop_duplicates("patient_id")
                .set_index("patient_id")["age"]
                .to_dict()
                if "age" in df_long.columns else {}
            )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Patients",     n_patients)
            col_b.metric("Markers used", pop["df_wide"].shape[1])
            col_c.metric("Types found",  n_clusters)
            st.caption(sample_size_label(n_patients))

            if pop["small_sample"]:
                st.warning(
                    f"Small sample ({n_patients} patients) — patient types are a starting point. "
                    "Results will stabilise around 200+ patients."
                )

            st.divider()

            # ── Patient map ──────────────────────────────────────────────────
            st.subheader("Patient map")
            st.info(
                "Each dot is a patient. **Patients close together have similar overall blood profiles.** "
                "Hover over a dot to see the patient ID and age.",
                icon="ℹ️",
            )

            has_age_data = bool(age_lookup)
            colour_by    = st.radio(
                "Colour by",
                ["Patient type", "Age"] if has_age_data else ["Patient type"],
                horizontal=True,
                key="scatter_colour",
            )

            X2   = pop["X_pca_2d"]
            var1 = pop["pca_var"][0] * 100
            var2 = pop["pca_var"][1] * 100

            hover_text = [
                f"{pid}" + (f" · Age {age_lookup[pid]}" if age_lookup.get(pid) is not None else "")
                for pid in patient_ids
            ]

            fig = go.Figure()

            if colour_by == "Age":
                ages = [age_lookup.get(pid) for pid in patient_ids]
                fig.add_trace(go.Scatter(
                    x=X2[:, 0], y=X2[:, 1],
                    mode="markers+text",
                    marker=dict(
                        size=12,
                        color=ages,
                        colorscale="RdYlBu_r",
                        showscale=True,
                        colorbar=dict(title="Age"),
                    ),
                    text=patient_ids,
                    customdata=hover_text,
                    textposition="top right",
                    textfont=dict(size=10),
                    hovertemplate="<b>%{customdata}</b><br>Similarity axis 1: %{x:.2f}<br>Similarity axis 2: %{y:.2f}<extra></extra>",
                    showlegend=False,
                ))
            else:
                for g in range(n_clusters):
                    mask = labels == g
                    idxs = np.where(mask)[0]
                    fig.add_trace(go.Scatter(
                        x=X2[mask, 0], y=X2[mask, 1],
                        mode="markers+text",
                        marker=dict(size=12, color=CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]),
                        text=[patient_ids[i] for i in idxs],
                        customdata=[hover_text[i] for i in idxs],
                        textposition="top right",
                        textfont=dict(size=10),
                        name=f"Type {g + 1}",
                        hovertemplate="<b>%{customdata}</b><br>Similarity axis 1: %{x:.2f}<br>Similarity axis 2: %{y:.2f}<extra></extra>",
                    ))

            fig.update_layout(
                xaxis_title=f"Similarity axis 1 ({var1:.1f}% of variation)",
                yaxis_title=f"Similarity axis 2 ({var2:.1f}% of variation)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── What defines each type ───────────────────────────────────────
            st.subheader("What defines each type?")
            st.write(
                "The markers below are what most distinguish each patient type from the rest of your population."
            )

            fp          = pop["fingerprint"].round(2)
            type_cols   = st.columns(n_clusters)
            for g in range(n_clusters):
                group_col   = f"Group {g + 1}"
                members     = [pid for pid, lbl in zip(patient_ids, labels) if lbl == g]
                bullets     = group_plain_english(fp, group_col, top_n=5)
                with type_cols[g]:
                    colour = CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]
                    st.markdown(
                        f"<h4 style='color:{colour}'>Type {g + 1}</h4>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{len(members)} patients")
                    if bullets:
                        st.markdown("\n".join(bullets))
                    else:
                        st.caption("No strong distinguishing markers found.")

            with st.expander("See full marker heatmap", expanded=False):
                st.caption(
                    "Red = this type scores higher than the population average for this marker. "
                    "Blue = lower than average. Darker = stronger signal."
                )
                fig_fp = go.Figure(go.Heatmap(
                    z=fp.values,
                    x=fp.columns.tolist(),
                    y=fp.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0, zmin=-2, zmax=2,
                    text=fp.values.round(2),
                    texttemplate="%{text}",
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
                    colorbar=dict(title="vs average"),
                ))
                fig_fp.update_layout(
                    height=max(300, len(fp.index) * 22),
                    margin=dict(l=160, r=40, t=20, b=40),
                    xaxis=dict(side="top"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_fp, use_container_width=True)

            st.divider()

            # ── Patient list ─────────────────────────────────────────────────
            st.subheader("Who is in each type?")
            gcols = st.columns(n_clusters)
            for g in range(n_clusters):
                members = [pid for pid, lbl in zip(patient_ids, labels) if lbl == g]
                gcols[g].markdown(f"**Type {g + 1}** — {len(members)} patients")
                for pid in members:
                    gcols[g].write(pid)

            with st.expander("Technical details", expanded=False):
                bic_str = " · ".join(
                    f"{n} types: {int(bic)}" for n, bic in sorted(pop["bic_scores"].items())
                )
                var_str = f"{pop['pca_var'][:pop['n_cluster_dims']].sum() * 100:.0f}%"
                st.caption(
                    f"Analysis used {pop['n_cluster_dims']} dimensions capturing {var_str} of total variation. "
                    f"Number of types selected by BIC (lower = better fit): {bic_str}."
                )


