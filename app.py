import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm as scipy_norm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import plotly.graph_objects as go

from thresholds import THRESHOLDS
from column_map import COLUMN_MAP, ID_COLUMN
from unit_conversions import (
    to_canonical,
    available_units, from_canonical, transform_for_display,
)
from gmm import fit_optimal_gmm, sort_gmm, get_boundaries, assign_clusters
from stub_data import generate_stub_data

@st.cache_data
def load_stub_data():
    return generate_stub_data()

st.set_page_config(page_title="Blood Test Classifier", layout="wide")

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Marker categories (for Patient View grouping) ─────────────────────────────
MARKER_CATEGORIES = {
    "Liver / Metabolic": ["Albumin", "ALP", "ALT", "GGT"],
    "Kidney":            ["eGFR"],
    "Iron / Blood":      ["Ferritin", "Haemoglobin", "Haematocrit (HCT)", "Platelet Count"],
    "Glucose":           ["HbA1C"],
    "Lipids":            ["Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol",
                          "Total Cholesterol:HDL Ratio"],
    "Thyroid":           ["TSH", "Free T4"],
    "Hormones":          ["Testosterone", "Free Testosterone", "SHBG", "Oestradiol",
                          "Prolactin", "FSH", "LH", "PSA"],
    "Blood Cells":       ["White Blood Cell Count", "Neutrophil Count",
                          "Basophil Count", "Eosinophil Count"],
}

def marker_category(test_name: str) -> str:
    for cat, markers in MARKER_CATEGORIES.items():
        if test_name in markers:
            return cat
    return "Other"


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


# ── Analysis functions (unchanged) ───────────────────────────────────────────

def analyse_upload(df_long: pd.DataFrame) -> dict:
    """Fit GMM per marker on uploaded data. Returns per-marker analysis dict."""
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
    """Attach group labels to every row. Computed once at load time."""
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
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
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


# ── CSV parsing ───────────────────────────────────────────────────────────────

def parse_upload(uploaded_file) -> tuple:
    """Wide-format CSV → long-format DataFrame with canonical units."""
    df_raw = pd.read_csv(uploaded_file)
    df_raw = df_raw[
        df_raw[ID_COLUMN].notna() &
        (df_raw[ID_COLUMN].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    rows = []
    for _, row in df_raw.iterrows():
        patient_id = str(row[ID_COLUMN])
        seen_tests = set()
        for col, mapping in COLUMN_MAP.items():
            if col not in df_raw.columns:
                continue
            raw = row.get(col)
            if pd.isna(raw):
                continue
            test_name = mapping["test"]
            if test_name in seen_tests:
                continue
            seen_tests.add(test_name)
            value = to_canonical(test_name, float(raw) * mapping["scale"])
            rows.append({
                "patient_id": patient_id,
                "test_name":  test_name,
                "value":      value,
                "unit":       THRESHOLDS[test_name]["unit"],
            })

    recognised   = [c for c in COLUMN_MAP if c in df_raw.columns]
    unrecognised = [c for c in df_raw.columns if c not in COLUMN_MAP and c != ID_COLUMN]
    return pd.DataFrame(rows), recognised, unrecognised


# ── App header ────────────────────────────────────────────────────────────────

st.title("Blood Test Classifier")
st.caption(
    "Discovers natural patterns in blood test results using unsupervised machine learning. "
    "Upload a population export and the app finds the groups that exist in your data — "
    "without pre-labelling them. **This tool is not a diagnostic instrument.** "
    "Findings should be interpreted by a qualified clinician."
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "How does my population look?",
    "What types of patient exist?",
    "Is this patient unusual?",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — How does my population look?
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("How does my population look?")
    st.write(
        "Each marker is analysed independently to find the natural groups in your data. "
        "Select a marker below to see how your patients are distributed."
    )

    uploaded = st.file_uploader("Upload a blood test CSV export", type="csv", key="upload")

    MULTI_UNIT_MARKERS = [m for m in THRESHOLDS if len(available_units(m)) > 1]

    with st.expander("Unit preferences", expanded=False):
        st.caption(
            "Some markers can be reported in different unit systems. "
            "Select the units your export uses — results will update instantly."
        )
        unit_prefs = {}
        cols = st.columns(3)
        for i, marker in enumerate(MULTI_UNIT_MARKERS):
            units = available_units(marker)
            unit_prefs[marker] = cols[i % 3].selectbox(
                marker, units, key=f"unit_{marker}"
            )
    st.session_state["unit_prefs"] = unit_prefs

    if uploaded:
        # Only reprocess if this is a new file
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("file_id") != file_id:
            with st.spinner("Reading your data and finding groups…"):
                df_long, recognised, unrecognised = parse_upload(uploaded)
                gmm_results  = analyse_upload(df_long)
                pop_results  = analyse_population(df_long)
                df_labelled  = build_labelled_df(df_long, gmm_results)
            st.session_state["df_long"]      = df_long
            st.session_state["gmm_results"]  = gmm_results
            st.session_state["pop_results"]  = pop_results
            st.session_state["df_labelled"]  = df_labelled
            st.session_state["is_demo"]      = False
            st.session_state["file_id"]      = file_id
            st.session_state["recognised"]   = recognised
            st.session_state["unrecognised"] = unrecognised

        with st.expander(
            f"Column mapping — {len(st.session_state['recognised'])} recognised, "
            f"{len(st.session_state['unrecognised'])} skipped",
            expanded=False,
        ):
            st.write(f"**Recognised:** {', '.join(COLUMN_MAP[c]['test'] for c in st.session_state['recognised'])}")
            if st.session_state["unrecognised"]:
                st.write(f"**Skipped (no mapping):** {', '.join(st.session_state['unrecognised'])}")
    else:
        if "df_long" not in st.session_state:
            with st.spinner("Loading demo data…"):
                df_long     = load_stub_data()
                gmm_results = analyse_upload(df_long)
                pop_results = analyse_population(df_long)
                df_labelled = build_labelled_df(df_long, gmm_results)
            st.session_state["df_long"]     = df_long
            st.session_state["gmm_results"] = gmm_results
            st.session_state["pop_results"] = pop_results
            st.session_state["df_labelled"] = df_labelled
            st.session_state["is_demo"]     = True
            st.session_state["file_id"]     = None

    if st.session_state.get("is_demo"):
        st.info(
            "**Demo mode** — showing 80 synthetic patients across two subgroups. "
            "Upload your own CSV above to analyse your real population.",
        )

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

            # Stats table
            n_comp = res["n_components"]
            labels = assign_clusters(res["values"], res["boundaries"])
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

            for marker, disp_unit in unit_prefs.items():
                mask = df_display["test_name"] == marker
                df_display.loc[mask, "value"] = df_display.loc[mask, "value"].apply(
                    lambda v, du=disp_unit, mn=marker: from_canonical(mn, v, du)
                )
                df_display.loc[mask, "unit"] = disp_unit

            df_display = df_display.rename(columns={
                "patient_id": "Patient", "test_name": "Test",
                "value": "Value", "unit": "Unit",
            })
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
    st.header("What types of patient exist?")
    st.write(
        "Rather than looking at one marker at a time, this combines every marker together "
        "to ask: which patients are similar to each other overall? "
        "The result is a set of patient types defined by their whole blood profile — "
        "not just one number."
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
                "Colour shows which group they belong to. Hover over a dot to see the patient ID.",
                icon="ℹ️",
            )

            X2   = pop["X_pca_2d"]
            var1 = pop["pca_var"][0] * 100
            var2 = pop["pca_var"][1] * 100

            fig = go.Figure()
            for g in range(n_clusters):
                mask = labels == g
                idxs = np.where(mask)[0]
                fig.add_trace(go.Scatter(
                    x=X2[mask, 0], y=X2[mask, 1],
                    mode="markers+text",
                    marker=dict(size=12, color=CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]),
                    text=[patient_ids[i] for i in idxs],
                    textposition="top right",
                    textfont=dict(size=10),
                    name=f"Type {g + 1}",
                    hovertemplate="<b>%{text}</b><br>Similarity axis 1: %{x:.2f}<br>Similarity axis 2: %{y:.2f}<extra></extra>",
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


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Is this patient unusual?
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Is this patient unusual?")
    st.write(
        "Select a patient to see how they compare to the rest of your population "
        "on each marker. Flagged markers are those where this patient sits in the "
        "smallest, most unusual group."
    )

    if "df_long" not in st.session_state:
        st.info("Upload a CSV in the first tab to get started.")
    else:
        df_long     = st.session_state["df_long"]
        gmm_results = st.session_state["gmm_results"]
        df_view     = st.session_state["df_labelled"]
        unit_prefs  = st.session_state.get("unit_prefs", {})

        # Minority cluster per marker
        minority_cluster = {}
        for test_name, res in gmm_results.items():
            if "error" in res or not res["cluster_stats"]:
                continue
            minority_cluster[test_name] = min(
                res["cluster_stats"], key=lambda r: r["Patients"]
            )["Group"]

        selected_patient = st.selectbox("Select patient", sorted(df_long["patient_id"].unique()))

        patient_df = df_view[df_view["patient_id"] == selected_patient].copy()
        patient_df["Unusual"] = patient_df.apply(
            lambda row: True if row["Group"] == minority_cluster.get(row["test_name"]) else False,
            axis=1,
        )
        patient_df["Category"] = patient_df["test_name"].apply(marker_category)

        # Apply display unit preferences
        for marker, disp_unit in unit_prefs.items():
            mask = patient_df["test_name"] == marker
            patient_df.loc[mask, "value"] = patient_df.loc[mask, "value"].apply(
                lambda v, du=disp_unit, mn=marker: from_canonical(mn, v, du)
            )
            patient_df.loc[mask, "unit"] = disp_unit

        patient_df["value"] = patient_df["value"].round(3)

        flagged   = patient_df[patient_df["Unusual"]]
        n_flagged = len(flagged)

        # ── Summary callout ──────────────────────────────────────────────────
        if n_flagged == 0:
            st.success(
                f"**{selected_patient}** does not fall in any unusual group across "
                f"{len(patient_df)} markers analysed."
            )
        else:
            # Group flagged markers by category
            flagged_by_cat = flagged.groupby("Category")["test_name"].apply(list).to_dict()
            cat_summaries  = [
                f"**{cat}**: {', '.join(markers)}"
                for cat, markers in flagged_by_cat.items()
            ]
            st.warning(
                f"**{selected_patient}** sits in an unusual group on **{n_flagged} of "
                f"{len(patient_df)} markers** — "
                + " · ".join(cat_summaries)
                + ". This means these values are atypical relative to the rest of this population. "
                "A clinician should determine whether this pattern is meaningful."
            )

        st.divider()

        # ── Per-category breakdown ───────────────────────────────────────────
        for category in MARKER_CATEGORIES:
            cat_rows = patient_df[patient_df["Category"] == category]
            if cat_rows.empty:
                continue

            n_flagged_cat = cat_rows["Unusual"].sum()
            header = f"**{category}**"
            if n_flagged_cat > 0:
                header += f" — {n_flagged_cat} unusual"

            with st.expander(header, expanded=(n_flagged_cat > 0)):
                display_rows = []
                for _, row in cat_rows.iterrows():
                    display_rows.append({
                        "Marker": row["test_name"],
                        "Value":  row["value"],
                        "Unit":   row["unit"],
                        "Group":  row["Group"],
                        "Note":   "⚠️ unusual — smallest group for this marker" if row["Unusual"] else "",
                    })
                st.dataframe(
                    pd.DataFrame(display_rows),
                    use_container_width=True,
                    hide_index=True,
                )
