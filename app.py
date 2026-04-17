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

st.set_page_config(page_title="Blood Test Classifier", layout="wide")
st.title("Blood Test Classifier")

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


@st.cache_data
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
                "Cluster":    f"Cluster {i + 1}",
                "Mean":       round(float(means[i]), 3),
                "Std Dev":    round(float(stds[i]), 3),
                "Min":        round(float(cv.min()), 3) if len(cv) else "—",
                "Max":        round(float(cv.max()), 3) if len(cv) else "—",
                "Patients":   int(mask.sum()),
                "% of Total": f"{mask.mean() * 100:.1f}%",
            })

        results[test_name] = {
            "n_components": n,
            "bic_scores":   bic_scores,
            "means":        means,
            "stds":         stds,
            "weights":      weights,
            "boundaries":   boundaries,
            "labels":       labels,
            "values":       values,
            "cluster_stats": cluster_stats,
            "small_sample": len(values) < 30,
        }

    return results


@st.cache_data
def analyse_population(df_long: pd.DataFrame) -> dict:
    """
    Multivariate patient clustering across all markers.
    Pipeline: wide pivot → median impute → StandardScaler → PCA → GMM.
    Returns enough to render scatter, fingerprint table, and patient list.
    """
    # Wide format: rows = patients, cols = markers
    df_wide = df_long.pivot_table(index="patient_id", columns="test_name", values="value")

    # Drop markers missing for more than half the patients
    df_wide = df_wide.dropna(thresh=int(len(df_wide) * 0.5), axis=1)

    # Impute remaining gaps with per-marker median
    df_wide = df_wide.fillna(df_wide.median())

    n_patients, n_markers = df_wide.shape
    if n_patients < 4:
        return {"error": f"Only {n_patients} patients — need at least 4 for population clustering."}
    if n_markers < 2:
        return {"error": "Not enough markers with sufficient coverage for population clustering."}

    # Normalise so markers on different scales contribute equally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_wide.values)

    # PCA — use enough components to explain 80% variance, minimum 2
    max_components = min(n_patients - 1, n_markers)
    pca = PCA(n_components=max_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_for_80 = int(np.searchsorted(cumvar, 0.80)) + 1
    n_cluster_dims = max(2, min(n_for_80, max_components))

    X_cluster = X_pca[:, :n_cluster_dims]

    # GMM with BIC to select cluster count (2–5, capped by patient count)
    max_n = min(5, n_patients - 1)
    best_n, best_bic, best_gmm, bic_scores = 2, np.inf, None, {}
    for n in range(2, max_n + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=10)
        gmm.fit(X_cluster)
        bic = gmm.bic(X_cluster)
        bic_scores[n] = float(bic)
        if bic < best_bic:
            best_bic, best_n, best_gmm = bic, n, gmm

    labels = best_gmm.predict(X_cluster)

    # Z-scores per marker per cluster (shows which markers define each group)
    z_scores = pd.DataFrame(X_scaled, index=df_wide.index, columns=df_wide.columns)
    fingerprint = z_scores.copy()
    fingerprint["Group"] = [f"Group {l + 1}" for l in labels]
    fingerprint = fingerprint.groupby("Group").mean().T  # markers × groups

    return {
        "patient_ids":    list(df_wide.index),
        "labels":         labels,
        "n_clusters":     best_n,
        "bic_scores":     bic_scores,
        "X_pca_2d":       X_pca[:, :2],
        "pca_var":        pca.explained_variance_ratio_,
        "n_cluster_dims": n_cluster_dims,
        "fingerprint":    fingerprint,       # DataFrame: markers × groups, values = mean z-score
        "df_wide":        df_wide,           # actual canonical values, for per-group means table
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


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Upload & Discover", "Population Groups", "Patient View"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload & Discover
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Upload & Discover")
    st.write(
        "Upload a blood test export. The app fits clusters to each marker from your data "
        "and lets you interpret what those clusters mean."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv", key="upload")

    # ── Unit preferences (shown whether or not a file has been uploaded) ─────
    MULTI_UNIT_MARKERS = [m for m in THRESHOLDS if len(available_units(m)) > 1]

    with st.expander("Unit preferences", expanded=False):
        st.caption(
            "These markers can appear in different unit systems. "
            "Pick the units your export uses — values will be displayed accordingly."
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
        with st.spinner("Parsing and fitting clusters…"):
            df_long, recognised, unrecognised = parse_upload(uploaded)
            gmm_results = analyse_upload(df_long)

        st.session_state["df_long"]     = df_long
        st.session_state["gmm_results"] = gmm_results
        st.session_state["is_demo"]     = False

        with st.expander(
            f"Column mapping — {len(recognised)} recognised, {len(unrecognised)} skipped",
            expanded=False,
        ):
            st.write(f"**Classified:** {', '.join(COLUMN_MAP[c]['test'] for c in recognised)}")
            if unrecognised:
                st.write(f"**No thresholds yet:** {', '.join(unrecognised)}")
    else:
        if "df_long" not in st.session_state or st.session_state.get("is_demo"):
            with st.spinner("Loading demo data…"):
                df_long     = generate_stub_data()
                gmm_results = analyse_upload(df_long)
            st.session_state["df_long"]     = df_long
            st.session_state["gmm_results"] = gmm_results
            st.session_state["is_demo"]     = True

    if st.session_state.get("is_demo"):
        st.info(
            "Showing demo data — 80 synthetic patients across two subgroups. "
            "Upload your own CSV above to analyse your population.",
        )

    if "df_long" in st.session_state:
        df_long     = st.session_state["df_long"]
        gmm_results = st.session_state["gmm_results"]

        # Build display table: attach cluster assignment to each row
        df_display = df_long.copy()
        df_display["Cluster"] = "—"
        for test_name, res in gmm_results.items():
            if "error" in res:
                continue
            mask = df_display["test_name"] == test_name
            labels = assign_clusters(df_display.loc[mask, "value"].values, res["boundaries"])
            df_display.loc[mask, "Cluster"] = [f"Cluster {l + 1}" for l in labels]

        # Apply display unit preferences
        for marker, disp_unit in unit_prefs.items():
            mask = df_display["test_name"] == marker
            df_display.loc[mask, "value"] = df_display.loc[mask, "value"].apply(
                lambda v, du=disp_unit, mn=marker: from_canonical(mn, v, du)
            )
            df_display.loc[mask, "unit"] = disp_unit

        df_display = df_display.rename(columns={
            "patient_id": "Patient", "test_name": "Test", "value": "Value", "unit": "Unit",
        })
        df_display["Value"] = df_display["Value"].round(3)

        n_patients = df_long["patient_id"].nunique()
        n_markers  = df_long["test_name"].nunique()
        st.subheader(f"Results — {n_patients} patients · {n_markers} markers")
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results CSV",
            data=df_display.to_csv(index=False),
            file_name="clustered_results.csv",
            mime="text/csv",
        )

        # ── Cluster Explorer ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Cluster Explorer")
        st.caption("Select a marker to see the discovered clusters and how they compare to reference ranges.")

        available = [t for t in gmm_results if "error" not in gmm_results[t]]
        errored   = [t for t in gmm_results if "error" in gmm_results[t]]

        if errored:
            st.caption(f"Skipped (too few data points): {', '.join(errored)}")

        selected_marker = st.selectbox("Marker", available)

        if selected_marker:
            res       = gmm_results[selected_marker]
            disp_unit = unit_prefs.get(selected_marker, THRESHOLDS[selected_marker]["unit"])

            if res["small_sample"]:
                st.warning(
                    f"Only {len(res['values'])} data points — clusters are indicative. "
                    "More data will improve reliability."
                )

            values_d, means_d, stds_d, boundaries_d = transform_for_display(
                selected_marker,
                res["values"], res["means"], res["stds"], res["boundaries"],
                disp_unit,
            )

            n_comp = res["n_components"]
            labels = assign_clusters(res["values"], res["boundaries"])
            display_stats = []
            for i in range(n_comp):
                mask = labels == i
                cv = values_d[mask]
                display_stats.append({
                    "Cluster":    f"Cluster {i + 1}",
                    "Mean":       round(float(means_d[i]), 3),
                    "Std Dev":    round(float(stds_d[i]), 3),
                    "Min":        round(float(cv.min()), 3) if len(cv) else "—",
                    "Max":        round(float(cv.max()), 3) if len(cv) else "—",
                    "Patients":   int(mask.sum()),
                    "% of Total": f"{mask.mean() * 100:.1f}%",
                })
            st.dataframe(pd.DataFrame(display_stats), use_container_width=True, hide_index=True)

            bic_str = " · ".join(
                f"n={n}: {int(bic)}" for n, bic in sorted(res["bic_scores"].items())
            )
            st.caption(
                f"BIC — {bic_str} (lower is better). "
                f"{res['n_components']} clusters selected."
            )

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
                    marker_color="steelblue", name="Data",
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=values_d, y=np.zeros(len(values_d)),
                    mode="markers", marker=dict(symbol="line-ns", size=16, line=dict(width=2, color="#333")),
                    name="Data points",
                ))

            for i, (m, s, w) in enumerate(zip(means_d, stds_d, weights)):
                colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
                fig.add_trace(go.Scatter(
                    x=x, y=w * scipy_norm.pdf(x, m, s),
                    mode="lines", line=dict(color=colour, width=2),
                    name=f"Cluster {i + 1} (mean={m:.2f})",
                ))

            total = sum(w * scipy_norm.pdf(x, m, s) for m, s, w in zip(means_d, stds_d, weights))
            fig.add_trace(go.Scatter(
                x=x, y=total,
                mode="lines", line=dict(color="black", width=1, dash="dash"),
                opacity=0.4, name="Combined fit",
            ))

            for b in boundaries_d:
                fig.add_vline(
                    x=b, line=dict(color="grey", width=1.5, dash="dash"),
                    annotation_text=f"Boundary: {b:.2f}", annotation_position="top",
                )

            rules  = THRESHOLDS[selected_marker]
            ref_nb = from_canonical(selected_marker, rules["normal"][1],     disp_unit)
            ref_ba = from_canonical(selected_marker, rules["borderline"][1], disp_unit)
            fig.add_vline(
                x=ref_nb, line=dict(color="green", width=1.5, dash="dot"),
                annotation_text=f"Ref N→B: {round(ref_nb, 2)}", annotation_position="top right",
            )
            fig.add_vline(
                x=ref_ba, line=dict(color="red", width=1.5, dash="dot"),
                annotation_text=f"Ref B→A: {round(ref_ba, 2)}", annotation_position="top right",
            )

            fig.update_layout(
                title=f"{selected_marker} — {res['n_components']} clusters discovered",
                xaxis_title=f"{selected_marker} ({disp_unit})",
                yaxis_title="Density",
                yaxis=dict(rangemode="tozero"),
                legend=dict(orientation="v", font=dict(size=11)),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Population Groups
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Population Groups")
    st.write(
        "Cluster patients by their full blood test profile — not marker by marker, "
        "but all markers together. Reveals which types of patient exist in your population."
    )

    if "df_long" not in st.session_state:
        st.info("Upload a CSV in the 'Upload & Discover' tab first.")
    else:
        df_long = st.session_state["df_long"]

        with st.spinner("Running population clustering…"):
            pop = analyse_population(df_long)

        if "error" in pop:
            st.warning(pop["error"])
        else:
            if pop["small_sample"]:
                st.warning(
                    f"Only {len(pop['patient_ids'])} patients — groups are indicative. "
                    "More data will improve reliability."
                )

            n_clusters  = pop["n_clusters"]
            labels      = pop["labels"]
            patient_ids = pop["patient_ids"]
            group_labels = [f"Group {l + 1}" for l in labels]

            # ── Summary metrics ──────────────────────────────────────────────
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Patients",        len(patient_ids))
            col_b.metric("Markers used",    pop["df_wide"].shape[1])
            col_c.metric("Groups found",    n_clusters)

            bic_str = " · ".join(
                f"n={n}: {int(bic)}" for n, bic in sorted(pop["bic_scores"].items())
            )
            var_str = f"{pop['pca_var'][:pop['n_cluster_dims']].sum() * 100:.0f}%"
            st.caption(
                f"Clustered on {pop['n_cluster_dims']} PCA components "
                f"({var_str} of variance explained) · "
                f"BIC — {bic_str} (lower is better)"
            )

            st.divider()

            # ── PCA scatter ──────────────────────────────────────────────────
            st.subheader("Patient map")
            st.caption(
                "Each point is a patient. Position reflects similarity across all markers — "
                "patients close together have similar overall profiles."
            )

            X2 = pop["X_pca_2d"]
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
                    name=f"Group {g + 1}",
                    hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                ))
            fig.update_layout(
                title="Patient clustering — all markers combined",
                xaxis_title=f"PC1 ({var1:.1f}% variance)",
                yaxis_title=f"PC2 ({var2:.1f}% variance)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Group fingerprint ────────────────────────────────────────────
            st.subheader("Group fingerprint")
            st.caption(
                "Mean z-score per marker per group. "
                "Positive (red) = above population average. "
                "Negative (blue) = below population average. "
                "The larger the value, the more that marker defines the group."
            )

            fp = pop["fingerprint"].round(2)
            fig_fp = go.Figure(go.Heatmap(
                z=fp.values,
                x=fp.columns.tolist(),
                y=fp.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-2, zmax=2,
                text=fp.values.round(2),
                texttemplate="%{text}",
                hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
                colorbar=dict(title="z-score"),
            ))
            fig_fp.update_layout(
                height=max(300, len(fp.index) * 22),
                margin=dict(l=160, r=40, t=20, b=40),
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_fp, use_container_width=True)

            st.divider()

            # ── Patient list per group ───────────────────────────────────────
            st.subheader("Group membership")
            gcols = st.columns(n_clusters)
            for g in range(n_clusters):
                members = [pid for pid, lbl in zip(patient_ids, labels) if lbl == g]
                gcols[g].markdown(f"**Group {g + 1}** ({len(members)} patients)")
                for pid in members:
                    gcols[g].write(pid)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Patient View
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Patient View")

    if "df_long" not in st.session_state:
        st.info("Upload a CSV in the 'Upload & Discover' tab first.")
    else:
        df_long     = st.session_state["df_long"]
        gmm_results = st.session_state["gmm_results"]

        # Identify the minority cluster per marker (fewest patients = most unusual)
        minority_cluster = {}
        for test_name, res in gmm_results.items():
            if "error" in res or not res["cluster_stats"]:
                continue
            minority_cluster[test_name] = min(
                res["cluster_stats"], key=lambda r: r["Patients"]
            )["Cluster"]

        # Attach cluster labels
        df_view = df_long.copy()
        df_view["Cluster"] = "—"
        for test_name, res in gmm_results.items():
            if "error" in res:
                continue
            mask   = df_view["test_name"] == test_name
            labels = assign_clusters(df_view.loc[mask, "value"].values, res["boundaries"])
            df_view.loc[mask, "Cluster"] = [f"Cluster {l + 1}" for l in labels]

        unit_prefs = st.session_state.get("unit_prefs", {})

        selected_patient = st.selectbox("Select patient", sorted(df_long["patient_id"].unique()))

        patient_df = df_view[df_view["patient_id"] == selected_patient].copy()
        patient_df["Note"] = patient_df.apply(
            lambda row: "⚠️ minority cluster" if row["Cluster"] == minority_cluster.get(row["test_name"]) else "",
            axis=1,
        )
        patient_df = (
            patient_df[["test_name", "value", "unit", "Cluster", "Note"]]
            .rename(columns={"test_name": "Test", "value": "Value", "unit": "Unit"})
        )

        # Apply display unit preferences
        for marker, disp_unit in unit_prefs.items():
            mask = patient_df["Test"] == marker
            patient_df.loc[mask, "Value"] = patient_df.loc[mask, "Value"].apply(
                lambda v, du=disp_unit, mn=marker: from_canonical(mn, v, du)
            )
            patient_df.loc[mask, "Unit"] = disp_unit

        patient_df["Value"] = patient_df["Value"].round(3)

        n_flagged = (patient_df["Note"] != "").sum()
        st.subheader(f"Patient {selected_patient}")
        st.caption(
            f"⚠️ flags markers where this patient is in the smallest cluster across all patients. "
            f"{n_flagged} of {len(patient_df)} markers flagged."
        )
        st.dataframe(patient_df, use_container_width=True, hide_index=True)
