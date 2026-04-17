import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from scipy.stats import norm as scipy_norm

from thresholds import classify_test, THRESHOLDS
from column_map import COLUMN_MAP, ID_COLUMN
from unit_conversions import to_canonical, unit_hint

st.set_page_config(page_title="Blood Test Classifier", layout="wide")
st.title("Blood Test Classifier")

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


# ── GMM fitting ───────────────────────────────────────────────────────────────

def fit_optimal_gmm(values: np.ndarray) -> tuple:
    """Try 2–4 components, return the GMM with lowest BIC."""
    max_n = min(4, max(2, len(values) // 5))
    best_n, best_bic, best_gmm, bic_scores = 2, np.inf, None, {}
    for n in range(2, max_n + 1):
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
        gmm.fit(values.reshape(-1, 1))
        bic = gmm.bic(values.reshape(-1, 1))
        bic_scores[n] = bic
        if bic < best_bic:
            best_bic, best_n, best_gmm = bic, n, gmm
    return best_gmm, best_n, bic_scores


def sort_gmm(gmm: GaussianMixture):
    """Return means, stds, weights sorted ascending by mean."""
    order = np.argsort(gmm.means_.ravel())
    means   = gmm.means_.ravel()[order]
    stds    = np.sqrt(gmm.covariances_.ravel()[order])
    weights = gmm.weights_[order]
    return means, stds, weights


def get_boundaries(means, stds, weights) -> list:
    """Intersection point between each pair of adjacent sorted components."""
    boundaries = []
    for i in range(len(means) - 1):
        try:
            b = brentq(
                lambda x: (weights[i]   * scipy_norm.pdf(x, means[i],   stds[i]) -
                           weights[i+1] * scipy_norm.pdf(x, means[i+1], stds[i+1])),
                means[i], means[i+1],
            )
        except ValueError:
            b = (means[i] + means[i+1]) / 2  # fallback: midpoint
        boundaries.append(b)
    return boundaries


def assign_clusters(values: np.ndarray, boundaries: list) -> np.ndarray:
    """0-indexed cluster per value based on sorted boundaries."""
    labels = np.zeros(len(values), dtype=int)
    for b in boundaries:
        labels[values > b] += 1
    return labels


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
            if test_name in seen_tests:     # skip duplicate column mappings (e.g. Haematocrit ADJ)
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

tab1, tab2, tab3 = st.tabs(["Reference Lookup", "Upload & Discover", "Patient View"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Reference Lookup (rules only, no ML)
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Reference Lookup")
    st.caption("Check a single value against male reference ranges.")

    col_a, col_b = st.columns(2)
    with col_a:
        selected_test = st.selectbox("Test", list(THRESHOLDS.keys()))
        hint = unit_hint(selected_test)
        value = st.number_input(f"Value ({hint})", format="%.4f", value=0.0)

    if st.button("Check"):
        canonical = to_canonical(selected_test, value)
        result = classify_test(selected_test, canonical)
        colour = {"Normal": "green", "Borderline": "orange", "Abnormal": "red"}[result]

        with col_b:
            st.subheader("Result")
            st.markdown(f"### :{colour}[{result}]")
            if canonical != value:
                st.caption(
                    f"Auto-converted: {value} → {round(canonical, 4)} "
                    f"{THRESHOLDS[selected_test]['unit']}"
                )

        rules = THRESHOLDS[selected_test]
        unit = rules["unit"]
        st.divider()
        st.caption(
            f"Normal: {rules['normal'][0]}–{rules['normal'][1]} {unit} · "
            f"Borderline: {rules['borderline'][0]}–{rules['borderline'][1]} {unit}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Upload & Discover
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("Upload & Discover")
    st.write(
        "Upload a blood test export. The app fits clusters to each marker from your data "
        "and lets you interpret what those clusters mean."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv", key="upload")

    if uploaded:
        with st.spinner("Parsing and fitting clusters…"):
            df_long, recognised, unrecognised = parse_upload(uploaded)
            gmm_results = analyse_upload(df_long)

        st.session_state["df_long"] = df_long
        st.session_state["gmm_results"] = gmm_results

        with st.expander(
            f"Column mapping — {len(recognised)} recognised, {len(unrecognised)} skipped",
            expanded=False,
        ):
            st.write(f"**Classified:** {', '.join(COLUMN_MAP[c]['test'] for c in recognised)}")
            if unrecognised:
                st.write(f"**No thresholds yet:** {', '.join(unrecognised)}")

        # Build display table: attach cluster assignment to each row
        df_display = df_long.copy()
        df_display["Cluster"] = "—"
        for test_name, res in gmm_results.items():
            if "error" in res:
                continue
            mask = df_display["test_name"] == test_name
            labels = assign_clusters(df_display.loc[mask, "value"].values, res["boundaries"])
            df_display.loc[mask, "Cluster"] = [f"Cluster {l + 1}" for l in labels]

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
            res  = gmm_results[selected_marker]
            unit = THRESHOLDS[selected_marker]["unit"]

            if res["small_sample"]:
                st.warning(
                    f"Only {len(res['values'])} data points — clusters are indicative. "
                    "More data will improve reliability."
                )

            # Cluster stats table
            st.dataframe(
                pd.DataFrame(res["cluster_stats"]),
                use_container_width=True,
                hide_index=True,
            )

            bic_str = " · ".join(
                f"n={n}: {int(bic)}" for n, bic in sorted(res["bic_scores"].items())
            )
            st.caption(
                f"BIC — {bic_str} (lower is better). "
                f"{res['n_components']} clusters selected."
            )

            # Histogram / rug + GMM curves
            values  = res["values"]
            means   = res["means"]
            stds    = res["stds"]
            weights = res["weights"]
            x = np.linspace(values.min() - 2 * values.std(), values.max() + 2 * values.std(), 500)

            fig, ax = plt.subplots(figsize=(10, 4))

            if len(values) >= 30:
                ax.hist(values, bins=20, density=True, alpha=0.25, color="steelblue", label="Data")
            else:
                # Rug plot is more honest for small n
                ax.plot(
                    values,
                    np.full_like(values, -0.01 / values.std()),
                    "|", color="#333", markersize=20, markeredgewidth=2, label="Data points",
                )

            for i, (m, s, w) in enumerate(zip(means, stds, weights)):
                colour = CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
                ax.plot(
                    x, w * scipy_norm.pdf(x, m, s),
                    color=colour, linewidth=2, label=f"Cluster {i + 1} (mean={m:.2f})",
                )

            total = sum(w * scipy_norm.pdf(x, m, s) for m, s, w in zip(means, stds, weights))
            ax.plot(x, total, "k--", linewidth=1, alpha=0.4, label="Combined fit")

            for b in res["boundaries"]:
                ax.axvline(b, color="grey", linestyle="--", linewidth=1.2,
                           label=f"Cluster boundary: {b:.3f}")

            # Reference ranges as context — NOT used to label clusters
            rules = THRESHOLDS[selected_marker]
            ax.axvline(rules["normal"][1], color="green", linestyle=":", linewidth=1.5,
                       label=f"Ref N→B: {rules['normal'][1]} {unit}")
            ax.axvline(rules["borderline"][1], color="red", linestyle=":", linewidth=1.5,
                       label=f"Ref B→A: {rules['borderline'][1]} {unit}")

            ax.set_xlabel(f"{selected_marker} ({unit})")
            ax.set_ylabel("Density")
            ax.set_title(f"{selected_marker} — {res['n_components']} clusters discovered")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


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
        patient_df["Value"] = patient_df["Value"].round(3)

        n_flagged = (patient_df["Note"] != "").sum()
        st.subheader(f"Patient {selected_patient}")
        st.caption(
            f"⚠️ flags markers where this patient is in the smallest cluster across all patients. "
            f"{n_flagged} of {len(patient_df)} markers flagged."
        )
        st.dataframe(patient_df, use_container_width=True, hide_index=True)
