import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

from thresholds import classify_test, THRESHOLDS
from column_map import COLUMN_MAP, ID_COLUMN

st.set_page_config(page_title="Blood Test Classifier", layout="wide")
st.title("Blood Test Classifier")

# ── Helpers ──────────────────────────────────────────────────────────────────

CATEGORY_COLOURS = {"Normal": "green", "Borderline": "orange", "Abnormal": "red"}

def colour_cell(val):
    colour = CATEGORY_COLOURS.get(val, "")
    return f"color: {colour}; font-weight: bold" if colour else ""


@st.cache_resource
def load_models():
    with open("blood_test_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("category_encoder.pkl", "rb") as f:
        cat_enc = pickle.load(f)
    with open("gmm_results.pkl", "rb") as f:
        gmm_results = pickle.load(f)
    test_name_enc = LabelEncoder()
    test_name_enc.fit(list(THRESHOLDS.keys()))
    return clf, cat_enc, gmm_results, test_name_enc


def predict_tree(test_name, value, clf, cat_enc, test_name_enc):
    enc = test_name_enc.transform([test_name])[0]
    X = pd.DataFrame([[enc, value]], columns=["Test Name Encoded", "value"])
    return cat_enc.inverse_transform(clf.predict(X))[0]


def predict_gmm(test_name, value, gmm_results):
    result = gmm_results[test_name]
    b0, b1 = result["boundaries"]
    component = 0 if value <= b0 else (1 if value <= b1 else 2)
    return result["label_map"].get(component, "Unknown")


def classify_row(row, clf, cat_enc, gmm_results, test_name_enc):
    records = []
    for col, mapping in COLUMN_MAP.items():
        if col not in row.index:
            continue
        raw = row[col]
        if pd.isna(raw):
            continue
        value = float(raw) * mapping["scale"]
        test_name = mapping["test"]
        records.append({
            "Test":       test_name,
            "Value":      round(value, 3),
            "Unit":       THRESHOLDS[test_name]["unit"],
            "Rules":      classify_test(test_name, value),
            "Tree":       predict_tree(test_name, value, clf, cat_enc, test_name_enc),
            "GMM":        predict_gmm(test_name, value, gmm_results),
        })
    return records


# ── Load models ───────────────────────────────────────────────────────────────

try:
    clf, cat_enc, gmm_results, test_name_enc = load_models()
    models_ready = True
except FileNotFoundError:
    models_ready = False
    st.error(
        "Model files not found. Run the following in your terminal first:\n\n"
        "```\npython3 generate_mock_data.py\n"
        "python3 train_classifier.py\n"
        "python3 train_gmm.py\n```"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Single Prediction", "CSV Upload", "Validation Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single Prediction
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.header("Single Prediction")
    st.write("Enter a test name and value to see how all three approaches classify it.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        test_names = list(THRESHOLDS.keys())
        selected_test = st.selectbox("Test", test_names)
        unit = THRESHOLDS[selected_test]["unit"]
        value = st.number_input(f"Value ({unit})", format="%.4f", value=0.0)

    if st.button("Classify", disabled=not models_ready):
        rules_pred  = classify_test(selected_test, value)
        tree_pred   = predict_tree(selected_test, value, clf, cat_enc, test_name_enc)
        gmm_pred    = predict_gmm(selected_test, value, gmm_results)

        with col_b:
            st.subheader("Results")
            for label, pred in [("Rules", rules_pred), ("Decision Tree", tree_pred), ("GMM", gmm_pred)]:
                colour = CATEGORY_COLOURS.get(pred, "grey")
                st.markdown(f"**{label}:** :{colour}[{pred}]")

            if rules_pred == tree_pred == gmm_pred:
                st.success("All three approaches agree.")
            else:
                st.warning("Approaches disagree — value may be near a boundary.")

        rules = THRESHOLDS[selected_test]
        st.divider()
        st.caption(
            f"Reference ranges — Normal: {rules['normal'][0]}–{rules['normal'][1]} {unit} | "
            f"Borderline: {rules['borderline'][0]}–{rules['borderline'][1]} {unit}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — CSV Upload
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header("CSV Upload")
    st.write("Upload a blood test export. Recognised columns will be classified automatically.")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        # Drop blank rows (rows where the ID column is empty)
        df_raw = df_raw[df_raw[ID_COLUMN].notna() & (df_raw[ID_COLUMN].astype(str).str.strip() != "")]
        df_raw = df_raw.reset_index(drop=True)

        # Show column mapping status
        recognised = [c for c in COLUMN_MAP if c in df_raw.columns]
        unrecognised = [c for c in df_raw.columns if c not in COLUMN_MAP and c != ID_COLUMN]

        with st.expander("Column mapping", expanded=False):
            st.write(f"**Recognised ({len(recognised)}):** {', '.join([COLUMN_MAP[c]['test'] for c in recognised])}")
            st.write(f"**Not yet in thresholds ({len(unrecognised)}):** {', '.join(unrecognised)}")

        if not recognised:
            st.error("No recognised columns found in this file.")
        elif not models_ready:
            st.error("Models not loaded — cannot classify.")
        else:
            # Build results table
            all_records = []
            for _, row in df_raw.iterrows():
                patient_id = row[ID_COLUMN]
                for rec in classify_row(row, clf, cat_enc, gmm_results, test_name_enc):
                    all_records.append({"Patient ID": patient_id, **rec})

            results_df = pd.DataFrame(all_records)

            # Store in session state for validation tab
            st.session_state["uploaded_results"] = results_df
            st.session_state["uploaded_raw"] = df_raw

            st.subheader(f"Results — {len(df_raw)} patients, {len(recognised)} tests")

            styled = (
                results_df.style
                .applymap(colour_cell, subset=["Rules", "Tree", "GMM"])
            )
            st.dataframe(styled, use_container_width=True)

            # Summary
            st.subheader("Summary")
            summary_cols = st.columns(3)
            for i, approach in enumerate(["Rules", "Tree", "GMM"]):
                counts = results_df[approach].value_counts()
                with summary_cols[i]:
                    st.write(f"**{approach}**")
                    for cat in ["Normal", "Borderline", "Abnormal"]:
                        n = counts.get(cat, 0)
                        colour = CATEGORY_COLOURS[cat]
                        st.markdown(f":{colour}[{cat}: {n}]")

            # Agreement flags
            disagree = results_df[
                (results_df["Rules"] != results_df["Tree"]) |
                (results_df["Rules"] != results_df["GMM"])
            ]
            if not disagree.empty:
                st.subheader("Disagreements")
                st.dataframe(disagree, use_container_width=True)

            # Download
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                "Download results as CSV",
                data=csv_out,
                file_name="classified_results.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Validation Dashboard
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Validation Dashboard")

    if not models_ready:
        st.error("Models not loaded.")
    else:
        source = "uploaded" if "uploaded_raw" in st.session_state else "mock"
        if source == "uploaded":
            st.info("Using uploaded patient data.")
            df_raw = st.session_state["uploaded_raw"]
            # Rebuild long-format data from recognised columns
            rows = []
            for _, row in df_raw.iterrows():
                for col, mapping in COLUMN_MAP.items():
                    if col not in df_raw.columns or pd.isna(row.get(col)):
                        continue
                    rows.append({
                        "test_name": mapping["test"],
                        "value":     float(row[col]) * mapping["scale"],
                    })
            df = pd.DataFrame(rows)
        else:
            st.info("No CSV uploaded — using mock data. Upload a file in the CSV tab to validate against real data.")
            from sqlalchemy import create_engine
            engine = create_engine("sqlite:///blood_tests.db")
            df = pd.read_sql("SELECT * FROM blood_tests", con=engine)

        # Agreement
        df["pred_rules"] = df.apply(lambda r: classify_test(r["test_name"], r["value"]), axis=1)
        enc = test_name_enc.transform(df["test_name"])
        X = pd.DataFrame({"Test Name Encoded": enc, "value": df["value"].values})
        df["pred_tree"] = cat_enc.inverse_transform(clf.predict(X))
        df["pred_gmm"]  = df.apply(
            lambda r: predict_gmm(r["test_name"], r["value"], gmm_results)
            if r["test_name"] in gmm_results else "Unknown", axis=1
        )

        def agree_pct(a, b):
            return f"{(a == b).mean() * 100:.1f}%"

        st.subheader("Agreement Between Approaches")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rules vs Tree", agree_pct(df["pred_rules"], df["pred_tree"]))
        col2.metric("Rules vs GMM",  agree_pct(df["pred_rules"], df["pred_gmm"]))
        col3.metric("Tree vs GMM",   agree_pct(df["pred_tree"],  df["pred_gmm"]))

        # Boundary comparison
        st.subheader("GMM Boundary vs Known Thresholds")
        boundary_rows = []
        for test_name in gmm_results:
            rules = THRESHOLDS[test_name]
            b0, b1 = gmm_results[test_name]["boundaries"]
            boundary_rows.append({
                "Test":         test_name,
                "Known N→B":    rules["normal"][1],
                "GMM N→B":      round(b0, 3),
                "Diff N→B":     f"{abs(b0 - rules['normal'][1]) / rules['normal'][1] * 100:.1f}%",
                "Known B→A":    rules["borderline"][1],
                "GMM B→A":      round(b1, 3),
                "Diff B→A":     f"{abs(b1 - rules['borderline'][1]) / rules['borderline'][1] * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(boundary_rows), use_container_width=True)

        # Histograms
        st.subheader("Distributions")
        available_tests = df["test_name"].unique().tolist()
        selected_plot = st.selectbox("Select test to plot", available_tests)

        values = df[df["test_name"] == selected_plot]["value"].values
        if selected_plot in gmm_results and len(values) > 0:
            res = gmm_results[selected_plot]
            means, stds, weights = res["means"], res["stds"], res["weights"]
            x = np.linspace(values.min() - values.std(), values.max() + values.std(), 500)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(values, bins=40, density=True, alpha=0.35, color="steelblue", label="Data")

            for i, (m, s, w) in enumerate(zip(means, stds, weights)):
                label = res["label_map"].get(i, f"Component {i}")
                colour = CATEGORY_COLOURS.get(label, "grey")
                ax.plot(x, w * norm.pdf(x, m, s), color=colour, linewidth=2, label=f"GMM: {label}")

            total = sum(w * norm.pdf(x, m, s) for m, s, w in zip(means, stds, weights))
            ax.plot(x, total, "k--", linewidth=1, alpha=0.5, label="GMM total")

            for boundary in res["boundaries"]:
                ax.axvline(boundary, color="orange", linestyle="--", linewidth=1.5,
                           label=f"GMM boundary: {boundary:.3f}")

            rules = THRESHOLDS[selected_plot]
            ax.axvline(rules["normal"][1], color="green", linestyle=":", linewidth=1.5,
                       label=f"Rules N→B: {rules['normal'][1]}")
            ax.axvline(rules["borderline"][1], color="red", linestyle=":", linewidth=1.5,
                       label=f"Rules B→A: {rules['borderline'][1]}")

            ax.set_xlabel(f"Value ({THRESHOLDS[selected_plot]['unit']})")
            ax.set_ylabel("Density")
            ax.set_title(selected_plot)
            ax.legend(fontsize=8)
            st.pyplot(fig)
        else:
            st.info("No GMM data available for this test yet.")
