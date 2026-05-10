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
from stub_data import generate_stub_data
from analysis import (
    analyse_upload, analyse_population, build_labelled_df, filter_long,
    most_separated_marker, strongest_marker_pair,
)

@st.cache_data
def load_stub_data():
    return generate_stub_data()

st.set_page_config(page_title="Classifier", layout="wide")

CLUSTER_COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Typography & rhythm ──────────────────────────────────────────────────────

st.markdown(
    """
<style>
.bt-headline {
    font-size: 2.0rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.015em;
    margin: 0.4rem 0 0.35rem;
}
.bt-sub {
    font-size: 1.05rem;
    line-height: 1.5;
    color: #4a4a4a;
    margin: 0 0 1.4rem;
    max-width: 70ch;
}
.bt-sub strong { color: #1f1f1f; }
.bt-next {
    margin: 1.4rem 0 0.4rem;
    padding: 0.6rem 0.9rem;
    border-left: 3px solid #4C72B0;
    background: rgba(76, 114, 176, 0.05);
    color: #2a2a2a;
    font-size: 0.97rem;
    line-height: 1.5;
    border-radius: 2px;
}
.bt-next strong { color: #1f1f1f; }
.bt-quiet { color: #6b6b6b; font-size: 0.9rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)


def headline(text: str) -> None:
    st.markdown(f'<div class="bt-headline">{text}</div>', unsafe_allow_html=True)


def sub(text: str) -> None:
    st.markdown(f'<div class="bt-sub">{text}</div>', unsafe_allow_html=True)


def next_step(text: str) -> None:
    st.markdown(f'<div class="bt-next">{text}</div>', unsafe_allow_html=True)


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


def group_plain_english(fp: pd.DataFrame, group_col: str, top_n: int = 5) -> list[str]:
    """Ranked plain-English bullets describing what defines a population group.

    Each bullet leads with the marker name, a strength word (notably / clearly /
    slightly), and the direction. The numeric effect size is appended as a
    quiet badge so an analyst can still see the magnitude without it dominating.
    """
    col = fp[group_col]
    col_sorted = col.abs().sort_values(ascending=False)
    lines = []
    for marker in col_sorted.index:
        z = col[marker]
        if abs(z) < 0.3:
            break
        direction = "higher" if z > 0 else "lower"
        strength = (
            "notably" if abs(z) > 1.0 else
            "clearly" if abs(z) > 0.6 else
            "slightly"
        )
        lines.append(
            f"- **{marker}** — {strength} {direction} than average "
            f"<span class='bt-quiet'>({z:+.1f})</span>"
        )
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
            "df_long_full":      df_long,
            "gmm_results_full":  gmm_results,
            "pop_results_full":  pop_results,
            "df_labelled_full":  df_labelled,
            "df_long":           df_long,
            "gmm_results":       gmm_results,
            "pop_results":       pop_results,
            "df_labelled":       df_labelled,
            "is_demo":           False,
            "file_id":           file_id,
            "recognised":        recognised,
            "unrecognised":      unrecognised,
            "current_filter_key": None,
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
    if "df_long_full" not in st.session_state:
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
        st.session_state.update({
            **_cached,
            "df_long_full":      _cached["df_long"],
            "gmm_results_full":  _cached["gmm_results"],
            "pop_results_full":  _cached["pop_results"],
            "df_labelled_full":  _cached["df_labelled"],
            "is_demo":           True,
            "file_id":           None,
            "current_filter_key": None,
        })

if st.session_state.get("is_demo"):
    st.info(
        "**Demo mode** — exploring 80 synthetic patients. Upload your own CSV in the top right."
    )

# ── Sidebar: cohort filters ──────────────────────────────────────────────────

def _reset_cohort_filters():
    for k in list(st.session_state.keys()):
        if k == "filter_age" or k == "filter_markers" or k.startswith("filter_marker_"):
            del st.session_state[k]

df_full = st.session_state["df_long_full"]
has_age_full = "age" in df_full.columns and df_full["age"].notna().any()

with st.sidebar:
    st.header("Cohort filters")
    st.caption("Restrict the analysis to a subset of patients. Both tabs re-run on the filtered cohort.")

    age_range = None
    if has_age_full:
        ages = pd.to_numeric(df_full.drop_duplicates("patient_id")["age"], errors="coerce").dropna()
        a_lo, a_hi = int(ages.min()), int(ages.max())
        if a_lo < a_hi:
            age_range = st.slider(
                "Age range", a_lo, a_hi, (a_lo, a_hi), key="filter_age",
            )
            if age_range == (a_lo, a_hi):
                age_range = None

    marker_options = sorted(df_full["test_name"].unique())
    chosen_markers = st.multiselect(
        "Filter by marker value", marker_options, key="filter_markers",
        help="Pick one or more markers, then drag each slider to the value range you want to keep.",
    )
    marker_ranges: dict = {}
    for m in chosen_markers:
        vals = df_full.loc[df_full["test_name"] == m, "value"].astype(float)
        m_lo, m_hi = float(vals.min()), float(vals.max())
        if m_lo >= m_hi:
            continue
        unit = THRESHOLDS[m]["unit"]
        sel = st.slider(
            f"{m} ({unit})", m_lo, m_hi, (m_lo, m_hi),
            key=f"filter_marker_{m}",
        )
        if sel != (m_lo, m_hi):
            marker_ranges[m] = sel

    if age_range is not None or marker_ranges:
        st.button("Reset filters", on_click=_reset_cohort_filters, use_container_width=True)

filter_key = (
    st.session_state.get("file_id"),
    age_range,
    tuple(sorted(marker_ranges.items())),
)

if st.session_state.get("current_filter_key") != filter_key:
    if age_range is None and not marker_ranges:
        st.session_state["df_long"]     = st.session_state["df_long_full"]
        st.session_state["gmm_results"] = st.session_state["gmm_results_full"]
        st.session_state["pop_results"] = st.session_state["pop_results_full"]
        st.session_state["df_labelled"] = st.session_state["df_labelled_full"]
    else:
        with st.spinner("Re-running analysis on filtered cohort…"):
            df_filtered = filter_long(
                st.session_state["df_long_full"],
                age_range=age_range,
                marker_ranges=marker_ranges,
            )
            n_filtered = df_filtered["patient_id"].nunique() if not df_filtered.empty else 0
            if n_filtered < 4:
                st.session_state["df_long"]     = df_filtered
                st.session_state["gmm_results"] = {}
                st.session_state["pop_results"] = {"error": f"Only {n_filtered} patients match the current filters — need at least 4."}
                st.session_state["df_labelled"] = df_filtered
            else:
                st.session_state["df_long"]     = df_filtered
                st.session_state["gmm_results"] = analyse_upload(df_filtered)
                st.session_state["pop_results"] = analyse_population(df_filtered)
                st.session_state["df_labelled"] = build_labelled_df(df_filtered, st.session_state["gmm_results"])
    st.session_state["current_filter_key"] = filter_key

n_full          = df_full["patient_id"].nunique()
n_filtered_now  = st.session_state["df_long"]["patient_id"].nunique() if not st.session_state["df_long"].empty else 0
filters_active  = age_range is not None or bool(marker_ranges)
if filters_active:
    st.success(
        f"**Filtered cohort:** analysing {n_filtered_now} of {n_full} patients. "
        "Use **Reset filters** in the sidebar to return to the full population.",
        icon="🔍",
    )

# ── Tab bar row: tabs on the left, unit preferences on the far right ──────────

_tab_col, _prefs_col = st.columns([7, 1])

with _prefs_col:
    unit_prefs = {}
    if MULTI_UNIT_MARKERS:
        with st.popover("⚙ Units", use_container_width=True):
            st.caption("Select the units your export uses.")
            for marker in MULTI_UNIT_MARKERS:
                units = available_units(marker)
                unit_prefs[marker] = st.selectbox(marker, units, key=f"unit_{marker}")
    else:
        # No multi-unit markers — still consume the column so layout is consistent
        for marker in MULTI_UNIT_MARKERS:
            unit_prefs[marker] = available_units(marker)[0]

with _tab_col:
    tab1, tab2, tab3 = st.tabs([
        "1. What patterns are in my data?",
        "2. What types of patient exist?",
        "3. Which markers move together?",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — How does my population look?
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    if "df_long" in st.session_state:
        df_long     = st.session_state["df_long"]
        gmm_results = st.session_state["gmm_results"]

        n_patients = df_long["patient_id"].nunique() if not df_long.empty else 0
        n_markers  = df_long["test_name"].nunique() if not df_long.empty else 0

    if "df_long" in st.session_state and (n_patients < 4 or not gmm_results):
        st.warning(
            f"Cohort too small for analysis ({n_patients} patients). "
            "Loosen or reset the filters in the sidebar."
        )
    elif "df_long" in st.session_state:
        available = [t for t in gmm_results if "error" not in gmm_results[t]]
        errored   = [t for t in gmm_results if "error" in gmm_results[t]]

        # Auto-pick the marker with the clearest cluster separation as the
        # default. The user can switch beneath the chart.
        pick = most_separated_marker(gmm_results)
        auto_choice = pick[0] if pick else (available[0] if available else None)
        if "tab1_selected_marker" not in st.session_state or \
                st.session_state["tab1_selected_marker"] not in available:
            st.session_state["tab1_selected_marker"] = auto_choice
        selected_marker = st.session_state["tab1_selected_marker"]

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

            n_comp = res["n_components"]
            weights = res["weights"]

            # ── Headline + sub-headline ──────────────────────────────────────
            if n_comp >= 2:
                groups_sorted = sorted(
                    [(float(means_d[i]), float(weights[i])) for i in range(n_comp)],
                    key=lambda x: -x[1],
                )
                biggest, second = groups_sorted[0], groups_sorted[1]
                why = (
                    " <span class='bt-quiet'>(it has the clearest split in your data)</span>"
                    if selected_marker == auto_choice else ""
                )
                headline(
                    f"Your {n_patients} patients form "
                    f"<strong>{n_comp} natural groups</strong> in {selected_marker}"
                )
                sub(
                    f"We started you on <strong>{selected_marker}</strong>{why}. "
                    f"The largest group ({biggest[1]*100:.0f}% of patients) sits around "
                    f"<strong>{biggest[0]:.2f} {disp_unit}</strong>; "
                    f"the next ({second[1]*100:.0f}%) around "
                    f"<strong>{second[0]:.2f} {disp_unit}</strong>. "
                    f"Switch markers below to explore others."
                )
            else:
                headline(f"<strong>{selected_marker}</strong> looks uniform across your patients")
                sub("No clear sub-groups in this marker — the values cluster around a single average.")

            if res["small_sample"]:
                st.warning(
                    f"Small sample ({len(res['values'])} patients) — "
                    "treat these groups as a starting point. Results will stabilise around 100+ patients."
                )

            # Chart — leading visual
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
                xaxis_title=f"{selected_marker} ({disp_unit})",
                yaxis_title="How many patients",
                yaxis=dict(rangemode="tozero"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
                height=440,
                margin=dict(t=40, l=10, r=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Marker switcher ──────────────────────────────────────────────
            sw_a, sw_b = st.columns([1, 4])
            sw_a.markdown(
                "<div style='padding-top:0.5rem;' class='bt-quiet'>Explore another marker</div>",
                unsafe_allow_html=True,
            )
            new_pick = sw_b.selectbox(
                "Marker", available,
                index=available.index(selected_marker),
                key="tab1_marker_picker", label_visibility="collapsed",
            )
            if new_pick != selected_marker:
                st.session_state["tab1_selected_marker"] = new_pick
                st.rerun()

            # ── Group statistics (folded) ────────────────────────────────────
            labels = res["labels"]
            with st.expander("Group statistics", expanded=False):
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

            with st.expander("Technical details", expanded=False):
                bic_str = " · ".join(
                    f"{n} groups: {int(bic)}" for n, bic in sorted(res["bic_scores"].items())
                )
                st.caption(
                    f"Group count selected by BIC scoring (lower = better fit): {bic_str}. "
                    f"{res['n_components']} groups selected."
                )
                if errored:
                    st.caption(f"Skipped (too few data points): {', '.join(errored)}")

            st.markdown(
                f"<div class='bt-quiet'>{n_patients} patients · {n_markers} markers · {sample_size_label(n_patients)}</div>",
                unsafe_allow_html=True,
            )

            next_step(
                "→ Now see how these groups map onto whole-patient profiles in "
                "<strong>2. What types of patient exist?</strong>"
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

            # ── Headline + sub-headline ──────────────────────────────────────
            fp_round = pop["fingerprint"].round(2)
            type_summaries = []
            for g in range(n_clusters):
                col = fp_round[f"Group {g + 1}"]
                top_marker = col.abs().sort_values(ascending=False).index[0]
                z = float(col[top_marker])
                direction = "higher" if z > 0 else "lower"
                n_in_group = int((labels == g).sum())
                ages_g = [
                    age_lookup.get(pid) for pid, lbl in zip(patient_ids, labels)
                    if lbl == g and age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid))
                ]
                age_bit = f", median age {int(np.median(ages_g))}" if ages_g else ""
                type_summaries.append(
                    f"<strong>Type {g + 1}</strong> ({n_in_group} patients{age_bit}) "
                    f"— most distinct in <strong>{top_marker}</strong> ({direction} than average)"
                )

            type_word = "type" if n_clusters == 1 else "distinct types"
            headline(
                f"Your population splits into <strong>{n_clusters} {type_word}</strong>"
            )
            sub(" &nbsp;·&nbsp; ".join(type_summaries))

            if pop["small_sample"]:
                st.warning(
                    f"Small sample ({n_patients} patients) — patient types are a starting point. "
                    "Results will stabilise around 200+ patients."
                )

            # ── Patient map ──────────────────────────────────────────────────
            st.markdown(
                "<div class='bt-quiet'>Each dot is a patient. Patients close together "
                "have similar overall blood profiles.</div>",
                unsafe_allow_html=True,
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
                ages_for_type = [
                    age_lookup.get(pid) for pid in members
                    if age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid))
                ]
                with type_cols[g]:
                    colour = CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]
                    st.markdown(
                        f"<h4 style='color:{colour}'>Type {g + 1}</h4>",
                        unsafe_allow_html=True,
                    )
                    if ages_for_type:
                        median_age = int(np.median(ages_for_type))
                        a_lo, a_hi = int(min(ages_for_type)), int(max(ages_for_type))
                        st.caption(
                            f"{len(members)} patients · median age **{median_age}** "
                            f"(range {a_lo}–{a_hi})"
                        )
                    else:
                        st.caption(f"{len(members)} patients")
                    if bullets:
                        st.markdown("\n".join(bullets), unsafe_allow_html=True)
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

            # ── Patients to investigate ──────────────────────────────────────
            st.subheader("Patients to investigate")
            st.write(
                "Patients on the boundary between types or with unusual overall profiles. "
                "These are the leads worth a closer look."
            )

            posts        = pop["posteriors"]
            ll           = pop["log_likelihood"]
            max_post     = posts.max(axis=1)
            second_post  = np.sort(posts, axis=1)[:, -2] if posts.shape[1] >= 2 else np.zeros(len(posts))
            outlier_thresh = float(np.percentile(ll, 5)) if len(ll) >= 20 else float(ll.min() - 1)

            inv_rows = []
            for i, pid in enumerate(patient_ids):
                reasons = []
                is_boundary = max_post[i] < 0.7 and posts.shape[1] >= 2
                is_outlier  = ll[i] <= outlier_thresh
                if is_boundary:
                    a, b = np.argsort(posts[i])[::-1][:2]
                    reasons.append(
                        f"borderline ({posts[i, a]*100:.0f}% Type {a+1} / {posts[i, b]*100:.0f}% Type {b+1})"
                    )
                if is_outlier:
                    reasons.append("multivariate outlier")
                if reasons:
                    inv_rows.append({
                        "Patient": pid,
                        "Age": int(age_lookup[pid]) if age_lookup.get(pid) is not None and pd.notna(age_lookup.get(pid)) else "—",
                        "Type": f"Type {labels[i] + 1}",
                        "Confidence": f"{max_post[i]*100:.0f}%",
                        "Why investigate": " · ".join(reasons),
                    })

            if inv_rows:
                inv_df = pd.DataFrame(inv_rows).sort_values("Confidence")
                st.dataframe(inv_df, use_container_width=True, hide_index=True)
                st.caption(
                    f"{sum(1 for r in inv_rows if 'borderline' in r['Why investigate'])} borderline · "
                    f"{sum(1 for r in inv_rows if 'outlier' in r['Why investigate'])} outliers · "
                    f"{len(inv_rows)} of {n_patients} patients flagged"
                )
            else:
                st.caption("No boundary cases or multivariate outliers — every patient sits squarely inside their type.")

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

            st.markdown(
                f"<div class='bt-quiet'>{n_patients} patients · {pop['df_wide'].shape[1]} markers used · {sample_size_label(n_patients)}</div>",
                unsafe_allow_html=True,
            )

            next_step(
                "→ Curious which markers move together across these patients? "
                "Open <strong>3. Which markers move together?</strong>"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Pairwise marker view
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    if "df_long" not in st.session_state or st.session_state["df_long"].empty:
        st.info("Upload a CSV in the first tab, or loosen the cohort filters in the sidebar.")
    else:
        df_long = st.session_state["df_long"]
        markers_in_data = sorted(df_long["test_name"].unique())

        if len(markers_in_data) < 2:
            st.warning("Need at least 2 markers in the cohort to compare pairs.")
        else:
            # Auto-pick the strongest correlation pair as the default.
            auto_pair = strongest_marker_pair(df_long)
            if "pair_x" not in st.session_state or st.session_state["pair_x"] not in markers_in_data:
                st.session_state["pair_x"] = auto_pair[0] if auto_pair else markers_in_data[0]
            if "pair_y" not in st.session_state or st.session_state["pair_y"] not in markers_in_data \
                    or st.session_state["pair_y"] == st.session_state["pair_x"]:
                st.session_state["pair_y"] = (
                    auto_pair[1] if auto_pair else
                    next((m for m in markers_in_data if m != st.session_state["pair_x"]), markers_in_data[0])
                )

            # ── Headline + sub-headline ──────────────────────────────────────
            current_x = st.session_state["pair_x"]
            current_y = st.session_state["pair_y"]
            wide_for_r = (
                df_long[df_long["test_name"].isin([current_x, current_y])]
                .pivot_table(index="patient_id", columns="test_name", values="value")
                .dropna()
            )
            current_r = (
                float(wide_for_r[current_x].corr(wide_for_r[current_y]))
                if len(wide_for_r) >= 4 and current_x in wide_for_r.columns and current_y in wide_for_r.columns
                else None
            )
            showing_auto = (
                auto_pair is not None and (
                    {current_x, current_y} == {auto_pair[0], auto_pair[1]}
                )
            )

            if current_r is not None:
                strength_word = (
                    "strong" if abs(current_r) > 0.6 else
                    "moderate" if abs(current_r) > 0.3 else
                    "weak"
                )
                dir_word = "positive" if current_r > 0 else "negative"
                why = (
                    " <span class='bt-quiet'>(it's the strongest relationship in your data)</span>"
                    if showing_auto else ""
                )
                opener = "We started you on this pair" if showing_auto else "Showing"
                headline(
                    f"<strong>{current_x}</strong> and <strong>{current_y}</strong> "
                    f"({'track each other' if abs(current_r) > 0.3 else 'show little relationship'}, "
                    f"r = {current_r:+.2f})"
                )
                sub(
                    f"{opener}{why}. The {strength_word} {dir_word} correlation "
                    f"means {('knowing one tells you something about the other.' if abs(current_r) > 0.3 else 'these two markers move largely independently in your population.')} "
                    f"Switch markers below to explore other pairs."
                )
            else:
                headline("Compare any two markers")
                sub(
                    "Pick an X and Y marker below. Coupled patterns "
                    "(e.g. testosterone &amp; SHBG, HbA1c &amp; HDL) jump out here in a way "
                    "the per-marker view cannot show."
                )

            sel_a, sel_b, sel_c = st.columns([3, 3, 2])
            x_marker = sel_a.selectbox("X axis", markers_in_data, key="pair_x")
            y_options = [m for m in markers_in_data if m != x_marker]
            if st.session_state["pair_y"] not in y_options:
                st.session_state["pair_y"] = y_options[0]
            y_marker = sel_b.selectbox("Y axis", y_options, key="pair_y")

            pop_for_colour = st.session_state.get("pop_results", {})
            colour_choices = ["Population type"] if "labels" in pop_for_colour else []
            colour_choices.append("None")
            colour_by = sel_c.radio("Colour by", colour_choices, key="pair_colour")

            wide = (
                df_long[df_long["test_name"].isin([x_marker, y_marker])]
                .pivot_table(index="patient_id", columns="test_name", values="value")
                .dropna()
            )

            if len(wide) < 4:
                st.warning(
                    f"Only {len(wide)} patients have both {x_marker} and {y_marker} measured. "
                    "Pick another pair or widen the cohort."
                )
            else:
                disp_x = unit_prefs.get(x_marker, THRESHOLDS[x_marker]["unit"])
                disp_y = unit_prefs.get(y_marker, THRESHOLDS[y_marker]["unit"])
                xs = wide[x_marker].apply(lambda v: from_canonical(x_marker, v, disp_x))
                ys = wide[y_marker].apply(lambda v: from_canonical(y_marker, v, disp_y))

                fig = go.Figure()

                if colour_by == "Population type" and "labels" in pop_for_colour:
                    pop_label_lookup = dict(zip(pop_for_colour["patient_ids"], pop_for_colour["labels"]))
                    n_clusters_p = pop_for_colour["n_clusters"]
                    for g in range(n_clusters_p):
                        mask = [pop_label_lookup.get(pid) == g for pid in wide.index]
                        if not any(mask):
                            continue
                        fig.add_trace(go.Scatter(
                            x=xs[mask], y=ys[mask],
                            mode="markers",
                            marker=dict(size=10, color=CLUSTER_COLOURS[g % len(CLUSTER_COLOURS)]),
                            name=f"Type {g + 1}",
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
                fig.add_vline(
                    x=ref_x, line=dict(color="green", width=1.5, dash="dot"),
                    annotation_text=f"{x_marker} ref: {ref_x:.2f}",
                    annotation_position="bottom right",
                )
                fig.add_hline(
                    y=ref_y, line=dict(color="green", width=1.5, dash="dot"),
                    annotation_text=f"{y_marker} ref: {ref_y:.2f}",
                    annotation_position="top left",
                )

                fig.update_layout(
                    xaxis_title=f"{x_marker} ({disp_x})",
                    yaxis_title=f"{y_marker} ({disp_y})",
                    height=520,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                corr = float(xs.corr(ys))
                strength = (
                    "strong" if abs(corr) > 0.6 else
                    "moderate" if abs(corr) > 0.3 else
                    "weak"
                )
                direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"
                st.markdown(
                    f"<div class='bt-quiet'>r = {corr:+.2f} · {strength} {direction} correlation "
                    f"across {len(wide)} patients with both markers measured.</div>",
                    unsafe_allow_html=True,
                )

                next_step(
                    "→ Want to slice this view to a specific cohort? "
                    "Open <strong>Cohort filters</strong> in the left sidebar."
                )
