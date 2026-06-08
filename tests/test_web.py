"""Web-layer integration tests covering routes, filter flows, upload, units,
and the methodology drawer.

Uses FastAPI's TestClient — no live server required. State is reset before
each test via the autouse `reset_state` fixture, so upload tests can mutate
the global AppState without bleeding into subsequent tests.
"""
from __future__ import annotations

import io
import re
import textwrap

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from web.main import app
from web.state import _filtered_data_cached, _load_demo, state

client = TestClient(app)


# ── Test isolation ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state() -> None:
    """Restore demo data and clear unit preferences before every test."""
    _load_demo()
    state.unit_prefs = {}
    state.upload_filename = None
    state.last_upload_error = None
    _filtered_data_cached.cache_clear()


@pytest.fixture
def demo_marker() -> str:
    """A marker guaranteed to exist in the demo cohort."""
    return sorted(state.df_long_full["test_name"].unique())[0]


@pytest.fixture
def demo_marker_pair() -> tuple[str, str]:
    markers = sorted(state.df_long_full["test_name"].unique())
    return markers[0], markers[1]


# ── /healthz ─────────────────────────────────────────────────────────────────

def test_healthz_returns_ok() -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


# ── Index page + tab fallback ────────────────────────────────────────────────

@pytest.mark.parametrize("tab", ["explorer", "population", "investigate", "pairs"])
def test_index_each_tab_renders(tab: str) -> None:
    """Full page renders end-to-end and dispatches to the requested tab.

    A 200 proves the shell + tab partial rendered without a template error.
    The tab strip marks exactly the requested tab `is-active` (driven by the
    same `active` var that selects the body partial), so it's the structural
    signal of correct dispatch — no need to pin tab-specific copy.
    """
    res = client.get(f"/?tab={tab}")
    assert res.status_code == 200
    assert re.search(rf'class="tab is-active"\s+href="/\?tab={tab}', res.text)


def test_index_default_tab_is_explorer() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert 'data-mdr-section="explorer"' in res.text


def test_index_unknown_tab_falls_back_to_explorer() -> None:
    res = client.get("/?tab=nonsense")
    assert res.status_code == 200
    assert 'data-mdr-section="explorer"' in res.text


def test_index_includes_demo_banner() -> None:
    res = client.get("/")
    assert "Demo mode" in res.text
    assert "80 synthetic blood tests" in res.text


# ── Unified toolbar ──────────────────────────────────────────────────────────

def test_toolbar_groups_controls_in_one_row() -> None:
    """The per-view control, cohort chip, and help icon share one toolbar
    between the tab strip and the body — not three stacked control bands."""
    html = client.get("/?tab=explorer").text
    assert html.count('class="toolbar"') == 1
    bar = html.find('class="toolbar"')
    assert html.find('class="tab-strip"') < bar < html.find('id="tab-body"')
    assert 'id="marker-picker"' in html              # per-view control (left)
    assert "cohort-summary" in html                  # cohort chip (right)
    assert 'aria-label="How this works"' in html     # help icon (right)


def test_outliers_tab_has_no_per_view_control() -> None:
    """Outliers has no view-specific control, so the toolbar's left slot is
    empty there — the cohort chip and help icon still render."""
    html = client.get("/?tab=investigate").text
    assert "toolbar-control" not in html
    assert "cohort-summary" in html


def test_refit_indicator_wired_to_results() -> None:
    """The results carry an hx-indicator + overlay so cohort recomputes dim
    the chart and show 'Updating…'. Controls inside #page-body inherit it."""
    html = client.get("/").text
    assert 'id="results-wrap"' in html
    assert 'hx-indicator="#results-wrap"' in html
    assert 'class="refit-overlay"' in html
    assert "Updating" in html


def test_cohort_popover_closed_by_default_open_while_editing() -> None:
    open_re = r'id="cohort"[^>]* open>'
    plain = client.get("/?tab=explorer").text
    assert not re.search(open_re, plain)             # collapsed on a normal load
    # Editing a filter re-renders the popover expanded so it stays open.
    edited = client.get("/filters/set?tab=explorer&age_min=35&age_max=55").text
    assert re.search(open_re, edited) and _is_filtered(edited)
    # Reset collapses it again.
    assert not re.search(open_re, client.get("/filters/reset?tab=explorer").text)


def test_investigate_sets_aside_and_annotates_imputed_patients() -> None:
    """M2: a mostly-imputed patient is set aside from flagging (and counted),
    while a partially-imputed flagged patient is annotated with its fraction."""
    import numpy as np

    from web.contexts import _investigate_context

    pop = {
        "patient_ids":   ["P0", "P1", "P2"],
        "labels":        np.array([0, 1, 0]),
        # P0 would be a boundary case (0.5/0.5); P2 is too (0.6/0.4)
        "posteriors":    np.array([[0.5, 0.5], [0.99, 0.01], [0.6, 0.4]]),
        "mahalanobis_sq": np.array([0.1, 0.1, 0.1]),
        "imputed_frac":  np.array([0.8, 0.0, 0.3]),   # P0 heavy, P2 partial
        "n_clusters":    2,
        "n_cluster_dims": 2,
    }
    df = pd.DataFrame([
        {"patient_id": p, "age": 40, "test_name": "X", "value": 1.0, "unit": "u"}
        for p in ["P0", "P1", "P2"]
    ])
    res = _investigate_context({"df_long": df, "pop_results": pop})

    assert res["n_low_data"] == 1                       # P0 set aside
    flagged = {r["patient"] for r in res["rows"]}
    assert "P0" not in flagged                          # heavily-imputed → not flagged
    p2 = next(r for r in res["rows"] if r["patient"] == "P2")
    assert p2["imputed_pct"] == 30                      # partial imputation annotated


def test_groups_toolbar_is_simplified() -> None:
    """The Groups toolbar is just the marker picker — no 'Showing' label and no
    'auto-picked … N groups · N tests' status text."""
    html = client.get("/?tab=explorer").text
    assert 'id="marker-picker"' in html          # the picker stays
    assert "Showing" not in html                 # label removed
    assert "auto-picked" not in html             # status text removed


def test_below_evidence_floor_copy_is_honest() -> None:
    """A cohort below the K>1 evidence floor must say only one cluster was
    *evaluated* — not imply BIC compared candidates and chose it."""
    # age 45-65 → 35 demo patients, below the 50-patient population floor
    html = client.get("/?tab=population&age_min=45&age_max=65").text
    assert "Only a single cluster was evaluated" in html
    assert "Number of clusters selected by BIC" not in html


# ── /tab/{name} partials ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tab", ["explorer", "population", "investigate", "pairs"])
def test_tab_partial_returns_page_body(tab: str) -> None:
    res = client.get(f"/tab/{tab}")
    assert res.status_code == 200
    # Partial fragments must not include a full HTML shell.
    assert "<html" not in res.text
    assert "<head" not in res.text
    assert 'id="tab-body"' in res.text
    assert f'data-mdr-section="{tab}"' in res.text


def test_tab_partial_unknown_falls_back_to_explorer() -> None:
    res = client.get("/tab/garbage")
    assert res.status_code == 200
    assert 'data-mdr-section="explorer"' in res.text


# ── /marker partial ──────────────────────────────────────────────────────────

def test_marker_partial_returns_chart(demo_marker: str) -> None:
    res = client.get(f"/marker?name={demo_marker}")
    assert res.status_code == 200
    # Plotly inlines the chart as a div.
    assert "plotly-graph-div" in res.text
    assert 'data-mdr-section="explorer"' in res.text


def test_marker_partial_unknown_marker_falls_back() -> None:
    res = client.get("/marker?name=ThisDoesNotExist")
    assert res.status_code == 200
    # Falls back to the first available marker, so we still get a chart.
    assert "plotly-graph-div" in res.text


# ── /population/scatter partial ──────────────────────────────────────────────

@pytest.mark.parametrize("colour_by", ["type", "age"])
def test_population_scatter_partial(colour_by: str) -> None:
    res = client.get(f"/population/scatter?colour_by={colour_by}")
    assert res.status_code == 200
    assert "plotly-graph-div" in res.text


def test_population_scatter_colour_by_changes_the_chart() -> None:
    """`type` and `age` must produce genuinely different charts: discrete
    cluster traces vs a continuous age colour-scale. A route that ignored
    `colour_by` would emit identical markup for both."""
    by_type = client.get("/population/scatter?colour_by=type")
    by_age = client.get("/population/scatter?colour_by=age")
    assert by_type.status_code == 200 and by_age.status_code == 200
    assert by_type.text != by_age.text
    # type → named cluster traces; age → a continuous "Age" colour bar.
    assert "Cluster 1" in by_type.text and "Cluster 1" not in by_age.text
    assert "Age" in by_age.text and "Age" not in by_type.text


def test_population_scatter_invalid_colour_by_defaults_to_type() -> None:
    garbage = client.get("/population/scatter?colour_by=garbage")
    assert garbage.status_code == 200
    # Invalid input must render as `type` (cluster traces, no age colour bar),
    # not silently fall through to `age`.
    assert "Cluster 1" in garbage.text and "Age" not in garbage.text


# ── Correlation robustness (#61) ──────────────────────────────────────────────

def _pair_df(x_vals, y_vals, xm="ALT", ym="ALP"):
    rows = []
    for i, (xv, yv) in enumerate(zip(x_vals, y_vals, strict=True)):
        rows.append({"patient_id": f"P{i}", "age": 40, "test_name": xm,
                     "value": float(xv), "unit": "U/L"})
        rows.append({"patient_id": f"P{i}", "age": 40, "test_name": ym,
                     "value": float(yv), "unit": "U/L"})
    return pd.DataFrame(rows)


def test_pair_flags_pearson_spearman_divergence() -> None:
    """A single bivariate outlier makes Pearson and Spearman disagree → flagged."""
    from web.contexts import _pair_context

    x = list(range(15))
    y = list(range(15))
    y[0] = 1000  # outlier: low x, huge y — wrecks Pearson, dents Spearman less
    ctx = _pair_context({"df_long": _pair_df(x, y), "pop_results": {}}, "ALT", "ALP")
    assert ctx["diverges"] is True
    assert abs(ctx["r"] - ctx["rho"]) > 0.2


def test_pair_strong_claim_requires_enough_overlap() -> None:
    """A clean correlation on too few points is flagged low-overlap (tentative)."""
    from web.contexts import _pair_context

    x = list(range(15))                      # n=15 < the 25-point "strong" floor
    ctx = _pair_context({"df_long": _pair_df(x, x), "pop_results": {}}, "ALT", "ALP")
    assert ctx["low_overlap"] is True


def test_pair_clean_large_sample_not_flagged() -> None:
    from web.contexts import _pair_context

    x = [float(i) for i in range(30)]        # perfect linear, n=30 ≥ floor
    ctx = _pair_context({"df_long": _pair_df(x, x), "pop_results": {}}, "ALT", "ALP")
    assert ctx["low_overlap"] is False
    assert ctx["diverges"] is False


# ── /pair partial ────────────────────────────────────────────────────────────

def test_pair_partial_no_params_picks_strongest_pair() -> None:
    res = client.get("/pair")
    assert res.status_code == 200
    assert "plotly-graph-div" in res.text
    assert 'data-mdr-section="pairs"' in res.text


def test_pair_partial_explicit_pair(demo_marker_pair: tuple[str, str]) -> None:
    x, y = demo_marker_pair
    res = client.get(f"/pair?x={x}&y={y}")
    assert res.status_code == 200
    # The chart must actually plot x on the x-axis and y on the y-axis.
    titles = _axis_titles(res.text)
    assert len(titles) >= 2
    assert titles[0].startswith(f"{x} (")
    assert titles[1].startswith(f"{y} (")


def test_pair_partial_respects_axis_assignment(
    demo_marker_pair: tuple[str, str],
) -> None:
    """Swapping x and y must swap the axes — proves the route reads both
    params rather than charting a fixed/strongest pair."""
    x, y = demo_marker_pair
    forward = _axis_titles(client.get(f"/pair?x={x}&y={y}").text)
    flipped = _axis_titles(client.get(f"/pair?x={y}&y={x}").text)
    assert forward[0].startswith(f"{x} (") and forward[1].startswith(f"{y} (")
    assert flipped[0].startswith(f"{y} (") and flipped[1].startswith(f"{x} (")


def test_pair_partial_same_marker_picks_alternate(demo_marker: str) -> None:
    res = client.get(f"/pair?x={demo_marker}&y={demo_marker}")
    assert res.status_code == 200
    # The route normalises y away from x; a chart still renders.
    assert "plotly-graph-div" in res.text


# ── Filter routes ────────────────────────────────────────────────────────────

def test_filters_set_age_range_marks_bar_filtered() -> None:
    res = client.get("/filters/set?tab=explorer&age_min=35&age_max=55")
    assert res.status_code == 200
    assert _is_filtered(res.text)
    assert _cohort_active_count(res.text) < state.df_long_full["patient_id"].nunique()


def test_filters_set_full_age_range_treated_as_no_filter() -> None:
    """Selecting the data's full age range should normalise to 'no filter'."""
    age_min = int(state.df_long_full["age"].min())
    age_max = int(state.df_long_full["age"].max())
    res = client.get(f"/filters/set?tab=explorer&age_min={age_min}&age_max={age_max}")
    assert res.status_code == 200
    assert not _is_filtered(res.text)


def test_filters_add_marker_shows_in_bar(demo_marker: str) -> None:
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert res.status_code == 200
    # The added marker appears as an editable pill in the filter bar.
    assert _is_filtered(res.text)
    assert demo_marker in res.text


def test_add_marker_is_fast_path_no_refit(demo_marker: str) -> None:
    """Adding a marker returns only the cohort popover — no chart, and the
    full-range filter leaves the cohort count unchanged. It must not refit."""
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert res.status_code == 200
    # Only the popover fragment: an editable pill, but no chart and no shell.
    assert 'id="cohort"' in res.text
    assert "plotly-graph-div" not in res.text
    assert "tab-strip" not in res.text
    # Full-range add changes nothing: cohort still the whole dataset.
    n_full = state.df_long_full["patient_id"].nunique()
    assert _cohort_active_count(res.text) == n_full
    # The new pill's lower bound is autofocused for immediate editing.
    assert "autofocus" in res.text


def test_add_marker_does_not_call_gmm(monkeypatch, demo_marker: str) -> None:
    """Hard proof the add fast-path skips the GMM refit: make the fit explode
    and confirm the add still succeeds."""
    import web.state as st

    def _boom(*_a, **_k):
        raise AssertionError("unexpected GMM refit on add")

    monkeypatch.setattr(st, "analyse_population", _boom)
    monkeypatch.setattr(st, "analyse_upload", _boom)
    st._filtered_data_cached.cache_clear()  # avoid a cache hit masking a refit
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert res.status_code == 200
    assert demo_marker in res.text


def test_filters_remove_marker_restores_full_cohort(demo_marker: str) -> None:
    """Removing a genuinely-narrowing marker filter must restore the full
    cohort. A no-op remove would leave it narrowed."""
    sub = state.df_long_full[state.df_long_full["test_name"] == demo_marker]["value"]
    lo, hi = float(sub.min()), float(sub.median())
    n_full = state.df_long_full["patient_id"].nunique()
    # Narrow first, and confirm it actually subset the cohort.
    narrowed = client.get(
        f"/filters/set-marker?tab=explorer&marker={demo_marker}&lo={lo}&hi={hi}"
    )
    active = _cohort_active_count(narrowed.text)
    assert active is not None and active < n_full
    # Removing it returns to the full cohort.
    removed = client.get(
        f"/filters/remove?tab=explorer&marker={demo_marker}&m={demo_marker}:{lo}:{hi}"
    )
    assert removed.status_code == 200
    assert not _is_filtered(removed.text)
    assert _cohort_active_count(removed.text) == n_full


def test_filters_reset_restores_full_cohort(demo_marker: str) -> None:
    """Reset must drop both age and marker filters back to the full cohort."""
    sub = state.df_long_full[state.df_long_full["test_name"] == demo_marker]["value"]
    lo, hi = float(sub.min()), float(sub.median())
    n_full = state.df_long_full["patient_id"].nunique()
    narrowed = client.get(
        f"/filters/set-marker?tab=explorer&age_min=35&age_max=55"
        f"&marker={demo_marker}&lo={lo}&hi={hi}"
    )
    assert _is_filtered(narrowed.text)
    active = _cohort_active_count(narrowed.text)
    assert active is not None and active < n_full
    reset = client.get("/filters/reset?tab=explorer")
    assert reset.status_code == 200
    assert not _is_filtered(reset.text)
    assert _cohort_active_count(reset.text) == n_full


def _cohort_active_count(html: str) -> int | None:
    """Pull `n_active` out of the filter bar's count ('N of M tests')."""
    m = re.search(r"(\d+)\s+of\s+(\d+)\s+tests", html)
    return int(m.group(1)) if m else None


def _is_filtered(html: str) -> bool:
    """The filter bar carries `is-filtered` exactly when a cohort filter is active."""
    return "is-filtered" in html


def _axis_titles(html: str) -> list[str]:
    """Plotly axis title texts in document order (x-axis first, then y-axis).

    Plotly serialises axis titles as `"title":{"text":"…"}`. Trace names and
    annotations use other keys, so on a charts with no colour-bar (the marker
    and pair charts) the only matches are the two axis titles.
    """
    return re.findall(r'"title":\{"text":"([^"]*)"\}', html)


def _marker_axis_unit(html: str, marker: str) -> str | None:
    """Extract the unit from a marker chart's x-axis title (`Marker (unit)`).

    Plotly escapes the slash in unit strings as `\\u002f`, so undo that.
    """
    m = re.search(re.escape(marker) + r" \(([^)]*)\)", html)
    return m.group(1).replace("\\u002f", "/") if m else None


def test_filters_add_marker_renders_editable_range_inputs(demo_marker: str) -> None:
    """Adding a marker filter must expose lo/hi inputs, not a read-only chip."""
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert res.status_code == 200
    assert 'name="lo"' in res.text and 'name="hi"' in res.text
    assert "/filters/set-marker" in res.text


def test_filters_set_marker_narrows_cohort(demo_marker: str) -> None:
    """A tightened marker range must subset the cohort (fewer active tests)."""
    sub = state.df_long_full[state.df_long_full["test_name"] == demo_marker]["value"]
    lo, hi = float(sub.min()), float(sub.median())
    n_full = state.df_long_full["patient_id"].nunique()
    res = client.get(
        f"/filters/set-marker?tab=explorer&marker={demo_marker}&lo={lo}&hi={hi}"
    )
    assert res.status_code == 200
    assert _is_filtered(res.text)
    active = _cohort_active_count(res.text)
    assert active is not None and active < n_full


def test_filters_set_marker_full_range_keeps_everyone(demo_marker: str) -> None:
    """The data's full range matches every test → cohort unchanged."""
    sub = state.df_long_full[state.df_long_full["test_name"] == demo_marker]["value"]
    lo, hi = float(sub.min()), float(sub.max())
    n_full = state.df_long_full["patient_id"].nunique()
    res = client.get(
        f"/filters/set-marker?tab=explorer&marker={demo_marker}&lo={lo}&hi={hi}"
    )
    assert res.status_code == 200
    assert _cohort_active_count(res.text) == n_full


def test_filters_set_marker_swaps_inverted_bounds(demo_marker: str) -> None:
    """lo > hi is tolerated by swapping, not by emptying the cohort."""
    sub = state.df_long_full[state.df_long_full["test_name"] == demo_marker]["value"]
    lo, hi = float(sub.min()), float(sub.median())
    res = client.get(
        f"/filters/set-marker?tab=explorer&marker={demo_marker}&lo={hi}&hi={lo}"
    )
    assert res.status_code == 200
    active = _cohort_active_count(res.text)
    assert active is not None and active > 0


def test_filters_set_marker_preserves_other_markers(
    demo_marker_pair: tuple[str, str],
) -> None:
    """Editing one marker must not drop the other active marker's filter.

    `set-marker` swaps page-body only (chips aren't re-rendered), so retention
    is verified through the cohort count: `second` stays tight, so even with
    `first` widened to a no-op the cohort must remain a strict subset.
    """
    first, second = demo_marker_pair
    f = state.df_long_full[state.df_long_full["test_name"] == first]["value"]
    s = state.df_long_full[state.df_long_full["test_name"] == second]["value"]
    n_full = state.df_long_full["patient_id"].nunique()
    res = client.get(
        f"/filters/set-marker?tab=explorer&marker={first}"
        f"&lo={float(f.min())}&hi={float(f.max())}"          # widen `first` (no-op)
        f"&m={second}:{float(s.min())}:{float(s.median())}"  # `second` stays tight
    )
    assert res.status_code == 200
    active = _cohort_active_count(res.text)
    assert active is not None and active < n_full


# ── URL bookmarkability ──────────────────────────────────────────────────────

def test_full_parameterised_url_restores_cohort_tab_and_drawer(
    demo_marker: str,
) -> None:
    """A fully parameterised GET should reproduce the same cohort and tab,
    and the drawer should be ready to deep-link via methodology=<tab>."""
    res = client.get(
        f"/?tab=population&age_min=35&age_max=55"
        f"&m={demo_marker}:0:9999&methodology=population"
    )
    assert res.status_code == 200
    assert 'data-mdr-section="population"' in res.text
    assert _is_filtered(res.text)


def test_filter_mutation_pushes_canonical_url(demo_marker: str) -> None:
    """Filter mutations must push the bookmarkable /?… page URL — not the bare
    /filters/* endpoint, which renders shell-less (no CSS) on reload."""
    res = client.get(f"/filters/set?tab=pairs&age_min=35&age_max=55&m={demo_marker}:0:9999")
    pushed = res.headers.get("HX-Push-Url")
    assert pushed is not None and pushed.startswith("/?tab=pairs")
    assert "age_min=35" in pushed and f"m={demo_marker}" in pushed
    assert "/filters/" not in pushed
    # And that pushed URL must actually serve the full styled shell.
    full = client.get(pushed)
    assert "/static/styles.css" in full.text and "<html" in full.text


def test_tab_switch_pushes_canonical_url() -> None:
    """Switching tabs must push /?tab=… (full page), not the /tab/* fragment."""
    res = client.get("/tab/population")
    pushed = res.headers.get("HX-Push-Url")
    assert pushed == "/?tab=population"
    assert "/static/styles.css" in client.get(pushed).text


# ── Upload ───────────────────────────────────────────────────────────────────

def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    """Tiny CSV builder for upload fixtures. Each row dict maps column → value."""
    if not rows:
        return b""
    cols = list(rows[0].keys())
    header = ",".join(cols)
    lines = [header]
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _valid_upload_csv(n_rows: int = 5) -> bytes:
    rows = []
    for i in range(n_rows):
        rows.append({
            "Blood Test Info Blood Test ID": f"P{i:03d}",
            "Current Age": str(30 + i),
            "Blood Test Info Albumin Levels": str(40 + i * 0.5),
            "Blood Test Info ALT Levels":     str(20 + i * 1.2),
            "Blood Test Info HBA1C Levels":   str(5.0 + i * 0.1),
            "Blood Test Info TSH Levels":     str(1.8 + i * 0.05),
        })
    return _csv_bytes(rows)


def test_upload_happy_path_replaces_demo_state() -> None:
    csv = _valid_upload_csv(n_rows=5)
    res = client.post(
        "/upload",
        files={"file": ("uploaded.csv", io.BytesIO(csv), "text/csv")},
    )
    assert res.status_code == 200
    assert res.headers.get("HX-Redirect") == "/"
    assert state.is_demo is False
    assert state.upload_filename == "uploaded.csv"
    assert state.df_long_full["patient_id"].nunique() == 5


def test_upload_handles_thousands_separators() -> None:
    """Values like '1,366.00' (thousands separator) must parse, not 500."""
    rows = []
    for i in range(5):
        rows.append({
            "Blood Test Info Blood Test ID":     f"P{i:03d}",
            "Current Age":                       str(40 + i),
            "Blood Test Info Ferritin Levels":   '"1,366.00"',
            "Blood Test Info Albumin Levels":    str(42 + i),
        })
    res = client.post(
        "/upload",
        files={"file": ("commas.csv", io.BytesIO(_csv_bytes(rows)), "text/csv")},
    )
    assert res.status_code == 200
    assert res.headers.get("HX-Redirect") == "/"   # parsed, not an inline error
    assert state.df_long_full.query("test_name == 'Ferritin'")["value"].max() == 1366.0


def test_upload_surfaces_detected_units() -> None:
    """After an upload, the rail shows the inferred source unit per marker.
    The fixture's HbA1C values (~5%) are detected as % and converted."""
    csv = _valid_upload_csv(n_rows=5)
    client.post("/upload", files={"file": ("u.csv", io.BytesIO(csv), "text/csv")})
    report = {u["marker"]: u for u in state.upload_unit_report}
    assert "HbA1C" in report and report["HbA1C"]["detected"] == "%"
    assert report["HbA1C"]["converted"] is True
    # And it's rendered for the user in the home page rail.
    home = client.get("/").text
    assert "Detected units" in home and "HbA1C" in home


def _testosterone_csv(values: list[float]) -> bytes:
    rows = []
    for i, v in enumerate(values):
        rows.append({
            "Blood Test Info Blood Test ID":        f"P{i:03d}",
            "Current Age":                          str(40 + i),
            "Blood Test Info Testosterone Levels":  str(v),
            "Blood Test Info Albumin Levels":       str(42 + i),
        })
    return _csv_bytes(rows)


def test_upload_unit_override_reconverts_without_reupload() -> None:
    """Forcing a marker's source unit re-converts from the stored CSV and
    re-runs the analysis — no re-upload."""
    # Testosterone all < 100 → auto-detected nmol/L (kept as-is)
    csv = _testosterone_csv([12, 15, 20, 9, 18, 11])
    client.post("/upload", files={"file": ("u.csv", io.BytesIO(csv), "text/csv")})
    before = state.df_long_full.query("test_name == 'Testosterone'")["value"].max()
    assert before > 5  # kept in nmol/L

    res = client.post("/upload/units", data={"marker": "Testosterone", "unit": "ng/dL"})
    assert res.status_code == 200
    assert res.headers.get("HX-Redirect") == "/"
    after = state.df_long_full.query("test_name == 'Testosterone'")["value"].max()
    assert after < before  # re-converted ng/dL → nmol/L (÷28.84)
    rep = {u["marker"]: u for u in state.upload_unit_report}
    assert rep["Testosterone"]["detected"] == "ng/dL" and rep["Testosterone"]["forced"]


def test_upload_unit_overrides_accumulate() -> None:
    """Sequential overrides on different markers both persist and both re-convert."""
    rows = []
    for i in range(6):
        rows.append({
            "Blood Test Info Blood Test ID":               f"P{i:03d}",
            "Current Age":                                 str(40 + i),
            "Blood Test Info Testosterone Levels":         str(12 + i),   # nmol/L kept
            "Blood Test Info Total Cholesterol Levels":    str(4 + i * 0.2),  # mmol/L kept
        })
    client.post("/upload", files={"file": ("u.csv", io.BytesIO(_csv_bytes(rows)), "text/csv")})
    t0 = state.df_long_full.query("test_name == 'Testosterone'")["value"].max()
    c0 = state.df_long_full.query("test_name == 'Total Cholesterol'")["value"].max()

    client.post("/upload/units", data={"marker": "Testosterone", "unit": "ng/dL"})
    client.post("/upload/units", data={"marker": "Total Cholesterol", "unit": "mg/dL"})

    assert state.upload_unit_overrides == {
        "Testosterone": "ng/dL", "Total Cholesterol": "mg/dL",
    }
    # both columns re-converted (divided down), proving both overrides applied
    assert state.df_long_full.query("test_name == 'Testosterone'")["value"].max() < t0
    assert state.df_long_full.query("test_name == 'Total Cholesterol'")["value"].max() < c0


def test_upload_unit_override_requires_prior_upload() -> None:
    """Overriding in demo mode (no stored CSV) shows a visible error, not a 500."""
    _load_demo()
    res = client.post("/upload/units", data={"marker": "Testosterone", "unit": "ng/dL"})
    assert res.status_code == 200
    assert "cohort-error" in res.text
    assert state.is_demo is True


# Upload errors return 200 (not 4xx) on purpose: HTMX does not swap the body
# of an error response, so a 4xx would show the user nothing. The error is
# rendered inline as a .cohort-error fragment instead.

def test_upload_too_few_rows_shows_inline_error() -> None:
    csv = _valid_upload_csv(n_rows=2)
    res = client.post(
        "/upload",
        files={"file": ("small.csv", io.BytesIO(csv), "text/csv")},
    )
    assert res.status_code == 200
    assert "cohort-error" in res.text and "need at least 4" in res.text
    assert state.is_demo is True  # demo state intact


def test_upload_unrecognised_columns_shows_inline_error() -> None:
    """A CSV with no recognised column maps still has the ID column, parses
    to an empty long-frame, and is rejected with a clear visible message."""
    csv = textwrap.dedent("""\
        Blood Test Info Blood Test ID,Wholly Unknown Column,Another Random One
        P001,1,2
        P002,3,4
        P003,5,6
        P004,7,8
        P005,9,10
    """).encode("utf-8")
    res = client.post(
        "/upload",
        files={"file": ("unknown.csv", io.BytesIO(csv), "text/csv")},
    )
    assert res.status_code == 200
    assert "cohort-error" in res.text and "No recognised columns" in res.text
    assert state.is_demo is True


def test_upload_unparseable_input_shows_inline_error() -> None:
    """Bytes that pandas can't read as CSV produce the parse-error branch."""
    bad = b"not,a,valid\ncsv,without,id\n"
    res = client.post(
        "/upload",
        files={"file": ("garbage.csv", io.BytesIO(bad), "text/csv")},
    )
    assert res.status_code == 200
    assert "cohort-error" in res.text and "Could not read CSV" in res.text
    assert state.is_demo is True


def test_upload_analysis_failure_shows_inline_error(monkeypatch) -> None:
    """If the analysis raises on an otherwise-valid CSV, the user gets a
    visible error — not a silent 500 that HTMX drops."""
    import web.main as wm

    def _boom(*_a, **_k):
        raise ValueError("degenerate marker")

    monkeypatch.setattr(wm, "analyse_upload", _boom)
    res = client.post(
        "/upload",
        files={"file": ("ok.csv", io.BytesIO(_valid_upload_csv(n_rows=5)), "text/csv")},
    )
    assert res.status_code == 200
    assert "cohort-error" in res.text and "Could not analyse" in res.text
    assert state.is_demo is True  # failed upload must not wipe the demo


def test_upload_reset_restores_demo() -> None:
    # First, upload something so we're out of demo mode.
    csv = _valid_upload_csv(n_rows=5)
    client.post("/upload", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
    assert state.is_demo is False
    # Now reset.
    res = client.post("/upload/reset")
    assert res.status_code == 200
    assert res.headers.get("HX-Redirect") == "/"
    assert state.is_demo is True
    assert state.upload_filename is None


# ── Display units ────────────────────────────────────────────────────────────

def test_units_set_switches_display_unit() -> None:
    """HbA1C supports multiple units (%, mmol/mol). Setting the alt should
    persist in state.unit_prefs."""
    # Pick a multi-unit marker present in the demo.
    from web.state import MULTI_UNIT_MARKERS

    from unit_conversions import available_units
    present = set(state.df_long_full["test_name"].unique())
    candidate = next((m for m in MULTI_UNIT_MARKERS if m in present), None)
    if candidate is None:
        pytest.skip("No multi-unit marker present in the demo cohort.")
    canonical = available_units(candidate)[0]
    alt = next(u for u in available_units(candidate) if u != canonical)

    res = client.get(f"/units/set?tab=explorer&u={candidate}:{alt}")
    assert res.status_code == 200
    assert state.unit_prefs.get(candidate) == alt


def test_units_set_changes_rendered_axis_label() -> None:
    """A unit switch must change the *rendered* chart axis label, not merely
    `state.unit_prefs`. Guards against a unit pref that never reaches a chart."""
    from web.state import MULTI_UNIT_MARKERS

    from unit_conversions import available_units
    present = set(state.df_long_full["test_name"].unique())
    candidate = next((m for m in MULTI_UNIT_MARKERS if m in present), None)
    if candidate is None:
        pytest.skip("No multi-unit marker present in the demo cohort.")
    canonical = available_units(candidate)[0]
    alt = next(u for u in available_units(candidate) if u != canonical)

    before = client.get(f"/marker?name={candidate}").text
    assert _marker_axis_unit(before, candidate) == canonical

    client.get(f"/units/set?tab=explorer&u={candidate}:{alt}")
    after = client.get(f"/marker?name={candidate}").text
    assert _marker_axis_unit(after, candidate) == alt


def test_units_set_ignores_invalid_marker_or_unit() -> None:
    res = client.get("/units/set?tab=explorer&u=NotAMarker:NotAUnit")
    assert res.status_code == 200
    assert state.unit_prefs == {}


# ── Drawer markup (preserved from #8) ────────────────────────────────────────

def test_drawer_contains_all_four_sections() -> None:
    res = client.get("/")
    html = res.text
    for anchor in ("methodology-gmm", "methodology-clusters",
                   "methodology-outliers", "methodology-correlations"):
        assert f'id="{anchor}"' in html


def test_methodology_param_is_passed_through_harmlessly() -> None:
    res = client.get("/?tab=explorer&methodology=explorer")
    assert res.status_code == 200
    assert 'class="methodology-trigger"' in res.text


def test_old_methodology_disclosure_is_gone() -> None:
    res = client.get("/")
    html = res.text
    assert "disclosure methodology" not in html
    assert "how this view is computed" not in html.lower()
