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
    res = client.get(f"/?tab={tab}")
    assert res.status_code == 200
    assert f'data-mdr-section="{tab}"' in res.text
    # The intro paragraph is the most stable per-tab content marker.
    intros = {
        "explorer":    "For each blood marker",
        "population":  "treats each blood test as a whole",
        "investigate": "don't fit cleanly into any cluster",
        "pairs":       "compares any two markers",
    }
    assert intros[tab] in res.text


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


def test_population_scatter_invalid_colour_by_defaults_to_type() -> None:
    res = client.get("/population/scatter?colour_by=garbage")
    assert res.status_code == 200
    assert "plotly-graph-div" in res.text


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
    assert "plotly-graph-div" in res.text


def test_pair_partial_same_marker_picks_alternate(demo_marker: str) -> None:
    res = client.get(f"/pair?x={demo_marker}&y={demo_marker}")
    assert res.status_code == 200
    # The route normalises y away from x; a chart still renders.
    assert "plotly-graph-div" in res.text


# ── Filter routes ────────────────────────────────────────────────────────────

def test_filters_set_age_range_shows_cohort_banner() -> None:
    res = client.get("/filters/set?tab=explorer&age_min=35&age_max=55")
    assert res.status_code == 200
    assert "cohort-banner" in res.text
    assert "Filtered cohort" in res.text


def test_filters_set_full_age_range_treated_as_no_filter() -> None:
    """Selecting the data's full age range should normalise to 'no filter'."""
    age_min = int(state.df_long_full["age"].min())
    age_max = int(state.df_long_full["age"].max())
    res = client.get(f"/filters/set?tab=explorer&age_min={age_min}&age_max={age_max}")
    assert res.status_code == 200
    assert "cohort-banner" not in res.text


def test_filters_add_marker_returns_full_render(demo_marker: str) -> None:
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert res.status_code == 200
    # full_render.html injects an OOB rail-state swap on top of page-body.
    assert 'id="rail-state"' in res.text
    assert "cohort-banner" in res.text
    assert demo_marker in res.text


def test_filters_remove_marker(demo_marker: str) -> None:
    # Establish a marker filter first (broad range covering all values).
    res = client.get(f"/filters/add?tab=explorer&marker={demo_marker}")
    assert "cohort-banner" in res.text
    # Now remove it.
    res = client.get(f"/filters/remove?tab=explorer&marker={demo_marker}&m={demo_marker}:0:9999")
    assert res.status_code == 200
    assert "cohort-banner" not in res.text


def test_filters_reset_clears_everything(demo_marker: str) -> None:
    res = client.get(
        f"/filters/reset?tab=explorer&age_min=35&age_max=55&m={demo_marker}:0:9999"
    )
    assert res.status_code == 200
    assert "cohort-banner" not in res.text


def _cohort_active_count(html: str) -> int | None:
    """Pull `n_active` out of the cohort banner ('analysing N of M blood tests')."""
    m = re.search(r"analysing\s+(\d+)\s+of\s+(\d+)\s+blood tests", html)
    return int(m.group(1)) if m else None


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
    assert "cohort-banner" in res.text
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
    assert "cohort-banner" in res.text
    assert "Filtered cohort" in res.text


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


def test_upload_too_few_rows_returns_400() -> None:
    csv = _valid_upload_csv(n_rows=2)
    res = client.post(
        "/upload",
        files={"file": ("small.csv", io.BytesIO(csv), "text/csv")},
    )
    assert res.status_code == 400
    assert "need at least 4" in res.text
    # Demo state should still be intact.
    assert state.is_demo is True


def test_upload_unrecognised_columns_returns_400() -> None:
    """A CSV with no recognised column maps still has the ID column, parses
    to an empty long-frame, and is rejected with a clear message."""
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
    assert res.status_code == 400
    assert "No recognised columns" in res.text
    assert state.is_demo is True


def test_upload_unparseable_input_returns_400() -> None:
    """Bytes that pandas can't read as CSV produce the parse-error branch."""
    # parse_csv expects the ID column; omitting it makes the row-filter raise.
    bad = b"not,a,valid\ncsv,without,id\n"
    res = client.post(
        "/upload",
        files={"file": ("garbage.csv", io.BytesIO(bad), "text/csv")},
    )
    assert res.status_code == 400
    assert "Could not read CSV" in res.text
    assert state.is_demo is True


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
