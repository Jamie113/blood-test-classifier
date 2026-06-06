"""FastAPI front-end for the blood-test classifier.

Thin route layer: parses query params, calls into web.state /
web.contexts / web.charts, and returns Jinja2 templates. No analysis,
chart-building, or context-shaping logic lives here.
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# web.* imports first so web/__init__.py is unambiguously responsible for
# putting PROJECT_ROOT on sys.path before any project-root module is touched.
from web.contexts import (
    _build_tab_ctx,
    _common,
    _marker_context,
    _pair_context,
    _population_context,
)
from web.filters import FilterSpec, _resolve_tab
from web.state import (
    MULTI_UNIT_MARKERS,
    _filtered_data,
    _filtered_data_cached,
    _load_demo,
    _marker_value_range,
    _normalise_age,
    state,
)

from analysis import analyse_population, analyse_upload, build_labelled_df
from parsing import parse_csv
from thresholds import THRESHOLDS
from unit_conversions import available_units

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Blood Test Classifier")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


def get_filter_spec(
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> FilterSpec:
    """FastAPI dependency: parse the cohort-filter query params into a spec.

    Page routes share this. Filter-mutation routes build the spec themselves
    because they normalise age and rewrite the marker list first."""
    return FilterSpec.from_request(age_min, age_max, m)


def _page_response(
    request: Request, spec: FilterSpec, tab: str, template: str,
) -> HTMLResponse:
    """Render a full tab view: filtered data → cross-tab + tab-specific context."""
    data = _filtered_data(spec)
    tab = _resolve_tab(tab)
    ctx = {"active": tab, **_common(spec, data), **_build_tab_ctx(data, tab)}
    return templates.TemplateResponse(request, template, ctx)


def _filter_response(
    request: Request, spec: FilterSpec, tab: str, full: bool,
) -> HTMLResponse:
    """Response for a filter mutation. `full=True` also OOB-swaps the rail
    filter section; otherwise just the page body."""
    template = "partials/full_render.html" if full else "partials/page_body.html"
    return _page_response(request, spec, tab, template)


# ── Page + tab routes ────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    spec: FilterSpec = Depends(get_filter_spec),
    tab: str = "explorer",
) -> HTMLResponse:
    return _page_response(request, spec, tab, "index.html")


@app.get("/tab/{name}", response_class=HTMLResponse)
def tab_partial(
    request: Request,
    name: str,
    spec: FilterSpec = Depends(get_filter_spec),
) -> HTMLResponse:
    return _page_response(request, spec, name, "partials/page_body.html")


@app.get("/marker", response_class=HTMLResponse)
def marker_partial(
    request: Request,
    name: str,
    spec: FilterSpec = Depends(get_filter_spec),
) -> HTMLResponse:
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
    spec: FilterSpec = Depends(get_filter_spec),
) -> HTMLResponse:
    if colour_by not in {"type", "age"}:
        colour_by = "type"
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
    spec: FilterSpec = Depends(get_filter_spec),
) -> HTMLResponse:
    data = _filtered_data(spec)
    pair = _pair_context(data, x, y) if not data.get("error") else None
    ctx: dict = {"active": "pairs", **_common(spec, data)}
    if pair:
        ctx["pair"] = pair
    return templates.TemplateResponse(request, "partials/page_body.html", ctx)


# ── Filter routes ────────────────────────────────────────────────────────────

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


@app.get("/filters/set-marker", response_class=HTMLResponse)
def set_marker_filter_partial(
    request: Request,
    marker: str,
    lo: float,
    hi: float,
    tab: str = "explorer",
    age_min: int | None = None,
    age_max: int | None = None,
    m: list[str] = Query(default_factory=list),
) -> HTMLResponse:
    """Update the value range of an already-active marker filter. The other
    active markers arrive as `m`; this one's bounds arrive as `lo`/`hi`."""
    age_min, age_max = _normalise_age(age_min, age_max)
    if hi < lo:
        lo, hi = hi, lo
    new_m = [s for s in m if not s.startswith(f"{marker}:")]
    if marker in state.df_long_full["test_name"].unique():
        new_m.append(f"{marker}:{lo}:{hi}")
    spec = FilterSpec.from_request(age_min, age_max, new_m)
    return _filter_response(request, spec, tab, full=False)


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
        df_long, _recognised, _unrecognised = parse_csv(io.BytesIO(contents))
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
