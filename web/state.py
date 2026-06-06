"""Application state, demo-cache loading, and the filtered-analysis LRU cache.

Single-process, single-user state model: one `state` singleton holds the
current cohort (demo or uploaded). Filtered analyses are computed lazily by
`_filtered_data_cached`, bounded by a 32-entry LRU cache. `_load_demo` and
the upload route both invalidate the cache.
"""
from __future__ import annotations

import functools
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from analysis import (
    analyse_population,
    analyse_upload,
    build_labelled_df,
    filter_long,
)
from thresholds import THRESHOLDS
from unit_conversions import available_units
from web.filters import FilterSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Markers with more than one supported display unit. Used by the units UI
# and by display-unit fallback logic.
MULTI_UNIT_MARKERS = sorted(m for m in THRESHOLDS if len(available_units(m)) > 1)


class AppState:
    df_long_full: Any = None
    gmm_results_full: dict = {}
    pop_results_full: dict = {}
    df_labelled_full: Any = None
    is_demo: bool = True
    upload_filename: str | None = None
    unit_prefs: dict[str, str] = {}  # marker → display unit
    last_upload_error: str | None = None


state = AppState()


def _load_demo() -> None:
    cache_path = PROJECT_ROOT / "demo_cache.pkl"
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    state.df_long_full     = cached["df_long"]
    state.gmm_results_full = cached["gmm_results"]
    state.pop_results_full = cached["pop_results"]
    state.df_labelled_full = cached["df_labelled"]
    state.is_demo          = True
    _filtered_data_cached.cache_clear()


def _full_data() -> dict:
    return {
        "df_long":     state.df_long_full,
        "gmm_results": state.gmm_results_full,
        "pop_results": state.pop_results_full,
        "df_labelled": state.df_labelled_full,
    }


@functools.lru_cache(maxsize=32)
def _filtered_data_cached(spec: FilterSpec) -> dict:
    """Compute and cache the filtered analysis. Bounded LRU keeps memory in
    check on the 512 MB Render free tier — each entry is ~400 KB. Callers
    must NOT mutate the returned dict; cache hits return the same object."""
    age_range: tuple | None = None
    if spec.age_min is not None or spec.age_max is not None:
        full_range = _age_full_range()
        lo = spec.age_min if spec.age_min is not None else (full_range[0] if full_range else 0)
        hi = spec.age_max if spec.age_max is not None else (full_range[1] if full_range else 200)
        age_range = (lo, hi)

    marker_ranges = {mf.marker: (mf.lo, mf.hi) for mf in spec.markers}

    df_filtered = filter_long(state.df_long_full, age_range=age_range, marker_ranges=marker_ranges)
    n = df_filtered["patient_id"].nunique() if not df_filtered.empty else 0

    if n < 4:
        msg = (
            f"Only {n} blood test{'s' if n != 1 else ''} match the current filters "
            "— need at least 4 to run analysis."
        )
        return {
            "error":       msg,
            "df_long":     df_filtered,
            "gmm_results": {},
            "pop_results": {"error": msg},
            "df_labelled": df_filtered,
        }

    gmm = analyse_upload(df_filtered)
    pop = analyse_population(df_filtered)
    return {
        "df_long":     df_filtered,
        "gmm_results": gmm,
        "pop_results": pop,
        "df_labelled": build_labelled_df(df_filtered, gmm),
    }


def _filtered_data(spec: FilterSpec) -> dict:
    """Return a {df_long, gmm_results, pop_results, df_labelled} dict for the
    cohort defined by `spec`."""
    if not spec.is_active():
        return _full_data()
    return _filtered_data_cached(spec)


def _age_full_range() -> tuple[int, int] | None:
    df = state.df_long_full
    if "age" not in df.columns:
        return None
    ages = pd.to_numeric(df.drop_duplicates("patient_id")["age"], errors="coerce").dropna()
    if ages.empty:
        return None
    return (int(ages.min()), int(ages.max()))


def _marker_value_range(marker: str) -> tuple[float, float] | None:
    df = state.df_long_full
    sub = df[df["test_name"] == marker]
    if sub.empty:
        return None
    return float(sub["value"].min()), float(sub["value"].max())


def _normalise_age(age_min: int | None, age_max: int | None) -> tuple[int | None, int | None]:
    """Treat (full_min, full_max) as 'no age filter' — the form's default state."""
    full = _age_full_range()
    if full is None:
        return None, None
    if age_min is not None and age_max is not None \
            and age_min <= full[0] and age_max >= full[1]:
        return None, None
    return age_min, age_max


def _display_unit(marker: str) -> str:
    """The unit to render `marker` in. Falls back to canonical if the user's
    preference isn't one of this marker's supported units."""
    canonical = THRESHOLDS[marker]["unit"]
    pref = state.unit_prefs.get(marker)
    if pref and pref in available_units(marker):
        return pref
    return canonical


def _filter_ui_context(spec: FilterSpec) -> dict:
    """Filter rail state: existing chips + the dropdown of addable markers."""
    age_full = _age_full_range()
    active_marker_names = {mf.marker for mf in spec.markers}
    addable_markers = [
        m for m in sorted(state.df_long_full["test_name"].unique())
        if m not in active_marker_names
    ]

    filter_chips = []
    for mf in spec.markers:
        full = _marker_value_range(mf.marker)
        unit = _display_unit(mf.marker) if mf.marker in THRESHOLDS else ""
        without = spec.without_marker(mf.marker)
        filter_chips.append({
            "marker":  mf.marker,
            "lo":      mf.lo,
            "hi":      mf.hi,
            "full_lo": full[0] if full else mf.lo,
            "full_hi": full[1] if full else mf.hi,
            "unit":    unit,
            "remove_qs": without.to_query_string(),
        })

    return {
        "age_full":        age_full,
        "addable_markers": addable_markers,
        "filter_chips":    filter_chips,
    }


def _units_ui_context() -> dict:
    """Per-marker display-unit options. Only markers with > 1 supported unit
    AND a value in the current cohort are shown."""
    present = (
        set(state.df_long_full["test_name"].unique())
        if state.df_long_full is not None
        else set()
    )
    rows: list[dict] = []
    for marker in MULTI_UNIT_MARKERS:
        if marker not in present:
            continue
        rows.append({
            "marker":  marker,
            "options": available_units(marker),
            "current": _display_unit(marker),
        })
    return {
        "unit_options":      rows,
        "any_unit_override": bool(state.unit_prefs),
        "n_unit_overrides":  len(state.unit_prefs),
    }


# Eagerly load the demo cohort on module import. Routes assume `state` is
# populated; without this, the first request after fresh boot would 500.
_load_demo()
