# unit_conversions.py
# Two responsibilities:
#   1. to_canonical()   — convert incoming values to canonical units (used at upload)
#   2. from_canonical() — convert canonical values to a chosen display unit (used in UI)

import numpy as np

# ── Conversion factors ────────────────────────────────────────────────────────

_TESTOSTERONE_NMOL_TO_NGDL  =  28.84
_CHOLESTEROL_MMOL_TO_MGDL   =  38.67
_OESTRADIOL_PG_TO_PMOL      =   3.671
_PROLACTIN_MIUL_TO_NGML     =   1 / 21.2
_FREE_T_NMOL_TO_PMOL        =  1000.0


def _hba1c_pct_to_mmol(pct: float) -> float:
    return (pct - 2.15) * 10.929

def _hba1c_mmol_to_pct(mmol: float) -> float:
    return (mmol / 10.929) + 2.15


# ── Canonical → display transforms ───────────────────────────────────────────
# Each entry: unit_name → {
#   "fn":    canonical value → display value (scalar or array)
#   "scale": linear scale factor (for transforming std devs, which don't shift)
#   "shift": additive shift (for transforming means and boundaries)
# }
# For a linear transform Y = scale * X + shift:
#   mean_display  = scale * mean_canonical  + shift
#   std_display   = |scale| * std_canonical (shift has no effect on spread)
#   boundary_disp = scale * boundary        + shift

_DISPLAY_TRANSFORMS: dict[str, dict] = {
    "Testosterone": {
        "nmol/L": {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "ng/dL":  {"fn": lambda x: x * _TESTOSTERONE_NMOL_TO_NGDL, "scale": _TESTOSTERONE_NMOL_TO_NGDL, "shift": 0},
    },
    "Free Testosterone": {
        "nmol/L": {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "pmol/L": {"fn": lambda x: x * _FREE_T_NMOL_TO_PMOL, "scale": _FREE_T_NMOL_TO_PMOL, "shift": 0},
    },
    "Total Cholesterol": {
        "mmol/L": {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "mg/dL":  {"fn": lambda x: x * _CHOLESTEROL_MMOL_TO_MGDL, "scale": _CHOLESTEROL_MMOL_TO_MGDL, "shift": 0},
    },
    "LDL Cholesterol": {
        "mmol/L": {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "mg/dL":  {"fn": lambda x: x * _CHOLESTEROL_MMOL_TO_MGDL, "scale": _CHOLESTEROL_MMOL_TO_MGDL, "shift": 0},
    },
    "HDL Cholesterol": {
        "mmol/L": {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "mg/dL":  {"fn": lambda x: x * _CHOLESTEROL_MMOL_TO_MGDL, "scale": _CHOLESTEROL_MMOL_TO_MGDL, "shift": 0},
    },
    "HbA1C": {
        "mmol/mol": {"fn": lambda x: x,                     "scale": 1,                    "shift": 0},
        "%":        {"fn": _hba1c_mmol_to_pct,              "scale": 1 / 10.929,           "shift": 2.15},
    },
    "Oestradiol": {
        "pg/mL":  {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "pmol/L": {"fn": lambda x: x * _OESTRADIOL_PG_TO_PMOL, "scale": _OESTRADIOL_PG_TO_PMOL, "shift": 0},
    },
    "Prolactin": {
        "mIU/L":  {"fn": lambda x: x,                       "scale": 1,                    "shift": 0},
        "ng/mL":  {"fn": lambda x: x * _PROLACTIN_MIUL_TO_NGML, "scale": _PROLACTIN_MIUL_TO_NGML, "shift": 0},
    },
}

# ── Incoming conversion (upload → canonical) ─────────────────────────────────
# Detection metadata for the markers that arrive in more than one unit system.
# `alt` is the non-canonical source unit; `alt_when` is the magnitude rule that
# signals it (`gt`/`lt` a threshold); `to_canon` converts an alt-unit value to
# canonical. Single source of truth for both the per-value `to_canonical` and
# the per-column `to_canonical_column`.
_INCOMING: dict[str, dict] = {
    "Testosterone":      {"canonical": "nmol/L",   "alt": "ng/dL",  "alt_when": ("gt", 100),
                          "to_canon": lambda v: v / _TESTOSTERONE_NMOL_TO_NGDL},
    "Free Testosterone": {"canonical": "nmol/L",   "alt": "pmol/L", "alt_when": ("gt", 5),
                          "to_canon": lambda v: v / _FREE_T_NMOL_TO_PMOL},
    "Total Cholesterol": {"canonical": "mmol/L",   "alt": "mg/dL",  "alt_when": ("gt", 15),
                          "to_canon": lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL},
    "LDL Cholesterol":   {"canonical": "mmol/L",   "alt": "mg/dL",  "alt_when": ("gt", 15),
                          "to_canon": lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL},
    "HDL Cholesterol":   {"canonical": "mmol/L",   "alt": "mg/dL",  "alt_when": ("gt", 15),
                          "to_canon": lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL},
    "HbA1C":             {"canonical": "mmol/mol", "alt": "%",      "alt_when": ("lt", 20),
                          "to_canon": _hba1c_pct_to_mmol},
    "Oestradiol":        {"canonical": "pg/mL",    "alt": "pmol/L", "alt_when": ("gt", 200),
                          "to_canon": lambda v: v / _OESTRADIOL_PG_TO_PMOL},
    "Prolactin":         {"canonical": "mIU/L",    "alt": "ng/mL",  "alt_when": ("lt", 50),
                          "to_canon": lambda v: v / _PROLACTIN_MIUL_TO_NGML},
}


def has_unit_detection(test_name: str) -> bool:
    """True if this marker can arrive in more than one unit system."""
    return test_name in _INCOMING


def _matches_alt(rule: tuple, arr: np.ndarray) -> np.ndarray:
    op, threshold = rule
    return arr > threshold if op == "gt" else arr < threshold


def _to_canonical(test_name: str, value: float) -> float:
    """Per-VALUE conversion by magnitude heuristic. Private: this is the
    column-splitting behaviour the per-column path replaced — kept only as the
    primitive the conversion-factor tests exercise. Production ingest must use
    `to_canonical_column`, never this."""
    info = _INCOMING.get(test_name)
    if info is None:
        return value
    op, threshold = info["alt_when"]
    is_alt = value > threshold if op == "gt" else value < threshold
    return info["to_canon"](value) if is_alt else value


# Detection thresholds are heuristics tuned to typical adult-male ranges (the
# same caveat as the male-only reference ranges in thresholds.py). For markers
# whose two unit systems overlap clinically (HbA1C, Prolactin) a whole column
# can still be mis-detected — heuristic detection can't resolve that. The upload
# summary surfaces every decision so the user can review and override (#62).


def detect_incoming_unit(test_name: str, values) -> tuple[str, bool]:
    """Decide the source unit for a whole marker column.

    Returns (detected_unit, ambiguous). The decision is the majority of values
    matching the alt-unit rule, so a column is never split across units; exact
    ties fall back to canonical. `ambiguous` flags any value on the opposite
    side of the threshold — the likely fingerprint of a data-entry error — for
    the user to check."""
    info = _INCOMING.get(test_name)
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if info is None:
        from thresholds import THRESHOLDS
        return THRESHOLDS.get(test_name, {}).get("unit", ""), False
    if arr.size == 0:
        return info["canonical"], False
    frac_alt = float(_matches_alt(info["alt_when"], arr).mean())
    detected = info["alt"] if frac_alt > 0.5 else info["canonical"]  # ties → canonical
    ambiguous = 0.0 < frac_alt < 1.0
    return detected, ambiguous


def to_canonical_column(test_name: str, values, force_unit: str | None = None) -> tuple[list, str, bool]:
    """Convert a whole marker column to canonical units using ONE unit decided
    for the column (per-column, not per-value).

    Returns (converted_values, detected_unit, ambiguous). `force_unit` skips
    detection (used by the override path); it must be the marker's canonical or
    alt unit — an unrecognised value raises rather than silently shipping the
    values raw under the wrong label."""
    info = _INCOMING.get(test_name)
    if info is None:
        detected, ambiguous = detect_incoming_unit(test_name, values)
        return list(values), detected, ambiguous
    if force_unit is not None:
        if force_unit not in (info["canonical"], info["alt"]):
            raise ValueError(
                f"Unknown source unit {force_unit!r} for {test_name} — "
                f"expected {info['canonical']!r} or {info['alt']!r}."
            )
        detected, ambiguous = force_unit, False
    else:
        detected, ambiguous = detect_incoming_unit(test_name, values)
    if detected == info["alt"]:
        converted = [info["to_canon"](float(v)) for v in values]
    else:
        converted = list(values)
    return converted, detected, ambiguous


# ── Display conversion (canonical → chosen display unit) ─────────────────────

def available_units(test_name: str) -> list[str]:
    """Return the list of display unit options for a test. Single-unit tests return one item."""
    from thresholds import THRESHOLDS
    if test_name in _DISPLAY_TRANSFORMS:
        return list(_DISPLAY_TRANSFORMS[test_name].keys())
    return [THRESHOLDS.get(test_name, {}).get("unit", "")]


def from_canonical(test_name: str, value: float, display_unit: str) -> float:
    """Convert a canonical value to the requested display unit."""
    transforms = _DISPLAY_TRANSFORMS.get(test_name, {})
    t = transforms.get(display_unit)
    return t["fn"](value) if t else value


def transform_for_display(
    test_name: str,
    values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    boundaries: list,
    display_unit: str,
) -> tuple:
    """
    Apply a display unit transform to all GMM plot quantities at once.
    Returns (values_d, means_d, stds_d, boundaries_d).
    All quantities stay in canonical units if display_unit matches canonical.
    """
    transforms = _DISPLAY_TRANSFORMS.get(test_name, {})
    t = transforms.get(display_unit)
    if not t:
        return values, means, stds, boundaries

    scale, shift = t["scale"], t["shift"]
    fn = t["fn"]

    values_d     = np.vectorize(fn)(values)
    means_d      = means * scale + shift
    stds_d       = stds  * abs(scale)
    boundaries_d = [b * scale + shift for b in boundaries]
    return values_d, means_d, stds_d, boundaries_d


def _assert_unit_config_consistent() -> None:
    """Guard the two sources of truth: a marker's canonical unit in `_INCOMING`
    must equal its reference unit in thresholds.py. Otherwise `to_canonical_column`
    would convert to canonical while parsing labels the row with the other unit —
    a silently inverted dataset with no test failure. Checked at import."""
    from thresholds import THRESHOLDS
    for marker, info in _INCOMING.items():
        ref_unit = THRESHOLDS.get(marker, {}).get("unit")
        if ref_unit != info["canonical"]:
            raise AssertionError(
                f"Unit config mismatch for {marker!r}: _INCOMING canonical "
                f"{info['canonical']!r} != thresholds.py unit {ref_unit!r}."
            )


_assert_unit_config_consistent()

