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
    return info["to_canon"](value) if _matches_alt(info["alt_when"], np.asarray(value)) else value


# Detection thresholds are heuristics tuned to typical adult-male ranges (the
# same caveat as the male-only reference ranges in thresholds.py). The `lt`
# markers in particular (HbA1C < 20, Prolactin < 50) sit near the clinical
# overlap between the two unit systems, so a column with an unusual case-mix can
# be misclassified — which is why we also flag any threshold-crosser below.
_MIN_DETECT_N = 3  # below this, the column is too short to trust the majority


def detect_incoming_unit(test_name: str, values) -> tuple[str, bool]:
    """Decide the source unit for a whole marker column.

    Returns (detected_unit, ambiguous). The decision is by majority of values
    matching the alt-unit rule — so a column is never split across units.

    `ambiguous` is True when ANY value falls on the opposite side of the
    threshold from the rest: a single rogue value (e.g. one mistyped ng/dL cell
    in an nmol/L column) is the smoking gun for a data-entry error and is left
    unconverted, so it must be surfaced rather than silently shipped into the
    GMM as a fake outlier. Columns too short to trust, and exact 50/50 ties,
    fall back to canonical (no conversion) and are flagged."""
    info = _INCOMING.get(test_name)
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if info is None:
        from thresholds import THRESHOLDS
        return THRESHOLDS.get(test_name, {}).get("unit", ""), False
    if arr.size == 0:
        return info["canonical"], False
    if arr.size < _MIN_DETECT_N:
        return info["canonical"], True  # too few values to convert confidently
    frac_alt = float(_matches_alt(info["alt_when"], arr).mean())
    detected = info["alt"] if frac_alt > 0.5 else info["canonical"]  # ties → canonical
    ambiguous = 0.0 < frac_alt < 1.0  # any value on the wrong side of the threshold
    return detected, ambiguous


def to_canonical_column(test_name: str, values, force_unit: str | None = None) -> tuple[list, str, bool]:
    """Convert a whole marker column to canonical units using ONE unit decided
    for the column (per-column, not per-value).

    Returns (converted_values, detected_unit, ambiguous). `force_unit` skips
    detection (used by the override path)."""
    info = _INCOMING.get(test_name)
    if info is None:
        detected, ambiguous = detect_incoming_unit(test_name, values)
        return list(values), detected, ambiguous
    if force_unit is not None:
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

