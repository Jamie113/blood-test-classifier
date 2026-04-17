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

_TO_CANONICAL = {
    "Testosterone":     lambda v: v / _TESTOSTERONE_NMOL_TO_NGDL if v > 100 else v,
    "Free Testosterone":lambda v: v / _FREE_T_NMOL_TO_PMOL if v > 5 else v,
    "Total Cholesterol":lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL if v > 15 else v,
    "LDL Cholesterol":  lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL if v > 15 else v,
    "HDL Cholesterol":  lambda v: v / _CHOLESTEROL_MMOL_TO_MGDL if v > 15 else v,
    "HbA1C":            lambda v: _hba1c_pct_to_mmol(v) if v < 20 else v,
    "Oestradiol":       lambda v: v / _OESTRADIOL_PG_TO_PMOL if v > 200 else v,
    "Prolactin":        lambda v: v / _PROLACTIN_MIUL_TO_NGML if v < 50 else v,
}


def to_canonical(test_name: str, value: float) -> float:
    """Convert an incoming value to canonical units (used at CSV upload time)."""
    fn = _TO_CANONICAL.get(test_name)
    return fn(value) if fn else value


# ── Display conversion (canonical → chosen display unit) ─────────────────────

def available_units(test_name: str) -> list[str]:
    """Return the list of display unit options for a test. Single-unit tests return one item."""
    from thresholds import THRESHOLDS
    if test_name in _DISPLAY_TRANSFORMS:
        return list(_DISPLAY_TRANSFORMS[test_name].keys())
    return [THRESHOLDS.get(test_name, {}).get("unit", "")]


def canonical_unit(test_name: str) -> str:
    """Return the canonical unit name for a test."""
    from thresholds import THRESHOLDS
    return THRESHOLDS.get(test_name, {}).get("unit", "")


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

