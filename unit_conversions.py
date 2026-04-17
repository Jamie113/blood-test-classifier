# unit_conversions.py
# Auto-detects which unit a value is in and converts to the canonical unit
# used in thresholds.py.
#
# Detection is based on value ranges that don't overlap between unit systems.
# Each entry documents the detection heuristic and conversion applied.

# Conversion factors
_TESTOSTERONE_NGDL_TO_NMOL = 1 / 28.84       # ng/dL → nmol/L
_CHOLESTEROL_MGDL_TO_MMOL  = 1 / 38.67       # mg/dL → mmol/L
_OESTRADIOL_PMOL_TO_PG     = 1 / 3.671       # pmol/L → pg/mL
_PROLACTIN_NGML_TO_MIUL    = 21.2            # ng/mL → mIU/L
_FREE_T_PMOL_TO_NMOL       = 1 / 1000        # pmol/L → nmol/L

# HbA1C: % (NGSP) → mmol/mol (IFCC)
def _hba1c_pct_to_mmol(pct: float) -> float:
    return (pct - 2.15) * 10.929


# Maps test_name → (detect_fn, canonical_unit, alt_unit)
# detect_fn: value → canonical value (converts if needed, passes through if already canonical)
_CONVERSIONS: dict[str, tuple] = {
    "Testosterone": (
        lambda v: v * _TESTOSTERONE_NGDL_TO_NMOL if v > 100 else v,
        "nmol/L", "ng/dL (auto-detected if value > 100)",
    ),
    "Free Testosterone": (
        lambda v: v * _FREE_T_PMOL_TO_NMOL if v > 5 else v,
        "nmol/L", "pmol/L (auto-detected if value > 5)",
    ),
    "Total Cholesterol": (
        lambda v: v * _CHOLESTEROL_MGDL_TO_MMOL if v > 15 else v,
        "mmol/L", "mg/dL (auto-detected if value > 15)",
    ),
    "LDL Cholesterol": (
        lambda v: v * _CHOLESTEROL_MGDL_TO_MMOL if v > 15 else v,
        "mmol/L", "mg/dL (auto-detected if value > 15)",
    ),
    "HDL Cholesterol": (
        lambda v: v * _CHOLESTEROL_MGDL_TO_MMOL if v > 15 else v,
        "mmol/L", "mg/dL (auto-detected if value > 15)",
    ),
    "HbA1C": (
        lambda v: _hba1c_pct_to_mmol(v) if v < 20 else v,
        "mmol/mol", "% NGSP (auto-detected if value < 20)",
    ),
    "Oestradiol": (
        lambda v: v * _OESTRADIOL_PMOL_TO_PG if v > 200 else v,
        "pg/mL", "pmol/L (auto-detected if value > 200)",
    ),
    "Prolactin": (
        lambda v: v * _PROLACTIN_NGML_TO_MIUL if v < 50 else v,
        "mIU/L", "ng/mL (auto-detected if value < 50)",
    ),
}


def to_canonical(test_name: str, value: float) -> float:
    """Convert value to canonical unit if a conversion rule exists for this test."""
    if test_name in _CONVERSIONS:
        fn, _, _ = _CONVERSIONS[test_name]
        return fn(value)
    return value


def unit_hint(test_name: str) -> str:
    """Return a display string describing accepted units for this test."""
    from thresholds import THRESHOLDS
    canonical = THRESHOLDS.get(test_name, {}).get("unit", "")
    if test_name in _CONVERSIONS:
        _, canon, alt = _CONVERSIONS[test_name]
        return f"{canon} or {alt}"
    return canonical
