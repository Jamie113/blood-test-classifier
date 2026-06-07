import pytest

from unit_conversions import (
    _to_canonical,
    detect_incoming_unit,
    to_canonical_column,
)

# ── Testosterone (nmol/L ↔ ng/dL, threshold > 100) ────────────────────────

def test_testosterone_ngdl_converted():
    # 346 ng/dL ÷ 28.84 ≈ 12.0 nmol/L
    result = _to_canonical("Testosterone", 346)
    assert abs(result - 12.0) < 0.1

def test_testosterone_nmol_unchanged():
    result = _to_canonical("Testosterone", 15.0)
    assert result == 15.0

def test_testosterone_at_boundary_unchanged():
    # Exactly 100 → not converted (rule is > 100)
    assert _to_canonical("Testosterone", 100.0) == 100.0

def test_testosterone_above_boundary_converted():
    # 100.1 is detected as ng/dL
    assert _to_canonical("Testosterone", 100.1) < 10


# ── HbA1C (mmol/mol ↔ %, threshold < 20) ─────────────────────────────────

def test_hba1c_pct_converted():
    # 6.5% → (6.5 - 2.15) × 10.929 ≈ 47.5 mmol/mol
    result = _to_canonical("HbA1C", 6.5)
    assert abs(result - 47.5) < 0.5

def test_hba1c_mmol_unchanged():
    result = _to_canonical("HbA1C", 45.0)
    assert result == 45.0

def test_hba1c_at_boundary_unchanged():
    # Exactly 20 → not converted (rule is < 20)
    assert _to_canonical("HbA1C", 20.0) == 20.0

def test_hba1c_below_boundary_converted():
    result = _to_canonical("HbA1C", 5.0)
    assert result > 20  # Should be in mmol/mol range


# ── Cholesterol — Total, LDL, HDL (mmol/L ↔ mg/dL, threshold > 15) ────────

def test_total_cholesterol_mgdl_converted():
    # 193 mg/dL ÷ 38.67 ≈ 4.99 mmol/L
    result = _to_canonical("Total Cholesterol", 193)
    assert abs(result - 4.99) < 0.05

def test_total_cholesterol_mmol_unchanged():
    result = _to_canonical("Total Cholesterol", 5.5)
    assert result == 5.5

def test_ldl_mgdl_converted():
    result = _to_canonical("LDL Cholesterol", 116)
    assert abs(result - 3.0) < 0.1

def test_ldl_mmol_unchanged():
    result = _to_canonical("LDL Cholesterol", 3.5)
    assert result == 3.5

def test_hdl_mgdl_converted():
    result = _to_canonical("HDL Cholesterol", 39)  # ≈ 1.0 mmol/L
    assert abs(result - 1.01) < 0.05

def test_hdl_mmol_unchanged():
    result = _to_canonical("HDL Cholesterol", 1.2)
    assert result == 1.2


# ── Oestradiol (pg/mL ↔ pmol/L, threshold > 200) ─────────────────────────

def test_oestradiol_pmol_converted():
    # 300 pmol/L ÷ 3.671 ≈ 81.7 pg/mL
    result = _to_canonical("Oestradiol", 300)
    assert abs(result - 81.7) < 1.0

def test_oestradiol_pgml_unchanged():
    result = _to_canonical("Oestradiol", 35.0)
    assert result == 35.0

def test_oestradiol_at_boundary_unchanged():
    assert _to_canonical("Oestradiol", 200.0) == 200.0


# ── Prolactin (mIU/L ↔ ng/mL, threshold < 50) ────────────────────────────

def test_prolactin_ngml_converted():
    # 10 ng/mL × 21.2 = 212 mIU/L
    result = _to_canonical("Prolactin", 10)
    assert abs(result - 212) < 2

def test_prolactin_miul_unchanged():
    result = _to_canonical("Prolactin", 150.0)
    assert result == 150.0

def test_prolactin_at_boundary_unchanged():
    assert _to_canonical("Prolactin", 50.0) == 50.0


# ── Free Testosterone (nmol/L ↔ pmol/L, threshold > 5) ───────────────────

def test_free_testosterone_pmol_converted():
    # 300 pmol/L ÷ 1000 = 0.3 nmol/L
    result = _to_canonical("Free Testosterone", 300)
    assert abs(result - 0.3) < 0.001

def test_free_testosterone_nmol_unchanged():
    result = _to_canonical("Free Testosterone", 0.4)
    assert result == 0.4


# ── Tests with no conversion rule should pass through unchanged ─────────────

@pytest.mark.parametrize("test_name,value", [
    ("TSH", 2.5),
    ("Albumin", 42.0),
    ("eGFR", 95.0),
    ("PSA", 1.2),
    ("Neutrophil Count", 4.5),
    ("White Blood Cell Count", 6.0),
])
def test_no_conversion_passthrough(test_name, value):
    assert _to_canonical(test_name, value) == value


# ── Per-column detection (the silent-corruption fix) ───────────────────────

def test_column_decides_one_unit_no_split():
    """A ng/dL Testosterone column converts as ONE unit — the old per-value
    rule left a low-but-valid 90 unconverted while its column-mates converted."""
    col = [300.0, 420.0, 250.0, 380.0, 410.0, 90.0]   # clearly ng/dL, one low reading
    converted, detected, ambiguous = to_canonical_column("Testosterone", col)
    assert detected == "ng/dL"
    # every value converted, including the 90 that per-value logic would keep
    assert all(abs(c - raw / 28.84) < 1e-6 for c, raw in zip(converted, col, strict=True))
    assert converted[-1] < 10           # the 90 is now ~3.1 nmol/L, not left as 90
    assert ambiguous                    # the boundary-crossing 90 is surfaced


def test_unanimous_column_not_flagged():
    """A column entirely on one side of the threshold is not ambiguous."""
    col = [300.0, 420.0, 250.0, 380.0]   # all ng/dL
    _, detected, ambiguous = to_canonical_column("Testosterone", col)
    assert detected == "ng/dL" and not ambiguous


def test_column_canonical_left_unchanged():
    col = [12.0, 18.0, 9.0, 25.0]          # all nmol/L, unanimous
    converted, detected, ambiguous = to_canonical_column("Testosterone", col)
    assert detected == "nmol/L"
    assert not ambiguous
    assert converted == col


def test_single_rogue_value_is_flagged_not_silently_kept():
    """Reviewer's inverse case: one mistyped ng/dL cell in an nmol/L column is
    left unconverted (it's the minority) but MUST be flagged ambiguous, not
    shipped silently into the GMM as a fake outlier."""
    col = [12.0, 15.0, 9.0, 20.0, 18.0, 300.0]   # one rogue 300
    converted, detected, ambiguous = to_canonical_column("Testosterone", col)
    assert detected == "nmol/L"          # majority is nmol/L
    assert ambiguous                     # the 300 crosser is surfaced
    assert converted[-1] == 300.0        # left unconverted — flagged for the user


def test_single_clear_value_converts_unflagged():
    """A unanimous solo reading (346 ng/dL) converts best-effort and is NOT
    flagged — leaving it raw would ship 346 into the GMM as 346 nmol/L."""
    converted, detected, ambiguous = to_canonical_column("Testosterone", [346.0])
    assert detected == "ng/dL"
    assert abs(converted[0] - 346 / 28.84) < 1e-6   # converted, not shipped raw
    assert not ambiguous                             # unanimous → no spurious flag


def test_exact_tie_falls_back_to_canonical():
    detected, ambiguous = detect_incoming_unit("Testosterone", [90.0, 90.0, 110.0, 110.0])
    assert detected == "nmol/L"          # 50/50 → canonical (no confident convert)
    assert ambiguous


def test_force_unit_overrides_detection():
    col = [300.0, 420.0]                    # would auto-detect ng/dL
    converted, detected, _ = to_canonical_column("Testosterone", col, force_unit="nmol/L")
    assert detected == "nmol/L"
    assert converted == col                 # forced canonical → no conversion


def test_force_unit_rejects_unknown():
    """The override API must reject a typo'd / unknown unit, not silently ship
    the values raw under the wrong label (the #62 contract)."""
    with pytest.raises(ValueError):
        to_canonical_column("Testosterone", [300.0], force_unit="ng/dl")  # wrong case


def test_unit_config_is_self_consistent():
    """_INCOMING canonical units must match thresholds.py (guards label inversion)."""
    from unit_conversions import _assert_unit_config_consistent
    _assert_unit_config_consistent()  # raises on mismatch


def test_non_convertible_marker_passthrough_column():
    converted, detected, ambiguous = to_canonical_column("TSH", [2.0, 3.0])
    assert converted == [2.0, 3.0]
    assert not ambiguous
