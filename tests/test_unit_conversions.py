import pytest
from unit_conversions import to_canonical


# ── Testosterone (nmol/L ↔ ng/dL, threshold > 100) ────────────────────────

def test_testosterone_ngdl_converted():
    # 346 ng/dL ÷ 28.84 ≈ 12.0 nmol/L
    result = to_canonical("Testosterone", 346)
    assert abs(result - 12.0) < 0.1

def test_testosterone_nmol_unchanged():
    result = to_canonical("Testosterone", 15.0)
    assert result == 15.0

def test_testosterone_at_boundary_unchanged():
    # Exactly 100 → not converted (rule is > 100)
    assert to_canonical("Testosterone", 100.0) == 100.0

def test_testosterone_above_boundary_converted():
    # 100.1 is detected as ng/dL
    assert to_canonical("Testosterone", 100.1) < 10


# ── HbA1C (mmol/mol ↔ %, threshold < 20) ─────────────────────────────────

def test_hba1c_pct_converted():
    # 6.5% → (6.5 - 2.15) × 10.929 ≈ 47.5 mmol/mol
    result = to_canonical("HbA1C", 6.5)
    assert abs(result - 47.5) < 0.5

def test_hba1c_mmol_unchanged():
    result = to_canonical("HbA1C", 45.0)
    assert result == 45.0

def test_hba1c_at_boundary_unchanged():
    # Exactly 20 → not converted (rule is < 20)
    assert to_canonical("HbA1C", 20.0) == 20.0

def test_hba1c_below_boundary_converted():
    result = to_canonical("HbA1C", 5.0)
    assert result > 20  # Should be in mmol/mol range


# ── Cholesterol — Total, LDL, HDL (mmol/L ↔ mg/dL, threshold > 15) ────────

def test_total_cholesterol_mgdl_converted():
    # 193 mg/dL ÷ 38.67 ≈ 4.99 mmol/L
    result = to_canonical("Total Cholesterol", 193)
    assert abs(result - 4.99) < 0.05

def test_total_cholesterol_mmol_unchanged():
    result = to_canonical("Total Cholesterol", 5.5)
    assert result == 5.5

def test_ldl_mgdl_converted():
    result = to_canonical("LDL Cholesterol", 116)
    assert abs(result - 3.0) < 0.1

def test_ldl_mmol_unchanged():
    result = to_canonical("LDL Cholesterol", 3.5)
    assert result == 3.5

def test_hdl_mgdl_converted():
    result = to_canonical("HDL Cholesterol", 39)  # ≈ 1.0 mmol/L
    assert abs(result - 1.01) < 0.05

def test_hdl_mmol_unchanged():
    result = to_canonical("HDL Cholesterol", 1.2)
    assert result == 1.2


# ── Oestradiol (pg/mL ↔ pmol/L, threshold > 200) ─────────────────────────

def test_oestradiol_pmol_converted():
    # 300 pmol/L ÷ 3.671 ≈ 81.7 pg/mL
    result = to_canonical("Oestradiol", 300)
    assert abs(result - 81.7) < 1.0

def test_oestradiol_pgml_unchanged():
    result = to_canonical("Oestradiol", 35.0)
    assert result == 35.0

def test_oestradiol_at_boundary_unchanged():
    assert to_canonical("Oestradiol", 200.0) == 200.0


# ── Prolactin (mIU/L ↔ ng/mL, threshold < 50) ────────────────────────────

def test_prolactin_ngml_converted():
    # 10 ng/mL × 21.2 = 212 mIU/L
    result = to_canonical("Prolactin", 10)
    assert abs(result - 212) < 2

def test_prolactin_miul_unchanged():
    result = to_canonical("Prolactin", 150.0)
    assert result == 150.0

def test_prolactin_at_boundary_unchanged():
    assert to_canonical("Prolactin", 50.0) == 50.0


# ── Free Testosterone (nmol/L ↔ pmol/L, threshold > 5) ───────────────────

def test_free_testosterone_pmol_converted():
    # 300 pmol/L ÷ 1000 = 0.3 nmol/L
    result = to_canonical("Free Testosterone", 300)
    assert abs(result - 0.3) < 0.001

def test_free_testosterone_nmol_unchanged():
    result = to_canonical("Free Testosterone", 0.4)
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
    assert to_canonical(test_name, value) == value
