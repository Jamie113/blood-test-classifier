import numpy as np
import pandas as pd
import pytest

from analysis import (
    DERIVED_MARKERS,
    analyse_population,
    analyse_upload,
    build_labelled_df,
    filter_long,
    most_separated_marker,
    strongest_marker_pair,
)
from stub_data import generate_stub_data


@pytest.fixture(scope="module")
def stub_df():
    return generate_stub_data()


# ── analyse_upload ────────────────────────────────────────────────────────────

def test_returns_dict_keyed_by_marker(stub_df):
    results = analyse_upload(stub_df)
    assert isinstance(results, dict)
    assert len(results) == stub_df["test_name"].nunique()


def test_each_result_has_required_keys(stub_df):
    results = analyse_upload(stub_df)
    required = {"n_components", "bic_scores", "means", "stds", "weights",
                "boundaries", "labels", "values", "small_sample",
                "gmm", "order_inverse"}
    for name, res in results.items():
        if "error" in res:
            continue
        assert required.issubset(res.keys()), f"{name} missing keys"


def test_labels_cover_all_values(stub_df):
    results = analyse_upload(stub_df)
    for _name, res in results.items():
        if "error" in res:
            continue
        assert len(res["labels"]) == len(res["values"])


def test_n_components_in_valid_range(stub_df):
    results = analyse_upload(stub_df)
    for name, res in results.items():
        if "error" in res:
            continue
        # K=1 is now allowed (the ΔBIC≥6 floor lets uniform markers report
        # a single Gaussian rather than overstate spurious sub-groups).
        assert 1 <= res["n_components"] <= 4, f"{name}: {res['n_components']} components"


def test_too_few_points_returns_error():
    df = pd.DataFrame([
        {"patient_id": "P1", "age": 30, "test_name": "TSH", "value": 2.0, "unit": "mIU/L"},
        {"patient_id": "P2", "age": 30, "test_name": "TSH", "value": 2.5, "unit": "mIU/L"},
    ])
    results = analyse_upload(df)
    assert "error" in results["TSH"]


# ── build_labelled_df ─────────────────────────────────────────────────────────

def test_labelled_df_has_group_column(stub_df):
    results = analyse_upload(stub_df)
    labelled = build_labelled_df(stub_df, results)
    assert "Group" in labelled.columns


def test_labelled_df_same_length_as_input(stub_df):
    results = analyse_upload(stub_df)
    labelled = build_labelled_df(stub_df, results)
    assert len(labelled) == len(stub_df)


def test_group_values_are_strings(stub_df):
    results = analyse_upload(stub_df)
    labelled = build_labelled_df(stub_df, results)
    assert pd.api.types.is_string_dtype(labelled["Group"])


# ── analyse_population ────────────────────────────────────────────────────────

def test_population_returns_expected_keys(stub_df):
    result = analyse_population(stub_df)
    required = {"patient_ids", "labels", "n_clusters", "bic_scores",
                "X_pca_2d", "pca_var", "n_cluster_dims", "fingerprint", "df_wide"}
    assert required.issubset(result.keys())


def test_population_n_clusters_in_range(stub_df):
    result = analyse_population(stub_df)
    # K=1 is allowed (ΔBIC≥6 over no-cluster null required for K>1).
    assert 1 <= result["n_clusters"] <= 5


def test_population_labels_match_patient_count(stub_df):
    result = analyse_population(stub_df)
    assert len(result["labels"]) == len(result["patient_ids"])


def test_population_pca_2d_shape(stub_df):
    result = analyse_population(stub_df)
    assert result["X_pca_2d"].shape == (len(result["patient_ids"]), 2)


def test_population_error_on_too_few_patients():
    df = pd.DataFrame([
        {"patient_id": f"P{i}", "age": 30, "test_name": "TSH", "value": 2.0, "unit": "mIU/L"}
        for i in range(3)
    ])
    result = analyse_population(df)
    assert "error" in result


def test_population_returns_posteriors_and_mahalanobis(stub_df):
    result = analyse_population(stub_df)
    n = len(result["patient_ids"])
    assert result["posteriors"].shape == (n, result["n_clusters"])
    np.testing.assert_allclose(result["posteriors"].sum(axis=1), np.ones(n))
    assert result["mahalanobis_sq"].shape == (n,)
    assert (result["mahalanobis_sq"] >= 0).all()


def test_population_bic_search_includes_k1_and_caps_by_size(stub_df):
    """The BIC search must always include K=1 (for the null comparison) and
    must cap K above by sample size to avoid over-clustering small cohorts."""
    result = analyse_population(stub_df)
    assert 1 in result["bic_scores"]
    n_patients = len(result["patient_ids"])
    expected_max_k = max(2, min(5, n_patients // 25))
    assert max(result["bic_scores"]) <= expected_max_k


def test_uniform_marker_reports_k1():
    """When a marker has no real sub-structure, the per-marker GMM should
    fall back to K=1 rather than over-fitting to two components."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame([
        {"patient_id": f"P{i}", "age": 40, "test_name": "TSH",
         "value": float(rng.normal(2.0, 0.3)), "unit": "mIU/L"}
        for i in range(120)
    ])
    results = analyse_upload(df)
    assert results["TSH"]["n_components"] == 1


def test_uniform_population_reports_k1():
    """A homogeneous multivariate cohort should report K=1, not be forced
    into spurious clusters."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        rows.append({"patient_id": f"P{i}", "age": 40,
                     "test_name": "TSH", "value": float(rng.normal(2.0, 0.3)), "unit": "mIU/L"})
        rows.append({"patient_id": f"P{i}", "age": 40, "test_name": "Free T4",
                     "value": float(rng.normal(15.0, 1.5)), "unit": "pmol/L"})
    df = pd.DataFrame(rows)
    result = analyse_population(df)
    assert result["n_clusters"] == 1


# ── filter_long ───────────────────────────────────────────────────────────────

def test_filter_long_no_filters_returns_input(stub_df):
    out = filter_long(stub_df)
    assert len(out) == len(stub_df)
    assert set(out["patient_id"]) == set(stub_df["patient_id"])


def test_filter_long_age_range(stub_df):
    out = filter_long(stub_df, age_range=(40, 50))
    ages = out.drop_duplicates("patient_id")["age"]
    assert ages.between(40, 50).all()
    assert len(ages) > 0
    assert len(ages) < stub_df["patient_id"].nunique()


def test_filter_long_marker_range(stub_df):
    # Restrict to patients with low HbA1C — should drop the elevated subgroup
    out = filter_long(stub_df, marker_ranges={"HbA1C": (0, 36)})
    out_hba1c = out[out["test_name"] == "HbA1C"]["value"]
    assert (out_hba1c <= 36).all()
    assert out["patient_id"].nunique() < stub_df["patient_id"].nunique()


def test_filter_long_combined_filters(stub_df):
    out = filter_long(stub_df, age_range=(45, 65), marker_ranges={"HbA1C": (40, 100)})
    assert out["patient_id"].nunique() > 0
    assert out["patient_id"].nunique() < stub_df["patient_id"].nunique()
    # All retained patients meet both criteria
    ages = out.drop_duplicates("patient_id")["age"]
    assert ages.between(45, 65).all()
    hba1c = out[out["test_name"] == "HbA1C"]["value"]
    assert (hba1c >= 40).all()


def test_filter_long_keeps_patients_without_filter_marker(stub_df):
    # Drop HbA1C from one patient, then filter by HbA1C range — that patient
    # should remain because they have no value to test against.
    df = stub_df.copy()
    a_patient = df["patient_id"].iloc[0]
    df = df[~((df["patient_id"] == a_patient) & (df["test_name"] == "HbA1C"))]
    out = filter_long(df, marker_ranges={"HbA1C": (50, 100)})
    assert a_patient in out["patient_id"].values


def test_filter_long_empty_input():
    df = pd.DataFrame(columns=["patient_id", "age", "test_name", "value", "unit"])
    out = filter_long(df, age_range=(0, 100))
    assert out.empty


# ── most_separated_marker ─────────────────────────────────────────────────────

def test_most_separated_marker_returns_marker_with_clearest_split(stub_df):
    results = analyse_upload(stub_df)
    pick = most_separated_marker(results)
    assert pick is not None
    name, score = pick
    assert name in results
    assert score > 0
    assert name not in DERIVED_MARKERS
    # All other multi-cluster, non-derived markers should score <= the picked one
    for _n, r in results.items():
        if _n in DERIVED_MARKERS or "error" in r or r.get("n_components", 0) < 2:
            continue
        pooled = float(np.mean(r["stds"]))
        if pooled <= 0:
            continue
        s = float(r["means"].max() - r["means"].min()) / pooled
        assert s <= score + 1e-9


def test_most_separated_marker_handles_empty_or_errored():
    assert most_separated_marker({}) is None
    assert most_separated_marker({"X": {"error": "too few"}}) is None


# ── strongest_marker_pair ─────────────────────────────────────────────────────

def test_strongest_marker_pair_returns_a_pair(stub_df):
    pick = strongest_marker_pair(stub_df)
    assert pick is not None
    a, b, r = pick
    assert a != b
    assert -1.0 <= r <= 1.0


def test_strongest_marker_pair_empty():
    df = pd.DataFrame(columns=["patient_id", "age", "test_name", "value", "unit"])
    assert strongest_marker_pair(df) is None


def test_strongest_marker_pair_single_marker():
    df = pd.DataFrame([
        {"patient_id": f"P{i}", "age": 40, "test_name": "TSH",
         "value": 2.0 + i * 0.1, "unit": "mIU/L"}
        for i in range(20)
    ])
    assert strongest_marker_pair(df) is None


# ── Derived-marker exclusion + deterministic selection (#59) ──────────────────

def _long_rows(by_marker: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for marker, vals in by_marker.items():
        for i, v in enumerate(vals):
            rows.append({"patient_id": f"P{i}", "age": 40,
                         "test_name": marker, "value": v, "unit": "x"})
    return pd.DataFrame(rows)


def test_strongest_pair_excludes_derived_even_when_it_correlates_hardest():
    """A derived marker that perfectly tracks its component must not be the
    headline pair — it's tautological."""
    base = [float(i) for i in range(20)]
    derived = next(iter(DERIVED_MARKERS))
    df = _long_rows({
        "Total Cholesterol": base,
        derived:             [v * 1.0001 for v in base],   # ~perfect corr (tautology)
        "SHBG":              [v * 0.8 + 3 for v in base],   # strong but genuine
    })
    a, b, _ = strongest_marker_pair(df)
    assert derived not in (a, b)


def test_strongest_pair_skips_structurally_correlated_lipids():
    """Total Cholesterol ↔ LDL/HDL correlate by the additive lipid identity, so
    a genuine (cross-system) pair must headline instead — even when the lipid
    pair has the higher |r|."""
    base = [float(i) for i in range(20)]
    df = _long_rows({
        "Total Cholesterol": base,
        "LDL Cholesterol":   [v + 0.001 for v in base],     # ~perfect (structural)
        "SHBG":              [v * 0.85 + 2 for v in base],   # strong, genuine
        "Albumin":           [v * 0.85 + 1 for v in base],
    })
    a, b, _ = strongest_marker_pair(df)
    assert {a, b} != {"Total Cholesterol", "LDL Cholesterol"}


def test_most_separated_excludes_derived():
    derived = next(iter(DERIVED_MARKERS))
    results = {
        "Albumin": {"n_components": 2, "means": np.array([1.0, 3.0]),
                    "stds": np.array([0.5, 0.5])},
        derived:   {"n_components": 2, "means": np.array([0.0, 99.0]),
                    "stds": np.array([0.5, 0.5])},
    }
    name, _ = most_separated_marker(results)
    assert name == "Albumin"   # the bigger-separation derived marker is skipped


def test_strongest_pair_tiebreak_is_deterministic():
    """When several pairs tie on |r|, the winner is a stable alphabetical key,
    not whatever the corr-matrix / insertion order happens to be."""
    base = [float(i) for i in range(15)]
    df = _long_rows({"CCC": base, "AAA": list(base), "BBB": list(base)})  # all r=1.0
    a, b, _ = strongest_marker_pair(df)
    assert (a, b) == ("AAA", "BBB")   # alphabetical, regardless of insert order


def test_population_evidence_floor_small_cohort_reports_one_cluster(stub_df):
    """L2 (#58): below ~50 patients (two clusters of ~25) the population fit
    must report a single cluster, not split a small cohort into 'distinct' ones."""
    keep = stub_df["patient_id"].drop_duplicates().head(40)
    small = stub_df[stub_df["patient_id"].isin(keep)]
    res = analyse_population(small)
    assert "error" not in res
    assert res["n_clusters"] == 1
    assert max(res["bic_scores"]) == 1  # K=2 not even attempted below the floor


def test_population_full_demo_still_reaches_two_clusters(stub_df):
    """The n=80 demo is well above the floor and still finds its two subgroups."""
    res = analyse_population(stub_df)
    assert res["n_clusters"] == 2
