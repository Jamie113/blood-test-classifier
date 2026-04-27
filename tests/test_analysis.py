import numpy as np
import pandas as pd
import pytest
from stub_data import generate_stub_data
from analysis import analyse_upload, analyse_population, build_labelled_df


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
                "boundaries", "labels", "values", "cluster_stats", "small_sample"}
    for name, res in results.items():
        if "error" in res:
            continue
        assert required.issubset(res.keys()), f"{name} missing keys"


def test_labels_cover_all_values(stub_df):
    results = analyse_upload(stub_df)
    for name, res in results.items():
        if "error" in res:
            continue
        assert len(res["labels"]) == len(res["values"])


def test_n_components_in_valid_range(stub_df):
    results = analyse_upload(stub_df)
    for name, res in results.items():
        if "error" in res:
            continue
        assert 2 <= res["n_components"] <= 4, f"{name}: {res['n_components']} components"


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
    assert 2 <= result["n_clusters"] <= 5


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
