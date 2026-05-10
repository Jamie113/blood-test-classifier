"""Shared CSV parser.

Wide-format blood-test export → long-format DataFrame with canonical units.
Used by both the Streamlit app and the FastAPI front-end.
"""
from __future__ import annotations

from typing import IO, Union

import pandas as pd

from column_map import AGE_COLUMN, COLUMN_MAP, ID_COLUMN
from thresholds import THRESHOLDS
from unit_conversions import to_canonical


def parse_csv(source: Union[str, bytes, IO]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Parse a wide-format CSV into a long-format DataFrame.

    `source` is anything pandas.read_csv accepts: a path, bytes-like object,
    or file-like object (e.g. an UploadFile.file from FastAPI).

    Returns:
        df_long       Long-format frame with columns
                      [patient_id, age, test_name, value, unit].
                      Values are stored in canonical units.
        recognised    CSV column names that mapped to a known test.
        unrecognised  CSV column names that were skipped (excluding the ID
                      column itself).
    """
    df_raw = pd.read_csv(source)
    df_raw = df_raw[
        df_raw[ID_COLUMN].notna()
        & (df_raw[ID_COLUMN].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    has_age = AGE_COLUMN in df_raw.columns

    frames: list[pd.DataFrame] = []
    seen_tests: set[str] = set()
    for col, mapping in COLUMN_MAP.items():
        if col not in df_raw.columns:
            continue
        test_name = mapping["test"]
        if test_name in seen_tests:
            continue  # skip duplicate column mappings (e.g. Haematocrit ADJ)
        seen_tests.add(test_name)

        sub = df_raw[[ID_COLUMN, col]].copy()
        if has_age:
            sub[AGE_COLUMN] = df_raw[AGE_COLUMN]
        sub = sub.dropna(subset=[col]).reset_index(drop=True)
        if sub.empty:
            continue

        sub["patient_id"] = sub[ID_COLUMN].astype(str)
        sub["age"] = (
            pd.to_numeric(sub[AGE_COLUMN], errors="coerce").astype("Int64")
            if has_age else None
        )
        sub["value"] = sub[col].astype(float) * mapping["scale"]
        sub["value"] = sub["value"].apply(lambda v, t=test_name: to_canonical(t, v))
        sub["test_name"] = test_name
        sub["unit"] = THRESHOLDS[test_name]["unit"]
        frames.append(sub[["patient_id", "age", "test_name", "value", "unit"]])

    if frames:
        df_long = pd.concat(frames, ignore_index=True)
    else:
        df_long = pd.DataFrame(columns=["patient_id", "age", "test_name", "value", "unit"])

    recognised = [c for c in COLUMN_MAP if c in df_raw.columns]
    unrecognised = [c for c in df_raw.columns if c not in COLUMN_MAP and c != ID_COLUMN]
    return df_long, recognised, unrecognised
