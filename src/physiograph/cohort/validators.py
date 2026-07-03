"""Cohort contract validators and assertion helpers.

Provides lightweight assertion functions that validate cohort DataFrames
against structural contracts (required columns, no duplicate stays,
minimum size).  These are used at pipeline boundaries to catch data
integrity issues early.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def assert_required_columns(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    context: str = "",
) -> None:
    """Assert that *df* contains all *required_cols*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required_cols:
        Column names that must be present.
    context:
        Optional label (e.g. dataset name) included in error messages.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        label = f" [{context}]" if context else ""
        raise ValueError(
            f"Missing required columns{label}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def assert_no_duplicate_stays(
    df: pd.DataFrame,
    stay_id_col: str = "stay_id",
    context: str = "",
) -> None:
    """Assert that *df* has no duplicate values in *stay_id_col*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    stay_id_col:
        Column name for the stay identifier (default ``"stay_id"``).
    context:
        Optional label included in error messages.

    Raises
    ------
    ValueError
        If duplicate stay IDs are found, with the count and examples.
    """
    if stay_id_col not in df.columns:
        raise ValueError(
            f"Stay ID column '{stay_id_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    duplicates = df[stay_id_col].duplicated()
    n_dupes = int(duplicates.sum())
    if n_dupes > 0:
        dupe_ids = df.loc[duplicates, stay_id_col].head(10).tolist()
        label = f" [{context}]" if context else ""
        raise ValueError(
            f"Found {n_dupes} duplicate stay IDs{label}. "
            f"Examples: {dupe_ids}"
        )


def assert_cohort_size(
    df: pd.DataFrame,
    min_size: int,
    context: str = "",
) -> None:
    """Assert that *df* has at least *min_size* rows.

    Parameters
    ----------
    df:
        DataFrame to validate.
    min_size:
        Minimum number of rows required.
    context:
        Optional label included in error messages.

    Raises
    ------
    ValueError
        If the DataFrame has fewer rows than *min_size*.
    """
    n = len(df)
    if n < min_size:
        label = f" [{context}]" if context else ""
        raise ValueError(
            f"Cohort has {n} rows{label}, but minimum required is {min_size}. "
            f"This may indicate a data loading or filtering error."
        )


def assert_flag_values(
    df: pd.DataFrame,
    flag_cols: Sequence[str],
    allowed_values: tuple[int, ...] = (0, 1),
    context: str = "",
) -> None:
    """Assert that flag columns contain only *allowed_values*.

    Parameters
    ----------
    df:
        DataFrame to validate.
    flag_cols:
        Column names that should contain only binary flag values.
    allowed_values:
        Tuple of allowed integer values (default ``(0, 1)``).
    context:
        Optional label included in error messages.

    Raises
    ------
    ValueError
        If any flag column contains values outside *allowed_values*.
    """
    present_cols = [c for c in flag_cols if c in df.columns]
    for col in present_cols:
        unique_vals = df[col].dropna().unique()
        invalid = [v for v in unique_vals if v not in allowed_values]
        if invalid:
            label = f" [{context}]" if context else ""
            raise ValueError(
                f"Column '{col}'{label} contains invalid values: {invalid}. "
                f"Allowed: {list(allowed_values)}"
            )


def assert_no_nulls_in_required(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    context: str = "",
) -> None:
    """Assert that *required_cols* in *df* have no null values.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required_cols:
        Column names that must not contain nulls.
    context:
        Optional label included in error messages.

    Raises
    ------
    ValueError
        If any required column contains null values, with counts.
    """
    present_cols = [c for c in required_cols if c in df.columns]
    null_counts = {c: int(df[c].isna().sum()) for c in present_cols}
    null_cols = {c: n for c, n in null_counts.items() if n > 0}
    if null_cols:
        label = f" [{context}]" if context else ""
        raise ValueError(
            f"Null values found in required columns{label}: {null_cols}"
        )