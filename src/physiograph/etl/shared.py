"""Shared utilities for ETL extractors (MIMIC and eICU).

Provides chunked CSV reading, event frame construction, text normalization,
time-window classification, and ICD prefix matching used by both extractors.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Time Window Constants
# ──────────────────────────────────────────────────────────────────────

OBSERVATION_HOURS: float = 4.0
OUTCOME_HOURS: float = 24.0
OUTCOME_WINDOW_END_HOURS: float = OBSERVATION_HOURS + OUTCOME_HOURS
TIME_STEP_HOURS: float = 0.25
TIME_STEP_MINUTES: int = int(TIME_STEP_HOURS * 60)
LANDMARK_MINUTES: int = int(OBSERVATION_HOURS * 60)
OUTCOME_WINDOW_END_MINUTES: int = int(OUTCOME_WINDOW_END_HOURS * 60)

# ──────────────────────────────────────────────────────────────────────
# Required Column Sets
# ──────────────────────────────────────────────────────────────────────

EVENT_REQUIRED_COLUMNS: list[str] = [
    "dataset",
    "stay_id",
    "event_family",
    "concept",
    "source_table",
    "raw_name",
    "offset_minutes",
    "window",
    "time_bin",
    "value_numeric",
    "value_text",
    "unit",
    "is_intervention",
    "is_outcome_event",
]

COHORT_REQUIRED_COLUMNS: list[str] = [
    "dataset",
    "stay_id",
    "person_id",
    "admit_time",
    "admit_year",
    "age",
    "is_male",
    "cohort_hf_flag",
    "shock_icd_flag",
    "early_icu_flag",
    "death_offset_minutes",
    "excluded_before_landmark_flag",
    "exclusion_reason",
]

LABEL_REQUIRED_COLUMNS: list[str] = [
    "dataset",
    "stay_id",
    "target",
    "pressor_24h_flag",
    "mcs_24h_flag",
    "escalation_24h_flag",
    "renal_injury_24h_flag",
    "hypoperfusion_24h_flag",
    "hepatic_injury_24h_flag",
    "end_organ_24h_flag",
    "mortality_24h_flag",
    "shock_progression_24h_flag",
    "landmark_lactate",
    "post_landmark_lactate_last",
    "lactate_clearance_24h_pct",
    "complete_lactate_clearance_24h_flag",
    "clearance_ge_64_24h_flag",
]

# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SourceExtraction:
    """Result of a source extraction step.

    Attributes:
        cohort_df: Cohort DataFrame with stay-level metadata.
        events_df: Long-format events DataFrame with timestamps.
    """

    cohort_df: pd.DataFrame
    events_df: pd.DataFrame


# ──────────────────────────────────────────────────────────────────────
# Chunked CSV Helpers
# ──────────────────────────────────────────────────────────────────────


def _iter_csv_chunks(
    path: Path,
    *,
    usecols: list[str],
    chunksize: int,
    max_chunks: int | None,
):
    """Yield DataFrames from a CSV file in memory-safe chunks.

    Args:
        path: Path to the CSV file.
        usecols: Columns to load (reduces memory per chunk).
        chunksize: Number of rows per chunk.
        max_chunks: Optional cap on the number of chunks to read.

    Yields:
        DataFrame chunks from the CSV.
    """
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    ):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        yield chunk


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate event frames or return an empty frame with required columns.

    Args:
        frames: List of event DataFrames.

    Returns:
        Concatenated DataFrame, or an empty frame with EVENT_REQUIRED_COLUMNS.
    """
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)


# ──────────────────────────────────────────────────────────────────────
# Event Frame Construction
# ──────────────────────────────────────────────────────────────────────


def build_event_frame(
    *,
    dataset: str,
    stay_id: pd.Series,
    event_family: str,
    concept: pd.Series | str,
    source_table: str,
    raw_name: pd.Series | str,
    offset_minutes: pd.Series,
    value_numeric: "pd.Series | object" = pd.NA,
    value_text: "pd.Series | object" = pd.NA,
    unit: "pd.Series | object" = pd.NA,
    is_intervention: int = 0,
) -> pd.DataFrame:
    """Build a standardized event DataFrame from raw extraction outputs.

    Args:
        dataset: Dataset identifier ("mimic" or "eicu").
        stay_id: Stay identifiers.
        event_family: Category ("lab", "vital", "intervention", "death").
        concept: Normalized concept name(s).
        source_table: Source CSV filename.
        raw_name: Raw identifier string(s) from the source.
        offset_minutes: Minutes from ICU anchor time.
        value_numeric: Numeric measurement values.
        value_text: Text measurement values.
        unit: Measurement unit.
        is_intervention: 1 if an intervention event, 0 otherwise.

    Returns:
        DataFrame with EVENT_REQUIRED_COLUMNS.
    """
    return pd.DataFrame(
        {
            "dataset": dataset,
            "stay_id": pd.to_numeric(stay_id, errors="coerce").astype("Int64"),
            "event_family": event_family,
            "concept": concept,
            "source_table": source_table,
            "raw_name": raw_name,
            "offset_minutes": offset_minutes,
            "window": "",
            "time_bin": pd.NA,
            "value_numeric": value_numeric,
            "value_text": value_text,
            "unit": unit,
            "is_intervention": int(is_intervention),
            "is_outcome_event": 0,
        }
    )


def _finalize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Assign time windows, bins, and outcome flags to raw events.

    Drops rows with missing offset_minutes or stay_id, and assigns
    observation/outcome windows based on the landmark clock.

    Args:
        events_df: Raw events DataFrame from extractors.

    Returns:
        Cleaned events DataFrame with window/time_bin/is_outcome_event filled.
    """
    if events_df.empty:
        return pd.DataFrame(columns=EVENT_REQUIRED_COLUMNS)
    events_df = events_df.copy()
    events_df["offset_minutes"] = pd.to_numeric(
        events_df["offset_minutes"], errors="coerce"
    )
    events_df = events_df.dropna(subset=["offset_minutes"])
    events_df["stay_id"] = pd.to_numeric(
        events_df["stay_id"], errors="coerce"
    ).astype("Int64")
    events_df = events_df.dropna(subset=["stay_id"])
    events_df["stay_id"] = events_df["stay_id"].astype(int)
    events_df["window"] = events_df["offset_minutes"].map(classify_offset_minutes)
    events_df["time_bin"] = events_df["offset_minutes"].map(observation_time_bin)
    events_df["is_outcome_event"] = events_df["offset_minutes"].map(
        lambda value: int(classify_offset_minutes(value) == "outcome")
    )
    return events_df[EVENT_REQUIRED_COLUMNS]


def build_death_events(
    cohort_df: pd.DataFrame,
    *,
    dataset: str,
    source_table: str,
    raw_name: str,
) -> pd.DataFrame:
    """Extract death events from the cohort's death_offset_minutes column.

    Args:
        cohort_df: Cohort DataFrame with death_offset_minutes.
        dataset: Dataset identifier.
        source_table: Source table filename for provenance.
        raw_name: Raw field name for provenance.

    Returns:
        Event DataFrame with death rows.
    """
    death_rows = cohort_df.loc[
        cohort_df["death_offset_minutes"].notna(),
        ["stay_id", "death_offset_minutes"],
    ].copy()
    return build_event_frame(
        dataset=dataset,
        stay_id=death_rows["stay_id"].astype(int),
        event_family="death",
        concept="death",
        source_table=source_table,
        raw_name=raw_name,
        offset_minutes=death_rows["death_offset_minutes"],
        value_numeric=pd.NA,
        value_text="death",
        unit=pd.NA,
        is_intervention=0,
    )


def event_stay_count(events_df: pd.DataFrame) -> int:
    """Return the number of unique stays in an events DataFrame.

    Args:
        events_df: Events DataFrame.

    Returns:
        Unique stay count (0 if empty).
    """
    if events_df.empty:
        return 0
    return int(events_df["stay_id"].nunique())


# ──────────────────────────────────────────────────────────────────────
# Text Normalization
# ──────────────────────────────────────────────────────────────────────

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: object) -> str:
    """Normalize text to lowercase alphanumeric with single spaces.

    Args:
        value: Any value (None/NaN → empty string).

    Returns:
        Normalized, lowercased, whitespace-collapsed string.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    lowered = str(value).strip().lower()
    return _NON_ALNUM_RE.sub(" ", lowered).strip()


def contains_any_token(value: object, tokens: list[str]) -> bool:
    """Check if a value contains any of the given tokens (normalized).

    Args:
        value: Text value to check.
        tokens: List of tokens to match.

    Returns:
        True if any token is found in the normalized value.
    """
    normalized = normalize_text(value)
    return any(normalize_text(token) in normalized for token in tokens)


def series_contains_any(series: pd.Series, tokens: list[str]) -> pd.Series:
    """Boolean mask for rows where text contains any of the given tokens.

    Uses regex OR matching for efficiency on large Series.

    Args:
        series: Text Series to match against.
        tokens: Token list to search for.

    Returns:
        Boolean Series where True means at least one token matched.
    """
    normalized_tokens = [normalize_text(token) for token in tokens]
    pattern = "|".join(re.escape(token) for token in normalized_tokens if token)
    if not pattern:
        return pd.Series(False, index=series.index)
    normalized_series = series.fillna("").map(normalize_text)
    return normalized_series.str.contains(pattern, regex=True, na=False)


# ──────────────────────────────────────────────────────────────────────
# ICD Code Matching
# ──────────────────────────────────────────────────────────────────────


def match_icd_prefix(
    icd_codes: pd.Series,
    icd_versions: pd.Series,
    pattern_map: dict[int, list[str]],
) -> pd.Series:
    """Find rows where ICD codes match version-specific regex patterns.

    Args:
        icd_codes: ICD code strings.
        icd_versions: ICD version integers (9 or 10).
        pattern_map: Mapping from version to list of regex patterns.

    Returns:
        Boolean Series where True means the ICD code matched.
    """
    icd_codes = icd_codes.fillna("").astype(str)
    out = pd.Series(False, index=icd_codes.index)
    for version, patterns in pattern_map.items():
        version_mask = icd_versions == version
        if not patterns:
            continue
        pattern = "|".join(f"(?:{value})" for value in patterns)
        out = out | (version_mask & icd_codes.str.match(pattern, na=False))
    return out


# ──────────────────────────────────────────────────────────────────────
# Time Window Functions
# ──────────────────────────────────────────────────────────────────────


def is_observation_offset_minutes(offset_minutes: float | int | None) -> bool:
    """Check if the offset falls within the observation window [0, 240)."""
    return offset_minutes is not None and 0 <= float(offset_minutes) < LANDMARK_MINUTES


def is_outcome_offset_minutes(offset_minutes: float | int | None) -> bool:
    """Check if the offset falls within the outcome window (240, 1680]."""
    return (
        offset_minutes is not None
        and LANDMARK_MINUTES < float(offset_minutes) <= OUTCOME_WINDOW_END_MINUTES
    )


def is_pre_or_at_landmark(offset_minutes: float | int | None) -> bool:
    """Check if the offset is at or before the landmark [0, 240]."""
    return offset_minutes is not None and 0 <= float(offset_minutes) <= LANDMARK_MINUTES


def classify_offset_minutes(offset_minutes: float | int | None) -> str:
    """Classify an offset in minutes into a time window category.

    Args:
        offset_minutes: Minutes from ICU anchor.

    Returns:
        One of "observation", "landmark", "outcome", or "outside".
    """
    if offset_minutes is None:
        return "outside"
    value = float(offset_minutes)
    if 0 <= value < LANDMARK_MINUTES:
        return "observation"
    if math.isclose(value, LANDMARK_MINUTES):
        return "landmark"
    if LANDMARK_MINUTES < value <= OUTCOME_WINDOW_END_MINUTES:
        return "outcome"
    return "outside"


def observation_time_bin(offset_minutes: float | int | None) -> int | None:
    """Map an observation-window offset to its 15-minute time bin index.

    Args:
        offset_minutes: Minutes from ICU anchor (must be in observation window).

    Returns:
        0-based bin index, or None if outside the observation window.
    """
    if not is_observation_offset_minutes(offset_minutes):
        return None
    max_bin = int(LANDMARK_MINUTES // TIME_STEP_MINUTES) - 1
    return min(int(float(offset_minutes) // TIME_STEP_MINUTES), max_bin)


# ──────────────────────────────────────────────────────────────────────
# Validation Helpers
# ──────────────────────────────────────────────────────────────────────


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise ValueError if required columns are missing.

    Args:
        df: DataFrame to check.
        columns: Required column names.
        name: Human-readable name for error messages.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def ordered_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Reorder DataFrame so required columns appear first.

    Args:
        df: DataFrame to reorder.
        required: Columns that must appear first.

    Returns:
        Reordered DataFrame (extra columns follow required ones).
    """
    extra = [col for col in df.columns if col not in required]
    return df[required + extra]


def sanitize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize events: convert Fahrenheit temperatures to Celsius.

    Temperatures above 50°C are assumed to be in Fahrenheit and are
    converted to Celsius using (F - 32) * 5/9.

    Args:
        events_df: Events DataFrame.

    Returns:
        Copy of events_df with sanitized temperature values.
    """
    events_df = events_df.copy()
    temp_mask = (events_df["concept"] == "temp") & (
        events_df["value_numeric"] > 50
    )
    events_df.loc[temp_mask, "value_numeric"] = (
        events_df.loc[temp_mask, "value_numeric"] - 32.0
    ) * (5.0 / 9.0)
    return events_df
