"""Parity tests for ETL pipeline shared utilities and audit logging.

Verifies that shared utility functions, constants, audit logging, and
data structures produce correct outputs with synthetic test data. No
actual MIMIC/eICU CSV files are required.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from physiograph.etl.audit import AuditLogger
from physiograph.etl.shared import (
    COHORT_REQUIRED_COLUMNS,
    EVENT_REQUIRED_COLUMNS,
    LANDMARK_MINUTES,
    OUTCOME_WINDOW_END_MINUTES,
    OBSERVATION_HOURS,
    OUTCOME_HOURS,
    TIME_STEP_MINUTES,
    SourceExtraction,
    build_event_frame,
    classify_offset_minutes,
    contains_any_token,
    match_icd_prefix,
    normalize_text,
    observation_time_bin,
    require_columns,
    sanitize_events,
    series_contains_any,
)


# ──────────────────────────────────────────────────────────────────────
# 1. CHUNK_SIZE parity
# ──────────────────────────────────────────────────────────────────────


def test_chunk_size():
    """Verify CHUNK_SIZE is 250,000 in both extractors."""
    from physiograph.etl.mimic_extractor import CHUNK_SIZE as MIMIC_CHUNK
    from physiograph.etl.eicu_extractor import CHUNK_SIZE as EICU_CHUNK

    assert MIMIC_CHUNK == 250_000, f"MIMIC CHUNK_SIZE={MIMIC_CHUNK}, expected 250000"
    assert EICU_CHUNK == 250_000, f"eICU CHUNK_SIZE={EICU_CHUNK}, expected 250000"
    assert MIMIC_CHUNK == EICU_CHUNK, "MIMIC and eICU CHUNK_SIZE must match"


# ──────────────────────────────────────────────────────────────────────
# 2. normalize_text
# ──────────────────────────────────────────────────────────────────────


class TestNormalizeText:
    """Tests for normalize_text utility."""

    def test_lowercase(self):
        assert normalize_text("Heart Failure") == "heart failure"

    def test_strip_whitespace(self):
        assert normalize_text("  lactate  ") == "lactate"

    def test_special_characters_removed(self):
        assert normalize_text("BUN/Creatinine") == "bun creatinine"

    def test_none_returns_empty(self):
        assert normalize_text(None) == ""

    def test_nan_returns_empty(self):
        assert normalize_text(float("nan")) == ""

    def test_multiple_spaces_collapsed(self):
        assert normalize_text("a   b   c") == "a b c"

    def test_numbers_preserved(self):
        assert normalize_text("ICD 9 Code") == "icd 9 code"


# ──────────────────────────────────────────────────────────────────────
# 3. match_icd_prefix
# ──────────────────────────────────────────────────────────────────────


class TestMatchIcdPrefix:
    """Tests for ICD-9/10 prefix matching."""

    def test_icd9_match(self):
        codes = pd.Series(["4280", "4289", "I509", "39891"])
        versions = pd.Series([9, 9, 10, 9])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True, False]

    def test_icd10_match(self):
        codes = pd.Series(["I509", "I500", "4280"])
        versions = pd.Series([10, 10, 9])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True]

    def test_no_match(self):
        codes = pd.Series(["A00", "B99"])
        versions = pd.Series([10, 10])
        pattern_map = {9: [r"^428"], 10: [r"^I50"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [False, False]

    def test_nan_codes(self):
        codes = pd.Series([None, "4280"])
        versions = pd.Series([9, 9])
        pattern_map = {9: [r"^428"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [False, True]

    def test_multiple_patterns_per_version(self):
        codes = pd.Series(["4280", "78551", "I509", "R570"])
        versions = pd.Series([9, 9, 10, 10])
        pattern_map = {9: [r"^428", r"^78551"], 10: [r"^I50", r"^R570"]}
        result = match_icd_prefix(codes, versions, pattern_map)
        assert result.tolist() == [True, True, True, True]


# ──────────────────────────────────────────────────────────────────────
# 4. classify_offset_minutes
# ──────────────────────────────────────────────────────────────────────


class TestClassifyOffsetMinutes:
    """Tests for time window classification."""

    def test_observation_window(self):
        assert classify_offset_minutes(0) == "observation"
        assert classify_offset_minutes(120) == "observation"
        assert classify_offset_minutes(239) == "observation"

    def test_landmark(self):
        assert classify_offset_minutes(240) == "landmark"

    def test_outcome_window(self):
        assert classify_offset_minutes(241) == "outcome"
        assert classify_offset_minutes(1000) == "outcome"
        assert classify_offset_minutes(1440) == "outcome"

    def test_outside_window(self):
        assert classify_offset_minutes(-1) == "outside"
        assert classify_offset_minutes(1681) == "outside"
        assert classify_offset_minutes(None) == "outside"

    def test_boundary_values(self):
        # Exactly 0 is observation
        assert classify_offset_minutes(0) == "observation"
        # Exactly 240 is landmark (not observation)
        assert classify_offset_minutes(240) == "landmark"
        # Exactly 1680 is outcome (inclusive end of window)
        assert classify_offset_minutes(1680) == "outcome"


# ──────────────────────────────────────────────────────────────────────
# 5. observation_time_bin
# ──────────────────────────────────────────────────────────────────────


class TestObservationTimeBin:
    """Tests for 15-minute time binning within observation window."""

    def test_bin_zero(self):
        assert observation_time_bin(0) == 0

    def test_bin_mid_window(self):
        # 120 minutes / 15 = bin 8
        assert observation_time_bin(120) == 8

    def test_bin_near_landmark(self):
        # 225 minutes / 15 = bin 15
        assert observation_time_bin(225) == 15

    def test_outside_observation_returns_none(self):
        assert observation_time_bin(240) is None
        assert observation_time_bin(300) is None
        assert observation_time_bin(-1) is None
        assert observation_time_bin(None) is None

    def test_max_bin_clamp(self):
        # 239 minutes / 15 = 15.93 → floor = 15, but max_bin = 15
        assert observation_time_bin(239) == 15

    def test_bin_count(self):
        # 4 hours = 240 minutes, 15-min steps → 16 bins (0..15)
        assert observation_time_bin(0) == 0
        assert observation_time_bin(239) == 15


# ──────────────────────────────────────────────────────────────────────
# 6. build_event_frame
# ──────────────────────────────────────────────────────────────────────


class TestBuildEventFrame:
    """Tests for event frame construction."""

    def test_basic_construction(self):
        df = build_event_frame(
            dataset="mimic",
            stay_id=pd.Series([1001, 1002]),
            event_family="vital",
            concept="hr",
            source_table="chartevents.csv",
            raw_name="Heart Rate",
            offset_minutes=pd.Series([10.0, 20.0]),
            value_numeric=pd.Series([80.0, 90.0]),
            value_text=pd.NA,
            unit="bpm",
            is_intervention=0,
        )
        assert len(df) == 2
        assert set(df.columns) == set(EVENT_REQUIRED_COLUMNS)
        assert df["dataset"].tolist() == ["mimic", "mimic"]
        assert df["event_family"].tolist() == ["vital", "vital"]
        assert df["concept"].tolist() == ["hr", "hr"]
        assert df["is_intervention"].tolist() == [0, 0]
        assert df["is_outcome_event"].tolist() == [0, 0]

    def test_string_concept(self):
        df = build_event_frame(
            dataset="eicu",
            stay_id=pd.Series([2001]),
            event_family="death",
            concept="death",
            source_table="patient.csv",
            raw_name="discharge_status",
            offset_minutes=pd.Series([500.0]),
            value_numeric=pd.NA,
            value_text="death",
            unit=pd.NA,
            is_intervention=0,
        )
        assert len(df) == 1
        assert df["concept"].iloc[0] == "death"

    def test_stay_id_numeric_coercion(self):
        df = build_event_frame(
            dataset="mimic",
            stay_id=pd.Series(["1001", "1002"]),
            event_family="lab",
            concept="lactate",
            source_table="labevents.csv",
            raw_name="Lactate",
            offset_minutes=pd.Series([5.0, 10.0]),
        )
        assert df["stay_id"].dtype == "Int64"
        assert df["stay_id"].iloc[0] == 1001


# ──────────────────────────────────────────────────────────────────────
# 7. AuditLogger
# ──────────────────────────────────────────────────────────────────────


class TestAuditLogger:
    """Tests for audit logging format and structure."""

    def test_entry_format(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_one", row_count=100, stay_count=50)
        assert len(logger.entries) == 1
        entry = logger.entries[0]
        assert "timestamp_utc" in entry
        assert entry["dataset"] == "mimic"
        assert entry["step"] == "step_one"
        assert entry["row_count"] == 100
        assert entry["stay_count"] == 50

    def test_optional_fields(self):
        logger = AuditLogger(dataset="eicu")
        logger.log("step_two", row_count=200)
        entry = logger.entries[0]
        assert "stay_count" not in entry
        assert entry["row_count"] == 200

    def test_details_field(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_three", row_count=50, details={"max_stays": 10})
        entry = logger.entries[0]
        assert entry["details"] == {"max_stays": 10}

    def test_multiple_entries(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_a", row_count=10)
        logger.log("step_b", row_count=20)
        assert len(logger.entries) == 2
        assert logger.entries[0]["step"] == "step_a"
        assert logger.entries[1]["step"] == "step_b"

    def test_timestamp_is_iso_format(self):
        logger = AuditLogger(dataset="mimic")
        logger.log("step_ts", row_count=1)
        ts = logger.entries[0]["timestamp_utc"]
        # ISO format should contain 'T' separator
        assert "T" in ts


# ──────────────────────────────────────────────────────────────────────
# 8. require_columns
# ──────────────────────────────────────────────────────────────────────


class TestRequireColumns:
    """Tests for column requirement assertion."""

    def test_passes_when_all_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        require_columns(df, ["a", "b"], "test_df")

    def test_raises_on_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            require_columns(df, ["a", "b", "c"], "test_df")

    def test_error_message_contains_name(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="my_table"):
            require_columns(df, ["missing_col"], "my_table")


# ──────────────────────────────────────────────────────────────────────
# 9. sanitize_events (Fahrenheit → Celsius)
# ──────────────────────────────────────────────────────────────────────


class TestSanitizeEvents:
    """Tests for Fahrenheit-to-Celsius temperature conversion."""

    def _make_events_df(self, concepts, values):
        return pd.DataFrame(
            {
                "dataset": "eicu",
                "stay_id": [1] * len(concepts),
                "event_family": "vital",
                "concept": concepts,
                "source_table": "vitalPeriodic.csv",
                "raw_name": "temperature",
                "offset_minutes": [10.0] * len(concepts),
                "window": "",
                "time_bin": pd.NA,
                "value_numeric": values,
                "value_text": pd.NA,
                "unit": pd.NA,
                "is_intervention": 0,
                "is_outcome_event": 0,
            }
        )

    def test_fahrenheit_to_celsius(self):
        # 98.6°F = 37.0°C
        df = self._make_events_df(["temp"], [98.6])
        result = sanitize_events(df)
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.1)

    def test_celsius_preserved(self):
        # 37.0°C should stay 37.0 (≤ 50 threshold)
        df = self._make_events_df(["temp"], [37.0])
        result = sanitize_events(df)
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.01)

    def test_non_temp_unaffected(self):
        # HR values should never be converted
        df = self._make_events_df(["hr"], [80.0])
        result = sanitize_events(df)
        assert result["value_numeric"].iloc[0] == 80.0

    def test_mixed_concepts(self):
        df = self._make_events_df(
            ["temp", "hr", "temp"],
            [98.6, 80.0, 104.0],
        )
        result = sanitize_events(df)
        # 98.6°F → ~37°C, 80.0 HR unchanged, 104°F → ~40°C
        assert math.isclose(result["value_numeric"].iloc[0], 37.0, abs_tol=0.1)
        assert result["value_numeric"].iloc[1] == 80.0
        assert math.isclose(result["value_numeric"].iloc[2], 40.0, abs_tol=0.1)


# ──────────────────────────────────────────────────────────────────────
# 10. contains_any_token
# ──────────────────────────────────────────────────────────────────────


class TestContainsAnyToken:
    """Tests for token matching in eICU diagnosis strings."""

    def test_single_token_match(self):
        assert contains_any_token("congestive heart failure", ["heart failure"]) is True

    def test_no_token_match(self):
        assert contains_any_token("pneumonia", ["heart failure"]) is False

    def test_multiple_tokens_any_match(self):
        assert (
            contains_any_token(
                "acute myocardial infarction",
                ["heart failure", "myocardial infarction"],
            )
            is True
        )

    def test_normalized_matching(self):
        # normalize_text lowercases and strips special chars
        assert contains_any_token("Cardiogenic Shock!", ["cardiogenic shock"]) is True

    def test_empty_value(self):
        assert contains_any_token("", ["heart failure"]) is False

    def test_empty_tokens(self):
        assert contains_any_token("heart failure", []) is False

    def test_series_contains_any(self):
        series = pd.Series(
            ["congestive heart failure", "pneumonia", "cardiogenic shock"]
        )
        result = series_contains_any(series, ["heart failure", "cardiogenic shock"])
        assert result.tolist() == [True, False, True]


# ──────────────────────────────────────────────────────────────────────
# 11. SourceExtraction dataclass
# ──────────────────────────────────────────────────────────────────────


class TestSourceExtraction:
    """Tests for SourceExtraction dataclass fields."""

    def test_has_expected_fields(self):
        field_names = {f.name for f in fields(SourceExtraction)}
        assert field_names == {"cohort_df", "events_df"}

    def test_instantiation(self):
        cohort = pd.DataFrame({"stay_id": [1, 2]})
        events = pd.DataFrame({"stay_id": [1], "concept": ["hr"]})
        extraction = SourceExtraction(cohort_df=cohort, events_df=events)
        assert len(extraction.cohort_df) == 2
        assert len(extraction.events_df) == 1


# ──────────────────────────────────────────────────────────────────────
# 12. COHORT_REQUIRED_COLUMNS
# ──────────────────────────────────────────────────────────────────────


class TestCohortRequiredColumns:
    """Tests for COHORT_REQUIRED_COLUMNS list."""

    def test_contains_required_fields(self):
        required = [
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
        for col in required:
            assert col in COHORT_REQUIRED_COLUMNS, f"Missing: {col}"

    def test_length(self):
        assert len(COHORT_REQUIRED_COLUMNS) == 13

    def test_event_required_columns(self):
        assert "stay_id" in EVENT_REQUIRED_COLUMNS
        assert "offset_minutes" in EVENT_REQUIRED_COLUMNS
        assert "concept" in EVENT_REQUIRED_COLUMNS
        assert "window" in EVENT_REQUIRED_COLUMNS
        assert "time_bin" in EVENT_REQUIRED_COLUMNS


# ──────────────────────────────────────────────────────────────────────
# Time window constants parity
# ──────────────────────────────────────────────────────────────────────


class TestTimeWindowConstants:
    """Verify time window constants match specification."""

    def test_observation_hours(self):
        assert OBSERVATION_HOURS == 4.0

    def test_outcome_hours(self):
        assert OUTCOME_HOURS == 24.0

    def test_landmark_minutes(self):
        assert LANDMARK_MINUTES == 240

    def test_outcome_window_end_minutes(self):
        assert OUTCOME_WINDOW_END_MINUTES == 1680

    def test_time_step_minutes(self):
        assert TIME_STEP_MINUTES == 15