"""Schema contract tests for PhysioGraph pandera schemas.

Validates that all 5 DataFrame schemas (MIMICCohortSchema, EICUCohortSchema,
FeatureSchema, LabelSchema, EventSchema) accept valid data and reject invalid
data, and that the validate_schema() helper correctly labels errors.
"""

from __future__ import annotations

import pandas as pd
import pytest
import pandera as pa

from physiograph.schema import (
    MIMICCohortSchema,
    EICUCohortSchema,
    FeatureSchema,
    LabelSchema,
    EventSchema,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic DataFrames matching each schema
# ---------------------------------------------------------------------------


def _make_cohort_df(n: int = 3) -> pd.DataFrame:
    """Return a valid MIMIC/eICU cohort DataFrame with *n* rows."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": range(100, 100 + n),
            "person_id": range(200, 200 + n),
            "admit_time": ["2020-01-01 00:00:00"] * n,
            "admit_year": [2020] * n,
            "age": [65.0, 72.0, 55.0],
            "is_male": [1, 0, 1],
            "cohort_hf_flag": [1, 0, 1],
            "shock_icd_flag": [0, 1, 0],
            "early_icu_flag": [1, 1, 0],
            "death_offset_minutes": [None, 1440.0, None],
            "excluded_before_landmark_flag": [0, 0, 0],
            "exclusion_reason": [None, None, None],
        }
    )


def _make_feature_df(n: int = 3) -> pd.DataFrame:
    """Return a valid feature DataFrame with *n* rows (all 50 columns)."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": range(100, 100 + n),
            "age": [65.0, 72.0, 55.0],
            "is_male": [1, 0, 1],
            "cohort_hf_flag": [1, 0, 1],
            "shock_icd_flag": [0, 1, 0],
            # Baseline vitals / labs
            "baseline_lactate": [2.1, 4.5, 1.8],
            "baseline_hr": [90.0, 110.0, 75.0],
            "baseline_sbp": [120.0, 85.0, 130.0],
            "baseline_map": [80.0, 60.0, 90.0],
            "baseline_ph": [7.35, 7.20, 7.40],
            "baseline_creatinine": [1.2, 2.5, 0.9],
            "baseline_bilirubin_total": [0.8, 2.1, 0.5],
            "baseline_spo2": [96.0, 92.0, 98.0],
            "baseline_resp_rate": [18.0, 24.0, 16.0],
            "baseline_temp": [36.8, 37.5, 36.5],
            # Binary flags
            "tachycardia_flag": [0, 1, 0],
            "hypotension_flag": [0, 1, 0],
            "severe_hypotension_flag": [0, 0, 0],
            "acidemia_flag": [0, 1, 0],
            "severe_acidemia_flag": [0, 0, 0],
            "modifier_burden": [0, 2, 1],
            # Lactate dynamics
            "lactate_delta": [0.5, -1.2, 0.3],
            "lactate_slope_per_hr": [0.1, -0.3, 0.05],
            "lactate_clearance_4h_pct": [10.0, 25.0, 5.0],
            "persistent_lactate_flag": [0, 1, 0],
            "renal_hypoperfusion_flag": [0, 1, 0],
            "hepatic_hypoperfusion_flag": [0, 0, 0],
            "lactate_ge_2_flag": [1, 1, 0],
            "lactate_ge_3_1_flag": [0, 1, 0],
            "lactate_ge_5_flag": [0, 0, 0],
            # Time-below thresholds
            "map_below_65_fraction": [0.0, 0.5, 0.0],
            "sbp_below_90_fraction": [0.0, 0.3, 0.0],
            "hr_above_100_fraction": [0.0, 0.6, 0.0],
            "ph_below_7_25_fraction": [0.0, 0.4, 0.0],
            # Interaction features
            "lactate_map_ratio": [0.03, 0.07, 0.02],
            "lactate_sbp_ratio": [0.02, 0.05, 0.01],
            "lactate_acidemia_interaction": [0.0, 4.5, 0.0],
            "lactate_hypotension_interaction": [0.0, 4.5, 0.0],
            "lactate_tachycardia_interaction": [0.0, 4.5, 0.0],
            # Composite scores
            "occult_hypoperfusion_flag": [0, 1, 0],
            "perfusion_burden_score": [0.0, 1.5, 0.2],
            # Categorical bins
            "baseline_lactate_bin": ["normal", "elevated", "normal"],
            "baseline_lactate_focus_bin": ["low", "high", "low"],
            "hr_band": ["normal", "tachycardic", "normal"],
            "hr_band_fine": ["60-100", "100-120", "60-100"],
            "lactate_scai_modifier": ["none", "persistent", "none"],
            "scai_stage": ["A", "C", "B"],
            "scai_stage_num": [1, 3, 2],
            "scai_stage_collapsed": ["A-B", "C-D-E", "A-B"],
        }
    )


def _make_label_df(n: int = 3) -> pd.DataFrame:
    """Return a valid label DataFrame with *n* rows (all 21 columns)."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": range(100, 100 + n),
            "target": [0, 1, 0],
            "pressor_24h_flag": [0, 1, 0],
            "mcs_24h_flag": [0, 0, 0],
            "escalation_24h_flag": [0, 1, 0],
            "renal_injury_24h_flag": [0, 1, 0],
            "hypoperfusion_24h_flag": [0, 1, 0],
            "hepatic_injury_24h_flag": [0, 0, 0],
            "end_organ_24h_flag": [0, 1, 0],
            "mortality_24h_flag": [0, 0, 0],
            "shock_progression_24h_flag": [0, 1, 0],
            "landmark_lactate": [1.8, 4.2, 1.5],
            "post_landmark_lactate_last": [1.5, 3.0, 1.2],
            "lactate_clearance_24h_pct": [16.7, 28.6, 20.0],
            "complete_lactate_clearance_24h_flag": [0, 0, 0],
            "clearance_ge_64_24h_flag": [0, 0, 0],
            "cohort_hf_flag": [1, 0, 1],
            "shock_icd_flag": [0, 1, 0],
            "age": [65.0, 72.0, 55.0],
            "is_male": [1, 0, 1],
        }
    )


def _make_event_df(n: int = 3) -> pd.DataFrame:
    """Return a valid event DataFrame with *n* rows (all 14 columns)."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": range(100, 100 + n),
            "event_family": ["intervention", "outcome", "intervention"],
            "concept": ["vasopressor", "death", "vasopressor"],
            "source_table": ["inputevents", "patients", "inputevents"],
            "raw_name": ["Norepinephrine", None, "Norepinephrine"],
            "offset_minutes": [30.0, 1440.0, 60.0],
            "window": ["observation", "outcome", "observation"],
            "time_bin": [0.5, 24.0, 1.0],
            "value_numeric": [0.5, None, 0.3],
            "value_text": [None, "expired", None],
            "unit": ["mcg/kg/min", None, "mcg/kg/min"],
            "is_intervention": [1, 0, 1],
            "is_outcome_event": [0, 1, 0],
        }
    )


# ===================================================================
# MIMICCohortSchema tests
# ===================================================================


class TestMIMICCohortSchema:
    """Contract tests for MIMICCohortSchema."""

    def test_mimic_cohort_schema_valid(self):
        """MIMICCohortSchema accepts a valid cohort DataFrame."""
        df = _make_cohort_df()
        result = MIMICCohortSchema.validate(df)
        assert len(result) == 3
        assert list(result.columns) == list(MIMICCohortSchema.columns.keys())

    def test_mimic_cohort_schema_invalid_wrong_types(self):
        """MIMICCohortSchema rejects data with wrong column types."""
        df = _make_cohort_df()
        # Replace int column with non-numeric strings
        df["stay_id"] = ["not", "an", "int"]
        with pytest.raises(pa.errors.SchemaErrors):
            MIMICCohortSchema.validate(df)

    def test_mimic_cohort_schema_invalid_missing_columns(self):
        """MIMICCohortSchema rejects data with missing required columns."""
        df = _make_cohort_df()
        df = df.drop(columns=["stay_id", "age"])
        with pytest.raises(pa.errors.SchemaError):
            MIMICCohortSchema.validate(df)

    def test_mimic_cohort_schema_invalid_flag_values(self):
        """MIMICCohortSchema rejects data with flag values outside {0, 1}."""
        df = _make_cohort_df()
        df["is_male"] = [2, 3, 4]  # not in {0, 1}
        with pytest.raises(pa.errors.SchemaError):
            MIMICCohortSchema.validate(df)


# ===================================================================
# EICUCohortSchema tests
# ===================================================================


class TestEICUCohortSchema:
    """Contract tests for EICUCohortSchema."""

    def test_eicu_cohort_schema_valid(self):
        """EICUCohortSchema accepts a valid cohort DataFrame."""
        df = _make_cohort_df()
        # eICU uses float ages (e.g., 89.5)
        df["dataset"] = ["eicu"] * len(df)
        df["age"] = [65.7, 72.3, 55.1]
        result = EICUCohortSchema.validate(df)
        assert len(result) == 3
        assert list(result.columns) == list(EICUCohortSchema.columns.keys())

    def test_eicu_cohort_schema_invalid_wrong_types(self):
        """EICUCohortSchema rejects data with wrong column types."""
        df = _make_cohort_df()
        df["dataset"] = ["eicu"] * len(df)
        df["admit_year"] = ["twenty-twenty", "twenty-twenty", "twenty-twenty"]
        with pytest.raises(pa.errors.SchemaErrors):
            EICUCohortSchema.validate(df)

    def test_eicu_cohort_schema_invalid_missing_columns(self):
        """EICUCohortSchema rejects data with missing required columns."""
        df = _make_cohort_df()
        df["dataset"] = ["eicu"] * len(df)
        df = df.drop(columns=["person_id", "excluded_before_landmark_flag"])
        with pytest.raises(pa.errors.SchemaError):
            EICUCohortSchema.validate(df)

    def test_eicu_cohort_schema_invalid_flag_values(self):
        """EICUCohortSchema rejects data with flag values outside {0, 1}."""
        df = _make_cohort_df()
        df["dataset"] = ["eicu"] * len(df)
        df["cohort_hf_flag"] = [5, 6, 7]  # not in {0, 1}
        with pytest.raises(pa.errors.SchemaError):
            EICUCohortSchema.validate(df)


# ===================================================================
# FeatureSchema tests
# ===================================================================


class TestFeatureSchema:
    """Contract tests for FeatureSchema (50 columns)."""

    def test_feature_schema_valid(self):
        """FeatureSchema accepts a valid 50-column feature DataFrame."""
        df = _make_feature_df()
        result = FeatureSchema.validate(df)
        assert len(result) == 3
        assert len(result.columns) == 50

    def test_feature_schema_invalid_wrong_columns(self):
        """FeatureSchema rejects data with missing or extra columns."""
        df = _make_feature_df()
        # Drop a required column
        df = df.drop(columns=["baseline_lactate"])
        with pytest.raises(pa.errors.SchemaError):
            FeatureSchema.validate(df)

    def test_feature_schema_invalid_flag_values(self):
        """FeatureSchema rejects data with binary flag values outside {0, 1}."""
        df = _make_feature_df()
        df["tachycardia_flag"] = [2, 3, 4]  # not in {0, 1}
        with pytest.raises(pa.errors.SchemaError):
            FeatureSchema.validate(df)

    def test_feature_schema_invalid_fraction_range(self):
        """FeatureSchema rejects data with fraction values outside [0, 1]."""
        df = _make_feature_df()
        df["map_below_65_fraction"] = [1.5, 2.0, -0.1]  # outside [0, 1]
        with pytest.raises(pa.errors.SchemaError):
            FeatureSchema.validate(df)

    def test_feature_schema_invalid_negative_vitals(self):
        """FeatureSchema rejects data with negative vital values (ge(0) check)."""
        df = _make_feature_df()
        df["baseline_hr"] = [-10.0, -5.0, -1.0]  # violates Check.ge(0)
        with pytest.raises(pa.errors.SchemaError):
            FeatureSchema.validate(df)


# ===================================================================
# LabelSchema tests
# ===================================================================


class TestLabelSchema:
    """Contract tests for LabelSchema (21 columns)."""

    def test_label_schema_valid(self):
        """LabelSchema accepts a valid label DataFrame."""
        df = _make_label_df()
        result = LabelSchema.validate(df)
        assert len(result) == 3
        assert len(result.columns) == 21

    def test_label_schema_invalid_wrong_columns(self):
        """LabelSchema rejects data with missing columns."""
        df = _make_label_df()
        df = df.drop(columns=["target", "age"])
        with pytest.raises(pa.errors.SchemaError):
            LabelSchema.validate(df)

    def test_label_schema_invalid_flag_values(self):
        """LabelSchema rejects data with target values outside {0, 1}."""
        df = _make_label_df()
        df["target"] = [2, 3, 4]  # not in {0, 1}
        with pytest.raises(pa.errors.SchemaError):
            LabelSchema.validate(df)

    def test_label_schema_invalid_negative_lactate(self):
        """LabelSchema rejects data with negative landmark_lactate (ge(0) check)."""
        df = _make_label_df()
        df["landmark_lactate"] = [-1.0, -2.0, -0.5]  # violates Check.ge(0)
        with pytest.raises(pa.errors.SchemaError):
            LabelSchema.validate(df)


# ===================================================================
# EventSchema tests
# ===================================================================


class TestEventSchema:
    """Contract tests for EventSchema (14 columns, long-format events)."""

    def test_event_schema_valid(self):
        """EventSchema accepts a valid long-format event DataFrame."""
        df = _make_event_df()
        result = EventSchema.validate(df)
        assert len(result) == 3
        assert len(result.columns) == 14

    def test_event_schema_invalid_wrong_columns(self):
        """EventSchema rejects data with missing columns."""
        df = _make_event_df()
        df = df.drop(columns=["concept", "offset_minutes"])
        with pytest.raises(pa.errors.SchemaError):
            EventSchema.validate(df)

    def test_event_schema_invalid_negative_offset(self):
        """EventSchema rejects data with negative offset_minutes (ge(0) check)."""
        df = _make_event_df()
        df["offset_minutes"] = [-10.0, -5.0, -1.0]  # violates Check.ge(0)
        with pytest.raises(pa.errors.SchemaError):
            EventSchema.validate(df)

    def test_event_schema_invalid_flag_values(self):
        """EventSchema rejects data with is_intervention values outside {0, 1}."""
        df = _make_event_df()
        df["is_intervention"] = [2, 3, 4]  # not in {0, 1}
        with pytest.raises(pa.errors.SchemaError):
            EventSchema.validate(df)


# ===================================================================
# validate_schema() helper tests
# ===================================================================


class TestValidateSchemaHelper:
    """Tests for the validate_schema() convenience function."""

    def test_validate_schema_passes_valid_data(self):
        """validate_schema returns validated DataFrame on success."""
        df = _make_cohort_df()
        result = validate_schema(df, MIMICCohortSchema, context="test")
        assert len(result) == 3

    def test_validate_schema_catches_and_labels_errors(self):
        """validate_schema re-raises SchemaErrors with context label."""
        df = _make_cohort_df()
        df["stay_id"] = ["not", "an", "int"]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_schema(df, MIMICCohortSchema, context="mimic_cohort_csv")

    def test_validate_schema_without_context(self):
        """validate_schema works without a context label."""
        df = _make_cohort_df()
        df["stay_id"] = ["not", "an", "int"]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_schema(df, MIMICCohortSchema)


# ===================================================================
# Schema coercion tests
# ===================================================================


class TestSchemaCoercion:
    """Tests for coerce=True behavior on schemas."""

    @pytest.mark.slow
    def test_coercion_int_age_to_float(self):
        """coerce=True converts int age values to float (MIMIC→float)."""
        df = _make_cohort_df()
        # Provide int ages — coerce should cast to float
        df["age"] = [65, 72, 55]  # int values, schema expects float
        result = MIMICCohortSchema.validate(df)
        # After coercion, age column should be float dtype
        assert result["age"].dtype in ("float64", "float32")

    @pytest.mark.slow
    def test_coercion_string_to_int_stay_id(self):
        """coerce=True converts numeric strings to int for stay_id."""
        df = _make_cohort_df()
        # Provide string stay_ids that are numeric — coerce should cast to int
        df["stay_id"] = ["100", "101", "102"]
        result = MIMICCohortSchema.validate(df)
        # After coercion, stay_id should be int dtype
        assert pd.api.types.is_integer_dtype(result["stay_id"])

    @pytest.mark.slow
    def test_coercion_eicu_float_age(self):
        """coerce=True handles float ages correctly for eICU schema."""
        df = _make_cohort_df()
        df["dataset"] = ["eicu"] * len(df)
        df["age"] = [65.7, 72.3, 55.1]
        result = EICUCohortSchema.validate(df)
        assert result["age"].dtype in ("float64", "float32")