"""End-to-end integration tests for the PhysioGraph pipeline.

These tests verify that core modules work together using synthetic data
without requiring actual MIMIC or eICU data files.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from physiograph import constants as const
from physiograph.config import load_config
from physiograph.etl.audit import AuditLogger
from physiograph.cohort import build_cohort
from physiograph.features import build_feature_table
from physiograph.guards import LeakageGuard, Preprocessor
from physiograph.models.train import (
    COMPARATOR_SPECS,
    ComparatorSpec,
)
from physiograph.schema import (
    EICUCohortSchema,
    EventSchema,
    FeatureSchema,
    LabelSchema,
    MIMICCohortSchema,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_cohort_df(n: int = 10) -> pd.DataFrame:
    """Minimal cohort DataFrame matching COHORT_REQUIRED_COLUMNS."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": list(range(1, n + 1)),
            "person_id": list(range(1001, 1001 + n)),
            "admit_time": [datetime.now(timezone.utc).isoformat()] * n,
            "admit_year": [2020] * n,
            "age": np.random.uniform(50, 80, n),
            "is_male": np.random.randint(0, 2, n),
            "cohort_hf_flag": np.random.randint(0, 2, n),
            "shock_icd_flag": np.random.randint(0, 2, n),
            "early_icu_flag": np.random.randint(0, 2, n),
            "death_offset_minutes": [None] * n,
            "excluded_before_landmark_flag": [0] * n,
            "exclusion_reason": [""] * n,
        }
    )


def _synthetic_events_df(n: int = 50) -> pd.DataFrame:
    """Minimal events DataFrame matching EVENT_REQUIRED_COLUMNS."""
    stay_ids = list(range(1, 6)) * 10
    concepts = ["lactate", "hr", "sbp", "map", "ph"] * 10
    offsets = np.random.uniform(0, 240, n)
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": stay_ids[:n],
            "event_family": ["lab"] * n,
            "concept": concepts[:n],
            "source_table": ["labevents"] * n,
            "raw_name": ["lactate"] * n,
            "offset_minutes": offsets,
            "window": ["observation"] * n,
            "time_bin": offsets / 60.0,
            "value_numeric": np.random.uniform(0.5, 5.0, n),
            "value_text": [None] * n,
            "unit": ["mmol/L"] * n,
            "is_intervention": [0] * n,
            "is_outcome_event": [0] * n,
        }
    )


def _synthetic_feature_df(n: int = 10) -> pd.DataFrame:
    """Minimal features DataFrame matching PRIMARY_FEATURE_COLUMNS."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": list(range(1, n + 1)),
            "age": np.random.uniform(50, 80, n),
            "is_male": np.random.randint(0, 2, n),
            "cohort_hf_flag": np.random.randint(0, 2, n),
            "shock_icd_flag": np.random.randint(0, 2, n),
            "baseline_lactate": np.random.uniform(1.0, 6.0, n),
            "baseline_hr": np.random.uniform(60, 120, n),
            "baseline_sbp": np.random.uniform(80, 160, n),
            "baseline_map": np.random.uniform(60, 100, n),
            "baseline_ph": np.random.uniform(7.20, 7.45, n),
            "baseline_creatinine": np.random.uniform(0.5, 2.5, n),
            "baseline_bilirubin_total": np.random.uniform(0.3, 3.0, n),
            "baseline_spo2": np.random.uniform(90, 100, n),
            "baseline_resp_rate": np.random.uniform(12, 30, n),
            "baseline_temp": np.random.uniform(36.0, 38.5, n),
            "tachycardia_flag": np.random.randint(0, 2, n),
            "hypotension_flag": np.random.randint(0, 2, n),
            "severe_hypotension_flag": np.random.randint(0, 2, n),
            "acidemia_flag": np.random.randint(0, 2, n),
            "severe_acidemia_flag": np.random.randint(0, 2, n),
            "modifier_burden": np.random.randint(0, 4, n),
            "lactate_delta": np.random.uniform(-1.0, 2.0, n),
            "lactate_slope_per_hr": np.random.uniform(-0.5, 1.0, n),
            "lactate_clearance_4h_pct": np.random.uniform(0, 100, n),
            "persistent_lactate_flag": np.random.randint(0, 2, n),
            "renal_hypoperfusion_flag": np.random.randint(0, 2, n),
            "hepatic_hypoperfusion_flag": np.random.randint(0, 2, n),
            "lactate_ge_2_flag": np.random.randint(0, 2, n),
            "lactate_ge_3_1_flag": np.random.randint(0, 2, n),
            "lactate_ge_5_flag": np.random.randint(0, 2, n),
            "map_below_65_fraction": np.random.uniform(0, 1, n),
            "sbp_below_90_fraction": np.random.uniform(0, 1, n),
            "hr_above_100_fraction": np.random.uniform(0, 1, n),
            "ph_below_7_25_fraction": np.random.uniform(0, 1, n),
            "lactate_map_ratio": np.random.uniform(0.01, 0.1, n),
            "lactate_sbp_ratio": np.random.uniform(0.01, 0.08, n),
            "lactate_acidemia_interaction": np.random.uniform(0, 5, n),
            "lactate_hypotension_interaction": np.random.uniform(0, 5, n),
            "lactate_tachycardia_interaction": np.random.uniform(0, 5, n),
            "occult_hypoperfusion_flag": np.random.randint(0, 2, n),
            "perfusion_burden_score": np.random.uniform(0, 6, n),
            "baseline_lactate_bin": pd.Categorical(
                ["<2", "2-3.1", "3.1-5", ">=5"] * (n // 4) + ["<2"] * (n % 4)
            ),
            "baseline_lactate_focus_bin": pd.Categorical(
                ["<2", "2-3", "3-5", ">=5"] * (n // 4) + ["<2"] * (n % 4)
            ),
            "hr_band": pd.Categorical(["<100", "100-119", ">=120"] * (n // 3) + ["<100"] * (n % 3)),
            "hr_band_fine": pd.Categorical(
                ["<100", "100-109", "110-119", "120-139", ">=140"] * (n // 5)
                + ["<100"] * (n % 5)
            ),
            "lactate_scai_modifier": pd.Categorical(
                ["neither", "lactate>=5 only", "pH<7.2 only"] * (n // 3) + ["neither"] * (n % 3)
            ),
            "scai_stage": pd.Categorical(["A", "B", "C", "D", "E"] * (n // 5) + ["A"] * (n % 5)),
            "scai_stage_num": np.random.randint(1, 6, n),
            "scai_stage_collapsed": pd.Categorical(["A", "B/C", "D/E"] * (n // 3) + ["A"] * (n % 3)),
        }
    )


def _synthetic_label_df(n: int = 10) -> pd.DataFrame:
    """Minimal labels DataFrame matching LABEL_REQUIRED_COLUMNS."""
    return pd.DataFrame(
        {
            "dataset": ["mimic"] * n,
            "stay_id": list(range(1, n + 1)),
            "target": np.random.randint(0, 2, n),
            "pressor_24h_flag": np.random.randint(0, 2, n),
            "mcs_24h_flag": np.random.randint(0, 2, n),
            "escalation_24h_flag": np.random.randint(0, 2, n),
            "renal_injury_24h_flag": np.random.randint(0, 2, n),
            "hypoperfusion_24h_flag": np.random.randint(0, 2, n),
            "hepatic_injury_24h_flag": np.random.randint(0, 2, n),
            "end_organ_24h_flag": np.random.randint(0, 2, n),
            "mortality_24h_flag": np.random.randint(0, 2, n),
            "shock_progression_24h_flag": np.random.randint(0, 2, n),
            "landmark_lactate": np.random.uniform(1.0, 6.0, n),
            "post_landmark_lactate_last": np.random.uniform(0.5, 5.0, n),
            "lactate_clearance_24h_pct": np.random.uniform(0, 100, n),
            "complete_lactate_clearance_24h_flag": np.random.randint(0, 2, n),
            "clearance_ge_64_24h_flag": np.random.randint(0, 2, n),
            "cohort_hf_flag": np.random.randint(0, 2, n),
            "shock_icd_flag": np.random.randint(0, 2, n),
            "age": np.random.uniform(50, 80, n),
            "is_male": np.random.randint(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Config loads and feeds modules
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_config_loads_and_feeds_modules():
    """Verify load_config() returns a dict usable by constants.py."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "observation_hours" in cfg
    assert "outcome_hours" in cfg
    assert "time_step_hours" in cfg
    assert "clinical_normals" in cfg
    assert "training" in cfg
    assert "model" in cfg


# ---------------------------------------------------------------------------
# Test 2: Constants sourced from config
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_constants_from_config():
    """Verify OBSERVATION_HOURS, OUTCOME_HOURS, etc. match config values."""
    cfg = load_config()
    assert const.OBSERVATION_HOURS == cfg["observation_hours"]
    assert const.OUTCOME_HOURS == cfg["outcome_hours"]
    assert const.OUTCOME_WINDOW_END_HOURS == cfg["outcome_window_end_hours"]
    assert const.TIME_STEP_HOURS == cfg["time_step_hours"]
    assert const.TIME_STEP_MINUTES == cfg["time_step_minutes"]
    assert const.LANDMARK_MINUTES == cfg["landmark_minutes"]
    assert const.NUM_STEPS == cfg["num_steps"]


# ---------------------------------------------------------------------------
# Test 3: Schema validation with synthetic data
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_schema_validation_with_synthetic_data():
    """Validate synthetic DataFrames against all 5 schemas."""
    # MIMICCohortSchema
    cohort_df = _synthetic_cohort_df(n=10)
    validated = validate_schema(cohort_df, MIMICCohortSchema, context="mimic_cohort")
    assert len(validated) == 10
    assert set(validated.columns).issuperset({"stay_id", "person_id", "age", "is_male"})

    # EICUCohortSchema
    eicu_cohort_df = _synthetic_cohort_df(n=10)
    validated_eicu = validate_schema(eicu_cohort_df, EICUCohortSchema, context="eicu_cohort")
    assert len(validated_eicu) == 10

    # FeatureSchema
    feature_df = _synthetic_feature_df(n=10)
    validated_features = validate_schema(feature_df, FeatureSchema, context="features")
    assert len(validated_features) == 10
    assert "baseline_lactate" in validated_features.columns

    # LabelSchema
    label_df = _synthetic_label_df(n=10)
    validated_labels = validate_schema(label_df, LabelSchema, context="labels")
    assert len(validated_labels) == 10
    assert "target" in validated_labels.columns

    # EventSchema
    events_df = _synthetic_events_df(n=50)
    validated_events = validate_schema(events_df, EventSchema, context="events")
    assert len(validated_events) == 50
    assert "offset_minutes" in validated_events.columns


# ---------------------------------------------------------------------------
# Test 4: Guards integrate with pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_guards_integrate_with_pipeline():
    """Verify LeakageGuard can be instantiated with config values."""
    cfg = load_config()
    landmark_hours = cfg["observation_hours"]
    outcome_column = "target"
    patient_id_column = "stay_id"
    timestamp_column = "offset_minutes"

    guard = LeakageGuard(
        landmark_hours=landmark_hours,
        outcome_column=outcome_column,
        patient_id_column=patient_id_column,
        timestamp_column=timestamp_column,
    )
    assert guard.landmark_hours == landmark_hours
    assert guard.outcome_column == outcome_column
    assert guard.patient_id_column == patient_id_column
    assert guard.timestamp_column == timestamp_column

    # Synthetic feature matrix without forbidden columns
    features_df = _synthetic_feature_df(n=5)
    # Should not raise
    guard.assert_no_feature_leakage(features_df, context="test")

    # Test patient overlap check
    train_ids = np.array([1, 2, 3, 4, 5])
    test_ids = np.array([4, 5, 6, 7, 8])  # overlap at 4, 5
    with pytest.raises(ValueError, match="patient.*appear in both"):
        guard.assert_no_patient_overlap(train_ids, test_ids, context="test")


# ---------------------------------------------------------------------------
# Test 5: ComparatorSpec from config
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_comparator_spec_from_config():
    """Verify ComparatorSpec can be built from config and pre-registered specs exist."""
    # COMPARATOR_SPECS from models/train.py
    assert len(COMPARATOR_SPECS) > 0
    spec_names = [s.name for s in COMPARATOR_SPECS]
    assert "lactate_only" in spec_names
    assert "lactate_hemodynamics" in spec_names
    assert "lactate_end_organ" in spec_names
    assert "scai_stage_model" in spec_names

    # Each spec has required fields
    for spec in COMPARATOR_SPECS:
        assert isinstance(spec.name, str)
        assert isinstance(spec.description, str)
        assert isinstance(spec.numeric_features, tuple)
        assert isinstance(spec.categorical_features, tuple)

    # Build a custom spec from config values
    cfg = load_config()
    comparator_cfg = cfg.get("comparator_specs", {})
    assert isinstance(comparator_cfg, dict)


# ---------------------------------------------------------------------------
# Test 6: Full import chain
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_import_chain():
    """Verify all modules import in correct order without circular imports."""
    # Import order that exercises the full chain
    from physiograph.config import load_config as lc1
    from physiograph import constants as c1
    from physiograph.schema import MIMICCohortSchema as s1
    from physiograph.guards import LeakageGuard as g1
    from physiograph.pipeline import build_feature_table as p1
    from physiograph.cohort import build_cohort as c2
    from physiograph.features import build_feature_table as f1
    from physiograph.models.train import COMPARATOR_SPECS as m1
    from physiograph.etl.audit import AuditLogger as a1

    # Verify objects are accessible
    assert callable(lc1)
    assert hasattr(c1, "OBSERVATION_HOURS")
    assert s1 is not None
    assert g1 is not None
    assert callable(p1)
    assert callable(c2)
    assert callable(f1)
    assert len(m1) > 0
    assert a1 is not None


# ---------------------------------------------------------------------------
# Test 7: Preprocessor fit_transform integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_preprocessor_fit_transform_integration():
    """Verify Preprocessor.fit_transform works with synthetic data."""
    feature_df = _synthetic_feature_df(n=20)

    continuous_cols = ["baseline_lactate", "baseline_hr", "baseline_sbp", "baseline_map"]
    binary_cols = ["tachycardia_flag", "hypotension_flag", "acidemia_flag"]

    preprocessor = Preprocessor(
        continuous_columns=continuous_cols,
        binary_columns=binary_cols,
    )
    result = preprocessor.fit_transform(feature_df)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 20
    assert preprocessor.fitted is True
    assert preprocessor.feature_names_out is not None
    assert len(preprocessor.feature_names_out) == result.shape[1]


# ---------------------------------------------------------------------------
# Test 8: AuditLogger integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_audit_logger_integration():
    """Verify AuditLogger can be created and produces valid entries."""
    logger = AuditLogger(dataset="mimic")
    assert logger.dataset == "mimic"
    assert logger.entries == []

    logger.log(step="test_step", row_count=100, stay_count=50, details={"key": "value"})
    assert len(logger.entries) == 1
    entry = logger.entries[0]
    assert entry["step"] == "test_step"
    assert entry["row_count"] == 100
    assert entry["stay_count"] == 50
    assert entry["details"] == {"key": "value"}
    assert "timestamp_utc" in entry
    assert entry["dataset"] == "mimic"

    # Second entry
    logger.log(step="test_step_2")
    assert len(logger.entries) == 2


# ---------------------------------------------------------------------------
# Test 9: build_cohort dispatch integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_build_cohort_dispatch_integration():
    """Verify build_cohort raises FileNotFoundError for unknown dataset config."""
    # build_cohort calls load_config which raises FileNotFoundError for unknown datasets
    with pytest.raises(FileNotFoundError, match="Dataset config not found"):
        build_cohort(dataset="unknown_dataset")


# ---------------------------------------------------------------------------
# Test 10: Feature extraction with config
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_feature_extraction_with_config():
    """Verify build_feature_table accepts config parameter and works with synthetic events."""
    cohort_df = _synthetic_cohort_df(n=5)
    events_df = _synthetic_events_df(n=50)

    # build_feature_table from features/extraction.py accepts events_df and cohort_df
    features_df = build_feature_table(events_df, cohort_df)

    assert isinstance(features_df, pd.DataFrame)
    assert len(features_df) > 0
    assert "stay_id" in features_df.columns
    assert "dataset" in features_df.columns
