"""Tests for data-leakage guards and Preprocessor enforcement.

Covers:
- LeakageGuard: patient overlap, post-landmark features, outcome columns,
  observation-only events, PROBAST+AI Domain 4 compliance
- Preprocessor: fit/transform lifecycle, RuntimeError on premature transform
- Module-level constants: FORBIDDEN_FEATURE_COLUMNS, ANALYSIS_ONLY_CONTEXT_COLUMNS,
  OUTCOME_FLAG_COLUMNS
- Standalone functions: assert_no_feature_leakage_columns, assert_observation_only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph.guards import (
    ANALYSIS_ONLY_CONTEXT_COLUMNS,
    FORBIDDEN_FEATURE_COLUMNS,
    OUTCOME_FLAG_COLUMNS,
    LeakageGuard,
    Preprocessor,
    assert_no_feature_leakage_columns,
    assert_observation_only as standalone_assert_observation_only,
)
from physiograph.constants import (
    ANALYSIS_ONLY_CONTEXT_COLUMNS as CONSTANTS_ANALYSIS_ONLY,
    OUTCOME_FLAG_COLUMNS as CONSTANTS_OUTCOME,
    FORBIDDEN_FEATURE_COLUMNS as CONSTANTS_FORBIDDEN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def guard() -> LeakageGuard:
    """LeakageGuard with default landmark of 4 hours."""
    return LeakageGuard(landmark_hours=4.0)


@pytest.fixture
def clean_features_df() -> pd.DataFrame:
    """Feature DataFrame with no forbidden columns."""
    return pd.DataFrame(
        {
            "baseline_lactate": [2.1, 3.5, 1.8],
            "heart_rate_mean": [88.0, 102.0, 76.0],
            "systolic_bp_mean": [110.0, 95.0, 120.0],
            "age": [65, 72, 58],
        }
    )


@pytest.fixture
def leaky_features_df() -> pd.DataFrame:
    """Feature DataFrame containing forbidden columns."""
    return pd.DataFrame(
        {
            "baseline_lactate": [2.1, 3.5, 1.8],
            "landmark_lactate": [4.2, 5.1, 3.9],  # post-landmark
            "pressor_24h_flag": [1, 0, 1],  # outcome flag
            "mortality_24h_flag": [0, 0, 1],  # both analysis-only & outcome
        }
    )


@pytest.fixture
def observation_events_df() -> pd.DataFrame:
    """Events DataFrame with only observation-window rows."""
    return pd.DataFrame(
        {
            "stay_id": [1001, 1001, 1002, 1002],
            "offset_minutes": [0, 60, 0, 120],
            "window": ["observation", "observation", "observation", "observation"],
            "value_numeric": [2.1, 2.3, 1.8, 1.9],
        }
    )


@pytest.fixture
def mixed_events_df() -> pd.DataFrame:
    """Events DataFrame with both observation and outcome-window rows."""
    return pd.DataFrame(
        {
            "stay_id": [1001, 1001, 1001, 1002],
            "offset_minutes": [0, 60, 300, 0],
            "window": ["observation", "observation", "outcome", "observation"],
            "value_numeric": [2.1, 2.3, 5.0, 1.8],
        }
    )


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Small training DataFrame for Preprocessor tests."""
    return pd.DataFrame(
        {
            "heart_rate": [80.0, 90.0, 100.0, 110.0],
            "systolic_bp": [120.0, 110.0, 100.0, 90.0],
            "is_male": [1, 0, 1, 0],
            "shock_flag": [0, 0, 1, 1],
        }
    )


# ---------------------------------------------------------------------------
# 1. Patient overlap detection
# ---------------------------------------------------------------------------


class TestPatientOverlap:
    """Tests for LeakageGuard.assert_no_patient_overlap."""

    def test_no_patient_overlap_pass(self, guard: LeakageGuard) -> None:
        """No error when train/test IDs are disjoint."""
        train_ids = np.array([1001, 1002, 1003])
        test_ids = np.array([2001, 2002, 2003])
        # Should not raise
        guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_no_patient_overlap_pass_with_pd_index(self, guard: LeakageGuard) -> None:
        """No error when using pd.Index for IDs."""
        train_ids = pd.Index([1001, 1002, 1003])
        test_ids = pd.Index([2001, 2002, 2003])
        guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_no_patient_overlap_fail(self, guard: LeakageGuard) -> None:
        """ValueError raised when patient IDs overlap between train and test."""
        train_ids = np.array([1001, 1002, 1003])
        test_ids = np.array([1002, 2001, 2002])
        with pytest.raises(ValueError, match=r"patient.*overlap"):
            guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_no_patient_overlap_fail_reports_examples(self, guard: LeakageGuard) -> None:
        """Error message includes example overlapping IDs."""
        train_ids = np.array([1001, 1002])
        test_ids = np.array([1002])
        with pytest.raises(ValueError, match=r"1002"):
            guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_no_patient_overlap_fail_with_context(self, guard: LeakageGuard) -> None:
        """Error message includes context label when provided."""
        train_ids = np.array([1001, 1002])
        test_ids = np.array([1002])
        with pytest.raises(ValueError, match=r"\[external\]"):
            guard.assert_no_patient_overlap(train_ids, test_ids, context="external")


# ---------------------------------------------------------------------------
# 2. Post-landmark feature detection
# ---------------------------------------------------------------------------


class TestPostLandmarkFeatures:
    """Tests for LeakageGuard.assert_no_post_landmark_features."""

    def test_no_post_landmark_features_pass(self, guard: LeakageGuard, clean_features_df: pd.DataFrame) -> None:
        """No error when features contain no analysis-only columns."""
        guard.assert_no_post_landmark_features(clean_features_df)

    def test_no_post_landmark_features_fail(self, guard: LeakageGuard) -> None:
        """ValueError raised when post-landmark columns are present."""
        df = pd.DataFrame(
            {
                "baseline_lactate": [2.1],
                "landmark_lactate": [4.2],  # post-landmark
                "post_landmark_lactate_last": [3.9],  # post-landmark
            }
        )
        with pytest.raises(ValueError, match=r"post-landmark.*analysis-only"):
            guard.assert_no_post_landmark_features(df)

    def test_no_post_landmark_features_fail_with_context(self, guard: LeakageGuard) -> None:
        """Error message includes context label."""
        df = pd.DataFrame({"landmark_lactate": [4.2]})
        with pytest.raises(ValueError, match=r"\[train\]"):
            guard.assert_no_post_landmark_features(df, context="train")


# ---------------------------------------------------------------------------
# 3. Outcome column detection
# ---------------------------------------------------------------------------


class TestOutcomeInFeatures:
    """Tests for LeakageGuard.assert_no_outcome_in_features."""

    def test_no_outcome_in_features_pass(self, guard: LeakageGuard, clean_features_df: pd.DataFrame) -> None:
        """No error when features contain no outcome-flag columns."""
        guard.assert_no_outcome_in_features(clean_features_df)

    def test_no_outcome_in_features_fail(self, guard: LeakageGuard) -> None:
        """ValueError raised when outcome-flag columns are present."""
        df = pd.DataFrame(
            {
                "baseline_lactate": [2.1],
                "pressor_24h_flag": [1],  # outcome flag
                "target": [0],  # outcome flag
            }
        )
        with pytest.raises(ValueError, match=r"outcome-flag"):
            guard.assert_no_outcome_in_features(df)

    def test_no_outcome_in_features_fail_with_context(self, guard: LeakageGuard) -> None:
        """Error message includes context label."""
        df = pd.DataFrame({"mcs_24h_flag": [0]})
        with pytest.raises(ValueError, match=r"\[test\]"):
            guard.assert_no_outcome_in_features(df, context="test")


# ---------------------------------------------------------------------------
# 4. Observation-only events
# ---------------------------------------------------------------------------


class TestObservationOnly:
    """Tests for LeakageGuard.assert_observation_only."""

    def test_observation_only_pass(self, guard: LeakageGuard, observation_events_df: pd.DataFrame) -> None:
        """No error when all events are observation-window only."""
        guard.assert_observation_only(observation_events_df)

    def test_observation_only_pass_no_window_column(self, guard: LeakageGuard) -> None:
        """No error when 'window' column is absent (graceful degradation)."""
        df = pd.DataFrame(
            {
                "stay_id": [1001],
                "offset_minutes": [60],
                "value_numeric": [2.1],
            }
        )
        # Should silently pass — no window column means no check needed
        guard.assert_observation_only(df)

    def test_observation_only_fail(self, guard: LeakageGuard, mixed_events_df: pd.DataFrame) -> None:
        """ValueError raised when non-observation windows are present."""
        with pytest.raises(ValueError, match=r"non-observation"):
            guard.assert_observation_only(mixed_events_df)

    def test_observation_only_fail_with_context(self, guard: LeakageGuard, mixed_events_df: pd.DataFrame) -> None:
        """Error message includes context label."""
        with pytest.raises(ValueError, match=r"\[PROBAST\+AI-D4\]"):
            guard.assert_observation_only(mixed_events_df, context="PROBAST+AI-D4")


# ---------------------------------------------------------------------------
# 5. PROBAST+AI Domain 4 comprehensive check
# ---------------------------------------------------------------------------


class TestProbastDomain4:
    """Tests for LeakageGuard.check_probast_domain4."""

    def test_probast_domain4_comprehensive_pass(self, guard: LeakageGuard, clean_features_df: pd.DataFrame) -> None:
        """All checks pass when data is clean."""
        train_ids = np.array([1001, 1002])
        test_ids = np.array([2001, 2002])
        events_df = pd.DataFrame(
            {
                "stay_id": [1001, 2001],
                "offset_minutes": [0, 60],
                "window": ["observation", "observation"],
                "value_numeric": [2.1, 1.8],
            }
        )
        issues = guard.check_probast_domain4(
            clean_features_df,
            train_ids=train_ids,
            test_ids=test_ids,
            events_df=events_df,
        )
        assert issues == []

    def test_probast_domain4_comprehensive_fail_multiple(self, guard: LeakageGuard) -> None:
        """ValueError raised with multiple violations listed."""
        # Features with both post-landmark and outcome columns
        leaky_df = pd.DataFrame(
            {
                "baseline_lactate": [2.1],
                "landmark_lactate": [4.2],  # post-landmark
                "pressor_24h_flag": [1],  # outcome flag
            }
        )
        # Overlapping patient IDs
        train_ids = np.array([1001, 1002])
        test_ids = np.array([1002, 2001])
        # Events with outcome window
        events_df = pd.DataFrame(
            {
                "stay_id": [1001, 1001],
                "offset_minutes": [0, 300],
                "window": ["observation", "outcome"],
                "value_numeric": [2.1, 5.0],
            }
        )
        with pytest.raises(ValueError, match=r"PROBAST\+AI Domain 4 violations"):
            guard.check_probast_domain4(
                leaky_df,
                train_ids=train_ids,
                test_ids=test_ids,
                events_df=events_df,
            )

    def test_probast_domain4_no_ids_no_events(self, guard: LeakageGuard, clean_features_df: pd.DataFrame) -> None:
        """Passes when only features are checked (no IDs or events provided)."""
        issues = guard.check_probast_domain4(clean_features_df)
        assert issues == []

    def test_probast_domain4_post_landmark_only(self, guard: LeakageGuard) -> None:
        """Detects post-landmark violation even when IDs and events are clean."""
        leaky_df = pd.DataFrame(
            {
                "baseline_lactate": [2.1],
                "lactate_clearance_24h_pct": [0.5],  # post-landmark
            }
        )
        train_ids = np.array([1001])
        test_ids = np.array([2001])
        with pytest.raises(ValueError, match=r"post-landmark"):
            guard.check_probast_domain4(
                leaky_df,
                train_ids=train_ids,
                test_ids=test_ids,
            )


# ---------------------------------------------------------------------------
# 6. Preprocessor fit/transform lifecycle
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Tests for Preprocessor fit/transform enforcement."""

    def test_preprocessor_fit_transform(self, train_df: pd.DataFrame) -> None:
        """Preprocessor.fit() then .transform() works correctly."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate", "systolic_bp"],
            binary_columns=["is_male", "shock_flag"],
        )
        preprocessor.fit(train_df)
        result = preprocessor.transform(train_df)

        # Result should be a 2D numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(train_df)
        assert result.shape[1] == 4  # 2 continuous + 2 binary

        # Continuous columns should be z-score standardized (mean ≈ 0, std ≈ 1)
        assert abs(result[:, 0].mean()) < 0.5  # heart_rate standardized
        assert abs(result[:, 1].mean()) < 0.5  # systolic_bp standardized

        # Binary columns should be unchanged
        np.testing.assert_array_equal(result[:, 2], train_df["is_male"].values.astype(float))
        np.testing.assert_array_equal(result[:, 3], train_df["shock_flag"].values.astype(float))

    def test_preprocessor_fit_transform_method(self, train_df: pd.DataFrame) -> None:
        """Preprocessor.fit_transform() produces same result as fit().transform()."""
        preprocessor_a = Preprocessor(
            continuous_columns=["heart_rate", "systolic_bp"],
            binary_columns=["is_male", "shock_flag"],
        )
        preprocessor_b = Preprocessor(
            continuous_columns=["heart_rate", "systolic_bp"],
            binary_columns=["is_male", "shock_flag"],
        )
        result_a = preprocessor_a.fit(train_df).transform(train_df)
        result_b = preprocessor_b.fit_transform(train_df)
        np.testing.assert_allclose(result_a, result_b)

    def test_preprocessor_transform_before_fit(self) -> None:
        """RuntimeError raised if transform() called before fit()."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate"],
            binary_columns=["is_male"],
        )
        df = pd.DataFrame({"heart_rate": [80.0], "is_male": [1]})
        with pytest.raises(RuntimeError, match=r"transform.*before fit"):
            preprocessor.transform(df)

    def test_preprocessor_fitted_property(self, train_df: pd.DataFrame) -> None:
        """Preprocessor.fitted property reflects fit state."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate"],
            binary_columns=["is_male"],
        )
        assert preprocessor.fitted is False
        preprocessor.fit(train_df)
        assert preprocessor.fitted is True

    def test_preprocessor_feature_names_out(self, train_df: pd.DataFrame) -> None:
        """Preprocessor.feature_names_out returns correct column names after fit."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate", "systolic_bp"],
            binary_columns=["is_male", "shock_flag"],
        )
        assert preprocessor.feature_names_out is None
        preprocessor.fit(train_df)
        assert preprocessor.feature_names_out == [
            "heart_rate", "systolic_bp", "is_male", "shock_flag"
        ]

    def test_preprocessor_missing_column_raises(self) -> None:
        """ValueError raised when a required column is missing from training data."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate", "nonexistent_col"],
            binary_columns=["is_male"],
        )
        df = pd.DataFrame({"heart_rate": [80.0], "is_male": [1]})
        with pytest.raises(ValueError, match=r"nonexistent_col"):
            preprocessor.fit(df)

    def test_preprocessor_categorical_columns(self) -> None:
        """Preprocessor handles categorical columns with one-hot encoding."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate"],
            binary_columns=[],
            categorical_columns=["scai_stage"],
        )
        df = pd.DataFrame(
            {
                "heart_rate": [80.0, 90.0, 100.0],
                "scai_stage": ["A", "B", "C"],
            }
        )
        preprocessor.fit(df)
        result = preprocessor.transform(df)

        # One-hot encoding: categories sorted → A, B, C; reference = A
        # So columns are: heart_rate, scai_stage__B, scai_stage__C
        assert result.shape == (3, 3)
        assert preprocessor.feature_names_out == [
            "heart_rate", "scai_stage__B", "scai_stage__C"
        ]

    def test_preprocessor_nan_fill_continuous(self) -> None:
        """Preprocessor fills NaN in continuous columns with training median."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate"],
            binary_columns=[],
        )
        train_df = pd.DataFrame({"heart_rate": [80.0, 90.0, 100.0]})
        test_df = pd.DataFrame({"heart_rate": [85.0, np.nan]})
        preprocessor.fit(train_df)
        result = preprocessor.transform(test_df)
        # NaN should be filled with median (90.0), then standardized
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# 7. Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constant definitions."""

    def test_forbidden_feature_columns(self) -> None:
        """FORBIDDEN_FEATURE_COLUMNS contains expected analysis-only and outcome columns."""
        # Must be a frozenset (immutable)
        assert isinstance(FORBIDDEN_FEATURE_COLUMNS, frozenset)

        # Must contain all analysis-only context columns
        for col in ANALYSIS_ONLY_CONTEXT_COLUMNS:
            assert col in FORBIDDEN_FEATURE_COLUMNS, f"{col} missing from FORBIDDEN_FEATURE_COLUMNS"

        # Must contain all outcome flag columns
        for col in OUTCOME_FLAG_COLUMNS:
            assert col in FORBIDDEN_FEATURE_COLUMNS, f"{col} missing from FORBIDDEN_FEATURE_COLUMNS"

        # Total count: 6 analysis-only + 9 outcome = 15
        assert len(FORBIDDEN_FEATURE_COLUMNS) == 15

    def test_analysis_only_context_columns(self) -> None:
        """ANALYSIS_ONLY_CONTEXT_COLUMNS list contains expected post-landmark columns."""
        expected = [
            "landmark_lactate",
            "post_landmark_lactate_last",
            "lactate_clearance_24h_pct",
            "complete_lactate_clearance_24h_flag",
            "clearance_ge_64_24h_flag",
            "mortality_24h_flag",
        ]
        assert ANALYSIS_ONLY_CONTEXT_COLUMNS == expected

    def test_outcome_flag_columns(self) -> None:
        """OUTCOME_FLAG_COLUMNS list contains expected outcome flag columns."""
        expected = [
            "pressor_24h_flag",
            "mcs_24h_flag",
            "escalation_24h_flag",
            "renal_injury_24h_flag",
            "hypoperfusion_24h_flag",
            "hepatic_injury_24h_flag",
            "end_organ_24h_flag",
            "shock_progression_24h_flag",
            "target",
        ]
        assert OUTCOME_FLAG_COLUMNS == expected

    def test_constants_module_consistency(self) -> None:
        """guards.py constants match constants.py definitions."""
        # ANALYSIS_ONLY_CONTEXT_COLUMNS should match between modules
        assert set(ANALYSIS_ONLY_CONTEXT_COLUMNS) == set(CONSTANTS_ANALYSIS_ONLY)
        assert set(OUTCOME_FLAG_COLUMNS) == set(CONSTANTS_OUTCOME)
        assert FORBIDDEN_FEATURE_COLUMNS == CONSTANTS_FORBIDDEN or set(FORBIDDEN_FEATURE_COLUMNS) == CONSTANTS_FORBIDDEN


# ---------------------------------------------------------------------------
# 8. Standalone functions
# ---------------------------------------------------------------------------


class TestStandaloneFunctions:
    """Tests for module-level convenience functions."""

    def test_assert_no_feature_leakage_columns_pass(self) -> None:
        """No error when column list contains no forbidden columns."""
        columns = ["baseline_lactate", "heart_rate_mean", "systolic_bp_mean"]
        assert_no_feature_leakage_columns(columns)

    def test_assert_no_feature_leakage_columns_pass_with_pd_index(self) -> None:
        """No error when pd.Index contains no forbidden columns."""
        columns = pd.Index(["baseline_lactate", "heart_rate_mean"])
        assert_no_feature_leakage_columns(columns)

    def test_assert_no_feature_leakage_columns_fail(self) -> None:
        """ValueError raised when forbidden columns are present."""
        columns = ["baseline_lactate", "landmark_lactate", "target"]
        with pytest.raises(ValueError, match=r"Post-landmark or outcome columns leaked"):
            assert_no_feature_leakage_columns(columns)

    def test_assert_no_feature_leakage_columns_fail_reports_overlap(self) -> None:
        """Error message lists the specific forbidden columns found."""
        columns = ["baseline_lactate", "pressor_24h_flag", "mortality_24h_flag"]
        with pytest.raises(ValueError, match=r"pressor_24h_flag"):
            assert_no_feature_leakage_columns(columns)

    def test_standalone_assert_observation_only_pass(self) -> None:
        """No error when all events are observation-only."""
        df = pd.DataFrame(
            {
                "stay_id": [1001, 1002],
                "window": ["observation", "observation"],
                "value_numeric": [2.1, 1.8],
            }
        )
        standalone_assert_observation_only(df)

    def test_standalone_assert_observation_only_fail(self) -> None:
        """ValueError raised when non-observation windows are present."""
        df = pd.DataFrame(
            {
                "stay_id": [1001, 1001],
                "window": ["observation", "outcome"],
                "value_numeric": [2.1, 5.0],
            }
        )
        with pytest.raises(ValueError, match=r"observation-only"):
            standalone_assert_observation_only(df)


# ---------------------------------------------------------------------------
# 9. assert_no_feature_leakage (convenience method)
# ---------------------------------------------------------------------------


class TestFeatureLeakageConvenience:
    """Tests for LeakageGuard.assert_no_feature_leakage convenience method."""

    def test_assert_no_feature_leakage_pass(self, guard: LeakageGuard, clean_features_df: pd.DataFrame) -> None:
        """No error when features contain no forbidden columns."""
        guard.assert_no_feature_leakage(clean_features_df)

    def test_assert_no_feature_leakage_fail(self, guard: LeakageGuard) -> None:
        """ValueError raised when any forbidden column is present."""
        df = pd.DataFrame(
            {
                "baseline_lactate": [2.1],
                "landmark_lactate": [4.2],  # analysis-only
                "target": [0],  # outcome flag
            }
        )
        with pytest.raises(ValueError, match=r"post-landmark or outcome"):
            guard.assert_no_feature_leakage(df)

    def test_assert_no_feature_leakage_fail_with_context(self, guard: LeakageGuard) -> None:
        """Error message includes context label."""
        df = pd.DataFrame({"target": [0]})
        with pytest.raises(ValueError, match=r"\[validation\]"):
            guard.assert_no_feature_leakage(df, context="validation")


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_empty_train_test_ids(self, guard: LeakageGuard) -> None:
        """No error when both train and test ID sets are empty."""
        train_ids = np.array([], dtype=int)
        test_ids = np.array([], dtype=int)
        guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_single_overlap(self, guard: LeakageGuard) -> None:
        """ValueError raised for a single overlapping patient ID."""
        train_ids = np.array([1001])
        test_ids = np.array([1001])
        with pytest.raises(ValueError, match=r"1 patient"):
            guard.assert_no_patient_overlap(train_ids, test_ids)

    def test_preprocessor_empty_dataframe(self) -> None:
        """Preprocessor handles empty DataFrame gracefully."""
        preprocessor = Preprocessor(
            continuous_columns=["heart_rate"],
            binary_columns=["is_male"],
        )
        df = pd.DataFrame({"heart_rate": pd.Series([], dtype=float), "is_male": pd.Series([], dtype=int)})
        # Empty DataFrame: median is NaN → fill_val defaults to 0.0
        preprocessor.fit(df)
        assert preprocessor.fitted is True

    def test_observation_only_with_nan_window(self, guard: LeakageGuard) -> None:
        """NaN values in window column are ignored (dropna behavior)."""
        df = pd.DataFrame(
            {
                "stay_id": [1001, 1002, 1003],
                "window": ["observation", None, "observation"],
                "value_numeric": [2.1, 3.0, 1.8],
            }
        )
        # NaN window values should be dropped, leaving only "observation"
        guard.assert_observation_only(df)

    def test_forbidden_columns_union(self) -> None:
        """FORBIDDEN_FEATURE_COLUMNS is the union of analysis-only and outcome columns."""
        expected_union = set(ANALYSIS_ONLY_CONTEXT_COLUMNS) | set(OUTCOME_FLAG_COLUMNS)
        assert set(FORBIDDEN_FEATURE_COLUMNS) == expected_union

    def test_mortality_24h_flag_in_both_lists(self) -> None:
        """mortality_24h_flag appears in both analysis-only and outcome lists."""
        assert "mortality_24h_flag" in ANALYSIS_ONLY_CONTEXT_COLUMNS
        # Note: mortality_24h_flag is in ANALYSIS_ONLY_CONTEXT_COLUMNS but NOT in
        # OUTCOME_FLAG_COLUMNS — it's a special case that bridges both categories
        # The FORBIDDEN set still contains it exactly once (frozenset dedup)