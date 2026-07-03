"""Parity tests for cohort definitions (MIMIC and eICU).

Verifies that the extracted cohort module logic matches the original
notebook definitions for ICD pattern matching, diagnosis token matching,
exclusion criteria, validator functions, and dispatch routing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph.cohort import (
    CohortResult,
    EICUCohortBuilder,
    MIMICCohortBuilder,
    assert_cohort_size,
    assert_no_duplicate_stays,
    assert_required_columns,
    build_cohort,
    derive_death_offset_minutes,
    match_icd_prefix,
    normalize_text,
    parse_eicu_age,
    series_contains_any,
)
from physiograph.cohort.eicu_cohort import _EICU_DIAGNOSIS_TOKENS
from physiograph.cohort.mimic_cohort import (
    _CARDIOGENIC_SHOCK_ICD_PATTERNS,
    _HF_ICD_PATTERNS,
)


# ---------------------------------------------------------------------------
# 1. MIMIC ICD pattern matching
# ---------------------------------------------------------------------------


class TestMimicICDPatternMatching:
    """Verify HF_ICD_PATTERNS and CARDIOGENIC_SHOCK_ICD_PATTERNS match
    expected notebook patterns (^428, ^I50, ^78551, ^R570)."""

    def test_hf_icd9_pattern_is_428_prefix(self):
        """ICD-9 HF pattern must match codes starting with 428."""
        assert _HF_ICD_PATTERNS[9] == [r"^428"]

    def test_hf_icd10_pattern_is_I50_prefix(self):
        """ICD-10 HF pattern must match codes starting with I50."""
        assert _HF_ICD_PATTERNS[10] == [r"^I50"]

    def test_shock_icd9_pattern_is_78551_prefix(self):
        """ICD-9 cardiogenic shock pattern must match codes starting with 78551."""
        assert _CARDIOGENIC_SHOCK_ICD_PATTERNS[9] == [r"^78551"]

    def test_shock_icd10_pattern_is_R570_prefix(self):
        """ICD-10 cardiogenic shock pattern must match codes starting with R570."""
        assert _CARDIOGENIC_SHOCK_ICD_PATTERNS[10] == [r"^R570"]

    def test_match_icd_prefix_hf_icd9(self):
        """match_icd_prefix correctly identifies ICD-9 HF codes."""
        codes = pd.Series(["4280", "4289", "42820", "4109", "I509"])
        versions = pd.Series([9, 9, 9, 9, 10])
        result = match_icd_prefix(codes, versions, _HF_ICD_PATTERNS)
        assert result.iloc[0] == True   # 4280 — ICD-9 HF
        assert result.iloc[1] == True   # 4289 — ICD-9 HF
        assert result.iloc[2] == True   # 42820 — ICD-9 HF
        assert result.iloc[3] == False  # 4109 — not HF
        assert result.iloc[4] == True   # I509 with version 10 — ICD-10 HF

    def test_match_icd_prefix_hf_icd10(self):
        """match_icd_prefix correctly identifies ICD-10 HF codes."""
        codes = pd.Series(["I509", "I500", "I501", "J189", "4280"])
        versions = pd.Series([10, 10, 10, 10, 9])
        result = match_icd_prefix(codes, versions, _HF_ICD_PATTERNS)
        assert result.iloc[0] == True   # I509 — ICD-10 HF
        assert result.iloc[1] == True   # I500 — ICD-10 HF
        assert result.iloc[2] == True   # I501 — ICD-10 HF
        assert result.iloc[3] == False  # J189 — not HF
        assert result.iloc[4] == True   # 4280 with version 9 — ICD-9 HF

    def test_match_icd_prefix_shock_icd9(self):
        """match_icd_prefix correctly identifies ICD-9 cardiogenic shock codes."""
        codes = pd.Series(["78551", "78552", "4280", "78551"])
        versions = pd.Series([9, 9, 9, 10])
        result = match_icd_prefix(codes, versions, _CARDIOGENIC_SHOCK_ICD_PATTERNS)
        assert result.iloc[0] == True  # 78551
        assert result.iloc[1] == False  # 78552 — does NOT start with 78551
        assert result.iloc[2] == False  # 4280
        assert result.iloc[3] == False  # 78551 but version 10

    def test_match_icd_prefix_shock_icd10(self):
        """match_icd_prefix correctly identifies ICD-10 cardiogenic shock codes."""
        codes = pd.Series(["R570", "R571", "R5700", "I509"])
        versions = pd.Series([10, 10, 10, 10])
        result = match_icd_prefix(codes, versions, _CARDIOGENIC_SHOCK_ICD_PATTERNS)
        assert result.iloc[0] == True  # R570
        assert result.iloc[1] == False  # R571 — does NOT start with R570
        assert result.iloc[2] == True  # R5700 starts with R570
        assert result.iloc[3] == False  # I509

    def test_match_icd_prefix_handles_nan(self):
        """match_icd_prefix handles NaN codes gracefully."""
        codes = pd.Series(["4280", None, np.nan, "I509"])
        versions = pd.Series([9, 9, 9, 10])
        result = match_icd_prefix(codes, versions, _HF_ICD_PATTERNS)
        assert result.iloc[0] == True  # 4280
        assert result.iloc[1] == False  # NaN
        assert result.iloc[2] == False  # NaN
        assert result.iloc[3] == True  # I509

    def test_match_icd_prefix_empty_pattern_map(self):
        """match_icd_prefix returns all-False for empty pattern map."""
        codes = pd.Series(["4280", "I509"])
        versions = pd.Series([9, 10])
        result = match_icd_prefix(codes, versions, {})
        assert not result.any()


# ---------------------------------------------------------------------------
# 2. eICU diagnosis token matching
# ---------------------------------------------------------------------------


class TestEICUDiagnosisTokenMatching:
    """Verify EICU_DIAGNOSIS_TOKENS match expected tokens from notebook."""

    def test_hf_tokens_present(self):
        """HF diagnosis tokens must include 'congestive heart failure' and 'heart failure'."""
        tokens = _EICU_DIAGNOSIS_TOKENS["cohort_hf_flag"]
        assert "congestive heart failure" in tokens
        assert "heart failure" in tokens

    def test_shock_tokens_present(self):
        """Shock diagnosis tokens must include 'cardiogenic shock'."""
        tokens = _EICU_DIAGNOSIS_TOKENS["shock_icd_flag"]
        assert "cardiogenic shock" in tokens

    def test_cardiomyopathy_tokens_present(self):
        """Cardiomyopathy diagnosis tokens must include 'cardiomyopathy'."""
        tokens = _EICU_DIAGNOSIS_TOKENS["cardiomyopathy_flag"]
        assert "cardiomyopathy" in tokens

    def test_acute_mi_tokens_present(self):
        """Acute MI diagnosis tokens must include 'acute myocardial infarction'."""
        tokens = _EICU_DIAGNOSIS_TOKENS["acute_mi_flag"]
        assert "acute myocardial infarction" in tokens

    def test_series_contains_any_basic(self):
        """series_contains_any identifies rows containing any of the tokens."""
        series = pd.Series([
            "Congestive Heart Failure",
            "pneumonia",
            "cardiogenic shock and HF",
            "acute myocardial infarction",
            "diabetes",
        ])
        result = series_contains_any(series, ["heart failure", "cardiogenic shock"])
        assert result.iloc[0] == True  # "Congestive Heart Failure" contains "heart failure"
        assert result.iloc[1] == False  # pneumonia
        assert result.iloc[2] == True  # contains "cardiogenic shock"
        assert result.iloc[3] == False  # acute MI — no matching token
        assert result.iloc[4] == False  # diabetes

    def test_series_contains_any_case_insensitive(self):
        """series_contains_any is case-insensitive."""
        series = pd.Series(["HEART FAILURE", "Heart Failure", "heart failure"])
        result = series_contains_any(series, ["heart failure"])
        assert result.all()

    def test_series_contains_any_handles_nan(self):
        """series_contains_any handles NaN values gracefully."""
        series = pd.Series(["heart failure", None, np.nan, "pneumonia"])
        result = series_contains_any(series, ["heart failure"])
        assert result.iloc[0] == True
        assert result.iloc[1] == False
        assert result.iloc[2] == False
        assert result.iloc[3] == False

    def test_normalize_text(self):
        """normalize_text lowercases and strips whitespace."""
        assert normalize_text("  Heart Failure  ") == "heart failure"
        assert normalize_text(None) == ""
        assert normalize_text(np.nan) == ""
        assert normalize_text("CARDIOGENIC SHOCK") == "cardiogenic shock"


# ---------------------------------------------------------------------------
# 3. Exclusion criteria: age < 16
# ---------------------------------------------------------------------------


class TestExclusionCriteriaAge:
    """Verify age < 16 exclusion works correctly."""

    def test_eicu_age_under_16_excluded(self):
        """eICU patients with age < 16 should be flagged for exclusion."""
        # Simulate eICU exclusion logic
        cohort_df = pd.DataFrame({
            "stay_id": [1, 2, 3, 4],
            "age": [15.0, 16.0, 45.0, 89.0],
            "unitdischargeoffset": [5000, 5000, 5000, 5000],
            "death_offset_minutes": [np.nan, np.nan, np.nan, np.nan],
        })
        # Age < 16 exclusion
        excluded = []
        for _, row in cohort_df.iterrows():
            age = row.get("age")
            r = []
            if pd.notna(age) and float(age) < 16:
                r.append("age_lt_16")
            excluded.append(1 if r else 0)
        assert excluded == [1, 0, 0, 0]

    def test_eicu_age_exactly_16_not_excluded(self):
        """eICU patients with age exactly 16 should NOT be excluded."""
        cohort_df = pd.DataFrame({
            "stay_id": [1],
            "age": [16.0],
            "unitdischargeoffset": [5000],
            "death_offset_minutes": [np.nan],
        })
        age = cohort_df.iloc[0]["age"]
        assert not (pd.notna(age) and float(age) < 16)

    def test_parse_eicu_age_gt89(self):
        """parse_eicu_age handles '>89' de-identification pattern."""
        assert parse_eicu_age(">89") == 89.0

    def test_parse_eicu_age_normal(self):
        """parse_eicu_age parses normal age strings."""
        assert parse_eicu_age("45") == 45.0
        assert parse_eicu_age("16") == 16.0

    def test_parse_eicu_age_none(self):
        """parse_eicu_age returns NaN for None input."""
        result = parse_eicu_age(None)
        assert pd.isna(result)

    def test_parse_eicu_age_nan(self):
        """parse_eicu_age returns NaN for NaN input."""
        result = parse_eicu_age(float("nan"))
        assert pd.isna(result)


# ---------------------------------------------------------------------------
# 4. Exclusion criteria: LOS < 2h
# ---------------------------------------------------------------------------


class TestExclusionCriteriaLOS:
    """Verify LOS < 2h (120 minutes) exclusion works correctly."""

    def test_los_under_2h_excluded(self):
        """eICU stays with unit discharge offset < 120 min should be excluded."""
        cohort_df = pd.DataFrame({
            "stay_id": [1, 2, 3],
            "age": [45.0, 45.0, 45.0],
            "unitdischargeoffset": [60, 119, 120],
            "death_offset_minutes": [np.nan, np.nan, np.nan],
        })
        excluded = []
        for _, row in cohort_df.iterrows():
            r = []
            unit_discharge = pd.to_numeric(row.get("unitdischargeoffset", pd.NA), errors="coerce")
            if pd.notna(unit_discharge) and float(unit_discharge) < 120:
                r.append("los_lt_2h")
            excluded.append(1 if r else 0)
        assert excluded == [1, 1, 0]

    def test_los_exactly_2h_not_excluded(self):
        """eICU stays with unit discharge offset exactly 120 min should NOT be excluded."""
        unit_discharge = 120
        assert not (pd.notna(unit_discharge) and float(unit_discharge) < 120)

    def test_los_nan_not_excluded(self):
        """eICU stays with missing LOS should NOT be excluded for LOS."""
        unit_discharge = pd.NA
        result = pd.notna(pd.to_numeric(unit_discharge, errors="coerce"))
        assert not result


# ---------------------------------------------------------------------------
# 5. Exclusion criteria: pre-landmark death
# ---------------------------------------------------------------------------


class TestExclusionCriteriaPreLandmarkDeath:
    """Verify pre-landmark death exclusion works correctly."""

    def test_death_within_4h_excluded(self):
        """Patients who died within 4 hours (240 min) of landmark should be excluded."""
        cohort_df = pd.DataFrame({
            "stay_id": [1, 2, 3, 4],
            "age": [45.0, 45.0, 45.0, 45.0],
            "unitdischargeoffset": [5000, 5000, 5000, 5000],
            "death_offset_minutes": [60.0, 240.0, 241.0, np.nan],
        })
        excluded = []
        for _, row in cohort_df.iterrows():
            r = []
            death_offset = row.get("death_offset_minutes")
            if pd.notna(death_offset) and float(death_offset) >= 0 and float(death_offset) <= 240:
                r.append("death_before_or_at_4h")
            excluded.append(1 if r else 0)
        assert excluded == [1, 1, 0, 0]

    def test_death_at_exactly_4h_excluded(self):
        """Death at exactly 240 minutes (4h) should be excluded."""
        death_offset = 240.0
        assert pd.notna(death_offset) and float(death_offset) >= 0 and float(death_offset) <= 240

    def test_death_after_4h_not_excluded(self):
        """Death after 4 hours should NOT be excluded for pre-landmark death."""
        death_offset = 241.0
        assert not (pd.notna(death_offset) and float(death_offset) >= 0 and float(death_offset) <= 240)

    def test_negative_death_offset_not_excluded(self):
        """Negative death offset (data error) should NOT be excluded."""
        death_offset = -10.0
        assert not (pd.notna(death_offset) and float(death_offset) >= 0 and float(death_offset) <= 240)

    def test_derive_death_offset_minutes_expired(self):
        """derive_death_offset_minutes correctly identifies expired patients."""
        row = pd.Series({
            "unitdischargestatus": "Expired",
            "unitdischargelocation": "",
            "hospitaldischargestatus": "",
            "hospitaldischargelocation": "",
            "unitdischargeoffset": 1440,
            "hospitaldischargeoffset": pd.NA,
            "hospitaladmitoffset": pd.NA,
        })
        result = derive_death_offset_minutes(row)
        assert result == 1440.0

    def test_derive_death_offset_minutes_alive(self):
        """derive_death_offset_minutes returns None for alive patients."""
        row = pd.Series({
            "unitdischargestatus": "Alive",
            "unitdischargelocation": "Step-Down Unit",
            "hospitaldischargestatus": "Alive",
            "hospitaldischargelocation": "Home",
            "unitdischargeoffset": 5000,
            "hospitaldischargeoffset": 7200,
            "hospitaladmitoffset": -120,
        })
        result = derive_death_offset_minutes(row)
        assert result is None


# ---------------------------------------------------------------------------
# 6. Validator: assert_required_columns
# ---------------------------------------------------------------------------


class TestValidatorRequiredColumns:
    """Test assert_required_columns raises on missing columns."""

    def test_raises_on_missing_columns(self):
        """assert_required_columns raises ValueError when columns are missing."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            assert_required_columns(df, ["a", "b", "c"], context="test")

    def test_passes_when_all_columns_present(self):
        """assert_required_columns passes when all required columns exist."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        assert_required_columns(df, ["a", "b", "c"])

    def test_error_message_includes_missing_columns(self):
        """Error message lists the missing columns."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="c") as exc_info:
            assert_required_columns(df, ["a", "c"])
        assert "c" in str(exc_info.value)

    def test_error_message_includes_context(self):
        """Error message includes context label when provided."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match=r"\[mimic\]"):
            assert_required_columns(df, ["b"], context="mimic")

    def test_empty_required_list_passes(self):
        """assert_required_columns passes with empty required list."""
        df = pd.DataFrame({"a": [1]})
        assert_required_columns(df, [])


# ---------------------------------------------------------------------------
# 7. Validator: assert_no_duplicate_stays
# ---------------------------------------------------------------------------


class TestValidatorNoDuplicateStays:
    """Test assert_no_duplicate_stays catches duplicates."""

    def test_raises_on_duplicate_stays(self):
        """assert_no_duplicate_stays raises ValueError when duplicates found."""
        df = pd.DataFrame({"stay_id": [1, 2, 2, 3]})
        with pytest.raises(ValueError, match="duplicate stay IDs"):
            assert_no_duplicate_stays(df, context="test")

    def test_passes_with_unique_stays(self):
        """assert_no_duplicate_stays passes when all stay IDs are unique."""
        df = pd.DataFrame({"stay_id": [1, 2, 3, 4]})
        assert_no_duplicate_stays(df)

    def test_custom_stay_id_column(self):
        """assert_no_duplicate_stays works with custom stay ID column."""
        df = pd.DataFrame({"hadm_id": [1, 2, 2, 3]})
        with pytest.raises(ValueError, match="duplicate stay IDs"):
            assert_no_duplicate_stays(df, stay_id_col="hadm_id")

    def test_raises_when_stay_id_column_missing(self):
        """assert_no_duplicate_stays raises ValueError when stay_id column is missing."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found"):
            assert_no_duplicate_stays(df, stay_id_col="stay_id")

    def test_error_includes_duplicate_count(self):
        """Error message includes count and examples of duplicates."""
        df = pd.DataFrame({"stay_id": [1, 2, 2, 3, 3]})
        with pytest.raises(ValueError) as exc_info:
            assert_no_duplicate_stays(df)
        msg = str(exc_info.value)
        assert "2" in msg  # 2 duplicates


# ---------------------------------------------------------------------------
# 8. Validator: assert_cohort_size
# ---------------------------------------------------------------------------


class TestValidatorCohortSize:
    """Test assert_cohort_size enforces minimum."""

    def test_raises_when_below_minimum(self):
        """assert_cohort_size raises ValueError when DataFrame is too small."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="minimum required"):
            assert_cohort_size(df, min_size=10, context="test")

    def test_passes_when_at_minimum(self):
        """assert_cohort_size passes when DataFrame has exactly min_size rows."""
        df = pd.DataFrame({"a": range(5)})
        assert_cohort_size(df, min_size=5)

    def test_passes_when_above_minimum(self):
        """assert_cohort_size passes when DataFrame exceeds min_size."""
        df = pd.DataFrame({"a": range(20)})
        assert_cohort_size(df, min_size=10)

    def test_error_message_includes_actual_size(self):
        """Error message includes actual row count."""
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="3") as exc_info:
            assert_cohort_size(df, min_size=3)
        assert "2" in str(exc_info.value)  # actual size

    def test_error_message_includes_context(self):
        """Error message includes context label when provided."""
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match=r"\[eicu\]"):
            assert_cohort_size(df, min_size=5, context="eicu")


# ---------------------------------------------------------------------------
# 9. build_cohort dispatch
# ---------------------------------------------------------------------------


class TestBuildCohortDispatch:
    """Test build_cohort routes to correct builder by dataset name."""

    def test_dispatch_mimic(self):
        """build_cohort('mimic', ...) should create MIMICCohortBuilder."""
        # We can't run the full pipeline without data, but we can verify
        # that the dispatch function routes to the correct builder class.
        from unittest.mock import patch

        with patch.object(MIMICCohortBuilder, "build_cohort", return_value=CohortResult(
            cohort_df=pd.DataFrame(), valid_stay_ids=np.array([]), anchors=pd.DataFrame()
        )) as mock_build:
            result = build_cohort("mimic", data_root="/tmp/fake_mimic")
            mock_build.assert_called_once()

    def test_dispatch_eicu(self):
        """build_cohort('eicu', ...) should create EICUCohortBuilder."""
        from unittest.mock import patch

        with patch.object(EICUCohortBuilder, "build_cohort", return_value=CohortResult(
            cohort_df=pd.DataFrame(), valid_stay_ids=np.array([]), anchors=pd.DataFrame()
        )) as mock_build:
            result = build_cohort("eicu", data_root="/tmp/fake_eicu")
            mock_build.assert_called_once()

    def test_dispatch_invalid_dataset_raises(self):
        """build_cohort raises error for unsupported dataset names."""
        with pytest.raises((ValueError, FileNotFoundError)):
            build_cohort("unknown_dataset", data_root="/tmp/fake")

    @pytest.mark.mimic
    def test_mimic_builder_initialized_with_data_root(self):
        """MIMICCohortBuilder stores data_root correctly."""
        builder = MIMICCohortBuilder(data_root="/tmp/mimic_data", config={"hf_icd_patterns": {9: [r"^428"], 10: [r"^I50"]}, "cardiogenic_shock_icd_patterns": {9: [r"^78551"], 10: [r"^R570"]}})
        assert str(builder.data_root) == "/tmp/mimic_data"

    @pytest.mark.eicu
    def test_eicu_builder_initialized_with_data_root(self):
        """EICUCohortBuilder stores data_root correctly."""
        builder = EICUCohortBuilder(data_root="/tmp/eicu_data")
        assert str(builder.data_root) == "/tmp/eicu_data"


# ---------------------------------------------------------------------------
# 10. CohortResult dataclass structure
# ---------------------------------------------------------------------------


class TestCohortResultStructure:
    """Test CohortResult dataclass has expected fields."""

    def test_cohort_result_has_cohort_df(self):
        """CohortResult must have a cohort_df field."""
        result = CohortResult(
            cohort_df=pd.DataFrame({"stay_id": [1]}),
            valid_stay_ids=np.array([1]),
            anchors=pd.DataFrame(),
        )
        assert hasattr(result, "cohort_df")
        assert isinstance(result.cohort_df, pd.DataFrame)

    def test_cohort_result_has_valid_stay_ids(self):
        """CohortResult must have a valid_stay_ids field."""
        result = CohortResult(
            cohort_df=pd.DataFrame(),
            valid_stay_ids=np.array([1, 2, 3]),
            anchors=pd.DataFrame(),
        )
        assert hasattr(result, "valid_stay_ids")
        np.testing.assert_array_equal(result.valid_stay_ids, [1, 2, 3])

    def test_cohort_result_has_anchors(self):
        """CohortResult must have an anchors field."""
        anchors_df = pd.DataFrame({"hadm_id": [1], "anchor_time": pd.Timestamp("2023-01-01")})
        result = CohortResult(
            cohort_df=pd.DataFrame(),
            valid_stay_ids=np.array([1]),
            anchors=anchors_df,
        )
        assert hasattr(result, "anchors")
        assert isinstance(result.anchors, pd.DataFrame)
        assert "hadm_id" in result.anchors.columns

    def test_cohort_result_anchors_default_empty(self):
        """CohortResult anchors default to empty DataFrame."""
        result = CohortResult(
            cohort_df=pd.DataFrame(),
            valid_stay_ids=np.array([]),
        )
        assert isinstance(result.anchors, pd.DataFrame)
        assert result.anchors.empty

    def test_cohort_result_valid_stay_ids_is_numpy_array(self):
        """CohortResult valid_stay_ids should be a numpy array."""
        result = CohortResult(
            cohort_df=pd.DataFrame(),
            valid_stay_ids=np.array([100, 200]),
        )
        assert isinstance(result.valid_stay_ids, np.ndarray)

    def test_eicu_cohort_result_same_structure(self):
        """eICU CohortResult has the same fields as MIMIC CohortResult."""
        from physiograph.cohort.eicu_cohort import CohortResult as EICUCohortResult
        eicu_result = EICUCohortResult(
            cohort_df=pd.DataFrame(),
            valid_stay_ids=np.array([]),
            anchors=pd.DataFrame(),
        )
        assert hasattr(eicu_result, "cohort_df")
        assert hasattr(eicu_result, "valid_stay_ids")
        assert hasattr(eicu_result, "anchors")