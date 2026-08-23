"""Data schema definitions and validation.

Defines pandera schemas for validating DataFrames at pipeline
boundaries (raw data, features, predictions).
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cohort schemas
# ---------------------------------------------------------------------------

_COHORT_COLUMNS: dict[str, Column] = {
    "dataset": Column(str, nullable=False, description="Dataset identifier (mimic/eicu)"),
    "stay_id": Column(int, nullable=False, description="ICU stay identifier"),
    "person_id": Column(int, nullable=False, description="Patient identifier"),
    "admit_time": Column(str, nullable=True, description="ICU admission timestamp"),
    "admit_year": Column(int, nullable=False, description="Year of admission"),
    "age": Column(float, nullable=False, description="Patient age at admission"),
    "is_male": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Male sex indicator"),
    "cohort_hf_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Heart failure cohort flag"),
    "shock_icd_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Shock ICD code flag"),
    "early_icu_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Early ICU admission flag"),
    "death_offset_minutes": Column(float, nullable=True, description="Minutes from admission to death"),
    "excluded_before_landmark_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Excluded before landmark flag"),
    "exclusion_reason": Column(str, nullable=True, description="Reason for exclusion"),
}

MIMICCohortSchema = DataFrameSchema(
    columns=_COHORT_COLUMNS,
    coerce=True,
    description="MIMIC-III cohort definition with patient demographics and flags",
)

EICUCohortSchema = DataFrameSchema(
    columns=_COHORT_COLUMNS,
    coerce=True,
    description="eICU cohort definition with patient demographics and flags",
)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FeatureSchema = DataFrameSchema(
    coerce=True,
    columns={
        "dataset": Column(str, nullable=False, description="Dataset identifier"),
        "stay_id": Column(int, nullable=False, description="ICU stay identifier"),
        "age": Column(float, nullable=False, description="Patient age"),
        "is_male": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Male sex indicator"),
        "cohort_hf_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Heart failure flag"),
        "shock_icd_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Shock ICD flag"),
        # Baseline vitals / labs
        "baseline_lactate": Column(float, nullable=True, checks=Check.ge(0), description="Baseline lactate (mmol/L)"),
        "baseline_hr": Column(float, nullable=True, checks=Check.ge(0), description="Baseline heart rate"),
        "baseline_sbp": Column(float, nullable=True, checks=Check.ge(0), description="Baseline systolic BP"),
        "baseline_map": Column(float, nullable=True, checks=Check.ge(0), description="Baseline mean arterial pressure"),
        "baseline_ph": Column(float, nullable=True, description="Baseline arterial pH"),
        "baseline_creatinine": Column(float, nullable=True, checks=Check.ge(0), description="Baseline creatinine"),
        "baseline_bilirubin_total": Column(float, nullable=True, checks=Check.ge(0), description="Baseline total bilirubin"),
        "baseline_spo2": Column(float, nullable=True, description="Baseline SpO2"),
        "baseline_resp_rate": Column(float, nullable=True, checks=Check.ge(0), description="Baseline respiratory rate"),
        "baseline_temp": Column(float, nullable=True, description="Baseline temperature"),
        # Binary flags
        "tachycardia_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Tachycardia flag"),
        "hypotension_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hypotension flag"),
        "severe_hypotension_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Severe hypotension flag"),
        "acidemia_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Acidemia flag"),
        "severe_acidemia_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Severe acidemia flag"),
        "modifier_burden": Column(int, nullable=False, checks=Check.ge(0), description="SCAI modifier burden count"),
        # Lactate dynamics
        "lactate_delta": Column(float, nullable=True, description="Lactate change from baseline"),
        "lactate_slope_per_hr": Column(float, nullable=True, description="Lactate slope per hour"),
        "lactate_clearance_4h_pct": Column(float, nullable=True, description="Lactate clearance at 4h (%)"),
        "persistent_lactate_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Persistent lactate elevation flag"),
        "renal_hypoperfusion_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Renal hypoperfusion flag"),
        "hepatic_hypoperfusion_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hepatic hypoperfusion flag"),
        "lactate_ge_2_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate >= 2 flag"),
        "lactate_ge_3_1_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate >= 3.1 flag"),
        "lactate_ge_5_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate >= 5 flag"),
        # Time-below thresholds
        "map_below_65_fraction": Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)], description="Fraction of time MAP < 65"),
        "sbp_below_90_fraction": Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)], description="Fraction of time SBP < 90"),
        "hr_above_100_fraction": Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)], description="Fraction of time HR > 100"),
        "ph_below_7_25_fraction": Column(float, nullable=False, checks=[Check.ge(0), Check.le(1)], description="Fraction of time pH < 7.25"),
        # Interaction features
        "lactate_map_ratio": Column(float, nullable=True, description="Lactate / MAP ratio"),
        "lactate_sbp_ratio": Column(float, nullable=True, description="Lactate / SBP ratio"),
        "lactate_acidemia_interaction": Column(float, nullable=True, description="Lactate x acidemia interaction"),
        "lactate_hypotension_interaction": Column(float, nullable=True, description="Lactate x hypotension interaction"),
        "lactate_tachycardia_interaction": Column(float, nullable=True, description="Lactate x tachycardia interaction"),
        # Composite scores
        "occult_hypoperfusion_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Occult hypoperfusion flag"),
        "perfusion_burden_score": Column(float, nullable=False, checks=Check.ge(0), description="Perfusion burden score"),
        # Categorical bins
        "baseline_lactate_bin": Column(str, nullable=False, description="Baseline lactate category bin"),
        "baseline_lactate_focus_bin": Column(str, nullable=False, description="Baseline lactate focus bin"),
        "hr_band": Column(str, nullable=False, description="Heart rate band"),
        "hr_band_fine": Column(str, nullable=False, description="Fine heart rate band"),
        "lactate_scai_modifier": Column(str, nullable=False, description="Lactate SCAI modifier category"),
        "scai_stage": Column(str, nullable=False, description="SCAI stage letter"),
        "scai_stage_num": Column(int, nullable=False, checks=[Check.ge(1), Check.le(5)], description="SCAI stage number"),
        "scai_stage_collapsed": Column(str, nullable=False, description="Collapsed SCAI stage"),
    },
    description="Engineered feature matrix for shock prediction models",
)

# ---------------------------------------------------------------------------
# Label schema
# ---------------------------------------------------------------------------

LabelSchema = DataFrameSchema(
    coerce=True,
    columns={
        "dataset": Column(str, nullable=False, description="Dataset identifier"),
        "stay_id": Column(int, nullable=False, description="ICU stay identifier"),
        "target": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Primary composite target"),
        "lactate_rise_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate rise within 12h"),
        "lactate_rise_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate rise within 24h"),
        "vis_rise_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="VIS rise within 12h"),
        "vis_rise_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="VIS rise within 24h"),
        "uo_decline_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Urine output decline within 12h"),
        "uo_decline_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Urine output decline within 24h"),
        "pressor_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Pressor use within 12h"),
        "pressor_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Pressor use within 24h"),
        "mcs_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="MCS use within 12h"),
        "mcs_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="MCS use within 24h"),
        "escalation_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Escalation within 12h"),
        "escalation_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Escalation within 24h"),
        "renal_injury_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Renal injury within 12h"),
        "renal_injury_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Renal injury within 24h"),
        "hypoperfusion_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hypoperfusion within 12h"),
        "hypoperfusion_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hypoperfusion within 24h"),
        "hepatic_injury_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hepatic injury within 12h"),
        "hepatic_injury_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Hepatic injury within 24h"),
        "end_organ_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="End-organ injury within 12h"),
        "end_organ_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="End-organ injury within 24h"),
        "mortality_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Mortality within 12h"),
        "mortality_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Mortality within 24h"),
        "shock_progression_12h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Shock progression within 12h"),
        "shock_progression_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Shock progression within 24h"),
        "landmark_lactate": Column(float, nullable=True, checks=Check.ge(0), description="Lactate at landmark time"),
        "post_landmark_lactate_last": Column(float, nullable=True, checks=Check.ge(0), description="Last lactate after landmark"),
        "lactate_clearance_24h_pct": Column(float, nullable=True, description="Lactate clearance at 24h (%)"),
        "complete_lactate_clearance_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Complete lactate clearance at 24h"),
        "clearance_ge_64_24h_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Lactate clearance >= 64% at 24h"),
        "cohort_hf_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Heart failure cohort flag"),
        "shock_icd_flag": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Shock ICD flag"),
        "age": Column(float, nullable=False, description="Patient age"),
        "is_male": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Male sex indicator"),
    },
    description="Outcome labels and target definitions for shock prediction",
)

# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

EventSchema = DataFrameSchema(
    coerce=True,
    columns={
        "dataset": Column(str, nullable=False, description="Dataset identifier"),
        "stay_id": Column(int, nullable=False, description="ICU stay identifier"),
        "event_family": Column(str, nullable=False, description="Event family (intervention/outcome)"),
        "concept": Column(str, nullable=False, description="Clinical concept name"),
        "source_table": Column(str, nullable=False, description="Source data table"),
        "raw_name": Column(str, nullable=True, description="Raw event name from source"),
        "offset_minutes": Column(float, nullable=False, checks=Check.ge(0), description="Minutes from ICU admission"),
        "window": Column(str, nullable=False, description="Time window (observation/outcome/outside)"),
        "time_bin": Column(float, nullable=True, description="Hourly time bin index"),
        "value_numeric": Column(float, nullable=True, description="Numeric event value"),
        "value_text": Column(str, nullable=True, description="Text event value"),
        "unit": Column(str, nullable=True, description="Measurement unit"),
        "is_intervention": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Intervention event flag"),
        "is_outcome_event": Column(int, nullable=False, checks=Check.isin([0, 1]), description="Outcome event flag"),
    },
    description="Long-format clinical events (vitals, labs, interventions, outcomes)",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MIMICCohortSchema",
    "EICUCohortSchema",
    "FeatureSchema",
    "LabelSchema",
    "EventSchema",
    "validate_schema",
]


def validate_schema(
    dataframe: pd.DataFrame,
    schema: DataFrameSchema,
    context: str = "",
) -> pd.DataFrame:
    """Validate *dataframe* against *schema* with clear error messages.

    Parameters
    ----------
    dataframe:
        DataFrame to validate.
    schema:
        Pandera ``DataFrameSchema`` to validate against.
    context:
        Optional label (e.g. file path) included in error messages.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame (may contain coerced types).

    Raises
    ------
    pa.errors.SchemaError
        Re-raised with additional context after logging.
    """
    label = f" [{context}]" if context else ""
    try:
        validated = schema.validate(dataframe, lazy=True)
        logger.debug("Schema validation passed%s – %d rows, %d cols", label, len(validated), len(validated.columns))
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error("Schema validation failed%s:\n%s", label, exc)
        raise
