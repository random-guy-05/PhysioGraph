"""Pipeline orchestration for PhysioGraph ETL workflows.

Coordinates cohort extraction, label derivation, feature engineering,
validation, and artifact writing for MIMIC-IV and eICU-CRD datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .etl.audit import AuditLogger
from .etl.eicu_extractor import extract_eicu
from .etl.mimic_extractor import extract_mimic
from .etl.shared import (
    COHORT_REQUIRED_COLUMNS,
    EVENT_REQUIRED_COLUMNS,
    LABEL_REQUIRED_COLUMNS,
    LANDMARK_MINUTES,
    OUTCOME_WINDOW_END_MINUTES,
    TIME_STEP_HOURS,
    classify_offset_minutes,
    is_outcome_offset_minutes,
    is_pre_or_at_landmark,
    observation_time_bin,
    ordered_columns,
    require_columns,
    sanitize_events,
)

# ──────────────────────────────────────────────────────────────────────
# Feature Schema Constants
# ──────────────────────────────────────────────────────────────────────

PORTABLE_CONTEXT_VARIABLES: list[str] = [
    "lactate",
    "hr",
    "sbp",
    "map",
    "ph",
    "creatinine",
    "bilirubin_total",
    "spo2",
    "resp_rate",
    "temp",
]

ANALYSIS_ONLY_CONTEXT_COLUMNS: list[str] = [
    "landmark_lactate",
    "post_landmark_lactate_last",
    "lactate_clearance_24h_pct",
    "complete_lactate_clearance_24h_flag",
    "clearance_ge_64_24h_flag",
    "mortality_24h_flag",
]

OUTCOME_FLAG_COLUMNS: list[str] = [
    "lactate_rise_12h_flag",
    "lactate_rise_24h_flag",
    "vis_rise_12h_flag",
    "vis_rise_24h_flag",
    "uo_decline_12h_flag",
    "uo_decline_24h_flag",
    "pressor_12h_flag",
    "pressor_24h_flag",
    "mcs_12h_flag",
    "mcs_24h_flag",
    "escalation_12h_flag",
    "escalation_24h_flag",
    "renal_injury_12h_flag",
    "renal_injury_24h_flag",
    "hypoperfusion_12h_flag",
    "hypoperfusion_24h_flag",
    "hepatic_injury_12h_flag",
    "hepatic_injury_24h_flag",
    "end_organ_12h_flag",
    "end_organ_24h_flag",
    "shock_progression_12h_flag",
    "shock_progression_24h_flag",
    "mortality_12h_flag",
    "mortality_24h_flag",
    "target",
]

PRIMARY_FEATURE_COLUMNS: list[str] = [
    "dataset",
    "stay_id",
    "age",
    "is_male",
    "cohort_hf_flag",
    "shock_icd_flag",
    "baseline_lactate",
    "baseline_hr",
    "baseline_sbp",
    "baseline_map",
    "baseline_ph",
    "baseline_creatinine",
    "baseline_bilirubin_total",
    "baseline_spo2",
    "baseline_resp_rate",
    "baseline_temp",
    "tachycardia_flag",
    "hypotension_flag",
    "severe_hypotension_flag",
    "acidemia_flag",
    "severe_acidemia_flag",
    "modifier_burden",
    "lactate_delta",
    "lactate_slope_per_hr",
    "lactate_clearance_4h_pct",
    "persistent_lactate_flag",
    "renal_hypoperfusion_flag",
    "hepatic_hypoperfusion_flag",
    "lactate_ge_2_flag",
    "lactate_ge_3_1_flag",
    "lactate_ge_5_flag",
    "map_below_65_fraction",
    "sbp_below_90_fraction",
    "hr_above_100_fraction",
    "ph_below_7_25_fraction",
    "lactate_map_ratio",
    "lactate_sbp_ratio",
    "lactate_acidemia_interaction",
    "lactate_hypotension_interaction",
    "lactate_tachycardia_interaction",
    "occult_hypoperfusion_flag",
    "perfusion_burden_score",
    "baseline_lactate_bin",
    "baseline_lactate_focus_bin",
    "hr_band",
    "hr_band_fine",
    "lactate_scai_modifier",
    "scai_stage",
    "scai_stage_num",
    "scai_stage_collapsed",
]

LACTATE_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.1, 5.0, np.inf]
LACTATE_BIN_LABELS: list[str] = ["<2", "2-3.1", "3.1-5", ">=5"]
FOCUS_LACTATE_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.0, 5.0, np.inf]
FOCUS_LACTATE_BIN_LABELS: list[str] = ["<2", "2-3", "3-5", ">=5"]
HR_FINE_BAND_EDGES: list[float] = [
    -np.inf,
    100.0,
    110.0,
    120.0,
    140.0,
    np.inf,
]
HR_FINE_BAND_LABELS: list[str] = [
    "<100",
    "100-109",
    "110-119",
    "120-139",
    ">=140",
]
SCAI_STAGE_LABELS: list[str] = ["A", "B", "C", "D", "E"]

FORBIDDEN_FEATURE_COLUMNS: set[str] = set(
    ANALYSIS_ONLY_CONTEXT_COLUMNS + OUTCOME_FLAG_COLUMNS
)


# ──────────────────────────────────────────────────────────────────────
# Leakage Guards
# ──────────────────────────────────────────────────────────────────────


def assert_no_feature_leakage_columns(columns: list[str] | pd.Index) -> None:
    """Raise ValueError if post-landmark or outcome columns appear in features.

    Args:
        columns: Column names to check.
    """
    overlap = sorted(FORBIDDEN_FEATURE_COLUMNS & set(columns))
    if overlap:
        raise ValueError(
            f"Post-landmark or outcome columns leaked into primary features: "
            f"{overlap}"
        )


def assert_observation_only(events_df: pd.DataFrame) -> None:
    """Raise ValueError if features use non-observation-window events.

    Args:
        events_df: Events DataFrame to check.
    """
    non_obs = sorted(
        set(
            events_df.loc[
                events_df["window"] != "observation", "window"
            ].dropna().tolist()
        )
    )
    if non_obs:
        raise ValueError(
            f"Primary features must use observation-only events, "
            f"found windows: {non_obs}"
        )


# ──────────────────────────────────────────────────────────────────────
# Validation Helpers
# ──────────────────────────────────────────────────────────────────────


def validate_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder cohort DataFrame columns.

    Args:
        df: Cohort DataFrame.

    Returns:
        Validated DataFrame with required columns first.
    """
    require_columns(df, COHORT_REQUIRED_COLUMNS, "cohort_df")
    return ordered_columns(df, COHORT_REQUIRED_COLUMNS)


def validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder labels DataFrame columns.

    Args:
        df: Labels DataFrame.

    Returns:
        Validated DataFrame with required columns first.
    """
    require_columns(df, LABEL_REQUIRED_COLUMNS, "labels_df")
    return ordered_columns(df, LABEL_REQUIRED_COLUMNS)


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder events DataFrame columns.

    Args:
        df: Events DataFrame.

    Returns:
        Validated DataFrame with required columns first.
    """
    require_columns(df, EVENT_REQUIRED_COLUMNS, "events_df")
    return ordered_columns(df, EVENT_REQUIRED_COLUMNS)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder features DataFrame columns.

    Args:
        df: Features DataFrame.

    Returns:
        Validated DataFrame with required columns first.
    """
    require_columns(df, PRIMARY_FEATURE_COLUMNS, "features_df")
    return ordered_columns(df, PRIMARY_FEATURE_COLUMNS)


# ──────────────────────────────────────────────────────────────────────
# Label Derivation
# ──────────────────────────────────────────────────────────────────────


def _series_from_table(
    frame: pd.DataFrame, column: str, index: pd.Index
) -> pd.Series:
    """Extract a column from a pivoted table, reindexed to the given index.

    Args:
        frame: Pivoted DataFrame.
        column: Column name to extract.
        index: Target index.

    Returns:
        Aligned Series (NaN-filled if column missing).
    """
    if column in frame.columns:
        return frame[column].reindex(index)
    return pd.Series(index=index, dtype=float)


def derive_labels(
    cohort_df: pd.DataFrame, events_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive outcome labels and apply pre-landmark exclusions.

    Computes 12-hour and 24-hour outcome flags (pressor, MCS, end-organ injury,
    mortality, shock progression, lactate rise, VIS rise, UO decline) from
    events and cohort metadata.
    Excludes stays with intervention or death on or before the 4-hour
    landmark.

    Args:
        cohort_df: Cohort DataFrame with stay-level metadata.
        events_df: Long-format events DataFrame.

    Returns:
        Tuple of (updated_cohort_df, labels_df):
        - cohort_df gains exclusion flags and reasons.
        - labels_df contains per-stay outcome flags for valid stays.
    """
    cohort_df = cohort_df.copy()
    events_df = sanitize_events(events_df)

    def is_outcome_12h_offset_minutes(offset_minutes: float | int | None) -> bool:
        return offset_minutes is not None and LANDMARK_MINUTES < float(offset_minutes) <= (12 * 60)

    interventions = events_df.loc[
        events_df["is_intervention"] == 1,
        ["stay_id", "concept", "offset_minutes"],
    ].copy()
    early_intervention_ids = set(
        interventions.loc[
            interventions["offset_minutes"].map(is_pre_or_at_landmark),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    early_death_ids = set(
        cohort_df.loc[
            cohort_df["death_offset_minutes"].notna()
            & cohort_df["death_offset_minutes"].map(is_pre_or_at_landmark),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )

    cohort_df["excluded_before_landmark_flag"] = (
        cohort_df["stay_id"]
        .isin(early_intervention_ids | early_death_ids)
        .astype(int)
    )

    def _exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if int(row["stay_id"]) in early_intervention_ids:
            reasons.append("intervention_before_or_at_4h")
        if int(row["stay_id"]) in early_death_ids:
            reasons.append("death_before_or_at_4h")
        return ";".join(reasons)

    cohort_df["exclusion_reason"] = cohort_df.apply(
        _exclusion_reason, axis=1
    )
    valid_df = cohort_df.loc[
        cohort_df["excluded_before_landmark_flag"] == 0
    ].copy()
    valid_index = pd.Index(
        valid_df["stay_id"].astype(int).tolist(), name="stay_id"
    )

    valid_events = events_df.loc[
        events_df["stay_id"].isin(valid_index)
    ].copy()
    labs = valid_events.loc[
        valid_events["event_family"] == "lab",
        ["stay_id", "concept", "offset_minutes", "value_numeric", "window"],
    ].copy()

    if not labs.empty:
        observation_labs = labs.loc[labs["window"] == "observation"].copy()
        future_labs = labs.loc[labs["window"] == "outcome"].copy()
        future_labs_12h = labs.loc[labs["offset_minutes"].map(is_outcome_12h_offset_minutes)].copy()

        baseline_last = (
            observation_labs.sort_values(["stay_id", "concept", "offset_minutes"])
            .groupby(["stay_id", "concept"])
            .tail(1)
            .pivot(index="stay_id", columns="concept", values="value_numeric")
            .reindex(valid_index)
        )
        future_max = (
            future_labs.pivot_table(
                index="stay_id",
                columns="concept",
                values="value_numeric",
                aggfunc="max",
            )
            .reindex(valid_index)
        )
        future_max_12h = (
            future_labs_12h.pivot_table(
                index="stay_id",
                columns="concept",
                values="value_numeric",
                aggfunc="max",
            )
            .reindex(valid_index)
        )
        future_min = (
            future_labs.pivot_table(
                index="stay_id",
                columns="concept",
                values="value_numeric",
                aggfunc="min",
            )
            .reindex(valid_index)
        )
        future_min_12h = (
            future_labs_12h.pivot_table(
                index="stay_id",
                columns="concept",
                values="value_numeric",
                aggfunc="min",
            )
            .reindex(valid_index)
        )
        future_last = (
            future_labs.sort_values(["stay_id", "concept", "offset_minutes"])
            .groupby(["stay_id", "concept"])
            .tail(1)
            .pivot(index="stay_id", columns="concept", values="value_numeric")
            .reindex(valid_index)
        )
    else:
        baseline_last = pd.DataFrame(index=valid_index)
        future_max = pd.DataFrame(index=valid_index)
        future_max_12h = pd.DataFrame(index=valid_index)
        future_min = pd.DataFrame(index=valid_index)
        future_min_12h = pd.DataFrame(index=valid_index)
        future_last = pd.DataFrame(index=valid_index)

    landmark_lactate = _series_from_table(baseline_last, "lactate", valid_index)
    post_landmark_lactate_last = _series_from_table(
        future_last, "lactate", valid_index
    )
    valid_landmark = landmark_lactate.where(landmark_lactate > 0)
    lactate_clearance_24h_pct = (
        ((valid_landmark - post_landmark_lactate_last) / valid_landmark)
        * 100.0
    ).replace([np.inf, -np.inf], np.nan)

    baseline_creatinine = _series_from_table(
        baseline_last, "creatinine", valid_index
    )
    future_creatinine = _series_from_table(future_max, "creatinine", valid_index)
    future_creatinine_12h = _series_from_table(future_max_12h, "creatinine", valid_index)
    future_lactate = _series_from_table(future_max, "lactate", valid_index)
    future_lactate_12h = _series_from_table(future_max_12h, "lactate", valid_index)
    future_ph = _series_from_table(future_min, "ph", valid_index)
    future_ph_12h = _series_from_table(future_min_12h, "ph", valid_index)
    future_bili = _series_from_table(
        future_max, "bilirubin_total", valid_index
    )
    future_bili_12h = _series_from_table(
        future_max_12h, "bilirubin_total", valid_index
    )
    future_ast = _series_from_table(future_max, "ast", valid_index)
    future_ast_12h = _series_from_table(future_max_12h, "ast", valid_index)
    future_alt = _series_from_table(future_max, "alt", valid_index)
    future_alt_12h = _series_from_table(future_max_12h, "alt", valid_index)

    renal_injury = (
        (future_creatinine >= (baseline_creatinine + 0.3))
        | (future_creatinine >= (baseline_creatinine * 1.5))
        | (future_creatinine >= 2.5)
    ).fillna(False)
    renal_injury_12h = (
        (future_creatinine_12h >= (baseline_creatinine + 0.3))
        | (future_creatinine_12h >= (baseline_creatinine * 1.5))
        | (future_creatinine_12h >= 2.5)
    ).fillna(False)

    hypoperfusion = (
        (future_lactate >= 4.0) | (future_ph < 7.20)
    ).fillna(False)
    hypoperfusion_12h = (
        (future_lactate_12h >= 4.0) | (future_ph_12h < 7.20)
    ).fillna(False)

    hepatic_injury = (
        (future_bili >= 2.0)
        | (future_ast >= 200.0)
        | (future_alt >= 200.0)
    ).fillna(False)
    hepatic_injury_12h = (
        (future_bili_12h >= 2.0)
        | (future_ast_12h >= 200.0)
        | (future_alt_12h >= 200.0)
    ).fillna(False)

    end_organ = (renal_injury | hypoperfusion | hepatic_injury).fillna(False)
    end_organ_12h = (renal_injury_12h | hypoperfusion_12h | hepatic_injury_12h).fillna(False)

    lactate_rise_24h = ((future_lactate - valid_landmark) >= 1.0) | ((future_lactate > 2.0) & (valid_landmark <= 2.0))
    lactate_rise_12h = ((future_lactate_12h - valid_landmark) >= 1.0) | ((future_lactate_12h > 2.0) & (valid_landmark <= 2.0))

    pressor_24h_ids = set(
        interventions.loc[
            (interventions["concept"] == "pressor")
            & interventions["offset_minutes"].map(is_outcome_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    pressor_12h_ids = set(
        interventions.loc[
            (interventions["concept"] == "pressor")
            & interventions["offset_minutes"].map(is_outcome_12h_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    mcs_24h_ids = set(
        interventions.loc[
            (interventions["concept"] == "mcs")
            & interventions["offset_minutes"].map(is_outcome_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    mcs_12h_ids = set(
        interventions.loc[
            (interventions["concept"] == "mcs")
            & interventions["offset_minutes"].map(is_outcome_12h_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    escalation_24h_ids = pressor_24h_ids | mcs_24h_ids
    escalation_12h_ids = pressor_12h_ids | mcs_12h_ids

    mortality_24h_ids = set(
        valid_df.loc[
            valid_df["death_offset_minutes"].notna()
            & valid_df["death_offset_minutes"].map(is_outcome_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )
    mortality_12h_ids = set(
        valid_df.loc[
            valid_df["death_offset_minutes"].notna()
            & valid_df["death_offset_minutes"].map(is_outcome_12h_offset_minutes),
            "stay_id",
        ]
        .astype(int)
        .tolist()
    )

    end_organ_ids = set(end_organ.index[end_organ].astype(int).tolist())
    end_organ_12h_ids = set(end_organ_12h.index[end_organ_12h].astype(int).tolist())
    shock_progression_ids = (
        escalation_24h_ids | end_organ_ids | mortality_24h_ids
    )
    shock_progression_12h_ids = (
        escalation_12h_ids | end_organ_12h_ids | mortality_12h_ids
    )

    labels_df = valid_df[
        [
            "dataset",
            "stay_id",
            "cohort_hf_flag",
            "shock_icd_flag",
            "age",
            "is_male",
        ]
    ].copy()
    labels_df["landmark_lactate"] = labels_df["stay_id"].map(landmark_lactate)
    labels_df["post_landmark_lactate_last"] = labels_df["stay_id"].map(
        post_landmark_lactate_last
    )
    labels_df["lactate_clearance_24h_pct"] = labels_df["stay_id"].map(
        lactate_clearance_24h_pct
    )
    labels_df["complete_lactate_clearance_24h_flag"] = labels_df[
        "stay_id"
    ].map(
        (post_landmark_lactate_last < 2.0).fillna(False).astype(int)
    )
    labels_df["clearance_ge_64_24h_flag"] = labels_df["stay_id"].map(
        (lactate_clearance_24h_pct >= 64.0).fillna(False).astype(int)
    )
    labels_df["pressor_24h_flag"] = (
        labels_df["stay_id"].isin(pressor_24h_ids).astype(int)
    )
    labels_df["pressor_12h_flag"] = (
        labels_df["stay_id"].isin(pressor_12h_ids).astype(int)
    )
    labels_df["mcs_24h_flag"] = (
        labels_df["stay_id"].isin(mcs_24h_ids).astype(int)
    )
    labels_df["mcs_12h_flag"] = (
        labels_df["stay_id"].isin(mcs_12h_ids).astype(int)
    )
    labels_df["escalation_24h_flag"] = (
        labels_df["stay_id"].isin(escalation_24h_ids).astype(int)
    )
    labels_df["escalation_12h_flag"] = (
        labels_df["stay_id"].isin(escalation_12h_ids).astype(int)
    )
    labels_df["renal_injury_24h_flag"] = labels_df["stay_id"].isin(
        set(renal_injury.index[renal_injury].astype(int).tolist())
    ).astype(int)
    labels_df["renal_injury_12h_flag"] = labels_df["stay_id"].isin(
        set(renal_injury_12h.index[renal_injury_12h].astype(int).tolist())
    ).astype(int)
    labels_df["hypoperfusion_24h_flag"] = labels_df["stay_id"].isin(
        set(hypoperfusion.index[hypoperfusion].astype(int).tolist())
    ).astype(int)
    labels_df["hypoperfusion_12h_flag"] = labels_df["stay_id"].isin(
        set(hypoperfusion_12h.index[hypoperfusion_12h].astype(int).tolist())
    ).astype(int)
    labels_df["hepatic_injury_24h_flag"] = labels_df["stay_id"].isin(
        set(hepatic_injury.index[hepatic_injury].astype(int).tolist())
    ).astype(int)
    labels_df["hepatic_injury_12h_flag"] = labels_df["stay_id"].isin(
        set(hepatic_injury_12h.index[hepatic_injury_12h].astype(int).tolist())
    ).astype(int)
    labels_df["end_organ_24h_flag"] = (
        labels_df["stay_id"].isin(end_organ_ids).astype(int)
    )
    labels_df["end_organ_12h_flag"] = (
        labels_df["stay_id"].isin(end_organ_12h_ids).astype(int)
    )
    labels_df["mortality_24h_flag"] = (
        labels_df["stay_id"].isin(mortality_24h_ids).astype(int)
    )
    labels_df["mortality_12h_flag"] = (
        labels_df["stay_id"].isin(mortality_12h_ids).astype(int)
    )
    labels_df["shock_progression_24h_flag"] = (
        labels_df["stay_id"].isin(shock_progression_ids).astype(int)
    )
    labels_df["shock_progression_12h_flag"] = (
        labels_df["stay_id"].isin(shock_progression_12h_ids).astype(int)
    )

    # Adding the specific decompensation endpoints
    labels_df["lactate_rise_24h_flag"] = labels_df["stay_id"].isin(
        set(lactate_rise_24h.index[lactate_rise_24h].astype(int).tolist())
    ).astype(int)
    labels_df["lactate_rise_12h_flag"] = labels_df["stay_id"].isin(
        set(lactate_rise_12h.index[lactate_rise_12h].astype(int).tolist())
    ).astype(int)

    # Actually compute VIS score rise and UO decline based on extracted events
    vis_events = events_df.loc[events_df["concept"] == "vis"].copy()
    if not vis_events.empty:
        obs_vis = vis_events.loc[vis_events["offset_minutes"].le(LANDMARK_MINUTES)].groupby("stay_id")["value_numeric"].max()
        out_vis_12 = vis_events.loc[vis_events["offset_minutes"].gt(LANDMARK_MINUTES) & vis_events["offset_minutes"].map(is_outcome_12h_offset_minutes)].groupby("stay_id")["value_numeric"].max()
        out_vis_24 = vis_events.loc[vis_events["offset_minutes"].gt(LANDMARK_MINUTES) & vis_events["offset_minutes"].map(is_outcome_offset_minutes)].groupby("stay_id")["value_numeric"].max()

        # Merge and calculate
        vis_df = valid_df[["stay_id"]].copy()
        vis_df["base"] = vis_df["stay_id"].map(obs_vis).fillna(0)
        vis_df["out_12"] = vis_df["stay_id"].map(out_vis_12).fillna(0)
        vis_df["out_24"] = vis_df["stay_id"].map(out_vis_24).fillna(0)

        # VIS rise if it goes up by at least 5 points, or new start of any pressor.
        labels_df["vis_rise_12h_flag"] = ((vis_df["out_12"] - vis_df["base"]) >= 5.0).astype(int) | labels_df["pressor_12h_flag"]
        labels_df["vis_rise_24h_flag"] = ((vis_df["out_24"] - vis_df["base"]) >= 5.0).astype(int) | labels_df["pressor_24h_flag"]
    else:
        # Fallback if no VIS events extracted
        labels_df["vis_rise_24h_flag"] = labels_df["pressor_24h_flag"]
        labels_df["vis_rise_12h_flag"] = labels_df["pressor_12h_flag"]

    uo_events = events_df.loc[events_df["concept"] == "urine_output"].copy()
    if not uo_events.empty:
        # Rate: ml/hr.
        # We define UO decline as sum over 12/24h period being < 0.5 ml/kg/hr.
        # Since we don't have weight easily accessible in the strict schema,
        # we use absolute thresholds: < 30ml/hr average as a common proxy for oliguria.
        # 12h = < 360ml total
        # 24h = < 720ml total
        out_uo_12 = uo_events.loc[uo_events["offset_minutes"].gt(LANDMARK_MINUTES) & uo_events["offset_minutes"].map(is_outcome_12h_offset_minutes)].groupby("stay_id")["value_numeric"].sum()
        out_uo_24 = uo_events.loc[uo_events["offset_minutes"].gt(LANDMARK_MINUTES) & uo_events["offset_minutes"].map(is_outcome_offset_minutes)].groupby("stay_id")["value_numeric"].sum()

        uo_df = valid_df[["stay_id"]].copy()
        # If no events, assume oliguric? Safest is to assume normal or absent data, so we don't flag unless strictly measured.
        # Better: calculate rate per hour over the window and flag if rate < 30 ml/hr.
        uo_12_rate = uo_df["stay_id"].map(out_uo_12) / 12.0
        uo_24_rate = uo_df["stay_id"].map(out_uo_24) / 24.0

        # Only flag if we have data and rate is low
        labels_df["uo_decline_12h_flag"] = (uo_12_rate.notna() & (uo_12_rate < 30.0)).astype(int)
        labels_df["uo_decline_24h_flag"] = (uo_24_rate.notna() & (uo_24_rate < 30.0)).astype(int)
    else:
        labels_df["uo_decline_24h_flag"] = 0
        labels_df["uo_decline_12h_flag"] = 0

    labels_df["target"] = labels_df["shock_progression_24h_flag"]
    return cohort_df, labels_df


# ──────────────────────────────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────────────────────────────


def build_feature_table(
    events_df: pd.DataFrame, cohort_df: pd.DataFrame
) -> pd.DataFrame:
    """Build the primary feature table from observation-window events.

    Aggregates landmark-aware features from observation-window events
    (0-4 hours), including baseline measurements, threshold flags,
    lactate dynamics, interaction terms, SCAI staging, and categorical
    binning. Strictly uses observation-only events to prevent leakage.

    Args:
        events_df: Long-format events DataFrame.
        cohort_df: Cohort DataFrame (used for demographic context).

    Returns:
        Feature DataFrame with PRIMARY_FEATURE_COLUMNS.
    """
    events_df = sanitize_events(events_df)
    observation_events = events_df.loc[
        (events_df["window"] == "observation")
        & events_df["concept"].isin(PORTABLE_CONTEXT_VARIABLES)
    ].copy()
    assert_observation_only(observation_events)

    summary = cohort_df[
        [
            "dataset",
            "stay_id",
            "age",
            "is_male",
            "cohort_hf_flag",
            "shock_icd_flag",
        ]
    ].copy()
    ordered = observation_events.sort_values(
        ["stay_id", "concept", "offset_minutes"]
    ).copy()

    for variable in PORTABLE_CONTEXT_VARIABLES:
        sub = ordered.loc[
            ordered["concept"] == variable,
            ["stay_id", "value_numeric", "offset_minutes"],
        ].copy()
        if sub.empty:
            continue
        grouped = sub.groupby("stay_id", sort=False)
        stats = pd.DataFrame(
            {
                "stay_id": grouped["value_numeric"].mean().index.astype(int),
                f"{variable}_count": grouped["value_numeric"].count().values,
                f"{variable}_first": grouped["value_numeric"].first().values,
                f"{variable}_last": grouped["value_numeric"].last().values,
                f"{variable}_min": grouped["value_numeric"].min().values,
                f"{variable}_max": grouped["value_numeric"].max().values,
                f"{variable}_mean": grouped["value_numeric"].mean().values,
                f"{variable}_first_time": grouped["offset_minutes"].first().values
                / 60.0,
                f"{variable}_last_time": grouped["offset_minutes"].last().values
                / 60.0,
            }
        )
        summary = summary.merge(stats, on="stay_id", how="left")

    def _col(name: str) -> pd.Series:
        if name in summary.columns:
            return summary[name]
        return pd.Series(np.nan, index=summary.index)

    def _measurement_fraction(
        variable: str, comparator: Any
    ) -> pd.Series:
        sub = ordered.loc[
            ordered["concept"] == variable,
            ["stay_id", "value_numeric"],
        ].copy()
        if sub.empty:
            return pd.Series(np.nan, index=summary.index)
        frac = (
            sub.assign(flag=comparator(sub["value_numeric"]).astype(float))
            .groupby("stay_id")["flag"]
            .mean()
        )
        return summary["stay_id"].map(frac)

    summary["baseline_lactate"] = _col("lactate_last").combine_first(
        _col("lactate_max")
    )
    summary["baseline_hr"] = _col("hr_last").combine_first(_col("hr_max"))
    summary["baseline_sbp"] = _col("sbp_last").combine_first(_col("sbp_min"))
    summary["baseline_map"] = _col("map_last").combine_first(_col("map_min"))
    summary["baseline_ph"] = _col("ph_last").combine_first(_col("ph_min"))
    summary["baseline_creatinine"] = _col(
        "creatinine_last"
    ).combine_first(_col("creatinine_max"))
    summary["baseline_bilirubin_total"] = _col(
        "bilirubin_total_last"
    ).combine_first(_col("bilirubin_total_max"))
    summary["baseline_spo2"] = _col("spo2_last").combine_first(
        _col("spo2_mean")
    )
    summary["baseline_resp_rate"] = _col("resp_rate_last").combine_first(
        _col("resp_rate_mean")
    )
    summary["baseline_temp"] = _col("temp_last").combine_first(
        _col("temp_mean")
    )

    summary["tachycardia_flag"] = (
        (_col("hr_max") >= 100).fillna(False)
    ).astype(int)
    summary["hypotension_flag"] = (
        ((_col("sbp_min") < 90) | (_col("map_min") < 65)).fillna(False)
    ).astype(int)
    summary["severe_hypotension_flag"] = (
        ((_col("sbp_min") < 80) | (_col("map_min") < 60)).fillna(False)
    ).astype(int)
    summary["acidemia_flag"] = (
        (_col("ph_min") < 7.25).fillna(False)
    ).astype(int)
    summary["severe_acidemia_flag"] = (
        (_col("ph_min") < 7.20).fillna(False)
    ).astype(int)
    summary["modifier_burden"] = (
        summary["tachycardia_flag"]
        + summary["hypotension_flag"]
        + summary["acidemia_flag"]
    ).astype(int)

    valid_lactate_first = _col("lactate_first").where(
        _col("lactate_first") > 0
    )
    lactate_span = (_col("lactate_last_time") - _col("lactate_first_time")).clip(
        lower=TIME_STEP_HOURS
    )
    summary["lactate_delta"] = _col("lactate_last") - _col("lactate_first")
    summary["lactate_slope_per_hr"] = (
        summary["lactate_delta"] / lactate_span
    ).replace([np.inf, -np.inf], np.nan)
    summary["lactate_clearance_4h_pct"] = (
        ((valid_lactate_first - _col("lactate_last")) / valid_lactate_first)
        * 100.0
    ).replace([np.inf, -np.inf], np.nan)
    summary["persistent_lactate_flag"] = (
        (
            (_col("lactate_count").fillna(0) >= 2)
            & (_col("lactate_first") >= 2.0)
            & (_col("lactate_last") >= 2.0)
        )
        | (
            (_col("lactate_count").fillna(0) >= 2)
            & (_col("lactate_mean") >= 2.5)
            & (summary["lactate_delta"] > 0.3)
        )
    ).fillna(False).astype(int)
    summary["renal_hypoperfusion_flag"] = (
        (_col("creatinine_max") >= 2.0).fillna(False)
    ).astype(int)
    summary["hepatic_hypoperfusion_flag"] = (
        (_col("bilirubin_total_max") >= 2.0).fillna(False)
    ).astype(int)
    summary["lactate_ge_2_flag"] = (
        (summary["baseline_lactate"] >= 2.0).fillna(False)
    ).astype(int)
    summary["lactate_ge_3_1_flag"] = (
        (summary["baseline_lactate"] >= 3.1).fillna(False)
    ).astype(int)
    summary["lactate_ge_5_flag"] = (
        (summary["baseline_lactate"] >= 5.0).fillna(False)
    ).astype(int)
    summary["map_below_65_fraction"] = _measurement_fraction(
        "map", lambda s: s < 65.0
    ).fillna(0.0)
    summary["sbp_below_90_fraction"] = _measurement_fraction(
        "sbp", lambda s: s < 90.0
    ).fillna(0.0)
    summary["hr_above_100_fraction"] = _measurement_fraction(
        "hr", lambda s: s >= 100.0
    ).fillna(0.0)
    summary["ph_below_7_25_fraction"] = _measurement_fraction(
        "ph", lambda s: s < 7.25
    ).fillna(0.0)
    summary["lactate_map_ratio"] = (
        summary["baseline_lactate"]
        / summary["baseline_map"].clip(lower=35.0)
    ).replace([np.inf, -np.inf], np.nan)
    summary["lactate_sbp_ratio"] = (
        summary["baseline_lactate"]
        / summary["baseline_sbp"].clip(lower=60.0)
    ).replace([np.inf, -np.inf], np.nan)
    summary["lactate_acidemia_interaction"] = (
        summary["baseline_lactate"].fillna(0.0)
        * summary["acidemia_flag"]
    )
    summary["lactate_hypotension_interaction"] = (
        summary["baseline_lactate"].fillna(0.0)
        * summary["hypotension_flag"]
    )
    summary["lactate_tachycardia_interaction"] = (
        summary["baseline_lactate"].fillna(0.0)
        * summary["tachycardia_flag"]
    )
    summary["occult_hypoperfusion_flag"] = (
        (summary["lactate_ge_2_flag"] == 1)
        & (summary["hypotension_flag"] == 0)
    ).astype(int)
    summary["perfusion_burden_score"] = (
        summary["modifier_burden"]
        + summary["renal_hypoperfusion_flag"]
        + summary["hepatic_hypoperfusion_flag"]
        + summary["lactate_ge_3_1_flag"]
    ).astype(float)

    summary["baseline_lactate_bin"] = pd.cut(
        summary["baseline_lactate"],
        bins=LACTATE_BIN_EDGES,
        labels=LACTATE_BIN_LABELS,
        right=False,
    )
    summary["baseline_lactate_bin"] = (
        summary["baseline_lactate_bin"].astype("object").fillna("missing")
    )
    summary["baseline_lactate_focus_bin"] = pd.cut(
        summary["baseline_lactate"],
        bins=FOCUS_LACTATE_BIN_EDGES,
        labels=FOCUS_LACTATE_BIN_LABELS,
        right=False,
    )
    summary["baseline_lactate_focus_bin"] = (
        summary["baseline_lactate_focus_bin"]
        .astype("object")
        .fillna("missing")
    )
    summary["hr_band"] = pd.cut(
        summary["baseline_hr"].combine_first(_col("hr_max")),
        bins=[-np.inf, 100.0, 120.0, np.inf],
        labels=["<100", "100-119", ">=120"],
        right=False,
    )
    summary["hr_band"] = summary["hr_band"].astype("object").fillna("missing")
    summary["hr_band_fine"] = pd.cut(
        summary["baseline_hr"].combine_first(_col("hr_max")),
        bins=HR_FINE_BAND_EDGES,
        labels=HR_FINE_BAND_LABELS,
        right=False,
    )
    summary["hr_band_fine"] = (
        summary["hr_band_fine"].astype("object").fillna("missing")
    )
    summary["lactate_scai_modifier"] = np.select(
        [
            (summary["baseline_lactate"] >= 5.0) & (_col("ph_min") < 7.20),
            summary["baseline_lactate"] >= 5.0,
            _col("ph_min") < 7.20,
        ],
        [
            "lactate>=5 & pH<7.2",
            "lactate>=5 only",
            "pH<7.2 only",
        ],
        default="neither",
    )

    def _assign_scai_stage(row: pd.Series) -> str:
        if bool(row["severe_acidemia_flag"]) or (
            bool(row["severe_hypotension_flag"])
            and int(row["modifier_burden"]) >= 2
        ):
            return "E"
        if bool(row["persistent_lactate_flag"]) or (
            (
                pd.notna(row["baseline_lactate"])
                and row["baseline_lactate"] >= 2.0
            )
            and int(row["modifier_burden"]) >= 2
        ):
            return "D"
        if (
            (pd.notna(row["baseline_lactate"]) and row["baseline_lactate"] >= 2.0)
            or bool(row["acidemia_flag"])
            or bool(row["renal_hypoperfusion_flag"])
            or bool(row["hepatic_hypoperfusion_flag"])
        ):
            return "C"
        if bool(row["tachycardia_flag"]) or bool(row["hypotension_flag"]):
            return "B"
        return "A"

    summary["scai_stage"] = summary.apply(_assign_scai_stage, axis=1)
    summary["scai_stage_num"] = summary["scai_stage"].map(
        {stage: idx + 1 for idx, stage in enumerate(SCAI_STAGE_LABELS)}
    )
    summary["scai_stage_collapsed"] = np.where(
        summary["scai_stage"].isin(["D", "E"]),
        "D/E",
        np.where(
            summary["scai_stage"].isin(["B", "C"]), "B/C", "A"
        ),
    )
    features_df = summary[PRIMARY_FEATURE_COLUMNS].copy()
    assert_no_feature_leakage_columns(features_df.columns)
    return features_df


# ──────────────────────────────────────────────────────────────────────
# Pipeline Orchestration
# ──────────────────────────────────────────────────────────────────────


def run_pipeline(
    dataset: str,
    *,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 250_000,
) -> dict[str, str]:
    """Run the full PhysioGraph ETL pipeline for a dataset.

    Orchestrates cohort extraction → label derivation → feature
    engineering → validation → artifact writing. All outputs are
    validated and audit-logged at every step.

    Args:
        dataset: Dataset identifier ("mimic" or "eicu").
        mimic_root: Path to MIMIC data directory (required for "mimic").
        eicu_root: Path to eICU data directory (required for "eicu").
        output_dir: Output directory (defaults to ./physiograph_outputs/<dataset>).
        max_stays: Optional cap on cohort size (for testing).
        max_chunks: Optional cap on chunks per streaming table.
        chunk_size: Rows per chunk for CSV streaming (default: 250,000).

    Returns:
        Dict mapping artifact names to file paths:
        cohort, labels, events, features, audit, manifest.

    Raises:
        ValueError: If dataset is unsupported or data root is missing.
    """
    dataset = dataset.lower()
    if dataset not in ("mimic", "eicu"):
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected 'mimic' or 'eicu'."
        )

    root: Path
    if dataset == "mimic":
        if mimic_root is None:
            raise ValueError(
                "mimic_root is required for MIMIC pipeline."
            )
        root = Path(mimic_root)
    else:
        if eicu_root is None:
            raise ValueError(
                "eicu_root is required for eICU pipeline."
            )
        root = Path(eicu_root)

    output_path = (
        Path(output_dir) if output_dir is not None
        else Path.cwd() / "physiograph_outputs" / dataset
    )
    output_path.mkdir(parents=True, exist_ok=True)

    audit = AuditLogger(dataset=dataset)

    if dataset == "mimic":
        extracted = extract_mimic(
            root,
            audit,
            max_stays=max_stays,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
        )
    else:
        extracted = extract_eicu(
            root,
            audit,
            max_stays=max_stays,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
        )

    cohort_df, labels_df = derive_labels(
        extracted.cohort_df, extracted.events_df
    )
    valid_cohort = cohort_df.loc[
        cohort_df["excluded_before_landmark_flag"] == 0
    ].copy()
    valid_cohort = valid_cohort.loc[
        valid_cohort["stay_id"].isin(labels_df["stay_id"])
    ].copy()
    features_df = build_feature_table(
        extracted.events_df.loc[
            extracted.events_df["stay_id"].isin(labels_df["stay_id"])
        ],
        valid_cohort,
    )

    cohort_df = validate_cohort(cohort_df)
    labels_df = validate_labels(labels_df)
    events_df = validate_events(extracted.events_df)
    features_df = validate_features(features_df)

    cohort_path = output_path / "cohort.csv"
    labels_path = output_path / "labels.csv"
    events_path = output_path / "events.csv"
    features_path = output_path / "features.csv"
    audit_path = output_path / "audit.json"
    manifest_path = output_path / "manifest.json"

    cohort_df.to_csv(cohort_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    events_df.to_csv(events_path, index=False)
    features_df.to_csv(features_path, index=False)

    manifest = {
        "dataset": dataset,
        "max_stays": max_stays,
        "max_chunks": max_chunks,
        "chunk_size": chunk_size,
        "artifacts": {
            "cohort": str(cohort_path),
            "labels": str(labels_path),
            "events": str(events_path),
            "features": str(features_path),
            "audit": str(audit_path),
        },
        "counts": {
            "cohort_rows": int(len(cohort_df)),
            "label_rows": int(len(labels_df)),
            "event_rows": int(len(events_df)),
            "feature_rows": int(len(features_df)),
        },
    }
    audit.entries.append(
        {
            "step": "artifacts_written",
            "details": manifest["counts"],
            "dataset": dataset,
        }
    )
    audit_path.write_text(
        json.dumps(audit.entries, indent=2, default=str)
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "cohort": str(cohort_path),
        "labels": str(labels_path),
        "events": str(events_path),
        "features": str(features_path),
        "audit": str(audit_path),
        "manifest": str(manifest_path),
    }
