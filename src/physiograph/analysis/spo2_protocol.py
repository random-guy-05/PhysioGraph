"""Leakage-safe analysis for SpO2 instability and early decompensation.

The clock is explicit throughout: predictors use [0, 240) minutes after ICU
admission and outcomes use (240, 240 + horizon] minutes.  A suffix such as
``_12h`` therefore means twelve hours *after the landmark*, not ICU hour 12.

Binary thresholds are sensitivity-friendly summaries of continuous outcomes;
the continuous deltas and availability indicators are always retained.  VIS is
only considered observed when an extractor supplied normalized ``concept=vis``
events.  Urine-output decline is named a proxy unless weight-normalized data
support the KDIGO oliguria threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

LANDMARK_MINUTES = 240
POST_LANDMARK_HORIZONS_HOURS = (12, 24)
LACTATE_RISE_ABSOLUTE = 0.5
CREATININE_AKI_ABSOLUTE = 0.3
CREATININE_AKI_RATIO = 1.5
BILIRUBIN_WORSENING_ABSOLUTE = 0.5
BILIRUBIN_WORSENING_RATIO = 1.5
URINE_RATE_DECLINE_FRACTION = 0.5
KDIGO_OLIGURIA_ML_KG_H = 0.5

PRIMARY_ENDPOINTS = (
    "lactate_rise_12h_flag",
    "lactate_rise_24h_flag",
    "vis_rise_12h_flag",
    "vis_rise_24h_flag",
)
SECONDARY_ENDPOINTS = (
    "death_12h_flag",
    "death_24h_flag",
    "pressor_initiation_12h_flag",
    "pressor_initiation_24h_flag",
    "urine_output_decline_proxy_12h_flag",
    "urine_output_decline_proxy_24h_flag",
    "oliguria_kdigo_proxy_12h_flag",
    "oliguria_kdigo_proxy_24h_flag",
    "aki_creatinine_12h_flag",
    "aki_creatinine_24h_flag",
    "hepatic_lab_worsening_12h_flag",
    "hepatic_lab_worsening_24h_flag",
    "hepatic_injury_extended_12h_flag",
    "hepatic_injury_extended_24h_flag",
    "platelet_injury_12h_flag",
    "platelet_injury_24h_flag",
    "troponin_relative_rise_12h_flag",
    "troponin_relative_rise_24h_flag",
    "mcs_12h_flag",
    "mcs_24h_flag",
    "early_decompensation_12h_flag",
    "early_decompensation_24h_flag",
    "early_decompensation_plus_vis_12h_flag",
    "early_decompensation_plus_vis_24h_flag",
)

ABSOLUTE_SPO2_FEATURES = (
    "spo2_mean",
    "spo2_min",
    "spo2_below_90_fraction",
)
SAMPLING_ADJUSTMENT_FEATURES = (
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_longest_gap_minutes",
)
INSTABILITY_FEATURES = (
    "spo2_sd",
    "spo2_rmssd",
    "spo2_iqr",
    "spo2_range",
    "spo2_mad",
    "spo2_abrupt_jump_rate_per_hr",
    "spo2_abrupt_jump_fraction",
    "spo2_drop_3_count",
    "spo2_drop_5_count",
    "spo2_slope_per_hr",
    "spo2_dynamics_proxy_score",
)
PARSIMONIOUS_INSTABILITY_FEATURES = (
    "spo2_rmssd",
    "spo2_abrupt_jump_fraction",
    "spo2_drop_3_count",
)
CLINICAL_CONTROL_CANDIDATES = (
    "age",
    "is_male",
    "sex_unknown_flag",
    "shock_icd_flag",
    "baseline_lactate",
    "baseline_creatinine",
    "baseline_map",
    "baseline_hr",
    "baseline_sbp",
    "baseline_ph",
    "baseline_bilirubin_total",
    "baseline_resp_rate",
    "baseline_vasoactive_flag",
    "baseline_mcs_flag",
    "fio2_max",
    "resp_support_any_flag",
    "mechanical_ventilation_flag",
    "noninvasive_ventilation_flag",
    "rrt_or_dialysis_flag",
    "dataset",
    "race",
    "ethnicity",
    "race_ethnicity",
    "hospital_id",
    "icu_type",
    "unit_admit_source",
    "admit_year",
)
MISSINGNESS_CONTROL_FEATURES = (
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_longest_gap_minutes",
    "spo2_plausible_count",
)
EXTERNAL_NONHARMONIZED_CONTROLS = {
    "dataset",
    "race",
    "ethnicity",
    "race_ethnicity",
    "hospital_id",
    "icu_type",
    "unit_admit_source",
    "admit_year",
}
CONTINUOUS_DECOMPENSATION_TRAJECTORIES = (
    "lactate_delta_12h",
    "lactate_delta_24h",
    "lactate_max_delta_12h",
    "lactate_max_delta_24h",
    "vis_delta_12h",
    "vis_delta_24h",
    "urine_output_decline_fraction_12h",
    "urine_output_decline_fraction_24h",
    "creatinine_delta_12h",
    "creatinine_delta_24h",
    "bilirubin_delta_12h",
    "bilirubin_delta_24h",
    "ast_ratio_12h",
    "ast_ratio_24h",
    "alt_ratio_12h",
    "alt_ratio_24h",
    "inr_ratio_12h",
    "inr_ratio_24h",
    "platelet_decline_fraction_12h",
    "platelet_decline_fraction_24h",
    "troponin_ratio_12h",
    "troponin_ratio_24h",
)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _keys_from(*frames: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty or "stay_id" not in frame:
            continue
        local = frame.copy()
        if "dataset" not in local:
            local["dataset"] = "unknown"
        pieces.append(local[["dataset", "stay_id"]].drop_duplicates())
    if not pieces:
        return pd.DataFrame(columns=["dataset", "stay_id"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _last_measurement(events: pd.DataFrame, concept: str) -> pd.DataFrame:
    selected = events.loc[events["concept"].eq(concept)].dropna(subset=["value_numeric"])
    if selected.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "value_numeric", "offset_minutes"])
    return (
        selected.sort_values(["dataset", "stay_id", "offset_minutes"])
        .groupby(["dataset", "stay_id"], as_index=False, sort=False)
        .tail(1)[["dataset", "stay_id", "value_numeric", "offset_minutes"]]
    )


def _max_measurement(events: pd.DataFrame, concept: str) -> pd.DataFrame:
    selected = events.loc[events["concept"].eq(concept)].dropna(subset=["value_numeric"])
    if selected.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    idx = selected.groupby(["dataset", "stay_id"])["value_numeric"].idxmax()
    return selected.loc[idx, ["dataset", "stay_id", "value_numeric"]]


def _last_measurement_any(
    events: pd.DataFrame, concepts: tuple[str, ...]
) -> pd.DataFrame:
    """Last measurement across assay-specific names for one clinical domain."""
    selected = events.loc[events["concept"].isin(concepts)].dropna(
        subset=["value_numeric"]
    )
    if selected.empty:
        return pd.DataFrame(
            columns=["dataset", "stay_id", "value_numeric", "offset_minutes"]
        )
    return (
        selected.sort_values(["dataset", "stay_id", "offset_minutes"])
        .groupby(["dataset", "stay_id"], as_index=False, sort=False)
        .tail(1)[["dataset", "stay_id", "value_numeric", "offset_minutes"]]
    )


def _max_measurement_any(
    events: pd.DataFrame, concepts: tuple[str, ...]
) -> pd.DataFrame:
    """Maximum measurement across assay-specific names for one domain."""
    selected = events.loc[events["concept"].isin(concepts)].dropna(
        subset=["value_numeric"]
    )
    if selected.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    idx = selected.groupby(["dataset", "stay_id"])["value_numeric"].idxmax()
    return selected.loc[idx, ["dataset", "stay_id", "value_numeric"]]


def _merge_value(
    result: pd.DataFrame,
    source: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:
    if source.empty:
        empty = result[["dataset", "stay_id"]].iloc[0:0].copy()
        empty[target_col] = pd.Series(dtype=float)
        return result.merge(empty, on=["dataset", "stay_id"], how="left")
    renamed = source[["dataset", "stay_id", source_col]].rename(columns={source_col: target_col})
    return result.merge(renamed, on=["dataset", "stay_id"], how="left")


def _binary_when_observed(condition: pd.Series, observed: pd.Series) -> pd.Series:
    values = pd.Series(np.nan, index=condition.index, dtype=float)
    values.loc[observed] = condition.loc[observed].astype(int)
    return values


def _binary_event_with_censoring(
    condition: pd.Series,
    ascertainable: pd.Series,
    complete_followup: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Keep confirmed pre-censor events; require full follow-up for negatives."""
    # A directly observed event remains known even when the source is only
    # partially complete (e.g., a deliberately truncated schema pilot).
    # Source/follow-up completeness is required to establish a non-event.
    positive = condition.fillna(False)
    observed = positive | (ascertainable & complete_followup)
    return _binary_when_observed(positive, observed), observed


def _event_flag(events: pd.DataFrame, concept: str, keys: pd.DataFrame) -> pd.Series:
    flagged = events.loc[events["concept"].eq(concept), ["dataset", "stay_id"]].drop_duplicates()
    if flagged.empty:
        return pd.Series(0, index=keys.index, dtype=int)
    merged = keys.merge(flagged.assign(_flag=1), on=["dataset", "stay_id"], how="left")
    return merged["_flag"].fillna(0).astype(int)


def _mcs_event_flag(events: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """Flag procedure-start or operational-device documentation MCS rows."""
    if events.empty or "concept" not in events:
        return pd.Series(0, index=keys.index, dtype=int)
    concept = events["concept"].fillna("").astype(str).str.lower()
    flagged = events.loc[
        concept.eq("mcs") | concept.str.startswith("mcs_context_"),
        ["dataset", "stay_id"],
    ].drop_duplicates()
    if flagged.empty:
        return pd.Series(0, index=keys.index, dtype=int)
    merged = keys.merge(
        flagged.assign(_flag=1), on=["dataset", "stay_id"], how="left"
    )
    return merged["_flag"].fillna(0).astype(int)


def _compute_early_decompensation_outcomes_legacy(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
    *,
    horizons: tuple[int, ...] = POST_LANDMARK_HORIZONS_HOURS,
) -> pd.DataFrame:
    """Derive prespecified 12/24-hour post-landmark outcomes and availability.

    ``lactate_rise`` is a last-value increase of at least 0.5 mmol/L from the
    last observation-window lactate; the continuous delta and a threshold-
    crossing sensitivity outcome are retained.  Creatinine follows the KDIGO
    absolute/relative criteria.  Bilirubin is explicitly named lab worsening,
    not acute liver failure.  Urine decline is a rate proxy; the KDIGO-style
    oliguria proxy is only populated when weight is observed.
    """
    keys = _keys_from(stay_index if stay_index is not None else cohort_df, cohort_df, events_df)
    result = keys.copy()
    if not cohort_df.empty and "person_id" in cohort_df:
        cohort_keys = cohort_df.copy()
        if "dataset" not in cohort_keys:
            cohort_keys["dataset"] = "unknown"
        result = result.merge(
            cohort_keys[["dataset", "stay_id", "person_id"]].drop_duplicates(),
            on=["dataset", "stay_id"],
            how="left",
        )

    events = events_df.copy()
    if events.empty:
        events = pd.DataFrame(columns=["dataset", "stay_id", "concept", "offset_minutes", "value_numeric"])
    if "dataset" not in events:
        events["dataset"] = "unknown"
    for column in ("concept", "offset_minutes", "value_numeric"):
        if column not in events:
            events[column] = np.nan if column != "concept" else ""
    events["concept"] = events["concept"].fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    events["value_numeric"] = pd.to_numeric(events["value_numeric"], errors="coerce")
    observation = events.loc[
        events["offset_minutes"].ge(0) & events["offset_minutes"].lt(LANDMARK_MINUTES)
    ].copy()

    baseline: dict[str, pd.DataFrame] = {
        concept: _last_measurement(observation, concept)
        for concept in ("lactate", "creatinine", "bilirubin_total", "weight", "vis")
    }
    for concept in ("lactate", "creatinine", "bilirubin_total", "weight"):
        result = _merge_value(result, baseline[concept], "value_numeric", f"baseline_{concept}_outcome")
    vis_datasets = set(events.loc[events["concept"].eq("vis"), "dataset"].dropna().astype(str))
    urine_datasets = set(events.loc[events["concept"].eq("urine_output"), "dataset"].dropna().astype(str))

    baseline_urine = observation.loc[
        observation["concept"].eq("urine_output") & observation["value_numeric"].gt(0)
    ]
    baseline_urine_sum = (
        baseline_urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
        if not baseline_urine.empty
        else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    )
    result = _merge_value(result, baseline_urine_sum, "value_numeric", "baseline_urine_output_ml")
    result["baseline_urine_output_rate_ml_h"] = (
        result["baseline_urine_output_ml"] / 4.0
    )

    for horizon in horizons:
        upper = LANDMARK_MINUTES + int(horizon * 60)
        result = result.copy()
        post = events.loc[
            events["offset_minutes"].gt(LANDMARK_MINUTES)
            & events["offset_minutes"].le(upper)
        ].copy()
        censor_end = pd.concat(
            [
                followup.fillna(float(upper)),
                death.fillna(float(upper)),
                pd.Series(float(upper), index=result.index),
            ],
            axis=1,
        ).min(axis=1)
        post = post.merge(
            result[["dataset", "stay_id"]].assign(_censor_end=censor_end),
            on=["dataset", "stay_id"],
            how="inner",
            validate="many_to_one",
        )
        post = post.loc[post["offset_minutes"].le(post["_censor_end"])].drop(
            columns="_censor_end"
        )

        for concept in ("lactate", "creatinine", "bilirubin_total"):
            last = _last_measurement(post, concept)
            maximum = _max_measurement(post, concept)
            result = _merge_value(result, last, "value_numeric", f"{concept}_last_{horizon}h")
            result = _merge_value(result, maximum, "value_numeric", f"{concept}_max_{horizon}h")

        lactate_observed = result["baseline_lactate_outcome"].notna() & result[f"lactate_last_{horizon}h"].notna()
        result[f"lactate_rise_{horizon}h_observed"] = lactate_observed.astype(int)
        result[f"lactate_delta_{horizon}h"] = result[f"lactate_last_{horizon}h"] - result["baseline_lactate_outcome"]
        result[f"lactate_max_delta_{horizon}h"] = result[f"lactate_max_{horizon}h"] - result["baseline_lactate_outcome"]
        result[f"lactate_rise_{horizon}h_flag"] = _binary_when_observed(
            result[f"lactate_delta_{horizon}h"].ge(LACTATE_RISE_ABSOLUTE), lactate_observed
        )
        result[f"lactate_cross_2_{horizon}h_flag"] = _binary_when_observed(
            result["baseline_lactate_outcome"].lt(2.0) & result[f"lactate_max_{horizon}h"].ge(2.0),
            lactate_observed,
        )

        creat_observed = result["baseline_creatinine_outcome"].gt(0) & result[f"creatinine_max_{horizon}h"].notna()
        result[f"creatinine_delta_{horizon}h"] = result[f"creatinine_max_{horizon}h"] - result["baseline_creatinine_outcome"]
        result[f"creatinine_ratio_{horizon}h"] = result[f"creatinine_max_{horizon}h"] / result["baseline_creatinine_outcome"]
        result[f"aki_creatinine_{horizon}h_observed"] = creat_observed.astype(int)
        result[f"aki_creatinine_{horizon}h_flag"] = _binary_when_observed(
            result[f"creatinine_delta_{horizon}h"].ge(CREATININE_AKI_ABSOLUTE)
            | result[f"creatinine_ratio_{horizon}h"].ge(CREATININE_AKI_RATIO),
            creat_observed,
        )

        bili_observed = result["baseline_bilirubin_total_outcome"].gt(0) & result[f"bilirubin_total_max_{horizon}h"].notna()
        result[f"bilirubin_delta_{horizon}h"] = result[f"bilirubin_total_max_{horizon}h"] - result["baseline_bilirubin_total_outcome"]
        result[f"bilirubin_ratio_{horizon}h"] = result[f"bilirubin_total_max_{horizon}h"] / result["baseline_bilirubin_total_outcome"]
        result[f"hepatic_lab_worsening_{horizon}h_observed"] = bili_observed.astype(int)
        result[f"hepatic_lab_worsening_{horizon}h_flag"] = _binary_when_observed(
            result[f"bilirubin_delta_{horizon}h"].ge(BILIRUBIN_WORSENING_ABSOLUTE)
            | result[f"bilirubin_ratio_{horizon}h"].ge(BILIRUBIN_WORSENING_RATIO),
            bili_observed,
        )

        result[f"pressor_initiation_{horizon}h_flag"] = _event_flag(post, "pressor", keys)
        result[f"mcs_{horizon}h_flag"] = _event_flag(post, "mcs", keys)

        vis_max = _max_measurement(post, "vis")
        result = _merge_value(result, vis_max, "value_numeric", f"vis_max_{horizon}h")
        vis_available = result["dataset"].astype(str).isin(vis_datasets)
        base_vis = pd.Series(0.0, index=result.index)
        if not baseline["vis"].empty:
            base_map = baseline["vis"].set_index(["dataset", "stay_id"])["value_numeric"]
            base_vis = pd.Series(
                [base_map.get((d, s), 0.0) for d, s in zip(result["dataset"], result["stay_id"])],
                index=result.index,
                dtype=float,
            )
        result[f"vis_observed_{horizon}h"] = vis_available.astype(int)
        result[f"vis_baseline_max_{horizon}h"] = base_vis.where(vis_available)
        result[f"vis_max_{horizon}h"] = result[f"vis_max_{horizon}h"].fillna(0).where(vis_available)
        result[f"vis_delta_{horizon}h"] = result[f"vis_max_{horizon}h"] - result[f"vis_baseline_max_{horizon}h"]
        result[f"vis_rise_{horizon}h_flag"] = _binary_when_observed(
            result[f"vis_delta_{horizon}h"].gt(0), vis_available
        )

        urine = post.loc[post["concept"].eq("urine_output") & post["value_numeric"].gt(0)]
        urine_sum = (
            urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
            if not urine.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
        )
        result = _merge_value(result, urine_sum, "value_numeric", f"urine_output_{horizon}h_ml")
        result[f"urine_output_{horizon}h_observed_hours"] = (
            result[f"urine_output_{horizon}h_covered_2h_bins"] * 2.0
        )
        result[f"urine_output_rate_{horizon}h_ml_h"] = (
            result[f"urine_output_{horizon}h_ml"]
            / result[f"urine_output_{horizon}h_observed_hours"].replace(
                0, np.nan
            )
        )
        urine_observed = (
            result["dataset"].astype(str).isin(urine_datasets)
            & result["baseline_urine_output_ml"].notna()
            & result[f"urine_output_{horizon}h_ml"].notna()
        )
        result[f"urine_output_{horizon}h_observed"] = urine_observed.astype(int)
        result[f"urine_output_decline_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"].le(
                result["baseline_urine_output_rate_ml_h"] * URINE_RATE_DECLINE_FRACTION
            ),
            urine_observed,
        )
        weight_observed = urine_observed & result["baseline_weight_outcome"].gt(0)
        result[f"oliguria_kdigo_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"].div(result["baseline_weight_outcome"]).lt(
                KDIGO_OLIGURIA_ML_KG_H
            ),
            weight_observed,
        )

        organ_components = [
            result[f"aki_creatinine_{horizon}h_flag"],
            result[f"hepatic_lab_worsening_{horizon}h_flag"],
        ]
        organ_frame = pd.concat(organ_components, axis=1)
        organ_observed = organ_frame.notna().any(axis=1)
        result[f"organ_lab_worsening_{horizon}h_observed"] = organ_observed.astype(int)
        result[f"organ_lab_worsening_{horizon}h_flag"] = _binary_when_observed(
            organ_frame.fillna(0).max(axis=1).gt(0), organ_observed
        )

        components = pd.concat(
            [
                result[f"lactate_rise_{horizon}h_flag"],
                result[f"vis_rise_{horizon}h_flag"],
                result[f"urine_output_decline_proxy_{horizon}h_flag"],
                result[f"organ_lab_worsening_{horizon}h_flag"],
                result[f"mcs_{horizon}h_flag"].astype(float),
            ],
            axis=1,
        )
        composite_observed = components.notna().any(axis=1)
        result[f"early_decompensation_{horizon}h_observed"] = composite_observed.astype(int)
        result[f"early_decompensation_{horizon}h_component_count"] = components.fillna(0).sum(axis=1)
        result[f"early_decompensation_{horizon}h_flag"] = _binary_when_observed(
            components.fillna(0).max(axis=1).gt(0), composite_observed
        )
        result[f"outcome_window_start_minutes_{horizon}h"] = LANDMARK_MINUTES
        result[f"outcome_window_end_minutes_{horizon}h"] = upper

    result["endpoint_definition_version"] = "spo2_protocol_v1.0"
    return result


def _conservative_any(flags: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    """Return (value, observed) without converting missing domains to zero.

    A positive component establishes a positive composite.  A negative
    composite is emitted only when every required component is observed.
    """
    frame = pd.concat(flags, axis=1)
    any_event = frame.eq(1).any(axis=1)
    complete = frame.notna().all(axis=1)
    observed = any_event | complete
    value = pd.Series(np.nan, index=frame.index, dtype=float)
    value.loc[any_event] = 1.0
    value.loc[complete & ~any_event] = 0.0
    return value, observed


def compute_early_decompensation_outcomes(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
    *,
    horizons: tuple[int, ...] = POST_LANDMARK_HORIZONS_HOURS,
) -> pd.DataFrame:
    """Derive censoring-aware, source-aware 12/24-hour outcomes.

    Exact landmark measurements belong to baseline.  Fixed-horizon outcomes
    are unavailable after ICU discharge/transfer or death before the horizon.
    Source completeness comes from per-stay cohort metadata; a missing clinical
    measurement is never silently converted to a non-event.
    """
    keys = _keys_from(stay_index if stay_index is not None else cohort_df, cohort_df)
    result = keys.copy()
    cohort = cohort_df.copy()
    if "dataset" not in cohort:
        cohort["dataset"] = "unknown"
    context_columns = [
        column
        for column in (
            "person_id",
            "followup_end_offset_minutes",
            "death_offset_minutes",
            "mcs_source_available",
            "pressor_source_available",
            "vis_source_available",
            "urine_output_source_available",
            "baseline_vasoactive_flag",
            "baseline_mcs_flag",
            "admission_weight_kg",
        )
        if column in cohort
    ]
    if context_columns:
        result = result.merge(
            cohort[["dataset", "stay_id", *context_columns]].drop_duplicates(
                ["dataset", "stay_id"]
            ),
            on=["dataset", "stay_id"],
            how="left",
            validate="one_to_one",
        )
    result["mcs_endpoint_method"] = np.where(
        result["dataset"].astype(str).eq("eicu"),
        "first_device_treatment_documentation_proxy",
        "device_line_procedure_start_or_first_operational_chart_documentation_proxy",
    )
    result["pressor_endpoint_method"] = np.where(
        result["dataset"].astype(str).eq("eicu"),
        "first_post_landmark_infusion_documentation_proxy_without_prelandmark_record",
        "infusion_off_to_on_transition_with_5_minute_restart_grace",
    )

    events = events_df.copy()
    if events.empty:
        events = pd.DataFrame(
            columns=["dataset", "stay_id", "concept", "offset_minutes", "value_numeric"]
        )
    if "dataset" not in events:
        events["dataset"] = "unknown"
    for column in ("concept", "offset_minutes", "value_numeric"):
        if column not in events:
            events[column] = "" if column == "concept" else np.nan
    events["concept"] = events["concept"].fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    events["value_numeric"] = pd.to_numeric(events["value_numeric"], errors="coerce")

    plausible_bounds: dict[str, tuple[float, float]] = {
        "lactate": (0.0, 30.0),
        "ph": (6.5, 8.0),
        "creatinine": (0.1, 30.0),
        "bilirubin_total": (0.0, 60.0),
        "ast": (0.0, 20_000.0),
        "alt": (0.0, 20_000.0),
        "inr": (0.3, 20.0),
        "platelets": (1.0, 2_000.0),
        "weight": (20.0, 400.0),
        "vis": (0.0, 1_000.0),
        "urine_output": (0.0, 5_000.0),
        "troponin_t": (0.0, 1_000.0),
        "troponin_i": (0.0, 1_000.0),
    }
    measurement_mask = pd.Series(True, index=events.index)
    for concept, (lower, upper) in plausible_bounds.items():
        concept_mask = events["concept"].eq(concept)
        measurement_mask.loc[concept_mask] = events.loc[
            concept_mask, "value_numeric"
        ].between(lower, upper, inclusive="both")
    events = events.loc[measurement_mask].copy()
    baseline_events = events.loc[
        events["offset_minutes"].ge(0)
        & events["offset_minutes"].le(LANDMARK_MINUTES)
    ].copy()

    baseline_concepts = (
        "lactate",
        "ph",
        "creatinine",
        "bilirubin_total",
        "ast",
        "alt",
        "inr",
        "platelets",
        "weight",
        "vis",
    )
    baseline = {
        concept: _last_measurement(baseline_events, concept)
        for concept in baseline_concepts
    }
    for concept in baseline_concepts:
        result = _merge_value(
            result,
            baseline[concept],
            "value_numeric",
            f"baseline_{concept}_outcome",
        )
    result = _merge_value(
        result,
        _last_measurement_any(baseline_events, ("troponin_t", "troponin_i")),
        "value_numeric",
        "baseline_troponin_outcome",
    )
    for assay in ("troponin_t", "troponin_i"):
        result = _merge_value(
            result,
            _last_measurement(baseline_events, assay),
            "value_numeric",
            f"baseline_{assay}_outcome",
        )
    if "admission_weight_kg" in result:
        result["baseline_weight_outcome"] = result["baseline_weight_outcome"].combine_first(
            pd.to_numeric(result["admission_weight_kg"], errors="coerce")
        )

    baseline_urine = baseline_events.loc[baseline_events["concept"].eq("urine_output")]
    baseline_urine_sum = (
        baseline_urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
        if not baseline_urine.empty
        else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    )
    baseline_urine_count = (
        baseline_urine.groupby(["dataset", "stay_id"], as_index=False).size()
        if not baseline_urine.empty
        else pd.DataFrame(columns=["dataset", "stay_id", "size"])
    )
    baseline_urine_bins = (
        baseline_urine.assign(
            # Right-closed two-hour intervals: [0,120], (120,240].
            # A measurement exactly on a boundary must not create a spurious
            # extra coverage bin.
            _bin=np.maximum(
                np.ceil(baseline_urine["offset_minutes"] / 120.0) - 1.0,
                0.0,
            )
        )
        .groupby(["dataset", "stay_id"], as_index=False)["_bin"]
        .nunique()
        if not baseline_urine.empty
        else pd.DataFrame(columns=["dataset", "stay_id", "_bin"])
    )
    result = _merge_value(result, baseline_urine_sum, "value_numeric", "baseline_urine_output_ml")
    result = _merge_value(result, baseline_urine_count, "size", "baseline_urine_output_count")
    result = _merge_value(
        result,
        baseline_urine_bins,
        "_bin",
        "baseline_urine_output_covered_2h_bins",
    )
    result["baseline_urine_output_coverage_fraction"] = (
        result["baseline_urine_output_covered_2h_bins"] / 2.0
    ).clip(upper=1.0)
    result["baseline_urine_output_observed_hours"] = (
        result["baseline_urine_output_covered_2h_bins"] * 2.0
    )
    result["baseline_urine_output_rate_ml_h"] = (
        result["baseline_urine_output_ml"]
        / result["baseline_urine_output_observed_hours"].replace(0, np.nan)
    )

    event_datasets = {
        concept: set(
            events.loc[events["concept"].eq(concept), "dataset"].dropna().astype(str)
        )
        for concept in ("vis", "mcs", "urine_output", "pressor", "pressor_initiation")
    }
    event_datasets["mcs"] = set(
        events.loc[
            events["concept"].eq("mcs")
            | events["concept"].str.startswith("mcs_context_"),
            "dataset",
        ]
        .dropna()
        .astype(str)
    )

    def source_available(column: str, concept: str) -> pd.Series:
        if column in result:
            return pd.to_numeric(result[column], errors="coerce").fillna(0).gt(0)
        return result["dataset"].astype(str).isin(event_datasets.get(concept, set()))

    for horizon in horizons:
        upper = LANDMARK_MINUTES + int(horizon * 60)
        followup = (
            pd.to_numeric(result["followup_end_offset_minutes"], errors="coerce")
            if "followup_end_offset_minutes" in result
            else pd.Series(np.inf, index=result.index)
        )
        death = (
            pd.to_numeric(result["death_offset_minutes"], errors="coerce")
            if "death_offset_minutes" in result
            else pd.Series(np.nan, index=result.index)
        )
        # A present-but-missing follow-up timestamp cannot establish a fixed-
        # horizon non-event.  When the column is entirely absent (synthetic or
        # legacy inputs), ``followup`` is explicitly set to infinity above.
        # A death timestamped exactly at the horizon occurs after the complete
        # outcome interval has been observed; only death strictly before the
        # boundary censors a fixed-window negative.
        complete_followup = followup.ge(upper) & (
            death.isna() | death.ge(upper)
        )
        result[f"complete_followup_{horizon}h"] = complete_followup.astype(int)
        death_source_available = "death_offset_minutes" in result.columns
        result[f"death_{horizon}h_observed"] = int(death_source_available)
        result[f"death_{horizon}h_flag"] = (
            death.gt(LANDMARK_MINUTES) & death.le(upper)
        ).astype(float) if death_source_available else np.nan
        result[f"death_{horizon}h_method"] = (
            "in_hospital_death_time_or_discharge_status"
            if death_source_available
            else "unavailable"
        )
        censor_limits = result[["dataset", "stay_id"]].copy()
        censor_limits["_outcome_observation_end"] = float(upper)
        valid_followup = followup.notna()
        censor_limits.loc[valid_followup, "_outcome_observation_end"] = np.minimum(
            censor_limits.loc[valid_followup, "_outcome_observation_end"],
            followup.loc[valid_followup],
        )
        valid_death = death.notna()
        censor_limits.loc[valid_death, "_outcome_observation_end"] = np.minimum(
            censor_limits.loc[valid_death, "_outcome_observation_end"],
            death.loc[valid_death],
        )
        post = events.loc[
            events["offset_minutes"].gt(LANDMARK_MINUTES)
            & events["offset_minutes"].le(upper)
        ].merge(
            censor_limits,
            on=["dataset", "stay_id"],
            how="inner",
            validate="many_to_one",
        )
        post = post.loc[
            post["offset_minutes"].le(post["_outcome_observation_end"])
        ].drop(columns="_outcome_observation_end")
        observed_through_horizon = events.loc[
            events["offset_minutes"].le(upper)
        ].merge(
            censor_limits,
            on=["dataset", "stay_id"],
            how="inner",
            validate="many_to_one",
        )
        observed_through_horizon = observed_through_horizon.loc[
            observed_through_horizon["offset_minutes"].le(
                observed_through_horizon["_outcome_observation_end"]
            )
        ].drop(columns="_outcome_observation_end")

        for concept in ("lactate", "creatinine", "bilirubin_total", "ast", "alt", "inr"):
            result = _merge_value(
                result,
                _last_measurement(post, concept),
                "value_numeric",
                f"{concept}_last_{horizon}h",
            )
            result = _merge_value(
                result,
                _max_measurement(post, concept),
                "value_numeric",
                f"{concept}_max_{horizon}h",
            )
        result = _merge_value(
            result,
            _max_measurement_any(post, ("troponin_t", "troponin_i")),
            "value_numeric",
            f"troponin_max_{horizon}h",
        )
        for assay in ("troponin_t", "troponin_i"):
            result = _merge_value(
                result,
                _max_measurement(post, assay),
                "value_numeric",
                f"{assay}_max_{horizon}h",
            )
        platelet = post.loc[post["concept"].eq("platelets")].dropna(subset=["value_numeric"])
        platelet_min = (
            platelet.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].min()
            if not platelet.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
        )
        result = _merge_value(
            result, platelet_min, "value_numeric", f"platelets_min_{horizon}h"
        )
        result = result.copy()

        lactate_ascertainable = (
            result["baseline_lactate_outcome"].notna()
            & result[f"lactate_last_{horizon}h"].notna()
        )
        result[f"lactate_delta_{horizon}h"] = (
            result[f"lactate_last_{horizon}h"] - result["baseline_lactate_outcome"]
        )
        result[f"lactate_max_delta_{horizon}h"] = (
            result[f"lactate_max_{horizon}h"] - result["baseline_lactate_outcome"]
        )
        # The locked primary lactate endpoint uses the *last* value in the
        # complete fixed window, so early censoring prevents classification in
        # either direction.  Monotone any-time/max variants below can retain a
        # confirmed event observed before censoring.
        lactate_observed = lactate_ascertainable & complete_followup
        lactate_flag = _binary_when_observed(
            result[f"lactate_delta_{horizon}h"].ge(LACTATE_RISE_ABSOLUTE),
            lactate_observed,
        )
        result[f"lactate_rise_{horizon}h_flag"] = lactate_flag
        result[f"lactate_rise_{horizon}h_observed"] = lactate_observed.astype(int)
        result[f"lactate_rise_1_0_{horizon}h_flag"] = _binary_when_observed(
            result[f"lactate_delta_{horizon}h"].ge(1.0),
            lactate_observed,
        )
        lactate_max_ascertainable = (
            result["baseline_lactate_outcome"].notna()
            & result[f"lactate_max_{horizon}h"].notna()
        )
        result[f"lactate_max_rise_{horizon}h_flag"] = _binary_event_with_censoring(
            result[f"lactate_max_delta_{horizon}h"].ge(LACTATE_RISE_ABSOLUTE),
            lactate_max_ascertainable,
            complete_followup,
        )[0]
        result[f"lactate_cross_2_{horizon}h_flag"] = _binary_event_with_censoring(
            result["baseline_lactate_outcome"].lt(2.0)
            & result[f"lactate_max_{horizon}h"].ge(2.0),
            lactate_max_ascertainable,
            complete_followup,
        )[0]

        creat_ascertainable = (
            result["baseline_creatinine_outcome"].gt(0)
            & result[f"creatinine_max_{horizon}h"].notna()
        )
        result[f"creatinine_delta_{horizon}h"] = (
            result[f"creatinine_max_{horizon}h"] - result["baseline_creatinine_outcome"]
        )
        result[f"creatinine_ratio_{horizon}h"] = (
            result[f"creatinine_max_{horizon}h"] / result["baseline_creatinine_outcome"]
        )
        creat_flag, creat_observed = _binary_event_with_censoring(
            result[f"creatinine_delta_{horizon}h"].ge(CREATININE_AKI_ABSOLUTE)
            | result[f"creatinine_ratio_{horizon}h"].ge(CREATININE_AKI_RATIO),
            creat_ascertainable,
            complete_followup,
        )
        result[f"aki_creatinine_{horizon}h_flag"] = creat_flag
        result[f"aki_creatinine_{horizon}h_observed"] = creat_observed.astype(int)

        # T and I are distinct assays with non-interchangeable scales.  Match
        # each post-landmark value to the same assay at baseline, then combine
        # the assay-specific event indicators; never divide T by I (or vice
        # versa) merely because it was the latest measurement.
        assay_ratios: list[pd.Series] = []
        assay_ascertainable: list[pd.Series] = []
        assay_events: list[pd.Series] = []
        for assay in ("troponin_t", "troponin_i"):
            baseline_column = f"baseline_{assay}_outcome"
            post_column = f"{assay}_max_{horizon}h"
            ratio_column = f"{assay}_ratio_{horizon}h"
            result[ratio_column] = result[post_column] / result[
                baseline_column
            ].replace(0, np.nan)
            ascertainable = result[baseline_column].gt(0) & result[post_column].notna()
            assay_ratios.append(result[ratio_column])
            assay_ascertainable.append(ascertainable)
            assay_events.append(ascertainable & result[ratio_column].ge(1.5))
        troponin_ascertainable = pd.concat(assay_ascertainable, axis=1).any(axis=1)
        troponin_event = pd.concat(assay_events, axis=1).any(axis=1)
        result[f"troponin_ratio_{horizon}h"] = pd.concat(
            assay_ratios, axis=1
        ).max(axis=1, skipna=True)
        troponin_flag, troponin_observed = _binary_event_with_censoring(
            troponin_event,
            troponin_ascertainable,
            complete_followup,
        )
        result[f"troponin_relative_rise_{horizon}h_flag"] = troponin_flag
        result[f"troponin_relative_rise_{horizon}h_observed"] = troponin_observed.astype(int)
        result[f"troponin_endpoint_method_{horizon}h"] = "assay_matched_relative_rise"
        result = result.copy()

        bili_ascertainable = (
            result["baseline_bilirubin_total_outcome"].notna()
            & result[f"bilirubin_total_max_{horizon}h"].notna()
        )
        result[f"bilirubin_delta_{horizon}h"] = (
            result[f"bilirubin_total_max_{horizon}h"]
            - result["baseline_bilirubin_total_outcome"]
        )
        result[f"bilirubin_ratio_{horizon}h"] = (
            result[f"bilirubin_total_max_{horizon}h"]
            / result["baseline_bilirubin_total_outcome"].replace(0, np.nan)
        )
        bili_flag, bili_observed = _binary_event_with_censoring(
            result[f"bilirubin_delta_{horizon}h"].ge(BILIRUBIN_WORSENING_ABSOLUTE)
            | result[f"bilirubin_ratio_{horizon}h"].ge(BILIRUBIN_WORSENING_RATIO),
            bili_ascertainable,
            complete_followup,
        )
        result[f"hepatic_lab_worsening_{horizon}h_flag"] = bili_flag
        result[f"hepatic_lab_worsening_{horizon}h_observed"] = bili_observed.astype(int)

        hepatic_flags = [result[f"hepatic_lab_worsening_{horizon}h_flag"]]
        for concept in ("ast", "alt"):
            ascertainable = (
                result[f"baseline_{concept}_outcome"].gt(0)
                & result[f"{concept}_max_{horizon}h"].notna()
            )
            delta_ratio = result[f"{concept}_max_{horizon}h"] / result[
                f"baseline_{concept}_outcome"
            ]
            result[f"{concept}_ratio_{horizon}h"] = delta_ratio
            result[f"{concept}_delta_{horizon}h"] = (
                result[f"{concept}_max_{horizon}h"]
                - result[f"baseline_{concept}_outcome"]
            )
            flag, observed = _binary_event_with_censoring(
                (
                    result[f"baseline_{concept}_outcome"].lt(200)
                    & result[f"{concept}_max_{horizon}h"].ge(200)
                )
                | delta_ratio.ge(2.0),
                ascertainable,
                complete_followup,
            )
            result[f"{concept}_injury_{horizon}h_observed"] = observed.astype(int)
            result[f"{concept}_injury_{horizon}h_flag"] = flag
            hepatic_flags.append(flag)
        inr_ascertainable = (
            result["baseline_inr_outcome"].gt(0)
            & result[f"inr_max_{horizon}h"].notna()
        )
        result[f"inr_delta_{horizon}h"] = (
            result[f"inr_max_{horizon}h"] - result["baseline_inr_outcome"]
        )
        result[f"inr_ratio_{horizon}h"] = (
            result[f"inr_max_{horizon}h"]
            / result["baseline_inr_outcome"].replace(0, np.nan)
        )
        inr_flag, inr_observed = _binary_event_with_censoring(
            result[f"inr_max_{horizon}h"].sub(result["baseline_inr_outcome"]).ge(0.3)
            | result[f"inr_max_{horizon}h"].div(result["baseline_inr_outcome"]).ge(1.5),
            inr_ascertainable,
            complete_followup,
        )
        result[f"inr_injury_{horizon}h_observed"] = inr_observed.astype(int)
        result[f"inr_injury_{horizon}h_flag"] = inr_flag
        hepatic_flags.append(inr_flag)
        hepatic_extended, hepatic_extended_observed = _conservative_any(hepatic_flags)
        result[f"hepatic_injury_extended_{horizon}h_flag"] = hepatic_extended
        result[f"hepatic_injury_extended_{horizon}h_observed"] = hepatic_extended_observed.astype(int)

        platelet_ascertainable = (
            result["baseline_platelets_outcome"].gt(0)
            & result[f"platelets_min_{horizon}h"].notna()
        )
        result[f"platelet_ratio_{horizon}h"] = (
            result[f"platelets_min_{horizon}h"]
            / result["baseline_platelets_outcome"].replace(0, np.nan)
        )
        result[f"platelet_decline_fraction_{horizon}h"] = (
            1.0 - result[f"platelet_ratio_{horizon}h"]
        )
        platelet_flag, platelet_observed = _binary_event_with_censoring(
            (
                result["baseline_platelets_outcome"].ge(100)
                & result[f"platelets_min_{horizon}h"].lt(100)
            )
            | result[f"platelets_min_{horizon}h"]
            .div(result["baseline_platelets_outcome"])
            .le(0.70),
            platelet_ascertainable,
            complete_followup,
        )
        result[f"platelet_injury_{horizon}h_observed"] = platelet_observed.astype(int)
        result[f"platelet_injury_{horizon}h_flag"] = platelet_flag
        result = result.copy()

        mcs_source = source_available("mcs_source_available", "mcs")
        if "baseline_mcs_flag" in result:
            baseline_mcs = pd.to_numeric(
                result["baseline_mcs_flag"], errors="coerce"
            ).fillna(0).gt(0)
        else:
            baseline_mcs = _mcs_event_flag(baseline_events, keys).astype(bool)
        mcs_at_risk = ~baseline_mcs
        result[f"mcs_incident_risk_set_{horizon}h_flag"] = mcs_at_risk.astype(int)
        mcs_event = _mcs_event_flag(post, keys).astype(bool) & mcs_at_risk
        mcs_flag, mcs_observed = _binary_event_with_censoring(
            mcs_event, mcs_source & mcs_at_risk, complete_followup
        )
        result[f"mcs_{horizon}h_observed"] = mcs_observed.astype(int)
        result[f"mcs_{horizon}h_flag"] = mcs_flag
        pressor_source = source_available("pressor_source_available", "pressor")
        pressor_event = _event_flag(post, "pressor_initiation", keys).astype(bool)
        pressor_flag, pressor_observed = _binary_event_with_censoring(
            pressor_event, pressor_source, complete_followup
        )
        result[f"pressor_initiation_{horizon}h_observed"] = pressor_observed.astype(int)
        result[f"pressor_initiation_{horizon}h_flag"] = pressor_flag

        vis_source = source_available("vis_source_available", "vis")
        incompatible_vis = observed_through_horizon.loc[
            observed_through_horizon["concept"].eq("vis_rate_unstandardized"),
            ["dataset", "stay_id"],
        ].drop_duplicates()
        if not incompatible_vis.empty:
            invalid_keys = set(
                zip(
                    incompatible_vis["dataset"].astype(str),
                    incompatible_vis["stay_id"],
                )
            )
            vis_source = vis_source & pd.Series(
                [
                    (str(dataset), stay_id) not in invalid_keys
                    for dataset, stay_id in zip(result["dataset"], result["stay_id"])
                ],
                index=result.index,
            )
        result[f"vis_rate_units_complete_{horizon}h_flag"] = vis_source.astype(int)
        vis_base = pd.Series(0.0, index=result.index, dtype=float)
        if not baseline["vis"].empty:
            vis_map = baseline["vis"].set_index(["dataset", "stay_id"])["value_numeric"]
            vis_base = pd.Series(
                [vis_map.get((dataset, stay_id), 0.0) for dataset, stay_id in zip(result["dataset"], result["stay_id"])],
                index=result.index,
                dtype=float,
            )
        result[f"vis_baseline_max_{horizon}h"] = vis_base.where(vis_source)
        result = _merge_value(
            result, _max_measurement(post, "vis"), "value_numeric", f"vis_max_{horizon}h"
        )
        result[f"vis_max_{horizon}h"] = result[f"vis_max_{horizon}h"].fillna(0).where(vis_source)
        result[f"vis_delta_{horizon}h"] = (
            result[f"vis_max_{horizon}h"] - result[f"vis_baseline_max_{horizon}h"]
        )
        vis_flag, vis_observed = _binary_event_with_censoring(
            result[f"vis_delta_{horizon}h"].gt(0),
            vis_source,
            complete_followup,
        )
        result[f"vis_observed_{horizon}h"] = vis_observed.astype(int)
        result[f"vis_rise_{horizon}h_flag"] = vis_flag
        result = result.copy()

        urine = post.loc[post["concept"].eq("urine_output")]
        urine_sum = (
            urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
            if not urine.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
        )
        urine_count = (
            urine.groupby(["dataset", "stay_id"], as_index=False).size()
            if not urine.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "size"])
        )
        urine_bins = (
            urine.assign(
                _bin=np.maximum(
                    np.ceil(
                        (urine["offset_minutes"] - LANDMARK_MINUTES) / 120.0
                    )
                    - 1.0,
                    0.0,
                )
            )
            .groupby(["dataset", "stay_id"], as_index=False)["_bin"]
            .nunique()
            if not urine.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "_bin"])
        )
        result = _merge_value(result, urine_sum, "value_numeric", f"urine_output_{horizon}h_ml")
        result = _merge_value(result, urine_count, "size", f"urine_output_{horizon}h_count")
        result = _merge_value(result, urine_bins, "_bin", f"urine_output_{horizon}h_covered_2h_bins")
        result[f"urine_output_{horizon}h_observed_hours"] = (
            result[f"urine_output_{horizon}h_covered_2h_bins"] * 2.0
        )
        result[f"urine_output_rate_{horizon}h_ml_h"] = (
            result[f"urine_output_{horizon}h_ml"]
            / result[f"urine_output_{horizon}h_observed_hours"].replace(
                0, np.nan
            )
        )
        result[f"urine_output_rate_ratio_{horizon}h"] = (
            result[f"urine_output_rate_{horizon}h_ml_h"]
            / result["baseline_urine_output_rate_ml_h"].replace(0, np.nan)
        )
        result[f"urine_output_decline_fraction_{horizon}h"] = (
            1.0 - result[f"urine_output_rate_ratio_{horizon}h"]
        )
        result[f"urine_output_{horizon}h_coverage_fraction"] = (
            result[f"urine_output_{horizon}h_covered_2h_bins"] / max(horizon / 2.0, 1.0)
        ).clip(upper=1.0)
        urine_source = source_available("urine_output_source_available", "urine_output")
        urine_measurement_observed = (
            complete_followup
            & urine_source
            & result[f"urine_output_{horizon}h_count"].gt(0)
        )
        urine_decline_observed = (
            urine_measurement_observed
            & result["baseline_urine_output_count"].gt(0)
            & result["baseline_urine_output_rate_ml_h"].gt(0)
            & result["baseline_urine_output_coverage_fraction"].ge(0.5)
            & result[f"urine_output_{horizon}h_coverage_fraction"].ge(0.5)
        )
        result[f"urine_output_{horizon}h_observed"] = urine_measurement_observed.astype(int)
        result[f"urine_output_decline_proxy_{horizon}h_observed"] = urine_decline_observed.astype(int)
        result[f"urine_output_decline_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"].le(
                result["baseline_urine_output_rate_ml_h"] * URINE_RATE_DECLINE_FRACTION
            ),
            urine_decline_observed,
        )
        kdigo_observed = (
            urine_measurement_observed
            & result["baseline_weight_outcome"].gt(0)
            & result[f"urine_output_{horizon}h_coverage_fraction"].ge(0.5)
        )
        result[f"oliguria_kdigo_proxy_{horizon}h_observed"] = kdigo_observed.astype(int)
        result[f"oliguria_kdigo_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"]
            .div(result["baseline_weight_outcome"])
            .lt(KDIGO_OLIGURIA_ML_KG_H),
            kdigo_observed,
        )
        result = result.copy()

        organ_flag, organ_observed = _conservative_any(
            [
                result[f"aki_creatinine_{horizon}h_flag"],
                result[f"hepatic_injury_extended_{horizon}h_flag"],
                result[f"platelet_injury_{horizon}h_flag"],
            ]
        )
        result[f"organ_lab_worsening_{horizon}h_flag"] = organ_flag
        result[f"organ_lab_worsening_{horizon}h_observed"] = organ_observed.astype(int)

        # Harmonized composite deliberately uses domains ascertainable in both
        # datasets. VIS and urine output remain separately reported endpoints.
        composite, composite_observed = _conservative_any(
            [
                result[f"lactate_rise_{horizon}h_flag"],
                result[f"organ_lab_worsening_{horizon}h_flag"],
                result[f"mcs_{horizon}h_flag"],
            ]
        )
        result[f"early_decompensation_{horizon}h_flag"] = composite
        result[f"early_decompensation_{horizon}h_observed"] = composite_observed.astype(int)
        result[f"early_decompensation_{horizon}h_component_count"] = pd.concat(
            [
                result[f"lactate_rise_{horizon}h_flag"],
                result[f"organ_lab_worsening_{horizon}h_flag"],
                result[f"mcs_{horizon}h_flag"],
            ],
            axis=1,
        ).eq(1).sum(axis=1)
        plus_vis, plus_vis_observed = _conservative_any(
            [composite, result[f"vis_rise_{horizon}h_flag"]]
        )
        result[f"early_decompensation_plus_vis_{horizon}h_flag"] = plus_vis
        result[f"early_decompensation_plus_vis_{horizon}h_observed"] = plus_vis_observed.astype(int)
        result[f"outcome_window_start_minutes_{horizon}h"] = LANDMARK_MINUTES
        result[f"outcome_window_end_minutes_{horizon}h"] = upper
        result = result.copy()

    result["endpoint_definition_version"] = "spo2_protocol_v2.2"
    return result


def build_endpoint_completeness_audit(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Report coverage and event floors for every prespecified endpoint."""
    tiers = {
        **{name: "primary" for name in PRIMARY_ENDPOINTS},
        **{name: "secondary_or_exploratory" for name in SECONDARY_ENDPOINTS},
    }
    rows: list[dict[str, Any]] = []
    datasets = sorted(analysis_df.get("dataset", pd.Series(["all"])).dropna().astype(str).unique())
    for dataset in datasets:
        frame = analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)] if "dataset" in analysis_df else analysis_df
        for endpoint, tier in tiers.items():
            if endpoint not in frame:
                rows.append({"dataset": dataset, "endpoint": endpoint, "tier": tier, "status": "unavailable", "reason": "column_missing"})
                continue
            y = pd.to_numeric(frame[endpoint], errors="coerce")
            n = int(y.notna().sum())
            events = int(y.fillna(0).sum())
            status = (
                "unavailable"
                if n == 0
                else "adequate"
                if n >= 200 and events >= 20 and (n - events) >= 20
                else "fragile_or_underpowered"
            )
            reason = (
                "no_observed_endpoint_values"
                if status == "unavailable"
                else ""
                if status == "adequate"
                else "below_minimum_200_observed_or_20_events_and_20_non_events"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "tier": tier,
                    "status": status,
                    "n_total": int(len(frame)),
                    "n_observed": n,
                    "coverage_fraction": float(n / len(frame)) if len(frame) else math.nan,
                    "events": events,
                    "non_events": int(n - events),
                    "event_rate": float(y.mean()) if n else math.nan,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _group_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    dataset = frame.get("dataset", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    if "person_id" in frame and frame["person_id"].notna().any():
        person = frame["person_id"].astype("string")
        if "stay_id" in frame:
            fallback = "stay:" + frame["stay_id"].astype("string")
        else:
            fallback = pd.Series(
                [f"row:{index}" for index in range(len(frame))],
                index=frame.index,
                dtype="string",
            )
        person = person.where(
            person.notna() & person.str.strip().ne(""), fallback
        )
        return (
            dataset + ":person:" + person.astype(str),
            "dataset_qualified_person_id_with_stay_fallback",
        )
    if "stay_id" not in frame:
        raise ValueError("Patient-grouped evaluation requires person_id or stay_id")
    return dataset + ":stay:" + frame["stay_id"].astype(str), "dataset_qualified_stay_id_fallback"


def _coerce_categorical_matrix(values: Any) -> np.ndarray:
    """Normalize heterogeneous source categories inside each model pipeline."""
    normalized = pd.DataFrame(values).astype("string").fillna("__missing__")
    return normalized.astype(str).to_numpy(dtype=object)


def _build_model_pipeline(frame: pd.DataFrame, columns: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    categorical = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = [column for column in columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "normalize_type",
                            FunctionTransformer(
                                _coerce_categorical_matrix,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")),
        ]
    )


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1] if index < bins - 1 else probability <= edges[index + 1]
        )
        if mask.any():
            total += float(mask.mean() * abs(y[mask].mean() - probability[mask].mean()))
    return total


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    intercept, slope = _calibration(y, probability)
    return {
        "auroc_oof": float(roc_auc_score(y, probability)),
        "auprc_oof": float(average_precision_score(y, probability)),
        "brier_oof": float(brier_score_loss(y, probability)),
        "ece_oof": _ece(y, probability),
        "calibration_intercept_oof": intercept,
        "calibration_slope_oof": slope,
    }


def _bootstrap_metric_ci(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    reference_probability: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    samples: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "brier": [],
        "ece": [],
        "calibration_intercept": [],
        "calibration_slope": [],
    }
    delta_aurocs: list[float] = []
    delta_auprcs: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_group[group] for group in sampled])
        y_boot = y[index]
        if np.unique(y_boot).size < 2:
            continue
        p_boot = probability[index]
        auroc = float(roc_auc_score(y_boot, p_boot))
        auprc = float(average_precision_score(y_boot, p_boot))
        samples["auroc"].append(auroc)
        samples["auprc"].append(auprc)
        samples["brier"].append(float(np.mean((p_boot - y_boot) ** 2)))
        samples["ece"].append(_ece(y_boot, p_boot))
        try:
            intercept, slope = _calibration(y_boot, p_boot)
            samples["calibration_intercept"].append(intercept)
            samples["calibration_slope"].append(slope)
        except Exception:
            pass
        if reference_probability is not None:
            ref = reference_probability[index]
            delta_aurocs.append(auroc - float(roc_auc_score(y_boot, ref)))
            delta_auprcs.append(auprc - float(average_precision_score(y_boot, ref)))

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return math.nan, math.nan
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    result: dict[str, float] = {
        "bootstrap_repetitions_requested": float(repetitions),
        "bootstrap_repetitions_valid": float(len(samples["auroc"])),
    }
    for metric, values in samples.items():
        low, high = interval(values)
        result[f"{metric}_ci95_low"] = low
        result[f"{metric}_ci95_high"] = high
    if reference_probability is not None:
        d_auc_low, d_auc_high = interval(delta_aurocs)
        d_pr_low, d_pr_high = interval(delta_auprcs)
        result.update(
            {
                "delta_auroc_vs_reference_ci95_low": d_auc_low,
                "delta_auroc_vs_reference_ci95_high": d_auc_high,
                "delta_auprc_vs_reference_ci95_low": d_pr_low,
                "delta_auprc_vs_reference_ci95_high": d_pr_high,
                # Compatibility aliases retained for earlier consumers. The
                # v2.2 reference also contains sampling/missingness controls.
                "delta_auroc_vs_absolute_ci95_low": d_auc_low,
                "delta_auroc_vs_absolute_ci95_high": d_auc_high,
                "delta_auprc_vs_absolute_ci95_low": d_pr_low,
                "delta_auprc_vs_absolute_ci95_high": d_pr_high,
            }
        )
    return result


def _prespecified_outcomes(frame: pd.DataFrame) -> list[str]:
    ordered = [*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS]
    available = [name for name in ordered if name in frame]
    if available:
        return available
    # Compatibility for archived artifacts and parity fixtures only.  These
    # endpoints are never selected when protocol-v1 endpoints are present.
    return [
        name
        for name in ("target", "mcs_or_death_168h_flag", "death_168h_flag")
        if name in frame
    ]


def fit_grouped_incremental_models(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = 20,
    n_splits: int = 5,
    bootstrap_repetitions: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit prespecified nested models per dataset, then pooled secondarily.

    Every nested specification uses the same outcome-observed,
    dynamics-eligible rows. Preprocessing is learned inside each patient-grouped
    training fold.  The parsimonious instability signature is primary; the
    collinear full block is retained and explicitly flagged with EPV.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    measured = analysis_df.copy()
    if "spo2_plausible_count" in measured:
        measured = measured.loc[pd.to_numeric(measured["spo2_plausible_count"], errors="coerce").fillna(0).gt(0)].copy()
    if "spo2_dynamics_eligible_flag" in measured:
        measured = measured.loc[
            pd.to_numeric(measured["spo2_dynamics_eligible_flag"], errors="coerce")
            .fillna(0)
            .eq(1)
        ].copy()
    if "dataset" not in measured:
        measured["dataset"] = "unknown"
    dataset_values = sorted(measured["dataset"].dropna().astype(str).unique())
    scopes: list[tuple[str, pd.DataFrame]] = [
        (dataset, measured.loc[measured["dataset"].astype(str).eq(dataset)].copy())
        for dataset in dataset_values
    ]
    if len(dataset_values) > 1:
        scopes.append(("pooled_secondary", measured.copy()))
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for analysis_scope, scope_frame in scopes:
        controls = [
            column
            for column in CLINICAL_CONTROL_CANDIDATES
            if column in scope_frame
            and scope_frame[column].notna().any()
            and not (analysis_scope != "pooled_secondary" and column == "dataset")
        ]
        absolute = [
            column
            for column in ABSOLUTE_SPO2_FEATURES
            if column in scope_frame and scope_frame[column].notna().any()
        ]
        sampling = [
            column
            for column in SAMPLING_ADJUSTMENT_FEATURES
            if column in scope_frame and scope_frame[column].notna().any()
        ]
        parsimonious = [
            column
            for column in PARSIMONIOUS_INSTABILITY_FEATURES
            if column in scope_frame and scope_frame[column].notna().any()
        ]
        instability = [
            column
            for column in INSTABILITY_FEATURES
            if column in scope_frame and scope_frame[column].notna().any()
        ]
        specifications = {
            "clinical_only": controls,
            # This is the actual reference for the instability hypothesis:
            # clinical context + absolute oxygenation + monitoring intensity.
            "absolute_spo2": _dedupe([*controls, *absolute, *sampling]),
            "parsimonious_spo2_instability": _dedupe(
                [*controls, *absolute, *sampling, *parsimonious]
            ),
            "full_spo2_instability": _dedupe(
                [*controls, *absolute, *sampling, *instability]
            ),
        }
        for outcome in _prespecified_outcomes(scope_frame):
            valid = pd.to_numeric(scope_frame[outcome], errors="coerce").isin([0, 1])
            frame = scope_frame.loc[valid].reset_index(drop=True)
            y = pd.to_numeric(frame[outcome], errors="coerce").astype(int)
            groups, grouping = _group_series(frame)
            observed_dataset_count = (
                int(frame["dataset"].dropna().astype(str).nunique())
                if "dataset" in frame
                else int(not frame.empty)
            )
            if analysis_scope == "pooled_secondary" and observed_dataset_count < 2:
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "outcome": outcome,
                        "status": "skipped",
                        "n": int(len(y)),
                        "events": int(y.sum()),
                        "non_events": int((1 - y).sum()),
                        "observed_dataset_count": observed_dataset_count,
                        "analysis_population": "spo2_dynamics_eligible_common_sample",
                        "reason": "pooled endpoint not observed in at least two datasets",
                    }
                )
                continue
            event_groups = int(groups.loc[y.eq(1)].nunique())
            non_event_groups = int(groups.loc[y.eq(0)].nunique())
            max_splits = min(
                n_splits,
                int(groups.nunique()),
                event_groups,
                non_event_groups,
            )
            if (
                len(y) < min_rows
                or int(y.sum()) < min_events
                or int((1 - y).sum()) < min_events
                or max_splits < 2
            ):
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "outcome": outcome,
                        "status": "skipped",
                        "n": int(len(y)),
                        "events": int(y.sum()),
                        "non_events": int((1 - y).sum()),
                        "n_groups": int(groups.nunique()),
                        "analysis_population": "spo2_dynamics_eligible_common_sample",
                        "reason": "insufficient rows/events/non-events/patient groups",
                    }
                )
                continue

            folds: list[tuple[np.ndarray, np.ndarray]] | None = None
            splits = max_splits
            chosen_seed = random_state
            for candidate_splits in range(max_splits, 1, -1):
                for seed_offset in range(20):
                    splitter = StratifiedGroupKFold(
                        n_splits=candidate_splits,
                        shuffle=True,
                        random_state=random_state + seed_offset,
                    )
                    candidate = list(splitter.split(frame, y, groups))
                    if all(
                        y.iloc[train].nunique() == 2 and y.iloc[test].nunique() == 2
                        for train, test in candidate
                    ):
                        folds = candidate
                        splits = candidate_splits
                        chosen_seed = random_state + seed_offset
                        break
                if folds is not None:
                    break
            if folds is None:
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "outcome": outcome,
                        "status": "failed",
                        "n": int(len(y)),
                        "events": int(y.sum()),
                        "reason": "unable to construct all-two-class patient-grouped folds",
                    }
                )
                continue

            predictions = {
                name: np.full(len(frame), np.nan, dtype=float) for name in specifications
            }
            feature_counts = {name: [] for name in specifications}
            failure: str | None = None
            for train_index, test_index in folds:
                for model_name, columns in specifications.items():
                    if not columns:
                        failure = f"no features for {model_name}"
                        break
                    pipeline = _build_model_pipeline(frame, columns)
                    try:
                        pipeline.fit(frame.iloc[train_index][columns], y.iloc[train_index])
                        predictions[model_name][test_index] = pipeline.predict_proba(
                            frame.iloc[test_index][columns]
                        )[:, 1]
                        feature_counts[model_name].append(
                            int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))
                        )
                    except Exception as exc:
                        failure = f"{model_name}: {exc}"
                        break
                if failure:
                    break
            if failure or any(
                not np.isfinite(values).all() for values in predictions.values()
            ):
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "outcome": outcome,
                        "status": "failed",
                        "n": int(len(y)),
                        "events": int(y.sum()),
                        "reason": failure or "incomplete OOF predictions",
                    }
                )
                continue

            y_array = y.to_numpy()
            group_array = groups.to_numpy()
            absolute_probability = predictions["absolute_spo2"]
            absolute_metrics = _metrics(y_array, absolute_probability)
            for model_name, probability in predictions.items():
                metrics = _metrics(y_array, probability)
                reference = (
                    absolute_probability
                    if model_name in {
                        "parsimonious_spo2_instability",
                        "full_spo2_instability",
                    }
                    else None
                )
                ci = _bootstrap_metric_ci(
                    y_array,
                    probability,
                    group_array,
                    repetitions=bootstrap_repetitions,
                    seed=random_state + len(rows),
                    reference_probability=reference,
                )
                feature_count = max(feature_counts.get(model_name, [0]) or [0])
                minority_events = min(int(y.sum()), int((1 - y).sum()))
                epv = minority_events / feature_count if feature_count else math.nan
                row: dict[str, Any] = {
                    "analysis_scope": analysis_scope,
                    "outcome": outcome,
                    "model": model_name,
                    "status": "fit",
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "non_events": int((1 - y).sum()),
                    "n_groups": int(groups.nunique()),
                    "n_stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else int(len(frame)),
                    "n_patients": int(groups.nunique()),
                    "observed_dataset_count": observed_dataset_count,
                    "grouping": grouping,
                    "cv_splitter": "StratifiedGroupKFold",
                    "cv_folds": splits,
                    "cv_random_state": chosen_seed,
                    "preprocessing_scope": "fit_within_each_training_fold",
                    "analysis_population": "spo2_dynamics_eligible_common_sample",
                    "incremental_reference": (
                        "clinical_absolute_spo2_and_sampling"
                        if model_name in {
                            "absolute_spo2",
                            "parsimonious_spo2_instability",
                            "full_spo2_instability",
                        }
                        else "clinical_only"
                    ),
                    "feature_columns": ",".join(specifications[model_name]),
                    "feature_count": feature_count,
                    "events_per_feature": epv,
                    "epv_status": "adequate_ge_10" if epv >= 10 else "fragile_lt_10",
                    "metric_scope": (
                        "pooled_secondary_patient_grouped_oof"
                        if analysis_scope == "pooled_secondary"
                        else "per_dataset_patient_grouped_oof_internal_validation"
                    ),
                    **metrics,
                    **ci,
                }
                if reference is not None:
                    row["delta_auroc_vs_reference"] = (
                        metrics["auroc_oof"] - absolute_metrics["auroc_oof"]
                    )
                    row["delta_auprc_vs_reference"] = (
                        metrics["auprc_oof"] - absolute_metrics["auprc_oof"]
                    )
                    row["delta_auroc_vs_absolute"] = row[
                        "delta_auroc_vs_reference"
                    ]
                    row["delta_auprc_vs_absolute"] = row[
                        "delta_auprc_vs_reference"
                    ]
                rows.append(row)
                identifiers = [
                    column for column in ("dataset", "stay_id", "person_id") if column in frame
                ]
                oof = frame[identifiers].copy()
                oof["analysis_scope"] = analysis_scope
                oof["outcome"] = outcome
                oof["model"] = model_name
                oof["y_true"] = y_array
                oof["probability"] = probability
                prediction_frames.append(oof)
    if not rows:
        return pd.DataFrame([{"status": "skipped", "reason": "no prespecified outcomes available"}])
    result = pd.DataFrame(rows)
    result.attrs["oof_predictions"] = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    return result


def fit_external_transportability(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 100,
    min_events: int = 20,
) -> pd.DataFrame:
    """Train nested models on MIMIC and evaluate untouched eICU."""
    if "dataset" not in analysis_df or not {"mimic", "eicu"}.issubset(set(analysis_df["dataset"].astype(str))):
        return pd.DataFrame([{"status": "unavailable", "reason": "both_mimic_and_eicu_required"}])
    measured = analysis_df.copy()
    if "spo2_plausible_count" in measured:
        measured = measured.loc[pd.to_numeric(measured["spo2_plausible_count"], errors="coerce").fillna(0).gt(0)].copy()
    if "spo2_dynamics_eligible_flag" in measured:
        measured = measured.loc[
            pd.to_numeric(measured["spo2_dynamics_eligible_flag"], errors="coerce")
            .fillna(0)
            .eq(1)
        ].copy()
    mimic_rows = measured["dataset"].astype(str).eq("mimic")
    controls = [
        column
        for column in CLINICAL_CONTROL_CANDIDATES
        if column in measured
        and column not in EXTERNAL_NONHARMONIZED_CONTROLS
        and measured.loc[mimic_rows, column].notna().any()
        and measured.loc[~mimic_rows, column].notna().any()
    ]
    absolute = [
        column
        for column in ABSOLUTE_SPO2_FEATURES
        if column in measured
        and measured.loc[mimic_rows, column].notna().any()
        and measured.loc[~mimic_rows, column].notna().any()
    ]
    sampling = [
        column
        for column in SAMPLING_ADJUSTMENT_FEATURES
        if column in measured
        and measured.loc[mimic_rows, column].notna().any()
        and measured.loc[~mimic_rows, column].notna().any()
    ]
    parsimonious = [
        column
        for column in PARSIMONIOUS_INSTABILITY_FEATURES
        if column in measured
        and measured.loc[mimic_rows, column].notna().any()
        and measured.loc[~mimic_rows, column].notna().any()
    ]
    instability = [
        column
        for column in INSTABILITY_FEATURES
        if column in measured
        and measured.loc[mimic_rows, column].notna().any()
        and measured.loc[~mimic_rows, column].notna().any()
    ]
    specifications = {
        "clinical_only": controls,
        "absolute_spo2": _dedupe([*controls, *absolute, *sampling]),
        "parsimonious_spo2_instability": _dedupe(
            [*controls, *absolute, *sampling, *parsimonious]
        ),
        "full_spo2_instability": _dedupe(
            [*controls, *absolute, *sampling, *instability]
        ),
    }
    rows: list[dict[str, Any]] = []
    for outcome in _prespecified_outcomes(measured):
        train = measured.loc[measured["dataset"].astype(str).eq("mimic")].copy()
        test = measured.loc[measured["dataset"].astype(str).eq("eicu")].copy()
        train = train.loc[pd.to_numeric(train[outcome], errors="coerce").isin([0, 1])]
        test = test.loc[pd.to_numeric(test[outcome], errors="coerce").isin([0, 1])]
        y_train = pd.to_numeric(train[outcome], errors="coerce").astype(int)
        y_test = pd.to_numeric(test[outcome], errors="coerce").astype(int)
        if len(train) < min_rows or len(test) < min_rows or y_train.sum() < min_events or y_test.sum() < min_events or y_train.nunique() < 2 or y_test.nunique() < 2:
            rows.append({"outcome": outcome, "status": "unavailable", "n_train": int(len(train)), "n_test": int(len(test)), "events_train": int(y_train.sum()), "events_test": int(y_test.sum()), "reason": "below external validation floors"})
            continue
        groups, grouping = _group_series(test)
        probabilities: dict[str, np.ndarray] = {}
        fitted: dict[str, Any] = {}
        failure: str | None = None
        for model_name, columns in specifications.items():
            if not columns:
                failure = f"no features for {model_name}"
                break
            try:
                pipeline = _build_model_pipeline(train, columns)
                pipeline.fit(train[columns], y_train)
                probabilities[model_name] = pipeline.predict_proba(test[columns])[:, 1]
                fitted[model_name] = pipeline
            except Exception as exc:
                failure = f"{model_name}: {exc}"
                break
        if failure:
            rows.append(
                {
                    "outcome": outcome,
                    "status": "failed",
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "reason": failure,
                }
            )
            continue
        absolute_probability = probabilities["absolute_spo2"]
        absolute_metrics = _metrics(y_test.to_numpy(), absolute_probability)
        for model_name, probability in probabilities.items():
            metrics = _metrics(y_test.to_numpy(), probability)
            reference = absolute_probability if model_name != "absolute_spo2" else None
            ci = _bootstrap_metric_ci(
                y_test.to_numpy(),
                probability,
                groups.to_numpy(),
                repetitions=500,
                seed=91 + len(rows),
                reference_probability=reference,
            )
            feature_count = int(
                len(fitted[model_name].named_steps["preprocess"].get_feature_names_out())
            )
            epv = min(int(y_train.sum()), int((1 - y_train).sum())) / max(feature_count, 1)
            row: dict[str, Any] = {
                "outcome": outcome,
                "model": model_name,
                "status": "fit",
                "train_dataset": "mimic",
                "test_dataset": "eicu",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "events_train": int(y_train.sum()),
                "events_test": int(y_test.sum()),
                "grouping": grouping,
                "preprocessing_scope": "fit_on_mimic_only",
                "analysis_population": "spo2_dynamics_eligible_common_sample",
                "incremental_reference": (
                    "clinical_absolute_spo2_and_sampling"
                    if model_name in {
                        "absolute_spo2",
                        "parsimonious_spo2_instability",
                        "full_spo2_instability",
                    }
                    else "clinical_only"
                ),
                "feature_columns": ",".join(specifications[model_name]),
                "feature_count": feature_count,
                "training_events_per_feature": epv,
                "epv_status": "adequate_ge_10" if epv >= 10 else "fragile_lt_10",
                "metric_scope": "external_dataset_transportability",
                **metrics,
                **ci,
            }
            if reference is not None:
                row["delta_auroc_vs_reference"] = (
                    metrics["auroc_oof"] - absolute_metrics["auroc_oof"]
                )
                row["delta_auprc_vs_reference"] = (
                    metrics["auprc_oof"] - absolute_metrics["auprc_oof"]
                )
                row["delta_auroc_vs_absolute"] = row[
                    "delta_auroc_vs_reference"
                ]
                row["delta_auprc_vs_absolute"] = row[
                    "delta_auprc_vs_reference"
                ]
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "unavailable", "reason": "no shared prespecified endpoints"}])


def fit_grouped_missingness_control(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Patient-grouped sampling/missingness-only measurement-intensity control."""
    features = [name for name in MISSINGNESS_CONTROL_FEATURES if name in analysis_df]
    if not features:
        return pd.DataFrame([{"status": "not_run", "reason": "missingness features unavailable"}])
    keep = _dedupe(
        column
        for column in (
            "dataset",
            "stay_id",
            "person_id",
            "spo2_plausible_count",
            "spo2_dynamics_eligible_flag",
            *PRIMARY_ENDPOINTS,
            *SECONDARY_ENDPOINTS,
            "target",
            "mcs_or_death_168h_flag",
            *features,
        )
        if column in analysis_df
    )
    proxy = analysis_df[keep].copy()
    # Map all sampling variables onto otherwise absent clinical slots so the
    # exact same grouped, fold-local evaluation engine is used.  Only the
    # clinical-only row is retained; no physiological SpO2 values enter.
    proxy_slots = ("age", "baseline_lactate", "baseline_map", "baseline_hr")
    for target, source in zip(proxy_slots, features):
        proxy[target] = proxy[source]
    result = fit_grouped_incremental_models(proxy, min_rows=50, min_events=10, bootstrap_repetitions=100)
    if "model" in result:
        result = result.loc[
            result["model"].eq("clinical_only") | result["model"].isna()
        ].copy()
        result.loc[result["model"].eq("clinical_only"), "model"] = "missingness_only"
    result["analysis"] = "measurement_intensity_control"
    result["measurement_intensity_features"] = ",".join(features)
    return result


def build_continuous_trajectory_associations(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_groups: int = 50,
    bootstrap_repetitions: int = 500,
    random_state: int = 20260830,
) -> pd.DataFrame:
    """Relate dynamics severity to continuous worsening trajectories.

    The primary coefficient is a partial Spearman correlation after rank-
    residualizing both the dynamics score and outcome against absolute SpO2
    summaries and sampling density. Patient-cluster bootstrap intervals retain
    repeated stays as a unit. These are descriptive trajectory associations,
    not causal effects or predictive-validation metrics.
    """

    if "spo2_dynamics_proxy_score" not in analysis_df:
        return pd.DataFrame(
            [{"status": "unavailable", "reason": "dynamics score unavailable"}]
        )
    outcomes = [
        column
        for column in CONTINUOUS_DECOMPENSATION_TRAJECTORIES
        if column in analysis_df
    ]
    if not outcomes:
        return pd.DataFrame(
            [{"status": "unavailable", "reason": "continuous trajectories unavailable"}]
        )
    frame = analysis_df.copy()
    if "dataset" not in frame:
        frame["dataset"] = "unknown"
    controls = [
        column
        for column in (
            "spo2_mean",
            "spo2_min",
            "spo2_below_90_fraction",
            *SAMPLING_ADJUSTMENT_FEATURES,
        )
        if column in frame
    ]

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 3 or np.nanstd(left) <= 0 or np.nanstd(right) <= 0:
            return math.nan
        return float(np.corrcoef(left, right)[0, 1])

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)

    def availability_column(outcome: str) -> str | None:
        horizon = "12h" if "_12h" in outcome else "24h" if "_24h" in outcome else None
        if horizon is None:
            return None
        if outcome.startswith("lactate_"):
            return f"lactate_rise_{horizon}_observed"
        if outcome.startswith("vis_"):
            return f"vis_observed_{horizon}"
        if outcome.startswith("urine_output_"):
            return f"urine_output_decline_proxy_{horizon}_observed"
        if outcome.startswith("creatinine_"):
            return f"aki_creatinine_{horizon}_observed"
        if outcome.startswith("bilirubin_"):
            return f"hepatic_lab_worsening_{horizon}_observed"
        if outcome.startswith("ast_"):
            return f"ast_injury_{horizon}_observed"
        if outcome.startswith("alt_"):
            return f"alt_injury_{horizon}_observed"
        if outcome.startswith("inr_"):
            return f"inr_injury_{horizon}_observed"
        if outcome.startswith("platelet_"):
            return f"platelet_injury_{horizon}_observed"
        if outcome.startswith("troponin_"):
            return f"troponin_relative_rise_{horizon}_observed"
        return None

    for dataset, dataset_frame in frame.groupby(
        frame["dataset"].astype(str), sort=True
    ):
        for outcome in outcomes:
            observed_column = availability_column(outcome)
            selected = [
                column
                for column in (
                    "dataset",
                    "stay_id",
                    "person_id",
                    "spo2_dynamics_eligible_flag",
                    "spo2_dynamics_proxy_score",
                    outcome,
                    observed_column,
                    *controls,
                )
                if column is not None and column in dataset_frame
            ]
            local = dataset_frame[selected].copy()
            local["_exposure"] = pd.to_numeric(
                local["spo2_dynamics_proxy_score"], errors="coerce"
            )
            local["_outcome"] = pd.to_numeric(local[outcome], errors="coerce")
            valid = local["_exposure"].notna() & local["_outcome"].notna()
            if observed_column is not None and observed_column in local:
                valid &= pd.to_numeric(
                    local[observed_column], errors="coerce"
                ).eq(1)
            if "spo2_dynamics_eligible_flag" in local:
                valid &= pd.to_numeric(
                    local["spo2_dynamics_eligible_flag"], errors="coerce"
                ).eq(1)
            local = local.loc[valid].reset_index(drop=True)
            groups, grouping = _group_series(local) if not local.empty else (
                pd.Series(dtype=str),
                "unavailable",
            )
            n = int(len(local))
            n_groups = int(groups.nunique()) if n else 0
            if n < 3:
                rows.append(
                    {
                        "dataset": dataset,
                        "trajectory": outcome,
                        "status": "unavailable",
                        "n": n,
                        "n_groups": n_groups,
                        "reason": "fewer than 3 complete dynamics/outcome rows",
                    }
                )
                continue

            def rank_residuals(
                sample: pd.DataFrame,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
                sample = sample.reset_index(drop=True)
                sample_n = len(sample)
                exposure_rank = sample["_exposure"].rank(
                    method="average"
                ).to_numpy(float)
                outcome_rank = sample["_outcome"].rank(
                    method="average"
                ).to_numpy(float)
                design_columns: list[np.ndarray] = [
                    np.ones(sample_n, dtype=float)
                ]
                used: list[str] = []
                for control in controls:
                    values = pd.to_numeric(sample[control], errors="coerce")
                    if not values.notna().any():
                        continue
                    values = values.fillna(values.median()).rank(method="average")
                    if values.nunique(dropna=False) <= 1:
                        continue
                    design_columns.append(values.to_numpy(float))
                    used.append(control)
                design = np.column_stack(design_columns)
                exposure_residual = exposure_rank - design @ np.linalg.lstsq(
                    design, exposure_rank, rcond=None
                )[0]
                outcome_residual = outcome_rank - design @ np.linalg.lstsq(
                    design, outcome_rank, rcond=None
                )[0]
                return (
                    exposure_rank,
                    outcome_rank,
                    exposure_residual,
                    outcome_residual,
                    used,
                )

            (
                exposure_rank,
                outcome_rank,
                exposure_residual,
                outcome_residual,
                used_controls,
            ) = rank_residuals(local)
            raw_rho = correlation(exposure_rank, outcome_rank)
            partial_rho = correlation(exposure_residual, outcome_residual)

            cluster_values = groups.to_numpy()
            unique_groups = np.unique(cluster_values)
            by_group = {
                group: np.flatnonzero(cluster_values == group)
                for group in unique_groups
            }
            boot: list[float] = []
            if bootstrap_repetitions > 0 and len(unique_groups) >= 2:
                for _ in range(bootstrap_repetitions):
                    sampled = rng.choice(
                        unique_groups, size=len(unique_groups), replace=True
                    )
                    index = np.concatenate([by_group[group] for group in sampled])
                    # Re-rank and re-fit the adjustment inside each clustered
                    # replicate. Reusing full-sample residuals would omit the
                    # uncertainty in the nuisance adjustment and make the CI
                    # artificially narrow.
                    _, _, boot_exposure, boot_outcome, _ = rank_residuals(
                        local.iloc[index]
                    )
                    value = correlation(
                        boot_exposure, boot_outcome
                    )
                    if math.isfinite(value):
                        boot.append(value)
            enough_bootstrap = len(boot) >= max(
                100, math.ceil(bootstrap_repetitions * 0.80)
            )
            sample_adequate = n >= min_rows and n_groups >= min_groups
            estimator_adequate = (
                math.isfinite(raw_rho)
                and math.isfinite(partial_rho)
                and enough_bootstrap
            )
            status = (
                "estimated"
                if sample_adequate and estimator_adequate
                else "underpowered" if not sample_adequate else "non_estimable"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "trajectory": outcome,
                    "trajectory_direction": "larger_value_is_more_decompensation",
                    "status": status,
                    "n": n,
                    "n_groups": n_groups,
                    "grouping": grouping,
                    "spearman_rho_unadjusted": raw_rho,
                    "partial_spearman_rho_beyond_absolute_spo2_and_sampling": partial_rho,
                    "partial_rho_cluster_boot_ci95_low": (
                        float(np.quantile(boot, 0.025))
                        if enough_bootstrap
                        else math.nan
                    ),
                    "partial_rho_cluster_boot_ci95_high": (
                        float(np.quantile(boot, 0.975))
                        if enough_bootstrap
                        else math.nan
                    ),
                    "bootstrap_repetitions_valid": int(len(boot)),
                    "adjustment_columns": ",".join(used_controls),
                    "trajectory_availability_column": observed_column or "not_available",
                    "claim_scope": "descriptive_continuous_association_not_causal_or_validated_prediction",
                    "reason": (
                        ""
                        if status == "estimated"
                        else "insufficient rows or patient groups"
                        if status == "underpowered"
                        else "non-finite correlation or fewer than 80% valid cluster bootstraps"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _first_instability(
    events: pd.DataFrame,
    *,
    lower_bound: float | None = 50.0,
    jump_threshold: float = 4.0,
    include_hypoxemia: bool = False,
) -> pd.DataFrame:
    """Find first protocol-consistent dynamic SpO2 event by stay.

    Same-time duplicates and 15-minute bins are collapsed exactly as in the
    feature engine.  Abrupt changes require observations no more than 30
    minutes apart; otherwise a multi-hour monitoring gap is not mislabeled as
    signal instability. ``lower_bound=None`` is the raw-value sensitivity.
    Hypoxemia can be included for the explicitly combined exposure, but it is
    excluded by default so dynamics are never conflated with absolute SpO2.
    """
    required = {"concept", "offset_minutes", "value_numeric", "stay_id"}
    if events.empty or not required.issubset(events):
        return pd.DataFrame(
            columns=["dataset", "stay_id", "instability_onset_minutes"]
        )
    local = events.copy()
    if "dataset" not in local:
        local["dataset"] = "unknown"
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    mask = (
        local["concept"].fillna("").astype(str).str.lower().eq("spo2")
        & local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(LANDMARK_MINUTES)
        & local["value_numeric"].le(100)
    )
    if lower_bound is not None:
        mask &= local["value_numeric"].ge(float(lower_bound))
    spo2 = local.loc[mask, ["dataset", "stay_id", "offset_minutes", "value_numeric"]]
    if spo2.empty:
        return pd.DataFrame(
            columns=["dataset", "stay_id", "instability_onset_minutes"]
        )
    same_time = (
        spo2.groupby(["dataset", "stay_id", "offset_minutes"], as_index=False)[
            "value_numeric"
        ]
        .median()
        .sort_values(["dataset", "stay_id", "offset_minutes"])
    )
    same_time["_bin"] = np.floor(same_time["offset_minutes"] / 15.0).astype(int)
    binned = (
        same_time.groupby(["dataset", "stay_id", "_bin"], as_index=False)
        .agg(
            value_numeric=("value_numeric", "median"),
            offset_minutes=("offset_minutes", "median"),
        )
        .sort_values(["dataset", "stay_id", "offset_minutes"])
    )
    grouped = binned.groupby(["dataset", "stay_id"], sort=False)
    difference = grouped["value_numeric"].diff().abs()
    gap = grouped["offset_minutes"].diff()
    valid_pair = gap.gt(0) & gap.le(30)
    support = binned[["dataset", "stay_id"]].copy()
    support["_valid_pair"] = valid_pair.astype(int)
    support["_bin_count"] = 1
    support = support.groupby(["dataset", "stay_id"], sort=False).agg(
        _valid_pair_count=("_valid_pair", "sum"),
        _bin_count=("_bin_count", "sum"),
    )
    eligible_keys = set(
        support.loc[
            support["_bin_count"].ge(3)
            & support["_valid_pair_count"].ge(2)
        ].index
    )
    dynamics_eligible = pd.Series(
        [
            (dataset, stay_id) in eligible_keys
            for dataset, stay_id in zip(binned["dataset"], binned["stay_id"])
        ],
        index=binned.index,
    )
    dynamic_event = (
        difference.ge(float(jump_threshold))
        & valid_pair
        & dynamics_eligible
    )
    binned["_unstable"] = dynamic_event
    if include_hypoxemia:
        binned["_unstable"] |= binned["value_numeric"].lt(90)
    unstable = binned.loc[binned["_unstable"]]
    return (
        unstable.groupby(["dataset", "stay_id"], as_index=False)["offset_minutes"]
        .min()
        .rename(columns={"offset_minutes": "instability_onset_minutes"})
    )


def build_temporal_precedence(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate time from first pre-landmark instability to outcome onset.

    This confirms temporal ordering under the landmark design; it does not
    identify a causal effect and must not be described as such.
    """
    events = events_df.copy()
    if events.empty:
        empty = pd.DataFrame([{"status": "unavailable", "reason": "no events"}])
        return empty, empty.copy()
    if "dataset" not in events:
        events["dataset"] = "unknown"
    events["concept"] = events.get("concept", "").fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    events["value_numeric"] = pd.to_numeric(events.get("value_numeric"), errors="coerce")
    # Use the same physiologic plausibility envelope as the fixed-horizon
    # outcome engine; otherwise a malformed lab value could create a lead-time
    # event that is absent from the corresponding endpoint label.
    temporal_bounds = {
        "lactate": (0.0, 30.0),
        "creatinine": (0.1, 30.0),
        "bilirubin_total": (0.0, 60.0),
        "ast": (0.0, 20_000.0),
        "alt": (0.0, 20_000.0),
        "inr": (0.3, 20.0),
        "platelets": (1.0, 2_000.0),
        "vis": (0.0, 1_000.0),
        "urine_output": (0.0, 5_000.0),
        "troponin_t": (0.0, 1_000.0),
        "troponin_i": (0.0, 1_000.0),
    }
    plausible = pd.Series(True, index=events.index)
    for concept, (lower, upper) in temporal_bounds.items():
        selected = events["concept"].eq(concept)
        plausible.loc[selected] = events.loc[selected, "value_numeric"].between(
            lower, upper, inclusive="both"
        )
    events = events.loc[plausible].copy()
    onset = _first_instability(events)
    cohort = cohort_df.copy()
    if "dataset" not in cohort:
        cohort["dataset"] = "unknown"
    cohort_columns = [
        column
        for column in (
            "dataset",
            "stay_id",
            "person_id",
            "followup_end_offset_minutes",
            "death_offset_minutes",
            "baseline_mcs_flag",
        )
        if column in cohort
    ]
    cohort_context = cohort[cohort_columns].drop_duplicates(["dataset", "stay_id"])
    baseline_window = events.loc[
        events["offset_minutes"].ge(0)
        & events["offset_minutes"].le(LANDMARK_MINUTES)
    ].copy()
    baseline_lactate = _last_measurement(
        baseline_window, "lactate"
    ).rename(columns={"value_numeric": "baseline_lactate"})
    baseline_creatinine = _last_measurement(
        baseline_window, "creatinine"
    ).rename(columns={"value_numeric": "baseline_creatinine"})
    baseline_bilirubin = _last_measurement(
        baseline_window, "bilirubin_total"
    ).rename(columns={"value_numeric": "baseline_bilirubin"})
    records: list[pd.DataFrame] = []
    post = events.loc[
        events["offset_minutes"].gt(LANDMARK_MINUTES)
        & events["offset_minutes"].le(LANDMARK_MINUTES + 24 * 60)
    ].copy()

    def append_first(frame: pd.DataFrame, outcome: str) -> None:
        if frame.empty:
            return
        eligible = frame.merge(
            cohort_context, on=["dataset", "stay_id"], how="left"
        ).reset_index(drop=True)
        followup = (
            pd.to_numeric(eligible["followup_end_offset_minutes"], errors="coerce")
            if "followup_end_offset_minutes" in eligible
            else pd.Series(np.nan, index=eligible.index)
        )
        death = (
            pd.to_numeric(eligible["death_offset_minutes"], errors="coerce")
            if "death_offset_minutes" in eligible
            else pd.Series(np.nan, index=eligible.index)
        )
        within_observed_stay = (
            (followup.isna() | eligible["offset_minutes"].le(followup))
            & (death.isna() | eligible["offset_minutes"].le(death))
        )
        eligible = eligible.loc[within_observed_stay]
        if eligible.empty:
            return
        first = eligible.loc[
            eligible.groupby(["dataset", "stay_id"])["offset_minutes"].idxmin()
        ].rename(columns={"offset_minutes": "outcome_onset_minutes"})
        keep_context = [
            column for column in ("dataset", "stay_id", "person_id", "outcome_onset_minutes")
            if column in first
        ]
        merged = first[keep_context].merge(
            onset, on=["dataset", "stay_id"], how="left", validate="one_to_one"
        )
        merged["outcome"] = outcome
        merged["lead_time_minutes"] = merged["outcome_onset_minutes"] - merged["instability_onset_minutes"]
        merged["instability_observed_flag"] = merged["instability_onset_minutes"].notna().astype(int)
        merged["preceded_by_instability_flag"] = (
            merged["instability_onset_minutes"].notna()
            & merged["instability_onset_minutes"].lt(merged["outcome_onset_minutes"])
        ).astype(int)
        records.append(merged)

    lact = post.loc[post["concept"].eq("lactate")].merge(
        baseline_lactate[["dataset", "stay_id", "baseline_lactate"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(lact.loc[lact["value_numeric"].ge(lact["baseline_lactate"] + LACTATE_RISE_ABSOLUTE)], "lactate_rise")
    append_first(post.loc[post["concept"].eq("pressor_initiation")], "pressor_initiation")
    mcs_candidates = post.loc[
        post["concept"].eq("mcs")
        | post["concept"].str.startswith("mcs_context_")
    ].copy()
    if "baseline_mcs_flag" in cohort_context and not mcs_candidates.empty:
        mcs_candidates = mcs_candidates.merge(
            cohort_context[["dataset", "stay_id", "baseline_mcs_flag"]],
            on=["dataset", "stay_id"],
            how="left",
            validate="many_to_one",
        )
        mcs_candidates = mcs_candidates.loc[
            pd.to_numeric(
                mcs_candidates["baseline_mcs_flag"], errors="coerce"
            ).fillna(0).eq(0)
        ].drop(columns="baseline_mcs_flag")
    append_first(mcs_candidates, "mcs_initiation")
    vis_base = _last_measurement(
        baseline_window, "vis"
    ).rename(columns={"value_numeric": "baseline_vis"})
    vis = post.loc[post["concept"].eq("vis")].merge(
        vis_base[["dataset", "stay_id", "baseline_vis"]],
        on=["dataset", "stay_id"],
        how="left",
    )
    incompatible_vis = events.loc[
        events["concept"].eq("vis_rate_unstandardized")
        & events["offset_minutes"].le(LANDMARK_MINUTES + 24 * 60),
        ["dataset", "stay_id"],
    ].drop_duplicates()
    if not incompatible_vis.empty and not vis.empty:
        vis = vis.merge(
            incompatible_vis.assign(_vis_incompatible=1),
            on=["dataset", "stay_id"],
            how="left",
        )
        vis = vis.loc[vis["_vis_incompatible"].isna()].drop(
            columns="_vis_incompatible"
        )
    vis["baseline_vis"] = vis["baseline_vis"].fillna(0)
    append_first(vis.loc[vis["value_numeric"].gt(vis["baseline_vis"])], "vis_rise")
    creat = post.loc[post["concept"].eq("creatinine")].merge(
        baseline_creatinine[["dataset", "stay_id", "baseline_creatinine"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        creat.loc[
            creat["value_numeric"].ge(creat["baseline_creatinine"] + CREATININE_AKI_ABSOLUTE)
            | creat["value_numeric"].ge(creat["baseline_creatinine"] * CREATININE_AKI_RATIO)
        ],
        "aki_creatinine",
    )
    troponin_rises: list[pd.DataFrame] = []
    for assay in ("troponin_t", "troponin_i"):
        baseline_name = f"baseline_{assay}"
        assay_baseline = _last_measurement(baseline_window, assay).rename(
            columns={"value_numeric": baseline_name}
        )
        assay_post = post.loc[post["concept"].eq(assay)].merge(
            assay_baseline[["dataset", "stay_id", baseline_name]],
            on=["dataset", "stay_id"],
            how="inner",
        )
        troponin_rises.append(
            assay_post.loc[
                assay_post[baseline_name].gt(0)
                & assay_post["value_numeric"].ge(assay_post[baseline_name] * 1.5)
            ].assign(troponin_assay=assay)
        )
    append_first(
        pd.concat(troponin_rises, ignore_index=True),
        "troponin_relative_rise",
    )
    bili = post.loc[post["concept"].eq("bilirubin_total")].merge(
        baseline_bilirubin[["dataset", "stay_id", "baseline_bilirubin"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        bili.loc[
            bili["value_numeric"].ge(bili["baseline_bilirubin"] + BILIRUBIN_WORSENING_ABSOLUTE)
            | bili["value_numeric"].ge(bili["baseline_bilirubin"] * BILIRUBIN_WORSENING_RATIO)
        ],
        "hepatic_lab_worsening",
    )
    for concept in ("ast", "alt"):
        baseline_name = f"baseline_{concept}"
        baseline = _last_measurement(
            baseline_window, concept
        ).rename(columns={"value_numeric": baseline_name})
        lab = post.loc[post["concept"].eq(concept)].merge(
            baseline[["dataset", "stay_id", baseline_name]],
            on=["dataset", "stay_id"],
            how="inner",
        )
        append_first(
            lab.loc[
                (
                    lab[baseline_name].lt(200)
                    & lab["value_numeric"].ge(200)
                )
                | lab["value_numeric"].ge(lab[baseline_name] * 2.0)
            ],
            f"{concept}_injury",
        )
    baseline_inr = _last_measurement(
        baseline_window, "inr"
    ).rename(columns={"value_numeric": "baseline_inr"})
    inr = post.loc[post["concept"].eq("inr")].merge(
        baseline_inr[["dataset", "stay_id", "baseline_inr"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        inr.loc[
            inr["value_numeric"].ge(inr["baseline_inr"] + 0.3)
            | inr["value_numeric"].ge(inr["baseline_inr"] * 1.5)
        ],
        "inr_injury",
    )
    baseline_platelets = _last_measurement(
        baseline_window, "platelets"
    ).rename(columns={"value_numeric": "baseline_platelets"})
    platelets = post.loc[post["concept"].eq("platelets")].merge(
        baseline_platelets[["dataset", "stay_id", "baseline_platelets"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        platelets.loc[
            (
                platelets["baseline_platelets"].ge(100)
                & platelets["value_numeric"].lt(100)
            )
            | platelets["value_numeric"].le(platelets["baseline_platelets"] * 0.70)
        ],
        "platelet_injury",
    )

    baseline_urine = events.loc[
        events["concept"].eq("urine_output")
        & events["offset_minutes"].ge(0)
        & events["offset_minutes"].le(LANDMARK_MINUTES)
    ]
    post_urine = post.loc[post["concept"].eq("urine_output")].copy()
    if not baseline_urine.empty and not post_urine.empty:
        baseline_urine = baseline_urine.assign(
            _two_hour_bin=np.maximum(
                np.ceil(baseline_urine["offset_minutes"] / 120.0) - 1.0,
                0.0,
            )
        )
        baseline_rate = baseline_urine.groupby(
            ["dataset", "stay_id"], as_index=False
        ).agg(
            baseline_urine_rate=("value_numeric", "sum"),
            baseline_covered_bins=("_two_hour_bin", "nunique"),
        )
        baseline_rate["baseline_urine_rate"] /= (
            baseline_rate["baseline_covered_bins"] * 2.0
        )
        post_urine["_two_hour_bin"] = np.maximum(
            np.ceil(
                (post_urine["offset_minutes"] - LANDMARK_MINUTES) / 120.0
            )
            - 1.0,
            0.0,
        )
        post_coverage = post_urine.groupby(
            ["dataset", "stay_id"], as_index=False
        )["_two_hour_bin"].nunique().rename(
            columns={"_two_hour_bin": "post_covered_bins"}
        )
        post_rate = post_urine.groupby(
            ["dataset", "stay_id"], as_index=False
        )["value_numeric"].sum().rename(
            columns={"value_numeric": "post_urine_total"}
        ).merge(
            post_coverage, on=["dataset", "stay_id"], how="inner"
        )
        post_rate["post_urine_rate"] = post_rate["post_urine_total"].div(
            post_rate["post_covered_bins"] * 2.0
        )
        urine_bins = (
            post_urine.groupby(
                ["dataset", "stay_id", "_two_hour_bin"], as_index=False
            )
            .agg(
                value_numeric=("value_numeric", "sum"),
                offset_minutes=("offset_minutes", "max"),
            )
            .merge(baseline_rate, on=["dataset", "stay_id"], how="inner")
            .merge(
                post_rate[
                    [
                        "dataset",
                        "stay_id",
                        "post_covered_bins",
                        "post_urine_rate",
                    ]
                ],
                on=["dataset", "stay_id"],
                how="inner",
            )
        )
        append_first(
            urine_bins.loc[
                urine_bins["baseline_covered_bins"].ge(1)
                & urine_bins["post_covered_bins"].ge(6)
                & urine_bins["post_urine_rate"].le(
                    urine_bins["baseline_urine_rate"]
                    * URINE_RATE_DECLINE_FRACTION
                )
                & urine_bins["value_numeric"].div(2.0).le(
                    urine_bins["baseline_urine_rate"]
                    * URINE_RATE_DECLINE_FRACTION
                )
            ],
            "urine_output_decline_proxy",
        )
    if not records:
        empty = pd.DataFrame([{"status": "unavailable", "reason": "no paired instability/outcome onsets"}])
        return empty, empty.copy()
    record_frame = pd.concat(records, ignore_index=True)
    summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (dataset, outcome), group in record_frame.groupby(["dataset", "outcome"]):
        paired = group.loc[group["preceded_by_instability_flag"].eq(1)].copy()
        values = paired["lead_time_minutes"].dropna().to_numpy(dtype=float)
        boot: list[float] = []
        if len(values) and bootstrap_repetitions > 0:
            cluster = (
                paired["person_id"].fillna(paired["stay_id"]).astype(str).to_numpy()
                if "person_id" in paired
                else paired["stay_id"].astype(str).to_numpy()
            )
            unique = np.unique(cluster)
            by_cluster = {item: np.flatnonzero(cluster == item) for item in unique}
            for _ in range(bootstrap_repetitions):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                index = np.concatenate([by_cluster[item] for item in sampled])
                boot.append(float(np.median(values[index])))
        if boot:
            median_low = float(np.quantile(boot, 0.025))
            median_high = float(np.quantile(boot, 0.975))
        else:  # bootstrap_repetitions == 0: caller supplies its own CI
            median_low = median_high = math.nan
        summaries.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "status": "descriptive_temporal_ordering",
                "n_outcome_events": int(len(group)),
                "n_preceded_by_instability": int(len(values)),
                "n_without_prior_instability": int(len(group) - len(values)),
                "preceded_fraction_all_outcome_events": float(len(values) / len(group)),
                "median_lead_time_minutes": float(np.median(values)) if len(values) else math.nan,
                "q1_lead_time_minutes": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                "q3_lead_time_minutes": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                "median_ci95_low": median_low,
                "median_ci95_high": median_high,
                "positive_lead_time_fraction_paired": float(np.mean(values > 0)) if len(values) else math.nan,
                "claim_scope": "landmark_ordering_design_enforced_not_causal_precedence",
                "inferential_lead_time_test": "not_performed_precedence_is_enforced_by_landmark_design",
            }
        )
    return record_frame, pd.DataFrame(summaries)


__all__ = [
    "CONTINUOUS_DECOMPENSATION_TRAJECTORIES",
    "PRIMARY_ENDPOINTS",
    "SECONDARY_ENDPOINTS",
    "POST_LANDMARK_HORIZONS_HOURS",
    "build_endpoint_completeness_audit",
    "build_continuous_trajectory_associations",
    "build_temporal_precedence",
    "compute_early_decompensation_outcomes",
    "fit_external_transportability",
    "fit_grouped_incremental_models",
    "fit_grouped_missingness_control",
]
