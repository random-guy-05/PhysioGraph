"""Multiscale, dynamics-first tests of charted SpO2 instability.

This module deliberately separates *instability* (successive change,
oscillation, recurrence, persistence, and temporal acceleration) from absolute
oxygenation and observation intensity.  It analyzes the immutable cached event
tables only; no raw-database extraction occurs here.

The primary within-database signal uses each database's native charting cadence
with an outcome-blind gap rule (twice the dataset median cadence, bounded at
10--90 minutes).  Fixed 10/30/90-minute raw-pair and 15/30/60-minute binned
sensitivities expose cadence dependence instead of concealing it.  All models
use patient-grouped out-of-fold prediction and every model for an endpoint uses
the same rows and folds.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .biomarker_benchmark import (
    KAPUR_SCAI_FEATURES,
    LACTATE_FEATURES,
    SBP_FEATURES,
    _metric_values,
    _percentile_interval,
    _valid_grouped_folds,
    _weighted_binary_metric_samples,
    build_biomarker_analysis_frame,
)
from .spo2_lactate_mechanistic import _bh_adjust
from .spo2_multiorgan_mechanistic import _clean_eicu_ventilation_proxy
from .spo2_protocol import (
    ABSOLUTE_SPO2_FEATURES,
    SAMPLING_ADJUSTMENT_FEATURES,
    _build_model_pipeline,
    _calibration,
    _ece,
    _group_series,
    compute_early_decompensation_outcomes,
)


PROTOCOL_VERSION = "spo2_instability_multiscale_v1"
LANDMARK_MINUTES = 240.0
RANDOM_SEED = 20260903
HORIZONS_HOURS = (1, 2, 4, 8, 12, 20, 24)
MIN_ROWS = 200
MIN_EVENTS = 20

CONTEXT_FEATURES = (
    "age",
    "is_male",
    "sex_unknown_flag",
    "shock_icd_flag",
)
LEVEL_FEATURES = tuple(ABSOLUTE_SPO2_FEATURES)
SAMPLING_FEATURES = (
    "adaptive_value_count",
    "adaptive_valid_pair_count",
    "adaptive_median_gap_minutes",
    "adaptive_longest_gap_minutes",
    "adaptive_observed_span_minutes",
)

CORE_INSTABILITY_FEATURES = (
    "adaptive_rmssd_10min_equivalent",
    "adaptive_diff_mad",
    "adaptive_abs_delta_p90",
    "adaptive_jump_ge4_rate_per_hr",
    "adaptive_confirmed_jump_ge4_rate_per_hr",
    "adaptive_reversal_fraction",
    "adaptive_detrended_residual_mad",
)
DIRECTIONAL_INSTABILITY_FEATURES = (
    "adaptive_drop_ge3_rate_per_hr",
    "adaptive_drop_ge5_rate_per_hr",
    "adaptive_rise_ge4_rate_per_hr",
    "adaptive_confirmed_drop_ge3_rate_per_hr",
    "adaptive_persistent_shift_rate_per_hr",
    "adaptive_max_drop_pp",
)
OSCILLATORY_INSTABILITY_FEATURES = (
    "adaptive_oscillation_rate_per_hr",
    "adaptive_reversal_fraction",
    "adaptive_direction_entropy",
    "adaptive_jump_burst_rate_per_hr",
)
TEMPORAL_INSTABILITY_FEATURES = (
    "hour1_rmssd_10min_equivalent",
    "hour2_rmssd_10min_equivalent",
    "hour3_rmssd_10min_equivalent",
    "hour4_rmssd_10min_equivalent",
    "hour1_jump_ge4_rate_per_hr",
    "hour2_jump_ge4_rate_per_hr",
    "hour3_jump_ge4_rate_per_hr",
    "hour4_jump_ge4_rate_per_hr",
    "adaptive_volatility_acceleration",
    "adaptive_late_jump_fraction",
)
THRESHOLD_INSTABILITY_FEATURES = (
    "adaptive_any_jump_ge4",
    "adaptive_recurrent_jump_ge4",
    "adaptive_any_confirmed_jump_ge4",
    "adaptive_any_confirmed_drop_ge3",
)
ARTIFACT_MORPHOLOGY_FEATURES = (
    "adaptive_return_to_baseline_spike_rate_per_hr",
    "adaptive_return_to_baseline_spike_fraction",
)
FULL_INSTABILITY_FEATURES = tuple(
    dict.fromkeys(
        [
            *CORE_INSTABILITY_FEATURES,
            *DIRECTIONAL_INSTABILITY_FEATURES,
            *OSCILLATORY_INSTABILITY_FEATURES,
            *TEMPORAL_INSTABILITY_FEATURES,
            *THRESHOLD_INSTABILITY_FEATURES,
        ]
    )
)

PRIMARY_ASSOCIATION_EXPOSURES = (
    "adaptive_instability_burden_score",
    "adaptive_rmssd_10min_equivalent",
    "adaptive_abs_delta_p90",
    "adaptive_jump_ge4_rate_per_hr",
    "adaptive_confirmed_jump_ge4_rate_per_hr",
    "adaptive_oscillation_rate_per_hr",
    "adaptive_persistent_shift_rate_per_hr",
    "adaptive_recurrent_jump_ge4",
    "adaptive_volatility_acceleration",
    "adaptive_return_to_baseline_spike_rate_per_hr",
)

PREDICTION_ENDPOINT_PREFIXES = (
    "lactate_rise_",
    "lactate_max_rise_",
    "vis_rise_",
    "death_",
    "pressor_initiation_",
    "mcs_",
    "urine_output_decline_proxy_",
    "oliguria_kdigo_proxy_",
    "aki_creatinine_",
    "hepatic_injury_extended_",
    "platelet_injury_",
    "troponin_relative_rise_",
    "early_decompensation_plus_vis_",
    "invasive_ventilation_",
)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _prepare_spo2_signal(events: pd.DataFrame) -> pd.DataFrame:
    required = {"stay_id", "concept", "offset_minutes", "value_numeric"}
    if events.empty or not required.issubset(events):
        return pd.DataFrame(
            columns=["dataset", "stay_id", "offset_minutes", "value_numeric"]
        )
    local = events.reindex(
        columns=["dataset", "stay_id", "concept", "offset_minutes", "value_numeric"]
    ).copy()
    local["dataset"] = local["dataset"].fillna("unknown").astype(str)
    local["concept"] = local["concept"].fillna("").astype(str).str.lower()
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    local = local.loc[
        local["concept"].eq("spo2")
        & local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(LANDMARK_MINUTES)
        & local["value_numeric"].between(50, 100, inclusive="both")
    ]
    return (
        local.groupby(["dataset", "stay_id", "offset_minutes"], as_index=False)[
            "value_numeric"
        ]
        .median()
        .sort_values(["dataset", "stay_id", "offset_minutes"])
        .reset_index(drop=True)
    )


def _dataset_cadence(signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in signal.groupby("dataset", sort=True):
        gaps = group.groupby("stay_id", sort=False)["offset_minutes"].diff()
        gaps = gaps.loc[gaps.gt(0) & gaps.le(LANDMARK_MINUTES)]
        median_gap = float(gaps.median()) if len(gaps) else math.nan
        adaptive = (
            float(np.clip(2.0 * median_gap, 10.0, 90.0))
            if math.isfinite(median_gap)
            else 30.0
        )
        rows.append(
            {
                "dataset": str(dataset),
                "median_native_gap_minutes": median_gap,
                "adaptive_max_pair_gap_minutes": adaptive,
                "rule": "twice_dataset_median_native_gap_bounded_10_to_90_minutes",
                "outcome_information_used": False,
            }
        )
    return pd.DataFrame(rows)


def _bin_signal(group: pd.DataFrame, width_minutes: float) -> tuple[np.ndarray, np.ndarray]:
    local = group.assign(
        _bin=np.floor(group["offset_minutes"].to_numpy(float) / width_minutes).astype(int)
    )
    binned = (
        local.groupby("_bin", as_index=False)
        .agg(
            offset_minutes=("offset_minutes", "median"),
            value_numeric=("value_numeric", "median"),
        )
        .sort_values("offset_minutes")
    )
    return (
        binned["offset_minutes"].to_numpy(float),
        binned["value_numeric"].to_numpy(float),
    )


def _entropy(categories: np.ndarray) -> float:
    if not len(categories):
        return math.nan
    _, counts = np.unique(categories, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    value = -float(np.sum(probabilities * np.log(probabilities)))
    return value / math.log(3.0) if len(counts) > 1 else 0.0


def _signal_summary(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    max_gap_minutes: float,
) -> dict[str, float]:
    """Summarize change dynamics while preserving gaps and episode morphology."""
    result: dict[str, float] = {
        "value_count": float(len(values)),
        "valid_pair_count": 0.0,
        "eligible_flag": 0.0,
    }
    if not len(values):
        return result
    result.update(
        {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "below_90_fraction": float(np.mean(values < 90.0)),
            "observed_span_minutes": (
                float(offsets[-1] - offsets[0]) if len(values) > 1 else 0.0
            ),
        }
    )
    if len(values) < 2:
        return result
    gaps_all = np.diff(offsets)
    deltas_all = np.diff(values)
    valid = (gaps_all > 0) & (gaps_all <= float(max_gap_minutes))
    pair_indices = np.flatnonzero(valid)
    gaps = gaps_all[valid]
    deltas = deltas_all[valid]
    result.update(
        {
            "median_gap_minutes": float(np.median(gaps_all[gaps_all > 0]))
            if np.any(gaps_all > 0)
            else math.nan,
            "longest_gap_minutes": float(np.max(gaps_all[gaps_all > 0]))
            if np.any(gaps_all > 0)
            else math.nan,
            "valid_pair_count": float(len(deltas)),
            "eligible_flag": float(len(values) >= 4 and len(deltas) >= 3),
        }
    )
    if not len(deltas):
        return result
    observed_hours = float(gaps.sum() / 60.0)
    abs_delta = np.abs(deltas)
    jump4 = abs_delta >= 4.0
    drop3 = deltas <= -3.0
    drop5 = deltas <= -5.0
    rise4 = deltas >= 4.0

    consecutive_transition = np.zeros(len(deltas), dtype=bool)
    if len(pair_indices) > 1:
        consecutive_transition[1:] = np.diff(pair_indices) == 1
    reversals = (
        consecutive_transition
        & (np.sign(deltas) != np.sign(np.roll(deltas, 1)))
        & (abs_delta >= 1.0)
        & (np.roll(abs_delta, 1) >= 1.0)
    )
    oscillations = (
        consecutive_transition
        & (np.sign(deltas) != np.sign(np.roll(deltas, 1)))
        & (abs_delta >= 2.0)
        & (np.roll(abs_delta, 1) >= 2.0)
    )

    artifact_like = np.zeros(len(deltas), dtype=bool)
    persistent = np.zeros(len(deltas), dtype=bool)
    confirmed_drop = np.zeros(len(deltas), dtype=bool)
    for position, original_index in enumerate(pair_indices):
        next_index = original_index + 1
        if next_index >= len(deltas_all) or not valid[next_index]:
            continue
        first, second = float(deltas_all[original_index]), float(deltas_all[next_index])
        pre, after = float(values[original_index]), float(values[original_index + 2])
        if (
            abs(first) >= 4.0
            and abs(second) >= 4.0
            and np.sign(first) != np.sign(second)
            and abs(after - pre) <= 2.0
        ):
            artifact_like[position] = True
            if (
                position + 1 < len(pair_indices)
                and pair_indices[position + 1] == next_index
            ):
                artifact_like[position + 1] = True
        if abs(first) >= 4.0 and (
            (first < 0 and after <= pre - 3.0)
            or (first > 0 and after >= pre + 3.0)
        ):
            persistent[position] = True
        if first <= -3.0 and after <= pre - 2.0:
            confirmed_drop[position] = True

    jump_times = offsets[1:][valid][jump4]
    burst_count = 0
    if len(jump_times) >= 2:
        run = 1
        for gap in np.diff(jump_times):
            if gap <= 30.0:
                run += 1
            else:
                burst_count += int(run >= 2)
                run = 1
        burst_count += int(run >= 2)

    residual_mad = math.nan
    if len(values) >= 3 and np.ptp(offsets) > 0:
        coefficients = np.polyfit(offsets / 60.0, values, 1)
        residual = values - np.polyval(coefficients, offsets / 60.0)
        residual_mad = float(np.median(np.abs(residual - np.median(residual))))

    pair_minutes = float(gaps.sum())
    rmssd_10 = (
        float(np.sqrt(np.sum(deltas**2) / pair_minutes * 10.0))
        if pair_minutes > 0
        else math.nan
    )
    denominator = observed_hours if observed_hours > 0 else math.nan
    confirmed_jump = jump4 & ~artifact_like
    direction_categories = np.where(deltas <= -2, -1, np.where(deltas >= 2, 1, 0))
    result.update(
        {
            "rmssd": float(np.sqrt(np.mean(deltas**2))),
            "rmssd_10min_equivalent": rmssd_10,
            "diff_mad": float(np.median(np.abs(deltas - np.median(deltas)))),
            "abs_delta_p90": float(np.quantile(abs_delta, 0.90)),
            "abs_delta_p95": float(np.quantile(abs_delta, 0.95)),
            "max_abs_delta_pp": float(np.max(abs_delta)),
            "max_drop_pp": float(max(-np.min(deltas), 0.0)),
            "max_rise_pp": float(max(np.max(deltas), 0.0)),
            "total_variation_per_hr": float(abs_delta.sum() / denominator),
            "jump_ge4_count": float(jump4.sum()),
            "jump_ge4_fraction": float(jump4.mean()),
            "jump_ge4_rate_per_hr": float(jump4.sum() / denominator),
            "confirmed_jump_ge4_rate_per_hr": float(confirmed_jump.sum() / denominator),
            "drop_ge3_rate_per_hr": float(drop3.sum() / denominator),
            "drop_ge5_rate_per_hr": float(drop5.sum() / denominator),
            "rise_ge4_rate_per_hr": float(rise4.sum() / denominator),
            "confirmed_drop_ge3_rate_per_hr": float(confirmed_drop.sum() / denominator),
            "persistent_shift_rate_per_hr": float(persistent.sum() / denominator),
            "return_to_baseline_spike_rate_per_hr": float(artifact_like.sum() / denominator),
            "return_to_baseline_spike_fraction": float(artifact_like.mean()),
            "reversal_fraction": float(reversals.sum() / max(consecutive_transition.sum(), 1)),
            "oscillation_rate_per_hr": float(oscillations.sum() / denominator),
            "direction_entropy": _entropy(direction_categories),
            "jump_burst_rate_per_hr": float(burst_count / denominator),
            "detrended_residual_mad": residual_mad,
            "any_jump_ge4": float(jump4.any()),
            "recurrent_jump_ge4": float(jump4.sum() >= 2),
            "any_confirmed_jump_ge4": float(confirmed_jump.any()),
            "any_confirmed_drop_ge3": float(confirmed_drop.any()),
            "first_jump_ge4_minutes": float(jump_times[0]) if len(jump_times) else math.nan,
            "last_jump_ge4_minutes": float(jump_times[-1]) if len(jump_times) else math.nan,
            "late_jump_fraction": (
                float(np.mean(jump_times >= 180.0)) if len(jump_times) else 0.0
            ),
        }
    )
    result["instability_burden_score"] = float(
        0.30 * (rmssd_10 / 4.0)
        + 0.20 * (result["abs_delta_p90"] / 4.0)
        + 0.20 * result["jump_ge4_fraction"]
        + 0.15 * min(result["oscillation_rate_per_hr"] / 2.0, 2.0)
        + 0.15 * min(result["persistent_shift_rate_per_hr"] / 2.0, 2.0)
    )
    return result


def _prefix(record: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in record.items()}


def build_multiscale_instability_features(
    events: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build native-cadence, fixed-gap, binned, and hour-localized dynamics."""
    signal = _prepare_spo2_signal(events)
    cadence = _dataset_cadence(signal)
    adaptive_gap = cadence.set_index("dataset")[
        "adaptive_max_pair_gap_minutes"
    ].to_dict()
    rows: list[dict[str, Any]] = []
    for (dataset, stay_id), group in signal.groupby(
        ["dataset", "stay_id"], sort=False
    ):
        offsets = group["offset_minutes"].to_numpy(float)
        values = group["value_numeric"].to_numpy(float)
        gap = float(adaptive_gap.get(str(dataset), 30.0))
        record: dict[str, Any] = {"dataset": str(dataset), "stay_id": stay_id}
        record.update(_prefix(_signal_summary(offsets, values, max_gap_minutes=gap), "adaptive"))

        for label, max_gap in (("raw10", 10.0), ("raw30", 30.0), ("raw90", 90.0)):
            summary = _signal_summary(offsets, values, max_gap_minutes=max_gap)
            retained = {
                name: summary.get(name, math.nan)
                for name in (
                    "eligible_flag",
                    "valid_pair_count",
                    "rmssd_10min_equivalent",
                    "abs_delta_p90",
                    "jump_ge4_rate_per_hr",
                    "confirmed_jump_ge4_rate_per_hr",
                    "drop_ge3_rate_per_hr",
                    "oscillation_rate_per_hr",
                    "persistent_shift_rate_per_hr",
                    "instability_burden_score",
                )
            }
            record.update(_prefix(retained, label))

        for label, width in (("bin15", 15.0), ("bin30", 30.0), ("bin60", 60.0)):
            bin_offsets, bin_values = _bin_signal(group, width)
            summary = _signal_summary(
                bin_offsets, bin_values, max_gap_minutes=2.0 * width
            )
            retained = {
                name: summary.get(name, math.nan)
                for name in (
                    "value_count",
                    "eligible_flag",
                    "valid_pair_count",
                    "rmssd_10min_equivalent",
                    "abs_delta_p90",
                    "jump_ge4_rate_per_hr",
                    "confirmed_jump_ge4_rate_per_hr",
                    "drop_ge3_rate_per_hr",
                    "oscillation_rate_per_hr",
                    "persistent_shift_rate_per_hr",
                    "instability_burden_score",
                )
            }
            record.update(_prefix(retained, label))

        hourly_rmssd: list[float] = []
        for hour in range(4):
            selected = (offsets >= hour * 60.0) & (offsets < (hour + 1) * 60.0)
            summary = _signal_summary(
                offsets[selected], values[selected], max_gap_minutes=gap
            )
            hour_record = {
                "rmssd_10min_equivalent": summary.get(
                    "rmssd_10min_equivalent", math.nan
                ),
                "jump_ge4_rate_per_hr": summary.get(
                    "jump_ge4_rate_per_hr", math.nan
                ),
                "valid_pair_count": summary.get("valid_pair_count", 0.0),
            }
            record.update(_prefix(hour_record, f"hour{hour + 1}"))
            hourly_rmssd.append(hour_record["rmssd_10min_equivalent"])
        early = np.asarray(hourly_rmssd[:3], dtype=float)
        early = early[np.isfinite(early)]
        last = hourly_rmssd[3]
        record["adaptive_volatility_acceleration"] = (
            float(last - np.mean(early))
            if math.isfinite(last) and len(early)
            else math.nan
        )
        rows.append(record)

    features = pd.DataFrame(rows)
    if stay_index is not None and {"dataset", "stay_id"}.issubset(stay_index):
        keys = stay_index[["dataset", "stay_id"]].drop_duplicates()
        features = keys.merge(
            features, on=["dataset", "stay_id"], how="left", validate="one_to_one"
        )
    features["instability_protocol_version"] = PROTOCOL_VERSION
    return features, cadence


def _derive_ventilation_outcomes(
    events: pd.DataFrame,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add incident invasive-ventilation outcomes on every tested horizon."""
    cleaned, audit = _clean_eicu_ventilation_proxy(events)
    local = cleaned.reindex(
        columns=["dataset", "stay_id", "concept", "offset_minutes"]
    ).copy()
    local["dataset"] = local["dataset"].fillna("unknown").astype(str)
    local["concept"] = local["concept"].fillna("").astype(str)
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    ventilation = local.loc[
        local["concept"].eq("mechanical_ventilation")
        & local["offset_minutes"].notna()
    ].copy()
    datasets_with_source = set(ventilation["dataset"].astype(str))
    baseline_keys = set(
        ventilation.loc[ventilation["offset_minutes"].le(LANDMARK_MINUTES), ["dataset", "stay_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    event_times = {
        (str(dataset), stay_id): np.sort(group["offset_minutes"].to_numpy(float))
        for (dataset, stay_id), group in ventilation.groupby(
            ["dataset", "stay_id"], sort=False
        )
    }
    result = frame.copy()
    followup = _numeric(result, "followup_end_offset_minutes")
    death = _numeric(result, "death_offset_minutes")
    for horizon in HORIZONS_HOURS:
        upper = LANDMARK_MINUTES + 60.0 * horizon
        values = np.full(len(result), np.nan, dtype=float)
        observed = np.zeros(len(result), dtype=int)
        at_risk_values = np.zeros(len(result), dtype=int)
        source_values = np.zeros(len(result), dtype=int)
        for position, (dataset, stay_id) in enumerate(
            result[["dataset", "stay_id"]].itertuples(index=False, name=None)
        ):
            key = (str(dataset), stay_id)
            source = str(dataset) in datasets_with_source
            source_values[position] = int(source)
            at_risk = source and key not in baseline_keys
            at_risk_values[position] = int(at_risk)
            if not at_risk:
                continue
            times = event_times.get(key, np.array([], dtype=float))
            left = int(np.searchsorted(times, LANDMARK_MINUTES, side="right"))
            right = int(np.searchsorted(times, upper, side="right"))
            event = right > left
            limit_candidates = [followup.iloc[position]]
            if pd.notna(death.iloc[position]):
                limit_candidates.append(death.iloc[position])
            finite = [float(value) for value in limit_candidates if pd.notna(value)]
            limit = min(finite) if finite else -math.inf
            is_observed = event or limit >= upper
            observed[position] = int(is_observed)
            if is_observed:
                values[position] = float(event)
        stem = f"invasive_ventilation_{horizon}h"
        result[f"{stem}_source_available"] = source_values
        result[f"{stem}_at_risk"] = at_risk_values
        result[f"{stem}_observed"] = observed
        result[f"{stem}_flag"] = values
    return result, audit


def build_multihorizon_instability_frame(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    instability_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach short and conventional horizons plus cleaned ventilation."""
    generated = compute_early_decompensation_outcomes(
        events,
        cohort,
        horizons=HORIZONS_HOURS,
    )
    benchmark = build_biomarker_analysis_frame(events, analysis_frame)
    keys = ["dataset", "stay_id"]
    generated_columns = [
        column
        for column in generated
        if column in keys
        or any(f"_{horizon}h" in column for horizon in HORIZONS_HOURS)
    ]
    generated = generated[generated_columns].copy()
    overlap = [
        column for column in generated.columns if column not in keys and column in benchmark
    ]
    benchmark = benchmark.drop(columns=overlap)
    result = benchmark.merge(
        generated, on=keys, how="left", validate="one_to_one"
    ).merge(
        instability_features.drop(columns=["instability_protocol_version"], errors="ignore"),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    result, ventilation_audit = _derive_ventilation_outcomes(events, result)
    result["instability_protocol_version"] = PROTOCOL_VERSION
    return result, ventilation_audit


def _endpoint_family(endpoint: str) -> str:
    if endpoint.startswith("lactate_"):
        return "lactate"
    if endpoint.startswith(("vis_", "pressor_", "mcs_", "invasive_ventilation_")):
        return "clinical_support"
    if endpoint.startswith("death_"):
        return "mortality"
    if endpoint.startswith(("urine_", "oliguria_")):
        return "urine_output"
    if endpoint.startswith(
        ("aki_", "hepatic_", "platelet_", "troponin_", "ast_", "alt_", "inr_")
    ):
        return "organ_injury"
    return "composite"


def _prediction_endpoints(frame: pd.DataFrame) -> list[str]:
    endpoints: list[str] = []
    for horizon in HORIZONS_HOURS:
        suffix = f"{horizon}h_flag"
        for prefix in PREDICTION_ENDPOINT_PREFIXES:
            column = f"{prefix}{suffix}"
            if column in frame:
                endpoints.append(column)
    return _dedupe(endpoints)


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame]


def _model_specifications(frame: pd.DataFrame) -> dict[str, list[str]]:
    context = _available(CONTEXT_FEATURES, frame)
    level = _available(LEVEL_FEATURES, frame)
    sampling = _available(SAMPLING_FEATURES, frame)
    core = _available(CORE_INSTABILITY_FEATURES, frame)
    full = _available(FULL_INSTABILITY_FEATURES, frame)
    temporal = _available(TEMPORAL_INSTABILITY_FEATURES, frame)
    threshold = _available(THRESHOLD_INSTABILITY_FEATURES, frame)
    artifact = _available(ARTIFACT_MORPHOLOGY_FEATURES, frame)
    conventional = _available(
        [*SBP_FEATURES, *LACTATE_FEATURES, *KAPUR_SCAI_FEATURES], frame
    )
    return {
        "instability_core_only": core,
        "clinical_context": context,
        "context_plus_instability_core": _dedupe([*context, *core]),
        "context_plus_instability_threshold": _dedupe([*context, *threshold]),
        "context_plus_instability_temporal": _dedupe([*context, *temporal]),
        "context_plus_level_sampling": _dedupe([*context, *level, *sampling]),
        "context_plus_level_sampling_plus_instability_core": _dedupe(
            [*context, *level, *sampling, *core]
        ),
        "context_plus_level_sampling_plus_instability_full": _dedupe(
            [*context, *level, *sampling, *full]
        ),
        "context_plus_level_sampling_plus_instability_nonlinear": _dedupe(
            [*context, *level, *sampling, *core]
        ),
        "context_plus_level_sampling_plus_artifact_morphology": _dedupe(
            [*context, *level, *sampling, *artifact]
        ),
        "context_plus_conventional": _dedupe([*context, *conventional]),
        "context_plus_conventional_plus_instability_core": _dedupe(
            [*context, *conventional, *core]
        ),
    }


MODEL_COMPARISONS = (
    ("context_plus_instability_core", "clinical_context", "instability_core_beyond_context"),
    ("context_plus_instability_threshold", "clinical_context", "episode_thresholds_beyond_context"),
    ("context_plus_instability_temporal", "clinical_context", "time_localized_instability_beyond_context"),
    (
        "context_plus_level_sampling_plus_instability_core",
        "context_plus_level_sampling",
        "instability_core_beyond_level_and_sampling",
    ),
    (
        "context_plus_level_sampling_plus_instability_full",
        "context_plus_level_sampling",
        "full_instability_beyond_level_and_sampling",
    ),
    (
        "context_plus_level_sampling_plus_instability_nonlinear",
        "context_plus_level_sampling",
        "nonlinear_instability_beyond_level_and_sampling",
    ),
    (
        "context_plus_level_sampling_plus_artifact_morphology",
        "context_plus_level_sampling",
        "artifact_morphology_beyond_level_and_sampling",
    ),
    (
        "context_plus_conventional_plus_instability_core",
        "context_plus_conventional",
        "instability_core_beyond_conventional",
    ),
    ("context_plus_instability_core", "context_plus_conventional", "instability_vs_conventional"),
)


def _build_nonlinear_pipeline(frame: pd.DataFrame, columns: list[str]):
    """Fold-local fixed-knot spline model; no outcome-driven tuning."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    dynamics = [column for column in CORE_INSTABILITY_FEATURES if column in columns]
    linear = [column for column in columns if column not in dynamics]
    transformers: list[tuple[str, Any, list[str]]] = []
    if linear:
        transformers.append(
            (
                "linear",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                linear,
            )
        )
    if dynamics:
        transformers.append(
            (
                "dynamics_spline",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=4,
                                degree=2,
                                knots="quantile",
                                include_bias=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                dynamics,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")),
        ]
    )


def _paired_bootstrap(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    group_codes, unique_groups = pd.factorize(groups, sort=True)
    rng = np.random.default_rng(seed)
    group_weights = np.empty((repetitions, len(unique_groups)), dtype=np.int32)
    for repetition in range(repetitions):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        group_weights[repetition] = np.bincount(
            sampled, minlength=len(unique_groups)
        )
    row_weights = group_weights[:, group_codes]
    valid = (row_weights[:, y == 1].sum(axis=1) > 0) & (
        row_weights[:, y == 0].sum(axis=1) > 0
    )
    row_weights = row_weights[valid]
    samples = {
        model: _weighted_binary_metric_samples(y, probability, row_weights)
        for model, probability in predictions.items()
    }
    points = {
        model: _metric_values(y, probability)
        for model, probability in predictions.items()
    }
    intervals: dict[str, dict[str, float]] = {}
    for model, metrics in samples.items():
        intervals[model] = {
            "bootstrap_repetitions_requested": float(repetitions),
            "bootstrap_repetitions_valid": float(len(row_weights)),
        }
        for metric, values in metrics.items():
            low, high = _percentile_interval(values.tolist())
            intervals[model][f"{metric}_ci95_low"] = low
            intervals[model][f"{metric}_ci95_high"] = high

    rows: list[dict[str, Any]] = []
    for candidate, reference, label in MODEL_COMPARISONS:
        if candidate not in predictions or reference not in predictions:
            continue
        row: dict[str, Any] = {
            "comparison": label,
            "candidate_model": candidate,
            "reference_model": reference,
            "bootstrap_repetitions_requested": repetitions,
            "bootstrap_repetitions_valid": int(len(row_weights)),
        }
        for metric in ("auroc", "auprc", "brier"):
            if metric == "brier":
                difference = samples[reference][metric] - samples[candidate][metric]
                point = points[reference][metric] - points[candidate][metric]
            else:
                difference = samples[candidate][metric] - samples[reference][metric]
                point = points[candidate][metric] - points[reference][metric]
            low, high = _percentile_interval(difference.tolist())
            row[f"delta_{metric}"] = point
            row[f"delta_{metric}_ci95_low"] = low
            row[f"delta_{metric}_ci95_high"] = high
            row[f"candidate_better_{metric}"] = bool(low > 0)
        rows.append(row)
    return intervals, rows


def fit_instability_prediction_models(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    bootstrap_repetitions: int = 300,
    min_rows: int = MIN_ROWS,
    min_events: int = MIN_EVENTS,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare genuine dynamics blocks on identical patient-grouped OOF folds."""
    eligible = frame.loc[_numeric(frame, "adaptive_eligible_flag").eq(1)].copy()
    performance_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    endpoints = _prediction_endpoints(eligible)
    for dataset, dataset_frame in eligible.groupby(
        eligible["dataset"].astype(str), sort=True
    ):
        specifications = _model_specifications(dataset_frame)
        for outcome in endpoints:
            valid = _numeric(dataset_frame, outcome).isin([0, 1])
            local = dataset_frame.loc[valid].reset_index(drop=True)
            y = _numeric(local, outcome).astype(int)
            groups, grouping = _group_series(local)
            events = int(y.sum())
            non_events = int(len(y) - events)
            base = {
                "dataset": str(dataset),
                "outcome": outcome,
                "endpoint_family": _endpoint_family(outcome),
                "n": int(len(y)),
                "events": events,
                "non_events": non_events,
                "n_patients": int(groups.nunique()) if len(groups) else 0,
                "protocol_version": PROTOCOL_VERSION,
            }
            if len(y) < min_rows or min(events, non_events) < min_events:
                performance_rows.append(
                    {
                        **base,
                        "status": "skipped",
                        "reason": "insufficient rows/events/non-events",
                    }
                )
                continue
            folds, fold_count, fold_seed = _valid_grouped_folds(
                local,
                y,
                groups,
                n_splits=n_splits,
                random_state=random_state,
            )
            if folds is None:
                performance_rows.append(
                    {**base, "status": "failed", "reason": "no valid grouped folds"}
                )
                continue
            predictions = {
                model: np.full(len(local), np.nan, dtype=float)
                for model, columns in specifications.items()
                if columns
            }
            feature_counts = {model: [] for model in predictions}
            failure: str | None = None
            for train_index, test_index in folds:
                for model, probability in predictions.items():
                    columns = specifications[model]
                    try:
                        pipeline = (
                            _build_nonlinear_pipeline(local, columns)
                            if model.endswith("_nonlinear")
                            else _build_model_pipeline(local, columns)
                        )
                        pipeline.fit(local.iloc[train_index][columns], y.iloc[train_index])
                        probability[test_index] = pipeline.predict_proba(
                            local.iloc[test_index][columns]
                        )[:, 1]
                        feature_counts[model].append(
                            int(
                                len(
                                    pipeline.named_steps[
                                        "preprocess"
                                    ].get_feature_names_out()
                                )
                            )
                        )
                    except Exception as exc:
                        failure = f"{model}: {type(exc).__name__}: {exc}"
                        break
                if failure:
                    break
            if failure or any(
                not np.isfinite(probability).all()
                for probability in predictions.values()
            ):
                performance_rows.append(
                    {
                        **base,
                        "status": "failed",
                        "reason": failure or "incomplete OOF predictions",
                    }
                )
                continue
            intervals, comparisons = _paired_bootstrap(
                y.to_numpy(),
                predictions,
                groups.to_numpy(),
                repetitions=bootstrap_repetitions,
                seed=random_state + len(comparison_rows),
            )
            for model, probability in predictions.items():
                metrics = _metric_values(y.to_numpy(), probability)
                feature_count = max(feature_counts[model] or [0])
                epv = min(events, non_events) / max(feature_count, 1)
                performance_rows.append(
                    {
                        **base,
                        "model": model,
                        "status": "fit",
                        "reason": "",
                        "grouping": grouping,
                        "cv_splitter": "StratifiedGroupKFold",
                        "cv_folds": fold_count,
                        "cv_random_state": fold_seed,
                        "feature_columns": ",".join(specifications[model]),
                        "feature_count": feature_count,
                        "events_per_feature": epv,
                        "epv_status": (
                            "adequate_ge_10" if epv >= 10 else "fragile_lt_10"
                        ),
                        **metrics,
                        **intervals[model],
                    }
                )
            for row in comparisons:
                comparison_rows.append({**base, **row})
    return pd.DataFrame(performance_rows), pd.DataFrame(comparison_rows)


def _fit_modified_poisson(
    frame: pd.DataFrame,
    *,
    outcome: str,
    exposure: str,
    adjustment: str,
    min_rows: int = 80,
    min_events: int = 10,
) -> dict[str, Any]:
    """Estimate a cluster-robust RR without selecting controls by p-value."""
    base = {
        "outcome": outcome,
        "endpoint_family": _endpoint_family(outcome),
        "exposure": exposure,
        "adjustment": adjustment,
    }
    selected = ["dataset", "stay_id", "person_id", outcome, exposure]
    controls: list[str] = []
    if adjustment == "context":
        controls = _available(CONTEXT_FEATURES, frame)
    elif adjustment == "level_sampling":
        controls = _available(
            [*CONTEXT_FEATURES, *LEVEL_FEATURES, *SAMPLING_FEATURES], frame
        )
    selected.extend(controls)
    local = frame.reindex(columns=_dedupe(selected)).copy()
    local[outcome] = _numeric(local, outcome)
    raw_exposure = _numeric(local, exposure)
    is_binary = set(raw_exposure.dropna().unique()).issubset({0.0, 1.0})
    center = float(raw_exposure.mean()) if raw_exposure.notna().any() else math.nan
    scale = float(raw_exposure.std(ddof=0)) if raw_exposure.notna().any() else math.nan
    if is_binary:
        local["_exposure"] = raw_exposure
        exposure_scale = "presence_vs_absence"
    elif math.isfinite(scale) and scale > 0:
        local["_exposure"] = (raw_exposure - center) / scale
        exposure_scale = "per_dataset_specific_1sd_higher_instability"
    else:
        local["_exposure"] = np.nan
        exposure_scale = "non_estimable_zero_variance"
    local = local.loc[
        local[outcome].isin([0, 1]) & local["_exposure"].notna()
    ].copy()
    events = int(local[outcome].sum())
    non_events = int(len(local) - events)
    common = {
        **base,
        "n": int(len(local)),
        "events": events,
        "non_events": non_events,
        "exposure_raw_mean": center,
        "exposure_raw_sd": scale,
        "exposure_scale": exposure_scale,
    }
    if len(local) < min_rows or min(events, non_events) < min_events:
        return {
            **common,
            "status": "underpowered",
            "reason": "insufficient rows/events/non-events",
        }
    try:
        import statsmodels.api as sm

        design = pd.DataFrame({"exposure": local["_exposure"]}, index=local.index)
        retained: list[str] = []
        for control in controls:
            values = _numeric(local, control)
            if values.notna().sum() < max(20, int(0.20 * len(local))):
                continue
            missing = values.isna()
            values = values.fillna(float(values.median()))
            if values.nunique(dropna=False) <= 1:
                continue
            if control not in {"is_male", "sex_unknown_flag", "shock_icd_flag"}:
                control_scale = float(values.std(ddof=0))
                if math.isfinite(control_scale) and control_scale > 0:
                    values = (values - float(values.mean())) / control_scale
            design[control] = values.astype(float)
            retained.append(control)
            if missing.any() and (~missing).any():
                design[f"{control}__missing"] = missing.astype(float)
                retained.append(f"{control}__missing")
        design = sm.add_constant(design.astype(float), has_constant="add")
        parameter_count = int(design.shape[1])
        epv = min(events, non_events) / max(parameter_count, 1)
        if epv < 5:
            return {
                **common,
                "status": "underpowered_low_information",
                "reason": "events_per_parameter_below_5",
                "parameter_count": parameter_count,
                "events_per_parameter": epv,
                "control_covariates": ",".join(retained),
            }
        groups, grouping = _group_series(local)
        model = sm.GLM(
            local[outcome].astype(int), design, family=sm.families.Poisson()
        )
        fit = (
            model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            if groups.nunique() > 1
            else model.fit(cov_type="HC0")
        )
        coefficient = float(fit.params["exposure"])
        low, high = fit.conf_int().loc["exposure"]
        rr, ci_low, ci_high = np.exp([coefficient, float(low), float(high)])
        return {
            **common,
            "status": "estimated",
            "reason": "",
            "risk_ratio": float(rr),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "p_value": float(fit.pvalues["exposure"]),
            "log_rr_se": float(fit.bse["exposure"]),
            "parameter_count": parameter_count,
            "events_per_parameter": epv,
            "claim_ready_epv10": bool(epv >= 10),
            "control_covariates": ",".join(retained),
            "grouping": grouping,
        }
    except Exception as exc:
        return {
            **common,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}"[:500],
        }


def fit_instability_associations(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the complete exposure grid with transparent multiplicity families."""
    eligible = frame.loc[_numeric(frame, "adaptive_eligible_flag").eq(1)].copy()
    endpoints = _prediction_endpoints(eligible)
    rows: list[dict[str, Any]] = []
    for dataset, dataset_frame in eligible.groupby(
        eligible["dataset"].astype(str), sort=True
    ):
        for outcome in endpoints:
            for exposure in _available(PRIMARY_ASSOCIATION_EXPOSURES, dataset_frame):
                for adjustment in ("unadjusted", "context", "level_sampling"):
                    rows.append(
                        {
                            "dataset": str(dataset),
                            "protocol_version": PROTOCOL_VERSION,
                            **_fit_modified_poisson(
                                dataset_frame,
                                outcome=outcome,
                                exposure=exposure,
                                adjustment=adjustment,
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    if result.empty or "p_value" not in result:
        return result
    result["q_value_within_exposure_family"] = np.nan
    result["q_value_global_dataset_adjustment"] = np.nan
    estimated = result["status"].eq("estimated") & _numeric(result, "p_value").notna()
    for _, index in result.loc[estimated].groupby(
        ["dataset", "adjustment", "exposure", "endpoint_family"], sort=False
    ).groups.items():
        result.loc[index, "q_value_within_exposure_family"] = _bh_adjust(
            result.loc[index, "p_value"]
        )
    for _, index in result.loc[estimated].groupby(
        ["dataset", "adjustment"], sort=False
    ).groups.items():
        result.loc[index, "q_value_global_dataset_adjustment"] = _bh_adjust(
            result.loc[index, "p_value"]
        )
    result["multiplicity"] = (
        "BH_within_dataset_adjustment_exposure_endpoint_family_and_global_dataset_adjustment"
    )
    return result


def build_instability_dose_response(frame: pd.DataFrame) -> pd.DataFrame:
    """Report risk across score quartiles without breaking tied values."""
    rows: list[dict[str, Any]] = []
    exposure = "adaptive_instability_burden_score"
    endpoints = _prediction_endpoints(frame)
    for dataset, dataset_frame in frame.groupby(frame["dataset"].astype(str), sort=True):
        local = dataset_frame.loc[
            _numeric(dataset_frame, "adaptive_eligible_flag").eq(1)
            & _numeric(dataset_frame, exposure).notna()
        ].copy()
        values = _numeric(local, exposure)
        if values.nunique() < 4:
            continue
        thresholds = np.unique(values.quantile([0.25, 0.50, 0.75]).to_numpy(float))
        local["_quartile"] = np.searchsorted(thresholds, values, side="left") + 1
        for outcome in endpoints:
            valid = _numeric(local, outcome).isin([0, 1])
            observed = local.loc[valid].copy()
            observed["_outcome"] = _numeric(observed, outcome).astype(int)
            if len(observed) < MIN_ROWS or observed["_outcome"].sum() < MIN_EVENTS:
                continue
            for quartile, group in observed.groupby("_quartile", sort=True):
                rows.append(
                    {
                        "dataset": str(dataset),
                        "outcome": outcome,
                        "endpoint_family": _endpoint_family(outcome),
                        "quartile": int(quartile),
                        "n": int(len(group)),
                        "events": int(group["_outcome"].sum()),
                        "risk": float(group["_outcome"].mean()),
                        "score_min": float(_numeric(group, exposure).min()),
                        "score_max": float(_numeric(group, exposure).max()),
                        "ties_preserved": True,
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["risk_ratio_vs_lowest_stratum"] = np.nan
    for _, index in result.groupby(["dataset", "outcome"], sort=False).groups.items():
        group = result.loc[index].sort_values("quartile")
        reference = float(group.iloc[0]["risk"])
        if reference > 0:
            result.loc[group.index, "risk_ratio_vs_lowest_stratum"] = (
                group["risk"] / reference
            )
    return result


def build_resolution_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify how cadence and smoothing change univariate discrimination."""
    from sklearn.metrics import roc_auc_score

    resolutions = (
        "adaptive",
        "raw10",
        "raw30",
        "raw90",
        "bin15",
        "bin30",
        "bin60",
    )
    endpoints = _prediction_endpoints(frame)
    rows: list[dict[str, Any]] = []
    for dataset, dataset_frame in frame.groupby(frame["dataset"].astype(str), sort=True):
        for resolution in resolutions:
            eligibility = f"{resolution}_eligible_flag"
            score = f"{resolution}_instability_burden_score"
            if eligibility not in dataset_frame or score not in dataset_frame:
                continue
            eligible = _numeric(dataset_frame, eligibility).eq(1) & _numeric(
                dataset_frame, score
            ).notna()
            for outcome in endpoints:
                valid = eligible & _numeric(dataset_frame, outcome).isin([0, 1])
                local = dataset_frame.loc[valid]
                y = _numeric(local, outcome).astype(int)
                events = int(y.sum())
                if len(local) < 80 or min(events, len(local) - events) < 10:
                    continue
                rows.append(
                    {
                        "dataset": str(dataset),
                        "resolution": resolution,
                        "outcome": outcome,
                        "endpoint_family": _endpoint_family(outcome),
                        "n": int(len(local)),
                        "events": events,
                        "eligible_fraction_dataset": float(eligible.mean()),
                        "univariate_auroc": float(
                            roc_auc_score(y, _numeric(local, score))
                        ),
                        "score_event_median": float(
                            _numeric(local.loc[y.eq(1)], score).median()
                        ),
                        "score_nonevent_median": float(
                            _numeric(local.loc[y.eq(0)], score).median()
                        ),
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
    return pd.DataFrame(rows)


def build_instability_coverage(
    frame: pd.DataFrame,
    cadence: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cadence_lookup = cadence.set_index("dataset").to_dict("index") if not cadence.empty else {}
    for dataset, group in frame.groupby(frame["dataset"].astype(str), sort=True):
        for resolution in (
            "adaptive",
            "raw10",
            "raw30",
            "raw90",
            "bin15",
            "bin30",
            "bin60",
        ):
            eligibility = _numeric(group, f"{resolution}_eligible_flag")
            rows.append(
                {
                    "dataset": str(dataset),
                    "resolution": resolution,
                    "n_stays": int(len(group)),
                    "n_eligible": int(eligibility.eq(1).sum()),
                    "eligible_fraction": float(eligibility.eq(1).mean()),
                    "median_valid_pairs": float(
                        _numeric(group, f"{resolution}_valid_pair_count").median()
                    ),
                    "median_native_gap_minutes": cadence_lookup.get(str(dataset), {}).get(
                        "median_native_gap_minutes", math.nan
                    ),
                    "adaptive_max_pair_gap_minutes": cadence_lookup.get(str(dataset), {}).get(
                        "adaptive_max_pair_gap_minutes", math.nan
                    ),
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def build_instability_feature_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    comparisons = _dedupe(
        [
            *PRIMARY_ASSOCIATION_EXPOSURES,
            *LEVEL_FEATURES,
            *SAMPLING_FEATURES,
            "spo2_rmssd",
            "spo2_abrupt_jump_fraction",
            "spo2_dynamics_proxy_score",
        ]
    )
    rows: list[dict[str, Any]] = []
    for dataset, group in frame.groupby(frame["dataset"].astype(str), sort=True):
        local = group.loc[_numeric(group, "adaptive_eligible_flag").eq(1)]
        target = _numeric(local, "adaptive_instability_burden_score")
        for column in _available(comparisons, local):
            values = _numeric(local, column)
            valid = target.notna() & values.notna()
            rows.append(
                {
                    "dataset": str(dataset),
                    "target": "adaptive_instability_burden_score",
                    "comparison_feature": column,
                    "n": int(valid.sum()),
                    "spearman_rho": (
                        float(target.loc[valid].corr(values.loc[valid], method="spearman"))
                        if valid.sum() >= 3
                        else math.nan
                    ),
                    "protocol_version": PROTOCOL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def build_instability_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "primary_signal",
                "definition": "same-time-deduplicated native charted SpO2",
                "purpose": "avoid 15-minute median smoothing",
            },
            {
                "component": "adaptive_gap",
                "definition": "2 x dataset median native cadence, bounded 10-90 minutes",
                "purpose": "retain source-appropriate transitions without outcome tuning",
            },
            {
                "component": "eligibility",
                "definition": ">=4 values and >=3 gap-qualified transitions",
                "purpose": "prevent two-point changes from masquerading as instability",
            },
            {
                "component": "threshold_free_dynamics",
                "definition": "RMSSD per sqrt(10 min), robust difference MAD, 90th percentile absolute change, detrended residual MAD",
                "purpose": "measure volatility without relying on one cutoff",
            },
            {
                "component": "episode_dynamics",
                "definition": ">=4-point jumps, recurrence, bursts, direction reversals, and oscillations per observed hour",
                "purpose": "measure abrupt and recurrent instability",
            },
            {
                "component": "physiologic_confirmation",
                "definition": "jump/drop remains displaced on the next gap-qualified value",
                "purpose": "separate persistent shifts from isolated one-point spikes",
            },
            {
                "component": "artifact_morphology_control",
                "definition": ">=4-point change followed by opposite >=4-point change returning within 2 points",
                "purpose": "flag, not automatically discard, spike-and-return morphology",
            },
            {
                "component": "temporal_localization",
                "definition": "hour-specific volatility plus hour-4 minus hours-1-to-3 acceleration",
                "purpose": "avoid averaging a late warning signal across four hours",
            },
            {
                "component": "resolution_grid",
                "definition": "raw <=10/30/90 min and 15/30/60-min median-bin sensitivities",
                "purpose": "show cadence and smoothing dependence explicitly",
            },
            {
                "component": "prediction",
                "definition": "shared rows/folds, grouped OOF linear, threshold, temporal, full, and fixed-knot nonlinear models",
                "purpose": "test both linear and nonlinear instability without p-value model selection",
            },
            {
                "component": "outcome_clock",
                "definition": "1/2/4/8/12/20/24 h after minute-240 landmark; 8 h and 20 h correspond to ICU admission hours 12 and 24",
                "purpose": "detect short-lived signal and resolve admission-vs-landmark ambiguity",
            },
        ]
    ).assign(protocol_version=PROTOCOL_VERSION)


def summarize_instability_analysis(
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    associations: pd.DataFrame,
    resolution: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fit = performance.loc[
        performance.get("status", pd.Series(index=performance.index, dtype=str)).eq("fit")
    ]
    for key, group in fit.groupby(["dataset", "model"], sort=True):
        rows.append(
            {
                "analysis": "patient_grouped_oof_prediction",
                "dataset": key[0],
                "estimand": key[1],
                "n_endpoints": int(group["outcome"].nunique()),
                "median_auroc": float(group["auroc"].median()),
                "median_auprc": float(group["auprc"].median()),
            }
        )
    for key, group in comparisons.groupby(["dataset", "comparison"], sort=True):
        rows.append(
            {
                "analysis": "paired_patient_bootstrap",
                "dataset": key[0],
                "estimand": key[1],
                "n_endpoints": int(group["outcome"].nunique()),
                "median_delta_auroc": float(group["delta_auroc"].median()),
                "positive_ci_excludes_zero": int(
                    _numeric(group, "delta_auroc_ci95_low").gt(0).sum()
                ),
                "negative_ci_excludes_zero": int(
                    _numeric(group, "delta_auroc_ci95_high").lt(0).sum()
                ),
            }
        )
    adjusted = associations.loc[
        associations.get("status", pd.Series(index=associations.index, dtype=str)).eq("estimated")
        & associations.get("adjustment", pd.Series(index=associations.index, dtype=str)).eq(
            "level_sampling"
        )
    ]
    for key, group in adjusted.groupby(["dataset", "exposure"], sort=True):
        rows.append(
            {
                "analysis": "adjusted_modified_poisson",
                "dataset": key[0],
                "estimand": key[1],
                "n_endpoints": int(group["outcome"].nunique()),
                "median_risk_ratio": float(group["risk_ratio"].median()),
                "positive_q_lt_0_05_within_family": int(
                    (
                        _numeric(group, "risk_ratio").gt(1)
                        & _numeric(group, "q_value_within_exposure_family").lt(0.05)
                    ).sum()
                ),
                "inverse_q_lt_0_05_within_family": int(
                    (
                        _numeric(group, "risk_ratio").lt(1)
                        & _numeric(group, "q_value_within_exposure_family").lt(0.05)
                    ).sum()
                ),
            }
        )
    if not resolution.empty:
        for key, group in resolution.groupby(["dataset", "resolution"], sort=True):
            rows.append(
                {
                    "analysis": "resolution_sensitivity",
                    "dataset": key[0],
                    "estimand": key[1],
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_univariate_auroc": float(group["univariate_auroc"].median()),
                }
            )

    key_rows: list[pd.DataFrame] = []
    if not comparisons.empty:
        positive_prediction = comparisons.loc[
            _numeric(comparisons, "delta_auroc_ci95_low").gt(0)
        ].copy()
        if not positive_prediction.empty:
            positive_prediction["evidence_type"] = "positive_paired_delta_auroc"
            key_rows.append(positive_prediction)
    if not associations.empty:
        positive_association = associations.loc[
            associations["status"].eq("estimated")
            & associations["adjustment"].eq("level_sampling")
            & _numeric(associations, "risk_ratio").gt(1)
            & _numeric(associations, "q_value_within_exposure_family").lt(0.05)
        ].copy()
        if not positive_association.empty:
            positive_association["evidence_type"] = "positive_adjusted_rr"
            key_rows.append(positive_association)
    key_results = (
        pd.concat(key_rows, ignore_index=True, sort=False)
        if key_rows
        else pd.DataFrame(
            [{"evidence_type": "none", "status": "no_positive_multiplicity_controlled_result"}]
        )
    )
    return pd.DataFrame(rows), key_results


def run_multiscale_instability_analysis(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    bootstrap_repetitions: int = 300,
) -> dict[str, pd.DataFrame]:
    """Run every multiscale instability layer from cached artifacts."""
    features, cadence = build_multiscale_instability_features(events, analysis_frame)
    frame, ventilation_audit = build_multihorizon_instability_frame(
        events, cohort, analysis_frame, features
    )
    performance, comparisons = fit_instability_prediction_models(
        frame,
        n_splits=n_splits,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    associations = fit_instability_associations(frame)
    dose_response = build_instability_dose_response(frame)
    resolution = build_resolution_sensitivity(frame)
    coverage = build_instability_coverage(frame, cadence)
    correlations = build_instability_feature_correlations(frame)
    summary, key_results = summarize_instability_analysis(
        performance, comparisons, associations, resolution
    )
    compact_columns = _dedupe(
        [
            "dataset",
            "stay_id",
            "person_id",
            "instability_protocol_version",
            *[column for column in frame if column.startswith(("adaptive_", "raw", "bin", "hour"))],
        ]
    )
    return {
        "instability_definitions": build_instability_definitions(),
        "instability_cadence_audit": cadence,
        "instability_features": frame.reindex(columns=compact_columns),
        "instability_coverage": coverage,
        "instability_feature_correlations": correlations,
        "instability_ventilation_audit": ventilation_audit,
        "instability_prediction_performance": performance,
        "instability_prediction_pairwise": comparisons,
        "instability_associations": associations,
        "instability_dose_response": dose_response,
        "instability_resolution_sensitivity": resolution,
        "instability_summary": summary,
        "instability_key_results": key_results,
    }


__all__ = [
    "PROTOCOL_VERSION",
    "build_instability_definitions",
    "build_multiscale_instability_features",
    "build_multihorizon_instability_frame",
    "fit_instability_associations",
    "fit_instability_prediction_models",
    "run_multiscale_instability_analysis",
]
