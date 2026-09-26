"""Episode-anchored SpO2-to-lactate analyses.

This module is a post-result protocol amendment.  It answers the mechanistic
question that a fixed four-hour landmark cannot answer cleanly: after a
specific SpO2 instability episode, does the *next* lactate rise during a
biologically plausible lag?  Every lactate baseline is strictly before the
episode.  The first and maximum subsequent lactates are reported separately.

The analysis remains observational and chart-resolution limited.  It keeps the
original landmark analysis intact, labels focused multiplicity explicitly, and
reports the lactate-observation process so informative remeasurement cannot be
mistaken for a biological null or positive result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


LANDMARK_MINUTES = 240.0
LACTATE_LOWER = 0.0
LACTATE_UPPER = 30.0
LACTATE_LOOKBACK_MINUTES = 360.0
PRIMARY_RISE_THRESHOLD = 0.5
SECONDARY_RISE_THRESHOLD = 1.0
MIN_ROWS = 40
MIN_EVENTS = 10
MIN_EVENTS_PER_PARAMETER = 5.0
BOOTSTRAP_SEED = 20260831

# Acute and delayed windows answer two distinct biological questions without
# averaging a short-lived effect away.  The non-overlapping delayed bins are
# localization analyses; 0-8 h is a cumulative companion.
LAG_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("0_to_1h", 0.0, 60.0, "focused_acute"),
    ("1_to_8h", 60.0, 480.0, "focused_delayed"),
    ("1_to_2h", 60.0, 120.0, "kinetic_localization_secondary"),
    ("2_to_4h", 120.0, 240.0, "kinetic_localization_secondary"),
    ("4_to_8h", 240.0, 480.0, "kinetic_localization_secondary"),
    ("8_to_12h", 480.0, 720.0, "late_companion"),
    ("0_to_8h", 0.0, 480.0, "cumulative_companion"),
)

# The binned absolute jump reconstructs the original exposure.  Directional
# definitions test the important biological possibility that recovery rises
# and deterioration drops cancel when combined.  Raw-time versions quantify
# masking introduced by 15-minute binning.
EPISODE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "episode_definition": "absolute_jump_ge4",
        "signal_resolution": "15_minute_median_bins",
        "kind": "absolute",
        "threshold": 4.0,
        "role": "revised_primary",
    },
    {
        "episode_definition": "desaturation_drop_ge3",
        "signal_resolution": "15_minute_median_bins",
        "kind": "drop",
        "threshold": 3.0,
        "role": "directional_sensitivity",
    },
    {
        "episode_definition": "desaturation_drop_ge5",
        "signal_resolution": "15_minute_median_bins",
        "kind": "drop",
        "threshold": 5.0,
        "role": "directional_sensitivity",
    },
    {
        "episode_definition": "recovery_rise_ge4",
        "signal_resolution": "15_minute_median_bins",
        "kind": "rise",
        "threshold": 4.0,
        "role": "direction_specificity_comparator",
    },
    {
        "episode_definition": "absolute_jump_ge4",
        "signal_resolution": "same_time_deduplicated_raw",
        "kind": "absolute",
        "threshold": 4.0,
        "role": "resolution_sensitivity",
    },
    {
        "episode_definition": "desaturation_drop_ge3",
        "signal_resolution": "same_time_deduplicated_raw",
        "kind": "drop",
        "threshold": 3.0,
        "role": "resolution_sensitivity",
    },
)

PRIMARY_FOCUSED_LAGS = {"0_to_1h", "1_to_8h"}
KINETIC_LOCALIZATION_LAGS = {"1_to_2h", "2_to_4h", "4_to_8h"}
ADJUSTED_LAGS = {
    *PRIMARY_FOCUSED_LAGS,
    *KINETIC_LOCALIZATION_LAGS,
    "0_to_8h",
}
META_ANALYSIS_LAGS = {*PRIMARY_FOCUSED_LAGS, *KINETIC_LOCALIZATION_LAGS}
POST_HOC_STATUS = "post_result_protocol_amendment_exploratory"


def _empty(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment preserving the caller's index."""
    numeric = pd.to_numeric(pvalues, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric)
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    if not valid.any():
        return result
    values = numeric.loc[valid].to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    previous = 1.0
    for rank in range(len(values), 0, -1):
        position = int(order[rank - 1])
        value = min(previous, values[position] * len(values) / rank)
        adjusted[position] = value
        previous = value
    result.loc[numeric.index[valid]] = adjusted
    return result


def _episode_mask(delta: pd.Series, *, kind: str, threshold: float) -> pd.Series:
    if kind == "absolute":
        return delta.abs().ge(threshold)
    if kind == "drop":
        return delta.le(-threshold)
    if kind == "rise":
        return delta.ge(threshold)
    raise ValueError(f"Unknown episode kind: {kind}")


def _prepare_spo2_transitions(
    events_df: pd.DataFrame,
    *,
    signal_resolution: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create gap-qualified adjacent transitions at one signal resolution."""
    required = {"stay_id", "concept", "offset_minutes", "value_numeric"}
    if events_df.empty or not required.issubset(events_df):
        return _empty(
            [
                "dataset",
                "stay_id",
                "anchor_offset_minutes",
                "previous_offset_minutes",
                "anchor_spo2",
                "previous_spo2",
                "spo2_delta",
                "transition_gap_minutes",
            ]
        ), _empty(
            [
                "dataset",
                "stay_id",
                "signal_value_count",
                "valid_transition_count",
                "strict_dynamics_eligible_flag",
            ]
        )
    local = events_df.copy()
    if "dataset" not in local:
        local["dataset"] = "unknown"
    local["concept"] = local["concept"].fillna("").astype(str).str.lower()
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    spo2 = local.loc[
        local["concept"].eq("spo2")
        & local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(LANDMARK_MINUTES)
        & local["value_numeric"].between(50, 100, inclusive="both"),
        ["dataset", "stay_id", "offset_minutes", "value_numeric"],
    ].copy()
    if spo2.empty:
        return _prepare_spo2_transitions(pd.DataFrame(), signal_resolution=signal_resolution)
    signal = (
        spo2.groupby(["dataset", "stay_id", "offset_minutes"], as_index=False)[
            "value_numeric"
        ]
        .median()
        .sort_values(["dataset", "stay_id", "offset_minutes"])
    )
    if signal_resolution == "15_minute_median_bins":
        signal["_bin"] = np.floor(signal["offset_minutes"] / 15.0).astype(int)
        signal = (
            signal.groupby(["dataset", "stay_id", "_bin"], as_index=False)
            .agg(
                offset_minutes=("offset_minutes", "median"),
                value_numeric=("value_numeric", "median"),
            )
            .sort_values(["dataset", "stay_id", "offset_minutes"])
        )
    elif signal_resolution != "same_time_deduplicated_raw":
        raise ValueError(f"Unknown signal resolution: {signal_resolution}")

    grouped = signal.groupby(["dataset", "stay_id"], sort=False)
    signal["previous_offset_minutes"] = grouped["offset_minutes"].shift(1)
    signal["previous_spo2"] = grouped["value_numeric"].shift(1)
    signal["transition_gap_minutes"] = (
        signal["offset_minutes"] - signal["previous_offset_minutes"]
    )
    signal["spo2_delta"] = signal["value_numeric"] - signal["previous_spo2"]
    valid = signal["transition_gap_minutes"].gt(0) & signal[
        "transition_gap_minutes"
    ].le(30)
    transitions = signal.loc[valid].rename(
        columns={
            "offset_minutes": "anchor_offset_minutes",
            "value_numeric": "anchor_spo2",
        }
    )
    transitions = transitions[
        [
            "dataset",
            "stay_id",
            "anchor_offset_minutes",
            "previous_offset_minutes",
            "anchor_spo2",
            "previous_spo2",
            "spo2_delta",
            "transition_gap_minutes",
        ]
    ].reset_index(drop=True)
    support = (
        signal.groupby(["dataset", "stay_id"], as_index=False)
        .agg(signal_value_count=("offset_minutes", "size"))
        .merge(
            transitions.groupby(["dataset", "stay_id"], as_index=False).agg(
                valid_transition_count=("anchor_offset_minutes", "size")
            ),
            on=["dataset", "stay_id"],
            how="left",
        )
    )
    support["valid_transition_count"] = support["valid_transition_count"].fillna(0).astype(int)
    support["strict_dynamics_eligible_flag"] = (
        support["signal_value_count"].ge(3)
        & support["valid_transition_count"].ge(2)
    ).astype(int)
    return transitions, support


def _select_anchor_per_stay(
    transitions: pd.DataFrame,
    support: pd.DataFrame,
    *,
    definition: dict[str, Any],
) -> pd.DataFrame:
    """Select the first episode or a time-comparable no-episode transition."""
    if transitions.empty:
        return _empty(
            [
                "dataset",
                "stay_id",
                "anchor_offset_minutes",
                "episode_exposed",
                "episode_definition",
                "signal_resolution",
            ]
        )
    local = transitions.copy()
    local["_episode"] = _episode_mask(
        local["spo2_delta"],
        kind=str(definition["kind"]),
        threshold=float(definition["threshold"]),
    )
    keys = ["dataset", "stay_id"]
    exposed = (
        local.loc[local["_episode"]]
        .sort_values([*keys, "anchor_offset_minutes"])
        .groupby(keys, as_index=False, sort=False)
        .head(1)
        .copy()
    )
    exposed["episode_exposed"] = 1
    exposed_keys = exposed[keys].drop_duplicates().assign(_ever_episode=1)
    candidates = local.merge(exposed_keys, on=keys, how="left")
    candidates = candidates.loc[candidates["_ever_episode"].isna()].copy()

    # Mirror the exposed anchor-time distribution instead of forcing every
    # control toward one dataset median.  Deterministic quantile assignment is
    # reproducible and prevents time-of-day/measurement-opportunity imbalance
    # from being hidden inside the exposure contrast.
    target_rows: list[pd.DataFrame] = []
    for dataset, dataset_candidates in candidates.groupby("dataset", sort=True):
        control_keys = dataset_candidates[keys].drop_duplicates().copy()
        control_keys["_stable_stay_sort"] = control_keys["stay_id"].astype(str)
        control_keys = control_keys.sort_values("_stable_stay_sort").reset_index(drop=True)
        exposed_times = np.sort(
            pd.to_numeric(
                exposed.loc[
                    exposed["dataset"].astype(str).eq(str(dataset)),
                    "anchor_offset_minutes",
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(float)
        )
        if len(exposed_times):
            quantiles = (np.arange(len(control_keys), dtype=float) + 0.5) / max(
                len(control_keys), 1
            )
            control_keys["_target_anchor"] = np.quantile(exposed_times, quantiles)
        else:
            control_keys["_target_anchor"] = LANDMARK_MINUTES / 2.0
        target_rows.append(control_keys.drop(columns="_stable_stay_sort"))
    targets = (
        pd.concat(target_rows, ignore_index=True)
        if target_rows
        else pd.DataFrame(columns=[*keys, "_target_anchor"])
    )
    candidates = candidates.merge(targets, on=keys, how="left")
    candidates["_target_anchor"] = candidates["_target_anchor"].fillna(
        LANDMARK_MINUTES / 2.0
    )
    candidates["_anchor_distance"] = (
        candidates["anchor_offset_minutes"] - candidates["_target_anchor"]
    ).abs()
    unexposed = (
        candidates.sort_values([*keys, "_anchor_distance", "anchor_offset_minutes"])
        .groupby(keys, as_index=False, sort=False)
        .head(1)
        .copy()
    )
    unexposed["episode_exposed"] = 0
    exposed["control_anchor_strategy"] = "first_episode_transition"
    unexposed["control_anchor_strategy"] = (
        "nearest_transition_to_exposed_anchor_time_quantile"
    )
    anchors = pd.concat([exposed, unexposed], ignore_index=True, sort=False)
    anchors = anchors.merge(support, on=keys, how="left")
    anchors["episode_definition"] = str(definition["episode_definition"])
    anchors["signal_resolution"] = str(definition["signal_resolution"])
    anchors["definition_role"] = str(definition["role"])
    anchors["episode_threshold_pp"] = float(definition["threshold"])
    anchors["episode_direction"] = str(definition["kind"])
    anchors["absolute_spo2_change"] = anchors["spo2_delta"].abs()
    anchors["preanchor_spo2_count"] = np.nan
    anchors["preanchor_sampling_density_per_hr"] = np.nan
    transition_groups = {
        key: group["anchor_offset_minutes"].sort_values().to_numpy(float)
        for key, group in transitions.groupby(keys, sort=False)
    }
    for index, row in anchors.iterrows():
        times = transition_groups.get((row["dataset"], row["stay_id"]), np.array([]))
        count = int(np.searchsorted(times, float(row["anchor_offset_minutes"]), side="right")) + 1
        anchors.at[index, "preanchor_spo2_count"] = count
        hours = max(float(row["anchor_offset_minutes"]) / 60.0, 0.25)
        anchors.at[index, "preanchor_sampling_density_per_hr"] = count / hours
    return anchors.drop(
        columns=[
            "_episode",
            "_ever_episode",
            "_target_anchor",
            "_anchor_distance",
            "_bin",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _merge_anchor_context(
    anchors: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None,
) -> pd.DataFrame:
    keys = ["dataset", "stay_id"]
    context_columns = (
        "person_id",
        "age",
        "is_male",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
        "baseline_mcs_flag",
        "followup_end_offset_minutes",
        "death_offset_minutes",
        "first_stay_per_person_flag",
    )
    result = anchors.copy()
    cohort = cohort_df.copy()
    if "dataset" not in cohort:
        cohort["dataset"] = "unknown"
    available = [*keys, *[column for column in context_columns if column in cohort]]
    if set(keys).issubset(cohort):
        result = result.merge(
            cohort[available].drop_duplicates(keys), on=keys, how="left"
        )
    if analysis_df is not None and not analysis_df.empty:
        analysis = analysis_df.copy()
        if "dataset" not in analysis:
            analysis["dataset"] = "unknown"
        extra_candidates = (
            "resp_support_any_flag",
            "mechanical_ventilation_flag",
            "spo2_below_90_fraction",
            "spo2_sampling_density_per_hr",
        )
        extras = [column for column in extra_candidates if column in analysis and column not in result]
        if extras:
            result = result.merge(
                analysis[[*keys, *extras]].drop_duplicates(keys),
                on=keys,
                how="left",
            )
    return result


def _lactate_groups(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> dict[tuple[str, Any], tuple[np.ndarray, np.ndarray]]:
    events = events_df.copy()
    if "dataset" not in events:
        events["dataset"] = "unknown"
    events["concept"] = events.get("concept", "").fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events.get("offset_minutes"), errors="coerce")
    events["value_numeric"] = pd.to_numeric(events.get("value_numeric"), errors="coerce")
    lactate = events.loc[
        events["concept"].eq("lactate")
        & events["offset_minutes"].notna()
        & events["value_numeric"].between(
            LACTATE_LOWER, LACTATE_UPPER, inclusive="both"
        ),
        ["dataset", "stay_id", "offset_minutes", "value_numeric"],
    ].copy()
    if lactate.empty:
        return {}
    lactate = (
        lactate.groupby(["dataset", "stay_id", "offset_minutes"], as_index=False)[
            "value_numeric"
        ]
        .median()
        .sort_values(["dataset", "stay_id", "offset_minutes"])
    )
    cohort = cohort_df.copy()
    if "dataset" not in cohort:
        cohort["dataset"] = "unknown"
    limits = [
        column
        for column in (
            "dataset",
            "stay_id",
            "followup_end_offset_minutes",
            "death_offset_minutes",
        )
        if column in cohort
    ]
    if set(("dataset", "stay_id")).issubset(cohort) and len(limits) > 2:
        lactate = lactate.merge(
            cohort[limits].drop_duplicates(["dataset", "stay_id"]),
            on=["dataset", "stay_id"],
            how="left",
        )
        followup = pd.to_numeric(
            lactate.get(
                "followup_end_offset_minutes",
                pd.Series(np.nan, index=lactate.index),
            ),
            errors="coerce",
        )
        death = pd.to_numeric(
            lactate.get(
                "death_offset_minutes", pd.Series(np.nan, index=lactate.index)
            ),
            errors="coerce",
        )
        observed = (followup.isna() | lactate["offset_minutes"].le(followup)) & (
            death.isna() | lactate["offset_minutes"].le(death)
        )
        lactate = lactate.loc[observed]
    return {
        key: (
            group["offset_minutes"].to_numpy(float),
            group["value_numeric"].to_numpy(float),
        )
        for key, group in lactate.groupby(["dataset", "stay_id"], sort=False)
    }


def _attach_lactate_lags(
    anchors: pd.DataFrame,
    lactates: dict[tuple[str, Any], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        anchor_time = float(anchor["anchor_offset_minutes"])
        times, values = lactates.get(
            (anchor["dataset"], anchor["stay_id"]),
            (np.array([], dtype=float), np.array([], dtype=float)),
        )
        prior_right = int(np.searchsorted(times, anchor_time, side="left"))
        prior_left = int(
            np.searchsorted(
                times,
                anchor_time - LACTATE_LOOKBACK_MINUTES,
                side="left",
            )
        )
        prior_available = prior_right > prior_left
        prior_value = float(values[prior_right - 1]) if prior_available else math.nan
        prior_time = float(times[prior_right - 1]) if prior_available else math.nan
        for lag_label, lower, upper, lag_role in LAG_WINDOWS:
            left = int(np.searchsorted(times, anchor_time + lower, side="right"))
            right = int(np.searchsorted(times, anchor_time + upper, side="right"))
            has_post = right > left
            first_value = float(values[left]) if has_post else math.nan
            first_time = float(times[left]) if has_post else math.nan
            if has_post:
                local_values = values[left:right]
                max_local = int(np.argmax(local_values))
                max_value = float(local_values[max_local])
                max_time = float(times[left + max_local])
            else:
                max_value = math.nan
                max_time = math.nan
            row = dict(anchor)
            row.update(
                {
                    "lag_window": lag_label,
                    "lag_lower_minutes_exclusive": lower,
                    "lag_upper_minutes_inclusive": upper,
                    "lag_role": lag_role,
                    "prior_lactate": prior_value,
                    "prior_lactate_offset_minutes": prior_time,
                    "prior_lactate_strictly_before_episode": int(prior_available),
                    "first_lactate": first_value,
                    "first_lactate_offset_minutes": first_time,
                    "maximum_lactate": max_value,
                    "maximum_lactate_offset_minutes": max_time,
                    "post_lactate_observed": int(has_post),
                    "complete_lactate_pair": int(prior_available and has_post),
                    "first_lactate_delta": (
                        first_value - prior_value
                        if prior_available and has_post
                        else math.nan
                    ),
                    "maximum_lactate_delta": (
                        max_value - prior_value
                        if prior_available and has_post
                        else math.nan
                    ),
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    for prefix in ("first", "maximum"):
        delta = pd.to_numeric(result.get(f"{prefix}_lactate_delta"), errors="coerce")
        observed = delta.notna()
        result[f"{prefix}_rise_gt0"] = np.where(observed, delta.gt(0).astype(float), np.nan)
        result[f"{prefix}_rise_ge0_5"] = np.where(
            observed, delta.ge(PRIMARY_RISE_THRESHOLD).astype(float), np.nan
        )
        result[f"{prefix}_rise_ge1_0"] = np.where(
            observed, delta.ge(SECONDARY_RISE_THRESHOLD).astype(float), np.nan
        )
    result["analysis_status"] = POST_HOC_STATUS
    result["causal_claim_permitted"] = False
    return result


def build_episode_lactate_records(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one episode/control anchor per stay and attach strict lactate lags."""
    transition_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    anchors: list[pd.DataFrame] = []
    for definition in EPISODE_DEFINITIONS:
        resolution = str(definition["signal_resolution"])
        if resolution not in transition_cache:
            transition_cache[resolution] = _prepare_spo2_transitions(
                events_df, signal_resolution=resolution
            )
        transitions, support = transition_cache[resolution]
        selected = _select_anchor_per_stay(
            transitions, support, definition=definition
        )
        if not selected.empty:
            anchors.append(selected)
    if not anchors:
        return _empty(
            [
                "dataset",
                "stay_id",
                "episode_definition",
                "signal_resolution",
                "lag_window",
                "analysis_status",
            ]
        )
    anchor_frame = pd.concat(anchors, ignore_index=True, sort=False)
    anchor_frame = _merge_anchor_context(anchor_frame, cohort_df, analysis_df)
    lactates = _lactate_groups(events_df, cohort_df)
    return _attach_lactate_lags(anchor_frame, lactates)


def _cluster_bootstrap_metrics(
    frame: pd.DataFrame,
    delta_column: str,
    *,
    repetitions: int,
    random_state: int,
) -> dict[str, float]:
    local = frame.loc[pd.to_numeric(frame[delta_column], errors="coerce").notna()].copy()
    if local.empty or repetitions <= 0:
        return {
            "median_ci95_low": math.nan,
            "median_ci95_high": math.nan,
            "rise_ge0_5_ci95_low": math.nan,
            "rise_ge0_5_ci95_high": math.nan,
            "bootstrap_repetitions_valid": 0,
        }
    if "person_id" in local and local["person_id"].notna().any():
        person = local["person_id"].astype("string")
        stay = local["stay_id"].astype("string")
        cluster = person.where(person.notna() & person.str.strip().ne(""), "stay:" + stay)
    else:
        cluster = "stay:" + local["stay_id"].astype("string")
    cluster = local["dataset"].astype(str) + ":" + cluster.astype(str)
    local["_cluster"] = cluster
    groups = {key: group for key, group in local.groupby("_cluster", sort=False)}
    keys = np.asarray(list(groups), dtype=object)
    if len(keys) < 2:
        return {
            "median_ci95_low": math.nan,
            "median_ci95_high": math.nan,
            "rise_ge0_5_ci95_low": math.nan,
            "rise_ge0_5_ci95_high": math.nan,
            "bootstrap_repetitions_valid": 0,
        }
    rng = np.random.default_rng(random_state)
    medians: list[float] = []
    proportions: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = np.concatenate(
            [pd.to_numeric(groups[key][delta_column], errors="coerce").dropna().to_numpy(float) for key in sampled]
        )
        if values.size:
            medians.append(float(np.median(values)))
            proportions.append(float(np.mean(values >= PRIMARY_RISE_THRESHOLD)))
    required = max(100, math.ceil(repetitions * 0.8))
    if len(medians) < required:
        return {
            "median_ci95_low": math.nan,
            "median_ci95_high": math.nan,
            "rise_ge0_5_ci95_low": math.nan,
            "rise_ge0_5_ci95_high": math.nan,
            "bootstrap_repetitions_valid": len(medians),
        }
    return {
        "median_ci95_low": float(np.quantile(medians, 0.025)),
        "median_ci95_high": float(np.quantile(medians, 0.975)),
        "rise_ge0_5_ci95_low": float(np.quantile(proportions, 0.025)),
        "rise_ge0_5_ci95_high": float(np.quantile(proportions, 0.975)),
        "bootstrap_repetitions_valid": len(medians),
    }


def _wilcoxon_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 5:
        return math.nan
    if np.allclose(values, 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return math.nan


def build_episode_lactate_paired_summary(
    records: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 500,
) -> pd.DataFrame:
    """Summarize within-episode strictly-before/after lactate changes."""
    if records.empty:
        return _empty(["status", "reason"])
    grouping = [
        "dataset",
        "episode_definition",
        "signal_resolution",
        "definition_role",
        "lag_window",
        "lag_role",
    ]
    rows: list[dict[str, Any]] = []
    for eligibility_scope in ("episode_pair_observed", "strict_signal_eligible"):
        eligible = records.loc[records["episode_exposed"].eq(1)].copy()
        if eligibility_scope == "strict_signal_eligible":
            eligible = eligible.loc[eligible["strict_dynamics_eligible_flag"].eq(1)]
        for keys, group in eligible.groupby(grouping, dropna=False, sort=False):
            base = dict(zip(grouping, keys))
            for outcome_kind, delta_column in (
                ("first_next_lactate", "first_lactate_delta"),
                ("maximum_lactate", "maximum_lactate_delta"),
            ):
                values = pd.to_numeric(group[delta_column], errors="coerce").dropna().to_numpy(float)
                complete = group.loc[pd.to_numeric(group[delta_column], errors="coerce").notna()]
                bootstrap_requested = (
                    bootstrap_repetitions
                    if base["episode_definition"] == "absolute_jump_ge4"
                    and base["signal_resolution"] == "15_minute_median_bins"
                    and eligibility_scope == "episode_pair_observed"
                    and outcome_kind == "first_next_lactate"
                    and base["lag_window"] in {*PRIMARY_FOCUSED_LAGS, "0_to_8h"}
                    else 0
                )
                bootstrap = _cluster_bootstrap_metrics(
                    complete,
                    delta_column,
                    repetitions=bootstrap_requested,
                    random_state=BOOTSTRAP_SEED + len(rows),
                )
                rows.append(
                    {
                        **base,
                        "eligibility_scope": eligibility_scope,
                        "outcome_kind": outcome_kind,
                        "status": "estimated" if len(values) >= 5 else "underpowered",
                        "n_episode_anchors": int(group[["dataset", "stay_id"]].drop_duplicates().shape[0]),
                        "n_prior_lactate": int(group["prior_lactate_strictly_before_episode"].eq(1).sum()),
                        "n_complete_pairs": int(len(values)),
                        "complete_pair_fraction": float(len(values) / len(group)) if len(group) else math.nan,
                        "median_delta_mmol_l": float(np.median(values)) if len(values) else math.nan,
                        "mean_delta_mmol_l": float(np.mean(values)) if len(values) else math.nan,
                        "iqr_delta_mmol_l": float(np.subtract(*np.quantile(values, [0.75, 0.25]))) if len(values) else math.nan,
                        "proportion_delta_gt0": float(np.mean(values > 0)) if len(values) else math.nan,
                        "proportion_rise_ge0_5": float(np.mean(values >= PRIMARY_RISE_THRESHOLD)) if len(values) else math.nan,
                        "proportion_rise_ge1_0": float(np.mean(values >= SECONDARY_RISE_THRESHOLD)) if len(values) else math.nan,
                        "paired_wilcoxon_two_sided_p": _wilcoxon_p(values),
                        "bootstrap_repetitions_requested": int(bootstrap_requested),
                        **bootstrap,
                        "analysis_status": POST_HOC_STATUS,
                        "interpretation_scope": "within_episode_change_not_a_controlled_causal_effect",
                    }
                )
    result = pd.DataFrame(rows)
    result["global_exploratory_q_value"] = _bh_adjust(
        result["paired_wilcoxon_two_sided_p"]
    )
    result["focused_q_value"] = np.nan
    focused = (
        result["episode_definition"].eq("absolute_jump_ge4")
        & result["signal_resolution"].eq("15_minute_median_bins")
        & result["eligibility_scope"].eq("episode_pair_observed")
        & result["outcome_kind"].eq("first_next_lactate")
        & result["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
    )
    for _, indexes in result.loc[focused].groupby("dataset").groups.items():
        result.loc[indexes, "focused_q_value"] = _bh_adjust(
            result.loc[indexes, "paired_wilcoxon_two_sided_p"]
        )
    result["focused_family"] = np.where(
        focused,
        "first_next_lactate_after_absolute_jump_ge4_acute_0_to_1h_and_delayed_1_to_8h",
        "not_in_focused_family",
    )
    result["kinetic_localization_q_value"] = np.nan
    localization = (
        result["episode_definition"].eq("absolute_jump_ge4")
        & result["signal_resolution"].eq("15_minute_median_bins")
        & result["eligibility_scope"].eq("episode_pair_observed")
        & result["outcome_kind"].eq("first_next_lactate")
        & result["lag_window"].isin(KINETIC_LOCALIZATION_LAGS)
    )
    for _, indexes in result.loc[localization].groupby("dataset").groups.items():
        result.loc[indexes, "kinetic_localization_q_value"] = _bh_adjust(
            result.loc[indexes, "paired_wilcoxon_two_sided_p"]
        )
    return result


def _risk_ratio(
    exposed_events: int,
    exposed_total: int,
    control_events: int,
    control_total: int,
) -> dict[str, float]:
    if exposed_total <= 0 or control_total <= 0:
        return {
            "risk_ratio": math.nan,
            "risk_ratio_ci95_low": math.nan,
            "risk_ratio_ci95_high": math.nan,
            "log_risk_ratio_se": math.nan,
        }
    a = float(exposed_events)
    b = float(exposed_total - exposed_events)
    c = float(control_events)
    d = float(control_total - control_events)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    risk_exposed = a / (a + b)
    risk_control = c / (c + d)
    if risk_control <= 0 or risk_exposed <= 0:
        return {
            "risk_ratio": math.nan,
            "risk_ratio_ci95_low": math.nan,
            "risk_ratio_ci95_high": math.nan,
            "log_risk_ratio_se": math.nan,
        }
    rr = risk_exposed / risk_control
    se = math.sqrt(max((1.0 / a) - (1.0 / (a + b)) + (1.0 / c) - (1.0 / (c + d)), 0.0))
    return {
        "risk_ratio": rr,
        "risk_ratio_ci95_low": math.exp(math.log(rr) - 1.96 * se),
        "risk_ratio_ci95_high": math.exp(math.log(rr) + 1.96 * se),
        "log_risk_ratio_se": se,
    }


def _fisher_p(a: int, b: int, c: int, d: int) -> float:
    try:
        from scipy.stats import fisher_exact

        return float(fisher_exact([[a, b], [c, d]], alternative="two-sided").pvalue)
    except Exception:
        return math.nan


def _mann_whitney_p(exposed: np.ndarray, controls: np.ndarray) -> float:
    if len(exposed) < 5 or len(controls) < 5:
        return math.nan
    try:
        from scipy.stats import mannwhitneyu

        return float(mannwhitneyu(exposed, controls, alternative="two-sided").pvalue)
    except Exception:
        return math.nan


def _adjusted_modified_poisson(
    frame: pd.DataFrame,
    *,
    outcome_column: str,
) -> dict[str, Any]:
    """Parsimonious modified-Poisson RR with a hard low-information gate."""
    result: dict[str, Any] = {
        "adjusted_status": "not_estimable",
        "adjusted_risk_ratio": math.nan,
        "adjusted_ci95_low": math.nan,
        "adjusted_ci95_high": math.nan,
        "adjusted_p_value": math.nan,
        "adjusted_log_rr_se": math.nan,
        "adjusted_parameter_count": 0,
        "adjusted_events_per_parameter": math.nan,
        "adjusted_covariates": "",
        "adjusted_reason": "",
    }
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        result["adjusted_reason"] = f"statsmodels unavailable: {exc}"
        return result
    local = frame.copy()
    local[outcome_column] = pd.to_numeric(local[outcome_column], errors="coerce")
    local = local.loc[
        local[outcome_column].isin([0, 1]) & local["episode_exposed"].isin([0, 1])
    ].copy()
    events = int(local[outcome_column].sum())
    non_events = int(len(local) - events)
    if len(local) < MIN_ROWS or min(events, non_events) < MIN_EVENTS:
        result["adjusted_status"] = "underpowered"
        result["adjusted_reason"] = "insufficient rows/events/non-events"
        return result
    covariate_candidates = (
        "prior_lactate",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
        "age",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
    )
    covariates = [column for column in covariate_candidates if column in local]
    design = pd.DataFrame(index=local.index)
    design["episode_exposed"] = pd.to_numeric(local["episode_exposed"], errors="coerce")
    retained: list[str] = []
    for column in covariates:
        values = pd.to_numeric(local[column], errors="coerce")
        if values.notna().sum() < max(20, int(len(local) * 0.2)):
            continue
        values = values.fillna(float(values.median()))
        if values.nunique(dropna=False) <= 1:
            continue
        if column not in {"shock_icd_flag", "baseline_vasoactive_flag"}:
            scale = float(values.std(ddof=0))
            if np.isfinite(scale) and scale > 0:
                values = (values - float(values.mean())) / scale
        design[column] = values
        retained.append(column)
    design = sm.add_constant(design.astype(float), has_constant="add")
    parameters = int(design.shape[1])
    epv = min(events, non_events) / max(parameters, 1)
    result.update(
        {
            "adjusted_parameter_count": parameters,
            "adjusted_events_per_parameter": epv,
            "adjusted_covariates": ",".join(retained),
        }
    )
    if epv < MIN_EVENTS_PER_PARAMETER:
        result["adjusted_status"] = "underpowered_low_information"
        result["adjusted_reason"] = "events_per_parameter_below_5_fail_closed"
        return result
    y = local[outcome_column].astype(int)
    if "person_id" in local and local["person_id"].notna().any():
        person = local["person_id"].astype("string")
        stay = local["stay_id"].astype("string")
        groups = person.where(person.notna() & person.str.strip().ne(""), "stay:" + stay)
    else:
        groups = "stay:" + local["stay_id"].astype("string")
    groups = local["dataset"].astype(str) + ":" + groups.astype(str)
    try:
        model = sm.GLM(y, design, family=sm.families.Poisson())
        fit = (
            model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            if groups.nunique() > 1
            else model.fit(cov_type="HC0")
        )
        coef = float(fit.params["episode_exposed"])
        low, high = fit.conf_int().loc["episode_exposed"]
        values = np.exp([coef, float(low), float(high)])
        if not np.isfinite(values).all() or not np.isfinite(fit.bse["episode_exposed"]):
            raise ValueError("non-finite modified-Poisson inference")
        result.update(
            {
                "adjusted_status": "estimated",
                "adjusted_risk_ratio": float(values[0]),
                "adjusted_ci95_low": float(values[1]),
                "adjusted_ci95_high": float(values[2]),
                "adjusted_p_value": float(fit.pvalues["episode_exposed"]),
                "adjusted_log_rr_se": float(fit.bse["episode_exposed"]),
                "adjusted_reason": "",
            }
        )
    except Exception as exc:
        result["adjusted_status"] = "non_estimable"
        result["adjusted_reason"] = str(exc)[:500]
    return result


def build_episode_lactate_controlled_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Compare episode anchors with time-aligned no-episode transition anchors."""
    if records.empty:
        return _empty(["status", "reason"])
    grouping = [
        "dataset",
        "episode_definition",
        "signal_resolution",
        "definition_role",
        "lag_window",
        "lag_role",
    ]
    rows: list[dict[str, Any]] = []
    for eligibility_scope in ("episode_pair_observed", "strict_signal_eligible"):
        eligible = records.copy()
        if eligibility_scope == "strict_signal_eligible":
            eligible = eligible.loc[eligible["strict_dynamics_eligible_flag"].eq(1)]
        for keys, group in eligible.groupby(grouping, dropna=False, sort=False):
            base = dict(zip(grouping, keys))
            for outcome_kind, delta_column in (
                ("first_next_lactate", "first_lactate_delta"),
                ("maximum_lactate", "maximum_lactate_delta"),
            ):
                complete = group.loc[pd.to_numeric(group[delta_column], errors="coerce").notna()].copy()
                complete["_delta"] = pd.to_numeric(complete[delta_column], errors="coerce")
                for rise_threshold in (PRIMARY_RISE_THRESHOLD, SECONDARY_RISE_THRESHOLD):
                    complete["_rise"] = complete["_delta"].ge(rise_threshold).astype(int)
                    exposed = complete.loc[complete["episode_exposed"].eq(1)]
                    controls = complete.loc[complete["episode_exposed"].eq(0)]
                    a = int(exposed["_rise"].sum())
                    n1 = int(len(exposed))
                    c = int(controls["_rise"].sum())
                    n0 = int(len(controls))
                    rr = _risk_ratio(a, n1, c, n0)
                    estimated = n1 >= 5 and n0 >= 5 and min(a + c, n1 + n0 - a - c) >= 2
                    adjusted_requested = (
                        base["episode_definition"] == "absolute_jump_ge4"
                        and base["signal_resolution"] == "15_minute_median_bins"
                        and eligibility_scope == "episode_pair_observed"
                        and outcome_kind == "first_next_lactate"
                        and rise_threshold == PRIMARY_RISE_THRESHOLD
                        and base["lag_window"] in ADJUSTED_LAGS
                    )
                    adjusted = (
                        _adjusted_modified_poisson(complete, outcome_column="_rise")
                        if adjusted_requested
                        else {
                            "adjusted_status": "not_requested_sensitivity",
                            "adjusted_risk_ratio": math.nan,
                            "adjusted_ci95_low": math.nan,
                            "adjusted_ci95_high": math.nan,
                            "adjusted_p_value": math.nan,
                            "adjusted_log_rr_se": math.nan,
                            "adjusted_parameter_count": 0,
                            "adjusted_events_per_parameter": math.nan,
                            "adjusted_covariates": "",
                            "adjusted_reason": "parsimonious adjustment reserved for focused and cumulative windows",
                        }
                    )
                    exposed_delta = exposed["_delta"].to_numpy(float)
                    control_delta = controls["_delta"].to_numpy(float)
                    rows.append(
                        {
                            **base,
                            "eligibility_scope": eligibility_scope,
                            "outcome_kind": outcome_kind,
                            "rise_threshold_mmol_l": rise_threshold,
                            "status": "estimated" if estimated else "underpowered",
                            "n_exposed": n1,
                            "n_unexposed": n0,
                            "events_exposed": a,
                            "events_unexposed": c,
                            "risk_exposed": a / n1 if n1 else math.nan,
                            "risk_unexposed": c / n0 if n0 else math.nan,
                            **rr,
                            "risk_difference": (a / n1) - (c / n0) if n1 and n0 else math.nan,
                            "fisher_two_sided_p": _fisher_p(a, n1 - a, c, n0 - c) if n1 and n0 else math.nan,
                            "median_delta_exposed": float(np.median(exposed_delta)) if n1 else math.nan,
                            "median_delta_unexposed": float(np.median(control_delta)) if n0 else math.nan,
                            "median_delta_difference": float(np.median(exposed_delta) - np.median(control_delta)) if n1 and n0 else math.nan,
                            "mann_whitney_two_sided_p": _mann_whitney_p(exposed_delta, control_delta),
                            **adjusted,
                            "analysis_status": POST_HOC_STATUS,
                            "control_anchor_definition": "no_episode_stay_transition_quantile_matched_to_exposed_anchor_time_distribution",
                            "interpretation_scope": "observational_time_aligned_episode_contrast_not_causal",
                        }
                    )
    result = pd.DataFrame(rows)
    for p_column, q_column in (
        ("fisher_two_sided_p", "global_exploratory_q_value"),
        ("mann_whitney_two_sided_p", "global_continuous_q_value"),
        ("adjusted_p_value", "global_adjusted_q_value"),
    ):
        result[q_column] = _bh_adjust(result[p_column])
    result["focused_q_value"] = np.nan
    result["focused_adjusted_q_value"] = np.nan
    focused = (
        result["episode_definition"].eq("absolute_jump_ge4")
        & result["signal_resolution"].eq("15_minute_median_bins")
        & result["eligibility_scope"].eq("episode_pair_observed")
        & result["outcome_kind"].eq("first_next_lactate")
        & result["rise_threshold_mmol_l"].eq(PRIMARY_RISE_THRESHOLD)
        & result["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
    )
    for _, indexes in result.loc[focused].groupby("dataset").groups.items():
        result.loc[indexes, "focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[indexes, "focused_adjusted_q_value"] = _bh_adjust(
            result.loc[indexes, "adjusted_p_value"]
        )
    result["focused_family"] = np.where(
        focused,
        "controlled_first_next_lactate_rise_ge0_5_after_absolute_jump_ge4_acute_and_delayed_windows",
        "not_in_focused_family",
    )
    result["kinetic_localization_q_value"] = np.nan
    result["kinetic_localization_adjusted_q_value"] = np.nan
    localization = (
        result["episode_definition"].eq("absolute_jump_ge4")
        & result["signal_resolution"].eq("15_minute_median_bins")
        & result["eligibility_scope"].eq("episode_pair_observed")
        & result["outcome_kind"].eq("first_next_lactate")
        & result["rise_threshold_mmol_l"].eq(PRIMARY_RISE_THRESHOLD)
        & result["lag_window"].isin(KINETIC_LOCALIZATION_LAGS)
    )
    for _, indexes in result.loc[localization].groupby("dataset").groups.items():
        result.loc[indexes, "kinetic_localization_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[indexes, "kinetic_localization_adjusted_q_value"] = _bh_adjust(
            result.loc[indexes, "adjusted_p_value"]
        )
    return result


def build_lactate_observation_process(records: pd.DataFrame) -> pd.DataFrame:
    """Audit whether instability changes the probability of lactate remeasurement."""
    if records.empty:
        return _empty(["status", "reason"])
    primary = records.loc[
        records["episode_definition"].eq("absolute_jump_ge4")
        & records["signal_resolution"].eq("15_minute_median_bins")
    ].copy()
    rows: list[dict[str, Any]] = []
    for (dataset, lag_window, lag_role), group in primary.groupby(
        ["dataset", "lag_window", "lag_role"], sort=False
    ):
        for observed_name, observed_column in (
            ("strict_prior_lactate_available", "prior_lactate_strictly_before_episode"),
            ("post_lactate_measured", "post_lactate_observed"),
            ("complete_strict_pair", "complete_lactate_pair"),
        ):
            exposed = group.loc[group["episode_exposed"].eq(1), observed_column].astype(int)
            controls = group.loc[group["episode_exposed"].eq(0), observed_column].astype(int)
            a, n1 = int(exposed.sum()), int(len(exposed))
            c, n0 = int(controls.sum()), int(len(controls))
            rows.append(
                {
                    "dataset": dataset,
                    "lag_window": lag_window,
                    "lag_role": lag_role,
                    "observation_target": observed_name,
                    "n_exposed_anchors": n1,
                    "n_unexposed_anchors": n0,
                    "observed_exposed": a,
                    "observed_unexposed": c,
                    "observation_rate_exposed": a / n1 if n1 else math.nan,
                    "observation_rate_unexposed": c / n0 if n0 else math.nan,
                    **_risk_ratio(a, n1, c, n0),
                    "fisher_two_sided_p": _fisher_p(a, n1 - a, c, n0 - c) if n1 and n0 else math.nan,
                    "analysis_status": POST_HOC_STATUS,
                    "interpretation": "measurement_process_diagnostic_not_biological_endpoint",
                }
            )
    result = pd.DataFrame(rows)
    result["focused_q_value"] = np.nan
    focused = result["lag_window"].isin(PRIMARY_FOCUSED_LAGS) & result[
        "observation_target"
    ].eq("complete_strict_pair")
    for _, indexes in result.loc[focused].groupby("dataset").groups.items():
        result.loc[indexes, "focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
    return result


def _measurement_weighted_row(
    group: pd.DataFrame,
    *,
    dataset: str,
    lag_window: str,
) -> dict[str, Any]:
    """Estimate a remeasurement-weighted RR conditional on a strict baseline."""
    base: dict[str, Any] = {
        "dataset": dataset,
        "lag_window": lag_window,
        "status": "not_estimable",
        "n_anchor_risk_set": 0,
        "n_complete_pairs": 0,
        "events_complete_pairs": 0,
        "selection_model_parameters": 0,
        "selection_events_per_parameter": math.nan,
        "outcome_model_parameters": 0,
        "outcome_events_per_parameter": math.nan,
        "weighted_risk_exposed": math.nan,
        "weighted_risk_unexposed": math.nan,
        "measurement_weighted_risk_ratio": math.nan,
        "measurement_weighted_ci95_low": math.nan,
        "measurement_weighted_ci95_high": math.nan,
        "measurement_weighted_p_value": math.nan,
        "effective_sample_size": math.nan,
        "predicted_observation_probability_min": math.nan,
        "predicted_observation_probability_max": math.nan,
        "probability_clipped_low_count": 0,
        "probability_clipped_high_count": 0,
        "positivity_status": "not_assessed",
        "stabilized_weight_p01": math.nan,
        "stabilized_weight_p99": math.nan,
        "covariates": "",
        "reason": "",
        "analysis_status": POST_HOC_STATUS,
        "measurement_model_scope": (
            "inverse_probability_of_post_lactate_remeasurement_"
            "conditional_on_strict_pre_episode_lactate"
        ),
        "claim_scope": "missing_outcome_sensitivity_not_causal_correction",
    }
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        base["reason"] = f"statsmodels unavailable: {exc}"
        return base

    risk_set = group.loc[
        pd.to_numeric(
            group["prior_lactate_strictly_before_episode"], errors="coerce"
        ).eq(1)
    ].copy()
    risk_set["_observed"] = pd.to_numeric(
        risk_set["complete_lactate_pair"], errors="coerce"
    )
    risk_set = risk_set.loc[
        risk_set["_observed"].isin([0, 1])
        & risk_set["episode_exposed"].isin([0, 1])
    ].copy()
    base["n_anchor_risk_set"] = int(len(risk_set))
    observed = int(risk_set["_observed"].sum())
    unobserved = int(len(risk_set) - observed)
    if len(risk_set) < MIN_ROWS or min(observed, unobserved) < MIN_EVENTS:
        base["status"] = "underpowered_selection_model"
        base["reason"] = "insufficient observed/unobserved post-lactate measurements"
        return base

    covariate_candidates = (
        "prior_lactate",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
        "age",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
    )

    def design_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        design = pd.DataFrame(index=frame.index)
        design["episode_exposed"] = pd.to_numeric(
            frame["episode_exposed"], errors="coerce"
        )
        retained: list[str] = []
        for column in covariate_candidates:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() < max(20, int(len(frame) * 0.2)):
                continue
            values = values.fillna(float(values.median()))
            if values.nunique(dropna=False) <= 1:
                continue
            if column not in {"shock_icd_flag", "baseline_vasoactive_flag"}:
                scale = float(values.std(ddof=0))
                if np.isfinite(scale) and scale > 0:
                    values = (values - float(values.mean())) / scale
            design[column] = values
            retained.append(column)
        return sm.add_constant(design.astype(float), has_constant="add"), retained

    selection_design, retained = design_matrix(risk_set)
    selection_parameters = int(selection_design.shape[1])
    selection_epp = min(observed, unobserved) / max(selection_parameters, 1)
    base.update(
        {
            "selection_model_parameters": selection_parameters,
            "selection_events_per_parameter": selection_epp,
            "covariates": ",".join(retained),
        }
    )
    if selection_epp < MIN_EVENTS_PER_PARAMETER:
        base["status"] = "underpowered_selection_model"
        base["reason"] = "selection_events_per_parameter_below_5_fail_closed"
        return base
    try:
        selection_fit = sm.GLM(
            risk_set["_observed"].astype(int),
            selection_design,
            family=sm.families.Binomial(),
        ).fit()
        probability = np.asarray(selection_fit.predict(selection_design), dtype=float)
    except Exception as exc:
        base["status"] = "non_estimable_selection_model"
        base["reason"] = str(exc)[:500]
        return base
    if not np.isfinite(probability).all():
        base["status"] = "non_estimable_selection_model"
        base["reason"] = "non-finite observation probabilities"
        return base
    low_count = int(np.sum(probability < 0.02))
    high_count = int(np.sum(probability > 0.98))
    base["probability_clipped_low_count"] = low_count
    base["probability_clipped_high_count"] = high_count
    base["positivity_status"] = (
        "limited_predictions_clipped"
        if low_count or high_count
        else "adequate_no_probability_clipping"
    )
    probability = np.clip(probability, 0.02, 0.98)
    risk_set["_observation_probability"] = probability
    base["predicted_observation_probability_min"] = float(probability.min())
    base["predicted_observation_probability_max"] = float(probability.max())

    complete = risk_set.loc[risk_set["_observed"].eq(1)].copy()
    complete["_rise"] = pd.to_numeric(
        complete["first_lactate_delta"], errors="coerce"
    ).ge(PRIMARY_RISE_THRESHOLD).astype(int)
    complete = complete.loc[
        pd.to_numeric(complete["first_lactate_delta"], errors="coerce").notna()
    ].copy()
    events = int(complete["_rise"].sum())
    non_events = int(len(complete) - events)
    base.update(
        {
            "n_complete_pairs": int(len(complete)),
            "events_complete_pairs": events,
        }
    )
    if len(complete) < MIN_ROWS or min(events, non_events) < MIN_EVENTS:
        base["status"] = "underpowered_outcome_model"
        base["reason"] = "insufficient complete-pair outcome events/non-events"
        return base

    numerator = risk_set.groupby("episode_exposed")["_observed"].mean()
    complete["_weight"] = (
        complete["episode_exposed"].map(numerator).astype(float)
        / complete["_observation_probability"].astype(float)
    )
    if not np.isfinite(complete["_weight"]).all() or complete["_weight"].le(0).any():
        base["status"] = "non_estimable_weights"
        base["reason"] = "non-finite or non-positive stabilized weights"
        return base
    lower_weight, upper_weight = np.quantile(complete["_weight"], [0.01, 0.99])
    complete["_weight"] = complete["_weight"].clip(lower_weight, upper_weight)
    base["stabilized_weight_p01"] = float(lower_weight)
    base["stabilized_weight_p99"] = float(upper_weight)
    weight = complete["_weight"].to_numpy(float)
    base["effective_sample_size"] = float(
        np.square(weight.sum()) / np.square(weight).sum()
    )
    weighted_risks: dict[int, float] = {}
    for exposure in (0, 1):
        local = complete.loc[complete["episode_exposed"].eq(exposure)]
        if local.empty or float(local["_weight"].sum()) <= 0:
            weighted_risks[exposure] = math.nan
        else:
            weighted_risks[exposure] = float(
                np.average(local["_rise"], weights=local["_weight"])
            )
    base["weighted_risk_exposed"] = weighted_risks[1]
    base["weighted_risk_unexposed"] = weighted_risks[0]

    outcome_design, outcome_covariates = design_matrix(complete)
    outcome_parameters = int(outcome_design.shape[1])
    outcome_epp = min(events, non_events) / max(outcome_parameters, 1)
    base["outcome_model_parameters"] = outcome_parameters
    base["outcome_events_per_parameter"] = outcome_epp
    base["covariates"] = ",".join(outcome_covariates)
    if outcome_epp < MIN_EVENTS_PER_PARAMETER:
        base["status"] = "underpowered_outcome_model"
        base["reason"] = "outcome_events_per_parameter_below_5_fail_closed"
        return base
    try:
        fit = sm.GLM(
            complete["_rise"].astype(int),
            outcome_design,
            family=sm.families.Poisson(),
            freq_weights=complete["_weight"].astype(float),
        ).fit(cov_type="HC0")
        coefficient = float(fit.params["episode_exposed"])
        low, high = fit.conf_int().loc["episode_exposed"]
        values = np.exp([coefficient, float(low), float(high)])
        if not np.isfinite(values).all():
            raise ValueError("non-finite weighted risk-ratio inference")
        base.update(
            {
                "status": "estimated",
                "measurement_weighted_risk_ratio": float(values[0]),
                "measurement_weighted_ci95_low": float(values[1]),
                "measurement_weighted_ci95_high": float(values[2]),
                "measurement_weighted_p_value": float(
                    fit.pvalues["episode_exposed"]
                ),
                "reason": "",
            }
        )
    except Exception as exc:
        base["status"] = "non_estimable_outcome_model"
        base["reason"] = str(exc)[:500]
    return base


def build_lactate_episode_measurement_weighted(
    records: pd.DataFrame,
) -> pd.DataFrame:
    """Sensitivity for informative post-episode lactate remeasurement."""
    if records.empty:
        return _empty(["status", "reason"])
    primary = records.loc[
        records["episode_definition"].eq("absolute_jump_ge4")
        & records["signal_resolution"].eq("15_minute_median_bins")
        & records["lag_window"].isin({*PRIMARY_FOCUSED_LAGS, "0_to_8h"})
    ].copy()
    rows = [
        _measurement_weighted_row(
            group,
            dataset=str(dataset),
            lag_window=str(lag_window),
        )
        for (dataset, lag_window), group in primary.groupby(
            ["dataset", "lag_window"], sort=False
        )
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["focused_q_value"] = np.nan
    focused = result["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
    for _, indexes in result.loc[focused].groupby("dataset").groups.items():
        result.loc[indexes, "focused_q_value"] = _bh_adjust(
            result.loc[indexes, "measurement_weighted_p_value"]
        )
    return result


def build_lactate_episode_strata(records: pd.DataFrame) -> pd.DataFrame:
    """Focused phenotype and measurement-density sensitivity strata."""
    if records.empty:
        return _empty(["status", "reason"])
    primary = records.loc[
        records["episode_definition"].eq("absolute_jump_ge4")
        & records["signal_resolution"].eq("15_minute_median_bins")
        & records["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
        & pd.to_numeric(records["first_lactate_delta"], errors="coerce").notna()
    ].copy()
    primary["_rise"] = pd.to_numeric(
        primary["first_lactate_delta"], errors="coerce"
    ).ge(PRIMARY_RISE_THRESHOLD).astype(int)
    sampling_median = primary.groupby("dataset")[
        "preanchor_sampling_density_per_hr"
    ].transform("median")
    primary["_sampling_high"] = primary[
        "preanchor_sampling_density_per_hr"
    ].ge(sampling_median)
    stratum_specs: list[tuple[str, str, pd.Series]] = [
        ("overall", "all", pd.Series(True, index=primary.index)),
    ]
    candidates = (
        ("shock", "shock_present", "shock_icd_flag", 1),
        ("shock", "shock_absent", "shock_icd_flag", 0),
        ("baseline_vasoactive", "present", "baseline_vasoactive_flag", 1),
        ("baseline_vasoactive", "absent", "baseline_vasoactive_flag", 0),
    )
    for variable, level, column, value in candidates:
        if column in primary:
            stratum_specs.append(
                (
                    variable,
                    level,
                    pd.to_numeric(primary[column], errors="coerce").eq(value),
                )
            )
    stratum_specs.extend(
        [
            ("strict_prior_lactate", "lte_2", primary["prior_lactate"].le(2.0)),
            ("strict_prior_lactate", "gt_2", primary["prior_lactate"].gt(2.0)),
            ("anchor_absolute_spo2", "lt_90", primary["anchor_spo2"].lt(90.0)),
            ("anchor_absolute_spo2", "ge_90", primary["anchor_spo2"].ge(90.0)),
            ("preanchor_sampling", "high", primary["_sampling_high"]),
            ("preanchor_sampling", "low", ~primary["_sampling_high"]),
        ]
    )
    rows: list[dict[str, Any]] = []
    for variable, level, mask in stratum_specs:
        selected = primary.loc[mask.fillna(False)]
        for (dataset, lag_window), group in selected.groupby(
            ["dataset", "lag_window"], sort=False
        ):
            exposed = group.loc[group["episode_exposed"].eq(1), "_rise"]
            controls = group.loc[group["episode_exposed"].eq(0), "_rise"]
            a, n1 = int(exposed.sum()), int(len(exposed))
            c, n0 = int(controls.sum()), int(len(controls))
            rows.append(
                {
                    "dataset": dataset,
                    "lag_window": lag_window,
                    "stratum_variable": variable,
                    "stratum_level": level,
                    "n_exposed": n1,
                    "n_unexposed": n0,
                    "events_exposed": a,
                    "events_unexposed": c,
                    **_risk_ratio(a, n1, c, n0),
                    "fisher_two_sided_p": _fisher_p(a, n1 - a, c, n0 - c) if n1 and n0 else math.nan,
                    "status": "estimated" if n1 >= 5 and n0 >= 5 else "underpowered",
                    "analysis_status": POST_HOC_STATUS,
                    "multiplicity_scope": "descriptive_heterogeneity_no_significance_claim",
                }
            )
    return pd.DataFrame(rows)


def build_lactate_episode_meta_analysis(controlled: pd.DataFrame) -> pd.DataFrame:
    """Combine harmonized MIMIC/eICU episode RRs and report heterogeneity."""
    if controlled.empty:
        return _empty(["status", "reason"])
    selected = controlled.loc[
        controlled["episode_definition"].eq("absolute_jump_ge4")
        & controlled["signal_resolution"].eq("15_minute_median_bins")
        & controlled["eligibility_scope"].eq("episode_pair_observed")
        & controlled["outcome_kind"].eq("first_next_lactate")
        & controlled["rise_threshold_mmol_l"].eq(PRIMARY_RISE_THRESHOLD)
        & controlled["lag_window"].isin(META_ANALYSIS_LAGS)
        & controlled["risk_ratio"].gt(0)
        & controlled["log_risk_ratio_se"].gt(0)
    ].copy()
    rows: list[dict[str, Any]] = []
    for lag_window, group in selected.groupby("lag_window", sort=False):
        yi = np.log(pd.to_numeric(group["risk_ratio"], errors="coerce").to_numpy(float))
        sei = pd.to_numeric(group["log_risk_ratio_se"], errors="coerce").to_numpy(float)
        valid = np.isfinite(yi) & np.isfinite(sei) & (sei > 0)
        yi, sei = yi[valid], sei[valid]
        if len(yi) == 0:
            continue
        weights = 1.0 / np.square(sei)
        fixed = float(np.sum(weights * yi) / np.sum(weights))
        fixed_se = float(math.sqrt(1.0 / np.sum(weights)))
        q = float(np.sum(weights * np.square(yi - fixed)))
        df = max(len(yi) - 1, 0)
        c = float(np.sum(weights) - np.sum(np.square(weights)) / np.sum(weights))
        tau2 = max((q - df) / c, 0.0) if df > 0 and c > 0 else 0.0
        random_weights = 1.0 / (np.square(sei) + tau2)
        random_effect = float(np.sum(random_weights * yi) / np.sum(random_weights))
        random_se = float(math.sqrt(1.0 / np.sum(random_weights)))
        try:
            from scipy.stats import norm

            fixed_p = float(2 * norm.sf(abs(fixed / fixed_se)))
            random_p = float(2 * norm.sf(abs(random_effect / random_se)))
        except Exception:  # pragma: no cover
            fixed_p = math.nan
            random_p = math.nan
        rows.append(
            {
                "lag_window": lag_window,
                "meta_family": (
                    "focused_acute_delayed"
                    if lag_window in PRIMARY_FOCUSED_LAGS
                    else "kinetic_localization_secondary"
                ),
                "datasets_contributing": int(len(yi)),
                "dataset_names": ",".join(group.loc[valid, "dataset"].astype(str)),
                "fixed_effect_risk_ratio": math.exp(fixed),
                "fixed_effect_ci95_low": math.exp(fixed - 1.96 * fixed_se),
                "fixed_effect_ci95_high": math.exp(fixed + 1.96 * fixed_se),
                "fixed_effect_p": fixed_p,
                "random_effect_risk_ratio": math.exp(random_effect),
                "random_effect_ci95_low": math.exp(random_effect - 1.96 * random_se),
                "random_effect_ci95_high": math.exp(random_effect + 1.96 * random_se),
                "random_effect_p": random_p,
                "cochran_q": q,
                "heterogeneity_df": df,
                "i_squared_percent": max((q - df) / q * 100.0, 0.0) if q > 0 and df > 0 else 0.0,
                "tau_squared": tau2,
                "status": "estimated" if len(yi) >= 2 else "single_dataset_only",
                "analysis_status": POST_HOC_STATUS,
                "claim_scope": "cross_dataset_synthesis_not_independent_preregistration",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["focused_fixed_q_value"] = np.nan
        result["focused_random_q_value"] = np.nan
        result["kinetic_localization_fixed_q_value"] = np.nan
        result["kinetic_localization_random_q_value"] = np.nan
        focused = result["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
        localization = result["lag_window"].isin(KINETIC_LOCALIZATION_LAGS)
        result.loc[focused, "focused_fixed_q_value"] = _bh_adjust(
            result.loc[focused, "fixed_effect_p"]
        )
        result.loc[focused, "focused_random_q_value"] = _bh_adjust(
            result.loc[focused, "random_effect_p"]
        )
        result.loc[localization, "kinetic_localization_fixed_q_value"] = _bh_adjust(
            result.loc[localization, "fixed_effect_p"]
        )
        result.loc[localization, "kinetic_localization_random_q_value"] = _bh_adjust(
            result.loc[localization, "random_effect_p"]
        )
    return result


def build_lactate_episode_evidence_summary(
    paired: pd.DataFrame,
    controlled: pd.DataFrame,
    meta: pd.DataFrame,
    measurement_weighted: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Grade separate evidence components without an all-or-nothing gate."""
    rows: list[dict[str, Any]] = []
    if not paired.empty:
        paired_focus = paired.loc[
            paired["episode_definition"].eq("absolute_jump_ge4")
            & paired["signal_resolution"].eq("15_minute_median_bins")
            & paired["eligibility_scope"].eq("episode_pair_observed")
            & paired["outcome_kind"].eq("first_next_lactate")
            & paired["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
        ]
        for row in paired_focus.to_dict("records"):
            positive = pd.notna(row.get("median_delta_mmol_l")) and row["median_delta_mmol_l"] > 0
            significant = pd.notna(row.get("focused_q_value")) and row["focused_q_value"] < 0.05
            grade = (
                "paired_change_supported"
                if positive and significant
                else "directionally_positive_inconclusive"
                if positive
                else "no_positive_paired_change"
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "lag_window": row["lag_window"],
                    "evidence_component": "strictly_pre_episode_paired_change",
                    "evidence_grade": grade,
                    "effect": row.get("median_delta_mmol_l"),
                    "ci95_low": row.get("median_ci95_low"),
                    "ci95_high": row.get("median_ci95_high"),
                    "focused_q_value": row.get("focused_q_value"),
                    "analysis_status": POST_HOC_STATUS,
                }
            )
    if not controlled.empty:
        control_focus = controlled.loc[
            controlled["episode_definition"].eq("absolute_jump_ge4")
            & controlled["signal_resolution"].eq("15_minute_median_bins")
            & controlled["eligibility_scope"].eq("episode_pair_observed")
            & controlled["outcome_kind"].eq("first_next_lactate")
            & controlled["rise_threshold_mmol_l"].eq(PRIMARY_RISE_THRESHOLD)
            & controlled["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
        ]
        for row in control_focus.to_dict("records"):
            positive = pd.notna(row.get("risk_ratio")) and row["risk_ratio"] > 1
            significant = pd.notna(row.get("focused_q_value")) and row["focused_q_value"] < 0.05
            grade = (
                "controlled_association_supported"
                if positive and significant
                else "directionally_positive_inconclusive"
                if positive
                else "no_positive_controlled_association"
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "lag_window": row["lag_window"],
                    "evidence_component": "time_aligned_controlled_risk_ratio",
                    "evidence_grade": grade,
                    "effect": row.get("risk_ratio"),
                    "ci95_low": row.get("risk_ratio_ci95_low"),
                    "ci95_high": row.get("risk_ratio_ci95_high"),
                    "focused_q_value": row.get("focused_q_value"),
                    "analysis_status": POST_HOC_STATUS,
                }
            )
            adjusted_estimable = row.get("adjusted_status") == "estimated"
            adjusted_positive = (
                adjusted_estimable
                and pd.notna(row.get("adjusted_risk_ratio"))
                and row["adjusted_risk_ratio"] > 1
            )
            adjusted_significant = (
                pd.notna(row.get("focused_adjusted_q_value"))
                and row["focused_adjusted_q_value"] < 0.05
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "lag_window": row["lag_window"],
                    "evidence_component": "covariate_adjusted_risk_ratio",
                    "evidence_grade": (
                        "not_estimable_underpowered"
                        if not adjusted_estimable
                        else "adjusted_association_supported"
                        if adjusted_positive and adjusted_significant
                        else "directionally_positive_inconclusive"
                        if adjusted_positive
                        else "no_positive_adjusted_association"
                    ),
                    "effect": row.get("adjusted_risk_ratio"),
                    "ci95_low": row.get("adjusted_ci95_low"),
                    "ci95_high": row.get("adjusted_ci95_high"),
                    "focused_q_value": row.get("focused_adjusted_q_value"),
                    "analysis_status": POST_HOC_STATUS,
                }
            )
    if measurement_weighted is not None and not measurement_weighted.empty:
        weighted_focus = measurement_weighted.loc[
            measurement_weighted["lag_window"].isin(PRIMARY_FOCUSED_LAGS)
        ]
        for row in weighted_focus.to_dict("records"):
            estimable = row.get("status") == "estimated"
            positive = (
                estimable
                and pd.notna(row.get("measurement_weighted_risk_ratio"))
                and row["measurement_weighted_risk_ratio"] > 1
            )
            significant = (
                pd.notna(row.get("focused_q_value"))
                and row["focused_q_value"] < 0.05
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "lag_window": row["lag_window"],
                    "evidence_component": "remeasurement_weighted_risk_ratio",
                    "evidence_grade": (
                        "not_estimable_underpowered"
                        if not estimable
                        else "measurement_weighted_association_supported"
                        if positive and significant
                        else "directionally_positive_inconclusive"
                        if positive
                        else "no_positive_measurement_weighted_association"
                    ),
                    "effect": row.get("measurement_weighted_risk_ratio"),
                    "ci95_low": row.get("measurement_weighted_ci95_low"),
                    "ci95_high": row.get("measurement_weighted_ci95_high"),
                    "focused_q_value": row.get("focused_q_value"),
                    "analysis_status": POST_HOC_STATUS,
                }
            )
    if not meta.empty:
        meta_focus = meta.loc[meta["lag_window"].isin(PRIMARY_FOCUSED_LAGS)]
        for row in meta_focus.to_dict("records"):
            positive = pd.notna(row.get("random_effect_risk_ratio")) and row[
                "random_effect_risk_ratio"
            ] > 1
            significant = pd.notna(row.get("focused_random_q_value")) and row[
                "focused_random_q_value"
            ] < 0.05
            rows.append(
                {
                    "dataset": "cross_dataset",
                    "lag_window": row["lag_window"],
                    "evidence_component": "random_effect_meta_analysis",
                    "evidence_grade": (
                        "cross_dataset_association_supported"
                        if positive and significant
                        else "directionally_positive_inconclusive"
                        if positive
                        else "no_positive_cross_dataset_association"
                    ),
                    "effect": row.get("random_effect_risk_ratio"),
                    "ci95_low": row.get("random_effect_ci95_low"),
                    "ci95_high": row.get("random_effect_ci95_high"),
                    "focused_q_value": row.get("focused_random_q_value"),
                    "analysis_status": POST_HOC_STATUS,
                }
            )
    return pd.DataFrame(rows)


def run_lactate_episode_analyses(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = 500,
) -> dict[str, pd.DataFrame]:
    """Run the complete episode-anchored lactate amendment."""
    records = build_episode_lactate_records(events_df, cohort_df, analysis_df)
    paired = build_episode_lactate_paired_summary(
        records, bootstrap_repetitions=bootstrap_repetitions
    )
    controlled = build_episode_lactate_controlled_summary(records)
    observation = build_lactate_observation_process(records)
    measurement_weighted = build_lactate_episode_measurement_weighted(records)
    strata = build_lactate_episode_strata(records)
    meta = build_lactate_episode_meta_analysis(controlled)
    evidence = build_lactate_episode_evidence_summary(
        paired,
        controlled,
        meta,
        measurement_weighted,
    )
    primary_records = records.loc[
        records.get("episode_definition", pd.Series(index=records.index, dtype=str)).eq(
            "absolute_jump_ge4"
        )
        & records.get(
            "signal_resolution", pd.Series(index=records.index, dtype=str)
        ).eq("15_minute_median_bins")
    ].copy()
    return {
        "lactate_episode_records": primary_records,
        "lactate_episode_paired_summary": paired,
        "lactate_episode_controlled_summary": controlled,
        "lactate_episode_observation_process": observation,
        "lactate_episode_measurement_weighted": measurement_weighted,
        "lactate_episode_stratified_sensitivity": strata,
        "lactate_episode_meta_analysis": meta,
        "lactate_episode_evidence_summary": evidence,
    }


__all__ = [
    "EPISODE_DEFINITIONS",
    "LAG_WINDOWS",
    "POST_HOC_STATUS",
    "build_episode_lactate_controlled_summary",
    "build_episode_lactate_paired_summary",
    "build_episode_lactate_records",
    "build_lactate_episode_evidence_summary",
    "build_lactate_episode_measurement_weighted",
    "build_lactate_episode_meta_analysis",
    "build_lactate_episode_strata",
    "build_lactate_observation_process",
    "run_lactate_episode_analyses",
]
