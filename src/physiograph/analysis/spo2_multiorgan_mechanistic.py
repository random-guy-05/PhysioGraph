"""Episode-anchored SpO2 analyses for all cached decompensation endpoints.

This post-result amendment generalizes the lactate design without searching for
the most favorable endpoint/window.  The same time-aligned episode/control
anchors are evaluated in a frozen grid of non-overlapping and cumulative lag
windows.  Laboratory baselines are strictly before the SpO2 episode; incident
interventions use anchor-specific risk sets; measurement and censoring
selection are reported and reweighted when estimable.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from physiograph.cohort.harmonization import (
    HARMONIZED_HF_PHENOTYPE,
    filter_to_harmonized_cohort,
    harmonize_hf_cohort,
)

from .spo2_lactate_mechanistic import (
    EPISODE_DEFINITIONS,
    LACTATE_LOOKBACK_MINUTES,
    MIN_EVENTS,
    MIN_EVENTS_PER_PARAMETER,
    MIN_ROWS,
    _bh_adjust,
    _fisher_p,
    _mann_whitney_p,
    _merge_anchor_context,
    _prepare_spo2_transitions,
    _risk_ratio,
    _select_anchor_per_stay,
)


POST_HOC_STATUS = "post_result_multiorgan_episode_amendment_exploratory"
PRIMARY_RISE_THRESHOLD = 0.5
BOOTSTRAP_SEED = 20260901

# A frozen grid prevents outcome-specific window shopping.  Focused windows are
# chosen by endpoint family below; every remaining row is an explicitly
# secondary localization/cumulative companion.
LAG_WINDOWS: tuple[tuple[str, float, float, str], ...] = (
    ("0_to_1h", 0.0, 60.0, "acute_localization"),
    ("1_to_4h", 60.0, 240.0, "acute_localization"),
    ("4_to_8h", 240.0, 480.0, "early_localization"),
    ("8_to_12h", 480.0, 720.0, "early_localization"),
    ("12_to_24h", 720.0, 1440.0, "late_localization"),
    ("1_to_8h", 60.0, 480.0, "lactate_delayed_companion"),
    ("0_to_4h", 0.0, 240.0, "cumulative_companion"),
    ("0_to_8h", 0.0, 480.0, "cumulative_companion"),
    ("0_to_12h", 0.0, 720.0, "cumulative_companion"),
    ("0_to_24h", 0.0, 1440.0, "cumulative_companion"),
    ("8_to_24h", 480.0, 1440.0, "late_cumulative_companion"),
    ("4_to_12h", 240.0, 720.0, "early_cumulative_companion"),
)
WINDOW_BY_NAME = {
    label: (lower, upper, role) for label, lower, upper, role in LAG_WINDOWS
}

FOCUSED_WINDOWS_BY_FAMILY: dict[str, tuple[str, str]] = {
    "lactate": ("0_to_1h", "1_to_8h"),
    "laboratory_organ_injury": ("0_to_8h", "8_to_24h"),
    "hemodynamic_support": ("0_to_4h", "4_to_12h"),
    "incident_intervention": ("0_to_4h", "4_to_12h"),
    "mortality": ("0_to_8h", "8_to_24h"),
    "urine_output": ("0_to_4h", "4_to_12h"),
    "hepatic_composite": ("0_to_8h", "8_to_24h"),
    "organ_injury_composite": ("0_to_8h", "8_to_24h"),
    "early_decompensation_composite": ("0_to_4h", "4_to_12h"),
    "all_decompensation_composite": ("0_to_8h", "8_to_24h"),
}

# Carry every frozen lactate-amendment exposure definition to every endpoint:
# absolute instability, clinically directional drops at two thresholds, a
# recovery-rise specificity comparator, and raw-time resolution sensitivities.
ANALYZED_DEFINITIONS = tuple(EPISODE_DEFINITIONS)


@dataclass(frozen=True)
class LabSpec:
    endpoint: str
    concept: str
    family: str
    direction: str
    primary_variant: str
    plausible_low: float
    plausible_high: float
    event_rule: Callable[[float, float], bool]
    change_scale: str


LAB_SPECS: tuple[LabSpec, ...] = (
    LabSpec(
        "lactate_rise",
        "lactate",
        "lactate",
        "high",
        "first_next",
        0.0,
        30.0,
        lambda baseline, post: post - baseline >= 0.5,
        "absolute_delta_mmol_l",
    ),
    LabSpec(
        "aki_creatinine",
        "creatinine",
        "laboratory_organ_injury",
        "high",
        "worst_in_window",
        0.05,
        30.0,
        lambda baseline, post: post - baseline >= 0.3 or post / baseline >= 1.5,
        "absolute_delta_mg_dl",
    ),
    LabSpec(
        "bilirubin_worsening",
        "bilirubin_total",
        "laboratory_organ_injury",
        "high",
        "worst_in_window",
        0.01,
        100.0,
        lambda baseline, post: post - baseline >= 0.5 or post / baseline >= 1.5,
        "absolute_delta_mg_dl",
    ),
    LabSpec(
        "ast_injury",
        "ast",
        "laboratory_organ_injury",
        "high",
        "worst_in_window",
        0.01,
        100_000.0,
        lambda baseline, post: (baseline < 200.0 and post >= 200.0)
        or post / baseline >= 2.0,
        "relative_ratio_minus_one",
    ),
    LabSpec(
        "alt_injury",
        "alt",
        "laboratory_organ_injury",
        "high",
        "worst_in_window",
        0.01,
        100_000.0,
        lambda baseline, post: (baseline < 200.0 and post >= 200.0)
        or post / baseline >= 2.0,
        "relative_ratio_minus_one",
    ),
    LabSpec(
        "inr_injury",
        "inr",
        "laboratory_organ_injury",
        "high",
        "worst_in_window",
        0.01,
        30.0,
        lambda baseline, post: post - baseline >= 0.3 or post / baseline >= 1.5,
        "absolute_delta",
    ),
    LabSpec(
        "platelet_injury",
        "platelets",
        "laboratory_organ_injury",
        "low",
        "worst_in_window",
        1.0,
        5_000.0,
        lambda baseline, post: (baseline >= 100.0 and post < 100.0)
        or post / baseline <= 0.70,
        "decline_fraction",
    ),
)


EVENT_SPECS: tuple[dict[str, str], ...] = (
    {
        "endpoint": "pressor_initiation",
        "concept": "pressor_initiation",
        "family": "hemodynamic_support",
        "source_column": "pressor_source_available",
        "baseline_flag": "baseline_vasoactive_flag",
    },
    {
        "endpoint": "mcs_initiation",
        "concept": "mcs",
        "family": "incident_intervention",
        "source_column": "mcs_source_available",
        "baseline_flag": "baseline_mcs_flag",
    },
    {
        "endpoint": "rrt_initiation",
        "concept": "rrt",
        "family": "incident_intervention",
        "source_column": "rrt_source_available",
        "baseline_flag": "rrt_or_dialysis_flag",
    },
    {
        "endpoint": "invasive_ventilation_initiation",
        "concept": "mechanical_ventilation",
        "family": "incident_intervention",
        "source_column": "respiratory_source_available",
        "baseline_flag": "mechanical_ventilation_flag",
        "requires_concept_presence": "1",
    },
    {
        "endpoint": "intubation_initiation",
        "concept": "intubation",
        "family": "incident_intervention",
        "source_column": "respiratory_source_available",
        "baseline_flag": "intubation_flag",
        "requires_concept_presence": "1",
    },
    {
        "endpoint": "death",
        "concept": "death",
        "family": "mortality",
        "source_column": "death_source_available",
        "baseline_flag": "",
    },
)


REQUIRED_CONCEPTS = {
    *(spec.concept for spec in LAB_SPECS),
    "troponin_t",
    "troponin_i",
    "vis",
    "urine_output",
    *(spec["concept"] for spec in EVENT_SPECS),
}


class EventLookup:
    """Compact per-concept/per-stay sorted arrays for repeated lag queries."""

    def __init__(self, events_df: pd.DataFrame) -> None:
        events = events_df.copy()
        if "dataset" not in events:
            events["dataset"] = "unknown"
        events["concept"] = events.get("concept", "").fillna("").astype(str).str.lower()
        events = events.loc[events["concept"].isin(REQUIRED_CONCEPTS)].copy()
        events["offset_minutes"] = pd.to_numeric(
            events.get("offset_minutes"), errors="coerce"
        )
        events["value_numeric"] = pd.to_numeric(
            events.get("value_numeric"), errors="coerce"
        )
        events = events.dropna(subset=["offset_minutes", "stay_id"])
        # Same-time laboratory duplicates must not create artificial extrema.
        events = (
            events.groupby(
                ["dataset", "stay_id", "concept", "offset_minutes"],
                as_index=False,
                dropna=False,
            )["value_numeric"]
            .median()
            .sort_values(["dataset", "stay_id", "concept", "offset_minutes"])
        )
        self._groups: dict[
            tuple[str, Any, str], tuple[np.ndarray, np.ndarray]
        ] = {
            (str(dataset), stay_id, str(concept)): (
                group["offset_minutes"].to_numpy(float),
                group["value_numeric"].to_numpy(float),
            )
            for (dataset, stay_id, concept), group in events.groupby(
                ["dataset", "stay_id", "concept"], sort=False
            )
        }
        self.datasets_by_concept: dict[str, set[str]] = {
            str(concept): set(group["dataset"].astype(str))
            for concept, group in events.groupby("concept", sort=False)
        }

    def get(
        self, dataset: str, stay_id: Any, concept: str
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._groups.get(
            (str(dataset), stay_id, str(concept)),
            (np.array([], dtype=float), np.array([], dtype=float)),
        )


def _merge_anchor_extras(
    anchors: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None,
) -> pd.DataFrame:
    keys = ["dataset", "stay_id"]
    extra_candidates = (
        "mcs_source_available",
        "pressor_source_available",
        "vis_source_available",
        "urine_output_source_available",
        "respiratory_source_available",
        "rrt_source_available",
        "rrt_or_dialysis_flag",
        "mechanical_ventilation_flag",
        "intubation_flag",
        "admission_weight_kg",
        "baseline_weight_outcome",
    )
    result = anchors
    for source in (cohort_df, analysis_df):
        if source is None or source.empty or not set(keys).issubset(source):
            continue
        extras = [
            column
            for column in extra_candidates
            if column in source and column not in result
        ]
        if extras:
            result = result.merge(
                source[[*keys, *extras]].drop_duplicates(keys),
                on=keys,
                how="left",
            )
    result["death_source_available"] = 1
    return result


def _build_anchor_sets(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None,
) -> dict[tuple[str, str], pd.DataFrame]:
    transition_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    anchors_by_definition: dict[tuple[str, str], pd.DataFrame] = {}
    for definition in ANALYZED_DEFINITIONS:
        resolution = str(definition["signal_resolution"])
        if resolution not in transition_cache:
            transition_cache[resolution] = _prepare_spo2_transitions(
                events_df, signal_resolution=resolution
            )
        transitions, support = transition_cache[resolution]
        anchors = _select_anchor_per_stay(
            transitions, support, definition=definition
        )
        if anchors.empty:
            continue
        anchors = _merge_anchor_context(anchors, cohort_df, analysis_df)
        anchors = _merge_anchor_extras(anchors, cohort_df, analysis_df)
        anchors_by_definition[
            (
                str(definition["episode_definition"]),
                str(definition["signal_resolution"]),
            )
        ] = anchors.reset_index(drop=True)
    return anchors_by_definition


def _followup_limit(anchor: dict[str, Any], *, include_death: bool = True) -> float:
    candidates = [
        pd.to_numeric(anchor.get("followup_end_offset_minutes"), errors="coerce")
    ]
    if include_death:
        candidates.append(
            pd.to_numeric(anchor.get("death_offset_minutes"), errors="coerce")
        )
    finite = [float(value) for value in candidates if pd.notna(value) and np.isfinite(value)]
    return min(finite) if finite else math.inf


def _change_value(spec: LabSpec, baseline: float, post: float) -> float:
    if spec.change_scale == "relative_ratio_minus_one":
        return post / baseline - 1.0
    if spec.change_scale == "decline_fraction":
        return 1.0 - post / baseline
    return post - baseline


def _base_record(
    anchor: dict[str, Any],
    *,
    endpoint: str,
    family: str,
    lag_window: str,
    lag_role: str,
    outcome_variant: str,
) -> dict[str, Any]:
    return {
        "dataset": str(anchor["dataset"]),
        "stay_id": anchor["stay_id"],
        "person_id": anchor.get("person_id"),
        "episode_definition": anchor.get("episode_definition"),
        "signal_resolution": anchor.get("signal_resolution"),
        "definition_role": anchor.get("definition_role"),
        "episode_exposed": int(anchor.get("episode_exposed", 0)),
        "anchor_offset_minutes": float(anchor["anchor_offset_minutes"]),
        "anchor_spo2": pd.to_numeric(anchor.get("anchor_spo2"), errors="coerce"),
        "preanchor_sampling_density_per_hr": pd.to_numeric(
            anchor.get("preanchor_sampling_density_per_hr"), errors="coerce"
        ),
        "age": pd.to_numeric(anchor.get("age"), errors="coerce"),
        "shock_icd_flag": pd.to_numeric(
            anchor.get("shock_icd_flag"), errors="coerce"
        ),
        "baseline_vasoactive_flag": pd.to_numeric(
            anchor.get("baseline_vasoactive_flag"), errors="coerce"
        ),
        "endpoint": endpoint,
        "endpoint_family": family,
        "lag_window": lag_window,
        "lag_role": lag_role,
        "outcome_variant": outcome_variant,
        "source_available": 1,
        "at_risk": 1,
        "baseline_observed": 0,
        "post_observed": 0,
        "outcome_observed": 0,
        "baseline_value": math.nan,
        "post_value": math.nan,
        "change_value": math.nan,
        "event": math.nan,
        "event_offset_minutes": math.nan,
        "measurement_concept": "",
        "analysis_status": POST_HOC_STATUS,
        "causal_claim_permitted": False,
    }


def _lab_records(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    spec: LabSpec,
    *,
    lag_window: str,
) -> pd.DataFrame:
    lower, upper, lag_role = WINDOW_BY_NAME[lag_window]
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        dataset, stay_id = str(anchor["dataset"]), anchor["stay_id"]
        times, values = lookup.get(dataset, stay_id, spec.concept)
        plausible = np.isfinite(values) & (values >= spec.plausible_low) & (
            values <= spec.plausible_high
        )
        times, values = times[plausible], values[plausible]
        anchor_time = float(anchor["anchor_offset_minutes"])
        prior_right = int(np.searchsorted(times, anchor_time, side="left"))
        prior_left = int(
            np.searchsorted(
                times,
                anchor_time - LACTATE_LOOKBACK_MINUTES,
                side="left",
            )
        )
        baseline_observed = prior_right > prior_left
        baseline = float(values[prior_right - 1]) if baseline_observed else math.nan
        limit = _followup_limit(anchor)
        left = int(np.searchsorted(times, anchor_time + lower, side="right"))
        right = int(
            np.searchsorted(
                times, min(anchor_time + upper, limit), side="right"
            )
        )
        post_observed = right > left
        first_post = float(values[left]) if post_observed else math.nan
        first_time = float(times[left]) if post_observed else math.nan
        if post_observed:
            local = values[left:right]
            worst_index = int(np.argmax(local) if spec.direction == "high" else np.argmin(local))
            worst_post = float(local[worst_index])
            worst_time = float(times[left + worst_index])
        else:
            worst_post, worst_time = math.nan, math.nan
        for variant, post, post_time in (
            ("first_next", first_post, first_time),
            ("worst_in_window", worst_post, worst_time),
        ):
            row = _base_record(
                anchor,
                endpoint=spec.endpoint,
                family=spec.family,
                lag_window=lag_window,
                lag_role=lag_role,
                outcome_variant=variant,
            )
            source_available = int(
                dataset in lookup.datasets_by_concept.get(spec.concept, set())
            )
            event_flag = (
                bool(spec.event_rule(baseline, post))
                if source_available and baseline_observed and post_observed
                else False
            )
            binary_observed = bool(
                source_available
                and baseline_observed
                and post_observed
                and (
                    variant == "first_next"
                    or event_flag
                    or limit >= anchor_time + upper
                )
            )
            row.update(
                {
                    "source_available": source_available,
                    "at_risk": source_available,
                    "baseline_observed": int(baseline_observed),
                    "post_observed": int(post_observed),
                    "outcome_observed": int(binary_observed),
                    "baseline_value": baseline,
                    "post_value": post,
                    "event_offset_minutes": post_time,
                    "measurement_concept": spec.concept,
                    "change_scale": spec.change_scale,
                }
            )
            if source_available and baseline_observed and post_observed:
                row["change_value"] = _change_value(spec, baseline, post)
                row["event"] = int(event_flag) if binary_observed else math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _troponin_records(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    *,
    lag_window: str,
) -> pd.DataFrame:
    lower, upper, lag_role = WINDOW_BY_NAME[lag_window]
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        anchor_time = float(anchor["anchor_offset_minutes"])
        limit = _followup_limit(anchor)
        dataset = str(anchor["dataset"])
        source_available = int(
            dataset in lookup.datasets_by_concept.get("troponin_t", set())
            or dataset in lookup.datasets_by_concept.get("troponin_i", set())
        )
        baseline_assays: set[str] = set()
        assay_pairs: dict[str, tuple[float, float, float, float, float]] = {}
        for assay in ("troponin_t", "troponin_i"):
            times, values = lookup.get(str(anchor["dataset"]), anchor["stay_id"], assay)
            plausible = np.isfinite(values) & (values > 0) & (values <= 1_000_000)
            times, values = times[plausible], values[plausible]
            prior_right = int(np.searchsorted(times, anchor_time, side="left"))
            prior_left = int(
                np.searchsorted(
                    times,
                    anchor_time - LACTATE_LOOKBACK_MINUTES,
                    side="left",
                )
            )
            if prior_right <= prior_left:
                continue
            baseline_assays.add(assay)
            baseline = float(values[prior_right - 1])
            left = int(np.searchsorted(times, anchor_time + lower, side="right"))
            right = int(
                np.searchsorted(times, min(anchor_time + upper, limit), side="right")
            )
            if right <= left:
                continue
            first = float(values[left])
            local = values[left:right]
            worst_index = int(np.argmax(local))
            worst = float(local[worst_index])
            assay_pairs[assay] = (
                baseline,
                first,
                float(times[left]),
                worst,
                float(times[left + worst_index]),
            )
        for variant in ("first_next", "worst_in_window"):
            row = _base_record(
                anchor,
                endpoint="troponin_relative_rise",
                family="laboratory_organ_injury",
                lag_window=lag_window,
                lag_role=lag_role,
                outcome_variant=variant,
            )
            row["change_scale"] = "assay_matched_relative_ratio_minus_one"
            row["source_available"] = source_available
            row["at_risk"] = source_available
            row["baseline_observed"] = int(bool(baseline_assays))
            row["post_observed"] = int(bool(assay_pairs))
            event_flag = False
            if assay_pairs:
                candidates = []
                for assay, (
                    baseline,
                    first,
                    first_time,
                    worst,
                    worst_time,
                ) in assay_pairs.items():
                    post = first if variant == "first_next" else worst
                    post_time = first_time if variant == "first_next" else worst_time
                    candidates.append(
                        (post / baseline, assay, baseline, post, post_time)
                    )
                ratio, assay, baseline, post, post_time = (
                    min(candidates, key=lambda value: value[4])
                    if variant == "first_next"
                    else max(candidates, key=lambda value: value[0])
                )
                event_flag = ratio >= 1.5
                binary_observed = bool(
                    source_available
                    and (
                        variant == "first_next"
                        or event_flag
                        or limit >= anchor_time + upper
                    )
                )
                row["outcome_observed"] = int(binary_observed)
                row.update(
                    {
                        "baseline_value": baseline,
                        "post_value": post,
                        "change_value": ratio - 1.0,
                        "event": int(event_flag) if binary_observed else math.nan,
                        "event_offset_minutes": post_time,
                        "measurement_concept": assay,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _source_flag(anchor: dict[str, Any], column: str, lookup: EventLookup, concept: str) -> int:
    if column in anchor and pd.notna(anchor[column]):
        return int(pd.to_numeric(anchor[column], errors="coerce") > 0)
    return int(str(anchor["dataset"]) in lookup.datasets_by_concept.get(concept, set()))


def _event_records(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    spec: dict[str, str],
    *,
    lag_window: str,
) -> pd.DataFrame:
    lower, upper, lag_role = WINDOW_BY_NAME[lag_window]
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        endpoint, concept = spec["endpoint"], spec["concept"]
        row = _base_record(
            anchor,
            endpoint=endpoint,
            family=spec["family"],
            lag_window=lag_window,
            lag_role=lag_role,
            outcome_variant="incident_event",
        )
        source = _source_flag(anchor, spec["source_column"], lookup, concept)
        if spec.get("requires_concept_presence") == "1":
            source = int(
                source
                and str(anchor["dataset"])
                in lookup.datasets_by_concept.get(concept, set())
            )
        row["source_available"] = source
        row["baseline_observed"] = source
        row["measurement_concept"] = concept
        if not source:
            row["at_risk"] = 0
            rows.append(row)
            continue
        anchor_time = float(anchor["anchor_offset_minutes"])
        times, _ = lookup.get(str(anchor["dataset"]), anchor["stay_id"], concept)
        start = anchor_time + lower
        end = anchor_time + upper
        prior_count = int(np.searchsorted(times, start, side="right"))
        baseline_flag_name = spec.get("baseline_flag", "")
        baseline_flag = (
            pd.to_numeric(anchor.get(baseline_flag_name), errors="coerce")
            if baseline_flag_name
            else 0
        )
        baseline_event = bool(pd.notna(baseline_flag) and baseline_flag > 0)
        at_risk = prior_count == 0 and not baseline_event
        row["at_risk"] = int(at_risk)
        if not at_risk:
            rows.append(row)
            continue
        left = int(np.searchsorted(times, start, side="right"))
        right = int(np.searchsorted(times, end, side="right"))
        event_time = float(times[left]) if right > left else math.nan
        if endpoint == "death":
            # In-hospital death time/discharge status makes the binary endpoint
            # observed even when ICU follow-up ends before the window boundary.
            observed = True
            event = bool(right > left)
        else:
            followup = _followup_limit(anchor)
            event = bool(right > left and event_time <= followup)
            observed = event or followup >= end
        row.update(
            {
                "post_observed": int(observed),
                "outcome_observed": int(observed),
                "event": int(event) if observed else math.nan,
                "event_offset_minutes": event_time if event else math.nan,
                "change_scale": "incident_binary_event",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _vis_records(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    *,
    lag_window: str,
) -> pd.DataFrame:
    lower, upper, lag_role = WINDOW_BY_NAME[lag_window]
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        row = _base_record(
            anchor,
            endpoint="vis_escalation",
            family="hemodynamic_support",
            lag_window=lag_window,
            lag_role=lag_role,
            outcome_variant="worst_in_window",
        )
        source = _source_flag(anchor, "vis_source_available", lookup, "vis")
        row["source_available"] = source
        row["measurement_concept"] = "vis"
        if not source:
            row["at_risk"] = 0
            rows.append(row)
            continue
        anchor_time = float(anchor["anchor_offset_minutes"])
        times, values = lookup.get(str(anchor["dataset"]), anchor["stay_id"], "vis")
        plausible = np.isfinite(values) & (values >= 0)
        times, values = times[plausible], values[plausible]
        prior_right = int(np.searchsorted(times, anchor_time, side="left"))
        baseline = float(np.max(values[:prior_right])) if prior_right else 0.0
        start = anchor_time + lower
        end = anchor_time + upper
        state_right = int(np.searchsorted(times, start, side="right"))
        # A non-overlapping lag is an onset window, not a persistence window:
        # if VIS already exceeded its strict pre-anchor maximum before this
        # window starts, this anchor is no longer at risk for a new escalation.
        interim = values[prior_right:state_right]
        prior_escalation = bool(len(interim) and np.max(interim) > baseline)
        row["at_risk"] = int(not prior_escalation)
        if prior_escalation:
            rows.append(row)
            continue
        right = int(np.searchsorted(times, end, side="right"))
        candidates = [baseline]
        if right > state_right:
            candidates.extend(values[state_right:right].tolist())
        post_max = float(max(candidates))
        escalation = post_max > baseline
        followup = _followup_limit(anchor)
        observed = escalation or followup >= end
        row.update(
            {
                "baseline_observed": 1,
                "post_observed": int(observed),
                "outcome_observed": int(observed),
                "baseline_value": baseline,
                "post_value": post_max if observed else math.nan,
                "change_value": post_max - baseline if observed else math.nan,
                "event": int(escalation) if observed else math.nan,
                "change_scale": "vis_absolute_increase",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _urine_records(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    *,
    lag_window: str,
) -> pd.DataFrame:
    lower, upper, lag_role = WINDOW_BY_NAME[lag_window]
    duration = upper - lower
    expected_bins = max(int(math.ceil(duration / 120.0)), 1)
    rows: list[dict[str, Any]] = []
    for anchor in anchors.to_dict("records"):
        source = _source_flag(
            anchor, "urine_output_source_available", lookup, "urine_output"
        )
        weight = pd.to_numeric(anchor.get("baseline_weight_outcome"), errors="coerce")
        if pd.isna(weight) or weight <= 0:
            weight = pd.to_numeric(anchor.get("admission_weight_kg"), errors="coerce")
        anchor_time = float(anchor["anchor_offset_minutes"])
        start, end = anchor_time + lower, anchor_time + upper
        times, values = lookup.get(
            str(anchor["dataset"]), anchor["stay_id"], "urine_output"
        )
        plausible = np.isfinite(values) & (values >= 0)
        times, values = times[plausible], values[plausible]

        baseline_duration = min(anchor_time, 240.0)
        baseline_start = anchor_time - baseline_duration
        baseline_expected_bins = max(int(math.ceil(baseline_duration / 120.0)), 1)
        baseline_left = int(np.searchsorted(times, baseline_start, side="right"))
        baseline_right = int(np.searchsorted(times, anchor_time, side="left"))
        baseline_times = times[baseline_left:baseline_right]
        baseline_values = values[baseline_left:baseline_right]
        baseline_bins = (
            np.floor(
                (baseline_times - baseline_start - np.finfo(float).eps) / 120.0
            )
            if len(baseline_times)
            else np.array([], dtype=float)
        )
        baseline_covered = int(
            len(
                np.unique(
                    np.clip(baseline_bins, 0, baseline_expected_bins - 1)
                )
            )
        )
        baseline_coverage = baseline_covered / baseline_expected_bins
        baseline_rate = (
            float(baseline_values.sum() / (baseline_covered * 2.0))
            if baseline_covered
            else math.nan
        )
        strict_baseline_observed = bool(
            baseline_duration >= 120.0
            and baseline_covered > 0
            and baseline_coverage >= 0.5
            and np.isfinite(baseline_rate)
            and baseline_rate > 0
        )

        left = int(np.searchsorted(times, start, side="right"))
        right = int(np.searchsorted(times, end, side="right"))
        local_times, local_values = times[left:right], values[left:right]
        bins = (
            np.floor((local_times - start - np.finfo(float).eps) / 120.0)
            if len(local_times)
            else np.array([], dtype=float)
        )
        covered_bins = int(len(np.unique(np.clip(bins, 0, expected_bins - 1))))
        coverage = covered_bins / expected_bins
        followup = _followup_limit(anchor)
        post_rate_observed = bool(
            source
            and duration >= 240.0
            and followup >= end
            and covered_bins > 0
            and coverage >= 0.5
        )
        rate_ml_h = math.nan
        if post_rate_observed:
            observed_hours = covered_bins * 2.0
            rate_ml_h = float(local_values.sum() / observed_hours)

        kdigo = _base_record(
            anchor,
            endpoint="oliguria_kdigo_proxy",
            family="urine_output",
            lag_window=lag_window,
            lag_role=lag_role,
            outcome_variant="absolute_rate",
        )
        kdigo.update(
            {
                "source_available": source,
                "at_risk": int(source and pd.notna(weight) and weight > 0),
                "measurement_concept": "urine_output",
                "baseline_value": weight,
                "baseline_observed": int(pd.notna(weight) and weight > 0),
                "post_observed": int(post_rate_observed),
                "outcome_observed": int(
                    post_rate_observed and pd.notna(weight) and weight > 0
                ),
                "change_scale": "negative_urine_rate_ml_kg_h",
                "urine_coverage_fraction": coverage,
                "urine_covered_2h_bins": covered_bins,
            }
        )
        if kdigo["outcome_observed"]:
            rate_ml_kg_h = rate_ml_h / float(weight)
            kdigo.update(
                {
                    "post_value": rate_ml_kg_h,
                    "change_value": -rate_ml_kg_h,
                    "event": int(rate_ml_kg_h < 0.5),
                }
            )
        rows.append(kdigo)

        decline = _base_record(
            anchor,
            endpoint="urine_output_decline_proxy",
            family="urine_output",
            lag_window=lag_window,
            lag_role=lag_role,
            outcome_variant="relative_decline",
        )
        decline.update(
            {
                "source_available": source,
                "at_risk": source,
                "measurement_concept": "urine_output",
                "baseline_value": baseline_rate,
                "baseline_observed": int(strict_baseline_observed),
                "post_observed": int(post_rate_observed),
                "outcome_observed": int(
                    strict_baseline_observed and post_rate_observed
                ),
                "change_scale": "urine_rate_decline_fraction",
                "urine_baseline_coverage_fraction": baseline_coverage,
                "urine_coverage_fraction": coverage,
                "urine_covered_2h_bins": covered_bins,
            }
        )
        if decline["outcome_observed"]:
            decline_fraction = 1.0 - rate_ml_h / baseline_rate
            decline.update(
                {
                    "post_value": rate_ml_h,
                    "change_value": decline_fraction,
                    "event": int(decline_fraction >= 0.5),
                }
            )
        rows.append(decline)
    return pd.DataFrame(rows)


PRIMARY_VARIANT_BY_ENDPOINT: dict[str, str] = {
    **{spec.endpoint: spec.primary_variant for spec in LAB_SPECS},
    "troponin_relative_rise": "worst_in_window",
    **{spec["endpoint"]: "incident_event" for spec in EVENT_SPECS},
    "vis_escalation": "worst_in_window",
    "oliguria_kdigo_proxy": "absolute_rate",
    "urine_output_decline_proxy": "relative_decline",
}

# Ordered because later composites can depend on earlier conservative-any
# composites.  This mirrors the fixed-window protocol before adding one broad,
# explicitly exploratory clinical composite.
COMPOSITE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "endpoint": "hepatic_injury_extended",
        "family": "hepatic_composite",
        "components": (
            "bilirubin_worsening",
            "ast_injury",
            "alt_injury",
            "inr_injury",
        ),
    },
    {
        "endpoint": "organ_lab_worsening",
        "family": "organ_injury_composite",
        "components": (
            "aki_creatinine",
            "hepatic_injury_extended",
            "platelet_injury",
        ),
    },
    {
        "endpoint": "early_decompensation",
        "family": "early_decompensation_composite",
        "components": (
            "lactate_rise",
            "organ_lab_worsening",
            "mcs_initiation",
        ),
    },
    {
        "endpoint": "early_decompensation_plus_vis",
        "family": "early_decompensation_composite",
        "components": ("early_decompensation", "vis_escalation"),
    },
    {
        "endpoint": "any_clinical_decompensation",
        "family": "all_decompensation_composite",
        "components": (
            "early_decompensation_plus_vis",
            "troponin_relative_rise",
            "oliguria_kdigo_proxy",
            "pressor_initiation",
            "rrt_initiation",
            "invasive_ventilation_initiation",
            "death",
        ),
    },
)
for _composite_spec in COMPOSITE_SPECS:
    PRIMARY_VARIANT_BY_ENDPOINT[_composite_spec["endpoint"]] = "conservative_any"

PRIMARY_DEFINITION = ("absolute_jump_ge4", "15_minute_median_bins")
LOCKED_EXTERNAL_VALIDATION_WINDOW = "4_to_12h"
LOCKED_EXTERNAL_VALIDATION_ENDPOINT = "invasive_ventilation_initiation"
LOCKED_ENDPOINT_TIER: dict[str, str] = {
    "invasive_ventilation_initiation": "transferred_primary",
    "intubation_initiation": "mimic_respiratory_specificity",
    "pressor_initiation": "clinical_escalation_secondary",
    "mcs_initiation": "clinical_escalation_secondary",
    "rrt_initiation": "clinical_escalation_secondary",
    "death": "clinical_escalation_secondary",
    "lactate_rise": "organ_injury_secondary",
    "aki_creatinine": "organ_injury_secondary",
    "bilirubin_worsening": "organ_injury_secondary",
    "ast_injury": "organ_injury_secondary",
    "alt_injury": "organ_injury_secondary",
    "inr_injury": "organ_injury_secondary",
    "platelet_injury": "organ_injury_secondary",
    "troponin_relative_rise": "organ_injury_secondary",
    "vis_escalation": "organ_injury_secondary",
    "oliguria_kdigo_proxy": "organ_injury_secondary",
    "urine_output_decline_proxy": "organ_injury_secondary",
    "hepatic_injury_extended": "exploratory_composite",
    "organ_lab_worsening": "exploratory_composite",
    "early_decompensation": "exploratory_composite",
    "early_decompensation_plus_vis": "exploratory_composite",
    "any_clinical_decompensation": "exploratory_composite",
}
SENSITIVITY_WINDOWS = tuple(
    label
    for label, *_ in LAG_WINDOWS
    if label
    in {
        window
        for windows in FOCUSED_WINDOWS_BY_FAMILY.values()
        for window in windows
    }
)
CLAIM_EVENTS_PER_PARAMETER = 10.0


def _component_record_frames(
    anchors: pd.DataFrame,
    lookup: EventLookup,
    *,
    lag_window: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for spec in LAB_SPECS:
        frames[spec.endpoint] = _lab_records(
            anchors, lookup, spec, lag_window=lag_window
        )
    frames["troponin_relative_rise"] = _troponin_records(
        anchors, lookup, lag_window=lag_window
    )
    for spec in EVENT_SPECS:
        frames[spec["endpoint"]] = _event_records(
            anchors, lookup, spec, lag_window=lag_window
        )
    frames["vis_escalation"] = _vis_records(
        anchors, lookup, lag_window=lag_window
    )
    urine = _urine_records(anchors, lookup, lag_window=lag_window)
    for endpoint, group in urine.groupby("endpoint", sort=False):
        frames[str(endpoint)] = group.reset_index(drop=True)
    return frames


def _select_primary_variant(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    variant = PRIMARY_VARIANT_BY_ENDPOINT[endpoint]
    return frame.loc[frame["outcome_variant"].eq(variant)].reset_index(drop=True)


def _conservative_any_records(
    component_frames: dict[str, pd.DataFrame],
    spec: dict[str, Any],
) -> pd.DataFrame:
    """Combine endpoints without treating an unobserved component as negative."""
    components = [
        component
        for component in spec["components"]
        if component in component_frames
    ]
    if not components:
        return pd.DataFrame()
    keys = ["dataset", "stay_id"]
    base = component_frames[components[0]].drop_duplicates(keys).set_index(keys)
    index = base.index
    available_count = np.zeros(len(index), dtype=int)
    eligible_count = np.zeros(len(index), dtype=int)
    observed_count = np.zeros(len(index), dtype=int)
    event_count = np.zeros(len(index), dtype=int)
    all_available_complete = np.ones(len(index), dtype=bool)
    for component in components:
        local = component_frames[component].drop_duplicates(keys).set_index(keys)
        local = local.reindex(index)
        source = pd.to_numeric(local["source_available"], errors="coerce").fillna(0).eq(1)
        at_risk = pd.to_numeric(local["at_risk"], errors="coerce").fillna(0).eq(1)
        observed = pd.to_numeric(local["outcome_observed"], errors="coerce").fillna(0).eq(1)
        event = pd.to_numeric(local["event"], errors="coerce").fillna(0).eq(1)
        available_count += source.to_numpy(int)
        eligible_count += (source & at_risk).to_numpy(int)
        observed_count += (source & at_risk & observed).to_numpy(int)
        event_count += (source & at_risk & observed & event).to_numpy(int)
        all_available_complete &= (~source | (at_risk & observed)).to_numpy(bool)
    any_event = event_count > 0
    source_available = available_count > 0
    outcome_observed = source_available & (any_event | all_available_complete)
    result = base.reset_index().copy()
    result["endpoint"] = str(spec["endpoint"])
    result["endpoint_family"] = str(spec["family"])
    result["outcome_variant"] = "conservative_any"
    result["measurement_concept"] = "+".join(components)
    result["change_scale"] = "conservative_any_binary"
    result["source_available"] = source_available.astype(int)
    result["at_risk"] = (eligible_count > 0).astype(int)
    result["baseline_observed"] = source_available.astype(int)
    result["post_observed"] = outcome_observed.astype(int)
    result["outcome_observed"] = outcome_observed.astype(int)
    result["event"] = np.where(outcome_observed, any_event.astype(int), np.nan)
    result["baseline_value"] = np.nan
    result["post_value"] = np.nan
    result["change_value"] = np.nan
    result["event_offset_minutes"] = np.nan
    result["component_count_available"] = available_count
    result["component_count_eligible"] = eligible_count
    result["component_count_observed"] = observed_count
    result["component_count_event"] = event_count
    return result.reset_index(drop=True)


def _add_composite_frames(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    primary_frames = {
        endpoint: _select_primary_variant(frame, endpoint)
        for endpoint, frame in frames.items()
    }
    for spec in COMPOSITE_SPECS:
        composite = _conservative_any_records(primary_frames, spec)
        if composite.empty:
            continue
        endpoint = str(spec["endpoint"])
        frames[endpoint] = composite
        primary_frames[endpoint] = composite
    return frames


def _empty_adjusted(reason: str, status: str = "not_requested") -> dict[str, Any]:
    return {
        "adjusted_status": status,
        "adjusted_risk_ratio": math.nan,
        "adjusted_ci95_low": math.nan,
        "adjusted_ci95_high": math.nan,
        "adjusted_p_value": math.nan,
        "adjusted_log_rr_se": math.nan,
        "adjusted_parameter_count": 0,
        "adjusted_events_per_parameter": math.nan,
        "adjusted_claim_ready_epv10": False,
        "adjusted_covariates": "",
        "adjusted_reason": reason,
    }


def _model_groups(frame: pd.DataFrame) -> pd.Series:
    if "person_id" in frame and frame["person_id"].notna().any():
        person = frame["person_id"].astype("string")
        stay = frame["stay_id"].astype("string")
        identity = person.where(
            person.notna() & person.str.strip().ne(""), "stay:" + stay
        )
    else:
        identity = "stay:" + frame["stay_id"].astype("string")
    return frame["dataset"].astype(str) + ":" + identity.astype(str)


def _design_matrix(
    frame: pd.DataFrame,
    *,
    information_count: int,
    minimum_epp: float = MIN_EVENTS_PER_PARAMETER,
) -> tuple[pd.DataFrame, list[str], Any]:
    import statsmodels.api as sm

    design = pd.DataFrame(index=frame.index)
    design["episode_exposed"] = pd.to_numeric(
        frame["episode_exposed"], errors="coerce"
    )
    # This priority is frozen and outcome-p-value independent.  The number
    # retained adapts only to the available information count, preventing a
    # needlessly over-parameterized model from suppressing an estimable signal.
    candidates = (
        "baseline_value",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
        "age",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
    )
    allowed_parameters = max(int(math.floor(information_count / minimum_epp)), 2)
    allowed_covariates = max(allowed_parameters - 2, 0)
    retained: list[str] = []
    for column in candidates:
        if len(retained) >= allowed_covariates or column not in frame:
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
    return sm.add_constant(design.astype(float), has_constant="add"), retained, sm


def _adjusted_modified_poisson_all_endpoints(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    result = _empty_adjusted("", "not_estimable")
    local = frame.copy()
    local["event"] = pd.to_numeric(local["event"], errors="coerce")
    local = local.loc[
        local["event"].isin([0, 1])
        & pd.to_numeric(local["episode_exposed"], errors="coerce").isin([0, 1])
    ].copy()
    events = int(local["event"].sum())
    non_events = int(len(local) - events)
    if len(local) < MIN_ROWS or min(events, non_events) < MIN_EVENTS:
        return _empty_adjusted(
            "insufficient rows/events/non-events", "underpowered"
        )
    try:
        design, retained, sm = _design_matrix(
            local, information_count=min(events, non_events)
        )
    except Exception as exc:  # pragma: no cover
        return _empty_adjusted(f"statsmodels/design unavailable: {exc}")
    parameters = int(design.shape[1])
    epp = min(events, non_events) / max(parameters, 1)
    result.update(
        {
            "adjusted_parameter_count": parameters,
            "adjusted_events_per_parameter": epp,
            "adjusted_claim_ready_epv10": bool(
                epp >= CLAIM_EVENTS_PER_PARAMETER
            ),
            "adjusted_covariates": ",".join(retained),
        }
    )
    if epp < MIN_EVENTS_PER_PARAMETER:
        result["adjusted_status"] = "underpowered_low_information"
        result["adjusted_reason"] = "events_per_parameter_below_5_fail_closed"
        return result
    try:
        model = sm.GLM(
            local["event"].astype(int),
            design,
            family=sm.families.Poisson(),
        )
        groups = _model_groups(local)
        fit = (
            model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            if groups.nunique() > 1
            else model.fit(cov_type="HC0")
        )
        coefficient = float(fit.params["episode_exposed"])
        low, high = fit.conf_int().loc["episode_exposed"]
        values = np.exp([coefficient, float(low), float(high)])
        if not np.isfinite(values).all():
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


def _bootstrap_median_difference(
    exposed: np.ndarray,
    unexposed: np.ndarray,
    *,
    repetitions: int,
    seed_key: str,
) -> tuple[float, float]:
    if repetitions < 50 or len(exposed) < 5 or len(unexposed) < 5:
        return math.nan, math.nan
    seed = BOOTSTRAP_SEED + int(zlib.crc32(seed_key.encode("utf-8")))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        estimates[index] = float(
            np.median(rng.choice(exposed, size=len(exposed), replace=True))
            - np.median(rng.choice(unexposed, size=len(unexposed), replace=True))
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _summarize_record_frame(
    frame: pd.DataFrame,
    *,
    primary_variant: bool,
    primary_definition: bool,
    bootstrap_repetitions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    endpoint = str(frame["endpoint"].iloc[0])
    family = str(frame["endpoint_family"].iloc[0])
    focused_windows = FOCUSED_WINDOWS_BY_FAMILY[family]
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        eligible = dataset_frame.loc[
            pd.to_numeric(dataset_frame["source_available"], errors="coerce").eq(1)
            & pd.to_numeric(dataset_frame["at_risk"], errors="coerce").eq(1)
            & pd.to_numeric(dataset_frame["outcome_observed"], errors="coerce").eq(1)
            & pd.to_numeric(dataset_frame["event"], errors="coerce").isin([0, 1])
        ].copy()
        eligible["event"] = pd.to_numeric(eligible["event"], errors="coerce").astype(int)
        exposed = eligible.loc[eligible["episode_exposed"].eq(1)]
        unexposed = eligible.loc[eligible["episode_exposed"].eq(0)]
        a, n1 = int(exposed["event"].sum()), int(len(exposed))
        c, n0 = int(unexposed["event"].sum()), int(len(unexposed))
        total_events = a + c
        total_non_events = n1 + n0 - total_events
        status = (
            "unavailable"
            if int(pd.to_numeric(dataset_frame["source_available"], errors="coerce").sum()) == 0
            else "estimated"
            if n1 >= 5 and n0 >= 5 and min(total_events, total_non_events) >= 2
            else "underpowered"
        )
        risk_exposed = a / n1 if n1 else math.nan
        risk_unexposed = c / n0 if n0 else math.nan
        rr = _risk_ratio(a, n1, c, n0)
        if n1 and n0:
            rd = risk_exposed - risk_unexposed
            rd_se = math.sqrt(
                risk_exposed * (1.0 - risk_exposed) / n1
                + risk_unexposed * (1.0 - risk_unexposed) / n0
            )
            rd_low, rd_high = max(rd - 1.96 * rd_se, -1.0), min(
                rd + 1.96 * rd_se, 1.0
            )
        else:
            rd = rd_low = rd_high = math.nan
        exposed_change = pd.to_numeric(
            exposed["change_value"], errors="coerce"
        ).dropna().to_numpy(float)
        unexposed_change = pd.to_numeric(
            unexposed["change_value"], errors="coerce"
        ).dropna().to_numpy(float)
        median_exposed = (
            float(np.median(exposed_change)) if len(exposed_change) else math.nan
        )
        median_unexposed = (
            float(np.median(unexposed_change))
            if len(unexposed_change)
            else math.nan
        )
        median_difference = (
            median_exposed - median_unexposed
            if len(exposed_change) and len(unexposed_change)
            else math.nan
        )
        focused = bool(
            primary_definition
            and primary_variant
            and str(frame["lag_window"].iloc[0]) in focused_windows
        )
        bootstrap_low, bootstrap_high = (
            _bootstrap_median_difference(
                exposed_change,
                unexposed_change,
                repetitions=bootstrap_repetitions,
                seed_key=(
                    f"{dataset}|{endpoint}|{frame['lag_window'].iloc[0]}|"
                    f"{frame['outcome_variant'].iloc[0]}"
                ),
            )
            if focused
            else (math.nan, math.nan)
        )
        adjusted = (
            _adjusted_modified_poisson_all_endpoints(eligible)
            if primary_definition and primary_variant
            else _empty_adjusted(
                "adjustment reserved for the primary episode definition and endpoint variant"
            )
        )
        rows.append(
            {
                "dataset": str(dataset),
                "endpoint": endpoint,
                "endpoint_family": family,
                "episode_definition": str(frame["episode_definition"].iloc[0]),
                "signal_resolution": str(frame["signal_resolution"].iloc[0]),
                "definition_role": str(frame["definition_role"].iloc[0]),
                "lag_window": str(frame["lag_window"].iloc[0]),
                "lag_role": str(frame["lag_role"].iloc[0]),
                "outcome_variant": str(frame["outcome_variant"].iloc[0]),
                "primary_endpoint_variant": bool(primary_variant),
                "definition_focused_window": bool(
                    primary_variant
                    and str(frame["lag_window"].iloc[0]) in focused_windows
                ),
                "focused_endpoint_window": focused,
                "status": status,
                "n_anchor_rows": int(len(dataset_frame)),
                "n_source_available": int(
                    pd.to_numeric(
                        dataset_frame["source_available"], errors="coerce"
                    ).fillna(0).sum()
                ),
                "n_at_risk": int(
                    (
                        pd.to_numeric(dataset_frame["source_available"], errors="coerce").eq(1)
                        & pd.to_numeric(dataset_frame["at_risk"], errors="coerce").eq(1)
                    ).sum()
                ),
                "n_outcome_observed": int(len(eligible)),
                "n_exposed": n1,
                "n_unexposed": n0,
                "events_exposed": a,
                "events_unexposed": c,
                "risk_exposed": risk_exposed,
                "risk_unexposed": risk_unexposed,
                **rr,
                "risk_difference": rd,
                "risk_difference_ci95_low": rd_low,
                "risk_difference_ci95_high": rd_high,
                "fisher_two_sided_p": (
                    _fisher_p(a, n1 - a, c, n0 - c)
                    if n1 and n0
                    else math.nan
                ),
                "n_continuous_exposed": int(len(exposed_change)),
                "n_continuous_unexposed": int(len(unexposed_change)),
                "median_change_exposed": median_exposed,
                "median_change_unexposed": median_unexposed,
                "median_change_difference": median_difference,
                "median_change_difference_ci95_low": bootstrap_low,
                "median_change_difference_ci95_high": bootstrap_high,
                "mann_whitney_two_sided_p": _mann_whitney_p(
                    exposed_change, unexposed_change
                ),
                **adjusted,
                "analysis_status": POST_HOC_STATUS,
                "control_anchor_definition": (
                    "no_episode_stay_transition_quantile_matched_to_"
                    "exposed_anchor_time_distribution"
                ),
                "interpretation_scope": (
                    "observational_time_aligned_episode_contrast_not_causal"
                ),
            }
        )
    return rows


def _apply_effect_multiplicity(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return effects
    result = effects.copy()
    for p_column, q_column in (
        ("fisher_two_sided_p", "global_binary_q_value"),
        ("mann_whitney_two_sided_p", "global_continuous_q_value"),
        ("adjusted_p_value", "global_adjusted_q_value"),
    ):
        result[q_column] = _bh_adjust(result[p_column])
    for column in (
        "definition_endpoint_q_value",
        "definition_endpoint_continuous_q_value",
        "endpoint_focused_q_value",
        "endpoint_focused_continuous_q_value",
        "endpoint_focused_adjusted_q_value",
        "family_focused_q_value",
        "family_focused_adjusted_q_value",
        "cross_endpoint_focused_q_value",
        "cross_endpoint_focused_continuous_q_value",
        "cross_endpoint_focused_adjusted_q_value",
    ):
        result[column] = np.nan

    definition_focus = result["definition_focused_window"].eq(True)  # noqa: E712
    grouping = [
        "dataset",
        "episode_definition",
        "signal_resolution",
        "endpoint",
    ]
    for _, indexes in result.loc[definition_focus].groupby(grouping).groups.items():
        result.loc[indexes, "definition_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[indexes, "definition_endpoint_continuous_q_value"] = _bh_adjust(
            result.loc[indexes, "mann_whitney_two_sided_p"]
        )

    primary_focus = result["focused_endpoint_window"].eq(True)  # noqa: E712
    for _, indexes in result.loc[primary_focus].groupby(
        ["dataset", "endpoint"]
    ).groups.items():
        result.loc[indexes, "endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[indexes, "endpoint_focused_continuous_q_value"] = _bh_adjust(
            result.loc[indexes, "mann_whitney_two_sided_p"]
        )
        result.loc[indexes, "endpoint_focused_adjusted_q_value"] = _bh_adjust(
            result.loc[indexes, "adjusted_p_value"]
        )
    for _, indexes in result.loc[primary_focus].groupby(
        ["dataset", "endpoint_family"]
    ).groups.items():
        result.loc[indexes, "family_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[indexes, "family_focused_adjusted_q_value"] = _bh_adjust(
            result.loc[indexes, "adjusted_p_value"]
        )
    for _, indexes in result.loc[primary_focus].groupby("dataset").groups.items():
        result.loc[indexes, "cross_endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
        result.loc[
            indexes, "cross_endpoint_focused_continuous_q_value"
        ] = _bh_adjust(result.loc[indexes, "mann_whitney_two_sided_p"])
        result.loc[indexes, "cross_endpoint_focused_adjusted_q_value"] = _bh_adjust(
            result.loc[indexes, "adjusted_p_value"]
        )
    return result


def _observation_process_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    endpoint = str(frame["endpoint"].iloc[0])
    family = str(frame["endpoint_family"].iloc[0])
    targets: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("source_available", "source_available", ()),
        ("anchor_at_risk", "at_risk", ("source_available",)),
        (
            "strict_baseline_available",
            "baseline_observed",
            ("source_available", "at_risk"),
        ),
        (
            "post_endpoint_measured_or_followed",
            "post_observed",
            ("source_available", "at_risk", "baseline_observed"),
        ),
        (
            "complete_endpoint_classification",
            "outcome_observed",
            ("source_available", "at_risk", "baseline_observed"),
        ),
    )
    for dataset, dataset_frame in frame.groupby("dataset", sort=False):
        for target, observed_column, denominator_columns in targets:
            denominator = pd.Series(True, index=dataset_frame.index)
            for column in denominator_columns:
                denominator &= pd.to_numeric(
                    dataset_frame[column], errors="coerce"
                ).fillna(0).eq(1)
            risk = dataset_frame.loc[denominator].copy()
            observed = pd.to_numeric(
                risk[observed_column], errors="coerce"
            ).fillna(0).eq(1)
            exposed_mask = risk["episode_exposed"].eq(1)
            unexposed_mask = risk["episode_exposed"].eq(0)
            a, n1 = int(observed.loc[exposed_mask].sum()), int(exposed_mask.sum())
            c, n0 = int(observed.loc[unexposed_mask].sum()), int(unexposed_mask.sum())
            rows.append(
                {
                    "dataset": str(dataset),
                    "endpoint": endpoint,
                    "endpoint_family": family,
                    "lag_window": str(frame["lag_window"].iloc[0]),
                    "lag_role": str(frame["lag_role"].iloc[0]),
                    "observation_target": target,
                    "denominator_scope": (
                        "+".join(denominator_columns)
                        if denominator_columns
                        else "all_episode_or_control_anchors"
                    ),
                    "n_exposed_denominator": n1,
                    "n_unexposed_denominator": n0,
                    "observed_exposed": a,
                    "observed_unexposed": c,
                    "observation_rate_exposed": a / n1 if n1 else math.nan,
                    "observation_rate_unexposed": c / n0 if n0 else math.nan,
                    **_risk_ratio(a, n1, c, n0),
                    "fisher_two_sided_p": (
                        _fisher_p(a, n1 - a, c, n0 - c)
                        if n1 and n0
                        else math.nan
                    ),
                    "status": (
                        "estimated" if n1 >= 5 and n0 >= 5 else "underpowered"
                    ),
                    "analysis_status": POST_HOC_STATUS,
                    "interpretation": (
                        "measurement_or_followup_process_diagnostic_not_"
                        "biological_endpoint"
                    ),
                }
            )
    return rows


def _apply_observation_multiplicity(observation: pd.DataFrame) -> pd.DataFrame:
    if observation.empty:
        return observation
    result = observation.copy()
    result["global_q_value"] = _bh_adjust(result["fisher_two_sided_p"])
    result["endpoint_focused_q_value"] = np.nan
    focused = result.apply(
        lambda row: row["lag_window"]
        in FOCUSED_WINDOWS_BY_FAMILY.get(str(row["endpoint_family"]), ()),
        axis=1,
    ) & result["observation_target"].eq("complete_endpoint_classification")
    for _, indexes in result.loc[focused].groupby(
        ["dataset", "endpoint"]
    ).groups.items():
        result.loc[indexes, "endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
    result["cross_endpoint_focused_q_value"] = np.nan
    for _, indexes in result.loc[focused].groupby("dataset").groups.items():
        result.loc[indexes, "cross_endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "fisher_two_sided_p"]
        )
    return result


def _empty_weighted_row(
    frame: pd.DataFrame,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": str(frame["dataset"].iloc[0]),
        "endpoint": str(frame["endpoint"].iloc[0]),
        "endpoint_family": str(frame["endpoint_family"].iloc[0]),
        "lag_window": str(frame["lag_window"].iloc[0]),
        "status": status,
        "n_anchor_risk_set": 0,
        "n_complete_outcomes": 0,
        "events_complete_outcomes": 0,
        "selection_model_parameters": 0,
        "selection_events_per_parameter": math.nan,
        "outcome_model_parameters": 0,
        "outcome_events_per_parameter": math.nan,
        "outcome_claim_ready_epv10": False,
        "weighted_risk_exposed": math.nan,
        "weighted_risk_unexposed": math.nan,
        "selection_weighted_risk_ratio": math.nan,
        "selection_weighted_ci95_low": math.nan,
        "selection_weighted_ci95_high": math.nan,
        "selection_weighted_p_value": math.nan,
        "selection_weighted_log_rr_se": math.nan,
        "effective_sample_size": math.nan,
        "predicted_observation_probability_min": math.nan,
        "predicted_observation_probability_max": math.nan,
        "probability_clipped_low_count": 0,
        "probability_clipped_high_count": 0,
        "positivity_status": "not_assessed",
        "stabilized_weight_p01": math.nan,
        "stabilized_weight_p99": math.nan,
        "selection_covariates": "",
        "outcome_covariates": "",
        "weighted_inference_method": "cluster_robust_gee_poisson_log_link",
        "reason": reason,
        "analysis_status": POST_HOC_STATUS,
        "weighting_scope": (
            "inverse_probability_of_endpoint_observation_conditional_on_"
            "source_at_risk_and_strict_baseline"
        ),
        "claim_scope": "missing_outcome_and_censoring_sensitivity_not_causal_correction",
    }


def _selection_weighted_row(frame: pd.DataFrame) -> dict[str, Any]:
    base = _empty_weighted_row(frame, status="not_estimable", reason="")
    risk_set = frame.loc[
        pd.to_numeric(frame["source_available"], errors="coerce").eq(1)
        & pd.to_numeric(frame["at_risk"], errors="coerce").eq(1)
        & pd.to_numeric(frame["baseline_observed"], errors="coerce").eq(1)
        & pd.to_numeric(frame["episode_exposed"], errors="coerce").isin([0, 1])
    ].copy().reset_index(drop=True)
    risk_set["_observed"] = pd.to_numeric(
        risk_set["outcome_observed"], errors="coerce"
    ).fillna(0).eq(1).astype(int)
    base["n_anchor_risk_set"] = int(len(risk_set))
    if len(risk_set) < MIN_ROWS:
        base["status"] = "underpowered_risk_set"
        base["reason"] = "fewer than minimum anchor-risk-set rows"
        return base
    observed = int(risk_set["_observed"].sum())
    unobserved = int(len(risk_set) - observed)
    selection_needed = unobserved > 0
    probability = np.ones(len(risk_set), dtype=float)
    if selection_needed:
        if min(observed, unobserved) < MIN_EVENTS:
            base["status"] = "underpowered_selection_model"
            base["reason"] = "insufficient observed/unobserved endpoint classifications"
            return base
        try:
            selection_design, retained, sm = _design_matrix(
                risk_set, information_count=min(observed, unobserved)
            )
            selection_parameters = int(selection_design.shape[1])
            selection_epp = min(observed, unobserved) / max(
                selection_parameters, 1
            )
            base.update(
                {
                    "selection_model_parameters": selection_parameters,
                    "selection_events_per_parameter": selection_epp,
                    "selection_covariates": ",".join(retained),
                }
            )
            if selection_epp < MIN_EVENTS_PER_PARAMETER:
                base["status"] = "underpowered_selection_model"
                base["reason"] = "selection_events_per_parameter_below_5_fail_closed"
                return base
            selection_fit = sm.GLM(
                risk_set["_observed"],
                selection_design,
                family=sm.families.Binomial(),
            ).fit()
            probability = np.asarray(
                selection_fit.predict(selection_design), dtype=float
            )
        except Exception as exc:
            base["status"] = "non_estimable_selection_model"
            base["reason"] = str(exc)[:500]
            return base
        if not np.isfinite(probability).all():
            base["status"] = "non_estimable_selection_model"
            base["reason"] = "non-finite endpoint-observation probabilities"
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
    else:
        base["positivity_status"] = "complete_observation_no_selection_weights_needed"
    risk_set["_observation_probability"] = probability
    base["predicted_observation_probability_min"] = float(probability.min())
    base["predicted_observation_probability_max"] = float(probability.max())
    complete = risk_set.loc[risk_set["_observed"].eq(1)].copy()
    complete["_event"] = pd.to_numeric(complete["event"], errors="coerce")
    complete = (
        complete.loc[complete["_event"].isin([0, 1])]
        .copy()
        .reset_index(drop=True)
    )
    complete["_event"] = complete["_event"].astype(int)
    events = int(complete["_event"].sum())
    non_events = int(len(complete) - events)
    base["n_complete_outcomes"] = int(len(complete))
    base["events_complete_outcomes"] = events
    if len(complete) < MIN_ROWS or min(events, non_events) < MIN_EVENTS:
        base["status"] = "underpowered_outcome_model"
        base["reason"] = "insufficient complete endpoint events/non-events"
        return base
    if selection_needed:
        numerator = risk_set.groupby("episode_exposed")["_observed"].mean()
        complete["_weight"] = (
            complete["episode_exposed"].map(numerator).astype(float)
            / complete["_observation_probability"].astype(float)
        )
    else:
        complete["_weight"] = 1.0
    if not np.isfinite(complete["_weight"]).all() or complete["_weight"].le(0).any():
        base["status"] = "non_estimable_weights"
        base["reason"] = "non-finite or non-positive stabilized weights"
        return base
    lower_weight, upper_weight = np.quantile(complete["_weight"], [0.01, 0.99])
    complete["_weight"] = complete["_weight"].clip(
        float(lower_weight), float(upper_weight)
    )
    base["stabilized_weight_p01"] = float(lower_weight)
    base["stabilized_weight_p99"] = float(upper_weight)
    weight = complete["_weight"].to_numpy(float)
    base["effective_sample_size"] = float(
        np.square(weight.sum()) / np.square(weight).sum()
    )
    for exposure, label in ((1, "weighted_risk_exposed"), (0, "weighted_risk_unexposed")):
        local = complete.loc[complete["episode_exposed"].eq(exposure)]
        base[label] = (
            float(np.average(local["_event"], weights=local["_weight"]))
            if len(local) and float(local["_weight"].sum()) > 0
            else math.nan
        )
    try:
        outcome_design, retained, sm = _design_matrix(
            complete, information_count=min(events, non_events)
        )
        parameters = int(outcome_design.shape[1])
        epp = min(events, non_events) / max(parameters, 1)
        base.update(
            {
                "outcome_model_parameters": parameters,
                "outcome_events_per_parameter": epp,
                "outcome_claim_ready_epv10": bool(
                    epp >= CLAIM_EVENTS_PER_PARAMETER
                ),
                "outcome_covariates": ",".join(retained),
            }
        )
        if epp < MIN_EVENTS_PER_PARAMETER:
            base["status"] = "underpowered_outcome_model"
            base["reason"] = "outcome_events_per_parameter_below_5_fail_closed"
            return base
        groups = _model_groups(complete)
        if groups.nunique() > 1:
            fit = sm.GEE(
                complete["_event"],
                outcome_design,
                groups=groups,
                family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Independence(),
                weights=complete["_weight"],
            ).fit()
        else:
            fit = sm.GLM(
                complete["_event"],
                outcome_design,
                family=sm.families.Poisson(),
                freq_weights=complete["_weight"],
            ).fit(cov_type="HC0")
        coefficient = float(fit.params["episode_exposed"])
        low, high = fit.conf_int().loc["episode_exposed"]
        values = np.exp([coefficient, float(low), float(high)])
        if not np.isfinite(values).all():
            raise ValueError("non-finite selection-weighted risk-ratio inference")
        base.update(
            {
                "status": (
                    "estimated"
                    if selection_needed
                    else "not_needed_complete_observation"
                ),
                "selection_weighted_risk_ratio": float(values[0]),
                "selection_weighted_ci95_low": float(values[1]),
                "selection_weighted_ci95_high": float(values[2]),
                "selection_weighted_p_value": float(
                    fit.pvalues["episode_exposed"]
                ),
                "selection_weighted_log_rr_se": float(
                    fit.bse["episode_exposed"]
                ),
                "reason": "",
            }
        )
    except Exception as exc:
        base["status"] = "non_estimable_outcome_model"
        base["reason"] = str(exc)[:500]
    return base


def _apply_weighted_multiplicity(weighted: pd.DataFrame) -> pd.DataFrame:
    if weighted.empty:
        return weighted
    result = weighted.copy()
    result["endpoint_focused_q_value"] = np.nan
    for _, indexes in result.groupby(["dataset", "endpoint"]).groups.items():
        result.loc[indexes, "endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "selection_weighted_p_value"]
        )
    result["cross_endpoint_focused_q_value"] = np.nan
    for _, indexes in result.groupby("dataset").groups.items():
        result.loc[indexes, "cross_endpoint_focused_q_value"] = _bh_adjust(
            result.loc[indexes, "selection_weighted_p_value"]
        )
    return result


def _meta_effect_rows(effects: pd.DataFrame) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    selected = effects.loc[effects["focused_endpoint_window"].eq(True)].copy()  # noqa: E712
    rows: list[dict[str, Any]] = []
    specifications = (
        (
            "unadjusted",
            "risk_ratio",
            "log_risk_ratio_se",
            "status",
            "estimated",
        ),
        (
            "covariate_adjusted",
            "adjusted_risk_ratio",
            "adjusted_log_rr_se",
            "adjusted_status",
            "estimated",
        ),
    )
    for effect_type, effect_column, se_column, status_column, valid_status in specifications:
        local = selected.loc[
            selected[status_column].eq(valid_status)
            & pd.to_numeric(selected[effect_column], errors="coerce").gt(0)
            & pd.to_numeric(selected[se_column], errors="coerce").gt(0)
        ].copy()
        for (endpoint, family, lag_window), group in local.groupby(
            ["endpoint", "endpoint_family", "lag_window"], sort=False
        ):
            yi = np.log(
                pd.to_numeric(group[effect_column], errors="coerce").to_numpy(float)
            )
            sei = pd.to_numeric(group[se_column], errors="coerce").to_numpy(float)
            valid = np.isfinite(yi) & np.isfinite(sei) & (sei > 0)
            yi, sei = yi[valid], sei[valid]
            if not len(yi):
                continue
            weights = 1.0 / np.square(sei)
            fixed = float(np.sum(weights * yi) / np.sum(weights))
            fixed_se = float(math.sqrt(1.0 / np.sum(weights)))
            cochran_q = float(np.sum(weights * np.square(yi - fixed)))
            df = max(len(yi) - 1, 0)
            denominator = float(
                np.sum(weights) - np.sum(np.square(weights)) / np.sum(weights)
            )
            tau_squared = (
                max((cochran_q - df) / denominator, 0.0)
                if df > 0 and denominator > 0
                else 0.0
            )
            random_weights = 1.0 / (np.square(sei) + tau_squared)
            random_effect = float(
                np.sum(random_weights * yi) / np.sum(random_weights)
            )
            random_se = float(math.sqrt(1.0 / np.sum(random_weights)))
            try:
                from scipy.stats import norm

                fixed_p = float(2.0 * norm.sf(abs(fixed / fixed_se)))
                random_p = float(2.0 * norm.sf(abs(random_effect / random_se)))
            except Exception:  # pragma: no cover
                fixed_p = random_p = math.nan
            rows.append(
                {
                    "endpoint": str(endpoint),
                    "endpoint_family": str(family),
                    "lag_window": str(lag_window),
                    "effect_type": effect_type,
                    "datasets_contributing": int(len(yi)),
                    "dataset_names": ",".join(
                        group.loc[valid, "dataset"].astype(str)
                    ),
                    "fixed_effect_risk_ratio": math.exp(fixed),
                    "fixed_effect_ci95_low": math.exp(fixed - 1.96 * fixed_se),
                    "fixed_effect_ci95_high": math.exp(fixed + 1.96 * fixed_se),
                    "fixed_effect_p": fixed_p,
                    "random_effect_risk_ratio": math.exp(random_effect),
                    "random_effect_ci95_low": math.exp(
                        random_effect - 1.96 * random_se
                    ),
                    "random_effect_ci95_high": math.exp(
                        random_effect + 1.96 * random_se
                    ),
                    "random_effect_p": random_p,
                    "cochran_q": cochran_q,
                    "heterogeneity_df": df,
                    "i_squared_percent": (
                        max((cochran_q - df) / cochran_q * 100.0, 0.0)
                        if cochran_q > 0 and df > 0
                        else 0.0
                    ),
                    "tau_squared": tau_squared,
                    "status": (
                        "estimated" if len(yi) >= 2 else "single_dataset_only"
                    ),
                    "analysis_status": POST_HOC_STATUS,
                    "claim_scope": (
                        "cross_dataset_synthesis_not_independent_preregistration"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for column in (
        "endpoint_fixed_q_value",
        "endpoint_random_q_value",
        "cross_endpoint_fixed_q_value",
        "cross_endpoint_random_q_value",
    ):
        result[column] = np.nan
    for _, indexes in result.groupby(["effect_type", "endpoint"]).groups.items():
        result.loc[indexes, "endpoint_fixed_q_value"] = _bh_adjust(
            result.loc[indexes, "fixed_effect_p"]
        )
        result.loc[indexes, "endpoint_random_q_value"] = _bh_adjust(
            result.loc[indexes, "random_effect_p"]
        )
    for _, indexes in result.groupby("effect_type").groups.items():
        result.loc[indexes, "cross_endpoint_fixed_q_value"] = _bh_adjust(
            result.loc[indexes, "fixed_effect_p"]
        )
        result.loc[indexes, "cross_endpoint_random_q_value"] = _bh_adjust(
            result.loc[indexes, "random_effect_p"]
        )
    return result


def _grade_effect(
    *,
    status: str,
    effect: Any,
    ci_low: Any,
    endpoint_q: Any,
    cross_q: Any,
    claim_ready: bool = True,
) -> str:
    value = pd.to_numeric(effect, errors="coerce")
    low = pd.to_numeric(ci_low, errors="coerce")
    endpoint_q_value = pd.to_numeric(endpoint_q, errors="coerce")
    cross_q_value = pd.to_numeric(cross_q, errors="coerce")
    if status == "unavailable":
        return "endpoint_unavailable"
    if pd.isna(value):
        return "not_estimable_or_underpowered"
    positive = value > 1.0
    if status not in {"estimated", "not_needed_complete_observation"}:
        return (
            "directionally_positive_but_underpowered"
            if positive
            else "underpowered_without_positive_direction"
        )
    if (
        positive
        and claim_ready
        and pd.notna(cross_q_value)
        and cross_q_value < 0.05
    ):
        return "supported_after_cross_endpoint_fdr"
    if (
        positive
        and claim_ready
        and pd.notna(endpoint_q_value)
        and endpoint_q_value < 0.05
    ):
        return "supported_within_endpoint_multiplicity_sensitive"
    if positive and pd.notna(low) and low > 1.0:
        return "nominal_positive_multiplicity_sensitive"
    if positive:
        return "directionally_positive_inconclusive"
    return "no_positive_association"


def _build_evidence_summary(
    effects: pd.DataFrame,
    weighted: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    focus = effects.loc[effects["focused_endpoint_window"].eq(True)].copy()  # noqa: E712
    for row in focus.to_dict("records"):
        common = {
            "dataset": row["dataset"],
            "endpoint": row["endpoint"],
            "endpoint_family": row["endpoint_family"],
            "lag_window": row["lag_window"],
            "analysis_status": POST_HOC_STATUS,
        }
        rows.append(
            {
                **common,
                "evidence_component": "time_aligned_unadjusted_risk_ratio",
                "evidence_grade": _grade_effect(
                    status=str(row["status"]),
                    effect=row.get("risk_ratio"),
                    ci_low=row.get("risk_ratio_ci95_low"),
                    endpoint_q=row.get("endpoint_focused_q_value"),
                    cross_q=row.get("cross_endpoint_focused_q_value"),
                ),
                "effect": row.get("risk_ratio"),
                "ci95_low": row.get("risk_ratio_ci95_low"),
                "ci95_high": row.get("risk_ratio_ci95_high"),
                "nominal_p_value": row.get("fisher_two_sided_p"),
                "endpoint_q_value": row.get("endpoint_focused_q_value"),
                "cross_endpoint_q_value": row.get(
                    "cross_endpoint_focused_q_value"
                ),
                "claim_ready_information": row.get("status") == "estimated",
            }
        )
        rows.append(
            {
                **common,
                "evidence_component": "covariate_adjusted_risk_ratio",
                "evidence_grade": _grade_effect(
                    status=str(row.get("adjusted_status")),
                    effect=row.get("adjusted_risk_ratio"),
                    ci_low=row.get("adjusted_ci95_low"),
                    endpoint_q=row.get("endpoint_focused_adjusted_q_value"),
                    cross_q=row.get("cross_endpoint_focused_adjusted_q_value"),
                    claim_ready=bool(row.get("adjusted_claim_ready_epv10")),
                ),
                "effect": row.get("adjusted_risk_ratio"),
                "ci95_low": row.get("adjusted_ci95_low"),
                "ci95_high": row.get("adjusted_ci95_high"),
                "nominal_p_value": row.get("adjusted_p_value"),
                "endpoint_q_value": row.get(
                    "endpoint_focused_adjusted_q_value"
                ),
                "cross_endpoint_q_value": row.get(
                    "cross_endpoint_focused_adjusted_q_value"
                ),
                "claim_ready_information": bool(
                    row.get("adjusted_claim_ready_epv10")
                ),
            }
        )
        continuous_effect = pd.to_numeric(
            row.get("median_change_difference"), errors="coerce"
        )
        rows.append(
            {
                **common,
                "evidence_component": "continuous_worsening_change",
                "evidence_grade": (
                    "continuous_change_supported_after_cross_endpoint_fdr"
                    if pd.notna(continuous_effect)
                    and continuous_effect > 0
                    and pd.to_numeric(
                        row.get("cross_endpoint_focused_continuous_q_value"),
                        errors="coerce",
                    )
                    < 0.05
                    else "directionally_positive_inconclusive"
                    if pd.notna(continuous_effect) and continuous_effect > 0
                    else "no_positive_continuous_change"
                    if pd.notna(continuous_effect)
                    else "continuous_endpoint_not_applicable_or_unobserved"
                ),
                "effect": continuous_effect,
                "ci95_low": row.get("median_change_difference_ci95_low"),
                "ci95_high": row.get("median_change_difference_ci95_high"),
                "nominal_p_value": row.get("mann_whitney_two_sided_p"),
                "endpoint_q_value": row.get(
                    "endpoint_focused_continuous_q_value"
                ),
                "cross_endpoint_q_value": row.get(
                    "cross_endpoint_focused_continuous_q_value"
                ),
                "claim_ready_information": bool(
                    row.get("n_continuous_exposed", 0) >= 5
                    and row.get("n_continuous_unexposed", 0) >= 5
                ),
            }
        )
    for row in weighted.to_dict("records"):
        rows.append(
            {
                "dataset": row["dataset"],
                "endpoint": row["endpoint"],
                "endpoint_family": row["endpoint_family"],
                "lag_window": row["lag_window"],
                "evidence_component": "selection_and_censoring_weighted_risk_ratio",
                "evidence_grade": _grade_effect(
                    status=str(row["status"]),
                    effect=row.get("selection_weighted_risk_ratio"),
                    ci_low=row.get("selection_weighted_ci95_low"),
                    endpoint_q=row.get("endpoint_focused_q_value"),
                    cross_q=row.get("cross_endpoint_focused_q_value"),
                    claim_ready=bool(row.get("outcome_claim_ready_epv10")),
                ),
                "effect": row.get("selection_weighted_risk_ratio"),
                "ci95_low": row.get("selection_weighted_ci95_low"),
                "ci95_high": row.get("selection_weighted_ci95_high"),
                "nominal_p_value": row.get("selection_weighted_p_value"),
                "endpoint_q_value": row.get("endpoint_focused_q_value"),
                "cross_endpoint_q_value": row.get(
                    "cross_endpoint_focused_q_value"
                ),
                "claim_ready_information": bool(
                    row.get("outcome_claim_ready_epv10")
                ),
                "analysis_status": POST_HOC_STATUS,
            }
        )
    for row in meta.to_dict("records"):
        rows.append(
            {
                "dataset": "cross_dataset",
                "endpoint": row["endpoint"],
                "endpoint_family": row["endpoint_family"],
                "lag_window": row["lag_window"],
                "evidence_component": f"{row['effect_type']}_random_effect_meta_analysis",
                "evidence_grade": _grade_effect(
                    status=str(row["status"]),
                    effect=row.get("random_effect_risk_ratio"),
                    ci_low=row.get("random_effect_ci95_low"),
                    endpoint_q=row.get("endpoint_random_q_value"),
                    cross_q=row.get("cross_endpoint_random_q_value"),
                    claim_ready=row.get("datasets_contributing", 0) >= 2,
                ),
                "effect": row.get("random_effect_risk_ratio"),
                "ci95_low": row.get("random_effect_ci95_low"),
                "ci95_high": row.get("random_effect_ci95_high"),
                "nominal_p_value": row.get("random_effect_p"),
                "endpoint_q_value": row.get("endpoint_random_q_value"),
                "cross_endpoint_q_value": row.get(
                    "cross_endpoint_random_q_value"
                ),
                "claim_ready_information": row.get("datasets_contributing", 0)
                >= 2,
                "analysis_status": POST_HOC_STATUS,
            }
        )
    return pd.DataFrame(rows)


def _build_key_results(
    effects: pd.DataFrame,
    weighted: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """One-row-per-dataset/endpoint focused results for notebook review."""
    focus = effects.loc[effects["focused_endpoint_window"].eq(True)].copy()  # noqa: E712
    if focus.empty:
        return focus
    weighted_columns = [
        "dataset",
        "endpoint",
        "lag_window",
        "status",
        "selection_weighted_risk_ratio",
        "selection_weighted_ci95_low",
        "selection_weighted_ci95_high",
        "selection_weighted_p_value",
        "endpoint_focused_q_value",
        "cross_endpoint_focused_q_value",
        "outcome_events_per_parameter",
        "outcome_claim_ready_epv10",
    ]
    weighted_local = weighted.reindex(columns=weighted_columns).rename(
        columns={
            "status": "weighted_status",
            "endpoint_focused_q_value": "weighted_endpoint_q_value",
            "cross_endpoint_focused_q_value": "weighted_cross_endpoint_q_value",
        }
    )
    result = focus.merge(
        weighted_local,
        on=["dataset", "endpoint", "lag_window"],
        how="left",
        validate="one_to_one",
    )
    meta_columns = [
        "endpoint",
        "lag_window",
        "datasets_contributing",
        "random_effect_risk_ratio",
        "random_effect_ci95_low",
        "random_effect_ci95_high",
        "random_effect_p",
        "endpoint_random_q_value",
        "cross_endpoint_random_q_value",
        "i_squared_percent",
    ]
    if meta.empty or not {"effect_type", "datasets_contributing"}.issubset(meta):
        meta_local = pd.DataFrame(columns=meta_columns)
    else:
        meta_local = meta.loc[
            meta["effect_type"].eq("unadjusted")
            & pd.to_numeric(meta["datasets_contributing"], errors="coerce").ge(2)
        ].reindex(columns=meta_columns)
    result = result.merge(
        meta_local,
        on=["endpoint", "lag_window"],
        how="left",
        validate="many_to_one",
    )
    result["unadjusted_cross_fdr_positive"] = (
        pd.to_numeric(result["risk_ratio"], errors="coerce").gt(1)
        & pd.to_numeric(
            result["cross_endpoint_focused_q_value"], errors="coerce"
        ).lt(0.05)
    )
    result["adjusted_claim_ready_positive"] = (
        result["adjusted_status"].eq("estimated")
        & result["adjusted_claim_ready_epv10"].eq(True)  # noqa: E712
        & pd.to_numeric(result["adjusted_risk_ratio"], errors="coerce").gt(1)
        & pd.to_numeric(result["adjusted_ci95_low"], errors="coerce").gt(1)
    )
    result["adjusted_cross_fdr_positive"] = (
        result["adjusted_claim_ready_positive"]
        & pd.to_numeric(
            result["cross_endpoint_focused_adjusted_q_value"], errors="coerce"
        ).lt(0.05)
    )
    result["weighted_claim_ready_positive"] = (
        result["weighted_status"].isin(
            ["estimated", "not_needed_complete_observation"]
        )
        & result["outcome_claim_ready_epv10"].eq(True)  # noqa: E712
        & pd.to_numeric(
            result["selection_weighted_risk_ratio"], errors="coerce"
        ).gt(1)
        & pd.to_numeric(
            result["selection_weighted_ci95_low"], errors="coerce"
        ).gt(1)
    )
    result["weighted_cross_fdr_positive"] = (
        result["weighted_claim_ready_positive"]
        & pd.to_numeric(
            result["weighted_cross_endpoint_q_value"], errors="coerce"
        ).lt(0.05)
    )
    result["replicated_random_meta_positive"] = (
        pd.to_numeric(result["datasets_contributing"], errors="coerce").ge(2)
        & pd.to_numeric(
            result["random_effect_risk_ratio"], errors="coerce"
        ).gt(1)
        & pd.to_numeric(
            result["cross_endpoint_random_q_value"], errors="coerce"
        ).lt(0.05)
    )

    def classify(row: pd.Series) -> str:
        if row["status"] == "unavailable":
            return "endpoint_unavailable"
        risk_ratio = pd.to_numeric(row.get("risk_ratio"), errors="coerce")
        if row["status"] != "estimated":
            return (
                "directionally_positive_but_underpowered"
                if pd.notna(risk_ratio) and risk_ratio > 1
                else "underpowered_without_positive_direction"
            )
        if (
            bool(row["unadjusted_cross_fdr_positive"])
            and bool(row["adjusted_cross_fdr_positive"])
            and bool(row["weighted_cross_fdr_positive"])
        ):
            return "robust_to_adjustment_weighting_and_cross_endpoint_fdr"
        if (
            bool(row["unadjusted_cross_fdr_positive"])
            and bool(row["adjusted_claim_ready_positive"])
            and bool(row["weighted_claim_ready_positive"])
        ):
            return "unadjusted_cross_fdr_with_adjusted_weighted_ci_support"
        if bool(row["replicated_random_meta_positive"]):
            return "cross_dataset_random_effect_fdr_signal"
        if bool(row["unadjusted_cross_fdr_positive"]):
            return "unadjusted_cross_endpoint_fdr_signal"
        endpoint_q = pd.to_numeric(
            row.get("endpoint_focused_q_value"), errors="coerce"
        )
        if pd.notna(risk_ratio) and risk_ratio > 1 and pd.notna(endpoint_q) and endpoint_q < 0.05:
            return "within_endpoint_fdr_signal"
        ci_low = pd.to_numeric(row.get("risk_ratio_ci95_low"), errors="coerce")
        if pd.notna(risk_ratio) and risk_ratio > 1 and pd.notna(ci_low) and ci_low > 1:
            return "nominal_positive_multiplicity_sensitive"
        if pd.notna(risk_ratio) and risk_ratio > 1:
            return "directionally_positive_inconclusive"
        return "no_positive_association"

    result["evidence_label"] = result.apply(classify, axis=1)
    keep = [
        "dataset",
        "endpoint",
        "endpoint_family",
        "lag_window",
        "status",
        "n_exposed",
        "n_unexposed",
        "events_exposed",
        "events_unexposed",
        "risk_ratio",
        "risk_ratio_ci95_low",
        "risk_ratio_ci95_high",
        "fisher_two_sided_p",
        "endpoint_focused_q_value",
        "cross_endpoint_focused_q_value",
        "adjusted_status",
        "adjusted_risk_ratio",
        "adjusted_ci95_low",
        "adjusted_ci95_high",
        "adjusted_p_value",
        "endpoint_focused_adjusted_q_value",
        "cross_endpoint_focused_adjusted_q_value",
        "adjusted_events_per_parameter",
        "adjusted_claim_ready_epv10",
        "weighted_status",
        "selection_weighted_risk_ratio",
        "selection_weighted_ci95_low",
        "selection_weighted_ci95_high",
        "selection_weighted_p_value",
        "weighted_endpoint_q_value",
        "weighted_cross_endpoint_q_value",
        "outcome_events_per_parameter",
        "outcome_claim_ready_epv10",
        "median_change_difference",
        "endpoint_focused_continuous_q_value",
        "cross_endpoint_focused_continuous_q_value",
        "datasets_contributing",
        "random_effect_risk_ratio",
        "random_effect_ci95_low",
        "random_effect_ci95_high",
        "cross_endpoint_random_q_value",
        "i_squared_percent",
        "unadjusted_cross_fdr_positive",
        "adjusted_claim_ready_positive",
        "adjusted_cross_fdr_positive",
        "weighted_claim_ready_positive",
        "weighted_cross_fdr_positive",
        "replicated_random_meta_positive",
        "evidence_label",
        "analysis_status",
    ]
    return result.reindex(columns=keep).sort_values(
        ["dataset", "endpoint", "lag_window"]
    ).reset_index(drop=True)


def _endpoint_definitions_table() -> pd.DataFrame:
    rule_text = {
        "lactate_rise": "first next lactate minus strict pre-anchor lactate >=0.5 mmol/L",
        "aki_creatinine": "window maximum minus baseline >=0.3 mg/dL or ratio >=1.5",
        "bilirubin_worsening": "window maximum minus baseline >=0.5 mg/dL or ratio >=1.5",
        "ast_injury": "baseline <200 and maximum >=200 U/L, or maximum/baseline >=2",
        "alt_injury": "baseline <200 and maximum >=200 U/L, or maximum/baseline >=2",
        "inr_injury": "window maximum minus baseline >=0.3 or ratio >=1.5",
        "platelet_injury": "baseline >=100 and minimum <100, or minimum/baseline <=0.70",
    }
    rows: list[dict[str, Any]] = []
    for spec in LAB_SPECS:
        rows.append(
            {
                "endpoint": spec.endpoint,
                "endpoint_family": spec.family,
                "primary_outcome_variant": spec.primary_variant,
                "binary_event_definition": rule_text[spec.endpoint],
                "continuous_estimand": spec.change_scale,
                "strict_preanchor_baseline": "last valid value strictly before anchor within 6 hours",
                "source_method": f"cached {spec.concept} laboratory events",
                "focused_windows": ",".join(
                    FOCUSED_WINDOWS_BY_FAMILY[spec.family]
                ),
                "definition_role": "protocol_threshold_carried_to_episode_anchor",
            }
        )
    rows.extend(
        [
            {
                "endpoint": "troponin_relative_rise",
                "endpoint_family": "laboratory_organ_injury",
                "primary_outcome_variant": "worst_in_window",
                "binary_event_definition": "maximum same-assay troponin T or I ratio >=1.5; injury marker, not adjudicated MI",
                "continuous_estimand": "maximum assay-matched ratio minus one",
                "strict_preanchor_baseline": "last same-assay value strictly before anchor within 6 hours",
                "source_method": "cached troponin T and I events kept assay-specific",
                "focused_windows": "0_to_8h,8_to_24h",
                "definition_role": "protocol_threshold_carried_to_episode_anchor",
            },
            {
                "endpoint": "vis_escalation",
                "endpoint_family": "hemodynamic_support",
                "primary_outcome_variant": "worst_in_window",
                "binary_event_definition": "new VIS maximum above strict pre-anchor maximum; delayed bins exclude prior escalation",
                "continuous_estimand": "window maximum VIS minus strict pre-anchor maximum",
                "strict_preanchor_baseline": "maximum valid VIS strictly before anchor; zero if no support",
                "source_method": "standardized cached VIS only",
                "focused_windows": "0_to_4h,4_to_12h",
                "definition_role": "incident_escalation_timing_amendment",
            },
            {
                "endpoint": "oliguria_kdigo_proxy",
                "endpoint_family": "urine_output",
                "primary_outcome_variant": "absolute_rate",
                "binary_event_definition": "window urine rate <0.5 mL/kg/h with >=50% 2-hour-bin coverage",
                "continuous_estimand": "negative future urine rate in mL/kg/h",
                "strict_preanchor_baseline": "admission/baseline weight strictly external to outcome window",
                "source_method": "cached urine-output events plus weight",
                "focused_windows": "0_to_4h,4_to_12h",
                "definition_role": "absolute_kdigo_rate_proxy",
            },
            {
                "endpoint": "urine_output_decline_proxy",
                "endpoint_family": "urine_output",
                "primary_outcome_variant": "relative_decline",
                "binary_event_definition": "future covered urine rate declines >=50% from covered strict pre-anchor rate",
                "continuous_estimand": "one minus future/pre-anchor urine-rate ratio",
                "strict_preanchor_baseline": "up to 4 hours strictly before anchor, >=2 hours and >=50% bin coverage",
                "source_method": "cached urine-output events",
                "focused_windows": "0_to_4h,4_to_12h",
                "definition_role": "strict_preepisode_decline_companion",
            },
        ]
    )
    for spec in EVENT_SPECS:
        rows.append(
            {
                "endpoint": spec["endpoint"],
                "endpoint_family": spec["family"],
                "primary_outcome_variant": "incident_event",
                "binary_event_definition": (
                    f"first cached {spec['concept']} event in lag among anchors with no event before lag start"
                ),
                "continuous_estimand": "not_applicable",
                "strict_preanchor_baseline": (
                    f"incident risk set; baseline flag {spec.get('baseline_flag') or 'none'}"
                ),
                "source_method": f"cached {spec['concept']} event source",
                "focused_windows": ",".join(
                    FOCUSED_WINDOWS_BY_FAMILY[spec["family"]]
                ),
                "definition_role": "incident_intervention_or_event",
            }
        )
    for spec in COMPOSITE_SPECS:
        rows.append(
            {
                "endpoint": spec["endpoint"],
                "endpoint_family": spec["family"],
                "primary_outcome_variant": "conservative_any",
                "binary_event_definition": (
                    "any observed component event; a composite non-event requires every available component to be at risk and observed"
                ),
                "continuous_estimand": "not_applicable_noncommensurable_components",
                "strict_preanchor_baseline": "component-specific strict baselines and incident risk sets",
                "source_method": "+".join(spec["components"]),
                "focused_windows": ",".join(
                    FOCUSED_WINDOWS_BY_FAMILY[spec["family"]]
                ),
                "definition_role": "conservative_any_composite",
            }
        )
    result = pd.DataFrame(rows)
    result["definition_version"] = "spo2_multiorgan_episode_v1"
    result["lag_grid_frozen_before_all_endpoint_run"] = True
    result["analysis_status"] = POST_HOC_STATUS
    return result


def _availability_table(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, endpoint, family, lag_window), group in records.groupby(
        ["dataset", "endpoint", "endpoint_family", "lag_window"], sort=False
    ):
        source = pd.to_numeric(group["source_available"], errors="coerce").fillna(0).eq(1)
        at_risk = source & pd.to_numeric(group["at_risk"], errors="coerce").fillna(0).eq(1)
        baseline = at_risk & pd.to_numeric(
            group["baseline_observed"], errors="coerce"
        ).fillna(0).eq(1)
        observed = at_risk & pd.to_numeric(
            group["outcome_observed"], errors="coerce"
        ).fillna(0).eq(1)
        n_source = int(source.sum())
        if n_source == 0:
            status = "unavailable"
            if str(dataset) == "mimic" and str(endpoint).startswith("urine_"):
                reason = "MIMIC cached outputevents/urine source unavailable"
            elif str(dataset) == "mimic" and endpoint == "oliguria_kdigo_proxy":
                reason = "MIMIC cached outputevents/urine source unavailable"
            elif str(dataset) == "eicu" and endpoint == "vis_escalation":
                reason = "eICU infusion-rate units are not standardized for quantitative VIS"
            elif (
                str(dataset) == "mimic"
                and endpoint == "invasive_ventilation_initiation"
            ):
                reason = "MIMIC cache has respiratory context but no harmonized incident mechanical-ventilation event"
            else:
                reason = "endpoint-specific cached source unavailable"
        else:
            status = "available" if int(observed.sum()) else "available_no_observed_pairs"
            reason = ""
        rows.append(
            {
                "dataset": str(dataset),
                "endpoint": str(endpoint),
                "endpoint_family": str(family),
                "lag_window": str(lag_window),
                "status": status,
                "reason": reason,
                "n_anchor_rows": int(len(group)),
                "n_source_available": n_source,
                "n_at_risk": int(at_risk.sum()),
                "n_strict_baseline_observed": int(baseline.sum()),
                "n_outcome_observed": int(observed.sum()),
                "outcome_observed_fraction_of_at_risk": (
                    float(observed.sum() / at_risk.sum())
                    if int(at_risk.sum())
                    else math.nan
                ),
                "analysis_status": POST_HOC_STATUS,
            }
        )
    return pd.DataFrame(rows)


def rebuild_multiorgan_weighted_sensitivity(
    records: pd.DataFrame,
    effects: pd.DataFrame,
    meta: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Rebuild only IPW/GEE and its evidence rows from focused records."""
    if records.empty:
        weighted = pd.DataFrame()
    else:
        weighted = _apply_weighted_multiplicity(
            pd.DataFrame(
                [
                    _selection_weighted_row(group)
                    for _, group in records.groupby(
                        ["dataset", "endpoint", "lag_window"], sort=False
                    )
                ]
            )
        )
    return {
        "multiorgan_episode_measurement_weighted": weighted,
        "multiorgan_episode_evidence_summary": _build_evidence_summary(
            effects, weighted, meta
        ),
        "multiorgan_episode_key_results": _build_key_results(
            effects, weighted, meta
        ),
    }


def _locked_validation_anchors(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None,
) -> pd.DataFrame:
    definition = next(
        definition
        for definition in ANALYZED_DEFINITIONS
        if (
            str(definition["episode_definition"]),
            str(definition["signal_resolution"]),
        )
        == PRIMARY_DEFINITION
    )
    transitions, support = _prepare_spo2_transitions(
        events_df,
        signal_resolution=str(definition["signal_resolution"]),
    )
    anchors = _select_anchor_per_stay(
        transitions,
        support,
        definition=definition,
    )
    if anchors.empty:
        return anchors
    anchors = _merge_anchor_context(anchors, cohort_df, analysis_df)
    return _merge_anchor_extras(anchors, cohort_df, analysis_df).reset_index(
        drop=True
    )


def _clean_eicu_ventilation_proxy(
    events_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove known false-positive eICU ventilation documentation rows."""
    events = events_df.copy()
    dataset = events.get("dataset", pd.Series(index=events.index, dtype=str)).astype(
        str
    )
    concept = events.get("concept", pd.Series(index=events.index, dtype=str)).astype(
        str
    )
    source = events.get(
        "source_table", pd.Series(index=events.index, dtype=str)
    ).astype(str)
    target = (
        dataset.eq("eicu")
        & concept.eq("mechanical_ventilation")
        & source.eq("treatment.csv")
    )
    text = (
        events.get("raw_name", pd.Series(index=events.index, dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    exact_segment = text.str.contains(
        r"(?:^|\|)mechanical ventilation(?:\||$)", regex=True, na=False
    )
    noninvasive = text.str.contains(
        r"non[ -]?invasive ventilation", regex=True, na=False
    )
    weaning = text.str.contains(r"ventilator weaning", regex=True, na=False)
    retain = target & exact_segment & ~noninvasive & ~weaning
    remove = target & ~retain
    audit = pd.DataFrame(
        [
            {
                "dataset": "eicu",
                "endpoint": "invasive_ventilation_initiation",
                "source_table": "treatment.csv",
                "input_documentation_rows": int(target.sum()),
                "retained_literal_mechanical_ventilation_rows": int(retain.sum()),
                "removed_total_rows": int(remove.sum()),
                "removed_noninvasive_rows": int((target & noninvasive).sum()),
                "removed_weaning_rows": int((target & weaning).sum()),
                "removed_nonliteral_false_matches": int(
                    (target & ~exact_segment & ~noninvasive & ~weaning).sum()
                ),
                "retained_rule": (
                    "literal_pipe_delimited_mechanical_ventilation_segment_"
                    "excluding_noninvasive_and_weaning"
                ),
                "time_semantics": (
                    "first_treatment_documentation_proxy_not_true_start"
                ),
            }
        ]
    )
    return events.loc[~remove].reset_index(drop=True), audit


def _recompute_locked_ventilation_baseline(
    analysis_df: pd.DataFrame | None,
    events_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Prevent a historical broad eICU text flag from leaking into risk sets."""
    if analysis_df is None:
        return None
    result = analysis_df.copy()
    concept = events_df.get(
        "concept", pd.Series(index=events_df.index, dtype=str)
    ).astype(str)
    offsets = pd.to_numeric(events_df.get("offset_minutes"), errors="coerce")
    ventilation = events_df.loc[concept.eq("mechanical_ventilation") & offsets.le(240)]
    keys = set(
        ventilation[["dataset", "stay_id"]]
        .astype({"dataset": str})
        .itertuples(index=False, name=None)
    )
    result["mechanical_ventilation_flag"] = [
        int((str(dataset), stay_id) in keys)
        for dataset, stay_id in result[["dataset", "stay_id"]].itertuples(
            index=False, name=None
        )
    ]
    return result


def _apply_locked_tier_multiplicity(
    frame: pd.DataFrame,
    p_columns: tuple[tuple[str, str], ...],
    *,
    extra_group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["locked_validation_tier"] = result["endpoint"].map(
        LOCKED_ENDPOINT_TIER
    ).fillna("unclassified_secondary")
    for p_column, q_column in p_columns:
        result[q_column] = np.nan
        grouping = [
            "dataset",
            "locked_validation_tier",
            *extra_group_columns,
        ]
        for _, indexes in result.groupby(grouping, dropna=False).groups.items():
            result.loc[indexes, q_column] = _bh_adjust(
                result.loc[indexes, p_column]
            )
    result["locked_transferred_primary"] = result["endpoint"].eq(
        LOCKED_EXTERNAL_VALIDATION_ENDPOINT
    )
    return result


def _locked_external_summary(
    effects: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    if effects.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    adjusted_meta = (
        meta.loc[meta["effect_type"].eq("covariate_adjusted")]
        if not meta.empty and "effect_type" in meta
        else pd.DataFrame()
    )
    for endpoint in effects["endpoint"].drop_duplicates():
        local = effects.loc[effects["endpoint"].eq(endpoint)]

        def dataset_row(dataset: str) -> dict[str, Any]:
            selected = local.loc[local["dataset"].eq(dataset)]
            return selected.iloc[0].to_dict() if len(selected) else {}

        eicu = dataset_row("eicu")
        mimic = dataset_row("mimic")
        meta_row = adjusted_meta.loc[adjusted_meta["endpoint"].eq(endpoint)]
        pooled = meta_row.iloc[0].to_dict() if len(meta_row) else {}

        def positive(row: dict[str, Any]) -> bool:
            return bool(
                row.get("adjusted_status") == "estimated"
                and pd.to_numeric(row.get("adjusted_ci95_low"), errors="coerce")
                > 1.0
            )

        eicu_positive = positive(eicu)
        mimic_positive = positive(mimic)
        both_direction_positive = bool(
            pd.to_numeric(eicu.get("adjusted_risk_ratio"), errors="coerce") > 1
            and pd.to_numeric(mimic.get("adjusted_risk_ratio"), errors="coerce")
            > 1
        )
        pooled_adjusted_tier_fdr_positive = bool(
            pd.to_numeric(
                pooled.get("random_effect_ci95_low"), errors="coerce"
            )
            > 1
            and pd.to_numeric(
                pooled.get("locked_tier_random_q_value"), errors="coerce"
            )
            < 0.05
        )
        tier = LOCKED_ENDPOINT_TIER.get(
            str(endpoint), "unclassified_secondary"
        )
        mimic_estimable = mimic.get("adjusted_status") == "estimated"
        if tier == "transferred_primary":
            if not mimic or mimic.get("status") == "unavailable":
                conclusion = "primary_mimic_endpoint_unavailable"
            elif eicu_positive and mimic_positive:
                conclusion = "transferred_primary_replicated"
            elif mimic_estimable and both_direction_positive:
                conclusion = "directionally_consistent_not_confirmed_in_mimic"
            elif mimic_estimable:
                conclusion = "transferred_primary_not_replicated"
            else:
                conclusion = "transferred_primary_underpowered_or_nonestimable"
        elif both_direction_positive and pooled_adjusted_tier_fdr_positive:
            conclusion = "secondary_adjusted_random_meta_tier_fdr_positive"
        elif eicu_positive and mimic_positive:
            conclusion = "secondary_cross_database_positive"
        elif both_direction_positive:
            conclusion = "secondary_directionally_consistent"
        elif mimic_positive:
            conclusion = "mimic_only_secondary_positive"
        elif eicu_positive:
            conclusion = "eicu_only_secondary_positive"
        else:
            conclusion = "secondary_not_positive_in_both_databases"
        rows.append(
            {
                "endpoint": endpoint,
                "locked_validation_tier": tier,
                "lag_window": LOCKED_EXTERNAL_VALIDATION_WINDOW,
                "eicu_adjusted_status": eicu.get("adjusted_status"),
                "eicu_adjusted_risk_ratio": eicu.get("adjusted_risk_ratio"),
                "eicu_adjusted_ci95_low": eicu.get("adjusted_ci95_low"),
                "eicu_adjusted_ci95_high": eicu.get("adjusted_ci95_high"),
                "eicu_adjusted_p_value": eicu.get("adjusted_p_value"),
                "eicu_locked_tier_adjusted_q_value": eicu.get(
                    "locked_tier_adjusted_q_value"
                ),
                "mimic_adjusted_status": mimic.get("adjusted_status"),
                "mimic_adjusted_risk_ratio": mimic.get("adjusted_risk_ratio"),
                "mimic_adjusted_ci95_low": mimic.get("adjusted_ci95_low"),
                "mimic_adjusted_ci95_high": mimic.get("adjusted_ci95_high"),
                "mimic_adjusted_p_value": mimic.get("adjusted_p_value"),
                "mimic_locked_tier_adjusted_q_value": mimic.get(
                    "locked_tier_adjusted_q_value"
                ),
                "both_adjusted_directions_positive": both_direction_positive,
                "pooled_datasets": pooled.get("datasets_contributing"),
                "pooled_adjusted_random_effect_rr": pooled.get(
                    "random_effect_risk_ratio"
                ),
                "pooled_adjusted_random_effect_ci95_low": pooled.get(
                    "random_effect_ci95_low"
                ),
                "pooled_adjusted_random_effect_ci95_high": pooled.get(
                    "random_effect_ci95_high"
                ),
                "pooled_adjusted_random_effect_p": pooled.get("random_effect_p"),
                "pooled_adjusted_locked_tier_random_q_value": pooled.get(
                    "locked_tier_random_q_value"
                ),
                "pooled_i_squared_percent": pooled.get("i_squared_percent"),
                "conclusion": conclusion,
            }
        )
    order = {
        "transferred_primary": 0,
        "mimic_respiratory_specificity": 1,
        "clinical_escalation_secondary": 2,
        "organ_injury_secondary": 3,
        "exploratory_composite": 4,
    }
    result = pd.DataFrame(rows)
    result["_tier_order"] = result["locked_validation_tier"].map(order).fillna(9)
    return result.sort_values(["_tier_order", "endpoint"]).drop(
        columns="_tier_order"
    ).reset_index(drop=True)


def _locked_external_design_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "design_element": "cohort",
                "locked_value": HARMONIZED_HF_PHENOTYPE,
                "role": "identical_fail_closed_rule_in_mimic_and_eicu",
            },
            {
                "design_element": "exposure",
                "locked_value": PRIMARY_DEFINITION[0],
                "role": "first_gap_qualified_absolute_spo2_jump_ge4",
            },
            {
                "design_element": "signal_resolution",
                "locked_value": PRIMARY_DEFINITION[1],
                "role": "same_as_eicu_discovery_analysis",
            },
            {
                "design_element": "lag_window",
                "locked_value": LOCKED_EXTERNAL_VALIDATION_WINDOW,
                "role": "transferred_unchanged_from_eicu",
            },
            {
                "design_element": "primary_endpoint",
                "locked_value": LOCKED_EXTERNAL_VALIDATION_ENDPOINT,
                "role": "single_confirmatory_mimic_test_no_endpoint_multiplicity",
            },
            {
                "design_element": "mimic_specificity_endpoint",
                "locked_value": "intubation_initiation",
                "role": "secondary_direct_procedure_sensitivity",
            },
            {
                "design_element": "eicu_primary_endpoint",
                "locked_value": (
                    "literal_pipe_delimited_mechanical_ventilation_treatment_"
                    "segment_excluding_noninvasive_and_weaning"
                ),
                "role": (
                    "incident_first_documentation_proxy_with_false_positive_"
                    "cleanup_audited"
                ),
            },
            {
                "design_element": "secondary_multiplicity",
                "locked_value": "benjamini_hochberg_within_dataset_and_tier",
                "role": "secondary_results_cannot_redefine_primary_replication",
            },
            {
                "design_element": "claim_scope",
                "locked_value": "observational_external_replication_after_eicu_discovery",
                "role": "not_prospective_preregistration_and_not_causal",
            },
        ]
    )


def run_locked_external_replication(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = 0,
) -> dict[str, pd.DataFrame]:
    """Test the exact eICU 4--12 h association in harmonized MIMIC HF.

    This is intentionally a single-definition, single-window runner.  It is
    suitable for a fast analysis-only refresh and cannot select an endpoint,
    exposure definition, or lag window using MIMIC results.
    """
    harmonized, cohort_audit = harmonize_hf_cohort(cohort_df)
    events = filter_to_harmonized_cohort(events_df, harmonized)
    events, eicu_endpoint_audit = _clean_eicu_ventilation_proxy(events)
    analysis = (
        filter_to_harmonized_cohort(analysis_df, harmonized)
        if analysis_df is not None
        else None
    )
    analysis = _recompute_locked_ventilation_baseline(analysis, events)
    lookup = EventLookup(events)
    anchors = _locked_validation_anchors(events, harmonized, analysis)
    if anchors.empty:
        raise RuntimeError(
            "No harmonized stays had enough first-four-hour SpO2 support for "
            "the locked external validation"
        )
    component_frames = _add_composite_frames(
        _component_record_frames(
            anchors,
            lookup,
            lag_window=LOCKED_EXTERNAL_VALIDATION_WINDOW,
        )
    )
    effect_rows: list[dict[str, Any]] = []
    record_frames: list[pd.DataFrame] = []
    weighted_rows: list[dict[str, Any]] = []
    for endpoint, frame in component_frames.items():
        if frame.empty or endpoint not in PRIMARY_VARIANT_BY_ENDPOINT:
            continue
        primary = _select_primary_variant(frame, endpoint)
        record_frames.append(primary)
        effect_rows.extend(
            _summarize_record_frame(
                primary,
                primary_variant=True,
                primary_definition=True,
                bootstrap_repetitions=bootstrap_repetitions,
            )
        )
        for _, dataset_frame in primary.groupby("dataset", sort=False):
            weighted_rows.append(_selection_weighted_row(dataset_frame))
    records = (
        pd.concat(record_frames, ignore_index=True, sort=False)
        if record_frames
        else pd.DataFrame()
    )
    effects = pd.DataFrame(effect_rows)
    if not effects.empty:
        # For this runner, the transferred 4--12 h window is the only tested
        # window for every endpoint.  Secondary tiers remain explicitly marked.
        effects["definition_focused_window"] = True
        effects["focused_endpoint_window"] = True
        effects = _apply_effect_multiplicity(effects)
        effects = _apply_locked_tier_multiplicity(
            effects,
            (
                ("fisher_two_sided_p", "locked_tier_binary_q_value"),
                ("adjusted_p_value", "locked_tier_adjusted_q_value"),
                (
                    "mann_whitney_two_sided_p",
                    "locked_tier_continuous_q_value",
                ),
            ),
        )
    weighted = _apply_weighted_multiplicity(pd.DataFrame(weighted_rows))
    if not weighted.empty:
        weighted = _apply_locked_tier_multiplicity(
            weighted,
            (("selection_weighted_p_value", "locked_tier_weighted_q_value"),),
        )
    meta = _meta_effect_rows(effects)
    if not meta.empty:
        meta["dataset"] = "cross_dataset"
        meta = _apply_locked_tier_multiplicity(
            meta,
            (
                ("fixed_effect_p", "locked_tier_fixed_q_value"),
                ("random_effect_p", "locked_tier_random_q_value"),
            ),
            extra_group_columns=("effect_type",),
        )
    key_results = _build_key_results(effects, weighted, meta)
    if not key_results.empty:
        key_results["locked_validation_tier"] = key_results["endpoint"].map(
            LOCKED_ENDPOINT_TIER
        ).fillna("unclassified_secondary")
        key_results["locked_transferred_primary"] = key_results["endpoint"].eq(
            LOCKED_EXTERNAL_VALIDATION_ENDPOINT
        )
    availability = _availability_table(records)
    if not availability.empty:
        availability["locked_validation_tier"] = availability["endpoint"].map(
            LOCKED_ENDPOINT_TIER
        ).fillna("unclassified_secondary")
    return {
        "locked_external_validation_records": records,
        "locked_external_validation_effects": effects,
        "locked_external_validation_measurement_weighted": weighted,
        "locked_external_validation_meta_analysis": meta,
        "locked_external_validation_key_results": key_results,
        "locked_external_validation_summary": _locked_external_summary(
            effects, meta
        ),
        "locked_external_validation_availability": availability,
        "locked_external_validation_hf_cohort_audit": cohort_audit,
        "locked_external_validation_eicu_endpoint_audit": eicu_endpoint_audit,
        "locked_external_validation_design": _locked_external_design_table(),
    }


def run_multiorgan_episode_analyses(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    analysis_df: pd.DataFrame | None = None,
    *,
    bootstrap_repetitions: int = 500,
) -> dict[str, pd.DataFrame]:
    """Run the cached-event, all-endpoint episode-anchored amendment."""
    lookup = EventLookup(events_df)
    anchors_by_definition = _build_anchor_sets(events_df, cohort_df, analysis_df)
    effect_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    weighted_rows: list[dict[str, Any]] = []
    stored_records: list[pd.DataFrame] = []
    record_columns = (
        "dataset",
        "stay_id",
        "person_id",
        "episode_definition",
        "signal_resolution",
        "definition_role",
        "episode_exposed",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
        "age",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
        "endpoint",
        "endpoint_family",
        "lag_window",
        "lag_role",
        "outcome_variant",
        "source_available",
        "at_risk",
        "baseline_observed",
        "post_observed",
        "outcome_observed",
        "baseline_value",
        "post_value",
        "change_value",
        "event",
        "event_offset_minutes",
        "measurement_concept",
        "change_scale",
        "component_count_available",
        "component_count_eligible",
        "component_count_observed",
        "component_count_event",
        "urine_baseline_coverage_fraction",
        "urine_coverage_fraction",
        "urine_covered_2h_bins",
        "analysis_status",
        "causal_claim_permitted",
    )
    for definition_key, anchors in anchors_by_definition.items():
        primary_definition = definition_key == PRIMARY_DEFINITION
        windows = (
            tuple(label for label, *_ in LAG_WINDOWS)
            if primary_definition
            else SENSITIVITY_WINDOWS
        )
        for lag_window in windows:
            frames = _add_composite_frames(
                _component_record_frames(
                    anchors, lookup, lag_window=lag_window
                )
            )
            for endpoint, endpoint_frame in frames.items():
                if endpoint_frame.empty:
                    continue
                family = str(endpoint_frame["endpoint_family"].iloc[0])
                if (
                    not primary_definition
                    and lag_window not in FOCUSED_WINDOWS_BY_FAMILY[family]
                ):
                    continue
                primary_variant_name = PRIMARY_VARIANT_BY_ENDPOINT[endpoint]
                variants = (
                    endpoint_frame.groupby("outcome_variant", sort=False)
                    if primary_definition
                    else [
                        (
                            primary_variant_name,
                            _select_primary_variant(endpoint_frame, endpoint),
                        )
                    ]
                )
                for variant, variant_frame in variants:
                    is_primary_variant = str(variant) == primary_variant_name
                    effect_rows.extend(
                        _summarize_record_frame(
                            variant_frame,
                            primary_variant=is_primary_variant,
                            primary_definition=primary_definition,
                            bootstrap_repetitions=bootstrap_repetitions,
                        )
                    )
                if not primary_definition:
                    continue
                primary_frame = _select_primary_variant(endpoint_frame, endpoint)
                observation_rows.extend(_observation_process_rows(primary_frame))
                if lag_window not in FOCUSED_WINDOWS_BY_FAMILY[family]:
                    continue
                stored_records.append(primary_frame.reindex(columns=record_columns))
                for _, dataset_frame in primary_frame.groupby("dataset", sort=False):
                    weighted_rows.append(_selection_weighted_row(dataset_frame))

    effects = _apply_effect_multiplicity(pd.DataFrame(effect_rows))
    observation = _apply_observation_multiplicity(
        pd.DataFrame(observation_rows)
    )
    weighted = _apply_weighted_multiplicity(pd.DataFrame(weighted_rows))
    records = (
        pd.concat(stored_records, ignore_index=True, sort=False)
        if stored_records
        else pd.DataFrame(columns=record_columns)
    )
    meta = _meta_effect_rows(effects)
    evidence = _build_evidence_summary(effects, weighted, meta)
    key_results = _build_key_results(effects, weighted, meta)
    return {
        "multiorgan_episode_records": records,
        "multiorgan_episode_effects": effects,
        "multiorgan_episode_observation_process": observation,
        "multiorgan_episode_measurement_weighted": weighted,
        "multiorgan_episode_meta_analysis": meta,
        "multiorgan_episode_evidence_summary": evidence,
        "multiorgan_episode_key_results": key_results,
        "multiorgan_episode_endpoint_definitions": _endpoint_definitions_table(),
        "multiorgan_episode_availability": _availability_table(records),
    }


__all__ = [
    "ANALYZED_DEFINITIONS",
    "COMPOSITE_SPECS",
    "FOCUSED_WINDOWS_BY_FAMILY",
    "LAG_WINDOWS",
    "POST_HOC_STATUS",
    "rebuild_multiorgan_weighted_sensitivity",
    "run_multiorgan_episode_analyses",
]
