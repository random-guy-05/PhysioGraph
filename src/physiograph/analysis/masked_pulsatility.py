"""Outcome-blind construction and support gates for masked pulsatility."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

PRPP_THRESHOLD = 0.25
OBSERVATION_END_MINUTES = 240.0
LACTATE_LOOKBACK_MINUTES = 240.0
MINIMUM_GATE = {
    "crossings": 100,
    "controls": 200,
    "crossings_followup_6h": 75,
    "controls_followup_6h": 150,
}
SOURCE_PRIORITY = {"invasive_arterial": 0, "noninvasive_cuff": 1, "unknown": 2}
REPAIR_SOURCE_PRIORITY = {
    "nibp_cuff_union": 0,
    "nibp_vitalAperiodic": 1,
    "nibp_nurseCharting": 2,
    "ibp_vitalPeriodic": 3,
    "other_validated_source": 4,
}
SUPPORT_BIN_EDGES = [0.0, 60.0, 120.0, 240.0, 360.0, 720.0]
SUPPORT_BIN_LABELS = ["0-1h", "1-2h", "2-4h", "4-6h", "6-12h"]
MIMIC_SUPPORT_DRUGS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221662: "dopamine",
    221653: "dobutamine",
    221749: "phenylephrine",
    222315: "vasopressin",
    221986: "milrinone",
}
EICU_SUPPORT_ALIASES = {
    "norepinephrine": ("norepinephrine", "noradrenaline", "levophed"),
    "epinephrine": ("epinephrine", "epinepherine", "adrenalin"),
    "dopamine": ("dopamine",),
    "dobutamine": ("dobutamine", "dobutrex"),
    "milrinone": ("milrinone", "primacor", "primacore"),
    "vasopressin": ("vasopressin", "pitressin"),
    "phenylephrine": (
        "phenylephrine",
        "neo-synephrine",
        "neosynephrine",
        "neosynsprine",
    ),
}


def proportional_pulse_pressure(sbp: Any, dbp: Any) -> float:
    systolic = pd.to_numeric(pd.Series([sbp]), errors="coerce").iloc[0]
    diastolic = pd.to_numeric(pd.Series([dbp]), errors="coerce").iloc[0]
    if pd.isna(systolic) or pd.isna(diastolic) or float(systolic) <= 0:
        return math.nan
    return (float(systolic) - float(diastolic)) / float(systolic)


def valid_bp_triplet(sbp: Any, dbp: Any, map_value: Any) -> bool:
    values = pd.to_numeric(pd.Series([sbp, dbp, map_value]), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        return False
    systolic, diastolic, mean = map(float, values)
    return (
        50.0 <= systolic <= 300.0
        and 20.0 <= diastolic <= 200.0
        and 20.0 <= mean <= 200.0
        and systolic > diastolic
        and diastolic <= mean <= systolic
    )


def valid_bp_pair(sbp: Any, dbp: Any) -> bool:
    values = pd.to_numeric(pd.Series([sbp, dbp]), errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        return False
    systolic, diastolic = map(float, values)
    return 50.0 <= systolic <= 300.0 and 20.0 <= diastolic <= 200.0 and systolic > diastolic


def derived_mean_arterial_pressure(sbp: Any, dbp: Any) -> float:
    if not valid_bp_pair(sbp, dbp):
        return math.nan
    return (float(sbp) + 2.0 * float(dbp)) / 3.0


def deduplicate_cuff_pairs(
    aperiodic: pd.DataFrame, nurse: pd.DataFrame, *, near_minutes: float = 1.0
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Prefer aperiodic cuff rows over identical nurse rows within one minute."""

    required = {"stay_id", "timestamp_min", "sbp", "dbp", "map"}
    if required - set(aperiodic) or required - set(nurse):
        raise ValueError("Cuff deduplication inputs lack required columns")
    left = aperiodic.reset_index(drop=True).reset_index(names="_aperiodic_index")
    right = nurse.reset_index(drop=True).reset_index(names="_nurse_index")
    candidates = right.merge(
        left,
        on=["stay_id", "sbp", "dbp"],
        how="inner",
        suffixes=("_nurse", "_aperiodic"),
    )
    if candidates.empty:
        combined = pd.concat([aperiodic, nurse], ignore_index=True)
        return combined, {
            "exact_duplicates": 0,
            "near_time_duplicates": 0,
            "value_identical_cross_source": 0,
            "nurse_rows_dropped": 0,
            "aperiodic_rows": len(aperiodic),
            "aperiodic_source_exclusive_rows": len(aperiodic),
            "nurse_source_exclusive_rows": len(nurse),
            "map_enriched_from_nurse": 0,
        }
    candidates["_delta"] = (
        candidates["timestamp_min_nurse"] - candidates["timestamp_min_aperiodic"]
    ).abs()
    nurse_map = pd.to_numeric(candidates["map_nurse"], errors="coerce")
    aperiodic_map = pd.to_numeric(candidates["map_aperiodic"], errors="coerce")
    candidates["_map_compatible"] = (
        nurse_map.isna()
        | aperiodic_map.isna()
        | np.isclose(nurse_map, aperiodic_map, equal_nan=True)
    )
    exact = candidates.loc[
        candidates["_delta"].eq(0)
        & np.isclose(nurse_map, aperiodic_map, equal_nan=True)
    ]
    near = candidates.loc[
        candidates["_delta"].le(float(near_minutes)) & candidates["_map_compatible"]
    ].copy()
    matched = (
        near.sort_values(["_nurse_index", "_delta", "_aperiodic_index"])
        .drop_duplicates("_nurse_index")
        .copy()
    )
    enriched = left.copy()
    enriched_count = 0
    for _, row in matched.iterrows():
        target = int(row["_aperiodic_index"])
        if pd.isna(enriched.loc[target, "map"]) and pd.notna(row["map_nurse"]):
            enriched.loc[target, "map"] = row["map_nurse"]
            enriched_count += 1
    kept_nurse = right.loc[~right["_nurse_index"].isin(matched["_nurse_index"])].copy()
    combined = pd.concat(
        [
            enriched.drop(columns="_aperiodic_index"),
            kept_nurse.drop(columns="_nurse_index"),
        ],
        ignore_index=True,
    )
    return combined, {
        "exact_duplicates": int(exact["_nurse_index"].nunique()),
        "near_time_duplicates": int(matched["_nurse_index"].nunique()),
        "value_identical_cross_source": int(candidates["_nurse_index"].nunique()),
        "nurse_rows_dropped": int(matched["_nurse_index"].nunique()),
        "aperiodic_rows": len(aperiodic),
        "aperiodic_source_exclusive_rows": int(
            len(aperiodic) - matched["_aperiodic_index"].nunique()
        ),
        "nurse_source_exclusive_rows": len(kept_nurse),
        "map_enriched_from_nurse": int(enriched_count),
    }


def build_source_hourly_bp_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "stay_id", "timestamp_min", "bp_source", "sbp", "dbp", "map"}
    missing = required - set(pairs)
    if missing:
        raise ValueError(f"BP pairs lack columns: {sorted(missing)}")
    local = pairs.copy()
    for column in ("timestamp_min", "sbp", "dbp", "map"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    valid_pairs = pd.Series(
        [valid_bp_pair(*values) for values in local[["sbp", "dbp"]].to_numpy()],
        index=local.index,
    )
    local = local.loc[
        local["timestamp_min"].ge(0)
        & local["timestamp_min"].lt(OBSERVATION_END_MINUTES)
        & valid_pairs
    ].copy()
    if local.empty:
        return pd.DataFrame()
    local["hour"] = np.floor(local["timestamp_min"] / 60.0).astype(int)
    local["derived_map"] = (local["sbp"] + 2.0 * local["dbp"]) / 3.0
    hourly = (
        local.groupby(["dataset", "stay_id", "bp_source", "hour"], as_index=False)
        .agg(
            anchor_minute=("timestamp_min", "max"),
            sbp=("sbp", "median"),
            dbp=("dbp", "median"),
            map=("map", "median"),
            derived_map=("derived_map", "median"),
            measurement_count=("timestamp_min", "size"),
        )
        .sort_values(["dataset", "stay_id", "bp_source", "hour"])
    )
    hourly["pp"] = hourly["sbp"] - hourly["dbp"]
    hourly["prpp"] = hourly["pp"] / hourly["sbp"]
    return hourly.reset_index(drop=True)


def detect_source_consistent_crossings(hourly: pd.DataFrame) -> tuple[pd.DataFrame, set[tuple[Any, ...]]]:
    rows: list[pd.Series] = []
    initially_low: set[tuple[Any, ...]] = set()
    for keys, group in hourly.groupby(["dataset", "stay_id", "bp_source"], sort=False):
        ordered = group.sort_values("hour")
        if float(ordered.iloc[0]["prpp"]) < PRPP_THRESHOLD:
            initially_low.add(tuple(keys))
            continue
        low = ordered.loc[ordered["prpp"].lt(PRPP_THRESHOLD)]
        if not low.empty:
            rows.append(low.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True), initially_low


def hospital_contribution_audit(crossings: pd.DataFrame) -> dict[str, float | int]:
    if crossings.empty:
        return {
            "hospital_n": 0,
            "hospitals_ge5_crossings": 0,
            "median_crossings_per_hospital": 0.0,
            "maximum_hospital_contribution": 0,
            "top1_crossing_pct": 0.0,
            "top5_crossing_pct": 0.0,
            "top10_crossing_pct": 0.0,
        }
    counts = crossings.groupby("hospital_id").size().sort_values(ascending=False)
    total = float(counts.sum())
    return {
        "hospital_n": len(counts),
        "hospitals_ge5_crossings": int(counts.ge(5).sum()),
        "median_crossings_per_hospital": float(counts.median()),
        "maximum_hospital_contribution": int(counts.max()),
        "top1_crossing_pct": float(100.0 * counts.head(1).sum() / total),
        "top5_crossing_pct": float(100.0 * counts.head(5).sum() / total),
        "top10_crossing_pct": float(100.0 * counts.head(10).sum() / total),
    }


def source_repair_feasibility(
    mimic_crossings: int,
    eicu_crossings: int,
    mimic_controls: int,
    eicu_controls: int,
    eicu_crossing_hospitals: int,
    *,
    severe_center_concentration: bool,
) -> tuple[str, str]:
    if min(mimic_crossings, eicu_crossings) < 75 or severe_center_concentration:
        return "fail", "stop_insufficient_routine_bp_support"
    if (
        min(mimic_crossings, eicu_crossings) >= 300
        and min(mimic_controls, eicu_controls) >= 500
        and eicu_crossing_hospitals >= 10
    ):
        return "strong", "proceed_to_protocol_freeze"
    if (
        min(mimic_crossings, eicu_crossings) >= 150
        and min(mimic_controls, eicu_controls) >= 300
        and eicu_crossing_hospitals >= 10
    ):
        return "acceptable", "proceed_to_protocol_freeze"
    return "weak", "stop_insufficient_routine_bp_support"


def build_hourly_bp_series(triplets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate same-time/source triplets and select one source per stay."""

    required = {"dataset", "stay_id", "event_minute", "source", "sbp", "dbp", "map"}
    missing = required - set(triplets)
    if missing:
        raise ValueError(f"BP triplets lack columns: {sorted(missing)}")
    local = triplets.copy()
    for column in ("event_minute", "sbp", "dbp", "map"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    local = local.loc[
        local["event_minute"].ge(0)
        & local["event_minute"].lt(OBSERVATION_END_MINUTES)
    ].copy()
    local = local.loc[
        [valid_bp_triplet(*values) for values in local[["sbp", "dbp", "map"]].to_numpy()]
    ].copy()
    if local.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "stay_id",
                "source",
                "hour",
                "anchor_minute",
                "sbp",
                "dbp",
                "map",
                "measurement_count",
                "pp",
                "prpp",
            ]
        )
    local["hour"] = np.floor(local["event_minute"] / 60.0).astype(int)
    hourly = (
        local.groupby(["dataset", "stay_id", "source", "hour"], as_index=False)
        .agg(
            anchor_minute=("event_minute", "max"),
            sbp=("sbp", "median"),
            dbp=("dbp", "median"),
            map=("map", "median"),
            measurement_count=("event_minute", "size"),
        )
        .sort_values(["dataset", "stay_id", "source", "hour"])
    )
    source_counts = (
        hourly.groupby(["dataset", "stay_id", "source"], as_index=False)
        .size()
        .rename(columns={"size": "valid_bins"})
    )
    source_counts["source_priority"] = source_counts["source"].map(SOURCE_PRIORITY).fillna(99)
    chosen = (
        source_counts.sort_values(
            ["dataset", "stay_id", "valid_bins", "source_priority", "source"],
            ascending=[True, True, False, True, True],
        )
        .drop_duplicates(["dataset", "stay_id"])
        [["dataset", "stay_id", "source"]]
    )
    hourly = hourly.merge(chosen, on=["dataset", "stay_id", "source"], how="inner")
    hourly["pp"] = hourly["sbp"] - hourly["dbp"]
    hourly["prpp"] = hourly["pp"] / hourly["sbp"]
    return hourly.sort_values(["dataset", "stay_id", "hour"]).reset_index(drop=True)


def detect_incident_crossings(hourly: pd.DataFrame) -> tuple[pd.DataFrame, set[Any]]:
    """Return first high-to-low crossing and stays initially below threshold."""

    rows: list[pd.Series] = []
    initially_low: set[Any] = set()
    for _, group in hourly.groupby(["dataset", "stay_id"], sort=False):
        ordered = group.sort_values("hour")
        if ordered.empty:
            continue
        if float(ordered.iloc[0]["prpp"]) < PRPP_THRESHOLD:
            initially_low.add(ordered.iloc[0]["stay_id"])
            continue
        low = ordered.loc[ordered["prpp"].lt(PRPP_THRESHOLD)]
        if not low.empty:
            rows.append(low.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True), initially_low


def latest_preanchor_lactate(
    measurements: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Select only lactates collected and available within the fixed lookback."""

    keys = ["dataset", "stay_id", "candidate_id", "anchor_minute"]
    required_measurements = {"dataset", "stay_id", "event_minute", "available_minute", "value"}
    required_candidates = set(keys)
    if required_measurements - set(measurements) or required_candidates - set(candidates):
        raise ValueError("Lactate selection inputs lack required columns")
    measurement_keys = ["dataset", "stay_id"]
    if "candidate_id" in measurements:
        measurement_keys.append("candidate_id")
    measurement_values = measurements.drop(columns=["anchor_minute"], errors="ignore")
    merged = candidates[keys].merge(measurement_values, on=measurement_keys, how="left")
    eligible = merged.loc[
        merged["event_minute"].ge(merged["anchor_minute"] - LACTATE_LOOKBACK_MINUTES)
        & merged["event_minute"].le(merged["anchor_minute"])
        & merged["available_minute"].le(merged["anchor_minute"])
        & pd.to_numeric(merged["value"], errors="coerce").gt(0)
        & pd.to_numeric(merged["value"], errors="coerce").le(30)
    ].copy()
    if eligible.empty:
        return candidates[keys].assign(
            baseline_lactate=np.nan,
            lactate_event_minute=np.nan,
            lactate_available_minute=np.nan,
            lactate_to_anchor_lag=np.nan,
        )
    same_time = (
        eligible.groupby([*keys, "event_minute"], as_index=False)
        .agg(value=("value", "median"), available_minute=("available_minute", "max"))
        .sort_values(["candidate_id", "event_minute", "available_minute"])
        .drop_duplicates("candidate_id", keep="last")
        .rename(
            columns={
                "value": "baseline_lactate",
                "event_minute": "lactate_event_minute",
                "available_minute": "lactate_available_minute",
            }
        )
    )
    result = candidates[keys].merge(same_time, on=keys, how="left", validate="one_to_one")
    result["lactate_to_anchor_lag"] = result["anchor_minute"] - result["lactate_event_minute"]
    return result


def intervention_free(anchor_minute: float, first_pressor: Any, first_mcs: Any) -> bool:
    pressor = pd.to_numeric(pd.Series([first_pressor]), errors="coerce").iloc[0]
    mcs = pd.to_numeric(pd.Series([first_mcs]), errors="coerce").iloc[0]
    return not (
        (pd.notna(pressor) and float(pressor) <= float(anchor_minute))
        or (pd.notna(mcs) and float(mcs) <= float(anchor_minute))
    )


def assign_pseudo_anchors(
    exposed: pd.DataFrame, eligible_control_bins: pd.DataFrame
) -> pd.DataFrame:
    """Match controls to exposed anchor-time quantiles without outcomes."""

    if exposed.empty or eligible_control_bins.empty:
        return eligible_control_bins.iloc[0:0].copy()
    rows: list[pd.Series] = []
    for dataset, candidates in eligible_control_bins.groupby("dataset", sort=True):
        times = np.sort(
            pd.to_numeric(
                exposed.loc[exposed["dataset"].astype(str).eq(str(dataset)), "anchor_minute"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(float)
        )
        if not len(times):
            continue
        stays = sorted(candidates["stay_id"].unique(), key=str)
        targets = np.quantile(times, (np.arange(len(stays)) + 0.5) / len(stays))
        for stay_id, target in zip(stays, targets):
            group = candidates.loc[candidates["stay_id"].eq(stay_id)].copy()
            group["_distance"] = (group["anchor_minute"] - float(target)).abs()
            picked = group.sort_values(["_distance", "anchor_minute", "candidate_id"]).iloc[0].copy()
            picked["target_anchor_minute"] = float(target)
            picked["control_anchor_strategy"] = "nearest_eligible_hour_to_exposed_anchor_time_quantile"
            rows.append(picked)
    return pd.DataFrame(rows).drop(columns="_distance", errors="ignore").reset_index(drop=True)


def followup_observation_flags(times: list[float] | np.ndarray, anchor_minute: float) -> tuple[bool, bool]:
    values = np.asarray(times, dtype=float) - float(anchor_minute)
    return bool(((values > 0) & (values <= 360)).any()), bool(
        ((values > 0) & (values <= 720)).any()
    )


def feasibility_gate(row: dict[str, Any]) -> bool:
    return all(int(row.get(key, 0)) >= threshold for key, threshold in MINIMUM_GATE.items())


def phase_a_decision(rows: list[dict[str, Any]], protocol_sha256: str, environment: str) -> dict[str, Any]:
    support = {str(row["dataset"]): feasibility_gate(row) for row in rows}
    passed = len(support) >= 2 and all(support.values())
    return {
        "feasibility_passed": passed,
        "protocol_frozen_before_outcomes": False,
        "mimic_supported": bool(support.get("mimic", False)),
        "eicu_supported": bool(support.get("eicu", False)),
        "mimic_primary_positive": False,
        "eicu_primary_positive": False,
        "cross_database_replication": False,
        "observation_robust": False,
        "sbp_map_independent_information": False,
        "meaningful_lead_time": False,
        "clinically_large_effect": False,
        "biological_claim_ready": False,
        "masked_pulsatility_feasible": passed,
        "recommended_action": "freeze_phase_b_protocol" if passed else "stop_no_viable_signal",
        "phase_reached": "phase_a_feasibility",
        "post_anchor_lactate_values_inspected": False,
        "protocol_sha256": protocol_sha256,
        "execution_environment": environment,
        "support": rows,
    }


def support_lead_bin(lead_minutes: Any) -> str | None:
    """Assign one of the five frozen support lead-time bins."""

    lead = pd.to_numeric(pd.Series([lead_minutes]), errors="coerce").iloc[0]
    if pd.isna(lead) or not 0.0 < float(lead) <= 720.0:
        return None
    for lower, upper, label in zip(
        SUPPORT_BIN_EDGES[:-1], SUPPORT_BIN_EDGES[1:], SUPPORT_BIN_LABELS
    ):
        if lower < float(lead) <= upper:
            return label
    return None


def canonical_eicu_support_drug(drug_name: Any) -> str | None:
    """Map eICU infusion text with the exact frozen alias set."""

    text = " ".join(str(drug_name).strip().lower().replace("_", " ").split())
    for canonical, aliases in EICU_SUPPORT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return None


def protocol_precedes_outcomes(freeze_utc: str, first_outcome_read_utc: str) -> bool:
    """Verify strict protocol-before-outcome temporal ordering."""

    freeze = pd.Timestamp(freeze_utc)
    first_read = pd.Timestamp(first_outcome_read_utc)
    return bool(freeze.tzinfo is not None and first_read.tzinfo is not None and freeze < first_read)


def audit_future_control_crossings(
    controls: pd.DataFrame, crossings: pd.DataFrame, *, window_end_minute: float = 240.0
) -> pd.DataFrame:
    """Identify control stays with a known later crossing in the locked window."""

    future = controls[["dataset", "stay_id", "candidate_id", "anchor_minute"]].merge(
        crossings[["dataset", "stay_id", "anchor_minute"]].rename(
            columns={"anchor_minute": "crossing_minute"}
        ),
        on=["dataset", "stay_id"],
        how="left",
        validate="one_to_many",
    )
    future = future.loc[
        future["crossing_minute"].gt(future["anchor_minute"])
        & future["crossing_minute"].lt(window_end_minute)
    ]
    return future.sort_values(["dataset", "stay_id", "crossing_minute"]).reset_index(drop=True)


def build_two_hour_landmark(
    anchors: pd.DataFrame, support_starts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach first support starts and apply the frozen two-hour landmark.

    ``support_starts`` contains absolute ICU-relative start minutes.  Only the
    first strictly post-anchor start is used.  Starts at exactly 120 minutes
    after anchor remain in the concurrent/blanking interval.
    """

    required = {
        "dataset",
        "stay_id",
        "candidate_id",
        "anchor_minute",
        "followup_end_offset_minutes",
        "death_offset_minutes",
    }
    missing = required - set(anchors)
    if missing:
        raise ValueError(f"Anchor rows lack columns: {sorted(missing)}")
    local = anchors.copy()
    starts = support_starts.copy()
    if starts.empty:
        starts = pd.DataFrame(columns=["dataset", "stay_id", "start_minute", "drug"])
    starts["start_minute"] = pd.to_numeric(starts["start_minute"], errors="coerce")
    joined = starts.merge(
        local[["dataset", "stay_id", "candidate_id", "anchor_minute"]],
        on=["dataset", "stay_id"],
        how="inner",
        validate="many_to_one",
    )
    joined["lead_minutes"] = joined["start_minute"] - joined["anchor_minute"]
    post = joined.loc[joined["lead_minutes"].gt(0)].sort_values(
        ["candidate_id", "lead_minutes", "drug"], na_position="last"
    )
    first = post.drop_duplicates("candidate_id", keep="first")
    local = local.merge(
        first[["candidate_id", "start_minute", "lead_minutes", "drug"]],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    local["support_lead_bin"] = local["lead_minutes"].map(support_lead_bin)
    local["blanking_support"] = local["lead_minutes"].gt(0) & local[
        "lead_minutes"
    ].le(120)
    local["delayed_support_2_12h"] = local["lead_minutes"].gt(120) & local[
        "lead_minutes"
    ].le(720)
    local["delayed_support_2_6h"] = local["lead_minutes"].gt(120) & local[
        "lead_minutes"
    ].le(360)
    landmark = local["anchor_minute"] + 120.0
    local["alive_observed_at_2h"] = local["followup_end_offset_minutes"].ge(
        landmark
    ) & (
        local["death_offset_minutes"].isna()
        | local["death_offset_minutes"].gt(landmark)
    )
    local["landmark_eligible"] = (
        local["alive_observed_at_2h"] & ~local["blanking_support"]
    )
    flow = (
        local.groupby(["dataset", "anchor_type"], dropna=False)
        .agg(
            pre_landmark_n=("candidate_id", "size"),
            blanking_support_n=("blanking_support", "sum"),
            not_alive_observed_2h_n=("alive_observed_at_2h", lambda x: int((~x).sum())),
            landmark_n=("landmark_eligible", "sum"),
            primary_events=("delayed_support_2_12h", lambda x: int(x[local.loc[x.index, "landmark_eligible"]].sum())),
        )
        .reset_index()
    )
    return local, flow


def preanchor_linear_slope(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    value_column: str = "value",
    lookback_minutes: float = 120.0,
) -> pd.DataFrame:
    """Compute an OLS slope per hour using only values at/before anchor."""

    merged = events.merge(
        anchors[["dataset", "stay_id", "candidate_id", "anchor_minute"]],
        on=["dataset", "stay_id"],
        how="inner",
        validate="many_to_many",
    )
    merged["event_minute"] = pd.to_numeric(merged["event_minute"], errors="coerce")
    merged[value_column] = pd.to_numeric(merged[value_column], errors="coerce")
    merged = merged.loc[
        merged["event_minute"].le(merged["anchor_minute"])
        & merged["event_minute"].ge(merged["anchor_minute"] - lookback_minutes)
    ].dropna(subset=["event_minute", value_column])
    rows: list[dict[str, Any]] = []
    for candidate_id, group in merged.groupby("candidate_id", sort=False):
        collapsed = group.groupby("event_minute", as_index=False)[value_column].median()
        if collapsed["event_minute"].nunique() < 2:
            slope = math.nan
            count = len(collapsed)
        else:
            x = (collapsed["event_minute"] - collapsed["event_minute"].max()) / 60.0
            slope = float(np.polyfit(x.to_numpy(float), collapsed[value_column], 1)[0])
            count = len(collapsed)
        rows.append({"candidate_id": candidate_id, "slope_per_hour": slope, "slope_points": count})
    return pd.DataFrame(rows)


def latest_preanchor_measurement(
    events: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    lookback_minutes: float,
    availability_column: str | None = None,
) -> pd.DataFrame:
    """Select the latest strictly pre-anchor-safe measurement per anchor."""

    merged = events.merge(
        anchors[["dataset", "stay_id", "candidate_id", "anchor_minute"]],
        on=["dataset", "stay_id"],
        how="inner",
        validate="many_to_many",
    )
    merged["event_minute"] = pd.to_numeric(merged["event_minute"], errors="coerce")
    valid = merged["event_minute"].le(merged["anchor_minute"]) & merged[
        "event_minute"
    ].ge(merged["anchor_minute"] - lookback_minutes)
    if availability_column:
        merged[availability_column] = pd.to_numeric(
            merged[availability_column], errors="coerce"
        )
        valid &= merged[availability_column].le(merged["anchor_minute"])
    selected = merged.loc[valid].sort_values(
        ["candidate_id", "event_minute"], ascending=[True, False]
    )
    return selected.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)


def merge_continuous_infusion_intervals(
    rows: pd.DataFrame, *, restart_grace_minutes: float = 5.0
) -> pd.DataFrame:
    """Collapse contiguous positive-rate MIMIC infusion rows into episodes."""

    required = {"dataset", "stay_id", "drug", "start_minute", "end_minute", "rate"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Infusion rows lack columns: {sorted(missing)}")
    local = rows.copy()
    for column in ("start_minute", "end_minute", "rate"):
        local[column] = pd.to_numeric(local[column], errors="coerce")
    local = local.loc[
        local["rate"].gt(0)
        & local["start_minute"].notna()
        & local["end_minute"].gt(local["start_minute"])
    ].sort_values(["dataset", "stay_id", "drug", "start_minute", "end_minute"])
    episodes: list[dict[str, Any]] = []
    for keys, group in local.groupby(["dataset", "stay_id", "drug"], sort=False):
        episode_start: float | None = None
        episode_end: float | None = None
        rows_merged = 0
        for row in group.itertuples(index=False):
            start, end = float(row.start_minute), float(row.end_minute)
            if episode_start is None or start > float(episode_end) + restart_grace_minutes:
                if episode_start is not None:
                    episodes.append(
                        {"dataset": keys[0], "stay_id": keys[1], "drug": keys[2], "start_minute": episode_start, "end_minute": episode_end, "rows_merged": rows_merged}
                    )
                episode_start, episode_end, rows_merged = start, end, 1
            else:
                episode_end = max(float(episode_end), end)
                rows_merged += 1
        if episode_start is not None:
            episodes.append(
                {"dataset": keys[0], "stay_id": keys[1], "drug": keys[2], "start_minute": episode_start, "end_minute": episode_end, "rows_merged": rows_merged}
            )
    return pd.DataFrame(episodes)


def prepare_frozen_covariates(
    frame: pd.DataFrame, covariates: list[str]
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Winsorize, median-impute, flag missingness, and standardize covariates."""

    local = frame.copy()
    diagnostics: list[dict[str, Any]] = []
    model_columns: list[str] = []
    for column in covariates:
        values = pd.to_numeric(local[column], errors="coerce")
        missing = values.isna()
        observed = values.dropna()
        lower = float(observed.quantile(0.01)) if len(observed) else 0.0
        upper = float(observed.quantile(0.99)) if len(observed) else 0.0
        median = float(observed.median()) if len(observed) else 0.0
        clipped = values.clip(lower, upper).fillna(median)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        name = f"z_{column}"
        local[name] = (clipped - mean) / std if std > 0 else 0.0
        if std > 0:
            model_columns.append(name)
        if missing.any() and missing.nunique() > 1:
            missing_name = f"missing_{column}"
            local[missing_name] = missing.astype(int)
            model_columns.append(missing_name)
        diagnostics.append(
            {"covariate": column, "n": len(values), "missing_n": int(missing.sum()), "winsor_low": lower, "winsor_high": upper, "imputation_median": median, "standardization_mean": mean, "standardization_sd": std}
        )
    return local, model_columns, pd.DataFrame(diagnostics)


def fit_modified_poisson(
    frame: pd.DataFrame,
    *,
    outcome: str,
    covariates: list[str],
    cluster_column: str | None = None,
    weights: str | None = None,
) -> tuple[dict[str, float | int | str], Any]:
    """Fit the frozen modified-Poisson model and standardized RR/RD."""

    columns = [outcome, "exposed", *covariates]
    if cluster_column:
        columns.append(cluster_column)
    if weights:
        columns.append(weights)
    local = frame[columns].dropna().copy()
    design = sm.add_constant(local[["exposed", *covariates]].astype(float), has_constant="add")
    kwargs: dict[str, Any] = {}
    if weights:
        kwargs["freq_weights"] = pd.to_numeric(local[weights], errors="coerce")
    model = sm.GLM(local[outcome].astype(int), design, family=sm.families.Poisson(), **kwargs)
    if cluster_column:
        fit = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": local[cluster_column].to_numpy(), "use_correction": True},
        )
    else:
        fit = model.fit(cov_type="HC0")
    beta = fit.params.to_numpy(float)
    covariance = np.asarray(fit.cov_params(), dtype=float)
    exposure_index = list(design.columns).index("exposed")
    log_rr = float(beta[exposure_index])
    se = float(math.sqrt(max(covariance[exposure_index, exposure_index], 0.0)))
    x0 = design.copy()
    x1 = design.copy()
    x0["exposed"] = 0.0
    x1["exposed"] = 1.0
    p0 = np.exp(x0.to_numpy(float) @ beta)
    p1 = np.exp(x1.to_numpy(float) @ beta)
    risk0, risk1 = float(p0.mean()), float(p1.mean())
    gradient0 = (p0[:, None] * x0.to_numpy(float)).mean(axis=0)
    gradient1 = (p1[:, None] * x1.to_numpy(float)).mean(axis=0)
    rd_gradient = gradient1 - gradient0
    rd = risk1 - risk0
    rd_se = float(math.sqrt(max(rd_gradient @ covariance @ rd_gradient, 0.0)))
    marginal_log_rr = math.log(risk1 / risk0)
    rr_gradient = gradient1 / risk1 - gradient0 / risk0
    marginal_se = float(math.sqrt(max(rr_gradient @ covariance @ rr_gradient, 0.0)))
    result: dict[str, float | int | str] = {
        "n": len(local),
        "events": int(local[outcome].sum()),
        "exposed_n": int(local["exposed"].sum()),
        "control_n": int((1 - local["exposed"]).sum()),
        "coefficient_rr": math.exp(log_rr),
        "coefficient_ci_low": math.exp(log_rr - 1.96 * se),
        "coefficient_ci_high": math.exp(log_rr + 1.96 * se),
        "adjusted_exposed_risk": risk1,
        "adjusted_control_risk": risk0,
        "adjusted_rd": rd,
        "adjusted_rd_ci_low": rd - 1.96 * rd_se,
        "adjusted_rd_ci_high": rd + 1.96 * rd_se,
        "adjusted_rr": math.exp(marginal_log_rr),
        "adjusted_rr_ci_low": math.exp(marginal_log_rr - 1.96 * marginal_se),
        "adjusted_rr_ci_high": math.exp(marginal_log_rr + 1.96 * marginal_se),
        "log_rr": log_rr,
        "log_rr_se": se,
        "aic": float(fit.aic),
        "covariance": "hospital_clustered" if cluster_column else "HC0",
    }
    return result, fit


def unadjusted_binary_risk(frame: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    """Return frozen unadjusted risks, RR/RD, and large-sample intervals."""

    table = frame.groupby("exposed")[outcome].agg(["sum", "count"])
    for group in (0, 1):
        if group not in table.index:
            raise ValueError("Both exposed and control groups are required")
    e1, n1 = float(table.loc[1, "sum"]), float(table.loc[1, "count"])
    e0, n0 = float(table.loc[0, "sum"]), float(table.loc[0, "count"])
    r1, r0 = e1 / n1, e0 / n0
    rd = r1 - r0
    rd_se = math.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    a, b = e1 + 0.5, e0 + 0.5
    rr = (a / (n1 + 1.0)) / (b / (n0 + 1.0))
    log_se = math.sqrt(1 / a - 1 / (n1 + 1.0) + 1 / b - 1 / (n0 + 1.0))
    return {
        "exposed_n": int(n1), "control_n": int(n0), "exposed_events": int(e1), "control_events": int(e0),
        "exposed_risk": r1, "control_risk": r0, "risk_difference": rd,
        "rd_ci_low": rd - 1.96 * rd_se, "rd_ci_high": rd + 1.96 * rd_se,
        "risk_ratio": rr, "rr_ci_low": math.exp(math.log(rr) - 1.96 * log_se), "rr_ci_high": math.exp(math.log(rr) + 1.96 * log_se),
    }


def overlap_weights(
    frame: pd.DataFrame, covariates: list[str]
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Fit the frozen exposure propensity and return overlap weights/SMDs."""

    design = sm.add_constant(frame[covariates].astype(float), has_constant="add")
    propensity = sm.GLM(frame["exposed"].astype(int), design, family=sm.families.Binomial()).fit().predict(design)
    propensity = pd.Series(np.clip(propensity, 0.001, 0.999), index=frame.index)
    weights = pd.Series(np.where(frame["exposed"].eq(1), 1 - propensity, propensity), index=frame.index)
    rows: list[dict[str, Any]] = []
    for column in covariates:
        values = frame[column].astype(float)
        exposed = frame["exposed"].eq(1)
        mean1, mean0 = values[exposed].mean(), values[~exposed].mean()
        pooled = math.sqrt((values[exposed].var(ddof=1) + values[~exposed].var(ddof=1)) / 2)
        before = (mean1 - mean0) / pooled if pooled > 0 else 0.0
        def weighted(
            group: pd.Series, current_values: pd.Series = values
        ) -> tuple[float, float]:
            x, w = (
                current_values[group].to_numpy(float),
                weights[group].to_numpy(float),
            )
            mean = float(np.average(x, weights=w))
            variance = float(np.average((x - mean) ** 2, weights=w))
            return mean, variance
        wm1, wv1 = weighted(exposed)
        wm0, wv0 = weighted(~exposed)
        denominator = math.sqrt((wv1 + wv0) / 2)
        after = (wm1 - wm0) / denominator if denominator > 0 else 0.0
        rows.append({"covariate": column, "smd_before": before, "smd_after": after})
    return propensity, weights, pd.DataFrame(rows)


def kish_effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(values.sum() ** 2 / np.square(values).sum()) if np.square(values).sum() else 0.0


def random_effects_meta(estimates: pd.DataFrame) -> dict[str, float | int]:
    """DerSimonian-Laird random-effects pooling of source log risk ratios."""

    y = estimates["log_rr"].to_numpy(float)
    variance = np.square(estimates["log_rr_se"].to_numpy(float))
    fixed_weights = 1.0 / variance
    fixed = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * np.square(y - fixed)))
    df = len(y) - 1
    c = float(np.sum(fixed_weights) - np.sum(np.square(fixed_weights)) / np.sum(fixed_weights))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    weights = 1.0 / (variance + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    se = math.sqrt(1.0 / np.sum(weights))
    i2 = max(0.0, 100.0 * (q - df) / q) if q > 0 else 0.0
    return {"database_n": len(y), "pooled_rr": math.exp(pooled), "ci_low": math.exp(pooled - 1.96 * se), "ci_high": math.exp(pooled + 1.96 * se), "q": q, "q_df": df, "tau2": tau2, "i2_percent": i2}


def delayed_support_classification(flags: dict[str, bool]) -> str:
    """Apply the frozen claim/failure-state precedence literally."""

    if flags.get("high_impact_claim_ready", False):
        return "CLAIM_READY_REPLICATED_SIGNAL"
    if flags.get("mimic_significant_positive", False) and not flags.get("eicu_confirmatory_positive", False):
        return "DISCOVERY_ONLY_NO_EXTERNAL_REPLICATION"
    if flags.get("eicu_concurrent_positive", False) and not flags.get("primary_delayed_support_positive", False):
        return "CONCURRENT_DETERIORATION_ONLY"
    if flags.get("eicu_unadjusted_delayed_positive", False) and not flags.get("effect_survives_sbp_map_adjustment", False):
        return "CONVENTIONAL_BP_EXPLAINS_SIGNAL"
    if flags.get("effect_survives_sbp_map_adjustment", False) and not flags.get("dynamic_beats_static", False):
        return "STATIC_PRPP_EXPLAINS_SIGNAL"
    return "NO_SIGNAL"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "EICU_SUPPORT_ALIASES",
    "LACTATE_LOOKBACK_MINUTES",
    "MIMIC_SUPPORT_DRUGS",
    "MINIMUM_GATE",
    "OBSERVATION_END_MINUTES",
    "PRPP_THRESHOLD",
    "REPAIR_SOURCE_PRIORITY",
    "SUPPORT_BIN_EDGES",
    "SUPPORT_BIN_LABELS",
    "assign_pseudo_anchors",
    "audit_future_control_crossings",
    "build_hourly_bp_series",
    "build_source_hourly_bp_pairs",
    "build_two_hour_landmark",
    "canonical_eicu_support_drug",
    "deduplicate_cuff_pairs",
    "delayed_support_classification",
    "derived_mean_arterial_pressure",
    "detect_incident_crossings",
    "detect_source_consistent_crossings",
    "feasibility_gate",
    "fit_modified_poisson",
    "followup_observation_flags",
    "hospital_contribution_audit",
    "intervention_free",
    "kish_effective_sample_size",
    "latest_preanchor_lactate",
    "latest_preanchor_measurement",
    "merge_continuous_infusion_intervals",
    "overlap_weights",
    "phase_a_decision",
    "preanchor_linear_slope",
    "prepare_frozen_covariates",
    "proportional_pulse_pressure",
    "protocol_precedes_outcomes",
    "random_effects_meta",
    "sha256_file",
    "source_repair_feasibility",
    "support_lead_bin",
    "unadjusted_binary_risk",
    "valid_bp_pair",
    "valid_bp_triplet",
]
