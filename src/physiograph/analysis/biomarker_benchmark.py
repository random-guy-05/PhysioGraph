"""Leakage-safe head-to-head biomarker benchmark for early decompensation.

All candidate predictors are restricted to ``[0, 240)`` minutes after ICU
admission and all outcomes retain the study's post-landmark 12/24-hour clock.
The benchmark uses identical outcome-observed rows, patient-grouped folds, and
preprocessing for SpO2, SBP, lactate, and a transparent EHR operationalization
of the CSWG-refined SCAI criteria in Kapur et al. (JACC 2022;
doi:10.1016/j.jacc.2022.04.049).

The SCAI variable is explicitly an EHR-derived severity stage, not clinician
adjudication.  OHCA is unavailable in the cached cross-database artifact and
eICU treatment timestamps are documentation proxies.  Component coverage and
stage lower-bound status are emitted so missing data cannot silently become
stage A.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .spo2_lactate_mechanistic import _bh_adjust
from .spo2_protocol import (
    ABSOLUTE_SPO2_FEATURES,
    PARSIMONIOUS_INSTABILITY_FEATURES,
    PRIMARY_ENDPOINTS,
    SAMPLING_ADJUSTMENT_FEATURES,
    SECONDARY_ENDPOINTS,
    _build_model_pipeline,
    _calibration,
    _ece,
    _group_series,
)


LANDMARK_MINUTES = 240.0
KAPUR_REFERENCE = (
    "Kapur NK et al. Criteria for Defining Stages of Cardiogenic Shock "
    "Severity. J Am Coll Cardiol. 2022;80(3):185-198. "
    "doi:10.1016/j.jacc.2022.04.049"
)
KAPUR_STAGE_VERSION = "kapur_cswg_2022_first_values_within_4h_ehr_operationalization_v1"
RANDOM_SEED = 20260902
MIN_ROWS = 200
MIN_EVENTS = 20
BOOTSTRAP_REPETITIONS = 300

ADDITIONAL_LACTATE_ENDPOINTS = (
    "lactate_rise_1_0_12h_flag",
    "lactate_rise_1_0_24h_flag",
    "lactate_max_rise_12h_flag",
    "lactate_max_rise_24h_flag",
    "lactate_cross_2_12h_flag",
    "lactate_cross_2_24h_flag",
)
BENCHMARK_ENDPOINTS = (
    *PRIMARY_ENDPOINTS,
    *SECONDARY_ENDPOINTS,
    *ADDITIONAL_LACTATE_ENDPOINTS,
)

NEUTRAL_CONTEXT_FEATURES = (
    "age",
    "is_male",
    "sex_unknown_flag",
    "shock_icd_flag",
)
SPO2_LEVEL_FEATURES = tuple(ABSOLUTE_SPO2_FEATURES)
SPO2_INSTABILITY_FEATURES = tuple(PARSIMONIOUS_INSTABILITY_FEATURES)
SPO2_SAMPLING_FEATURES = tuple(SAMPLING_ADJUSTMENT_FEATURES)
SBP_FEATURES = (
    "sbp_first_4h",
    "sbp_last_4h",
    "sbp_mean_4h",
    "sbp_min_4h",
    "sbp_sd_4h",
    "sbp_rmssd_4h",
    "sbp_slope_per_hr_4h",
    "sbp_below_90_fraction_4h",
    "sbp_count_4h",
)
LACTATE_FEATURES = (
    "lactate_first_4h",
    "lactate_last_4h",
    "lactate_max_4h",
    "lactate_delta_4h",
    "lactate_slope_per_hr_4h",
    "lactate_ge_2_fraction_4h",
    "lactate_count_4h",
)
KAPUR_SCAI_FEATURES = (
    "kapur_scai_stage_num_first4h",
    "kapur_scai_core_observed",
    "kapur_scai_full_component_observed",
)

DRUG_PATTERNS: dict[str, re.Pattern[str]] = {
    "norepinephrine": re.compile(r"norepinephrine|noradrenaline|levophed", re.I),
    "epinephrine": re.compile(r"(?<!nor)epinephrine|adrenalin", re.I),
    "phenylephrine": re.compile(r"phenylephrine|neosynephrine|neo-synephrine", re.I),
    "dopamine": re.compile(r"dopamine", re.I),
    "dobutamine": re.compile(r"dobutamine|dobutrex", re.I),
    "milrinone": re.compile(r"milrinone|primacor|primacore", re.I),
    "vasopressin": re.compile(r"vasopressin", re.I),
}
DEVICE_PATTERNS: dict[str, re.Pattern[str]] = {
    "iabp": re.compile(r"\biabp\b|intra\s*aortic balloon", re.I),
    "impella": re.compile(r"\bimpella\b", re.I),
    "ecmo": re.compile(r"\becmo\b|extracorporeal membrane", re.I),
    "lvad": re.compile(r"\blvad\b|left ventricular assist", re.I),
    "rvad": re.compile(r"\brvad\b|right ventricular assist", re.I),
    "bivad": re.compile(r"\bbivad\b|biventricular assist", re.I),
}


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame and frame[column].notna().any()]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _endpoint_tier(endpoint: str) -> str:
    if endpoint in PRIMARY_ENDPOINTS:
        return "primary"
    if endpoint in ADDITIONAL_LACTATE_ENDPOINTS:
        return "prespecified_lactate_sensitivity"
    return "secondary"


def _between(value: float, low: float, high: float) -> bool:
    return bool(np.isfinite(value) and low <= value <= high)


def assign_kapur_scai_stage(
    *,
    sbp: float = math.nan,
    map_value: float = math.nan,
    lactate: float = math.nan,
    alt: float = math.nan,
    ph: float = math.nan,
    drug_count: float = math.nan,
    device_count: float = math.nan,
    persistent_abnormality: bool = False,
    ohca: float = math.nan,
) -> tuple[str | None, float, str, bool]:
    """Assign the highest observed Kapur/CSWG stage and its evidence.

    Returns ``(stage, number, evidence, is_lower_bound)``.  Normal but
    incomplete physiology is never called stage A.  When severe evidence is
    observed despite missing components, the observed stage is retained and
    marked as a lower bound.
    """
    values = [sbp, map_value, lactate, alt, ph, drug_count, device_count, ohca]
    sbp, map_value, lactate, alt, ph, drug_count, device_count, ohca = [
        float(value) if value is not None and not pd.isna(value) else math.nan
        for value in values
    ]
    bp_observed = np.isfinite(sbp) or np.isfinite(map_value)
    perfusion_observed = np.isfinite(lactate) or np.isfinite(alt)
    treatment_observed = np.isfinite(drug_count) and np.isfinite(device_count)
    core_complete = bool(bp_observed and perfusion_observed and treatment_observed)
    stage_a_complete = bool(core_complete and np.isfinite(ph) and np.isfinite(ohca))

    severe_hypotension = bool(
        (np.isfinite(sbp) and sbp < 60.0)
        or (np.isfinite(map_value) and map_value < 50.0)
    )
    hypotension = bool(
        _between(sbp, 60.0, 90.0) or _between(map_value, 50.0, 65.0)
    )
    mild_hypoperfusion = bool(
        _between(lactate, 2.0, 5.0) or _between(alt, 200.0, 500.0)
    )
    worsened_hypoperfusion = bool(
        _between(lactate, 5.0, 10.0) or (np.isfinite(alt) and alt > 500.0)
    )
    severe_hypoperfusion = bool(
        (np.isfinite(lactate) and lactate > 10.0)
        or (np.isfinite(ph) and ph <= 7.2)
    )
    hypoperfusion = bool(
        mild_hypoperfusion or worsened_hypoperfusion or severe_hypoperfusion
    )
    treatment_total = (
        int(drug_count) + int(device_count) if treatment_observed else None
    )

    stage: str | None
    evidence: str
    if (
        severe_hypotension
        or severe_hypoperfusion
        or (np.isfinite(drug_count) and drug_count > 3)
        or (np.isfinite(device_count) and device_count >= 3)
        or (np.isfinite(ohca) and ohca > 0)
    ):
        stage, evidence = "E", "extremis_threshold_or_treatment_intensity_or_ohca"
    elif hypotension and worsened_hypoperfusion:
        stage, evidence = "D", "hypotension_and_worsened_hypoperfusion"
    elif treatment_total is not None and treatment_total >= 2:
        stage, evidence = "D", "two_or_more_drugs_or_devices_below_stage_E_threshold"
    elif treatment_total == 1 and persistent_abnormality:
        stage, evidence = "D", "one_therapy_with_4h_persistent_abnormality_proxy"
    elif hypotension and hypoperfusion:
        stage, evidence = "C", "hypotension_and_hypoperfusion"
    elif treatment_total == 1:
        stage, evidence = "C", "one_drug_or_device"
    elif hypotension or hypoperfusion:
        stage, evidence = "B", "isolated_hypotension_or_hypoperfusion"
    elif stage_a_complete and treatment_total == 0:
        stage, evidence = "A", "core_components_observed_without_B_to_E_criteria"
    else:
        stage, evidence = None, "insufficient_components_for_stage_A"
    number = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0}.get(
        stage, math.nan
    )
    return stage, number, evidence, not stage_a_complete


def _aggregate_numeric_concept(
    events: pd.DataFrame,
    *,
    concept: str,
    low: float,
    high: float,
) -> pd.DataFrame:
    keys = ["dataset", "stay_id"]
    local = events.loc[events["concept"].eq(concept), [*keys, "offset_minutes", "value_numeric"]].copy()
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    local = local.loc[
        local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(LANDMARK_MINUTES)
        & local["value_numeric"].between(low, high, inclusive="both")
    ]
    if local.empty:
        return pd.DataFrame(columns=keys)
    # Multiple source rows at the same instant are reduced before dynamics so
    # duplicate invasive/noninvasive charting cannot inflate variability.
    local = (
        local.groupby([*keys, "offset_minutes"], as_index=False)["value_numeric"]
        .median()
        .sort_values([*keys, "offset_minutes"])
    )
    grouped = local.groupby(keys, sort=False)
    result = grouped.agg(
        first=("value_numeric", "first"),
        last=("value_numeric", "last"),
        mean=("value_numeric", "mean"),
        min=("value_numeric", "min"),
        max=("value_numeric", "max"),
        sd=("value_numeric", "std"),
        count=("value_numeric", "count"),
        first_offset=("offset_minutes", "first"),
        last_offset=("offset_minutes", "last"),
    ).reset_index()
    difference = grouped["value_numeric"].diff()
    rmssd = (
        difference.pow(2)
        .groupby([local["dataset"], local["stay_id"]])
        .mean()
        .pow(0.5)
        .rename("rmssd")
        .reset_index()
    )
    slope_values = local.assign(
        _x=local["offset_minutes"] / 60.0,
        _y=local["value_numeric"],
    )
    slope_values["_xy"] = slope_values["_x"] * slope_values["_y"]
    slope_values["_x2"] = slope_values["_x"] ** 2
    sums = slope_values.groupby(keys, sort=False).agg(
        _n=("_x", "count"),
        _sx=("_x", "sum"),
        _sy=("_y", "sum"),
        _sxy=("_xy", "sum"),
        _sx2=("_x2", "sum"),
    )
    denominator = sums["_n"] * sums["_sx2"] - sums["_sx"] ** 2
    sums["slope_per_hr"] = (
        sums["_n"] * sums["_sxy"] - sums["_sx"] * sums["_sy"]
    ) / denominator.where(denominator.abs().gt(1e-12))
    result = result.merge(rmssd, on=keys, how="left", validate="one_to_one")
    result = result.merge(
        sums[["slope_per_hr"]].reset_index(),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    return result.rename(
        columns={
            column: f"{concept}_{column}_4h"
            for column in result.columns
            if column not in keys
        }
    )


def _threshold_fraction(
    events: pd.DataFrame,
    *,
    concept: str,
    output: str,
    predicate: Any,
    low: float,
    high: float,
) -> pd.DataFrame:
    keys = ["dataset", "stay_id"]
    local = events.loc[events["concept"].eq(concept), [*keys, "offset_minutes", "value_numeric"]].copy()
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    local = local.loc[
        local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(LANDMARK_MINUTES)
        & local["value_numeric"].between(low, high, inclusive="both")
    ]
    if local.empty:
        return pd.DataFrame(columns=[*keys, output])
    local = local.groupby([*keys, "offset_minutes"], as_index=False)["value_numeric"].median()
    local["_threshold"] = predicate(local["value_numeric"]).astype(float)
    return local.groupby(keys, as_index=False)["_threshold"].mean().rename(
        columns={"_threshold": output}
    )


def _canonical_count(text: str, patterns: dict[str, re.Pattern[str]]) -> int:
    value = str(text or "")
    return int(sum(bool(pattern.search(value)) for pattern in patterns.values()))


def _landmark_treatment_counts(
    events: pd.DataFrame,
    analysis_frame: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["dataset", "stay_id"]
    result = analysis_frame.reindex(
        columns=[
            *keys,
            "pressor_source_available",
            "mcs_source_available",
            "baseline_vasoactive_flag",
            "baseline_mcs_flag",
        ]
    ).drop_duplicates(keys)
    local = events.reindex(
        columns=[*keys, "concept", "offset_minutes", "value_numeric", "raw_name", "value_text"]
    ).copy()
    local["offset_minutes"] = pd.to_numeric(local["offset_minutes"], errors="coerce")
    local["value_numeric"] = pd.to_numeric(local["value_numeric"], errors="coerce")
    status = local.loc[
        local["concept"].isin(["pressor_active", "mcs_active"])
        & np.isclose(local["offset_minutes"], LANDMARK_MINUTES, equal_nan=False)
        & local["value_numeric"].gt(0)
    ].copy()
    status["_text"] = status["raw_name"].fillna("").astype(str) + "+" + status["value_text"].fillna("").astype(str)
    for concept, output, patterns in (
        ("pressor_active", "kapur_active_drug_count_4h", DRUG_PATTERNS),
        ("mcs_active", "kapur_active_device_count_4h", DEVICE_PATTERNS),
    ):
        selected = status.loc[status["concept"].eq(concept)]
        if selected.empty:
            counts = pd.DataFrame(columns=[*keys, output])
        else:
            text = selected.groupby(keys, sort=False)["_text"].agg("+".join)
            counts = text.map(lambda value: _canonical_count(value, patterns)).rename(output).reset_index()
        result = result.merge(counts, on=keys, how="left", validate="one_to_one")

    pressor_known = pd.to_numeric(result["pressor_source_available"], errors="coerce").eq(1)
    mcs_known = pd.to_numeric(result["mcs_source_available"], errors="coerce").eq(1)
    result["kapur_active_drug_count_4h"] = pd.to_numeric(
        result["kapur_active_drug_count_4h"], errors="coerce"
    ).where(pressor_known)
    result["kapur_active_device_count_4h"] = pd.to_numeric(
        result["kapur_active_device_count_4h"], errors="coerce"
    ).where(mcs_known)
    result.loc[pressor_known, "kapur_active_drug_count_4h"] = result.loc[
        pressor_known, "kapur_active_drug_count_4h"
    ].fillna(0)
    result.loc[mcs_known, "kapur_active_device_count_4h"] = result.loc[
        mcs_known, "kapur_active_device_count_4h"
    ].fillna(0)
    result.loc[
        pressor_known
        & pd.to_numeric(result["baseline_vasoactive_flag"], errors="coerce").eq(1),
        "kapur_active_drug_count_4h",
    ] = result.loc[
        pressor_known
        & pd.to_numeric(result["baseline_vasoactive_flag"], errors="coerce").eq(1),
        "kapur_active_drug_count_4h",
    ].clip(lower=1)
    result.loc[
        mcs_known & pd.to_numeric(result["baseline_mcs_flag"], errors="coerce").eq(1),
        "kapur_active_device_count_4h",
    ] = result.loc[
        mcs_known & pd.to_numeric(result["baseline_mcs_flag"], errors="coerce").eq(1),
        "kapur_active_device_count_4h",
    ].clip(lower=1)
    result["kapur_treatment_components_observed"] = (pressor_known & mcs_known).astype(int)
    result["kapur_treatment_semantics"] = np.where(
        result["dataset"].astype(str).eq("mimic"),
        "interval_active_at_minute_240",
        "documented_by_minute_240_proxy",
    )
    return result.drop(
        columns=[
            "pressor_source_available",
            "mcs_source_available",
            "baseline_vasoactive_flag",
            "baseline_mcs_flag",
        ]
    )


def build_four_hour_biomarker_features(
    events: pd.DataFrame,
    analysis_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe SBP/lactate and Kapur-SCAI comparator features."""
    required_events = {"dataset", "stay_id", "concept", "offset_minutes", "value_numeric"}
    required_frame = {"dataset", "stay_id"}
    if missing := required_events.difference(events.columns):
        raise ValueError(f"Events lack benchmark columns: {sorted(missing)}")
    if missing := required_frame.difference(analysis_frame.columns):
        raise ValueError(f"Analysis frame lacks benchmark keys: {sorted(missing)}")
    keys = ["dataset", "stay_id"]
    base = analysis_frame[keys].drop_duplicates().copy()
    if len(base) != len(analysis_frame):
        raise ValueError("Biomarker benchmark requires one analysis row per dataset/stay")
    local = events.copy()
    local["dataset"] = local["dataset"].astype(str)
    local["concept"] = local["concept"].fillna("").astype(str).str.lower()
    local = local.loc[
        local["concept"].isin(
            [
                "sbp",
                "map",
                "lactate",
                "alt",
                "ph",
                "pressor_active",
                "mcs_active",
            ]
        )
    ].merge(base, on=keys, how="inner", validate="many_to_one")

    numeric_specs = {
        "sbp": (30.0, 300.0),
        "map": (20.0, 200.0),
        "lactate": (0.0, 30.0),
        "alt": (0.0, 50_000.0),
        "ph": (6.5, 8.0),
    }
    result = base.copy()
    for concept, (low, high) in numeric_specs.items():
        aggregate = _aggregate_numeric_concept(
            local, concept=concept, low=low, high=high
        )
        result = result.merge(aggregate, on=keys, how="left", validate="one_to_one")

    thresholds = (
        ("sbp", "sbp_below_90_fraction_4h", lambda values: values < 90.0, 30.0, 300.0),
        ("sbp", "sbp_below_60_fraction_4h", lambda values: values < 60.0, 30.0, 300.0),
        ("map", "map_below_65_fraction_4h", lambda values: values < 65.0, 20.0, 200.0),
        ("map", "map_below_50_fraction_4h", lambda values: values < 50.0, 20.0, 200.0),
        ("lactate", "lactate_ge_2_fraction_4h", lambda values: values >= 2.0, 0.0, 30.0),
        ("lactate", "lactate_ge_5_fraction_4h", lambda values: values >= 5.0, 0.0, 30.0),
        ("lactate", "lactate_gt_10_fraction_4h", lambda values: values > 10.0, 0.0, 30.0),
    )
    for concept, output, predicate, low, high in thresholds:
        fraction = _threshold_fraction(
            local,
            concept=concept,
            output=output,
            predicate=predicate,
            low=low,
            high=high,
        )
        result = result.merge(fraction, on=keys, how="left", validate="one_to_one")

    result["lactate_delta_4h"] = (
        _numeric_series(result, "lactate_last_4h")
        - _numeric_series(result, "lactate_first_4h")
    )
    treatment = _landmark_treatment_counts(local, analysis_frame)
    result = result.merge(treatment, on=keys, how="left", validate="one_to_one")

    component_flags = {
        "kapur_sbp_observed": "sbp_first_4h",
        "kapur_map_observed": "map_first_4h",
        "kapur_lactate_observed": "lactate_first_4h",
        "kapur_alt_observed": "alt_first_4h",
        "kapur_ph_observed": "ph_first_4h",
    }
    for flag, column in component_flags.items():
        result[flag] = _numeric_series(result, column).notna().astype(int)
    result["kapur_bp_domain_observed"] = result[
        ["kapur_sbp_observed", "kapur_map_observed"]
    ].max(axis=1)
    result["kapur_hypoperfusion_domain_observed"] = result[
        ["kapur_lactate_observed", "kapur_alt_observed"]
    ].max(axis=1)
    result["kapur_scai_core_observed"] = (
        result["kapur_bp_domain_observed"].eq(1)
        & result["kapur_hypoperfusion_domain_observed"].eq(1)
        & result["kapur_treatment_components_observed"].eq(1)
    ).astype(int)
    result["kapur_scai_full_component_observed"] = (
        result[list(component_flags)].eq(1).all(axis=1)
        & result["kapur_treatment_components_observed"].eq(1)
    ).astype(int)
    result["kapur_ohca_source_available"] = 0
    result["kapur_ohca_flag"] = np.nan

    last_abnormal = (
        _numeric_series(result, "sbp_last_4h").le(90.0)
        | _numeric_series(result, "map_last_4h").le(65.0)
        | _numeric_series(result, "lactate_last_4h").ge(2.0)
        | _numeric_series(result, "alt_last_4h").ge(200.0)
    ).fillna(False)
    result["kapur_persistent_abnormality_4h_proxy"] = last_abnormal.astype(int)

    first_stage_rows: list[dict[str, Any]] = []
    worst_stage_rows: list[dict[str, Any]] = []
    for row in result.itertuples(index=False):
        first = assign_kapur_scai_stage(
            sbp=getattr(row, "sbp_first_4h", math.nan),
            map_value=getattr(row, "map_first_4h", math.nan),
            lactate=getattr(row, "lactate_first_4h", math.nan),
            alt=getattr(row, "alt_first_4h", math.nan),
            ph=getattr(row, "ph_first_4h", math.nan),
            drug_count=getattr(row, "kapur_active_drug_count_4h", math.nan),
            device_count=getattr(row, "kapur_active_device_count_4h", math.nan),
            persistent_abnormality=bool(
                getattr(row, "kapur_persistent_abnormality_4h_proxy", 0)
            ),
            ohca=math.nan,
        )
        worst = assign_kapur_scai_stage(
            sbp=getattr(row, "sbp_min_4h", math.nan),
            map_value=getattr(row, "map_min_4h", math.nan),
            lactate=getattr(row, "lactate_max_4h", math.nan),
            alt=getattr(row, "alt_max_4h", math.nan),
            ph=getattr(row, "ph_min_4h", math.nan),
            drug_count=getattr(row, "kapur_active_drug_count_4h", math.nan),
            device_count=getattr(row, "kapur_active_device_count_4h", math.nan),
            persistent_abnormality=bool(last_abnormal.iloc[len(first_stage_rows)]),
            ohca=math.nan,
        )
        first_stage_rows.append(
            {
                "kapur_scai_stage_first4h": first[0],
                "kapur_scai_stage_num_first4h": first[1],
                "kapur_scai_evidence_first4h": first[2],
                "kapur_scai_lower_bound_first4h": int(first[3]),
            }
        )
        worst_stage_rows.append(
            {
                "kapur_scai_stage_worst4h": worst[0],
                "kapur_scai_stage_num_worst4h": worst[1],
                "kapur_scai_evidence_worst4h": worst[2],
                "kapur_scai_lower_bound_worst4h": int(worst[3]),
            }
        )
    result = pd.concat(
        [
            result.reset_index(drop=True),
            pd.DataFrame(first_stage_rows),
            pd.DataFrame(worst_stage_rows),
        ],
        axis=1,
    )
    result["kapur_scai_definition_version"] = KAPUR_STAGE_VERSION
    result["kapur_scai_reference"] = KAPUR_REFERENCE
    result["kapur_scai_limitations"] = (
        "OHCA_unavailable;physical_exam_unavailable;first_values_within_4h_not_24h;"
        "one_therapy_persistence_uses_last_4h_abnormality_proxy;"
        "eICU_treatment_is_documentation_proxy"
    )
    return result


def build_biomarker_analysis_frame(
    events: pd.DataFrame,
    analysis_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach comparator features to the immutable endpoint analysis frame."""
    features = build_four_hour_biomarker_features(events, analysis_frame)
    result = analysis_frame.merge(
        features,
        on=["dataset", "stay_id"],
        how="left",
        validate="one_to_one",
    )
    derived = pd.DataFrame(
        {
            "sbp_severity_index": -_numeric_series(result, "sbp_min_4h"),
            "lactate_severity_index": _numeric_series(result, "lactate_max_4h"),
            "spo2_absolute_deficit_index": -_numeric_series(result, "spo2_min"),
            "spo2_instability_index": _numeric_series(
                result, "spo2_dynamics_proxy_score"
            ),
            "kapur_scai_severity_index": _numeric_series(
                result, "kapur_scai_stage_num_first4h"
            ),
        },
        index=result.index,
    )
    return pd.concat([result, derived], axis=1)


def build_biomarker_coverage_audit(
    frame: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Report marker coverage, including the shared complete-case denominator."""
    rows: list[dict[str, Any]] = []
    source_tables = set(events.get("source_table", pd.Series(dtype=str)).dropna().astype(str))
    for dataset, group in frame.groupby(frame["dataset"].astype(str), sort=True):
        populations = {
            "harmonized_hf": group,
            "shock_code_positive": group.loc[
                _numeric_series(group, "shock_icd_flag").eq(1)
            ],
        }
        for population, local in populations.items():
            marker_columns = {
                "spo2_instability": "spo2_instability_index",
                "spo2_absolute_level": "spo2_absolute_deficit_index",
                "sbp": "sbp_severity_index",
                "lactate": "lactate_severity_index",
                "kapur_scai_first4h": "kapur_scai_severity_index",
                "kapur_scai_core_complete": "kapur_scai_core_observed",
                "kapur_scai_all_components_complete": "kapur_scai_full_component_observed",
            }
            for marker, column in marker_columns.items():
                values = _numeric_series(local, column)
                if marker.endswith("complete"):
                    observed = values.eq(1)
                else:
                    observed = values.notna()
                rows.append(
                    {
                        "dataset": dataset,
                        "population": population,
                        "marker": marker,
                        "n_population": int(len(local)),
                        "n_observed": int(observed.sum()),
                        "coverage_fraction": float(observed.mean()) if len(local) else math.nan,
                        "vital_aperiodic_source_present": bool(
                            dataset != "eicu" or "vitalAperiodic.csv" in source_tables
                        ),
                    }
                )
            shared = local[
                [
                    "spo2_instability_index",
                    "sbp_severity_index",
                    "lactate_severity_index",
                    "kapur_scai_severity_index",
                ]
            ].notna().all(axis=1)
            rows.append(
                {
                    "dataset": dataset,
                    "population": population,
                    "marker": "all_four_shared_complete_case",
                    "n_population": int(len(local)),
                    "n_observed": int(shared.sum()),
                    "coverage_fraction": float(shared.mean()) if len(local) else math.nan,
                    "vital_aperiodic_source_present": bool(
                        dataset != "eicu" or "vitalAperiodic.csv" in source_tables
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_kapur_scai_stage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Emit stage distributions without conflating missing components with A."""
    rows: list[dict[str, Any]] = []
    for dataset, group in frame.groupby(frame["dataset"].astype(str), sort=True):
        for population, local in {
            "harmonized_hf": group,
            "shock_code_positive": group.loc[
                _numeric_series(group, "shock_icd_flag").eq(1)
            ],
        }.items():
            stage = local.get("kapur_scai_stage_first4h", pd.Series(index=local.index, dtype=object)).fillna("unclassified")
            for category in ("A", "B", "C", "D", "E", "unclassified"):
                mask = stage.eq(category)
                rows.append(
                    {
                        "dataset": dataset,
                        "population": population,
                        "stage": category,
                        "n_population": int(len(local)),
                        "n": int(mask.sum()),
                        "fraction": float(mask.mean()) if len(local) else math.nan,
                        "lower_bound_fraction_within_stage": (
                            float(
                                pd.to_numeric(
                                    local.loc[mask, "kapur_scai_lower_bound_first4h"],
                                    errors="coerce",
                                ).mean()
                            )
                            if mask.any()
                            else math.nan
                        ),
                        "definition_version": KAPUR_STAGE_VERSION,
                    }
                )
    return pd.DataFrame(rows)


def build_biomarker_definition_table() -> pd.DataFrame:
    """Machine-readable definitions for every compared bedside signal."""
    return pd.DataFrame(
        [
            {
                "marker": "SpO2 level",
                "window": "[0,240) minutes",
                "features": ",".join(SPO2_LEVEL_FEATURES),
                "direction": "lower saturation is worse",
                "role": "absolute oxygenation comparator",
                "caveat": "pulse-oximeter and oxygen-treatment dependent",
                "reference": "PhysioGraph prespecified protocol",
            },
            {
                "marker": "SpO2 instability",
                "window": "[0,240) minutes; 15-minute median bins",
                "features": ",".join(SPO2_INSTABILITY_FEATURES),
                "direction": "higher RMSSD/jump/drop burden is worse",
                "role": "primary dynamic signal",
                "caveat": "requires at least three valid bins and two transitions",
                "reference": "PhysioGraph prespecified protocol",
            },
            {
                "marker": "SBP",
                "window": "[0,240) minutes",
                "features": ",".join(SBP_FEATURES),
                "direction": "lower pressure and greater instability are worse",
                "role": "hemodynamic comparator",
                "caveat": "eICU cuff BP unavailable when vitalAperiodic.csv is absent",
                "reference": "clinical comparator",
            },
            {
                "marker": "lactate",
                "window": "[0,240) minutes",
                "features": ",".join(LACTATE_FEATURES),
                "direction": "higher level/rising trajectory is worse",
                "role": "hypoperfusion comparator",
                "caveat": "sparse and selectively measured; mathematical coupling for lactate-rise outcomes",
                "reference": "clinical comparator",
            },
            {
                "marker": "Kapur-CSWG SCAI stage",
                "window": "first values in [0,240) minutes; treatment at minute 240",
                "features": ",".join(KAPUR_SCAI_FEATURES),
                "direction": "A through E",
                "role": "multicomponent staging comparator",
                "caveat": "EHR operationalization; OHCA and physical examination unavailable",
                "reference": KAPUR_REFERENCE,
            },
        ]
    )


def _model_specifications(
    frame: pd.DataFrame,
    *,
    include_dataset: bool,
) -> dict[str, list[str]]:
    context = _available(NEUTRAL_CONTEXT_FEATURES, frame)
    if include_dataset and "dataset" in frame:
        context.append("dataset")
    spo2_level = _available(SPO2_LEVEL_FEATURES, frame)
    spo2_instability = _available(SPO2_INSTABILITY_FEATURES, frame)
    spo2_sampling = _available(SPO2_SAMPLING_FEATURES, frame)
    sbp = _available(SBP_FEATURES, frame)
    lactate = _available(LACTATE_FEATURES, frame)
    scai = _available(KAPUR_SCAI_FEATURES, frame)
    return {
        "clinical_context": context,
        "clinical_plus_spo2_level": _dedupe([*context, *spo2_level, *spo2_sampling]),
        "clinical_plus_spo2_instability": _dedupe(
            [*context, *spo2_level, *spo2_sampling, *spo2_instability]
        ),
        "clinical_plus_sbp": _dedupe([*context, *sbp]),
        "clinical_plus_lactate": _dedupe([*context, *lactate]),
        "clinical_plus_kapur_scai": _dedupe([*context, *scai]),
        "clinical_plus_raw_conventional": _dedupe([*context, *sbp, *lactate]),
        "clinical_plus_raw_conventional_plus_spo2": _dedupe(
            [*context, *sbp, *lactate, *spo2_level, *spo2_sampling, *spo2_instability]
        ),
        "clinical_plus_kapur_scai_plus_spo2": _dedupe(
            [*context, *scai, *spo2_level, *spo2_sampling, *spo2_instability]
        ),
        "clinical_plus_all_conventional": _dedupe([*context, *sbp, *lactate, *scai]),
        "clinical_plus_all_conventional_plus_spo2": _dedupe(
            [
                *context,
                *sbp,
                *lactate,
                *scai,
                *spo2_level,
                *spo2_sampling,
                *spo2_instability,
            ]
        ),
    }


PRIMARY_PAIRWISE_COMPARISONS = (
    ("clinical_plus_spo2_level", "clinical_context", "absolute_SpO2_vs_context"),
    (
        "clinical_plus_spo2_instability",
        "clinical_plus_spo2_level",
        "SpO2_instability_increment_beyond_level_and_sampling",
    ),
    ("clinical_plus_spo2_instability", "clinical_context", "SpO2_signal_vs_context"),
    ("clinical_plus_sbp", "clinical_context", "SBP_vs_context"),
    ("clinical_plus_lactate", "clinical_context", "lactate_vs_context"),
    ("clinical_plus_kapur_scai", "clinical_context", "Kapur_SCAI_vs_context"),
    ("clinical_plus_spo2_instability", "clinical_plus_sbp", "SpO2_vs_SBP"),
    ("clinical_plus_spo2_instability", "clinical_plus_lactate", "SpO2_vs_lactate"),
    ("clinical_plus_spo2_instability", "clinical_plus_kapur_scai", "SpO2_vs_Kapur_SCAI"),
    (
        "clinical_plus_raw_conventional_plus_spo2",
        "clinical_plus_raw_conventional",
        "SpO2_increment_beyond_SBP_and_lactate",
    ),
    (
        "clinical_plus_kapur_scai_plus_spo2",
        "clinical_plus_kapur_scai",
        "SpO2_increment_beyond_Kapur_SCAI",
    ),
    (
        "clinical_plus_all_conventional_plus_spo2",
        "clinical_plus_all_conventional",
        "SpO2_increment_beyond_all_conventional",
    ),
)


def _metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    intercept, slope = _calibration(y, probability)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": float(_ece(y, probability)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _weighted_binary_metric_samples(
    y: np.ndarray,
    probability: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized binary metrics for rows of non-negative sample weights."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    weights = np.asarray(sample_weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != len(y):
        raise ValueError("sample_weights must have shape (samples, observations)")
    if len(probability) != len(y):
        raise ValueError("probability and y must have equal length")
    if np.any(weights < 0):
        raise ValueError("sample weights must be non-negative")

    total = weights.sum(axis=1)
    positive_total = weights @ y.astype(float)
    negative_total = weights @ (1 - y).astype(float)
    brier = np.divide(
        weights @ ((probability - y) ** 2),
        total,
        out=np.full(len(weights), np.nan),
        where=total > 0,
    )

    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_y = y[order]
    sorted_weights = weights[:, order]
    group_starts = np.r_[
        0,
        np.flatnonzero(sorted_probability[1:] != sorted_probability[:-1]) + 1,
    ]
    positive_by_score = np.add.reduceat(
        sorted_weights * sorted_y,
        group_starts,
        axis=1,
    )
    negative_by_score = np.add.reduceat(
        sorted_weights * (1 - sorted_y),
        group_starts,
        axis=1,
    )

    negative_below = np.cumsum(negative_by_score, axis=1) - negative_by_score
    concordant = np.sum(
        positive_by_score * (negative_below + 0.5 * negative_by_score),
        axis=1,
    )
    auc_denominator = positive_total * negative_total
    auroc = np.divide(
        concordant,
        auc_denominator,
        out=np.full(len(weights), np.nan),
        where=auc_denominator > 0,
    )

    positive_descending = positive_by_score[:, ::-1]
    negative_descending = negative_by_score[:, ::-1]
    cumulative_positive = np.cumsum(positive_descending, axis=1)
    cumulative_total = cumulative_positive + np.cumsum(negative_descending, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    auprc = np.divide(
        np.sum(precision * positive_descending, axis=1),
        positive_total,
        out=np.full(len(weights), np.nan),
        where=positive_total > 0,
    )
    return {"auroc": auroc, "auprc": auprc, "brier": brier}


def _cluster_bootstrap_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Patient-cluster bootstrap all models jointly for paired contrasts."""
    group_codes, unique = pd.factorize(groups, sort=True)
    n_groups = len(unique)
    rng = np.random.default_rng(seed)
    group_weight_samples = np.empty((repetitions, n_groups), dtype=np.int32)
    for repetition in range(repetitions):
        # Uniform integer draws plus bincount are exactly a non-parametric
        # cluster bootstrap and are substantially faster than multinomial's
        # sequential-binomial implementation for thousands of patients.
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        group_weight_samples[repetition] = np.bincount(
            sampled_groups, minlength=n_groups
        )
    row_weight_samples = group_weight_samples[:, group_codes]
    positive_weight = row_weight_samples[:, y == 1].sum(axis=1)
    negative_weight = row_weight_samples[:, y == 0].sum(axis=1)
    valid = (positive_weight > 0) & (negative_weight > 0)
    row_weight_samples = row_weight_samples[valid]
    valid_repetitions = int(valid.sum())
    metric_samples = {
        model: _weighted_binary_metric_samples(y, probability, row_weight_samples)
        for model, probability in predictions.items()
    }

    intervals: dict[str, dict[str, float]] = {}
    point_metrics = {
        model: _metric_values(y, probability)
        for model, probability in predictions.items()
    }
    for model, samples in metric_samples.items():
        intervals[model] = {
            "bootstrap_repetitions_requested": float(repetitions),
            "bootstrap_repetitions_valid": float(valid_repetitions),
        }
        for metric, values in samples.items():
            low, high = _percentile_interval(values)
            intervals[model][f"{metric}_ci95_low"] = low
            intervals[model][f"{metric}_ci95_high"] = high

    comparisons: list[dict[str, Any]] = []
    for candidate, reference, label in PRIMARY_PAIRWISE_COMPARISONS:
        if candidate not in predictions or reference not in predictions:
            continue
        row: dict[str, Any] = {
            "comparison": label,
            "candidate_model": candidate,
            "reference_model": reference,
            "bootstrap_repetitions_requested": repetitions,
            "bootstrap_repetitions_valid": valid_repetitions,
        }
        for metric in ("auroc", "auprc", "brier"):
            if metric == "brier":
                values = metric_samples[reference][metric] - metric_samples[candidate][metric]
            else:
                values = metric_samples[candidate][metric] - metric_samples[reference][metric]
            low, high = _percentile_interval(values)
            candidate_point = point_metrics[candidate][metric]
            reference_point = point_metrics[reference][metric]
            point = (
                reference_point - candidate_point
                if metric == "brier"
                else candidate_point - reference_point
            )
            row[f"delta_{metric}"] = point
            row[f"delta_{metric}_ci95_low"] = low
            row[f"delta_{metric}_ci95_high"] = high
            row[f"candidate_better_{metric}"] = bool(low > 0)
        comparisons.append(row)
    return intervals, comparisons


def _valid_grouped_folds(
    frame: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]] | None, int, int]:
    from sklearn.model_selection import StratifiedGroupKFold

    event_groups = int(groups.loc[y.eq(1)].nunique())
    non_event_groups = int(groups.loc[y.eq(0)].nunique())
    maximum = min(n_splits, int(groups.nunique()), event_groups, non_event_groups)
    for candidate_splits in range(maximum, 1, -1):
        for seed_offset in range(20):
            seed = random_state + seed_offset
            splitter = StratifiedGroupKFold(
                n_splits=candidate_splits, shuffle=True, random_state=seed
            )
            candidate = list(splitter.split(frame, y, groups))
            if all(
                y.iloc[train].nunique() == 2 and y.iloc[test].nunique() == 2
                for train, test in candidate
            ):
                return candidate, candidate_splits, seed
    return None, maximum, random_state


def _analysis_populations(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # SCAI validity in shock-coded patients is assessed in the dedicated
    # association sensitivity. Repeating every five-fold predictive model in
    # that small subgroup adds no new head-to-head estimand and doubles runtime.
    return {"harmonized_hf": frame}


def fit_biomarker_prediction_benchmark(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    min_rows: int = MIN_ROWS,
    min_events: int = MIN_EVENTS,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare all markers on identical patient-grouped OOF folds."""
    measured = frame.copy()
    if "spo2_dynamics_eligible_flag" in measured:
        measured = measured.loc[
            pd.to_numeric(measured["spo2_dynamics_eligible_flag"], errors="coerce").eq(1)
        ].copy()
    datasets = sorted(measured["dataset"].dropna().astype(str).unique())
    performance_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for population, population_frame in _analysis_populations(measured).items():
        scopes = [
            (dataset, population_frame.loc[population_frame["dataset"].astype(str).eq(dataset)].copy())
            for dataset in datasets
        ]
        if len(datasets) > 1:
            scopes.append(("pooled_secondary", population_frame.copy()))
        for analysis_scope, scope_frame in scopes:
            specifications = _model_specifications(
                scope_frame, include_dataset=analysis_scope == "pooled_secondary"
            )
            for outcome in [name for name in BENCHMARK_ENDPOINTS if name in scope_frame]:
                valid = pd.to_numeric(scope_frame[outcome], errors="coerce").isin([0, 1])
                local = scope_frame.loc[valid].reset_index(drop=True)
                y = pd.to_numeric(local[outcome], errors="coerce").astype(int)
                groups, grouping = _group_series(local)
                events = int(y.sum())
                non_events = int(len(y) - events)
                base = {
                    "population": population,
                    "analysis_scope": analysis_scope,
                    "outcome": outcome,
                    "endpoint_tier": _endpoint_tier(outcome),
                    "n": int(len(y)),
                    "events": events,
                    "non_events": non_events,
                    "n_patients": int(groups.nunique()) if len(groups) else 0,
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
                        {
                            **base,
                            "status": "failed",
                            "reason": "unable to construct all-two-class patient-grouped folds",
                        }
                    )
                    continue
                predictions = {
                    model: np.full(len(local), np.nan, dtype=float)
                    for model in specifications
                }
                fold_ids = np.full(len(local), -1, dtype=int)
                feature_counts = {model: [] for model in specifications}
                failure: str | None = None
                for fold_id, (train_index, test_index) in enumerate(folds):
                    fold_ids[test_index] = fold_id
                    for model, columns in specifications.items():
                        if not columns:
                            failure = f"no available features for {model}"
                            break
                        try:
                            pipeline = _build_model_pipeline(local, columns)
                            pipeline.fit(local.iloc[train_index][columns], y.iloc[train_index])
                            predictions[model][test_index] = pipeline.predict_proba(
                                local.iloc[test_index][columns]
                            )[:, 1]
                            feature_counts[model].append(
                                int(len(pipeline.named_steps["preprocess"].get_feature_names_out()))
                            )
                        except Exception as exc:
                            failure = f"{model}: {exc}"
                            break
                    if failure:
                        break
                if failure or any(not np.isfinite(value).all() for value in predictions.values()):
                    performance_rows.append(
                        {
                            **base,
                            "status": "failed",
                            "reason": failure or "incomplete OOF predictions",
                        }
                    )
                    continue

                intervals, comparisons = _cluster_bootstrap_metrics(
                    y.to_numpy(),
                    predictions,
                    groups.to_numpy(),
                    repetitions=bootstrap_repetitions,
                    seed=random_state + len(comparison_rows),
                )
                context_metrics = _metric_values(y.to_numpy(), predictions["clinical_context"])
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
                            "preprocessing_scope": "fit_within_each_training_fold",
                            "analysis_population": "common_spo2_dynamics_eligible_outcome_observed_rows",
                            "feature_columns": ",".join(specifications[model]),
                            "feature_count": feature_count,
                            "events_per_feature": epv,
                            "epv_status": "adequate_ge_10" if epv >= 10 else "fragile_lt_10",
                            **metrics,
                            **intervals[model],
                            "delta_auroc_vs_context": metrics["auroc"] - context_metrics["auroc"],
                            "delta_auprc_vs_context": metrics["auprc"] - context_metrics["auprc"],
                            "brier_improvement_vs_context": context_metrics["brier"] - metrics["brier"],
                        }
                    )
                for row in comparisons:
                    comparison_rows.append({**base, **row})

                identifiers = [
                    column for column in ("dataset", "stay_id", "person_id") if column in local
                ]
                prediction = local[identifiers].copy()
                prediction["population"] = population
                prediction["analysis_scope"] = analysis_scope
                prediction["outcome"] = outcome
                prediction["endpoint_tier"] = _endpoint_tier(outcome)
                prediction["fold_id"] = fold_ids
                prediction["y_true"] = y.to_numpy()
                for model, probability in predictions.items():
                    prediction[f"probability__{model}"] = probability
                prediction_rows.append(prediction)

    performance = pd.DataFrame(performance_rows)
    comparisons = pd.DataFrame(comparison_rows)
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows
        else pd.DataFrame()
    )
    return performance, comparisons, predictions


def fit_cross_database_biomarker_transportability(
    frame: pd.DataFrame,
    *,
    min_rows: int = 100,
    min_events: int = MIN_EVENTS,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    random_state: int = RANDOM_SEED + 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train in one database and evaluate unchanged models in the other."""
    measured = frame.loc[
        _numeric_series(frame, "spo2_dynamics_eligible_flag").eq(1)
    ].copy()
    if not {"mimic", "eicu"}.issubset(set(measured["dataset"].astype(str))):
        unavailable = pd.DataFrame(
            [{"status": "unavailable", "reason": "mimic_and_eicu_required"}]
        )
        return unavailable, unavailable.copy(), pd.DataFrame()
    performance_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for train_dataset, test_dataset in (("mimic", "eicu"), ("eicu", "mimic")):
        train_all = measured.loc[measured["dataset"].astype(str).eq(train_dataset)].copy()
        test_all = measured.loc[measured["dataset"].astype(str).eq(test_dataset)].copy()
        specifications = {
            model: [
                column
                for column in columns
                if train_all[column].notna().any() and test_all[column].notna().any()
            ]
            for model, columns in _model_specifications(
                measured, include_dataset=False
            ).items()
        }
        for outcome in [name for name in BENCHMARK_ENDPOINTS if name in measured]:
            train = train_all.loc[
                pd.to_numeric(train_all[outcome], errors="coerce").isin([0, 1])
            ].reset_index(drop=True)
            test = test_all.loc[
                pd.to_numeric(test_all[outcome], errors="coerce").isin([0, 1])
            ].reset_index(drop=True)
            y_train = pd.to_numeric(train[outcome], errors="coerce").astype(int)
            y_test = pd.to_numeric(test[outcome], errors="coerce").astype(int)
            base = {
                "population": "harmonized_hf",
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "outcome": outcome,
                "endpoint_tier": _endpoint_tier(outcome),
                "n_train": int(len(train)),
                "events_train": int(y_train.sum()),
                "n_test": int(len(test)),
                "events_test": int(y_test.sum()),
            }
            if (
                len(train) < min_rows
                or len(test) < min_rows
                or min(int(y_train.sum()), int(len(y_train) - y_train.sum())) < min_events
                or min(int(y_test.sum()), int(len(y_test) - y_test.sum())) < min_events
            ):
                performance_rows.append(
                    {
                        **base,
                        "status": "skipped",
                        "reason": "below cross-database rows/events floor",
                    }
                )
                continue
            probabilities: dict[str, np.ndarray] = {}
            feature_counts: dict[str, int] = {}
            failure: str | None = None
            for model, columns in specifications.items():
                if not columns:
                    failure = f"no available features for {model}"
                    break
                try:
                    pipeline = _build_model_pipeline(train, columns)
                    pipeline.fit(train[columns], y_train)
                    probabilities[model] = pipeline.predict_proba(test[columns])[:, 1]
                    feature_counts[model] = int(
                        len(pipeline.named_steps["preprocess"].get_feature_names_out())
                    )
                except Exception as exc:
                    failure = f"{model}: {exc}"
                    break
            if failure or any(not np.isfinite(value).all() for value in probabilities.values()):
                performance_rows.append(
                    {
                        **base,
                        "status": "failed",
                        "reason": failure or "non-finite external predictions",
                    }
                )
                continue
            groups, grouping = _group_series(test)
            intervals, comparisons = _cluster_bootstrap_metrics(
                y_test.to_numpy(),
                probabilities,
                groups.to_numpy(),
                repetitions=bootstrap_repetitions,
                seed=random_state + len(comparison_rows),
            )
            context_metrics = _metric_values(
                y_test.to_numpy(), probabilities["clinical_context"]
            )
            for model, probability in probabilities.items():
                metrics = _metric_values(y_test.to_numpy(), probability)
                feature_count = feature_counts[model]
                epv = min(
                    int(y_train.sum()), int(len(y_train) - y_train.sum())
                ) / max(feature_count, 1)
                performance_rows.append(
                    {
                        **base,
                        "model": model,
                        "status": "fit",
                        "reason": "",
                        "grouping": grouping,
                        "preprocessing_scope": f"fit_on_{train_dataset}_only",
                        "metric_scope": "untouched_external_database",
                        "feature_columns": ",".join(specifications[model]),
                        "feature_count": feature_count,
                        "training_events_per_feature": epv,
                        "epv_status": "adequate_ge_10" if epv >= 10 else "fragile_lt_10",
                        **metrics,
                        **intervals[model],
                        "delta_auroc_vs_context": metrics["auroc"] - context_metrics["auroc"],
                        "delta_auprc_vs_context": metrics["auprc"] - context_metrics["auprc"],
                        "brier_improvement_vs_context": context_metrics["brier"] - metrics["brier"],
                    }
                )
            for row in comparisons:
                comparison_rows.append({**base, **row})
            identifiers = [
                column for column in ("dataset", "stay_id", "person_id") if column in test
            ]
            prediction = test[identifiers].copy()
            prediction["train_dataset"] = train_dataset
            prediction["test_dataset"] = test_dataset
            prediction["outcome"] = outcome
            prediction["endpoint_tier"] = _endpoint_tier(outcome)
            prediction["y_true"] = y_test.to_numpy()
            for model, probability in probabilities.items():
                prediction[f"probability__{model}"] = probability
            prediction_rows.append(prediction)
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(comparison_rows),
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame(),
    )


def build_decision_curve_table(
    predictions: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Calculate out-of-fold net benefit without generating decorative figures."""
    if predictions.empty:
        return pd.DataFrame()
    probability_columns = [
        column for column in predictions if column.startswith("probability__")
    ]
    group_columns = [
        column
        for column in (
            "population",
            "analysis_scope",
            "train_dataset",
            "test_dataset",
            "outcome",
            "endpoint_tier",
        )
        if column in predictions
    ]
    rows: list[dict[str, Any]] = []
    for key, local in predictions.groupby(group_columns, sort=False, dropna=False):
        metadata = dict(zip(group_columns, key if isinstance(key, tuple) else (key,)))
        y = pd.to_numeric(local["y_true"], errors="coerce").to_numpy(int)
        n = len(y)
        prevalence = float(y.mean()) if n else math.nan
        for threshold in thresholds:
            treat_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
            rows.append(
                {
                    **metadata,
                    "model": "treat_all",
                    "threshold": threshold,
                    "n": n,
                    "event_rate": prevalence,
                    "net_benefit": treat_all,
                    "net_benefit_vs_treat_all": 0.0,
                    "net_benefit_vs_treat_none": treat_all,
                }
            )
            for column in probability_columns:
                probability = pd.to_numeric(local[column], errors="coerce").to_numpy(float)
                predicted = probability >= threshold
                tp = int(np.sum(predicted & (y == 1)))
                fp = int(np.sum(predicted & (y == 0)))
                net_benefit = tp / n - fp / n * threshold / (1.0 - threshold)
                rows.append(
                    {
                        **metadata,
                        "model": column.removeprefix("probability__"),
                        "threshold": threshold,
                        "n": n,
                        "event_rate": prevalence,
                        "true_positives": tp,
                        "false_positives": fp,
                        "net_benefit": net_benefit,
                        "net_benefit_vs_treat_all": net_benefit - treat_all,
                        "net_benefit_vs_treat_none": net_benefit,
                    }
                )
    return pd.DataFrame(rows)


ASSOCIATION_EXPOSURES = {
    "spo2_instability": "spo2_instability_index",
    "spo2_absolute_deficit": "spo2_absolute_deficit_index",
    "sbp_severity": "sbp_severity_index",
    "lactate_severity": "lactate_severity_index",
    "kapur_scai_severity": "kapur_scai_severity_index",
}
ASSOCIATION_CONTROL_CANDIDATES = (
    "age",
    "is_male",
    "sex_unknown_flag",
    "shock_icd_flag",
)


def _standardized_exposures(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    selected_columns = _dedupe(
        [
            "dataset",
            "stay_id",
            "person_id",
            *BENCHMARK_ENDPOINTS,
            *ASSOCIATION_EXPOSURES.values(),
            *ASSOCIATION_CONTROL_CANDIDATES,
        ]
    )
    local = frame.reindex(columns=selected_columns).copy()
    audit: dict[str, dict[str, float]] = {}
    for label, column in ASSOCIATION_EXPOSURES.items():
        values = _numeric_series(local, column)
        mean = float(values.mean()) if values.notna().any() else math.nan
        sd = float(values.std(ddof=0)) if values.notna().any() else math.nan
        output = f"exposure__{label}"
        local[output] = (values - mean) / sd if np.isfinite(sd) and sd > 0 else np.nan
        audit[output] = {
            "raw_mean": mean,
            "raw_sd": sd,
            "n_observed": float(values.notna().sum()),
        }
    return local, audit


def _association_groups(frame: pd.DataFrame) -> pd.Series:
    groups, _ = _group_series(frame)
    return groups


def _fit_modified_poisson_exposures(
    frame: pd.DataFrame,
    *,
    outcome: str,
    exposures: list[str],
    model_name: str,
    min_rows: int,
    min_events: int,
) -> list[dict[str, Any]]:
    base = {
        "model": model_name,
        "outcome": outcome,
        "endpoint_tier": _endpoint_tier(outcome),
        "exposure_set": ",".join(exposures),
    }
    local = frame.copy()
    local[outcome] = pd.to_numeric(local[outcome], errors="coerce")
    for exposure in exposures:
        local[exposure] = pd.to_numeric(local[exposure], errors="coerce")
    local = local.loc[
        local[outcome].isin([0, 1]) & local[exposures].notna().all(axis=1)
    ].copy()
    events = int(local[outcome].sum())
    non_events = int(len(local) - events)
    information = min(events, non_events)
    common = {
        **base,
        "n": int(len(local)),
        "events": events,
        "non_events": non_events,
    }
    if len(local) < min_rows or information < min_events:
        return [
            {
                **common,
                "status": "underpowered",
                "reason": "insufficient complete rows/events/non-events",
            }
        ]
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        return [{**common, "status": "failed", "reason": str(exc)}]

    design = pd.DataFrame(index=local.index)
    for exposure in exposures:
        design[exposure] = local[exposure].astype(float)
    retained: list[str] = []
    for column in ASSOCIATION_CONTROL_CANDIDATES:
        if column not in local:
            continue
        values = pd.to_numeric(local[column], errors="coerce")
        if values.notna().sum() < max(20, int(0.20 * len(local))):
            continue
        missing = values.isna()
        values = values.fillna(float(values.median()))
        if values.nunique(dropna=False) <= 1:
            continue
        if column not in {
            "is_male",
            "sex_unknown_flag",
            "shock_icd_flag",
            "acute_mi_flag",
            "cardiomyopathy_flag",
        }:
            scale = float(values.std(ddof=0))
            if np.isfinite(scale) and scale > 0:
                values = (values - float(values.mean())) / scale
        design[column] = values.astype(float)
        retained.append(column)
        if missing.any() and missing.any() and (~missing).any():
            design[f"{column}__missing"] = missing.astype(float)
            retained.append(f"{column}__missing")
    design = sm.add_constant(design.astype(float), has_constant="add")
    parameter_count = int(design.shape[1])
    epv = information / max(parameter_count, 1)
    if epv < 5:
        return [
            {
                **common,
                "status": "underpowered_low_information",
                "reason": "events_per_parameter_below_5",
                "parameter_count": parameter_count,
                "events_per_parameter": epv,
                "control_covariates": ",".join(retained),
            }
        ]
    try:
        model = sm.GLM(
            local[outcome].astype(int), design, family=sm.families.Poisson()
        )
        groups = _association_groups(local)
        fit = (
            model.fit(cov_type="cluster", cov_kwds={"groups": groups})
            if groups.nunique() > 1
            else model.fit(cov_type="HC0")
        )
    except Exception as exc:
        return [
            {
                **common,
                "status": "failed",
                "reason": str(exc)[:500],
                "parameter_count": parameter_count,
                "events_per_parameter": epv,
                "control_covariates": ",".join(retained),
            }
        ]
    rows: list[dict[str, Any]] = []
    for exposure in exposures:
        coefficient = float(fit.params[exposure])
        low, high = fit.conf_int().loc[exposure]
        values = np.exp([coefficient, float(low), float(high)])
        rows.append(
            {
                **common,
                "exposure": exposure.removeprefix("exposure__"),
                "status": "estimated",
                "reason": "",
                "risk_ratio_per_1sd_worse": float(values[0]),
                "ci95_low": float(values[1]),
                "ci95_high": float(values[2]),
                "p_value": float(fit.pvalues[exposure]),
                "log_rr_se": float(fit.bse[exposure]),
                "parameter_count": parameter_count,
                "events_per_parameter": epv,
                "claim_ready_epv10": bool(epv >= 10),
                "control_covariates": ",".join(retained),
                "grouping": "dataset_qualified_patient_with_stay_fallback",
                "estimand": "adjusted_risk_ratio_per_dataset_specific_1sd_worsening",
            }
        )
    return rows


def fit_biomarker_associations(
    frame: pd.DataFrame,
    *,
    min_rows: int = 80,
    min_events: int = 10,
) -> pd.DataFrame:
    """Estimate marker-specific and mutually adjusted prognostic RRs."""
    rows: list[dict[str, Any]] = []
    for dataset, dataset_frame in frame.groupby(frame["dataset"].astype(str), sort=True):
        for population, population_frame in {
            "harmonized_hf": dataset_frame,
            "shock_code_positive_sensitivity": dataset_frame.loc[
                _numeric_series(dataset_frame, "shock_icd_flag").eq(1)
            ],
        }.items():
            local, scale_audit = _standardized_exposures(population_frame)
            model_sets = {
                "marker_specific_spo2_instability": ["exposure__spo2_instability"],
                "marker_specific_spo2_absolute": ["exposure__spo2_absolute_deficit"],
                "marker_specific_sbp": ["exposure__sbp_severity"],
                "marker_specific_lactate": ["exposure__lactate_severity"],
                "marker_specific_kapur_scai": ["exposure__kapur_scai_severity"],
                "joint_raw_biomarkers": [
                    "exposure__spo2_instability",
                    "exposure__spo2_absolute_deficit",
                    "exposure__sbp_severity",
                    "exposure__lactate_severity",
                ],
                "kapur_scai_plus_spo2": [
                    "exposure__spo2_instability",
                    "exposure__spo2_absolute_deficit",
                    "exposure__kapur_scai_severity",
                ],
            }
            for outcome in [name for name in BENCHMARK_ENDPOINTS if name in local]:
                for model_name, exposures in model_sets.items():
                    estimates = _fit_modified_poisson_exposures(
                        local,
                        outcome=outcome,
                        exposures=exposures,
                        model_name=model_name,
                        min_rows=min_rows,
                        min_events=min_events,
                    )
                    for estimate in estimates:
                        exposure_key = (
                            f"exposure__{estimate.get('exposure')}"
                            if estimate.get("exposure")
                            else exposures[0]
                        )
                        rows.append(
                            {
                                "dataset": dataset,
                                "population": population,
                                **estimate,
                                "exposure_raw_mean": scale_audit.get(exposure_key, {}).get("raw_mean"),
                                "exposure_raw_sd": scale_audit.get(exposure_key, {}).get("raw_sd"),
                                "exposure_observed_in_population": scale_audit.get(exposure_key, {}).get("n_observed"),
                            }
                        )
    result = pd.DataFrame(rows)
    if result.empty or "p_value" not in result:
        return result
    result["q_value_bh"] = np.nan
    estimated = result["status"].eq("estimated") & pd.to_numeric(
        result["p_value"], errors="coerce"
    ).notna()
    for _, index in result.loc[estimated].groupby(
        ["dataset", "population", "model", "endpoint_tier"], sort=False
    ).groups.items():
        result.loc[index, "q_value_bh"] = _bh_adjust(result.loc[index, "p_value"])
    result["multiplicity_family"] = (
        "BH_within_dataset_population_model_and_endpoint_tier"
    )
    return result


def summarize_biomarker_benchmark(
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    associations: pd.DataFrame,
    external_performance: pd.DataFrame,
    external_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Create compact, non-selective summaries across all eligible endpoints."""
    rows: list[dict[str, Any]] = []
    fit = performance.loc[
        performance.get("status", pd.Series(index=performance.index, dtype=str)).eq("fit")
    ].copy()
    if not fit.empty:
        for key, group in fit.groupby(
            ["population", "analysis_scope", "model"], sort=True
        ):
            rows.append(
                {
                    "analysis": "internal_patient_grouped_oof",
                    "population": key[0],
                    "scope": key[1],
                    "model_or_exposure": key[2],
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_auroc": float(group["auroc"].median()),
                    "median_auprc": float(group["auprc"].median()),
                    "median_delta_auroc_vs_context": float(
                        group["delta_auroc_vs_context"].median()
                    ),
                    "median_delta_auprc_vs_context": float(
                        group["delta_auprc_vs_context"].median()
                    ),
                }
            )
    if not comparisons.empty:
        for key, group in comparisons.groupby(
            ["population", "analysis_scope", "comparison"], sort=True
        ):
            rows.append(
                {
                    "analysis": "internal_paired_patient_bootstrap",
                    "population": key[0],
                    "scope": key[1],
                    "model_or_exposure": key[2],
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_delta_auroc": float(group["delta_auroc"].median()),
                    "n_delta_auroc_ci_excludes_zero_positive": int(
                        pd.to_numeric(group["delta_auroc_ci95_low"], errors="coerce").gt(0).sum()
                    ),
                    "n_delta_auroc_ci_excludes_zero_negative": int(
                        pd.to_numeric(group["delta_auroc_ci95_high"], errors="coerce").lt(0).sum()
                    ),
                }
            )
    estimated = associations.loc[
        associations.get("status", pd.Series(index=associations.index, dtype=str)).eq("estimated")
    ].copy()
    if not estimated.empty:
        for key, group in estimated.groupby(
            ["dataset", "population", "model", "exposure"], sort=True
        ):
            rows.append(
                {
                    "analysis": "modified_poisson_association",
                    "population": key[1],
                    "scope": key[0],
                    "model_or_exposure": f"{key[2]}:{key[3]}",
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_rr_per_1sd_worse": float(
                        group["risk_ratio_per_1sd_worse"].median()
                    ),
                    "n_positive_q_lt_0_05": int(
                        (
                            pd.to_numeric(group["q_value_bh"], errors="coerce").lt(0.05)
                            & pd.to_numeric(group["risk_ratio_per_1sd_worse"], errors="coerce").gt(1)
                        ).sum()
                    ),
                    "n_inverse_q_lt_0_05": int(
                        (
                            pd.to_numeric(group["q_value_bh"], errors="coerce").lt(0.05)
                            & pd.to_numeric(group["risk_ratio_per_1sd_worse"], errors="coerce").lt(1)
                        ).sum()
                    ),
                }
            )
    external_fit = external_performance.loc[
        external_performance.get(
            "status", pd.Series(index=external_performance.index, dtype=str)
        ).eq("fit")
    ].copy()
    if not external_fit.empty:
        for key, group in external_fit.groupby(
            ["train_dataset", "test_dataset", "model"], sort=True
        ):
            rows.append(
                {
                    "analysis": "external_database_transportability",
                    "population": "harmonized_hf",
                    "scope": f"{key[0]}_to_{key[1]}",
                    "model_or_exposure": key[2],
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_auroc": float(group["auroc"].median()),
                    "median_delta_auroc_vs_context": float(
                        group["delta_auroc_vs_context"].median()
                    ),
                }
            )
    if not external_comparisons.empty:
        for key, group in external_comparisons.groupby(
            ["train_dataset", "test_dataset", "comparison"], sort=True
        ):
            rows.append(
                {
                    "analysis": "external_paired_patient_bootstrap",
                    "population": "harmonized_hf",
                    "scope": f"{key[0]}_to_{key[1]}",
                    "model_or_exposure": key[2],
                    "n_endpoints": int(group["outcome"].nunique()),
                    "median_delta_auroc": float(group["delta_auroc"].median()),
                    "n_delta_auroc_ci_excludes_zero_positive": int(
                        pd.to_numeric(group["delta_auroc_ci95_low"], errors="coerce").gt(0).sum()
                    ),
                    "n_delta_auroc_ci_excludes_zero_negative": int(
                        pd.to_numeric(group["delta_auroc_ci95_high"], errors="coerce").lt(0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compact_feature_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain comparator provenance without duplicating the 400-column frame."""
    explicit = [
        "dataset",
        "stay_id",
        "person_id",
        "age",
        "is_male",
        "shock_icd_flag",
        "spo2_dynamics_eligible_flag",
        *SPO2_LEVEL_FEATURES,
        *SPO2_SAMPLING_FEATURES,
        *SPO2_INSTABILITY_FEATURES,
        *SBP_FEATURES,
        *LACTATE_FEATURES,
        *KAPUR_SCAI_FEATURES,
        "sbp_severity_index",
        "lactate_severity_index",
        "spo2_absolute_deficit_index",
        "spo2_instability_index",
        "kapur_scai_severity_index",
    ]
    generated = [
        column
        for column in frame
        if column.startswith("kapur_")
        or (
            column.startswith(("map_", "alt_", "ph_"))
            and column.endswith("_4h")
        )
    ]
    return frame.reindex(columns=_dedupe([*explicit, *generated])).copy()


def run_biomarker_benchmark(
    events: pd.DataFrame,
    analysis_frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, pd.DataFrame]:
    """Run the complete cached-data benchmark and return auditable tables."""
    frame = build_biomarker_analysis_frame(events, analysis_frame)
    coverage = build_biomarker_coverage_audit(frame, events)
    stage_audit = build_kapur_scai_stage_audit(frame)
    performance, comparisons, predictions = fit_biomarker_prediction_benchmark(
        frame,
        n_splits=n_splits,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    external_performance, external_comparisons, external_predictions = (
        fit_cross_database_biomarker_transportability(
            frame,
            bootstrap_repetitions=bootstrap_repetitions,
        )
    )
    associations = fit_biomarker_associations(frame)
    decision_curve = build_decision_curve_table(predictions)
    external_decision_curve = build_decision_curve_table(external_predictions)
    summary = summarize_biomarker_benchmark(
        performance,
        comparisons,
        associations,
        external_performance,
        external_comparisons,
    )
    return {
        "biomarker_benchmark_features": _compact_feature_audit(frame),
        "biomarker_benchmark_definitions": build_biomarker_definition_table(),
        "biomarker_benchmark_coverage": coverage,
        "biomarker_benchmark_kapur_scai_audit": stage_audit,
        "biomarker_benchmark_performance": performance,
        "biomarker_benchmark_pairwise": comparisons,
        "biomarker_benchmark_oof_predictions": predictions,
        "biomarker_benchmark_decision_curve": decision_curve,
        "biomarker_benchmark_associations": associations,
        "biomarker_benchmark_external_performance": external_performance,
        "biomarker_benchmark_external_pairwise": external_comparisons,
        "biomarker_benchmark_external_predictions": external_predictions,
        "biomarker_benchmark_external_decision_curve": external_decision_curve,
        "biomarker_benchmark_summary": summary,
    }


__all__ = [
    "BENCHMARK_ENDPOINTS",
    "KAPUR_REFERENCE",
    "KAPUR_STAGE_VERSION",
    "assign_kapur_scai_stage",
    "build_biomarker_analysis_frame",
    "build_biomarker_coverage_audit",
    "build_biomarker_definition_table",
    "build_decision_curve_table",
    "build_four_hour_biomarker_features",
    "build_kapur_scai_stage_audit",
    "fit_biomarker_associations",
    "fit_biomarker_prediction_benchmark",
    "fit_cross_database_biomarker_transportability",
    "run_biomarker_benchmark",
    "summarize_biomarker_benchmark",
]
