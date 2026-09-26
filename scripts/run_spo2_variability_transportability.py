"""Run the post-hoc precision and measurement-transportability audit locally."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physiograph.analysis.shock_signal_discovery import (
    pair_domain_events,
    rolling_oliguria_events,
    sustained_hypotension_events,
    trajectory_features,
)
from physiograph.analysis.spo2_variability_transportability import (
    FROZEN_FEATURE_SPEC,
    apply_locked_timing_degradation,
    canonical_sha256,
    clinical_context_rows,
    cross_database_synthesis,
    degradation_agreement,
    derive_degradation_specification,
    feature_distribution_rows,
    frozen_exposure_from_raw,
    measurement_process_rows,
    precision_analysis,
    require_degradation_lock_before_outcomes,
    sha256_file,
    standardize_with_mimic,
    write_immutable_degradation_lock,
)
from physiograph.analysis.spo2_variability_validation import fit_fixed_scale_model

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT.parents[1]
PRIVATE = DRIVE / "Data/PhysioGraph_Biological_Discovery_20260905"
MIMIC_DB = PRIVATE / "objective_shock_single_signal_discovery.duckdb"
EICU_DB = PRIVATE / "spo2_variability_validation.duckdb"
LOCKED = ROOT / "research/spo2_variability_validation"
OUT = ROOT / "research/spo2_variability_transportability"
FIGURES = OUT / "figures"
DEGRADATION_LOCK = OUT / "measurement_degradation_lock.json"
BASE_COVARIATES = [
    "age", "male_sex", "hr_level", "sbp_level", "map_level",
    "resp_rate_level", "spo2_level", "temperature_c_level",
    "baseline_creatinine", "baseline_lactate", "lactate_observed",
    "bp_measurement_density", "urine_measurement_density",
]
DECISIONS = {
    "THIRD_COHORT_STRONGLY_JUSTIFIED",
    "THIRD_COHORT_REASONABLE_BUT_LOW_EXPECTED_YIELD",
    "MEASUREMENT_TRANSPORTABILITY_CONCERN",
    "TRUE_EFFECT_LIKELY_ATTENUATED",
    "SIGNAL_UNLIKELY_TO_JUSTIFY_FURTHER_VALIDATION",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def verify_frozen_result() -> dict[str, Any]:
    config_path = LOCKED / "external_validation_config.json"
    lock_path = LOCKED / "external_validation_lock.json"
    config = json.loads(config_path.read_text())
    lock = json.loads(lock_path.read_text())
    checks: list[dict[str, Any]] = []

    def add(item: str, actual: str, expected: str) -> None:
        checks.append({"item": item, "actual": actual, "expected": expected, "match": actual == expected})

    add("external_validation_config", sha256_file(config_path), lock["config_sha256"])
    add("protocol", sha256_file(LOCKED / "protocol.json"), config["protocol_sha256"])
    add("protocol_documentation", sha256_file(ROOT / "docs/SPO2_VARIABILITY_TARGETED_VALIDATION_PROTOCOL.md"), config["documentation_sha256"])
    for relative, expected in config["code_hashes"].items():
        add(relative, sha256_file(ROOT / relative), expected)
    if not all(row["match"] for row in checks):
        raise RuntimeError("LOCK_INTEGRITY_FAILURE")
    result = json.loads((LOCKED / "validation_results.json").read_text())
    return {
        "classification": result["classification"],
        "lock_sha256": lock["config_sha256"],
        "lock_utc": lock["lock_utc"],
        "first_eicu_association_read_utc": lock["first_eicu_association_read_utc"],
        "exposure": FROZEN_FEATURE_SPEC,
        "exposure_parameters": config["exposure_parameters"],
        "covariates": BASE_COVARIATES,
        "outcomes": {
            "primary": "constructed pressure/support plus hypoperfusion composite, >360 through 960 minutes",
            "later": "same constructed composite, >480 through 960 minutes",
        },
        "support": result["support"],
        "effects": {"mimic_primary": result["mimic"]["primary"], "mimic_later": result["mimic"]["lead"], "eicu_primary": result["primary"], "eicu_later": result["lead"]},
        "hash_checks": checks,
    }


def _load_table(database: Path, query: str) -> pd.DataFrame:
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(query).df()
    finally:
        connection.close()


def _perfusion_events(labs: pd.DataFrame, baseline: pd.DataFrame, time_column: str) -> pd.DataFrame:
    merged = labs.merge(baseline, on="stay_id", how="left")
    events: list[pd.DataFrame] = []
    for marker, group in merged.loc[merged[time_column].gt(240)].groupby("marker"):
        if marker == "lactate":
            mask = group["value"].ge(4) | (group["baseline_lactate"].notna() & group["value"].ge(2) & group["value"].sub(group["baseline_lactate"]).ge(0.5))
        elif marker == "creatinine":
            mask = group["baseline_creatinine"].notna() & (group["value"].sub(group["baseline_creatinine"]).ge(0.3) | group["value"].div(group["baseline_creatinine"]).ge(1.5))
        elif marker == "alt":
            mask = group["baseline_alt"].notna() & ((group["baseline_alt"].le(200) & group["value"].gt(200)) | group["value"].div(group["baseline_alt"]).ge(3))
        else:
            mask = group["baseline_ph"].notna() & group["baseline_ph"].ge(7.2) & group["value"].lt(7.2)
        selected = group.loc[mask, ["stay_id", time_column]].rename(columns={time_column: "event_minute"})
        selected["component"] = marker
        events.append(selected)
    return pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=["stay_id", "event_minute", "component"])


def _baseline_from_labs(labs: pd.DataFrame, time_column: str) -> pd.DataFrame:
    baseline = labs.loc[labs[time_column].le(240)].sort_values(time_column).groupby(["stay_id", "marker"]).tail(1)
    if baseline.empty:
        return pd.DataFrame(columns=["stay_id", "baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"])
    return baseline.pivot(index="stay_id", columns="marker", values="value").add_prefix("baseline_").reset_index()


def _collapse_urine(urine: pd.DataFrame) -> pd.DataFrame:
    return urine.groupby(["stay_id", "hour"], as_index=False).agg(available_minute=("available_minute", "max"), value=("value", "sum"))


def _eicu_prelandmark_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    connection = duckdb.connect(str(EICU_DB), read_only=True)
    try:
        cohort = connection.execute("SELECT * FROM cohort").df()
        periodic = connection.execute('SELECT * FROM periodic WHERE "minute" BETWEEN 0 AND 240').df()
        cuff = connection.execute('SELECT * FROM cuff_hourly WHERE "hour" BETWEEN 0 AND 6').df()
        labs = connection.execute('SELECT * FROM labs WHERE "minute" <= 360').df()
        urine = _collapse_urine(connection.execute('SELECT * FROM urine WHERE "hour" BETWEEN 0 AND 6').df())
        support = connection.execute("SELECT * FROM support WHERE event_minute <= 360 UNION ALL SELECT * FROM mcs WHERE event_minute <= 360").df()
    finally:
        connection.close()
    pressure = pd.concat([sustained_hypotension_events(cuff), support], ignore_index=True)
    baseline = _baseline_from_labs(labs, "minute")
    perfusion = pd.concat([_perfusion_events(labs, baseline, "minute"), rolling_oliguria_events(urine, start_hour=0, end_hour=6)], ignore_index=True)
    excluded = set(pressure.loc[pressure["event_minute"].le(360), "stay_id"]) | set(perfusion.loc[perfusion["event_minute"].le(360), "stay_id"])
    at_risk = cohort.loc[cohort["followup_end_offset_minutes"].gt(360) & ~cohort["death_offset_minutes"].between(0, 360).fillna(False) & ~cohort["stay_id"].isin(excluded)].copy()
    valid_periodic = periodic.loc[periodic["stay_id"].isin(at_risk["stay_id"])].copy()
    raw = valid_periodic.loc[valid_periodic["spo2"].between(50, 100), ["stay_id", "minute", "spo2"]].copy()
    hourly_parts: list[pd.DataFrame] = []
    for signal, bounds in {"spo2": (50, 100), "hr": (20, 250), "resp_rate": (4, 80), "temperature_c": (30, 43)}.items():
        part = valid_periodic.loc[valid_periodic[signal].between(*bounds), ["stay_id", "minute", signal]].copy()
        part["hour"] = np.floor(part["minute"] / 60).astype(int)
        part = part.groupby(["stay_id", "hour"], as_index=False)[signal].median().rename(columns={signal: "value"})
        part["signal"] = signal
        hourly_parts.append(part[["stay_id", "signal", "hour", "value"]])
    early_cuff = cuff.loc[cuff["stay_id"].isin(at_risk["stay_id"]) & cuff["hour"].between(0, 4) & cuff["available_minute"].le(240)]
    for signal in ("sbp", "map"):
        part = early_cuff.loc[early_cuff[signal].notna(), ["stay_id", "hour", signal]].rename(columns={signal: "value"})
        part["signal"] = signal
        hourly_parts.append(part[["stay_id", "signal", "hour", "value"]])
    trajectory = trajectory_features(pd.concat(hourly_parts, ignore_index=True))
    levels = trajectory.pivot(index="stay_id", columns="signal", values="level").add_suffix("_level").reset_index()
    early_baseline = baseline.loc[baseline["stay_id"].isin(at_risk["stay_id"])]
    context = at_risk[["stay_id", "hospital_id", "age", "male_sex"]].merge(levels, on="stay_id", how="left").merge(early_baseline, on="stay_id", how="left")
    spo2_context = raw.groupby("stay_id")["spo2"].agg(spo2_min="min", spo2_mean="mean").reset_index()
    context = context.merge(spo2_context, on="stay_id", how="left")
    return at_risk, raw, context, periodic


def load_measurement_only() -> dict[str, Any]:
    connection = duckdb.connect(str(MIMIC_DB), read_only=True)
    try:
        mimic_ids = connection.execute("SELECT stay_id FROM model_base").df()["stay_id"]
        mimic_raw = connection.execute("""SELECT v.stay_id,v.event_minute AS minute,v.value AS spo2 FROM vital_rows v JOIN model_base m USING(stay_id) WHERE v.signal='spo2' AND v.event_minute BETWEEN 0 AND 240 AND v.available_minute<=240 AND v.value BETWEEN 50 AND 100""").df()
        context_columns = ["stay_id", "age", "male_sex", "spo2_level", "sbp_level", "map_level", "hr_level", "resp_rate_level", "baseline_lactate", "baseline_creatinine"]
        mimic_context = connection.execute(f"SELECT {','.join(context_columns)} FROM model_base").df()
        locked_feature = connection.execute("SELECT stay_id,value AS raw_rms_residual,bins,span_hours FROM feature_long WHERE feature='spo2_variability'").df()
    finally:
        connection.close()
    mimic_spo2_context = mimic_raw.groupby("stay_id")["spo2"].agg(spo2_min="min", spo2_mean="mean").reset_index()
    mimic_context = mimic_context.merge(mimic_spo2_context, on="stay_id", how="left")
    eicu_at_risk, eicu_raw, eicu_context, _ = _eicu_prelandmark_inputs()
    return {"mimic_ids": mimic_ids, "mimic_raw": mimic_raw, "mimic_context": mimic_context, "mimic_locked_feature": locked_feature, "eicu_at_risk": eicu_at_risk, "eicu_raw": eicu_raw, "eicu_context": eicu_context}


def _save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _measurement_figures(mimic_stay: pd.DataFrame, eicu_stay: pd.DataFrame, mimic_feature: pd.DataFrame, eicu_feature: pd.DataFrame, hospital_feature: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.log1p(mimic_stay["raw_observations"]), bins=40, alpha=0.6, label="MIMIC", density=True)
    ax.hist(np.log1p(eicu_stay["raw_observations"]), bins=40, alpha=0.6, label="eICU", density=True)
    ax.set_xlabel("log(1 + raw SpO2 observations in minutes 0–240)"); ax.set_ylabel("Density"); ax.legend()
    _save_figure(fig, "measurement_density_comparison")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mimic_feature["standardized_exposure"].dropna(), bins=50, alpha=0.6, label="MIMIC", density=True)
    ax.hist(eicu_feature["standardized_exposure"].dropna(), bins=50, alpha=0.6, label="eICU", density=True)
    ax.set_xlabel("Frozen exposure, MIMIC SD units"); ax.set_ylabel("Density"); ax.legend()
    _save_figure(fig, "feature_distribution_comparison")
    groups: list[pd.Series] = []; labels: list[str] = []
    for dataset, frame in [("MIMIC", mimic_feature), ("eICU", eicu_feature)]:
        for bins in sorted(frame["bins"].dropna().unique()):
            values = frame.loc[frame["bins"].eq(bins), "standardized_exposure"].dropna()
            if len(values): groups.append(values); labels.append(f"{dataset}\n{int(bins)} bins")
    fig, ax = plt.subplots(figsize=(9, 4)); ax.boxplot(groups, labels=labels, showfliers=False); ax.set_ylabel("Frozen exposure, MIMIC SD units")
    _save_figure(fig, "exposure_by_bin_count")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(hospital_feature["median_exposure"].dropna(), bins=35); ax.set_xlabel("eICU hospital median frozen exposure"); ax.set_ylabel("Hospitals")
    _save_figure(fig, "eicu_hospital_distribution")


def write_precision_artifacts(verification: dict[str, Any]) -> dict[str, Any]:
    locked_result = json.loads((LOCKED / "validation_results.json").read_text())
    precision = precision_analysis(locked_result)
    write_json(OUT / "eicu_precision_analysis.json", precision)
    estimate = precision["estimate"]
    power_lines = "\n".join(f"- True RR {row['true_rr']:.2f}: {row['power']:.1%} power" for row in precision["power"])
    event_lines = "\n".join(f"- RR {row['true_rr']:.2f}, {row['target_power']:.0%} power: approximately {row['required_events']} events" for row in precision["required_events"])
    report = f"""# eICU precision audit

Execution environment: **LOCAL**. No model was refit for this phase; precision is derived from the frozen hospital-clustered estimate.

- Locked RR: {estimate['rr']:.3f}
- log(RR): {estimate['log_rr']:.6f}
- implied cluster-robust SE: {estimate['cluster_robust_se']:.6f}
- 95% CI: {estimate['ci_low']:.3f}–{estimate['ci_high']:.3f}
- multiplicative 95% CI factor: ×/{estimate['multiplicative_95_ci_factor']:.3f}
- RR-scale CI width: {estimate['rr_scale_ci_width']:.3f}

## Approximate power at 85 events

{power_lines}

## Approximate event requirements

{event_lines}

These are explicitly normal-approximation calculations holding the observed exposure distribution and hospital-clustering design effect constant through the observed robust SE. They are a precision audit, not an excuse for the failed validation and not a replacement for the frozen external decision.
"""
    (OUT / "eicu_precision_report.md").write_text(report)
    write_json(OUT / "frozen_result_verification.json", verification)
    return precision


def measurement_lock_stage() -> None:
    verification = verify_frozen_result()
    OUT.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    precision = write_precision_artifacts(verification)
    inputs = load_measurement_only()
    if len(inputs["eicu_at_risk"]) != int(verification["support"]["at_risk"]):
        raise RuntimeError("eICU at-risk reconstruction differs from frozen support count")
    mimic_features = frozen_exposure_from_raw(inputs["mimic_raw"])
    eicu_features = frozen_exposure_from_raw(inputs["eicu_raw"])
    parity = mimic_features.merge(inputs["mimic_locked_feature"], on="stay_id", suffixes=("_new", "_locked"))
    valid = parity[["raw_rms_residual_new", "raw_rms_residual_locked"]].dropna()
    maximum_difference = float((valid["raw_rms_residual_new"] - valid["raw_rms_residual_locked"]).abs().max())
    if maximum_difference > 1e-10 or int(mimic_features["raw_rms_residual"].notna().sum()) != 5957:
        raise RuntimeError("Frozen MIMIC exposure parity failed")
    if int(eicu_features["raw_rms_residual"].notna().sum()) != int(verification["support"]["spo2_observed"]):
        raise RuntimeError("Frozen eICU exposure support parity failed")
    parameters = verification["exposure_parameters"]
    measurement_mimic, mimic_stay = measurement_process_rows(inputs["mimic_raw"], inputs["mimic_ids"], "MIMIC")
    measurement_eicu, eicu_stay = measurement_process_rows(inputs["eicu_raw"], inputs["eicu_at_risk"]["stay_id"], "eICU")
    pd.concat([measurement_mimic, measurement_eicu], ignore_index=True).to_csv(OUT / "measurement_process_comparison.csv", index=False)
    feature_mimic_rows, mimic_scaled = feature_distribution_rows(mimic_features, "MIMIC", parameters)
    feature_eicu_rows, eicu_scaled = feature_distribution_rows(eicu_features, "eICU", parameters)
    hospital_feature = eicu_scaled.merge(inputs["eicu_at_risk"][["stay_id", "hospital_id"]], on="stay_id", how="left").groupby("hospital_id")["standardized_exposure"].agg(n="count", mean_exposure="mean", median_exposure="median", sd_exposure="std").reset_index()
    hospital_rows: list[dict[str, Any]] = []
    for row in hospital_feature.itertuples(index=False):
        for statistic in ("mean_exposure", "median_exposure", "sd_exposure"):
            hospital_rows.append({"dataset": "eICU", "stratum": f"hospital:{row.hospital_id}", "scale": "mimic_standardized", "statistic": statistic, "value": getattr(row, statistic), "n": row.n})
    feature_table = pd.concat([feature_mimic_rows, feature_eicu_rows, pd.DataFrame(hospital_rows)], ignore_index=True)
    feature_table.to_csv(OUT / "feature_distribution_comparison.csv", index=False)
    clinical = pd.concat([clinical_context_rows(inputs["mimic_context"], "MIMIC"), clinical_context_rows(inputs["eicu_context"], "eICU")], ignore_index=True)
    clinical.to_csv(OUT / "clinical_context_comparison.csv", index=False)
    _measurement_figures(mimic_stay, eicu_stay, mimic_scaled, eicu_scaled, hospital_feature)
    specification = derive_degradation_specification(inputs["mimic_raw"], inputs["eicu_raw"], mimic_features, eicu_features)
    lock_payload = {
        "lock_utc": datetime.now(timezone.utc).isoformat(),
        "status": "outcome_independent_measurement_degradation_specification",
        "frozen_external_validation_lock_sha256": verification["lock_sha256"],
        "phase_c_artifact_hashes": {name: sha256_file(OUT / name) for name in ["measurement_process_comparison.csv", "feature_distribution_comparison.csv", "clinical_context_comparison.csv"]},
        "exact_mimic_exposure_parity_max_abs_difference": maximum_difference,
        "specification_sha256": canonical_sha256(specification),
        "specification": specification,
    }
    whole_hash = write_immutable_degradation_lock(DEGRADATION_LOCK, lock_payload)
    DEGRADATION_LOCK.with_suffix(".sha256").write_text(whole_hash + "\n")
    write_json(OUT / "measurement_stage_status.json", {"status": "complete", "environment": "LOCAL", "precision": precision["estimate"], "mimic_valid_exposure": int(mimic_features["raw_rms_residual"].notna().sum()), "eicu_valid_exposure": int(eicu_features["raw_rms_residual"].notna().sum()), "degradation_lock_sha256": whole_hash})
    print(json.dumps(json.loads((OUT / "measurement_stage_status.json").read_text()), indent=2))


def _full_components(dataset: str) -> dict[str, Any]:
    if dataset == "MIMIC":
        connection = duckdb.connect(str(MIMIC_DB), read_only=True)
        try:
            model = connection.execute("SELECT * FROM model_base").df()
            hourly = connection.execute("SELECT * FROM vital_hourly WHERE signal IN ('sbp','map')").df()
            cuff = hourly.pivot_table(index=["stay_id", "hour"], columns="signal", values="value", aggfunc="first").reset_index()
            availability = hourly.groupby(["stay_id", "hour"], as_index=False)["available_minute"].max()
            cuff = cuff.merge(availability, on=["stay_id", "hour"], how="left")
            support = connection.execute("SELECT * FROM support_events UNION ALL SELECT * FROM mcs_events").df()
            labs = connection.execute("SELECT * FROM lab_rows").df()
            urine = connection.execute("SELECT * FROM urine_hourly").df()
        finally:
            connection.close()
        baseline = model[["stay_id", "baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"]]
        pressure = pd.concat([sustained_hypotension_events(cuff), support], ignore_index=True)
        perfusion = pd.concat([_perfusion_events(labs, baseline, "event_minute"), rolling_oliguria_events(urine)], ignore_index=True)
        at_risk = model
        hospital_column = None
    else:
        connection = duckdb.connect(str(EICU_DB), read_only=True)
        try:
            cohort = connection.execute("SELECT * FROM cohort").df()
            cuff = connection.execute("SELECT * FROM cuff_hourly").df()
            support = connection.execute("SELECT * FROM support UNION ALL SELECT * FROM mcs").df()
            labs = connection.execute("SELECT * FROM labs").df()
            urine = _collapse_urine(connection.execute("SELECT * FROM urine").df())
        finally:
            connection.close()
        baseline = _baseline_from_labs(labs, "minute")
        pressure = pd.concat([sustained_hypotension_events(cuff), support], ignore_index=True)
        perfusion = pd.concat([_perfusion_events(labs, baseline, "minute"), rolling_oliguria_events(urine, start_hour=0, end_hour=16)], ignore_index=True)
        excluded = set(pressure.loc[pressure["event_minute"].le(360), "stay_id"]) | set(perfusion.loc[perfusion["event_minute"].le(360), "stay_id"])
        at_risk = cohort.loc[cohort["followup_end_offset_minutes"].gt(360) & ~cohort["death_offset_minutes"].between(0, 360).fillna(False) & ~cohort["stay_id"].isin(excluded)].copy()
        hospital_column = "hospital_id"
    endpoint = pair_domain_events(pressure, perfusion, lower_minute=360, upper_minute=960)
    return {"at_risk": at_risk, "pressure": pressure, "perfusion": perfusion, "endpoint": endpoint, "hospital_column": hospital_column}


def _composition_rows(dataset: str, components: dict[str, Any]) -> pd.DataFrame:
    at_risk = components["at_risk"]; ids = set(at_risk["stay_id"]); endpoint = components["endpoint"].loc[lambda x: x["stay_id"].isin(ids)]
    pressure = components["pressure"].loc[lambda x: x["stay_id"].isin(ids) & x["event_minute"].gt(360) & x["event_minute"].le(960)]
    perfusion = components["perfusion"].loc[lambda x: x["stay_id"].isin(ids) & x["event_minute"].gt(360) & x["event_minute"].le(960)]
    rows = [
        {"dataset": dataset, "section": "risk_set", "component": "at_risk", "count": len(at_risk), "denominator": len(at_risk), "fraction": 1.0},
        {"dataset": dataset, "section": "endpoint", "component": "constructed_pressure_support_plus_hypoperfusion", "count": endpoint["stay_id"].nunique(), "denominator": len(at_risk), "fraction": endpoint["stay_id"].nunique() / len(at_risk)},
        {"dataset": dataset, "section": "domain", "component": "any_pressure_or_support", "count": pressure["stay_id"].nunique(), "denominator": len(at_risk), "fraction": pressure["stay_id"].nunique() / len(at_risk)},
        {"dataset": dataset, "section": "domain", "component": "any_hypoperfusion", "count": perfusion["stay_id"].nunique(), "denominator": len(at_risk), "fraction": perfusion["stay_id"].nunique() / len(at_risk)},
    ]
    for section, frame in [("pressure_component", pressure), ("hypoperfusion_component", perfusion)]:
        for component, count in frame.groupby("component")["stay_id"].nunique().items():
            rows.append({"dataset": dataset, "section": section, "component": component, "count": int(count), "denominator": len(at_risk), "fraction": float(count / len(at_risk))})
    for column, section in [("pressure_component", "endpoint_completing_pressure"), ("perfusion_component", "endpoint_completing_hypoperfusion")]:
        for component, count in endpoint[column].value_counts().items():
            rows.append({"dataset": dataset, "section": section, "component": component, "count": int(count), "denominator": len(endpoint), "fraction": float(count / len(endpoint))})
    pairs = endpoint.groupby(["pressure_component", "perfusion_component"]).size()
    for (pressure_component, perfusion_component), count in pairs.items():
        rows.append({"dataset": dataset, "section": "endpoint_component_overlap", "component": f"{pressure_component}|{perfusion_component}", "count": int(count), "denominator": len(endpoint), "fraction": float(count / len(endpoint))})
    return pd.DataFrame(rows)


def _hospital_heterogeneity(eicu_components: dict[str, Any], eicu_features: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    at_risk = eicu_components["at_risk"][["stay_id", "hospital_id"]]
    endpoint_ids = set(eicu_components["endpoint"]["stay_id"])
    valid_ids = set(eicu_features.loc[eicu_features["raw_rms_residual"].notna(), "stay_id"])
    frame = at_risk.copy(); frame["valid_exposure"] = frame["stay_id"].isin(valid_ids); frame["event"] = frame["stay_id"].isin(endpoint_ids); frame["modeled_event"] = frame["event"] & frame["valid_exposure"]
    table = frame.groupby("hospital_id").agg(at_risk_stays=("stay_id", "size"), valid_exposure_stays=("valid_exposure", "sum"), events=("event", "sum"), modeled_events=("modeled_event", "sum")).reset_index()
    table["event_prevalence"] = table["events"] / table["at_risk_stays"]
    table["modeled_event_prevalence"] = table["modeled_events"] / table["valid_exposure_stays"].replace(0, np.nan)
    ordered = table.sort_values("modeled_events", ascending=False)
    total = int(ordered["modeled_events"].sum()); summaries = []
    for percent in (0.01, 0.05, 0.10):
        hospitals = max(1, math.ceil(len(ordered) * percent))
        contribution = int(ordered.head(hospitals)["modeled_events"].sum())
        summaries.append({"top_hospital_fraction": percent, "hospitals": hospitals, "events": contribution, "fraction_of_modeled_events": contribution / total if total else math.nan})
    return table, summaries


def locked_outcome_stage() -> None:
    verification = verify_frozen_result()
    require_degradation_lock_before_outcomes(DEGRADATION_LOCK, ["objective_shock"])
    if (OUT / "mimic_degradation_results.json").exists():
        raise FileExistsError("The one-time degraded MIMIC outcome fit already exists")
    lock_hash = sha256_file(DEGRADATION_LOCK)
    lock = json.loads(DEGRADATION_LOCK.read_text())
    if canonical_sha256(lock["specification"]) != lock["specification_sha256"]:
        raise RuntimeError("Degradation specification hash mismatch")
    inputs = load_measurement_only()
    mimic_original = frozen_exposure_from_raw(inputs["mimic_raw"])
    eicu_original = frozen_exposure_from_raw(inputs["eicu_raw"])
    valid_mimic_ids = mimic_original.loc[mimic_original["raw_rms_residual"].notna(), "stay_id"]
    degraded_raw = apply_locked_timing_degradation(inputs["mimic_raw"], inputs["eicu_raw"], valid_mimic_ids, lock["specification"])
    degraded_feature = frozen_exposure_from_raw(degraded_raw)
    agreement = degradation_agreement(mimic_original, degraded_feature)
    connection = duckdb.connect(str(MIMIC_DB), read_only=True)
    try:
        model = connection.execute("SELECT * FROM model_base").df()
    finally:
        connection.close()
    require_degradation_lock_before_outcomes(DEGRADATION_LOCK, list(model.columns))
    result_frame = model.merge(degraded_feature[["stay_id", "raw_rms_residual"]].rename(columns={"raw_rms_residual": "spo2_variability_degraded"}), on="stay_id", how="left")
    receipt = {"outcome_read_utc": datetime.now(timezone.utc).isoformat(), "degradation_lock_sha256": lock_hash, "fit": "single frozen MIMIC primary model with degraded exposure", "attempt": 1}
    write_json(OUT / "degradation_outcome_read_receipt.json", receipt)
    degraded_model, _ = fit_fixed_scale_model(result_frame, feature="spo2_variability_degraded", outcome="objective_shock", adjustment_covariates=BASE_COVARIATES, exposure_parameters=verification["exposure_parameters"])
    original = verification["effects"]["mimic_primary"]
    attenuation = 100 * (1 - degraded_model["log_rr"] / original["log_rr"])
    degradation_result = {"status": "POST_HOC_MEASUREMENT_PROCESS_DIAGNOSTIC", "degradation_lock_sha256": lock_hash, "agreement": agreement, "original_locked_mimic": original, "degraded_mimic_single_fit": degraded_model, "percent_attenuation_of_log_rr": attenuation, "outcome_fit_count": 1, "interpretation_guard": "The degradation rule was not modified after this effect estimate."}
    write_json(OUT / "mimic_degradation_results.json", degradation_result)
    mimic_scaled = standardize_with_mimic(mimic_original["raw_rms_residual"], verification["exposure_parameters"]); mimic_plot = mimic_original[["stay_id"]].join(mimic_scaled["standardized_exposure"]).rename(columns={"standardized_exposure": "original"})
    degraded_scaled = standardize_with_mimic(degraded_feature["raw_rms_residual"], verification["exposure_parameters"]); degraded_plot = degraded_feature[["stay_id"]].join(degraded_scaled["standardized_exposure"]).rename(columns={"standardized_exposure": "degraded"})
    paired = mimic_plot.merge(degraded_plot, on="stay_id").dropna(); fig, ax = plt.subplots(figsize=(5, 5)); ax.hexbin(paired["original"], paired["degraded"], gridsize=40, mincnt=1); limits = [float(paired[["original", "degraded"]].min().min()), float(paired[["original", "degraded"]].max().max())]; ax.plot(limits, limits, color="red", lw=1); ax.set_xlabel("Original frozen exposure"); ax.set_ylabel("Degraded frozen exposure"); _save_figure(fig, "original_vs_degraded_mimic")
    mimic_components = _full_components("MIMIC"); eicu_components = _full_components("eICU")
    composition = pd.concat([_composition_rows("MIMIC", mimic_components), _composition_rows("eICU", eicu_components)], ignore_index=True); composition.to_csv(OUT / "outcome_composition_comparison.csv", index=False)
    hospital_table, hospital_summary = _hospital_heterogeneity(eicu_components, eicu_original); hospital_table.to_csv(OUT / "eicu_hospital_heterogeneity.csv", index=False)
    synthesis = cross_database_synthesis(json.loads((LOCKED / "validation_results.json").read_text())); write_json(OUT / "cross_database_effect_synthesis.json", synthesis)
    studies = synthesis["studies"]; fixed = synthesis["fixed_effect"]; fig, ax = plt.subplots(figsize=(7, 4)); labels = [row["dataset"] for row in studies] + ["Fixed-effect summary\n(not validation)"]; rr = [verification["effects"]["mimic_primary"]["rr_per_mimic_sd"], verification["effects"]["eicu_primary"]["rr_per_mimic_sd"], fixed["rr"]]; lo = [verification["effects"]["mimic_primary"]["ci_low"], verification["effects"]["eicu_primary"]["ci_low"], fixed["ci_low"]]; hi = [verification["effects"]["mimic_primary"]["ci_high"], verification["effects"]["eicu_primary"]["ci_high"], fixed["ci_high"]]; y = np.arange(len(labels))[::-1]; ax.errorbar(rr, y, xerr=[np.array(rr)-np.array(lo), np.array(hi)-np.array(rr)], fmt="o"); ax.axvline(1, color="black", lw=0.8); ax.set_yticks(y, labels); ax.set_xlabel("Adjusted RR per frozen MIMIC SD"); _save_figure(fig, "locked_mimic_eicu_forest")
    write_json(OUT / "outcome_stage_status.json", {"status": "complete", "environment": "LOCAL", "degradation": degradation_result, "hospital_event_concentration": hospital_summary, "mimic_endpoint_events": int(mimic_components["endpoint"]["stay_id"].nunique()), "eicu_endpoint_events": int(eicu_components["endpoint"]["stay_id"].nunique())})
    print(json.dumps(json.loads((OUT / "outcome_stage_status.json").read_text()), indent=2))


def resume_reporting_after_locked_fit() -> None:
    """Finish descriptive artifacts without repeating the one-time degraded fit."""
    verification = verify_frozen_result()
    require_degradation_lock_before_outcomes(DEGRADATION_LOCK, ["objective_shock"])
    degradation_path = OUT / "mimic_degradation_results.json"
    receipt_path = OUT / "degradation_outcome_read_receipt.json"
    if not degradation_path.exists() or not receipt_path.exists():
        raise RuntimeError("No completed one-time degradation fit is available to resume")
    degradation_result = json.loads(degradation_path.read_text())
    if degradation_result.get("outcome_fit_count") != 1:
        raise RuntimeError("Expected exactly one completed degradation fit")
    receipt = json.loads(receipt_path.read_text())
    lock_hash = sha256_file(DEGRADATION_LOCK)
    if receipt.get("degradation_lock_sha256") != lock_hash or degradation_result.get("degradation_lock_sha256") != lock_hash:
        raise RuntimeError("Completed degradation fit does not match the immutable lock")
    inputs = load_measurement_only()
    eicu_original = frozen_exposure_from_raw(inputs["eicu_raw"])
    mimic_components = _full_components("MIMIC")
    eicu_components = _full_components("eICU")
    composition = pd.concat(
        [_composition_rows("MIMIC", mimic_components), _composition_rows("eICU", eicu_components)],
        ignore_index=True,
    )
    composition.to_csv(OUT / "outcome_composition_comparison.csv", index=False)
    hospital_table, hospital_summary = _hospital_heterogeneity(eicu_components, eicu_original)
    hospital_table.to_csv(OUT / "eicu_hospital_heterogeneity.csv", index=False)
    synthesis = cross_database_synthesis(json.loads((LOCKED / "validation_results.json").read_text()))
    write_json(OUT / "cross_database_effect_synthesis.json", synthesis)
    fixed = synthesis["fixed_effect"]
    labels = [row["dataset"] for row in synthesis["studies"]] + ["Fixed-effect summary\n(not validation)"]
    rr = [verification["effects"]["mimic_primary"]["rr_per_mimic_sd"], verification["effects"]["eicu_primary"]["rr_per_mimic_sd"], fixed["rr"]]
    lo = [verification["effects"]["mimic_primary"]["ci_low"], verification["effects"]["eicu_primary"]["ci_low"], fixed["ci_low"]]
    hi = [verification["effects"]["mimic_primary"]["ci_high"], verification["effects"]["eicu_primary"]["ci_high"], fixed["ci_high"]]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(rr, y, xerr=[np.array(rr) - np.array(lo), np.array(hi) - np.array(rr)], fmt="o")
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Adjusted RR per frozen MIMIC SD")
    _save_figure(fig, "locked_mimic_eicu_forest")
    write_json(
        OUT / "outcome_stage_status.json",
        {
            "status": "complete_after_interrupted_reporting_resume",
            "environment": "LOCAL",
            "degradation": degradation_result,
            "hospital_event_concentration": hospital_summary,
            "mimic_endpoint_events": int(mimic_components["endpoint"]["stay_id"].nunique()),
            "eicu_endpoint_events": int(eicu_components["endpoint"]["stay_id"].nunique()),
            "degraded_outcome_model_refit_during_resume": False,
        },
    )
    write_json(
        OUT / "execution_continuation.json",
        {
            "reason": "The tool session ended after the immutable one-time degradation result and its outcome-read receipt were written, before descriptive reporting completed.",
            "preserved_degradation_result_sha256": sha256_file(degradation_path),
            "preserved_outcome_receipt_sha256": sha256_file(receipt_path),
            "degraded_outcome_model_refit": False,
            "resume_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(json.loads((OUT / "outcome_stage_status.json").read_text()), indent=2))


def repair_outcome_status_counts() -> None:
    """Repair risk-set denominators in status metadata from the validated CSV."""
    composition = pd.read_csv(OUT / "outcome_composition_comparison.csv")
    status_path = OUT / "outcome_stage_status.json"
    status = json.loads(status_path.read_text())
    endpoint = composition.loc[
        composition["section"].eq("endpoint")
        & composition["component"].eq("constructed_pressure_support_plus_hypoperfusion")
    ].set_index("dataset")["count"]
    mimic_count = int(endpoint["MIMIC"])
    eicu_count = int(endpoint["eICU"])
    if mimic_count != 238 or eicu_count != 95:
        raise RuntimeError("Frozen endpoint parity failed in composition CSV")
    status["mimic_endpoint_events"] = mimic_count
    status["eicu_endpoint_events"] = eicu_count
    status["status_metadata_correction"] = "risk-set-restricted counts read from validated composition CSV; no model refit"
    write_json(status_path, status)
    write_json(
        OUT / "implementation_correction_001.json",
        {
            "scope": "descriptive status metadata only",
            "failure": "Initial outcome-stage status counted paired-domain events before restricting to the frozen at-risk cohorts.",
            "invalid_status_values": {"MIMIC": 1716, "eICU": 377},
            "validated_risk_set_values": {"MIMIC": mimic_count, "eICU": eicu_count},
            "source": "outcome_composition_comparison.csv",
            "association_models_refit": False,
            "degradation_rule_changed": False,
            "frozen_validation_changed": False,
        },
    )
    print(json.dumps(status, indent=2))


def _third_cohort_protocol(decision: str, verification: dict[str, Any]) -> str:
    parameters = verification["exposure_parameters"]
    return f"""# Frozen third-cohort validation protocol template

Status: **FUTURE, NOT EXECUTED**  
Decision basis: `{decision}`  
Created after the completed MIMIC/eICU audit. No third-cohort outcome was inspected.

## Eligibility

- Adults with explicit heart failure, using the first ICU stay per hospital encounter.
- Alive and observable beyond ICU minute 360.
- Exclude pressure/support or hypoperfusion events through minute 360 using the same frozen component definitions.
- Require at least three hourly SpO2 bins spanning at least two hours in minutes 0–240.

## Frozen exposure

- Restrict SpO2 to 50–100%.
- Compute hourly medians during minutes 0–240.
- Fit ordinary least squares SpO2 on hour within each stay.
- Exposure is the root-mean-square residual around that line.
- Clip at `{parameters['lower']}` and `{parameters['upper']}`.
- Standardize with MIMIC center `{parameters['center']}` and SD `{parameters['scale']}`.
- No dataset-specific centering, scaling, threshold optimization, or alternative SpO2 feature.

## Outcomes and chronology

- Primary: the constructed pressure/support-plus-hypoperfusion composite strictly after minute 360 through 960, with domains paired within six hours.
- Later-window sensitivity: the same endpoint strictly after minute 480 through 960.
- This is not adjudicated cardiogenic shock.

## Adjustment and missingness

- Preserve the clinical adjustment philosophy: age, sex, latest pre-240-minute HR, cuff SBP, cuff MAP, respiratory rate, SpO2 level, temperature, creatinine, lactate, lactate-observed status, BP density, and urine-output density.
- Missing continuous covariates use the same deterministic preparation as the frozen validation; missing outcomes remain missing rather than negative.
- Use the MIMIC exposure transform without refitting it.

## Inference and support gates

- Modified Poisson RR per frozen MIMIC SD.
- Cluster-robust variance by hospital/site; HC0 only as a labeled sensitivity.
- Minimum 70% exposure coverage, 50 modeled primary events, and 20 contributing hospitals.
- Confirmation requires RR >1, two-sided clustered 95% CI excluding 1, p<0.05, and later-window RR >1.
- Stop without rescue if feasibility gates fail or the primary confirmation rule fails.
- No subgroup, hospital, endpoint, time-window, feature, or threshold search.

## Future secondary benchmark

Run only if basic exposure/outcome feasibility passes, on identical patients for every comparison:

1. Clinical baseline
2. Clinical + baseline lactate
3. Clinical + frozen SpO2 variability
4. Clinical + baseline lactate + frozen SpO2 variability

Pre-specify a lactate-normal subgroup as baseline lactate <2.0 mmol/L. This benchmark is secondary, does not alter the primary RR decision, and must not be run in unavailable data.
"""


def finalize(decision: str) -> None:
    if decision not in DECISIONS:
        raise ValueError(f"Decision must be one of {sorted(DECISIONS)}")
    verification = verify_frozen_result(); precision = json.loads((OUT / "eicu_precision_analysis.json").read_text()); degradation = json.loads((OUT / "mimic_degradation_results.json").read_text()); degradation_lock = json.loads(DEGRADATION_LOCK.read_text()); synthesis = json.loads((OUT / "cross_database_effect_synthesis.json").read_text()); outcome_status = json.loads((OUT / "outcome_stage_status.json").read_text())
    power = {str(row["true_rr"]): row["power"] for row in precision["power"]}
    evidence = {
        "eicu_power_rr_1_20": power["1.2"], "eicu_power_rr_1_25": power["1.25"],
        "eicu_ci_contains_mimic_rr": verification["effects"]["eicu_primary"]["ci_low"] <= verification["effects"]["mimic_primary"]["rr_per_mimic_sd"] <= verification["effects"]["eicu_primary"]["ci_high"],
        "directionally_concordant": verification["effects"]["eicu_primary"]["rr_per_mimic_sd"] > 1,
        "mimic_bootstrap_positive_fraction": json.loads((LOCKED / "mimic_targeted_followup.json").read_text())["bootstrap"]["sign_consistency"],
        "degradation_percent_log_rr_attenuation": degradation["percent_attenuation_of_log_rr"],
        "degradation_pearson": degradation["agreement"]["pearson_correlation"],
        "database_interaction_p": synthesis["database_by_exposure_interaction"]["p_value"],
    }
    rationale_map = {
        "THIRD_COHORT_STRONGLY_JUSTIFIED": "eICU precision is demonstrably modest for RR 1.20–1.25, its interval contains the MIMIC effect, direction is concordant, and the MIMIC signal is bootstrap-stable; a higher-event, high-fidelity cohort can materially adjudicate transportability.",
        "THIRD_COHORT_REASONABLE_BUT_LOW_EXPECTED_YIELD": "Residual uncertainty remains, but the combined precision, measurement, and degradation evidence suggests limited expected information gain.",
        "MEASUREMENT_TRANSPORTABILITY_CONCERN": "Outcome-independent measurement differences and the locked degradation experiment show material feature unreliability or log-effect attenuation, so eICU is not a clean measurement-equivalent replication.",
        "TRUE_EFFECT_LIKELY_ATTENUATED": "Measurement processes are sufficiently comparable and degradation does not materially attenuate the MIMIC effect, while eICU points to a smaller effect.",
        "SIGNAL_UNLIKELY_TO_JUSTIFY_FURTHER_VALIDATION": "eICU was sufficiently precise for clinically relevant effects and neither measurement degradation nor case mix plausibly explains the null-compatible estimate.",
    }
    decision_artifact = {"classification": decision, "rationale": rationale_map[decision], "evidence": evidence, "guard": "The frozen eICU result remains POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED."}
    write_json(OUT / "third_cohort_decision.json", decision_artifact)
    if decision != "SIGNAL_UNLIKELY_TO_JUSTIFY_FURTHER_VALIDATION":
        (OUT / "THIRD_COHORT_VALIDATION_PROTOCOL.md").write_text(_third_cohort_protocol(decision, verification))
    else:
        (OUT / "THIRD_COHORT_VALIDATION_PROTOCOL.md").write_text("# Third-cohort protocol\n\nNot created because the audit classified the signal as unlikely to justify further validation.\n")
    fixed = synthesis["fixed_effect"]; random = synthesis["random_effects_dl"]; hospital = outcome_status["hospital_event_concentration"]
    measurement = pd.read_csv(OUT / "measurement_process_comparison.csv")
    feature = pd.read_csv(OUT / "feature_distribution_comparison.csv")
    context = pd.read_csv(OUT / "clinical_context_comparison.csv")
    composition = pd.read_csv(OUT / "outcome_composition_comparison.csv")

    def measurement_value(dataset: str, metric: str, statistic: str) -> float:
        row = measurement.loc[(measurement["dataset"] == dataset) & (measurement["metric"] == metric) & (measurement["statistic"] == statistic), "value"]
        if len(row) != 1:
            raise RuntimeError(f"Expected one measurement value for {dataset}/{metric}/{statistic}; found {len(row)}")
        return float(row.iloc[0])

    def feature_value(dataset: str, scale: str, statistic: str) -> float:
        row = feature.loc[(feature["dataset"] == dataset) & (feature["stratum"] == "all_valid") & (feature["scale"] == scale) & (feature["statistic"] == statistic), "value"]
        if len(row) != 1:
            raise RuntimeError(f"Expected one feature value for {dataset}/{scale}/{statistic}; found {len(row)}")
        return float(row.iloc[0])

    def context_value(dataset: str, variable: str, statistic: str) -> float:
        row = context.loc[(context["dataset"] == dataset) & (context["variable"] == variable) & (context["statistic"] == statistic), "value"]
        if len(row) != 1:
            raise RuntimeError(f"Expected one context value for {dataset}/{variable}/{statistic}; found {len(row)}")
        return float(row.iloc[0])

    def outcome_fraction(dataset: str, section: str, component: str) -> float:
        row = composition.loc[(composition["dataset"] == dataset) & (composition["section"] == section) & (composition["component"] == component), "fraction"]
        if len(row) != 1:
            raise RuntimeError(f"Expected one outcome fraction for {dataset}/{section}/{component}; found {len(row)}")
        return float(row.iloc[0])

    required = {(row["true_rr"], row["target_power"]): row["required_events"] for row in precision["required_events"]}
    report = f"""# SpO2-variability transportability and precision audit

Execution environment: **LOCAL**. This is a post-hoc audit, not a rescue analysis. The frozen result remains **POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED**.

## Decision

**{decision}**

{rationale_map[decision]}

This decision does **not** reclassify the frozen eICU analysis. It answers only whether another locked validation is worth the access and engineering cost.

## Frozen analysis guard

- Feature: `spo2_variability` from hours 0–4 only, with hourly medians, a linear residual over observed bins, RMS residual, frozen 1st/99th-percentile winsorization, and MIMIC center/scale.
- Frozen MIMIC RR: {verification['effects']['mimic_primary']['rr_per_mimic_sd']:.3f} ({verification['effects']['mimic_primary']['ci_low']:.3f}–{verification['effects']['mimic_primary']['ci_high']:.3f}); frozen eICU RR: {verification['effects']['eicu_primary']['rr_per_mimic_sd']:.3f} ({verification['effects']['eicu_primary']['ci_low']:.3f}–{verification['effects']['eicu_primary']['ci_high']:.3f}).
- All frozen protocol, configuration, documentation, and code hashes passed. No alternate feature, threshold, subgroup, hospital exclusion, or outcome definition was fitted.

## Precision

The eICU log RR was {precision['estimate']['log_rr']:.4f} with hospital-cluster-robust SE {precision['estimate']['cluster_robust_se']:.4f}. Its RR interval width was {precision['estimate']['rr_scale_ci_width']:.3f}, equivalent to a multiplicative 95% CI factor of {precision['estimate']['multiplicative_95_ci_factor']:.3f}. With 85 modeled events, approximate two-sided power was {power['1.1']:.1%}, {power['1.15']:.1%}, {power['1.2']:.1%}, {power['1.25']:.1%}, and {power['1.3']:.1%} for true RRs 1.10, 1.15, 1.20, 1.25, and 1.30, respectively. A true RR of 1.20 would require approximately {required[(1.2, 0.8)]} events for 80% power and {required[(1.2, 0.9)]} for 90%; RR 1.25 would require {required[(1.25, 0.8)]} and {required[(1.25, 0.9)]}, respectively.

These are **normal approximations using the observed clustered SE and inverse event scaling**, not simulation-based guarantees. They show that the eICU result was not precise enough to separate a small effect near 1.10 from a clinically relevant effect near 1.20–1.25.

## Measurement process and feature transport

The databases did not differ by simple scarcity in eICU. Median raw SpO2 observations per stay were {measurement_value('MIMIC', 'raw_observations_per_stay', 'median'):.0f} in MIMIC and {measurement_value('eICU', 'raw_observations_per_stay', 'median'):.0f} in eICU; median spacing was {measurement_value('MIMIC', 'time_between_observations_minutes', 'median'):.0f} versus {measurement_value('eICU', 'time_between_observations_minutes', 'median'):.0f} minutes, and median observations within an occupied hour were {measurement_value('MIMIC', 'within_hour_observations', 'median'):.0f} versus {measurement_value('eICU', 'within_hour_observations', 'median'):.0f}. However, consecutive identical readings were {measurement_value('MIMIC', 'consecutive_observations_identical', 'fraction'):.1%} versus {measurement_value('eICU', 'consecutive_observations_identical', 'fraction'):.1%}, and the median maximum identical run was {measurement_value('MIMIC', 'maximum_identical_run_per_stay', 'median'):.0f} versus {measurement_value('eICU', 'maximum_identical_run_per_stay', 'median'):.0f}. Thus eICU was denser but substantially more repetitive, consistent with a different charting/device process rather than simply better temporal resolution.

On the frozen MIMIC scale, the MIMIC exposure had mean {feature_value('MIMIC', 'mimic_standardized', 'mean'):.3f}, SD {feature_value('MIMIC', 'mimic_standardized', 'sd'):.3f}, median {feature_value('MIMIC', 'mimic_standardized', 'median'):.3f}, and IQR {feature_value('MIMIC', 'mimic_standardized', 'iqr'):.3f}; eICU had mean {feature_value('eICU', 'mimic_standardized', 'mean'):.3f}, SD {feature_value('eICU', 'mimic_standardized', 'sd'):.3f}, median {feature_value('eICU', 'mimic_standardized', 'median'):.3f}, and IQR {feature_value('eICU', 'mimic_standardized', 'iqr'):.3f}. eICU therefore showed a lower, narrower frozen-feature distribution. Hospital-specific summaries are preserved in `feature_distribution_comparison.csv`; no hospital was excluded.

## Measurement degradation

The outcome-independent eICU timing-template operator was fully specified and hashed before outcome access. It sampled empirical eICU timing templates with replacement under seed {degradation_lock['specification']['seed']} and mapped each scheduled time to the latest actual MIMIC value at or before that time (earliest later value only when no prior value existed); it added no Gaussian noise and introduced no arbitrary tuning constant. Across {degradation['agreement']['n_paired']:,} paired MIMIC stays, original-versus-degraded Pearson correlation was {degradation['agreement']['pearson_correlation']:.3f}, Spearman correlation was {degradation['agreement']['spearman_rank_correlation']:.3f}, reliability slope was {degradation['agreement']['reliability_slope_degraded_on_original']:.3f}, exact-quartile preservation was {degradation['agreement']['same_quartile_fraction']:.1%}, and {degradation['agreement']['moved_two_or_more_quartiles_fraction']:.1%} moved by at least two quartiles.

The locked original MIMIC RR was {degradation['original_locked_mimic']['rr_per_mimic_sd']:.3f}; the **single permitted** degraded-exposure fit gave RR {degradation['degraded_mimic_single_fit']['rr_per_mimic_sd']:.3f} ({degradation['degraded_mimic_single_fit']['ci_low']:.3f}–{degradation['degraded_mimic_single_fit']['ci_high']:.3f}), p={degradation['degraded_mimic_single_fit']['p_value']:.3f}. Log-RR attenuation was {degradation['percent_attenuation_of_log_rr']:.1f}%. This is strong evidence that the frozen statistic is sensitive to measurement-process transport, but it is a post-hoc diagnostic and does not prove that measurement alone caused the eICU attenuation.

## Outcome and case mix

The constructed endpoint is pressure/support plus hypoperfusion, **not adjudicated cardiogenic shock**. MIMIC had {outcome_status['mimic_endpoint_events']} events among 8,196 at-risk stays ({outcome_fraction('MIMIC', 'endpoint', 'constructed_pressure_support_plus_hypoperfusion'):.2%}); eICU had {outcome_status['eicu_endpoint_events']} among 8,743 ({outcome_fraction('eICU', 'endpoint', 'constructed_pressure_support_plus_hypoperfusion'):.2%}). Sustained hypotension completed the pressure side in {outcome_fraction('MIMIC', 'endpoint_completing_pressure', 'sustained_hypotension'):.1%} of MIMIC endpoints and {outcome_fraction('eICU', 'endpoint_completing_pressure', 'sustained_hypotension'):.1%} of eICU endpoints. Oliguria completed the perfusion side in {outcome_fraction('MIMIC', 'endpoint_completing_hypoperfusion', 'oliguria'):.1%} versus {outcome_fraction('eICU', 'endpoint_completing_hypoperfusion', 'oliguria'):.1%}; lactate did so in {outcome_fraction('MIMIC', 'endpoint_completing_hypoperfusion', 'lactate'):.1%} versus {outcome_fraction('eICU', 'endpoint_completing_hypoperfusion', 'lactate'):.1%}. These differences show meaningful outcome-composition and event-rate transport issues.

Age medians were {context_value('MIMIC', 'age', 'median'):.0f} and {context_value('eICU', 'age', 'median'):.0f}; baseline lactate medians were {context_value('MIMIC', 'baseline_lactate', 'median'):.1f} and {context_value('eICU', 'baseline_lactate', 'median'):.1f}, but availability differed ({context_value('MIMIC', 'baseline_lactate', 'availability'):.1%} versus {context_value('eICU', 'baseline_lactate', 'availability'):.1%}). Ventilation, FiO2, and oxygen-support variables were not harmonized in either frozen frame, so respiratory context and heart-failure severity could **not** be directly adjudicated. No unsupported clinical equivalence claim is made.

Among the 85 modeled eICU events, the top 1%, 5%, and 10% of hospitals contributed {hospital[0]['fraction_of_modeled_events']:.1%}, {hospital[1]['fraction_of_modeled_events']:.1%}, and {hospital[2]['fraction_of_modeled_events']:.1%}, respectively. This concentration is descriptively important; an association refit after dropping hospitals was intentionally not performed because hospital exclusion was prohibited. Full component and hospital tables are retained in `outcome_composition_comparison.csv` and `eicu_hospital_heterogeneity.csv`.

## Cross-database synthesis

The fixed-effect descriptive summary RR was {fixed['rr']:.3f} ({fixed['ci_low']:.3f}–{fixed['ci_high']:.3f}); the two-study DerSimonian–Laird estimate was {random['rr']:.3f} ({random['ci_low']:.3f}–{random['ci_high']:.3f}). The database-by-exposure interaction test gave z={synthesis['database_by_exposure_interaction']['z']:.3f}, p={synthesis['database_by_exposure_interaction']['p_value']:.3f}. Q={synthesis['heterogeneity']['q']:.3f}, I²={synthesis['heterogeneity']['i_squared_percent']:.1f}%, and tau²={random['tau_squared']:.4f}. These heterogeneity estimates are unstable and weakly informative with only two datasets.

## Third-cohort requirements

A third cohort should be pursued only with the frozen protocol in `THIRD_COHORT_VALIDATION_PROTOCOL.md`, high-fidelity timestamped SpO2 provenance, the same outcome construction, and site-clustered inference. The precision target should be at least {required[(1.2, 0.8)]} modeled endpoint events for approximately 80% power at RR 1.20 (preferably {required[(1.2, 0.9)]} for 90%), not merely a large patient count. Feasibility and support gates must be assessed without reading the exposure–outcome association; a failed confirmatory test ends the program without rescue.

**Pooling does not convert a failed external validation into successful external validation.** Neither this audit nor the frozen study establishes causality, clinical utility, or adjudicated cardiogenic shock.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    summary = {"status": "complete", "environment": "LOCAL", "frozen_validation_classification": verification["classification"], "transportability_decision": decision, "precision": precision, "degradation": degradation, "synthesis": synthesis, "outcome_hospital_summary": outcome_status, "artifact_hashes": {path.name: sha256_file(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "transportability_summary.json"}}
    write_json(OUT / "transportability_summary.json", summary)
    print(json.dumps(decision_artifact, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--measurement-lock", action="store_true")
    group.add_argument("--run-locked-outcomes", action="store_true")
    group.add_argument("--resume-reporting", action="store_true")
    group.add_argument("--repair-status-counts", action="store_true")
    group.add_argument("--finalize", choices=sorted(DECISIONS))
    args = parser.parse_args()
    if args.measurement_lock:
        measurement_lock_stage()
    elif args.run_locked_outcomes:
        locked_outcome_stage()
    elif args.resume_reporting:
        resume_reporting_after_locked_fit()
    elif args.repair_status_counts:
        repair_outcome_status_counts()
    else:
        finalize(args.finalize)


if __name__ == "__main__":
    main()
