"""Run one prospectively amended NWICU SpO2-variability analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physiograph.analysis.shock_signal_discovery import (
    pair_domain_events,
    sustained_hypotension_events,
    trajectory_features,
)
from physiograph.analysis.spo2_variability_nwicu_validation import (
    FROZEN_EXPOSURE_PARAMETERS,
    MODIFIED_ENDPOINT,
    NWICU_MAPPING,
    assert_association_authorized,
    canonical_sha256,
    compute_frozen_exposure,
    evaluate_frozen_feasibility,
    lock_payload,
    sha256_file,
    verify_source_manifest,
)
from physiograph.analysis.spo2_variability_validation import fit_fixed_scale_model

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT.parents[1]
NWICU = DRIVE / "Data/NWICU"
DATA = NWICU / "data"
OUT = ROOT / "research/spo2_variability_nwicu_validation"
PROTOCOL = OUT / "protocol_amendment.json"
DOCUMENTATION = ROOT / "docs/SPO2_VARIABILITY_NWICU_MODIFIED_ENDPOINT.md"
LOCKED_THIRD_PROTOCOL = ROOT / "research/spo2_variability_transportability/THIRD_COHORT_VALIDATION_PROTOCOL.md"
FROZEN_EXTERNAL_LOCK = ROOT / "research/spo2_variability_validation/external_validation_lock.json"
LOCK = OUT / "analysis_lock.json"
RESULTS = OUT / "validation_results.json"
CODE_FILES = [
    "scripts/run_spo2_variability_nwicu_validation.py",
    "src/physiograph/analysis/spo2_variability_nwicu_validation.py",
    "src/physiograph/analysis/spo2_variability_transportability.py",
    "src/physiograph/analysis/spo2_variability_validation.py",
    "src/physiograph/analysis/shock_signal_discovery.py",
    "tests/unit/test_spo2_variability_nwicu_validation.py",
]
BASE_COVARIATES = [
    "age", "male_sex", "hr_level", "sbp_level", "map_level",
    "resp_rate_level", "spo2_level", "temperature_c_level",
    "baseline_creatinine", "baseline_lactate", "lactate_observed",
    "bp_measurement_density", "urine_measurement_density",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_path(group: str, name: str) -> Path:
    return DATA / group / f"{name}.csv.gz"


def csv_scan(path: Path) -> str:
    return f"read_csv({sql_path(path)},header=true,all_varchar=true,quote='\"',escape='\"',sample_size=10000)"


def protocol_payload() -> dict[str, Any]:
    return {
        "status": "prospectively_amended_before_nwicu_association",
        "dataset": "NWICU",
        "modified_endpoint": MODIFIED_ENDPOINT,
        "mapping": NWICU_MAPPING,
        "exposure_parameters": FROZEN_EXPOSURE_PARAMETERS,
        "waivers": {
            "minimum_70_percent_exposure_coverage": True,
            "minimum_20_contributing_sites": True,
            "minimum_50_modeled_primary_events": False,
        },
        "covariance": "HC0",
        "association_inspected": False,
    }


def verify_frozen_ancestors() -> dict[str, str]:
    frozen = json.loads(FROZEN_EXTERNAL_LOCK.read_text())
    config = ROOT / "research/spo2_variability_validation/external_validation_config.json"
    if sha256_file(config) != frozen["config_sha256"]:
        raise RuntimeError("Frozen external validation lock mismatch")
    return {
        "frozen_external_validation_config_sha256": frozen["config_sha256"],
        "third_cohort_protocol_sha256": sha256_file(LOCKED_THIRD_PROTOCOL),
    }


def verify_dictionary_mapping() -> dict[str, Any]:
    """Fail closed if the supplied NWICU dictionaries do not match the amendment."""
    item_dictionary = pd.read_csv(source_path("nw_icu", "d_items"), compression="gzip")
    lab_dictionary = pd.read_csv(source_path("nw_hosp", "d_labitems"), compression="gzip")
    expected_items = {
        320045: "PULSE",
        320179: "BP SYSTOLIC",
        320180: "BP DIASTOLIC",
        320210: "RESPIRATIONS",
        320277: "PULSE OXIMETRY",
        323761: "TEMPERATURE",
        736876: "ECMO PUMP SETTINGS",
    }
    expected_labs = {
        100002: ("Creatinine", "Blood"),
        100031: ("Lactate", "Blood"),
        100042: ("Alanine Aminotransferase (ALT)", "Blood"),
        100339: ("pH", "Blood"),
    }
    actual_items = item_dictionary.loc[item_dictionary["itemid"].isin(expected_items)].groupby("itemid")["label"].apply(lambda values: sorted(set(values))).to_dict()
    for itemid, expected in expected_items.items():
        if expected not in actual_items.get(itemid, []):
            raise RuntimeError(f"NWICU item mapping mismatch for {itemid}: expected {expected}")
    actual_labs = lab_dictionary.set_index("itemid")[["label", "fluid"]].to_dict("index")
    for itemid, (label, fluid) in expected_labs.items():
        actual = actual_labs.get(itemid)
        if actual != {"label": label, "fluid": fluid}:
            raise RuntimeError(f"NWICU lab mapping mismatch for {itemid}: expected {label}/{fluid}")
    return {
        "vital_and_mcs_items": {str(key): value for key, value in expected_items.items()},
        "laboratory_items": {str(key): {"label": value[0], "fluid": value[1]} for key, value in expected_labs.items()},
        "map_cuff_item": None,
        "continuous_infusion_rate_source": None,
        "urine_output_source": None,
        "site_identifier": None,
    }


def write_lock() -> None:
    if LOCK.exists() or RESULTS.exists():
        raise FileExistsError("NWICU analysis lock or result already exists")
    source_checks = verify_source_manifest(NWICU)
    dictionary_verification = verify_dictionary_mapping()
    ancestors = verify_frozen_ancestors()
    protocol = protocol_payload()
    write_json(PROTOCOL, protocol)
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in CODE_FILES}
    payload = lock_payload(
        protocol_sha256=sha256_file(PROTOCOL),
        frozen_external_lock_sha256=ancestors["frozen_external_validation_config_sha256"],
        code_hashes=code_hashes,
        source_manifest_sha256=sha256_file(NWICU / "SHA256SUMS.txt"),
    )
    payload.update(
        {
            "lock_utc": utc_now(),
            "documentation_sha256": sha256_file(DOCUMENTATION),
            "third_cohort_protocol_sha256": ancestors["third_cohort_protocol_sha256"],
            "source_checks": source_checks,
            "dictionary_verification": dictionary_verification,
            "first_nwicu_association_read_utc": None,
            "nwicu_association_inspected": False,
        }
    )
    write_json(LOCK, payload)
    (LOCK.with_suffix(".sha256")).write_text(sha256_file(LOCK) + "\n")
    print(json.dumps({"status": payload["status"], "analysis_lock_sha256": sha256_file(LOCK)}, indent=2))


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    sidecar = LOCK.with_suffix(".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != sha256_file(LOCK):
        raise RuntimeError("NWICU lock whole-file hash mismatch")
    if canonical_sha256(lock["specification"]) != lock["specification_sha256"]:
        raise RuntimeError("NWICU specification hash mismatch")
    if sha256_file(PROTOCOL) != lock["specification"]["protocol_sha256"]:
        raise RuntimeError("NWICU protocol hash mismatch")
    if sha256_file(DOCUMENTATION) != lock["documentation_sha256"]:
        raise RuntimeError("NWICU documentation hash mismatch")
    if any(sha256_file(ROOT / relative) != expected for relative, expected in lock["code_hashes"].items()):
        raise RuntimeError("NWICU locked code hash mismatch")
    if any(not row["match"] for row in verify_source_manifest(NWICU)):
        raise RuntimeError("NWICU source verification failed")
    return lock


def build_inputs() -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET enable_progress_bar=false")
    paths = {
        "admissions": source_path("nw_hosp", "admissions"),
        "diagnoses": source_path("nw_hosp", "diagnoses_icd"),
        "patients": source_path("nw_hosp", "patients"),
        "icustays": source_path("nw_icu", "icustays"),
        "chartevents": source_path("nw_icu", "chartevents"),
        "labevents": source_path("nw_hosp", "labevents"),
        "procedureevents": source_path("nw_icu", "procedureevents"),
    }
    for name, path in paths.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM {csv_scan(path)}")
    con.execute("""
        CREATE TABLE cohort AS
        WITH hf AS (
            SELECT DISTINCT hadm_id FROM diagnoses
            WHERE (try_cast(icd_version AS INTEGER)=9 AND starts_with(icd_code,'428'))
               OR (try_cast(icd_version AS INTEGER)=10 AND starts_with(upper(icd_code),'I50'))
        ), ranked AS (
            SELECT i.*,row_number() OVER(
                PARTITION BY i.hadm_id ORDER BY try_cast(i.intime AS TIMESTAMP),try_cast(i.stay_id AS BIGINT)
            ) AS rn FROM icustays i
        )
        SELECT try_cast(i.stay_id AS BIGINT) AS stay_id,i.subject_id,i.hadm_id,
               i.first_careunit,try_cast(i.intime AS TIMESTAMP) AS intime,
               try_cast(p.anchor_age AS DOUBLE)+year(try_cast(a.admittime AS TIMESTAMP))-try_cast(p.anchor_year AS DOUBLE) AS age,
               CASE upper(trim(p.gender)) WHEN 'M' THEN 1.0 WHEN 'F' THEN 0.0 END AS male_sex,
               epoch(try_cast(i.outtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60 AS followup_end_offset_minutes,
               epoch(try_cast(a.deathtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60 AS death_offset_minutes
        FROM ranked i JOIN admissions a USING(subject_id,hadm_id) JOIN patients p USING(subject_id)
        WHERE i.rn=1 AND i.hadm_id IN (SELECT hadm_id FROM hf)
          AND epoch(try_cast(i.intime AS TIMESTAMP)-try_cast(a.admittime AS TIMESTAMP))/60 BETWEEN 0 AND 1440
          AND try_cast(p.anchor_age AS DOUBLE)+year(try_cast(a.admittime AS TIMESTAMP))-try_cast(p.anchor_year AS DOUBLE)>=18
          AND epoch(try_cast(i.outtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60>360
          AND epoch(try_cast(a.dischtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60>360
          AND (a.deathtime IS NULL OR epoch(try_cast(a.deathtime AS TIMESTAMP)-try_cast(i.intime AS TIMESTAMP))/60>360)
    """)
    con.execute("""
        CREATE TABLE vital_rows AS
        SELECT c.stay_id,
               epoch(try_cast(v.charttime AS TIMESTAMP)-c.intime)/60 AS event_minute,
               greatest(epoch(try_cast(v.charttime AS TIMESTAMP)-c.intime)/60,
                        epoch(try_cast(v.storetime AS TIMESTAMP)-c.intime)/60) AS available_minute,
               CASE try_cast(v.itemid AS BIGINT)
                   WHEN 320045 THEN 'hr' WHEN 320179 THEN 'sbp' WHEN 320180 THEN 'dbp'
                   WHEN 320210 THEN 'resp_rate' WHEN 320277 THEN 'spo2' WHEN 323761 THEN 'temperature_c'
               END AS signal,
               CASE WHEN try_cast(v.itemid AS BIGINT)=323761 THEN (try_cast(v.valuenum AS DOUBLE)-32)*5/9
                    ELSE try_cast(v.valuenum AS DOUBLE) END AS value
        FROM chartevents v JOIN cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
        WHERE try_cast(v.itemid AS BIGINT) IN (320045,320179,320180,320210,320277,323761)
          AND epoch(try_cast(v.charttime AS TIMESTAMP)-c.intime)/60 BETWEEN 0 AND 960
          AND CASE try_cast(v.itemid AS BIGINT)
              WHEN 320045 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 20 AND 250
              WHEN 320179 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 40 AND 260
              WHEN 320180 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 20 AND 180
              WHEN 320210 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 4 AND 80
              WHEN 320277 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 50 AND 100
              WHEN 323761 THEN (try_cast(v.valuenum AS DOUBLE)-32)*5/9 BETWEEN 30 AND 43 ELSE false END
    """)
    con.execute("""
        CREATE TABLE vital_hourly AS
        SELECT stay_id,signal,floor(event_minute/60) AS hour,median(value) AS value,
               max(available_minute) AS available_minute,count(*) AS measurements
        FROM vital_rows GROUP BY 1,2,3
    """)
    con.execute("""
        CREATE TABLE lab_rows AS
        SELECT c.stay_id,
               CASE try_cast(l.itemid AS BIGINT) WHEN 100031 THEN 'lactate' WHEN 100002 THEN 'creatinine'
                    WHEN 100042 THEN 'alt' WHEN 100339 THEN 'ph' END AS marker,
               greatest(epoch(try_cast(l.charttime AS TIMESTAMP)-c.intime)/60,
                        epoch(try_cast(l.storetime AS TIMESTAMP)-c.intime)/60) AS event_minute,
               try_cast(l.valuenum AS DOUBLE) AS value
        FROM labevents l JOIN cohort c ON l.hadm_id=c.hadm_id
        WHERE try_cast(l.itemid AS BIGINT) IN (100031,100002,100042,100339)
          AND greatest(epoch(try_cast(l.charttime AS TIMESTAMP)-c.intime)/60,
                       epoch(try_cast(l.storetime AS TIMESTAMP)-c.intime)/60) BETWEEN -1440 AND 960
          AND CASE try_cast(l.itemid AS BIGINT)
              WHEN 100031 THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 0.1 AND 30
              WHEN 100002 THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 0.1 AND 30
              WHEN 100042 THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 1 AND 10000
              WHEN 100339 THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 6.5 AND 8.0 ELSE false END
    """)
    con.execute("""
        CREATE TABLE mcs_events AS
        SELECT c.stay_id,epoch(try_cast(p.starttime AS TIMESTAMP)-c.intime)/60 AS event_minute,'mcs' AS component
        FROM procedureevents p JOIN cohort c ON try_cast(p.stay_id AS BIGINT)=c.stay_id
        WHERE try_cast(p.itemid AS BIGINT)=736876
          AND epoch(try_cast(p.starttime AS TIMESTAMP)-c.intime)/60<=960
          AND try_cast(p.endtime AS TIMESTAMP)>c.intime
    """)
    cohort = con.execute("SELECT * FROM cohort").df()
    early = con.execute("SELECT * FROM vital_rows WHERE event_minute BETWEEN 0 AND 240 AND available_minute<=240").df()
    raw_spo2 = early.loc[early["signal"].eq("spo2"), ["stay_id", "event_minute", "value"]].rename(
        columns={"event_minute": "minute", "value": "spo2"}
    )
    exposure = compute_frozen_exposure(raw_spo2)
    candidate_parts: list[pd.DataFrame] = []
    for signal in ("hr", "sbp", "resp_rate", "spo2", "temperature_c"):
        part = early.loc[early["signal"].eq(signal), ["stay_id", "event_minute", "value"]].copy()
        part["hour"] = np.floor(part["event_minute"] / 60).astype(int)
        part = part.groupby(["stay_id", "hour"], as_index=False)["value"].median()
        part["signal"] = signal
        candidate_parts.append(part[["stay_id", "signal", "hour", "value"]])
    trajectories = trajectory_features(pd.concat(candidate_parts, ignore_index=True))
    levels = trajectories.pivot(index="stay_id", columns="signal", values="level").add_suffix("_level").reset_index()
    levels["map_level"] = np.nan
    bp = con.execute("""
        SELECT stay_id,hour,max(value) FILTER(WHERE signal='sbp') AS sbp,
               cast(NULL AS DOUBLE) AS map,max(available_minute) AS available_minute
        FROM vital_hourly WHERE signal='sbp' GROUP BY 1,2
    """).df()
    pressure = pd.concat(
        [sustained_hypotension_events(bp), con.execute("SELECT * FROM mcs_events").df()],
        ignore_index=True,
    )
    lab_rows = con.execute("SELECT * FROM lab_rows").df()
    baseline_rows = lab_rows.loc[lab_rows["event_minute"].le(240)].sort_values("event_minute").groupby(
        ["stay_id", "marker"]
    ).tail(1)
    if baseline_rows.empty:
        baseline = pd.DataFrame(columns=["stay_id", "baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"])
    else:
        baseline = baseline_rows.pivot(index="stay_id", columns="marker", values="value").add_prefix("baseline_").reset_index()
    for column in ("baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"):
        if column not in baseline:
            baseline[column] = np.nan
    labs = lab_rows.loc[lab_rows["event_minute"].gt(240)].merge(baseline, on="stay_id", how="left")
    perfusion_parts: list[pd.DataFrame] = []
    for marker, group in labs.groupby("marker"):
        if marker == "lactate":
            mask = group["value"].ge(4) | (group["baseline_lactate"].notna() & group["value"].ge(2) & group["value"].sub(group["baseline_lactate"]).ge(0.5))
        elif marker == "creatinine":
            mask = group["baseline_creatinine"].notna() & (group["value"].sub(group["baseline_creatinine"]).ge(0.3) | group["value"].div(group["baseline_creatinine"]).ge(1.5))
        elif marker == "alt":
            mask = group["baseline_alt"].notna() & ((group["baseline_alt"].le(200) & group["value"].gt(200)) | group["value"].div(group["baseline_alt"]).ge(3))
        else:
            mask = group["baseline_ph"].notna() & group["baseline_ph"].ge(7.2) & group["value"].lt(7.2)
        selected = group.loc[mask, ["stay_id", "event_minute"]].copy()
        selected["component"] = marker
        perfusion_parts.append(selected)
    perfusion = pd.concat(perfusion_parts, ignore_index=True) if perfusion_parts else pd.DataFrame(columns=["stay_id", "event_minute", "component"])
    direct_pre = lab_rows.loc[
        lab_rows["event_minute"].le(360)
        & ((lab_rows["marker"].eq("lactate") & lab_rows["value"].ge(2))
           | (lab_rows["marker"].eq("alt") & lab_rows["value"].gt(200))
           | (lab_rows["marker"].eq("ph") & lab_rows["value"].lt(7.2))),
        "stay_id",
    ]
    pre_pressure = set(pressure.loc[pressure["event_minute"].le(360), "stay_id"])
    pre_perfusion = set(direct_pre) | set(perfusion.loc[perfusion["event_minute"].le(360), "stay_id"])
    at_risk = cohort.loc[~cohort["stay_id"].isin(pre_pressure | pre_perfusion)].copy()
    primary_events = pair_domain_events(pressure, perfusion, lower_minute=360, upper_minute=960)
    later_events = pair_domain_events(pressure, perfusion, lower_minute=480, upper_minute=960)
    outcomes = at_risk[["stay_id"]].merge(
        primary_events[["stay_id", "event_minute"]].rename(columns={"event_minute": "modified_endpoint_minute"}),
        on="stay_id", how="left",
    ).merge(
        later_events[["stay_id", "event_minute"]].rename(columns={"event_minute": "modified_endpoint_later_minute"}),
        on="stay_id", how="left",
    )
    outcomes["modified_endpoint"] = outcomes["modified_endpoint_minute"].notna().astype(int)
    outcomes["modified_endpoint_later"] = outcomes["modified_endpoint_later_minute"].notna().astype(int)
    bp_density = early.loc[early["signal"].eq("sbp")].assign(
        hour=lambda frame: np.floor(frame["event_minute"] / 60).astype(int)
    ).groupby("stay_id")["hour"].nunique().rename("bp_measurement_density")
    model = at_risk.merge(outcomes, on="stay_id").merge(levels, on="stay_id", how="left").merge(
        exposure[["stay_id", "bins", "span_hours", "raw_rms_residual"]].rename(
            columns={"bins": "spo2_bins", "span_hours": "spo2_span_hours", "raw_rms_residual": "spo2_variability"}
        ), on="stay_id", how="left",
    ).merge(baseline, on="stay_id", how="left").merge(bp_density, on="stay_id", how="left")
    model["lactate_observed"] = model["baseline_lactate"].notna().astype(int)
    model["urine_measurement_density"] = np.nan
    component_counts = {
        "pressure": pressure.loc[pressure["event_minute"].gt(360) & pressure["event_minute"].le(960), "component"].value_counts().to_dict(),
        "laboratory_hypoperfusion": perfusion.loc[perfusion["event_minute"].gt(360) & perfusion["event_minute"].le(960), "component"].value_counts().to_dict(),
        "paired_pressure": primary_events["pressure_component"].value_counts().to_dict(),
        "paired_laboratory_hypoperfusion": primary_events["perfusion_component"].value_counts().to_dict(),
    }
    con.close()
    return {
        "model": model,
        "exposure": exposure,
        "preliminary_cohort_n": len(cohort),
        "at_risk_n": len(at_risk),
        "pre_pressure_excluded_n": len(pre_pressure),
        "pre_perfusion_excluded_n": len(pre_perfusion),
        "primary_events": primary_events,
        "later_events": later_events,
        "component_counts": component_counts,
    }


def run_locked_analysis() -> None:
    lock = verify_lock()
    if RESULTS.exists():
        raise FileExistsError("NWICU association result already exists; repeat fitting is prohibited")
    inputs = build_inputs()
    model = inputs["model"]
    valid = model["spo2_variability"].notna()
    modeled_events = int(model.loc[valid, "modified_endpoint"].sum())
    feasibility = evaluate_frozen_feasibility(
        preliminary_eligible_stays=len(model),
        preliminary_valid_exposures=int(valid.sum()),
        modified_endpoint_observable=True,
        site_identifier_available=False,
        contributing_sites=None,
        modeled_primary_events=modeled_events,
    )
    support = {
        "preliminary_cohort_before_available_pre360_exclusions": inputs["preliminary_cohort_n"],
        "modified_endpoint_at_risk": inputs["at_risk_n"],
        "pre_pressure_excluded_unique_stays": inputs["pre_pressure_excluded_n"],
        "pre_laboratory_hypoperfusion_excluded_unique_stays": inputs["pre_perfusion_excluded_n"],
        "valid_frozen_exposure": int(valid.sum()),
        "exposure_coverage": float(valid.mean()),
        "modified_endpoint_events_at_risk": int(model["modified_endpoint"].sum()),
        "modified_endpoint_events_modeled": modeled_events,
        "later_window_events_modeled": int(model.loc[valid, "modified_endpoint_later"].sum()),
        "released_site_identifiers": 0,
        "component_counts": inputs["component_counts"],
        "structurally_unavailable": ["continuous vasoactive-infusion support", "oliguria"],
    }
    write_json(OUT / "support_counts.json", support)
    write_json(OUT / "feasibility.json", feasibility)
    assert_association_authorized(feasibility)
    read_utc = utc_now()
    receipt = {
        "first_nwicu_association_read_utc": read_utc,
        "analysis_lock_sha256": sha256_file(LOCK),
        "attempt": 1,
        "endpoint": MODIFIED_ENDPOINT["name"],
    }
    write_json(OUT / "association_read_receipt.json", receipt)
    primary, _ = fit_fixed_scale_model(
        model,
        feature="spo2_variability",
        outcome="modified_endpoint",
        adjustment_covariates=BASE_COVARIATES,
        exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    later, _ = fit_fixed_scale_model(
        model,
        feature="spo2_variability",
        outcome="modified_endpoint_later",
        adjustment_covariates=BASE_COVARIATES,
        exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    supportive = bool(primary["rr_per_mimic_sd"] > 1 and primary["ci_low"] > 1 and primary["p_value"] < 0.05 and later["rr_per_mimic_sd"] > 1)
    classification = "MODIFIED_NWICU_ENDPOINT_DIRECTIONALLY_SUPPORTIVE" if supportive else "MODIFIED_NWICU_ENDPOINT_NOT_DIRECTIONALLY_SUPPORTIVE"
    result = {
        "classification": classification,
        "not_original_external_validation": True,
        "endpoint": MODIFIED_ENDPOINT,
        "primary": primary,
        "later_window": later,
        "support": support,
        "feasibility": feasibility,
        "first_nwicu_association_read_utc": read_utc,
        "outcome_fit_count": 2,
        "covariance": "HC0_single_source",
        "waivers": lock["specification"]["user_authorized_waivers"],
    }
    write_json(RESULTS, result)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ["NWICU modified endpoint", "NWICU later window"]
    rr = np.array([primary["rr_per_mimic_sd"], later["rr_per_mimic_sd"]])
    lo = np.array([primary["ci_low"], later["ci_low"]])
    hi = np.array([primary["ci_high"], later["ci_high"]])
    ax.errorbar(rr, [1, 0], xerr=[rr - lo, hi - rr], fmt="o", color="#2457C5", capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks([1, 0], labels)
    ax.set_xlabel("Adjusted RR per frozen MIMIC SD")
    ax.set_title("NWICU prospectively amended analysis")
    fig.tight_layout()
    fig.savefig(OUT / "nwicu_modified_endpoint_forest.png", dpi=180)
    plt.close(fig)
    report = f"""# NWICU SpO2-variability modified-endpoint analysis

**{classification}**

This is a **LOCAL**, single-source analysis of the prospectively amended NWICU endpoint. It is not external validation of the original frozen composite and does not change the frozen eICU classification.

## Result

- Primary modified endpoint: adjusted HC0 RR **{primary['rr_per_mimic_sd']:.3f}** (95% CI **{primary['ci_low']:.3f}–{primary['ci_high']:.3f}**), p={primary['p_value']:.4g}; {primary['n']:,} modeled stays and {primary['events']} events.
- Later window: adjusted HC0 RR **{later['rr_per_mimic_sd']:.3f}** (95% CI **{later['ci_low']:.3f}–{later['ci_high']:.3f}**), p={later['p_value']:.4g}; {later['events']} events.

## Support and deviations from the original target

- {support['modified_endpoint_at_risk']:,} stays remained at risk after available pre-360 exclusions; {support['valid_frozen_exposure']:,} had the exact frozen exposure: **{support['exposure_coverage']:.1%} coverage**, below the original 70% gate.
- The user explicitly authorized proceeding below 70% coverage and without the original ≥20-site gate. NWICU exposes no hospital/site identifier; HC0 variance is used, and care unit is not substituted as a site.
- The endpoint contains sustained cuff-SBP hypotension and recorded ECMO support paired with lactate, creatinine, blood pH, or ALT hypoperfusion. Continuous vasoactive-infusion support and oliguria are structurally unavailable.
- The original minimum 50 modeled-event gate was retained and passed with {modeled_events} events.

The result concerns a constructed, modified endpoint—not adjudicated cardiogenic shock. Association does not establish causality or clinical utility. No alternate exposure, scaling, subgroup, care-unit selection, endpoint, or threshold was tested.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    artifacts = [path for path in OUT.iterdir() if path.is_file() and path.name != "artifact_hashes.json"]
    write_json(OUT / "artifact_hashes.json", {path.name: sha256_file(path) for path in sorted(artifacts)})
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--lock", action="store_true")
    stage.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.lock:
        write_lock()
    else:
        run_locked_analysis()


if __name__ == "__main__":
    main()
