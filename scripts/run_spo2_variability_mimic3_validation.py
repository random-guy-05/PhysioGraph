"""Run the locked MIMIC-III historical/database-version replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from physiograph.analysis.spo2_variability_mimic3_validation import (
    FROZEN_ENDPOINT,
    FROZEN_EXPOSURE_PARAMETERS,
    FROZEN_GATES,
    MIMIC3_MAPPING,
    assert_association_authorized,
    canonical_sha256,
    compute_frozen_exposure,
    evaluate_frozen_feasibility,
    sha256_file,
    verify_source_manifest,
)
from physiograph.analysis.spo2_variability_validation import fit_fixed_scale_model

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT.parents[1]
MIMIC3 = DRIVE / "Data/MIMIC III"
OUT = ROOT / "research/spo2_variability_mimic3_validation"
PROTOCOL = OUT / "protocol.json"
LOCK = OUT / "analysis_lock.json"
SUPPORT = OUT / "support_counts.json"
FEASIBILITY = OUT / "feasibility.json"
RUN_STATUS = OUT / "run_status.json"
RESULTS = OUT / "validation_results.json"
RECEIPT = OUT / "association_read_receipt.json"
DOCUMENTATION = ROOT / "docs/SPO2_VARIABILITY_MIMIC3_REPLICATION.md"
THIRD_PROTOCOL = ROOT / "research/spo2_variability_transportability/THIRD_COHORT_VALIDATION_PROTOCOL.md"
EXTERNAL_CONFIG = ROOT / "research/spo2_variability_validation/external_validation_config.json"
EXTERNAL_LOCK = ROOT / "research/spo2_variability_validation/external_validation_lock.json"
PROTECTED_DIRS = [
    "research/spo2_variability_validation",
    "research/spo2_variability_transportability",
    "research/spo2_variability_nwicu_validation",
    "research/spo2_variability_inspire_validation",
]
CODE_FILES = [
    "scripts/run_spo2_variability_mimic3_validation.py",
    "scripts/report_spo2_variability_mimic3_validation.py",
    "src/physiograph/analysis/spo2_variability_mimic3_validation.py",
    "src/physiograph/analysis/spo2_variability_nwicu_validation.py",
    "src/physiograph/analysis/spo2_variability_validation.py",
    "src/physiograph/analysis/shock_signal_discovery.py",
    "tests/unit/test_spo2_variability_mimic3_validation.py",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def csv_scan(name: str) -> str:
    path = MIMIC3 / f"{name}.csv.gz"
    return (
        f"read_csv({sql_path(path)},header=true,all_varchar=true,quote='\"',"
        "escape='\"',strict_mode=false,sample_size=10000)"
    )


def csv_ints(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {relative: tree_sha256(ROOT / relative) for relative in PROTECTED_DIRS}


def protocol_payload() -> dict[str, Any]:
    return {
        "status": "FROZEN_BEFORE_MIMIC3_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "dataset": "MIMIC-III_v1.4",
        "interpretation": "same_institution_historical_database_version_replication_not_independent_external_validation",
        "population": "adult_explicit_HF_first_ICU_per_hospital_admission",
        "timeline_origin": "ICUSTAYS.intime",
        "exposure": {
            "source": "CHARTEVENTS itemid 646",
            "window_minutes": [0, 240],
            "value_bounds": [50, 100],
            "hourly_aggregation": "median",
            "minimum_bins": 3,
            "minimum_span_hours": 2,
            "feature": "RMS residual around within-stay OLS line",
            "transform": FROZEN_EXPOSURE_PARAMETERS,
        },
        "endpoint": FROZEN_ENDPOINT,
        "mapping": MIMIC3_MAPPING,
        "adjustment_covariates": BASE_COVARIATES,
        "gates": {**FROZEN_GATES, "exact_composite_required": True},
        "user_authorized_waivers": {
            "minimum_70_percent_exposure_coverage": True,
            "minimum_20_contributing_sites": True,
            "minimum_50_modeled_primary_events": False,
            "exact_composite_required": False,
        },
        "primary_covariance": "HC0_due_single_hospital_and_user_site_waiver",
        "confirmation": {
            "primary_rr_gt_1": True,
            "primary_two_sided_95_ci_excludes_1": True,
            "primary_p_lt": 0.05,
            "later_window_rr_gt_1": True,
        },
        "prohibited": [
            "alternate_exposure", "dataset_specific_exposure_scaling",
            "alternate_endpoint", "threshold_or_time_window_search",
            "subgroup_or_unit_rescue",
        ],
        "association_inspected": False,
    }


def verify_frozen_ancestors() -> dict[str, str]:
    lock = json.loads(EXTERNAL_LOCK.read_text())
    if sha256_file(EXTERNAL_CONFIG) != lock["config_sha256"]:
        raise RuntimeError("Frozen external-validation config no longer matches its lock")
    return {
        "external_validation_config_sha256": sha256_file(EXTERNAL_CONFIG),
        "external_validation_lock_sha256": sha256_file(EXTERNAL_LOCK),
        "third_cohort_protocol_sha256": sha256_file(THIRD_PROTOCOL),
    }


def verify_dictionary_mapping() -> dict[str, Any]:
    items = pd.read_csv(MIMIC3 / "D_ITEMS.csv.gz")
    labs = pd.read_csv(MIMIC3 / "D_LABITEMS.csv.gz")
    actual_items = set(pd.to_numeric(items["itemid"], errors="coerce").dropna().astype(int))
    actual_labs = set(pd.to_numeric(labs["itemid"], errors="coerce").dropna().astype(int))
    item_groups = [
        *MIMIC3_MAPPING["vitals"].values(),
        MIMIC3_MAPPING["continuous_support"]["itemids"],
        MIMIC3_MAPPING["mcs"]["itemids"],
        MIMIC3_MAPPING["urine_output"]["itemids"],
    ]
    missing_items = sorted(set().union(*map(set, item_groups)) - actual_items)
    missing_labs = sorted(
        set().union(*map(set, MIMIC3_MAPPING["labs"].values())) - actual_labs
    )
    if missing_items or missing_labs:
        raise RuntimeError(f"MIMIC-III mapping missing itemids: items={missing_items}, labs={missing_labs}")
    mv_inputs_header_only = (MIMIC3 / "INPUTEVENTS_MV.csv.gz").stat().st_size < 1024
    mv_procedures_header_only = (MIMIC3 / "PROCEDUREEVENTS_MV.csv.gz").stat().st_size < 1024
    if not (mv_inputs_header_only and mv_procedures_header_only):
        raise RuntimeError("Unexpected MetaVision content; frozen CareVue-only mapping is invalid")
    return {
        "mapped_itemids_present": True,
        "mapped_lab_itemids_present": True,
        "inputevents_mv_header_only": True,
        "procedureevents_mv_header_only": True,
        "released_site_count": 1,
    }


def write_lock() -> None:
    if LOCK.exists() or RESULTS.exists() or RECEIPT.exists():
        raise FileExistsError("MIMIC-III lock or association artifact already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    source_checks = verify_source_manifest(MIMIC3)
    dictionary = verify_dictionary_mapping()
    ancestors = verify_frozen_ancestors()
    protocol = protocol_payload()
    write_json(PROTOCOL, protocol)
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in CODE_FILES}
    specification = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "documentation_sha256": sha256_file(DOCUMENTATION),
        "source_manifest_sha256": sha256_file(MIMIC3 / "SHA256SUMS.txt"),
        "frozen_ancestors": ancestors,
        "protected_artifact_tree_hashes": protected_hashes(),
        "endpoint": FROZEN_ENDPOINT,
        "mapping": MIMIC3_MAPPING,
        "exposure_parameters": FROZEN_EXPOSURE_PARAMETERS,
        "code_hashes": code_hashes,
    }
    payload = {
        "status": "LOCKED_BEFORE_MIMIC3_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "lock_utc": utc_now(),
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "source_checks": source_checks,
        "dictionary_verification": dictionary,
        "first_mimic3_association_read_utc": None,
        "mimic3_association_inspected": False,
    }
    write_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(sha256_file(LOCK) + "\n")
    print(json.dumps({"status": payload["status"], "analysis_lock_sha256": sha256_file(LOCK)}, indent=2))


def verify_lock(*, verify_sources: bool = True) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    if LOCK.with_suffix(".sha256").read_text().strip() != sha256_file(LOCK):
        raise RuntimeError("MIMIC-III whole-file lock hash mismatch")
    specification = lock["specification"]
    if canonical_sha256(specification) != lock["specification_sha256"]:
        raise RuntimeError("MIMIC-III specification hash mismatch")
    if sha256_file(PROTOCOL) != specification["protocol_sha256"]:
        raise RuntimeError("MIMIC-III protocol hash mismatch")
    if sha256_file(DOCUMENTATION) != specification["documentation_sha256"]:
        raise RuntimeError("MIMIC-III documentation hash mismatch")
    for relative, expected in specification["code_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"MIMIC-III locked code changed: {relative}")
    if protected_hashes() != specification["protected_artifact_tree_hashes"]:
        raise RuntimeError("A pre-existing frozen validation artifact changed")
    verify_frozen_ancestors()
    if verify_sources:
        verify_source_manifest(MIMIC3)
    return lock


def count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def build_inputs() -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET enable_progress_bar=false")
    temporary = Path(os.environ.get("MIMIC3_DUCKDB_TEMP", "/tmp/physiograph-mimic3-duckdb"))
    temporary.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={sql_path(temporary)}")
    for table in ("ICUSTAYS", "ADMISSIONS", "PATIENTS", "DIAGNOSES_ICD"):
        con.execute(f"CREATE VIEW {table.lower()} AS SELECT * FROM {csv_scan(table)}")
    con.execute("""
        CREATE TABLE cohort AS
        WITH hf_hadm AS (
            SELECT DISTINCT try_cast(hadm_id AS BIGINT) AS hadm_id
            FROM diagnoses_icd WHERE starts_with(replace(trim(icd9_code),'.',''),'428')
        ), candidates AS (
            SELECT try_cast(i.icustay_id AS BIGINT) AS stay_id,
                   try_cast(i.subject_id AS BIGINT) AS subject_id,
                   try_cast(i.hadm_id AS BIGINT) AS hadm_id,
                   try_cast(i.intime AS TIMESTAMP) AS intime,
                   try_cast(i.outtime AS TIMESTAMP) AS outtime,
                   try_cast(a.admittime AS TIMESTAMP) AS admittime,
                   try_cast(a.dischtime AS TIMESTAMP) AS dischtime,
                   try_cast(a.deathtime AS TIMESTAMP) AS deathtime,
                   CASE WHEN date_diff('day',try_cast(p.dob AS TIMESTAMP),try_cast(a.admittime AS TIMESTAMP))/365.2425>89
                        THEN 90.0 ELSE date_diff('day',try_cast(p.dob AS TIMESTAMP),try_cast(a.admittime AS TIMESTAMP))/365.2425 END AS age,
                   CASE upper(trim(p.gender)) WHEN 'M' THEN 1.0 WHEN 'F' THEN 0.0 END AS male_sex,
                   row_number() OVER(PARTITION BY try_cast(i.hadm_id AS BIGINT)
                                     ORDER BY try_cast(i.intime AS TIMESTAMP),try_cast(i.icustay_id AS BIGINT)) AS rn
            FROM icustays i JOIN admissions a USING(subject_id,hadm_id)
            JOIN patients p USING(subject_id) JOIN hf_hadm h ON try_cast(i.hadm_id AS BIGINT)=h.hadm_id
        )
        SELECT *,date_diff('second',intime,outtime)/60.0 AS followup_end_offset_minutes,
               date_diff('second',intime,deathtime)/60.0 AS death_offset_minutes
        FROM candidates WHERE rn=1 AND age>=18
          AND outtime>intime+INTERVAL 360 MINUTE AND dischtime>intime+INTERVAL 360 MINUTE
          AND (deathtime IS NULL OR deathtime>intime+INTERVAL 360 MINUTE)
    """)
    vital_map = MIMIC3_MAPPING["vitals"]
    mcs_ids = MIMIC3_MAPPING["mcs"]["itemids"]
    all_chart_ids = sorted(set().union(*map(set, vital_map.values()), set(mcs_ids)))
    con.execute(f"""
        CREATE TABLE chart_rows AS
        SELECT c.stay_id,date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 AS event_minute,
               CASE
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['hr'])}) THEN 'hr'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['sbp_cuff'])}) THEN 'sbp'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['map_cuff'])}) THEN 'map'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['resp_rate'])}) THEN 'resp_rate'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['spo2'])}) THEN 'spo2'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['temperature_c'])}) THEN 'temperature_c'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(vital_map['temperature_f'])}) THEN 'temperature_f'
                 WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(mcs_ids)}) THEN 'mcs' END AS signal,
               try_cast(x.valuenum AS DOUBLE) AS raw_value,trim(x.value) AS text_value
        FROM {csv_scan('CHARTEVENTS')} x JOIN cohort c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({csv_ints(all_chart_ids)})
          AND try_cast(x.charttime AS TIMESTAMP) BETWEEN c.intime AND least(c.outtime,c.intime+INTERVAL 960 MINUTE)
          AND coalesce(try_cast(x.error AS INTEGER),0)=0
    """)
    con.execute("""
        CREATE TABLE vital_rows AS
        SELECT stay_id,event_minute,
               CASE WHEN signal='temperature_f' THEN 'temperature_c' ELSE signal END AS signal,
               CASE WHEN signal='temperature_f' THEN (raw_value-32.0)*5.0/9.0 ELSE raw_value END AS value
        FROM chart_rows WHERE signal!='mcs' AND CASE signal
          WHEN 'hr' THEN raw_value BETWEEN 20 AND 250
          WHEN 'sbp' THEN raw_value BETWEEN 40 AND 260
          WHEN 'map' THEN raw_value BETWEEN 30 AND 200
          WHEN 'resp_rate' THEN raw_value BETWEEN 4 AND 80
          WHEN 'spo2' THEN raw_value BETWEEN 50 AND 100
          WHEN 'temperature_c' THEN raw_value BETWEEN 30 AND 43
          WHEN 'temperature_f' THEN raw_value BETWEEN 86 AND 109.4 ELSE false END
    """)
    con.execute("""
        CREATE TABLE vital_hourly AS
        SELECT stay_id,signal,floor(event_minute/60) AS hour,median(value) AS value,
               max(event_minute) AS available_minute,count(*) AS measurements
        FROM vital_rows GROUP BY 1,2,3
    """)
    support_ids = MIMIC3_MAPPING["continuous_support"]["itemids"]
    con.execute(f"""
        CREATE TABLE continuous_support AS
        SELECT c.stay_id,date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 AS event_minute,
               'continuous_support' AS component
        FROM {csv_scan('INPUTEVENTS_CV')} x JOIN cohort c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({csv_ints(support_ids)})
          AND coalesce(try_cast(x.rate AS DOUBLE),try_cast(x.originalrate AS DOUBLE))>0
          AND coalesce(lower(trim(x.stopped)),'') NOT IN ('stopped','discontinued')
          AND try_cast(x.charttime AS TIMESTAMP) BETWEEN c.intime AND least(c.outtime,c.intime+INTERVAL 960 MINUTE)
    """)
    urine_ids = MIMIC3_MAPPING["urine_output"]["itemids"]
    con.execute(f"""
        CREATE TABLE urine_rows AS
        SELECT c.stay_id,date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 AS event_minute,
               try_cast(x.value AS DOUBLE) AS value
        FROM {csv_scan('OUTPUTEVENTS')} x JOIN cohort c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({csv_ints(urine_ids)})
          AND try_cast(x.value AS DOUBLE) BETWEEN 0 AND 5000
          AND coalesce(try_cast(x.iserror AS INTEGER),0)=0
          AND try_cast(x.charttime AS TIMESTAMP) BETWEEN c.intime AND least(c.outtime,c.intime+INTERVAL 960 MINUTE)
    """)
    lab_map = MIMIC3_MAPPING["labs"]
    lab_ids = sorted(set().union(*map(set, lab_map.values())))
    con.execute(f"""
        CREATE TABLE lab_rows AS
        SELECT c.stay_id,
               CASE WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['lactate'])}) THEN 'lactate'
                    WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['creatinine'])}) THEN 'creatinine'
                    WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['alt'])}) THEN 'alt'
                    WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['ph'])}) THEN 'ph' END AS marker,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 AS event_minute,
               try_cast(x.valuenum AS DOUBLE) AS value
        FROM {csv_scan('LABEVENTS')} x JOIN cohort c
          ON try_cast(x.hadm_id AS BIGINT)=c.hadm_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_ids)})
          AND try_cast(x.charttime AS TIMESTAMP) BETWEEN greatest(c.admittime,c.intime-INTERVAL 1440 MINUTE)
                                                   AND least(c.dischtime,c.outtime,c.intime+INTERVAL 960 MINUTE)
          AND CASE
            WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['lactate'])}) THEN try_cast(x.valuenum AS DOUBLE) BETWEEN 0.1 AND 30
            WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['creatinine'])}) THEN try_cast(x.valuenum AS DOUBLE) BETWEEN 0.1 AND 30
            WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['alt'])}) THEN try_cast(x.valuenum AS DOUBLE) BETWEEN 1 AND 10000
            WHEN try_cast(x.itemid AS INTEGER) IN ({csv_ints(lab_map['ph'])}) THEN try_cast(x.valuenum AS DOUBLE) BETWEEN 6.5 AND 8.0 ELSE false END
    """)
    cohort = con.execute("SELECT stay_id,subject_id,hadm_id,age,male_sex FROM cohort").df()
    early = con.execute("SELECT * FROM vital_rows WHERE event_minute BETWEEN 0 AND 240").df()
    raw_spo2 = early.loc[early["signal"].eq("spo2"), ["stay_id", "event_minute", "value"]].rename(
        columns={"event_minute": "minute", "value": "spo2"}
    )
    exposure = compute_frozen_exposure(raw_spo2)
    trajectory_parts: list[pd.DataFrame] = []
    for signal in ("hr", "sbp", "map", "resp_rate", "spo2", "temperature_c"):
        part = early.loc[early["signal"].eq(signal), ["stay_id", "event_minute", "value"]].copy()
        if part.empty:
            continue
        part["hour"] = np.floor(part["event_minute"] / 60).astype(int)
        part = part.groupby(["stay_id", "hour"], as_index=False)["value"].median()
        part["signal"] = signal
        trajectory_parts.append(part[["stay_id", "signal", "hour", "value"]])
    trajectories = trajectory_features(pd.concat(trajectory_parts, ignore_index=True))
    levels = trajectories.pivot(index="stay_id", columns="signal", values="level").add_suffix("_level").reset_index()
    for column in ("hr_level", "sbp_level", "map_level", "resp_rate_level", "spo2_level", "temperature_c_level"):
        if column not in levels:
            levels[column] = np.nan
    bp = con.execute("""
        SELECT stay_id,hour,max(value) FILTER(WHERE signal='sbp') AS sbp,
               max(value) FILTER(WHERE signal='map') AS map,max(available_minute) AS available_minute
        FROM vital_hourly WHERE signal IN ('sbp','map') GROUP BY 1,2
    """).df()
    hypotension = sustained_hypotension_events(bp)
    support = con.execute("SELECT * FROM continuous_support").df()
    mcs = con.execute("""
        SELECT stay_id,event_minute,'mcs' AS component FROM chart_rows
        WHERE signal='mcs' AND (raw_value IS NOT NULL OR nullif(text_value,'') IS NOT NULL)
          AND coalesce(lower(text_value),'') NOT IN ('off','no','none','removed','discontinued')
    """).df()
    pressure = pd.concat([hypotension, support, mcs], ignore_index=True)
    labs = con.execute("SELECT * FROM lab_rows").df()
    baseline_rows = labs.loc[labs["event_minute"].le(240)].sort_values("event_minute").groupby(
        ["stay_id", "marker"]
    ).tail(1)
    baseline = (
        baseline_rows.pivot(index="stay_id", columns="marker", values="value").add_prefix("baseline_").reset_index()
        if not baseline_rows.empty else pd.DataFrame(columns=["stay_id"])
    )
    for column in ("baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"):
        if column not in baseline:
            baseline[column] = np.nan
    labs_after_baseline = labs.loc[labs["event_minute"].gt(240)].merge(baseline, on="stay_id", how="left")
    perfusion_parts: list[pd.DataFrame] = []
    for marker, group in labs_after_baseline.groupby("marker"):
        if marker == "lactate":
            mask = group["value"].ge(4) | (
                group["baseline_lactate"].notna() & group["value"].ge(2)
                & group["value"].sub(group["baseline_lactate"]).ge(0.5)
            )
        elif marker == "creatinine":
            mask = group["baseline_creatinine"].notna() & (
                group["value"].sub(group["baseline_creatinine"]).ge(0.3)
                | group["value"].div(group["baseline_creatinine"]).ge(1.5)
            )
        elif marker == "alt":
            mask = group["baseline_alt"].notna() & (
                (group["baseline_alt"].le(200) & group["value"].gt(200))
                | group["value"].div(group["baseline_alt"]).ge(3)
            )
        else:
            mask = group["baseline_ph"].notna() & group["baseline_ph"].ge(7.2) & group["value"].lt(7.2)
        selected = group.loc[mask, ["stay_id", "event_minute"]].copy()
        selected["component"] = marker
        perfusion_parts.append(selected)
    lab_perfusion = pd.concat(perfusion_parts, ignore_index=True) if perfusion_parts else pd.DataFrame(
        columns=["stay_id", "event_minute", "component"]
    )
    urine = con.execute("""
        SELECT stay_id,floor(event_minute/60) AS hour,max(event_minute) AS available_minute,sum(value) AS value
        FROM urine_rows GROUP BY 1,2
    """).df()
    oliguria = rolling_oliguria_events(urine, start_hour=0, end_hour=16)
    perfusion = pd.concat([lab_perfusion, oliguria], ignore_index=True)
    direct_pre = labs.loc[
        labs["event_minute"].le(360) & (
            (labs["marker"].eq("lactate") & labs["value"].ge(2))
            | (labs["marker"].eq("alt") & labs["value"].gt(200))
            | (labs["marker"].eq("ph") & labs["value"].lt(7.2))
        ), "stay_id"
    ]
    pre_pressure = set(pressure.loc[pressure["event_minute"].le(360), "stay_id"])
    pre_perfusion = set(direct_pre) | set(perfusion.loc[perfusion["event_minute"].le(360), "stay_id"])
    at_risk = cohort.loc[~cohort["stay_id"].isin(pre_pressure | pre_perfusion)].copy()
    primary_events = pair_domain_events(pressure, perfusion, lower_minute=360, upper_minute=960)
    later_events = pair_domain_events(pressure, perfusion, lower_minute=480, upper_minute=960)
    outcomes = at_risk[["stay_id"]].merge(
        primary_events[["stay_id", "event_minute"]].rename(columns={"event_minute": "endpoint_minute"}),
        on="stay_id", how="left",
    ).merge(
        later_events[["stay_id", "event_minute"]].rename(columns={"event_minute": "later_endpoint_minute"}),
        on="stay_id", how="left",
    )
    outcomes["objective_shock"] = outcomes["endpoint_minute"].notna().astype(int)
    outcomes["objective_shock_later"] = outcomes["later_endpoint_minute"].notna().astype(int)
    bp_density = early.loc[early["signal"].isin(["sbp", "map"])].assign(
        hour=lambda frame: np.floor(frame["event_minute"] / 60).astype(int)
    ).groupby("stay_id")["hour"].nunique().rename("bp_measurement_density")
    urine_early = con.execute("SELECT * FROM urine_rows WHERE event_minute BETWEEN 0 AND 240").df()
    urine_density = urine_early.assign(
        hour=lambda frame: np.floor(frame["event_minute"] / 60).astype(int)
    ).groupby("stay_id")["hour"].nunique().rename("urine_measurement_density")
    model = at_risk.merge(outcomes, on="stay_id").merge(levels, on="stay_id", how="left").merge(
        exposure[["stay_id", "bins", "span_hours", "raw_rms_residual"]].rename(
            columns={"bins": "spo2_bins", "span_hours": "spo2_span_hours", "raw_rms_residual": "spo2_variability"}
        ), on="stay_id", how="left",
    ).merge(baseline, on="stay_id", how="left").merge(bp_density, on="stay_id", how="left").merge(
        urine_density, on="stay_id", how="left"
    )
    model["lactate_observed"] = model["baseline_lactate"].notna().astype(int)
    component_counts = {
        "pressure_support_post360": count_values(
            pressure.loc[pressure["event_minute"].gt(360) & pressure["event_minute"].le(960)], "component"
        ),
        "hypoperfusion_post360": count_values(
            perfusion.loc[perfusion["event_minute"].gt(360) & perfusion["event_minute"].le(960)], "component"
        ),
        "paired_pressure_support": count_values(primary_events, "pressure_component"),
        "paired_hypoperfusion": count_values(primary_events, "perfusion_component"),
    }
    con.close()
    return {
        "model": model,
        "preliminary_cohort_n": len(cohort), "at_risk_n": len(at_risk),
        "pre_pressure_excluded_n": len(pre_pressure), "pre_perfusion_excluded_n": len(pre_perfusion),
        "primary_events": primary_events, "later_events": later_events,
        "component_counts": component_counts,
    }


def write_artifact_hashes() -> None:
    artifacts = [path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"]
    write_json(OUT / "artifact_hashes.json", {str(path.relative_to(OUT)): sha256_file(path) for path in sorted(artifacts)})


def write_support_figure(support_counts: dict[str, Any]) -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = ["At risk", "Frozen exposure", "Endpoint events"]
    values = [support_counts["at_risk"], support_counts["valid_frozen_exposure"], support_counts["primary_events_modeled"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#446E9B", "#729FCF", "#CC4C4C"])
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Stays")
    ax.set_title("MIMIC-III frozen replication support")
    if min(values) > 0:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(figure_dir / "mimic3_support_funnel.png", dpi=180)
    fig.savefig(figure_dir / "mimic3_support_funnel.pdf")
    plt.close(fig)


def run_locked_analysis() -> None:
    verify_lock()
    if any(path.exists() for path in (SUPPORT, FEASIBILITY, RUN_STATUS, RESULTS, RECEIPT)):
        raise FileExistsError("MIMIC-III run artifacts already exist; repeat execution is prohibited")
    inputs = build_inputs()
    model = inputs["model"]
    valid = model["spo2_variability"].notna()
    modeled_events = int(model.loc[valid, "objective_shock"].sum())
    feasibility = evaluate_frozen_feasibility(
        eligible_stays=len(model), valid_exposures=int(valid.sum()),
        modeled_primary_events=modeled_events, exact_composite_observable=True,
    )
    support_counts = {
        "preliminary_cohort_before_pre360_exclusions": inputs["preliminary_cohort_n"],
        "at_risk": inputs["at_risk_n"],
        "pre_pressure_support_excluded_unique_stays": inputs["pre_pressure_excluded_n"],
        "pre_hypoperfusion_excluded_unique_stays": inputs["pre_perfusion_excluded_n"],
        "valid_frozen_exposure": int(valid.sum()),
        "exposure_coverage": float(valid.mean()) if len(model) else 0.0,
        "primary_events_at_risk": int(model["objective_shock"].sum()),
        "primary_events_modeled": modeled_events,
        "later_events_modeled": int(model.loc[valid, "objective_shock_later"].sum()),
        "released_sites": 1,
        "component_counts": inputs["component_counts"],
    }
    write_json(SUPPORT, support_counts)
    write_json(FEASIBILITY, feasibility)
    write_support_figure(support_counts)
    if not feasibility["association_authorized"]:
        status = {
            "status": "completed_without_association", "classification": feasibility["classification"],
            "environment": "LOCAL_STANDALONE_RUNTIME", "association_inspected": False,
            "association_models_fitted": 0, "reason": ", ".join(feasibility["failures"]),
        }
        write_json(RUN_STATUS, status)
        report = f"""# MIMIC-III SpO2-variability replication attempt

**{feasibility['classification']}**

The exact frozen endpoint was constructed, but the retained event gate stopped execution before any exposure-outcome association was fitted or inspected.

- At-risk stays: **{support_counts['at_risk']:,}**
- Exact frozen exposure: **{support_counts['valid_frozen_exposure']:,}** ({support_counts['exposure_coverage']:.1%})
- Modeled primary endpoint events: **{support_counts['primary_events_modeled']}**
- Required modeled events: **50**
- Released hospitals: **1**; the prior coverage and site waivers are noted.

MIMIC-III and MIMIC-IV derive from the same hospital and may overlap, so this is a historical database-version replication, not independent external validation. Execution environment: **LOCAL standalone runtime**, not Colab. Prior MIMIC-IV, eICU, NWICU, and INSPIRE results remain unchanged.
"""
        (OUT / "FINAL_REPORT.md").write_text(report)
        verify_lock(verify_sources=False)
        write_artifact_hashes()
        print(json.dumps({"support": support_counts, "feasibility": feasibility, "run_status": status}, indent=2))
        return
    assert_association_authorized(feasibility)
    read_utc = utc_now()
    write_json(RECEIPT, {
        "first_mimic3_association_read_utc": read_utc,
        "analysis_lock_sha256": sha256_file(LOCK), "attempt": 1,
        "endpoint": FROZEN_ENDPOINT["name"],
    })
    primary, _ = fit_fixed_scale_model(
        model, feature="spo2_variability", outcome="objective_shock",
        adjustment_covariates=BASE_COVARIATES, exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    later, _ = fit_fixed_scale_model(
        model, feature="spo2_variability", outcome="objective_shock_later",
        adjustment_covariates=BASE_COVARIATES, exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    confirmed = bool(
        primary["rr_per_mimic_sd"] > 1 and primary["ci_low"] > 1
        and primary["p_value"] < 0.05 and later["rr_per_mimic_sd"] > 1
    )
    classification = (
        "MIMIC3_HISTORICAL_REPLICATION_CONFIRMED_POST_HOC_SIGNAL"
        if confirmed else "MIMIC3_HISTORICAL_REPLICATION_DID_NOT_CONFIRM_POST_HOC_SIGNAL"
    )
    result = {
        "classification": classification, "dataset": "MIMIC-III_v1.4",
        "interpretation": "same_institution_historical_database_version_replication_not_independent_external_validation",
        "primary": primary, "later_window": later, "support": support_counts,
        "feasibility": feasibility, "first_mimic3_association_read_utc": read_utc,
        "association_models_fitted": 2, "covariance": "HC0_single_hospital",
        "frozen_eicu_classification_unchanged": "POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED",
        "nwicu_result_unchanged": True, "inspire_result_unchanged": True,
    }
    write_json(RESULTS, result)
    write_json(RUN_STATUS, {
        "status": "completed", "classification": classification,
        "environment": "LOCAL_STANDALONE_RUNTIME", "association_inspected": True,
        "association_models_fitted": 2,
    })
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    rr = np.array([primary["rr_per_mimic_sd"], later["rr_per_mimic_sd"]])
    lo = np.array([primary["ci_low"], later["ci_low"]])
    hi = np.array([primary["ci_high"], later["ci_high"]])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(rr, [1, 0], xerr=[rr - lo, hi - rr], fmt="o", color="#2457C5", capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks([1, 0], ["Primary window", "Later window"])
    ax.set_xlabel("Adjusted RR per frozen MIMIC-IV SD")
    ax.set_title("MIMIC-III historical replication")
    fig.tight_layout()
    fig.savefig(figure_dir / "mimic3_replication_forest.png", dpi=180)
    fig.savefig(figure_dir / "mimic3_replication_forest.pdf")
    plt.close(fig)
    report = f"""# MIMIC-III replication of SpO2 variability

**{classification}**

The primary HC0 adjusted RR was **{primary['rr_per_mimic_sd']:.3f}** (95% CI **{primary['ci_low']:.3f}–{primary['ci_high']:.3f}**), p={primary['p_value']:.4g}, using {primary['n']:,} stays and {primary['events']} endpoint events. The later-window RR was **{later['rr_per_mimic_sd']:.3f}** (95% CI **{later['ci_low']:.3f}–{later['ci_high']:.3f}**).

Exposure coverage was **{support_counts['exposure_coverage']:.1%}**. The prior coverage and site waivers and HC0 inference are explicit. MIMIC-III and MIMIC-IV derive from the same hospital and may overlap, so this is a historical database-version replication—not independent external validation. It does not change the frozen eICU classification, establish transportability to a new institution, or establish causality or clinical utility.

Execution environment: **LOCAL standalone runtime**, not Colab. No alternate feature, endpoint, threshold, subgroup, unit, or time-window rescue was used.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    verify_lock(verify_sources=False)
    write_artifact_hashes()
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--lock", action="store_true")
    stage.add_argument("--run", action="store_true")
    args = parser.parse_args()
    write_lock() if args.lock else run_locked_analysis()


if __name__ == "__main__":
    main()
