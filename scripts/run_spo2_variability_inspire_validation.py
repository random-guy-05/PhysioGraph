"""Run the locked INSPIRE external validation of SpO2 variability."""

from __future__ import annotations

import argparse
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
from physiograph.analysis.spo2_variability_inspire_validation import (
    FROZEN_ENDPOINT,
    FROZEN_EXPOSURE_PARAMETERS,
    FROZEN_GATES,
    INSPIRE_MAPPING,
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
INSPIRE = DRIVE / "Data/INSPIRE"
OUT = ROOT / "research/spo2_variability_inspire_validation"
PROTOCOL = OUT / "protocol.json"
LOCK = OUT / "analysis_lock.json"
RESULTS = OUT / "validation_results.json"
RECEIPT = OUT / "association_read_receipt.json"
SUPPORT = OUT / "support_counts.json"
FEASIBILITY = OUT / "feasibility.json"
RUN_STATUS = OUT / "run_status.json"
DOCUMENTATION = ROOT / "docs/SPO2_VARIABILITY_INSPIRE_EXTERNAL_VALIDATION.md"
THIRD_PROTOCOL = ROOT / "research/spo2_variability_transportability/THIRD_COHORT_VALIDATION_PROTOCOL.md"
EXTERNAL_CONFIG = ROOT / "research/spo2_variability_validation/external_validation_config.json"
EXTERNAL_LOCK = ROOT / "research/spo2_variability_validation/external_validation_lock.json"
CODE_FILES = [
    "scripts/run_spo2_variability_inspire_validation.py",
    "src/physiograph/analysis/spo2_variability_inspire_validation.py",
    "src/physiograph/analysis/spo2_variability_nwicu_validation.py",
    "src/physiograph/analysis/spo2_variability_transportability.py",
    "src/physiograph/analysis/spo2_variability_validation.py",
    "src/physiograph/analysis/shock_signal_discovery.py",
    "tests/unit/test_spo2_variability_inspire_validation.py",
]
BASE_COVARIATES = [
    "age",
    "male_sex",
    "hr_level",
    "sbp_level",
    "map_level",
    "resp_rate_level",
    "spo2_level",
    "temperature_c_level",
    "baseline_creatinine",
    "baseline_lactate",
    "lactate_observed",
    "bp_measurement_density",
    "urine_measurement_density",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def csv_scan(path: Path) -> str:
    return (
        f"read_csv({sql_path(path)},header=true,all_varchar=true,"
        "quote='\"',escape='\"',strict_mode=false,sample_size=10000)"
    )


def protocol_payload() -> dict[str, Any]:
    return {
        "status": "FROZEN_BEFORE_INSPIRE_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "dataset": "INSPIRE_v1.4.2",
        "population": "adult_explicit_HF_first_ICU_bearing_operation_per_hospital_admission",
        "timeline_origin": "operations.icuin_time",
        "exposure": {
            "source": "ward_vitals.spo2",
            "window_minutes": [0, 240],
            "value_bounds": [50, 100],
            "hourly_aggregation": "median",
            "minimum_bins": 3,
            "minimum_span_hours": 2,
            "feature": "RMS residual around within-stay OLS line",
            "transform": FROZEN_EXPOSURE_PARAMETERS,
        },
        "endpoint": FROZEN_ENDPOINT,
        "mapping": INSPIRE_MAPPING,
        "adjustment_covariates": BASE_COVARIATES,
        "gates": {**FROZEN_GATES, "exact_composite_required": True},
        "user_authorized_waivers": {
            "minimum_70_percent_exposure_coverage": True,
            "minimum_20_contributing_sites": True,
            "minimum_50_modeled_primary_events": False,
            "exact_composite_required": False,
        },
        "primary_covariance": "HC0_due_single_released_site_and_user_site_waiver",
        "confirmation": {
            "primary_rr_gt_1": True,
            "primary_two_sided_95_ci_excludes_1": True,
            "primary_p_lt": 0.05,
            "later_window_rr_gt_1": True,
        },
        "prohibited": [
            "alternate_exposure",
            "dataset_specific_exposure_scaling",
            "alternate_endpoint",
            "threshold_or_time_window_search",
            "subgroup_or_department_rescue",
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
    parameters = pd.read_csv(INSPIRE / "parameters.csv", encoding="utf-8-sig")
    expected = {
        "ward_vitals": {"hr", "nibp_sbp", "nibp_mbp", "rr", "spo2", "bt", "uo", "iabp", "ecmo"},
        "labs": {"creatinine", "lacate", "alt", "ph"},
    }
    for table, labels in expected.items():
        actual = set(parameters.loc[parameters["Table"].eq(table), "Label"].astype(str))
        missing = labels - actual
        if missing:
            raise RuntimeError(f"INSPIRE mapping missing {table} labels: {sorted(missing)}")
    schema = pd.read_csv(INSPIRE / "schema.csv", encoding="utf-8-sig")
    required_tables = {"operations", "diagnosis", "vitals", "ward_vitals", "labs", "medications"}
    table_names = set(schema["Table"].dropna().astype(str))
    if not required_tables.issubset(table_names):
        raise RuntimeError("INSPIRE schema is missing a required table")
    return {
        "parameter_labels": {table: sorted(labels) for table, labels in expected.items()},
        "medication_semantics": "administered medication at chart_time",
        "continuous_support_mapping": INSPIRE_MAPPING["continuous_support"],
        "released_site_count": 1,
    }


def write_lock() -> None:
    if LOCK.exists() or RESULTS.exists() or RECEIPT.exists():
        raise FileExistsError("INSPIRE lock or association artifact already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    source_checks = verify_source_manifest(INSPIRE)
    dictionary = verify_dictionary_mapping()
    ancestors = verify_frozen_ancestors()
    protocol = protocol_payload()
    write_json(PROTOCOL, protocol)
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in CODE_FILES}
    specification = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "documentation_sha256": sha256_file(DOCUMENTATION),
        "source_manifest_sha256": sha256_file(INSPIRE / "SHA256SUMS.txt"),
        "frozen_ancestors": ancestors,
        "endpoint": FROZEN_ENDPOINT,
        "mapping": INSPIRE_MAPPING,
        "exposure_parameters": FROZEN_EXPOSURE_PARAMETERS,
        "code_hashes": code_hashes,
    }
    payload = {
        "status": "LOCKED_BEFORE_INSPIRE_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "lock_utc": utc_now(),
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "source_checks": source_checks,
        "dictionary_verification": dictionary,
        "first_inspire_association_read_utc": None,
        "inspire_association_inspected": False,
    }
    write_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(sha256_file(LOCK) + "\n")
    print(json.dumps({"status": payload["status"], "analysis_lock_sha256": sha256_file(LOCK)}, indent=2))


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text())
    sidecar = LOCK.with_suffix(".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != sha256_file(LOCK):
        raise RuntimeError("INSPIRE whole-file lock hash mismatch")
    specification = lock["specification"]
    if canonical_sha256(specification) != lock["specification_sha256"]:
        raise RuntimeError("INSPIRE specification hash mismatch")
    if sha256_file(PROTOCOL) != specification["protocol_sha256"]:
        raise RuntimeError("INSPIRE protocol hash mismatch")
    if sha256_file(DOCUMENTATION) != specification["documentation_sha256"]:
        raise RuntimeError("INSPIRE documentation hash mismatch")
    for relative, expected in specification["code_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"INSPIRE locked code changed: {relative}")
    verify_frozen_ancestors()
    verify_source_manifest(INSPIRE)
    return lock


def count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def build_inputs() -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET enable_progress_bar=false")
    temporary = Path(os.environ.get("INSPIRE_DUCKDB_TEMP", "/tmp/physiograph-inspire-duckdb"))
    temporary.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={sql_path(temporary)}")
    for table in ("operations", "diagnosis", "ward_vitals", "labs", "medications"):
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM {csv_scan(INSPIRE / (table + '.csv.gz'))}")
    con.execute("""
        CREATE TABLE operation_rows AS
        SELECT try_cast(op_id AS BIGINT) AS op_id,
               try_cast(subject_id AS BIGINT) AS subject_id,
               try_cast(hadm_id AS BIGINT) AS hadm_id,
               try_cast(age AS DOUBLE) AS age,
               CASE upper(trim(sex)) WHEN 'M' THEN 1.0 WHEN 'F' THEN 0.0 END AS male_sex,
               try_cast(admission_time AS DOUBLE) AS admission_time,
               try_cast(discharge_time AS DOUBLE) AS discharge_time,
               try_cast(icuin_time AS DOUBLE) AS icuin_time,
               try_cast(icuout_time AS DOUBLE) AS icuout_time,
               try_cast(inhosp_death_time AS DOUBLE) AS inhosp_death_time
        FROM operations
    """)
    con.execute("""
        CREATE TABLE hf_hadm AS
        SELECT DISTINCT o.hadm_id
        FROM operation_rows o JOIN diagnosis d
          ON try_cast(d.subject_id AS BIGINT)=o.subject_id
        WHERE starts_with(upper(trim(d.icd10_cm)), 'I50')
          AND try_cast(d.chart_time AS DOUBLE) BETWEEN o.admission_time AND o.discharge_time
    """)
    con.execute("""
        CREATE TABLE cohort AS
        WITH candidates AS (
            SELECT o.*,row_number() OVER(
                PARTITION BY hadm_id ORDER BY icuin_time,op_id
            ) AS rn
            FROM operation_rows o
            WHERE icuin_time IS NOT NULL AND icuout_time IS NOT NULL
              AND age>=18 AND hadm_id IN (SELECT hadm_id FROM hf_hadm)
        )
        SELECT op_id AS stay_id,subject_id,hadm_id,age,male_sex,
               admission_time,discharge_time,icuin_time,icuout_time,
               icuout_time-icuin_time AS followup_end_offset_minutes,
               inhosp_death_time-icuin_time AS death_offset_minutes
        FROM candidates WHERE rn=1
          AND icuout_time-icuin_time>360
          AND discharge_time-icuin_time>360
          AND (inhosp_death_time IS NULL OR inhosp_death_time-icuin_time>360)
    """)
    con.execute("""
        CREATE TABLE ward_rows AS
        SELECT c.stay_id,
               try_cast(w.chart_time AS DOUBLE)-c.icuin_time AS event_minute,
               CASE lower(trim(w.item_name))
                   WHEN 'hr' THEN 'hr' WHEN 'nibp_sbp' THEN 'sbp'
                   WHEN 'nibp_mbp' THEN 'map' WHEN 'rr' THEN 'resp_rate'
                   WHEN 'spo2' THEN 'spo2' WHEN 'bt' THEN 'temperature_c'
                   WHEN 'uo' THEN 'urine_output' WHEN 'iabp' THEN 'iabp'
                   WHEN 'ecmo' THEN 'ecmo' END AS signal,
               try_cast(w.value AS DOUBLE) AS value
        FROM ward_vitals w JOIN cohort c ON try_cast(w.subject_id AS BIGINT)=c.subject_id
        WHERE try_cast(w.chart_time AS DOUBLE) BETWEEN c.icuin_time AND least(c.icuout_time,c.icuin_time+960)
          AND lower(trim(w.item_name)) IN ('hr','nibp_sbp','nibp_mbp','rr','spo2','bt','uo','iabp','ecmo')
          AND CASE lower(trim(w.item_name))
              WHEN 'hr' THEN try_cast(w.value AS DOUBLE) BETWEEN 20 AND 250
              WHEN 'nibp_sbp' THEN try_cast(w.value AS DOUBLE) BETWEEN 40 AND 260
              WHEN 'nibp_mbp' THEN try_cast(w.value AS DOUBLE) BETWEEN 30 AND 200
              WHEN 'rr' THEN try_cast(w.value AS DOUBLE) BETWEEN 4 AND 80
              WHEN 'spo2' THEN try_cast(w.value AS DOUBLE) BETWEEN 50 AND 100
              WHEN 'bt' THEN try_cast(w.value AS DOUBLE) BETWEEN 30 AND 43
              WHEN 'uo' THEN try_cast(w.value AS DOUBLE) BETWEEN 0 AND 5000
              WHEN 'iabp' THEN try_cast(w.value AS DOUBLE)>0
              WHEN 'ecmo' THEN try_cast(w.value AS DOUBLE)>0 ELSE false END
    """)
    con.execute("""
        CREATE TABLE ward_hourly AS
        SELECT stay_id,signal,floor(event_minute/60) AS hour,median(value) AS value,
               max(event_minute) AS available_minute,count(*) AS measurements
        FROM ward_rows GROUP BY 1,2,3
    """)
    con.execute("""
        CREATE TABLE lab_rows AS
        SELECT c.stay_id,
               CASE lower(trim(l.item_name)) WHEN 'lacate' THEN 'lactate'
                    WHEN 'creatinine' THEN 'creatinine' WHEN 'alt' THEN 'alt'
                    WHEN 'ph' THEN 'ph' END AS marker,
               try_cast(l.chart_time AS DOUBLE)-c.icuin_time AS event_minute,
               try_cast(l.value AS DOUBLE) AS value
        FROM labs l JOIN cohort c ON try_cast(l.subject_id AS BIGINT)=c.subject_id
        WHERE try_cast(l.chart_time AS DOUBLE)-c.icuin_time BETWEEN -1440 AND 960
          AND try_cast(l.chart_time AS DOUBLE) BETWEEN c.admission_time AND least(c.discharge_time,c.icuin_time+960)
          AND lower(trim(l.item_name)) IN ('lacate','creatinine','alt','ph')
          AND CASE lower(trim(l.item_name))
              WHEN 'lacate' THEN try_cast(l.value AS DOUBLE) BETWEEN 0.1 AND 30
              WHEN 'creatinine' THEN try_cast(l.value AS DOUBLE) BETWEEN 0.1 AND 30
              WHEN 'alt' THEN try_cast(l.value AS DOUBLE) BETWEEN 1 AND 10000
              WHEN 'ph' THEN try_cast(l.value AS DOUBLE) BETWEEN 6.5 AND 8.0 ELSE false END
    """)
    pressor_names = ",".join(f"'{name}'" for name in INSPIRE_MAPPING["continuous_support"]["drug_names"])
    con.execute(f"""
        CREATE TABLE continuous_support AS
        SELECT c.stay_id,try_cast(m.chart_time AS DOUBLE)-c.icuin_time AS event_minute,
               'continuous_support' AS component
        FROM medications m JOIN cohort c ON try_cast(m.subject_id AS BIGINT)=c.subject_id
        WHERE lower(trim(m.route))='iv' AND lower(trim(m.drug_name)) IN ({pressor_names})
          AND try_cast(m.chart_time AS DOUBLE) BETWEEN c.icuin_time AND least(c.icuout_time,c.icuin_time+960)
    """)
    cohort = con.execute("SELECT * FROM cohort").df()
    early = con.execute("SELECT * FROM ward_rows WHERE event_minute BETWEEN 0 AND 240").df()
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
               max(value) FILTER(WHERE signal='map') AS map,
               max(available_minute) AS available_minute
        FROM ward_hourly WHERE signal IN ('sbp','map') GROUP BY 1,2
    """).df()
    hypotension = sustained_hypotension_events(bp)
    support = con.execute("SELECT * FROM continuous_support").df()
    mcs = con.execute("""
        SELECT stay_id,event_minute,'mcs' AS component FROM ward_rows
        WHERE signal IN ('iabp','ecmo') AND value>0
    """).df()
    pressure = pd.concat([hypotension, support, mcs], ignore_index=True)
    lab_rows = con.execute("SELECT * FROM lab_rows").df()
    baseline_rows = lab_rows.loc[lab_rows["event_minute"].le(240)].sort_values("event_minute").groupby(
        ["stay_id", "marker"]
    ).tail(1)
    baseline = (
        baseline_rows.pivot(index="stay_id", columns="marker", values="value").add_prefix("baseline_").reset_index()
        if not baseline_rows.empty else pd.DataFrame(columns=["stay_id"])
    )
    for column in ("baseline_lactate", "baseline_creatinine", "baseline_alt", "baseline_ph"):
        if column not in baseline:
            baseline[column] = np.nan
    labs_after_baseline = lab_rows.loc[lab_rows["event_minute"].gt(240)].merge(baseline, on="stay_id", how="left")
    perfusion_parts: list[pd.DataFrame] = []
    for marker, group in labs_after_baseline.groupby("marker"):
        if marker == "lactate":
            mask = group["value"].ge(4) | (
                group["baseline_lactate"].notna()
                & group["value"].ge(2)
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
    lab_perfusion = (
        pd.concat(perfusion_parts, ignore_index=True)
        if perfusion_parts else pd.DataFrame(columns=["stay_id", "event_minute", "component"])
    )
    urine = con.execute("""
        SELECT stay_id,hour,max(available_minute) AS available_minute,sum(value) AS value
        FROM ward_hourly WHERE signal='urine_output' GROUP BY 1,2
    """).df()
    oliguria = rolling_oliguria_events(urine, start_hour=0, end_hour=16)
    perfusion = pd.concat([lab_perfusion, oliguria], ignore_index=True)
    direct_pre = lab_rows.loc[
        lab_rows["event_minute"].le(360)
        & (
            (lab_rows["marker"].eq("lactate") & lab_rows["value"].ge(2))
            | (lab_rows["marker"].eq("alt") & lab_rows["value"].gt(200))
            | (lab_rows["marker"].eq("ph") & lab_rows["value"].lt(7.2))
        ),
        "stay_id",
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
    urine_density = early.loc[early["signal"].eq("urine_output")].assign(
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
        "preliminary_cohort_n": len(cohort),
        "at_risk_n": len(at_risk),
        "pre_pressure_excluded_n": len(pre_pressure),
        "pre_perfusion_excluded_n": len(pre_perfusion),
        "primary_events": primary_events,
        "later_events": later_events,
        "component_counts": component_counts,
    }


def write_artifact_hashes() -> None:
    artifacts = [path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"]
    write_json(OUT / "artifact_hashes.json", {str(path.relative_to(OUT)): sha256_file(path) for path in sorted(artifacts)})


def write_support_figure(support_counts: dict[str, Any]) -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = ["At risk", "Frozen exposure", "Endpoint events"]
    values = [
        support_counts["at_risk"],
        support_counts["valid_frozen_exposure"],
        support_counts["primary_events_modeled"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#446E9B", "#729FCF", "#CC4C4C"])
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Stays")
    ax.set_title("INSPIRE external-validation support")
    if min(values) > 0:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(figure_dir / "inspire_support_funnel.png", dpi=180)
    fig.savefig(figure_dir / "inspire_support_funnel.pdf")
    plt.close(fig)


def run_locked_analysis() -> None:
    verify_lock()
    if any(path.exists() for path in (SUPPORT, FEASIBILITY, RUN_STATUS, RESULTS, RECEIPT)):
        raise FileExistsError("INSPIRE run artifacts already exist; repeat execution is prohibited")
    inputs = build_inputs()
    model = inputs["model"]
    valid = model["spo2_variability"].notna()
    modeled_events = int(model.loc[valid, "objective_shock"].sum())
    feasibility = evaluate_frozen_feasibility(
        eligible_stays=len(model),
        valid_exposures=int(valid.sum()),
        modeled_primary_events=modeled_events,
        exact_composite_observable=True,
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
            "status": "completed_without_association",
            "classification": feasibility["classification"],
            "environment": "LOCAL_STANDALONE_RUNTIME",
            "association_inspected": False,
            "association_models_fitted": 0,
            "reason": ", ".join(feasibility["failures"]),
        }
        write_json(RUN_STATUS, status)
        report = f"""# INSPIRE SpO2-variability external-validation attempt

**{feasibility['classification']}**

The original frozen endpoint was constructed, but the retained feasibility gate stopped execution before any exposure-outcome association was fitted or inspected.

- At-risk stays: **{support_counts['at_risk']:,}**
- Exact frozen exposure: **{support_counts['valid_frozen_exposure']:,}** ({support_counts['exposure_coverage']:.1%})
- Modeled primary endpoint events: **{support_counts['primary_events_modeled']}**
- Required modeled events: **50**
- Released sites: **1**; the earlier user-authorized site waiver is noted.

Execution environment: **LOCAL standalone runtime**, not Colab. The MIMIC, eICU, and NWICU results remain unchanged.
"""
        (OUT / "FINAL_REPORT.md").write_text(report)
        write_artifact_hashes()
        print(json.dumps({"support": support_counts, "feasibility": feasibility, "run_status": status}, indent=2))
        return
    assert_association_authorized(feasibility)
    read_utc = utc_now()
    receipt = {
        "first_inspire_association_read_utc": read_utc,
        "analysis_lock_sha256": sha256_file(LOCK),
        "attempt": 1,
        "endpoint": FROZEN_ENDPOINT["name"],
    }
    write_json(RECEIPT, receipt)
    primary, _ = fit_fixed_scale_model(
        model,
        feature="spo2_variability",
        outcome="objective_shock",
        adjustment_covariates=BASE_COVARIATES,
        exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    later, _ = fit_fixed_scale_model(
        model,
        feature="spo2_variability",
        outcome="objective_shock_later",
        adjustment_covariates=BASE_COVARIATES,
        exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    confirmed = bool(
        primary["rr_per_mimic_sd"] > 1
        and primary["ci_low"] > 1
        and primary["p_value"] < 0.05
        and later["rr_per_mimic_sd"] > 1
    )
    classification = (
        "INSPIRE_EXTERNALLY_VALIDATED_POST_HOC_SIGNAL"
        if confirmed else "INSPIRE_POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED"
    )
    result = {
        "classification": classification,
        "dataset": "INSPIRE_v1.4.2",
        "primary": primary,
        "later_window": later,
        "support": support_counts,
        "feasibility": feasibility,
        "first_inspire_association_read_utc": read_utc,
        "association_models_fitted": 2,
        "covariance": "HC0_single_released_site",
        "frozen_eicu_classification_unchanged": "POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED",
        "nwicu_result_unchanged": True,
    }
    write_json(RESULTS, result)
    status = {
        "status": "completed",
        "classification": classification,
        "environment": "LOCAL_STANDALONE_RUNTIME",
        "association_inspected": True,
        "association_models_fitted": 2,
    }
    write_json(RUN_STATUS, status)
    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    rr = np.array([primary["rr_per_mimic_sd"], later["rr_per_mimic_sd"]])
    lo = np.array([primary["ci_low"], later["ci_low"]])
    hi = np.array([primary["ci_high"], later["ci_high"]])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(rr, [1, 0], xerr=[rr - lo, hi - rr], fmt="o", color="#2457C5", capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks([1, 0], ["Primary window", "Later window"])
    ax.set_xlabel("Adjusted RR per frozen MIMIC SD")
    ax.set_title("INSPIRE external validation")
    fig.tight_layout()
    fig.savefig(figure_dir / "inspire_external_validation_forest.png", dpi=180)
    fig.savefig(figure_dir / "inspire_external_validation_forest.pdf")
    plt.close(fig)
    report = f"""# INSPIRE external validation of SpO2 variability

**{classification}**

The primary HC0 adjusted RR was **{primary['rr_per_mimic_sd']:.3f}** (95% CI **{primary['ci_low']:.3f}–{primary['ci_high']:.3f}**), p={primary['p_value']:.4g}, using {primary['n']:,} stays and {primary['events']} endpoint events. The later-window RR was **{later['rr_per_mimic_sd']:.3f}** (95% CI **{later['ci_low']:.3f}–{later['ci_high']:.3f}**).

Exposure coverage was **{support_counts['exposure_coverage']:.1%}**. INSPIRE is single-center, so the user-authorized site waiver and HC0 inference are explicitly retained. This is an external cohort validation of the original constructed pressure/support-plus-hypoperfusion endpoint, not adjudicated cardiogenic shock. It does not change the frozen eICU classification or establish causality or clinical utility.

Execution environment: **LOCAL standalone runtime**, not Colab. No alternate feature, endpoint, threshold, subgroup, department, or time-window rescue was used.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    write_artifact_hashes()
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
