"""Forensic, non-rescue audit of the frozen MIMIC-III SpO2 replication."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, spearmanr

from physiograph.analysis.shock_signal_discovery import (
    pair_domain_events,
    rolling_oliguria_events,
    sustained_hypotension_events,
    trajectory_features,
)
from physiograph.analysis.spo2_variability_mimic3_validation import (
    FROZEN_EXPOSURE_PARAMETERS,
    MIMIC3_MAPPING,
    compute_frozen_exposure,
    sha256_file,
)
from physiograph.analysis.spo2_variability_transportability import (
    feature_distribution_rows,
    measurement_process_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT.parents[1]
M3 = DRIVE / "Data/MIMIC III"
M4 = DRIVE / "Data/MIMIC/Full"
M4_DB = (
    DRIVE
    / "Data/PhysioGraph_Biological_Discovery_20260905/objective_shock_single_signal_discovery.duckdb"
)
M4_CONTEXT_DB = (
    DRIVE / "Data/PhysioGraph_Biological_Discovery_20260905/spo2_context.duckdb"
)
OUT = ROOT / "research/spo2_variability_mimic3_forensic"
FIGURES = OUT / "figures"
FROZEN_M3 = ROOT / "research/spo2_variability_mimic3_validation"
FROZEN_M4 = ROOT / "research/spo2_variability_validation"
TRANSPORT = ROOT / "research/spo2_variability_transportability"
DB_PATH = Path("/tmp/physiograph-mimic3-forensic.duckdb")
PROTECTED = [FROZEN_M3, FROZEN_M4, TRANSPORT]
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


def q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def scan(path: Path) -> str:
    return f"read_csv({q(path)},header=true,all_varchar=true,parallel=true,strict_mode=false,sample_size=10000)"


def ints(values: list[int]) -> str:
    return ",".join(map(str, values))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(file for file in path.rglob("*") if file.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): tree_hash(path) for path in PROTECTED}


def create_m3_tables(con: duckdb.DuckDBPyConnection) -> None:
    for name in ("ICUSTAYS", "ADMISSIONS", "PATIENTS", "DIAGNOSES_ICD"):
        con.execute(
            f"CREATE OR REPLACE VIEW m3_{name.lower()} AS SELECT * FROM {scan(M3 / (name + '.csv.gz'))}"
        )
    con.execute("""
        CREATE OR REPLACE TABLE cohort_m3 AS
        WITH dx AS (
          SELECT try_cast(hadm_id AS BIGINT) hadm_id,count(*) diagnosis_count,
                 count(*) FILTER(WHERE starts_with(replace(trim(icd9_code),'.',''),'428')) hf_code_count
          FROM m3_diagnoses_icd GROUP BY 1
        ), candidates AS (
          SELECT try_cast(i.icustay_id AS BIGINT) stay_id,try_cast(i.subject_id AS BIGINT) subject_id,
                 try_cast(i.hadm_id AS BIGINT) hadm_id,try_cast(i.intime AS TIMESTAMP) intime,
                 try_cast(i.outtime AS TIMESTAMP) outtime,try_cast(a.admittime AS TIMESTAMP) admittime,
                 try_cast(a.dischtime AS TIMESTAMP) dischtime,try_cast(a.deathtime AS TIMESTAMP) deathtime,
                 i.first_careunit,
                 CASE WHEN date_diff('day',try_cast(p.dob AS TIMESTAMP),try_cast(a.admittime AS TIMESTAMP))/365.2425>89
                      THEN 90.0 ELSE date_diff('day',try_cast(p.dob AS TIMESTAMP),try_cast(a.admittime AS TIMESTAMP))/365.2425 END age,
                 CASE upper(trim(p.gender)) WHEN 'M' THEN 1.0 WHEN 'F' THEN 0.0 END male_sex,
                 dx.diagnosis_count,dx.hf_code_count,
                 row_number() OVER(PARTITION BY try_cast(i.hadm_id AS BIGINT)
                                   ORDER BY try_cast(i.intime AS TIMESTAMP),try_cast(i.icustay_id AS BIGINT)) rn
          FROM m3_icustays i JOIN m3_admissions a USING(subject_id,hadm_id)
          JOIN m3_patients p USING(subject_id) JOIN dx ON try_cast(i.hadm_id AS BIGINT)=dx.hadm_id
          WHERE dx.hf_code_count>0
        )
        SELECT *,date_diff('second',intime,outtime)/60.0 followup_end_offset_minutes,
               date_diff('second',intime,deathtime)/60.0 death_offset_minutes,
               date_diff('second',admittime,intime)/60.0 admission_to_icu_minutes
        FROM candidates WHERE rn=1 AND age>=18
          AND outtime>intime+INTERVAL 360 MINUTE AND dischtime>intime+INTERVAL 360 MINUTE
          AND (deathtime IS NULL OR deathtime>intime+INTERVAL 360 MINUTE)
    """)
    vital = MIMIC3_MAPPING["vitals"]
    mcs = MIMIC3_MAPPING["mcs"]["itemids"]
    all_chart = sorted(set().union(*map(set, vital.values()), set(mcs)))
    con.execute(f"""
        CREATE OR REPLACE TABLE chart_m3 AS
        SELECT c.stay_id,c.hadm_id,try_cast(x.itemid AS INTEGER) itemid,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 event_minute,
               greatest(date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0,
                        date_diff('second',c.intime,try_cast(x.storetime AS TIMESTAMP))/60.0) store_available_minute,
               try_cast(x.valuenum AS DOUBLE) raw_value,trim(x.value) text_value,x.valueuom,
               coalesce(try_cast(x.error AS INTEGER),0) AS error_flag
        FROM {scan(M3 / "CHARTEVENTS.csv.gz")} x JOIN cohort_m3 c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({ints(all_chart)})
          AND date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 960
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE input_m3 AS
        SELECT c.stay_id,try_cast(x.itemid AS INTEGER) itemid,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 event_minute,
               try_cast(x.rate AS DOUBLE) rate,try_cast(x.originalrate AS DOUBLE) originalrate,
               x.rateuom,x.originalrateuom,x.stopped
        FROM {scan(M3 / "INPUTEVENTS_CV.csv.gz")} x JOIN cohort_m3 c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({ints(MIMIC3_MAPPING["continuous_support"]["itemids"])})
          AND date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 960
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE urine_m3 AS
        SELECT c.stay_id,try_cast(x.itemid AS INTEGER) itemid,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 event_minute,
               date_diff('second',c.intime,try_cast(x.storetime AS TIMESTAMP))/60.0 store_available_minute,
               try_cast(x.value AS DOUBLE) AS "value",x.valueuom,coalesce(try_cast(x.iserror AS INTEGER),0) AS error_flag
        FROM {scan(M3 / "OUTPUTEVENTS.csv.gz")} x JOIN cohort_m3 c
          ON try_cast(x.icustay_id AS BIGINT)=c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({ints(MIMIC3_MAPPING["urine_output"]["itemids"])})
          AND date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 960
    """)
    lab_ids = sorted(set().union(*map(set, MIMIC3_MAPPING["labs"].values())))
    con.execute(f"""
        CREATE OR REPLACE TABLE labs_m3 AS
        SELECT c.stay_id,c.hadm_id,try_cast(x.itemid AS INTEGER) itemid,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 event_minute,
               date_diff('second',c.intime,try_cast(x.charttime AS TIMESTAMP))/60.0 frozen_available_minute,
               try_cast(x.valuenum AS DOUBLE) AS "value",x.valueuom,x.flag
        FROM {scan(M3 / "LABEVENTS.csv.gz")} x JOIN cohort_m3 c
          ON try_cast(x.hadm_id AS BIGINT)=c.hadm_id
        WHERE try_cast(x.itemid AS INTEGER) IN ({ints(lab_ids)})
          AND try_cast(x.charttime AS TIMESTAMP) BETWEEN greatest(c.admittime,c.intime-INTERVAL 1440 MINUTE)
                                                   AND least(c.dischtime,c.outtime,c.intime+INTERVAL 960 MINUTE)
    """)


def create_m4_tables(con: duckdb.DuckDBPyConnection) -> None:
    if not M4_DB.exists():
        raise FileNotFoundError(
            f"Validated MIMIC-IV discovery database missing: {M4_DB}"
        )
    if not M4_CONTEXT_DB.exists():
        raise FileNotFoundError(
            f"Validated MIMIC-IV context database missing: {M4_CONTEXT_DB}"
        )
    con.execute(f"ATTACH {q(M4_DB)} AS frozen_m4 (READ_ONLY)")
    con.execute(f"ATTACH {q(M4_CONTEXT_DB)} AS context_m4 (READ_ONLY)")
    con.execute("""
        CREATE OR REPLACE TABLE cohort_m4 AS
        SELECT c.stay_id,c.person_id AS subject_id,c.hadm_id,c.admit_time AS intime,
               NULL::TIMESTAMP AS outtime,NULL::TIMESTAMP AS admittime,
               NULL::TIMESTAMP AS dischtime,NULL::TIMESTAMP AS deathtime,
               c.icu_type AS first_careunit,c.age,c.is_male AS male_sex,
               NULL::BIGINT AS diagnosis_count,NULL::BIGINT AS hf_code_count,
               c.followup_end_offset_minutes,c.death_offset_minutes,
               NULL::DOUBLE AS admission_to_icu_minutes,c.early_icu_flag,
               c.shock_icd_flag,c.respiratory_source_available
        FROM context_m4.mimic_cohort c
        JOIN frozen_m4.cohort f USING(stay_id)
    """)
    con.execute("""
        CREATE OR REPLACE VIEW chart_m4 AS
        SELECT v.stay_id,c.hadm_id,
               CASE v.signal
                 WHEN 'hr' THEN 220045 WHEN 'sbp' THEN 220179
                 WHEN 'dbp' THEN 220180 WHEN 'map' THEN 220181
                 WHEN 'resp_rate' THEN 220210 WHEN 'spo2' THEN 220277
                 WHEN 'temperature_c' THEN 223762
               END::INTEGER AS itemid,
               v.event_minute,v.available_minute AS store_available_minute,
               v.value AS raw_value,NULL::VARCHAR AS text_value,
               NULL::VARCHAR AS valueuom,0::INTEGER AS error_flag
        FROM frozen_m4.vital_rows v JOIN cohort_m4 c USING(stay_id)
    """)
    con.execute("""
        CREATE OR REPLACE VIEW input_m4 AS
        SELECT stay_id,NULL::INTEGER AS itemid,event_minute,1.0::DOUBLE AS rate,
               NULL::DOUBLE AS originalrate,NULL::VARCHAR AS rateuom,
               NULL::VARCHAR AS originalrateuom,NULL::VARCHAR AS stopped
        FROM frozen_m4.support_events
    """)
    con.execute("""
        CREATE OR REPLACE VIEW mcs_m4 AS
        SELECT stay_id,NULL::INTEGER AS itemid,event_minute
        FROM frozen_m4.mcs_events
    """)
    con.execute("""
        CREATE OR REPLACE VIEW urine_m4 AS
        SELECT stay_id,NULL::INTEGER AS itemid,hour*60.0 AS event_minute,
               available_minute AS store_available_minute,value,
               NULL::VARCHAR AS valueuom,0::INTEGER AS error_flag
        FROM frozen_m4.urine_hourly
    """)
    con.execute("""
        CREATE OR REPLACE VIEW labs_m4 AS
        SELECT l.stay_id,c.hadm_id,
               CASE l.marker WHEN 'lactate' THEN 50813 WHEN 'creatinine' THEN 50912
                 WHEN 'alt' THEN 50861 WHEN 'ph' THEN 50820 END::INTEGER AS itemid,
               l.event_minute,l.event_minute AS frozen_available_minute,
               l.value,NULL::VARCHAR AS valueuom,NULL::VARCHAR AS flag
        FROM frozen_m4.lab_rows l JOIN cohort_m4 c USING(stay_id)
    """)


def normalized_vitals(con: duckdb.DuckDBPyConnection, dataset: str) -> pd.DataFrame:
    if dataset == "MIMIC-III":
        mapping = {
            211: "hr",
            455: "sbp",
            456: "map",
            614: "resp_rate",
            615: "resp_rate",
            618: "resp_rate",
            646: "spo2",
            676: "temperature_c",
            677: "temperature_c",
            678: "temperature_f",
            679: "temperature_f",
        }
        source = con.execute("SELECT * FROM chart_m3 WHERE error_flag=0").df()
        source["signal"] = source["itemid"].map(mapping)
        source["value"] = source["raw_value"]
        f = source["signal"].eq("temperature_f")
        source.loc[f, "value"] = (source.loc[f, "value"] - 32) * 5 / 9
        source.loc[f, "signal"] = "temperature_c"
        source["available_minute"] = source["event_minute"]
    else:
        mapping = {
            220045: "hr",
            220179: "sbp",
            220180: "dbp",
            220181: "map",
            220210: "resp_rate",
            224689: "resp_rate",
            224690: "resp_rate",
            220277: "spo2",
            223761: "temperature_f",
            223762: "temperature_c",
        }
        source = con.execute("SELECT * FROM chart_m4").df()
        source["signal"] = source["itemid"].map(mapping)
        source["value"] = source["raw_value"]
        f = source["signal"].eq("temperature_f")
        source.loc[f, "value"] = (source.loc[f, "value"] - 32) * 5 / 9
        source.loc[f, "signal"] = "temperature_c"
        source["available_minute"] = source["store_available_minute"]
    bounds = {
        "hr": (20, 250),
        "sbp": (40, 260),
        "dbp": (20, 180),
        "map": (30, 200),
        "resp_rate": (4, 80),
        "spo2": (50, 100),
        "temperature_c": (30, 43),
    }
    valid = pd.Series(False, index=source.index)
    for signal, (low, high) in bounds.items():
        valid |= source["signal"].eq(signal) & source["value"].between(low, high)
    return source.loc[
        valid,
        ["stay_id", "event_minute", "available_minute", "signal", "value", "itemid"],
    ].copy()


def assemble(con: duckdb.DuckDBPyConnection, dataset: str) -> dict[str, Any]:
    key = "m3" if dataset == "MIMIC-III" else "m4"
    cohort = con.execute(f"SELECT * FROM cohort_{key}").df()
    vital = normalized_vitals(con, dataset)
    early = vital.loc[vital["event_minute"].between(0, 240)].copy()
    if dataset == "MIMIC-IV":
        early = early.loc[early["available_minute"].le(240)].copy()
    raw_spo2 = early.loc[
        early["signal"].eq("spo2"), ["stay_id", "event_minute", "value"]
    ].rename(columns={"event_minute": "minute", "value": "spo2"})
    if dataset == "MIMIC-IV":
        exposure = con.execute("""
            SELECT stay_id,bins,span_hours,value AS raw_rms_residual
            FROM frozen_m4.feature_long
            WHERE signal='spo2' AND dynamic='variability'
        """).df()
    else:
        exposure = compute_frozen_exposure(raw_spo2)
    pieces: list[pd.DataFrame] = []
    for signal in ("hr", "sbp", "map", "resp_rate", "spo2", "temperature_c"):
        part = early.loc[
            early["signal"].eq(signal), ["stay_id", "event_minute", "value"]
        ].copy()
        if part.empty:
            continue
        part["hour"] = np.floor(part["event_minute"] / 60).astype(int)
        part = part.groupby(["stay_id", "hour"], as_index=False)["value"].median()
        part["signal"] = signal
        pieces.append(part[["stay_id", "signal", "hour", "value"]])
    if dataset == "MIMIC-IV":
        frozen_levels = con.execute("""
            SELECT stay_id,feature,value FROM frozen_m4.feature_long
            WHERE dynamic='level'
        """).df()
        levels = frozen_levels.pivot(
            index="stay_id", columns="feature", values="value"
        ).reset_index()
    else:
        trajectories = trajectory_features(pd.concat(pieces, ignore_index=True))
        levels = (
            trajectories.pivot(index="stay_id", columns="signal", values="level")
            .add_suffix("_level")
            .reset_index()
        )
    for column in (
        "hr_level",
        "sbp_level",
        "map_level",
        "resp_rate_level",
        "spo2_level",
        "temperature_c_level",
    ):
        if column not in levels:
            levels[column] = np.nan
    hourly = (
        vital.assign(hour=np.floor(vital["event_minute"] / 60).astype(int))
        .groupby(["stay_id", "signal", "hour"], as_index=False)
        .agg(value=("value", "median"), available_minute=("available_minute", "max"))
    )
    bp = (
        hourly.loc[hourly["signal"].isin(["sbp", "map"])]
        .pivot_table(
            index=["stay_id", "hour"], columns="signal", values="value", aggfunc="first"
        )
        .reset_index()
    )
    availability = (
        hourly.loc[hourly["signal"].isin(["sbp", "map"])]
        .groupby(["stay_id", "hour"], as_index=False)["available_minute"]
        .max()
    )
    bp = bp.merge(availability, on=["stay_id", "hour"], how="left")
    for name in ("sbp", "map"):
        if name not in bp:
            bp[name] = np.nan
    hypotension = sustained_hypotension_events(bp)
    if dataset == "MIMIC-III":
        input_rows = con.execute("SELECT * FROM input_m3").df()
        support = input_rows.loc[
            input_rows[["rate", "originalrate"]].bfill(axis=1).iloc[:, 0].gt(0)
            & ~input_rows["stopped"]
            .fillna("")
            .str.lower()
            .str.strip()
            .isin(["stopped", "discontinued"]),
            ["stay_id", "event_minute"],
        ].copy()
        support["component"] = "continuous_support"
        chart = con.execute("SELECT * FROM chart_m3").df()
        mcs = chart.loc[
            chart["itemid"].isin(MIMIC3_MAPPING["mcs"]["itemids"])
            & (
                chart["raw_value"].notna()
                | chart["text_value"].fillna("").str.strip().ne("")
            )
            & ~chart["text_value"]
            .fillna("")
            .str.lower()
            .isin(["off", "no", "none", "removed", "discontinued"]),
            ["stay_id", "event_minute"],
        ].copy()
        mcs["component"] = "mcs"
    else:
        support = con.execute(
            "SELECT stay_id,event_minute,'continuous_support' component FROM input_m4"
        ).df()
        mcs = con.execute(
            "SELECT stay_id,event_minute,'mcs' component FROM mcs_m4"
        ).df()
    pressure = pd.concat([hypotension, support, mcs], ignore_index=True)
    labs = con.execute(f"SELECT * FROM labs_{key}").df()
    if dataset == "MIMIC-III":
        lab_map = {50813: "lactate", 50912: "creatinine", 50861: "alt", 50820: "ph"}
    else:
        lab_map = {
            50813: "lactate",
            52442: "lactate",
            53154: "lactate",
            50912: "creatinine",
            52024: "creatinine",
            52546: "creatinine",
            50861: "alt",
            53084: "alt",
            50820: "ph",
        }
    labs["marker"] = labs["itemid"].map(lab_map)
    bounds = {
        "lactate": (0.1, 30),
        "creatinine": (0.1, 30),
        "alt": (1, 10000),
        "ph": (6.5, 8.0),
    }
    lab_valid = pd.Series(False, index=labs.index)
    for marker, (low, high) in bounds.items():
        lab_valid |= labs["marker"].eq(marker) & labs["value"].between(low, high)
    labs = labs.loc[lab_valid].copy()
    if dataset == "MIMIC-IV":
        baseline = con.execute("""
            SELECT stay_id,
                   arg_max(value,event_minute) FILTER(WHERE marker='lactate' AND event_minute<=240) baseline_lactate,
                   arg_max(value,event_minute) FILTER(WHERE marker='creatinine' AND event_minute<=240) baseline_creatinine,
                   arg_max(value,event_minute) FILTER(WHERE marker='alt' AND event_minute<=240) baseline_alt,
                   arg_max(value,event_minute) FILTER(WHERE marker='ph' AND event_minute<=240) baseline_ph
            FROM frozen_m4.lab_rows GROUP BY stay_id
        """).df()
    else:
        baseline_rows = (
            labs.loc[labs["event_minute"].le(240)]
            .sort_values("event_minute")
            .groupby(["stay_id", "marker"])
            .tail(1)
        )
        baseline = (
            baseline_rows.pivot(index="stay_id", columns="marker", values="value")
            .add_prefix("baseline_")
            .reset_index()
        )
    for column in (
        "baseline_lactate",
        "baseline_creatinine",
        "baseline_alt",
        "baseline_ph",
    ):
        if column not in baseline:
            baseline[column] = np.nan
    post = labs.loc[labs["event_minute"].gt(240)].merge(
        baseline, on="stay_id", how="left"
    )
    perfusion_parts: list[pd.DataFrame] = []
    for marker, group in post.groupby("marker"):
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
            mask = (
                group["baseline_ph"].notna()
                & group["baseline_ph"].ge(7.2)
                & group["value"].lt(7.2)
            )
        current = group.loc[mask, ["stay_id", "event_minute"]].copy()
        current["component"] = marker
        perfusion_parts.append(current)
    lab_perfusion = pd.concat(perfusion_parts, ignore_index=True)
    urine_rows = con.execute(f"SELECT * FROM urine_{key}").df()
    urine_rows = urine_rows.loc[
        urine_rows["value"].between(0, 5000) & urine_rows["error_flag"].eq(0)
    ].copy()
    urine_rows["hour"] = np.floor(urine_rows["event_minute"] / 60).astype(int)
    urine = urine_rows.groupby(["stay_id", "hour"], as_index=False).agg(
        available_minute=(
            ("event_minute" if dataset == "MIMIC-III" else "store_available_minute"),
            "max",
        ),
        value=("value", "sum"),
    )
    if dataset == "MIMIC-IV":
        oliguria = rolling_oliguria_events(urine)
    else:
        oliguria = rolling_oliguria_events(urine, start_hour=0, end_hour=16)
    perfusion = pd.concat([lab_perfusion, oliguria], ignore_index=True)
    direct_pre = labs.loc[
        labs["event_minute"].le(360)
        & (
            (labs["marker"].eq("lactate") & labs["value"].ge(2))
            | (labs["marker"].eq("alt") & labs["value"].gt(200))
            | (labs["marker"].eq("ph") & labs["value"].lt(7.2))
        ),
        "stay_id",
    ]
    pre_pressure = set(pressure.loc[pressure["event_minute"].le(360), "stay_id"])
    pre_perfusion = set(direct_pre) | set(
        perfusion.loc[perfusion["event_minute"].le(360), "stay_id"]
    )
    if dataset == "MIMIC-IV":
        pre_urine = rolling_oliguria_events(
            urine.loc[urine["hour"].lt(6), ["stay_id", "hour", "value"]],
            start_hour=0,
            end_hour=6,
        )
        pre_perfusion |= set(pre_urine["stay_id"])
    risk_mask = ~cohort["stay_id"].isin(pre_pressure | pre_perfusion)
    if dataset == "MIMIC-IV":
        risk_mask &= cohort["followup_end_offset_minutes"].gt(360)
        risk_mask &= ~cohort["death_offset_minutes"].between(0, 360).fillna(False)
    at_risk = cohort.loc[risk_mask].copy()
    risk_set_replay_n = len(at_risk)
    if dataset == "MIMIC-IV":
        frozen_at_risk_ids = con.execute(
            "SELECT stay_id FROM frozen_m4.model_base"
        ).df()["stay_id"]
        at_risk = cohort.loc[cohort["stay_id"].isin(frozen_at_risk_ids)].copy()
    primary_all = pair_domain_events(
        pressure, perfusion, lower_minute=360, upper_minute=960
    )
    if dataset == "MIMIC-IV":
        later_all = primary_all.loc[primary_all["event_minute"].gt(480)].copy()
    else:
        later_all = pair_domain_events(
            pressure, perfusion, lower_minute=480, upper_minute=960
        )
    primary = primary_all.loc[primary_all["stay_id"].isin(at_risk["stay_id"])].copy()
    later = later_all.loc[later_all["stay_id"].isin(at_risk["stay_id"])].copy()
    outcomes = at_risk[["stay_id"]].copy()
    outcomes["objective_shock"] = (
        outcomes["stay_id"].isin(primary["stay_id"]).astype(int)
    )
    outcomes["objective_shock_later"] = (
        outcomes["stay_id"].isin(later["stay_id"]).astype(int)
    )
    bp_density = (
        early.loc[early["signal"].isin(["sbp", "map"])]
        .assign(hour=lambda x: np.floor(x["event_minute"] / 60).astype(int))
        .groupby("stay_id")["hour"]
        .nunique()
        .rename("bp_measurement_density")
    )
    urine_density = (
        urine_rows.loc[urine_rows["event_minute"].between(0, 240)]
        .groupby("stay_id")["hour"]
        .nunique()
        .rename("urine_measurement_density")
    )
    model = (
        at_risk.merge(outcomes, on="stay_id")
        .merge(levels, on="stay_id", how="left")
        .merge(
            exposure[["stay_id", "bins", "span_hours", "raw_rms_residual"]].rename(
                columns={"raw_rms_residual": "spo2_variability", "bins": "spo2_bins"}
            ),
            on="stay_id",
            how="left",
        )
        .merge(baseline, on="stay_id", how="left")
        .merge(bp_density, on="stay_id", how="left")
        .merge(urine_density, on="stay_id", how="left")
    )
    model["lactate_observed"] = model["baseline_lactate"].notna().astype(int)
    return {
        "cohort": cohort,
        "at_risk": at_risk,
        "model": model,
        "raw_spo2": raw_spo2,
        "exposure": exposure,
        "pressure": pressure,
        "perfusion": perfusion,
        "primary": primary,
        "later": later,
        "labs": labs,
        "urine": urine,
        "support": support,
        "mcs": mcs,
        "risk_set_replay_n": risk_set_replay_n,
    }


def source_mapping_audit(
    con: duckdb.DuckDBPyConnection, m3: dict[str, Any]
) -> pd.DataFrame:
    items = pd.read_csv(M3 / "D_ITEMS.csv.gz")
    labitems = pd.read_csv(M3 / "D_LABITEMS.csv.gz")
    n_cohort = len(m3["cohort"])
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, list[int], str, tuple[float, float] | None, str]] = [
        (
            "CHARTEVENTS",
            "SpO2 exposure/level",
            [646],
            "0-240 min",
            (50, 100),
            "raw_value",
        ),
        ("CHARTEVENTS", "heart rate", [211], "0-240 min", (20, 250), "raw_value"),
        ("CHARTEVENTS", "cuff SBP/level", [455], "0-960 min", (40, 260), "raw_value"),
        ("CHARTEVENTS", "cuff MAP/level", [456], "0-960 min", (30, 200), "raw_value"),
        (
            "CHARTEVENTS",
            "respiratory rate",
            [614, 615, 618],
            "0-240 min",
            (4, 80),
            "raw_value",
        ),
        (
            "CHARTEVENTS",
            "temperature",
            [676, 677, 678, 679],
            "0-240 min",
            None,
            "raw_value",
        ),
        (
            "CHARTEVENTS",
            "recorded MCS",
            MIMIC3_MAPPING["mcs"]["itemids"],
            "0-960 min",
            None,
            "raw_value",
        ),
        (
            "INPUTEVENTS_CV",
            "continuous vasoactive/inotrope support",
            MIMIC3_MAPPING["continuous_support"]["itemids"],
            "0-960 min",
            (0, math.inf),
            "rate",
        ),
        (
            "OUTPUTEVENTS",
            "urine output",
            MIMIC3_MAPPING["urine_output"]["itemids"],
            "0-960 min",
            (0, 5000),
            "value",
        ),
        ("LABEVENTS", "lactate", [50813], "-1440-960 min", (0.1, 30), "value"),
        ("LABEVENTS", "pH", [50820], "-1440-960 min", (6.5, 8), "value"),
        ("LABEVENTS", "ALT", [50861], "-1440-960 min", (1, 10000), "value"),
        ("LABEVENTS", "creatinine", [50912], "-1440-960 min", (0.1, 30), "value"),
    ]
    table_frames = {
        "CHARTEVENTS": con.execute("SELECT * FROM chart_m3").df(),
        "INPUTEVENTS_CV": con.execute("SELECT * FROM input_m3").df(),
        "OUTPUTEVENTS": con.execute("SELECT * FROM urine_m3").df(),
        "LABEVENTS": con.execute("SELECT * FROM labs_m3").df(),
    }
    for table, role, itemids, time_range, bounds, value_column in specs:
        frame = table_frames[table]
        for itemid in itemids:
            local = frame.loc[frame["itemid"].eq(itemid)].copy()
            if role == "SpO2 exposure/level" or role in (
                "heart rate",
                "respiratory rate",
                "temperature",
            ):
                local = local.loc[local["event_minute"].between(0, 240)]
            dictionary = labitems if table == "LABEVENTS" else items
            match = dictionary.loc[
                pd.to_numeric(dictionary["itemid"], errors="coerce").eq(itemid)
            ]
            label = str(match.iloc[0]["label"]) if len(match) else "UNMAPPED"
            dictionary_unit = (
                str(match.iloc[0].get("unitname", "")) if len(match) else ""
            )
            values = pd.to_numeric(local.get(value_column), errors="coerce")
            if table == "INPUTEVENTS_CV":
                values = local[["rate", "originalrate"]].bfill(axis=1).iloc[:, 0]
                units = local["rateuom"].fillna(local["originalrateuom"])
            else:
                units = local.get("valueuom", pd.Series(dtype=object))
            if bounds is None:
                implausible = 0
            elif math.isinf(bounds[1]):
                implausible = int(values.notna().sum() - values.gt(bounds[0]).sum())
            else:
                implausible = int(values.notna().sum() - values.between(*bounds).sum())
            common = (
                local.get("text_value", values)
                .astype(str)
                .value_counts()
                .head(5)
                .to_dict()
                if len(local)
                else {}
            )
            rows.append(
                {
                    "variable_role": role,
                    "source_table": table,
                    "itemid": itemid,
                    "label": label,
                    "dictionary_unit": dictionary_unit
                    if dictionary_unit != "nan"
                    else "",
                    "observed_units": json.dumps(
                        units.dropna().astype(str).value_counts().head(8).to_dict(),
                        sort_keys=True,
                    ),
                    "valid_time_range": time_range,
                    "rows": len(local),
                    "patients_or_stays": int(local["stay_id"].nunique()),
                    "percentage_missing_stays": 100
                    * (1 - local["stay_id"].nunique() / n_cohort),
                    "numeric_missing_rows": int(values.isna().sum()),
                    "min": values.min(),
                    "median": values.median(),
                    "max": values.max(),
                    "common_values": json.dumps(common, sort_keys=True),
                    "implausible_values": implausible,
                    "impossible_timestamp_rows": int(
                        (
                            ~local["event_minute"].between(
                                -1440 if table == "LABEVENTS" else 0, 960
                            )
                        ).sum()
                    ),
                    "manual_mapping_assessment": (
                        "Peripheral pulse-ox saturation only; label is SpO2 and observed unit is %. No SaO2/venous O2/FiO2 item is mixed."
                        if itemid == 646
                        else "Dictionary label reviewed against frozen mapping."
                    ),
                }
            )
    cohort = m3["cohort"]
    for variable, table in (
        ("age", "PATIENTS+ADMISSIONS"),
        ("male_sex", "PATIENTS"),
        ("first_careunit", "ICUSTAYS"),
        ("diagnosis_count", "DIAGNOSES_ICD"),
        ("hf_code_count", "DIAGNOSES_ICD"),
    ):
        values = (
            pd.to_numeric(cohort[variable], errors="coerce")
            if variable != "first_careunit"
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "variable_role": variable,
                "source_table": table,
                "itemid": "",
                "label": variable,
                "dictionary_unit": "",
                "observed_units": "",
                "valid_time_range": "admission/stay level",
                "rows": len(cohort),
                "patients_or_stays": int(cohort["stay_id"].nunique()),
                "percentage_missing_stays": 100 * cohort[variable].isna().mean(),
                "numeric_missing_rows": int(cohort[variable].isna().sum()),
                "min": values.min() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "max": values.max() if len(values) else np.nan,
                "common_values": json.dumps(
                    cohort[variable].astype(str).value_counts().head(8).to_dict(),
                    sort_keys=True,
                ),
                "implausible_values": 0,
                "impossible_timestamp_rows": 0,
                "manual_mapping_assessment": "Admission-level linkage reviewed.",
            }
        )
    return pd.DataFrame(rows)


def extra_measurement_rows(
    raw: pd.DataFrame,
    ids: pd.Series,
    dataset: str,
    features: pd.DataFrame,
    per_stay: pd.DataFrame,
) -> pd.DataFrame:
    clean = (
        raw.loc[raw["stay_id"].isin(ids) & raw["spo2"].between(50, 100)]
        .sort_values(["stay_id", "minute"])
        .copy()
    )
    clean["hour"] = np.floor(clean["minute"] / 60).astype(int)
    rows: list[dict[str, Any]] = []
    gap = clean.groupby("stay_id")["minute"].diff()
    longest = (
        gap.groupby(clean["stay_id"]).max().reindex(pd.Index(ids.unique())).fillna(0)
    )
    for stat, value in (
        ("median", longest.median()),
        ("p25", longest.quantile(0.25)),
        ("p75", longest.quantile(0.75)),
        ("p95", longest.quantile(0.95)),
    ):
        rows.append(
            {
                "dataset": dataset,
                "metric": "longest_gap_per_stay",
                "statistic": stat,
                "value": float(value),
                "n": len(longest),
                "unit": "minutes",
            }
        )
    merged = per_stay.merge(
        features[["stay_id", "raw_rms_residual"]], on="stay_id", how="left"
    )
    fraction = float(
        ((merged["raw_observations"] > 4) & (merged["valid_bins"] == 3)).mean()
    )
    rows.append(
        {
            "dataset": dataset,
            "metric": "more_than_4_raw_but_only_3_bins",
            "statistic": "fraction",
            "value": fraction,
            "n": len(merged),
            "unit": "fraction",
        }
    )
    for digit in range(10):
        fraction_digit = (
            float((np.round(clean["spo2"]).astype(int) % 10 == digit).mean())
            if len(clean)
            else np.nan
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": "terminal_digit_distribution",
                "statistic": str(digit),
                "value": fraction_digit,
                "n": len(clean),
                "unit": "fraction",
            }
        )
    for variable in ("raw_observations", "valid_bins", "gap_max", "identical_fraction"):
        rho, p = spearmanr(
            merged[variable], merged["raw_rms_residual"], nan_policy="omit"
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": f"exposure_vs_{variable}",
                "statistic": "spearman_rho",
                "value": float(rho),
                "n": int(merged[[variable, "raw_rms_residual"]].dropna().shape[0]),
                "unit": "correlation",
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": f"exposure_vs_{variable}",
                "statistic": "p_value_descriptive",
                "value": float(p),
                "n": int(merged[[variable, "raw_rms_residual"]].dropna().shape[0]),
                "unit": "p_value",
            }
        )
    return pd.DataFrame(rows)


def smd(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    try:
        mean_a, sd_a, mean_b, sd_b = map(float, (mean_a, sd_a, mean_b, sd_b))
    except (TypeError, ValueError):
        return math.nan
    if not all(math.isfinite(value) for value in (mean_a, sd_a, mean_b, sd_b)):
        return math.nan
    denominator = math.sqrt((sd_a**2 + sd_b**2) / 2)
    return (mean_b - mean_a) / denominator if denominator > 0 else math.nan


def coarse_icu_group(value: object) -> str | None:
    if pd.isna(value):
        return None
    label = str(value).strip().lower()
    if not label:
        return None
    if "medical/surgical" in label or "micu/sicu" in label:
        return "mixed_medical_surgical"
    if "cardiac vascular" in label or "cvicu" in label or label == "csru":
        return "cardiac_surgical"
    if "coronary" in label or label == "ccu":
        return "cardiac"
    if "trauma" in label or "tsicu" in label:
        return "trauma_surgical"
    if "surgical intensive" in label or label == "sicu":
        return "surgical"
    if "medical intensive" in label or label == "micu":
        return "medical"
    if "neuro" in label:
        return "neurologic"
    if "stepdown" in label or "intermediate" in label:
        return "intermediate_stepdown"
    if "pacu" in label:
        return "perioperative"
    return "other_or_general"


def cohort_comparison(m4: dict[str, Any], m3: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stages = {
        "preliminary": (m4["cohort"], m3["cohort"]),
        "at_risk": (m4["model"], m3["model"]),
        "exposure_observed": (
            m4["model"].dropna(subset=["spo2_variability"]),
            m3["model"].dropna(subset=["spo2_variability"]),
        ),
        "modeled": (
            m4["model"].dropna(subset=["spo2_variability"]),
            m3["model"].dropna(subset=["spo2_variability"]),
        ),
    }
    variables = [
        "age",
        "male_sex",
        "spo2_level",
        "sbp_level",
        "map_level",
        "hr_level",
        "resp_rate_level",
        "baseline_lactate",
        "baseline_creatinine",
        "diagnosis_count",
        "hf_code_count",
    ]
    for stage, (a, b) in stages.items():
        for variable in variables:
            if variable not in a or variable not in b:
                continue
            av = pd.to_numeric(a[variable], errors="coerce")
            bv = pd.to_numeric(b[variable], errors="coerce")
            a_mean = float(av.mean()) if av.notna().any() else math.nan
            a_sd = float(av.std(ddof=0)) if av.notna().any() else math.nan
            b_mean = float(bv.mean()) if bv.notna().any() else math.nan
            b_sd = float(bv.std(ddof=0)) if bv.notna().any() else math.nan
            rows.append(
                {
                    "stage": stage,
                    "variable": variable,
                    "mimic4_n": int(av.notna().sum()),
                    "mimic4_mean_or_prevalence": a_mean,
                    "mimic4_sd": a_sd,
                    "mimic3_n": int(bv.notna().sum()),
                    "mimic3_mean_or_prevalence": b_mean,
                    "mimic3_sd": b_sd,
                    "standardized_mean_difference_m3_minus_m4": smd(
                        a_mean, a_sd, b_mean, b_sd
                    ),
                    "note": "continuous/binary standardized mean difference",
                }
            )
        a_icu = a["first_careunit"].map(coarse_icu_group)
        b_icu = b["first_careunit"].map(coarse_icu_group)
        categories = sorted(set(a_icu.dropna()) | set(b_icu.dropna()))
        for category in categories:
            pa = a_icu.eq(category).mean()
            pb = b_icu.eq(category).mean()
            rows.append(
                {
                    "stage": stage,
                    "variable": f"icu_group:{category}",
                    "mimic4_n": len(a),
                    "mimic4_mean_or_prevalence": pa,
                    "mimic4_sd": math.sqrt(pa * (1 - pa)),
                    "mimic3_n": len(b),
                    "mimic3_mean_or_prevalence": pb,
                    "mimic3_sd": math.sqrt(pb * (1 - pb)),
                    "standardized_mean_difference_m3_minus_m4": smd(
                        pa, math.sqrt(pa * (1 - pa)), pb, math.sqrt(pb * (1 - pb))
                    ),
                    "note": "coarsely harmonized ICU category proportion; source taxonomies differ by era",
                }
            )
        for variable in ("ventilation", "respiratory_support", "fio2"):
            rows.append(
                {
                    "stage": stage,
                    "variable": variable,
                    "mimic4_n": 0,
                    "mimic4_mean_or_prevalence": np.nan,
                    "mimic4_sd": np.nan,
                    "mimic3_n": 0,
                    "mimic3_mean_or_prevalence": np.nan,
                    "mimic3_sd": np.nan,
                    "standardized_mean_difference_m3_minus_m4": np.nan,
                    "note": "not harmonized in either frozen model frame",
                }
            )
        respiratory_available = pd.to_numeric(
            a.get("respiratory_source_available", pd.Series(dtype=float)),
            errors="coerce",
        )
        rows.append(
            {
                "stage": stage,
                "variable": "respiratory_source_available",
                "mimic4_n": int(respiratory_available.notna().sum()),
                "mimic4_mean_or_prevalence": (
                    float(respiratory_available.mean())
                    if respiratory_available.notna().any()
                    else np.nan
                ),
                "mimic4_sd": (
                    float(respiratory_available.std(ddof=0))
                    if respiratory_available.notna().any()
                    else np.nan
                ),
                "mimic3_n": 0,
                "mimic3_mean_or_prevalence": np.nan,
                "mimic3_sd": np.nan,
                "standardized_mean_difference_m3_minus_m4": np.nan,
                "note": "MIMIC-IV source-availability flag; no harmonized MIMIC-III respiratory-support variable in the frozen frame",
            }
        )
        for label, obj in (("MIMIC-IV", m4), ("MIMIC-III", m3)):
            stage_ids = set((a if label == "MIMIC-IV" else b)["stay_id"])
            early_support = obj["support"].loc[
                obj["support"]["stay_id"].isin(stage_ids)
                & obj["support"]["event_minute"].between(0, 240),
                "stay_id",
            ].nunique() / max(len(stage_ids), 1)
            column = (
                "mimic4_mean_or_prevalence"
                if label == "MIMIC-IV"
                else "mimic3_mean_or_prevalence"
            )
            rows.append(
                {
                    "stage": stage,
                    "variable": f"early_vasoactive_prevalence:{label}",
                    "mimic4_n": len(a),
                    "mimic4_mean_or_prevalence": early_support
                    if label == "MIMIC-IV"
                    else np.nan,
                    "mimic4_sd": np.nan,
                    "mimic3_n": len(b),
                    "mimic3_mean_or_prevalence": early_support
                    if label == "MIMIC-III"
                    else np.nan,
                    "mimic3_sd": np.nan,
                    "standardized_mean_difference_m3_minus_m4": np.nan,
                    "note": f"dataset-specific row; value stored in {column}",
                }
            )
    return pd.DataFrame(rows)


def endpoint_rows(dataset: str, obj: dict[str, Any]) -> pd.DataFrame:
    ids = set(obj["at_risk"]["stay_id"])
    pressure = obj["pressure"].loc[
        obj["pressure"]["stay_id"].isin(ids)
        & obj["pressure"]["event_minute"].gt(360)
        & obj["pressure"]["event_minute"].le(960)
    ]
    perfusion = obj["perfusion"].loc[
        obj["perfusion"]["stay_id"].isin(ids)
        & obj["perfusion"]["event_minute"].gt(360)
        & obj["perfusion"]["event_minute"].le(960)
    ]
    endpoint = obj["primary"]
    n = len(ids)
    rows = [
        {
            "dataset": dataset,
            "section": "endpoint",
            "component": "primary",
            "count": len(endpoint),
            "denominator": n,
        },
        {
            "dataset": dataset,
            "section": "endpoint",
            "component": "later",
            "count": len(obj["later"]),
            "denominator": n,
        },
        {
            "dataset": dataset,
            "section": "domain",
            "component": "pressure_or_support_positive",
            "count": pressure["stay_id"].nunique(),
            "denominator": n,
        },
        {
            "dataset": dataset,
            "section": "domain",
            "component": "hypoperfusion_positive",
            "count": perfusion["stay_id"].nunique(),
            "denominator": n,
        },
    ]
    for section, frame in (
        ("pressure_component", pressure),
        ("hypoperfusion_component", perfusion),
    ):
        for component, count in frame.groupby("component")["stay_id"].nunique().items():
            rows.append(
                {
                    "dataset": dataset,
                    "section": section,
                    "component": component,
                    "count": int(count),
                    "denominator": n,
                }
            )
    for column, section in (
        ("pressure_component", "endpoint_completing_pressure"),
        ("perfusion_component", "endpoint_completing_hypoperfusion"),
    ):
        for component, count in endpoint[column].value_counts().items():
            rows.append(
                {
                    "dataset": dataset,
                    "section": section,
                    "component": component,
                    "count": int(count),
                    "denominator": len(endpoint),
                }
            )
    for (pc, hc), count in (
        endpoint.groupby(["pressure_component", "perfusion_component"]).size().items()
    ):
        rows.append(
            {
                "dataset": dataset,
                "section": "overlap",
                "component": f"{pc}|{hc}",
                "count": int(count),
                "denominator": len(endpoint),
            }
        )
    result = pd.DataFrame(rows)
    result["fraction"] = result["count"] / result["denominator"]
    result["availability"] = "available"
    result["timestamp_fidelity"] = (
        "chart/store availability or procedure start"
        if dataset == "MIMIC-IV"
        else "CareVue charttime; repeated infusion/device documentation"
    )
    result["structurally_weaker_than_other_era"] = dataset == "MIMIC-III"
    return result


def precision_and_heterogeneity() -> tuple[dict[str, Any], dict[str, Any]]:
    m3 = json.loads((FROZEN_M3 / "validation_results.json").read_text())["primary"]
    m4 = json.loads((FROZEN_M4 / "validation_results.json").read_text())["mimic"][
        "primary"
    ]
    beta3 = float(m3["log_rr"])
    beta4 = float(m4["log_rr"])
    se3 = (math.log(m3["ci_high"]) - math.log(m3["ci_low"])) / (2 * 1.96)
    se4 = (math.log(m4["ci_high"]) - math.log(m4["ci_low"])) / (2 * 1.96)
    z_alpha = NormalDist().inv_cdf(0.975)
    powers = []
    for rr in (1.10, 1.15, 1.20, 1.25, 1.30):
        ncp = math.log(rr) / se3
        power = norm.sf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        powers.append({"true_rr": rr, "approximate_power": float(power)})
    required = []
    for rr in (1.20, 1.25):
        for target in (0.80, 0.90):
            events = int(
                math.ceil(
                    m3["events"]
                    * (
                        ((z_alpha + NormalDist().inv_cdf(target)) * se3 / math.log(rr))
                        ** 2
                    )
                )
            )
            required.append(
                {"true_rr": rr, "target_power": target, "required_events": events}
            )
    precision = {
        "log_rr": beta3,
        "robust_se": se3,
        "rr_ci_width": float(m3["ci_high"] - m3["ci_low"]),
        "log_ci_width": float(math.log(m3["ci_high"]) - math.log(m3["ci_low"])),
        "multiplicative_95_ci_factor": math.exp(1.96 * se3),
        "events": int(m3["events"]),
        "power": powers,
        "required_events": required,
        "mimic_iv_rr_inside_mimic_iii_ci": bool(
            m3["ci_low"] <= m4["rr_per_mimic_sd"] <= m3["ci_high"]
        ),
        "method": "two-sided normal approximation using observed HC0 SE; inverse-event information scaling",
        "guard": "The 50-event gate is a minimum credibility gate, not proof of adequate power.",
    }
    difference = beta4 - beta3
    se_difference = math.sqrt(se4**2 + se3**2)
    z = difference / se_difference
    q_stat = z**2
    p = float(chi2.sf(q_stat, 1))
    i2 = max(0.0, (q_stat - 1) / q_stat) * 100 if q_stat > 0 else 0.0
    heterogeneity = {
        "mimic_iv": {"log_rr": beta4, "se": se4, "rr": m4["rr_per_mimic_sd"]},
        "mimic_iii": {"log_rr": beta3, "se": se3, "rr": m3["rr_per_mimic_sd"]},
        "log_rr_difference_mimic_iv_minus_mimic_iii": difference,
        "se_of_difference": se_difference,
        "z": z,
        "p_value": p,
        "q": q_stat,
        "df": 1,
        "i_squared_percent": i2,
        "formal_cross_era_heterogeneity_present_at_0_05": p < 0.05,
        "interpretation": "Two-study I-squared is unstable; no pooling is used for confirmation.",
    }
    return precision, heterogeneity


def parity_matrix() -> pd.DataFrame:
    rows = [
        (
            "Exposure window",
            "charttime 0-240 inclusive plus storetime<=240",
            "charttime 0-240 inclusive; no storetime gate",
            "materially different",
            "M4 vital_candidates vs M3 early filter",
            "M3 permits measurements stored after landmark",
        ),
        (
            "Time origin",
            "ICUSTAYS.intime",
            "ICUSTAYS.intime",
            "identical",
            "both subtract ICU intime",
            "none",
        ),
        (
            "Hourly bins",
            "floor(minute/60); hourly median",
            "floor(minute/60); hourly median",
            "identical",
            "shared trajectory_features",
            "minute 240 creates hour 4 in both",
        ),
        (
            "Duplicate timestamps",
            "retained then absorbed by hourly median",
            "retained then absorbed by hourly median",
            "identical",
            "no pre-hour deduplication",
            "duplicate feeds can weight median in both",
        ),
        (
            "Support for exposure",
            ">=3 bins and span>=2 h",
            ">=3 bins and span>=2 h",
            "identical",
            "shared trajectory_features",
            "none",
        ),
        (
            "Trend/RMS",
            "OLS with intercept on hour; sqrt(mean(residual^2))",
            "same shared implementation",
            "identical",
            "numpy.linalg.lstsq and mean denominator n",
            "none",
        ),
        (
            "Clipping/scaling",
            "M4 p1/p99, clipped mean/SD",
            "exact frozen constants; no recentering",
            "identical",
            json.dumps(FROZEN_EXPOSURE_PARAMETERS),
            "none",
        ),
        (
            "SpO2 validity",
            "50<=SpO2<=100; item 220277",
            "50<=SpO2<=100; item 646",
            "equivalent",
            "dictionary labels both SpO2",
            "different EHR generations/device processes",
        ),
        (
            "Cohort observation",
            "preliminary follow-up >240 min; at-risk follow-up/death beyond 360 min; ICU within 24 h or shock ICD",
            "preliminary follow-up/death beyond 360 min; no ICU-entry-time restriction",
            "materially different",
            "cohort SQL",
            "preliminary selection and ICU-entry phenotype differ; neither era requires observation through minute 960",
        ),
        (
            "HF phenotype",
            "ICD-9 428* or ICD-10 I50*",
            "ICD-9 428*",
            "equivalent",
            "era-appropriate diagnosis codes",
            "both use retrospective admission diagnoses",
        ),
        (
            "First ICU",
            "first stay per hadm by intime/stay_id",
            "first stay per hadm by intime/icustay_id",
            "equivalent",
            "row_number per hospital admission",
            "identifier names differ only",
        ),
        (
            "BP timing",
            "charttime with max(charttime,storetime) availability",
            "charttime only",
            "materially different",
            "sustained hypotension inputs",
            "M3 may recognize late-entered BP earlier",
        ),
        (
            "Lab timing",
            "max(charttime,storetime)",
            "charttime only; LABEVENTS has no storetime",
            "materially different",
            "normalized lab SQL",
            "availability chronology not equivalent",
        ),
        (
            "Urine timing",
            "max(charttime,storetime)",
            "charttime only",
            "materially different",
            "urine hourly availability",
            "M3 may recognize late-entered output earlier",
        ),
        (
            "Oliguria surveillance start",
            "shared helper default start_hour=4, end_hour=16",
            "shared helper explicitly start_hour=0, end_hour=16",
            "materially different",
            "frozen M4 discovery endpoint cell vs frozen M3 validation script",
            "different eligible six-hour urine windows and endpoint timing",
        ),
        (
            "Vasoactive support",
            "positive interval starttime; end>start",
            "positive CareVue rate row at charttime",
            "materially different",
            "INPUTEVENTS source semantics",
            "initiation versus repeated documentation",
        ),
        (
            "MCS",
            "procedure start itemids",
            "IABP/ECMO/VAD chart-field presence",
            "materially different",
            "procedureevents vs CareVue chartevents",
            "documentation frequency and onset fidelity differ",
        ),
        (
            "Hypoperfusion thresholds",
            "same lactate/creatinine/ALT/pH/oliguria rules",
            "same shared thresholds/helpers",
            "identical",
            "endpoint reconstruction parity",
            "source timing remains different",
        ),
        (
            "Composite logic",
            "pressure AND hypoperfusion; both >360 <=960; <=360 min separation",
            "same",
            "identical",
            "shared pair_domain_events",
            "none at logical level",
        ),
        (
            "Later endpoint",
            "primary composite event time >480; a component may occur during minutes 360-480",
            "composite re-paired with both components strictly >480",
            "materially different",
            "M4 objective_shock_lead4 vs M3 later_events construction",
            "later-event counts are descriptive but not definitionally interchangeable",
        ),
        (
            "Covariate list",
            ",".join(BASE_COVARIATES),
            ",".join(BASE_COVARIATES),
            "identical",
            "locked protocols",
            "none",
        ),
        (
            "Covariate preparation",
            "within-cohort p1/p99, median imputation, z scaling, missing flags",
            "same shared function",
            "identical",
            "prepare_adjustment_covariates",
            "dataset-specific covariate transforms occur in both by design",
        ),
        (
            "Model",
            "modified Poisson log link, HC0",
            "modified Poisson log link, HC0",
            "identical",
            "fit_fixed_scale_model",
            "single-hospital inference in both",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "component",
            "MIMIC-IV definition",
            "MIMIC-III definition",
            "identical / equivalent / materially different",
            "evidence",
            "consequence",
        ],
    )


def figures(
    measurement: pd.DataFrame, exposure: pd.DataFrame, endpoint: pd.DataFrame
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    def save(fig: plt.Figure, name: str) -> None:
        for suffix in ("png", "pdf"):
            fig.savefig(FIGURES / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    median_obs = measurement.loc[
        (measurement.metric == "raw_observations_per_stay")
        & (measurement.statistic == "median")
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(median_obs.dataset, median_obs.value, color=["#446E9B", "#CC6677"])
    ax.set_ylabel("Median raw SpO2 observations/stay")
    save(fig, "measurement_density")
    reps = measurement.loc[
        measurement.metric.isin(
            ["consecutive_observations_identical", "raw_observations_equal_100"]
        )
        & measurement.statistic.eq("fraction")
    ]
    pivot = reps.pivot(index="dataset", columns="metric", values="value")
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("Fraction")
    ax.legend(["Consecutive identical", "Equal to 100"], fontsize=8)
    save(fig, "repetition_quantization")
    dist = exposure.loc[
        (exposure.stratum == "all_valid")
        & (exposure.scale == "mimic_standardized")
        & exposure.statistic.isin(["mean", "sd", "median", "iqr"])
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    dist.pivot(index="dataset", columns="statistic", values="value").plot.bar(ax=ax)
    ax.set_ylabel("Frozen MIMIC-IV SD units")
    save(fig, "exposure_distribution")
    comp = endpoint.loc[(endpoint.section == "endpoint_completing_hypoperfusion")]
    fig, ax = plt.subplots(figsize=(8, 4))
    comp.pivot(index="component", columns="dataset", values="fraction").fillna(
        0
    ).plot.bar(ax=ax)
    ax.set_ylabel("Fraction of primary endpoints")
    save(fig, "endpoint_composition")
    m4 = json.loads((FROZEN_M4 / "validation_results.json").read_text())["mimic"][
        "primary"
    ]
    m3 = json.loads((FROZEN_M3 / "validation_results.json").read_text())["primary"]
    rr = np.array([m4["rr_per_mimic_sd"], m3["rr_per_mimic_sd"]])
    lo = np.array([m4["ci_low"], m3["ci_low"]])
    hi = np.array([m4["ci_high"], m3["ci_high"]])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.errorbar(rr, [1, 0], xerr=[rr - lo, hi - rr], fmt="o", capsize=3)
    ax.axvline(1, color="black", lw=0.8)
    ax.set_yticks([1, 0], ["MIMIC-IV", "MIMIC-III"])
    ax.set_xlabel("Adjusted RR per frozen MIMIC-IV SD")
    save(fig, "mimic4_mimic3_forest")


def main() -> None:
    before = protected_hashes()
    lock = json.loads((FROZEN_M3 / "analysis_lock.json").read_text())
    assert (
        sha256_file(FROZEN_M3 / "analysis_lock.json")
        == (FROZEN_M3 / "analysis_lock.sha256").read_text().strip()
    )
    for relative, expected in lock["specification"]["code_hashes"].items():
        assert sha256_file(ROOT / relative) == expected
    Path("/tmp/physiograph-mimic3-forensic-temp").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("SET threads=4")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET temp_directory='/tmp/physiograph-mimic3-forensic-temp'")
    create_m3_tables(con)
    create_m4_tables(con)
    m3 = assemble(con, "MIMIC-III")
    m4 = assemble(con, "MIMIC-IV")
    reconstructed = {
        "mimic3": {
            "preliminary": len(m3["cohort"]),
            "at_risk": len(m3["at_risk"]),
            "valid_exposure": int(m3["model"]["spo2_variability"].notna().sum()),
            "primary_events": len(m3["primary"]),
            "modeled_events": int(
                m3["model"]
                .loc[m3["model"]["spo2_variability"].notna(), "objective_shock"]
                .sum()
            ),
            "later_modeled_events": int(
                m3["model"]
                .loc[m3["model"]["spo2_variability"].notna(), "objective_shock_later"]
                .sum()
            ),
        },
        "mimic4": {
            "preliminary": len(m4["cohort"]),
            "at_risk": len(m4["at_risk"]),
            "valid_exposure": int(m4["model"]["spo2_variability"].notna().sum()),
            "primary_events": len(m4["primary"]),
            "modeled_events": int(
                m4["model"]
                .loc[m4["model"]["spo2_variability"].notna(), "objective_shock"]
                .sum()
            ),
        },
    }
    assert reconstructed["mimic3"] == {
        "preliminary": 5405,
        "at_risk": 2368,
        "valid_exposure": 1799,
        "primary_events": 103,
        "modeled_events": 72,
        "later_modeled_events": 53,
    }
    assert reconstructed["mimic4"]["preliminary"] == 17758
    assert m4["risk_set_replay_n"] == 8196
    assert reconstructed["mimic4"]["at_risk"] == 8196
    assert reconstructed["mimic4"]["valid_exposure"] == 5957
    assert reconstructed["mimic4"]["primary_events"] == 238
    assert reconstructed["mimic4"]["modeled_events"] == 158
    mapping = source_mapping_audit(con, m3)
    measurement_parts = []
    exposure_parts = []
    measurement_details: dict[str, pd.DataFrame] = {}
    for label, obj in (("MIMIC-IV", m4), ("MIMIC-III", m3)):
        metrics, per_stay = measurement_process_rows(
            obj["raw_spo2"], obj["at_risk"]["stay_id"], label
        )
        extra = extra_measurement_rows(
            obj["raw_spo2"], obj["at_risk"]["stay_id"], label, obj["exposure"], per_stay
        )
        measurement_parts.extend([metrics, extra])
        measurement_details[label] = per_stay
        exposure_for_distribution = obj["exposure"].loc[
            obj["exposure"]["stay_id"].isin(obj["at_risk"]["stay_id"]),
            ["stay_id", "bins", "span_hours", "raw_rms_residual"],
        ]
        dist, scaled = feature_distribution_rows(
            exposure_for_distribution,
            label,
            FROZEN_EXPOSURE_PARAMETERS,
        )
        # Correct clipping fractions to use only valid exposure rows.
        valid = scaled["raw_rms_residual"].notna()
        for stat, col in (
            ("fraction_clipped_low_valid", "clipped_low"),
            ("fraction_clipped_high_valid", "clipped_high"),
        ):
            dist = pd.concat(
                [
                    dist,
                    pd.DataFrame(
                        [
                            {
                                "dataset": label,
                                "stratum": "all_valid",
                                "scale": "raw",
                                "statistic": stat,
                                "value": float(scaled.loc[valid, col].mean()),
                                "n": int(valid.sum()),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        merged_icu = scaled.merge(
            obj["cohort"][["stay_id", "first_careunit"]], on="stay_id", how="left"
        )
        for unit, group in merged_icu.loc[valid].groupby("first_careunit"):
            if len(group) < 20:
                continue
            for stat, value in (
                ("mean", group["standardized_exposure"].mean()),
                ("sd", group["standardized_exposure"].std(ddof=0)),
                ("median", group["standardized_exposure"].median()),
                (
                    "iqr",
                    group["standardized_exposure"].quantile(0.75)
                    - group["standardized_exposure"].quantile(0.25),
                ),
            ):
                dist = pd.concat(
                    [
                        dist,
                        pd.DataFrame(
                            [
                                {
                                    "dataset": label,
                                    "stratum": f"icu_type:{unit}",
                                    "scale": "mimic_standardized",
                                    "statistic": stat,
                                    "value": float(value),
                                    "n": len(group),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
        exposure_parts.append(dist)
    measurement = pd.concat(measurement_parts, ignore_index=True)
    exposure = pd.concat(exposure_parts, ignore_index=True)
    cohort = cohort_comparison(m4, m3)
    endpoint = pd.concat(
        [endpoint_rows("MIMIC-IV", m4), endpoint_rows("MIMIC-III", m3)],
        ignore_index=True,
    )
    precision, heterogeneity = precision_and_heterogeneity()
    model3 = m3["model"].dropna(subset=["spo2_variability"])
    sanity = {"status": "implementation_sanity_checks_not_discovery", "checks": []}
    for variable, expected in (
        ("spo2_level", "plausible_range_only"),
        ("sbp_level", "lower_in_events"),
        ("map_level", "lower_in_events"),
        ("baseline_lactate", "higher_in_events"),
    ):
        event = pd.to_numeric(
            model3.loc[model3.objective_shock.eq(1), variable], errors="coerce"
        )
        nonevent = pd.to_numeric(
            model3.loc[model3.objective_shock.eq(0), variable], errors="coerce"
        )
        difference = event.median() - nonevent.median()
        direction = (
            True
            if expected == "plausible_range_only"
            else (difference < 0 if expected == "lower_in_events" else difference > 0)
        )
        sanity["checks"].append(
            {
                "variable": variable,
                "event_median": event.median(),
                "nonevent_median": nonevent.median(),
                "event_minus_nonevent": difference,
                "expected": expected,
                "direction_plausible": bool(direction),
                "inferential_test": False,
            }
        )
    paired = m3["primary"]
    sanity["endpoint_temporal_alignment"] = {
        "events": len(paired),
        "all_endpoint_times_after_360": bool(paired["event_minute"].gt(360).all()),
        "all_endpoint_times_by_960": bool(paired["event_minute"].le(960).all()),
        "pairing_helper_requires_each_component_after_360": True,
        "pairing_helper_requires_each_component_by_960": True,
        "pairing_helper_requires_separation_le_360": True,
        "evidence": "pair_domain_events source guard plus exact 103-stay frozen MIMIC-III primary replay",
    }
    chart3 = con.execute(
        "SELECT * FROM chart_m3 WHERE itemid=646 AND event_minute BETWEEN 0 AND 240"
    ).df()
    store_delay = chart3["store_available_minute"] - chart3["event_minute"]
    measurement_process_comparable = bool(
        measurement.loc[
            (measurement.dataset == "MIMIC-III")
            & (measurement.metric == "raw_observations_per_stay")
            & (measurement.statistic == "median"),
            "value",
        ].iloc[0]
        <= 2
        * measurement.loc[
            (measurement.dataset == "MIMIC-IV")
            & (measurement.metric == "raw_observations_per_stay")
            & (measurement.statistic == "median"),
            "value",
        ].iloc[0]
        and abs(
            measurement.loc[
                (measurement.dataset == "MIMIC-III")
                & (measurement.metric == "consecutive_observations_identical"),
                "value",
            ].iloc[0]
            - measurement.loc[
                (measurement.dataset == "MIMIC-IV")
                & (measurement.metric == "consecutive_observations_identical"),
                "value",
            ].iloc[0]
        )
        < 0.15
    )
    m3_power = {row["true_rr"]: row["approximate_power"] for row in precision["power"]}
    summary = {
        "status": "complete",
        "environment": "LOCAL_STANDALONE_RUNTIME",
        "input_provenance": {
            "mimic3": "independent reconstruction from original MIMIC-III CSV.GZ sources",
            "mimic4": "validated read-only frozen discovery and SpO2-context DuckDB tables after raw Google Drive CSV timeout",
        },
        "frozen_results_changed": False,
        "primary_classification": "IMPLEMENTATION_OR_MAPPING_CONCERN",
        "reconstructed_counts": reconstructed,
        "lock_and_code_hashes_verified": True,
        "dominant_reason": "A material endpoint-parity defect is present: frozen MIMIC-IV oliguria surveillance uses the shared helper's hour-4 start, whereas frozen MIMIC-III explicitly starts at hour 0; the two later endpoints also use different pairing semantics. CareVue availability semantics and ICU-entry phenotype differences compound this concern.",
        "secondary_flags": {
            "exposure_definition_identical": False,
            "scaling_identical": True,
            "time_window_identical": True,
            "endpoint_logically_identical": False,
            "endpoint_observability_comparable": False,
            "cohort_comparable": False,
            "measurement_process_comparable": measurement_process_comparable,
            "sufficient_power_for_RR_1_25": m3_power[1.25] >= 0.80,
            "sufficient_power_for_RR_1_20": m3_power[1.20] >= 0.80,
            "MIMIC_IV_effect_inside_MIMIC_III_CI": precision[
                "mimic_iv_rr_inside_mimic_iii_ci"
            ],
            "formal_cross_era_heterogeneity_present": heterogeneity[
                "formal_cross_era_heterogeneity_present_at_0_05"
            ],
            "coding_error_found": True,
            "source_mapping_error_found": False,
        },
        "storetime_audit": {
            "mimic3_spo2_rows_0_240": len(chart3),
            "storetime_after_minute_240_count": int(
                chart3["store_available_minute"].gt(240).sum()
            ),
            "storetime_after_minute_240_fraction": float(
                chart3["store_available_minute"].gt(240).mean()
            ),
            "median_store_delay_minutes": float(store_delay.median()),
            "p95_store_delay_minutes": float(store_delay.quantile(0.95)),
            "guard": "Descriptive implementation audit only; frozen exposure was not recalculated with a new availability rule.",
        },
        "implementation_parity_findings": {
            "mimic4_oliguria_start_hour": 4,
            "mimic3_oliguria_start_hour": 0,
            "mimic4_at_risk_replay_n": m4["risk_set_replay_n"],
            "mimic4_later_semantics": "primary composite event time >480",
            "mimic3_later_semantics": "both components re-paired strictly after 480",
            "frozen_results_recomputed_or_changed": False,
        },
        "degradation_phase": {
            "triggered": False,
            "reason": "The dominant diagnosed limitation is endpoint/cohort observability rather than a major outcome-blind SpO2 cadence or quantization shift requiring a new degradation operator.",
        },
        "association_models_fitted_in_forensic_audit": 0,
        "frozen_mimic3_association_models_fitted_unchanged": 2,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    parity_matrix().to_csv(OUT / "parity_matrix.csv", index=False)
    mapping.to_csv(OUT / "source_mapping_audit.csv", index=False)
    measurement.to_csv(OUT / "measurement_process_comparison.csv", index=False)
    exposure.to_csv(OUT / "exposure_distribution_comparison.csv", index=False)
    cohort.to_csv(OUT / "cohort_transportability.csv", index=False)
    endpoint.to_csv(OUT / "endpoint_component_comparison.csv", index=False)
    write_json(OUT / "precision_analysis.json", precision)
    write_json(OUT / "heterogeneity_analysis.json", heterogeneity)
    write_json(OUT / "sanity_checks.json", sanity)
    figures(measurement, exposure, endpoint)
    report = f"""# Forensic audit of the MIMIC-III SpO2-variability replication

## Primary classification

**IMPLEMENTATION_OR_MAPPING_CONCERN**

The frozen numerical record is internally reproducible: an independent reconstruction recovered the exact MIMIC-III counts (5,405 preliminary; 2,368 at risk; 1,799 exposed; 103 at-risk primary events; 72 modeled primary events; 53 modeled later events) and the exact MIMIC-IV support counts (17,758 preliminary; 8,196 at risk; 5,957 exposed; 238 at-risk primary events; 158 modeled events). Lock, source, and code hashes pass, no source item-mapping error was found, and no frozen result was changed.

The dominant forensic finding is a material implementation-parity concern in the endpoint. Frozen MIMIC-IV calls the six-hour oliguria helper with its default `start_hour=4`, whereas frozen MIMIC-III explicitly calls it with `start_hour=0`; therefore MIMIC-III admits urine windows that overlap more of the exposure/landmark period. The later endpoints are also not identical: MIMIC-IV labels a primary composite whose event time is after minute 480, while MIMIC-III re-pairs the domains with both components required strictly after minute 480. These findings are documented only; the locked MIMIC-III outcomes and two fitted models were not corrected or rerun.

Independent of that defect, endpoint observability is materially non-equivalent. MIMIC-IV uses availability-aware chart/store timestamps, interval starts for infusions, and procedure starts for MCS. MIMIC-III uses CareVue charttime alone, repeated positive-rate infusion rows, and device-chart documentation. Both at-risk cohorts require observation only beyond minute 360 for an endpoint extending through minute 960, so incomplete late follow-up is a shared limitation rather than a cross-era difference. Preliminary selection still differs: MIMIC-IV requires ICU entry within 24 hours unless a shock code is present, whereas MIMIC-III has no ICU-entry-time restriction.

The exposure computation is mathematically identical after extraction and uses the exact MIMIC-IV clipping/scale. However, extraction is not byte-equivalent because MIMIC-IV requires storetime <=240 while MIMIC-III uses charttime only. In MIMIC-III, {summary["storetime_audit"]["storetime_after_minute_240_count"]:,}/{summary["storetime_audit"]["mimic3_spo2_rows_0_240"]:,} early SpO2 rows ({summary["storetime_audit"]["storetime_after_minute_240_fraction"]:.2%}) were stored after minute 240; this is documented but not used to alter or refit the frozen exposure.

MIMIC-III was independently reconstructed from its original source files. After Google Drive timed out on a raw MIMIC-IV admissions read, the MIMIC-IV reference side used the already validated read-only frozen discovery and SpO2-context databases. Those tables replayed the exact 8,196-stay at-risk set and all 238 primary endpoint IDs; no MIMIC-IV estimate was refitted.

## Cohort transportability

The harmonized modeled cohorts do not look radically different by age or sex: mean age is 72.48 years in MIMIC-IV versus 72.56 in MIMIC-III (SMD 0.005), and male prevalence is 53.6% versus 54.4% (SMD 0.016). Differences are more visible for baseline SpO2 (96.38 versus 97.12; SMD 0.245) and MAP (83.57 versus 80.80; SMD -0.196). A coarse cross-era ICU grouping shows more medical-ICU representation in MIMIC-III (47.1% versus 27.2%), while MIMIC-IV includes a 15.5% mixed medical/surgical group and broader neuro/intermediate units. Ventilation, FiO2, MIMIC-III respiratory-support status, and cross-era HF coding density/comorbidity burden were not harmonized in the frozen frames. Thus the available data do not show a substantially different CareVue HF population on age/sex, but ICU mix and unavailable context prevent declaring the cohorts fully comparable.

## Precision and cross-era compatibility

MIMIC-III log RR was {precision["log_rr"]:.4f} with HC0 SE {precision["robust_se"]:.4f}; its RR-scale CI width was {precision["rr_ci_width"]:.3f} and its multiplicative 95% factor was ×/{precision["multiplicative_95_ci_factor"]:.3f}. Approximate power was {m3_power[1.20]:.1%} for a true RR of 1.20 and {m3_power[1.25]:.1%} for 1.25. The MIMIC-IV RR 1.251 lies just outside the MIMIC-III 95% CI upper bound 1.234. The cross-era difference test gives z={heterogeneity["z"]:.3f}, p={heterogeneity["p_value"]:.4f}, Q={heterogeneity["q"]:.3f}, and I²={heterogeneity["i_squared_percent"]:.1f}%; with two studies, I² is unstable. The effects are not formally heterogeneous at alpha .05, but the divergence is suggestive and the MIMIC-III null is not precise enough to call a true negative.

## Measurement and sanity checks

The source audit confirms CareVue item 646 is labeled SpO2 with percent units and does not mix arterial SaO2, venous saturation, or FiO2. Every mapped itemid is listed with counts, units, missingness, distributions, common values, and implausibility counts. Frozen endpoint pairs all satisfy the >360, <=960, and <=360-minute separation rules. Descriptive checks of SBP, MAP, lactate, and absolute SpO2 are recorded only as pipeline sanity checks, not discoveries.

No degradation experiment was triggered: the dominant limitations are endpoint and cohort observability, not a sufficiently dominant outcome-blind measurement-process shift that would justify another one-off MIMIC-IV model. No association model was fitted in this forensic audit.

## Evidentiary conclusion

The MIMIC-III result is numerically reproducible, but it is not an exact endpoint replication. It still argues against casually assuming the MIMIC-IV association is era-invariant; however, its evidentiary weight as biological non-confirmation is limited first by the material endpoint-parity defect, and additionally by weaker CareVue observability and a different ICU-entry phenotype. The correct conclusion remains **historical non-confirmation with an implementation concern**, not rescue, not proof of no effect, and not successful external validation.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    after = protected_hashes()
    assert before == after
    summary["protected_artifact_tree_hashes_before"] = before
    summary["protected_artifact_tree_hashes_after"] = after
    summary["artifact_hashes"] = {
        str(path.relative_to(OUT)): sha256_file(path)
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name != "forensic_summary.json"
    }
    write_json(OUT / "forensic_summary.json", summary)
    con.close()
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
