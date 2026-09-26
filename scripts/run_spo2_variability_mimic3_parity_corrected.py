#!/usr/bin/env python3
"""Locked one-time MIMIC-III endpoint-parity correction analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "research" / "spo2_variability_mimic3_parity_corrected"
sys.path.insert(0, str(ROOT))

from physiograph.analysis.spo2_variability_mimic3_parity_corrected import (  # noqa: E402
    EXPOSURE_AVAILABILITY_BRANCH,
    EXPOSURE_ITEMID,
    FORENSIC_CLASSIFICATION,
    FROZEN_EXPOSURE_PARAMETERS,
    MINIMUM_MODELED_EVENTS,
    ORIGINAL_CLASSIFICATION,
    AuthorizedFitLedger,
    assert_exposure_branch_matches_lock,
    assert_lock_precedes_association,
    assert_preserved_classifications,
    compare_endpoint_ids,
    compute_locked_exposure,
    corrected_later_events,
    corrected_oliguria_events,
    corrected_primary_events,
    prelandmark_oliguria_exclusions,
    verify_protected_hash_manifest,
)
from physiograph.analysis.spo2_variability_mimic3_validation import (  # noqa: E402
    canonical_sha256,
    verify_source_manifest,
)
from physiograph.analysis.spo2_variability_validation import (  # noqa: E402
    fit_fixed_scale_model,
)
from scripts.run_spo2_variability_mimic3_forensic import (  # noqa: E402
    M3,
    assemble as assemble_original,
    create_m3_tables,
    scan,
)

PROTOCOL = OUTPUT_DIR / "protocol.json"
LOCK = OUTPUT_DIR / "analysis_lock.json"
LOCK_HASH = OUTPUT_DIR / "analysis_lock.sha256"
AVAILABILITY_DECISION = OUTPUT_DIR / "availability_semantics_decision.md"
BASELINE = OUTPUT_DIR / "artifact_hashes.json"
SUPPORT = OUTPUT_DIR / "support_counts.json"
ENDPOINT_COMPARISON = OUTPUT_DIR / "endpoint_id_comparison.json"
RESULTS = OUTPUT_DIR / "validation_results.json"
PRECISION = OUTPUT_DIR / "precision_analysis.json"
HETEROGENEITY = OUTPUT_DIR / "heterogeneity_analysis.json"
RUN_STATUS = OUTPUT_DIR / "run_status.json"
RECEIPT = OUTPUT_DIR / "association_read_receipt.json"
REPORT = OUTPUT_DIR / "FINAL_REPORT.md"
FIGURES = OUTPUT_DIR / "figures"
NOTEBOOK = ROOT / "PhysioGraph_Final_Clean.ipynb"
DB_PATH = Path("/tmp/physiograph-mimic3-parity-corrected.duckdb")
TEMP_DIR = Path("/tmp/physiograph-mimic3-parity-corrected-temp")

ORIGINAL_DIR = ROOT / "research/spo2_variability_mimic3_validation"
FORENSIC_DIR = ROOT / "research/spo2_variability_mimic3_forensic"
MIMIC4_DIR = ROOT / "research/spo2_variability_validation"
ORIGINAL_RESULT = ORIGINAL_DIR / "validation_results.json"
FORENSIC_SUMMARY = FORENSIC_DIR / "forensic_summary.json"
MIMIC4_RESULT = MIMIC4_DIR / "validation_results.json"

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

CODE_FILES = [
    "scripts/run_spo2_variability_mimic3_parity_corrected.py",
    "scripts/run_spo2_variability_mimic3_forensic.py",
    "scripts/run_spo2_variability_mimic3_validation.py",
    "scripts/shock_signal_discovery_cells.py",
    "src/physiograph/analysis/shock_signal_discovery.py",
    "src/physiograph/analysis/spo2_variability_validation.py",
    "src/physiograph/analysis/spo2_variability_mimic3_validation.py",
    "src/physiograph/analysis/spo2_variability_mimic3_parity_corrected.py",
    "tests/unit/test_spo2_variability_mimic3_parity_corrected.py",
]

PROTECTED_ROOTS = (
    "research/spo2_variability_validation",
    "research/spo2_variability_transportability",
    "research/spo2_variability_nwicu_validation",
    "research/spo2_variability_inspire_validation",
    "research/spo2_variability_mimic3_validation",
    "research/spo2_variability_mimic3_forensic",
    "docs/SPO2_VARIABILITY_TARGETED_VALIDATION_PROTOCOL.md",
    "docs/SPO2_VARIABILITY_INSPIRE_EXTERNAL_VALIDATION.md",
    "docs/SPO2_VARIABILITY_NWICU_MODIFIED_ENDPOINT.md",
    "docs/SPO2_VARIABILITY_MIMIC3_REPLICATION.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonignored_protected_files() -> list[str]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        *PROTECTED_ROOTS,
    ]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    return sorted(line for line in output.splitlines() if line)


def establish_protected_baseline() -> None:
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"One-time output directory already exists: {OUTPUT_DIR}")
    files = nonignored_protected_files()
    hashes = {relative: sha256_file(ROOT / relative) for relative in files}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    payload = {
        "manifest_role": "pre_analysis_protected_artifact_baseline",
        "algorithm": "sha256",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protected_artifact_count": len(hashes),
        "protected_artifacts": hashes,
    }
    (OUTPUT_DIR / "artifact_hashes.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"BASELINE_WRITTEN count={len(hashes)}")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def load_preserved_results() -> tuple[dict[str, Any], dict[str, Any]]:
    original = json.loads(ORIGINAL_RESULT.read_text(encoding="utf-8"))
    forensic = json.loads(FORENSIC_SUMMARY.read_text(encoding="utf-8"))
    assert_preserved_classifications(original, forensic)
    return original, forensic


def configure_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(DB_PATH))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute("SET threads=4")
    con.execute("SET memory_limit='5GB'")
    con.execute("SET enable_progress_bar=false")
    con.execute(f"SET temp_directory='{str(TEMP_DIR).replace(chr(39), chr(39) * 2)}'")
    return con


def create_corrected_spo2_source(con: duckdb.DuckDBPyConnection) -> None:
    """Materialize the single pre-locked CareVue availability branch."""

    con.execute(
        f"""
        CREATE OR REPLACE TABLE corrected_spo2_m3 AS
        SELECT
            c.stay_id,
            try_cast(x.itemid AS INTEGER) AS itemid,
            date_diff('second', c.intime, try_cast(x.charttime AS TIMESTAMP)) / 60.0
                AS event_minute,
            CASE
                WHEN try_cast(x.storetime AS TIMESTAMP) IS NULL THEN NULL
                ELSE greatest(
                    date_diff('second', c.intime, try_cast(x.charttime AS TIMESTAMP)) / 60.0,
                    date_diff('second', c.intime, try_cast(x.storetime AS TIMESTAMP)) / 60.0
                )
            END AS store_available_minute,
            try_cast(x.valuenum AS DOUBLE) AS raw_value,
            coalesce(try_cast(x.error AS INTEGER), 0) AS error_flag
        FROM {scan(M3 / 'CHARTEVENTS.csv.gz')} x
        JOIN cohort_m3 c
          ON try_cast(x.icustay_id AS BIGINT) = c.stay_id
        WHERE try_cast(x.itemid AS INTEGER) = {EXPOSURE_ITEMID}
          AND date_diff('second', c.intime, try_cast(x.charttime AS TIMESTAMP)) / 60.0
              BETWEEN 0 AND 240
        """
    )


def availability_statistics(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    row = con.execute(
        """
        SELECT
            count(*) AS candidate_rows,
            count(*) FILTER (WHERE store_available_minute IS NULL) AS storetime_missing,
            count(*) FILTER (WHERE store_available_minute > 240) AS stored_after_240,
            count(*) FILTER (
                WHERE store_available_minute <= 240
                  AND raw_value BETWEEN 50 AND 100
                  AND error_flag = 0
            ) AS valid_available_rows,
            median(store_available_minute - event_minute)
                FILTER (WHERE store_available_minute IS NOT NULL) AS median_delay,
            quantile_cont(store_available_minute - event_minute, 0.95)
                FILTER (WHERE store_available_minute IS NOT NULL) AS p95_delay
        FROM corrected_spo2_m3
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    values = dict(zip(names, row, strict=True))
    n = int(values["candidate_rows"])
    return {
        "mimic3_carevue_spo2_rows_charttime_0_240": n,
        "storetime_missing_count": int(values["storetime_missing"]),
        "storetime_missing_fraction": float(values["storetime_missing"] / n),
        "storetime_after_minute_240_count": int(values["stored_after_240"]),
        "storetime_after_minute_240_fraction": float(values["stored_after_240"] / n),
        "valid_rows_available_by_240": int(values["valid_available_rows"]),
        "median_chart_to_store_delay_minutes": float(values["median_delay"]),
        "p95_chart_to_store_delay_minutes": float(values["p95_delay"]),
    }


def protocol_payload() -> dict[str, Any]:
    original_protocol = json.loads(
        (ORIGINAL_DIR / "protocol.json").read_text(encoding="utf-8")
    )
    return {
        "status": "FROZEN_BEFORE_CORRECTED_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "analysis_character": (
            "post_hoc_to_original_mimic3_null_prelocked_before_corrected_associations"
        ),
        "dataset": "MIMIC-III_v1.4_CareVue",
        "interpretation": (
            "same_institution_historical_database_version_replication_"
            "not_independent_external_validation"
        ),
        "cohort": {
            "population": "adult_explicit_HF_first_ICU_per_hospital_admission",
            "heart_failure_phenotype": "DIAGNOSES_ICD ICD-9 prefix 428",
            "first_icu_logic": (
                "first ICU stay per hospital admission ordered by ICUSTAYS.intime,icustay_id"
            ),
            "timeline_origin": "ICUSTAYS.intime",
            "early_icu_entry_rule": "none_in_frozen_MIMIC-III_protocol",
            "minimum_observation": (
                "ICU and hospital survival/observation strictly beyond minute 360"
            ),
            "at_risk_exclusions": (
                "unchanged frozen pressure/support or hypoperfusion evidence through minute 360; "
                "pre-landmark oliguria retained as a separate hour-0-to-6 exclusion"
            ),
        },
        "exposure": {
            "source": "CHARTEVENTS CareVue itemid 646 only",
            "availability_branch": EXPOSURE_AVAILABILITY_BRANCH,
            "eligibility": (
                "charttime ICU minutes 0-240 inclusive and nonmissing STORETIME-derived "
                "availability minute <=240"
            ),
            "value_bounds": [50, 100],
            "error_rule": "coalesced error flag equals 0",
            "hourly_binning": "floor(ICU minute/60), within-hour median",
            "minimum_bins": 3,
            "minimum_span_hours": 2,
            "detrending": "within-stay ordinary least squares with intercept",
            "feature": "sqrt(mean(squared OLS residuals))",
            "clipping_and_scaling": FROZEN_EXPOSURE_PARAMETERS,
            "dataset_specific_recentering_or_rescaling": False,
        },
        "endpoint": {
            "pressure_support_components": original_protocol["endpoint"][
                "pressure_support_components"
            ],
            "hypoperfusion_components": original_protocol["endpoint"][
                "hypoperfusion_components"
            ],
            "maximum_domain_separation_minutes": 360,
            "corrected_oliguria": {
                "start_hour": 4,
                "end_hour": 16,
                "window_hours": 6,
                "minimum_documented_hours": 4,
                "threshold": "mean urine output <30 mL/hour",
                "timestamp": "six-hour rolling-window end or maximum availability, whichever later",
                "all_other_parameters": "unchanged shared frozen helper",
            },
            "primary": (
                "first pressure/support plus hypoperfusion pairing with both components >360 "
                "and <=960 and separation <=360 minutes"
            ),
            "later": (
                "the already-constructed corrected primary composite has completion time >480"
            ),
            "pre_landmark_exclusions": "unchanged frozen definitions",
            "endpoint_observability_fully_equivalent": False,
            "observability_note": (
                "CareVue records source-era-specific support and availability documentation; "
                "logical construction is corrected but source capture cannot be identical across eras"
            ),
        },
        "model": {
            "adjustment_covariates": BASE_COVARIATES,
            "missing_data": (
                "unchanged prepare_adjustment_covariates behavior: median numeric imputation, "
                "missingness indicators, deterministic rank-deficiency removal"
            ),
            "family": "Poisson",
            "link": "log",
            "covariance": "HC0_due_single_hospital",
            "effect_scale": "RR per frozen MIMIC-IV SD",
        },
        "support": {
            "minimum_modeled_primary_events": MINIMUM_MODELED_EVENTS,
            "minimum_modeled_later_events": MINIMUM_MODELED_EVENTS,
            "minimum_exposure_coverage": 0.70,
            "minimum_contributing_sites": 20,
            "retained_waivers": {
                "minimum_exposure_coverage": True,
                "minimum_contributing_sites": True,
            },
            "new_waivers_created": False,
        },
        "authorized_association_fits": {
            "maximum": 2,
            "outcomes": [
                "objective_shock_corrected",
                "objective_shock_corrected_later",
            ],
            "later_only_if_modeled_events_at_least": MINIMUM_MODELED_EVENTS,
        },
        "confirmation": {
            "primary_rr_gt_1": True,
            "primary_ci_excludes_1_positive_direction": True,
            "primary_p_lt": 0.05,
            "direction_consistent_with_mimic4": True,
        },
        "prohibited": [
            "alternate_exposure",
            "alternate_availability_branch",
            "alternate_oliguria",
            "alternate_endpoint",
            "renal_free_or_without_oliguria_endpoint",
            "unadjusted_or_alternate_covariate_model",
            "threshold_or_time_window_search",
            "subgroup_analysis",
        ],
        "association_inspected": False,
    }


def write_availability_decision(stats: dict[str, Any]) -> None:
    decision = f"""# CareVue STORETIME availability decision

**Locked branch: `{EXPOSURE_AVAILABILITY_BRANCH}`**

This decision was made from field semantics and measurement timing before any corrected SpO2–outcome association was accessed. MIMIC-III documents `CHARTEVENTS.CHARTTIME` as the time the observation was made and `STORETIME` as the time it was manually input or validated. MIMIC-IV gives the same operational definitions for its `chartevents` fields. Those definitions support using CareVue `STORETIME` as the closest available source-era analogue of the frozen MIMIC-IV availability field.

Sources:

- MIMIC-III `CHARTEVENTS`: https://mimic.mit.edu/docs/iii/tables/chartevents.html
- MIMIC-IV `chartevents`: https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html

## Locked implementation

- Exposure source: CareVue `CHARTEVENTS.ITEMID = {EXPOSURE_ITEMID}` only.
- Observation time: ICU-relative `CHARTTIME`, required to be within minutes 0–240 inclusive.
- Availability time: the later of ICU-relative `CHARTTIME` and `STORETIME`.
- Eligibility: availability time must be nonmissing and no later than minute 240.
- No alternative availability branch will be computed or compared.

## Source audit

- Candidate item-646 rows with charttime in minutes 0–240: **{stats['mimic3_carevue_spo2_rows_charttime_0_240']:,}**.
- Missing STORETIME: **{stats['storetime_missing_count']:,}** ({stats['storetime_missing_fraction']:.2%}).
- Stored after minute 240: **{stats['storetime_after_minute_240_count']:,}** ({stats['storetime_after_minute_240_fraction']:.2%}).
- Valid item-646 rows available by minute 240: **{stats['valid_rows_available_by_240']:,}**.
- Median chart-to-store delay: **{stats['median_chart_to_store_delay_minutes']:.1f} minutes**.
- 95th-percentile chart-to-store delay: **{stats['p95_chart_to_store_delay_minutes']:.1f} minutes**.

## Qualification

The fields are sufficiently comparable for the specified landmark-availability restriction, but they are not proof of identical real-world latency or capture workflows across CareVue and MetaVision. The corrected analysis therefore treats exposure availability parity as supportable and applied while retaining cross-era measurement-process limitations. This choice is not based on an association estimate.
"""
    AVAILABILITY_DECISION.write_text(decision, encoding="utf-8")


def prepare_lock() -> None:
    if not BASELINE.exists():
        raise FileNotFoundError("Phase-0 protected artifact baseline is missing")
    if any(path.exists() for path in (PROTOCOL, LOCK, LOCK_HASH, RESULTS, RECEIPT)):
        raise FileExistsError("Corrected lock/association artifact already exists")
    verify_protected_hash_manifest(ROOT, BASELINE)
    original, forensic = load_preserved_results()
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = configure_connection()
    try:
        create_m3_tables(con)
        create_corrected_spo2_source(con)
        stats = availability_statistics(con)
    finally:
        con.close()
    expected_forensic = forensic["storetime_audit"]
    if stats["mimic3_carevue_spo2_rows_charttime_0_240"] != int(
        expected_forensic["mimic3_spo2_rows_0_240"]
    ):
        raise RuntimeError("STORETIME source-audit candidate count changed")
    if stats["storetime_after_minute_240_count"] != int(
        expected_forensic["storetime_after_minute_240_count"]
    ):
        raise RuntimeError("STORETIME source-audit late-row count changed")
    write_availability_decision(stats)
    write_json(PROTOCOL, protocol_payload())
    source_checks = verify_source_manifest(M3)
    code_hashes = {relative: sha256_file(ROOT / relative) for relative in CODE_FILES}
    specification = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "availability_semantics_decision_sha256": sha256_file(
            AVAILABILITY_DECISION
        ),
        "protected_artifact_baseline_sha256": sha256_file(BASELINE),
        "source_manifest_sha256": sha256_file(M3 / "SHA256SUMS.txt"),
        "git_revision": git_revision(),
        "code_hashes": code_hashes,
        "availability_branch": EXPOSURE_AVAILABILITY_BRANCH,
        "availability_statistics": stats,
        "corrected_oliguria_start_hour": 4,
        "corrected_primary_window": {
            "lower_exclusive": 360,
            "upper_inclusive": 960,
            "maximum_domain_separation_minutes": 360,
        },
        "corrected_later_rule": "corrected_primary_composite_completion_minute_gt_480",
        "exposure_itemid": EXPOSURE_ITEMID,
        "exposure_parameters": FROZEN_EXPOSURE_PARAMETERS,
        "model_covariates": BASE_COVARIATES,
        "maximum_authorized_association_fits": 2,
        "minimum_modeled_primary_events": MINIMUM_MODELED_EVENTS,
        "minimum_modeled_later_events": MINIMUM_MODELED_EVENTS,
        "retained_coverage_waiver": True,
        "retained_site_waiver": True,
        "new_waiver_created": False,
        "original_classification": original["classification"],
        "forensic_classification": forensic["primary_classification"],
    }
    payload = {
        "status": "LOCKED_BEFORE_CORRECTED_ENDPOINT_SUPPORT_OR_ASSOCIATION",
        "lock_utc": utc_now(),
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "source_checks": source_checks,
        "first_corrected_association_read_utc": None,
        "corrected_association_inspected": False,
    }
    write_json(LOCK, payload)
    LOCK_HASH.write_text(sha256_file(LOCK) + "\n", encoding="utf-8")
    verify_lock(verify_sources=False)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "availability_branch": EXPOSURE_AVAILABILITY_BRANCH,
                "analysis_lock_sha256": sha256_file(LOCK),
            },
            indent=2,
        )
    )


def verify_lock(*, verify_sources: bool) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if LOCK_HASH.read_text(encoding="utf-8").strip() != sha256_file(LOCK):
        raise RuntimeError("Corrected whole-file lock hash mismatch")
    specification = lock["specification"]
    if canonical_sha256(specification) != lock["specification_sha256"]:
        raise RuntimeError("Corrected specification hash mismatch")
    for path, expected in (
        (PROTOCOL, specification["protocol_sha256"]),
        (
            AVAILABILITY_DECISION,
            specification["availability_semantics_decision_sha256"],
        ),
        (BASELINE, specification["protected_artifact_baseline_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"Locked artifact changed: {path.name}")
    for relative, expected in specification["code_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Locked corrected code changed: {relative}")
    assert_exposure_branch_matches_lock(specification["availability_branch"])
    original, forensic = load_preserved_results()
    if specification["original_classification"] != original["classification"]:
        raise RuntimeError("Original MIMIC-III result was overwritten")
    if specification["forensic_classification"] != forensic["primary_classification"]:
        raise RuntimeError("Forensic classification was overwritten")
    verify_protected_hash_manifest(ROOT, BASELINE)
    if verify_sources:
        verify_source_manifest(M3)
    return lock


def count_components(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def assemble_corrected(
    con: duckdb.DuckDBPyConnection,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build corrected endpoints/exposure without fitting an association."""

    original = assemble_original(con, "MIMIC-III")
    expected = {
        "preliminary": 5405,
        "at_risk": 2368,
        "valid_exposure": 1799,
        "primary_events": 103,
        "modeled_events": 72,
        "later_modeled_events": 53,
    }
    observed = {
        "preliminary": len(original["cohort"]),
        "at_risk": len(original["at_risk"]),
        "valid_exposure": int(original["model"]["spo2_variability"].notna().sum()),
        "primary_events": len(original["primary"]),
        "modeled_events": int(
            original["model"]
            .loc[original["model"]["spo2_variability"].notna(), "objective_shock"]
            .sum()
        ),
        "later_modeled_events": int(
            original["model"]
            .loc[
                original["model"]["spo2_variability"].notna(),
                "objective_shock_later",
            ]
            .sum()
        ),
    }
    if observed != expected:
        raise RuntimeError(f"Original frozen reconstruction changed: {observed}")

    lab_perfusion = original["perfusion"].loc[
        original["perfusion"]["component"].ne("oliguria")
    ].copy()
    corrected_oliguria = corrected_oliguria_events(original["urine"])
    corrected_perfusion = pd.concat(
        [lab_perfusion, corrected_oliguria], ignore_index=True
    )

    labs = original["labs"]
    direct_pre = labs.loc[
        labs["event_minute"].le(360)
        & (
            (labs["marker"].eq("lactate") & labs["value"].ge(2))
            | (labs["marker"].eq("alt") & labs["value"].gt(200))
            | (labs["marker"].eq("ph") & labs["value"].lt(7.2))
        ),
        "stay_id",
    ]
    pre_pressure = set(
        original["pressure"].loc[
            original["pressure"]["event_minute"].le(360), "stay_id"
        ]
    )
    pre_urine = prelandmark_oliguria_exclusions(original["urine"])
    pre_perfusion = set(direct_pre) | set(
        corrected_perfusion.loc[
            corrected_perfusion["event_minute"].le(360), "stay_id"
        ]
    )
    pre_perfusion |= set(pre_urine["stay_id"])
    at_risk = original["cohort"].loc[
        ~original["cohort"]["stay_id"].isin(pre_pressure | pre_perfusion)
    ].copy()
    if set(at_risk["stay_id"]) != set(original["at_risk"]["stay_id"]):
        raise RuntimeError("Corrected implementation changed frozen at-risk exclusions")

    corrected_primary_all = corrected_primary_events(
        original["pressure"], corrected_perfusion
    )
    corrected_primary = corrected_primary_all.loc[
        corrected_primary_all["stay_id"].isin(at_risk["stay_id"])
    ].copy()
    corrected_later = corrected_later_events(corrected_primary)

    exposure_rows = con.execute("SELECT * FROM corrected_spo2_m3").df()
    corrected_exposure = compute_locked_exposure(exposure_rows)
    drop_columns = [
        column
        for column in (
            "objective_shock",
            "objective_shock_later",
            "spo2_variability",
            "spo2_bins",
            "spo2_span_hours",
            "span_hours",
        )
        if column in original["model"].columns
    ]
    model = original["model"].drop(columns=drop_columns).copy()
    exposure_columns = corrected_exposure[
        ["stay_id", "bins", "span_hours", "raw_rms_residual"]
    ].rename(
        columns={
            "bins": "spo2_bins",
            "span_hours": "spo2_span_hours",
            "raw_rms_residual": "spo2_variability",
        }
    )
    model = model.merge(exposure_columns, on="stay_id", how="left")
    model["objective_shock_corrected"] = model["stay_id"].isin(
        corrected_primary["stay_id"]
    ).astype(int)
    model["objective_shock_corrected_later"] = model["stay_id"].isin(
        corrected_later["stay_id"]
    ).astype(int)

    valid = model["spo2_variability"].notna()
    support = {
        "preliminary_cohort": len(original["cohort"]),
        "at_risk_cohort": len(at_risk),
        "corrected_primary_events_at_risk": len(corrected_primary),
        "valid_frozen_exposure_n": int(valid.sum()),
        "exposure_coverage": float(valid.mean()),
        "modeled_primary_events": int(
            model.loc[valid, "objective_shock_corrected"].sum()
        ),
        "corrected_later_window_events_at_risk": len(corrected_later),
        "modeled_later_window_events": int(
            model.loc[valid, "objective_shock_corrected_later"].sum()
        ),
        "released_sites": 1,
        "retained_support_gates": {
            "minimum_modeled_primary_events": MINIMUM_MODELED_EVENTS,
            "minimum_modeled_later_events": MINIMUM_MODELED_EVENTS,
            "minimum_exposure_coverage": 0.70,
            "minimum_contributing_sites": 20,
        },
        "retained_waivers": {
            "minimum_exposure_coverage": True,
            "minimum_contributing_sites": True,
        },
        "new_waiver_created": False,
        "primary_event_gate_passed": bool(
            model.loc[valid, "objective_shock_corrected"].sum()
            >= MINIMUM_MODELED_EVENTS
        ),
        "later_event_gate_passed": bool(
            model.loc[valid, "objective_shock_corrected_later"].sum()
            >= MINIMUM_MODELED_EVENTS
        ),
        "corrected_oliguria_events_all_stays": len(corrected_oliguria),
        "corrected_component_counts": {
            "paired_pressure_support": count_components(
                corrected_primary, "pressure_component"
            ),
            "paired_hypoperfusion": count_components(
                corrected_primary, "perfusion_component"
            ),
        },
    }
    comparison = {
        "primary": compare_endpoint_ids(original["primary"], corrected_primary),
        "later": compare_endpoint_ids(original["later"], corrected_later),
        "guard": (
            "Endpoint-membership comparison only; no SpO2 association was estimated in this phase."
        ),
    }
    return {
        "original": original,
        "model": model,
        "at_risk": at_risk,
        "corrected_oliguria": corrected_oliguria,
        "corrected_perfusion": corrected_perfusion,
        "corrected_primary": corrected_primary,
        "corrected_later": corrected_later,
        "corrected_exposure": corrected_exposure,
    }, {"support": support, "comparison": comparison}


def write_endpoint_figure(comparison: dict[str, Any]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = comparison["primary"]
    labels = ["Original only", "Shared", "Corrected only"]
    values = [
        primary["original_only"],
        primary["shared"],
        primary["corrected_only"],
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(labels, values, color=["#777777", "#2A6F97", "#D97706"])
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("At-risk stays with primary endpoint")
    ax.set_title("Original versus parity-corrected MIMIC-III endpoint IDs")
    ax.text(
        0.99,
        0.96,
        f"Jaccard = {primary['jaccard_overlap']:.3f}\n"
        f"Changed shared times = {primary['shared_event_time_assignments_changed']}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "endpoint_overlap.png", dpi=180)
    fig.savefig(FIGURES / "endpoint_overlap.svg")
    plt.close(fig)


def precision_analysis(result: dict[str, Any]) -> dict[str, Any]:
    se = (
        math.log(result["ci_high"]) - math.log(result["ci_low"])
    ) / (2 * 1.96)
    z_alpha = NormalDist().inv_cdf(0.975)
    powers = []
    for rr in (1.10, 1.15, 1.20, 1.25, 1.30):
        ncp = math.log(rr) / se
        power = norm.sf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
        powers.append({"true_rr": rr, "approximate_power": float(power)})
    required = []
    for rr in (1.20, 1.25):
        for target in (0.80, 0.90):
            events = int(
                math.ceil(
                    result["events"]
                    * (
                        (z_alpha + NormalDist().inv_cdf(target))
                        * se
                        / math.log(rr)
                    )
                    ** 2
                )
            )
            required.append(
                {"true_rr": rr, "target_power": target, "required_events": events}
            )
    return {
        "log_rr": float(result["log_rr"]),
        "robust_se": float(se),
        "rr_95_ci": [float(result["ci_low"]), float(result["ci_high"])],
        "rr_ci_width": float(result["ci_high"] - result["ci_low"]),
        "log_ci_width": float(
            math.log(result["ci_high"]) - math.log(result["ci_low"])
        ),
        "multiplicative_95_ci_factor": float(math.exp(1.96 * se)),
        "modeled_events": int(result["events"]),
        "power": powers,
        "required_events": required,
        "method": (
            "two-sided normal approximation using corrected observed HC0 robust SE; "
            "inverse-event information scaling"
        ),
        "guard": (
            "Passing the retained 50-event credibility gate does not establish adequate power."
        ),
    }


def heterogeneity_analysis(
    corrected: dict[str, Any], original: dict[str, Any], mimic4: dict[str, Any]
) -> dict[str, Any]:
    beta_corrected = float(corrected["log_rr"])
    beta_original = float(original["log_rr"])
    beta_mimic4 = float(mimic4["log_rr"])
    se_corrected = (
        math.log(corrected["ci_high"]) - math.log(corrected["ci_low"])
    ) / (2 * 1.96)
    se_mimic4 = (math.log(mimic4["ci_high"]) - math.log(mimic4["ci_low"])) / (
        2 * 1.96
    )
    difference = beta_corrected - beta_mimic4
    se_difference = math.sqrt(se_corrected**2 + se_mimic4**2)
    z = difference / se_difference
    q_stat = z**2
    p_value = float(chi2.sf(q_stat, 1))
    i2 = max(0.0, (q_stat - 1) / q_stat) * 100 if q_stat > 0 else 0.0
    return {
        "mimic4_primary": {
            "rr": float(mimic4["rr_per_mimic_sd"]),
            "log_rr": beta_mimic4,
            "robust_se": se_mimic4,
        },
        "mimic3_parity_corrected_primary": {
            "rr": float(corrected["rr_per_mimic_sd"]),
            "log_rr": beta_corrected,
            "robust_se": se_corrected,
        },
        "difference_in_log_rr_corrected_minus_mimic4": difference,
        "se_of_difference_assuming_independent_estimates": se_difference,
        "heterogeneity_z": z,
        "heterogeneity_p_value": p_value,
        "q": q_stat,
        "df": 1,
        "i_squared_percent": i2,
        "two_study_instability_warning": (
            "I-squared from only two database-era estimates is unstable and must not be "
            "used as proof of homogeneity or heterogeneity. No pooled estimate is used "
            "to claim replication."
        ),
        "original_vs_corrected_mimic3": {
            "original_log_rr": beta_original,
            "corrected_log_rr": beta_corrected,
            "corrected_minus_original_log_rr": beta_corrected - beta_original,
            "comparison": (
                "descriptive only; estimates arise from overlapping/dependent MIMIC-III "
                "risk sets, so no independence-based p-value is computed"
            ),
        },
    }


def classify_corrected(primary: dict[str, Any]) -> str:
    rr = float(primary["rr_per_mimic_sd"])
    if (
        rr > 1
        and float(primary["ci_low"]) > 1
        and float(primary["p_value"]) < 0.05
    ):
        return "MIMIC3_PARITY_CORRECTED_CONFIRMS_SIGNAL"
    if rr > 1:
        return "MIMIC3_PARITY_CORRECTED_DIRECTIONALLY_SUPPORTIVE_NOT_CONFIRMED"
    if rr < 1:
        return "MIMIC3_PARITY_CORRECTED_INVERSE"
    return "MIMIC3_PARITY_CORRECTED_NULL"


def flags_payload(
    support: dict[str, Any], fitted: list[str], *, outcome_peek: bool
) -> dict[str, bool]:
    return {
        "oliguria_parity_corrected": True,
        "later_endpoint_parity_corrected": True,
        "exposure_mathematics_identical": True,
        "exposure_availability_parity_applied": True,
        "exposure_availability_parity_supportable": True,
        "endpoint_logically_equivalent_after_correction": True,
        "endpoint_observability_fully_equivalent": False,
        "corrected_primary_event_gate_passed": bool(
            support["primary_event_gate_passed"]
        ),
        "corrected_later_event_gate_passed": bool(
            support["later_event_gate_passed"]
        ),
        "corrected_primary_association_fitted": (
            "objective_shock_corrected" in fitted
        ),
        "corrected_later_association_fitted": (
            "objective_shock_corrected_later" in fitted
        ),
        "original_replication_preserved": True,
        "forensic_classification_preserved": True,
        "outcome_peek_before_lock": outcome_peek,
        "unauthorized_models_fitted": False,
    }


def write_forest_figure(
    original: dict[str, Any],
    mimic4: dict[str, Any],
    corrected: dict[str, Any] | None,
) -> None:
    rows: list[tuple[str, dict[str, Any], str]] = [
        ("MIMIC-IV frozen primary", mimic4["primary"], "#1B4965"),
        ("MIMIC-IV frozen later", mimic4["lead"], "#1B4965"),
        ("MIMIC-III original primary", original["primary"], "#777777"),
        ("MIMIC-III original later", original["later_window"], "#777777"),
    ]
    if corrected is not None:
        rows.append(
            ("MIMIC-III corrected primary", corrected["primary"], "#D97706")
        )
    if corrected is not None and corrected.get("later_window") is not None:
        rows.append(
            ("MIMIC-III corrected later", corrected["later_window"], "#D97706")
        )
    labels = [row[0] for row in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for index, (_, result, color) in enumerate(rows):
        yi = y[index]
        rr = float(result["rr_per_mimic_sd"])
        ax.errorbar(
            rr,
            yi,
            xerr=[[rr - float(result["ci_low"])], [float(result["ci_high"]) - rr]],
            fmt="o",
            color=color,
            capsize=3,
        )
    ax.axvline(1, color="black", lw=0.9, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Adjusted RR per frozen MIMIC-IV SD (95% CI)")
    ax.set_title("Frozen and parity-corrected SpO₂-variability estimates")
    if corrected is None:
        ax.text(
            0.99,
            0.06,
            "Corrected association not fitted: primary event gate failed",
            transform=ax.transAxes,
            ha="right",
            color="#A61B1B",
            fontsize=9,
        )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "mimic4_original_mimic3_corrected_forest.png", dpi=180)
    fig.savefig(FIGURES / "mimic4_original_mimic3_corrected_forest.svg")
    plt.close(fig)


def result_line(label: str, result: dict[str, Any] | None) -> str:
    if result is None:
        return f"- {label}: not fitted because its retained 50-event gate did not pass."
    return (
        f"- {label}: RR **{result['rr_per_mimic_sd']:.3f}** "
        f"(95% CI **{result['ci_low']:.3f}–{result['ci_high']:.3f}**), "
        f"p={result['p_value']:.4g}; n={result['n']:,}, events={result['events']}."
    )


def write_final_report(
    *,
    classification: str,
    support: dict[str, Any],
    comparison: dict[str, Any],
    results: dict[str, Any] | None,
    precision: dict[str, Any] | None,
    heterogeneity: dict[str, Any] | None,
    flags: dict[str, bool],
) -> None:
    primary = results["primary"] if results else None
    later = results.get("later_window") if results else None
    primary_comparison = comparison["primary"]
    effect_section = (
        f"""## Locked effect estimates

- Original MIMIC-III primary: RR **0.969** (95% CI **0.761–1.234**), p=0.801.
- Original MIMIC-III later: RR **0.877** (95% CI **0.647–1.190**), p=0.400.
{result_line('Parity-corrected MIMIC-III primary', primary)}
{result_line('Parity-corrected MIMIC-III later', later)}
- MIMIC-IV frozen primary: RR **1.251** (95% CI **1.104–1.418**), p=0.00045.
- MIMIC-IV frozen later: RR **1.248** (95% CI **1.095–1.422**), p=0.00090.
"""
        if results
        else """## Association stop

The corrected primary modeled-event count was below 50. Per the pre-locked gate, no corrected association was fitted or inspected and no RR was estimated.
"""
    )
    precision_section = ""
    if precision and heterogeneity:
        power_text = ", ".join(
            f"RR {row['true_rr']:.2f}: {row['approximate_power']:.1%}"
            for row in precision["power"]
        )
        required_text = ", ".join(
            f"{row['target_power']:.0%} at RR {row['true_rr']:.2f}: {row['required_events']} events"
            for row in precision["required_events"]
        )
        precision_section = f"""## Formal comparison and precision

The corrected primary log RR was **{precision['log_rr']:.4f}** with HC0 robust SE **{precision['robust_se']:.4f}**. Relative to MIMIC-IV, corrected-minus-MIMIC-IV log RR was **{heterogeneity['difference_in_log_rr_corrected_minus_mimic4']:.4f}** (SE **{heterogeneity['se_of_difference_assuming_independent_estimates']:.4f}**), heterogeneity z=**{heterogeneity['heterogeneity_z']:.3f}**, p=**{heterogeneity['heterogeneity_p_value']:.4g}**, Q=**{heterogeneity['q']:.3f}**, and I²=**{heterogeneity['i_squared_percent']:.1f}%**. I² based on only two database-era estimates is unstable; no pooled estimate is used to claim replication.

The original-versus-corrected MIMIC-III log-RR change was **{heterogeneity['original_vs_corrected_mimic3']['corrected_minus_original_log_rr']:.4f}**. This is descriptive only because the estimates use overlapping, dependent stays.

Approximate two-sided power from the corrected model's observed HC0 SE: {power_text}. Approximate event requirements: {required_text}. Passing the 50-event gate is not evidence of adequate power.
"""
    flag_text = "\n".join(
        f"- `{name}`: `{str(value).lower()}`" for name, value in flags.items()
    )
    report = f"""# One-time locked MIMIC-III endpoint-parity correction

**{classification}**

## What was corrected

Only the two endpoint defects established by the zero-association forensic audit were corrected: MIMIC-III oliguria surveillance now begins at hour 4, and the later endpoint is the corrected primary composite whose completion time is after minute 480. The SpO₂ exposure uses CareVue item 646 only and the one pre-locked `{EXPOSURE_AVAILABILITY_BRANCH}` branch. CareVue STORETIME was judged sufficiently comparable to impose availability by minute 240 based on the documented MIMIC-III and MIMIC-IV field semantics; no second branch was run.

## Feasibility and endpoint membership

- Preliminary cohort: **{support['preliminary_cohort']:,}**.
- At-risk cohort: **{support['at_risk_cohort']:,}**.
- Corrected primary events among all at-risk stays: **{support['corrected_primary_events_at_risk']}**.
- Valid frozen exposures: **{support['valid_frozen_exposure_n']:,}** ({support['exposure_coverage']:.1%}).
- Modeled corrected primary events: **{support['modeled_primary_events']}**.
- Corrected later events among all at-risk stays: **{support['corrected_later_window_events_at_risk']}**.
- Modeled corrected later events: **{support['modeled_later_window_events']}**.
- Primary endpoint IDs—original only: **{primary_comparison['original_only']}**; corrected only: **{primary_comparison['corrected_only']}**; shared: **{primary_comparison['shared']}**; Jaccard: **{primary_comparison['jaccard_overlap']:.3f}**.
- Shared endpoint completion times changed: **{primary_comparison['shared_event_time_assignments_changed']}**.
- Released sites: **1**. The prior coverage and site waivers are retained and explicitly noted; no new waiver was created.

{effect_section}

{precision_section}

## Interpretation

The parity-corrected analysis is post hoc with respect to the previously observed MIMIC-III non-confirmation. It was undertaken because a zero-association forensic audit identified objective implementation non-equivalence in the replication endpoint. It does not erase or replace the original locked replication result.

The historical original classification remains **{ORIGINAL_CLASSIFICATION}**. The forensic classification remains **{FORENSIC_CLASSIFICATION}**. MIMIC-III and MIMIC-IV come from the same institution and may overlap, so this is a historical database-version replication—not independent external validation. Logical endpoint parity was restored as specified, but source-era endpoint observability is not fully identical. No alternative exposure, endpoint, availability branch, subgroup, threshold, covariate set, or time window was fitted.

Execution was **LOCAL using a standalone runtime**, not Google Colab.

## Required flags

{flag_text}
"""
    REPORT.write_text(report, encoding="utf-8")


def update_notebook_section(classification: str) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    marker = "mimic3_parity_corrected"
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("physiograph_section") != marker
    ]
    markdown = f"""## One-time MIMIC-III endpoint-parity correction

**LOCAL standalone-runtime execution; not Colab.** This analysis is post hoc to the original MIMIC-III null, but its corrected association specification was locked and hashed before either corrected model was accessed.

1. **Original locked MIMIC-III result:** `MIMIC3_HISTORICAL_REPLICATION_DID_NOT_CONFIRM_POST_HOC_SIGNAL` (primary RR 0.969, 95% CI 0.761–1.234).
2. **Forensic finding:** `IMPLEMENTATION_OR_MAPPING_CONCERN`—oliguria began at hour 0 instead of 4, and the later endpoint was incorrectly re-paired after minute 480.
3. **Parity-corrected analysis:** hour-4 oliguria; later status based on corrected-primary completion after minute 480; CareVue item 646 restricted by charttime and STORETIME availability through minute 240.
4. **Final corrected classification:** `{classification}`.

The cell below only displays frozen local artifacts. It does not fit or refit any association.
"""
    code = """from pathlib import Path
import json
from IPython.display import Markdown, display

root = Path.cwd()
out = root / "research" / "spo2_variability_mimic3_parity_corrected"
support = json.loads((out / "support_counts.json").read_text())
status = json.loads((out / "run_status.json").read_text())
lines = [
    "### Frozen local parity-correction result",
    f"- Classification: **{status['classification']}**",
    f"- Preliminary / at risk / exposed: **{support['preliminary_cohort']:,} / {support['at_risk_cohort']:,} / {support['valid_frozen_exposure_n']:,}**",
    f"- Corrected primary events (at risk / modeled): **{support['corrected_primary_events_at_risk']} / {support['modeled_primary_events']}**",
    f"- Corrected later events (at risk / modeled): **{support['corrected_later_window_events_at_risk']} / {support['modeled_later_window_events']}**",
    f"- Association models fitted: **{status['association_models_fitted']}**",
]
result_path = out / "validation_results.json"
if result_path.exists():
    result = json.loads(result_path.read_text())
    primary = result["primary"]
    lines.append(f"- Corrected primary RR: **{primary['rr_per_mimic_sd']:.3f} ({primary['ci_low']:.3f}–{primary['ci_high']:.3f})**, p={primary['p_value']:.4g}")
    if result.get("later_window"):
        later = result["later_window"]
        lines.append(f"- Corrected later RR: **{later['rr_per_mimic_sd']:.3f} ({later['ci_low']:.3f}–{later['ci_high']:.3f})**, p={later['p_value']:.4g}")
lines.append("- Execution label: **LOCAL standalone runtime; not Colab**")
display(Markdown("\\n".join(lines)))
"""
    metadata = {"physiograph_section": marker}
    notebook["cells"].extend(
        [
            {
                "cell_type": "markdown",
                "metadata": metadata,
                "source": [line + "\n" for line in markdown.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": metadata,
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ]
    )
    NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def run_locked_analysis() -> None:
    verify_lock(verify_sources=True)
    if not DB_PATH.exists():
        raise FileNotFoundError("Pre-lock corrected source database is missing")
    phase4_outputs = (SUPPORT, ENDPOINT_COMPARISON, RESULTS, RUN_STATUS, RECEIPT)
    if any(path.exists() for path in phase4_outputs):
        raise FileExistsError("One-time corrected run artifacts already exist")
    original_result, _ = load_preserved_results()
    mimic4_result = json.loads(MIMIC4_RESULT.read_text(encoding="utf-8"))["mimic"]
    con = configure_connection()
    try:
        inputs, phase4 = assemble_corrected(con)
    finally:
        con.close()
    support = phase4["support"]
    comparison = phase4["comparison"]
    write_json(SUPPORT, support)
    write_json(ENDPOINT_COMPARISON, comparison)
    write_endpoint_figure(comparison)

    if not support["primary_event_gate_passed"]:
        classification = (
            "MIMIC3_PARITY_CORRECTED_ANALYSIS_NOT_RUN_INSUFFICIENT_EVENTS"
        )
        fitted: list[str] = []
        flags = flags_payload(support, fitted, outcome_peek=False)
        status = {
            "status": "completed_without_association",
            "classification": classification,
            "environment": "LOCAL_STANDALONE_RUNTIME_NOT_COLAB",
            "association_models_fitted": 0,
            "corrected_association_inspected": False,
            "reason": "corrected_modeled_primary_events_below_retained_50_event_gate",
            "flags": flags,
            "protected_artifacts_verified_unchanged": True,
        }
        write_json(RUN_STATUS, status)
        write_forest_figure(original_result, mimic4_result, None)
        write_final_report(
            classification=classification,
            support=support,
            comparison=comparison,
            results=None,
            precision=None,
            heterogeneity=None,
            flags=flags,
        )
        update_notebook_section(classification)
        verify_lock(verify_sources=False)
        print(json.dumps(status, indent=2))
        return

    ledger = AuthorizedFitLedger()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    association_read_utc = utc_now()
    assert_lock_precedes_association(lock["lock_utc"], association_read_utc)
    write_json(
        RECEIPT,
        {
            "first_corrected_association_read_utc": association_read_utc,
            "analysis_lock_sha256": sha256_file(LOCK),
            "outcome_peek_before_lock": False,
            "authorized_model_limit": 2,
        },
    )
    ledger.register("objective_shock_corrected")
    primary, _ = fit_fixed_scale_model(
        inputs["model"],
        feature="spo2_variability",
        outcome="objective_shock_corrected",
        adjustment_covariates=BASE_COVARIATES,
        exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
    )
    later: dict[str, Any] | None = None
    if support["later_event_gate_passed"]:
        ledger.register("objective_shock_corrected_later")
        later, _ = fit_fixed_scale_model(
            inputs["model"],
            feature="spo2_variability",
            outcome="objective_shock_corrected_later",
            adjustment_covariates=BASE_COVARIATES,
            exposure_parameters=FROZEN_EXPOSURE_PARAMETERS,
        )
    classification = classify_corrected(primary)
    flags = flags_payload(support, ledger.fitted, outcome_peek=False)
    result = {
        "classification": classification,
        "analysis_character": (
            "post_hoc_to_original_mimic3_null_prelocked_before_corrected_associations"
        ),
        "dataset": "MIMIC-III_v1.4_CareVue",
        "interpretation": (
            "same_institution_historical_database_version_replication_"
            "not_independent_external_validation"
        ),
        "availability_branch": EXPOSURE_AVAILABILITY_BRANCH,
        "primary": primary,
        "later_window": later,
        "support": support,
        "endpoint_id_comparison": comparison,
        "association_models_fitted": len(ledger.fitted),
        "authorized_models_fitted": ledger.fitted,
        "first_corrected_association_read_utc": association_read_utc,
        "original_mimic3_classification_preserved": ORIGINAL_CLASSIFICATION,
        "forensic_classification_preserved": FORENSIC_CLASSIFICATION,
        "flags": flags,
    }
    precision = precision_analysis(primary)
    heterogeneity = heterogeneity_analysis(
        primary, original_result["primary"], mimic4_result["primary"]
    )
    write_json(RESULTS, result)
    write_json(PRECISION, precision)
    write_json(HETEROGENEITY, heterogeneity)
    status = {
        "status": "completed",
        "classification": classification,
        "environment": "LOCAL_STANDALONE_RUNTIME_NOT_COLAB",
        "association_models_fitted": len(ledger.fitted),
        "corrected_association_inspected": True,
        "flags": flags,
        "protected_artifacts_verified_unchanged": True,
    }
    write_json(RUN_STATUS, status)
    write_forest_figure(original_result, mimic4_result, result)
    write_final_report(
        classification=classification,
        support=support,
        comparison=comparison,
        results=result,
        precision=precision,
        heterogeneity=heterogeneity,
        flags=flags,
    )
    update_notebook_section(classification)
    verify_lock(verify_sources=False)
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--baseline", action="store_true")
    stage.add_argument("--prepare-lock", action="store_true")
    stage.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.baseline:
        establish_protected_baseline()
    elif args.prepare_lock:
        prepare_lock()
    else:
        run_locked_analysis()


if __name__ == "__main__":
    main()
