#!/usr/bin/env python3
"""Fail closed when a committed PhysioGraph analysis is internally inconsistent."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physiograph_colab_core import (
    _artifact_ready,
    _current_project_code_hashes,
    _file_sha256,
)
from physiograph.analysis.spo2_multiorgan_mechanistic import (
    ANALYZED_DEFINITIONS,
    FOCUSED_WINDOWS_BY_FAMILY,
    LAG_WINDOWS as MULTIORGAN_LAG_WINDOWS,
    PRIMARY_DEFINITION,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path)
    parser.add_argument("--eicu-root", type=Path)
    return parser


def _fingerprint_matches(fingerprint: dict) -> bool:
    path = Path(str(fingerprint.get("path", "")))
    return bool(
        path.is_file()
        and int(fingerprint.get("size_bytes", -1)) == int(path.stat().st_size)
        and fingerprint.get("sha256") == _file_sha256(path)
    )


def _true(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().eq("true")


def main() -> None:
    args = _parser().parse_args()
    root = args.output_root.expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    run_status_path = root / "run_status.json"
    run_status = (
        json.loads(run_status_path.read_text(encoding="utf-8"))
        if run_status_path.is_file()
        else {}
    )
    run_status_manifest_matches = (
        run_status.get("spo2_manifest_sha256") == _file_sha256(manifest_path)
    )
    if not run_status_manifest_matches:
        failures.append("top-level run status points to a stale analysis manifest hash")

    if manifest.get("code_sha256") != _current_project_code_hashes():
        failures.append("analysis manifest code hashes differ from current source")
    bad_outputs = [
        name
        for name, fingerprint in manifest.get("output_fingerprints", {}).items()
        if not _fingerprint_matches(fingerprint)
    ]
    bad_figures = [
        str(fingerprint.get("path", ""))
        for fingerprint in manifest.get("figure_fingerprints", [])
        if not _fingerprint_matches(fingerprint)
    ]
    if bad_outputs:
        failures.append(f"output fingerprint failures: {bad_outputs}")
    if bad_figures:
        failures.append(f"figure fingerprint failures: {bad_figures}")

    roots = {
        "mimic": args.mimic_root.expanduser().resolve() if args.mimic_root else None,
        "eicu": args.eicu_root.expanduser().resolve() if args.eicu_root else None,
    }
    dataset_ready: dict[str, bool] = {}
    for dataset in ("mimic", "eicu"):
        dataset_ready[dataset] = _artifact_ready(
            root / dataset,
            dataset=dataset,
            data_root=roots[dataset],
            max_stays=manifest.get("max_stays"),
            max_chunks=manifest.get("max_chunks"),
            chunk_size=int(manifest.get("chunk_size", 500_000)),
            require_current_code_hash=False,
        )
        if not dataset_ready[dataset]:
            failures.append(f"{dataset} cached artifact validation failed")
        input_manifest = root / dataset / "manifest.json"
        committed = manifest.get("input_manifests", {}).get(dataset, {}).get("sha256")
        if committed != _file_sha256(input_manifest):
            failures.append(f"{dataset} input-manifest hash mismatch")

    claims = pd.read_csv(output_dir / "claims_linter_warnings.csv")
    claims_errors = int(claims.get("severity", pd.Series(dtype=str)).eq("error").sum())
    if claims_errors:
        failures.append(f"claims linter contains {claims_errors} error(s)")

    low_information_fit_rows: dict[str, int] = {}
    for filename in (
        "spo2_or_pvalues_adjusted_per_feature.csv",
        "spo2_variability_or_ci_models.csv",
    ):
        table = pd.read_csv(output_dir / filename, low_memory=False)
        epp = pd.to_numeric(table.get("events_per_parameter"), errors="coerce")
        fitted_below_five = table.get("status", pd.Series(index=table.index, dtype=str)).eq(
            "fit"
        ) & epp.lt(5)
        low_information_fit_rows[filename] = int(fitted_below_five.sum())
        if fitted_below_five.any():
            failures.append(f"{filename} contains fitted rows below five events/parameter")

    refresh_status_path = output_dir / "lactate_episode_refresh_status.json"
    refresh_status = json.loads(refresh_status_path.read_text(encoding="utf-8"))
    if refresh_status.get("status") != "complete":
        failures.append("episode component refresh is not complete")
    if refresh_status.get("raw_data_rebuilt") is not False:
        failures.append("episode refresh does not explicitly attest no raw rebuild")

    conclusion_refresh_path = output_dir / "endpoint_conclusions_refresh_status.json"
    conclusion_refresh = (
        json.loads(conclusion_refresh_path.read_text(encoding="utf-8"))
        if conclusion_refresh_path.is_file()
        else {}
    )
    if conclusion_refresh.get("status") != "complete":
        failures.append("endpoint-conclusion component refresh is not complete")
    if conclusion_refresh.get("raw_data_rebuilt") is not False:
        failures.append(
            "endpoint-conclusion refresh does not explicitly attest no raw rebuild"
        )

    multiorgan_refresh_path = output_dir / "multiorgan_episode_refresh_status.json"
    multiorgan_refresh = (
        json.loads(multiorgan_refresh_path.read_text(encoding="utf-8"))
        if multiorgan_refresh_path.is_file()
        else {}
    )
    if multiorgan_refresh.get("status") != "complete":
        failures.append("all-endpoint episode refresh is not complete")
    if multiorgan_refresh.get("analysis_only") is not True:
        failures.append("all-endpoint episode refresh is not marked analysis-only")
    if multiorgan_refresh.get("raw_data_rebuilt") is not False:
        failures.append("all-endpoint episode refresh does not attest no raw rebuild")

    weighted_refresh_path = output_dir / "multiorgan_weighted_refresh_status.json"
    weighted_refresh = (
        json.loads(weighted_refresh_path.read_text(encoding="utf-8"))
        if weighted_refresh_path.is_file()
        else {}
    )
    if weighted_refresh.get("status") != "complete":
        failures.append("all-endpoint weighted refresh is not complete")
    if weighted_refresh.get("analysis_only") is not True:
        failures.append("all-endpoint weighted refresh is not marked analysis-only")
    if weighted_refresh.get("raw_data_rebuilt") is not False:
        failures.append("all-endpoint weighted refresh does not attest no raw rebuild")
    if weighted_refresh.get("focused_records_rebuilt") is not False:
        failures.append("weighted refresh unexpectedly rebuilt focused records")

    advanced_refresh_path = output_dir / "advanced_episode_refresh_status.json"
    advanced_refresh = (
        json.loads(advanced_refresh_path.read_text(encoding="utf-8"))
        if advanced_refresh_path.is_file()
        else {}
    )
    if advanced_refresh.get("status") != "complete":
        failures.append("advanced episode inference refresh is not complete")
    if advanced_refresh.get("analysis_only") is not True:
        failures.append("advanced episode inference is not marked analysis-only")
    if advanced_refresh.get("raw_data_rebuilt") is not False:
        failures.append("advanced episode inference does not attest no raw rebuild")
    if advanced_refresh.get("focused_records_rebuilt") is not False:
        failures.append("advanced episode inference unexpectedly rebuilt focused records")

    multiorgan_files = {
        "records": "spo2_multiorgan_episode_records.csv",
        "effects": "spo2_multiorgan_episode_effects.csv",
        "observation": "spo2_multiorgan_episode_observation_process.csv",
        "weighted": "spo2_multiorgan_episode_measurement_weighted.csv",
        "meta": "spo2_multiorgan_episode_meta_analysis.csv",
        "evidence": "spo2_multiorgan_episode_evidence_summary.csv",
        "key": "spo2_multiorgan_episode_key_results.csv",
        "definitions": "spo2_multiorgan_episode_endpoint_definitions.csv",
        "availability": "spo2_multiorgan_episode_availability.csv",
    }
    multiorgan_tables: dict[str, pd.DataFrame] = {}
    for table_name, filename in multiorgan_files.items():
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing all-endpoint output: {filename}")
            continue
        table = pd.read_csv(path, low_memory=False)
        multiorgan_tables[table_name] = table
        if table.empty:
            failures.append(f"empty all-endpoint output: {filename}")

    coverage_failures: dict[str, int] = {}
    required_table_names = {
        "records",
        "effects",
        "observation",
        "weighted",
        "meta",
        "evidence",
        "key",
        "definitions",
        "availability",
    }
    if required_table_names.issubset(multiorgan_tables):
        effects = multiorgan_tables["effects"]
        definitions = multiorgan_tables["definitions"]
        weighted = multiorgan_tables["weighted"]
        key = multiorgan_tables["key"]
        endpoints = set(definitions["endpoint"].astype(str))
        datasets = set(effects["dataset"].astype(str))
        lag_windows = {str(row[0]) for row in MULTIORGAN_LAG_WINDOWS}
        primary = effects.loc[
            effects["episode_definition"].eq(PRIMARY_DEFINITION[0])
            & effects["signal_resolution"].eq(PRIMARY_DEFINITION[1])
            & _true(effects["primary_endpoint_variant"])
        ]
        actual_primary = set(
            primary[["dataset", "endpoint", "lag_window"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        expected_primary = {
            (dataset, endpoint, window)
            for dataset in datasets
            for endpoint in endpoints
            for window in lag_windows
        }
        missing_primary = expected_primary - actual_primary
        coverage_failures["missing_primary_endpoint_windows"] = len(missing_primary)
        if missing_primary:
            failures.append(
                f"all-endpoint primary grid is missing {len(missing_primary)} dataset/endpoint/window rows"
            )

        endpoint_family = definitions.set_index("endpoint")["endpoint_family"].astype(str)
        sensitivity_definitions = {
            (str(item["episode_definition"]), str(item["signal_resolution"]))
            for item in ANALYZED_DEFINITIONS
            if (
                str(item["episode_definition"]),
                str(item["signal_resolution"]),
            )
            != PRIMARY_DEFINITION
        }
        expected_sensitivity = {
            (dataset, endpoint, definition, resolution, window)
            for dataset in datasets
            for endpoint in endpoints
            for definition, resolution in sensitivity_definitions
            for window in FOCUSED_WINDOWS_BY_FAMILY[endpoint_family.loc[endpoint]]
        }
        actual_sensitivity = set(
            effects.loc[
                ~(
                    effects["episode_definition"].eq(PRIMARY_DEFINITION[0])
                    & effects["signal_resolution"].eq(PRIMARY_DEFINITION[1])
                )
                & _true(effects["primary_endpoint_variant"]),
                [
                    "dataset",
                    "endpoint",
                    "episode_definition",
                    "signal_resolution",
                    "lag_window",
                ],
            ]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        missing_sensitivity = expected_sensitivity - actual_sensitivity
        coverage_failures["missing_sensitivity_endpoint_windows"] = len(
            missing_sensitivity
        )
        if missing_sensitivity:
            failures.append(
                f"all-endpoint sensitivity grid is missing {len(missing_sensitivity)} rows"
            )

        expected_focused = {
            (dataset, endpoint, window)
            for dataset in datasets
            for endpoint in endpoints
            for window in FOCUSED_WINDOWS_BY_FAMILY[endpoint_family.loc[endpoint]]
        }
        actual_weighted = set(
            weighted[["dataset", "endpoint", "lag_window"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        actual_key = set(
            key[["dataset", "endpoint", "lag_window"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        coverage_failures["missing_weighted_focused_rows"] = len(
            expected_focused - actual_weighted
        )
        coverage_failures["missing_key_result_rows"] = len(
            expected_focused - actual_key
        )
        if expected_focused != actual_weighted:
            failures.append("weighted table does not exactly cover the focused grid")
        if expected_focused != actual_key:
            failures.append("key-results table does not exactly cover the focused grid")

        invalid_weighted_status = weighted["status"].astype(str).str.contains(
            "error|failed", case=False, regex=True
        )
        if invalid_weighted_status.any():
            failures.append(
                f"weighted table contains {int(invalid_weighted_status.sum())} model error row(s)"
            )
        estimated_weighted = weighted["status"].isin(
            ["estimated", "not_needed_complete_observation"]
        )
        invalid_gee = estimated_weighted & ~weighted[
            "weighted_inference_method"
        ].eq("cluster_robust_gee_poisson_log_link")
        if invalid_gee.any():
            failures.append(
                f"weighted table contains {int(invalid_gee.sum())} estimated row(s) without cluster-robust GEE"
            )

        required_key_flags = {
            "unadjusted_cross_fdr_positive",
            "adjusted_claim_ready_positive",
            "adjusted_cross_fdr_positive",
            "weighted_claim_ready_positive",
            "weighted_cross_fdr_positive",
            "replicated_random_meta_positive",
        }
        missing_key_flags = required_key_flags - set(key.columns)
        if missing_key_flags:
            failures.append(
                f"key-results table lacks evidence flags: {sorted(missing_key_flags)}"
            )
        else:
            robust = key["evidence_label"].eq(
                "robust_to_adjustment_weighting_and_cross_endpoint_fdr"
            )
            robust_support = (
                _true(key["unadjusted_cross_fdr_positive"])
                & _true(key["adjusted_cross_fdr_positive"])
                & _true(key["weighted_cross_fdr_positive"])
            )
            invalid_key_robust = int((robust & ~robust_support).sum())
            if invalid_key_robust:
                failures.append(
                    f"key results contain {invalid_key_robust} overgraded robust label(s)"
                )
    else:
        missing_tables = sorted(required_table_names - set(multiorgan_tables))
        coverage_failures["missing_tables"] = len(missing_tables)

    advanced_files = {
        "preanchor": "spo2_advanced_episode_preanchor_covariates.csv",
        "weights": "spo2_advanced_episode_anchor_weights.csv",
        "balance": "spo2_advanced_episode_covariate_balance.csv",
        "overlap_summary": "spo2_advanced_episode_overlap_summary.csv",
        "overlap_effects": "spo2_advanced_episode_overlap_weighted_effects.csv",
        "aipw": "spo2_advanced_episode_aipw_effects.csv",
        "continuous_aipw": "spo2_advanced_episode_continuous_aipw_effects.csv",
        "site": "spo2_advanced_episode_site_effects.csv",
        "multicenter": "spo2_advanced_episode_multicenter_summary.csv",
        "evalues": "spo2_advanced_episode_evalues.csv",
        "key": "spo2_advanced_episode_key_results.csv",
        "design": "spo2_advanced_episode_design.csv",
    }
    advanced_tables: dict[str, pd.DataFrame] = {}
    for table_name, filename in advanced_files.items():
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing advanced-inference output: {filename}")
            continue
        table = pd.read_csv(path, low_memory=False)
        advanced_tables[table_name] = table
        if table.empty:
            failures.append(f"empty advanced-inference output: {filename}")

    required_advanced = set(advanced_files)
    if required_advanced.issubset(advanced_tables) and "expected_focused" in locals():
        for table_name in (
            "overlap_effects",
            "aipw",
            "continuous_aipw",
            "multicenter",
            "key",
        ):
            table = advanced_tables[table_name]
            actual = set(
                table[["dataset", "endpoint", "lag_window"]]
                .astype(str)
                .itertuples(index=False, name=None)
            )
            missing = expected_focused - actual
            extra = actual - expected_focused
            coverage_failures[f"advanced_{table_name}_missing_focused_rows"] = len(
                missing
            )
            coverage_failures[f"advanced_{table_name}_extra_focused_rows"] = len(extra)
            if missing or extra:
                failures.append(
                    f"advanced {table_name} does not exactly cover focused grid "
                    f"(missing={len(missing)}, extra={len(extra)})"
                )

        preanchor = advanced_tables["preanchor"]
        records = multiorgan_tables["records"]
        expected_anchors = records[["dataset", "stay_id"]].drop_duplicates()
        actual_anchors = preanchor[["dataset", "stay_id"]].drop_duplicates()
        if len(preanchor) != len(actual_anchors):
            failures.append("advanced pre-anchor table is not one row per stay anchor")
        if set(expected_anchors.astype(str).itertuples(index=False, name=None)) != set(
            actual_anchors.astype(str).itertuples(index=False, name=None)
        ):
            failures.append("advanced pre-anchor table does not cover every focused anchor")
        boundary = preanchor.get(
            "preanchor_boundary_rule", pd.Series(index=preanchor.index, dtype=str)
        ).eq("offset_minutes>=0_and_strictly_less_than_anchor")
        if not boundary.all():
            failures.append("advanced covariates do not all attest strict pre-anchor timing")

        overlap_summary = advanced_tables["overlap_summary"]
        remaining_imbalanced = pd.to_numeric(
            overlap_summary.get("covariates_above_0_10_overlap_weighted"),
            errors="coerce",
        ).fillna(1)
        if remaining_imbalanced.gt(0).any():
            failures.append("overlap weighting leaves covariates above absolute SMD 0.10")
        overlap_ess = pd.to_numeric(
            overlap_summary.get("overlap_effective_sample_size"), errors="coerce"
        )
        if overlap_ess.isna().any() or overlap_ess.le(0).any():
            failures.append("overlap effective sample size is missing or non-positive")

        for table_name in ("overlap_effects", "aipw", "continuous_aipw"):
            table = advanced_tables[table_name]
            computational_failure = table["status"].astype(str).str.contains(
                "nuisance_model|nuisance_prediction|observation_model|outcome_model|cluster_variance|weights",
                case=False,
                regex=True,
            )
            if computational_failure.any():
                failures.append(
                    f"advanced {table_name} contains "
                    f"{int(computational_failure.sum())} computational model failure row(s)"
                )
        aipw = advanced_tables["aipw"]
        targeted = aipw["status"].eq("estimated")
        invalid_targeted_risk = targeted & ~(
            pd.to_numeric(aipw["tmle_risk_exposed"], errors="coerce").between(
                0, 1, inclusive="neither"
            )
            & pd.to_numeric(
                aipw["tmle_risk_unexposed"], errors="coerce"
            ).between(0, 1, inclusive="neither")
            & pd.to_numeric(aipw["tmle_risk_ratio"], errors="coerce").gt(0)
        )
        if invalid_targeted_risk.any():
            failures.append(
                f"advanced targeted estimator contains {int(invalid_targeted_risk.sum())} invalid bounded-risk row(s)"
            )
        targeting_score = pd.concat(
            [
                pd.to_numeric(
                    aipw["tmle_efficient_influence_mean_exposed"], errors="coerce"
                ).abs(),
                pd.to_numeric(
                    aipw["tmle_efficient_influence_mean_unexposed"], errors="coerce"
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)
        invalid_targeting = targeted & targeting_score.gt(1e-6)
        if invalid_targeting.any():
            failures.append(
                f"advanced targeted estimator contains {int(invalid_targeting.sum())} unconverged targeting row(s)"
            )
        boundary_claim = (
            _true(
                aipw.get(
                    "tmle_targeting_boundary_hit",
                    pd.Series(False, index=aipw.index),
                )
            )
            & _true(
                aipw.get(
                    "tmle_claim_ready_positive",
                    pd.Series(False, index=aipw.index),
                )
            )
        )
        if boundary_claim.any():
            failures.append("advanced targeted estimator claims a boundary-hit fit")

        for table_name, claim_column in (
            ("overlap_effects", "overlap_claim_ready_positive"),
            ("aipw", "tmle_claim_ready_positive"),
            ("continuous_aipw", "continuous_claim_ready_positive"),
        ):
            table = advanced_tables[table_name]
            unsupported_claim = _true(
                table.get(claim_column, pd.Series(False, index=table.index))
            ) & ~_true(
                table.get(
                    "claim_ready_weight_diagnostics",
                    pd.Series(False, index=table.index),
                )
            )
            if unsupported_claim.any():
                failures.append(
                    f"advanced {table_name} contains a positive claim that fails "
                    "the prespecified weight-support diagnostics"
                )

        design = advanced_tables["design"].iloc[0]
        if str(design.get("preanchor_rule")) != (
            "offset_minutes>=0_and_strictly_less_than_anchor"
        ):
            failures.append("advanced design does not enforce strict pre-anchor timing")
        if str(design.get("concurrent_spo2_use")) != (
            "outcome_and_observation_models_only_not_propensity_or_balance"
        ):
            failures.append("advanced design mishandles concurrent SpO2 in propensity")
        if str(design.get("observation_clip")) != "0.02_1.0":
            failures.append(
                "advanced design spuriously upper-truncates valid observation probabilities"
            )

        advanced_key = advanced_tables["key"]
        required_advanced_flags = {
            "overlap_cross_endpoint_fdr_positive",
            "tmle_cross_endpoint_fdr_positive",
            "continuous_cross_endpoint_fdr_positive",
            "multicenter_cross_endpoint_fdr_positive",
            "advanced_cross_fdr_methods",
            "advanced_evidence_label",
        }
        missing_flags = required_advanced_flags - set(advanced_key.columns)
        if missing_flags:
            failures.append(
                f"advanced key results lack evidence fields: {sorted(missing_flags)}"
            )
        else:
            strongest = advanced_key["advanced_evidence_label"].eq(
                "multicenter_overlap_and_targeted_dr_cross_fdr_support"
            )
            support = (
                _true(advanced_key["overlap_cross_endpoint_fdr_positive"])
                & _true(advanced_key["tmle_cross_endpoint_fdr_positive"])
                & _true(advanced_key["multicenter_cross_endpoint_fdr_positive"])
            )
            if (strongest & ~support).any():
                failures.append("advanced key results contain an overgraded strongest label")
            any_advanced_cross_fdr = (
                _true(advanced_key["overlap_cross_endpoint_fdr_positive"])
                | _true(advanced_key["tmle_cross_endpoint_fdr_positive"])
                | _true(advanced_key["continuous_cross_endpoint_fdr_positive"])
                | _true(advanced_key["multicenter_cross_endpoint_fdr_positive"])
            )
            mislabeled_supported = any_advanced_cross_fdr & (
                advanced_key["advanced_cross_fdr_methods"].fillna("").eq("")
                | advanced_key["advanced_evidence_label"].isin(
                    {
                        "advanced_ci_support_but_cross_endpoint_fdr_sensitive",
                        "prior_robust_signal_not_reinforced_by_advanced_estimators",
                        "directionally_positive_advanced_but_inconclusive",
                        "directionally_positive_inconclusive",
                        "no_positive_association",
                    }
                )
            )
            if mislabeled_supported.any():
                failures.append(
                    "advanced key results undergrade one or more cross-endpoint-FDR signals"
                )

    locked_status_path = output_dir / "locked_external_validation_refresh_status.json"
    locked_status = (
        json.loads(locked_status_path.read_text(encoding="utf-8"))
        if locked_status_path.is_file()
        else {}
    )
    if locked_status.get("status") != "complete":
        failures.append("locked eICU-to-MIMIC validation refresh is not complete")
    if locked_status.get("analysis_only") is not True:
        failures.append("locked validation is not marked analysis-only")
    if locked_status.get("raw_data_rebuilt") is not False:
        failures.append("locked validation does not attest no raw rebuild")
    expected_targeted_sources = {"d_items.csv", "procedureevents.csv"}
    if set(locked_status.get("targeted_raw_tables_read", [])) != expected_targeted_sources:
        failures.append("locked validation targeted-source declaration is incomplete")

    locked_files = {
        "effects": "spo2_locked_external_validation_effects.csv",
        "summary": "spo2_locked_external_validation_summary.csv",
        "cohort_audit": "spo2_locked_external_validation_hf_cohort_audit.csv",
        "inventory": "spo2_locked_external_validation_mimic_endpoint_inventory.csv",
        "eicu_endpoint_audit": "spo2_locked_external_validation_eicu_endpoint_audit.csv",
        "advanced": "spo2_locked_external_validation_advanced_key_results.csv",
    }
    locked_tables: dict[str, pd.DataFrame] = {}
    for table_name, filename in locked_files.items():
        path = output_dir / filename
        if not path.is_file():
            failures.append(f"missing locked-validation output: {filename}")
            continue
        locked_tables[table_name] = pd.read_csv(path, low_memory=False)
    if set(locked_files).issubset(locked_tables):
        locked_effects = locked_tables["effects"]
        invalid_design = ~(
            locked_effects["episode_definition"].eq("absolute_jump_ge4")
            & locked_effects["signal_resolution"].eq("15_minute_median_bins")
            & locked_effects["lag_window"].eq("4_to_12h")
        )
        if invalid_design.any():
            failures.append("locked validation contains a non-locked exposure or window")
        primary = locked_effects.loc[
            locked_effects["endpoint"].eq("invasive_ventilation_initiation")
        ]
        if set(primary["dataset"].astype(str)) != {"mimic", "eicu"}:
            failures.append("locked ventilation primary is not present in both datasets")
        if primary["status"].eq("unavailable").any() or pd.to_numeric(
            primary["n_outcome_observed"], errors="coerce"
        ).le(0).any():
            failures.append("locked ventilation primary is unavailable or eventless")
        inventory = locked_tables["inventory"].set_index("endpoint")
        for endpoint in (
            "invasive_ventilation_initiation",
            "intubation_initiation",
        ):
            if endpoint not in inventory.index or pd.to_numeric(
                inventory.loc[endpoint, "procedure_rows_in_harmonized_hf_cohort"],
                errors="coerce",
            ) <= 0:
                failures.append(f"direct MIMIC endpoint inventory is empty: {endpoint}")
        audit = locked_tables["cohort_audit"]
        if set(audit["dataset"].astype(str)) != {"mimic", "eicu"} or pd.to_numeric(
            audit["included_encounters"], errors="coerce"
        ).le(0).any():
            failures.append("harmonized HF cohort audit is incomplete")
        eicu_endpoint_audit = locked_tables["eicu_endpoint_audit"]
        if len(eicu_endpoint_audit) != 1:
            failures.append("eICU ventilation-cleaning audit is not one explicit rule")
        else:
            eicu_row = eicu_endpoint_audit.iloc[0]
            if pd.to_numeric(
                eicu_row.get("retained_literal_mechanical_ventilation_rows"),
                errors="coerce",
            ) <= 0:
                failures.append("eICU ventilation cleaner retained no valid rows")
            if pd.to_numeric(
                eicu_row.get("removed_total_rows"), errors="coerce"
            ) <= 0:
                failures.append("eICU ventilation cleaner did not audit false positives")
            if eicu_row.get("time_semantics") != (
                "first_treatment_documentation_proxy_not_true_start"
            ):
                failures.append("eICU ventilation time semantics are not explicit")
        advanced_locked = locked_tables["advanced"]
        expected_locked_pairs = set(
            locked_effects[["dataset", "endpoint", "lag_window"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        actual_locked_pairs = set(
            advanced_locked[["dataset", "endpoint", "lag_window"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if expected_locked_pairs != actual_locked_pairs:
            failures.append("locked advanced inference does not cover every endpoint")

    conclusions = pd.read_csv(output_dir / "endpoint_conclusion_matrix.csv")
    robust = conclusions.get(
        "classification", pd.Series(index=conclusions.index, dtype=str)
    ).eq("robust_positive")
    internal_epv = conclusions.get(
        "incremental_epv_adequate_for_claim",
        pd.Series(False, index=conclusions.index),
    ).fillna(False).astype(str).str.lower().eq("true")
    external_epv = conclusions.get(
        "external_training_epv_adequate_for_claim",
        pd.Series(False, index=conclusions.index),
    ).fillna(False).astype(str).str.lower().eq("true")
    invalid_robust_claims = int((robust & ~(internal_epv & external_epv)).sum())
    if invalid_robust_claims:
        failures.append(
            f"endpoint conclusions contain {invalid_robust_claims} robust claim(s) without EPV gates"
        )

    report = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "analysis_outputs_checked": len(manifest.get("output_fingerprints", {})),
        "figures_checked": len(manifest.get("figure_fingerprints", [])),
        "dataset_artifacts_ready": dataset_ready,
        "claims_linter_errors": claims_errors,
        "fitted_rows_below_five_events_per_parameter": low_information_fit_rows,
        "episode_refresh_duration_seconds": refresh_status.get("duration_seconds", math.nan),
        "raw_data_rebuilt_for_episode_refresh": refresh_status.get("raw_data_rebuilt"),
        "endpoint_conclusion_refresh_duration_seconds": conclusion_refresh.get(
            "duration_seconds", math.nan
        ),
        "raw_data_rebuilt_for_endpoint_conclusion_refresh": conclusion_refresh.get(
            "raw_data_rebuilt"
        ),
        "robust_claims_without_epv_gates": invalid_robust_claims,
        "run_status_manifest_hash_matches": run_status_manifest_matches,
        "multiorgan_episode_refresh_duration_seconds": multiorgan_refresh.get(
            "duration_seconds", math.nan
        ),
        "multiorgan_weighted_refresh_duration_seconds": weighted_refresh.get(
            "duration_seconds", math.nan
        ),
        "raw_data_rebuilt_for_multiorgan_refresh": multiorgan_refresh.get(
            "raw_data_rebuilt"
        ),
        "raw_data_rebuilt_for_multiorgan_weighted_refresh": weighted_refresh.get(
            "raw_data_rebuilt"
        ),
        "advanced_episode_refresh_duration_seconds": advanced_refresh.get(
            "duration_seconds", math.nan
        ),
        "raw_data_rebuilt_for_advanced_episode_refresh": advanced_refresh.get(
            "raw_data_rebuilt"
        ),
        "multiorgan_coverage_failures": coverage_failures,
        "locked_external_validation_refresh_duration_seconds": locked_status.get(
            "duration_seconds", math.nan
        ),
        "locked_external_validation_raw_data_rebuilt": locked_status.get(
            "raw_data_rebuilt"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
