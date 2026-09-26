"""Minimal Colab runner for the PhysioGraph final workflow.

The notebook should only mount Drive, set paths, call ``run_physiograph_colab``,
and display the artifacts written here.  This module keeps the reusable logic in
plain Python so it can be tested outside Colab and rerun without copying large
code blocks into the notebook.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from physiograph.analysis.spo2_protocol import (
    POST_LANDMARK_HORIZONS_HOURS,
    PRIMARY_ENDPOINTS as PROTOCOL_PRIMARY_ENDPOINTS,
    SECONDARY_ENDPOINTS as PROTOCOL_SECONDARY_ENDPOINTS,
    build_endpoint_completeness_audit,
    build_continuous_trajectory_associations,
    build_temporal_precedence,
    compute_early_decompensation_outcomes,
    fit_external_transportability,
    fit_grouped_incremental_models,
    fit_grouped_missingness_control,
)
from physiograph.analysis.spo2_lactate_mechanistic import (
    run_lactate_episode_analyses,
)
from physiograph.analysis.spo2_multiorgan_mechanistic import (
    rebuild_multiorgan_weighted_sensitivity,
    run_locked_external_replication,
    run_multiorgan_episode_analyses,
)
from physiograph.analysis.spo2_advanced_inference import (
    run_advanced_episode_inference,
)
from physiograph.analysis.biomarker_benchmark import run_biomarker_benchmark
from physiograph.analysis.spo2_instability import (
    HORIZONS_HOURS as INSTABILITY_HORIZONS_HOURS,
    PROTOCOL_VERSION as INSTABILITY_PROTOCOL_VERSION,
    run_multiscale_instability_analysis,
)
from physiograph.pipeline import PIPELINE_SCHEMA_VERSION
from physiograph.cohort.harmonization import harmonize_hf_cohort
from physiograph.etl.mimic_extractor import (
    extract_mimic_respiratory_procedure_events,
)


LANDMARK_MINUTES = 240
OBSERVATION_HOURS = 4.0
OBSERVATION_BINS = 16
# Long horizons are retained only for backwards-compatible exploratory tables.
HORIZON_HOURS = (48, 72, 96, 168)
LOW_SPO2_THRESHOLDS = (88, 90, 92, 95)
REQUIRED_ARTIFACT_FILES = ("cohort.csv", "events.csv", "features.csv", "labels.csv")
PRIMARY_ENDPOINT = "lactate_rise_24h_flag"
PRIMARY_ENDPOINTS = PROTOCOL_PRIMARY_ENDPOINTS
SECONDARY_ENDPOINTS = PROTOCOL_SECONDARY_ENDPOINTS
ANALYSIS_LAYERS = (
    "descriptive",
    "association_or",
    "predictive_cv",
    "respiratory_stratified",
    "availability_audit",
    "external_cross_dataset",
    "measurement_intensity_control",
    "episode_anchored_lactate_mechanistic",
    "episode_anchored_multiorgan_mechanistic",
    "episode_anchored_advanced_inference",
    "cross_dataset_meta_analysis",
    "head_to_head_biomarker_benchmark",
    "multiscale_dynamics_first_instability",
    "limitations",
)
MIN_OR_EVENTS = 20
DATASET_HOLDOUT_MIN_ROWS = 100
DATASET_HOLDOUT_MIN_EVENTS = 20
MISSINGNESS_CONTROL_FEATURES = (
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_longest_gap_minutes",
    "spo2_plausible_count",
)
MANIFEST_WARNINGS = (
    "structured_respiratory_rrt_fields_used_where_available_text_fields_remain_proxies",
    "eicu_quantitative_vis_unavailable_for_unstandardized_infusion_rates",
    "mimic_urine_endpoints_unavailable_when_outputevents_source_absent",
    "proxy_signal_quality_when_probe_flags_absent",
    "charted_spo2_is_not_raw_plethysmographic_waveform",
    "per_dataset_cv_is_internal_not_external_validation",
    "pooled_results_are_secondary_only",
    "clif_out_of_scope",
    "no_causal_language",
)


def _execution_provenance_note(manifest: dict[str, Any]) -> str:
    if manifest.get("analysis_only", False):
        return "analysis_only_rerun_from_committed_etl_artifacts_not_fresh_colab_extraction"
    if manifest.get("fresh_colab_execution", False):
        return "fresh_colab_execution"
    if manifest.get("build_new", False):
        environment = str(manifest.get("execution_environment", "local"))
        return f"fresh_{environment}_execution_not_colab"
    return "cached_or_precomputed_execution_not_colab"
PRIMARY_OUTCOME_COLUMNS = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
PRIMARY_SPO2_MODEL_FEATURES = (
    "spo2_min",
    "spo2_mean",
    "spo2_below_90_fraction",
    "spo2_sd",
    "spo2_rmssd",
    "spo2_abrupt_jump_fraction",
    "spo2_drop_3_count",
    "spo2_dynamics_proxy_score",
)
CONTROL_FEATURE_CANDIDATES = (
    "age",
    "is_male",
    "sex_unknown_flag",
    "shock_icd_flag",
    "baseline_lactate",
    "baseline_creatinine",
    "baseline_map",
    "baseline_hr",
    "baseline_sbp",
    "baseline_ph",
    "baseline_bilirubin_total",
    "baseline_resp_rate",
    "baseline_vasoactive_flag",
    "baseline_mcs_flag",
    "fio2_max",
    "resp_support_any_flag",
    "mechanical_ventilation_flag",
    "noninvasive_ventilation_flag",
    "rrt_or_dialysis_flag",
    "dataset",
    "race",
    "ethnicity",
    "race_ethnicity",
    "hospital_id",
    "icu_type",
    "unit_admit_source",
    "admit_year",
)
PARSIMONIOUS_CONTROL_CANDIDATES = (
    "age",
    "is_male",
    "shock_icd_flag",
    "baseline_lactate",
    "baseline_creatinine",
    "baseline_map",
    "baseline_vasoactive_flag",
    "resp_support_any_flag",
)
SPO2_VARIABILITY_FEATURES = (
    "spo2_sd",
    "spo2_rmssd",
    "spo2_iqr",
    "spo2_range",
    "spo2_mad",
    "spo2_abrupt_jump_rate_per_hr",
    "spo2_abrupt_jump_count",
    "spo2_abrupt_jump_fraction",
    "spo2_drop_3_count",
    "spo2_drop_5_count",
    "spo2_slope_per_hr",
    "spo2_dynamics_proxy_score",
)
SPO2_VARIABILITY_OUTCOMES = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
SPO2_OXYGENATION_COVARIATES = (
    "spo2_mean",
    "spo2_min",
    "spo2_below_90_fraction",
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_longest_gap_minutes",
)

SPO2_SIGNAL_COLUMNS = [
    "spo2_measurement_count",
    "spo2_unique_timestamp_count",
    "spo2_plausible_count",
    "spo2_implausible_count",
    "spo2_observed_bin_count",
    "spo2_valid_bin_count",
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_monitoring_fraction",
    "spo2_monitoring_span_minutes",
    "spo2_variability_pair_count",
    "spo2_dynamics_eligible_flag",
    "spo2_mean",
    "spo2_median",
    "spo2_min",
    "spo2_max",
    "spo2_first",
    "spo2_last",
    "spo2_sd",
    "spo2_iqr",
    "spo2_mad",
    "spo2_range",
    "spo2_rmssd",
    "spo2_median_gap_minutes",
    "spo2_longest_gap_minutes",
    "spo2_slope_per_hr",
    "spo2_below_88_fraction",
    "spo2_below_90_fraction",
    "spo2_below_92_fraction",
    "spo2_below_95_fraction",
    "spo2_deficit_below_88_mean",
    "spo2_deficit_below_90_mean",
    "spo2_deficit_below_92_mean",
    "spo2_deficit_below_95_mean",
    "spo2_abrupt_jump_count",
    "spo2_abrupt_jump_rate_per_hr",
    "spo2_abrupt_jump_fraction",
    "spo2_sustained_abrupt_jump_episode_count",
    "spo2_jump_3_count",
    "spo2_jump_5_count",
    "spo2_drop_3_count",
    "spo2_drop_5_count",
    "spo2_below_90_duration_minutes",
    "spo2_below_90_deficit_auc",
    "spo2_desaturation_episode_count",
    "spo2_sustained_desaturation_episode_count",
    "spo2_longest_desaturation_minutes",
    "spo2_signal_quality_proxy_score",
    "spo2_dynamics_proxy_score",
    "spo2_instability_proxy_score",
]

RESPIRATORY_SUPPORT_COLUMNS = [
    "fio2_measurement_count",
    "fio2_max",
    "fio2_mean",
    "fio2_above_room_air_flag",
    "intubation_flag",
    "mechanical_ventilation_flag",
    "noninvasive_ventilation_flag",
    "nasal_cannula_flag",
    "oxygen_device_flag",
    "resp_support_any_flag",
]

RRT_COLUMNS = [
    "baseline_rrt_flag",
    "baseline_dialysis_flag",
    "rrt_or_dialysis_flag",
]


def _ensure_project_imports(project_root: Path) -> None:
    """Make project-root scripts and ``src`` imports work in Colab."""
    for candidate in (project_root, project_root / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so interrupted analyses cannot look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def _file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """Hash a cached artifact incrementally for integrity validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _synchronize_run_status_manifest(
    output_root: Path,
    manifest_path: Path,
    *,
    refreshed_at_utc: str,
) -> None:
    """Keep the top-level run receipt aligned after component-only refreshes."""
    status_path = output_root / "run_status.json"
    if not status_path.is_file():
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["spo2_manifest"] = str(manifest_path)
    status["spo2_manifest_sha256"] = _file_sha256(manifest_path)
    status["last_component_refresh_at_utc"] = refreshed_at_utc
    _write_json(status_path, status)


def _current_project_code_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parent
    code_files = [
        project_root / "physiograph_colab_core.py",
        *sorted((project_root / "src" / "physiograph").rglob("*.py")),
    ]
    return {
        str(path.relative_to(project_root)): _file_sha256(path)
        for path in code_files
        if path.is_file()
    }


def _artifact_ready(
    dataset_dir: Path,
    *,
    dataset: str | None = None,
    data_root: Path | None = None,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int | None = None,
    require_current_code_hash: bool = True,
) -> bool:
    """Validate an artifact set against its immutable build manifest."""
    required = (*REQUIRED_ARTIFACT_FILES, "manifest.json", "audit.json")
    if not all((dataset_dir / name).is_file() for name in required):
        return False
    try:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        return False
    if dataset is not None and manifest.get("dataset") != dataset:
        return False
    expected = {
        "max_stays": max_stays,
        "max_chunks": max_chunks,
    }
    if chunk_size is not None:
        expected["chunk_size"] = chunk_size
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    parameters = manifest.get("parameters", {})
    if not isinstance(parameters, dict) or not parameters:
        return False
    expected_parameters_digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if manifest.get("parameters_sha256") != expected_parameters_digest:
        return False
    if data_root is not None:
        if parameters.get("root") != str(data_root.expanduser().resolve()):
            return False
    counts = manifest.get("counts", {})
    for key in ("cohort_rows", "label_rows", "event_rows", "feature_rows"):
        if not isinstance(counts.get(key), int) or counts[key] < 0:
            return False
    if max_chunks is None and counts.get("cohort_rows", 0) > 0:
        if counts.get("spo2_plausible_rows", 0) <= 0:
            return False
    if data_root is not None:
        fingerprints = manifest.get("source_fingerprints", {})
        if not fingerprints:
            return False
        resolved_root = data_root.expanduser().resolve()
        for fingerprint in fingerprints.values():
            source = Path(str(fingerprint.get("path", ""))).expanduser()
            try:
                source.resolve().relative_to(resolved_root)
            except Exception:
                return False
            if not source.is_file():
                return False
            stat = source.stat()
            if int(fingerprint.get("size_bytes", -1)) != int(stat.st_size):
                return False
            if int(fingerprint.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
                return False
        if dataset is not None:
            from physiograph.config import load_config

            config = load_config(dataset=dataset)
            config_digest = hashlib.sha256(
                json.dumps(config, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            if manifest.get("configuration_sha256") != config_digest:
                return False
            expected_sources = set(config.get("source_tables", {}).values())
            if dataset == "mimic":
                expected_sources.update({"d_items.csv", "outputevents.csv"})
            elif dataset == "eicu":
                expected_sources.update(
                    {"intakeOutput.csv", "respiratoryCharting.csv", "hospital.csv"}
                )
            for name in expected_sources:
                if (resolved_root / name).is_file() != (name in fingerprints):
                    return False
    code_hashes = manifest.get("code_sha256", {})
    if not code_hashes:
        return False
    if require_current_code_hash and code_hashes != _current_project_code_hashes():
        return False
    artifact_fingerprints = manifest.get("artifact_fingerprints", {})
    if not artifact_fingerprints:
        return False
    for name in (*REQUIRED_ARTIFACT_FILES, "audit.json"):
        artifact = dataset_dir / name
        fingerprint = artifact_fingerprints.get(name, {})
        if not fingerprint or not artifact.is_file():
            return False
        stat = artifact.stat()
        if int(fingerprint.get("size_bytes", -1)) != int(stat.st_size):
            return False
        if int(fingerprint.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
            return False
        if fingerprint.get("sha256") != _file_sha256(artifact):
            return False
    return True


def _artifact_paths(dataset_dir: Path) -> dict[str, Path]:
    return {
        "cohort": dataset_dir / "cohort.csv",
        "events": dataset_dir / "events.csv",
        "features": dataset_dir / "features.csv",
        "labels": dataset_dir / "labels.csv",
        "manifest": dataset_dir / "manifest.json",
        "audit": dataset_dir / "audit.json",
    }


def _load_dataset_artifacts(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    paths = _artifact_paths(dataset_dir)
    return {
        "cohort": _read_csv(paths["cohort"]),
        "events": _read_csv(paths["events"]),
        "features": _read_csv(paths["features"]),
        "labels": _read_csv(paths["labels"]),
    }


def _run_or_load_dataset(
    dataset: str,
    *,
    dataset_dir: Path,
    mimic_root: Path | None,
    eicu_root: Path | None,
    build_new: bool,
    max_stays: int | None,
    max_chunks: int | None,
    chunk_size: int,
    analysis_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load cached artifacts or rebuild a dataset ETL output."""
    data_root = mimic_root if dataset == "mimic" else eicu_root
    if _artifact_ready(
        dataset_dir,
        dataset=dataset,
        data_root=data_root,
        max_stays=max_stays,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        require_current_code_hash=not analysis_only,
    ) and not build_new:
        return _load_dataset_artifacts(dataset_dir)

    if not build_new:
        missing = [
            name
            for name in (*REQUIRED_ARTIFACT_FILES, "manifest.json", "audit.json")
            if not (dataset_dir / name).exists()
        ]
        raise FileNotFoundError(
            f"{dataset_dir} is missing or incompatible ({missing or 'manifest mismatch'}). "
            "Set BUILD_NEW=True to rebuild ETL with the current schema and sources, or "
            "set analysis_only=True to reuse fingerprint-verified ETL artifacts after "
            "analysis-code changes."
        )

    from physiograph.pipeline import run_pipeline

    if dataset == "mimic":
        if mimic_root is None or not mimic_root.exists():
            raise FileNotFoundError(f"MIMIC root is not available: {mimic_root}")
        run_pipeline(
            dataset="mimic",
            mimic_root=mimic_root,
            output_dir=dataset_dir,
            max_stays=max_stays,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
        )
    elif dataset == "eicu":
        if eicu_root is None or not eicu_root.exists():
            raise FileNotFoundError(f"eICU root is not available: {eicu_root}")
        run_pipeline(
            dataset="eicu",
            eicu_root=eicu_root,
            output_dir=dataset_dir,
            max_stays=max_stays,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return _load_dataset_artifacts(dataset_dir)


def _stay_index_from(*frames: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty or "stay_id" not in frame.columns:
            continue
        cols = ["stay_id"]
        if "dataset" in frame.columns:
            cols = ["dataset", "stay_id"]
        pieces.append(frame[cols].drop_duplicates())
    if not pieces:
        return pd.DataFrame(columns=["dataset", "stay_id"])
    keys = pd.concat(pieces, ignore_index=True).drop_duplicates()
    if "dataset" not in keys.columns:
        keys["dataset"] = "unknown"
    return keys[["dataset", "stay_id"]].drop_duplicates().reset_index(drop=True)


def _event_text(events_df: pd.DataFrame) -> pd.Series:
    text_cols = [col for col in ("concept", "raw_name", "value_text", "source_table") if col in events_df]
    if not text_cols:
        return pd.Series("", index=events_df.index, dtype=str)
    return (
        events_df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )


def _observation_events(events_df: pd.DataFrame) -> pd.DataFrame:
    events = events_df.copy()
    if "offset_minutes" not in events:
        return events.iloc[0:0].copy()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    return events.loc[
        events["offset_minutes"].ge(0) & events["offset_minutes"].lt(LANDMARK_MINUTES)
    ].copy()


def _nan_record(columns: Iterable[str]) -> dict[str, float]:
    return {column: math.nan for column in columns}


def _summarize_spo2_group(group: pd.DataFrame) -> dict[str, float]:
    group = group.copy()
    group["value_numeric"] = pd.to_numeric(group["value_numeric"], errors="coerce")
    group["offset_minutes"] = pd.to_numeric(group["offset_minutes"], errors="coerce")
    group = group.dropna(subset=["value_numeric", "offset_minutes"])
    group = group.loc[
        group["offset_minutes"].ge(0) & group["offset_minutes"].lt(LANDMARK_MINUTES)
    ].sort_values("offset_minutes")

    record = _nan_record(SPO2_SIGNAL_COLUMNS)
    record["spo2_measurement_count"] = float(len(group))
    record["spo2_unique_timestamp_count"] = float(group["offset_minutes"].nunique())
    plausible_mask = group["value_numeric"].between(50, 100, inclusive="both")
    record["spo2_plausible_count"] = float(plausible_mask.sum())
    record["spo2_implausible_count"] = float((~plausible_mask).sum())
    record["spo2_sampling_density_per_hr"] = float(plausible_mask.sum() / OBSERVATION_HOURS)
    if not plausible_mask.any():
        record["spo2_observed_bin_count"] = 0.0
        record["spo2_valid_bin_count"] = 0.0
        record["spo2_missing_bin_count"] = float(OBSERVATION_BINS)
        record["spo2_monitoring_fraction"] = 0.0
        record["spo2_signal_quality_proxy_score"] = float(
            1.0 + (record["spo2_implausible_count"] / max(record["spo2_measurement_count"], 1.0))
        )
        return record

    plausible = group.loc[plausible_mask].copy()
    # Same-time duplicates are collapsed before any transition calculation.
    plausible = (
        plausible.groupby("offset_minutes", as_index=False)["value_numeric"]
        .median()
        .sort_values("offset_minutes")
    )
    plausible["analysis_bin"] = np.floor(plausible["offset_minutes"] / 15.0).astype(int)
    binned = (
        plausible.groupby("analysis_bin", as_index=False)
        .agg(value_numeric=("value_numeric", "median"), offset_minutes=("offset_minutes", "median"))
        .sort_values("analysis_bin")
    )
    values = binned["value_numeric"].astype(float).to_numpy()
    offsets = binned["offset_minutes"].astype(float).to_numpy()
    bins = binned["analysis_bin"].astype(int).to_numpy()
    diffs_all = np.diff(values)
    gaps_all = np.diff(offsets)
    valid_pair = (gaps_all > 0) & (gaps_all <= 30.0)
    diffs = diffs_all[valid_pair]
    gaps = gaps_all[valid_pair]
    record["spo2_observed_bin_count"] = float(len(bins))
    record["spo2_valid_bin_count"] = float(len(bins))
    record["spo2_missing_bin_count"] = float(max(OBSERVATION_BINS - len(bins), 0))
    record["spo2_monitoring_fraction"] = float(len(bins) / OBSERVATION_BINS)
    record["spo2_monitoring_span_minutes"] = float(offsets[-1] - offsets[0]) if len(offsets) > 1 else 0.0
    record["spo2_variability_pair_count"] = float(valid_pair.sum())
    record["spo2_dynamics_eligible_flag"] = float(len(values) >= 3 and valid_pair.sum() >= 2)

    record.update(
        {
            "spo2_mean": float(np.mean(values)),
            "spo2_median": float(np.median(values)),
            "spo2_min": float(np.min(values)),
            "spo2_max": float(np.max(values)),
            "spo2_first": float(values[0]),
            "spo2_last": float(values[-1]),
            "spo2_sd": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
            "spo2_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "spo2_mad": float(np.median(np.abs(values - np.median(values)))),
            "spo2_range": float(np.max(values) - np.min(values)),
            "spo2_rmssd": float(np.sqrt(np.mean(diffs**2))) if len(diffs) else math.nan,
            "spo2_median_gap_minutes": float(np.median(gaps_all)) if len(gaps_all) else math.nan,
            "spo2_longest_gap_minutes": float(np.max(gaps_all)) if len(gaps_all) else math.nan,
        }
    )
    for threshold in LOW_SPO2_THRESHOLDS:
        record[f"spo2_below_{threshold}_fraction"] = float(np.mean(values < threshold))
        record[f"spo2_deficit_below_{threshold}_mean"] = float(
            np.mean(np.maximum(threshold - values, 0))
        )

    if len(np.unique(offsets)) >= 3:
        record["spo2_slope_per_hr"] = float(np.polyfit(offsets / 60.0, values, 1)[0])
    abrupt = np.abs(diffs) >= 4.0
    record["spo2_abrupt_jump_count"] = float(abrupt.sum())
    monitored_hours = float(gaps.sum() / 60.0) if len(gaps) else math.nan
    record["spo2_abrupt_jump_rate_per_hr"] = (
        float(abrupt.sum() / monitored_hours) if monitored_hours > 0 else math.nan
    )
    record["spo2_abrupt_jump_fraction"] = float(abrupt.mean()) if len(diffs) else math.nan
    record["spo2_jump_3_count"] = float((np.abs(diffs) >= 3.0).sum())
    record["spo2_jump_5_count"] = float((np.abs(diffs) >= 5.0).sum())
    record["spo2_drop_3_count"] = float((diffs <= -3.0).sum())
    record["spo2_drop_5_count"] = float((diffs <= -5.0).sum())
    abrupt_all = (np.abs(diffs_all) >= 4.0) & valid_pair
    abrupt_runs: list[int] = []
    current_abrupt_run = 0
    for is_abrupt in abrupt_all:
        if is_abrupt:
            current_abrupt_run += 1
        elif current_abrupt_run:
            abrupt_runs.append(current_abrupt_run)
            current_abrupt_run = 0
    if current_abrupt_run:
        abrupt_runs.append(current_abrupt_run)
    record["spo2_sustained_abrupt_jump_episode_count"] = float(
        sum(length >= 2 for length in abrupt_runs)
    )

    low = values < 90.0
    record["spo2_below_90_duration_minutes"] = float(low.sum() * 15.0)
    record["spo2_below_90_deficit_auc"] = float(np.maximum(90.0 - values, 0.0).sum() * 15.0)
    episodes: list[int] = []
    current = 0
    previous_bin: int | None = None
    for bin_index, is_low in zip(bins, low):
        consecutive = previous_bin is not None and int(bin_index) == previous_bin + 1
        if is_low and (current == 0 or consecutive):
            current += 1
        elif is_low:
            if current:
                episodes.append(current)
            current = 1
        elif current:
            episodes.append(current)
            current = 0
        previous_bin = int(bin_index)
    if current:
        episodes.append(current)
    record["spo2_desaturation_episode_count"] = float(len(episodes))
    record["spo2_sustained_desaturation_episode_count"] = float(sum(length >= 2 for length in episodes))
    record["spo2_longest_desaturation_minutes"] = float(max(episodes, default=0) * 15.0)
    implausible_fraction = record["spo2_implausible_count"] / max(record["spo2_measurement_count"], 1.0)
    record["spo2_signal_quality_proxy_score"] = float(
        record["spo2_missing_bin_count"] / OBSERVATION_BINS
        + implausible_fraction
        + (1.0 if record.get("spo2_longest_gap_minutes", 0) > 45 else 0.0)
    )
    if np.isfinite(record["spo2_rmssd"]):
        record["spo2_dynamics_proxy_score"] = float(
            record["spo2_rmssd"] / 2.0
            + (
                record["spo2_abrupt_jump_fraction"]
                if np.isfinite(record["spo2_abrupt_jump_fraction"])
                else 0.0
            )
            + min(record["spo2_drop_3_count"], 4.0) / 4.0
        )
        record["spo2_instability_proxy_score"] = float(
            record["spo2_rmssd"] / 2.0
            + (record["spo2_abrupt_jump_fraction"] if np.isfinite(record["spo2_abrupt_jump_fraction"]) else 0.0)
            + record["spo2_below_90_fraction"]
            + min(record["spo2_desaturation_episode_count"], 4.0) / 4.0
        )
    if record["spo2_dynamics_eligible_flag"] != 1.0:
        # The frozen protocol requires >=3 valid bins and >=2 transitions.
        # Absolute oxygenation burden remains descriptive, but two-point
        # differences must not masquerade as an eligible instability signal.
        for feature in (
            "spo2_sd",
            "spo2_iqr",
            "spo2_mad",
            "spo2_range",
            "spo2_rmssd",
            "spo2_slope_per_hr",
            "spo2_abrupt_jump_count",
            "spo2_abrupt_jump_rate_per_hr",
            "spo2_abrupt_jump_fraction",
            "spo2_sustained_abrupt_jump_episode_count",
            "spo2_jump_3_count",
            "spo2_jump_5_count",
            "spo2_drop_3_count",
            "spo2_drop_5_count",
            "spo2_dynamics_proxy_score",
            "spo2_instability_proxy_score",
        ):
            record[feature] = math.nan
    return record


def compute_spo2_features(
    events_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute observation-window SpO2 granularity features by stay."""
    keys = _stay_index_from(stay_index if stay_index is not None else events_df)
    obs = _observation_events(events_df)
    if "concept" not in obs.columns or obs.empty:
        return _merge_default_feature_frame(keys, SPO2_SIGNAL_COLUMNS)

    spo2 = obs.loc[obs["concept"].astype(str).str.lower().eq("spo2")].copy()
    if spo2.empty:
        return _merge_default_feature_frame(keys, SPO2_SIGNAL_COLUMNS)

    if "dataset" not in spo2.columns:
        spo2["dataset"] = "unknown"
    rows: list[dict[str, Any]] = []
    for (dataset, stay_id), group in spo2.groupby(["dataset", "stay_id"], sort=False):
        row = {"dataset": dataset, "stay_id": stay_id}
        row.update(_summarize_spo2_group(group))
        rows.append(row)
    features = pd.DataFrame(rows)
    result = keys.merge(features, on=["dataset", "stay_id"], how="left")
    count_cols = [
        "spo2_measurement_count",
        "spo2_unique_timestamp_count",
        "spo2_plausible_count",
        "spo2_implausible_count",
        "spo2_observed_bin_count",
        "spo2_valid_bin_count",
        "spo2_missing_bin_count",
        "spo2_sampling_density_per_hr",
        "spo2_dynamics_eligible_flag",
    ]
    for col in set(count_cols) & set(result.columns):
        result[col] = result[col].fillna(0)
    return result


def _merge_default_feature_frame(keys: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    result = keys.copy()
    for column in value_columns:
        result[column] = np.nan
    for column in [
        "spo2_measurement_count",
        "spo2_unique_timestamp_count",
        "spo2_plausible_count",
        "spo2_implausible_count",
        "spo2_observed_bin_count",
        "spo2_valid_bin_count",
        "spo2_sampling_density_per_hr",
        "spo2_dynamics_eligible_flag",
    ]:
        if column in result.columns:
            result[column] = 0.0
    if "spo2_missing_bin_count" in result.columns:
        result["spo2_missing_bin_count"] = float(OBSERVATION_BINS)
    return result


def compute_respiratory_support_features(
    events_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract available respiratory support controls from event text."""
    keys = _stay_index_from(stay_index if stay_index is not None else events_df)
    obs = _observation_events(events_df)
    if obs.empty:
        return _zero_feature_frame(keys, RESPIRATORY_SUPPORT_COLUMNS)
    if "dataset" not in obs.columns:
        obs["dataset"] = "unknown"

    text = _event_text(obs)
    obs = obs.assign(_text=text)
    if "value_numeric" in obs.columns:
        value = pd.to_numeric(obs["value_numeric"], errors="coerce")
    else:
        value = pd.Series(np.nan, index=obs.index)
    fio2_mask = obs.get("concept", pd.Series("", index=obs.index)).astype(str).str.lower().eq("fio2") | obs["_text"].str.contains(
        r"fio2|fraction inspired oxygen|inspired o2|oxygen concentration",
        regex=True,
        na=False,
    )
    normalized_fio2 = value.where(~value.between(0.20, 1.0), value * 100.0)
    fio2_rows = obs.loc[fio2_mask & normalized_fio2.between(21, 100)].assign(
        _fio2=normalized_fio2.loc[fio2_mask & normalized_fio2.between(21, 100)]
    )
    fio2_summary = fio2_rows.groupby(["dataset", "stay_id"], sort=False)["_fio2"].agg(
        fio2_measurement_count="count",
        fio2_max="max",
        fio2_mean="mean",
    )

    flag_patterns = {
        "intubation_flag": r"intubat|endotracheal|\bett\b",
        "mechanical_ventilation_flag": r"mechanical ventil|ventilator|assist control|simv|volume control|pressure control",
        "noninvasive_ventilation_flag": r"bipap|cpap|\bniv\b|noninvasive",
        "nasal_cannula_flag": r"nasal cannula|\bnc\b|high flow nasal|hfnc",
        "oxygen_device_flag": r"oxygen device|o2 flow|oxygen flow|face mask|nonrebreather|nasal cannula|hfnc",
    }
    value_text = obs.get("value_text", pd.Series("", index=obs.index)).fillna("").astype(str).str.lower()
    inactive = value_text.str.contains(
        r"\broom air\b|\bnone\b|\boff\b|discontinu|standby|not in use|no oxygen|spontaneous",
        regex=True,
        na=False,
    )
    flag_frames: list[pd.DataFrame] = []
    for column, pattern in flag_patterns.items():
        flagged = obs.loc[
            obs["_text"].str.contains(pattern, regex=True, na=False) & ~inactive
        ]
        if flagged.empty:
            continue
        flag_frames.append(
            flagged[["dataset", "stay_id"]].drop_duplicates().assign(**{column: 1})
        )

    result = _zero_feature_frame(keys, RESPIRATORY_SUPPORT_COLUMNS)
    if not fio2_summary.empty:
        result = result.merge(
            fio2_summary.reset_index(),
            on=["dataset", "stay_id"],
            how="left",
            suffixes=("", "_new"),
        )
        for base in ("fio2_measurement_count", "fio2_max", "fio2_mean"):
            new_col = f"{base}_new"
            if new_col in result.columns:
                result[base] = result[new_col].combine_first(result[base])
                result = result.drop(columns=[new_col])
    for flags in flag_frames:
        result = result.merge(flags, on=["dataset", "stay_id"], how="left", suffixes=("", "_new"))
        for column in flag_patterns:
            new_col = f"{column}_new"
            if new_col in result:
                result[column] = result[new_col].combine_first(result[column])
                result = result.drop(columns=[new_col])
    support_cols = [
        "intubation_flag",
        "mechanical_ventilation_flag",
        "noninvasive_ventilation_flag",
        "nasal_cannula_flag",
        "oxygen_device_flag",
    ]
    result["fio2_above_room_air_flag"] = (
        pd.to_numeric(result["fio2_max"], errors="coerce").gt(21.0)
    ).astype(int)
    result["resp_support_any_flag"] = (
        result[[*support_cols, "fio2_above_room_air_flag"]].fillna(0).sum(axis=1) > 0
    ).astype(int)
    result["fio2_measurement_count"] = result["fio2_measurement_count"].fillna(0)
    for col in support_cols:
        result[col] = result[col].fillna(0).astype(int)
    return result[["dataset", "stay_id", *RESPIRATORY_SUPPORT_COLUMNS]]


def compute_rrt_features(
    events_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract baseline dialysis/RRT controls when available."""
    keys = _stay_index_from(stay_index if stay_index is not None else events_df)
    obs = _observation_events(events_df)
    result = _zero_feature_frame(keys, RRT_COLUMNS)
    if obs.empty:
        return result[["dataset", "stay_id", *RRT_COLUMNS]]
    if "dataset" not in obs:
        obs["dataset"] = "unknown"
    text = _event_text(obs)
    rrt_mask = text.str.contains(
        r"dialysis|renal replacement|\brrt\b|crrt|cvvh|cvvhd|hemofiltration|hemodialysis",
        regex=True,
        na=False,
    )
    dialysis_mask = text.str.contains(r"dialysis|hemodialysis", regex=True, na=False)
    rrt_rows = obs.loc[rrt_mask, ["dataset", "stay_id"]].drop_duplicates()
    dialysis_rows = obs.loc[dialysis_mask, ["dataset", "stay_id"]].drop_duplicates()
    if not rrt_rows.empty:
        result = result.merge(rrt_rows.assign(baseline_rrt_flag=1), on=["dataset", "stay_id"], how="left", suffixes=("", "_new"))
        result["baseline_rrt_flag"] = result["baseline_rrt_flag_new"].combine_first(result["baseline_rrt_flag"])
        result = result.drop(columns=["baseline_rrt_flag_new"])
    if not dialysis_rows.empty:
        result = result.merge(dialysis_rows.assign(baseline_dialysis_flag=1), on=["dataset", "stay_id"], how="left", suffixes=("", "_new"))
        result["baseline_dialysis_flag"] = result["baseline_dialysis_flag_new"].combine_first(result["baseline_dialysis_flag"])
        result = result.drop(columns=["baseline_dialysis_flag_new"])
    result["baseline_rrt_flag"] = result["baseline_rrt_flag"].fillna(0).astype(int)
    result["baseline_dialysis_flag"] = result["baseline_dialysis_flag"].fillna(0).astype(int)
    result["rrt_or_dialysis_flag"] = (
        result[["baseline_rrt_flag", "baseline_dialysis_flag"]].sum(axis=1) > 0
    ).astype(int)
    return result[["dataset", "stay_id", *RRT_COLUMNS]]


def _zero_feature_frame(keys: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    result = keys.copy()
    for column in value_columns:
        result[column] = 0
    for column in ("fio2_max", "fio2_mean"):
        if column in result.columns:
            result[column] = np.nan
    return result


def compute_horizon_outcomes(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZON_HOURS,
) -> pd.DataFrame:
    """Create legacy exploratory labels at hours from ICU admission.

    The authoritative hypothesis endpoints are derived separately on the
    post-landmark 12/24-hour clock. These 48--168-hour columns retain their
    historical admission-clock meaning and are never selected when protocol
    v2.2 endpoints are present.
    """
    keys = _stay_index_from(stay_index if stay_index is not None else cohort_df, cohort_df, events_df)
    result = keys.copy()
    for horizon in horizons:
        for concept in ("pressor", "mcs", "death"):
            result[f"{concept}_{horizon}h_flag"] = 0
        result[f"escalation_{horizon}h_flag"] = 0
        result[f"mcs_or_death_{horizon}h_flag"] = 0

    events = events_df.copy()
    if events.empty:
        return _add_death_from_cohort(result, cohort_df, horizons)
    if "dataset" not in events:
        events["dataset"] = "unknown"
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    event_text = _event_text(events)
    concept = events.get("concept", pd.Series("", index=events.index)).fillna("").astype(str).str.lower()

    for horizon in horizons:
        upper = horizon * 60
        window_mask = events["offset_minutes"].gt(LANDMARK_MINUTES) & events["offset_minutes"].le(upper)
        pressor_mask = window_mask & concept.eq("pressor")
        mcs_mask = window_mask & (
            concept.eq("mcs")
            | concept.str.startswith("mcs_context_")
            | event_text.str.contains(
                r"iabp|impella|ecmo|lvad|rvad|bivad|ventricular assist|"
                r"mechanical circulatory|\bmcs\b",
                regex=True,
                na=False,
            )
        )
        death_mask = window_mask & concept.eq("death")
        result = _mark_flag(result, events.loc[pressor_mask], f"pressor_{horizon}h_flag")
        result = _mark_flag(result, events.loc[mcs_mask], f"mcs_{horizon}h_flag")
        result = _mark_flag(result, events.loc[death_mask], f"death_{horizon}h_flag")

    result = _add_death_from_cohort(result, cohort_df, horizons)
    for horizon in horizons:
        result[f"escalation_{horizon}h_flag"] = (
            result[[f"pressor_{horizon}h_flag", f"mcs_{horizon}h_flag"]].sum(axis=1) > 0
        ).astype(int)
        result[f"mcs_or_death_{horizon}h_flag"] = (
            result[[f"mcs_{horizon}h_flag", f"death_{horizon}h_flag"]].sum(axis=1) > 0
        ).astype(int)
    return result


def _mark_flag(target: pd.DataFrame, source: pd.DataFrame, column: str) -> pd.DataFrame:
    if source.empty:
        return target
    keys = source[["dataset", "stay_id"]].drop_duplicates().assign(_flag=1)
    merged = target.merge(keys, on=["dataset", "stay_id"], how="left")
    merged[column] = np.maximum(merged[column].fillna(0).astype(int), merged["_flag"].fillna(0).astype(int))
    return merged.drop(columns=["_flag"])


def _add_death_from_cohort(
    result: pd.DataFrame,
    cohort_df: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if cohort_df.empty or "death_offset_minutes" not in cohort_df.columns:
        return result
    cohort = cohort_df.copy()
    if "dataset" not in cohort:
        cohort["dataset"] = "unknown"
    cohort["death_offset_minutes"] = pd.to_numeric(cohort["death_offset_minutes"], errors="coerce")
    for horizon in horizons:
        upper = horizon * 60
        death_rows = cohort.loc[
            cohort["death_offset_minutes"].gt(LANDMARK_MINUTES)
            & cohort["death_offset_minutes"].le(upper),
            ["dataset", "stay_id"],
        ]
        result = _mark_flag(result, death_rows, f"death_{horizon}h_flag")
    return result


def assemble_spo2_analysis_frame(
    *,
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge baseline features, outcomes, and SpO2/therapy controls."""
    eligible_cohort = cohort
    if "excluded_before_landmark_flag" in cohort:
        eligible_cohort = cohort.loc[
            pd.to_numeric(
                cohort["excluded_before_landmark_flag"], errors="coerce"
            )
            .fillna(0)
            .eq(0)
        ]
    # Cohort eligibility is authoritative.  Taking a union with feature/label
    # keys would silently reintroduce excluded stays whenever upstream files
    # still contain them.  Only fall back to artifact keys for legacy fixtures
    # that do not supply a cohort index.
    stay_index = _stay_index_from(eligible_cohort)
    if stay_index.empty:
        stay_index = _stay_index_from(features, labels)
    spo2 = compute_spo2_features(events, stay_index)
    resp = compute_respiratory_support_features(events, stay_index)
    rrt = compute_rrt_features(events, stay_index)
    early_outcomes = compute_early_decompensation_outcomes(
        events,
        cohort,
        stay_index,
        horizons=POST_LANDMARK_HORIZONS_HOURS,
    )
    exploratory_long_horizons = compute_horizon_outcomes(events, cohort, stay_index)

    feature_frame = features.copy()
    if "dataset" not in feature_frame:
        feature_frame["dataset"] = (
            stay_index["dataset"].iloc[0] if not stay_index.empty else "unknown"
        )
    # Preserve every landmark-eligible cohort/label key even if a feature
    # builder regresses and omits a row. Missing predictors must appear in
    # cohort flow and be excluded explicitly, never vanish during assembly.
    frame = stay_index.merge(
        feature_frame,
        on=["dataset", "stay_id"],
        how="left",
        validate="one_to_one",
    )
    label_cols = ["dataset", "stay_id"] + [
        col for col in labels.columns if col not in features.columns and col not in ("dataset", "stay_id")
    ]
    frame = frame.merge(labels[label_cols], on=["dataset", "stay_id"], how="left")
    for extra_name, extra in (
        ("spo2", spo2),
        ("respiratory", resp),
        ("rrt", rrt),
        ("protocol_early_outcomes", early_outcomes),
        ("exploratory_long_horizons", exploratory_long_horizons),
    ):
        duplicate_non_keys = [
            column
            for column in extra.columns
            if column in frame.columns and column not in {"dataset", "stay_id"}
        ]
        if duplicate_non_keys:
            if extra_name == "protocol_early_outcomes":
                # Protocol v2.2 is authoritative. In particular, the legacy
                # label table also contains ``mcs_24h_flag`` but lacks the
                # incident-risk-set and censoring rules used here.
                frame = frame.drop(columns=duplicate_non_keys)
            else:
                extra = extra.drop(columns=duplicate_non_keys)
        frame = frame.merge(extra, on=["dataset", "stay_id"], how="left")

    cohort_context_candidates = {
        "person_id",
        "age",
        "is_male",
        "cohort_hf_flag",
        "shock_icd_flag",
        "cardiomyopathy_flag",
        "acute_mi_flag",
        "early_icu_flag",
        "race",
        "ethnicity",
        "race_ethnicity",
        "hospital_id",
        "ward_id",
        "icu_type",
        "unit_admit_source",
        "admit_year",
        "sex_unknown_flag",
        "baseline_vasoactive_flag",
        "baseline_vasoactive_method",
        "baseline_mcs_flag",
        "baseline_intervention_flag",
        "first_stay_per_person_flag",
        "followup_end_offset_minutes",
        "unit_discharge_offset_minutes",
        "hospital_discharge_offset_minutes",
        "death_offset_minutes",
        "admission_weight_kg",
        "mcs_source_available",
        "pressor_source_available",
        "vis_source_available",
        "urine_output_source_available",
        "respiratory_source_available",
        "rrt_source_available",
    }
    cohort_context = [col for col in cohort.columns if col in cohort_context_candidates]
    if cohort_context:
        cohort_context_frame = cohort[
            ["dataset", "stay_id", *cohort_context]
        ].drop_duplicates(["dataset", "stay_id"])
        cohort_context_frame = cohort_context_frame.rename(
            columns={
                column: f"__cohort_{column}" for column in cohort_context
            }
        )
        frame = frame.merge(
            cohort_context_frame,
            on=["dataset", "stay_id"],
            how="left",
            validate="many_to_one",
        )
        for column in cohort_context:
            source = f"__cohort_{column}"
            if column in frame:
                frame[column] = frame[source].combine_first(frame[column])
                frame = frame.drop(columns=source)
            else:
                frame = frame.rename(columns={source: column})
    if "respiratory_source_available" in frame:
        respiratory_available = pd.to_numeric(
            frame["respiratory_source_available"], errors="coerce"
        ).fillna(0).gt(0)
        respiratory_columns = [
            column for column in RESPIRATORY_SUPPORT_COLUMNS if column in frame
        ]
        frame.loc[~respiratory_available, respiratory_columns] = np.nan
    if "rrt_source_available" in frame:
        rrt_available = pd.to_numeric(
            frame["rrt_source_available"], errors="coerce"
        ).fillna(0).gt(0)
        rrt_columns = [column for column in RRT_COLUMNS if column in frame]
        frame.loc[~rrt_available, rrt_columns] = np.nan
    # Lab measurements timestamped exactly at minute 240 are available at the
    # landmark and are part of the frozen outcome baseline.  Use the same
    # inclusive baseline for adjustment variables so the predictor and outcome
    # definitions cannot disagree at the boundary.
    for concept in ("lactate", "creatinine", "bilirubin_total", "ph"):
        inclusive = f"baseline_{concept}_outcome"
        target = f"baseline_{concept}"
        if inclusive in frame:
            if target in frame:
                frame[target] = pd.to_numeric(
                    frame[inclusive], errors="coerce"
                ).combine_first(pd.to_numeric(frame[target], errors="coerce"))
            else:
                frame[target] = pd.to_numeric(frame[inclusive], errors="coerce")
    race_cols = [col for col in cohort_context if col.lower() in {"race", "ethnicity", "race_ethnicity"}]

    def positive_rows(column: str) -> int:
        source = (
            frame[column]
            if column in frame
            else pd.Series(0.0, index=frame.index, dtype=float)
        )
        return int(pd.to_numeric(source, errors="coerce").fillna(0).gt(0).sum())

    availability = {
        "n_rows": int(len(frame)),
        "spo2_rows_with_measurements": positive_rows("spo2_plausible_count"),
        "fio2_rows_with_measurements": positive_rows("fio2_measurement_count"),
        "respiratory_support_rows": positive_rows("resp_support_any_flag"),
        "rrt_or_dialysis_rows": positive_rows("rrt_or_dialysis_flag"),
        "race_ethnicity_columns": race_cols,
        "race_ethnicity_available": bool(race_cols),
        "person_id_available": bool("person_id" in frame and frame["person_id"].notna().any()),
        "outcome_clock": "hours_after_4h_landmark",
        "signal_quality_note": (
            "Explicit pulse-ox probe quality flags are used only if present in source events; "
            "otherwise sampling gaps, implausible values, and abrupt jumps are proxy measures."
        ),
    }
    return frame, availability




def _filter_measured_spo2_rows(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with at least one plausible SpO2 measurement."""
    if "spo2_plausible_count" not in analysis_df.columns:
        return analysis_df.copy()
    counts = pd.to_numeric(analysis_df["spo2_plausible_count"], errors="coerce").fillna(0)
    return analysis_df.loc[counts.gt(0)].copy()


def _resolve_cv_group_key(analysis_df: pd.DataFrame) -> tuple[str | None, str]:
    """Return (group column, cv_scope label) for grouped CV."""
    if "person_id" in analysis_df.columns and analysis_df["person_id"].notna().any():
        return "person_id", "internal_grouped_cv_by_person_id"
    if "stay_id" in analysis_df.columns:
        return "stay_id", "internal_grouped_cv_by_stay_id"
    return None, "internal_row_cv_with_group_limitation"


def _get_grouped_cv_splitter(
    y: pd.Series,
    groups: pd.Series | None,
    *,
    n_splits: int = 5,
):
    """Build grouped CV splitter when possible."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

    if groups is not None and groups.nunique() >= n_splits:
        try:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42), "StratifiedGroupKFold"
        except Exception:
            return GroupKFold(n_splits=n_splits), "GroupKFold"
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), "StratifiedKFold_row_level"


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute ECE from out-of-fold predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return math.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.empty(m, dtype=float)
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        adj = min(prev, float(p_values[idx]) * m / rank)
        ranked[idx] = adj
        prev = adj
    return ranked.tolist()


def _fragility_label(n_events: int, *, min_events: int = 20) -> str:
    if n_events < 5:
        return "very_fragile_lt5_events"
    if n_events < min_events:
        return "fragile_lt20_events"
    return "moderate_event_count"


def _observed_dataset_count(frame: pd.DataFrame, outcome: str) -> int:
    """Count sources with an actually observed binary endpoint."""
    if outcome not in frame:
        return 0
    observed = pd.to_numeric(frame[outcome], errors="coerce").isin([0, 1])
    if not observed.any():
        return 0
    if "dataset" not in frame:
        return 1
    return int(frame.loc[observed, "dataset"].dropna().astype(str).nunique())


def _mask_single_source_pooled_outcomes(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """Mask endpoints that cannot support cross-dataset pooled analysis."""
    pooled = frame.copy()
    unsupported = {
        outcome
        for outcome in _available_outcomes(frame)
        if _observed_dataset_count(frame, outcome) < 2
    }
    for outcome in unsupported:
        pooled[outcome] = math.nan
    return pooled, unsupported


def build_descriptive_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SpO2 signals by prespecified outcome and dataset."""
    metric_cols = [
        col for col in [
            "spo2_min",
            "spo2_mean",
            "spo2_sd",
            "spo2_rmssd",
            "spo2_abrupt_jump_fraction",
            "spo2_drop_3_count",
            "spo2_below_90_fraction",
            "spo2_sampling_density_per_hr",
            "spo2_missing_bin_count",
            "spo2_longest_gap_minutes",
            "spo2_dynamics_proxy_score",
            "spo2_instability_proxy_score",
        ] if col in analysis_df.columns
    ]
    group_cols = _available_outcomes(analysis_df)
    if "dataset" in analysis_df:
        scopes = [
            (str(dataset), group.copy())
            for dataset, group in analysis_df.groupby(
                analysis_df["dataset"].astype(str), sort=True
            )
        ]
        if len(scopes) > 1:
            scopes.append(("pooled_secondary", analysis_df.copy()))
    else:
        scopes = [("unknown", analysis_df.copy())]
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in scopes:
        for group_col in group_cols:
            if (
                analysis_scope == "pooled_secondary"
                and _observed_dataset_count(scope_frame, group_col) < 2
            ):
                continue
            y = pd.to_numeric(scope_frame[group_col], errors="coerce")
            for value in (0, 1):
                group = scope_frame.loc[y.eq(value)]
                if group.empty:
                    continue
                row = {
                    "analysis_scope": analysis_scope,
                    "group_column": group_col,
                    "group_value": value,
                    "n": int(len(group)),
                }
                for metric in metric_cols:
                    values = pd.to_numeric(group[metric], errors="coerce")
                    row[f"{metric}_n"] = int(values.notna().sum())
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_median"] = float(values.median())
                rows.append(row)
    return pd.DataFrame(rows)


def build_lactate_negative_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SpO2-outcome patterns when baseline lactate is below 2."""
    if "baseline_lactate" not in analysis_df.columns:
        return pd.DataFrame([{"status": "skipped", "reason": "baseline_lactate unavailable"}])
    subset = _filter_measured_spo2_rows(
        analysis_df.loc[pd.to_numeric(analysis_df["baseline_lactate"], errors="coerce") < 2].copy()
    )
    outcomes = _available_outcomes(subset)
    if "dataset" in subset:
        scopes = [
            (str(dataset), group.copy())
            for dataset, group in subset.groupby(
                subset["dataset"].astype(str), sort=True
            )
        ]
        if len(scopes) > 1:
            scopes.append(("pooled_secondary", subset.copy()))
    else:
        scopes = [("unknown", subset)]
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in scopes:
        for outcome in outcomes:
            if (
                analysis_scope == "pooled_secondary"
                and _observed_dataset_count(scope_frame, outcome) < 2
            ):
                continue
            y = pd.to_numeric(scope_frame[outcome], errors="coerce")
            events = int(y.sum()) if y.notna().any() else 0
            rows.append(
                {
                    "analysis_scope": analysis_scope,
                    "outcome": outcome,
                    "n_lactate_negative": int(y.notna().sum()),
                    "n_events": events,
                    "event_rate": float(y.mean()) if y.notna().any() else math.nan,
                    "fragility_label": _fragility_label(events, min_events=MIN_OR_EVENTS),
                    "analysis_note": "descriptive_subgroup_only_not_network_discovery",
                    "mean_spo2_min": float(pd.to_numeric(scope_frame.get("spo2_min"), errors="coerce").mean()),
                    "mean_spo2_rmssd": float(pd.to_numeric(scope_frame.get("spo2_rmssd"), errors="coerce").mean()),
                    "mean_spo2_below_90_fraction": float(
                        pd.to_numeric(scope_frame.get("spo2_below_90_fraction"), errors="coerce").mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_spo2_association_proxy(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Describe unadjusted SpO2/outcome correlations by dataset."""
    outcome_cols = _available_outcomes(analysis_df)
    if "dataset" in analysis_df:
        scopes = [
            (str(dataset), group.copy())
            for dataset, group in analysis_df.groupby(
                analysis_df["dataset"].astype(str), sort=True
            )
        ]
        if len(scopes) > 1:
            scopes.append(("pooled_secondary", analysis_df.copy()))
    else:
        scopes = [("unknown", analysis_df)]
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in scopes:
        for feature in [col for col in SPO2_SIGNAL_COLUMNS if col in scope_frame.columns]:
            x = pd.to_numeric(scope_frame[feature], errors="coerce")
            for outcome in outcome_cols:
                if (
                    analysis_scope == "pooled_secondary"
                    and _observed_dataset_count(scope_frame, outcome) < 2
                ):
                    continue
                y = pd.to_numeric(scope_frame[outcome], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 3 or y.loc[valid].nunique() < 2:
                    corr = math.nan
                else:
                    corr = float(np.corrcoef(x.loc[valid], y.loc[valid])[0, 1])
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "feature": feature,
                        "outcome": outcome,
                        "n": int(valid.sum()),
                        "pearson_correlation": corr,
                        "absolute_correlation": abs(corr) if not math.isnan(corr) else math.nan,
                        "analysis_note": "unadjusted_descriptive_not_inference",
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["analysis_scope", "outcome", "absolute_correlation"],
        ascending=[True, True, False],
        na_position="last",
    )


def _available_outcomes(analysis_df: pd.DataFrame) -> list[str]:
    available = [col for col in PRIMARY_OUTCOME_COLUMNS if col in analysis_df.columns]
    if available:
        return available
    return [
        col
        for col in ("target", "mcs_or_death_168h_flag", "death_168h_flag")
        if col in analysis_df.columns
    ]


def _available_spo2_model_features(analysis_df: pd.DataFrame) -> list[str]:
    return [col for col in PRIMARY_SPO2_MODEL_FEATURES if col in analysis_df.columns]


def _available_controls(analysis_df: pd.DataFrame) -> list[str]:
    """Return the post-result parsimonious confounder set.

    High-cardinality site/year/unit fields and long lists of partially observed
    physiologic covariates made the earlier exploratory GLMs grossly
    overparameterized.  The broad candidate list remains in Table 1 and the
    predictive pipeline; focal inferential GLMs use this clinically compact set
    and still fail closed below five events per fitted parameter.
    """
    return [
        col for col in PARSIMONIOUS_CONTROL_CANDIDATES if col in analysis_df.columns
    ]


def _available_spo2_variability_features(analysis_df: pd.DataFrame) -> list[str]:
    return [col for col in SPO2_VARIABILITY_FEATURES if col in analysis_df.columns]


def _prepare_design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    x = frame[columns].copy()
    categorical = {
        col
        for col in columns
        if not pd.api.types.is_numeric_dtype(x[col])
    }
    for col in columns:
        if col in categorical:
            x[col] = x[col].fillna("missing").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            median = float(x[col].median()) if x[col].notna().any() else 0.0
            x[col] = x[col].fillna(median)
    x = pd.get_dummies(x, columns=sorted(categorical), dummy_na=False, drop_first=True, dtype=float)
    if not x.empty:
        nunique = x.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            x = x.drop(columns=constant_cols)
    return x


def _retain_full_rank_columns(
    matrix: pd.DataFrame,
    *,
    protected: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop only redundant adjustment columns while retaining focal terms."""
    protected = [column for column in protected if column in matrix]
    remaining = [column for column in matrix if column not in protected]
    base = np.column_stack(
        [
            np.ones(len(matrix), dtype=float),
            *[matrix[column].to_numpy(float) for column in protected],
        ]
    )
    if np.linalg.matrix_rank(base) < base.shape[1]:
        raise ValueError("protected focal terms are rank deficient")
    if not remaining:
        return matrix[protected].copy(), []

    adjustment = matrix[remaining].to_numpy(float)
    residual = adjustment - base @ np.linalg.lstsq(
        base, adjustment, rcond=None
    )[0]
    try:
        from scipy.linalg import qr

        _, upper, pivot = qr(residual, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(upper))
        scale = float(diagonal.max()) if diagonal.size else 0.0
        tolerance = max(residual.shape) * np.finfo(float).eps * scale
        rank = int((diagonal > tolerance).sum())
        selected = {remaining[int(index)] for index in pivot[:rank]}
    except Exception:  # pragma: no cover - dependency-light fallback
        selected = set()
        current = base
        current_rank = int(np.linalg.matrix_rank(current))
        for column in remaining:
            candidate = np.column_stack([current, matrix[column].to_numpy(float)])
            candidate_rank = int(np.linalg.matrix_rank(candidate))
            if candidate_rank > current_rank:
                selected.add(column)
                current = candidate
                current_rank = candidate_rank
    keep = [*protected, *[column for column in remaining if column in selected]]
    dropped = [column for column in remaining if column not in selected]
    return matrix[keep].copy(), dropped


def _patient_cluster_key(frame: pd.DataFrame) -> pd.Series:
    """Build dataset-qualified patient clusters with a stay-level fallback."""
    dataset = frame.get(
        "dataset", pd.Series("unknown", index=frame.index)
    ).fillna("unknown").astype(str)
    if "stay_id" in frame:
        stay = frame["stay_id"].astype("string")
    else:
        stay = pd.Series(
            [f"row:{index}" for index in range(len(frame))],
            index=frame.index,
            dtype="string",
        )
    if "person_id" in frame and frame["person_id"].notna().any():
        person = frame["person_id"].astype("string")
        person = person.where(
            person.notna() & person.str.strip().ne(""), "stay:" + stay
        )
        return dataset + ":person:" + person.astype(str)
    return dataset + ":stay:" + stay.astype(str)


class _NonEstimableInferenceError(RuntimeError):
    """Raised when a fitted GLM cannot support finite inference."""


def _fit_binomial_glm_inference(
    sm: Any,
    y: pd.Series,
    design: pd.DataFrame,
    cluster: pd.Series,
    *,
    required_terms: list[str] | tuple[str, ...] | None = None,
) -> Any:
    """Fit a binomial GLM and require finite inference for focal terms.

    Sparse nuisance categories can have undefined robust standard errors even
    when the prespecified focal coefficient and its robust interval are finite.
    Such nuisance-only failures are retained as explicit fit metadata rather
    than incorrectly discarding an otherwise estimable focal association.
    """
    glm = sm.GLM(y, design, family=sm.families.Binomial())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if cluster.nunique() > 1:
                fit = glm.fit(cov_type="cluster", cov_kwds={"groups": cluster})
            else:
                fit = glm.fit(cov_type="HC0")
        except Exception as exc:
            detail = str(exc)
            lower = detail.lower()
            if any(
                marker in lower
                for marker in (
                    "perfect separation",
                    "singular matrix",
                    "svd did not converge",
                    "hessian inversion",
                )
            ):
                raise _NonEstimableInferenceError(
                    f"non-estimable binomial fit: {detail}"
                ) from exc
            raise
    unstable_messages: list[str] = []
    for warning in caught:
        message = str(warning.message)
        lower = message.lower()
        category = warning.category.__name__.lower()
        if (
            "perfectseparation" in category
            or "perfect separation" in lower
            or "divide by zero" in lower
            or "invalid value" in lower
            or "overflow" in lower
            or "failed to converge" in lower
        ):
            unstable_messages.append(message)
    if unstable_messages:
        detail = "; ".join(dict.fromkeys(unstable_messages))[:500]
        raise _NonEstimableInferenceError(f"unstable binomial fit: {detail}")
    if not bool(getattr(fit, "converged", True)):
        raise _NonEstimableInferenceError("binomial fit did not converge")
    confidence = fit.conf_int()
    available_terms = list(getattr(fit.params, "index", design.columns))
    if required_terms is None:
        checked_terms = available_terms
    else:
        checked_terms = list(dict.fromkeys(required_terms))
        missing_terms = [term for term in checked_terms if term not in available_terms]
        if missing_terms:
            raise _NonEstimableInferenceError(
                "binomial fit omitted required focal terms: "
                + ",".join(missing_terms)
            )
    inference_arrays = (
        np.asarray(fit.params.loc[checked_terms], dtype=float),
        np.asarray(fit.bse.loc[checked_terms], dtype=float),
        np.asarray(fit.pvalues.loc[checked_terms], dtype=float),
        np.asarray(confidence.loc[checked_terms], dtype=float),
    )
    if not all(np.isfinite(values).all() for values in inference_arrays):
        raise _NonEstimableInferenceError(
            "binomial fit returned non-finite required-term inference"
        )
    all_inference = pd.DataFrame(
        {
            "parameter": pd.to_numeric(fit.params, errors="coerce"),
            "standard_error": pd.to_numeric(fit.bse, errors="coerce"),
            "p_value": pd.to_numeric(fit.pvalues, errors="coerce"),
            "ci_low": pd.to_numeric(confidence.iloc[:, 0], errors="coerce"),
            "ci_high": pd.to_numeric(confidence.iloc[:, 1], errors="coerce"),
        }
    )
    nonfinite_mask = ~np.isfinite(all_inference.to_numpy(float)).all(axis=1)
    required_set = set(checked_terms)
    fit._physiograph_nonfinite_nuisance_terms = tuple(
        str(term)
        for term in all_inference.index[nonfinite_mask]
        if str(term) not in required_set
    )
    return fit


def _finite_odds_ratio_triplet(
    coef: float,
    ci_low: float,
    ci_high: float,
) -> tuple[float, float, float]:
    """Exponentiate log-odds inference only when the OR scale is finite."""
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.exp(np.asarray([coef, ci_low, ci_high], dtype=float))
    if not np.isfinite(values).all():
        raise _NonEstimableInferenceError(
            "binomial fit produced non-finite odds-ratio inference"
        )
    return tuple(float(value) for value in values)


def fit_spo2_models(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = MIN_OR_EVENTS,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Fit the prespecified patient-grouped, fold-local incremental models."""
    result = fit_grouped_incremental_models(
        analysis_df,
        min_rows=min_rows,
        min_events=min_events,
        n_splits=n_splits,
    )
    oof_predictions = result.attrs.get("oof_predictions", pd.DataFrame())
    if "n" in result:
        result["n_input_rows"] = int(len(analysis_df))
        measured = _filter_measured_spo2_rows(analysis_df)
        result["n_measured_spo2_rows"] = int(len(measured))
    result.attrs["oof_predictions"] = oof_predictions
    return result


def _fit_spo2_or_pvalue_tables_single(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = MIN_OR_EVENTS,
) -> pd.DataFrame:
    """Estimate per-1SD OR with HC0 robust SE, BH-FDR, and skip reasons."""
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover - environment fallback
        return pd.DataFrame([{"status": "skipped", "reason": f"statsmodels unavailable: {exc}"}])

    measured = _filter_measured_spo2_rows(analysis_df)
    outcomes = _available_outcomes(measured)
    controls = _available_controls(measured)
    oxygenation_and_sampling = [
        column for column in SPO2_OXYGENATION_COVARIATES if column in measured
    ]
    sampling_only = [
        column for column in MISSINGNESS_CONTROL_FEATURES[:3] if column in measured
    ]
    endpoint_family = "spo2_primary_features_adjusted_logistic_or"
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for feature in _available_spo2_model_features(measured):
            is_dynamics = feature in SPO2_VARIABILITY_FEATURES
            adjustments = list(
                dict.fromkeys(
                    [
                        *controls,
                        *(
                            oxygenation_and_sampling
                            if is_dynamics
                            else sampling_only
                        ),
                    ]
                )
            )
            adjustments = [column for column in adjustments if column != feature]
            identity = [
                column
                for column in ("dataset", "person_id", "stay_id")
                if column in measured
                and column not in {outcome, feature, *adjustments}
            ]
            frame = measured[
                list(dict.fromkeys([outcome, feature, *adjustments, *identity]))
            ].copy()
            frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            valid = frame[outcome].notna() & frame[feature].notna()
            if feature in SPO2_VARIABILITY_FEATURES and "spo2_dynamics_eligible_flag" in measured:
                valid &= pd.to_numeric(
                    measured.loc[frame.index, "spo2_dynamics_eligible_flag"], errors="coerce"
                ).eq(1)
            frame = frame.loc[valid].copy()
            n = int(len(frame))
            events = int(frame[outcome].sum()) if n else 0
            non_events = int(n - events)
            if (
                n < min_rows
                or events < min_events
                or non_events < min_events
                or frame[outcome].nunique() < 2
            ):
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "skipped",
                        "n": n,
                        "events": events,
                        "non_events": non_events,
                        "endpoint_family": endpoint_family,
                        "reason": "insufficient rows/events/non-events/classes",
                    }
                )
                continue
            feature_sd = float(frame[feature].std(ddof=0))
            if not np.isfinite(feature_sd) or feature_sd <= 0:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "skipped",
                        "n": n,
                        "events": events,
                        "endpoint_family": endpoint_family,
                        "reason": "feature variance is zero",
                    }
                )
                continue
            feature_mean = float(frame[feature].mean())
            z_feature = f"{feature}_z"
            frame[z_feature] = (frame[feature] - feature_mean) / feature_sd
            columns = [z_feature, *adjustments]
            x = _prepare_design_matrix(frame, columns)
            if z_feature not in x.columns or x.empty:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "skipped",
                        "n": n,
                        "events": events,
                        "endpoint_family": endpoint_family,
                        "reason": "design matrix missing feature",
                    }
                )
                continue
            try:
                x, collinear_dropped = _retain_full_rank_columns(
                    x, protected=[z_feature]
                )
            except ValueError as exc:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "skipped",
                        "n": n,
                        "events": events,
                        "endpoint_family": endpoint_family,
                        "reason": str(exc),
                    }
                )
                continue
            y = frame[outcome].astype(int)
            design = sm.add_constant(x, has_constant="add")
            events_per_parameter = min(events, non_events) / max(
                design.shape[1], 1
            )
            if events_per_parameter < 5:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "underpowered_low_information",
                        "n": n,
                        "events": events,
                        "non_events": non_events,
                        "endpoint_family": endpoint_family,
                        "parameter_count": int(design.shape[1]),
                        "events_per_parameter": events_per_parameter,
                        "reason": "events_per_parameter_below_5_fail_closed",
                    }
                )
                continue
            try:
                cluster = _patient_cluster_key(frame)
                fit = _fit_binomial_glm_inference(
                    sm,
                    y,
                    design,
                    cluster,
                    required_terms=[z_feature],
                )
                nuisance_terms = getattr(
                    fit, "_physiograph_nonfinite_nuisance_terms", ()
                )
                coef = float(fit.params[z_feature])
                ci_low, ci_high = fit.conf_int().loc[z_feature]
                p_val = float(fit.pvalues[z_feature])
                odds_ratio, odds_ci_low, odds_ci_high = _finite_odds_ratio_triplet(
                    coef, float(ci_low), float(ci_high)
                )
            except _NonEstimableInferenceError as exc:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "non_estimable",
                        "n": n,
                        "events": events,
                        "non_events": non_events,
                        "endpoint_family": endpoint_family,
                        "reason": str(exc),
                    }
                )
                continue
            except Exception as exc:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "failed",
                        "n": n,
                        "events": events,
                        "non_events": non_events,
                        "endpoint_family": endpoint_family,
                        "reason": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "model": "adjusted",
                    "status": "fit",
                    "n": n,
                    "events": events,
                    "non_events": non_events,
                    "endpoint_family": endpoint_family,
                    "adjustment_set": (
                        "clinical_absolute_spo2_and_sampling"
                        if is_dynamics
                        else "clinical_and_sampling"
                    ),
                    "adjustment_columns": ",".join(adjustments),
                    "or_per_1sd": odds_ratio,
                    "ci95_low": odds_ci_low,
                    "ci95_high": odds_ci_high,
                    "p_value": p_val,
                    "mean_feature": feature_mean,
                    "sd_feature": feature_sd,
                    "control_count": int(x.shape[1] - 1),
                    "collinear_adjustment_columns_dropped": ",".join(
                        collinear_dropped
                    ),
                    "nonfinite_nuisance_inference_count": len(nuisance_terms),
                    "nonfinite_nuisance_inference_terms": ",".join(
                        nuisance_terms
                    ),
                    "covariance": "patient_cluster_robust" if cluster.nunique() > 1 else "HC0",
                    "events_per_parameter": min(events, n - events) / max(design.shape[1], 1),
                    "information_status": (
                        "adequate_ge10_events_per_parameter"
                        if events_per_parameter >= 10
                        else "fragile_5_to_10_events_per_parameter"
                    ),
                }
            )

    result = pd.DataFrame(rows)
    fit_mask = result["status"].eq("fit") if not result.empty and "status" in result.columns else pd.Series(dtype=bool)
    if fit_mask.any():
        pvals = result.loc[fit_mask, "p_value"].astype(float).tolist()
        adj = _benjamini_hochberg(pvals)
        result.loc[fit_mask, "p_value_adj"] = adj
        result.loc[fit_mask, "global_exploratory_q_value"] = adj
        result.loc[fit_mask, "n_tests"] = int(fit_mask.sum())
        result.loc[fit_mask, "multiplicity_scope"] = endpoint_family
        result["focused_endpoint_q_value"] = np.nan
        for _, indexes in result.loc[fit_mask].groupby("outcome").groups.items():
            result.loc[indexes, "focused_endpoint_q_value"] = _benjamini_hochberg(
                result.loc[indexes, "p_value"].astype(float).tolist()
            )
        result.loc[fit_mask, "focused_multiplicity_scope"] = (
            "features_within_one_clinical_endpoint_post_result_amendment"
        )
    if result.empty:
        return result
    order = [col for col in ("outcome", "p_value", "feature") if col in result.columns]
    if order:
        return result.sort_values(order, na_position="last").reset_index(drop=True)
    return result.reset_index(drop=True)


def fit_spo2_or_pvalue_tables(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = MIN_OR_EVENTS,
) -> pd.DataFrame:
    """Per-dataset adjusted OR tables with a pooled secondary analysis."""
    if "dataset" not in analysis_df:
        result = _fit_spo2_or_pvalue_tables_single(
            analysis_df, min_rows=min_rows, min_events=min_events
        )
        result["analysis_scope"] = "unknown"
        return result
    pieces: list[pd.DataFrame] = []
    datasets = sorted(analysis_df["dataset"].dropna().astype(str).unique())
    for dataset in datasets:
        local = _fit_spo2_or_pvalue_tables_single(
            analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)].copy(),
            min_rows=min_rows,
            min_events=min_events,
        )
        local["analysis_scope"] = dataset
        pieces.append(local)
    if len(datasets) > 1:
        pooled_frame, unsupported = _mask_single_source_pooled_outcomes(
            analysis_df
        )
        pooled = _fit_spo2_or_pvalue_tables_single(
            pooled_frame, min_rows=min_rows, min_events=min_events
        )
        if unsupported and "outcome" in pooled:
            mask = pooled["outcome"].isin(unsupported)
            pooled.loc[mask, "reason"] = (
                "pooled endpoint not observed in at least two datasets"
            )
        pooled["analysis_scope"] = "pooled_secondary"
        pieces.append(pooled)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _fit_spo2_variability_or_tables_single(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = 20,
) -> pd.DataFrame:
    """Estimate variability-specific OR/CI tables with staged adjustment sets."""
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover - environment fallback
        return pd.DataFrame([{"status": "skipped", "reason": f"statsmodels unavailable: {exc}"}])

    outcomes = _available_outcomes(analysis_df)
    variability_features = _available_spo2_variability_features(analysis_df)
    oxygen_covariates = [col for col in SPO2_OXYGENATION_COVARIATES if col in analysis_df.columns]
    clinical_controls = _available_controls(analysis_df)
    model_specs: dict[str, list[str]] = {
        "unadjusted": [],
        "oxygen_adjusted": oxygen_covariates,
        "clinical_adjusted": [*oxygen_covariates, *clinical_controls],
    }
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for feature in variability_features:
            for model_name, covariates in model_specs.items():
                covariates = [col for col in covariates if col != feature]
                identity = [
                    column
                    for column in ("dataset", "person_id", "stay_id")
                    if column in analysis_df and column not in {outcome, feature, *covariates}
                ]
                frame = analysis_df[
                    list(dict.fromkeys([outcome, feature, *covariates, *identity]))
                ].copy()
                frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
                valid = frame[outcome].notna() & frame[feature].notna()
                if "spo2_dynamics_eligible_flag" in analysis_df:
                    valid &= pd.to_numeric(
                        analysis_df.loc[frame.index, "spo2_dynamics_eligible_flag"],
                        errors="coerce",
                    ).eq(1)
                frame = frame.loc[valid].copy()
                n = int(len(frame))
                events = int(frame[outcome].sum()) if n else 0
                non_events = int(n - events)
                if (
                    n < min_rows
                    or events < min_events
                    or non_events < min_events
                    or frame[outcome].nunique() < 2
                ):
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "non_events": non_events,
                            "reason": "insufficient rows/events/non-events/classes",
                        }
                    )
                    continue
                feature_sd = float(frame[feature].std(ddof=0))
                if not np.isfinite(feature_sd) or feature_sd <= 0:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "feature variance is zero",
                        }
                    )
                    continue
                feature_mean = float(frame[feature].mean())
                z_feature = f"{feature}_z"
                frame[z_feature] = (frame[feature] - feature_mean) / feature_sd
                x = _prepare_design_matrix(frame, [z_feature, *covariates])
                if z_feature not in x.columns or x.empty:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "design matrix missing feature",
                        }
                    )
                    continue
                try:
                    x, collinear_dropped = _retain_full_rank_columns(
                        x, protected=[z_feature]
                    )
                except ValueError as exc:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": str(exc),
                        }
                    )
                    continue
                y = frame[outcome].astype(int)
                design = sm.add_constant(x, has_constant="add")
                events_per_parameter = min(events, non_events) / max(
                    design.shape[1], 1
                )
                if events_per_parameter < 5:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "underpowered_low_information",
                            "n": n,
                            "events": events,
                            "non_events": non_events,
                            "parameter_count": int(design.shape[1]),
                            "events_per_parameter": events_per_parameter,
                            "reason": "events_per_parameter_below_5_fail_closed",
                        }
                    )
                    continue
                try:
                    cluster = _patient_cluster_key(frame)
                    fit = _fit_binomial_glm_inference(
                        sm,
                        y,
                        design,
                        cluster,
                        required_terms=[z_feature],
                    )
                    nuisance_terms = getattr(
                        fit, "_physiograph_nonfinite_nuisance_terms", ()
                    )
                    coef = float(fit.params[z_feature])
                    ci_low, ci_high = fit.conf_int().loc[z_feature]
                    p_val = float(fit.pvalues[z_feature])
                    odds_ratio, odds_ci_low, odds_ci_high = _finite_odds_ratio_triplet(
                        coef, float(ci_low), float(ci_high)
                    )
                except _NonEstimableInferenceError as exc:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "non_estimable",
                            "n": n,
                            "events": events,
                            "non_events": non_events,
                            "reason": str(exc),
                        }
                    )
                    continue
                except Exception as exc:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "failed",
                            "n": n,
                            "events": events,
                            "non_events": non_events,
                            "reason": str(exc),
                        }
                    )
                    continue
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": model_name,
                        "status": "fit",
                        "n": n,
                        "events": events,
                        "non_events": non_events,
                        "or_per_1sd": odds_ratio,
                        "ci95_low": odds_ci_low,
                        "ci95_high": odds_ci_high,
                        "p_value": p_val,
                        "mean_feature": feature_mean,
                        "sd_feature": feature_sd,
                        "covariate_count": int(x.shape[1] - 1),
                        "collinear_adjustment_columns_dropped": ",".join(
                            collinear_dropped
                        ),
                        "nonfinite_nuisance_inference_count": len(
                            nuisance_terms
                        ),
                        "nonfinite_nuisance_inference_terms": ",".join(
                            nuisance_terms
                        ),
                        "covariance": "patient_cluster_robust" if cluster.nunique() > 1 else "HC0",
                        "events_per_parameter": min(events, n - events) / max(design.shape[1], 1),
                        "information_status": (
                            "adequate_ge10_events_per_parameter"
                            if events_per_parameter >= 10
                            else "fragile_5_to_10_events_per_parameter"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Sparse pilots and genuinely unavailable endpoints may yield only
    # skipped/failed rows.  Keep the output contract stable so downstream
    # sorting and CSV consumers do not require at least one successful fit.
    for column in (
        "or_per_1sd",
        "ci95_low",
        "ci95_high",
        "p_value",
        "p_value_adj",
        "n_tests",
        "mean_feature",
        "sd_feature",
        "covariate_count",
        "events_per_parameter",
    ):
        if column not in result:
            result[column] = math.nan
    for column in ("multiplicity_scope", "covariance"):
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="object")
    fit_mask = result["status"].eq("fit")
    if fit_mask.any():
        global_adjusted = _benjamini_hochberg(
            result.loc[fit_mask, "p_value"].astype(float).tolist()
        )
        result.loc[fit_mask, "p_value_adj"] = global_adjusted
        result.loc[fit_mask, "global_exploratory_q_value"] = global_adjusted
        result.loc[fit_mask, "n_tests"] = int(fit_mask.sum())
        result.loc[fit_mask, "multiplicity_scope"] = "all_variability_features_models_endpoints"
        result["focused_endpoint_model_q_value"] = np.nan
        for _, indexes in result.loc[fit_mask].groupby(["outcome", "model"]).groups.items():
            result.loc[indexes, "focused_endpoint_model_q_value"] = _benjamini_hochberg(
                result.loc[indexes, "p_value"].astype(float).tolist()
            )
        result.loc[fit_mask, "focused_multiplicity_scope"] = (
            "variability_features_within_one_endpoint_and_adjustment_model_post_result_amendment"
        )
    sort_cols = ["outcome", "model", "p_value", "feature"]
    return result.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def fit_spo2_variability_or_tables(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = 20,
) -> pd.DataFrame:
    """Per-dataset variability ORs with pooled estimates marked secondary."""
    if "dataset" not in analysis_df:
        result = _fit_spo2_variability_or_tables_single(
            analysis_df, min_rows=min_rows, min_events=min_events
        )
        result["analysis_scope"] = "unknown"
        return result
    pieces: list[pd.DataFrame] = []
    datasets = sorted(analysis_df["dataset"].dropna().astype(str).unique())
    for dataset in datasets:
        local = _fit_spo2_variability_or_tables_single(
            analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)].copy(),
            min_rows=min_rows,
            min_events=min_events,
        )
        local["analysis_scope"] = dataset
        pieces.append(local)
    if len(datasets) > 1:
        pooled_frame, unsupported = _mask_single_source_pooled_outcomes(
            analysis_df
        )
        pooled = _fit_spo2_variability_or_tables_single(
            pooled_frame, min_rows=min_rows, min_events=min_events
        )
        if unsupported and "outcome" in pooled:
            mask = pooled["outcome"].isin(unsupported)
            pooled.loc[mask, "reason"] = (
                "pooled endpoint not observed in at least two datasets"
            )
        pooled["analysis_scope"] = "pooled_secondary"
        pieces.append(pooled)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_spo2_variability_group_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw variability differences by outcome and dataset."""
    outcomes = _available_outcomes(analysis_df)
    features = [
        col
        for col in (
            "spo2_sd",
            "spo2_rmssd",
            "spo2_iqr",
            "spo2_range",
            "spo2_mad",
            "spo2_abrupt_jump_fraction",
            "spo2_drop_3_count",
            "spo2_dynamics_proxy_score",
        )
        if col in analysis_df.columns
    ]
    if "dataset" in analysis_df:
        scopes = [
            (str(dataset), group.copy())
            for dataset, group in analysis_df.groupby(
                analysis_df["dataset"].astype(str), sort=True
            )
        ]
        if len(scopes) > 1:
            scopes.append(("pooled_secondary", analysis_df.copy()))
    else:
        scopes = [("unknown", analysis_df)]
    rows: list[dict[str, Any]] = []
    for analysis_scope, scope_frame in scopes:
        for outcome in outcomes:
            if (
                analysis_scope == "pooled_secondary"
                and _observed_dataset_count(scope_frame, outcome) < 2
            ):
                continue
            y = pd.to_numeric(scope_frame[outcome], errors="coerce")
            for feature in features:
                x = pd.to_numeric(scope_frame[feature], errors="coerce")
                valid = y.isin([0, 1]) & x.notna()
                if valid.sum() < 10:
                    continue
                grp = pd.DataFrame({"y": y.loc[valid].astype(int), "x": x.loc[valid].astype(float)})
                mean0 = float(grp.loc[grp["y"].eq(0), "x"].mean()) if (grp["y"] == 0).any() else math.nan
                mean1 = float(grp.loc[grp["y"].eq(1), "x"].mean()) if (grp["y"] == 1).any() else math.nan
                median0 = float(grp.loc[grp["y"].eq(0), "x"].median()) if (grp["y"] == 0).any() else math.nan
                median1 = float(grp.loc[grp["y"].eq(1), "x"].median()) if (grp["y"] == 1).any() else math.nan
                if np.isfinite(mean0) and mean0 != 0 and np.isfinite(mean1):
                    pct_change = float((mean1 - mean0) / mean0 * 100.0)
                    ratio = float(mean1 / mean0)
                else:
                    pct_change = math.nan
                    ratio = math.nan
                rows.append(
                    {
                        "analysis_scope": analysis_scope,
                        "outcome": outcome,
                        "feature": feature,
                        "n_total": int(len(grp)),
                        "n_non_event": int((grp["y"] == 0).sum()),
                        "n_event": int((grp["y"] == 1).sum()),
                        "mean_non_event": mean0,
                        "mean_event": mean1,
                        "median_non_event": median0,
                        "median_event": median1,
                        "mean_ratio_event_vs_non_event": ratio,
                        "mean_pct_change_event_vs_non_event": pct_change,
                    }
                )
    if not rows:
        return pd.DataFrame([{"status": "skipped", "reason": "no valid variability/outcome group comparisons"}])
    return pd.DataFrame(rows).sort_values(
        ["analysis_scope", "outcome", "feature"]
    ).reset_index(drop=True)




def build_respiratory_context_tables(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Stratified SpO2 burden by respiratory support context with interaction terms."""
    measured = _filter_measured_spo2_rows(analysis_df)
    outcome = PRIMARY_ENDPOINT if PRIMARY_ENDPOINT in measured.columns else (
        "death_168h_flag" if "death_168h_flag" in measured.columns else None
    )
    if outcome is None:
        return pd.DataFrame([{"status": "skipped", "reason": "primary outcome unavailable"}])

    strata_specs: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=measured.index))]
    if "resp_support_any_flag" in measured.columns:
        resp = pd.to_numeric(measured["resp_support_any_flag"], errors="coerce").fillna(0)
        strata_specs.append(("no_respiratory_support", resp.eq(0)))
        mech = (
            pd.to_numeric(measured["mechanical_ventilation_flag"], errors="coerce").fillna(0)
            if "mechanical_ventilation_flag" in measured.columns
            else pd.Series(0, index=measured.index)
        )
        strata_specs.append(("no_mechanical_ventilation", mech.eq(0)))
    else:
        return pd.DataFrame([
            {"status": "skipped", "stratum": "all", "reason": "resp_support_any_flag unavailable"},
        ])

    rows: list[dict[str, Any]] = []
    for stratum, mask in strata_specs:
        subset = measured.loc[mask].copy()
        y = pd.to_numeric(subset[outcome], errors="coerce")
        valid = y.notna()
        n = int(valid.sum())
        events = int(y.loc[valid].sum()) if n else 0
        non_events = int(n - events)
        if n < 20 or events < MIN_OR_EVENTS or non_events < MIN_OR_EVENTS:
            rows.append({
                "stratum": stratum,
                "outcome": outcome,
                "status": "skipped",
                "n": n,
                "events": events,
                "non_events": non_events,
                "reason": "insufficient rows/events/non-events",
            })
            continue
        burden = pd.to_numeric(subset.get("spo2_below_90_fraction"), errors="coerce")
        rows.append({
            "stratum": stratum,
            "outcome": outcome,
            "status": "descriptive",
            "n": n,
            "events": events,
            "non_events": non_events,
            "mean_spo2_below_90_fraction": float(burden.mean()) if burden.notna().any() else math.nan,
            "event_rate": float(y.loc[valid].mean()),
        })

    # Interaction models
    interaction_specs = [
        ("spo2_dynamics_x_resp_support", "spo2_dynamics_proxy_score", "resp_support_any_flag"),
        ("spo2_dynamics_x_fio2_max", "spo2_dynamics_proxy_score", "fio2_max"),
        ("spo2_below_90_fraction_x_resp_support", "spo2_below_90_fraction", "resp_support_any_flag"),
        ("spo2_below_90_fraction_x_fio2_max", "spo2_below_90_fraction", "fio2_max"),
    ]
    try:
        import statsmodels.api as sm
    except Exception as exc:
        rows.append({"status": "skipped", "analysis": "interaction", "reason": f"statsmodels unavailable: {exc}"})
        return pd.DataFrame(rows)

    controls = _available_controls(measured)
    for label, feat_a, feat_b in interaction_specs:
        if feat_a not in measured.columns or feat_b not in measured.columns:
            rows.append({"status": "skipped", "analysis": label, "reason": f"missing {feat_a} or {feat_b}"})
            continue
        model_columns = list(dict.fromkeys([outcome, feat_a, feat_b, *controls]))
        frame = measured[model_columns].copy()
        for col in [outcome, feat_a, feat_b]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        valid = frame[outcome].notna() & frame[feat_a].notna() & frame[feat_b].notna()
        if feat_a == "spo2_dynamics_proxy_score" and "spo2_dynamics_eligible_flag" in measured:
            valid &= pd.to_numeric(
                measured.loc[frame.index, "spo2_dynamics_eligible_flag"], errors="coerce"
            ).eq(1)
        frame = frame.loc[valid].copy()
        n = int(len(frame))
        events = int(frame[outcome].sum()) if n else 0
        non_events = int(n - events)
        if (
            n < 200
            or events < MIN_OR_EVENTS
            or non_events < MIN_OR_EVENTS
            or frame[outcome].nunique() < 2
        ):
            rows.append({
                "status": "skipped",
                "analysis": label,
                "n": n,
                "events": events,
                "non_events": non_events,
                "reason": "insufficient rows/events/non-events/classes",
            })
            continue
        inter_col = f"{feat_a}_x_{feat_b}"
        frame[inter_col] = frame[feat_a] * frame[feat_b]
        design_columns = list(dict.fromkeys([feat_a, feat_b, inter_col, *controls]))
        x = _prepare_design_matrix(frame, design_columns)
        if inter_col not in x.columns:
            rows.append({"status": "skipped", "analysis": label, "reason": "interaction term dropped from design matrix"})
            continue
        try:
            x, collinear_dropped = _retain_full_rank_columns(
                x, protected=[feat_a, feat_b, inter_col]
            )
        except ValueError as exc:
            rows.append(
                {"status": "skipped", "analysis": label, "reason": str(exc)}
            )
            continue
        y = frame[outcome].astype(int)
        try:
            design = sm.add_constant(x, has_constant="add")
            cluster = _patient_cluster_key(frame)
            fit = _fit_binomial_glm_inference(
                sm,
                y,
                design,
                cluster,
                required_terms=[inter_col],
            )
            nuisance_terms = getattr(
                fit, "_physiograph_nonfinite_nuisance_terms", ()
            )
            coef = float(fit.params[inter_col])
            ci_low, ci_high = fit.conf_int().loc[inter_col]
            odds_ratio, odds_ci_low, odds_ci_high = _finite_odds_ratio_triplet(
                coef, float(ci_low), float(ci_high)
            )
            rows.append({
                "status": "fit",
                "analysis": label,
                "outcome": outcome,
                "n": n,
                "events": events,
                "non_events": non_events,
                "or_interaction": odds_ratio,
                "ci95_low": odds_ci_low,
                "ci95_high": odds_ci_high,
                "p_value": float(fit.pvalues[inter_col]),
                "covariance": "patient_cluster_robust" if cluster.nunique() > 1 else "HC0",
                "collinear_adjustment_columns_dropped": ",".join(
                    collinear_dropped
                ),
                "nonfinite_nuisance_inference_count": len(nuisance_terms),
                "nonfinite_nuisance_inference_terms": ",".join(
                    nuisance_terms
                ),
                "analysis_note": "diagnostic_primary_s4_s6_are_dataset_stratified",
            })
        except _NonEstimableInferenceError as exc:
            rows.append({
                "status": "non_estimable",
                "analysis": label,
                "outcome": outcome,
                "n": n,
                "events": events,
                "non_events": non_events,
                "reason": str(exc),
            })
        except Exception as exc:
            rows.append({
                "status": "failed",
                "analysis": label,
                "outcome": outcome,
                "n": n,
                "events": events,
                "non_events": non_events,
                "reason": str(exc),
            })
    result = pd.DataFrame(rows)
    fit_mask = result.get("status", pd.Series(index=result.index, dtype=str)).eq("fit")
    if fit_mask.any():
        result.loc[fit_mask, "p_value_adj"] = _benjamini_hochberg(
            pd.to_numeric(result.loc[fit_mask, "p_value"], errors="coerce").tolist()
        )
        result.loc[fit_mask, "multiplicity_scope"] = "respiratory_interaction_diagnostics"
    return result


def build_availability_audit(
    dataset_artifacts: dict[str, dict[str, pd.DataFrame]],
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-dataset availability audit for SpO2, controls, race, and composite endpoints."""
    rows: list[dict[str, Any]] = []
    for dataset, artifacts in dataset_artifacts.items():
        if "dataset" in analysis_df:
            frame = analysis_df.loc[
                analysis_df["dataset"].astype(str).eq(str(dataset))
            ].copy()
        elif len(dataset_artifacts) == 1:
            frame = analysis_df.copy()
        else:
            frame = analysis_df.iloc[0:0].copy()
        cohort = artifacts["cohort"]
        race_cols = [c for c in cohort.columns if c.lower() in {"race", "ethnicity", "race_ethnicity"}]

        def numeric_or_zero(column: str) -> pd.Series:
            source = (
                frame[column]
                if column in frame
                else pd.Series(0.0, index=frame.index, dtype=float)
            )
            return pd.to_numeric(source, errors="coerce").fillna(0)

        def positive_count(column: str) -> int:
            return int(numeric_or_zero(column).gt(0).sum())

        plausible = numeric_or_zero("spo2_plausible_count")
        zero_spo2 = int((plausible.eq(0)).sum())
        row: dict[str, Any] = {
            "dataset": dataset,
            "n_rows": int(len(frame)),
            "spo2_measured_rows": int(plausible.gt(0).sum()),
            "spo2_zero_plausible_rows": zero_spo2,
            "fio2_available_rows": positive_count("fio2_measurement_count"),
            "respiratory_support_rows": positive_count("resp_support_any_flag"),
            "mechanical_ventilation_rows": positive_count("mechanical_ventilation_flag"),
            "rrt_or_dialysis_rows": positive_count("rrt_or_dialysis_flag"),
            "vis_12h_observed_rows": positive_count("vis_observed_12h"),
            "vis_24h_observed_rows": positive_count("vis_observed_24h"),
            "urine_output_12h_observed_rows": positive_count("urine_output_12h_observed"),
            "urine_output_24h_observed_rows": positive_count("urine_output_24h_observed"),
            "person_id_available": bool("person_id" in frame and frame["person_id"].notna().any()),
            "race_ethnicity_status": "present" if race_cols else "absent_explicit",
            "race_ethnicity_columns": ",".join(race_cols) if race_cols else "",
        }
        for outcome in (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS):
            if outcome in frame.columns:
                y = pd.to_numeric(frame[outcome], errors="coerce")
                row[f"{outcome}_event_rate"] = float(y.mean()) if y.notna().any() else math.nan
                row[f"{outcome}_events"] = int(y.sum()) if y.notna().any() else 0
        rows.append(row)
    return pd.DataFrame(rows)


def fit_external_cross_dataset_holdout(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Train on MIMIC and evaluate untouched eICU using train-only preprocessing."""
    result = fit_external_transportability(
        analysis_df,
        min_rows=DATASET_HOLDOUT_MIN_ROWS,
        min_events=DATASET_HOLDOUT_MIN_EVENTS,
    )
    result["analysis"] = "external_cross_dataset_transportability"
    return result


def fit_missingness_negative_control(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper for the measurement-intensity control models."""
    return fit_grouped_missingness_control(analysis_df)


def lint_claims_and_outputs(
    *,
    model_metrics: pd.DataFrame,
    manifest: dict[str, Any],
    output_paths: dict[str, Path],
) -> pd.DataFrame:
    """Flag PI-facing outputs that contain apparent metrics or in-sample calibration."""
    warnings: list[dict[str, Any]] = []
    forbidden_cols = {"auroc_apparent", "auprc_apparent", "brier_apparent", "ece_apparent"}
    if not model_metrics.empty:
        bad = [c for c in model_metrics.columns if c in forbidden_cols]
        if bad:
            warnings.append({
                "severity": "error",
                "check": "apparent_metrics_in_model_output",
                "detail": f"forbidden columns present: {bad}",
            })
        if "auroc_oof" not in model_metrics.columns and model_metrics.get("status", pd.Series()).eq("fit").any():
            warnings.append({
                "severity": "warning",
                "check": "missing_oof_metrics",
                "detail": "fit rows lack out-of-fold AUROC",
            })
    if not manifest.get("fresh_colab_execution", False):
        warnings.append({
            "severity": "warning",
            "check": "fresh_colab_execution",
            "detail": _execution_provenance_note(manifest),
        })
    warnings.append(
        {
            "severity": "info",
            "check": "execution_provenance",
            "detail": _execution_provenance_note(manifest),
        }
    )
    for w in MANIFEST_WARNINGS:
        warnings.append({"severity": "info", "check": "manifest_policy", "detail": w})
    for key, p in output_paths.items():
        if p.exists() and p.suffix == ".csv":
            head = p.read_text(encoding="utf-8")[:4000]
            if "auroc_apparent" in head or "brier_apparent" in head:
                warnings.append({
                    "severity": "error",
                    "check": "apparent_metrics_in_csv",
                    "detail": f"{key}: {p.name}",
                })
    return pd.DataFrame(warnings)


def build_analysis_manifest(
    *,
    build_new: bool,
    fresh_colab_execution: bool,
    paths: dict[str, Path],
    datasets: list[str],
    rows: int,
    claims_warnings: pd.DataFrame,
    execution_environment: str = "local",
    run_comparator_requested: bool = False,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 250_000,
    require_all_requested_datasets: bool = True,
    analysis_only: bool = False,
) -> dict[str, Any]:
    """Build provenance manifest with explicit limitations."""
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "analysis": "SpO2 drilldown from cached or freshly rebuilt PhysioGraph artifacts",
        "fresh_colab_execution": bool(fresh_colab_execution),
        "build_new": bool(build_new),
        "analysis_only": bool(analysis_only),
        "execution_environment": str(execution_environment),
        "run_comparator_requested": bool(run_comparator_requested),
        "max_stays": max_stays,
        "max_chunks": max_chunks,
        "chunk_size": int(chunk_size),
        "require_all_requested_datasets": bool(require_all_requested_datasets),
        "output_fingerprint_integrity_basis": "sha256_and_size_bytes",
        "output_mtime_ns_role": "informational_cloud_sync_may_change_it",
        "raw_source_fingerprint_integrity_basis": "size_bytes_and_mtime_ns_without_full_source_hashing",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "outcome_clock": "12_and_24_hours_after_4h_landmark",
        "analysis_layers": list(ANALYSIS_LAYERS),
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "datasets": datasets,
        "rows": int(rows),
        "warnings": [
            _execution_provenance_note(
                {
                    "fresh_colab_execution": fresh_colab_execution,
                    "build_new": build_new,
                    "analysis_only": analysis_only,
                    "execution_environment": execution_environment,
                }
            ),
            *MANIFEST_WARNINGS,
        ],
        "claims_linter": claims_warnings.to_dict(orient="records") if not claims_warnings.empty else [],
        "notes": [
            "No Google Colab execution is claimed unless the notebook is run in Colab.",
            "Primary grouped-CV metrics are estimated separately within MIMIC and eICU; pooled estimates are secondary.",
            "MIMIC-to-eICU transportability is reported separately from within-dataset validation.",
            "Preprocessing is fitted independently within each patient-grouped training fold.",
            "Calibration/ECE uses out-of-fold predictions only.",
            "Structured respiratory/RRT fields are used where available; text-derived fields remain labeled proxies.",
            "CLIF dataset is out of scope.",
            "Episode-anchored lactate analyses are a transparent post-result protocol amendment and are exploratory until prospectively replicated.",
        ],
    }


def fit_spo2_dragged_horizon_models(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 500,
    min_events: int = 30,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compatibility wrapper around the rigorous incremental-value analysis."""
    result = fit_grouped_incremental_models(
        analysis_df,
        min_rows=min_rows,
        min_events=min_events,
        n_splits=n_splits,
    )
    result["analysis"] = "prespecified_12_24h_incremental_value"
    return result


def collect_raw_spo2_events(dataset_artifacts: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Collect observation-window raw SpO2 events across datasets."""
    frames: list[pd.DataFrame] = []
    for dataset, artifacts in dataset_artifacts.items():
        events = _observation_events(artifacts["events"])
        if events.empty or "concept" not in events.columns:
            continue
        spo2 = events.loc[events["concept"].astype(str).str.lower().eq("spo2")].copy()
        if spo2.empty:
            continue
        if "dataset" not in spo2.columns:
            spo2["dataset"] = dataset
        else:
            spo2["dataset"] = spo2["dataset"].fillna(dataset)
        spo2["value_numeric"] = pd.to_numeric(spo2["value_numeric"], errors="coerce")
        spo2["offset_minutes"] = pd.to_numeric(spo2["offset_minutes"], errors="coerce")
        spo2 = spo2.dropna(subset=["value_numeric", "offset_minutes"]).copy()
        if "time_bin" in spo2.columns:
            spo2["time_bin"] = pd.to_numeric(spo2["time_bin"], errors="coerce")
        else:
            spo2["time_bin"] = np.floor(
                spo2["offset_minutes"].clip(lower=0, upper=LANDMARK_MINUTES - 1) / 15.0
            )
        spo2["time_bin"] = spo2["time_bin"].clip(lower=0, upper=OBSERVATION_BINS - 1).fillna(0).astype(int)
        frames.append(spo2[["dataset", "stay_id", "offset_minutes", "time_bin", "value_numeric"]])
    if not frames:
        return pd.DataFrame(columns=["dataset", "stay_id", "offset_minutes", "time_bin", "value_numeric"])
    return pd.concat(frames, ignore_index=True)


def build_spo2_raw_event_summary(raw_spo2: pd.DataFrame, analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize signal quality and protocol-qualified transitions."""
    if raw_spo2.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no raw SpO2 events"}])
    frame = raw_spo2.copy().sort_values(["dataset", "stay_id", "offset_minutes"])
    frame["plausible"] = frame["value_numeric"].between(50, 100, inclusive="both")
    binned = (
        frame.loc[frame["plausible"]]
        .groupby(["dataset", "stay_id", "time_bin"], as_index=False)
        .agg(
            value_numeric=("value_numeric", "median"),
            offset_minutes=("offset_minutes", "median"),
        )
        .sort_values(["dataset", "stay_id", "time_bin"])
    )
    binned["below_90"] = binned["value_numeric"].lt(90)
    binned["below_88"] = binned["value_numeric"].lt(88)
    binned["delta"] = binned.groupby(["dataset", "stay_id"])[
        "value_numeric"
    ].diff()
    binned["gap_minutes"] = binned.groupby(["dataset", "stay_id"])[
        "offset_minutes"
    ].diff()
    binned["qualified_transition"] = binned["gap_minutes"].gt(0) & binned[
        "gap_minutes"
    ].le(30)
    binned["abrupt_jump"] = (
        binned["qualified_transition"] & binned["delta"].abs().ge(4)
    )

    outcome_cols = _available_outcomes(analysis_df)
    lookup = analysis_df[["dataset", "stay_id", *outcome_cols]].drop_duplicates(["dataset", "stay_id"])
    frame = frame.merge(lookup, on=["dataset", "stay_id"], how="left")
    binned = binned.merge(lookup, on=["dataset", "stay_id"], how="left")

    rows: list[dict[str, Any]] = []

    def summarize(
        event_subset: pd.DataFrame,
        bin_subset: pd.DataFrame,
        *,
        scope: str,
        group_col: str = "all",
        group_value: Any = "all",
    ) -> None:
        if event_subset.empty:
            return
        transitions = bin_subset.loc[bin_subset["qualified_transition"]]
        rows.append(
            {
                "scope": scope,
                "group_column": group_col,
                "group_value": group_value,
                "n_events": int(len(event_subset)),
                "n_stays": int(event_subset[["dataset", "stay_id"]].drop_duplicates().shape[0]),
                "n_stay_bins": int(len(bin_subset)),
                "n_qualified_transitions": int(len(transitions)),
                "plausible_fraction": float(event_subset["plausible"].mean()),
                "below_90_fraction": float(bin_subset["below_90"].mean()) if not bin_subset.empty else math.nan,
                "below_88_fraction": float(bin_subset["below_88"].mean()) if not bin_subset.empty else math.nan,
                "abrupt_jump_fraction": float(transitions["abrupt_jump"].mean()) if not transitions.empty else math.nan,
                "median_gap_minutes": float(transitions["gap_minutes"].median()) if not transitions.empty else math.nan,
                "mean_spo2": float(bin_subset["value_numeric"].mean()) if not bin_subset.empty else math.nan,
                "median_spo2": float(bin_subset["value_numeric"].median()) if not bin_subset.empty else math.nan,
                "spo2_p10": float(bin_subset["value_numeric"].quantile(0.10)) if not bin_subset.empty else math.nan,
                "spo2_p90": float(bin_subset["value_numeric"].quantile(0.90)) if not bin_subset.empty else math.nan,
                "distribution_weighting": "one_median_per_stay_15_minute_bin",
                "transition_definition": "same_as_primary_gap_le_30_minutes",
            }
        )

    summarize(frame, binned, scope="overall")
    for dataset, ds in frame.groupby("dataset", sort=False):
        bins = binned.loc[binned["dataset"].astype(str).eq(str(dataset))]
        summarize(ds, bins, scope="dataset", group_col="dataset", group_value=dataset)
        for outcome in outcome_cols:
            for value in (0, 1):
                event_sub = ds.loc[pd.to_numeric(ds[outcome], errors="coerce").eq(value)]
                bin_sub = bins.loc[pd.to_numeric(bins[outcome], errors="coerce").eq(value)]
                summarize(
                    event_sub,
                    bin_sub,
                    scope=str(dataset),
                    group_col=outcome,
                    group_value=value,
                )
    return pd.DataFrame(rows)


def build_spo2_trajectory_summary(
    raw_spo2: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    outcome: str | None = None,
) -> pd.DataFrame:
    """Aggregate stay-weighted 15-minute SpO2 trajectories by outcome."""
    if raw_spo2.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no raw SpO2 events"}])
    outcomes = (
        [outcome]
        if outcome is not None and outcome in analysis_df
        else _available_outcomes(analysis_df)
    )
    if outcome is not None and outcome not in analysis_df:
        return pd.DataFrame([{"status": "skipped", "reason": f"outcome missing: {outcome}"}])
    if not outcomes:
        return pd.DataFrame([{"status": "skipped", "reason": "no prespecified outcomes available"}])
    frame = raw_spo2.loc[
        pd.to_numeric(raw_spo2["value_numeric"], errors="coerce").between(50, 100)
    ].copy()
    frame["value_numeric"] = pd.to_numeric(frame["value_numeric"], errors="coerce")
    frame = (
        frame.groupby(["dataset", "stay_id", "time_bin"], as_index=False)[
            "value_numeric"
        ]
        .median()
    )
    frame = frame.merge(
        analysis_df[["dataset", "stay_id", *outcomes]].drop_duplicates(["dataset", "stay_id"]),
        on=["dataset", "stay_id"],
        how="left",
    )
    if frame.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no valid trajectories after filters"}])
    scopes = [
        (str(dataset), group.copy())
        for dataset, group in frame.groupby(frame["dataset"].astype(str), sort=True)
    ]
    if len(scopes) > 1:
        scopes.append(("pooled_secondary", frame.copy()))
    rows: list[pd.DataFrame] = []
    for analysis_scope, scope_frame in scopes:
        for endpoint in outcomes:
            if (
                analysis_scope == "pooled_secondary"
                and _observed_dataset_count(scope_frame, endpoint) < 2
            ):
                continue
            local = scope_frame.copy()
            local[endpoint] = pd.to_numeric(local[endpoint], errors="coerce")
            local = local.loc[local[endpoint].isin([0, 1])]
            if local.empty:
                continue
            grouped = local.groupby([endpoint, "time_bin"], sort=True)["value_numeric"]
            summary = grouped.agg(["count", "mean", "median", "std"]).rename(
                columns={"count": "n_stays"}
            ).reset_index()
            summary["q25"] = grouped.quantile(0.25).to_numpy()
            summary["q75"] = grouped.quantile(0.75).to_numpy()
            summary["below_90_fraction"] = (
                local.assign(_low=local["value_numeric"].lt(90))
                .groupby([endpoint, "time_bin"], sort=True)["_low"]
                .mean()
                .to_numpy()
            )
            summary["minutes_from_landmark_bin_start"] = summary["time_bin"].astype(int) * 15
            summary = summary.rename(columns={endpoint: "outcome_value"})
            summary.insert(0, "outcome", endpoint)
            summary.insert(0, "analysis_scope", analysis_scope)
            summary["weighting"] = "one_median_per_stay_bin"
            rows.append(summary)
    if not rows:
        return pd.DataFrame([{"status": "skipped", "reason": "no valid outcome trajectories"}])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["analysis_scope", "outcome", "outcome_value", "time_bin"]
    ).reset_index(drop=True)


def build_cohort_flow_table(
    dataset_artifacts: dict[str, dict[str, pd.DataFrame]],
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create an auditable analysis-inclusion flow for each dataset."""
    rows: list[dict[str, Any]] = []
    for dataset, artifacts in dataset_artifacts.items():
        cohort = artifacts["cohort"]
        frame = analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)].copy()
        total = int(len(cohort))
        excluded_source = (
            cohort["excluded_before_landmark_flag"]
            if "excluded_before_landmark_flag" in cohort
            else pd.Series(0, index=cohort.index, dtype=float)
        )
        excluded = pd.to_numeric(
            excluded_source, errors="coerce"
        )
        plausible = (
            frame["spo2_plausible_count"]
            if "spo2_plausible_count" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        dynamics = (
            frame["spo2_dynamics_eligible_flag"]
            if "spo2_dynamics_eligible_flag" in frame
            else pd.Series(0, index=frame.index, dtype=float)
        )
        steps = [
            ("cohort_after_clinical_eligibility", total),
            ("alive_and_in_icu_at_4h_landmark", int(excluded.fillna(0).eq(0).sum())),
            ("analysis_frame_assembled", int(len(frame))),
            (
                "at_least_one_plausible_spo2_0_to_4h",
                int(pd.to_numeric(plausible, errors="coerce").fillna(0).gt(0).sum()),
            ),
            (
                "spo2_dynamics_eligible_at_least_3_bins_2_pairs",
                int(pd.to_numeric(dynamics, errors="coerce").fillna(0).eq(1).sum()),
            ),
        ]
        for horizon in POST_LANDMARK_HORIZONS_HOURS:
            followup_source = (
                frame[f"complete_followup_{horizon}h"]
                if f"complete_followup_{horizon}h" in frame
                else pd.Series(0, index=frame.index, dtype=float)
            )
            steps.append(
                (
                    f"complete_fixed_followup_{horizon}h_after_landmark",
                    int(pd.to_numeric(followup_source, errors="coerce").fillna(0).eq(1).sum()),
                )
            )
        for order, (step, count) in enumerate(steps, start=1):
            rows.append(
                {
                    "dataset": dataset,
                    "step_order": order,
                    "step": step,
                    "n": count,
                    "fraction_of_clinically_eligible_cohort": count / total if total else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_table1(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form baseline characteristics with dataset-specific denominators."""
    variables = [
        column
        for column in (
            "age", "is_male", "shock_icd_flag", "baseline_lactate",
            "baseline_creatinine", "baseline_map", "baseline_hr", "baseline_sbp",
            "baseline_ph", "baseline_bilirubin_total", "baseline_vasoactive_flag",
            "baseline_mcs_flag", "resp_support_any_flag",
            "mechanical_ventilation_flag", "rrt_or_dialysis_flag", "spo2_mean",
            "spo2_min", "spo2_rmssd", "spo2_below_90_fraction",
            "spo2_abrupt_jump_fraction", "spo2_drop_3_count",
            "spo2_dynamics_proxy_score", "spo2_sampling_density_per_hr",
            "spo2_missing_bin_count", "spo2_longest_gap_minutes",
        )
        if column in analysis_df
    ]
    binary = {
        "is_male", "shock_icd_flag", "baseline_vasoactive_flag",
        "baseline_mcs_flag", "resp_support_any_flag",
        "mechanical_ventilation_flag", "rrt_or_dialysis_flag",
    }
    rows: list[dict[str, Any]] = []
    for dataset, group in analysis_df.groupby(analysis_df["dataset"].astype(str)):
        for variable in variables:
            values = pd.to_numeric(group[variable], errors="coerce")
            observed = values.dropna()
            row: dict[str, Any] = {
                "dataset": dataset,
                "variable": variable,
                "n_total": int(len(group)),
                "n_observed": int(len(observed)),
                "missing_fraction": float(values.isna().mean()),
                "summary_type": "binary" if variable in binary else "continuous",
            }
            if variable in binary:
                row["count_positive"] = int(observed.eq(1).sum())
                row["proportion_positive"] = float(observed.eq(1).mean()) if len(observed) else math.nan
            else:
                row.update(
                    {
                        "mean": float(observed.mean()) if len(observed) else math.nan,
                        "sd": float(observed.std(ddof=1)) if len(observed) > 1 else math.nan,
                        "median": float(observed.median()) if len(observed) else math.nan,
                        "q1": float(observed.quantile(0.25)) if len(observed) else math.nan,
                        "q3": float(observed.quantile(0.75)) if len(observed) else math.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_missingness_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Report predictor availability separately for MIMIC and eICU."""
    candidates = list(dict.fromkeys([
        *CONTROL_FEATURE_CANDIDATES,
        *PRIMARY_SPO2_MODEL_FEATURES,
        *SPO2_VARIABILITY_FEATURES,
        *MISSINGNESS_CONTROL_FEATURES,
    ]))
    rows: list[dict[str, Any]] = []
    for dataset, group in analysis_df.groupby(analysis_df["dataset"].astype(str)):
        for variable in candidates:
            observed = int(group[variable].notna().sum()) if variable in group else 0
            status = (
                "column_unavailable"
                if variable not in group
                else "available"
                if observed
                else "all_missing"
            )
            rows.append(
                {
                    "dataset": dataset,
                    "feature": variable,
                    "status": status,
                    "n_total": int(len(group)),
                    "n_observed": observed,
                    "missing_fraction": 1.0 - observed / len(group) if len(group) else math.nan,
                    "reason": (
                        "column_not_present"
                        if status == "column_unavailable"
                        else "no_observed_values"
                        if status == "all_missing"
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_oof_performance_curves(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build ROC, precision-recall, and calibration points from OOF predictions."""
    if oof_predictions is None or oof_predictions.empty:
        return pd.DataFrame([{"status": "unavailable", "reason": "no fitted OOF predictions"}])
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    rows: list[dict[str, Any]] = []
    for (scope, outcome, model), group in oof_predictions.groupby(
        ["analysis_scope", "outcome", "model"], dropna=False
    ):
        y = pd.to_numeric(group["y_true"], errors="coerce")
        probability = pd.to_numeric(group["probability"], errors="coerce")
        valid = y.isin([0, 1]) & probability.notna()
        if valid.sum() < 2 or y.loc[valid].nunique() < 2:
            continue
        yv = y.loc[valid].astype(int).to_numpy()
        pv = probability.loc[valid].to_numpy(dtype=float)
        fpr, tpr, roc_threshold = roc_curve(yv, pv)
        precision, recall, pr_threshold = precision_recall_curve(yv, pv)
        fraction_positive, mean_predicted = calibration_curve(
            yv,
            pv,
            n_bins=min(10, max(2, int(np.sqrt(len(yv))))),
            strategy="quantile",
        )
        for index in range(len(fpr)):
            rows.append({"analysis_scope": scope, "outcome": outcome, "model": model, "curve": "roc", "x": float(fpr[index]), "y": float(tpr[index]), "threshold": float(roc_threshold[index])})
        for index in range(len(precision)):
            threshold = float(pr_threshold[index]) if index < len(pr_threshold) else math.nan
            rows.append({"analysis_scope": scope, "outcome": outcome, "model": model, "curve": "precision_recall", "x": float(recall[index]), "y": float(precision[index]), "threshold": threshold})
        for x_value, y_value in zip(mean_predicted, fraction_positive):
            rows.append({"analysis_scope": scope, "outcome": outcome, "model": model, "curve": "calibration", "x": float(x_value), "y": float(y_value), "threshold": math.nan})
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "unavailable", "reason": "no two-class OOF groups"}])


def build_endpoint_conclusions(
    endpoint_audit: pd.DataFrame,
    model_metrics: pd.DataFrame,
    risk_tables: pd.DataFrame,
    sensitivity: pd.DataFrame,
    external_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify endpoint evidence without converting fragility into a claim."""
    rows: list[dict[str, Any]] = []
    for audit in endpoint_audit.itertuples(index=False):
        dataset, endpoint = str(audit.dataset), str(audit.endpoint)
        model_row = model_metrics.loc[
            model_metrics.get("analysis_scope", pd.Series(index=model_metrics.index, dtype=str)).astype(str).eq(dataset)
            & model_metrics.get("outcome", pd.Series(index=model_metrics.index, dtype=str)).astype(str).eq(endpoint)
            & model_metrics.get("model", pd.Series(index=model_metrics.index, dtype=str)).eq("parsimonious_spo2_instability")
            & model_metrics.get("status", pd.Series(index=model_metrics.index, dtype=str)).eq("fit")
        ]
        risk_row = risk_tables.loc[
            risk_tables.get("dataset", pd.Series(index=risk_tables.index, dtype=str)).astype(str).eq(dataset)
            & risk_tables.get("endpoint", pd.Series(index=risk_tables.index, dtype=str)).astype(str).eq(endpoint)
            & risk_tables.get("status", pd.Series(index=risk_tables.index, dtype=str)).eq("estimated")
        ]
        model_auc_positive = (
            not model_row.empty
            and pd.to_numeric(
                model_row.iloc[0].get(
                    "delta_auroc_vs_reference_ci95_low",
                    model_row.iloc[0].get("delta_auroc_vs_absolute_ci95_low"),
                ),
                errors="coerce",
            )
            > 0
        )
        model_auprc_positive = (
            not model_row.empty
            and pd.to_numeric(
                model_row.iloc[0].get(
                    "delta_auprc_vs_reference_ci95_low",
                    model_row.iloc[0].get("delta_auprc_vs_absolute_ci95_low"),
                ),
                errors="coerce",
            )
            > 0
        )
        model_bootstrap_valid = (
            not model_row.empty
            and pd.to_numeric(
                model_row.iloc[0].get("bootstrap_repetitions_valid"),
                errors="coerce",
            )
            >= 0.8
            * pd.to_numeric(
                model_row.iloc[0].get("bootstrap_repetitions_requested"),
                errors="coerce",
            )
        )
        model_epv_adequate = (
            not model_row.empty
            and str(model_row.iloc[0].get("epv_status", ""))
            == "adequate_ge_10"
        )
        model_robust = (
            model_auc_positive
            and model_auprc_positive
            and model_bootstrap_valid
            and model_epv_adequate
        )
        model_suggestive = model_epv_adequate and (
            model_auc_positive or model_auprc_positive
        )
        external = (
            external_metrics
            if external_metrics is not None
            else pd.DataFrame()
        )
        external_mask = (
            external.get(
                "outcome", pd.Series(index=external.index, dtype=str)
            ).astype(str).eq(endpoint)
            & external.get(
                "model", pd.Series(index=external.index, dtype=str)
            ).eq("parsimonious_spo2_instability")
            & external.get(
                "status", pd.Series(index=external.index, dtype=str)
            ).eq("fit")
        )
        # A transport result validates the model trained in ``dataset``.  The
        # locked external analysis is MIMIC -> eICU, so it can support a MIMIC
        # claim but must never be recycled as validation of an eICU-trained
        # model.
        if "train_dataset" in external:
            external_mask &= external["train_dataset"].astype(str).eq(dataset)
        if "test_dataset" in external:
            external_mask &= ~external["test_dataset"].astype(str).eq(dataset)
        external_row = external.loc[external_mask]
        external_auc_positive = (
            not external_row.empty
            and pd.to_numeric(
                external_row.iloc[0].get(
                    "delta_auroc_vs_reference_ci95_low",
                    external_row.iloc[0].get(
                        "delta_auroc_vs_absolute_ci95_low"
                    ),
                ),
                errors="coerce",
            )
            > 0
        )
        external_auprc_positive = (
            not external_row.empty
            and pd.to_numeric(
                external_row.iloc[0].get(
                    "delta_auprc_vs_reference_ci95_low",
                    external_row.iloc[0].get(
                        "delta_auprc_vs_absolute_ci95_low"
                    ),
                ),
                errors="coerce",
            )
            > 0
        )
        external_bootstrap_valid = (
            not external_row.empty
            and pd.to_numeric(
                external_row.iloc[0].get("bootstrap_repetitions_valid"),
                errors="coerce",
            )
            >= 0.8
            * pd.to_numeric(
                external_row.iloc[0].get("bootstrap_repetitions_requested"),
                errors="coerce",
            )
        )
        external_epv_adequate = (
            not external_row.empty
            and str(external_row.iloc[0].get("epv_status", ""))
            == "adequate_ge_10"
        )
        external_robust = (
            external_auc_positive
            and external_auprc_positive
            and external_bootstrap_valid
            and external_epv_adequate
        )
        epi_positive = (
            not risk_row.empty
            and pd.to_numeric(risk_row.iloc[0].get("risk_ratio"), errors="coerce") > 1
            and str(risk_row.iloc[0].get("risk_ratio_ci_method", ""))
            == "patient_cluster_bootstrap"
            and pd.to_numeric(
                risk_row.iloc[0].get("risk_ratio_ci95_low"), errors="coerce"
            )
            > 1
            and pd.to_numeric(risk_row.iloc[0].get("fisher_p_bh_adjusted"), errors="coerce") < 0.05
        )
        sensitivity_rows = sensitivity.loc[
            sensitivity.get("dataset", pd.Series(index=sensitivity.index, dtype=str)).astype(str).eq(dataset)
            & sensitivity.get("endpoint", pd.Series(index=sensitivity.index, dtype=str)).astype(str).eq(endpoint)
            & sensitivity.get("status", pd.Series(index=sensitivity.index, dtype=str)).eq("estimated")
        ]
        # Robustness of the dynamics hypothesis must not be helped or harmed by
        # sensitivity rows whose exposure is absolute hypoxemia/desaturation or
        # monitoring intensity. Those rows remain reported, but are diagnostics
        # rather than replications of the dynamics exposure.
        if "exposure" in sensitivity_rows:
            sensitivity_rows = sensitivity_rows.loc[
                ~sensitivity_rows["exposure"].astype(str).isin(
                    {
                        "exposure_hypoxemia_or_dynamics",
                        "exposure_sustained_low",
                        "exposure_sampling_high",
                    }
                )
            ]
        effect_source = (
            sensitivity_rows["effect"]
            if "effect" in sensitivity_rows
            else pd.Series(dtype=float)
        )
        effects = pd.to_numeric(effect_source, errors="coerce").dropna()
        sensitivity_fraction = float(effects.gt(1).mean()) if len(effects) else math.nan
        sensitivity_consistent = len(effects) >= 3 and sensitivity_fraction >= 0.75
        model_analysis_available = not model_row.empty
        epidemiology_available = not risk_row.empty
        if audit.status == "unavailable":
            classification = "unavailable"
        elif audit.status != "adequate":
            classification = "fragile"
        elif not model_analysis_available and not epidemiology_available:
            classification = "unresolved_analysis_underpowered_or_failed"
        elif (
            model_robust
            and epi_positive
            and sensitivity_consistent
            and external_robust
        ):
            classification = "robust_positive"
        elif model_suggestive or epi_positive:
            classification = "suggestive_positive"
        else:
            classification = "null_or_no_incremental_value"
        rows.append(
            {
                "dataset": dataset,
                "endpoint": endpoint,
                "tier": getattr(audit, "tier", ""),
                "endpoint_status": audit.status,
                "classification": classification,
                "incremental_auroc_ci_excludes_zero": bool(model_auc_positive),
                "incremental_auprc_ci_excludes_zero": bool(model_auprc_positive),
                "incremental_bootstrap_at_least_80pct_valid": bool(
                    model_bootstrap_valid
                ),
                "incremental_epv_adequate_for_claim": bool(model_epv_adequate),
                "epidemiology_cluster_rr_ci_excludes_one": bool(epi_positive),
                "external_incremental_auroc_ci_excludes_zero": bool(
                    external_auc_positive
                ),
                "external_incremental_auprc_ci_excludes_zero": bool(
                    external_auprc_positive
                ),
                "external_bootstrap_at_least_80pct_valid": bool(
                    external_bootstrap_valid
                ),
                "external_training_epv_adequate_for_claim": bool(
                    external_epv_adequate
                ),
                "external_validation_required_for_robust": True,
                "incremental_model_analysis_available": bool(
                    model_analysis_available
                ),
                "epidemiology_analysis_available": bool(epidemiology_available),
                "sensitivity_estimates": int(effects.notna().sum()),
                "sensitivity_fraction_direction_gt_1": sensitivity_fraction,
                "sensitivity_consistent": bool(sensitivity_consistent),
                "sensitivity_consistency_rule": "at_least_3_dynamics_exposure_estimates_and_at_least_75_percent_rr_gt_1",
                "claim_scope": "observational_prediction_and_landmark_ordering_not_causal",
            }
        )
    return pd.DataFrame(rows)


def write_oof_performance_figures(curves: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write ROC/PR/calibration panels for every fitted per-dataset endpoint."""
    if curves.empty or "curve" not in curves:
        return []
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    per_dataset = curves.loc[~curves["analysis_scope"].astype(str).eq("pooled_secondary")]
    for (scope, outcome), group in per_dataset.groupby(["analysis_scope", "outcome"]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for model, model_group in group.groupby("model"):
            for axis, curve_name in zip(axes, ("roc", "precision_recall", "calibration")):
                selected = model_group.loc[model_group["curve"].eq(curve_name)]
                if not selected.empty:
                    axis.plot(selected["x"], selected["y"], marker="o" if curve_name == "calibration" else None, label=model)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[2].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        axes[2].set(xlabel="Mean predicted risk", ylabel="Observed risk", title="Calibration")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{scope}: {outcome} (patient-grouped OOF)")
        fig.tight_layout()
        safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in f"{scope}_{outcome}")
        path = output_dir / f"oof_{safe}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_spo2_figures(analysis_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write compact diagnostic PNGs when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    available_outcomes = _available_outcomes(analysis_df)
    outcome = PRIMARY_ENDPOINT if PRIMARY_ENDPOINT in analysis_df.columns else (
        available_outcomes[0] if available_outcomes else ""
    )
    if outcome not in analysis_df.columns:
        return figures

    for metric, filename, ylabel in [
        ("spo2_below_90_fraction", "spo2_low_burden_by_outcome.png", "Fraction below 90%"),
        ("spo2_rmssd", "spo2_variability_by_outcome.png", "RMSSD"),
        ("spo2_sampling_density_per_hr", "spo2_sampling_density_by_outcome.png", "Measurements per hour"),
    ]:
        if metric not in analysis_df.columns:
            continue
        groups = []
        labels = []
        for value, group in analysis_df.groupby(outcome):
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            if not series.empty:
                groups.append(series.to_numpy())
                labels.append(str(value))
        if not groups:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        # matplotlib >= 3.9 renamed `labels` -> `tick_labels`; support both.
        try:
            ax.boxplot(groups, tick_labels=labels, showfliers=False)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(groups, labels=labels, showfliers=False)
        ax.set_title(f"{metric} by {outcome}")
        ax.set_xlabel(outcome)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    return figures


def write_archive_candidates(project_root: Path, output_dir: Path) -> Path:
    """Record legacy/V2 files to archive after the clean path is accepted."""
    candidates = [
        "Old",
        "PhysioGraph_Final_V2.ipynb",
        "PhysioGraph_Final_V2.ipynb - Colab.pdf",
        "PhysioGraph_V2_PLAN.md",
        "PhysioGraph_HF_Shock.ipynb",
        "PhysioGraph_HF_Shock.ipynb - Colab.pdf",
        "PhysioGraph_External_Pipeline.ipynb",
        "PhysioGraph_External_Pipeline.ipynb - Colab.pdf",
        "processed_data/hf_shock/physiograph_v2",
    ]
    rows = []
    for rel in candidates:
        path = project_root / rel
        rows.append(
            {
                "path": rel,
                "exists": path.exists(),
                "type": "directory" if path.is_dir() else "file",
                "action_after_acceptance": "archive_or_delete_after_clean_notebook_verification",
            }
        )
    out = output_dir / "archive_candidates.json"
    _write_json(out, rows)
    return out


def run_model_minimal_epidemiology(
    analysis_df: pd.DataFrame,
    all_events: pd.DataFrame,
    all_cohorts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Model-minimal companion analyses (ANALYSIS_PLAN.md section 8).

    Classical epidemiological estimators only — no fitted model: risk
    ratios with cluster-bootstrap CIs, Mantel-Haenszel pooled ORs,
    Cochran-Armitage dose-response, paired precedence sign tests, and a
    specificity matrix with a prespecified bilirubin comparator.
    """
    from physiograph.analysis.spo2_epidemiology import run_epidemiology_analyses

    return run_epidemiology_analyses(analysis_df, all_events, all_cohorts)



def _checkpoint_csv(output_dir: Path, name: str, frame: pd.DataFrame) -> None:
    """Write a stage result immediately so partial runs keep completed stages."""
    try:
        path = output_dir / f"{name}.csv"
        _atomic_to_csv(frame, path)
        print(f"[checkpoint] {name}: {len(frame)} rows -> {path.name}", flush=True)
    except Exception as exc:  # checkpointing must never kill the run
        print(f"[checkpoint] {name}: FAILED ({exc})", flush=True)


def run_spo2_drilldown(
    dataset_artifacts: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    *,
    mimic_root: str | Path | None = None,
    build_new: bool = False,
    fresh_colab_execution: bool = False,
    execution_environment: str = "local",
    run_comparator_requested: bool = False,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 250_000,
    require_all_requested_datasets: bool = True,
    analysis_only: bool = False,
) -> dict[str, Any]:
    """Run the SpO2 drilldown across available datasets and write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_code_hashes = _current_project_code_hashes()
    frames: list[pd.DataFrame] = []
    availability: dict[str, Any] = {}
    for dataset, artifacts in dataset_artifacts.items():
        frame, dataset_availability = assemble_spo2_analysis_frame(
            cohort=artifacts["cohort"],
            events=artifacts["events"],
            features=artifacts["features"],
            labels=artifacts["labels"],
        )
        frames.append(frame)
        availability[dataset] = dataset_availability

    analysis_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raw_spo2 = collect_raw_spo2_events(dataset_artifacts)
    all_events = pd.concat(
        [artifacts["events"] for artifacts in dataset_artifacts.values()],
        ignore_index=True,
    )
    all_cohorts = pd.concat(
        [artifacts["cohort"] for artifacts in dataset_artifacts.values()],
        ignore_index=True,
    )
    locked_events = all_events
    has_direct_mimic_ventilation = bool(
        (
            all_events.get("dataset", pd.Series(dtype=str))
            .astype(str)
            .eq("mimic")
            & all_events.get("concept", pd.Series(dtype=str))
            .astype(str)
            .eq("mechanical_ventilation")
        ).any()
    )
    if mimic_root is not None and not has_direct_mimic_ventilation:
        harmonized_cohort, _ = harmonize_hf_cohort(all_cohorts)
        direct_respiratory, _ = extract_mimic_respiratory_procedure_events(
            Path(mimic_root).expanduser().resolve(),
            harmonized_cohort,
        )
        locked_events = pd.concat(
            [all_events, direct_respiratory], ignore_index=True, sort=False
        ).drop_duplicates(
            [
                "dataset",
                "stay_id",
                "concept",
                "source_table",
                "offset_minutes",
                "value_numeric",
            ],
            keep="first",
        )
    descriptive = build_descriptive_summary(analysis_df)
    _checkpoint_csv(output_dir, "spo2_descriptive_summary", descriptive)
    lactate_negative = build_lactate_negative_summary(analysis_df)
    _checkpoint_csv(output_dir, "spo2_lactate_negative_summary", lactate_negative)
    association = build_spo2_association_proxy(analysis_df)
    _checkpoint_csv(output_dir, "spo2_association_proxy", association)
    model_metrics = fit_spo2_models(analysis_df)
    _checkpoint_csv(output_dir, "spo2_model_metrics", model_metrics)
    oof_predictions = model_metrics.attrs.get("oof_predictions", pd.DataFrame())
    oof_curves = build_oof_performance_curves(oof_predictions)
    _checkpoint_csv(output_dir, "spo2_oof_predictions", oof_predictions)
    _checkpoint_csv(output_dir, "spo2_oof_performance_curves", oof_curves)
    or_pvalues = fit_spo2_or_pvalue_tables(analysis_df)
    _checkpoint_csv(output_dir, "spo2_or_pvalue_tables", or_pvalues)
    respiratory_context = build_respiratory_context_tables(analysis_df)
    _checkpoint_csv(output_dir, "spo2_respiratory_context", respiratory_context)
    availability_audit = build_availability_audit(dataset_artifacts, analysis_df)
    _checkpoint_csv(output_dir, "spo2_availability_audit", availability_audit)
    cross_dataset = fit_external_cross_dataset_holdout(analysis_df)
    _checkpoint_csv(output_dir, "spo2_external_cross_dataset_holdout", cross_dataset)
    negative_control = fit_missingness_negative_control(analysis_df)
    _checkpoint_csv(output_dir, "spo2_missingness_negative_control", negative_control)
    variability_or = fit_spo2_variability_or_tables(analysis_df)
    _checkpoint_csv(output_dir, "spo2_variability_or_tables", variability_or)
    variability_summary = build_spo2_variability_group_summary(analysis_df)
    _checkpoint_csv(output_dir, "spo2_variability_group_summary", variability_summary)
    # The compatibility table is the same prespecified 12/24-hour model fit.
    # Reuse its OOF estimates instead of silently performing a second stochastic fit.
    dragged_models = model_metrics.copy()
    dragged_models["analysis"] = "prespecified_12_24h_incremental_value"
    _checkpoint_csv(output_dir, "spo2_dragged_horizon_models", dragged_models)
    raw_event_summary = build_spo2_raw_event_summary(raw_spo2, analysis_df)
    _checkpoint_csv(output_dir, "spo2_raw_event_summary", raw_event_summary)
    trajectory_summary = build_spo2_trajectory_summary(raw_spo2, analysis_df)
    _checkpoint_csv(output_dir, "spo2_trajectory_summary", trajectory_summary)
    continuous_trajectory_associations = build_continuous_trajectory_associations(
        analysis_df
    )
    _checkpoint_csv(
        output_dir,
        "spo2_continuous_trajectory_associations",
        continuous_trajectory_associations,
    )
    endpoint_audit = build_endpoint_completeness_audit(analysis_df)
    _checkpoint_csv(output_dir, "spo2_endpoint_completeness_early", endpoint_audit)
    cohort_flow = build_cohort_flow_table(dataset_artifacts, analysis_df)
    table1 = build_table1(analysis_df)
    feature_missingness = build_feature_missingness_table(analysis_df)
    _checkpoint_csv(output_dir, "cohort_flow", cohort_flow)
    _checkpoint_csv(output_dir, "table1_baseline_characteristics", table1)
    _checkpoint_csv(output_dir, "feature_missingness_by_dataset", feature_missingness)
    leadtime_records, leadtime_summary = build_temporal_precedence(all_events, all_cohorts)
    _checkpoint_csv(output_dir, "spo2_leadtime_summary", leadtime_summary)
    epidemiology = run_model_minimal_epidemiology(analysis_df, all_events, all_cohorts)
    for _table_name, _table_frame in epidemiology.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    lactate_episode = run_lactate_episode_analyses(
        all_events,
        all_cohorts,
        analysis_df,
    )
    for _table_name, _table_frame in lactate_episode.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    multiorgan_episode = run_multiorgan_episode_analyses(
        all_events,
        all_cohorts,
        analysis_df,
    )
    for _table_name, _table_frame in multiorgan_episode.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    locked_external = run_locked_external_replication(
        locked_events,
        all_cohorts,
        analysis_df,
        bootstrap_repetitions=0,
    )
    for _table_name, _table_frame in locked_external.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    advanced_episode = run_advanced_episode_inference(
        all_events,
        multiorgan_episode["multiorgan_episode_records"],
        multiorgan_episode["multiorgan_episode_key_results"],
        all_cohorts,
    )
    for _table_name, _table_frame in advanced_episode.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    biomarker_benchmark = run_biomarker_benchmark(
        all_events,
        analysis_df,
    )
    for _table_name, _table_frame in biomarker_benchmark.items():
        _checkpoint_csv(output_dir, f"ckpt_{_table_name}", _table_frame)
    endpoint_conclusions = build_endpoint_conclusions(
        endpoint_audit,
        model_metrics,
        epidemiology["stratified_risk_tables"],
        epidemiology["sensitivity_matrix"],
        cross_dataset,
    )
    _checkpoint_csv(output_dir, "endpoint_conclusion_matrix", endpoint_conclusions)
    figures = write_spo2_figures(analysis_df, output_dir / "figures")
    figures.extend(
        write_oof_performance_figures(
            oof_curves,
            output_dir / "figures" / "model_performance",
        )
    )

    paths = {
        "analysis_frame": output_dir / "spo2_analysis_frame.csv",
        "descriptive_summary": output_dir / "spo2_descriptive_summary.csv",
        "lactate_negative_summary": output_dir / "spo2_lactate_negative_results.csv",
        "association_proxy": output_dir / "spo2_signal_association_proxy.csv",
        "model_metrics": output_dir / "spo2_model_metrics.csv",
        "oof_predictions": output_dir / "spo2_oof_predictions.csv",
        "oof_curves": output_dir / "spo2_oof_roc_pr_calibration_curves.csv",
        "or_pvalues": output_dir / "spo2_or_pvalues_adjusted_per_feature.csv",
        "respiratory_context": output_dir / "spo2_respiratory_context.csv",
        "availability_audit": output_dir / "spo2_availability_audit.csv",
        "cross_dataset_holdout": output_dir / "spo2_external_cross_dataset.csv",
        "negative_control": output_dir / "spo2_missingness_negative_control.csv",
        "variability_or_models": output_dir / "spo2_variability_or_ci_models.csv",
        "variability_group_summary": output_dir / "spo2_variability_group_summary.csv",
        "dragged_models": output_dir / "spo2_horizon_dragged_model_metrics.csv",
        "raw_event_summary": output_dir / "spo2_raw_event_summary.csv",
        "trajectory_summary": output_dir / "spo2_trajectory_by_outcome.csv",
        "continuous_trajectory_associations": output_dir / "spo2_continuous_trajectory_associations.csv",
        "endpoint_completeness": output_dir / "spo2_endpoint_completeness.csv",
        "endpoint_conclusions": output_dir / "endpoint_conclusion_matrix.csv",
        "cohort_flow": output_dir / "cohort_flow.csv",
        "table1": output_dir / "table1_baseline_characteristics.csv",
        "feature_missingness": output_dir / "feature_missingness_by_dataset.csv",
        "leadtime_records": output_dir / "spo2_leadtime_records.csv",
        "leadtime_summary": output_dir / "spo2_leadtime_summary.csv",
        "epi_risk_tables": output_dir / "epi_stratified_risk_tables.csv",
        "epi_mantel_haenszel": output_dir / "epi_mantel_haenszel.csv",
        "epi_dose_response": output_dir / "epi_dose_response.csv",
        "epi_paired_precedence": output_dir / "epi_paired_precedence.csv",
        "epi_specificity_matrix": output_dir / "epi_specificity_matrix.csv",
        "sensitivity_matrix": output_dir / "spo2_sensitivity_matrix_s1_s12.csv",
        "lactate_episode_records": output_dir / "spo2_lactate_episode_records.csv",
        "lactate_episode_paired_summary": output_dir / "spo2_lactate_episode_paired_summary.csv",
        "lactate_episode_controlled_summary": output_dir / "spo2_lactate_episode_controlled_summary.csv",
        "lactate_episode_observation_process": output_dir / "spo2_lactate_episode_observation_process.csv",
        "lactate_episode_measurement_weighted": output_dir / "spo2_lactate_episode_measurement_weighted.csv",
        "lactate_episode_stratified_sensitivity": output_dir / "spo2_lactate_episode_stratified_sensitivity.csv",
        "lactate_episode_meta_analysis": output_dir / "spo2_lactate_episode_meta_analysis.csv",
        "lactate_episode_evidence_summary": output_dir / "spo2_lactate_episode_evidence_summary.csv",
        "multiorgan_episode_records": output_dir / "spo2_multiorgan_episode_records.csv",
        "multiorgan_episode_effects": output_dir / "spo2_multiorgan_episode_effects.csv",
        "multiorgan_episode_observation_process": output_dir / "spo2_multiorgan_episode_observation_process.csv",
        "multiorgan_episode_measurement_weighted": output_dir / "spo2_multiorgan_episode_measurement_weighted.csv",
        "multiorgan_episode_meta_analysis": output_dir / "spo2_multiorgan_episode_meta_analysis.csv",
        "multiorgan_episode_evidence_summary": output_dir / "spo2_multiorgan_episode_evidence_summary.csv",
        "multiorgan_episode_key_results": output_dir / "spo2_multiorgan_episode_key_results.csv",
        "multiorgan_episode_endpoint_definitions": output_dir / "spo2_multiorgan_episode_endpoint_definitions.csv",
        "multiorgan_episode_availability": output_dir / "spo2_multiorgan_episode_availability.csv",
        "advanced_episode_preanchor_covariates": output_dir / "spo2_advanced_episode_preanchor_covariates.csv",
        "advanced_episode_anchor_weights": output_dir / "spo2_advanced_episode_anchor_weights.csv",
        "advanced_episode_covariate_balance": output_dir / "spo2_advanced_episode_covariate_balance.csv",
        "advanced_episode_overlap_summary": output_dir / "spo2_advanced_episode_overlap_summary.csv",
        "advanced_episode_overlap_weighted_effects": output_dir / "spo2_advanced_episode_overlap_weighted_effects.csv",
        "advanced_episode_aipw_effects": output_dir / "spo2_advanced_episode_aipw_effects.csv",
        "advanced_episode_continuous_aipw_effects": output_dir / "spo2_advanced_episode_continuous_aipw_effects.csv",
        "advanced_episode_site_effects": output_dir / "spo2_advanced_episode_site_effects.csv",
        "advanced_episode_multicenter_summary": output_dir / "spo2_advanced_episode_multicenter_summary.csv",
        "advanced_episode_evalues": output_dir / "spo2_advanced_episode_evalues.csv",
        "advanced_episode_key_results": output_dir / "spo2_advanced_episode_key_results.csv",
        "advanced_episode_design": output_dir / "spo2_advanced_episode_design.csv",
        "availability": output_dir / "spo2_adjustment_availability.json",
        "claims_linter": output_dir / "claims_linter_warnings.csv",
        "manifest": output_dir / "manifest.json",
    }
    paths.update(
        {
            name: output_dir / f"spo2_{name}.csv"
            for name in locked_external
        }
    )
    paths.update(
        {
            name: output_dir / f"spo2_{name}.csv"
            for name in biomarker_benchmark
        }
    )
    output_frames = {
        "analysis_frame": analysis_df,
        "descriptive_summary": descriptive,
        "lactate_negative_summary": lactate_negative,
        "association_proxy": association,
        "model_metrics": model_metrics,
        "oof_predictions": oof_predictions,
        "oof_curves": oof_curves,
        "or_pvalues": or_pvalues,
        "respiratory_context": respiratory_context,
        "availability_audit": availability_audit,
        "cross_dataset_holdout": cross_dataset,
        "negative_control": negative_control,
        "variability_or_models": variability_or,
        "variability_group_summary": variability_summary,
        "dragged_models": dragged_models,
        "raw_event_summary": raw_event_summary,
        "trajectory_summary": trajectory_summary,
        "continuous_trajectory_associations": continuous_trajectory_associations,
        "endpoint_completeness": endpoint_audit,
        "endpoint_conclusions": endpoint_conclusions,
        "cohort_flow": cohort_flow,
        "table1": table1,
        "feature_missingness": feature_missingness,
        "leadtime_records": leadtime_records,
        "leadtime_summary": leadtime_summary,
        "epi_risk_tables": epidemiology["stratified_risk_tables"],
        "epi_mantel_haenszel": epidemiology["mantel_haenszel"],
        "epi_dose_response": epidemiology["dose_response"],
        "epi_paired_precedence": epidemiology["paired_precedence"],
        "epi_specificity_matrix": epidemiology["specificity_matrix"],
        "sensitivity_matrix": epidemiology["sensitivity_matrix"],
        **lactate_episode,
        **multiorgan_episode,
        **locked_external,
        **advanced_episode,
        **biomarker_benchmark,
    }
    for name, frame in output_frames.items():
        _atomic_to_csv(frame, paths[name])
    _write_json(paths["availability"], availability)

    claims_warnings = lint_claims_and_outputs(
        model_metrics=model_metrics,
        manifest={
            "fresh_colab_execution": bool(fresh_colab_execution),
            "build_new": bool(build_new),
            "analysis_only": bool(analysis_only),
            "execution_environment": execution_environment,
        },
        output_paths=paths,
    )
    endpoint_warnings = endpoint_audit.loc[
        endpoint_audit.get("tier", pd.Series(index=endpoint_audit.index, dtype=str)).eq("primary")
        & ~endpoint_audit.get("status", pd.Series(index=endpoint_audit.index, dtype=str)).eq("adequate")
    ]
    if not endpoint_warnings.empty:
        coverage_claims = pd.DataFrame(
            [
                {
                    "severity": "warning",
                    "check": "primary_endpoint_not_adequately_observed",
                    "detail": f"{row.dataset}: {row.endpoint} ({row.status})",
                }
                for row in endpoint_warnings.itertuples()
            ]
        )
        claims_warnings = pd.concat([claims_warnings, coverage_claims], ignore_index=True)
    manifest = build_analysis_manifest(
        build_new=build_new,
        fresh_colab_execution=fresh_colab_execution,
        paths=paths,
        datasets=sorted(dataset_artifacts),
        rows=int(len(analysis_df)),
        claims_warnings=claims_warnings,
        execution_environment=execution_environment,
        run_comparator_requested=run_comparator_requested,
        max_stays=max_stays,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        require_all_requested_datasets=require_all_requested_datasets,
        analysis_only=analysis_only,
    )
    manifest["figures"] = [str(path) for path in figures]
    manifest["raw_spo2_event_rows"] = int(len(raw_spo2))
    manifest["post_landmark_horizons_hours"] = list(POST_LANDMARK_HORIZONS_HOURS)
    manifest["exploratory_long_horizons_hours_from_icu_admission"] = list(HORIZON_HOURS)
    manifest["endpoint_definition_version"] = "spo2_protocol_v2.3_post_result_amendment"
    manifest["primary_endpoints"] = list(PRIMARY_ENDPOINTS)
    manifest["secondary_endpoints"] = list(SECONDARY_ENDPOINTS)
    manifest["leadtime_claim_scope"] = "landmark_ordering_design_enforced_not_causal_precedence"
    manifest["epidemiology_layer"] = "model_minimal_companion_v1"
    manifest["epidemiology_specificity_comparator_endpoint"] = "hepatic_lab_worsening_12h_flag"
    manifest["sensitivity_axes_completed"] = [f"S{index}" for index in range(1, 13)]
    manifest["episode_anchored_lactate_amendment"] = {
        "status": "post_result_protocol_amendment_exploratory",
        "exposure_anchor": "actual_gap_qualified_spo2_transition_in_first_4h",
        "lactate_baseline": "last_lactate_strictly_before_episode_within_6h",
        "lag_windows": ["0_to_1h", "1_to_8h", "1_to_2h", "2_to_4h", "4_to_8h", "8_to_12h", "0_to_8h"],
        "focused_family": "acute_0_to_1h_and_delayed_1_to_8h_two_window_family",
        "kinetic_localization_family": "1_to_2h_2_to_4h_4_to_8h_secondary",
        "control_anchor": "nearest_transition_to_exposed_anchor_time_quantile",
        "observation_bias_sensitivity": "inverse_probability_of_post_lactate_remeasurement_conditional_on_strict_prior_lactate",
        "directional_sensitivities": ["desaturation_drop_ge3", "desaturation_drop_ge5", "recovery_rise_ge4"],
        "resolution_sensitivity": "same_time_deduplicated_raw_vs_15_minute_median_bins",
        "claim_scope": "observational_not_causal_not_prospectively_preregistered",
    }
    manifest["episode_anchored_multiorgan_amendment"] = {
        "status": "post_result_multiorgan_episode_amendment_exploratory",
        "exposure_anchor": "actual_gap_qualified_spo2_transition_in_first_4h",
        "control_anchor": "nearest_transition_to_exposed_anchor_time_quantile",
        "endpoint_specific_focused_windows": True,
        "nonoverlapping_and_cumulative_lag_grid": True,
        "strict_preanchor_laboratory_baselines": True,
        "incident_intervention_risk_sets": True,
        "continuous_and_thresholded_estimands": True,
        "measurement_and_censoring_weighted_sensitivity": True,
        "cross_endpoint_fdr_reported": True,
        "claim_scope": "observational_not_causal_not_prospectively_preregistered",
    }
    manifest["episode_anchored_advanced_inference"] = {
        "status": "post_result_advanced_episode_robustness_amendment_exploratory",
        "raw_data_rebuilt": False,
        "strict_preanchor_covariates": True,
        "concurrent_spo2_in_propensity_model": False,
        "concurrent_spo2_in_observation_and_outcome_models": True,
        "exposure_balance": "outcome_independent_overlap_weights_with_full_smd_audit",
        "doubly_robust_estimator": "bounded_cross_validated_tmle_with_one_step_aipw_diagnostic",
        "continuous_estimand": "crossfit_aipw_mean_worsening_in_clinical_units",
        "outcome_observation_model": "crossfit_inverse_probability_correction",
        "multicenter_method": "hospital_specific_rr_paule_mandel_hksj_and_leave_one_hospital_out",
        "residual_confounding_sensitivity": "e_values_for_each_estimable_focused_rr",
        "multiplicity": "within_endpoint_and_cross_endpoint_BH_per_dataset",
        "claim_scope": "observational_not_causal_not_prospectively_preregistered",
    }
    manifest["locked_external_validation"] = {
        "status": "completed_external_replication_after_eicu_discovery",
        "cohort": "adult_explicit_hf_and_(icu_within_24h_or_cardiogenic_shock)",
        "exposure": "absolute_jump_ge4_in_15_minute_median_bins",
        "lag_window": "4_to_12h_after_episode_anchor",
        "transferred_primary_endpoint": "invasive_ventilation_initiation",
        "mimic_primary_source": (
            "procedureevents_itemid_225792_invasive_ventilation_interval_start"
        ),
        "mimic_specificity_endpoint": "procedureevents_itemid_224385_intubation",
        "eicu_primary_source": (
            "treatment_literal_pipe_delimited_mechanical_ventilation_segment"
        ),
        "eicu_primary_exclusions": [
            "noninvasive_ventilation",
            "ventilator_weaning",
            "nonliteral_substring_matches",
        ],
        "eicu_time_semantics": "first_treatment_documentation_proxy_not_true_start",
        "secondary_multiplicity": "BH_within_dataset_and_endpoint_tier",
        "claim_scope": (
            "observational_external_replication_after_eicu_discovery_not_causal"
        ),
    }
    manifest["head_to_head_biomarker_benchmark"] = {
        "status": "completed",
        "observation_window": "[0,240)_minutes_after_icu_admission",
        "outcome_clock": "(240,240+12h]_and_(240,240+24h]",
        "comparators": [
            "absolute_spo2",
            "spo2_instability",
            "sbp",
            "lactate",
            "kapur_cswg_scai_stage",
        ],
        "kapur_reference": (
            "Kapur_NK_et_al_JACC_2022_80_3_185_198_"
            "doi_10.1016/j.jacc.2022.04.049"
        ),
        "evaluation": (
            "identical_patient_grouped_oof_folds_paired_cluster_bootstrap_"
            "and_bidirectional_external_database_transportability"
        ),
        "scai_claim_scope": (
            "four_hour_ehr_operationalization_not_clinician_adjudication_"
            "and_not_complete_without_OHCA"
        ),
        "eicu_aperiodic_bp_source_available": bool(
            "vitalAperiodic.csv"
            in set(all_events.get("source_table", pd.Series(dtype=str)).astype(str))
        ),
        "claim_scope": "prognostic_association_and_prediction_not_causal",
    }
    _atomic_to_csv(claims_warnings, paths["claims_linter"])
    if _current_project_code_hashes() != analysis_code_hashes:
        raise RuntimeError(
            "PhysioGraph source code changed during analysis; manifest not committed."
        )
    manifest["code_sha256"] = analysis_code_hashes
    input_manifests: dict[str, Any] = {}
    for dataset in sorted(dataset_artifacts):
        input_path = output_dir.parent / dataset / "manifest.json"
        if not input_path.is_file():
            raise RuntimeError(f"Missing committed ETL manifest for {dataset}: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        input_manifests[dataset] = {
            "path": str(input_path),
            "sha256": _file_sha256(input_path),
            "run_id": payload.get("run_id"),
            "schema_version": payload.get("schema_version"),
        }
    manifest["input_manifests"] = input_manifests
    manifest["output_fingerprints"] = {
        name: {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
            "sha256": _file_sha256(path),
        }
        for name, path in paths.items()
        if name != "manifest" and path.is_file()
    }
    manifest["figure_fingerprints"] = [
        {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
            "sha256": _file_sha256(path),
        }
        for path in figures
        if path.is_file()
    ]
    # The manifest is the final commit marker for the complete analysis layer.
    _write_json(paths["manifest"], manifest)
    return {
        "output_dir": output_dir,
        "paths": paths,
        "figures": figures,
        "manifest": manifest,
    }


def refresh_lactate_episode_analysis(
    output_root: str | Path,
    *,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 500_000,
    bootstrap_repetitions: int = 500,
) -> dict[str, Any]:
    """Refresh only episode-anchored lactate outputs from immutable artifacts.

    This component-only path validates the committed cohort artifacts and the
    analysis frame, then regenerates the episode tables and atomically updates
    their manifest fingerprints.  It never invokes either raw-data extractor.
    """
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "lactate_episode_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "episode_anchored_lactate_mechanistic",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")

        analysis_path = output_dir / "spo2_analysis_frame.csv"
        analysis_fingerprint = manifest.get("output_fingerprints", {}).get(
            "analysis_frame", {}
        )
        if not analysis_path.is_file():
            raise FileNotFoundError(f"Missing cached analysis frame: {analysis_path}")
        if (
            int(analysis_fingerprint.get("size_bytes", -1))
            != int(analysis_path.stat().st_size)
            or analysis_fingerprint.get("sha256") != _file_sha256(analysis_path)
        ):
            raise RuntimeError("Cached analysis frame failed manifest integrity validation")

        events: list[pd.DataFrame] = []
        cohorts: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        data_roots = {
            "mimic": Path(mimic_root).expanduser().resolve() if mimic_root else None,
            "eicu": Path(eicu_root).expanduser().resolve() if eicu_root else None,
        }
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                data_root=data_roots[dataset],
                max_stays=max_stays,
                max_chunks=max_chunks,
                chunk_size=chunk_size,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            events.append(_read_csv(dataset_dir / "events.csv"))
            cohorts.append(_read_csv(dataset_dir / "cohort.csv"))

        analysis_frame = _read_csv(analysis_path)
        outputs = run_lactate_episode_analyses(
            pd.concat(events, ignore_index=True, sort=False),
            pd.concat(cohorts, ignore_index=True, sort=False),
            analysis_frame,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during component refresh"
            )

        filenames = {
            "lactate_episode_records": "spo2_lactate_episode_records.csv",
            "lactate_episode_paired_summary": "spo2_lactate_episode_paired_summary.csv",
            "lactate_episode_controlled_summary": "spo2_lactate_episode_controlled_summary.csv",
            "lactate_episode_observation_process": "spo2_lactate_episode_observation_process.csv",
            "lactate_episode_measurement_weighted": "spo2_lactate_episode_measurement_weighted.csv",
            "lactate_episode_stratified_sensitivity": "spo2_lactate_episode_stratified_sensitivity.csv",
            "lactate_episode_meta_analysis": "spo2_lactate_episode_meta_analysis.csv",
            "lactate_episode_evidence_summary": "spo2_lactate_episode_evidence_summary.csv",
        }
        output_paths = {name: output_dir / filename for name, filename in filenames.items()}
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "episode_anchored_lactate_mechanistic",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "input_analysis_frame_sha256": _file_sha256(analysis_path),
            }
        )
        manifest["episode_anchored_lactate_amendment"] = {
            "status": "post_result_protocol_amendment_exploratory",
            "exposure_anchor": "actual_gap_qualified_spo2_transition_in_first_4h",
            "control_anchor": "nearest_transition_to_exposed_anchor_time_quantile",
            "lactate_baseline": "last_lactate_strictly_before_episode_within_6h",
            "focused_family": "acute_0_to_1h_and_delayed_1_to_8h_two_window_family",
            "kinetic_localization_family": "1_to_2h_2_to_4h_4_to_8h_secondary",
            "measurement_bias_sensitivity": (
                "inverse_probability_of_post_lactate_remeasurement_"
                "conditional_on_strict_prior_lactate"
            ),
            "claim_scope": "observational_not_causal_not_prospectively_preregistered",
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )

        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_multiorgan_episode_analysis(
    output_root: str | Path,
    *,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 500_000,
    bootstrap_repetitions: int = 500,
) -> dict[str, Any]:
    """Refresh all episode-anchored endpoints without invoking raw ETL."""
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "multiorgan_episode_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "episode_anchored_multiorgan_mechanistic",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")
        analysis_path = output_dir / "spo2_analysis_frame.csv"
        analysis_fingerprint = manifest.get("output_fingerprints", {}).get(
            "analysis_frame", {}
        )
        if not analysis_path.is_file():
            raise FileNotFoundError(f"Missing cached analysis frame: {analysis_path}")
        if (
            int(analysis_fingerprint.get("size_bytes", -1))
            != int(analysis_path.stat().st_size)
            or analysis_fingerprint.get("sha256") != _file_sha256(analysis_path)
        ):
            raise RuntimeError("Cached analysis frame failed manifest integrity validation")

        events: list[pd.DataFrame] = []
        cohorts: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        data_roots = {
            "mimic": Path(mimic_root).expanduser().resolve() if mimic_root else None,
            "eicu": Path(eicu_root).expanduser().resolve() if eicu_root else None,
        }
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                data_root=data_roots[dataset],
                max_stays=max_stays,
                max_chunks=max_chunks,
                chunk_size=chunk_size,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            events.append(_read_csv(dataset_dir / "events.csv"))
            cohorts.append(_read_csv(dataset_dir / "cohort.csv"))

        outputs = run_multiorgan_episode_analyses(
            pd.concat(events, ignore_index=True, sort=False),
            pd.concat(cohorts, ignore_index=True, sort=False),
            _read_csv(analysis_path),
            bootstrap_repetitions=bootstrap_repetitions,
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during component refresh"
            )
        filenames = {
            "multiorgan_episode_records": "spo2_multiorgan_episode_records.csv",
            "multiorgan_episode_effects": "spo2_multiorgan_episode_effects.csv",
            "multiorgan_episode_observation_process": "spo2_multiorgan_episode_observation_process.csv",
            "multiorgan_episode_measurement_weighted": "spo2_multiorgan_episode_measurement_weighted.csv",
            "multiorgan_episode_meta_analysis": "spo2_multiorgan_episode_meta_analysis.csv",
            "multiorgan_episode_evidence_summary": "spo2_multiorgan_episode_evidence_summary.csv",
            "multiorgan_episode_key_results": "spo2_multiorgan_episode_key_results.csv",
            "multiorgan_episode_endpoint_definitions": "spo2_multiorgan_episode_endpoint_definitions.csv",
            "multiorgan_episode_availability": "spo2_multiorgan_episode_availability.csv",
        }
        output_paths = {
            name: output_dir / filename for name, filename in filenames.items()
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "episode_anchored_multiorgan_mechanistic",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "input_analysis_frame_sha256": _file_sha256(analysis_path),
            }
        )
        manifest["episode_anchored_multiorgan_amendment"] = {
            "status": "post_result_multiorgan_episode_amendment_exploratory",
            "exposure_definitions": [
                "absolute_jump_ge4_binned_primary",
                "desaturation_drop_ge3_binned",
                "desaturation_drop_ge5_binned",
                "recovery_rise_ge4_specificity",
                "absolute_jump_ge4_raw_resolution",
                "desaturation_drop_ge3_raw_resolution",
            ],
            "lag_design": "frozen_nonoverlapping_and_cumulative_grid_with_endpoint_specific_focused_pairs",
            "strict_baselines": "laboratory_and_relative_urine_outcomes_strictly_preanchor",
            "intervention_design": "incident_anchor_specific_risk_sets",
            "measurement_bias_sensitivity": "inverse_probability_of_endpoint_observation_or_censoring",
            "multiplicity": "endpoint_family_cross_endpoint_and_global_fdr_all_reported",
            "claim_scope": "observational_not_causal_not_prospectively_preregistered",
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def _mimic_respiratory_endpoint_inventory(
    respiratory_events: pd.DataFrame,
) -> pd.DataFrame:
    specifications = (
        (
            "invasive_ventilation_initiation",
            "mechanical_ventilation",
            "225792",
            "Invasive Ventilation",
            "procedure interval start",
            "transferred primary endpoint",
        ),
        (
            "intubation_initiation",
            "intubation",
            "224385",
            "Intubation",
            "procedure timestamp",
            "MIMIC respiratory specificity endpoint",
        ),
    )
    rows: list[dict[str, Any]] = []
    for endpoint, concept, itemid, label, timing, role in specifications:
        local = respiratory_events.loc[
            respiratory_events.get(
                "concept", pd.Series(index=respiratory_events.index, dtype=str)
            ).eq(concept)
        ].copy()
        offsets = pd.to_numeric(local.get("offset_minutes"), errors="coerce")
        rows.append(
            {
                "dataset": "mimic",
                "endpoint": endpoint,
                "concept": concept,
                "itemid": itemid,
                "dictionary_label": label,
                "source_table": "procedureevents.csv",
                "time_semantics": timing,
                "validation_role": role,
                "procedure_rows_in_harmonized_hf_cohort": int(len(local)),
                "stays_with_procedure": int(local.get("stay_id", pd.Series(dtype=float)).nunique()),
                "minimum_offset_minutes": float(offsets.min()) if offsets.notna().any() else np.nan,
                "maximum_offset_minutes": float(offsets.max()) if offsets.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def refresh_locked_external_validation(
    output_root: str | Path,
    *,
    mimic_root: str | Path,
    run_advanced: bool = True,
    bootstrap_repetitions: int = 0,
) -> dict[str, Any]:
    """Run the locked eICU-to-MIMIC test without rebuilding cached ETL.

    Only MIMIC ``d_items`` and ``procedureevents`` are read from the raw data
    root to recover direct ventilation/intubation starts.  The immutable cached
    cohort, feature, label, and event artifacts are never changed.
    """
    root = Path(output_root).expanduser().resolve()
    mimic_data_root = Path(mimic_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "locked_external_validation_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "locked_eicu_to_mimic_external_validation",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "targeted_raw_tables_read": ["d_items.csv", "procedureevents.csv"],
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")
        analysis_path = output_dir / "spo2_analysis_frame.csv"
        analysis_fingerprint = manifest.get("output_fingerprints", {}).get(
            "analysis_frame", {}
        )
        if not analysis_path.is_file() or (
            int(analysis_fingerprint.get("size_bytes", -1))
            != int(analysis_path.stat().st_size)
            or analysis_fingerprint.get("sha256") != _file_sha256(analysis_path)
        ):
            raise RuntimeError(
                "Cached analysis frame is missing or failed manifest validation"
            )
        source_paths = {
            name: mimic_data_root / name
            for name in ("d_items.csv", "procedureevents.csv")
        }
        missing_sources = [
            str(path) for path in source_paths.values() if not path.is_file()
        ]
        if missing_sources:
            raise FileNotFoundError(
                "Missing targeted MIMIC respiratory source(s): "
                + ", ".join(missing_sources)
            )

        event_frames: list[pd.DataFrame] = []
        cohort_frames: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            event_frames.append(_read_csv(dataset_dir / "events.csv"))
            cohort_frames.append(_read_csv(dataset_dir / "cohort.csv"))

        cohorts = pd.concat(cohort_frames, ignore_index=True, sort=False)
        harmonized_cohort, cohort_audit = harmonize_hf_cohort(cohorts)
        mimic_respiratory, extraction_counts = (
            extract_mimic_respiratory_procedure_events(
                mimic_data_root,
                harmonized_cohort,
            )
        )
        direct_starts = mimic_respiratory.loc[
            mimic_respiratory["concept"].isin(
                ["mechanical_ventilation", "intubation"]
            )
        ]
        if direct_starts.empty:
            raise RuntimeError(
                "Targeted MIMIC scan found no direct invasive-ventilation or "
                "intubation rows in the harmonized HF cohort"
            )
        cached_events = pd.concat(event_frames, ignore_index=True, sort=False)
        events = pd.concat(
            [cached_events, mimic_respiratory], ignore_index=True, sort=False
        ).drop_duplicates(
            [
                "dataset",
                "stay_id",
                "concept",
                "source_table",
                "offset_minutes",
                "value_numeric",
            ],
            keep="first",
        )
        outputs = run_locked_external_replication(
            events,
            harmonized_cohort,
            _read_csv(analysis_path),
            bootstrap_repetitions=bootstrap_repetitions,
        )
        # Preserve the pre-extraction audit and make direct source discovery an
        # explicit, inspectable result rather than an implicit code assumption.
        outputs["locked_external_validation_hf_cohort_audit"] = cohort_audit
        outputs["locked_external_validation_mimic_endpoint_inventory"] = (
            _mimic_respiratory_endpoint_inventory(mimic_respiratory)
        )

        if run_advanced:
            advanced = run_advanced_episode_inference(
                events,
                outputs["locked_external_validation_records"],
                outputs["locked_external_validation_key_results"],
                harmonized_cohort,
            )
            for name, frame in advanced.items():
                outputs[
                    name.replace(
                        "advanced_episode_",
                        "locked_external_validation_advanced_",
                    )
                ] = frame

        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during locked validation refresh"
            )
        output_paths = {
            name: output_dir / f"spo2_{name}.csv" for name in outputs
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        source_fingerprints = {
            name: {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
            for name, path in source_paths.items()
        }
        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "locked_eicu_to_mimic_external_validation",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "targeted_raw_tables_read": list(source_paths),
                "run_advanced": bool(run_advanced),
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "input_analysis_frame_sha256": _file_sha256(analysis_path),
                "mimic_respiratory_extraction_counts": extraction_counts,
            }
        )
        manifest["locked_external_validation"] = {
            "status": "completed_external_replication_after_eicu_discovery",
            "cohort": (
                "age_ge_18_and_explicit_hf_and_"
                "(early_icu_within_24h_or_cardiogenic_shock)"
            ),
            "exposure": "absolute_jump_ge4_in_15_minute_median_bins",
            "lag_window": "4_to_12h_after_episode_anchor",
            "transferred_primary_endpoint": "invasive_ventilation_initiation",
            "mimic_primary_source": (
                "procedureevents_itemid_225792_invasive_ventilation_interval_start"
            ),
            "mimic_specificity_endpoint": (
                "procedureevents_itemid_224385_intubation"
            ),
            "eicu_primary_source": (
                "treatment_literal_pipe_delimited_mechanical_ventilation_segment"
            ),
            "eicu_primary_exclusions": [
                "noninvasive_ventilation",
                "ventilator_weaning",
                "nonliteral_substring_matches",
            ],
            "eicu_time_semantics": (
                "first_treatment_documentation_proxy_not_true_start"
            ),
            "secondary_multiplicity": "BH_within_dataset_and_endpoint_tier",
            "raw_cached_artifacts_mutated": False,
            "advanced_all_endpoint_robustness": bool(
                run_advanced
                or manifest.get("locked_external_validation", {}).get(
                    "advanced_all_endpoint_robustness", False
                )
            ),
            "claim_scope": (
                "observational_external_replication_after_eicu_discovery_not_causal"
            ),
            "targeted_source_fingerprints": source_fingerprints,
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "mimic_respiratory_extraction_counts": extraction_counts,
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_multiorgan_weighted_analysis(
    output_root: str | Path,
) -> dict[str, Any]:
    """Refresh IPW/GEE and evidence grading from committed focused records."""
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "multiorgan_weighted_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "multiorgan_selection_weighted_sensitivity",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "focused_records_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")
        required = {
            "multiorgan_episode_records": "spo2_multiorgan_episode_records.csv",
            "multiorgan_episode_effects": "spo2_multiorgan_episode_effects.csv",
            "multiorgan_episode_meta_analysis": "spo2_multiorgan_episode_meta_analysis.csv",
        }
        inputs: dict[str, pd.DataFrame] = {}
        committed = manifest.get("output_fingerprints", {})
        for name, filename in required.items():
            path = output_dir / filename
            fingerprint = committed.get(name, {})
            if not path.is_file():
                raise FileNotFoundError(f"Missing committed analysis table: {path}")
            if (
                int(fingerprint.get("size_bytes", -1)) != int(path.stat().st_size)
                or fingerprint.get("sha256") != _file_sha256(path)
            ):
                raise RuntimeError(f"Committed {name} table failed integrity validation")
            inputs[name] = _read_csv(path)
        outputs = rebuild_multiorgan_weighted_sensitivity(
            inputs["multiorgan_episode_records"],
            inputs["multiorgan_episode_effects"],
            inputs["multiorgan_episode_meta_analysis"],
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during component refresh"
            )
        output_paths = {
            "multiorgan_episode_measurement_weighted": output_dir
            / "spo2_multiorgan_episode_measurement_weighted.csv",
            "multiorgan_episode_evidence_summary": output_dir
            / "spo2_multiorgan_episode_evidence_summary.csv",
            "multiorgan_episode_key_results": output_dir
            / "spo2_multiorgan_episode_key_results.csv",
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)
        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "multiorgan_selection_weighted_sensitivity",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "focused_records_rebuilt": False,
                "weighted_inference": "cluster_robust_gee_poisson_log_link",
            }
        )
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        manifest.setdefault("episode_anchored_multiorgan_amendment", {})[
            "weighted_inference"
        ] = "cluster_robust_gee_poisson_log_link"
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_advanced_episode_inference(
    output_root: str | Path,
    *,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 500_000,
) -> dict[str, Any]:
    """Refresh advanced inference from committed events and episode records only."""
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "advanced_episode_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "episode_anchored_advanced_inference",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "focused_records_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")

        required = {
            "multiorgan_episode_records": "spo2_multiorgan_episode_records.csv",
            "multiorgan_episode_key_results": "spo2_multiorgan_episode_key_results.csv",
        }
        committed = manifest.get("output_fingerprints", {})
        analysis_inputs: dict[str, pd.DataFrame] = {}
        for name, filename in required.items():
            path = output_dir / filename
            fingerprint = committed.get(name, {})
            if not path.is_file():
                raise FileNotFoundError(f"Missing committed analysis table: {path}")
            if (
                int(fingerprint.get("size_bytes", -1)) != int(path.stat().st_size)
                or fingerprint.get("sha256") != _file_sha256(path)
            ):
                raise RuntimeError(f"Committed {name} table failed integrity validation")
            analysis_inputs[name] = _read_csv(path)

        events: list[pd.DataFrame] = []
        cohorts: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        data_roots = {
            "mimic": Path(mimic_root).expanduser().resolve() if mimic_root else None,
            "eicu": Path(eicu_root).expanduser().resolve() if eicu_root else None,
        }
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                data_root=data_roots[dataset],
                max_stays=max_stays,
                max_chunks=max_chunks,
                chunk_size=chunk_size,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            events.append(_read_csv(dataset_dir / "events.csv"))
            cohorts.append(_read_csv(dataset_dir / "cohort.csv"))

        outputs = run_advanced_episode_inference(
            pd.concat(events, ignore_index=True, sort=False),
            analysis_inputs["multiorgan_episode_records"],
            analysis_inputs["multiorgan_episode_key_results"],
            pd.concat(cohorts, ignore_index=True, sort=False),
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during component refresh"
            )
        filenames = {
            "advanced_episode_preanchor_covariates": "spo2_advanced_episode_preanchor_covariates.csv",
            "advanced_episode_anchor_weights": "spo2_advanced_episode_anchor_weights.csv",
            "advanced_episode_covariate_balance": "spo2_advanced_episode_covariate_balance.csv",
            "advanced_episode_overlap_summary": "spo2_advanced_episode_overlap_summary.csv",
            "advanced_episode_overlap_weighted_effects": "spo2_advanced_episode_overlap_weighted_effects.csv",
            "advanced_episode_aipw_effects": "spo2_advanced_episode_aipw_effects.csv",
            "advanced_episode_continuous_aipw_effects": "spo2_advanced_episode_continuous_aipw_effects.csv",
            "advanced_episode_site_effects": "spo2_advanced_episode_site_effects.csv",
            "advanced_episode_multicenter_summary": "spo2_advanced_episode_multicenter_summary.csv",
            "advanced_episode_evalues": "spo2_advanced_episode_evalues.csv",
            "advanced_episode_key_results": "spo2_advanced_episode_key_results.csv",
            "advanced_episode_design": "spo2_advanced_episode_design.csv",
        }
        output_paths = {
            name: output_dir / filename for name, filename in filenames.items()
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "episode_anchored_advanced_inference",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "focused_records_rebuilt": False,
                "input_records_sha256": committed[
                    "multiorgan_episode_records"
                ].get("sha256"),
                "input_key_results_sha256": committed[
                    "multiorgan_episode_key_results"
                ].get("sha256"),
            }
        )
        manifest["episode_anchored_advanced_inference"] = {
            "status": "post_result_advanced_episode_robustness_amendment_exploratory",
            "strict_preanchor_covariates": True,
            "concurrent_spo2_in_propensity_model": False,
            "concurrent_spo2_in_observation_and_outcome_models": True,
            "exposure_balance": "outcome_independent_overlap_weights_with_full_smd_audit",
            "doubly_robust_estimator": "bounded_cross_validated_tmle_with_one_step_aipw_diagnostic",
            "continuous_estimand": "crossfit_aipw_mean_worsening_in_clinical_units",
            "outcome_observation_model": "crossfit_inverse_probability_correction",
            "multicenter_method": "hospital_specific_rr_paule_mandel_hksj_and_leave_one_hospital_out",
            "residual_confounding_sensitivity": "e_values_for_each_estimable_focused_rr",
            "multiplicity": "within_endpoint_and_cross_endpoint_BH_per_dataset",
            "claim_scope": "observational_not_causal_not_prospectively_preregistered",
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_biomarker_benchmark(
    output_root: str | Path,
    *,
    bootstrap_repetitions: int = 300,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Rerun only the SpO2/SBP/lactate/Kapur-SCAI comparison.

    The function validates and reads committed event and analysis artifacts;
    it never invokes the raw MIMIC or eICU extractors.  Outputs and manifest
    fingerprints are committed atomically after every benchmark layer finishes.
    """
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "biomarker_benchmark_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "head_to_head_biomarker_benchmark",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "n_splits": int(n_splits),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")
        analysis_path = output_dir / "spo2_analysis_frame.csv"
        analysis_fingerprint = manifest.get("output_fingerprints", {}).get(
            "analysis_frame", {}
        )
        if not analysis_path.is_file() or (
            int(analysis_fingerprint.get("size_bytes", -1))
            != int(analysis_path.stat().st_size)
            or analysis_fingerprint.get("sha256") != _file_sha256(analysis_path)
        ):
            raise RuntimeError(
                "Cached analysis frame is missing or failed manifest validation"
            )

        event_frames: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        source_availability: dict[str, Any] = {}
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest_path = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest_path):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            dataset_manifest = json.loads(
                dataset_manifest_path.read_text(encoding="utf-8")
            )
            source_availability[dataset] = dataset_manifest.get(
                "source_availability", {}
            )
            event_frames.append(_read_csv(dataset_dir / "events.csv"))

        events = pd.concat(event_frames, ignore_index=True, sort=False)
        analysis_frame = _read_csv(analysis_path)
        outputs = run_biomarker_benchmark(
            events,
            analysis_frame,
            n_splits=n_splits,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during biomarker benchmark refresh"
            )

        output_paths = {
            name: output_dir / f"spo2_{name}.csv" for name in outputs
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        eicu_aperiodic = source_availability.get("eicu", {}).get(
            "vitalAperiodic.csv", {}
        )
        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "head_to_head_biomarker_benchmark",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "n_splits": int(n_splits),
                "input_analysis_frame_sha256": _file_sha256(analysis_path),
                "eicu_vital_aperiodic_source_available": bool(
                    eicu_aperiodic.get("exists", False)
                ),
            }
        )
        manifest["head_to_head_biomarker_benchmark"] = {
            "status": "completed_analysis_only",
            "observation_window": "[0,240)_minutes_after_icu_admission",
            "outcome_clock": "(240,240+12h]_and_(240,240+24h]",
            "comparators": [
                "absolute_spo2",
                "spo2_instability",
                "sbp",
                "lactate",
                "kapur_cswg_scai_stage",
            ],
            "kapur_reference": (
                "Kapur_NK_et_al_JACC_2022_80_3_185_198_"
                "doi_10.1016/j.jacc.2022.04.049"
            ),
            "evaluation": (
                "identical_patient_grouped_oof_folds_paired_cluster_bootstrap_"
                "bidirectional_external_database_transportability_modified_"
                "poisson_associations_and_decision_curve_analysis"
            ),
            "scai_claim_scope": (
                "four_hour_ehr_operationalization_not_clinician_adjudication_"
                "all_assigned_stages_are_lower_bounds_when_OHCA_is_unavailable"
            ),
            "eicu_vital_aperiodic_source_available": bool(
                eicu_aperiodic.get("exists", False)
            ),
            "raw_cached_artifacts_mutated": False,
            "claim_scope": "prognostic_association_and_prediction_not_causal",
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "eicu_vital_aperiodic_source_available": bool(
                    eicu_aperiodic.get("exists", False)
                ),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_multiscale_instability_analysis(
    output_root: str | Path,
    *,
    bootstrap_repetitions: int = 300,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Rerun only the dynamics-first instability analysis from cached artifacts."""
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "instability_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "multiscale_dynamics_first_instability",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "n_splits": int(n_splits),
        "protocol_version": INSTABILITY_PROTOCOL_VERSION,
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")
        analysis_path = output_dir / "spo2_analysis_frame.csv"
        analysis_fingerprint = manifest.get("output_fingerprints", {}).get(
            "analysis_frame", {}
        )
        if not analysis_path.is_file() or (
            int(analysis_fingerprint.get("size_bytes", -1))
            != int(analysis_path.stat().st_size)
            or analysis_fingerprint.get("sha256") != _file_sha256(analysis_path)
        ):
            raise RuntimeError(
                "Cached analysis frame is missing or failed manifest validation"
            )

        event_frames: list[pd.DataFrame] = []
        cohort_frames: list[pd.DataFrame] = []
        input_manifests = manifest.get("input_manifests", {})
        for dataset in ("mimic", "eicu"):
            dataset_dir = root / dataset
            if not _artifact_ready(
                dataset_dir,
                dataset=dataset,
                require_current_code_hash=False,
            ):
                raise RuntimeError(
                    f"Cached {dataset} artifacts failed immutable-manifest validation"
                )
            dataset_manifest_path = dataset_dir / "manifest.json"
            committed_sha = input_manifests.get(dataset, {}).get("sha256")
            if committed_sha and committed_sha != _file_sha256(dataset_manifest_path):
                raise RuntimeError(
                    f"Cached {dataset} manifest differs from the analysis input manifest"
                )
            event_frames.append(_read_csv(dataset_dir / "events.csv"))
            cohort_frames.append(_read_csv(dataset_dir / "cohort.csv"))

        events = pd.concat(event_frames, ignore_index=True, sort=False)
        cohorts = pd.concat(cohort_frames, ignore_index=True, sort=False)
        analysis_frame = _read_csv(analysis_path)
        outputs = run_multiscale_instability_analysis(
            events,
            cohorts,
            analysis_frame,
            n_splits=n_splits,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during instability refresh"
            )

        output_paths = {
            name: output_dir / f"spo2_{name}.csv" for name in outputs
        }
        for name, frame in outputs.items():
            _atomic_to_csv(frame, output_paths[name])
            _checkpoint_csv(output_dir, f"ckpt_{name}", frame)

        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "multiscale_dynamics_first_instability",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "bootstrap_repetitions": int(bootstrap_repetitions),
                "n_splits": int(n_splits),
                "input_analysis_frame_sha256": _file_sha256(analysis_path),
                "protocol_version": INSTABILITY_PROTOCOL_VERSION,
            }
        )
        manifest["multiscale_instability_analysis"] = {
            "status": "completed_analysis_only",
            "protocol_version": INSTABILITY_PROTOCOL_VERSION,
            "observation_window": "[0,240)_minutes_after_icu_admission",
            "horizons_hours_after_landmark": list(INSTABILITY_HORIZONS_HOURS),
            "admission_clock_mapping": {"8h_after_landmark": 12, "20h_after_landmark": 24},
            "instability_domains": [
                "threshold_free_volatility",
                "abrupt_transitions",
                "recurrence_and_bursts",
                "directional_drops",
                "oscillation_and_reversals",
                "persistent_shift_confirmation",
                "hour_specific_acceleration",
                "spike_and_return_morphology_control",
            ],
            "resolutions": [
                "native_adaptive_gap",
                "raw_gap_le10",
                "raw_gap_le30",
                "raw_gap_le90",
                "15_minute_medians",
                "30_minute_medians",
                "60_minute_medians",
            ],
            "claim_scope": "prognostic_association_and_prediction_not_causal",
            "raw_cached_artifacts_mutated": False,
            "waveform_claim_permitted": False,
        }
        manifest.setdefault("outputs", {})
        manifest.setdefault("output_fingerprints", {})
        for name, path in output_paths.items():
            manifest["outputs"][name] = str(path)
            manifest["output_fingerprints"][name] = {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "sha256": _file_sha256(path),
            }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )
        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": {name: int(len(frame)) for name, frame in outputs.items()},
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "paths": output_paths,
            "manifest": manifest,
            "status": status,
            "frames": outputs,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def refresh_endpoint_conclusions(output_root: str | Path) -> dict[str, Any]:
    """Refresh only the evidence-classification table from committed outputs.

    This path is intentionally independent of raw MIMIC/eICU artifacts.  It
    validates every summary table consumed by ``build_endpoint_conclusions``,
    rebuilds the conclusion matrix atomically, and records the component
    refresh in the analysis manifest.
    """
    root = Path(output_root).expanduser().resolve()
    output_dir = root / "spo2_drilldown"
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "endpoint_conclusions_refresh_status.json"
    started = datetime.now(timezone.utc)
    analysis_code_hashes = _current_project_code_hashes()
    status: dict[str, Any] = {
        "status": "running",
        "component": "endpoint_evidence_classification",
        "analysis_only": True,
        "raw_data_rebuilt": False,
        "started_at_utc": started.isoformat(),
        "output_root": str(root),
    }
    _write_json(status_path, status)
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise RuntimeError("Analysis manifest schema does not match this project")

        required = {
            "endpoint_completeness": "spo2_endpoint_completeness.csv",
            "model_metrics": "spo2_model_metrics.csv",
            "epi_risk_tables": "epi_stratified_risk_tables.csv",
            "sensitivity_matrix": "spo2_sensitivity_matrix_s1_s12.csv",
            "cross_dataset_holdout": "spo2_external_cross_dataset.csv",
        }
        frames: dict[str, pd.DataFrame] = {}
        committed = manifest.get("output_fingerprints", {})
        for name, filename in required.items():
            path = output_dir / filename
            fingerprint = committed.get(name, {})
            if not path.is_file():
                raise FileNotFoundError(f"Missing committed analysis table: {path}")
            if (
                int(fingerprint.get("size_bytes", -1)) != int(path.stat().st_size)
                or fingerprint.get("sha256") != _file_sha256(path)
            ):
                raise RuntimeError(f"Committed {name} table failed integrity validation")
            frames[name] = _read_csv(path)

        conclusions = build_endpoint_conclusions(
            frames["endpoint_completeness"],
            frames["model_metrics"],
            frames["epi_risk_tables"],
            frames["sensitivity_matrix"],
            frames["cross_dataset_holdout"],
        )
        if _current_project_code_hashes() != analysis_code_hashes:
            raise RuntimeError(
                "PhysioGraph source code changed during component refresh"
            )

        output_path = output_dir / "endpoint_conclusion_matrix.csv"
        _atomic_to_csv(conclusions, output_path)
        _checkpoint_csv(output_dir, "endpoint_conclusions", conclusions)

        refreshed = datetime.now(timezone.utc)
        manifest["code_sha256"] = analysis_code_hashes
        manifest["last_component_refresh_at_utc"] = refreshed.isoformat()
        manifest.setdefault("component_refreshes", []).append(
            {
                "component": "endpoint_evidence_classification",
                "completed_at_utc": refreshed.isoformat(),
                "analysis_only": True,
                "raw_data_rebuilt": False,
                "claim_gates": (
                    "adequate_internal_epv_and_dataset-matched_external_training_epv"
                ),
            }
        )
        manifest.setdefault("outputs", {})["endpoint_conclusions"] = str(
            output_path
        )
        manifest.setdefault("output_fingerprints", {})[
            "endpoint_conclusions"
        ] = {
            "path": str(output_path),
            "size_bytes": int(output_path.stat().st_size),
            "mtime_ns": int(output_path.stat().st_mtime_ns),
            "sha256": _file_sha256(output_path),
        }
        _write_json(manifest_path, manifest)
        _synchronize_run_status_manifest(
            root,
            manifest_path,
            refreshed_at_utc=refreshed.isoformat(),
        )

        completed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "complete",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - started).total_seconds(),
                "rows": int(len(conclusions)),
                "manifest": str(manifest_path),
                "error": None,
            }
        )
        _write_json(status_path, status)
        return {
            "output_dir": output_dir,
            "path": output_path,
            "manifest": manifest,
            "status": status,
            "frame": conclusions,
        }
    except Exception as exc:
        failed = datetime.now(timezone.utc)
        status.update(
            {
                "status": "failed",
                "completed_at_utc": failed.isoformat(),
                "duration_seconds": (failed - started).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(status_path, status)
        raise


def run_comparator_if_available(
    output_root: Path,
    *,
    build_new: bool,
    analysis_only: bool = False,
) -> dict[str, Any]:
    """Run or load comparator validation when MIMIC and eICU artifacts exist."""
    mimic_dir = output_root / "mimic"
    eicu_dir = output_root / "eicu"
    comparator_dir = output_root / "locked_comparator_validation"
    metrics_path = comparator_dir / "metrics.csv"
    if not (
        _artifact_ready(
            mimic_dir, require_current_code_hash=not analysis_only
        )
        and _artifact_ready(
            eicu_dir, require_current_code_hash=not analysis_only
        )
    ):
        return {"status": "skipped", "reason": "mimic/eicu artifacts unavailable"}
    if metrics_path.exists() and not build_new:
        return {"status": "cached", "metrics": metrics_path}

    from physiograph.validation.locked_inference import run_locked_comparator_validation

    # The locked comparator is legacy/supporting analysis; a class-poor pilot
    # cohort or eligibility collapse must be reported, never crash the run.
    try:
        artifacts = run_locked_comparator_validation(
            internal_artifact_dir=mimic_dir,
            external_artifact_dir=eicu_dir,
            output_dir=comparator_dir,
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"legacy comparator validation failed: {exc}"}
    return {"status": "ran", "artifacts": artifacts}


def run_physiograph_colab(
    *,
    project_root: str | Path,
    build_new: bool = False,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    output_root: str | Path | None = None,
    run_comparator: bool = True,
    max_stays: int | None = None,
    max_chunks: int | None = None,
    chunk_size: int = 250_000,
    execution_environment: str = "local",
    require_all_requested_datasets: bool = True,
    analysis_only: bool = False,
) -> dict[str, Any]:
    """Run the minimal Colab-facing PhysioGraph workflow."""
    if analysis_only and build_new:
        raise ValueError("analysis_only=True is incompatible with build_new=True")
    project_root = Path(project_root).expanduser().resolve()
    _ensure_project_imports(project_root)
    output_root = Path(output_root) if output_root is not None else project_root / "physiograph_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    mimic_root = Path(mimic_root).expanduser() if mimic_root is not None else None
    eicu_root = Path(eicu_root).expanduser() if eicu_root is not None else None
    requested_datasets = {
        dataset
        for dataset, root in (("mimic", mimic_root), ("eicu", eicu_root))
        if root is not None
    }
    run_started = datetime.now(timezone.utc)
    status_path = output_root / "run_status.json"
    status_base: dict[str, Any] = {
        "status": "running",
        "started_at_utc": run_started.isoformat(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "build_new": bool(build_new),
        "analysis_only": bool(analysis_only),
        "execution_environment": execution_environment,
        "run_comparator_requested": bool(run_comparator),
        "max_stays": max_stays,
        "max_chunks": max_chunks,
        "chunk_size": int(chunk_size),
        "require_all_requested_datasets": bool(require_all_requested_datasets),
        "requested_datasets": sorted(requested_datasets),
        "datasets": {},
    }
    _write_json(status_path, status_base)

    dataset_artifacts: dict[str, dict[str, pd.DataFrame]] = {}
    load_status: dict[str, Any] = {}
    for dataset in ("mimic", "eicu"):
        try:
            dataset_artifacts[dataset] = _run_or_load_dataset(
                dataset,
                dataset_dir=output_root / dataset,
                mimic_root=mimic_root,
                eicu_root=eicu_root,
                build_new=build_new,
                max_stays=max_stays,
                max_chunks=max_chunks,
                chunk_size=chunk_size,
                analysis_only=analysis_only,
            )
            load_status[dataset] = {
                "status": (
                    "rebuilt"
                    if build_new
                    else "loaded_cached_analysis_only"
                    if analysis_only
                    else "loaded_cached"
                ),
                "artifact_dir": str(output_root / dataset),
            }
        except Exception as exc:
            load_status[dataset] = {"status": "skipped_or_failed", "reason": str(exc)}
        _write_json(status_path, {**status_base, "datasets": load_status})

    missing_requested = sorted(requested_datasets.difference(dataset_artifacts))
    if require_all_requested_datasets and missing_requested:
        completed = datetime.now(timezone.utc)
        _write_json(
            status_path,
            {
                **status_base,
                "status": "failed",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - run_started).total_seconds(),
                "datasets": load_status,
                "missing_requested_datasets": missing_requested,
                "error_type": "RuntimeError",
                "reason": "one or more requested dataset extractions failed",
            },
        )
        raise RuntimeError(
            "Requested dataset extraction failed; refusing to report a partial run. "
            f"Missing: {', '.join(missing_requested)}. See {status_path}."
        )

    if not dataset_artifacts:
        completed = datetime.now(timezone.utc)
        _write_json(
            status_path,
            {
                **status_base,
                "status": "failed",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - run_started).total_seconds(),
                "datasets": load_status,
                "error_type": "FileNotFoundError",
                "reason": "no MIMIC or eICU artifacts were available",
            },
        )
        raise FileNotFoundError(
            "No MIMIC or eICU artifacts were available. Run with BUILD_NEW=True after "
            "confirming Drive data paths, or place cached cohort/events/features/labels "
            f"under {output_root}/mimic and/or {output_root}/eicu."
        )

    fresh_colab_execution = bool(
        execution_environment == "colab"
        and build_new
        and dataset_artifacts
        and all(item.get("status") == "rebuilt" for item in load_status.values())
    )
    try:
        comparator_status = (
            run_comparator_if_available(
                output_root,
                build_new=build_new,
                analysis_only=analysis_only,
            )
            if run_comparator
            else {"status": "disabled"}
        )
        spo2 = run_spo2_drilldown(
            dataset_artifacts,
            output_root / "spo2_drilldown",
            mimic_root=mimic_root,
            build_new=build_new,
            fresh_colab_execution=fresh_colab_execution,
            execution_environment=execution_environment,
            run_comparator_requested=run_comparator,
            max_stays=max_stays,
            max_chunks=max_chunks,
            chunk_size=chunk_size,
            require_all_requested_datasets=require_all_requested_datasets,
            analysis_only=analysis_only,
        )
        archive_candidates = write_archive_candidates(
            project_root, output_root / "cleanup"
        )
    except Exception as exc:
        completed = datetime.now(timezone.utc)
        _write_json(
            status_path,
            {
                **status_base,
                "status": "failed",
                "completed_at_utc": completed.isoformat(),
                "duration_seconds": (completed - run_started).total_seconds(),
                "datasets": load_status,
                "error_type": type(exc).__name__,
                "reason": str(exc),
                "fresh_colab_execution": fresh_colab_execution,
            },
        )
        raise

    completed = datetime.now(timezone.utc)
    status = {
        **status_base,
        "status": "complete",
        "completed_at_utc": completed.isoformat(),
        "duration_seconds": (completed - run_started).total_seconds(),
        "datasets": load_status,
        "comparator": comparator_status,
        "spo2_manifest": str(spo2["paths"]["manifest"]),
        "spo2_manifest_sha256": _file_sha256(spo2["paths"]["manifest"]),
        "archive_candidates": str(archive_candidates),
        "fresh_colab_execution": fresh_colab_execution,
    }
    _write_json(status_path, status)
    return {
        "status_path": status_path,
        "output_root": output_root,
        "spo2": spo2,
        "comparator": comparator_status,
        "archive_candidates": archive_candidates,
        "dataset_status": load_status,
    }


__all__ = [
    "build_spo2_raw_event_summary",
    "build_spo2_trajectory_summary",
    "build_spo2_variability_group_summary",
    "collect_raw_spo2_events",
    "compute_horizon_outcomes",
    "compute_respiratory_support_features",
    "compute_rrt_features",
    "compute_spo2_features",
    "fit_spo2_dragged_horizon_models",
    "fit_spo2_or_pvalue_tables",
    "fit_spo2_variability_or_tables",
    "refresh_advanced_episode_inference",
    "refresh_biomarker_benchmark",
    "refresh_endpoint_conclusions",
    "refresh_lactate_episode_analysis",
    "refresh_locked_external_validation",
    "refresh_multiscale_instability_analysis",
    "run_physiograph_colab",
    "run_spo2_drilldown",
    "build_analysis_manifest",
    "lint_claims_and_outputs",
    "fit_missingness_negative_control",
    "fit_external_cross_dataset_holdout",
    "build_availability_audit",
    "build_respiratory_context_tables",
]
