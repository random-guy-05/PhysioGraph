"""Minimal Colab runner for the PhysioGraph final workflow.

The notebook should only mount Drive, set paths, call ``run_physiograph_colab``,
and display the artifacts written here.  This module keeps the reusable logic in
plain Python so it can be tested outside Colab and rerun without copying large
code blocks into the notebook.
"""

from __future__ import annotations

import json
import math
import sys
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
    build_temporal_precedence,
    compute_early_decompensation_outcomes,
    fit_external_transportability,
    fit_grouped_incremental_models,
    fit_grouped_missingness_control,
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
    "negative_control",
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
    "cached_or_precomputed_unless_fresh_colab_execution",
    "text_derived_respiratory_rrt_controls_are_heuristic_proxies",
    "proxy_signal_quality_when_probe_flags_absent",
    "pooled_cv_is_internal_not_external_validation",
    "clif_out_of_scope",
    "no_causal_language",
)
PRIMARY_OUTCOME_COLUMNS = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
PRIMARY_SPO2_MODEL_FEATURES = (
    "spo2_min",
    "spo2_mean",
    "spo2_sd",
    "spo2_rmssd",
    "spo2_sampling_density_per_hr",
    "spo2_below_90_fraction",
    "spo2_instability_proxy_score",
)
CONTROL_FEATURE_CANDIDATES = (
    "age",
    "is_male",
    "baseline_lactate",
    "baseline_creatinine",
    "baseline_map",
    "baseline_resp_rate",
    "fio2_max",
    "resp_support_any_flag",
    "mechanical_ventilation_flag",
    "noninvasive_ventilation_flag",
    "rrt_or_dialysis_flag",
    "dataset",
    "race",
    "ethnicity",
    "race_ethnicity",
)
SPO2_VARIABILITY_FEATURES = (
    "spo2_sd",
    "spo2_rmssd",
    "spo2_iqr",
    "spo2_range",
    "spo2_mad",
    "spo2_abrupt_jump_rate_per_hr",
    "spo2_abrupt_jump_count",
    "spo2_instability_proxy_score",
)
SPO2_VARIABILITY_OUTCOMES = (*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS)
SPO2_OXYGENATION_COVARIATES = (
    "spo2_mean",
    "spo2_min",
    "spo2_below_90_fraction",
    "spo2_sampling_density_per_hr",
)

SPO2_SIGNAL_COLUMNS = [
    "spo2_measurement_count",
    "spo2_plausible_count",
    "spo2_implausible_count",
    "spo2_observed_bin_count",
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
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
    "spo2_instability_proxy_score",
]

RESPIRATORY_SUPPORT_COLUMNS = [
    "fio2_measurement_count",
    "fio2_max",
    "fio2_mean",
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
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _artifact_ready(dataset_dir: Path) -> bool:
    return all((dataset_dir / name).exists() for name in REQUIRED_ARTIFACT_FILES)


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
) -> dict[str, pd.DataFrame]:
    """Load cached artifacts or rebuild a dataset ETL output."""
    if _artifact_ready(dataset_dir) and not build_new:
        return _load_dataset_artifacts(dataset_dir)

    if not build_new:
        missing = [name for name in REQUIRED_ARTIFACT_FILES if not (dataset_dir / name).exists()]
        raise FileNotFoundError(
            f"{dataset_dir} is missing {missing}. Set BUILD_NEW=True to rebuild ETL."
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
    if "offset_minutes" in events:
        events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    if "window" in events.columns:
        return events.loc[events["window"].eq("observation")].copy()
    return events.loc[
        events["offset_minutes"].ge(0) & events["offset_minutes"].lt(LANDMARK_MINUTES)
    ].copy()


def _nan_record(columns: Iterable[str]) -> dict[str, float]:
    return {column: math.nan for column in columns}


def _summarize_spo2_group(group: pd.DataFrame) -> dict[str, float]:
    group = group.copy()
    group["value_numeric"] = pd.to_numeric(group["value_numeric"], errors="coerce")
    group["offset_minutes"] = pd.to_numeric(group["offset_minutes"], errors="coerce")
    group = group.dropna(subset=["value_numeric", "offset_minutes"]).sort_values("offset_minutes")

    record = _nan_record(SPO2_SIGNAL_COLUMNS)
    record["spo2_measurement_count"] = float(len(group))
    if "time_bin" in group.columns:
        bins = pd.to_numeric(group["time_bin"], errors="coerce").dropna().unique()
    else:
        bins = np.floor(group["offset_minutes"].clip(lower=0, upper=LANDMARK_MINUTES - 1) / 15).unique()
    record["spo2_observed_bin_count"] = float(len(bins))
    record["spo2_missing_bin_count"] = float(max(OBSERVATION_BINS - len(bins), 0))

    plausible_mask = group["value_numeric"].between(50, 100, inclusive="both")
    record["spo2_plausible_count"] = float(plausible_mask.sum())
    record["spo2_implausible_count"] = float((~plausible_mask).sum())
    record["spo2_sampling_density_per_hr"] = float(plausible_mask.sum() / OBSERVATION_HOURS)
    if not plausible_mask.any():
        record["spo2_instability_proxy_score"] = (
            record["spo2_implausible_count"] + record["spo2_missing_bin_count"] / OBSERVATION_BINS
        )
        return record

    plausible = group.loc[plausible_mask].copy()
    values = plausible["value_numeric"].astype(float).to_numpy()
    offsets = plausible["offset_minutes"].astype(float).to_numpy()
    diffs = np.diff(values)
    gaps = np.diff(offsets)

    record.update(
        {
            "spo2_mean": float(np.mean(values)),
            "spo2_median": float(np.median(values)),
            "spo2_min": float(np.min(values)),
            "spo2_max": float(np.max(values)),
            "spo2_first": float(values[0]),
            "spo2_last": float(values[-1]),
            "spo2_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "spo2_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "spo2_mad": float(np.median(np.abs(values - np.median(values)))),
            "spo2_range": float(np.max(values) - np.min(values)),
            "spo2_rmssd": float(np.sqrt(np.mean(diffs**2))) if len(diffs) else 0.0,
            "spo2_median_gap_minutes": float(np.median(gaps)) if len(gaps) else math.nan,
            "spo2_longest_gap_minutes": float(np.max(gaps)) if len(gaps) else math.nan,
        }
    )
    for threshold in LOW_SPO2_THRESHOLDS:
        record[f"spo2_below_{threshold}_fraction"] = float(np.mean(values < threshold))
        record[f"spo2_deficit_below_{threshold}_mean"] = float(
            np.mean(np.maximum(threshold - values, 0))
        )

    if len(np.unique(offsets)) > 1:
        record["spo2_slope_per_hr"] = float(np.polyfit(offsets / 60.0, values, 1)[0])
    else:
        record["spo2_slope_per_hr"] = 0.0

    abrupt = np.abs(diffs) >= 4.0
    record["spo2_abrupt_jump_count"] = float(abrupt.sum())
    record["spo2_abrupt_jump_rate_per_hr"] = float(abrupt.sum() / OBSERVATION_HOURS)
    record["spo2_instability_proxy_score"] = float(
        record["spo2_implausible_count"]
        + record["spo2_abrupt_jump_count"]
        + record["spo2_missing_bin_count"] / OBSERVATION_BINS
        + (1.0 if record.get("spo2_longest_gap_minutes", 0) > 45 else 0.0)
    )
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
    count_cols = [col for col in result.columns if col.endswith("_count") or col.endswith("_flag")]
    count_cols += ["spo2_observed_bin_count", "spo2_missing_bin_count", "spo2_sampling_density_per_hr"]
    for col in set(count_cols) & set(result.columns):
        result[col] = result[col].fillna(0)
    return result


def _merge_default_feature_frame(keys: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    result = keys.copy()
    for column in value_columns:
        result[column] = np.nan
    for column in ["spo2_plausible_count", "spo2_sampling_density_per_hr"]:
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
    fio2_mask = obs["_text"].str.contains(
        r"fio2|fraction inspired oxygen|inspired o2|oxygen concentration",
        regex=True,
        na=False,
    )
    fio2_rows = obs.loc[fio2_mask].assign(_fio2=value.loc[fio2_mask])
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
    flag_frames: list[pd.DataFrame] = []
    for column, pattern in flag_patterns.items():
        flagged = obs.loc[obs["_text"].str.contains(pattern, regex=True, na=False)]
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
    result["resp_support_any_flag"] = (result[support_cols].fillna(0).sum(axis=1) > 0).astype(int)
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
    """Create post-landmark MCS/death/escalation labels at requested horizons."""
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
            | event_text.str.contains(r"iabp|impella|ecmo|mechanical circulatory|\bmcs\b", regex=True, na=False)
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
    stay_index = _stay_index_from(features, labels, cohort)
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

    frame = features.copy()
    if "dataset" not in frame:
        frame["dataset"] = stay_index["dataset"].iloc[0] if not stay_index.empty else "unknown"
    label_cols = ["dataset", "stay_id"] + [
        col for col in labels.columns if col not in features.columns and col not in ("dataset", "stay_id")
    ]
    frame = frame.merge(labels[label_cols], on=["dataset", "stay_id"], how="left")
    for extra in (spo2, resp, rrt, early_outcomes, exploratory_long_horizons):
        duplicate_non_keys = [
            column
            for column in extra.columns
            if column in frame.columns and column not in {"dataset", "stay_id"}
        ]
        if duplicate_non_keys:
            extra = extra.drop(columns=duplicate_non_keys)
        frame = frame.merge(extra, on=["dataset", "stay_id"], how="left")

    cohort_context = [
        col
        for col in cohort.columns
        if col == "person_id" or col.lower() in {"race", "ethnicity", "race_ethnicity"}
    ]
    missing_context = [column for column in cohort_context if column not in frame.columns]
    if missing_context:
        frame = frame.merge(
            cohort[["dataset", "stay_id", *missing_context]].drop_duplicates(),
            on=["dataset", "stay_id"],
            how="left",
            validate="many_to_one",
        )
    race_cols = [col for col in cohort_context if col.lower() in {"race", "ethnicity", "race_ethnicity"}]

    availability = {
        "n_rows": int(len(frame)),
        "spo2_rows_with_measurements": int((frame.get("spo2_plausible_count", 0) > 0).sum()),
        "fio2_rows_with_measurements": int((frame.get("fio2_measurement_count", 0) > 0).sum()),
        "respiratory_support_rows": int((frame.get("resp_support_any_flag", 0) > 0).sum()),
        "rrt_or_dialysis_rows": int((frame.get("rrt_or_dialysis_flag", 0) > 0).sum()),
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
        return "person_id", "internal_pooled_grouped_cv_by_person_id"
    if "stay_id" in analysis_df.columns:
        return "stay_id", "internal_pooled_grouped_cv_by_stay_id"
    return None, "internal_pooled_row_cv_with_group_limitation"


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


def build_descriptive_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SpO2 signals by available outcomes."""
    metric_cols = [
        col for col in [
            "spo2_min",
            "spo2_mean",
            "spo2_sd",
            "spo2_rmssd",
            "spo2_below_90_fraction",
            "spo2_sampling_density_per_hr",
            "spo2_instability_proxy_score",
        ] if col in analysis_df.columns
    ]
    group_cols = [
        col for col in [
            "target",
            "mcs_48h_flag",
            "mcs_72h_flag",
            "mcs_96h_flag",
            "mcs_168h_flag",
            "death_168h_flag",
            "mcs_or_death_168h_flag",
        ] if col in analysis_df.columns
    ]
    rows: list[dict[str, Any]] = []
    for group_col in group_cols:
        for value, group in analysis_df.groupby(group_col, dropna=False):
            row = {
                "group_column": group_col,
                "group_value": value,
                "n": int(len(group)),
            }
            for metric in metric_cols:
                row[f"{metric}_mean"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
                row[f"{metric}_median"] = float(pd.to_numeric(group[metric], errors="coerce").median())
            rows.append(row)
    return pd.DataFrame(rows)


def build_lactate_negative_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SpO2-outcome patterns when baseline lactate is below 2."""
    if "baseline_lactate" not in analysis_df.columns:
        return pd.DataFrame([{"status": "skipped", "reason": "baseline_lactate unavailable"}])
    subset = _filter_measured_spo2_rows(
        analysis_df.loc[pd.to_numeric(analysis_df["baseline_lactate"], errors="coerce") < 2].copy()
    )
    outcomes = [
        col for col in [
            "target",
            "mcs_48h_flag",
            "mcs_72h_flag",
            "mcs_96h_flag",
            "mcs_168h_flag",
            "death_168h_flag",
            "mcs_or_death_168h_flag",
        ] if col in subset.columns
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        y = pd.to_numeric(subset[outcome], errors="coerce")
        events = int(y.sum()) if y.notna().any() else 0
        rows.append(
            {
                "outcome": outcome,
                "n_lactate_negative": int(y.notna().sum()),
                "n_events": events,
                "event_rate": float(y.mean()) if y.notna().any() else math.nan,
                "fragility_label": _fragility_label(events, min_events=MIN_OR_EVENTS),
                "analysis_note": "descriptive_subgroup_only_not_network_discovery",
                "mean_spo2_min": float(pd.to_numeric(subset.get("spo2_min"), errors="coerce").mean()),
                "mean_spo2_rmssd": float(pd.to_numeric(subset.get("spo2_rmssd"), errors="coerce").mean()),
                "mean_spo2_below_90_fraction": float(
                    pd.to_numeric(subset.get("spo2_below_90_fraction"), errors="coerce").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_spo2_association_proxy(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Correlate SpO2 components with outcomes as a centrality proxy."""
    outcome_cols = [
        col for col in [
            "target",
            "mcs_48h_flag",
            "mcs_72h_flag",
            "mcs_96h_flag",
            "mcs_168h_flag",
            "death_168h_flag",
            "mcs_or_death_168h_flag",
        ] if col in analysis_df.columns
    ]
    rows: list[dict[str, Any]] = []
    for feature in [col for col in SPO2_SIGNAL_COLUMNS if col in analysis_df.columns]:
        x = pd.to_numeric(analysis_df[feature], errors="coerce")
        for outcome in outcome_cols:
            y = pd.to_numeric(analysis_df[outcome], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3 or y.loc[valid].nunique() < 2:
                corr = math.nan
            else:
                corr = float(np.corrcoef(x.loc[valid], y.loc[valid])[0, 1])
            rows.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n": int(valid.sum()),
                    "pearson_correlation": corr,
                    "absolute_correlation": abs(corr) if not math.isnan(corr) else math.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["outcome", "absolute_correlation"], ascending=[True, False], na_position="last"
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
    return [col for col in CONTROL_FEATURE_CANDIDATES if col in analysis_df.columns]


def _available_spo2_variability_features(analysis_df: pd.DataFrame) -> list[str]:
    return [col for col in SPO2_VARIABILITY_FEATURES if col in analysis_df.columns]


def _prepare_design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    x = frame[columns].copy()
    categorical = {
        col
        for col in columns
        if col in {"dataset", "race", "ethnicity", "race_ethnicity"}
        or str(x[col].dtype) in {"object", "category", "string"}
    }
    for col in columns:
        if col in categorical:
            x[col] = x[col].fillna("missing").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            median = float(x[col].median()) if x[col].notna().any() else 0.0
            x[col] = x[col].fillna(median)
    x = pd.get_dummies(x, columns=sorted(categorical), dummy_na=False, drop_first=False, dtype=float)
    if not x.empty:
        nunique = x.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            x = x.drop(columns=constant_cols)
    return x


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
    if "n" in result:
        result["n_input_rows"] = int(len(analysis_df))
        measured = _filter_measured_spo2_rows(analysis_df)
        result["n_measured_spo2_rows"] = int(len(measured))
    return result


def fit_spo2_or_pvalue_tables(
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
    endpoint_family = "spo2_primary_features_adjusted_logistic_or"
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        for feature in _available_spo2_model_features(measured):
            frame = measured[[outcome, feature, *controls]].copy()
            frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
            valid = frame[outcome].notna() & frame[feature].notna()
            frame = frame.loc[valid].copy()
            n = int(len(frame))
            events = int(frame[outcome].sum()) if n else 0
            if n < min_rows or events < min_events or frame[outcome].nunique() < 2:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "skipped",
                        "n": n,
                        "events": events,
                        "endpoint_family": endpoint_family,
                        "reason": "insufficient rows/events/classes",
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
            columns = [z_feature, *controls]
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
            y = frame[outcome].astype(int)
            try:
                glm = sm.GLM(
                    y,
                    sm.add_constant(x, has_constant="add"),
                    family=sm.families.Binomial(),
                )
                fit = glm.fit(cov_type="HC0")
                coef = float(fit.params[z_feature])
                ci_low, ci_high = fit.conf_int().loc[z_feature]
                p_val = float(fit.pvalues[z_feature])
            except Exception as exc:
                rows.append(
                    {
                        "outcome": outcome,
                        "feature": feature,
                        "model": "adjusted",
                        "status": "failed",
                        "n": n,
                        "events": events,
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
                    "endpoint_family": endpoint_family,
                    "or_per_1sd": float(np.exp(coef)),
                    "ci95_low": float(np.exp(ci_low)),
                    "ci95_high": float(np.exp(ci_high)),
                    "p_value": p_val,
                    "mean_feature": feature_mean,
                    "sd_feature": feature_sd,
                    "control_count": int(x.shape[1] - 1),
                }
            )

    result = pd.DataFrame(rows)
    fit_mask = result["status"].eq("fit") if not result.empty and "status" in result.columns else pd.Series(dtype=bool)
    if fit_mask.any():
        pvals = result.loc[fit_mask, "p_value"].astype(float).tolist()
        adj = _benjamini_hochberg(pvals)
        result.loc[fit_mask, "p_value_adj"] = adj
        result.loc[fit_mask, "n_tests"] = int(fit_mask.sum())
        result.loc[fit_mask, "multiplicity_scope"] = endpoint_family
    if result.empty:
        return result
    order = [col for col in ("outcome", "p_value", "feature") if col in result.columns]
    if order:
        return result.sort_values(order, na_position="last").reset_index(drop=True)
    return result.reset_index(drop=True)


def fit_spo2_variability_or_tables(
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
                frame = analysis_df[[outcome, feature, *covariates]].copy()
                frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
                valid = frame[outcome].notna() & frame[feature].notna()
                frame = frame.loc[valid].copy()
                n = int(len(frame))
                events = int(frame[outcome].sum()) if n else 0
                if n < min_rows or events < min_events or frame[outcome].nunique() < 2:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "insufficient rows/events/classes",
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
                y = frame[outcome].astype(int)
                try:
                    glm = sm.GLM(
                        y,
                        sm.add_constant(x, has_constant="add"),
                        family=sm.families.Binomial(),
                    )
                    fit = glm.fit(cov_type="HC0")
                    coef = float(fit.params[z_feature])
                    ci_low, ci_high = fit.conf_int().loc[z_feature]
                    p_val = float(fit.pvalues[z_feature])
                except Exception as exc:
                    rows.append(
                        {
                            "outcome": outcome,
                            "feature": feature,
                            "model": model_name,
                            "status": "failed",
                            "n": n,
                            "events": events,
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
                        "or_per_1sd": float(np.exp(coef)),
                        "ci95_low": float(np.exp(ci_low)),
                        "ci95_high": float(np.exp(ci_high)),
                        "p_value": p_val,
                        "mean_feature": feature_mean,
                        "sd_feature": feature_sd,
                        "covariate_count": int(x.shape[1] - 1),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    fit_mask = result["status"].eq("fit")
    if fit_mask.any():
        result.loc[fit_mask, "p_value_adj"] = _benjamini_hochberg(
            result.loc[fit_mask, "p_value"].astype(float).tolist()
        )
        result.loc[fit_mask, "n_tests"] = int(fit_mask.sum())
        result.loc[fit_mask, "multiplicity_scope"] = "all_variability_features_models_endpoints"
    sort_cols = ["outcome", "model", "p_value", "feature"]
    return result.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def build_spo2_variability_group_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw variability differences between event and non-event groups."""
    outcomes = _available_outcomes(analysis_df)
    features = [
        col
        for col in ("spo2_sd", "spo2_rmssd", "spo2_iqr", "spo2_range", "spo2_mad")
        if col in analysis_df.columns
    ]
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        y = pd.to_numeric(analysis_df[outcome], errors="coerce")
        for feature in features:
            x = pd.to_numeric(analysis_df[feature], errors="coerce")
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
    return pd.DataFrame(rows).sort_values(["outcome", "feature"]).reset_index(drop=True)




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
        if n < 20 or events < MIN_OR_EVENTS:
            rows.append({
                "stratum": stratum,
                "outcome": outcome,
                "status": "skipped",
                "n": n,
                "events": events,
                "reason": "insufficient rows/events",
            })
            continue
        burden = pd.to_numeric(subset.get("spo2_below_90_fraction"), errors="coerce")
        rows.append({
            "stratum": stratum,
            "outcome": outcome,
            "status": "descriptive",
            "n": n,
            "events": events,
            "mean_spo2_below_90_fraction": float(burden.mean()) if burden.notna().any() else math.nan,
            "event_rate": float(y.loc[valid].mean()),
        })

    # Interaction models
    interaction_specs = [
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
        frame = frame.loc[valid].copy()
        n = int(len(frame))
        events = int(frame[outcome].sum()) if n else 0
        if n < 200 or events < MIN_OR_EVENTS or frame[outcome].nunique() < 2:
            rows.append({
                "status": "skipped",
                "analysis": label,
                "n": n,
                "events": events,
                "reason": "insufficient rows/events/classes",
            })
            continue
        inter_col = f"{feat_a}_x_{feat_b}"
        frame[inter_col] = frame[feat_a] * frame[feat_b]
        design_columns = list(dict.fromkeys([feat_a, feat_b, inter_col, *controls]))
        x = _prepare_design_matrix(frame, design_columns)
        if inter_col not in x.columns:
            rows.append({"status": "skipped", "analysis": label, "reason": "interaction term dropped from design matrix"})
            continue
        y = frame[outcome].astype(int)
        try:
            glm = sm.GLM(y, sm.add_constant(x, has_constant="add"), family=sm.families.Binomial())
            fit = glm.fit(cov_type="HC0")
            coef = float(fit.params[inter_col])
            ci_low, ci_high = fit.conf_int().loc[inter_col]
            rows.append({
                "status": "fit",
                "analysis": label,
                "outcome": outcome,
                "n": n,
                "events": events,
                "or_interaction": float(np.exp(coef)),
                "ci95_low": float(np.exp(ci_low)),
                "ci95_high": float(np.exp(ci_high)),
                "p_value": float(fit.pvalues[inter_col]),
            })
        except Exception as exc:
            rows.append({"status": "failed", "analysis": label, "reason": str(exc)})
    return pd.DataFrame(rows)


def build_availability_audit(
    dataset_artifacts: dict[str, dict[str, pd.DataFrame]],
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-dataset availability audit for SpO2, controls, race, and composite endpoints."""
    rows: list[dict[str, Any]] = []
    for dataset, artifacts in dataset_artifacts.items():
        frame, _ = assemble_spo2_analysis_frame(
            cohort=artifacts["cohort"],
            events=artifacts["events"],
            features=artifacts["features"],
            labels=artifacts["labels"],
        )
        cohort = artifacts["cohort"]
        race_cols = [c for c in cohort.columns if c.lower() in {"race", "ethnicity", "race_ethnicity"}]
        plausible = pd.to_numeric(frame.get("spo2_plausible_count", 0), errors="coerce").fillna(0)
        zero_spo2 = int((plausible.eq(0)).sum())
        row: dict[str, Any] = {
            "dataset": dataset,
            "n_rows": int(len(frame)),
            "spo2_measured_rows": int(plausible.gt(0).sum()),
            "spo2_zero_plausible_rows": zero_spo2,
            "fio2_available_rows": int(pd.to_numeric(frame.get("fio2_measurement_count", 0), errors="coerce").fillna(0).gt(0).sum()),
            "respiratory_support_rows": int(pd.to_numeric(frame.get("resp_support_any_flag", 0), errors="coerce").fillna(0).gt(0).sum()),
            "mechanical_ventilation_rows": int(pd.to_numeric(frame.get("mechanical_ventilation_flag", 0), errors="coerce").fillna(0).gt(0).sum()),
            "rrt_or_dialysis_rows": int(pd.to_numeric(frame.get("rrt_or_dialysis_flag", 0), errors="coerce").fillna(0).gt(0).sum()),
            "vis_12h_observed_rows": int(pd.to_numeric(frame.get("vis_observed_12h", 0), errors="coerce").fillna(0).gt(0).sum()),
            "vis_24h_observed_rows": int(pd.to_numeric(frame.get("vis_observed_24h", 0), errors="coerce").fillna(0).gt(0).sum()),
            "urine_output_12h_observed_rows": int(pd.to_numeric(frame.get("urine_output_12h_observed", 0), errors="coerce").fillna(0).gt(0).sum()),
            "urine_output_24h_observed_rows": int(pd.to_numeric(frame.get("urine_output_24h_observed", 0), errors="coerce").fillna(0).gt(0).sum()),
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
    """Patient-grouped missingness/sampling negative-control models."""
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
            "detail": "fresh_colab_execution is False; metrics are local/precomputed",
        })
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
) -> dict[str, Any]:
    """Build provenance manifest with explicit limitations."""
    return {
        "analysis": "SpO2 drilldown from cached or freshly rebuilt PhysioGraph artifacts",
        "fresh_colab_execution": bool(fresh_colab_execution),
        "build_new": bool(build_new),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "outcome_clock": "12_and_24_hours_after_4h_landmark",
        "analysis_layers": list(ANALYSIS_LAYERS),
        "outputs": {key: str(value) for key, value in paths.items() if key != "manifest"},
        "datasets": datasets,
        "rows": int(rows),
        "warnings": MANIFEST_WARNINGS,
        "claims_linter": claims_warnings.to_dict(orient="records") if not claims_warnings.empty else [],
        "notes": [
            "No Google Colab execution is claimed unless the notebook is run in Colab.",
            "Grouped CV metrics are internal pooled estimates, not external validation.",
            "Preprocessing is fitted independently within each patient-grouped training fold.",
            "Calibration/ECE uses out-of-fold predictions only.",
            "Text-derived respiratory/RRT controls are heuristic proxies.",
            "CLIF dataset is out of scope.",
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
    """Summarize raw SpO2 signal quality and burden by dataset/outcome strata."""
    if raw_spo2.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no raw SpO2 events"}])
    frame = raw_spo2.copy().sort_values(["dataset", "stay_id", "offset_minutes"])
    frame["plausible"] = frame["value_numeric"].between(50, 100, inclusive="both")
    frame["below_90"] = frame["value_numeric"] < 90
    frame["below_88"] = frame["value_numeric"] < 88
    frame["delta"] = frame.groupby(["dataset", "stay_id"])["value_numeric"].diff()
    frame["gap_minutes"] = frame.groupby(["dataset", "stay_id"])["offset_minutes"].diff()
    frame["abrupt_jump"] = frame["delta"].abs().ge(4).fillna(False)

    outcome_cols = [col for col in ("target", "death_168h_flag", "mcs_or_death_168h_flag") if col in analysis_df]
    lookup = analysis_df[["dataset", "stay_id", *outcome_cols]].drop_duplicates(["dataset", "stay_id"])
    frame = frame.merge(lookup, on=["dataset", "stay_id"], how="left")

    rows: list[dict[str, Any]] = []

    def summarize(subset: pd.DataFrame, *, scope: str, group_col: str = "all", group_value: Any = "all") -> None:
        if subset.empty:
            return
        plausible = subset.loc[subset["plausible"]]
        rows.append(
            {
                "scope": scope,
                "group_column": group_col,
                "group_value": group_value,
                "n_events": int(len(subset)),
                "n_stays": int(subset[["dataset", "stay_id"]].drop_duplicates().shape[0]),
                "plausible_fraction": float(subset["plausible"].mean()),
                "below_90_fraction": float(plausible["below_90"].mean()) if not plausible.empty else math.nan,
                "below_88_fraction": float(plausible["below_88"].mean()) if not plausible.empty else math.nan,
                "abrupt_jump_fraction": float(subset["abrupt_jump"].mean()),
                "median_gap_minutes": float(subset["gap_minutes"].median()) if subset["gap_minutes"].notna().any() else math.nan,
                "mean_spo2": float(plausible["value_numeric"].mean()) if not plausible.empty else math.nan,
                "median_spo2": float(plausible["value_numeric"].median()) if not plausible.empty else math.nan,
                "spo2_p10": float(plausible["value_numeric"].quantile(0.10)) if not plausible.empty else math.nan,
                "spo2_p90": float(plausible["value_numeric"].quantile(0.90)) if not plausible.empty else math.nan,
            }
        )

    summarize(frame, scope="overall")
    for dataset, ds in frame.groupby("dataset", sort=False):
        summarize(ds, scope="dataset", group_col="dataset", group_value=dataset)
    for outcome in outcome_cols:
        for value in (0, 1):
            sub = frame.loc[pd.to_numeric(frame[outcome], errors="coerce").eq(value)]
            summarize(sub, scope="outcome", group_col=outcome, group_value=value)
    return pd.DataFrame(rows)


def build_spo2_trajectory_summary(
    raw_spo2: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    outcome: str = "mcs_or_death_168h_flag",
) -> pd.DataFrame:
    """Aggregate raw SpO2 trajectories (15-minute bins) by outcome group."""
    if raw_spo2.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no raw SpO2 events"}])
    if outcome not in analysis_df.columns:
        return pd.DataFrame([{"status": "skipped", "reason": f"outcome missing: {outcome}"}])
    frame = raw_spo2.copy()
    frame = frame.merge(
        analysis_df[["dataset", "stay_id", outcome]].drop_duplicates(["dataset", "stay_id"]),
        on=["dataset", "stay_id"],
        how="left",
    )
    frame = frame.loc[frame["value_numeric"].between(50, 100, inclusive="both")].copy()
    frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
    frame = frame.loc[frame[outcome].isin([0, 1])].copy()
    if frame.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no valid trajectories after filters"}])
    grouped = frame.groupby([outcome, "time_bin"], sort=True)["value_numeric"]
    summary = grouped.agg(["count", "mean", "median", "std"]).rename(columns={"count": "n_events"}).reset_index()
    summary["q25"] = grouped.quantile(0.25).to_numpy()
    summary["q75"] = grouped.quantile(0.75).to_numpy()
    summary["below_90_fraction"] = (
        frame.assign(_low=frame["value_numeric"] < 90)
        .groupby([outcome, "time_bin"], sort=True)["_low"]
        .mean()
        .to_numpy()
    )
    summary["minutes_from_landmark_bin_start"] = summary["time_bin"].astype(int) * 15
    summary = summary.rename(columns={outcome: "outcome_value"})
    summary.insert(0, "outcome", outcome)
    return summary.sort_values(["outcome_value", "time_bin"]).reset_index(drop=True)


def write_spo2_figures(analysis_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write compact diagnostic PNGs when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    outcome = "mcs_or_death_168h_flag" if "mcs_or_death_168h_flag" in analysis_df.columns else "target"
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


def run_spo2_drilldown(
    dataset_artifacts: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    *,
    build_new: bool = False,
    fresh_colab_execution: bool = False,
) -> dict[str, Any]:
    """Run the SpO2 drilldown across available datasets and write outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
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
    descriptive = build_descriptive_summary(analysis_df)
    lactate_negative = build_lactate_negative_summary(analysis_df)
    association = build_spo2_association_proxy(analysis_df)
    model_metrics = fit_spo2_models(analysis_df)
    or_pvalues = fit_spo2_or_pvalue_tables(analysis_df)
    respiratory_context = build_respiratory_context_tables(analysis_df)
    availability_audit = build_availability_audit(dataset_artifacts, analysis_df)
    cross_dataset = fit_external_cross_dataset_holdout(analysis_df)
    negative_control = fit_missingness_negative_control(analysis_df)
    variability_or = fit_spo2_variability_or_tables(analysis_df)
    variability_summary = build_spo2_variability_group_summary(analysis_df)
    dragged_models = fit_spo2_dragged_horizon_models(analysis_df)
    raw_event_summary = build_spo2_raw_event_summary(raw_spo2, analysis_df)
    trajectory_summary = build_spo2_trajectory_summary(raw_spo2, analysis_df)
    endpoint_audit = build_endpoint_completeness_audit(analysis_df)
    leadtime_records, leadtime_summary = build_temporal_precedence(all_events, all_cohorts)
    figures = write_spo2_figures(analysis_df, output_dir / "figures")

    paths = {
        "analysis_frame": output_dir / "spo2_analysis_frame.csv",
        "descriptive_summary": output_dir / "spo2_descriptive_summary.csv",
        "lactate_negative_summary": output_dir / "spo2_lactate_negative_results.csv",
        "association_proxy": output_dir / "spo2_signal_association_proxy.csv",
        "model_metrics": output_dir / "spo2_model_metrics.csv",
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
        "endpoint_completeness": output_dir / "spo2_endpoint_completeness.csv",
        "leadtime_records": output_dir / "spo2_leadtime_records.csv",
        "leadtime_summary": output_dir / "spo2_leadtime_summary.csv",
        "availability": output_dir / "spo2_adjustment_availability.json",
        "manifest": output_dir / "manifest.json",
    }
    analysis_df.to_csv(paths["analysis_frame"], index=False)
    descriptive.to_csv(paths["descriptive_summary"], index=False)
    lactate_negative.to_csv(paths["lactate_negative_summary"], index=False)
    association.to_csv(paths["association_proxy"], index=False)
    model_metrics.to_csv(paths["model_metrics"], index=False)
    or_pvalues.to_csv(paths["or_pvalues"], index=False)
    respiratory_context.to_csv(paths["respiratory_context"], index=False)
    availability_audit.to_csv(paths["availability_audit"], index=False)
    cross_dataset.to_csv(paths["cross_dataset_holdout"], index=False)
    negative_control.to_csv(paths["negative_control"], index=False)
    variability_or.to_csv(paths["variability_or_models"], index=False)
    variability_summary.to_csv(paths["variability_group_summary"], index=False)
    dragged_models.to_csv(paths["dragged_models"], index=False)
    raw_event_summary.to_csv(paths["raw_event_summary"], index=False)
    trajectory_summary.to_csv(paths["trajectory_summary"], index=False)
    endpoint_audit.to_csv(paths["endpoint_completeness"], index=False)
    leadtime_records.to_csv(paths["leadtime_records"], index=False)
    leadtime_summary.to_csv(paths["leadtime_summary"], index=False)
    _write_json(paths["availability"], availability)

    claims_warnings = lint_claims_and_outputs(
        model_metrics=model_metrics,
        manifest={"fresh_colab_execution": bool(fresh_colab_execution)},
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
    )
    manifest["figures"] = [str(path) for path in figures]
    manifest["raw_spo2_event_rows"] = int(len(raw_spo2))
    manifest["post_landmark_horizons_hours"] = list(POST_LANDMARK_HORIZONS_HOURS)
    manifest["exploratory_long_horizons_hours_from_icu_admission"] = list(HORIZON_HOURS)
    manifest["endpoint_definition_version"] = "spo2_protocol_v1.0"
    manifest["primary_endpoints"] = list(PRIMARY_ENDPOINTS)
    manifest["secondary_endpoints"] = list(SECONDARY_ENDPOINTS)
    manifest["leadtime_claim_scope"] = "landmark_ordering_not_causal_precedence"
    _write_json(paths["manifest"], manifest)
    claims_warnings.to_csv(output_dir / "claims_linter_warnings.csv", index=False)
    return {
        "output_dir": output_dir,
        "paths": paths,
        "figures": figures,
        "manifest": manifest,
    }


def run_comparator_if_available(
    output_root: Path,
    *,
    build_new: bool,
) -> dict[str, Any]:
    """Run or load comparator validation when MIMIC and eICU artifacts exist."""
    mimic_dir = output_root / "mimic"
    eicu_dir = output_root / "eicu"
    comparator_dir = output_root / "locked_comparator_validation"
    metrics_path = comparator_dir / "metrics.csv"
    if not (_artifact_ready(mimic_dir) and _artifact_ready(eicu_dir)):
        return {"status": "skipped", "reason": "mimic/eicu artifacts unavailable"}
    if metrics_path.exists() and not build_new:
        return {"status": "cached", "metrics": metrics_path}

    from physiograph.validation.locked_inference import run_locked_comparator_validation

    artifacts = run_locked_comparator_validation(
        internal_artifact_dir=mimic_dir,
        external_artifact_dir=eicu_dir,
        output_dir=comparator_dir,
    )
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
) -> dict[str, Any]:
    """Run the minimal Colab-facing PhysioGraph workflow."""
    project_root = Path(project_root).expanduser().resolve()
    _ensure_project_imports(project_root)
    output_root = Path(output_root) if output_root is not None else project_root / "physiograph_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    mimic_root = Path(mimic_root).expanduser() if mimic_root is not None else None
    eicu_root = Path(eicu_root).expanduser() if eicu_root is not None else None

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
            )
            load_status[dataset] = {
                "status": "rebuilt" if build_new else "loaded_cached",
                "artifact_dir": str(output_root / dataset),
            }
        except Exception as exc:
            load_status[dataset] = {"status": "skipped_or_failed", "reason": str(exc)}

    if not dataset_artifacts:
        status_path = output_root / "run_status.json"
        _write_json(status_path, {"datasets": load_status})
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
    comparator_status = (
        run_comparator_if_available(output_root, build_new=build_new)
        if run_comparator
        else {"status": "disabled"}
    )
    spo2 = run_spo2_drilldown(
        dataset_artifacts,
        output_root / "spo2_drilldown",
        build_new=build_new,
        fresh_colab_execution=fresh_colab_execution,
    )
    archive_candidates = write_archive_candidates(project_root, output_root / "cleanup")

    status = {
        "project_root": str(project_root),
        "output_root": str(output_root),
        "build_new": bool(build_new),
        "execution_environment": execution_environment,
        "datasets": load_status,
        "comparator": comparator_status,
        "spo2_manifest": str(spo2["paths"]["manifest"]),
        "archive_candidates": str(archive_candidates),
        "fresh_colab_execution": fresh_colab_execution,
    }
    status_path = output_root / "run_status.json"
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
    "run_physiograph_colab",
    "run_spo2_drilldown",
    "build_analysis_manifest",
    "lint_claims_and_outputs",
    "fit_missingness_negative_control",
    "fit_external_cross_dataset_holdout",
    "build_availability_audit",
    "build_respiratory_context_tables",
]
