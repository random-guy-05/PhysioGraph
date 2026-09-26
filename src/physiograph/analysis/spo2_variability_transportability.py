"""Outcome-guarded transportability audit for the frozen SpO2 variability signal."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, skew, spearmanr

from .shock_signal_discovery import trajectory_features

FROZEN_EXPOSURE = "detrended_hourly_spo2_rms_residual_0_240"
FROZEN_FEATURE_SPEC = {
    "window_minutes": [0, 240],
    "aggregation": "hourly_median",
    "minimum_bins": 3,
    "minimum_span_hours": 2,
    "detrending": "ordinary_least_squares_spo2_on_hour",
    "summary": "root_mean_square_residual",
    "valid_spo2_range": [50.0, 100.0],
}
OUTCOME_TOKENS = ("outcome", "objective_shock", "event_label", "case_status")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible object using a stable serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_frozen_feature_spec(spec: dict[str, Any]) -> None:
    """Fail closed if any part of the frozen feature definition changes."""
    if spec != FROZEN_FEATURE_SPEC:
        raise ValueError("Alternative SpO2 feature definitions are prohibited")


def assert_prelandmark_measurements(raw: pd.DataFrame) -> None:
    """Reject, rather than silently trim, any post-landmark measurement."""
    required = {"stay_id", "minute", "spo2"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing raw SpO2 columns: {sorted(missing)}")
    minute = pd.to_numeric(raw["minute"], errors="coerce")
    if minute.dropna().lt(0).any() or minute.dropna().gt(240).any():
        raise ValueError("SpO2 measurements must be confined to minutes 0 through 240")


def frozen_exposure_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the exact locked hourly-median, detrended RMS-residual feature."""
    assert_prelandmark_measurements(raw)
    validate_frozen_feature_spec(FROZEN_FEATURE_SPEC)
    clean = raw.copy()
    clean["minute"] = pd.to_numeric(clean["minute"], errors="coerce")
    clean["spo2"] = pd.to_numeric(clean["spo2"], errors="coerce")
    clean = clean.dropna(subset=["stay_id", "minute", "spo2"])
    clean = clean.loc[clean["spo2"].between(50, 100)].copy()
    clean["hour"] = np.floor(clean["minute"] / 60).astype(int)
    hourly = clean.groupby(["stay_id", "hour"], as_index=False)["spo2"].median()
    hourly["signal"] = "spo2"
    features = trajectory_features(
        hourly.rename(columns={"spo2": "value"})[["stay_id", "signal", "hour", "value"]]
    )
    return features.rename(columns={"variability": "raw_rms_residual"})[
        ["stay_id", "bins", "span_hours", "level", "raw_rms_residual"]
    ]


def standardize_with_mimic(
    values: pd.Series, parameters: dict[str, float]
) -> pd.DataFrame:
    """Apply only the frozen MIMIC clipping and scaling constants."""
    required = {"lower", "upper", "center", "scale"}
    if set(parameters) != required or parameters["scale"] <= 0:
        raise ValueError("Invalid or non-MIMIC exposure scaling parameters")
    numeric = pd.to_numeric(values, errors="coerce")
    clipped = numeric.clip(parameters["lower"], parameters["upper"])
    return pd.DataFrame(
        {
            "raw_rms_residual": numeric,
            "clipped_rms_residual": clipped,
            "standardized_exposure": (clipped - parameters["center"]) / parameters["scale"],
            "clipped_low": numeric.lt(parameters["lower"]),
            "clipped_high": numeric.gt(parameters["upper"]),
        },
        index=values.index,
    )


def implied_se(log_rr: float, ci_low: float, ci_high: float) -> float:
    """Recover a log-scale standard error from a symmetric Wald interval."""
    lower_se = (log_rr - math.log(ci_low)) / 1.96
    upper_se = (math.log(ci_high) - log_rr) / 1.96
    return float((lower_se + upper_se) / 2)


def normal_approximation_power(log_rr: float, se: float, alpha: float = 0.05) -> float:
    """Two-sided Wald power with SE held at the observed clustered value."""
    critical = NormalDist().inv_cdf(1 - alpha / 2)
    noncentrality = abs(log_rr) / se
    return float(norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality))


def precision_analysis(locked_result: dict[str, Any]) -> dict[str, Any]:
    """Quantify external precision using the locked clustered estimate."""
    primary = locked_result["primary"]
    assert_eicu_hospital_clustering(primary)
    beta = float(primary["log_rr"])
    se = implied_se(beta, float(primary["ci_low"]), float(primary["ci_high"]))
    effects = [1.10, 1.15, 1.20, 1.25, 1.30]
    powers = [
        {
            "true_rr": rr,
            "log_rr": math.log(rr),
            "power": normal_approximation_power(math.log(rr), se),
            "method": "two_sided_normal_approximation_observed_cluster_robust_se",
        }
        for rr in effects
    ]
    current_events = int(primary["events"])
    required: list[dict[str, Any]] = []
    z_alpha = NormalDist().inv_cdf(0.975)
    for rr in (1.20, 1.25):
        for target in (0.80, 0.90):
            z_power = NormalDist().inv_cdf(target)
            events = current_events * ((z_alpha + z_power) * se / math.log(rr)) ** 2
            required.append(
                {
                    "true_rr": rr,
                    "target_power": target,
                    "required_events": math.ceil(events),
                    "event_multiplier_vs_observed": float(events / current_events),
                    "method": "inverse_event_scaling_of_observed_cluster_robust_se",
                }
            )
    contrasts = []
    for rr in (1.20, 1.25):
        difference = math.log(rr) - math.log(1.10)
        contrasts.append(
            {
                "rr_a": 1.10,
                "rr_b": rr,
                "log_scale_difference": difference,
                "difference_in_observed_se_units": difference / se,
            }
        )
    return {
        "source": "locked_eicu_hospital_clustered_model",
        "estimate": {
            "rr": float(primary["rr_per_mimic_sd"]),
            "log_rr": beta,
            "cluster_robust_se": se,
            "ci_low": float(primary["ci_low"]),
            "ci_high": float(primary["ci_high"]),
            "rr_scale_ci_width": float(primary["ci_high"] - primary["ci_low"]),
            "log_scale_ci_width": float(math.log(primary["ci_high"]) - math.log(primary["ci_low"])),
            "multiplicative_95_ci_factor": float(math.exp(1.96 * se)),
            "events": current_events,
            "hospitals": int(primary["hospitals"]),
        },
        "power": powers,
        "required_events": required,
        "distinguish_rr_1_10_from": contrasts,
        "limitations": [
            "Power is a labeled normal approximation, not a new fitted or simulated outcome model.",
            "It holds the observed exposure distribution, covariate information, event prevalence relationship, and hospital-clustering design effect constant through the observed robust SE.",
            "Inverse-event scaling assumes information grows linearly with event count and a similar hospital structure.",
        ],
    }


def assert_eicu_hospital_clustering(primary: dict[str, Any]) -> None:
    """Require the locked eICU primary model to retain hospital clustering."""
    if primary.get("covariance") != "cluster_hospital" or int(primary.get("hospitals", 0)) < 1:
        raise ValueError("eICU primary inference must retain hospital-clustered covariance")


def _summaries(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if numeric.empty:
        return {key: math.nan for key in ("mean", "sd", "p1", "p5", "p25", "median", "p75", "p95", "p99", "iqr")}
    quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(numeric.mean()),
        "sd": float(numeric.std(ddof=0)),
        "p1": float(quantiles.loc[0.01]),
        "p5": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
    }


def measurement_process_rows(
    raw: pd.DataFrame, eligible_ids: pd.Series, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return standardized measurement-process summaries and per-stay diagnostics."""
    assert_prelandmark_measurements(raw)
    eligible = pd.Index(pd.Series(eligible_ids).dropna().unique(), name="stay_id")
    clean = raw.loc[raw["stay_id"].isin(eligible)].copy()
    clean["minute"] = pd.to_numeric(clean["minute"], errors="coerce")
    clean["spo2"] = pd.to_numeric(clean["spo2"], errors="coerce")
    clean = clean.dropna(subset=["stay_id", "minute", "spo2"])
    clean = clean.loc[clean["spo2"].between(50, 100)].sort_values(["stay_id", "minute"])
    clean["hour"] = np.floor(clean["minute"] / 60).astype(int)
    counts = clean.groupby("stay_id").size().reindex(eligible, fill_value=0).rename("raw_observations")
    unique_values = clean.groupby("stay_id")["spo2"].nunique().reindex(eligible, fill_value=0).rename("unique_values")
    spans = clean.groupby("stay_id")["minute"].agg(lambda x: x.max() - x.min()).reindex(eligible, fill_value=0).rename("span_minutes")
    bins = clean.groupby("stay_id")["hour"].nunique().reindex(eligible, fill_value=0).rename("valid_bins")
    differences = clean.groupby("stay_id")["minute"].diff()
    same = clean.groupby("stay_id")["spo2"].diff().eq(0)
    gap_stats = clean.assign(gap=differences).groupby("stay_id")["gap"].agg(["median", "max"]).reindex(eligible)
    identical = same.groupby(clean["stay_id"]).mean().reindex(eligible).fillna(0).rename("identical_fraction")
    run_lengths: dict[Any, int] = {}
    for stay_id, group in clean.groupby("stay_id", sort=False):
        change = group["spo2"].ne(group["spo2"].shift()).cumsum()
        run_lengths[stay_id] = int(group.groupby(change).size().max())
    max_run = pd.Series(run_lengths, name="max_identical_run").reindex(eligible).fillna(0)
    per_stay = pd.concat([counts, unique_values, spans, bins, gap_stats.add_prefix("gap_"), identical, max_run], axis=1).reset_index()
    within_hour = clean.groupby(["stay_id", "hour"]).size().rename("within_hour_observations")
    integer = np.isclose(clean["spo2"], np.round(clean["spo2"]), atol=1e-8)
    integer_values = np.round(clean.loc[integer, "spo2"]).astype(int)
    terminal_five = integer_values.mod(5).eq(0)
    metric_values: dict[str, tuple[float, int, str]] = {}
    for metric, series, unit in [
        ("raw_observations_per_stay", counts, "count"),
        ("observations_per_hour_per_stay", counts / 4.0, "count_per_hour"),
        ("time_between_observations_minutes", differences, "minutes"),
        ("within_hour_observations", within_hour, "count"),
        ("observation_span_minutes", spans, "minutes"),
        ("unique_spo2_values_per_stay", unique_values, "count"),
        ("maximum_identical_run_per_stay", max_run, "count"),
    ]:
        for statistic, value in _summaries(pd.Series(series)).items():
            metric_values[f"{metric}:{statistic}"] = (value, int(pd.Series(series).notna().sum()), unit)
    fractions = {
        "stays_with_gap_over_60_minutes": float(gap_stats["max"].gt(60).mean()),
        "stays_with_gap_over_120_minutes": float(gap_stats["max"].gt(120).mean()),
        "stays_with_exactly_3_valid_bins": float(bins.eq(3).mean()),
        "stays_with_exactly_4_valid_bins": float(bins.eq(4).mean()),
        "stays_with_exactly_5_valid_bins": float(bins.eq(5).mean()),
        "raw_observations_integer_valued": float(integer.mean()) if len(integer) else math.nan,
        "integer_observations_terminal_0_or_5": float(terminal_five.mean()) if len(terminal_five) else math.nan,
        "consecutive_observations_identical": float(same.mean()) if len(same) else math.nan,
        "raw_observations_equal_100": float(clean["spo2"].eq(100).mean()) if len(clean) else math.nan,
    }
    rows: list[dict[str, Any]] = []
    for key, (value, n, unit) in metric_values.items():
        metric, statistic = key.split(":", 1)
        rows.append({"dataset": dataset, "metric": metric, "statistic": statistic, "value": value, "n": n, "unit": unit})
    for metric, value in fractions.items():
        rows.append({"dataset": dataset, "metric": metric, "statistic": "fraction", "value": value, "n": len(eligible), "unit": "fraction"})
    for bin_count, count in bins.value_counts().sort_index().items():
        rows.append({"dataset": dataset, "metric": "valid_bin_distribution", "statistic": str(int(bin_count)), "value": int(count), "n": len(eligible), "unit": "stays"})
    return pd.DataFrame(rows), per_stay


def feature_distribution_rows(
    features: pd.DataFrame,
    dataset: str,
    parameters: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize raw and frozen-standardized exposure distributions."""
    local = features.copy()
    scaled = standardize_with_mimic(local["raw_rms_residual"], parameters)
    local = pd.concat([local.reset_index(drop=True), scaled.reset_index(drop=True).drop(columns="raw_rms_residual")], axis=1)
    rows: list[dict[str, Any]] = []
    for scale_name, column in [("raw", "raw_rms_residual"), ("mimic_standardized", "standardized_exposure")]:
        values = local[column].dropna()
        for statistic, value in _summaries(values).items():
            rows.append({"dataset": dataset, "stratum": "all_valid", "scale": scale_name, "statistic": statistic, "value": value, "n": len(values)})
        rows.append({"dataset": dataset, "stratum": "all_valid", "scale": scale_name, "statistic": "skewness", "value": float(skew(values, bias=False)) if len(values) > 2 else math.nan, "n": len(values)})
    for label, column in [("fraction_clipped_low", "clipped_low"), ("fraction_clipped_high", "clipped_high")]:
        rows.append({"dataset": dataset, "stratum": "all_valid", "scale": "raw", "statistic": label, "value": float(local[column].mean()), "n": int(local[column].notna().sum())})
    for bins, group in local.dropna(subset=["raw_rms_residual"]).groupby("bins"):
        for statistic, value in _summaries(group["raw_rms_residual"]).items():
            rows.append({"dataset": dataset, "stratum": f"bins_{int(bins)}", "scale": "raw", "statistic": statistic, "value": value, "n": len(group)})
    return pd.DataFrame(rows), local


def clinical_context_rows(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Standardize descriptive summaries of harmonized pre-landmark variables."""
    variables = [
        "age", "male_sex", "spo2_level", "spo2_min", "spo2_mean", "sbp_level",
        "map_level", "hr_level", "resp_rate_level", "baseline_lactate",
        "baseline_creatinine", "ventilation", "fio2", "oxygen_support",
    ]
    rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable not in frame:
            rows.append({"dataset": dataset, "variable": variable, "statistic": "availability", "value": 0.0, "n": 0, "note": "not harmonized in frozen validation frame"})
            continue
        values = pd.to_numeric(frame[variable], errors="coerce")
        rows.append({"dataset": dataset, "variable": variable, "statistic": "availability", "value": float(values.notna().mean()), "n": int(values.notna().sum()), "note": "pre-landmark descriptive"})
        for statistic, value in _summaries(values).items():
            rows.append({"dataset": dataset, "variable": variable, "statistic": statistic, "value": value, "n": int(values.notna().sum()), "note": "pre-landmark descriptive"})
    return pd.DataFrame(rows)


def measurement_pattern_hash(raw: pd.DataFrame) -> str:
    """Hash outcome-independent timing and valid-value patterns."""
    local = raw[["stay_id", "minute", "spo2"]].copy().sort_values(["stay_id", "minute", "spo2"])
    hashed = pd.util.hash_pandas_object(local, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def derive_degradation_specification(
    mimic_raw: pd.DataFrame,
    eicu_raw: pd.DataFrame,
    mimic_features: pd.DataFrame,
    eicu_features: pd.DataFrame,
    *,
    seed: int = 20260908,
) -> dict[str, Any]:
    """Define an eICU timing-template operator without outcome information."""
    assert_prelandmark_measurements(mimic_raw)
    assert_prelandmark_measurements(eicu_raw)
    mimic_integer = float(np.isclose(mimic_raw["spo2"], np.round(mimic_raw["spo2"]), atol=1e-8).mean())
    eicu_integer = float(np.isclose(eicu_raw["spo2"], np.round(eicu_raw["spo2"]), atol=1e-8).mean())
    quantization = "nearest_integer" if eicu_integer - mimic_integer >= 0.05 else "none"
    valid_eicu = set(eicu_features.loc[eicu_features["raw_rms_residual"].notna(), "stay_id"])
    template_rows = eicu_raw.loc[eicu_raw["stay_id"].isin(valid_eicu), ["stay_id", "minute", "spo2"]]
    if not valid_eicu or mimic_features["raw_rms_residual"].notna().sum() == 0:
        raise ValueError("Insufficient measurement-only support for degradation templates")
    return {
        "feature_specification": FROZEN_FEATURE_SPEC,
        "operator": "eicu_empirical_timing_template_with_mimic_last_observation_carried_forward",
        "seed": seed,
        "eicu_template_stays": len(valid_eicu),
        "template_pattern_sha256": measurement_pattern_hash(template_rows),
        "donor_assignment": "sorted MIMIC stay IDs; seeded eICU-template sampling with replacement",
        "value_mapping": "most recent actual MIMIC observation at or before each eICU template time; earliest subsequent actual observation only when no prior value exists",
        "missing_hour_pattern": "inherited exactly from the sampled eICU template",
        "cadence": "all valid 0-240 minute timestamps from the sampled eICU template",
        "quantization": quantization,
        "arbitrary_noise": False,
        "outcome_information_used": False,
        "justification": {
            "mimic_raw_observations_median": float(mimic_raw.groupby("stay_id").size().median()),
            "eicu_raw_observations_median": float(eicu_raw.groupby("stay_id").size().median()),
            "mimic_valid_bins_median": float(mimic_features["bins"].median()),
            "eicu_valid_bins_median": float(eicu_features["bins"].median()),
            "mimic_integer_fraction": mimic_integer,
            "eicu_integer_fraction": eicu_integer,
        },
        "prohibited": ["outcome_tuning", "effect_targeting", "gaussian_noise", "alternative_spo2_feature", "threshold_search"],
    }


def write_immutable_degradation_lock(path: Path, payload: dict[str, Any]) -> str:
    """Write the degradation lock once and return its whole-file hash."""
    if path.exists():
        raise FileExistsError(f"Degradation lock already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return sha256_file(path)


def require_degradation_lock_before_outcomes(lock_path: Path, requested_columns: list[str]) -> None:
    """Block outcome access until an immutable degradation specification exists."""
    if not lock_path.exists():
        raise RuntimeError("Outcome access prohibited before measurement-degradation lock")
    lowered = [column.lower() for column in requested_columns]
    if not any(any(token in column for token in OUTCOME_TOKENS) for column in lowered):
        return
    sidecar = lock_path.with_suffix(".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != sha256_file(lock_path):
        raise RuntimeError("Measurement-degradation lock hash is missing or invalid")


def apply_locked_timing_degradation(
    mimic_raw: pd.DataFrame,
    eicu_raw: pd.DataFrame,
    mimic_stay_ids: pd.Series,
    specification: dict[str, Any],
) -> pd.DataFrame:
    """Apply the single seed-locked eICU timing template to MIMIC observations."""
    assert_prelandmark_measurements(mimic_raw)
    assert_prelandmark_measurements(eicu_raw)
    if specification.get("operator") != "eicu_empirical_timing_template_with_mimic_last_observation_carried_forward":
        raise ValueError("Unrecognized or altered degradation operator")
    if specification.get("outcome_information_used") is not False:
        raise ValueError("Outcome-dependent degradation is prohibited")
    validate_frozen_feature_spec(specification["feature_specification"])
    eicu_features = frozen_exposure_from_raw(eicu_raw)
    valid_templates = sorted(
        eicu_features.loc[eicu_features["raw_rms_residual"].notna(), "stay_id"].unique()
    )
    template_rows = eicu_raw.loc[eicu_raw["stay_id"].isin(valid_templates), ["stay_id", "minute", "spo2"]]
    if measurement_pattern_hash(template_rows) != specification["template_pattern_sha256"]:
        raise RuntimeError("eICU measurement template hash changed after degradation lock")
    rng = np.random.default_rng(int(specification["seed"]))
    source_ids = sorted(pd.Series(mimic_stay_ids).dropna().unique())
    selected = rng.choice(np.asarray(valid_templates), size=len(source_ids), replace=True)
    eicu_groups = {stay: group.sort_values("minute") for stay, group in eicu_raw.groupby("stay_id", sort=False)}
    mimic_groups = {stay: group.sort_values("minute") for stay, group in mimic_raw.groupby("stay_id", sort=False)}
    rows: list[pd.DataFrame] = []
    for mimic_stay, template_stay in zip(source_ids, selected, strict=True):
        source = mimic_groups.get(mimic_stay)
        if source is None or source.empty:
            continue
        template = eicu_groups[int(template_stay)]
        source_times = source["minute"].to_numpy(float)
        source_values = source["spo2"].to_numpy(float)
        template_times = template["minute"].to_numpy(float)
        positions = np.searchsorted(source_times, template_times, side="right") - 1
        positions = np.where(positions < 0, 0, positions)
        mapped = source_values[positions]
        if specification["quantization"] == "nearest_integer":
            mapped = np.round(mapped)
        rows.append(pd.DataFrame({"stay_id": mimic_stay, "minute": template_times, "spo2": mapped}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["stay_id", "minute", "spo2"])


def degradation_agreement(original: pd.DataFrame, degraded: pd.DataFrame) -> dict[str, Any]:
    """Quantify reliability and rank preservation of the degraded exposure."""
    paired = original[["stay_id", "raw_rms_residual"]].rename(columns={"raw_rms_residual": "original"}).merge(
        degraded[["stay_id", "raw_rms_residual"]].rename(columns={"raw_rms_residual": "degraded"}),
        on="stay_id", how="inner",
    ).dropna()
    if len(paired) < 3:
        raise ValueError("Insufficient paired exposures")
    pearson = float(paired[["original", "degraded"]].corr().iloc[0, 1])
    rank = float(spearmanr(paired["original"], paired["degraded"]).statistic)
    slope = float(np.polyfit(paired["original"], paired["degraded"], 1)[0])
    paired["original_quartile"] = pd.qcut(paired["original"], 4, labels=False, duplicates="drop")
    paired["degraded_quartile"] = pd.qcut(paired["degraded"], 4, labels=False, duplicates="drop")
    return {
        "n_paired": len(paired),
        "pearson_correlation": pearson,
        "spearman_rank_correlation": rank,
        "reliability_slope_degraded_on_original": slope,
        "original_sd": float(paired["original"].std(ddof=0)),
        "degraded_sd": float(paired["degraded"].std(ddof=0)),
        "original_iqr": float(paired["original"].quantile(0.75) - paired["original"].quantile(0.25)),
        "degraded_iqr": float(paired["degraded"].quantile(0.75) - paired["degraded"].quantile(0.25)),
        "same_quartile_fraction": float(paired["original_quartile"].eq(paired["degraded_quartile"]).mean()),
        "moved_two_or_more_quartiles_fraction": float(paired["original_quartile"].sub(paired["degraded_quartile"]).abs().ge(2).mean()),
    }


def cross_database_synthesis(locked_result: dict[str, Any]) -> dict[str, Any]:
    """Compute descriptive two-database meta-analytic quantities without rescue claims."""
    mimic = locked_result["mimic"]["primary"]
    eicu = locked_result["primary"]
    beta = np.array([float(mimic["log_rr"]), float(eicu["log_rr"])])
    se = np.array([
        implied_se(beta[0], float(mimic["ci_low"]), float(mimic["ci_high"])),
        implied_se(beta[1], float(eicu["ci_low"]), float(eicu["ci_high"])),
    ])
    weights = 1 / se**2
    fixed_beta = float(np.sum(weights * beta) / np.sum(weights))
    fixed_se = float(np.sqrt(1 / np.sum(weights)))
    q = float(np.sum(weights * (beta - fixed_beta) ** 2))
    df = 1
    c_value = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau2 = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
    random_weights = 1 / (se**2 + tau2)
    random_beta = float(np.sum(random_weights * beta) / np.sum(random_weights))
    random_se = float(np.sqrt(1 / np.sum(random_weights)))
    interaction_z = float((beta[0] - beta[1]) / np.sqrt(np.sum(se**2)))
    return {
        "studies": [
            {"dataset": "MIMIC", "log_rr": float(beta[0]), "se": float(se[0]), "rr": math.exp(beta[0])},
            {"dataset": "eICU", "log_rr": float(beta[1]), "se": float(se[1]), "rr": math.exp(beta[1])},
        ],
        "fixed_effect": {"log_rr": fixed_beta, "se": fixed_se, "rr": math.exp(fixed_beta), "ci_low": math.exp(fixed_beta - 1.96 * fixed_se), "ci_high": math.exp(fixed_beta + 1.96 * fixed_se)},
        "random_effects_dl": {"log_rr": random_beta, "se": random_se, "rr": math.exp(random_beta), "ci_low": math.exp(random_beta - 1.96 * random_se), "ci_high": math.exp(random_beta + 1.96 * random_se), "tau_squared": tau2},
        "heterogeneity": {"q": q, "df": df, "p_value": float(chi2.sf(q, df)), "i_squared_percent": max(0.0, (q - df) / q * 100) if q > 0 else 0.0, "warning": "I-squared and tau-squared are highly unstable with only two datasets."},
        "database_by_exposure_interaction": {"z": interaction_z, "p_value": float(2 * norm.sf(abs(interaction_z))), "method": "independent_summary_estimate_difference"},
        "interpretation_guard": "Pooling does not convert a failed external validation into successful external validation.",
    }
