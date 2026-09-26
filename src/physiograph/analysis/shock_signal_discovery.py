"""Finite MIMIC screen for one routine precursor of objective shock progression."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

CANDIDATE_SIGNALS = (
    "hr",
    "sbp",
    "dbp",
    "map",
    "pulse_pressure",
    "resp_rate",
    "spo2",
    "temperature_c",
    "urine_output_ml_h",
)
CANDIDATE_DYNAMICS = ("level", "slope", "change", "variability")
BP_SIGNALS = frozenset({"sbp", "dbp", "map", "pulse_pressure"})


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return monotone Benjamini-Hochberg q-values with missing values retained."""

    numeric = pd.to_numeric(p_values, errors="coerce")
    valid = numeric.dropna().clip(0.0, 1.0)
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    if valid.empty:
        return result
    ordered = valid.sort_values()
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    adjusted = ordered.to_numpy(float) * len(p_values) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[ordered.index] = np.minimum(adjusted, 1.0)
    return result


def trajectory_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build the four frozen trajectory features for each stay and signal."""

    required = {"stay_id", "signal", "hour", "value"}
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"Missing hourly columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    clean = hourly.copy()
    clean["hour"] = pd.to_numeric(clean["hour"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna(subset=["stay_id", "signal", "hour", "value"])
    clean = clean.loc[clean["hour"].between(0, 4)]
    for (stay_id, signal), group in clean.groupby(["stay_id", "signal"], sort=False):
        summary = group.groupby("hour", as_index=False)["value"].median().sort_values("hour")
        hours = summary["hour"].to_numpy(float)
        values = summary["value"].to_numpy(float)
        row: dict[str, Any] = {
            "stay_id": stay_id,
            "signal": signal,
            "bins": len(values),
            "span_hours": float(hours[-1] - hours[0]) if len(hours) else 0.0,
            "level": float(values[-1]) if len(values) else np.nan,
            "slope": np.nan,
            "change": np.nan,
            "variability": np.nan,
        }
        if len(values) >= 3 and row["span_hours"] >= 2:
            design = np.column_stack([np.ones(len(hours)), hours])
            beta = np.linalg.lstsq(design, values, rcond=None)[0]
            residual = values - design @ beta
            row.update(
                slope=float(beta[1]),
                change=float(values[-1] - values[0]),
                variability=float(np.sqrt(np.mean(residual**2))),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sustained_hypotension_events(hourly_bp: pd.DataFrame) -> pd.DataFrame:
    """Find the second of two consecutive hourly hypotensive cuff bins."""

    required = {"stay_id", "hour", "sbp", "map"}
    missing = required - set(hourly_bp.columns)
    if missing:
        raise ValueError(f"Missing BP columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for stay_id, group in hourly_bp.sort_values("hour").groupby("stay_id", sort=False):
        local = group.copy()
        local["low"] = local["sbp"].lt(90) | local["map"].lt(65)
        prior_low = local["low"].shift(fill_value=False)
        consecutive = local["hour"].diff().eq(1)
        events = local.loc[local["low"] & prior_low & consecutive]
        for index, event in events.iterrows():
            hour = event["hour"]
            available = float((hour + 1) * 60)
            if "available_minute" in local:
                prior_available = local["available_minute"].shift().loc[index]
                available = max(available, float(event["available_minute"]), float(prior_available))
            rows.append(
                {
                    "stay_id": stay_id,
                    "event_minute": available,
                    "component": "sustained_hypotension",
                }
            )
    return pd.DataFrame(rows, columns=["stay_id", "event_minute", "component"])


def rolling_oliguria_events(
    hourly_urine: pd.DataFrame,
    *,
    start_hour: int = 4,
    end_hour: int = 16,
    window_hours: int = 6,
    minimum_documented_hours: int = 4,
    threshold_ml_h: float = 30.0,
) -> pd.DataFrame:
    """Return low documented urine-output windows without treating missing as zero."""

    rows: list[dict[str, Any]] = []
    for stay_id, group in hourly_urine.groupby("stay_id", sort=False):
        values = group.set_index("hour")["value"].astype(float)
        for window_end in range(start_hour + window_hours, end_hour + 1):
            window = values.reindex(range(window_end - window_hours, window_end)).dropna()
            if len(window) < minimum_documented_hours:
                continue
            if float(window.mean()) < threshold_ml_h:
                available = float(window_end * 60)
                if "available_minute" in group:
                    recorded = group.set_index("hour")["available_minute"].reindex(window.index)
                    available = max(available, float(recorded.max()))
                rows.append(
                    {
                        "stay_id": stay_id,
                        "event_minute": available,
                        "component": "oliguria",
                    }
                )
                break
    return pd.DataFrame(rows, columns=["stay_id", "event_minute", "component"])


def pair_domain_events(
    pressure_events: pd.DataFrame,
    perfusion_events: pd.DataFrame,
    *,
    lower_minute: float,
    upper_minute: float,
    max_separation_minutes: float = 360.0,
) -> pd.DataFrame:
    """Return first composite time when pressure/support and perfusion pair."""

    columns = ["stay_id", "event_minute", "pressure_component", "perfusion_component"]
    if pressure_events.empty or perfusion_events.empty:
        return pd.DataFrame(columns=columns)
    left = pressure_events.rename(
        columns={"event_minute": "pressure_minute", "component": "pressure_component"}
    )
    right = perfusion_events.rename(
        columns={"event_minute": "perfusion_minute", "component": "perfusion_component"}
    )
    paired = left.merge(right, on="stay_id", how="inner")
    paired["event_minute"] = paired[["pressure_minute", "perfusion_minute"]].max(axis=1)
    paired = paired.loc[
        paired["event_minute"].gt(lower_minute)
        & paired["event_minute"].le(upper_minute)
        & paired["pressure_minute"].gt(lower_minute)
        & paired["perfusion_minute"].gt(lower_minute)
        & paired["pressure_minute"].sub(paired["perfusion_minute"]).abs().le(max_separation_minutes)
    ].copy()
    if paired.empty:
        return pd.DataFrame(columns=columns)
    paired = paired.sort_values(
        ["stay_id", "event_minute", "pressure_component", "perfusion_component"]
    ).drop_duplicates("stay_id")
    return paired[columns].reset_index(drop=True)


def prepare_adjustment_covariates(
    frame: pd.DataFrame,
    covariates: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply frozen winsorization, imputation, standardization, and missing flags."""

    local = frame.copy()
    columns: list[str] = []
    for column in covariates:
        values = pd.to_numeric(local[column], errors="coerce")
        observed = values.dropna()
        lower = float(observed.quantile(0.01)) if len(observed) else 0.0
        upper = float(observed.quantile(0.99)) if len(observed) else 0.0
        median = float(observed.median()) if len(observed) else 0.0
        clipped = values.clip(lower, upper).fillna(median)
        std = float(clipped.std(ddof=0))
        model_name = f"z_{column}"
        local[model_name] = (clipped - float(clipped.mean())) / std if std > 0 else 0.0
        if std > 1e-12:
            columns.append(model_name)
        if values.isna().any() and values.notna().any():
            missing_name = f"missing_{column}"
            local[missing_name] = values.isna().astype(int)
            columns.append(missing_name)
    return local, columns


def fit_candidate_model(
    frame: pd.DataFrame,
    *,
    feature: str,
    outcome: str,
    adjustment_covariates: list[str],
) -> tuple[dict[str, float | int | str], Any]:
    """Fit one prespecified robust modified-Poisson candidate model."""

    observed = frame.dropna(subset=[feature, outcome]).copy()
    raw_feature = pd.to_numeric(observed[feature], errors="coerce")
    lower, upper = raw_feature.quantile([0.01, 0.99])
    clipped = raw_feature.clip(float(lower), float(upper))
    standard_deviation = float(clipped.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        raise ValueError(f"Candidate {feature} has zero variance")
    if observed[outcome].nunique() < 2:
        raise ValueError("Model requires events and nonevents")
    observed["candidate_z"] = (clipped - float(clipped.mean())) / standard_deviation
    prepared, columns = prepare_adjustment_covariates(observed, adjustment_covariates)
    design = sm.add_constant(prepared[["candidate_z", *columns]].astype(float), has_constant="add")
    # Keep the intercept and candidate, then remove only algebraically redundant
    # adjustment columns (e.g. lactate_observed == 1 - missing_baseline_lactate).
    retained: list[str] = []
    for column in design.columns:
        trial = design[[*retained, column]]
        if np.linalg.matrix_rank(trial.to_numpy(), tol=1e-10) > len(retained):
            retained.append(column)
    design = design[retained]
    model = sm.GLM(prepared[outcome].astype(int), design, family=sm.families.Poisson())
    fit = model.fit(cov_type="HC0")
    beta = float(fit.params["candidate_z"])
    se = float(fit.bse["candidate_z"])
    result: dict[str, float | int | str] = {
        "n": len(prepared),
        "events": int(prepared[outcome].sum()),
        "coverage": len(prepared) / len(frame),
        "rr_per_sd": math.exp(beta),
        "ci_low": math.exp(beta - 1.96 * se),
        "ci_high": math.exp(beta + 1.96 * se),
        "p_value": float(fit.pvalues["candidate_z"]),
        "log_rr": beta,
        "log_rr_se": se,
        "effect_magnitude": math.exp(abs(beta)),
        "feature_sd": standard_deviation,
        "covariance": "HC0",
    }
    return result, fit


def candidate_adjustment_set(signal: str, base_covariates: list[str]) -> list[str]:
    """Remove only an exactly duplicated current-level adjustment variable."""

    duplicate = {
        "hr": "hr_level",
        "sbp": "sbp_level",
        "map": "map_level",
        "resp_rate": "resp_rate_level",
        "spo2": "spo2_level",
        "temperature_c": "temperature_c_level",
        "urine_output_ml_h": "urine_output_ml_h_level",
    }.get(signal)
    return [column for column in base_covariates if column != duplicate]


def assemble_candidate_frame(
    model_base: pd.DataFrame,
    feature_long: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    """Attach a candidate unless its level value is already present in the model base."""

    if feature in model_base.columns:
        return model_base.copy()
    values = feature_long.loc[
        feature_long["feature"].eq(feature), ["stay_id", "value"]
    ].rename(columns={"value": feature})
    return model_base.merge(values, on="stay_id", how="left", validate="one_to_one")


def bootstrap_candidate(
    frame: pd.DataFrame,
    *,
    feature: str,
    outcome: str,
    adjustment_covariates: list[str],
    replicates: int = 200,
    seed: int = 20260907,
) -> dict[str, float | int]:
    """Outcome-stratified stay-level bootstrap for one already-gated candidate."""

    local = frame.dropna(subset=[feature, outcome]).copy()
    event = local.loc[local[outcome].eq(1)].index.to_numpy()
    nonevent = local.loc[local[outcome].eq(0)].index.to_numpy()
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled = np.concatenate(
            [rng.choice(event, size=len(event), replace=True), rng.choice(nonevent, size=len(nonevent), replace=True)]
        )
        boot = local.loc[sampled].reset_index(drop=True)
        try:
            result, _ = fit_candidate_model(
                boot,
                feature=feature,
                outcome=outcome,
                adjustment_covariates=adjustment_covariates,
            )
            estimates.append(float(result["log_rr"]))
        except (ValueError, np.linalg.LinAlgError):
            continue
    estimate_array = np.asarray(estimates, dtype=float)
    if not len(estimate_array):
        return {
            "bootstrap_replicates": replicates,
            "bootstrap_successful": 0,
            "bootstrap_success_fraction": 0.0,
            "bootstrap_sign_consistency": 0.0,
            "bootstrap_median_rr": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }
    reference_sign = np.sign(float(fit_candidate_model(
        local,
        feature=feature,
        outcome=outcome,
        adjustment_covariates=adjustment_covariates,
    )[0]["log_rr"]))
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_successful": len(estimate_array),
        "bootstrap_success_fraction": len(estimate_array) / replicates,
        "bootstrap_sign_consistency": float(np.mean(np.sign(estimate_array) == reference_sign)),
        "bootstrap_median_rr": float(np.exp(np.median(estimate_array))),
        "bootstrap_ci_low": float(np.exp(np.quantile(estimate_array, 0.025))),
        "bootstrap_ci_high": float(np.exp(np.quantile(estimate_array, 0.975))),
    }


def statistical_winner_gate(row: pd.Series) -> bool:
    """Evaluate all frozen non-novelty winner criteria."""

    return bool(
        row["coverage"] >= 0.70
        and row["cohort_events"] >= 100
        and row["events"] >= 50
        and row["q_value"] < 0.05
        and (row["ci_low"] > 1 or row["ci_high"] < 1)
    ) and bool(
        row["effect_magnitude"] >= 1.30
        and row["lead_q_value"] < 0.05
        and (row["lead_ci_low"] > 1 or row["lead_ci_high"] < 1)
        and np.sign(row["log_rr"]) == np.sign(row["lead_log_rr"])
        and row["lead_effect_magnitude"] >= 1.20
    )


def final_winner_gate(row: pd.Series) -> bool:
    """Evaluate stability and anti-circularity after the statistical gate."""

    return bool(
        statistical_winner_gate(row)
        and row["bootstrap_success_fraction"] >= 0.95
        and row["bootstrap_sign_consistency"] >= 0.90
        and (row["bootstrap_ci_low"] > 1 or row["bootstrap_ci_high"] < 1)
        and np.sign(row["log_rr"]) == np.sign(row["leave_domain_log_rr"])
        and (row["leave_domain_ci_low"] > 1 or row["leave_domain_ci_high"] < 1)
        and row["leave_domain_effect_magnitude"] >= 1.20
    )


def classify_discovery(*, eligible_winners: int, novelty_passed: bool | None) -> str:
    """Return the frozen discovery state without substituting a runner-up."""

    if eligible_winners == 0:
        return "NO_DISCOVERY_SIGNAL"
    if novelty_passed is None:
        return "PROVISIONAL_WINNER_REQUIRES_NOVELTY_REVIEW"
    if not novelty_passed:
        return "KNOWN_SIGNAL_NO_NOVEL_WINNER"
    return "ONE_SIGNAL_READY_FOR_EXTERNAL_FREEZE"
