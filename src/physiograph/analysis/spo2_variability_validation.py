"""Locked helpers for targeted SpO2-variability replication."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .shock_signal_discovery import prepare_adjustment_covariates


def fixed_exposure_parameters(values: pd.Series) -> dict[str, float]:
    """Return the development-cohort winsorization and scaling constants."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    lower, upper = numeric.quantile([0.01, 0.99])
    clipped = numeric.clip(float(lower), float(upper))
    return {"lower": float(lower), "upper": float(upper), "center": float(clipped.mean()), "scale": float(clipped.std(ddof=0))}


def fit_fixed_scale_model(
    frame: pd.DataFrame,
    *,
    feature: str,
    outcome: str,
    adjustment_covariates: list[str],
    exposure_parameters: dict[str, float],
    cluster: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Fit modified Poisson using the MIMIC-frozen exposure transformation."""
    observed = frame.dropna(subset=[feature, outcome, *([cluster] if cluster else [])]).copy()
    raw = pd.to_numeric(observed[feature], errors="coerce")
    clipped = raw.clip(exposure_parameters["lower"], exposure_parameters["upper"])
    observed["candidate_z"] = (clipped - exposure_parameters["center"]) / exposure_parameters["scale"]
    prepared, columns = prepare_adjustment_covariates(observed, adjustment_covariates)
    design = sm.add_constant(prepared[["candidate_z", *columns]].astype(float), has_constant="add")
    retained: list[str] = []
    for column in design.columns:
        trial = design[[*retained, column]].to_numpy()
        if np.linalg.matrix_rank(trial, tol=1e-10) > len(retained):
            retained.append(column)
    design = design[retained]
    model = sm.GLM(prepared[outcome].astype(int), design, family=sm.families.Poisson())
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": prepared[cluster]}) if cluster else model.fit(cov_type="HC0")
    beta = float(fit.params["candidate_z"]); se = float(fit.bse["candidate_z"])
    return {
        "n": len(prepared), "events": int(prepared[outcome].sum()),
        "coverage": float(len(prepared) / len(frame)), "rr_per_mimic_sd": math.exp(beta),
        "ci_low": math.exp(beta - 1.96 * se), "ci_high": math.exp(beta + 1.96 * se),
        "p_value": float(fit.pvalues["candidate_z"]), "log_rr": beta,
        "hospitals": int(prepared[cluster].nunique()) if cluster else None,
        "covariance": "cluster_hospital" if cluster else "HC0",
    }, fit


def stratified_bootstrap_fixed(
    frame: pd.DataFrame, *, feature: str, outcome: str, adjustment_covariates: list[str],
    exposure_parameters: dict[str, float], replicates: int = 1000, seed: int = 20260907,
) -> dict[str, Any]:
    """Outcome-stratified bootstrap with the fixed MIMIC transformation."""
    local = frame.dropna(subset=[feature, outcome]).copy(); rng = np.random.default_rng(seed)
    event = local.index[local[outcome].eq(1)].to_numpy(); nonevent = local.index[local[outcome].eq(0)].to_numpy(); estimates=[]
    for _ in range(replicates):
        sample = np.concatenate([rng.choice(event, len(event), replace=True), rng.choice(nonevent, len(nonevent), replace=True)])
        try:
            result, _ = fit_fixed_scale_model(local.loc[sample].reset_index(drop=True), feature=feature, outcome=outcome, adjustment_covariates=adjustment_covariates, exposure_parameters=exposure_parameters)
            estimates.append(result["log_rr"])
        except (ValueError, np.linalg.LinAlgError):
            continue
    values=np.asarray(estimates); return {"replicates": replicates, "successful": len(values), "success_fraction": float(len(values)/replicates), "sign_consistency": float(np.mean(values>0)), "median_rr": float(np.exp(np.median(values))), "ci_low": float(np.exp(np.quantile(values,.025))), "ci_high": float(np.exp(np.quantile(values,.975)))}
