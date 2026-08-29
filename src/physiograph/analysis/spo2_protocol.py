"""Leakage-safe analysis for SpO2 instability and early decompensation.

The clock is explicit throughout: predictors use [0, 240) minutes after ICU
admission and outcomes use (240, 240 + horizon] minutes.  A suffix such as
``_12h`` therefore means twelve hours *after the landmark*, not ICU hour 12.

Binary thresholds are sensitivity-friendly summaries of continuous outcomes;
the continuous deltas and availability indicators are always retained.  VIS is
only considered observed when an extractor supplied normalized ``concept=vis``
events.  Urine-output decline is named a proxy unless weight-normalized data
support the KDIGO oliguria threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

LANDMARK_MINUTES = 240
POST_LANDMARK_HORIZONS_HOURS = (12, 24)
LACTATE_RISE_ABSOLUTE = 0.5
CREATININE_AKI_ABSOLUTE = 0.3
CREATININE_AKI_RATIO = 1.5
BILIRUBIN_WORSENING_ABSOLUTE = 0.5
BILIRUBIN_WORSENING_RATIO = 1.5
URINE_RATE_DECLINE_FRACTION = 0.5
KDIGO_OLIGURIA_ML_KG_H = 0.5

PRIMARY_ENDPOINTS = (
    "lactate_rise_12h_flag",
    "lactate_rise_24h_flag",
    "vis_rise_12h_flag",
    "vis_rise_24h_flag",
)
SECONDARY_ENDPOINTS = (
    "urine_output_decline_proxy_12h_flag",
    "urine_output_decline_proxy_24h_flag",
    "aki_creatinine_12h_flag",
    "aki_creatinine_24h_flag",
    "hepatic_lab_worsening_12h_flag",
    "hepatic_lab_worsening_24h_flag",
    "mcs_12h_flag",
    "mcs_24h_flag",
    "early_decompensation_12h_flag",
    "early_decompensation_24h_flag",
)

ABSOLUTE_SPO2_FEATURES = (
    "spo2_mean",
    "spo2_min",
    "spo2_below_90_fraction",
    "spo2_sampling_density_per_hr",
)
INSTABILITY_FEATURES = (
    "spo2_sd",
    "spo2_rmssd",
    "spo2_iqr",
    "spo2_range",
    "spo2_mad",
    "spo2_abrupt_jump_rate_per_hr",
    "spo2_instability_proxy_score",
)
CLINICAL_CONTROL_CANDIDATES = (
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
MISSINGNESS_CONTROL_FEATURES = (
    "spo2_sampling_density_per_hr",
    "spo2_missing_bin_count",
    "spo2_longest_gap_minutes",
    "spo2_plausible_count",
)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _keys_from(*frames: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or frame.empty or "stay_id" not in frame:
            continue
        local = frame.copy()
        if "dataset" not in local:
            local["dataset"] = "unknown"
        pieces.append(local[["dataset", "stay_id"]].drop_duplicates())
    if not pieces:
        return pd.DataFrame(columns=["dataset", "stay_id"])
    return pd.concat(pieces, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _last_measurement(events: pd.DataFrame, concept: str) -> pd.DataFrame:
    selected = events.loc[events["concept"].eq(concept)].dropna(subset=["value_numeric"])
    if selected.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "value_numeric", "offset_minutes"])
    return (
        selected.sort_values(["dataset", "stay_id", "offset_minutes"])
        .groupby(["dataset", "stay_id"], as_index=False, sort=False)
        .tail(1)[["dataset", "stay_id", "value_numeric", "offset_minutes"]]
    )


def _max_measurement(events: pd.DataFrame, concept: str) -> pd.DataFrame:
    selected = events.loc[events["concept"].eq(concept)].dropna(subset=["value_numeric"])
    if selected.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    idx = selected.groupby(["dataset", "stay_id"])["value_numeric"].idxmax()
    return selected.loc[idx, ["dataset", "stay_id", "value_numeric"]]


def _merge_value(
    result: pd.DataFrame,
    source: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:
    if source.empty:
        result[target_col] = np.nan
        return result
    renamed = source[["dataset", "stay_id", source_col]].rename(columns={source_col: target_col})
    return result.merge(renamed, on=["dataset", "stay_id"], how="left")


def _binary_when_observed(condition: pd.Series, observed: pd.Series) -> pd.Series:
    values = pd.Series(np.nan, index=condition.index, dtype=float)
    values.loc[observed] = condition.loc[observed].astype(int)
    return values


def _event_flag(events: pd.DataFrame, concept: str, keys: pd.DataFrame) -> pd.Series:
    flagged = events.loc[events["concept"].eq(concept), ["dataset", "stay_id"]].drop_duplicates()
    if flagged.empty:
        return pd.Series(0, index=keys.index, dtype=int)
    merged = keys.merge(flagged.assign(_flag=1), on=["dataset", "stay_id"], how="left")
    return merged["_flag"].fillna(0).astype(int)


def compute_early_decompensation_outcomes(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    stay_index: pd.DataFrame | None = None,
    *,
    horizons: tuple[int, ...] = POST_LANDMARK_HORIZONS_HOURS,
) -> pd.DataFrame:
    """Derive prespecified 12/24-hour post-landmark outcomes and availability.

    ``lactate_rise`` is a last-value increase of at least 0.5 mmol/L from the
    last observation-window lactate; the continuous delta and a threshold-
    crossing sensitivity outcome are retained.  Creatinine follows the KDIGO
    absolute/relative criteria.  Bilirubin is explicitly named lab worsening,
    not acute liver failure.  Urine decline is a rate proxy; the KDIGO-style
    oliguria proxy is only populated when weight is observed.
    """
    keys = _keys_from(stay_index if stay_index is not None else cohort_df, cohort_df, events_df)
    result = keys.copy()
    if not cohort_df.empty and "person_id" in cohort_df:
        cohort_keys = cohort_df.copy()
        if "dataset" not in cohort_keys:
            cohort_keys["dataset"] = "unknown"
        result = result.merge(
            cohort_keys[["dataset", "stay_id", "person_id"]].drop_duplicates(),
            on=["dataset", "stay_id"],
            how="left",
        )

    events = events_df.copy()
    if events.empty:
        events = pd.DataFrame(columns=["dataset", "stay_id", "concept", "offset_minutes", "value_numeric"])
    if "dataset" not in events:
        events["dataset"] = "unknown"
    for column in ("concept", "offset_minutes", "value_numeric"):
        if column not in events:
            events[column] = np.nan if column != "concept" else ""
    events["concept"] = events["concept"].fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    events["value_numeric"] = pd.to_numeric(events["value_numeric"], errors="coerce")
    observation = events.loc[
        events["offset_minutes"].ge(0) & events["offset_minutes"].lt(LANDMARK_MINUTES)
    ].copy()

    baseline: dict[str, pd.DataFrame] = {
        concept: _last_measurement(observation, concept)
        for concept in ("lactate", "creatinine", "bilirubin_total", "weight", "vis")
    }
    for concept in ("lactate", "creatinine", "bilirubin_total", "weight"):
        result = _merge_value(result, baseline[concept], "value_numeric", f"baseline_{concept}_outcome")
    vis_datasets = set(events.loc[events["concept"].eq("vis"), "dataset"].dropna().astype(str))
    urine_datasets = set(events.loc[events["concept"].eq("urine_output"), "dataset"].dropna().astype(str))

    baseline_urine = observation.loc[
        observation["concept"].eq("urine_output") & observation["value_numeric"].gt(0)
    ]
    baseline_urine_sum = (
        baseline_urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
        if not baseline_urine.empty
        else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
    )
    result = _merge_value(result, baseline_urine_sum, "value_numeric", "baseline_urine_output_ml")
    result["baseline_urine_output_rate_ml_h"] = result["baseline_urine_output_ml"] / 4.0

    for horizon in horizons:
        upper = LANDMARK_MINUTES + int(horizon * 60)
        post = events.loc[
            events["offset_minutes"].gt(LANDMARK_MINUTES)
            & events["offset_minutes"].le(upper)
        ].copy()

        for concept in ("lactate", "creatinine", "bilirubin_total"):
            last = _last_measurement(post, concept)
            maximum = _max_measurement(post, concept)
            result = _merge_value(result, last, "value_numeric", f"{concept}_last_{horizon}h")
            result = _merge_value(result, maximum, "value_numeric", f"{concept}_max_{horizon}h")

        lactate_observed = result["baseline_lactate_outcome"].notna() & result[f"lactate_last_{horizon}h"].notna()
        result[f"lactate_rise_{horizon}h_observed"] = lactate_observed.astype(int)
        result[f"lactate_delta_{horizon}h"] = result[f"lactate_last_{horizon}h"] - result["baseline_lactate_outcome"]
        result[f"lactate_max_delta_{horizon}h"] = result[f"lactate_max_{horizon}h"] - result["baseline_lactate_outcome"]
        result[f"lactate_rise_{horizon}h_flag"] = _binary_when_observed(
            result[f"lactate_delta_{horizon}h"].ge(LACTATE_RISE_ABSOLUTE), lactate_observed
        )
        result[f"lactate_cross_2_{horizon}h_flag"] = _binary_when_observed(
            result["baseline_lactate_outcome"].lt(2.0) & result[f"lactate_max_{horizon}h"].ge(2.0),
            lactate_observed,
        )

        creat_observed = result["baseline_creatinine_outcome"].gt(0) & result[f"creatinine_max_{horizon}h"].notna()
        result[f"creatinine_delta_{horizon}h"] = result[f"creatinine_max_{horizon}h"] - result["baseline_creatinine_outcome"]
        result[f"creatinine_ratio_{horizon}h"] = result[f"creatinine_max_{horizon}h"] / result["baseline_creatinine_outcome"]
        result[f"aki_creatinine_{horizon}h_observed"] = creat_observed.astype(int)
        result[f"aki_creatinine_{horizon}h_flag"] = _binary_when_observed(
            result[f"creatinine_delta_{horizon}h"].ge(CREATININE_AKI_ABSOLUTE)
            | result[f"creatinine_ratio_{horizon}h"].ge(CREATININE_AKI_RATIO),
            creat_observed,
        )

        bili_observed = result["baseline_bilirubin_total_outcome"].gt(0) & result[f"bilirubin_total_max_{horizon}h"].notna()
        result[f"bilirubin_delta_{horizon}h"] = result[f"bilirubin_total_max_{horizon}h"] - result["baseline_bilirubin_total_outcome"]
        result[f"bilirubin_ratio_{horizon}h"] = result[f"bilirubin_total_max_{horizon}h"] / result["baseline_bilirubin_total_outcome"]
        result[f"hepatic_lab_worsening_{horizon}h_observed"] = bili_observed.astype(int)
        result[f"hepatic_lab_worsening_{horizon}h_flag"] = _binary_when_observed(
            result[f"bilirubin_delta_{horizon}h"].ge(BILIRUBIN_WORSENING_ABSOLUTE)
            | result[f"bilirubin_ratio_{horizon}h"].ge(BILIRUBIN_WORSENING_RATIO),
            bili_observed,
        )

        result[f"pressor_initiation_{horizon}h_flag"] = _event_flag(post, "pressor", keys)
        result[f"mcs_{horizon}h_flag"] = _event_flag(post, "mcs", keys)

        vis_max = _max_measurement(post, "vis")
        result = _merge_value(result, vis_max, "value_numeric", f"vis_max_{horizon}h")
        vis_available = result["dataset"].astype(str).isin(vis_datasets)
        base_vis = pd.Series(0.0, index=result.index)
        if not baseline["vis"].empty:
            base_map = baseline["vis"].set_index(["dataset", "stay_id"])["value_numeric"]
            base_vis = pd.Series(
                [base_map.get((d, s), 0.0) for d, s in zip(result["dataset"], result["stay_id"])],
                index=result.index,
                dtype=float,
            )
        result[f"vis_observed_{horizon}h"] = vis_available.astype(int)
        result[f"vis_baseline_max_{horizon}h"] = base_vis.where(vis_available)
        result[f"vis_max_{horizon}h"] = result[f"vis_max_{horizon}h"].fillna(0).where(vis_available)
        result[f"vis_delta_{horizon}h"] = result[f"vis_max_{horizon}h"] - result[f"vis_baseline_max_{horizon}h"]
        result[f"vis_rise_{horizon}h_flag"] = _binary_when_observed(
            result[f"vis_delta_{horizon}h"].gt(0), vis_available
        )

        urine = post.loc[post["concept"].eq("urine_output") & post["value_numeric"].gt(0)]
        urine_sum = (
            urine.groupby(["dataset", "stay_id"], as_index=False)["value_numeric"].sum()
            if not urine.empty
            else pd.DataFrame(columns=["dataset", "stay_id", "value_numeric"])
        )
        result = _merge_value(result, urine_sum, "value_numeric", f"urine_output_{horizon}h_ml")
        result[f"urine_output_rate_{horizon}h_ml_h"] = result[f"urine_output_{horizon}h_ml"] / float(horizon)
        urine_observed = (
            result["dataset"].astype(str).isin(urine_datasets)
            & result["baseline_urine_output_ml"].notna()
            & result[f"urine_output_{horizon}h_ml"].notna()
        )
        result[f"urine_output_{horizon}h_observed"] = urine_observed.astype(int)
        result[f"urine_output_decline_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"].le(
                result["baseline_urine_output_rate_ml_h"] * URINE_RATE_DECLINE_FRACTION
            ),
            urine_observed,
        )
        weight_observed = urine_observed & result["baseline_weight_outcome"].gt(0)
        result[f"oliguria_kdigo_proxy_{horizon}h_flag"] = _binary_when_observed(
            result[f"urine_output_rate_{horizon}h_ml_h"].div(result["baseline_weight_outcome"]).lt(
                KDIGO_OLIGURIA_ML_KG_H
            ),
            weight_observed,
        )

        organ_components = [
            result[f"aki_creatinine_{horizon}h_flag"],
            result[f"hepatic_lab_worsening_{horizon}h_flag"],
        ]
        organ_frame = pd.concat(organ_components, axis=1)
        organ_observed = organ_frame.notna().any(axis=1)
        result[f"organ_lab_worsening_{horizon}h_observed"] = organ_observed.astype(int)
        result[f"organ_lab_worsening_{horizon}h_flag"] = _binary_when_observed(
            organ_frame.fillna(0).max(axis=1).gt(0), organ_observed
        )

        components = pd.concat(
            [
                result[f"lactate_rise_{horizon}h_flag"],
                result[f"vis_rise_{horizon}h_flag"],
                result[f"urine_output_decline_proxy_{horizon}h_flag"],
                result[f"organ_lab_worsening_{horizon}h_flag"],
                result[f"mcs_{horizon}h_flag"].astype(float),
            ],
            axis=1,
        )
        composite_observed = components.notna().any(axis=1)
        result[f"early_decompensation_{horizon}h_observed"] = composite_observed.astype(int)
        result[f"early_decompensation_{horizon}h_component_count"] = components.fillna(0).sum(axis=1)
        result[f"early_decompensation_{horizon}h_flag"] = _binary_when_observed(
            components.fillna(0).max(axis=1).gt(0), composite_observed
        )
        result[f"outcome_window_start_minutes_{horizon}h"] = LANDMARK_MINUTES
        result[f"outcome_window_end_minutes_{horizon}h"] = upper

    result["endpoint_definition_version"] = "spo2_protocol_v1.0"
    return result


def build_endpoint_completeness_audit(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Report coverage and event floors for every prespecified endpoint."""
    tiers = {
        **{name: "primary" for name in PRIMARY_ENDPOINTS},
        **{name: "secondary_or_exploratory" for name in SECONDARY_ENDPOINTS},
    }
    rows: list[dict[str, Any]] = []
    datasets = sorted(analysis_df.get("dataset", pd.Series(["all"])).dropna().astype(str).unique())
    for dataset in datasets:
        frame = analysis_df.loc[analysis_df["dataset"].astype(str).eq(dataset)] if "dataset" in analysis_df else analysis_df
        for endpoint, tier in tiers.items():
            if endpoint not in frame:
                rows.append({"dataset": dataset, "endpoint": endpoint, "tier": tier, "status": "unavailable", "reason": "column_missing"})
                continue
            y = pd.to_numeric(frame[endpoint], errors="coerce")
            n = int(y.notna().sum())
            events = int(y.fillna(0).sum())
            status = "adequate" if n >= 200 and events >= 20 and (n - events) >= 20 else "fragile_or_underpowered"
            rows.append(
                {
                    "dataset": dataset,
                    "endpoint": endpoint,
                    "tier": tier,
                    "status": status,
                    "n_total": int(len(frame)),
                    "n_observed": n,
                    "coverage_fraction": float(n / len(frame)) if len(frame) else math.nan,
                    "events": events,
                    "non_events": int(n - events),
                    "event_rate": float(y.mean()) if n else math.nan,
                }
            )
    return pd.DataFrame(rows)


def _group_series(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    dataset = frame.get("dataset", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    if "person_id" in frame and frame["person_id"].notna().any():
        person = frame["person_id"].fillna(frame["stay_id"]).astype(str)
        return dataset + ":person:" + person, "dataset_qualified_person_id"
    if "stay_id" not in frame:
        raise ValueError("Patient-grouped evaluation requires person_id or stay_id")
    return dataset + ":stay:" + frame["stay_id"].astype(str), "dataset_qualified_stay_id_fallback"


def _build_model_pipeline(frame: pd.DataFrame, columns: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = [
        column
        for column in columns
        if column in {"dataset", "race", "ethnicity", "race_ethnicity"}
        or str(frame[column].dtype) in {"object", "category", "string"}
    ]
    numeric = [column for column in columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")),
        ]
    )


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability < edges[index + 1] if index < bins - 1 else probability <= edges[index + 1]
        )
        if mask.any():
            total += float(mask.mean() * abs(y[mask].mean() - probability[mask].mean()))
    return total


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    intercept, slope = _calibration(y, probability)
    return {
        "auroc_oof": float(roc_auc_score(y, probability)),
        "auprc_oof": float(average_precision_score(y, probability)),
        "brier_oof": float(brier_score_loss(y, probability)),
        "ece_oof": _ece(y, probability),
        "calibration_intercept_oof": intercept,
        "calibration_slope_oof": slope,
    }


def _bootstrap_metric_ci(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    reference_probability: np.ndarray | None = None,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    aurocs: list[float] = []
    auprcs: list[float] = []
    delta_aurocs: list[float] = []
    delta_auprcs: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_group[group] for group in sampled])
        y_boot = y[index]
        if np.unique(y_boot).size < 2:
            continue
        p_boot = probability[index]
        auroc = float(roc_auc_score(y_boot, p_boot))
        auprc = float(average_precision_score(y_boot, p_boot))
        aurocs.append(auroc)
        auprcs.append(auprc)
        if reference_probability is not None:
            ref = reference_probability[index]
            delta_aurocs.append(auroc - float(roc_auc_score(y_boot, ref)))
            delta_auprcs.append(auprc - float(average_precision_score(y_boot, ref)))

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return math.nan, math.nan
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    auroc_low, auroc_high = interval(aurocs)
    auprc_low, auprc_high = interval(auprcs)
    result = {
        "auroc_ci95_low": auroc_low,
        "auroc_ci95_high": auroc_high,
        "auprc_ci95_low": auprc_low,
        "auprc_ci95_high": auprc_high,
        "bootstrap_repetitions_valid": len(aurocs),
    }
    if reference_probability is not None:
        d_auc_low, d_auc_high = interval(delta_aurocs)
        d_pr_low, d_pr_high = interval(delta_auprcs)
        result.update(
            {
                "delta_auroc_vs_absolute_ci95_low": d_auc_low,
                "delta_auroc_vs_absolute_ci95_high": d_auc_high,
                "delta_auprc_vs_absolute_ci95_low": d_pr_low,
                "delta_auprc_vs_absolute_ci95_high": d_pr_high,
            }
        )
    return result


def _prespecified_outcomes(frame: pd.DataFrame) -> list[str]:
    ordered = [*PRIMARY_ENDPOINTS, *SECONDARY_ENDPOINTS]
    available = [name for name in ordered if name in frame]
    if available:
        return available
    # Compatibility for archived artifacts and parity fixtures only.  These
    # endpoints are never selected when protocol-v1 endpoints are present.
    return [
        name
        for name in ("target", "mcs_or_death_168h_flag", "death_168h_flag")
        if name in frame
    ]


def fit_grouped_incremental_models(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_events: int = 20,
    n_splits: int = 5,
    bootstrap_repetitions: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """Patient-grouped, fold-local models for absolute and dynamic SpO2 value."""
    from sklearn.model_selection import StratifiedGroupKFold

    measured = analysis_df.copy()
    if "spo2_plausible_count" in measured:
        measured = measured.loc[pd.to_numeric(measured["spo2_plausible_count"], errors="coerce").fillna(0).gt(0)].copy()
    outcomes = _prespecified_outcomes(measured)
    controls = [column for column in CLINICAL_CONTROL_CANDIDATES if column in measured]
    absolute = [column for column in ABSOLUTE_SPO2_FEATURES if column in measured]
    instability = [column for column in INSTABILITY_FEATURES if column in measured]
    specifications = {
        "clinical_only": controls,
        "absolute_spo2": _dedupe([*controls, *absolute]),
        "clinical_plus_spo2_instability": _dedupe([*controls, *absolute, *instability]),
    }
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        valid = pd.to_numeric(measured[outcome], errors="coerce").isin([0, 1])
        frame = measured.loc[valid].copy()
        y = pd.to_numeric(frame[outcome], errors="coerce").astype(int).reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        groups, grouping = _group_series(frame)
        event_groups = groups.loc[y.eq(1)].nunique()
        non_event_groups = groups.loc[y.eq(0)].nunique()
        splits = min(n_splits, int(groups.nunique()), int(event_groups), int(non_event_groups))
        if len(y) < min_rows or int(y.sum()) < min_events or int((1 - y).sum()) < min_events or splits < 2:
            rows.append(
                {
                    "outcome": outcome,
                    "status": "skipped",
                    "n": int(len(y)),
                    "events": int(y.sum()),
                    "n_groups": int(groups.nunique()),
                    "reason": "insufficient rows/events/non-events/patient groups",
                }
            )
            continue
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)
        predictions: dict[str, np.ndarray] = {
            name: np.full(len(frame), np.nan, dtype=float) for name in specifications
        }
        feature_counts: dict[str, int] = {}
        failure: str | None = None
        for train_index, test_index in splitter.split(frame, y, groups):
            if y.iloc[train_index].nunique() < 2 or y.iloc[test_index].nunique() < 2:
                failure = "single-class train or validation fold"
                break
            for model_name, columns in specifications.items():
                if not columns:
                    failure = f"no features for {model_name}"
                    break
                pipeline = _build_model_pipeline(frame, columns)
                try:
                    pipeline.fit(frame.iloc[train_index][columns], y.iloc[train_index])
                    predictions[model_name][test_index] = pipeline.predict_proba(
                        frame.iloc[test_index][columns]
                    )[:, 1]
                    feature_counts[model_name] = int(
                        len(pipeline.named_steps["preprocess"].get_feature_names_out())
                    )
                except Exception as exc:
                    failure = f"{model_name}: {exc}"
                    break
            if failure:
                break
        if failure or any(not np.isfinite(values).all() for values in predictions.values()):
            rows.append({"outcome": outcome, "status": "failed", "n": int(len(y)), "events": int(y.sum()), "reason": failure or "incomplete OOF predictions"})
            continue
        y_array = y.to_numpy()
        group_array = groups.to_numpy()
        absolute_probability = predictions["absolute_spo2"]
        absolute_metrics = _metrics(y_array, absolute_probability)
        for model_name, probability in predictions.items():
            metrics = _metrics(y_array, probability)
            reference = absolute_probability if model_name == "clinical_plus_spo2_instability" else None
            ci = _bootstrap_metric_ci(
                y_array,
                probability,
                group_array,
                repetitions=bootstrap_repetitions,
                seed=random_state + len(rows),
                reference_probability=reference,
            )
            row: dict[str, Any] = {
                "outcome": outcome,
                "model": model_name,
                "status": "fit",
                "n": int(len(y)),
                "events": int(y.sum()),
                "non_events": int((1 - y).sum()),
                "n_groups": int(groups.nunique()),
                "n_stays": int(frame["stay_id"].nunique()) if "stay_id" in frame else int(len(frame)),
                "n_patients": int(frame["person_id"].nunique()) if "person_id" in frame else math.nan,
                "grouping": grouping,
                "cv_splitter": "StratifiedGroupKFold",
                "cv_folds": splits,
                "preprocessing_scope": "fit_within_each_training_fold",
                "feature_count": feature_counts.get(model_name, 0),
                "metric_scope": "patient_grouped_out_of_fold_internal_validation",
                **metrics,
                **ci,
            }
            if model_name == "clinical_plus_spo2_instability":
                row["delta_auroc_vs_absolute"] = metrics["auroc_oof"] - absolute_metrics["auroc_oof"]
                row["delta_auprc_vs_absolute"] = metrics["auprc_oof"] - absolute_metrics["auprc_oof"]
            rows.append(row)
    if not rows:
        return pd.DataFrame([{"status": "skipped", "reason": "no prespecified outcomes available"}])
    return pd.DataFrame(rows)


def fit_external_transportability(
    analysis_df: pd.DataFrame,
    *,
    min_rows: int = 100,
    min_events: int = 20,
) -> pd.DataFrame:
    """Train on MIMIC and evaluate untouched eICU with train-only preprocessing."""
    if "dataset" not in analysis_df or not {"mimic", "eicu"}.issubset(set(analysis_df["dataset"].astype(str))):
        return pd.DataFrame([{"status": "unavailable", "reason": "both_mimic_and_eicu_required"}])
    measured = analysis_df.copy()
    if "spo2_plausible_count" in measured:
        measured = measured.loc[pd.to_numeric(measured["spo2_plausible_count"], errors="coerce").fillna(0).gt(0)].copy()
    controls = [column for column in CLINICAL_CONTROL_CANDIDATES if column in measured and column != "dataset"]
    columns = _dedupe(
        [
            *controls,
            *[column for column in ABSOLUTE_SPO2_FEATURES if column in measured],
            *[column for column in INSTABILITY_FEATURES if column in measured],
        ]
    )
    rows: list[dict[str, Any]] = []
    for outcome in _prespecified_outcomes(measured):
        train = measured.loc[measured["dataset"].astype(str).eq("mimic")].copy()
        test = measured.loc[measured["dataset"].astype(str).eq("eicu")].copy()
        train = train.loc[pd.to_numeric(train[outcome], errors="coerce").isin([0, 1])]
        test = test.loc[pd.to_numeric(test[outcome], errors="coerce").isin([0, 1])]
        y_train = pd.to_numeric(train[outcome], errors="coerce").astype(int)
        y_test = pd.to_numeric(test[outcome], errors="coerce").astype(int)
        if len(train) < min_rows or len(test) < min_rows or y_train.sum() < min_events or y_test.sum() < min_events or y_train.nunique() < 2 or y_test.nunique() < 2:
            rows.append({"outcome": outcome, "status": "unavailable", "n_train": int(len(train)), "n_test": int(len(test)), "events_train": int(y_train.sum()), "events_test": int(y_test.sum()), "reason": "below external validation floors"})
            continue
        pipeline = _build_model_pipeline(train, columns)
        pipeline.fit(train[columns], y_train)
        probability = pipeline.predict_proba(test[columns])[:, 1]
        metrics = _metrics(y_test.to_numpy(), probability)
        groups, grouping = _group_series(test)
        ci = _bootstrap_metric_ci(y_test.to_numpy(), probability, groups.to_numpy(), repetitions=500, seed=91)
        rows.append(
            {
                "outcome": outcome,
                "status": "fit",
                "train_dataset": "mimic",
                "test_dataset": "eicu",
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "events_train": int(y_train.sum()),
                "events_test": int(y_test.sum()),
                "grouping": grouping,
                "preprocessing_scope": "fit_on_mimic_only",
                "metric_scope": "external_dataset_transportability",
                **metrics,
                **ci,
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "unavailable", "reason": "no shared prespecified endpoints"}])


def fit_grouped_missingness_control(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Patient-grouped sampling/missingness-only negative-control models."""
    features = [name for name in MISSINGNESS_CONTROL_FEATURES if name in analysis_df]
    if not features:
        return pd.DataFrame([{"status": "not_run", "reason": "missingness features unavailable"}])
    keep = _dedupe(
        column
        for column in (
            "dataset",
            "stay_id",
            "person_id",
            "spo2_plausible_count",
            *PRIMARY_ENDPOINTS,
            *SECONDARY_ENDPOINTS,
            "target",
            "mcs_or_death_168h_flag",
            *features,
        )
        if column in analysis_df
    )
    proxy = analysis_df[keep].copy()
    # Map sampling variables onto the fixed absolute-model slots so the same
    # fold-local engine is used without any physiological SpO2 values.
    for target, source in zip(ABSOLUTE_SPO2_FEATURES, features):
        proxy[target] = proxy[source]
    result = fit_grouped_incremental_models(proxy, min_rows=50, min_events=10, bootstrap_repetitions=100)
    result["analysis"] = "missingness_only_negative_control"
    result["negative_control_features"] = ",".join(features)
    return result


def _first_instability(events: pd.DataFrame) -> pd.DataFrame:
    spo2 = events.loc[
        events["concept"].eq("spo2")
        & events["offset_minutes"].ge(0)
        & events["offset_minutes"].lt(LANDMARK_MINUTES)
        & events["value_numeric"].between(50, 100)
    ].sort_values(["dataset", "stay_id", "offset_minutes"])
    if spo2.empty:
        return pd.DataFrame(columns=["dataset", "stay_id", "instability_onset_minutes"])
    spo2["absolute_change"] = spo2.groupby(["dataset", "stay_id"])["value_numeric"].diff().abs()
    unstable = spo2.loc[spo2["value_numeric"].lt(90) | spo2["absolute_change"].ge(4)]
    return (
        unstable.groupby(["dataset", "stay_id"], as_index=False)["offset_minutes"]
        .min()
        .rename(columns={"offset_minutes": "instability_onset_minutes"})
    )


def build_temporal_precedence(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 1000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate time from first pre-landmark instability to outcome onset.

    This confirms temporal ordering under the landmark design; it does not
    identify a causal effect and must not be described as such.
    """
    events = events_df.copy()
    if events.empty:
        empty = pd.DataFrame([{"status": "unavailable", "reason": "no events"}])
        return empty, empty.copy()
    if "dataset" not in events:
        events["dataset"] = "unknown"
    events["concept"] = events.get("concept", "").fillna("").astype(str).str.lower()
    events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
    events["value_numeric"] = pd.to_numeric(events.get("value_numeric"), errors="coerce")
    onset = _first_instability(events)
    baseline_lactate = _last_measurement(
        events.loc[events["offset_minutes"].lt(LANDMARK_MINUTES)], "lactate"
    ).rename(columns={"value_numeric": "baseline_lactate"})
    baseline_creatinine = _last_measurement(
        events.loc[events["offset_minutes"].lt(LANDMARK_MINUTES)], "creatinine"
    ).rename(columns={"value_numeric": "baseline_creatinine"})
    baseline_bilirubin = _last_measurement(
        events.loc[events["offset_minutes"].lt(LANDMARK_MINUTES)], "bilirubin_total"
    ).rename(columns={"value_numeric": "baseline_bilirubin"})
    records: list[pd.DataFrame] = []
    post = events.loc[
        events["offset_minutes"].gt(LANDMARK_MINUTES)
        & events["offset_minutes"].le(LANDMARK_MINUTES + 24 * 60)
    ].copy()

    def append_first(frame: pd.DataFrame, outcome: str) -> None:
        if frame.empty:
            return
        first = frame.groupby(["dataset", "stay_id"], as_index=False)["offset_minutes"].min()
        merged = onset.merge(first, on=["dataset", "stay_id"], how="inner").rename(columns={"offset_minutes": "outcome_onset_minutes"})
        if merged.empty:
            return
        merged["outcome"] = outcome
        merged["lead_time_minutes"] = merged["outcome_onset_minutes"] - merged["instability_onset_minutes"]
        records.append(merged)

    lact = post.loc[post["concept"].eq("lactate")].merge(
        baseline_lactate[["dataset", "stay_id", "baseline_lactate"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(lact.loc[lact["value_numeric"].ge(lact["baseline_lactate"] + LACTATE_RISE_ABSOLUTE)], "lactate_rise")
    append_first(post.loc[post["concept"].eq("pressor")], "pressor_initiation")
    append_first(post.loc[post["concept"].eq("mcs")], "mcs_initiation")
    vis_base = _max_measurement(events.loc[events["offset_minutes"].lt(LANDMARK_MINUTES)], "vis").rename(columns={"value_numeric": "baseline_vis"})
    vis = post.loc[post["concept"].eq("vis")].merge(vis_base, on=["dataset", "stay_id"], how="left")
    vis["baseline_vis"] = vis["baseline_vis"].fillna(0)
    append_first(vis.loc[vis["value_numeric"].gt(vis["baseline_vis"])], "vis_rise")
    creat = post.loc[post["concept"].eq("creatinine")].merge(
        baseline_creatinine[["dataset", "stay_id", "baseline_creatinine"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        creat.loc[
            creat["value_numeric"].ge(creat["baseline_creatinine"] + CREATININE_AKI_ABSOLUTE)
            | creat["value_numeric"].ge(creat["baseline_creatinine"] * CREATININE_AKI_RATIO)
        ],
        "aki_creatinine",
    )
    bili = post.loc[post["concept"].eq("bilirubin_total")].merge(
        baseline_bilirubin[["dataset", "stay_id", "baseline_bilirubin"]],
        on=["dataset", "stay_id"],
        how="inner",
    )
    append_first(
        bili.loc[
            bili["value_numeric"].ge(bili["baseline_bilirubin"] + BILIRUBIN_WORSENING_ABSOLUTE)
            | bili["value_numeric"].ge(bili["baseline_bilirubin"] * BILIRUBIN_WORSENING_RATIO)
        ],
        "hepatic_lab_worsening",
    )
    if not records:
        empty = pd.DataFrame([{"status": "unavailable", "reason": "no paired instability/outcome onsets"}])
        return empty, empty.copy()
    record_frame = pd.concat(records, ignore_index=True)
    summaries: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (dataset, outcome), group in record_frame.groupby(["dataset", "outcome"]):
        values = group["lead_time_minutes"].to_numpy(dtype=float)
        boot = [float(np.median(rng.choice(values, size=len(values), replace=True))) for _ in range(bootstrap_repetitions)]
        if boot:
            median_low = float(np.quantile(boot, 0.025))
            median_high = float(np.quantile(boot, 0.975))
        else:  # bootstrap_repetitions == 0: caller supplies its own CI
            median_low = median_high = math.nan
        summaries.append(
            {
                "dataset": dataset,
                "outcome": outcome,
                "status": "descriptive_temporal_ordering",
                "n": int(len(values)),
                "median_lead_time_minutes": float(np.median(values)),
                "q1_lead_time_minutes": float(np.quantile(values, 0.25)),
                "q3_lead_time_minutes": float(np.quantile(values, 0.75)),
                "median_ci95_low": median_low,
                "median_ci95_high": median_high,
                "positive_lead_time_fraction": float(np.mean(values > 0)),
                "claim_scope": "landmark_ordering_not_causal_precedence",
            }
        )
    return record_frame, pd.DataFrame(summaries)


__all__ = [
    "PRIMARY_ENDPOINTS",
    "SECONDARY_ENDPOINTS",
    "POST_LANDMARK_HORIZONS_HOURS",
    "build_endpoint_completeness_audit",
    "build_temporal_precedence",
    "compute_early_decompensation_outcomes",
    "fit_external_transportability",
    "fit_grouped_incremental_models",
    "fit_grouped_missingness_control",
]
