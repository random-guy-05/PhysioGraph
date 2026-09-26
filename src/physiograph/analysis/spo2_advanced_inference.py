"""High-rigor robustness analyses for episode-anchored SpO2 associations.

This module strengthens the post-result multiorgan amendment without choosing
thresholds, covariates, endpoints, or estimators from observed p-values.  It
constructs strictly pre-anchor physiology, audits overlap-weight balance,
estimates bounded cross-validated targeted maximum-likelihood (TMLE) risks plus
one-step AIPW diagnostics while accounting for outcome observation, estimates
cross-fitted continuous AIPW effects, quantifies multicenter heterogeneity, and
reports E-values for residual unmeasured confounding.

The estimands remain observational associations.  Double robustness does not
remove unmeasured confounding or make the SpO2 episode a manipulable treatment.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .spo2_lactate_mechanistic import _bh_adjust, _prepare_spo2_transitions


ADVANCED_ANALYSIS_STATUS = (
    "post_result_advanced_episode_robustness_amendment_exploratory"
)
PRIMARY_DEFINITION = "absolute_jump_ge4"
PRIMARY_RESOLUTION = "15_minute_median_bins"
PROPENSITY_CLIP = (0.02, 0.98)
# Only very small observation probabilities destabilize inverse-observation
# weights.  A probability of one is valid (and exact for fully observed
# endpoints), so it must not be spuriously truncated or counted as a support
# violation.
OBSERVATION_CLIP = (0.02, 1.0)
MAX_PROPENSITY_CLIP_FRACTION = 0.10
MAX_OBSERVATION_LOW_CLIP_FRACTION = 0.10
MIN_EFFECTIVE_SAMPLE_SIZE_PER_ARM = 50.0
MIN_ROWS = 80
MIN_EVENTS = 10
N_FOLDS = 5
RANDOM_SEED = 20260902


# Every value is required to occur strictly before the episode/control anchor.
# Bounds are deliberately broad and mirror the production physiological guards.
PREANCHOR_CONCEPT_SPECS: dict[str, dict[str, Any]] = {
    "spo2": {"low": 50.0, "high": 100.0, "stats": ("last", "mean", "min", "max", "std", "count")},
    "hr": {"low": 20.0, "high": 250.0, "stats": ("last", "mean", "min", "max", "count")},
    "map": {"low": 20.0, "high": 200.0, "stats": ("last", "mean", "min", "max", "count")},
    "sbp": {"low": 30.0, "high": 300.0, "stats": ("last", "mean", "min", "max", "count")},
    "resp_rate": {"low": 2.0, "high": 80.0, "stats": ("last", "mean", "min", "max", "count")},
    "fio2": {"low": 20.0, "high": 100.0, "stats": ("last", "mean", "max", "count")},
    "lactate": {"low": 0.0, "high": 30.0, "stats": ("last", "mean", "max", "count")},
    "ph": {"low": 6.5, "high": 8.0, "stats": ("last", "mean", "min", "count")},
    "creatinine": {"low": 0.05, "high": 30.0, "stats": ("last", "mean", "max", "count")},
    "temp": {"low": 25.0, "high": 45.0, "stats": ("last", "mean", "min", "max", "count")},
}

PREANCHOR_BINARY_CONCEPTS: dict[str, tuple[str, ...]] = {
    "preanchor_pressor_support": ("pressor", "pressor_active", "pressor_initiation"),
    "preanchor_mcs_support": ("mcs", "mcs_active"),
    "preanchor_rrt_support": ("rrt", "rrt_active"),
    "preanchor_invasive_ventilation": ("mechanical_ventilation", "airway_device"),
    "preanchor_noninvasive_ventilation": ("noninvasive_ventilation",),
    "preanchor_oxygen_device": ("oxygen_device", "oxygen_flow"),
}

BASE_NUMERIC_COVARIATES: tuple[str, ...] = (
    "age",
    "anchor_offset_minutes",
    "previous_spo2",
    "preanchor_sampling_density_per_hr",
    "pre_spo2_last",
    "pre_spo2_mean",
    "pre_spo2_min",
    "pre_spo2_std",
    "pre_hr_last",
    "pre_hr_max",
    "pre_map_last",
    "pre_map_min",
    "pre_sbp_last",
    "pre_sbp_min",
    "pre_resp_rate_last",
    "pre_resp_rate_max",
    "pre_fio2_last",
    "pre_fio2_max",
    "pre_lactate_last",
    "pre_lactate_max",
    "pre_ph_last",
    "pre_ph_min",
    "pre_creatinine_last",
    "pre_temp_last",
    "preanchor_numeric_event_count",
)

# The current saturation is recorded at the same instant as the instability
# transition and partly defines the exposure.  It is therefore excluded from
# propensity/balance models (where it would induce deterministic separation),
# but may enter post-anchor observation and outcome regressions to test whether
# dynamics add information beyond the concurrent absolute SpO2 level.
CONCURRENT_NUMERIC_COVARIATES: tuple[str, ...] = ("anchor_spo2",)

BASE_BINARY_COVARIATES: tuple[str, ...] = (
    "is_male",
    "shock_icd_flag",
    "acute_mi_flag",
    "cardiomyopathy_flag",
    "baseline_vasoactive_flag",
    "baseline_mcs_flag",
    *PREANCHOR_BINARY_CONCEPTS.keys(),
)

BASE_CATEGORICAL_COVARIATES: tuple[str, ...] = (
    "icu_type",
    "unit_admit_source",
    "hospital_id",
)


def _available(columns: Iterable[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in frame]


def _is_true(value: Any) -> bool:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _unique_primary_anchors(records: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "stay_id",
        "episode_exposed",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
    }
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Focused records lack anchor fields: {sorted(missing)}")
    columns = [
        "dataset",
        "stay_id",
        "person_id",
        "episode_exposed",
        "anchor_offset_minutes",
        "anchor_spo2",
        "preanchor_sampling_density_per_hr",
        "age",
        "shock_icd_flag",
        "baseline_vasoactive_flag",
    ]
    anchors = records.reindex(columns=columns).drop_duplicates(
        ["dataset", "stay_id"]
    )
    multiplicity = records.groupby(["dataset", "stay_id"], sort=False)[
        "anchor_offset_minutes"
    ].nunique(dropna=False)
    if multiplicity.gt(1).any():
        raise ValueError("Focused records contain multiple primary anchors per stay")
    anchors["dataset"] = anchors["dataset"].astype(str)
    anchors["anchor_offset_minutes"] = pd.to_numeric(
        anchors["anchor_offset_minutes"], errors="coerce"
    )
    return anchors.reset_index(drop=True)


def build_episode_preanchor_covariates(
    events: pd.DataFrame,
    focused_records: pd.DataFrame,
    cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Build one strictly pre-anchor confounder row per primary anchor."""
    anchors = _unique_primary_anchors(focused_records)
    keys = ["dataset", "stay_id"]
    result = anchors.copy()

    transitions, _ = _prepare_spo2_transitions(
        events, signal_resolution=PRIMARY_RESOLUTION
    )
    transition_columns = [
        "dataset",
        "stay_id",
        "anchor_offset_minutes",
        "previous_spo2",
        "spo2_delta",
        "transition_gap_minutes",
    ]
    transition_context = transitions.reindex(columns=transition_columns)
    result = result.merge(
        transition_context,
        on=[*keys, "anchor_offset_minutes"],
        how="left",
        validate="one_to_one",
    )

    local = events.copy()
    if "dataset" not in local:
        local["dataset"] = "unknown"
    local["dataset"] = local["dataset"].astype(str)
    if "concept" not in local:
        local["concept"] = ""
    local["concept"] = local["concept"].fillna("").astype(str).str.lower()
    local["offset_minutes"] = pd.to_numeric(
        local.get("offset_minutes"), errors="coerce"
    )
    local["value_numeric"] = pd.to_numeric(
        local.get("value_numeric"), errors="coerce"
    )
    relevant_concepts = {
        *PREANCHOR_CONCEPT_SPECS,
        *(
            concept
            for concepts in PREANCHOR_BINARY_CONCEPTS.values()
            for concept in concepts
        ),
    }
    local = local.loc[local["concept"].isin(relevant_concepts)].merge(
        anchors[[*keys, "anchor_offset_minutes"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    local = local.loc[
        local["offset_minutes"].ge(0)
        & local["offset_minutes"].lt(local["anchor_offset_minutes"])
    ].copy()
    result["preanchor_numeric_event_count"] = 0

    for concept, spec in PREANCHOR_CONCEPT_SPECS.items():
        concept_rows = local.loc[
            local["concept"].eq(concept)
            & local["value_numeric"].between(
                float(spec["low"]), float(spec["high"]), inclusive="both"
            )
        ].copy()
        if concept_rows.empty:
            for statistic in spec["stats"]:
                result[f"pre_{concept}_{statistic}"] = np.nan
            continue
        concept_rows = concept_rows.sort_values([*keys, "offset_minutes"])
        grouped = concept_rows.groupby(keys, sort=False)["value_numeric"]
        aggregates: dict[str, pd.Series] = {
            "last": concept_rows.groupby(keys, sort=False).tail(1).set_index(keys)[
                "value_numeric"
            ],
            "mean": grouped.mean(),
            "min": grouped.min(),
            "max": grouped.max(),
            "std": grouped.std(ddof=1),
            "count": grouped.count().astype(float),
        }
        for statistic in spec["stats"]:
            values = aggregates[str(statistic)].rename(
                f"pre_{concept}_{statistic}"
            )
            result = result.merge(
                values.reset_index(), on=keys, how="left", validate="one_to_one"
            )
        counts = aggregates["count"].rename("_concept_count")
        result = result.merge(
            counts.reset_index(), on=keys, how="left", validate="one_to_one"
        )
        result["preanchor_numeric_event_count"] += result.pop(
            "_concept_count"
        ).fillna(0).astype(int)

    for output_column, concepts in PREANCHOR_BINARY_CONCEPTS.items():
        positive = (
            local.loc[local["concept"].isin(concepts), keys]
            .drop_duplicates()
            .assign(**{output_column: 1})
        )
        result = result.merge(positive, on=keys, how="left", validate="one_to_one")
        result[output_column] = result[output_column].fillna(0).astype(int)

    cohort_columns = [
        *keys,
        "age",
        "is_male",
        "shock_icd_flag",
        "acute_mi_flag",
        "cardiomyopathy_flag",
        "baseline_vasoactive_flag",
        "baseline_mcs_flag",
        "hospital_id",
        "icu_type",
        "unit_admit_source",
    ]
    context = cohort.reindex(columns=cohort_columns).drop_duplicates(keys)
    duplicate_context = [
        column for column in context if column in result and column not in keys
    ]
    context = context.drop(columns=duplicate_context)
    result = result.merge(context, on=keys, how="left", validate="one_to_one")
    for column in ("hospital_id", "icu_type", "unit_admit_source"):
        if column in result:
            result[column] = result[column].fillna("unknown").astype(str)
    result["preanchor_boundary_rule"] = "offset_minutes>=0_and_strictly_less_than_anchor"
    result["analysis_status"] = ADVANCED_ANALYSIS_STATUS
    return result.sort_values(keys).reset_index(drop=True)


def _raw_design_frame(
    frame: pd.DataFrame,
    *,
    include_exposure: bool = False,
    include_baseline_value: bool = False,
    include_site: bool = True,
    include_concurrent_spo2: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric = _available(BASE_NUMERIC_COVARIATES, frame)
    if include_concurrent_spo2:
        numeric.extend(_available(CONCURRENT_NUMERIC_COVARIATES, frame))
    binary = _available(BASE_BINARY_COVARIATES, frame)
    categorical = _available(BASE_CATEGORICAL_COVARIATES, frame)
    if not include_site:
        categorical = [column for column in categorical if column != "hospital_id"]
    if include_baseline_value and "baseline_value" in frame:
        baseline = pd.to_numeric(frame["baseline_value"], errors="coerce")
        if baseline.notna().sum() >= max(20, int(0.2 * len(frame))):
            numeric.append("baseline_value")
    if include_exposure:
        binary.append("episode_exposed")
    columns = list(dict.fromkeys([*numeric, *binary, *categorical]))
    return frame.reindex(columns=columns).copy(), numeric + binary, categorical


def _make_binary_pipeline(
    frame: pd.DataFrame,
    *,
    include_exposure: bool = False,
    include_baseline_value: bool = False,
    include_site: bool = True,
    include_concurrent_spo2: bool = False,
) -> tuple[Any, list[str]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    _, numeric, categorical = _raw_design_frame(
        frame,
        include_exposure=include_exposure,
        include_baseline_value=include_baseline_value,
        include_site=include_site,
        include_concurrent_spo2=include_concurrent_spo2,
    )
    # Entirely absent columns cannot be median-imputed and add no information.
    numeric = [
        column
        for column in numeric
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    categorical = [
        column for column in categorical if frame[column].notna().any()
    ]
    transformers: list[tuple[str, Any, list[str]]] = []
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
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=10,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable pre-anchor covariates")
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    return pipeline, [*numeric, *categorical]


def _make_continuous_pipeline(
    frame: pd.DataFrame,
    *,
    include_exposure: bool = True,
    include_baseline_value: bool = True,
    include_site: bool = True,
    include_concurrent_spo2: bool = True,
) -> tuple[Any, list[str]]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    _, numeric, categorical = _raw_design_frame(
        frame,
        include_exposure=include_exposure,
        include_baseline_value=include_baseline_value,
        include_site=include_site,
        include_concurrent_spo2=include_concurrent_spo2,
    )
    numeric = [
        column
        for column in numeric
        if pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    categorical = [
        column for column in categorical if frame[column].notna().any()
    ]
    transformers: list[tuple[str, Any, list[str]]] = []
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
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=10,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No usable pre-anchor covariates")
    return (
        Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(transformers, remainder="drop"),
                ),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        [*numeric, *categorical],
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return math.nan
    return float(np.average(values[finite], weights=weights[finite]))


def _standardized_difference(
    values: np.ndarray,
    exposure: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    values = np.asarray(values, dtype=float)
    exposure = np.asarray(exposure, dtype=int)
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    treated = finite & (exposure == 1)
    control = finite & (exposure == 0)
    if treated.sum() < 2 or control.sum() < 2:
        return math.nan
    mean_t = _weighted_mean(values[treated], weights[treated])
    mean_c = _weighted_mean(values[control], weights[control])

    def variance(mask: np.ndarray) -> float:
        local_w = weights[mask]
        local_x = values[mask]
        center = _weighted_mean(local_x, local_w)
        return float(np.average(np.square(local_x - center), weights=local_w))

    pooled = math.sqrt(max((variance(treated) + variance(control)) / 2.0, 0.0))
    if not np.isfinite(pooled) or pooled <= 0:
        return 0.0 if np.isclose(mean_t, mean_c) else math.nan
    return float((mean_t - mean_c) / pooled)


def _balance_design(frame: pd.DataFrame) -> pd.DataFrame:
    raw, numeric, categorical = _raw_design_frame(frame, include_site=True)
    parts: list[pd.DataFrame] = []
    for column in numeric:
        values = pd.to_numeric(raw[column], errors="coerce")
        if not values.notna().any():
            continue
        missing = values.isna()
        values = values.fillna(float(values.median()))
        scale = float(values.std(ddof=0))
        standardized = (
            (values - float(values.mean())) / scale if scale > 0 else values * 0.0
        )
        parts.append(pd.DataFrame({column: standardized}, index=frame.index))
        if missing.any():
            parts.append(
                pd.DataFrame(
                    {f"{column}__missing": missing.astype(float)}, index=frame.index
                )
            )
    if categorical:
        cats = raw[categorical].fillna("unknown").astype(str)
        parts.append(
            pd.get_dummies(
                cats,
                prefix=categorical,
                prefix_sep="=",
                drop_first=True,
                dtype=float,
            )
        )
    if not parts:
        raise ValueError("No covariates available for balance analysis")
    design = pd.concat(parts, axis=1)
    design = design.loc[:, design.nunique(dropna=False).gt(1)]
    return design.astype(float)


def build_episode_overlap_weights(
    preanchor_covariates: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Fit fixed, outcome-independent overlap weights and expose all balance."""
    from sklearn.linear_model import LogisticRegression

    weight_rows: list[pd.DataFrame] = []
    balance_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset, frame in preanchor_covariates.groupby("dataset", sort=False):
        local = frame.copy().reset_index(drop=True)
        exposure = pd.to_numeric(
            local["episode_exposed"], errors="coerce"
        ).astype(int).to_numpy()
        design = _balance_design(local)
        model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
        model.fit(design, exposure)
        propensity = model.predict_proba(design)[:, 1]
        propensity = np.clip(propensity, 1e-6, 1.0 - 1e-6)
        overlap = np.where(exposure == 1, 1.0 - propensity, propensity)
        keys = local[["dataset", "stay_id", "person_id", "episode_exposed"]].copy()
        keys["episode_propensity"] = propensity
        keys["overlap_weight"] = overlap
        keys["analysis_status"] = ADVANCED_ANALYSIS_STATUS
        weight_rows.append(keys)

        for column in design:
            values = design[column].to_numpy(float)
            before = _standardized_difference(values, exposure)
            after = _standardized_difference(values, exposure, overlap)
            balance_rows.append(
                {
                    "dataset": str(dataset),
                    "covariate": str(column),
                    "standardized_mean_difference_unweighted": before,
                    "standardized_mean_difference_overlap_weighted": after,
                    "absolute_smd_unweighted": abs(before) if np.isfinite(before) else math.nan,
                    "absolute_smd_overlap_weighted": abs(after) if np.isfinite(after) else math.nan,
                    "analysis_status": ADVANCED_ANALYSIS_STATUS,
                }
            )
        finite_balance = [
            row
            for row in balance_rows
            if row["dataset"] == str(dataset)
            and np.isfinite(row["absolute_smd_overlap_weighted"])
        ]
        finite_before = [
            float(row["absolute_smd_unweighted"])
            for row in finite_balance
            if np.isfinite(row["absolute_smd_unweighted"])
        ]
        ess = float(np.square(overlap.sum()) / np.square(overlap).sum())
        summary_rows.append(
            {
                "dataset": str(dataset),
                "n_anchors": int(len(local)),
                "n_exposed": int(exposure.sum()),
                "n_unexposed": int(len(exposure) - exposure.sum()),
                "propensity_min": float(propensity.min()),
                "propensity_p01": float(np.quantile(propensity, 0.01)),
                "propensity_p99": float(np.quantile(propensity, 0.99)),
                "propensity_max": float(propensity.max()),
                "outside_0_05_0_95_count": int(
                    ((propensity < 0.05) | (propensity > 0.95)).sum()
                ),
                "overlap_effective_sample_size": ess,
                "covariates_assessed": int(len(finite_balance)),
                "max_absolute_smd_unweighted": max(finite_before, default=math.nan),
                "max_absolute_smd_overlap_weighted": max(
                    (
                        row["absolute_smd_overlap_weighted"]
                        for row in finite_balance
                    ),
                    default=math.nan,
                ),
                "covariates_above_0_10_unweighted": int(
                    sum(value > 0.10 for value in finite_before)
                ),
                "covariates_above_0_10_overlap_weighted": int(
                    sum(
                        row["absolute_smd_overlap_weighted"] > 0.10
                        for row in finite_balance
                    )
                ),
                "balance_target": "average_treatment_effect_in_overlap_population",
                "analysis_status": ADVANCED_ANALYSIS_STATUS,
            }
        )
    return {
        "advanced_episode_anchor_weights": pd.concat(
            weight_rows, ignore_index=True, sort=False
        ),
        "advanced_episode_covariate_balance": pd.DataFrame(balance_rows),
        "advanced_episode_overlap_summary": pd.DataFrame(summary_rows),
    }


def _identity_groups(frame: pd.DataFrame) -> pd.Series:
    person = frame.get("person_id", pd.Series(index=frame.index, dtype=object)).astype(
        "string"
    )
    stay = frame["stay_id"].astype("string")
    identity = person.where(
        person.notna() & person.str.strip().ne(""), "stay:" + stay
    )
    return frame["dataset"].astype(str) + ":" + identity.astype(str)


def _crossfit_splits(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    exposure = pd.to_numeric(frame["episode_exposed"], errors="coerce").astype(int)
    class_count = int(exposure.value_counts().min()) if exposure.nunique() == 2 else 0
    groups = _identity_groups(frame)
    n_splits = min(N_FOLDS, class_count, int(groups.nunique()))
    if n_splits < 3:
        return []
    if groups.nunique() < len(frame):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
        )
        return list(splitter.split(frame, exposure, groups))
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    return list(splitter.split(frame, exposure))


def _empty_aipw_row(frame: pd.DataFrame, status: str, reason: str) -> dict[str, Any]:
    first = frame.iloc[0]
    return {
        "dataset": str(first["dataset"]),
        "endpoint": str(first["endpoint"]),
        "endpoint_family": str(first["endpoint_family"]),
        "lag_window": str(first["lag_window"]),
        "status": status,
        "reason": reason,
        "n_anchor_rows": int(len(frame)),
        "n_risk_set": 0,
        "n_outcome_observed": 0,
        "events_observed": 0,
        "events_exposed": 0,
        "events_unexposed": 0,
        "non_events_exposed": 0,
        "non_events_unexposed": 0,
        "crossfit_folds": 0,
        "nuisance_covariates": 0,
        "aipw_risk_exposed": math.nan,
        "aipw_risk_unexposed": math.nan,
        "aipw_risk_ratio": math.nan,
        "aipw_risk_ratio_ci95_low": math.nan,
        "aipw_risk_ratio_ci95_high": math.nan,
        "aipw_log_rr_se": math.nan,
        "aipw_risk_ratio_p_value": math.nan,
        "aipw_risk_difference": math.nan,
        "aipw_risk_difference_ci95_low": math.nan,
        "aipw_risk_difference_ci95_high": math.nan,
        "aipw_risk_difference_p_value": math.nan,
        "exposure_propensity_min": math.nan,
        "exposure_propensity_max": math.nan,
        "exposure_propensity_clipped_count": 0,
        "exposure_propensity_clipped_fraction": math.nan,
        "observation_probability_min": math.nan,
        "observation_probability_max": math.nan,
        "observation_probability_clipped_count": 0,
        "observation_probability_lower_clipped_count": 0,
        "observation_probability_lower_clipped_fraction": math.nan,
        "aipw_effective_sample_size_exposed": math.nan,
        "aipw_effective_sample_size_unexposed": math.nan,
        "claim_ready_propensity_support": False,
        "claim_ready_observation_support": False,
        "claim_ready_effective_sample_size": False,
        "claim_ready_weight_diagnostics": False,
        "one_step_aipw_status": "not_estimated",
        "tmle_risk_exposed": math.nan,
        "tmle_risk_unexposed": math.nan,
        "tmle_risk_ratio": math.nan,
        "tmle_risk_ratio_ci95_low": math.nan,
        "tmle_risk_ratio_ci95_high": math.nan,
        "tmle_log_rr_se": math.nan,
        "tmle_risk_ratio_p_value": math.nan,
        "tmle_risk_difference": math.nan,
        "tmle_risk_difference_ci95_low": math.nan,
        "tmle_risk_difference_ci95_high": math.nan,
        "tmle_risk_difference_p_value": math.nan,
        "tmle_targeting_epsilon_exposed": math.nan,
        "tmle_targeting_epsilon_unexposed": math.nan,
        "tmle_targeting_boundary_hit": False,
        "tmle_efficient_influence_mean_exposed": math.nan,
        "tmle_efficient_influence_mean_unexposed": math.nan,
        "claim_ready_min_10_events_per_exposure_arm": False,
        "claim_ready_min_10_events_and_non_events_per_arm": False,
        "estimand": "marginal_risk_ratio_and_difference_in_anchor_risk_set",
        "assumptions": (
            "consistency+conditional_exchangeability+positivity+missing_at_random;"
            "not_a_causal_claim"
        ),
        "analysis_status": ADVANCED_ANALYSIS_STATUS,
    }


def _cluster_standard_error(
    influence: np.ndarray,
    groups: pd.Series,
) -> float:
    local = pd.DataFrame(
        {"group": groups.astype(str).to_numpy(), "influence": influence}
    )
    cluster_sums = local.groupby("group", sort=False)["influence"].sum().to_numpy()
    clusters = len(cluster_sums)
    if clusters < 2:
        return math.nan
    variance = (
        clusters
        / (clusters - 1.0)
        * float(np.square(cluster_sums).sum())
        / float(len(influence) ** 2)
    )
    return math.sqrt(max(variance, 0.0))


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]
    if not len(weights) or float(np.square(weights).sum()) <= 0:
        return math.nan
    return float(np.square(weights.sum()) / np.square(weights).sum())


def _weight_support_diagnostics(
    propensity: np.ndarray,
    observation_probability: np.ndarray,
    exposed_weights: np.ndarray,
    unexposed_weights: np.ndarray,
) -> dict[str, Any]:
    """Return fixed, outcome-independent positivity and precision gates."""
    propensity = np.asarray(propensity, dtype=float)
    observation_probability = np.asarray(observation_probability, dtype=float)
    n = max(len(propensity), 1)
    propensity_outside = (
        (propensity < PROPENSITY_CLIP[0])
        | (propensity > PROPENSITY_CLIP[1])
    )
    observation_below = observation_probability < OBSERVATION_CLIP[0]
    propensity_fraction = float(propensity_outside.sum() / n)
    observation_fraction = float(observation_below.sum() / n)
    ess_exposed = _effective_sample_size(exposed_weights)
    ess_unexposed = _effective_sample_size(unexposed_weights)
    propensity_ready = propensity_fraction <= MAX_PROPENSITY_CLIP_FRACTION
    observation_ready = (
        observation_fraction <= MAX_OBSERVATION_LOW_CLIP_FRACTION
    )
    ess_ready = bool(
        np.isfinite(ess_exposed)
        and np.isfinite(ess_unexposed)
        and min(ess_exposed, ess_unexposed)
        >= MIN_EFFECTIVE_SAMPLE_SIZE_PER_ARM
    )
    return {
        "exposure_propensity_clipped_count": int(propensity_outside.sum()),
        "exposure_propensity_clipped_fraction": propensity_fraction,
        "observation_probability_lower_clipped_count": int(
            observation_below.sum()
        ),
        "observation_probability_lower_clipped_fraction": observation_fraction,
        "claim_ready_propensity_support": bool(propensity_ready),
        "claim_ready_observation_support": bool(observation_ready),
        "claim_ready_effective_sample_size": ess_ready,
        "claim_ready_weight_diagnostics": bool(
            propensity_ready and observation_ready and ess_ready
        ),
        "effective_sample_size_exposed": ess_exposed,
        "effective_sample_size_unexposed": ess_unexposed,
    }


def _tmle_target_arm(
    initial_counterfactual: np.ndarray,
    exposure_probability: np.ndarray,
    observation_probability: np.ndarray,
    exposure: np.ndarray,
    observed: np.ndarray,
    outcome: np.ndarray,
    *,
    arm: int,
) -> tuple[np.ndarray, float]:
    """Apply a bounded logistic fluctuation for one treatment-specific risk."""
    from scipy.optimize import brentq
    from scipy.special import expit, logit

    initial = np.clip(np.asarray(initial_counterfactual, dtype=float), 1e-6, 1 - 1e-6)
    treatment_probability = (
        exposure_probability if arm == 1 else 1.0 - exposure_probability
    )
    clever = 1.0 / (treatment_probability * observation_probability)
    factual = (exposure == arm) & (observed == 1)
    if not factual.any():
        raise ValueError(f"no observed outcomes in TMLE arm {arm}")
    offset = logit(initial[factual])
    clever_factual = clever[factual]
    outcome_factual = outcome[factual]

    def score(epsilon: float) -> float:
        updated = expit(offset + epsilon * clever_factual)
        return float(np.sum(clever_factual * (outcome_factual - updated)))

    lower_score, upper_score = score(-50.0), score(50.0)
    if lower_score == 0:
        epsilon = -50.0
    elif upper_score == 0:
        epsilon = 50.0
    elif lower_score * upper_score < 0:
        epsilon = float(brentq(score, -50.0, 50.0, maxiter=200))
    else:
        # Complete separation can put the optimum at the fluctuation boundary.
        epsilon = -50.0 if abs(lower_score) < abs(upper_score) else 50.0
    targeted = expit(logit(initial) + epsilon * clever)
    return np.clip(targeted, 1e-8, 1.0 - 1e-8), epsilon


def _crossfit_aipw_row(frame: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import norm

    result = _empty_aipw_row(frame, "not_estimable", "")
    local = frame.loc[
        pd.to_numeric(frame["source_available"], errors="coerce").eq(1)
        & pd.to_numeric(frame["at_risk"], errors="coerce").eq(1)
        & pd.to_numeric(frame["baseline_observed"], errors="coerce").eq(1)
        & pd.to_numeric(frame["episode_exposed"], errors="coerce").isin([0, 1])
    ].copy().reset_index(drop=True)
    local["episode_exposed"] = pd.to_numeric(
        local["episode_exposed"], errors="coerce"
    ).astype(int)
    local["_observed"] = pd.to_numeric(
        local["outcome_observed"], errors="coerce"
    ).fillna(0).eq(1).astype(int)
    local["_event"] = pd.to_numeric(local["event"], errors="coerce")
    observed = local.loc[
        local["_observed"].eq(1) & local["_event"].isin([0, 1])
    ].copy()
    observed["_event"] = observed["_event"].astype(int)
    events_exposed = int(
        observed.loc[observed["episode_exposed"].eq(1), "_event"].sum()
    )
    events_unexposed = int(
        observed.loc[observed["episode_exposed"].eq(0), "_event"].sum()
    )
    observed_exposed = int(observed["episode_exposed"].eq(1).sum())
    observed_unexposed = int(observed["episode_exposed"].eq(0).sum())
    non_events_exposed = observed_exposed - events_exposed
    non_events_unexposed = observed_unexposed - events_unexposed
    result.update(
        {
            "n_risk_set": int(len(local)),
            "n_outcome_observed": int(len(observed)),
            "events_observed": int(observed["_event"].sum()),
            "events_exposed": events_exposed,
            "events_unexposed": events_unexposed,
            "non_events_exposed": non_events_exposed,
            "non_events_unexposed": non_events_unexposed,
            "claim_ready_min_10_events_per_exposure_arm": bool(
                min(events_exposed, events_unexposed) >= MIN_EVENTS
            ),
            "claim_ready_min_10_events_and_non_events_per_arm": bool(
                min(
                    events_exposed,
                    events_unexposed,
                    non_events_exposed,
                    non_events_unexposed,
                )
                >= MIN_EVENTS
            ),
        }
    )
    if len(local) < MIN_ROWS or local["episode_exposed"].nunique() < 2:
        result["status"] = "underpowered_risk_set"
        result["reason"] = "insufficient rows or exposure variation"
        return result
    if (
        len(observed) < MIN_ROWS
        or observed["_event"].nunique() < 2
        or min(events_exposed, events_unexposed) < 3
    ):
        result["status"] = "underpowered_outcome"
        result["reason"] = "insufficient observed outcome variation by exposure arm"
        return result
    splits = _crossfit_splits(local)
    if not splits:
        result["status"] = "underpowered_crossfit"
        result["reason"] = "fewer than three patient-grouped cross-fitting folds"
        return result
    n = len(local)
    propensity = np.full(n, np.nan, dtype=float)
    observation_probability = np.ones(n, dtype=float)
    observation0 = np.ones(n, dtype=float)
    observation1 = np.ones(n, dtype=float)
    outcome0 = np.full(n, np.nan, dtype=float)
    outcome1 = np.full(n, np.nan, dtype=float)
    raw_design, numeric, categorical = _raw_design_frame(
        local,
        include_baseline_value=True,
        include_site=True,
        include_concurrent_spo2=True,
    )
    result["nuisance_covariates"] = int(len(numeric) + len(categorical))
    result["crossfit_folds"] = int(len(splits))
    try:
        for train_index, test_index in splits:
            train = local.iloc[train_index].copy()
            test = local.iloc[test_index].copy()

            propensity_model, propensity_columns = _make_binary_pipeline(
                train,
                include_exposure=False,
                include_baseline_value=True,
                include_site=True,
            )
            propensity_model.fit(
                train[propensity_columns], train["episode_exposed"]
            )
            propensity[test_index] = propensity_model.predict_proba(
                test[propensity_columns]
            )[:, 1]

            if local["_observed"].nunique() > 1:
                if train["_observed"].nunique() < 2:
                    constant_observation = float(
                        np.clip(train["_observed"].mean(), *OBSERVATION_CLIP)
                    )
                    observation_probability[test_index] = constant_observation
                    observation0[test_index] = constant_observation
                    observation1[test_index] = constant_observation
                else:
                    selection_model, selection_columns = _make_binary_pipeline(
                        train,
                        include_exposure=True,
                        include_baseline_value=True,
                        include_site=True,
                        include_concurrent_spo2=True,
                    )
                    selection_model.fit(
                        train[selection_columns], train["_observed"]
                    )
                    test0, test1 = test.copy(), test.copy()
                    test0["episode_exposed"] = 0
                    test1["episode_exposed"] = 1
                    observation0[test_index] = selection_model.predict_proba(
                        test0[selection_columns]
                    )[:, 1]
                    observation1[test_index] = selection_model.predict_proba(
                        test1[selection_columns]
                    )[:, 1]
                    observation_probability[test_index] = np.where(
                        test["episode_exposed"].to_numpy(int) == 1,
                        observation1[test_index],
                        observation0[test_index],
                    )

            outcome_train = train.loc[
                train["_observed"].eq(1) & train["_event"].isin([0, 1])
            ].copy()
            test0, test1 = test.copy(), test.copy()
            test0["episode_exposed"] = 0
            test1["episode_exposed"] = 1
            if outcome_train["_event"].nunique() < 2:
                smoothed_rate = float(
                    (outcome_train["_event"].sum() + 0.5)
                    / (len(outcome_train) + 1.0)
                )
                outcome0[test_index] = smoothed_rate
                outcome1[test_index] = smoothed_rate
            else:
                outcome_model, outcome_columns = _make_binary_pipeline(
                    outcome_train,
                    include_exposure=True,
                    include_baseline_value=True,
                    include_site=True,
                    include_concurrent_spo2=True,
                )
                outcome_model.fit(
                    outcome_train[outcome_columns], outcome_train["_event"].astype(int)
                )
                outcome0[test_index] = outcome_model.predict_proba(
                    test0[outcome_columns]
                )[:, 1]
                outcome1[test_index] = outcome_model.predict_proba(
                    test1[outcome_columns]
                )[:, 1]
    except Exception as exc:
        result["status"] = "non_estimable_nuisance_model"
        result["reason"] = str(exc)[:500]
        return result

    if not all(
        np.isfinite(values).all()
        for values in (
            propensity,
            observation_probability,
            observation0,
            observation1,
            outcome0,
            outcome1,
        )
    ):
        result["status"] = "non_estimable_nuisance_prediction"
        result["reason"] = "non-finite cross-fitted nuisance prediction"
        return result
    exposure_low, exposure_high = PROPENSITY_CLIP
    observation_low, observation_high = OBSERVATION_CLIP
    result["exposure_propensity_min"] = float(propensity.min())
    result["exposure_propensity_max"] = float(propensity.max())
    result["exposure_propensity_clipped_count"] = int(
        ((propensity < exposure_low) | (propensity > exposure_high)).sum()
    )
    result["observation_probability_min"] = float(observation_probability.min())
    result["observation_probability_max"] = float(observation_probability.max())
    result["observation_probability_clipped_count"] = int(
        (
            (observation_probability < observation_low)
            | (observation_probability > observation_high)
        ).sum()
    )
    raw_propensity = propensity.copy()
    raw_observation_probability = observation_probability.copy()
    propensity = np.clip(propensity, exposure_low, exposure_high)
    observation_probability = np.clip(
        observation_probability, observation_low, observation_high
    )
    observation0 = np.clip(observation0, observation_low, observation_high)
    observation1 = np.clip(observation1, observation_low, observation_high)
    exposure = local["episode_exposed"].to_numpy(int)
    response_observed = local["_observed"].to_numpy(int)
    outcome = local["_event"].fillna(0).to_numpy(float)
    weight1 = (
        (exposure == 1).astype(float)
        * response_observed
        / (propensity * observation_probability)
    )
    weight0 = (
        (exposure == 0).astype(float)
        * response_observed
        / ((1.0 - propensity) * observation_probability)
    )
    phi1 = outcome1 + weight1 * (outcome - outcome1)
    phi0 = outcome0 + weight0 * (outcome - outcome0)
    risk1, risk0 = float(phi1.mean()), float(phi0.mean())
    groups = _identity_groups(local)
    support = _weight_support_diagnostics(
        raw_propensity,
        raw_observation_probability,
        weight1,
        weight0,
    )
    result.update(
        {
            "aipw_risk_exposed": risk1,
            "aipw_risk_unexposed": risk0,
            "aipw_effective_sample_size_exposed": support.pop(
                "effective_sample_size_exposed"
            ),
            "aipw_effective_sample_size_unexposed": support.pop(
                "effective_sample_size_unexposed"
            ),
            **support,
        }
    )
    if 0.0 < risk1 < 1.0 and 0.0 < risk0 < 1.0:
        log_rr = math.log(risk1 / risk0)
        influence_log_rr = (phi1 - risk1) / risk1 - (phi0 - risk0) / risk0
        log_rr_se = _cluster_standard_error(influence_log_rr, groups)
        risk_difference = risk1 - risk0
        influence_rd = (phi1 - risk1) - (phi0 - risk0)
        risk_difference_se = _cluster_standard_error(influence_rd, groups)
        if np.isfinite(log_rr_se) and np.isfinite(risk_difference_se):
            result.update(
                {
                    "one_step_aipw_status": "estimated_bounded",
                    "aipw_risk_ratio": math.exp(log_rr),
                    "aipw_risk_ratio_ci95_low": math.exp(
                        log_rr - 1.96 * log_rr_se
                    ),
                    "aipw_risk_ratio_ci95_high": math.exp(
                        log_rr + 1.96 * log_rr_se
                    ),
                    "aipw_log_rr_se": log_rr_se,
                    "aipw_risk_ratio_p_value": float(
                        2.0 * norm.sf(abs(log_rr / log_rr_se))
                    )
                    if log_rr_se > 0
                    else math.nan,
                    "aipw_risk_difference": risk_difference,
                    "aipw_risk_difference_ci95_low": risk_difference
                    - 1.96 * risk_difference_se,
                    "aipw_risk_difference_ci95_high": risk_difference
                    + 1.96 * risk_difference_se,
                    "aipw_risk_difference_p_value": float(
                        2.0 * norm.sf(abs(risk_difference / risk_difference_se))
                    )
                    if risk_difference_se > 0
                    else math.nan,
                }
            )
        else:
            result["one_step_aipw_status"] = "non_finite_cluster_variance"
    else:
        result["one_step_aipw_status"] = "out_of_bounds_retained_as_diagnostic"

    try:
        targeted1, epsilon1 = _tmle_target_arm(
            outcome1,
            propensity,
            observation1,
            exposure,
            response_observed,
            outcome,
            arm=1,
        )
        targeted0, epsilon0 = _tmle_target_arm(
            outcome0,
            propensity,
            observation0,
            exposure,
            response_observed,
            outcome,
            arm=0,
        )
    except Exception as exc:
        result["status"] = "non_estimable_tmle_targeting"
        result["reason"] = str(exc)[:500]
        return result
    tmle_risk1, tmle_risk0 = float(targeted1.mean()), float(targeted0.mean())
    tmle_ic1 = weight1 * (outcome - targeted1) + targeted1 - tmle_risk1
    tmle_ic0 = weight0 * (outcome - targeted0) + targeted0 - tmle_risk0
    tmle_log_rr = math.log(tmle_risk1 / tmle_risk0)
    tmle_log_rr_se = _cluster_standard_error(
        tmle_ic1 / tmle_risk1 - tmle_ic0 / tmle_risk0,
        groups,
    )
    tmle_rd = tmle_risk1 - tmle_risk0
    tmle_rd_se = _cluster_standard_error(tmle_ic1 - tmle_ic0, groups)
    if not np.isfinite(tmle_log_rr_se) or not np.isfinite(tmle_rd_se):
        result["status"] = "non_estimable_tmle_cluster_variance"
        result["reason"] = "non-finite patient-cluster targeted influence variance"
        return result
    result.update(
        {
            "status": "estimated",
            "reason": "",
            "tmle_risk_exposed": tmle_risk1,
            "tmle_risk_unexposed": tmle_risk0,
            "tmle_risk_ratio": math.exp(tmle_log_rr),
            "tmle_risk_ratio_ci95_low": math.exp(
                tmle_log_rr - 1.96 * tmle_log_rr_se
            ),
            "tmle_risk_ratio_ci95_high": math.exp(
                tmle_log_rr + 1.96 * tmle_log_rr_se
            ),
            "tmle_log_rr_se": tmle_log_rr_se,
            "tmle_risk_ratio_p_value": float(
                2.0 * norm.sf(abs(tmle_log_rr / tmle_log_rr_se))
            )
            if tmle_log_rr_se > 0
            else math.nan,
            "tmle_risk_difference": tmle_rd,
            "tmle_risk_difference_ci95_low": tmle_rd - 1.96 * tmle_rd_se,
            "tmle_risk_difference_ci95_high": tmle_rd + 1.96 * tmle_rd_se,
            "tmle_risk_difference_p_value": float(
                2.0 * norm.sf(abs(tmle_rd / tmle_rd_se))
            )
            if tmle_rd_se > 0
            else math.nan,
            "tmle_targeting_epsilon_exposed": epsilon1,
            "tmle_targeting_epsilon_unexposed": epsilon0,
            "tmle_targeting_boundary_hit": bool(
                abs(epsilon1) >= 49.99 or abs(epsilon0) >= 49.99
            ),
            "tmle_efficient_influence_mean_exposed": float(tmle_ic1.mean()),
            "tmle_efficient_influence_mean_unexposed": float(tmle_ic0.mean()),
        }
    )
    return result


def build_crossfit_aipw_effects(
    focused_records: pd.DataFrame,
    preanchor_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate all focused binary endpoints with cross-fitted missingness AIPW."""
    keys = ["dataset", "stay_id"]
    context_columns = [
        column
        for column in preanchor_covariates.columns
        if column not in {"episode_exposed", "analysis_status"}
    ]
    merged = focused_records.merge(
        preanchor_covariates[context_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_preanchor"),
    )
    grouped = merged.groupby(["dataset", "endpoint", "lag_window"], sort=False)
    group_count = int(grouped.ngroups)
    rows: list[dict[str, Any]] = []
    for index, (group_key, group) in enumerate(grouped, start=1):
        rows.append(_crossfit_aipw_row(group))
        if index == 1 or index % 10 == 0 or index == group_count:
            print(
                f"[advanced] binary AIPW/TMLE {index}/{group_count}: {group_key}",
                flush=True,
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["one_step_aipw_endpoint_q_value"] = np.nan
    result["one_step_aipw_cross_endpoint_q_value"] = np.nan
    result["tmle_endpoint_q_value"] = np.nan
    result["tmle_cross_endpoint_q_value"] = np.nan
    for _, indexes in result.groupby(["dataset", "endpoint"]).groups.items():
        result.loc[indexes, "one_step_aipw_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "aipw_risk_ratio_p_value"]
        )
        result.loc[indexes, "tmle_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "tmle_risk_ratio_p_value"]
        )
    for _, indexes in result.groupby("dataset").groups.items():
        result.loc[indexes, "one_step_aipw_cross_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "aipw_risk_ratio_p_value"]
        )
        result.loc[indexes, "tmle_cross_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "tmle_risk_ratio_p_value"]
        )
    # Generic q-value aliases point to the bounded targeted estimator.
    result["endpoint_q_value"] = result["tmle_endpoint_q_value"]
    result["cross_endpoint_q_value"] = result["tmle_cross_endpoint_q_value"]
    result["aipw_claim_ready_positive"] = (
        result["one_step_aipw_status"].eq("estimated_bounded")
        & result["claim_ready_min_10_events_and_non_events_per_arm"].eq(True)  # noqa: E712
        & result["claim_ready_weight_diagnostics"].eq(True)  # noqa: E712
        & pd.to_numeric(result["aipw_risk_ratio"], errors="coerce").gt(1)
        & pd.to_numeric(result["aipw_risk_ratio_ci95_low"], errors="coerce").gt(1)
    )
    result["aipw_cross_endpoint_fdr_positive"] = (
        result["aipw_claim_ready_positive"]
        & pd.to_numeric(
            result["one_step_aipw_cross_endpoint_q_value"], errors="coerce"
        ).lt(0.05)
    )
    result["tmle_claim_ready_positive"] = (
        result["status"].eq("estimated")
        & result["claim_ready_min_10_events_and_non_events_per_arm"].eq(True)  # noqa: E712
        & result["claim_ready_weight_diagnostics"].eq(True)  # noqa: E712
        & result["tmle_targeting_boundary_hit"].eq(False)  # noqa: E712
        & pd.to_numeric(result["tmle_risk_ratio"], errors="coerce").gt(1)
        & pd.to_numeric(result["tmle_risk_ratio_ci95_low"], errors="coerce").gt(1)
    )
    result["tmle_cross_endpoint_fdr_positive"] = (
        result["tmle_claim_ready_positive"]
        & pd.to_numeric(result["tmle_cross_endpoint_q_value"], errors="coerce").lt(
            0.05
        )
    )
    return result.sort_values(["dataset", "endpoint", "lag_window"]).reset_index(
        drop=True
    )


def _empty_continuous_aipw_row(
    frame: pd.DataFrame, status: str, reason: str
) -> dict[str, Any]:
    first = frame.iloc[0]
    return {
        "dataset": str(first["dataset"]),
        "endpoint": str(first["endpoint"]),
        "endpoint_family": str(first["endpoint_family"]),
        "lag_window": str(first["lag_window"]),
        "change_scale": str(first.get("change_scale", "")),
        "status": status,
        "reason": reason,
        "n_risk_set": 0,
        "n_continuous_observed": 0,
        "n_continuous_exposed": 0,
        "n_continuous_unexposed": 0,
        "crossfit_folds": 0,
        "nuisance_covariates": 0,
        "aipw_mean_change_exposed": math.nan,
        "aipw_mean_change_unexposed": math.nan,
        "aipw_mean_change_difference": math.nan,
        "aipw_mean_change_difference_ci95_low": math.nan,
        "aipw_mean_change_difference_ci95_high": math.nan,
        "aipw_mean_change_difference_se": math.nan,
        "aipw_mean_change_p_value": math.nan,
        "aipw_standardized_mean_difference": math.nan,
        "aipw_standardized_ci95_low": math.nan,
        "aipw_standardized_ci95_high": math.nan,
        "observed_change_standard_deviation": math.nan,
        "exposure_propensity_min": math.nan,
        "exposure_propensity_max": math.nan,
        "exposure_propensity_clipped_count": 0,
        "exposure_propensity_clipped_fraction": math.nan,
        "observation_probability_min": math.nan,
        "observation_probability_max": math.nan,
        "observation_probability_clipped_count": 0,
        "observation_probability_lower_clipped_count": 0,
        "observation_probability_lower_clipped_fraction": math.nan,
        "effective_sample_size_exposed": math.nan,
        "effective_sample_size_unexposed": math.nan,
        "claim_ready_propensity_support": False,
        "claim_ready_observation_support": False,
        "claim_ready_effective_sample_size": False,
        "claim_ready_weight_diagnostics": False,
        "claim_ready_min_30_continuous_per_exposure_arm": False,
        "continuous_change_direction": "higher_is_worse",
        "outcome_regression": "crossfit_ridge_alpha_1",
        "robustness_reference": (
            "unadjusted_patient_bootstrap_median_difference_in_multiorgan_effects"
        ),
        "estimand": "marginal_mean_worsening_difference_in_anchor_risk_set",
        "assumptions": (
            "consistency+conditional_exchangeability+positivity+missing_at_random;"
            "not_a_causal_claim"
        ),
        "analysis_status": ADVANCED_ANALYSIS_STATUS,
    }


def _crossfit_continuous_aipw_row(frame: pd.DataFrame) -> dict[str, Any]:
    from scipy.stats import norm

    result = _empty_continuous_aipw_row(frame, "not_estimable", "")
    local = frame.loc[
        pd.to_numeric(frame["source_available"], errors="coerce").eq(1)
        & pd.to_numeric(frame["at_risk"], errors="coerce").eq(1)
        & pd.to_numeric(frame["baseline_observed"], errors="coerce").eq(1)
        & pd.to_numeric(frame["episode_exposed"], errors="coerce").isin([0, 1])
    ].copy().reset_index(drop=True)
    local["episode_exposed"] = pd.to_numeric(
        local["episode_exposed"], errors="coerce"
    ).astype(int)
    local["_change"] = pd.to_numeric(local["change_value"], errors="coerce")
    local["_observed"] = (
        pd.to_numeric(local["outcome_observed"], errors="coerce")
        .fillna(0)
        .eq(1)
        & local["_change"].notna()
    ).astype(int)
    observed = local.loc[local["_observed"].eq(1)].copy()
    n_exposed = int(observed["episode_exposed"].eq(1).sum())
    n_unexposed = int(observed["episode_exposed"].eq(0).sum())
    result.update(
        {
            "n_risk_set": int(len(local)),
            "n_continuous_observed": int(len(observed)),
            "n_continuous_exposed": n_exposed,
            "n_continuous_unexposed": n_unexposed,
            "claim_ready_min_30_continuous_per_exposure_arm": bool(
                min(n_exposed, n_unexposed) >= 30
            ),
        }
    )
    if not len(local) or local["_change"].notna().sum() == 0:
        result["status"] = "continuous_outcome_not_defined"
        result["reason"] = "primary endpoint variant has no continuous change estimand"
        return result
    if len(local) < MIN_ROWS or local["episode_exposed"].nunique() < 2:
        result["status"] = "underpowered_risk_set"
        result["reason"] = "insufficient rows or exposure variation"
        return result
    if len(observed) < MIN_ROWS or min(n_exposed, n_unexposed) < 20:
        result["status"] = "underpowered_continuous_outcome"
        result["reason"] = "fewer than 80 observations or 20 per exposure arm"
        return result
    splits = _crossfit_splits(local)
    if not splits:
        result["status"] = "underpowered_crossfit"
        result["reason"] = "fewer than three patient-grouped cross-fitting folds"
        return result
    n = len(local)
    propensity = np.full(n, np.nan, dtype=float)
    observation_probability = np.ones(n, dtype=float)
    outcome0 = np.full(n, np.nan, dtype=float)
    outcome1 = np.full(n, np.nan, dtype=float)
    _, numeric, categorical = _raw_design_frame(
        local,
        include_baseline_value=True,
        include_site=True,
        include_concurrent_spo2=True,
    )
    result["nuisance_covariates"] = int(len(numeric) + len(categorical))
    result["crossfit_folds"] = int(len(splits))
    try:
        for train_index, test_index in splits:
            train = local.iloc[train_index].copy()
            test = local.iloc[test_index].copy()
            propensity_model, propensity_columns = _make_binary_pipeline(
                train,
                include_exposure=False,
                include_baseline_value=True,
                include_site=True,
            )
            propensity_model.fit(
                train[propensity_columns], train["episode_exposed"]
            )
            propensity[test_index] = propensity_model.predict_proba(
                test[propensity_columns]
            )[:, 1]
            if local["_observed"].nunique() > 1:
                if train["_observed"].nunique() < 2:
                    observation_probability[test_index] = float(
                        np.clip(train["_observed"].mean(), *OBSERVATION_CLIP)
                    )
                else:
                    observation_model, observation_columns = _make_binary_pipeline(
                        train,
                        include_exposure=True,
                        include_baseline_value=True,
                        include_site=True,
                        include_concurrent_spo2=True,
                    )
                    observation_model.fit(
                        train[observation_columns], train["_observed"]
                    )
                    observation_probability[test_index] = (
                        observation_model.predict_proba(test[observation_columns])[
                            :, 1
                        ]
                    )
            outcome_train = train.loc[train["_observed"].eq(1)].copy()
            outcome_model, outcome_columns = _make_continuous_pipeline(
                outcome_train,
                include_exposure=True,
                include_baseline_value=True,
                include_site=True,
                include_concurrent_spo2=True,
            )
            outcome_model.fit(
                outcome_train[outcome_columns], outcome_train["_change"]
            )
            test0, test1 = test.copy(), test.copy()
            test0["episode_exposed"] = 0
            test1["episode_exposed"] = 1
            outcome0[test_index] = outcome_model.predict(test0[outcome_columns])
            outcome1[test_index] = outcome_model.predict(test1[outcome_columns])
    except Exception as exc:
        result["status"] = "non_estimable_nuisance_model"
        result["reason"] = str(exc)[:500]
        return result
    if not all(
        np.isfinite(values).all()
        for values in (propensity, observation_probability, outcome0, outcome1)
    ):
        result["status"] = "non_estimable_nuisance_prediction"
        result["reason"] = "non-finite cross-fitted nuisance prediction"
        return result
    exposure_low, exposure_high = PROPENSITY_CLIP
    observation_low, observation_high = OBSERVATION_CLIP
    result["exposure_propensity_min"] = float(propensity.min())
    result["exposure_propensity_max"] = float(propensity.max())
    result["exposure_propensity_clipped_count"] = int(
        ((propensity < exposure_low) | (propensity > exposure_high)).sum()
    )
    result["observation_probability_min"] = float(observation_probability.min())
    result["observation_probability_max"] = float(observation_probability.max())
    result["observation_probability_clipped_count"] = int(
        (
            (observation_probability < observation_low)
            | (observation_probability > observation_high)
        ).sum()
    )
    raw_propensity = propensity.copy()
    raw_observation_probability = observation_probability.copy()
    propensity = np.clip(propensity, exposure_low, exposure_high)
    observation_probability = np.clip(
        observation_probability, observation_low, observation_high
    )
    exposure = local["episode_exposed"].to_numpy(int)
    response_observed = local["_observed"].to_numpy(int)
    outcome = local["_change"].fillna(0).to_numpy(float)
    weight1 = (
        (exposure == 1).astype(float)
        * response_observed
        / (propensity * observation_probability)
    )
    weight0 = (
        (exposure == 0).astype(float)
        * response_observed
        / ((1.0 - propensity) * observation_probability)
    )
    phi1 = outcome1 + weight1 * (outcome - outcome1)
    phi0 = outcome0 + weight0 * (outcome - outcome0)
    mean1, mean0 = float(phi1.mean()), float(phi0.mean())
    difference = mean1 - mean0
    influence = (phi1 - mean1) - (phi0 - mean0)
    standard_error = _cluster_standard_error(influence, _identity_groups(local))
    if not np.isfinite(standard_error):
        result["status"] = "non_estimable_cluster_variance"
        result["reason"] = "non-finite patient-cluster influence variance"
        return result
    observed_sd = float(observed["_change"].std(ddof=1))
    standardized = difference / observed_sd if observed_sd > 0 else math.nan
    standardized_se = standard_error / observed_sd if observed_sd > 0 else math.nan
    support = _weight_support_diagnostics(
        raw_propensity,
        raw_observation_probability,
        weight1,
        weight0,
    )
    result.update(
        {
            "status": "estimated",
            "reason": "",
            "aipw_mean_change_exposed": mean1,
            "aipw_mean_change_unexposed": mean0,
            "aipw_mean_change_difference": difference,
            "aipw_mean_change_difference_ci95_low": difference
            - 1.96 * standard_error,
            "aipw_mean_change_difference_ci95_high": difference
            + 1.96 * standard_error,
            "aipw_mean_change_difference_se": standard_error,
            "aipw_mean_change_p_value": float(
                2.0 * norm.sf(abs(difference / standard_error))
            )
            if standard_error > 0
            else math.nan,
            "aipw_standardized_mean_difference": standardized,
            "aipw_standardized_ci95_low": standardized
            - 1.96 * standardized_se
            if np.isfinite(standardized_se)
            else math.nan,
            "aipw_standardized_ci95_high": standardized
            + 1.96 * standardized_se
            if np.isfinite(standardized_se)
            else math.nan,
            "observed_change_standard_deviation": observed_sd,
            **support,
        }
    )
    return result


def build_crossfit_continuous_aipw_effects(
    focused_records: pd.DataFrame,
    preanchor_covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate graded endpoint worsening without relying on binary thresholds."""
    keys = ["dataset", "stay_id"]
    context_columns = [
        column
        for column in preanchor_covariates.columns
        if column not in {"episode_exposed", "analysis_status"}
    ]
    merged = focused_records.merge(
        preanchor_covariates[context_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_preanchor"),
    )
    grouped = merged.groupby(["dataset", "endpoint", "lag_window"], sort=False)
    group_count = int(grouped.ngroups)
    rows: list[dict[str, Any]] = []
    for index, (group_key, group) in enumerate(grouped, start=1):
        rows.append(_crossfit_continuous_aipw_row(group))
        if index == 1 or index % 10 == 0 or index == group_count:
            print(
                f"[advanced] continuous AIPW {index}/{group_count}: {group_key}",
                flush=True,
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["endpoint_q_value"] = np.nan
    result["cross_endpoint_q_value"] = np.nan
    for _, indexes in result.groupby(["dataset", "endpoint"]).groups.items():
        result.loc[indexes, "endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "aipw_mean_change_p_value"]
        )
    for _, indexes in result.groupby("dataset").groups.items():
        result.loc[indexes, "cross_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "aipw_mean_change_p_value"]
        )
    result["continuous_claim_ready_positive"] = (
        result["status"].eq("estimated")
        & result["claim_ready_min_30_continuous_per_exposure_arm"].eq(True)  # noqa: E712
        & result["claim_ready_weight_diagnostics"].eq(True)  # noqa: E712
        & pd.to_numeric(
            result["aipw_mean_change_difference_ci95_low"], errors="coerce"
        ).gt(0)
    )
    result["continuous_cross_endpoint_fdr_positive"] = (
        result["continuous_claim_ready_positive"]
        & pd.to_numeric(result["cross_endpoint_q_value"], errors="coerce").lt(0.05)
    )
    return result.sort_values(["dataset", "endpoint", "lag_window"]).reset_index(
        drop=True
    )


def _empty_overlap_effect_row(
    frame: pd.DataFrame, status: str, reason: str
) -> dict[str, Any]:
    first = frame.iloc[0]
    return {
        "dataset": str(first["dataset"]),
        "endpoint": str(first["endpoint"]),
        "endpoint_family": str(first["endpoint_family"]),
        "lag_window": str(first["lag_window"]),
        "status": status,
        "reason": reason,
        "n_risk_set": 0,
        "n_outcome_observed": 0,
        "events_exposed": 0,
        "events_unexposed": 0,
        "non_events_exposed": 0,
        "non_events_unexposed": 0,
        "overlap_weighted_risk_exposed": math.nan,
        "overlap_weighted_risk_unexposed": math.nan,
        "overlap_weighted_risk_ratio": math.nan,
        "overlap_weighted_ci95_low": math.nan,
        "overlap_weighted_ci95_high": math.nan,
        "overlap_weighted_p_value": math.nan,
        "exposure_propensity_min": math.nan,
        "exposure_propensity_max": math.nan,
        "exposure_propensity_clipped_count": 0,
        "exposure_propensity_clipped_fraction": math.nan,
        "selection_probability_min": math.nan,
        "selection_probability_max": math.nan,
        "selection_probability_clipped_count": 0,
        "observation_probability_lower_clipped_count": 0,
        "observation_probability_lower_clipped_fraction": math.nan,
        "combined_weight_max": math.nan,
        "effective_sample_size_exposed": math.nan,
        "effective_sample_size_unexposed": math.nan,
        "claim_ready_propensity_support": False,
        "claim_ready_observation_support": False,
        "claim_ready_effective_sample_size": False,
        "claim_ready_weight_diagnostics": False,
        "claim_ready_min_10_events_per_exposure_arm": False,
        "claim_ready_min_10_events_and_non_events_per_arm": False,
        "estimand": "marginal_risk_ratio_in_overlap_population",
        "assumptions": (
            "conditional_exchangeability+positivity+missing_at_random;"
            "not_a_causal_claim"
        ),
        "analysis_status": ADVANCED_ANALYSIS_STATUS,
    }


def _crossfit_observation_probability(frame: pd.DataFrame) -> np.ndarray:
    observed = pd.to_numeric(frame["_observed"], errors="coerce").fillna(0).astype(int)
    if observed.nunique() < 2:
        return np.ones(len(frame), dtype=float)
    splits = _crossfit_splits(frame)
    if not splits:
        raise ValueError("fewer than three patient-grouped observation folds")
    probabilities = np.full(len(frame), np.nan, dtype=float)
    for train_index, test_index in splits:
        train = frame.iloc[train_index].copy()
        test = frame.iloc[test_index].copy()
        if train["_observed"].nunique() < 2:
            probabilities[test_index] = float(train["_observed"].mean())
            continue
        model, columns = _make_binary_pipeline(
            train,
            include_exposure=True,
            include_baseline_value=True,
            include_site=True,
            include_concurrent_spo2=True,
        )
        model.fit(train[columns], train["_observed"])
        probabilities[test_index] = model.predict_proba(test[columns])[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite cross-fitted observation probabilities")
    return probabilities


def _overlap_weighted_effect_row(frame: pd.DataFrame) -> dict[str, Any]:
    import statsmodels.api as sm

    result = _empty_overlap_effect_row(frame, "not_estimable", "")
    local = frame.loc[
        pd.to_numeric(frame["source_available"], errors="coerce").eq(1)
        & pd.to_numeric(frame["at_risk"], errors="coerce").eq(1)
        & pd.to_numeric(frame["baseline_observed"], errors="coerce").eq(1)
        & pd.to_numeric(frame["episode_exposed"], errors="coerce").isin([0, 1])
        & pd.to_numeric(frame["overlap_weight"], errors="coerce").gt(0)
    ].copy().reset_index(drop=True)
    local["episode_exposed"] = pd.to_numeric(
        local["episode_exposed"], errors="coerce"
    ).astype(int)
    local["_observed"] = (
        pd.to_numeric(local["outcome_observed"], errors="coerce")
        .fillna(0)
        .eq(1)
        .astype(int)
    )
    local["_event"] = pd.to_numeric(local["event"], errors="coerce")
    observed = local.loc[
        local["_observed"].eq(1) & local["_event"].isin([0, 1])
    ].copy()
    observed["_event"] = observed["_event"].astype(int)
    events_exposed = int(
        observed.loc[observed["episode_exposed"].eq(1), "_event"].sum()
    )
    events_unexposed = int(
        observed.loc[observed["episode_exposed"].eq(0), "_event"].sum()
    )
    observed_exposed = int(observed["episode_exposed"].eq(1).sum())
    observed_unexposed = int(observed["episode_exposed"].eq(0).sum())
    non_events_exposed = observed_exposed - events_exposed
    non_events_unexposed = observed_unexposed - events_unexposed
    result.update(
        {
            "n_risk_set": int(len(local)),
            "n_outcome_observed": int(len(observed)),
            "events_exposed": events_exposed,
            "events_unexposed": events_unexposed,
            "non_events_exposed": non_events_exposed,
            "non_events_unexposed": non_events_unexposed,
            "claim_ready_min_10_events_per_exposure_arm": bool(
                min(events_exposed, events_unexposed) >= MIN_EVENTS
            ),
            "claim_ready_min_10_events_and_non_events_per_arm": bool(
                min(
                    events_exposed,
                    events_unexposed,
                    non_events_exposed,
                    non_events_unexposed,
                )
                >= MIN_EVENTS
            ),
        }
    )
    if len(local) < MIN_ROWS or local["episode_exposed"].nunique() < 2:
        result["status"] = "underpowered_risk_set"
        result["reason"] = "insufficient rows or exposure variation"
        return result
    if (
        len(observed) < MIN_ROWS
        or observed["_event"].nunique() < 2
        or min(events_exposed, events_unexposed) < 3
    ):
        result["status"] = "underpowered_outcome"
        result["reason"] = "insufficient observed outcome variation by exposure arm"
        return result
    try:
        observation_probability = _crossfit_observation_probability(local)
    except Exception as exc:
        result["status"] = "non_estimable_observation_model"
        result["reason"] = str(exc)[:500]
        return result
    propensity = pd.to_numeric(
        local["episode_propensity"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(propensity).all():
        result["status"] = "non_estimable_weights"
        result["reason"] = "non-finite exposure propensity"
        return result
    result["exposure_propensity_min"] = float(propensity.min())
    result["exposure_propensity_max"] = float(propensity.max())
    low, high = OBSERVATION_CLIP
    result["selection_probability_min"] = float(observation_probability.min())
    result["selection_probability_max"] = float(observation_probability.max())
    result["selection_probability_clipped_count"] = int(
        ((observation_probability < low) | (observation_probability > high)).sum()
    )
    local["_observation_probability"] = np.clip(
        observation_probability, low, high
    )
    complete = local.loc[
        local["_observed"].eq(1) & local["_event"].isin([0, 1])
    ].copy().reset_index(drop=True)
    complete["_combined_weight"] = (
        pd.to_numeric(complete["overlap_weight"], errors="coerce")
        / complete["_observation_probability"]
    )
    if (
        not np.isfinite(complete["_combined_weight"]).all()
        or complete["_combined_weight"].le(0).any()
    ):
        result["status"] = "non_estimable_weights"
        result["reason"] = "non-finite or non-positive combined weights"
        return result
    result["combined_weight_max"] = float(complete["_combined_weight"].max())
    exposure = complete["episode_exposed"].to_numpy(int)
    event = complete["_event"].to_numpy(float)
    weight = complete["_combined_weight"].to_numpy(float)
    risk1 = _weighted_mean(event[exposure == 1], weight[exposure == 1])
    risk0 = _weighted_mean(event[exposure == 0], weight[exposure == 0])
    result.update(
        _weight_support_diagnostics(
            propensity,
            observation_probability,
            weight[exposure == 1],
            weight[exposure == 0],
        )
    )
    if not (np.isfinite(risk1) and np.isfinite(risk0) and risk1 > 0 and risk0 > 0):
        result["status"] = "non_estimable_zero_risk"
        result["reason"] = "weighted risk was zero or non-finite"
        return result
    design = sm.add_constant(
        complete[["episode_exposed"]].astype(float), has_constant="add"
    )
    try:
        groups = _identity_groups(complete)
        if groups.nunique() > 1:
            fit = sm.GEE(
                complete["_event"].astype(int),
                design,
                groups=groups,
                family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Independence(),
                weights=complete["_combined_weight"],
            ).fit()
        else:
            fit = sm.GLM(
                complete["_event"].astype(int),
                design,
                family=sm.families.Poisson(),
                freq_weights=complete["_combined_weight"],
            ).fit(cov_type="HC0")
        coefficient = float(fit.params["episode_exposed"])
        ci_low, ci_high = fit.conf_int().loc["episode_exposed"]
        inference = np.exp([coefficient, float(ci_low), float(ci_high)])
        if not np.isfinite(inference).all():
            raise ValueError("non-finite overlap-weighted inference")
    except Exception as exc:
        result["status"] = "non_estimable_outcome_model"
        result["reason"] = str(exc)[:500]
        return result
    result.update(
        {
            "status": "estimated",
            "reason": "",
            "overlap_weighted_risk_exposed": risk1,
            "overlap_weighted_risk_unexposed": risk0,
            "overlap_weighted_risk_ratio": float(inference[0]),
            "overlap_weighted_ci95_low": float(inference[1]),
            "overlap_weighted_ci95_high": float(inference[2]),
            "overlap_weighted_p_value": float(fit.pvalues["episode_exposed"]),
        }
    )
    return result


def build_overlap_weighted_effects(
    focused_records: pd.DataFrame,
    preanchor_covariates: pd.DataFrame,
    anchor_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate focused endpoint RRs in the pre-anchor overlap population."""
    keys = ["dataset", "stay_id"]
    weight_context = preanchor_covariates.merge(
        anchor_weights.reindex(
            columns=[*keys, "episode_propensity", "overlap_weight"]
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    context_columns = [
        column
        for column in weight_context.columns
        if column not in {"episode_exposed", "analysis_status"}
    ]
    merged = focused_records.merge(
        weight_context[context_columns],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_preanchor"),
    )
    rows = [
        _overlap_weighted_effect_row(group)
        for _, group in merged.groupby(
            ["dataset", "endpoint", "lag_window"], sort=False
        )
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["endpoint_q_value"] = np.nan
    result["cross_endpoint_q_value"] = np.nan
    for _, indexes in result.groupby(["dataset", "endpoint"]).groups.items():
        result.loc[indexes, "endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "overlap_weighted_p_value"]
        )
    for _, indexes in result.groupby("dataset").groups.items():
        result.loc[indexes, "cross_endpoint_q_value"] = _bh_adjust(
            result.loc[indexes, "overlap_weighted_p_value"]
        )
    result["overlap_claim_ready_positive"] = (
        result["status"].eq("estimated")
        & result["claim_ready_min_10_events_and_non_events_per_arm"].eq(True)  # noqa: E712
        & result["claim_ready_weight_diagnostics"].eq(True)  # noqa: E712
        & pd.to_numeric(result["overlap_weighted_risk_ratio"], errors="coerce").gt(1)
        & pd.to_numeric(result["overlap_weighted_ci95_low"], errors="coerce").gt(1)
    )
    result["overlap_cross_endpoint_fdr_positive"] = (
        result["overlap_claim_ready_positive"]
        & pd.to_numeric(result["cross_endpoint_q_value"], errors="coerce").lt(0.05)
    )
    return result.sort_values(["dataset", "endpoint", "lag_window"]).reset_index(
        drop=True
    )


def _site_log_risk_ratio(
    events_exposed: int,
    n_exposed: int,
    events_unexposed: int,
    n_unexposed: int,
) -> tuple[float, float, bool]:
    cells = (
        events_exposed,
        n_exposed - events_exposed,
        events_unexposed,
        n_unexposed - events_unexposed,
    )
    correction = 0.5 if any(cell == 0 for cell in cells) else 0.0
    a = events_exposed + correction
    n1 = n_exposed + 2.0 * correction
    c = events_unexposed + correction
    n0 = n_unexposed + 2.0 * correction
    if min(a, n1, c, n0) <= 0:
        return math.nan, math.nan, bool(correction)
    log_rr = math.log((a / n1) / (c / n0))
    variance = 1.0 / a - 1.0 / n1 + 1.0 / c - 1.0 / n0
    return log_rr, math.sqrt(max(variance, 0.0)), bool(correction)


def _paule_mandel_hksj(
    effects: np.ndarray,
    variances: np.ndarray,
) -> dict[str, float]:
    from scipy.stats import t

    effects = np.asarray(effects, dtype=float)
    variances = np.asarray(variances, dtype=float)
    finite = np.isfinite(effects) & np.isfinite(variances) & (variances > 0)
    effects, variances = effects[finite], variances[finite]
    k = len(effects)
    if k < 3:
        return {
            "mean": math.nan,
            "se": math.nan,
            "low": math.nan,
            "high": math.nan,
            "p": math.nan,
            "tau_squared": math.nan,
            "i_squared_percent": math.nan,
            "cochran_q": math.nan,
        }

    def weighted(tau_squared: float) -> tuple[float, float, float]:
        weights = 1.0 / (variances + tau_squared)
        mean = float(np.sum(weights * effects) / np.sum(weights))
        q = float(np.sum(weights * np.square(effects - mean)))
        return mean, q, float(np.sum(weights))

    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * effects) / np.sum(fixed_weights))
    cochran_q = float(np.sum(fixed_weights * np.square(effects - fixed_mean)))
    i_squared = max((cochran_q - (k - 1)) / max(cochran_q, 1e-12), 0.0) * 100.0
    _, q_zero, _ = weighted(0.0)
    if q_zero <= k - 1:
        tau_squared = 0.0
    else:
        lower, upper = 0.0, max(float(np.var(effects, ddof=1)), 1e-6)
        while weighted(upper)[1] > k - 1 and upper < 1e6:
            upper *= 2.0
        for _ in range(100):
            midpoint = (lower + upper) / 2.0
            if weighted(midpoint)[1] > k - 1:
                lower = midpoint
            else:
                upper = midpoint
        tau_squared = (lower + upper) / 2.0
    mean, q_random, weight_sum = weighted(tau_squared)
    hksj_scale = max(q_random / (k - 1), 1.0)
    se = math.sqrt(hksj_scale / weight_sum)
    critical = float(t.ppf(0.975, df=k - 1))
    p_value = float(2.0 * t.sf(abs(mean / se), df=k - 1)) if se > 0 else math.nan
    return {
        "mean": mean,
        "se": se,
        "low": mean - critical * se,
        "high": mean + critical * se,
        "p": p_value,
        "tau_squared": tau_squared,
        "i_squared_percent": i_squared,
        "cochran_q": cochran_q,
    }


def build_multicenter_robustness(
    focused_records: pd.DataFrame,
    cohort: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Estimate hospital-specific and conservative random-effects associations."""
    keys = ["dataset", "stay_id"]
    site = cohort.reindex(columns=[*keys, "hospital_id"]).drop_duplicates(keys)
    merged = focused_records.merge(site, on=keys, how="left", validate="many_to_one")
    merged["hospital_id"] = merged["hospital_id"].fillna("unknown").astype(str)
    site_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (dataset, endpoint, family, lag_window), group in merged.groupby(
        ["dataset", "endpoint", "endpoint_family", "lag_window"], sort=False
    ):
        eligible = group.loc[
            pd.to_numeric(group["source_available"], errors="coerce").eq(1)
            & pd.to_numeric(group["at_risk"], errors="coerce").eq(1)
            & pd.to_numeric(group["outcome_observed"], errors="coerce").eq(1)
            & pd.to_numeric(group["event"], errors="coerce").isin([0, 1])
        ].copy()
        eligible["event"] = pd.to_numeric(eligible["event"], errors="coerce").astype(int)
        local_site_rows: list[dict[str, Any]] = []
        for hospital, hospital_frame in eligible.groupby("hospital_id", sort=False):
            exposed = hospital_frame["episode_exposed"].eq(1)
            unexposed = hospital_frame["episode_exposed"].eq(0)
            n1, n0 = int(exposed.sum()), int(unexposed.sum())
            a = int(hospital_frame.loc[exposed, "event"].sum())
            c = int(hospital_frame.loc[unexposed, "event"].sum())
            total_events = a + c
            total_non_events = n1 + n0 - total_events
            estimable = n1 >= 5 and n0 >= 5 and total_events >= 2 and total_non_events >= 2
            log_rr, log_se, corrected = (
                _site_log_risk_ratio(a, n1, c, n0)
                if estimable
                else (math.nan, math.nan, False)
            )
            row = {
                "dataset": str(dataset),
                "endpoint": str(endpoint),
                "endpoint_family": str(family),
                "lag_window": str(lag_window),
                "hospital_id": str(hospital),
                "status": "estimated" if estimable else "underpowered_site",
                "n_exposed": n1,
                "n_unexposed": n0,
                "events_exposed": a,
                "events_unexposed": c,
                "risk_exposed": a / n1 if n1 else math.nan,
                "risk_unexposed": c / n0 if n0 else math.nan,
                "risk_ratio": math.exp(log_rr) if np.isfinite(log_rr) else math.nan,
                "risk_ratio_ci95_low": math.exp(log_rr - 1.96 * log_se)
                if np.isfinite(log_rr) and np.isfinite(log_se)
                else math.nan,
                "risk_ratio_ci95_high": math.exp(log_rr + 1.96 * log_se)
                if np.isfinite(log_rr) and np.isfinite(log_se)
                else math.nan,
                "log_risk_ratio": log_rr,
                "log_risk_ratio_se": log_se,
                "continuity_correction_used": corrected,
                "direction_positive": bool(n1 and n0 and a / n1 > c / n0),
                "analysis_status": ADVANCED_ANALYSIS_STATUS,
            }
            site_rows.append(row)
            local_site_rows.append(row)

        estimates = pd.DataFrame(local_site_rows)
        if estimates.empty:
            estimates = pd.DataFrame(
                columns=[
                    "status",
                    "hospital_id",
                    "log_risk_ratio",
                    "log_risk_ratio_se",
                    "direction_positive",
                ]
            )
        else:
            estimates = estimates.loc[estimates["status"].eq("estimated")].copy()
        meta = _paule_mandel_hksj(
            pd.to_numeric(estimates["log_risk_ratio"], errors="coerce").to_numpy(),
            np.square(
                pd.to_numeric(
                    estimates["log_risk_ratio_se"], errors="coerce"
                ).to_numpy()
            ),
        )
        leave_one_out: list[float] = []
        if len(estimates) >= 4:
            for hospital in estimates["hospital_id"]:
                subset = estimates.loc[~estimates["hospital_id"].eq(hospital)]
                loo = _paule_mandel_hksj(
                    subset["log_risk_ratio"].to_numpy(float),
                    np.square(subset["log_risk_ratio_se"].to_numpy(float)),
                )
                if np.isfinite(loo["mean"]):
                    leave_one_out.append(math.exp(loo["mean"]))
        status = (
            "estimated"
            if len(estimates) >= 3 and np.isfinite(meta["mean"])
            else "underpowered_multicenter"
        )
        summary_rows.append(
            {
                "dataset": str(dataset),
                "endpoint": str(endpoint),
                "endpoint_family": str(family),
                "lag_window": str(lag_window),
                "status": status,
                "hospitals_with_observed_outcomes": int(
                    eligible["hospital_id"].nunique()
                ),
                "hospitals_estimable": int(len(estimates)),
                "patients_observed": int(len(eligible)),
                "events_observed": int(eligible["event"].sum()),
                "direction_positive_hospitals": int(
                    estimates["direction_positive"].sum()
                )
                if len(estimates)
                else 0,
                "direction_positive_fraction": float(
                    estimates["direction_positive"].mean()
                )
                if len(estimates)
                else math.nan,
                "random_effect_risk_ratio": math.exp(meta["mean"])
                if np.isfinite(meta["mean"])
                else math.nan,
                "random_effect_ci95_low_hksj": math.exp(meta["low"])
                if np.isfinite(meta["low"])
                else math.nan,
                "random_effect_ci95_high_hksj": math.exp(meta["high"])
                if np.isfinite(meta["high"])
                else math.nan,
                "random_effect_p_value_hksj": meta["p"],
                "tau_squared_paule_mandel": meta["tau_squared"],
                "i_squared_percent": meta["i_squared_percent"],
                "cochran_q": meta["cochran_q"],
                "leave_one_hospital_out_rr_min": min(leave_one_out)
                if leave_one_out
                else math.nan,
                "leave_one_hospital_out_rr_max": max(leave_one_out)
                if leave_one_out
                else math.nan,
                "claim_ready_multicenter": bool(
                    len(estimates) >= 5
                    and int(eligible["event"].sum()) >= 40
                ),
                "meta_method": "paule_mandel_tau2_modified_hartung_knapp_ci",
                "analysis_status": ADVANCED_ANALYSIS_STATUS,
            }
        )
    summaries = pd.DataFrame(summary_rows)
    if not summaries.empty:
        summaries["endpoint_q_value"] = np.nan
        summaries["cross_endpoint_q_value"] = np.nan
        for _, indexes in summaries.groupby(["dataset", "endpoint"]).groups.items():
            summaries.loc[indexes, "endpoint_q_value"] = _bh_adjust(
                summaries.loc[indexes, "random_effect_p_value_hksj"]
            )
        for _, indexes in summaries.groupby("dataset").groups.items():
            summaries.loc[indexes, "cross_endpoint_q_value"] = _bh_adjust(
                summaries.loc[indexes, "random_effect_p_value_hksj"]
            )
        summaries["multicenter_cross_endpoint_fdr_positive"] = (
            summaries["status"].eq("estimated")
            & summaries["claim_ready_multicenter"].eq(True)  # noqa: E712
            & pd.to_numeric(
                summaries["random_effect_ci95_low_hksj"], errors="coerce"
            ).gt(1)
            & pd.to_numeric(
                summaries["cross_endpoint_q_value"], errors="coerce"
            ).lt(0.05)
        )
    return {
        "advanced_episode_site_effects": pd.DataFrame(site_rows),
        "advanced_episode_multicenter_summary": summaries,
    }


def _e_value(risk_ratio: float) -> float:
    """Return the VanderWeele-Ding E-value for a positive risk ratio."""
    if not np.isfinite(risk_ratio) or risk_ratio <= 0:
        return math.nan
    strength = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return float(strength + math.sqrt(strength * (strength - 1.0)))


def _e_value_confidence_limit(
    risk_ratio: float,
    ci_low: float,
    ci_high: float,
) -> float:
    if not all(np.isfinite(value) for value in (risk_ratio, ci_low, ci_high)):
        return math.nan
    if risk_ratio >= 1.0:
        return _e_value(ci_low) if ci_low > 1.0 else 1.0
    return _e_value(ci_high) if 0.0 < ci_high < 1.0 else 1.0


def build_evalue_sensitivity(
    key_results: pd.DataFrame,
    overlap_effects: pd.DataFrame,
    aipw_effects: pd.DataFrame,
    multicenter_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify residual-confounding strength for each estimable focused RR."""
    specifications = (
        (
            "covariate_adjusted",
            key_results,
            "adjusted_status",
            ("estimated",),
            "adjusted_risk_ratio",
            "adjusted_ci95_low",
            "adjusted_ci95_high",
            "cross_endpoint_focused_adjusted_q_value",
            ("adjusted_claim_ready_epv10",),
        ),
        (
            "outcome_observation_weighted",
            key_results,
            "weighted_status",
            ("estimated", "not_needed_complete_observation"),
            "selection_weighted_risk_ratio",
            "selection_weighted_ci95_low",
            "selection_weighted_ci95_high",
            "weighted_cross_endpoint_q_value",
            ("outcome_claim_ready_epv10",),
        ),
        (
            "exposure_overlap_and_observation_weighted",
            overlap_effects,
            "status",
            ("estimated",),
            "overlap_weighted_risk_ratio",
            "overlap_weighted_ci95_low",
            "overlap_weighted_ci95_high",
            "cross_endpoint_q_value",
            (
                "claim_ready_min_10_events_and_non_events_per_arm",
                "claim_ready_weight_diagnostics",
            ),
        ),
        (
            "crossfit_one_step_aipw",
            aipw_effects,
            "one_step_aipw_status",
            ("estimated_bounded",),
            "aipw_risk_ratio",
            "aipw_risk_ratio_ci95_low",
            "aipw_risk_ratio_ci95_high",
            "one_step_aipw_cross_endpoint_q_value",
            (
                "claim_ready_min_10_events_and_non_events_per_arm",
                "claim_ready_weight_diagnostics",
            ),
        ),
        (
            "crossfit_targeted_maximum_likelihood",
            aipw_effects,
            "status",
            ("estimated",),
            "tmle_risk_ratio",
            "tmle_risk_ratio_ci95_low",
            "tmle_risk_ratio_ci95_high",
            "tmle_cross_endpoint_q_value",
            (
                "claim_ready_min_10_events_and_non_events_per_arm",
                "claim_ready_weight_diagnostics",
            ),
        ),
        (
            "multicenter_random_effect",
            multicenter_summary,
            "status",
            ("estimated",),
            "random_effect_risk_ratio",
            "random_effect_ci95_low_hksj",
            "random_effect_ci95_high_hksj",
            "cross_endpoint_q_value",
            ("claim_ready_multicenter",),
        ),
    )
    rows: list[dict[str, Any]] = []
    for (
        estimator,
        frame,
        status_column,
        valid_statuses,
        estimate_column,
        low_column,
        high_column,
        q_column,
        readiness_columns,
    ) in specifications:
        required = {
            "dataset",
            "endpoint",
            "lag_window",
            status_column,
            estimate_column,
            low_column,
            high_column,
        }
        if frame.empty or not required.issubset(frame.columns):
            continue
        for row in frame.loc[frame[status_column].isin(valid_statuses)].to_dict(
            "records"
        ):
            rr = float(pd.to_numeric(row.get(estimate_column), errors="coerce"))
            low = float(pd.to_numeric(row.get(low_column), errors="coerce"))
            high = float(pd.to_numeric(row.get(high_column), errors="coerce"))
            if not (np.isfinite(rr) and np.isfinite(low) and np.isfinite(high) and rr > 0):
                continue
            support_ready = all(
                _is_true(row.get(column, False)) for column in readiness_columns
            )
            targeting_boundary = bool(
                estimator == "crossfit_targeted_maximum_likelihood"
                and _is_true(row.get("tmle_targeting_boundary_hit", False))
            )
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "endpoint": str(row["endpoint"]),
                    "endpoint_family": str(row.get("endpoint_family", "")),
                    "lag_window": str(row["lag_window"]),
                    "estimator": estimator,
                    "risk_ratio": rr,
                    "risk_ratio_ci95_low": low,
                    "risk_ratio_ci95_high": high,
                    "cross_endpoint_q_value": pd.to_numeric(
                        row.get(q_column), errors="coerce"
                    ),
                    "e_value_point_estimate": _e_value(rr),
                    "e_value_confidence_limit": _e_value_confidence_limit(
                        rr, low, high
                    ),
                    "confidence_interval_excludes_null": bool(
                        low > 1.0 or high < 1.0
                    ),
                    "claim_ready_information_and_weight_support": bool(
                        support_ready and not targeting_boundary
                    ),
                    "positive_claim_ready": bool(
                        support_ready
                        and not targeting_boundary
                        and rr > 1.0
                        and low > 1.0
                    ),
                    "interpretation": (
                        "minimum_risk_ratio_association_an_unmeasured_confounder_"
                        "would_need_with_exposure_and_outcome_to_explain_away_result"
                    ),
                    "analysis_status": ADVANCED_ANALYSIS_STATUS,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "endpoint", "lag_window", "estimator"]
    ).reset_index(drop=True) if rows else pd.DataFrame()


def build_advanced_key_results(
    key_results: pd.DataFrame,
    overlap_effects: pd.DataFrame,
    aipw_effects: pd.DataFrame,
    continuous_aipw_effects: pd.DataFrame,
    multicenter_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Join estimators without replacing or weakening the original evidence gates."""
    keys = ["dataset", "endpoint", "lag_window"]
    result = key_results.copy().rename(
        columns={
            "evidence_label": "prior_evidence_label",
            "random_effect_risk_ratio": "cross_dataset_random_effect_risk_ratio",
            "random_effect_ci95_low": "cross_dataset_random_effect_ci95_low",
            "random_effect_ci95_high": "cross_dataset_random_effect_ci95_high",
            "i_squared_percent": "cross_dataset_i_squared_percent",
        }
    )
    if result.empty:
        return result
    overlap_columns = [
        *keys,
        "status",
        "overlap_weighted_risk_ratio",
        "overlap_weighted_ci95_low",
        "overlap_weighted_ci95_high",
        "overlap_weighted_p_value",
        "cross_endpoint_q_value",
        "overlap_claim_ready_positive",
        "overlap_cross_endpoint_fdr_positive",
        "effective_sample_size_exposed",
        "effective_sample_size_unexposed",
        "exposure_propensity_clipped_fraction",
        "observation_probability_lower_clipped_fraction",
        "claim_ready_weight_diagnostics",
    ]
    result = result.merge(
        overlap_effects.reindex(columns=overlap_columns).rename(
            columns={
                "status": "overlap_status",
                "cross_endpoint_q_value": "overlap_cross_endpoint_q_value",
                "effective_sample_size_exposed": "overlap_ess_exposed",
                "effective_sample_size_unexposed": "overlap_ess_unexposed",
                "exposure_propensity_clipped_fraction": (
                    "overlap_exposure_propensity_clipped_fraction"
                ),
                "observation_probability_lower_clipped_fraction": (
                    "overlap_observation_probability_lower_clipped_fraction"
                ),
                "claim_ready_weight_diagnostics": (
                    "overlap_claim_ready_weight_diagnostics"
                ),
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    aipw_columns = [
        *keys,
        "status",
        "one_step_aipw_status",
        "aipw_risk_exposed",
        "aipw_risk_unexposed",
        "aipw_risk_ratio",
        "aipw_risk_ratio_ci95_low",
        "aipw_risk_ratio_ci95_high",
        "aipw_risk_ratio_p_value",
        "one_step_aipw_cross_endpoint_q_value",
        "aipw_claim_ready_positive",
        "aipw_cross_endpoint_fdr_positive",
        "tmle_risk_exposed",
        "tmle_risk_unexposed",
        "tmle_risk_ratio",
        "tmle_risk_ratio_ci95_low",
        "tmle_risk_ratio_ci95_high",
        "tmle_risk_ratio_p_value",
        "tmle_cross_endpoint_q_value",
        "tmle_claim_ready_positive",
        "tmle_cross_endpoint_fdr_positive",
        "tmle_targeting_boundary_hit",
        "aipw_effective_sample_size_exposed",
        "aipw_effective_sample_size_unexposed",
        "exposure_propensity_clipped_fraction",
        "observation_probability_lower_clipped_fraction",
        "claim_ready_weight_diagnostics",
    ]
    result = result.merge(
        aipw_effects.reindex(columns=aipw_columns).rename(
            columns={
                "status": "doubly_robust_status",
                "exposure_propensity_clipped_fraction": (
                    "dr_exposure_propensity_clipped_fraction"
                ),
                "observation_probability_lower_clipped_fraction": (
                    "dr_observation_probability_lower_clipped_fraction"
                ),
                "claim_ready_weight_diagnostics": (
                    "dr_claim_ready_weight_diagnostics"
                ),
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    continuous_columns = [
        *keys,
        "status",
        "change_scale",
        "aipw_mean_change_exposed",
        "aipw_mean_change_unexposed",
        "aipw_mean_change_difference",
        "aipw_mean_change_difference_ci95_low",
        "aipw_mean_change_difference_ci95_high",
        "aipw_mean_change_p_value",
        "aipw_standardized_mean_difference",
        "aipw_standardized_ci95_low",
        "aipw_standardized_ci95_high",
        "cross_endpoint_q_value",
        "continuous_claim_ready_positive",
        "continuous_cross_endpoint_fdr_positive",
        "exposure_propensity_clipped_fraction",
        "observation_probability_lower_clipped_fraction",
        "claim_ready_weight_diagnostics",
    ]
    result = result.merge(
        continuous_aipw_effects.reindex(columns=continuous_columns).rename(
            columns={
                "status": "continuous_aipw_status",
                "cross_endpoint_q_value": "continuous_aipw_cross_endpoint_q_value",
                "exposure_propensity_clipped_fraction": (
                    "continuous_exposure_propensity_clipped_fraction"
                ),
                "observation_probability_lower_clipped_fraction": (
                    "continuous_observation_probability_lower_clipped_fraction"
                ),
                "claim_ready_weight_diagnostics": (
                    "continuous_claim_ready_weight_diagnostics"
                ),
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    multicenter_columns = [
        *keys,
        "status",
        "hospitals_estimable",
        "direction_positive_fraction",
        "random_effect_risk_ratio",
        "random_effect_ci95_low_hksj",
        "random_effect_ci95_high_hksj",
        "random_effect_p_value_hksj",
        "cross_endpoint_q_value",
        "i_squared_percent",
        "leave_one_hospital_out_rr_min",
        "leave_one_hospital_out_rr_max",
        "claim_ready_multicenter",
        "multicenter_cross_endpoint_fdr_positive",
    ]
    result = result.merge(
        multicenter_summary.reindex(columns=multicenter_columns).rename(
            columns={
                "status": "multicenter_status",
                "cross_endpoint_q_value": "multicenter_cross_endpoint_q_value",
                "random_effect_risk_ratio": "multicenter_random_effect_risk_ratio",
                "random_effect_ci95_low_hksj": "multicenter_random_effect_ci95_low_hksj",
                "random_effect_ci95_high_hksj": "multicenter_random_effect_ci95_high_hksj",
                "random_effect_p_value_hksj": "multicenter_random_effect_p_value_hksj",
                "i_squared_percent": "multicenter_i_squared_percent",
            }
        ),
        on=keys,
        how="left",
        validate="one_to_one",
    )
    balance_columns = [
        "dataset",
        "overlap_effective_sample_size",
        "max_absolute_smd_unweighted",
        "max_absolute_smd_overlap_weighted",
        "covariates_above_0_10_unweighted",
        "covariates_above_0_10_overlap_weighted",
        "outside_0_05_0_95_count",
    ]
    result = result.merge(
        overlap_summary.reindex(columns=balance_columns),
        on="dataset",
        how="left",
        validate="many_to_one",
    )
    boolean_columns = (
        "overlap_claim_ready_positive",
        "overlap_cross_endpoint_fdr_positive",
        "aipw_claim_ready_positive",
        "aipw_cross_endpoint_fdr_positive",
        "tmle_claim_ready_positive",
        "tmle_cross_endpoint_fdr_positive",
        "continuous_claim_ready_positive",
        "continuous_cross_endpoint_fdr_positive",
        "overlap_claim_ready_weight_diagnostics",
        "dr_claim_ready_weight_diagnostics",
        "continuous_claim_ready_weight_diagnostics",
        "tmle_targeting_boundary_hit",
        "claim_ready_multicenter",
        "multicenter_cross_endpoint_fdr_positive",
        "unadjusted_cross_fdr_positive",
        "adjusted_cross_fdr_positive",
        "weighted_cross_fdr_positive",
    )
    for column in boolean_columns:
        if column not in result:
            result[column] = False
        result[column] = result[column].fillna(False).astype(bool)
    result["advanced_estimators_estimable"] = (
        result["overlap_status"].eq("estimated").astype(int)
        + result["doubly_robust_status"].eq("estimated").astype(int)
        + result["multicenter_status"].eq("estimated").astype(int)
    )
    result["advanced_positive_ci_estimators"] = (
        result["overlap_claim_ready_positive"].astype(int)
        + result["tmle_claim_ready_positive"].astype(int)
        + (
            result["claim_ready_multicenter"]
            & pd.to_numeric(
                result["multicenter_random_effect_ci95_low_hksj"], errors="coerce"
            ).gt(1)
        ).astype(int)
    )
    result["advanced_cross_fdr_estimators"] = (
        result["overlap_cross_endpoint_fdr_positive"].astype(int)
        + result["tmle_cross_endpoint_fdr_positive"].astype(int)
        + result["multicenter_cross_endpoint_fdr_positive"].astype(int)
    )
    result["advanced_cross_fdr_methods"] = result.apply(
        lambda row: ";".join(
            method
            for method, supported in (
                (
                    "overlap_weighted",
                    bool(row["overlap_cross_endpoint_fdr_positive"]),
                ),
                (
                    "targeted_double_robust",
                    bool(row["tmle_cross_endpoint_fdr_positive"]),
                ),
                (
                    "continuous_double_robust",
                    bool(row["continuous_cross_endpoint_fdr_positive"]),
                ),
                (
                    "multicenter_unadjusted_random_effect",
                    bool(row["multicenter_cross_endpoint_fdr_positive"]),
                ),
            )
            if supported
        ),
        axis=1,
    )

    def classify(row: pd.Series) -> str:
        if str(row.get("status")) == "unavailable":
            return "endpoint_unavailable"
        prior_robust = all(
            bool(row.get(column, False))
            for column in (
                "unadjusted_cross_fdr_positive",
                "adjusted_cross_fdr_positive",
                "weighted_cross_fdr_positive",
            )
        )
        if bool(row["multicenter_cross_endpoint_fdr_positive"]) and bool(
            row["tmle_cross_endpoint_fdr_positive"]
        ) and bool(row["overlap_cross_endpoint_fdr_positive"]):
            return "multicenter_overlap_and_targeted_dr_cross_fdr_support"
        if bool(row["multicenter_cross_endpoint_fdr_positive"]) and bool(
            row["tmle_cross_endpoint_fdr_positive"]
        ):
            return "multicenter_and_targeted_dr_cross_fdr_support"
        if bool(row["multicenter_cross_endpoint_fdr_positive"]) and bool(
            row["overlap_cross_endpoint_fdr_positive"]
        ):
            return "multicenter_and_overlap_cross_fdr_support"
        if bool(row["tmle_cross_endpoint_fdr_positive"]) and bool(
            row["overlap_cross_endpoint_fdr_positive"]
        ):
            return "overlap_and_targeted_dr_cross_fdr_support"
        if bool(row["tmle_cross_endpoint_fdr_positive"]):
            return "targeted_double_robust_cross_endpoint_fdr_support"
        if bool(row["overlap_cross_endpoint_fdr_positive"]):
            return "overlap_weighted_cross_endpoint_fdr_support"
        if bool(row["continuous_cross_endpoint_fdr_positive"]):
            return "graded_worsening_double_robust_cross_endpoint_fdr_support"
        if bool(row["multicenter_cross_endpoint_fdr_positive"]):
            return (
                "prior_robust_and_multicenter_cross_endpoint_fdr_signal"
                if prior_robust
                else "multicenter_unadjusted_cross_endpoint_fdr_signal"
            )
        if prior_robust:
            return "prior_robust_signal_not_reinforced_by_advanced_estimators"
        if int(row["advanced_positive_ci_estimators"]) > 0:
            return "advanced_ci_support_but_cross_endpoint_fdr_sensitive"
        advanced_rrs = pd.to_numeric(
            pd.Series(
                [
                    row.get("overlap_weighted_risk_ratio"),
                    row.get("tmle_risk_ratio"),
                    row.get("multicenter_random_effect_risk_ratio"),
                ]
            ),
            errors="coerce",
        ).dropna()
        if len(advanced_rrs) and advanced_rrs.gt(1).any():
            return "directionally_positive_advanced_but_inconclusive"
        return str(row.get("prior_evidence_label", "no_positive_association"))

    result["advanced_evidence_label"] = result.apply(classify, axis=1)
    result["advanced_claim_scope"] = (
        "observational_episode_association_under_measured_confounding_and_"
        "missing_at_random_assumptions_not_causal"
    )
    result["analysis_status"] = ADVANCED_ANALYSIS_STATUS
    return result.sort_values(keys).reset_index(drop=True)


def _advanced_design_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exposure_definition": PRIMARY_DEFINITION,
                "signal_resolution": PRIMARY_RESOLUTION,
                "anchor_window": "first_4_hours_after_icu_admission",
                "preanchor_rule": "offset_minutes>=0_and_strictly_less_than_anchor",
                "exposure_estimand": "episode_vs_time_aligned_no_episode_anchor",
                "overlap_target": "average_treatment_effect_in_overlap_population",
                "binary_dr_target": "bounded_cross_validated_tmle_marginal_risk_in_anchor_risk_set",
                "one_step_aipw_role": "diagnostic_retained_even_when_out_of_bounds",
                "continuous_dr_target": "crossfit_aipw_marginal_mean_worsening_difference",
                "crossfit_folds": N_FOLDS,
                "crossfit_group": "person_id_with_stay_fallback",
                "propensity_clip": f"{PROPENSITY_CLIP[0]}_{PROPENSITY_CLIP[1]}",
                "observation_clip": f"{OBSERVATION_CLIP[0]}_{OBSERVATION_CLIP[1]}",
                "max_propensity_clip_fraction_for_claim": (
                    MAX_PROPENSITY_CLIP_FRACTION
                ),
                "max_observation_low_clip_fraction_for_claim": (
                    MAX_OBSERVATION_LOW_CLIP_FRACTION
                ),
                "min_effective_sample_size_per_arm_for_claim": (
                    MIN_EFFECTIVE_SAMPLE_SIZE_PER_ARM
                ),
                "multiplicity": "BH_within_endpoint_and_across_all_focused_endpoints_per_dataset",
                "multicenter_method": "hospital_specific_RR_then_Paule_Mandel_HKSJ",
                "covariates": ",".join(
                    [
                        *BASE_NUMERIC_COVARIATES,
                        *CONCURRENT_NUMERIC_COVARIATES,
                        *BASE_BINARY_COVARIATES,
                        *BASE_CATEGORICAL_COVARIATES,
                    ]
                ),
                "concurrent_spo2_use": (
                    "outcome_and_observation_models_only_not_propensity_or_balance"
                ),
                "analysis_status": ADVANCED_ANALYSIS_STATUS,
                "outcome_independent_specification": True,
            }
        ]
    )


def run_advanced_episode_inference(
    events_df: pd.DataFrame,
    focused_records: pd.DataFrame,
    key_results: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run advanced inference entirely from cached events and episode records."""
    preanchor = build_episode_preanchor_covariates(
        events_df, focused_records, cohort_df
    )
    overlap_outputs = build_episode_overlap_weights(preanchor)
    overlap_effects = build_overlap_weighted_effects(
        focused_records,
        preanchor,
        overlap_outputs["advanced_episode_anchor_weights"],
    )
    aipw = build_crossfit_aipw_effects(focused_records, preanchor)
    continuous_aipw = build_crossfit_continuous_aipw_effects(
        focused_records, preanchor
    )
    multicenter_outputs = build_multicenter_robustness(focused_records, cohort_df)
    evalues = build_evalue_sensitivity(
        key_results,
        overlap_effects,
        aipw,
        multicenter_outputs["advanced_episode_multicenter_summary"],
    )
    advanced_key = build_advanced_key_results(
        key_results,
        overlap_effects,
        aipw,
        continuous_aipw,
        multicenter_outputs["advanced_episode_multicenter_summary"],
        overlap_outputs["advanced_episode_overlap_summary"],
    )
    return {
        "advanced_episode_preanchor_covariates": preanchor,
        **overlap_outputs,
        "advanced_episode_overlap_weighted_effects": overlap_effects,
        "advanced_episode_aipw_effects": aipw,
        "advanced_episode_continuous_aipw_effects": continuous_aipw,
        **multicenter_outputs,
        "advanced_episode_evalues": evalues,
        "advanced_episode_key_results": advanced_key,
        "advanced_episode_design": _advanced_design_table(),
    }


__all__ = [
    "ADVANCED_ANALYSIS_STATUS",
    "build_advanced_key_results",
    "build_crossfit_aipw_effects",
    "build_crossfit_continuous_aipw_effects",
    "build_episode_overlap_weights",
    "build_episode_preanchor_covariates",
    "build_evalue_sensitivity",
    "build_multicenter_robustness",
    "build_overlap_weighted_effects",
    "run_advanced_episode_inference",
]
