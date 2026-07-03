"""Locked (frozen) comparator model inference and validation.

Implements the full comparator validation pipeline: data loading,
complete-case eligibility filtering, deterministic train/validation
splitting, mean/scale preprocessing, logistic regression fitting,
and metric computation for internal and external evaluation.

All formulas and protocols are extracted verbatim from the
PhysioGraph notebooks to ensure exact numerical parity with the
ground-truth outputs in ``tests/ground_truth/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physiograph.validation.evaluate import (
    binary_classification_metrics,
    calibration_table_rows,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VALIDATION_FRACTION = 0.25
DEFAULT_REGULARIZATION_GRID = (0.0, 0.01, 0.1, 1.0)

PRIMARY_FEATURE_COLUMNS = [
    "dataset", "stay_id", "age", "is_male", "cohort_hf_flag", "shock_icd_flag",
    "baseline_lactate", "baseline_hr", "baseline_sbp", "baseline_map",
    "baseline_ph", "baseline_creatinine", "baseline_bilirubin_total",
    "baseline_spo2", "baseline_resp_rate", "baseline_temp",
    "tachycardia_flag", "hypotension_flag", "severe_hypotension_flag",
    "acidemia_flag", "severe_acidemia_flag", "modifier_burden",
    "lactate_delta", "lactate_slope_per_hr", "lactate_clearance_4h_pct",
    "persistent_lactate_flag", "renal_hypoperfusion_flag",
    "hepatic_hypoperfusion_flag", "lactate_ge_2_flag",
    "lactate_ge_3_1_flag", "lactate_ge_5_flag",
    "map_below_65_fraction", "sbp_below_90_fraction",
    "hr_above_100_fraction", "ph_below_7_25_fraction",
    "lactate_map_ratio", "lactate_sbp_ratio",
    "lactate_acidemia_interaction", "lactate_hypotension_interaction",
    "lactate_tachycardia_interaction", "occult_hypoperfusion_flag",
    "perfusion_burden_score", "baseline_lactate_bin",
    "baseline_lactate_focus_bin", "hr_band", "hr_band_fine",
    "lactate_scai_modifier", "scai_stage", "scai_stage_num",
    "scai_stage_collapsed",
]

LABEL_REQUIRED_COLUMNS = [
    "dataset", "stay_id", "target", "pressor_24h_flag", "mcs_24h_flag",
    "escalation_24h_flag", "renal_injury_24h_flag",
    "hypoperfusion_24h_flag", "hepatic_injury_24h_flag",
    "end_organ_24h_flag", "mortality_24h_flag",
    "shock_progression_24h_flag", "landmark_lactate",
    "post_landmark_lactate_last", "lactate_clearance_24h_pct",
    "complete_lactate_clearance_24h_flag", "clearance_ge_64_24h_flag",
    "cohort_hf_flag", "shock_icd_flag", "age", "is_male",
]

ANALYSIS_ONLY_CONTEXT_COLUMNS = [
    "landmark_lactate", "post_landmark_lactate_last",
    "lactate_clearance_24h_pct", "complete_lactate_clearance_24h_flag",
    "clearance_ge_64_24h_flag", "mortality_24h_flag",
]

OUTCOME_FLAG_COLUMNS = [
    "pressor_24h_flag", "mcs_24h_flag", "escalation_24h_flag",
    "renal_injury_24h_flag", "hypoperfusion_24h_flag",
    "hepatic_injury_24h_flag", "end_organ_24h_flag",
    "shock_progression_24h_flag", "target",
]

FORBIDDEN_FEATURE_COLUMNS = set(ANALYSIS_ONLY_CONTEXT_COLUMNS + OUTCOME_FLAG_COLUMNS)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparatorSpec:
    """Specification for a comparator model.

    Parameters
    ----------
    name:
        Unique identifier for the comparator.
    description:
        Human-readable description.
    numeric_features:
        Tuple of numeric feature column names.
    categorical_features:
        Tuple of categorical feature column names.
    """

    name: str
    description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumericTransform:
    """Mean/scale standardization parameters for a numeric feature."""

    mean: float
    scale: float


@dataclass(frozen=True)
class CategoricalTransform:
    """One-hot encoding parameters for a categorical feature."""

    categories: tuple[str, ...]
    fill_value: str
    reference_category: str


@dataclass(frozen=True)
class Preprocessor:
    """Fitted preprocessing transforms for numeric and categorical features."""

    numeric: dict[str, NumericTransform]
    categorical: dict[str, CategoricalTransform]


@dataclass(frozen=True)
class LogisticModel:
    """Fitted logistic regression model with optimization metadata."""

    coefficients: np.ndarray
    intercept: float
    l2_penalty: float
    iterations: int
    converged: bool
    optimization_trace: tuple[float, ...]


# ---------------------------------------------------------------------------
# Comparator specifications
# ---------------------------------------------------------------------------

COMPARATOR_SPECS: tuple[ComparatorSpec, ...] = (
    ComparatorSpec(
        name="lactate_only",
        description="Baseline landmark lactate only.",
        numeric_features=("baseline_lactate",),
    ),
    ComparatorSpec(
        name="lactate_hemodynamics",
        description="Baseline lactate with hemodynamic context.",
        numeric_features=(
            "baseline_lactate", "baseline_hr", "baseline_sbp", "baseline_map",
            "tachycardia_flag", "hypotension_flag", "severe_hypotension_flag",
            "hr_above_100_fraction", "sbp_below_90_fraction",
            "map_below_65_fraction", "lactate_map_ratio",
            "lactate_sbp_ratio", "lactate_hypotension_interaction",
            "lactate_tachycardia_interaction",
        ),
    ),
    ComparatorSpec(
        name="lactate_end_organ",
        description="Baseline lactate with acidemia and end-organ injury markers.",
        numeric_features=(
            "baseline_lactate", "baseline_ph", "baseline_creatinine",
            "baseline_bilirubin_total", "acidemia_flag",
            "severe_acidemia_flag", "renal_hypoperfusion_flag",
            "hepatic_hypoperfusion_flag", "perfusion_burden_score",
            "lactate_acidemia_interaction",
        ),
    ),
    ComparatorSpec(
        name="scai_stage_model",
        description="SCAI-like stage comparator using harmonized stage categories.",
        numeric_features=("baseline_lactate",),
        categorical_features=("scai_stage",),
    ),
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Clips inputs to [-35, 35] to prevent overflow in ``exp``.
    """
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def logistic_objective(
    design: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
    *,
    l2_penalty: float,
) -> float:
    """Compute penalized logistic regression negative log-likelihood.

    The intercept (``coefficients[0]``) is exempt from L2 penalty.
    """
    logits = design @ coefficients
    probabilities = np.clip(sigmoid(logits), 1e-8, 1.0 - 1e-8)
    negative_log_likelihood = -np.mean(
        targets * np.log(probabilities)
        + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    penalty = 0.5 * float(l2_penalty) * float(np.sum(coefficients[1:] ** 2))
    return float(negative_log_likelihood + penalty)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def fit_preprocessor(frame: pd.DataFrame, spec: ComparatorSpec) -> Preprocessor:
    """Fit mean/scale transforms for numeric features and category maps
    for categorical features from *frame*.

    Raises
    ------
    ValueError
        If any numeric feature still contains missing values.
    """
    numeric: dict[str, NumericTransform] = {}
    categorical: dict[str, CategoricalTransform] = {}

    for feature in spec.numeric_features:
        series = (
            pd.to_numeric(frame[feature], errors="coerce")
            if feature in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        if series.isna().any():
            raise ValueError(
                f"Numeric feature '{feature}' still contains missing values "
                f"after eligibility filtering."
            )
        filled = series.to_numpy(dtype=float)
        mean = float(filled.mean()) if len(filled) else 0.0
        scale = float(filled.std()) if len(filled) else 1.0
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        numeric[feature] = NumericTransform(mean=mean, scale=scale)

    for feature in spec.categorical_features:
        raw = (
            frame[feature]
            if feature in frame.columns
            else pd.Series(index=frame.index, dtype="object")
        )
        filled = raw.astype("object").where(raw.notna(), "missing").astype(str)
        categories = tuple(sorted(pd.unique(filled).tolist()))
        if not categories:
            categories = ("missing",)
        categorical[feature] = CategoricalTransform(
            categories=categories,
            fill_value="missing",
            reference_category=categories[0],
        )

    return Preprocessor(numeric=numeric, categorical=categorical)


def transform_features(
    frame: pd.DataFrame,
    preprocessor: Preprocessor,
) -> tuple[np.ndarray, list[str]]:
    """Apply fitted preprocessing transforms to *frame*.

    Numeric features are standardized (mean-centered, scale-divided).
    Categorical features are one-hot encoded with the reference
    category dropped.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Design matrix and corresponding expanded feature names.

    Raises
    ------
    ValueError
        If any numeric feature contains missing values during scoring.
    """
    columns: list[np.ndarray] = []
    feature_names: list[str] = []

    for feature, transform in preprocessor.numeric.items():
        series = (
            pd.to_numeric(frame[feature], errors="coerce")
            if feature in frame.columns
            else pd.Series(np.nan, index=frame.index)
        )
        if series.isna().any():
            raise ValueError(
                f"Numeric feature '{feature}' contains missing values during scoring."
            )
        filled = series.to_numpy(dtype=float)
        scaled = (filled - transform.mean) / transform.scale
        columns.append(scaled)
        feature_names.append(feature)

    for feature, transform in preprocessor.categorical.items():
        raw = (
            frame[feature]
            if feature in frame.columns
            else pd.Series(index=frame.index, dtype="object")
        )
        filled = (
            raw.astype("object")
            .where(raw.notna(), transform.fill_value)
            .astype(str)
        )
        if transform.fill_value in transform.categories:
            filled = filled.where(filled.isin(transform.categories), transform.fill_value)
        else:
            filled = filled.where(
                filled.isin(transform.categories), transform.reference_category
            )
        for category in transform.categories:
            if category == transform.reference_category:
                continue
            columns.append((filled == category).to_numpy(dtype=float))
            feature_names.append(f"{feature}__{category}")

    if not columns:
        return np.empty((len(frame), 0), dtype=float), feature_names
    return np.column_stack(columns).astype(float), feature_names


# ---------------------------------------------------------------------------
# Model fitting and prediction
# ---------------------------------------------------------------------------

def fit_logistic_regression(
    x_matrix: np.ndarray,
    y_vector: np.ndarray,
    *,
    l2_penalty: float,
    max_iter: int = 200,
    tolerance: float = 1e-8,
) -> LogisticModel:
    """Fit a logistic regression model via Newton–Raphson with L2 penalty.

    Uses line-search backtracking and falls back to gradient descent
    when the Hessian is singular.  The intercept is initialized to the
    log-odds of the base rate and is exempt from L2 regularization.

    Parameters
    ----------
    x_matrix:
        Feature matrix (rows = samples, columns = features).
    y_vector:
        Binary target vector (0 or 1).
    l2_penalty:
        L2 regularization strength.
    max_iter:
        Maximum Newton–Raphson iterations (default 200).
    tolerance:
        Convergence tolerance on loss and step norm (default 1e-8).

    Returns
    -------
    LogisticModel
        Fitted model with coefficients, intercept, and optimization trace.

    Raises
    ------
    ValueError
        If inputs are empty, targets are not binary, or penalty is negative.
    """
    x_matrix = np.asarray(x_matrix, dtype=float)
    y_vector = np.asarray(y_vector, dtype=float)
    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("x_matrix and y_vector must have the same number of rows.")
    if x_matrix.shape[0] == 0:
        raise ValueError("Cannot fit logistic regression on an empty matrix.")
    if set(np.unique(y_vector)) != {0.0, 1.0}:
        raise ValueError(
            "Logistic regression requires binary outcomes containing both 0 and 1."
        )
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative.")

    design = np.column_stack([np.ones(x_matrix.shape[0], dtype=float), x_matrix])
    beta = np.zeros(design.shape[1], dtype=float)
    base_rate = np.clip(float(y_vector.mean()), 1e-6, 1.0 - 1e-6)
    beta[0] = np.log(base_rate / (1.0 - base_rate))

    regularization = np.zeros((design.shape[1], design.shape[1]), dtype=float)
    if design.shape[1] > 1:
        regularization[1:, 1:] = (
            np.eye(design.shape[1] - 1, dtype=float) * float(l2_penalty)
        )

    loss_trace: list[float] = []
    current_loss = logistic_objective(
        design, y_vector, beta, l2_penalty=l2_penalty
    )
    loss_trace.append(current_loss)
    converged = False

    for iteration in range(1, max_iter + 1):
        logits = design @ beta
        probabilities = sigmoid(logits)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
        gradient = (design.T @ (probabilities - y_vector)) / float(len(y_vector))
        gradient[1:] += float(l2_penalty) * beta[1:]

        hessian = (design.T * weights) @ design / float(len(y_vector))
        hessian += regularization

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        step_size = 1.0
        candidate_beta = beta - step_size * step
        candidate_loss = logistic_objective(
            design, y_vector, candidate_beta, l2_penalty=l2_penalty
        )
        while candidate_loss > current_loss and step_size > 1e-6:
            step_size *= 0.5
            candidate_beta = beta - step_size * step
            candidate_loss = logistic_objective(
                design, y_vector, candidate_beta, l2_penalty=l2_penalty
            )

        if candidate_loss > current_loss:
            fallback_step = 0.1 * gradient
            candidate_beta = beta - fallback_step
            candidate_loss = logistic_objective(
                design, y_vector, candidate_beta, l2_penalty=l2_penalty
            )
            step_norm = float(np.max(np.abs(fallback_step)))
        else:
            step_norm = float(np.max(np.abs(step_size * step)))

        beta = candidate_beta
        loss_trace.append(candidate_loss)
        if abs(current_loss - candidate_loss) <= tolerance or step_norm <= tolerance:
            converged = True
            current_loss = candidate_loss
            break
        current_loss = candidate_loss

    return LogisticModel(
        coefficients=beta[1:].copy(),
        intercept=float(beta[0]),
        l2_penalty=float(l2_penalty),
        iterations=iteration,
        converged=converged,
        optimization_trace=tuple(float(v) for v in loss_trace),
    )


def predict_probabilities(model: LogisticModel, x_matrix: np.ndarray) -> np.ndarray:
    """Compute predicted probabilities from a fitted logistic model.

    Parameters
    ----------
    model:
        Fitted :class:`LogisticModel`.
    x_matrix:
        Feature matrix (already preprocessed).

    Returns
    -------
    np.ndarray
        Predicted probabilities.
    """
    x_matrix = np.asarray(x_matrix, dtype=float)
    logits = x_matrix @ model.coefficients + model.intercept
    return sigmoid(logits)


def select_logistic_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    regularization_grid: tuple[float, ...],
) -> tuple[LogisticModel, list[dict[str, Any]]]:
    """Select the best logistic model from a grid of L2 penalties.

    Selects the model with the lowest Brier score on the validation
    set, breaking ties by highest AUPRC, then highest AUROC, then
    lowest penalty.

    Parameters
    ----------
    x_train:
        Training feature matrix.
    y_train:
        Training target vector.
    x_validation:
        Validation feature matrix.
    y_validation:
        Validation target vector.
    regularization_grid:
        Tuple of L2 penalty values to try.

    Returns
    -------
    tuple[LogisticModel, list[dict[str, Any]]]
        Selected model and list of candidate evaluation rows.

    Raises
    ------
    ValueError
        If *regularization_grid* is empty.
    """
    if not regularization_grid:
        raise ValueError("regularization_grid must contain at least one penalty value.")

    selected_model: LogisticModel | None = None
    selected_key: tuple[float, float, float, float] | None = None
    candidate_rows: list[dict[str, Any]] = []

    for penalty in regularization_grid:
        model = fit_logistic_regression(x_train, y_train, l2_penalty=float(penalty))
        validation_probabilities = predict_probabilities(model, x_validation)
        metrics = binary_classification_metrics(y_validation, validation_probabilities)
        candidate_row = {
            "l2_penalty": float(penalty),
            "iterations": int(model.iterations),
            "converged": bool(model.converged),
            **metrics,
        }
        candidate_rows.append(candidate_row)
        selection_key = (
            float(metrics["brier"]),
            -float(metrics["auprc"]),
            -float(metrics["auroc"]),
            float(penalty),
        )
        if selected_key is None or selection_key < selected_key:
            selected_key = selection_key
            selected_model = model

    if selected_model is None:
        raise RuntimeError("Logistic model selection failed to produce a candidate.")
    return selected_model, candidate_rows


# ---------------------------------------------------------------------------
# Eligibility and splitting
# ---------------------------------------------------------------------------

def select_complete_cases(
    frame: pd.DataFrame,
    spec: ComparatorSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter *frame* to complete cases for *spec*'s numeric features.

    Rows with any missing value in the spec's numeric features are
    excluded.  Returns the eligible DataFrame and an eligibility report.

    Raises
    ------
    ValueError
        If eligible rows do not contain both target classes.
    """
    numeric_missing = pd.DataFrame(index=frame.index)
    for feature in spec.numeric_features:
        if feature in frame.columns:
            numeric_missing[feature] = pd.to_numeric(
                frame[feature], errors="coerce"
            ).isna()
        else:
            numeric_missing[feature] = True
    missing_mask = (
        numeric_missing.any(axis=1)
        if not numeric_missing.empty
        else pd.Series(False, index=frame.index)
    )
    eligible = (
        frame.loc[~missing_mask]
        .copy()
        .sort_values(["dataset", "stay_id"])
        .reset_index(drop=True)
    )
    if set(eligible["target"].unique()) != {0, 1}:
        raise ValueError(
            f"Eligible rows for comparator '{spec.name}' must contain both "
            f"target classes after complete-case filtering."
        )
    excluded = (
        frame.loc[missing_mask]
        .copy()
        .sort_values(["dataset", "stay_id"])
        .reset_index(drop=True)
    )
    return eligible, {
        "raw_rows": int(len(frame)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(len(excluded)),
        "excluded_by_feature": {
            feature: int(numeric_missing[feature].sum())
            for feature in numeric_missing.columns
        },
    }


def deterministic_internal_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> dict[str, Any]:
    """Create a deterministic train/validation split stratified by target.

    Uses evenly-spaced position selection to ensure reproducibility
    across runs without relying on random seeds.

    Raises
    ------
    ValueError
        If *validation_fraction* is out of range or either class has < 2 rows.
    """
    if not 0 < validation_fraction < 0.5:
        raise ValueError(
            "validation_fraction must be greater than 0 and less than 0.5."
        )
    if set(frame["target"].unique()) != {0, 1}:
        raise ValueError(
            "Internal comparator training requires both target classes."
        )

    validation_keys: set[tuple[str, int]] = set()
    for target_value in (0, 1):
        class_rows = frame.loc[
            frame["target"] == target_value, ["dataset", "stay_id"]
        ].copy()
        class_rows = class_rows.sort_values(["dataset", "stay_id"]).reset_index(
            drop=True
        )
        if len(class_rows) < 2:
            raise ValueError(
                f"Target class {target_value} has fewer than 2 rows; "
                f"deterministic train/validation split is impossible."
            )
        validation_count = min(
            max(1, int(round(len(class_rows) * validation_fraction))),
            len(class_rows) - 1,
        )
        validation_positions = _select_validation_positions(
            len(class_rows), validation_count
        )
        for position in validation_positions:
            row = class_rows.iloc[int(position)]
            validation_keys.add((str(row["dataset"]), int(row["stay_id"])))

    keys = list(
        zip(frame["dataset"].astype(str), frame["stay_id"].astype(int))
    )
    validation_mask = np.array(
        [key in validation_keys for key in keys], dtype=bool
    )
    train_mask = ~validation_mask
    if not validation_mask.any() or not train_mask.any():
        raise ValueError(
            "Deterministic train/validation split produced an empty partition."
        )

    train_df = (
        frame.loc[train_mask, ["dataset", "stay_id", "target"]]
        .sort_values(["dataset", "stay_id"])
        .reset_index(drop=True)
    )
    validation_df = (
        frame.loc[validation_mask, ["dataset", "stay_id", "target"]]
        .sort_values(["dataset", "stay_id"])
        .reset_index(drop=True)
    )
    return {
        "validation_fraction": float(validation_fraction),
        "train": train_df[["dataset", "stay_id"]].to_dict(orient="records"),
        "validation": validation_df[["dataset", "stay_id"]].to_dict(
            orient="records"
        ),
        "class_balance": {
            "train_positive_rate": float(train_df["target"].mean()),
            "validation_positive_rate": float(validation_df["target"].mean()),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
        },
    }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_preprocessor(preprocessor: Preprocessor) -> dict[str, Any]:
    """Convert a :class:`Preprocessor` to a JSON-serializable dict."""
    return {
        "numeric": {
            feature: {"mean": transform.mean, "scale": transform.scale}
            for feature, transform in preprocessor.numeric.items()
        },
        "categorical": {
            feature: {
                "categories": list(transform.categories),
                "fill_value": transform.fill_value,
                "reference_category": transform.reference_category,
            }
            for feature, transform in preprocessor.categorical.items()
        },
    }


def _pythonize(value: Any) -> Any:
    """Recursively convert NumPy types to Python builtins for JSON serialization."""
    if isinstance(value, dict):
        return {str(key): _pythonize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_pythonize(item) for item in value]
    if isinstance(value, tuple):
        return [_pythonize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _subset_by_records(
    frame: pd.DataFrame, records: list[dict[str, Any]]
) -> pd.DataFrame:
    """Subset *frame* to rows matching (dataset, stay_id) pairs in *records*."""
    key_set = {
        (str(record["dataset"]), int(record["stay_id"])) for record in records
    }
    subset = frame.loc[
        [
            (str(dataset), int(stay_id)) in key_set
            for dataset, stay_id in zip(
                frame["dataset"].astype(str), frame["stay_id"].astype(int)
            )
        ]
    ].copy()
    return subset.sort_values(["dataset", "stay_id"]).reset_index(drop=True)


def _select_validation_positions(
    total_rows: int, validation_count: int
) -> list[int]:
    """Select evenly-spaced validation positions for deterministic splitting."""
    if validation_count <= 0 or validation_count >= total_rows:
        raise ValueError(
            "validation_count must be between 1 and total_rows - 1."
        )
    raw_positions = np.linspace(
        0, total_rows - 1, num=validation_count + 2
    )[1:-1]
    positions: list[int] = []
    for value in raw_positions:
        candidate = int(round(float(value)))
        if candidate >= total_rows:
            candidate = total_rows - 1
        while candidate in positions and candidate + 1 < total_rows:
            candidate += 1
        while candidate in positions and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in positions:
            positions.append(candidate)
    for candidate in range(total_rows):
        if len(positions) == validation_count:
            break
        if candidate not in positions:
            positions.append(candidate)
    return sorted(positions)


def _metrics_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    excluded_rows: int = 0,
) -> dict[str, Any]:
    """Build a metrics dict row with model/split metadata."""
    metrics = binary_classification_metrics(y_true, probabilities)
    return {
        "model": model_name,
        "split": split_name,
        "excluded_rows": int(excluded_rows),
        **metrics,
    }


def _prediction_frame(
    frame: pd.DataFrame,
    model_name: str,
    split_name: str,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Build a prediction DataFrame from model outputs."""
    return pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "dataset": frame["dataset"].astype(str).to_numpy(),
            "stay_id": frame["stay_id"].astype(int).to_numpy(),
            "target": frame["target"].astype(int).to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _ordered_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    extra = [c for c in df.columns if c not in required]
    return df[required + extra]


def _validate_features(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, PRIMARY_FEATURE_COLUMNS, "features_df")
    return _ordered_columns(df, PRIMARY_FEATURE_COLUMNS)


def _validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, LABEL_REQUIRED_COLUMNS, "labels_df")
    return _ordered_columns(df, LABEL_REQUIRED_COLUMNS)


def _assert_no_feature_leakage_columns(columns: list[str] | pd.Index) -> None:
    overlap = sorted(FORBIDDEN_FEATURE_COLUMNS & set(columns))
    if overlap:
        raise ValueError(
            f"Post-landmark or outcome columns leaked into primary features: {overlap}"
        )


def load_harmonized_model_frame(
    artifact_dir: str | Path,
    *,
    target_column: str = "target",
) -> pd.DataFrame:
    """Load and merge features and labels from an artifact directory.

    Reads ``features.csv`` and ``labels.csv`` from *artifact_dir*,
    validates column schemas, checks for feature leakage, merges on
    ``(dataset, stay_id)``, and returns a single DataFrame with a
    binary ``target`` column.

    Parameters
    ----------
    artifact_dir:
        Path to the directory containing ``features.csv`` and
        ``labels.csv``.
    target_column:
        Name of the target column in labels (default ``"target"``).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame sorted by ``(dataset, stay_id)``.

    Raises
    ------
    ValueError
        If keys mismatch, target column is missing, or target is not
        binary.
    """
    artifact_path = Path(artifact_dir)
    features_path = artifact_path / "features.csv"
    labels_path = artifact_path / "labels.csv"
    features_df = _validate_features(pd.read_csv(features_path))
    labels_df = _validate_labels(pd.read_csv(labels_path))
    _assert_no_feature_leakage_columns(features_df.columns)

    key_columns = ["dataset", "stay_id"]
    feature_keys = set(
        map(
            tuple,
            features_df[key_columns]
            .astype({"dataset": str, "stay_id": int})
            .to_records(index=False),
        )
    )
    label_keys = set(
        map(
            tuple,
            labels_df[key_columns]
            .astype({"dataset": str, "stay_id": int})
            .to_records(index=False),
        )
    )
    if feature_keys != label_keys:
        raise ValueError(
            "features.csv and labels.csv must contain the same "
            "dataset/stay_id keys for comparator modeling."
        )

    if target_column not in labels_df.columns:
        raise ValueError(
            f"labels.csv does not contain requested target column '{target_column}'."
        )
    target_df = labels_df[key_columns + [target_column]].copy().rename(
        columns={target_column: "target"}
    )
    merged = features_df.merge(target_df, on=key_columns, how="inner", validate="one_to_one")
    merged = merged.sort_values(key_columns).reset_index(drop=True)
    merged["target"] = merged["target"].astype(int)
    if set(merged["target"].unique()) - {0, 1}:
        raise ValueError(
            "Comparator modeling requires a binary 'target' column with values 0/1."
        )
    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_locked_comparator_validation(
    internal_artifact_dir: str | Path,
    external_artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    regularization_grid: tuple[float, ...] = DEFAULT_REGULARIZATION_GRID,
    target_column: str = "target",
) -> dict[str, str]:
    """Run the locked comparator validation pipeline.

    Loads harmonized feature/label data, trains four frozen comparator
    models on the internal dataset with a deterministic train/validation
    split, evaluates each on the external dataset, and writes all
    artifacts (metrics, predictions, calibration, model specs) to
    *output_dir*.

    Parameters
    ----------
    internal_artifact_dir:
        Path to the internal dataset artifact directory (containing
        ``features.csv`` and ``labels.csv``).
    external_artifact_dir:
        Path to the external dataset artifact directory.
    output_dir:
        Directory where output files are written.
    validation_fraction:
        Fraction of internal data used for validation (default 0.25).
    regularization_grid:
        Tuple of L2 penalty values for model selection.
    target_column:
        Name of the target column in labels (default ``"target"``).

    Returns
    -------
    dict[str, str]
        Mapping of artifact names to file paths.
    """
    internal_df = load_harmonized_model_frame(
        internal_artifact_dir, target_column=target_column
    )
    external_df = load_harmonized_model_frame(
        external_artifact_dir, target_column=target_column
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_artifacts: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    split_artifacts: dict[str, Any] = {}

    for spec in COMPARATOR_SPECS:
        eligible_internal_df, internal_eligibility = select_complete_cases(
            internal_df, spec
        )
        eligible_external_df, external_eligibility = select_complete_cases(
            external_df, spec
        )
        split_artifact = deterministic_internal_split(
            eligible_internal_df, validation_fraction=validation_fraction
        )
        split_artifacts[spec.name] = split_artifact

        train_df = _subset_by_records(
            eligible_internal_df, split_artifact["train"]
        )
        validation_df = _subset_by_records(
            eligible_internal_df, split_artifact["validation"]
        )
        preprocessor = fit_preprocessor(train_df, spec)
        x_train, expanded_feature_names = transform_features(train_df, preprocessor)
        x_validation, _ = transform_features(validation_df, preprocessor)
        x_external, _ = transform_features(eligible_external_df, preprocessor)

        y_train = train_df["target"].to_numpy(dtype=float)
        y_validation = validation_df["target"].to_numpy(dtype=float)
        y_external = eligible_external_df["target"].to_numpy(dtype=float)

        selected_model, candidate_rows = select_logistic_model(
            x_train,
            y_train,
            x_validation,
            y_validation,
            regularization_grid=regularization_grid,
        )

        train_probabilities = predict_probabilities(selected_model, x_train)
        validation_probabilities = predict_probabilities(selected_model, x_validation)
        external_probabilities = predict_probabilities(selected_model, x_external)

        metric_rows.extend(
            [
                _metrics_row(
                    spec.name,
                    "internal_train",
                    y_train,
                    train_probabilities,
                    excluded_rows=0,
                ),
                _metrics_row(
                    spec.name,
                    "internal_validation",
                    y_validation,
                    validation_probabilities,
                    excluded_rows=0,
                ),
                _metrics_row(
                    spec.name,
                    "external",
                    y_external,
                    external_probabilities,
                    excluded_rows=external_eligibility["excluded_rows"],
                ),
            ]
        )
        calibration_rows.extend(
            calibration_table_rows(
                spec.name, "internal_train", y_train, train_probabilities
            )
        )
        calibration_rows.extend(
            calibration_table_rows(
                spec.name, "internal_validation", y_validation, validation_probabilities
            )
        )
        calibration_rows.extend(
            calibration_table_rows(
                spec.name, "external", y_external, external_probabilities
            )
        )

        prediction_frames.extend(
            [
                _prediction_frame(
                    train_df, spec.name, "internal_train", train_probabilities
                ),
                _prediction_frame(
                    validation_df,
                    spec.name,
                    "internal_validation",
                    validation_probabilities,
                ),
                _prediction_frame(
                    eligible_external_df,
                    spec.name,
                    "external",
                    external_probabilities,
                ),
            ]
        )

        model_artifacts[spec.name] = {
            "description": spec.description,
            "family": "numpy_logistic_regression",
            "numeric_features": list(spec.numeric_features),
            "categorical_features": list(spec.categorical_features),
            "expanded_feature_names": expanded_feature_names,
            "selected_regularization": selected_model.l2_penalty,
            "intercept": selected_model.intercept,
            "coefficients": selected_model.coefficients.tolist(),
            "iterations": selected_model.iterations,
            "converged": selected_model.converged,
            "optimization_trace": list(selected_model.optimization_trace),
            "target_column": target_column,
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "external_rows": int(len(eligible_external_df)),
            "frozen_for_external_validation": True,
            "missingness_strategy": "complete_case_by_model_numeric_features",
            "preprocessing": serialize_preprocessor(preprocessor),
            "eligibility": {
                "internal": internal_eligibility,
                "external": external_eligibility,
            },
            "validation_candidates": candidate_rows,
        }

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["model", "split"])
        .reset_index(drop=True)
    )
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["model", "split", "dataset", "stay_id"])
        .reset_index(drop=True)
    )
    calibration_df = (
        pd.DataFrame(calibration_rows)
        .sort_values(["model", "split", "bin_index"])
        .reset_index(drop=True)
    )

    metrics_csv_path = output_path / "metrics.csv"
    metrics_json_path = output_path / "metrics.json"
    predictions_csv_path = output_path / "predictions.csv"
    calibration_csv_path = output_path / "calibration.csv"
    models_json_path = output_path / "models.json"
    split_json_path = output_path / "internal_split.json"
    manifest_json_path = output_path / "manifest.json"

    metrics_df.to_csv(metrics_csv_path, index=False)
    predictions_df.to_csv(predictions_csv_path, index=False)
    calibration_df.to_csv(calibration_csv_path, index=False)
    metrics_json_path.write_text(
        json.dumps(_pythonize(metric_rows), indent=2, sort_keys=True)
    )
    models_json_path.write_text(
        json.dumps(_pythonize(model_artifacts), indent=2, sort_keys=True)
    )
    split_json_path.write_text(
        json.dumps(_pythonize(split_artifacts), indent=2, sort_keys=True)
    )

    manifest = {
        "internal_artifact_dir": str(Path(internal_artifact_dir)),
        "external_artifact_dir": str(Path(external_artifact_dir)),
        "locked_external_validation": True,
        "target_column": target_column,
        "validation_fraction": float(validation_fraction),
        "regularization_grid": list(regularization_grid),
        "models": [spec.name for spec in COMPARATOR_SPECS],
        "counts": {
            "internal_rows": int(len(internal_df)),
            "external_rows": int(len(external_df)),
            "metric_rows": int(len(metrics_df)),
            "prediction_rows": int(len(predictions_df)),
        },
        "split_counts_by_model": {
            spec.name: {
                "train_rows": int(model_artifacts[spec.name]["train_rows"]),
                "validation_rows": int(model_artifacts[spec.name]["validation_rows"]),
                "external_rows": int(model_artifacts[spec.name]["external_rows"]),
            }
            for spec in COMPARATOR_SPECS
        },
        "artifacts": {
            "metrics_csv": str(metrics_csv_path),
            "metrics_json": str(metrics_json_path),
            "predictions_csv": str(predictions_csv_path),
            "calibration_csv": str(calibration_csv_path),
            "models_json": str(models_json_path),
            "internal_split_json": str(split_json_path),
        },
    }
    manifest_json_path.write_text(
        json.dumps(_pythonize(manifest), indent=2, sort_keys=True)
    )
    return {
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "predictions_csv": str(predictions_csv_path),
        "calibration_csv": str(calibration_csv_path),
        "models_json": str(models_json_path),
        "internal_split_json": str(split_json_path),
        "manifest_json": str(manifest_json_path),
    }