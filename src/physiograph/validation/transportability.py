"""Cross-dataset transportability assessment.

Provides functions for comparing model performance across internal
and external datasets, assessing calibration drift, and summarizing
generalization gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from physiograph.validation.evaluate import (
    binary_classification_metrics,
    calibration_table_rows,
    expected_calibration_error,
)


def compare_splits(
    metrics: list[dict[str, Any]],
    *,
    internal_splits: tuple[str, ...] = (
        "internal_train",
        "internal_validation",
    ),
    external_split: str = "external",
) -> pd.DataFrame:
    """Compare internal vs external performance across models.

    For each model, selects the internal validation split as the
    reference and computes the generalization gap (external − internal)
    for AUROC, AUPRC, Brier score, and calibration-in-the-large.

    Parameters
    ----------
    metrics:
        List of metric dicts as produced by
        :func:`~physiograph.validation.evaluate.binary_classification_metrics`.
    internal_splits:
        Split names considered internal.
    external_split:
        Split name for the external dataset.

    Returns
    -------
    pd.DataFrame
        One row per model with internal metrics, external metrics,
        generalization gaps, and exclusion rates.
    """
    metrics_df = pd.DataFrame(metrics)
    models = metrics_df["model"].unique()
    rows: list[dict[str, Any]] = []

    for model in sorted(models):
        model_df = metrics_df[metrics_df["model"] == model]
        internal_rows = model_df[model_df["split"].isin(internal_splits)]
        external_rows = model_df[model_df["split"] == external_split]

        if internal_rows.empty or external_rows.empty:
            continue

        val_row = internal_rows[internal_rows["split"] == "internal_validation"]
        if val_row.empty:
            val_row = internal_rows.iloc[0:1]
        val_row = val_row.iloc[0]

        ext_row = external_rows.iloc[0]

        row: dict[str, Any] = {"model": model}
        for metric in ("auroc", "auprc", "brier", "calibration_in_the_large"):
            val_val = val_row.get(metric)
            ext_val = ext_row.get(metric)
            row[f"internal_{metric}"] = val_val
            row[f"external_{metric}"] = ext_val
            if val_val is not None and ext_val is not None:
                row[f"{metric}_gap"] = float(ext_val) - float(val_val)
            else:
                row[f"{metric}_gap"] = None

        row["internal_n"] = int(val_row.get("n", 0))
        row["external_n"] = int(ext_row.get("n", 0))
        row["internal_excluded"] = int(val_row.get("excluded_rows", 0))
        row["external_excluded"] = int(ext_row.get("excluded_rows", 0))

        total_internal = int(val_row.get("n", 0)) + int(val_row.get("excluded_rows", 0))
        total_external = int(ext_row.get("n", 0)) + int(ext_row.get("excluded_rows", 0))
        row["exclusion_rate_internal"] = (
            float(val_row["excluded_rows"]) / total_internal if total_internal > 0 else None
        )
        row["exclusion_rate_external"] = (
            float(ext_row["excluded_rows"]) / total_external if total_external > 0 else None
        )
        rows.append(row)

    return pd.DataFrame(rows)


def calibration_drift(
    y_true_internal: np.ndarray,
    probs_internal: np.ndarray,
    y_true_external: np.ndarray,
    probs_external: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Assess calibration drift between internal and external datasets.

    Computes the expected calibration error (ECE) and
    calibration-in-the-large for both datasets, then quantifies the
    drift as the absolute difference in calibration-in-the-large.

    Parameters
    ----------
    y_true_internal:
        Binary labels for the internal dataset.
    probs_internal:
        Predicted probabilities for the internal dataset.
    y_true_external:
        Binary labels for the external dataset.
    probs_external:
        Predicted probabilities for the external dataset.
    n_bins:
        Number of calibration bins (default 10).

    Returns
    -------
    dict[str, Any]
        Dictionary with internal/external ECE, calibration-in-the-large,
        and drift metrics.
    """
    internal_metrics = binary_classification_metrics(y_true_internal, probs_internal)
    external_metrics = binary_classification_metrics(y_true_external, probs_external)

    internal_ece = expected_calibration_error(
        y_true_internal, probs_internal, n_bins=n_bins
    )
    external_ece = expected_calibration_error(
        y_true_external, probs_external, n_bins=n_bins
    )

    cal_internal = internal_metrics.get("calibration_in_the_large", 0.0) or 0.0
    cal_external = external_metrics.get("calibration_in_the_large", 0.0) or 0.0

    return {
        "internal_ece": internal_ece,
        "external_ece": external_ece,
        "ece_drift": float(external_ece - internal_ece),
        "internal_calibration_in_the_large": float(cal_internal),
        "external_calibration_in_the_large": float(cal_external),
        "calibration_drift": float(abs(cal_external - cal_internal)),
        "internal_brier": internal_metrics["brier"],
        "external_brier": external_metrics["brier"],
        "brier_drift": (
            float(external_metrics["brier"] - internal_metrics["brier"])
            if internal_metrics["brier"] is not None
            and external_metrics["brier"] is not None
            else None
        ),
        "internal_auroc": internal_metrics["auroc"],
        "external_auroc": external_metrics["auroc"],
        "auroc_drift": (
            float(external_metrics["auroc"] - internal_metrics["auroc"])
            if internal_metrics["auroc"] is not None
            and external_metrics["auroc"] is not None
            else None
        ),
        "internal_n": internal_metrics["n"],
        "external_n": external_metrics["n"],
    }


def cross_dataset_evaluation(
    y_true_internal: np.ndarray,
    probs_internal: np.ndarray,
    y_true_external: np.ndarray,
    probs_external: np.ndarray,
    *,
    model_name: str = "",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Full cross-dataset evaluation comparing internal and external performance.

    Combines binary classification metrics, calibration drift assessment,
    and calibration table rows for both datasets.

    Parameters
    ----------
    y_true_internal:
        Binary labels for the internal dataset.
    probs_internal:
        Predicted probabilities for the internal dataset.
    y_true_external:
        Binary labels for the external dataset.
    probs_external:
        Predicted probabilities for the external dataset.
    model_name:
        Name of the model (for labeling calibration rows).
    n_bins:
        Number of calibration bins (default 10).

    Returns
    -------
    dict[str, Any]
        Dictionary with ``internal_metrics``, ``external_metrics``,
        ``drift``, and ``calibration`` keys.
    """
    internal_metrics = binary_classification_metrics(y_true_internal, probs_internal)
    external_metrics = binary_classification_metrics(y_true_external, probs_external)
    drift = calibration_drift(
        y_true_internal,
        probs_internal,
        y_true_external,
        probs_external,
        n_bins=n_bins,
    )

    internal_cal = calibration_table_rows(
        model_name, "internal", y_true_internal, probs_internal, n_bins=n_bins
    )
    external_cal = calibration_table_rows(
        model_name, "external", y_true_external, probs_external, n_bins=n_bins
    )

    return {
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "drift": drift,
        "calibration": internal_cal + external_cal,
    }