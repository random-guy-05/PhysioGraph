"""Evaluation metrics for binary classification in locked comparator validation.

Provides AUROC, AUPRC, Brier score, calibration-in-the-large, and
calibration table generation.  All metrics are computed via pure numpy
(no scikit-learn dependency) so the comparator models are fully portable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Compute standard binary classification metrics.

    Returns sample counts, event rates, Brier score, AUROC, AUPRC,
    expected-to-observed event ratio, and calibration-in-the-large.

    Args:
        y_true: 1-D array of binary ground-truth labels.
        probabilities: 1-D array of predicted probabilities.

    Returns:
        Dictionary with keys ``n``, ``positive_rate``, ``brier``,
        ``auroc``, ``auprc``, ``observed_events``, ``expected_events``,
        ``expected_to_observed_ratio``, ``calibration_in_the_large``.

    Raises:
        ValueError: If *y_true* and *probabilities* have different lengths.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "y_true and probabilities must have the same length."
        )

    auroc: float | None = None
    auprc: float | None = None
    if len(y_true):
        try:
            auroc = float(binary_auroc(y_true, probabilities))
        except ValueError:
            auroc = None
        try:
            auprc = float(binary_average_precision(y_true, probabilities))
        except ValueError:
            auprc = None

    observed_events = float(np.sum(y_true)) if len(y_true) else None
    expected_events = float(np.sum(probabilities)) if len(y_true) else None
    if observed_events in (None, 0.0):
        expected_to_observed_ratio = None
    else:
        expected_to_observed_ratio = float(
            expected_events / observed_events
        )

    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "brier": (
            float(np.mean((probabilities - y_true) ** 2))
            if len(y_true)
            else None
        ),
        "auroc": auroc,
        "auprc": auprc,
        "observed_events": observed_events,
        "expected_events": expected_events,
        "expected_to_observed_ratio": expected_to_observed_ratio,
        "calibration_in_the_large": (
            float(probabilities.mean() - y_true.mean())
            if len(y_true)
            else None
        ),
    }


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Wilcoxon-Mann-Whitney statistic.

    Computes AUROC by averaging the ranks of positive examples without
    building the full ROC curve.

    Args:
        y_true: 1-D binary labels.
        scores: 1-D continuous scores (higher = more positive).

    Returns:
        AUROC value in ``[0, 1]``.

    Raises:
        ValueError: If either class is absent.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError(
            "AUROC is undefined unless both classes are present."
        )
    ranks = average_ranks(scores)
    positive_ranks = float(np.sum(ranks[y_true == 1]))
    return (
        positive_ranks - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def binary_average_precision(
    y_true: np.ndarray, scores: np.ndarray
) -> float:
    """Area under the precision-recall curve (average precision).

    Args:
        y_true: 1-D binary labels.
        scores: 1-D continuous scores (higher = more positive).

    Returns:
        AUPRC value in ``[0, 1]``.

    Raises:
        ValueError: If there are no positive examples.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive_total = int(np.sum(y_true == 1))
    if positive_total == 0:
        raise ValueError(
            "AUPRC is undefined when there are no positive examples."
        )
    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = y_true[order]
    true_positives = np.cumsum(sorted_truth == 1)
    false_positives = np.cumsum(sorted_truth == 0)
    precision = true_positives / np.maximum(
        true_positives + false_positives, 1
    )
    recall = true_positives / float(positive_total)
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous_recall) * precision))


def calibration_table_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Build calibration table rows for reliability diagrams.

    Partitions predicted probabilities into *n_bins* equal-width bins and
    computes the mean prediction and observed event rate per bin.

    Args:
        model_name: Comparator model name.
        split_name: Split label (e.g. ``"external"``).
        y_true: 1-D binary ground-truth labels.
        probabilities: 1-D predicted probabilities.
        n_bins: Number of equal-width bins.

    Returns:
        List of dicts with keys ``model``, ``split``, ``bin_index``,
        ``bin_left``, ``bin_right``, ``n``,
        ``mean_predicted_probability``, ``observed_event_rate``.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0:
        return []

    bins = np.linspace(0.0, 1.0, num=n_bins + 1)
    rows: list[dict[str, Any]] = []
    for bin_index in range(n_bins):
        left = float(bins[bin_index])
        right = float(bins[bin_index + 1])
        if bin_index == n_bins - 1:
            mask = (probabilities >= left) & (probabilities <= right)
        else:
            mask = (probabilities >= left) & (probabilities < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "bin_index": int(bin_index),
                "bin_left": left,
                "bin_right": right,
                "n": int(np.sum(mask)),
                "mean_predicted_probability": float(
                    np.mean(probabilities[mask])
                ),
                "observed_event_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def compute_calibration(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics including binned statistics.

    Convenience wrapper around :func:`calibration_table_rows` that also
    returns calibration-in-the-large and Brier score.

    Args:
        y_true: 1-D binary labels.
        probabilities: 1-D predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Dictionary with keys ``brier``, ``calibration_in_the_large``,
        ``bins`` (list of bin dicts).
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    base_metrics = binary_classification_metrics(y_true, probabilities)
    return {
        "brier": base_metrics["brier"],
        "calibration_in_the_large": base_metrics[
            "calibration_in_the_large"
        ],
        "bins": calibration_table_rows(
            "model", "split", y_true, probabilities, n_bins=n_bins
        ),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for equal-value groups.

    Args:
        values: 1-D array of numeric scores.

    Returns:
        1-D array of average ranks (1-based).
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def logistic_objective(
    design: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
    *,
    l2_penalty: float,
) -> float:
    """Logistic regression objective (negative log-likelihood + L2 penalty).

    Args:
        design: Augmented design matrix ``[ones, X]`` of shape
            ``(n_samples, n_features + 1)``.
        targets: Binary target vector.
        coefficients: Coefficient vector (including intercept as first
            element).
        l2_penalty: L2 regularization strength applied to all coefficients
            except the intercept.

    Returns:
        Scalar objective value.
    """
    logits = design @ coefficients
    probabilities = np.clip(sigmoid(logits), 1e-8, 1.0 - 1e-8)
    negative_log_likelihood = -np.mean(
        targets * np.log(probabilities)
        + (1.0 - targets) * np.log(1.0 - probabilities)
    )
    penalty = (
        0.5
        * float(l2_penalty)
        * float(np.sum(coefficients[1:] ** 2))
    )
    return float(negative_log_likelihood + penalty)


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Args:
        values: Input array.

    Returns:
        Sigmoid-transformed array in ``(0, 1)``.
    """
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))
