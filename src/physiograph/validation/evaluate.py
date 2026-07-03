"""Classification metrics, bootstrap confidence intervals, and calibration.

Provides pure-NumPy metric computation (AUROC, AUPRC, Brier score,
calibration-in-the-large) along with bootstrap confidence intervals
and temperature-scaling calibration with guard conditions.

All metric formulas are extracted verbatim from the PhysioGraph notebooks
to ensure exact numerical parity with the ground-truth outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Rank-based AUROC and AUPRC (no sklearn dependency)
# ---------------------------------------------------------------------------

def average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute average ranks for an array, handling ties.

    Uses the mid-rank method: tied values receive the average of the
    ranks they would occupy.

    Parameters
    ----------
    values:
        1-D array of values to rank.

    Returns
    -------
    np.ndarray
        Array of average ranks (1-based).
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC using the Wilcoxon–Mann–Whitney statistic.

    Parameters
    ----------
    y_true:
        Binary labels (0 or 1).
    scores:
        Predicted probabilities or scores.

    Returns
    -------
    float
        Area under the receiver operating characteristic curve.

    Raises
    ------
    ValueError
        If both classes are not present in *y_true*.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC is undefined unless both classes are present.")
    ranks = average_ranks(scores)
    positive_ranks = float(np.sum(ranks[y_true == 1]))
    return (positive_ranks - positives * (positives + 1) / 2.0) / float(
        positives * negatives
    )


def binary_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision (area under the precision-recall curve).

    Uses the step-function interpolation method (same as
    ``sklearn.metrics.average_precision_score`` with
    ``average='macro'``).

    Parameters
    ----------
    y_true:
        Binary labels (0 or 1).
    scores:
        Predicted probabilities or scores.

    Returns
    -------
    float
        Average precision score.

    Raises
    ------
    ValueError
        If there are no positive examples in *y_true*.
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
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / float(positive_total)
    previous_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous_recall) * precision))


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def binary_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Compute binary classification metrics.

    Computes AUROC, AUPRC, Brier score, calibration-in-the-large,
    observed/expected event counts, and positive rate.  The metric
    formulas match the ground-truth values in
    ``tests/ground_truth/metrics.json`` exactly.

    Parameters
    ----------
    y_true:
        Binary labels (0 or 1).
    probabilities:
        Predicted probabilities (values between 0 and 1).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: ``n``, ``positive_rate``, ``brier``,
        ``auroc``, ``auprc``, ``observed_events``, ``expected_events``,
        ``expected_to_observed_ratio``, ``calibration_in_the_large``.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must have the same length.")

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
        expected_to_observed_ratio = float(expected_events / observed_events)

    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "brier": float(np.mean((probabilities - y_true) ** 2)) if len(y_true) else None,
        "auroc": auroc,
        "auprc": auprc,
        "observed_events": observed_events,
        "expected_events": expected_events,
        "expected_to_observed_ratio": expected_to_observed_ratio,
        "calibration_in_the_large": float(probabilities.mean() - y_true.mean()) if len(y_true) else None,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_table_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute calibration table rows for equal-width probability bins.

    Parameters
    ----------
    model_name:
        Name of the model.
    split_name:
        Name of the data split (e.g. ``"internal_train"``).
    y_true:
        Binary labels.
    probabilities:
        Predicted probabilities.
    n_bins:
        Number of equal-width bins (default 10).

    Returns
    -------
    list[dict[str, Any]]
        List of dicts, one per non-empty bin, with keys:
        ``model``, ``split``, ``bin_index``, ``bin_left``, ``bin_right``,
        ``n``, ``mean_predicted_probability``, ``observed_event_rate``.
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
                "mean_predicted_probability": float(np.mean(probabilities[mask])),
                "observed_event_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the expected calibration error (ECE).

    ECE is the weighted average of absolute differences between
    observed event rates and mean predicted probabilities across
    equal-width bins.

    Parameters
    ----------
    y_true:
        Binary labels (0 or 1).
    y_prob:
        Predicted probabilities.
    n_bins:
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        Expected calibration error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for AUROC and AUPRC.

    Resamples with replacement and computes metrics on each bootstrap
    sample.  Returns (mean, lower, upper) tuples for each metric.

    Parameters
    ----------
    y_true:
        Binary labels (0 or 1).
    y_probs:
        Predicted probabilities.
    n_bootstraps:
        Number of bootstrap resamples (default 1000).
    alpha:
        Significance level for confidence intervals (default 0.05).
    seed:
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Dictionary mapping metric names to ``(mean, lower, upper)``
        confidence interval tuples.  Keys are ``"auroc"`` and/or
        ``"auprc"`` depending on which metrics could be computed.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)

    auroc_scores: list[float] = []
    auprc_scores: list[float] = []
    rng = np.random.RandomState(seed)

    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_probs), len(y_probs), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auroc_scores.append(float(binary_auroc(y_true[idx], y_probs[idx])))
        except ValueError:
            pass
        try:
            auprc_scores.append(float(binary_average_precision(y_true[idx], y_probs[idx])))
        except ValueError:
            pass

    def _ci(scores: list[float]) -> tuple[float, float, float]:
        arr = np.array(scores)
        return (
            float(np.mean(arr)),
            float(np.percentile(arr, (alpha / 2) * 100)),
            float(np.percentile(arr, (1 - alpha / 2) * 100)),
        )

    result: dict[str, tuple[float, float, float]] = {}
    if auroc_scores:
        result["auroc"] = _ci(auroc_scores)
    if auprc_scores:
        result["auprc"] = _ci(auprc_scores)
    return result


# ---------------------------------------------------------------------------
# Probability spread statistics
# ---------------------------------------------------------------------------

def probability_spread_stats(y_prob: np.ndarray) -> dict[str, float]:
    """Compute summary statistics of a probability distribution.

    Parameters
    ----------
    y_prob:
        Predicted probabilities.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``mean``, ``std``, ``p10``, ``p50``, ``p90``.
    """
    y_prob = np.asarray(y_prob)
    return {
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
        "p10": float(np.percentile(y_prob, 10)),
        "p50": float(np.percentile(y_prob, 50)),
        "p90": float(np.percentile(y_prob, 90)),
    }


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def temperature_scale(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
    lr: float = 0.1,
) -> tuple[float, dict[str, Any]]:
    """Learn a positive temperature scaling factor via LBFGS optimization.

    Uses softplus parameterization (``softplus(log_temperature) + 1e-4``)
    to ensure the temperature is strictly positive, and applies guard
    conditions to reject calibration that worsens Brier score or
    compresses probability spread too aggressively.

    Guard conditions (all must hold to accept calibration):

    1. Learned temperature is finite and positive.
    2. Calibrated Brier score ≤ raw Brier score + 1e-4.
    3. Calibrated probability spread ≥ 85% of raw spread (or raw
       spread is near-zero).

    Parameters
    ----------
    logits:
        Raw model logits (pre-sigmoid outputs).
    labels:
        Binary labels (0 or 1).
    max_iter:
        Maximum LBFGS iterations (default 200).
    lr:
        LBFGS learning rate (default 0.1).

    Returns
    -------
    tuple[float, dict[str, Any]]
        A tuple of ``(learned_temperature, diagnostics)`` where
        *diagnostics* contains Brier/ECE before and after, spread
        statistics, and the ``use_calibrated`` decision.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    log_temperature = nn.Parameter(torch.tensor(0.0))
    optimizer = optim.LBFGS(
        [log_temperature], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def _current_temperature() -> torch.Tensor:
        return F.softplus(log_temperature) + 1e-4

    def _eval() -> torch.Tensor:
        optimizer.zero_grad()
        temp = _current_temperature()
        loss = F.binary_cross_entropy_with_logits(
            logits_tensor / temp, labels_tensor
        )
        loss.backward()
        return loss

    optimizer.step(_eval)
    learned_temperature = float(_current_temperature().item())

    with torch.no_grad():
        raw_probs = torch.sigmoid(logits_tensor).numpy()
        calibrated_probs = torch.sigmoid(logits_tensor / learned_temperature).numpy()

    raw_brier = float(np.mean((raw_probs - labels) ** 2))
    cal_brier = float(np.mean((calibrated_probs - labels) ** 2))
    raw_ece = expected_calibration_error(labels, raw_probs)
    cal_ece = expected_calibration_error(labels, calibrated_probs)

    raw_spread = probability_spread_stats(raw_probs)
    cal_spread = probability_spread_stats(calibrated_probs)

    spread_guard_ok = (
        raw_spread["std"] < 1e-6
        or cal_spread["std"] >= 0.85 * raw_spread["std"]
    )

    use_calibrated = (
        np.isfinite(learned_temperature)
        and learned_temperature > 0
        and cal_brier <= raw_brier + 1e-4
        and spread_guard_ok
    )

    diagnostics: dict[str, Any] = {
        "learned_temperature": learned_temperature,
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "raw_spread": raw_spread,
        "calibrated_spread": cal_spread,
        "spread_guard_ok": spread_guard_ok,
        "use_calibrated": use_calibrated,
    }

    return learned_temperature, diagnostics