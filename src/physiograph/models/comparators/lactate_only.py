"""Frozen lactate_only comparator model.

Single-feature logistic regression using only baseline landmark lactate.
This is the simplest comparator — it preserves the most patients (12.4% of
eICU) and serves as the clinical baseline that all richer models must
outperform.

The model was fit via Newton-Raphson with L2 penalty grid search over
``[0.0, 0.01, 0.1, 1.0]`` on MIMIC-III, then frozen for locked external
validation on eICU.
"""

from __future__ import annotations

import numpy as np

from physiograph.models.evaluate import sigmoid
from physiograph.models.train import (
    ComparatorSpec,
    LogisticModel,
    NumericTransform,
    Preprocessor,
)

NAME: str = "lactate_only"
"""Unique model identifier."""

DESCRIPTION: str = "Baseline landmark lactate only."
"""Clinical rationale for the model."""

SPEC = ComparatorSpec(
    name=NAME,
    description=DESCRIPTION,
    numeric_features=("baseline_lactate",),
)
"""Comparator specification with feature declarations."""

PREPROCESSOR = Preprocessor(
    numeric={
        "baseline_lactate": NumericTransform(
            mean=2.3230565760470245,
            scale=1.9384395514610107,
        ),
    },
    categorical={},
)
"""Frozen preprocessing parameters fitted on the MIMIC-III training set."""

COEFFICIENTS = np.array([0.7668485192226935], dtype=float)
"""Frozen model coefficients (excluding intercept)."""

INTERCEPT: float = 0.12734433998421354
"""Frozen model intercept."""

SELECTED_REGULARIZATION: float = 0.01
"""L2 penalty selected by the Brier → AUPRC → AUROC → penalty cascade."""

MODEL = LogisticModel(
    coefficients=COEFFICIENTS,
    intercept=INTERCEPT,
    l2_penalty=SELECTED_REGULARIZATION,
    iterations=5,
    converged=True,
    optimization_trace=(
        0.6925507830484897,
        0.6557259159740172,
        0.6517820239382957,
        0.6516941657547858,
        0.6516941217026501,
        0.6516941217026389,
    ),
)
"""Frozen logistic regression model ready for inference."""

EXPANDED_FEATURE_NAMES: list[str] = ["baseline_lactate"]
"""Feature names in the order expected by the design matrix."""

TRAIN_ROWS: int = 4083
"""Number of internal (MIMIC-III) training rows."""

VALIDATION_ROWS: int = 1361
"""Number of internal (MIMIC-III) validation rows."""

EXTERNAL_ROWS: int = 3008
"""Number of external (eICU) eligible rows (12.4% of 24,194)."""

ELIGIBILITY: dict = {
    "internal": {
        "raw_rows": 14553,
        "eligible_rows": 5444,
        "excluded_rows": 9109,
        "excluded_by_feature": {"baseline_lactate": 9109},
    },
    "external": {
        "raw_rows": 24194,
        "eligible_rows": 3008,
        "excluded_rows": 21186,
        "excluded_by_feature": {"baseline_lactate": 21186},
    },
}
"""Eligibility statistics for internal and external datasets."""


def predict_probabilities(x_matrix: np.ndarray) -> np.ndarray:
    """Compute predicted probabilities from the frozen model.

    Args:
        x_matrix: Design matrix of shape ``(n_samples, 1)``.  The single
            column must be standardized ``baseline_lactate`` using
            :data:`PREPROCESSOR`.

    Returns:
        1-D array of predicted probabilities in ``[0, 1]``.
    """
    logits = (
        np.asarray(x_matrix, dtype=float) @ COEFFICIENTS + INTERCEPT
    )
    return sigmoid(logits)
