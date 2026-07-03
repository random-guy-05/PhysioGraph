"""Frozen scai_stage_model comparator model.

5-coefficient logistic regression that combines baseline lactate with
SCAI-like shock stages (A–E).  Stages B–E are one-hot encoded relative to
reference stage A, yielding four dummy variables plus the continuous lactate
term.

This model requires only baseline lactate (same eligibility as
``lactate_only``, preserving 12.4% of eICU patients) but adds the clinical
structure of SCAI staging.

Fit on MIMIC-III, frozen for locked external validation on eICU.
"""

from __future__ import annotations

import numpy as np

from physiograph.models.evaluate import sigmoid
from physiograph.models.train import (
    CategoricalTransform,
    ComparatorSpec,
    LogisticModel,
    NumericTransform,
    Preprocessor,
)

NAME: str = "scai_stage_model"
"""Unique model identifier."""

DESCRIPTION: str = (
    "SCAI-like stage comparator using harmonized stage categories."
)
"""Clinical rationale for the model."""

SPEC = ComparatorSpec(
    name=NAME,
    description=DESCRIPTION,
    numeric_features=("baseline_lactate",),
    categorical_features=("scai_stage",),
)
"""Comparator specification with feature declarations."""

PREPROCESSOR = Preprocessor(
    numeric={
        "baseline_lactate": NumericTransform(
            mean=2.3230565760470245,
            scale=1.9384395514610107,
        ),
    },
    categorical={
        "scai_stage": CategoricalTransform(
            categories=("A", "B", "C", "D", "E"),
            fill_value="missing",
            reference_category="A",
        ),
    },
)
"""Frozen preprocessing parameters fitted on the MIMIC-III training set.

The categorical transform one-hot encodes SCAI stage relative to stage A.
The resulting design matrix has columns for baseline_lactate and
scai_stage__B, scai_stage__C, scai_stage__D, scai_stage__E.
"""

COEFFICIENTS = np.array(
    [
        0.42860241391661424,
        -0.30362102270027486,
        1.025355853525069,
        1.1730640372812164,
        1.242921734188985,
    ],
    dtype=float,
)
"""Frozen model coefficients.

Order: ``[baseline_lactate, scai_stage__B, scai_stage__C, scai_stage__D,
scai_stage__E]``.
"""

INTERCEPT: float = -0.5569028052556948
"""Frozen model intercept."""

SELECTED_REGULARIZATION: float = 0.0
"""L2 penalty selected (no regularization)."""

MODEL = LogisticModel(
    coefficients=COEFFICIENTS,
    intercept=INTERCEPT,
    l2_penalty=SELECTED_REGULARIZATION,
    iterations=5,
    converged=True,
    optimization_trace=(
        0.6925507830484897,
        0.6166021423300598,
        0.6150641015760759,
        0.6150327408006688,
        0.6150327224865176,
        0.6150327224865113,
    ),
)
"""Frozen logistic regression model ready for inference."""

EXPANDED_FEATURE_NAMES: list[str] = [
    "baseline_lactate",
    "scai_stage__B",
    "scai_stage__C",
    "scai_stage__D",
    "scai_stage__E",
]
"""Feature names in the order expected by the design matrix.

These are the column names after one-hot encoding: baseline_lactate followed
by dummy variables for SCAI stages B, C, D, E (stage A is the reference).
"""

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
        x_matrix: Design matrix of shape ``(n_samples, 5)``.  Columns must
            be in the order given by :data:`EXPANDED_FEATURE_NAMES`: one
            standardized lactate column followed by four one-hot SCAI stage
            dummy columns.

    Returns:
        1-D array of predicted probabilities in ``[0, 1]``.
    """
    logits = (
        np.asarray(x_matrix, dtype=float) @ COEFFICIENTS + INTERCEPT
    )
    return sigmoid(logits)
