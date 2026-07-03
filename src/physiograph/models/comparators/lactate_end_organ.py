"""Frozen lactate_end_organ comparator model.

10-feature logistic regression combining baseline lactate with acidemia
status (pH, acidemia flags) and end-organ injury markers (creatinine,
bilirubin, perfusion scores).  This model has the highest AUROC (0.842 on
internal validation) but excludes 97.5% of eICU patients because it requires
pH, creatinine, and bilirubin measurements.

Fit on MIMIC-III, frozen for locked external validation on eICU.
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

NAME: str = "lactate_end_organ"
"""Unique model identifier."""

DESCRIPTION: str = (
    "Baseline lactate with acidemia and end-organ injury markers."
)
"""Clinical rationale for the model."""

SPEC = ComparatorSpec(
    name=NAME,
    description=DESCRIPTION,
    numeric_features=(
        "baseline_lactate",
        "baseline_ph",
        "baseline_creatinine",
        "baseline_bilirubin_total",
        "acidemia_flag",
        "severe_acidemia_flag",
        "renal_hypoperfusion_flag",
        "hepatic_hypoperfusion_flag",
        "perfusion_burden_score",
        "lactate_acidemia_interaction",
    ),
)
"""Comparator specification with feature declarations."""

PREPROCESSOR = Preprocessor(
    numeric={
        "baseline_lactate": NumericTransform(
            mean=2.463360902255639,
            scale=2.301955278199876,
        ),
        "baseline_ph": NumericTransform(
            mean=7.355082706766917,
            scale=0.08946222331348697,
        ),
        "baseline_creatinine": NumericTransform(
            mean=2.063082706766917,
            scale=1.7536422593022554,
        ),
        "baseline_bilirubin_total": NumericTransform(
            mean=1.1944360902255637,
            scale=1.9353495977379418,
        ),
        "acidemia_flag": NumericTransform(
            mean=0.15413533834586465,
            scale=0.36107843444170196,
        ),
        "severe_acidemia_flag": NumericTransform(
            mean=0.07969924812030076,
            scale=0.2708270259212686,
        ),
        "renal_hypoperfusion_flag": NumericTransform(
            mean=0.35714285714285715,
            scale=0.4791574237499549,
        ),
        "hepatic_hypoperfusion_flag": NumericTransform(
            mean=0.12105263157894737,
            scale=0.3261884301546562,
        ),
        "perfusion_burden_score": NumericTransform(
            mean=1.6774436090225564,
            scale=1.217856896253574,
        ),
        "lactate_acidemia_interaction": NumericTransform(
            mean=0.6629323308270677,
            scale=2.2399923744348205,
        ),
    },
    categorical={},
)
"""Frozen preprocessing parameters fitted on the MIMIC-III training set."""

COEFFICIENTS = np.array(
    [
        0.7578862345939249,
        -0.10603272937343151,
        1.7287225776394848,
        0.5798001050462739,
        0.049585218371101414,
        0.1647194703387689,
        0.017910233536774508,
        0.386505033098813,
        0.1450599897310304,
        0.35787210243396056,
    ],
    dtype=float,
)
"""Frozen model coefficients (10 features, excluding intercept)."""

INTERCEPT: float = 0.9954122936428967
"""Frozen model intercept."""

SELECTED_REGULARIZATION: float = 0.0
"""L2 penalty selected (no regularization)."""

MODEL = LogisticModel(
    coefficients=COEFFICIENTS,
    intercept=INTERCEPT,
    l2_penalty=SELECTED_REGULARIZATION,
    iterations=7,
    converged=True,
    optimization_trace=(
        0.6806294878916634,
        0.5154378196444378,
        0.4858287245705327,
        0.4771150865307627,
        0.47576783019264657,
        0.4756967691354353,
        0.4756962497759298,
        0.4756962497384419,
    ),
)
"""Frozen logistic regression model ready for inference."""

EXPANDED_FEATURE_NAMES: list[str] = [
    "baseline_lactate",
    "baseline_ph",
    "baseline_creatinine",
    "baseline_bilirubin_total",
    "acidemia_flag",
    "severe_acidemia_flag",
    "renal_hypoperfusion_flag",
    "hepatic_hypoperfusion_flag",
    "perfusion_burden_score",
    "lactate_acidemia_interaction",
]
"""Feature names in the order expected by the design matrix."""

TRAIN_ROWS: int = 1330
"""Number of internal (MIMIC-III) training rows."""

VALIDATION_ROWS: int = 444
"""Number of internal (MIMIC-III) validation rows."""

EXTERNAL_ROWS: int = 612
"""Number of external (eICU) eligible rows (2.5% of 24,194)."""

ELIGIBILITY: dict = {
    "internal": {
        "raw_rows": 14553,
        "eligible_rows": 1774,
        "excluded_rows": 12779,
        "excluded_by_feature": {
            "baseline_lactate": 9109,
            "baseline_ph": 8873,
            "baseline_creatinine": 6312,
            "baseline_bilirubin_total": 10600,
            "acidemia_flag": 0,
            "severe_acidemia_flag": 0,
            "renal_hypoperfusion_flag": 0,
            "hepatic_hypoperfusion_flag": 0,
            "perfusion_burden_score": 0,
            "lactate_acidemia_interaction": 0,
        },
    },
    "external": {
        "raw_rows": 24194,
        "eligible_rows": 612,
        "excluded_rows": 23582,
        "excluded_by_feature": {
            "baseline_lactate": 21186,
            "baseline_ph": 19109,
            "baseline_creatinine": 16944,
            "baseline_bilirubin_total": 20912,
            "acidemia_flag": 0,
            "severe_acidemia_flag": 0,
            "renal_hypoperfusion_flag": 0,
            "hepatic_hypoperfusion_flag": 0,
            "perfusion_burden_score": 0,
            "lactate_acidemia_interaction": 0,
        },
    },
}
"""Eligibility statistics for internal and external datasets."""


def predict_probabilities(x_matrix: np.ndarray) -> np.ndarray:
    """Compute predicted probabilities from the frozen model.

    Args:
        x_matrix: Design matrix of shape ``(n_samples, 10)``.  Columns must
            be in the order given by :data:`EXPANDED_FEATURE_NAMES` and
            standardized using :data:`PREPROCESSOR`.

    Returns:
        1-D array of predicted probabilities in ``[0, 1]``.
    """
    logits = (
        np.asarray(x_matrix, dtype=float) @ COEFFICIENTS + INTERCEPT
    )
    return sigmoid(logits)
