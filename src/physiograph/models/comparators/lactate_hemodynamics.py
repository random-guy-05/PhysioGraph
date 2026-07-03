"""Frozen lactate_hemodynamics comparator model.

14-feature logistic regression that pairs baseline lactate with hemodynamic
context (heart rate, blood pressure, derived interaction terms).  This model
excludes 97.6% of eICU patients because it requires complete MAP and SBP
measurements during the observation window.

The model was fit on MIMIC-III and frozen for locked external validation on
eICU.  L2 penalty selected via the Brier → AUPRC → AUROC → penalty cascade.
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

NAME: str = "lactate_hemodynamics"
"""Unique model identifier."""

DESCRIPTION: str = "Baseline lactate with hemodynamic context."
"""Clinical rationale for the model."""

SPEC = ComparatorSpec(
    name=NAME,
    description=DESCRIPTION,
    numeric_features=(
        "baseline_lactate",
        "baseline_hr",
        "baseline_sbp",
        "baseline_map",
        "tachycardia_flag",
        "hypotension_flag",
        "severe_hypotension_flag",
        "hr_above_100_fraction",
        "sbp_below_90_fraction",
        "map_below_65_fraction",
        "lactate_map_ratio",
        "lactate_sbp_ratio",
        "lactate_hypotension_interaction",
        "lactate_tachycardia_interaction",
    ),
)
"""Comparator specification with feature declarations."""

PREPROCESSOR = Preprocessor(
    numeric={
        "baseline_lactate": NumericTransform(
            mean=2.25687858117326,
            scale=1.8979117332105535,
        ),
        "baseline_hr": NumericTransform(
            mean=86.28731241473398,
            scale=19.85237783928534,
        ),
        "baseline_sbp": NumericTransform(
            mean=115.80518417462483,
            scale=23.13719035427496,
        ),
        "baseline_map": NumericTransform(
            mean=78.38390177353342,
            scale=24.1937092282741,
        ),
        "tachycardia_flag": NumericTransform(
            mean=0.3844474761255116,
            scale=0.4864644018078155,
        ),
        "hypotension_flag": NumericTransform(
            mean=0.4504774897680764,
            scale=0.49754147664323334,
        ),
        "severe_hypotension_flag": NumericTransform(
            mean=0.2747612551159618,
            scale=0.44639389310682015,
        ),
        "hr_above_100_fraction": NumericTransform(
            mean=0.2688366829956632,
            scale=0.39406489944701073,
        ),
        "sbp_below_90_fraction": NumericTransform(
            mean=0.08947763969733151,
            scale=0.195239562471511,
        ),
        "map_below_65_fraction": NumericTransform(
            mean=0.1732886666859617,
            scale=0.2690709555146608,
        ),
        "lactate_map_ratio": NumericTransform(
            mean=0.03037020423996791,
            scale=0.0272613643429421,
        ),
        "lactate_sbp_ratio": NumericTransform(
            mean=0.020573546730643648,
            scale=0.019246055326194462,
        ),
        "lactate_hypotension_interaction": NumericTransform(
            mean=1.0870668485675308,
            scale=1.8532804857789864,
        ),
        "lactate_tachycardia_interaction": NumericTransform(
            mean=0.9641909959072306,
            scale=1.7519736698313992,
        ),
    },
    categorical={},
)
"""Frozen preprocessing parameters fitted on the MIMIC-III training set."""

COEFFICIENTS = np.array(
    [
        -0.1691639586611817,
        0.04715031642897572,
        0.16283589510045543,
        0.08267888934001917,
        -0.14883970934709514,
        -0.014261623670402585,
        0.07905941849546057,
        -0.03223982205055024,
        0.11286940929627379,
        0.12616530511230972,
        0.030302555197426383,
        0.9769752654506229,
        -0.05203736660325015,
        0.20808010241793892,
    ],
    dtype=float,
)
"""Frozen model coefficients (14 features, excluding intercept)."""

INTERCEPT: float = 0.04613573444000704
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
        0.6930718002382373,
        0.6510490859511485,
        0.6456556638918272,
        0.6451954756570173,
        0.6451919797981175,
        0.6451919795921514,
    ),
)
"""Frozen logistic regression model ready for inference."""

EXPANDED_FEATURE_NAMES: list[str] = [
    "baseline_lactate",
    "baseline_hr",
    "baseline_sbp",
    "baseline_map",
    "tachycardia_flag",
    "hypotension_flag",
    "severe_hypotension_flag",
    "hr_above_100_fraction",
    "sbp_below_90_fraction",
    "map_below_65_fraction",
    "lactate_map_ratio",
    "lactate_sbp_ratio",
    "lactate_hypotension_interaction",
    "lactate_tachycardia_interaction",
]
"""Feature names in the order expected by the design matrix."""

TRAIN_ROWS: int = 3665
"""Number of internal (MIMIC-III) training rows."""

VALIDATION_ROWS: int = 1221
"""Number of internal (MIMIC-III) validation rows."""

EXTERNAL_ROWS: int = 589
"""Number of external (eICU) eligible rows (2.4% of 24,194)."""

ELIGIBILITY: dict = {
    "internal": {
        "raw_rows": 14553,
        "eligible_rows": 4886,
        "excluded_rows": 9667,
        "excluded_by_feature": {
            "baseline_lactate": 9109,
            "baseline_hr": 624,
            "baseline_sbp": 775,
            "baseline_map": 767,
            "tachycardia_flag": 0,
            "hypotension_flag": 0,
            "severe_hypotension_flag": 0,
            "hr_above_100_fraction": 0,
            "sbp_below_90_fraction": 0,
            "map_below_65_fraction": 0,
            "lactate_map_ratio": 9661,
            "lactate_sbp_ratio": 9664,
            "lactate_hypotension_interaction": 0,
            "lactate_tachycardia_interaction": 0,
        },
    },
    "external": {
        "raw_rows": 24194,
        "eligible_rows": 589,
        "excluded_rows": 23605,
        "excluded_by_feature": {
            "baseline_lactate": 21186,
            "baseline_hr": 775,
            "baseline_sbp": 21173,
            "baseline_map": 21135,
            "tachycardia_flag": 0,
            "hypotension_flag": 0,
            "severe_hypotension_flag": 0,
            "hr_above_100_fraction": 0,
            "sbp_below_90_fraction": 0,
            "map_below_65_fraction": 0,
            "lactate_map_ratio": 23589,
            "lactate_sbp_ratio": 23605,
            "lactate_hypotension_interaction": 0,
            "lactate_tachycardia_interaction": 0,
        },
    },
}
"""Eligibility statistics for internal and external datasets."""


def predict_probabilities(x_matrix: np.ndarray) -> np.ndarray:
    """Compute predicted probabilities from the frozen model.

    Args:
        x_matrix: Design matrix of shape ``(n_samples, 14)``.  Columns must
            be in the order given by :data:`EXPANDED_FEATURE_NAMES` and
            standardized using :data:`PREPROCESSOR`.

    Returns:
        1-D array of predicted probabilities in ``[0, 1]``.
    """
    logits = (
        np.asarray(x_matrix, dtype=float) @ COEFFICIENTS + INTERCEPT
    )
    return sigmoid(logits)
