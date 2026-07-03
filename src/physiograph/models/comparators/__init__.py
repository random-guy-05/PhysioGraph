"""Frozen comparator models for locked external validation.

This package contains the four pre-registered logistic regression comparator
models extracted from the PhysioGraph notebooks.  All models were fit on
MIMIC-III, frozen, and validated on eICU.

Models
------
- :mod:`~physiograph.models.comparators.lactate_only`: Baseline lactate (1
  feature, 12.4% eICU preservation).
- :mod:`~physiograph.models.comparators.lactate_hemodynamics`: Lactate +
  hemodynamics (14 features, 2.4% eICU preservation).
- :mod:`~physiograph.models.comparators.lactate_end_organ`: Lactate +
  acidemia + end-organ markers (10 features, 2.5% eICU preservation,
  highest AUROC).
- :mod:`~physiograph.models.comparators.scai_stage_model`: Lactate + SCAI
  stage (5 features, 12.4% eICU preservation).

Usage
-----
.. code-block:: python

    from physiograph.models.comparators.lactate_only import (
        predict_probabilities,
    )

    # x is a (n_samples, 1) numpy array of standardized baseline_lactate
    probs = predict_probabilities(x)
"""

from physiograph.models.comparators import (
    lactate_end_organ,
    lactate_hemodynamics,
    lactate_only,
    scai_stage_model,
)

__all__ = [
    "lactate_end_organ",
    "lactate_hemodynamics",
    "lactate_only",
    "scai_stage_model",
]
