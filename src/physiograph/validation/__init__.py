"""Validation utilities for data quality and model performance.

Submodules:

- :mod:`~physiograph.validation.evaluate` — Metric computation, bootstrap CIs, calibration
- :mod:`~physiograph.validation.locked_inference` — Frozen comparator model pipeline
- :mod:`~physiograph.validation.transportability` — Cross-dataset evaluation
"""

from physiograph.validation import evaluate, locked_inference, transportability

__all__ = ["evaluate", "locked_inference", "transportability"]