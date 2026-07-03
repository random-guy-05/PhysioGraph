"""Missingness taxonomy and temporal imputation for clinical time-series.

Provides:

1. A three-category missingness taxonomy (structural / informative / random)
   for clinical variables in ICU time-series.
2. Forward-fill with exponential decay toward clinical normals — the
   temporal imputation strategy used by PhysioGraph's tensor pipeline.
3. Clinical-normals fallback for values that remain missing after
   forward-fill and decay.

The forward-fill + decay algorithm is extracted from
``PhysioGraph_External_Pipeline.ipynb`` Cell 74 and
``PhysioGraph_HF_Shock.ipynb`` Cell 7.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Missingness taxonomy
# ---------------------------------------------------------------------------

MISSINGNESS_STRUCTURAL: str = "structural"
"""Value was never measured because the measurement is not standard care
for this patient (e.g. no BNP ordered for a low-risk patient)."""

MISSINGNESS_INFORMATIVE: str = "informative"
"""Absence carries clinical signal — the measurement was deliberately
omitted because the patient was deemed too unstable, already received
treatment that renders the measurement uninterpretable, or other
clinical judgment.  Also called MNAR (missing not at random)."""

MISSINGNESS_RANDOM: str = "random"
"""Missing due to scheduling gaps, sensor disconnection, or lab batching.
Also called MCAR (missing completely at random) or MAR (missing at
random) in clinical contexts where the absence is unrelated to the
patient's underlying state."""

MISSINGNESS_TYPES: tuple[str, str, str] = (
    MISSINGNESS_STRUCTURAL,
    MISSINGNESS_INFORMATIVE,
    MISSINGNESS_RANDOM,
)
"""All recognised missingness categories."""


def classify_missingness(
    variable: str,
    measurement_count: int,
    clinical_context: Optional[str] = None,
) -> str:
    """Classify the likely missingness mechanism for a clinical variable.

    Classification is based on the number of measurements recorded
    during the observation window and the clinical role of the variable.

    Parameters
    ----------
    variable : str
        Clinical variable name (e.g. ``"lactate"``, ``"bnp"``).
    measurement_count : int
        Number of measurements for this variable in the observation
        window.
    clinical_context : str, optional
        Additional clinical context (e.g. diagnosis group).  Not yet
        used but reserved for future refinement.

    Returns
    -------
    str
        One of :data:`MISSINGNESS_STRUCTURAL`,
        :data:`MISSINGNESS_INFORMATIVE`, or :data:`MISSINGNESS_RANDOM`.

    Notes
    -----
    This is a heuristic classification.  In practice, the true
    missingness mechanism is unknown and must be inferred from domain
    knowledge.  The labels here are intended for documentation and
    sensitivity analysis, not for automated imputation decisions.
    """
    # Variables that are almost always measured in ICU: vitals + core labs
    _ubiquitous_vars = {
        "hr", "sbp", "dbp", "map", "resp_rate", "spo2", "temp",
        "lactate", "ph", "creatinine",
    }
    # Variables that are selectively ordered based on clinical suspicion
    _selective_vars = {
        "bnp", "troponin_t", "alt", "ast", "bilirubin_total",
        "albumin", "inr",
    }

    if measurement_count == 0:
        if variable in _selective_vars:
            return MISSINGNESS_STRUCTURAL
        return MISSINGNESS_INFORMATIVE
    if measurement_count < 2:
        return MISSINGNESS_RANDOM
    return MISSINGNESS_RANDOM  # sufficient measurements → random gaps only


# ---------------------------------------------------------------------------
# Forward-fill with exponential decay
# ---------------------------------------------------------------------------


def apply_forward_fill_and_decay(
    values: np.ndarray,
    mask: np.ndarray,
    node_normals: np.ndarray,
    *,
    step_hours: float = 0.25,
    decay_start_hours: float = 2.0,
    decay_rate: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill missing values with exponential decay toward normals.

    This is the primary temporal imputation strategy for the PhysioGraph
    tensor pipeline.  For each time step *t*:

    1. **Forward-fill**: carry forward the last observed value for each
       node.  This preserves temporal continuity within the observation
       window.

    2. **Delta tracking**: maintain a per-node counter of how long it
       has been since the last real observation.  The counter resets to 0
       whenever a new measurement arrives.

    3. **Decay**: after ``decay_start_hours`` of consecutive missingness,
       the imputed value decays exponentially toward the clinical normal
       value for that node:

       .. math::

          v_t = v_{t-1} \\cdot (1 - \\gamma) + n \\cdot \\gamma

       where :math:`\\gamma` is ``decay_rate`` and :math:`n` is the
       clinical normal.

    Parameters
    ----------
    values : np.ndarray
        3-D array of shape ``(patients, nodes, time_steps)`` containing
        the raw (forward-filled) measurements.
    mask : np.ndarray
        3-D boolean array of shape ``(patients, nodes, time_steps)``
        where ``True`` indicates a real (non-imputed) observation.
    node_normals : np.ndarray
        1-D array of shape ``(nodes,)`` with the clinical normal value
        for each node.
    step_hours : float
        Duration of one time step in hours (default 0.25 = 15 min).
    decay_start_hours : float
        Hours of consecutive missingness before decay begins (default 2.0).
    decay_rate : float
        Fraction of the gap toward clinical normal that is closed per time
        step (default 0.1 = 10%).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(values_decayed, deltas)`` where:
        - *values_decayed* is the values array with decay applied in-place.
        - *deltas* is an array of shape ``(patients, nodes, time_steps)``
          tracking the time since the last real observation.

    Raises
    ------
    ValueError
        If ``node_normals`` length does not match the node dimension of
        *values*.

    Notes
    -----
    Extracted verbatim from ``PhysioGraph_External_Pipeline.ipynb``
    Cell 74 (``apply_forward_fill_and_decay``) and
    ``PhysioGraph_HF_Shock.ipynb`` Cell 7 (decay logic inside
    ``sanitize_and_tensorize``).
    """
    if node_normals.shape[0] != values.shape[1]:
        raise ValueError(
            "node_normals length mismatch. "
            f"node_normals={node_normals.shape[0]}, nodes={values.shape[1]}"
        )

    values = values.copy()
    deltas: np.ndarray = np.zeros_like(mask, dtype=np.float32)
    normals = node_normals.reshape(1, -1)

    for t in range(1, values.shape[2]):
        # Accumulate delta since last real observation.
        deltas[:, :, t] = deltas[:, :, t - 1] + step_hours

        # Reset delta to 0 where a real measurement exists.
        seen_now = mask[:, :, t] > 0
        deltas[:, :, t][seen_now] = 0.0

        # Forward-fill: carry forward the previous value where missing.
        missing = mask[:, :, t] == 0
        values[:, :, t][missing] = values[:, :, t - 1][missing]

        # Apply decay after decay_start_hours of consecutive missingness.
        decay_mask = (deltas[:, :, t] >= decay_start_hours) & missing
        if np.any(decay_mask):
            prev_val = values[:, :, t - 1]
            decayed = prev_val * (1.0 - decay_rate) + normals * decay_rate
            values[:, :, t][decay_mask] = decayed[decay_mask]

    return values, deltas


def fill_clinical_normals(
    values: np.ndarray,
    node_names: list[str],
    clinical_normals: dict[str, float],
) -> np.ndarray:
    """Fill any remaining NaN values with clinical normal values.

    This is the final imputation step applied after forward-fill and
    decay.  It guarantees that no NaN values remain in the tensor before
    model input.

    Parameters
    ----------
    values : np.ndarray
        2-D array of shape ``(patients * nodes, time_steps)`` or
        3-D array of shape ``(patients, nodes, time_steps)`` containing
        values that may still contain NaN.
    node_names : list[str]
        Ordered list of node names matching the node dimension of *values*.
    clinical_normals : dict[str, float]
        Mapping from node name to clinical normal value.

    Returns
    -------
    np.ndarray
        *values* with NaN entries replaced by the corresponding clinical
        normal, modified in-place.

    Notes
    -----
    Extracted from ``PhysioGraph_HF_Shock.ipynb`` Cell 7 (the
    loop ``for node in NODE_NAMES`` that fills remaining NaN).
    """
    # Handle 2-D input with shape (patients*nodes, time_steps).
    if values.ndim == 2:
        n_nodes = len(node_names)
        n_total = values.shape[0]
        n_patients = n_total // n_nodes
        for i, node in enumerate(node_names):
            row_slice = slice(i, n_total, n_nodes)
            val = clinical_normals.get(node, 0.0)
            nan_mask = np.isnan(values[row_slice, :])
            if np.any(nan_mask):
                values[row_slice, :][nan_mask] = val
    elif values.ndim == 3:
        for i, node in enumerate(node_names):
            val = clinical_normals.get(node, 0.0)
            nan_mask = np.isnan(values[:, i, :])
            if np.any(nan_mask):
                values[:, i, :][nan_mask] = val
    else:
        raise ValueError(
            f"Expected 2-D or 3-D array, got shape {values.shape}"
        )

    return values


def compute_missingness_flags_df(
    observation_events_df: "pd.DataFrame",  # type: ignore[valid-type]
    variables: list[str],
) -> "pd.DataFrame":  # type: ignore[valid-type]
    """Compute per-stay missingness flags for each clinical variable.

    For each (stay_id, variable) pair a missingness type is assigned
    using :func:`classify_missingness`.  The result can be merged with
    the feature table to enable sensitivity analysis.

    Parameters
    ----------
    observation_events_df : pd.DataFrame
        Events DataFrame filtered to the observation window.  Must
        contain columns ``stay_id``, ``concept``, and ``value_numeric``.
    variables : list[str]
        List of clinical variables to report missingness for.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``stay_id`` and one column per variable
        containing the missingness classification string.
    """
    import pandas as pd

    # Group by stay and concept to count measurements.
    if "concept" not in observation_events_df.columns:
        raise ValueError(
            "observation_events_df must have a 'concept' column"
        )

    counts = (
        observation_events_df.groupby(["stay_id", "concept"], sort=False)
        .size()
        .reset_index(name="n_measurements")
    )
    counts_pivot = counts.pivot(
        index="stay_id", columns="concept", values="n_measurements"
    ).fillna(0).astype(int)

    # Build a DataFrame with missingness classifications.
    result = pd.DataFrame(
        {"stay_id": counts_pivot.index.astype(int)}
    )

    for var in variables:
        if var in counts_pivot.columns:
            result[f"{var}_missingness"] = counts_pivot[var].apply(
                lambda n: classify_missingness(var, n)
            )
        else:
            result[f"{var}_missingness"] = classify_missingness(var, 0)

    return result
