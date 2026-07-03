"""Lactate-specific feature extraction, binning, and clearance metrics.

All functions are extracted from the PhysioGraph notebooks and preserve
the exact computation logic (bin edges, thresholds, interaction terms).

Constants (bin edges, labels, thresholds) default to the values in
``configs/default.yaml`` but can be overridden at call time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default constants (mirror configs/default.yaml)
# ---------------------------------------------------------------------------

DEFAULT_LACTATE_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.1, 5.0, np.inf]
DEFAULT_LACTATE_BIN_LABELS: list[str] = ["<2", "2-3.1", "3.1-5", ">=5"]

DEFAULT_FOCUS_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.0, 5.0, np.inf]
DEFAULT_FOCUS_BIN_LABELS: list[str] = ["<2", "2-3", "3-5", ">=5"]

DEFAULT_SEVERE_ACIDEMIA_PH: float = 7.20
"""pH threshold for severe acidemia in the lactate-SCAI modifier."""


# ---------------------------------------------------------------------------
# Lactate binning
# ---------------------------------------------------------------------------


def bin_lactate(
    baseline_lactate: pd.Series,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    right: bool = False,
) -> pd.Series:
    """Categorise baseline lactate into clinical bins.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Series of baseline lactate values (mmol/L).
    bin_edges : list[float], optional
        Bin edge list (default: ``[-inf, 2, 3.1, 5, inf]``).
    bin_labels : list[str], optional
        Bin labels (default: ``["<2", "2-3.1", "3.1-5", ">=5"]``).
    right : bool
        Whether intervals are right-closed (default ``False`` = left-closed
        ``[a, b)`` to match ``pd.cut(right=False)`` in notebooks).

    Returns
    -------
    pd.Series
        Categorical Series with missing values filled as ``"missing"``.
    """
    edges = bin_edges if bin_edges is not None else DEFAULT_LACTATE_BIN_EDGES
    labels = bin_labels if bin_labels is not None else DEFAULT_LACTATE_BIN_LABELS
    result = pd.cut(
        baseline_lactate,
        bins=edges,
        labels=labels,
        right=right,
    )
    return result.astype("object").fillna("missing")


def bin_lactate_focus(
    baseline_lactate: pd.Series,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    right: bool = False,
) -> pd.Series:
    """Categorise baseline lactate using the fine-grained "focus" bins.

    Default bin edges are ``[-inf, 2, 3, 5, inf]``, providing a narrower
    2-3 mmol/L subgroup used for sensitivity analysis.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Series of baseline lactate values (mmol/L).
    bin_edges : list[float], optional
        Bin edge list (default: ``[-inf, 2, 3, 5, inf]``).
    bin_labels : list[str], optional
        Bin labels (default: ``["<2", "2-3", "3-5", ">=5"]``).
    right : bool
        Whether intervals are right-closed.

    Returns
    -------
    pd.Series
        Categorical Series with missing values filled as ``"missing"``.
    """
    edges = bin_edges if bin_edges is not None else DEFAULT_FOCUS_BIN_EDGES
    labels = bin_labels if bin_labels is not None else DEFAULT_FOCUS_BIN_LABELS
    result = pd.cut(
        baseline_lactate,
        bins=edges,
        labels=labels,
        right=right,
    )
    return result.astype("object").fillna("missing")


# ---------------------------------------------------------------------------
# Lactate dynamics: clearance, slope, persistence
# ---------------------------------------------------------------------------


def compute_lactate_clearance_4h_pct(
    lactate_first: pd.Series,
    lactate_last: pd.Series,
    valid_first_mask: pd.Series | None = None,
) -> pd.Series:
    """Compute 4-hour lactate clearance as percentage of first value.

    .. math::

       \\text{clearance} = \\frac{\\text{first} - \\text{last}}{\\text{first}} \\times 100

    Parameters
    ----------
    lactate_first : pd.Series
        First lactate measurement in observation window.
    lactate_last : pd.Series
        Last lactate measurement in observation window.
    valid_first_mask : pd.Series, optional
        Boolean mask for rows where ``lactate_first`` is valid (> 0).  If
        ``None``, the mask is computed as ``lactate_first > 0``.

    Returns
    -------
    pd.Series
        Clearance percentage.  Infinite values are replaced with NaN.
    """
    valid = lactate_first.where(
        valid_first_mask if valid_first_mask is not None else (lactate_first > 0)
    )
    clearance = ((valid - lactate_last) / valid) * 100.0
    return clearance.replace([np.inf, -np.inf], np.nan)


def compute_lactate_slope_per_hr(
    lactate_delta: pd.Series,
    lactate_time_span_hours: pd.Series,
    min_time_span_hours: float = 0.25,
) -> pd.Series:
    """Compute lactate slope in mmol/L per hour.

    Parameters
    ----------
    lactate_delta : pd.Series
        Difference (last - first) in mmol/L.
    lactate_time_span_hours : pd.Series
        Hours between first and last lactate measurement.
    min_time_span_hours : float
        Minimum time span to avoid division by zero (default 0.25 = 15 min).

    Returns
    -------
    pd.Series
        Slope in mmol/L/hr.  Infinite values replaced with NaN.
    """
    span = lactate_time_span_hours.clip(lower=min_time_span_hours)
    slope = lactate_delta / span
    return slope.replace([np.inf, -np.inf], np.nan)


def compute_persistent_lactate_flag(
    lactate_count: pd.Series,
    lactate_first: pd.Series,
    lactate_last: pd.Series,
    lactate_mean: pd.Series,
    lactate_delta: pd.Series,
) -> pd.Series:
    """Flag stays with persistent lactate elevation.

    A stay is flagged if EITHER:

    1. At least 2 lactate measurements, first >= 2.0, last >= 2.0
    2. At least 2 lactate measurements, mean >= 2.5, and delta > 0.3 (rising)

    Parameters
    ----------
    lactate_count : pd.Series
        Number of lactate measurements in observation window.
    lactate_first : pd.Series
        First lactate measurement value.
    lactate_last : pd.Series
        Last lactate measurement value.
    lactate_mean : pd.Series
        Mean lactate across all measurements.
    lactate_delta : pd.Series
        Lactate change (last - first).

    Returns
    -------
    pd.Series
        Binary flag (0/1) as integer.
    """
    count_filled = lactate_count.fillna(0)
    has_multiple = count_filled >= 2

    condition_a = (
        has_multiple & (lactate_first >= 2.0) & (lactate_last >= 2.0)
    )
    condition_b = (
        has_multiple & (lactate_mean >= 2.5) & (lactate_delta > 0.3)
    )
    return (condition_a | condition_b).fillna(False).astype(int)


# ---------------------------------------------------------------------------
# Lactate threshold flags
# ---------------------------------------------------------------------------


def compute_lactate_threshold_flags(
    baseline_lactate: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute binary flags for lactate exceeding clinical thresholds.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline (last) lactate value.
    thresholds : dict[str, float], optional
        Mapping ``{flag_suffix: threshold}``.  Default::

            {"ge_2": 2.0, "ge_3_1": 3.1, "ge_5": 5.0}

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``lactate_ge_2_flag``,
        ``lactate_ge_3_1_flag``, ``lactate_ge_5_flag``.
    """
    if thresholds is None:
        thresholds = {"ge_2": 2.0, "ge_3_1": 3.1, "ge_5": 5.0}
    result = {}
    for suffix, thresh in thresholds.items():
        result[f"lactate_{suffix}_flag"] = (
            (baseline_lactate >= thresh).fillna(False).astype(int)
        )
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Lactate-pH interaction (SCAI modifier)
# ---------------------------------------------------------------------------


def compute_lactate_scai_modifier(
    baseline_lactate: pd.Series,
    ph_min: pd.Series,
    severe_acidemia_ph: float = DEFAULT_SEVERE_ACIDEMIA_PH,
) -> pd.Series:
    """Classify the lactate-pH interaction for SCAI staging.

    Categories (in priority order):

    * ``"lactate>=5 & pH<7.2"`` — both severe
    * ``"lactate>=5 only"``
    * ``"pH<7.2 only"``
    * ``"neither"``

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate value.
    ph_min : pd.Series
        Minimum pH value in observation window.
    severe_acidemia_ph : float
        pH threshold for severe acidemia (default 7.20).

    Returns
    -------
    pd.Series
        Categorical modifier label.
    """
    is_high_lactate = baseline_lactate >= 5.0
    is_severe_acidemia = ph_min < severe_acidemia_ph

    return np.select(
        [
            is_high_lactate & is_severe_acidemia,
            is_high_lactate,
            is_severe_acidemia,
        ],
        [
            "lactate>=5 & pH<7.2",
            "lactate>=5 only",
            "pH<7.2 only",
        ],
        default="neither",
    )


# ---------------------------------------------------------------------------
# Lactate interaction terms
# ---------------------------------------------------------------------------


def compute_lactate_acidemia_interaction(
    baseline_lactate: pd.Series,
    acidemia_flag: pd.Series,
) -> pd.Series:
    """Compute lactate × acidemia interaction term.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate values.  NaN is filled with 0.0 before multiplication.
    acidemia_flag : pd.Series
        Binary acidemia flag (0/1).

    Returns
    -------
    pd.Series
        Interaction product.
    """
    return baseline_lactate.fillna(0.0) * acidemia_flag


def compute_lactate_hypotension_interaction(
    baseline_lactate: pd.Series,
    hypotension_flag: pd.Series,
) -> pd.Series:
    """Compute lactate × hypotension interaction term.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate values.  NaN is filled with 0.0.
    hypotension_flag : pd.Series
        Binary hypotension flag (0/1).

    Returns
    -------
    pd.Series
        Interaction product.
    """
    return baseline_lactate.fillna(0.0) * hypotension_flag


def compute_lactate_tachycardia_interaction(
    baseline_lactate: pd.Series,
    tachycardia_flag: pd.Series,
) -> pd.Series:
    """Compute lactate × tachycardia interaction term.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate values.  NaN is filled with 0.0.
    tachycardia_flag : pd.Series
        Binary tachycardia flag (0/1).

    Returns
    -------
    pd.Series
        Interaction product.
    """
    return baseline_lactate.fillna(0.0) * tachycardia_flag


def compute_occult_hypoperfusion_flag(
    lactate_ge_2_flag: pd.Series,
    hypotension_flag: pd.Series,
) -> pd.Series:
    """Flag occult hypoperfusion: elevated lactate without hypotension.

    Parameters
    ----------
    lactate_ge_2_flag : pd.Series
        Binary flag for lactate >= 2.0.
    hypotension_flag : pd.Series
        Binary flag for hypotension.

    Returns
    -------
    pd.Series
        Binary flag (0/1) as integer.
    """
    return ((lactate_ge_2_flag == 1) & (hypotension_flag == 0)).astype(int)
