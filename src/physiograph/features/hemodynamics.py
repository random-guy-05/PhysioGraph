"""Hemodynamic feature extraction: hypotension, tachycardia, and
time-below-threshold fractions.

All computation logic is extracted verbatim from the PhysioGraph
notebooks and preserves the exact thresholds and interaction terms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hypotension flags
# ---------------------------------------------------------------------------


def compute_hypotension_flag(
    sbp_min: pd.Series,
    map_min: pd.Series,
    sbp_threshold: float = 90.0,
    map_threshold: float = 65.0,
) -> pd.Series:
    """Flag stays with hypotension (SBP < 90 mmHg OR MAP < 65 mmHg).

    Parameters
    ----------
    sbp_min : pd.Series
        Minimum systolic blood pressure in observation window.
    map_min : pd.Series
        Minimum mean arterial pressure in observation window.
    sbp_threshold : float
        SBP hypotension threshold (default 90 mmHg).
    map_threshold : float
        MAP hypotension threshold (default 65 mmHg).

    Returns
    -------
    pd.Series
        Binary flag (0/1) as integer.
    """
    return (
        ((sbp_min < sbp_threshold) | (map_min < map_threshold))
        .fillna(False)
        .astype(int)
    )


def compute_severe_hypotension_flag(
    sbp_min: pd.Series,
    map_min: pd.Series,
    sbp_threshold: float = 80.0,
    map_threshold: float = 60.0,
) -> pd.Series:
    """Flag stays with severe hypotension (SBP < 80 mmHg OR MAP < 60 mmHg).

    Parameters
    ----------
    sbp_min : pd.Series
        Minimum systolic blood pressure in observation window.
    map_min : pd.Series
        Minimum mean arterial pressure in observation window.
    sbp_threshold : float
        Severe SBP threshold (default 80 mmHg).
    map_threshold : float
        Severe MAP threshold (default 60 mmHg).

    Returns
    -------
    pd.Series
        Binary flag (0/1) as integer.
    """
    return (
        ((sbp_min < sbp_threshold) | (map_min < map_threshold))
        .fillna(False)
        .astype(int)
    )


# ---------------------------------------------------------------------------
# Tachycardia flag
# ---------------------------------------------------------------------------


def compute_tachycardia_flag(
    hr_max: pd.Series,
    threshold: float = 100.0,
) -> pd.Series:
    """Flag stays with tachycardia (HR >= 100 bpm).

    Parameters
    ----------
    hr_max : pd.Series
        Maximum heart rate in observation window.
    threshold : float
        Tachycardia threshold in bpm (default 100).

    Returns
    -------
    pd.Series
        Binary flag (0/1) as integer.
    """
    return (hr_max >= threshold).fillna(False).astype(int)


# ---------------------------------------------------------------------------
# Time-below / time-above threshold fractions
# ---------------------------------------------------------------------------


def compute_time_below_65_fraction(
    map_measurements: pd.Series,
) -> float:
    """Compute fraction of MAP measurements below 65 mmHg.

    Parameters
    ----------
    map_measurements : pd.Series
        MAP values for a single stay.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if len(map_measurements) == 0:
        return 0.0
    return float((map_measurements < 65.0).mean())


def compute_time_below_90_fraction(
    sbp_measurements: pd.Series,
) -> float:
    """Compute fraction of SBP measurements below 90 mmHg.

    Parameters
    ----------
    sbp_measurements : pd.Series
        SBP values for a single stay.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if len(sbp_measurements) == 0:
        return 0.0
    return float((sbp_measurements < 90.0).mean())


def compute_time_above_100_fraction(
    hr_measurements: pd.Series,
) -> float:
    """Compute fraction of HR measurements above 100 bpm.

    Parameters
    ----------
    hr_measurements : pd.Series
        HR values for a single stay.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if len(hr_measurements) == 0:
        return 0.0
    return float((hr_measurements >= 100.0).mean())


def compute_measurement_fraction_series(
    ordered_events: pd.DataFrame,
    variable: str,
    comparator: "callable",  # type: ignore[type-arg]
) -> pd.Series:
    """Compute per-stay fraction of measurements satisfying a comparator.

    This replicates the ``measurement_fraction`` inner function from the
    notebook's ``build_feature_table`` / ``summarize_observation_window``.

    Parameters
    ----------
    ordered_events : pd.DataFrame
        Events DataFrame with columns ``stay_id``, ``concept``,
        ``value_numeric``.  Should already be filtered to the observation
        window.
    variable : str
        Clinical variable name (e.g. ``"map"``).
    comparator : callable
        A function applied to ``value_numeric`` that returns a boolean
        Series or array (e.g. ``lambda s: s < 65.0``).

    Returns
    -------
    pd.Series
        Per-stay fraction indexed by ``stay_id``.  Stays with no
        measurements of *variable* will have NaN.

    Notes
    -----
    This function uses ``.groupby("stay_id")["flag"].mean()``, exactly
    matching the notebook computation.  Stays without any measurement
    of the variable receive NaN and should be filled downstream (the
    notebooks fill with 0.0 for these fractions).
    """
    sub = ordered_events.loc[
        ordered_events["concept"] == variable, ["stay_id", "value_numeric"]
    ].copy()
    if sub.empty:
        unique_stays = ordered_events["stay_id"].unique()
        return pd.Series(np.nan, index=pd.Index(unique_stays, name="stay_id"))
    frac = (
        sub.assign(flag=comparator(sub["value_numeric"]).astype(float))
        .groupby("stay_id")["flag"]
        .mean()
    )
    return frac


# ---------------------------------------------------------------------------
# Lactate-hemodynamics interaction ratios
# ---------------------------------------------------------------------------


def compute_lactate_map_ratio(
    baseline_lactate: pd.Series,
    baseline_map: pd.Series,
    map_clip_lower: float = 35.0,
) -> pd.Series:
    """Compute lactate / MAP ratio.

    MAP is clipped at *map_clip_lower* to prevent division by extremely
    low values (e.g. MAP of 0 from continuous monitoring artifact).

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate (mmol/L).
    baseline_map : pd.Series
        Baseline MAP (mmHg).
    map_clip_lower : float
        Minimum MAP value to use (default 35 mmHg).

    Returns
    -------
    pd.Series
        Lactate / MAP ratio.  Infinite values replaced with NaN.
    """
    ratio = baseline_lactate / baseline_map.clip(lower=map_clip_lower)
    return ratio.replace([np.inf, -np.inf], np.nan)


def compute_lactate_sbp_ratio(
    baseline_lactate: pd.Series,
    baseline_sbp: pd.Series,
    sbp_clip_lower: float = 60.0,
) -> pd.Series:
    """Compute lactate / SBP ratio.

    SBP is clipped at *sbp_clip_lower* to prevent division by extremely
    low values.

    Parameters
    ----------
    baseline_lactate : pd.Series
        Baseline lactate (mmol/L).
    baseline_sbp : pd.Series
        Baseline SBP (mmHg).
    sbp_clip_lower : float
        Minimum SBP value to use (default 60 mmHg).

    Returns
    -------
    pd.Series
        Lactate / SBP ratio.  Infinite values replaced with NaN.
    """
    ratio = baseline_lactate / baseline_sbp.clip(lower=sbp_clip_lower)
    return ratio.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Heart rate banding
# ---------------------------------------------------------------------------

DEFAULT_HR_BAND_EDGES: list[float] = [-np.inf, 100.0, 120.0, np.inf]
DEFAULT_HR_BAND_LABELS: list[str] = ["<100", "100-119", ">=120"]

DEFAULT_HR_FINE_BAND_EDGES: list[float] = [
    -np.inf, 100.0, 110.0, 120.0, 140.0, np.inf
]
DEFAULT_HR_FINE_BAND_LABELS: list[str] = [
    "<100", "100-109", "110-119", "120-139", ">=140"
]


def bin_heart_rate(
    hr_values: pd.Series,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    right: bool = False,
) -> pd.Series:
    """Categorise heart rate into clinical bands.

    Parameters
    ----------
    hr_values : pd.Series
        Heart rate values (bpm).  Typically ``baseline_hr.combine_first(hr_max)``.
    bin_edges : list[float], optional
        Bin edges (default: ``[-inf, 100, 120, inf]`` = coarse bands).
    bin_labels : list[str], optional
        Bin labels (default: ``["<100", "100-119", ">=120"]``).
    right : bool
        Whether intervals are right-closed.

    Returns
    -------
    pd.Series
        Categorical Series with missing values filled as ``"missing"``.
    """
    edges = bin_edges if bin_edges is not None else DEFAULT_HR_BAND_EDGES
    labels = bin_labels if bin_labels is not None else DEFAULT_HR_BAND_LABELS
    result = pd.cut(hr_values, bins=edges, labels=labels, right=right)
    return result.astype("object").fillna("missing")


def bin_heart_rate_fine(
    hr_values: pd.Series,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    right: bool = False,
) -> pd.Series:
    """Categorise heart rate into fine-grained bands.

    Parameters
    ----------
    hr_values : pd.Series
        Heart rate values (bpm).
    bin_edges : list[float], optional
        Bin edges (default: ``[-inf, 100, 110, 120, 140, inf]``).
    bin_labels : list[str], optional
        Bin labels (default: ``["<100", "100-109", "110-119", "120-139", ">=140"]``).
    right : bool
        Whether intervals are right-closed.

    Returns
    -------
    pd.Series
        Categorical Series with missing values filled as ``"missing"``.
    """
    edges = bin_edges if bin_edges is not None else DEFAULT_HR_FINE_BAND_EDGES
    labels = bin_labels if bin_labels is not None else DEFAULT_HR_FINE_BAND_LABELS
    result = pd.cut(hr_values, bins=edges, labels=labels, right=right)
    return result.astype("object").fillna("missing")


# ---------------------------------------------------------------------------
# Perfusion burden score
# ---------------------------------------------------------------------------


def compute_perfusion_burden_score(
    modifier_burden: pd.Series,
    renal_hypoperfusion_flag: pd.Series,
    hepatic_hypoperfusion_flag: pd.Series,
    lactate_ge_3_1_flag: pd.Series,
) -> pd.Series:
    """Compute perfusion burden score as sum of clinical indicators.

    Parameters
    ----------
    modifier_burden : pd.Series
        SCAI modifier burden count (tachycardia + hypotension + acidemia).
    renal_hypoperfusion_flag : pd.Series
        Renal hypoperfusion flag (creatinine >= 2.0).
    hepatic_hypoperfusion_flag : pd.Series
        Hepatic hypoperfusion flag (bilirubin >= 2.0).
    lactate_ge_3_1_flag : pd.Series
        Lactate >= 3.1 flag.

    Returns
    -------
    pd.Series
        Perfusion burden score as float.
    """
    return (
        modifier_burden
        + renal_hypoperfusion_flag
        + hepatic_hypoperfusion_flag
        + lactate_ge_3_1_flag
    ).astype(float)
