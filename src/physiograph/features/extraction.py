"""Landmark-aware feature extraction from clinical events.

Provides the primary feature engineering pipeline that aggregates
observation-window events into per-stay baseline features, clinical
flags, interaction terms, and SCAI shock stages.

The core function :func:`build_feature_table` is extracted verbatim from
``PhysioGraph_External_Pipeline.ipynb`` Cell 28 and preserves the exact
computation logic (thresholds, bin edges, interaction definitions).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from physiograph.features.hemodynamics import (
    bin_heart_rate,
    bin_heart_rate_fine,
    compute_hypotension_flag,
    compute_lactate_map_ratio,
    compute_lactate_sbp_ratio,
    compute_measurement_fraction_series,
    compute_perfusion_burden_score,
    compute_severe_hypotension_flag,
    compute_tachycardia_flag,
)
from physiograph.features.lactate import (
    bin_lactate,
    bin_lactate_focus,
    compute_lactate_acidemia_interaction,
    compute_lactate_clearance_4h_pct,
    compute_lactate_hypotension_interaction,
    compute_lactate_scai_modifier,
    compute_lactate_slope_per_hr,
    compute_lactate_tachycardia_interaction,
    compute_occult_hypoperfusion_flag,
    compute_persistent_lactate_flag,
    compute_lactate_threshold_flags,
)
from physiograph.guards import assert_no_feature_leakage_columns, assert_observation_only

# ---------------------------------------------------------------------------
# Default constants (mirror configs/default.yaml)
# ---------------------------------------------------------------------------

DEFAULT_PORTABLE_VARIABLES: list[str] = [
    "lactate", "hr", "sbp", "map", "ph", "creatinine",
    "bilirubin_total", "spo2", "resp_rate", "temp",
]

DEFAULT_LACTATE_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.1, 5.0, np.inf]
DEFAULT_LACTATE_BIN_LABELS: list[str] = ["<2", "2-3.1", "3.1-5", ">=5"]
DEFAULT_FOCUS_BIN_EDGES: list[float] = [-np.inf, 2.0, 3.0, 5.0, np.inf]
DEFAULT_FOCUS_BIN_LABELS: list[str] = ["<2", "2-3", "3-5", ">=5"]

DEFAULT_HR_FINE_BAND_EDGES: list[float] = [-np.inf, 100.0, 110.0, 120.0, 140.0, np.inf]
DEFAULT_HR_FINE_BAND_LABELS: list[str] = ["<100", "100-109", "110-119", "120-139", ">=140"]

DEFAULT_SCAI_STAGE_LABELS: list[str] = ["A", "B", "C", "D", "E"]

DEFAULT_TIME_STEP_HOURS: float = 0.25

DEFAULT_PRIMARY_FEATURE_COLUMNS: list[str] = [
    "dataset", "stay_id", "age", "is_male", "cohort_hf_flag", "shock_icd_flag",
    "baseline_lactate", "baseline_hr", "baseline_sbp", "baseline_map",
    "baseline_ph", "baseline_creatinine", "baseline_bilirubin_total",
    "baseline_spo2", "baseline_resp_rate", "baseline_temp",
    "tachycardia_flag", "hypotension_flag", "severe_hypotension_flag",
    "acidemia_flag", "severe_acidemia_flag", "modifier_burden",
    "lactate_delta", "lactate_slope_per_hr", "lactate_clearance_4h_pct",
    "persistent_lactate_flag", "renal_hypoperfusion_flag",
    "hepatic_hypoperfusion_flag",
    "lactate_ge_2_flag", "lactate_ge_3_1_flag", "lactate_ge_5_flag",
    "map_below_65_fraction", "sbp_below_90_fraction",
    "hr_above_100_fraction", "ph_below_7_25_fraction",
    "lactate_map_ratio", "lactate_sbp_ratio",
    "lactate_acidemia_interaction", "lactate_hypotension_interaction",
    "lactate_tachycardia_interaction",
    "occult_hypoperfusion_flag", "perfusion_burden_score",
    "baseline_lactate_bin", "baseline_lactate_focus_bin",
    "hr_band", "hr_band_fine",
    "lactate_scai_modifier",
    "scai_stage", "scai_stage_num", "scai_stage_collapsed",
]


# ---------------------------------------------------------------------------
# Helpers (extracted from notebook inner functions)
# ---------------------------------------------------------------------------


def _col(summary: pd.DataFrame, name: str) -> pd.Series:
    """Return a column from *summary* or a NaN Series if missing."""
    if name in summary.columns:
        return summary[name]
    return pd.Series(np.nan, index=summary.index)


def _measurement_fraction(
    ordered: pd.DataFrame,
    summary: pd.DataFrame,
    variable: str,
    comparator: Callable[[pd.Series], pd.Series | np.ndarray],
) -> pd.Series:
    """Compute per-stay fraction of measurements satisfying *comparator*.

    Exact replica of the ``measurement_fraction`` closure inside
    ``build_feature_table`` (External Pipeline Cell 28).
    """
    sub = ordered.loc[
        ordered["concept"] == variable, ["stay_id", "value_numeric"]
    ].copy()
    if sub.empty:
        return pd.Series(np.nan, index=summary.index)
    frac = (
        sub.assign(flag=comparator(sub["value_numeric"]).astype(float))
        .groupby("stay_id")["flag"]
        .mean()
    )
    return summary["stay_id"].map(frac)


def _assign_scai_stage(row: pd.Series, stage_labels: list[str]) -> str:
    """Classify a single stay row into a SCAI shock stage.

    Exact replica of the ``assign_scai_stage`` inner function from
    ``build_feature_table`` (External Pipeline Cell 28).
    """
    severe_acidemia = bool(row.get("severe_acidemia_flag", False))
    severe_hypotension = bool(row.get("severe_hypotension_flag", False))
    modifier_burden = int(row.get("modifier_burden", 0))
    persistent_lactate = bool(row.get("persistent_lactate_flag", False))
    lactate = row.get("baseline_lactate", np.nan)
    has_elevated_lactate = pd.notna(lactate) and lactate >= 2.0
    acidemia = bool(row.get("acidemia_flag", False))
    renal = bool(row.get("renal_hypoperfusion_flag", False))
    hepatic = bool(row.get("hepatic_hypoperfusion_flag", False))
    tachycardia = bool(row.get("tachycardia_flag", False))
    hypotension = bool(row.get("hypotension_flag", False))

    if severe_acidemia or (severe_hypotension and modifier_burden >= 2):
        return stage_labels[4]  # E
    if persistent_lactate or (has_elevated_lactate and modifier_burden >= 2):
        return stage_labels[3]  # D
    if has_elevated_lactate or acidemia or renal or hepatic:
        return stage_labels[2]  # C
    if tachycardia or hypotension:
        return stage_labels[1]  # B
    return stage_labels[0]  # A


# ---------------------------------------------------------------------------
# Legacy helper (Fahrenheit → Celsius conversion from notebook Cell 26)
# ---------------------------------------------------------------------------


def _sanitize_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Clean events DataFrame before feature extraction.

    Currently only converts Fahrenheit temperatures (> 50°F) to Celsius.
    Extracted from ``PhysioGraph_External_Pipeline.ipynb`` Cell 26.
    """
    events_df = events_df.copy()
    temperature_mask = (
        (events_df["concept"] == "temp")
        & (events_df["value_numeric"] > 50)
    )
    events_df.loc[temperature_mask, "value_numeric"] = (
        events_df.loc[temperature_mask, "value_numeric"] - 32.0
    ) * (5.0 / 9.0)
    return events_df


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def _extract_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Extract feature-engineering constants from a config dict.

    Returns a flat dict with defaults for any missing key, so callers
    can pass ``None`` and get the notebook defaults.
    """
    if cfg is None:
        cfg = {}

    return {
        "variables": cfg.get(
            "portable_context_variables", DEFAULT_PORTABLE_VARIABLES
        ),
        "time_step_hours": cfg.get(
            "time_step_hours", DEFAULT_TIME_STEP_HOURS
        ),
        "lactate_bin_edges": cfg.get(
            "lactate_bin_edges", DEFAULT_LACTATE_BIN_EDGES
        ),
        "lactate_bin_labels": cfg.get(
            "lactate_bin_labels", DEFAULT_LACTATE_BIN_LABELS
        ),
        "focus_bin_edges": cfg.get(
            "focus_lactate_bin_edges", DEFAULT_FOCUS_BIN_EDGES
        ),
        "focus_bin_labels": cfg.get(
            "focus_lactate_bin_labels", DEFAULT_FOCUS_BIN_LABELS
        ),
        "hr_fine_band_edges": cfg.get(
            "hr_fine_band_edges", DEFAULT_HR_FINE_BAND_EDGES
        ),
        "hr_fine_band_labels": cfg.get(
            "hr_fine_band_labels", DEFAULT_HR_FINE_BAND_LABELS
        ),
        "scai_stage_labels": cfg.get(
            "scai_stage_labels", DEFAULT_SCAI_STAGE_LABELS
        ),
        "primary_feature_columns": cfg.get(
            "primary_feature_columns", DEFAULT_PRIMARY_FEATURE_COLUMNS
        ),
    }


# ---------------------------------------------------------------------------
# build_feature_table
# ---------------------------------------------------------------------------


def build_feature_table(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate observation-window events into a per-stay feature matrix.

    This is the primary feature engineering function for the PhysioGraph
    pipeline.  It computes 50+ features per ICU stay from raw clinical
    events observed during the landmark window (first 4 hours).

    Features computed (in order):

    **Baseline values** (last observation, fallback to max/min as
    appropriate):
    lactate, hr, sbp, map, ph, creatinine, bilirubin_total, spo2,
    resp_rate, temp.

    **Clinical flags:**
    tachycardia (HR >= 100), hypotension (SBP < 90 or MAP < 65),
    severe hypotension (SBP < 80 or MAP < 60), acidemia (pH < 7.25),
    severe acidemia (pH < 7.20).

    **Lactate dynamics:**
    delta, slope per hour, 4-hour clearance %, persistent elevation flag,
    threshold flags (>=2, >=3.1, >=5).

    **End-organ flags:**
    renal hypoperfusion (creatinine >= 2.0), hepatic hypoperfusion
    (bilirubin >= 2.0).

    **Time-below-threshold fractions:**
    MAP < 65, SBP < 90, HR > 100, pH < 7.25.

    **Interaction terms:**
    lactate/MAP ratio, lactate/SBP ratio,
    lactate × acidemia, lactate × hypotension, lactate × tachycardia.

    **Composite scores:**
    SCAI modifier burden, occult hypoperfusion, perfusion burden.

    **Categorical bins:**
    lactate bins (standard + focus), HR bands (coarse + fine),
    lactate-SCAI modifier.

    **SCAI staging:**
    Stage letter (A-E), stage number (1-5), collapsed stage (A, B/C, D/E).

    Parameters
    ----------
    events_df : pd.DataFrame
        Long-format events DataFrame.  Must contain columns
        ``stay_id``, ``concept``, ``value_numeric``, ``offset_minutes``,
        and ``window``.  The ``window`` column must contain only
        ``"observation"`` rows (non-observation rows raise an error).
    cohort_df : pd.DataFrame
        Cohort metadata DataFrame.  Must contain columns ``dataset``,
        ``stay_id``, ``age``, ``is_male``, ``cohort_hf_flag``,
        ``shock_icd_flag``.
    config : dict, optional
        Configuration dict (as returned by :func:`physiograph.config.load_config`).
        If ``None``, all constants default to the notebook values.

    Returns
    -------
    pd.DataFrame
        Feature matrix with columns matching
        :data:`DEFAULT_PRIMARY_FEATURE_COLUMNS` (subsetted from all
        computed columns).  Indexed by row number; ``stay_id`` is an
        ordinary column.

    Raises
    ------
    ValueError
        If observation-only assertion fails (non-observation windows
        present) or if forbidden columns appear in the final feature set.

    Notes
    -----
    This function is extracted verbatim from
    ``PhysioGraph_External_Pipeline.ipynb`` Cell 28.  The computation
    logic (thresholds, bin edges, fallback rules, SCAI stage rules) is
    preserved exactly as in the notebook.
    """
    par = _extract_config(config)

    # Sanitize and filter to observation-only events.
    events_df = _sanitize_events(events_df)
    observation_events = events_df.loc[
        (events_df["window"] == "observation")
        & events_df["concept"].isin(par["variables"])
    ].copy()
    assert_observation_only(observation_events)

    # Start with cohort demographics.
    summary = cohort_df[[
        "dataset", "stay_id", "age", "is_male",
        "cohort_hf_flag", "shock_icd_flag",
    ]].copy()
    ordered = observation_events.sort_values(
        ["stay_id", "concept", "offset_minutes"]
    ).copy()

    # Per-variable statistics.
    for variable in par["variables"]:
        sub = ordered.loc[
            ordered["concept"] == variable,
            ["stay_id", "value_numeric", "offset_minutes"],
        ].copy()
        if sub.empty:
            continue
        grouped = sub.groupby("stay_id", sort=False)
        stats = pd.DataFrame({
            "stay_id": grouped["value_numeric"].mean().index.astype(int),
            f"{variable}_count": grouped["value_numeric"].count().values,
            f"{variable}_first": grouped["value_numeric"].first().values,
            f"{variable}_last": grouped["value_numeric"].last().values,
            f"{variable}_min": grouped["value_numeric"].min().values,
            f"{variable}_max": grouped["value_numeric"].max().values,
            f"{variable}_mean": grouped["value_numeric"].mean().values,
            f"{variable}_first_time": (
                grouped["offset_minutes"].first().values / 60.0
            ),
            f"{variable}_last_time": (
                grouped["offset_minutes"].last().values / 60.0
            ),
        })
        summary = summary.merge(stats, on="stay_id", how="left")

    # Baseline values (last → fallback).
    summary["baseline_lactate"] = (
        _col(summary, "lactate_last").combine_first(_col(summary, "lactate_max"))
    )
    summary["baseline_hr"] = (
        _col(summary, "hr_last").combine_first(_col(summary, "hr_max"))
    )
    summary["baseline_sbp"] = (
        _col(summary, "sbp_last").combine_first(_col(summary, "sbp_min"))
    )
    summary["baseline_map"] = (
        _col(summary, "map_last").combine_first(_col(summary, "map_min"))
    )
    summary["baseline_ph"] = (
        _col(summary, "ph_last").combine_first(_col(summary, "ph_min"))
    )
    summary["baseline_creatinine"] = (
        _col(summary, "creatinine_last").combine_first(
            _col(summary, "creatinine_max")
        )
    )
    summary["baseline_bilirubin_total"] = (
        _col(summary, "bilirubin_total_last").combine_first(
            _col(summary, "bilirubin_total_max")
        )
    )
    summary["baseline_spo2"] = (
        _col(summary, "spo2_last").combine_first(_col(summary, "spo2_mean"))
    )
    summary["baseline_resp_rate"] = (
        _col(summary, "resp_rate_last").combine_first(
            _col(summary, "resp_rate_mean")
        )
    )
    summary["baseline_temp"] = (
        _col(summary, "temp_last").combine_first(_col(summary, "temp_mean"))
    )

    # Clinical flags.
    summary["tachycardia_flag"] = compute_tachycardia_flag(
        _col(summary, "hr_max")
    )
    summary["hypotension_flag"] = compute_hypotension_flag(
        _col(summary, "sbp_min"), _col(summary, "map_min")
    )
    summary["severe_hypotension_flag"] = compute_severe_hypotension_flag(
        _col(summary, "sbp_min"), _col(summary, "map_min")
    )
    summary["acidemia_flag"] = (
        (_col(summary, "ph_min") < 7.25).fillna(False).astype(int)
    )
    summary["severe_acidemia_flag"] = (
        (_col(summary, "ph_min") < 7.20).fillna(False).astype(int)
    )
    summary["modifier_burden"] = (
        summary["tachycardia_flag"]
        + summary["hypotension_flag"]
        + summary["acidemia_flag"]
    ).astype(int)

    # Lactate dynamics.
    lactate_span = (
        _col(summary, "lactate_last_time")
        - _col(summary, "lactate_first_time")
    ).clip(lower=par["time_step_hours"])
    summary["lactate_delta"] = (
        _col(summary, "lactate_last") - _col(summary, "lactate_first")
    )
    summary["lactate_slope_per_hr"] = compute_lactate_slope_per_hr(
        summary["lactate_delta"], lactate_span,
        min_time_span_hours=par["time_step_hours"],
    )
    summary["lactate_clearance_4h_pct"] = compute_lactate_clearance_4h_pct(
        _col(summary, "lactate_first"), _col(summary, "lactate_last")
    )
    summary["persistent_lactate_flag"] = compute_persistent_lactate_flag(
        _col(summary, "lactate_count"),
        _col(summary, "lactate_first"),
        _col(summary, "lactate_last"),
        _col(summary, "lactate_mean"),
        summary["lactate_delta"],
    )
    summary["renal_hypoperfusion_flag"] = (
        (_col(summary, "creatinine_max") >= 2.0).fillna(False).astype(int)
    )
    summary["hepatic_hypoperfusion_flag"] = (
        (_col(summary, "bilirubin_total_max") >= 2.0).fillna(False).astype(int)
    )

    # Lactate threshold flags.
    lactate_flags = compute_lactate_threshold_flags(
        summary["baseline_lactate"]
    )
    for col_name in lactate_flags.columns:
        summary[col_name] = lactate_flags[col_name]

    # Time-below-threshold fractions.
    summary["map_below_65_fraction"] = _measurement_fraction(
        ordered, summary, "map", lambda s: s < 65.0
    ).fillna(0.0)
    summary["sbp_below_90_fraction"] = _measurement_fraction(
        ordered, summary, "sbp", lambda s: s < 90.0
    ).fillna(0.0)
    summary["hr_above_100_fraction"] = _measurement_fraction(
        ordered, summary, "hr", lambda s: s >= 100.0
    ).fillna(0.0)
    summary["ph_below_7_25_fraction"] = _measurement_fraction(
        ordered, summary, "ph", lambda s: s < 7.25
    ).fillna(0.0)

    # Interaction ratios.
    summary["lactate_map_ratio"] = compute_lactate_map_ratio(
        summary["baseline_lactate"], summary["baseline_map"]
    )
    summary["lactate_sbp_ratio"] = compute_lactate_sbp_ratio(
        summary["baseline_lactate"], summary["baseline_sbp"]
    )
    summary["lactate_acidemia_interaction"] = (
        compute_lactate_acidemia_interaction(
            summary["baseline_lactate"], summary["acidemia_flag"]
        )
    )
    summary["lactate_hypotension_interaction"] = (
        compute_lactate_hypotension_interaction(
            summary["baseline_lactate"], summary["hypotension_flag"]
        )
    )
    summary["lactate_tachycardia_interaction"] = (
        compute_lactate_tachycardia_interaction(
            summary["baseline_lactate"], summary["tachycardia_flag"]
        )
    )

    # Composite scores.
    summary["occult_hypoperfusion_flag"] = compute_occult_hypoperfusion_flag(
        summary["lactate_ge_2_flag"], summary["hypotension_flag"]
    )
    summary["perfusion_burden_score"] = compute_perfusion_burden_score(
        summary["modifier_burden"],
        summary["renal_hypoperfusion_flag"],
        summary["hepatic_hypoperfusion_flag"],
        summary["lactate_ge_3_1_flag"],
    )

    # Categorical bins.
    summary["baseline_lactate_bin"] = bin_lactate(
        summary["baseline_lactate"],
        bin_edges=par["lactate_bin_edges"],
        bin_labels=par["lactate_bin_labels"],
    )
    summary["baseline_lactate_focus_bin"] = bin_lactate_focus(
        summary["baseline_lactate"],
        bin_edges=par["focus_bin_edges"],
        bin_labels=par["focus_bin_labels"],
    )
    summary["hr_band"] = bin_heart_rate(
        summary["baseline_hr"].combine_first(_col(summary, "hr_max"))
    )
    summary["hr_band_fine"] = bin_heart_rate_fine(
        summary["baseline_hr"].combine_first(_col(summary, "hr_max")),
        bin_edges=par["hr_fine_band_edges"],
        bin_labels=par["hr_fine_band_labels"],
    )

    # Lactate-SCAI modifier.
    summary["lactate_scai_modifier"] = compute_lactate_scai_modifier(
        summary["baseline_lactate"], _col(summary, "ph_min")
    )

    # SCAI staging.
    summary["scai_stage"] = summary.apply(
        lambda r: _assign_scai_stage(r, par["scai_stage_labels"]), axis=1
    )
    summary["scai_stage_num"] = summary["scai_stage"].map({
        stage: idx + 1
        for idx, stage in enumerate(par["scai_stage_labels"])
    })
    summary["scai_stage_collapsed"] = np.where(
        summary["scai_stage"].isin(["D", "E"]),
        "D/E",
        np.where(
            summary["scai_stage"].isin(["B", "C"]),
            "B/C",
            "A",
        ),
    )

    # Subset to primary feature columns and validate.
    features_df = summary[par["primary_feature_columns"]].copy()
    assert_no_feature_leakage_columns(features_df.columns)

    return features_df


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------


def extract_features(
    events_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Alias for :func:`build_feature_table`.

    Provides a discoverable name for the extraction step.  Behaviour is
    identical to ``build_feature_table``.
    """
    return build_feature_table(events_df, cohort_df, config)
