"""Outcome-blind support rules for the SpO2/venous-oxygen reserve audit."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .spo2_lactate_mechanistic import (
    _prepare_spo2_transitions,
    _select_anchor_per_stay,
)

PRIMARY_EPISODE_DEFINITION: dict[str, Any] = {
    "episode_definition": "absolute_jump_ge4",
    "signal_resolution": "15_minute_median_bins",
    "kind": "absolute",
    "threshold": 4.0,
    "role": "established_primary",
}
VENOUS_TYPES = ("mixed_venous", "central_venous", "venous_unspecified")
PRIMARY_GATE = {"total": 300, "exposed": 100, "controls": 100}
EXPLORATORY_GATE = {"total": 200, "exposed": 75, "controls": 75}


def _normalized_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def classify_venous_measurement(label: Any) -> str | None:
    """Classify only explicitly venous oxygen-saturation labels.

    A generic ``Oxygen Saturation`` label is deliberately not classified as
    venous.  ``SvO2`` is treated as mixed venous; central/PreSep labels always
    take precedence over the shorter ``SvO2`` token.
    """

    raw = str(label or "").lower()
    normalized = _normalized_label(label)
    compact = re.sub(r"[^a-z0-9]", "", raw)
    if not normalized or any(
        token in normalized
        for token in (" alarm", "alarm ", " sqi", "sqi ", "calibrat", "catheter")
    ):
        return None
    saturation_like = (
        "saturation" in normalized
        or "sat" in normalized.split()
        or compact.startswith(("scvo2", "svo2"))
    )
    if not saturation_like:
        return None
    if (
        "central venous" in normalized
        or "scvo2" in compact
        or "presep" in compact
    ):
        return "central_venous"
    if "mixed venous" in normalized or compact == "svo2" or compact.startswith("svo2"):
        return "mixed_venous"
    if "venous" in normalized:
        return "venous_unspecified"
    return None


def normalize_venous_saturation(value: Any, unit: Any) -> tuple[float, bool]:
    """Return percentage saturation and whether its explicit unit is valid."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        return math.nan, False
    raw_unit = str(unit or "").strip().lower()
    cleaned = "%" if raw_unit == "%" else _normalized_label(unit).replace(" ", "")
    if cleaned in {"%", "percent", "percentage", "pct"}:
        result = float(numeric)
    elif cleaned in {"fraction", "ratio", "0to1"}:
        result = float(numeric) * 100.0
    else:
        return math.nan, False
    if not 20.0 <= result <= 100.0:
        return math.nan, False
    return result, True


def build_primary_spo2_anchors(events: pd.DataFrame) -> pd.DataFrame:
    """Reuse the established first-episode/quantile-matched control anchors."""

    transitions, support = _prepare_spo2_transitions(
        events,
        signal_resolution=PRIMARY_EPISODE_DEFINITION["signal_resolution"],
    )
    return _select_anchor_per_stay(
        transitions,
        support,
        definition=PRIMARY_EPISODE_DEFINITION,
    )


def pre_post_window_flags(
    measurement_minutes: Iterable[float], anchor_minute: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen PRE [-60, 0) and POST (+30, +120] boundaries."""

    minutes = np.asarray(list(measurement_minutes), dtype=float)
    relative = minutes - float(anchor_minute)
    pre = (relative >= -60.0) & (relative < 0.0)
    post = (relative > 30.0) & (relative <= 120.0)
    return pre, post


def support_tier(total: int, exposed: int, controls: int) -> str:
    values = {"total": int(total), "exposed": int(exposed), "controls": int(controls)}
    if all(values[key] >= PRIMARY_GATE[key] for key in PRIMARY_GATE):
        return "primary"
    if all(values[key] >= EXPLORATORY_GATE[key] for key in EXPLORATORY_GATE):
        return "exploratory"
    return "inadequate"


def choose_venous_strategy(rows: pd.DataFrame) -> dict[str, Any]:
    """Choose a compartment using coverage only, never an association."""

    required = {"dataset", "venous_type", "valid_pre_post_pair", "exposed", "controls"}
    missing = required - set(rows)
    if missing:
        raise ValueError(f"Feasibility table lacks columns: {sorted(missing)}")
    local = rows.copy()
    local["support_tier"] = [
        support_tier(total, exposed, controls)
        for total, exposed, controls in local[
            ["valid_pre_post_pair", "exposed", "controls"]
        ].itertuples(index=False, name=None)
    ]
    datasets = sorted(local["dataset"].astype(str).unique())
    if len(datasets) < 2:
        return {
            "venous_o2_feasible": False,
            "analysis_tier": "inadequate",
            "strategy": "fallback",
            "primary_measurement": None,
            "same_measure_support": False,
            "reason": "fewer_than_two_databases",
        }

    for measurement in ("mixed_venous", "central_venous"):
        selected = local.loc[local["venous_type"].eq(measurement)]
        tiers = dict(zip(selected["dataset"].astype(str), selected["support_tier"]))
        if all(tiers.get(dataset) == "primary" for dataset in datasets):
            return {
                "venous_o2_feasible": True,
                "analysis_tier": "primary",
                "strategy": "same_measure",
                "primary_measurement": measurement,
                "same_measure_support": True,
                "reason": "primary_gate_passed_in_each_database",
            }
    for measurement in ("mixed_venous", "central_venous"):
        selected = local.loc[local["venous_type"].eq(measurement)]
        tiers = dict(zip(selected["dataset"].astype(str), selected["support_tier"]))
        if all(tiers.get(dataset) in {"primary", "exploratory"} for dataset in datasets):
            return {
                "venous_o2_feasible": True,
                "analysis_tier": "exploratory",
                "strategy": "same_measure",
                "primary_measurement": measurement,
                "same_measure_support": True,
                "reason": "exploratory_gate_passed_in_each_database",
            }

    choices: dict[str, str] = {}
    tiers: list[str] = []
    for dataset in datasets:
        source = local.loc[
            local["dataset"].astype(str).eq(dataset)
            & local["venous_type"].isin(["mixed_venous", "central_venous"])
            & local["support_tier"].isin(["primary", "exploratory"])
        ].copy()
        if source.empty:
            break
        source["_rank"] = source["support_tier"].map({"primary": 0, "exploratory": 1})
        source["_type_rank"] = source["venous_type"].map(
            {"mixed_venous": 0, "central_venous": 1}
        )
        picked = source.sort_values(
            ["_rank", "_type_rank", "valid_pre_post_pair"],
            ascending=[True, True, False],
        ).iloc[0]
        choices[dataset] = str(picked["venous_type"])
        tiers.append(str(picked["support_tier"]))
    if len(choices) == len(datasets):
        return {
            "venous_o2_feasible": True,
            "analysis_tier": "primary" if all(value == "primary" for value in tiers) else "exploratory",
            "strategy": "source_specific",
            "primary_measurement": choices,
            "same_measure_support": False,
            "reason": "different_supported_compartments_by_database",
        }
    return {
        "venous_o2_feasible": False,
        "analysis_tier": "inadequate",
        "strategy": "fallback",
        "primary_measurement": None,
        "same_measure_support": False,
        "reason": "insufficient_outcome_blind_pre_post_support",
    }


def phase_a_decision_summary(strategy: Mapping[str, Any]) -> dict[str, Any]:
    feasible = bool(strategy.get("venous_o2_feasible", False))
    return {
        "venous_o2_feasible": feasible,
        "physiology_gate_passed": False,
        "injury_gate_passed": False,
        "low_output_gate_passed": False,
        "prediction_gate_passed": False,
        "cross_database_replication": False,
        "biological_claim_ready": False,
        "paradigm_shifting_result": False,
        "recommended_action": "freeze_phase_b_protocol" if feasible else "run_predefined_fallback",
        "phase_reached": "phase_a_feasibility",
        "effect_estimates_inspected": False,
        "strategy": dict(strategy),
    }


__all__ = [
    "EXPLORATORY_GATE",
    "PRIMARY_EPISODE_DEFINITION",
    "PRIMARY_GATE",
    "VENOUS_TYPES",
    "build_primary_spo2_anchors",
    "choose_venous_strategy",
    "classify_venous_measurement",
    "normalize_venous_saturation",
    "phase_a_decision_summary",
    "pre_post_window_flags",
    "support_tier",
]
