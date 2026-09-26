"""Locked helpers for the one-time MIMIC-III endpoint-parity correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .shock_signal_discovery import pair_domain_events, rolling_oliguria_events
from .spo2_variability_mimic3_validation import (
    FROZEN_EXPOSURE_PARAMETERS,
    compute_frozen_exposure,
)


EXPOSURE_AVAILABILITY_BRANCH = "AVAILABILITY_PARITY_APPLIED"
EXPOSURE_ITEMID = 646
CORRECTED_OLIGURIA_START_HOUR = 4
OLIGURIA_END_HOUR = 16
PRIMARY_LOWER_MINUTE = 360
PRIMARY_UPPER_MINUTE = 960
LATER_COMPLETION_LOWER_MINUTE = 480
MINIMUM_MODELED_EVENTS = 50
AUTHORIZED_OUTCOMES = ("objective_shock_corrected", "objective_shock_corrected_later")
ORIGINAL_CLASSIFICATION = "MIMIC3_HISTORICAL_REPLICATION_DID_NOT_CONFIRM_POST_HOC_SIGNAL"
FORENSIC_CLASSIFICATION = "IMPLEMENTATION_OR_MAPPING_CONCERN"


def select_frozen_exposure_rows(chart_rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the locked CareVue item, value, charttime, and availability rules."""

    required = {
        "stay_id",
        "itemid",
        "event_minute",
        "store_available_minute",
        "raw_value",
        "error_flag",
    }
    missing = required - set(chart_rows.columns)
    if missing:
        raise ValueError(f"Missing exposure columns: {sorted(missing)}")
    selected = chart_rows.loc[
        chart_rows["itemid"].eq(EXPOSURE_ITEMID)
        & chart_rows["event_minute"].between(0, 240)
        & chart_rows["store_available_minute"].le(240)
        & pd.to_numeric(chart_rows["raw_value"], errors="coerce").between(50, 100)
        & chart_rows["error_flag"].eq(0),
        ["stay_id", "event_minute", "raw_value"],
    ].copy()
    selected = selected.rename(
        columns={"event_minute": "minute", "raw_value": "spo2"}
    )
    return selected


def compute_locked_exposure(chart_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute the unchanged RMS feature with the frozen MIMIC-IV transform."""

    return compute_frozen_exposure(select_frozen_exposure_rows(chart_rows))


def corrected_oliguria_events(hourly_urine: pd.DataFrame) -> pd.DataFrame:
    """Use the exact frozen MIMIC-IV post-landmark surveillance start."""

    return rolling_oliguria_events(
        hourly_urine,
        start_hour=CORRECTED_OLIGURIA_START_HOUR,
        end_hour=OLIGURIA_END_HOUR,
    )


def prelandmark_oliguria_exclusions(hourly_urine: pd.DataFrame) -> pd.DataFrame:
    """Preserve the frozen pre-landmark six-hour exclusion through minute 360."""

    return rolling_oliguria_events(
        hourly_urine.loc[hourly_urine["hour"].lt(6), ["stay_id", "hour", "value"]],
        start_hour=0,
        end_hour=6,
    )


def corrected_primary_events(
    pressure_events: pd.DataFrame, perfusion_events: pd.DataFrame
) -> pd.DataFrame:
    """Construct the corrected primary composite exactly once."""

    return pair_domain_events(
        pressure_events,
        perfusion_events,
        lower_minute=PRIMARY_LOWER_MINUTE,
        upper_minute=PRIMARY_UPPER_MINUTE,
    )


def corrected_later_events(primary_events: pd.DataFrame) -> pd.DataFrame:
    """Select primary composites completing after minute 480; never re-pair."""

    return primary_events.loc[
        pd.to_numeric(primary_events["event_minute"], errors="coerce").gt(
            LATER_COMPLETION_LOWER_MINUTE
        )
    ].copy()


def compare_endpoint_ids(
    original: pd.DataFrame, corrected: pd.DataFrame
) -> dict[str, Any]:
    """Compare endpoint membership and completion-time assignment."""

    original_times = original.set_index("stay_id")["event_minute"]
    corrected_times = corrected.set_index("stay_id")["event_minute"]
    original_ids = set(original_times.index)
    corrected_ids = set(corrected_times.index)
    shared = original_ids & corrected_ids
    changed_times = sum(
        not np.isclose(float(original_times.loc[stay]), float(corrected_times.loc[stay]))
        for stay in shared
    )
    union = original_ids | corrected_ids
    return {
        "original_event_ids": len(original_ids),
        "corrected_event_ids": len(corrected_ids),
        "original_only": len(original_ids - corrected_ids),
        "corrected_only": len(corrected_ids - original_ids),
        "shared": len(shared),
        "jaccard_overlap": len(shared) / len(union) if union else 1.0,
        "shared_event_time_assignments_changed": int(changed_times),
    }


def assert_lock_precedes_association(lock_utc: str, association_read_utc: str) -> None:
    """Fail if an association-access timestamp is not strictly after the lock."""

    if datetime.fromisoformat(association_read_utc) <= datetime.fromisoformat(lock_utc):
        raise RuntimeError("Corrected association access did not occur after lock")


def assert_preserved_classifications(
    original_result: dict[str, Any], forensic_result: dict[str, Any]
) -> None:
    """Protect the historical result and forensic classification."""

    if original_result.get("classification") != ORIGINAL_CLASSIFICATION:
        raise RuntimeError("Original MIMIC-III classification changed")
    if forensic_result.get("primary_classification") != FORENSIC_CLASSIFICATION:
        raise RuntimeError("Forensic classification changed")


def assert_exposure_branch_matches_lock(locked_branch: str) -> None:
    """Fail closed if execution diverges from the one locked availability branch."""

    if locked_branch != EXPOSURE_AVAILABILITY_BRANCH:
        raise RuntimeError("Exposure branch differs from locked availability decision")


def verify_protected_hash_manifest(root: Path, manifest_path: Path) -> None:
    """Fail if any file in the Phase-0 protected baseline has changed."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["protected_artifacts"].items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            raise RuntimeError(f"Protected artifact changed: {relative}")


@dataclass
class AuthorizedFitLedger:
    """Enforce the one-primary/one-later corrected-model authorization."""

    fitted: list[str] = field(default_factory=list)

    def register(self, outcome: str) -> None:
        if outcome not in AUTHORIZED_OUTCOMES:
            raise RuntimeError(f"Unauthorized corrected endpoint: {outcome}")
        if outcome in self.fitted:
            raise RuntimeError(f"Corrected endpoint already fitted: {outcome}")
        if len(self.fitted) >= 2:
            raise RuntimeError("More than two corrected association models requested")
        self.fitted.append(outcome)


__all__ = [
    "AUTHORIZED_OUTCOMES",
    "AuthorizedFitLedger",
    "CORRECTED_OLIGURIA_START_HOUR",
    "EXPOSURE_AVAILABILITY_BRANCH",
    "EXPOSURE_ITEMID",
    "FORENSIC_CLASSIFICATION",
    "FROZEN_EXPOSURE_PARAMETERS",
    "LATER_COMPLETION_LOWER_MINUTE",
    "MINIMUM_MODELED_EVENTS",
    "ORIGINAL_CLASSIFICATION",
    "compare_endpoint_ids",
    "compute_locked_exposure",
    "corrected_later_events",
    "corrected_oliguria_events",
    "corrected_primary_events",
    "prelandmark_oliguria_exclusions",
    "select_frozen_exposure_rows",
    "assert_lock_precedes_association",
    "assert_preserved_classifications",
    "assert_exposure_branch_matches_lock",
    "verify_protected_hash_manifest",
]
