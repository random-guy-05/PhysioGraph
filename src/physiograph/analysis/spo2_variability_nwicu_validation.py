"""Guardrails for the frozen NWICU SpO2-variability validation attempt."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .spo2_variability_transportability import (
    FROZEN_FEATURE_SPEC,
    frozen_exposure_from_raw,
    standardize_with_mimic,
)

FROZEN_EXPOSURE_PARAMETERS = {
    "lower": 1.160311428702309e-14,
    "upper": 4.300141942983509,
    "center": 0.8776215724100497,
    "scale": 0.8227000137550243,
}

NWICU_MAPPING = {
    "heart_failure": {"icd9_prefix": "428", "icd10_prefix": "I50"},
    "vitals": {
        "hr": 320045,
        "sbp_cuff": 320179,
        "map_cuff": None,
        "resp_rate": 320210,
        "spo2": 320277,
        "temperature_f": 323761,
    },
    "labs": {
        "creatinine_blood": 100002,
        "lactate_blood": 100031,
        "alt_blood": 100042,
        "ph_blood": 100339,
    },
    "outcome_sources": {
        "urine_output": None,
        "continuous_infusion_rate": None,
        "hospital_or_site_identifier": None,
        "mcs_itemids": [736876],
    },
}

MODIFIED_ENDPOINT = {
    "name": "modified_nwicu_pressure_support_plus_laboratory_hypoperfusion",
    "primary_window_minutes": {"lower_exclusive": 360, "upper_inclusive": 960},
    "later_window_minutes": {"lower_exclusive": 480, "upper_inclusive": 960},
    "maximum_domain_separation_minutes": 360,
    "pressure_support_components": ["sustained_cuff_sbp_hypotension", "mcs"],
    "laboratory_hypoperfusion_components": ["lactate", "creatinine", "ph", "alt"],
    "structurally_unavailable_components": ["continuous_support", "oliguria"],
    "user_authorized_utc_date": "2026-09-08",
}

FROZEN_GATES = {
    "minimum_exposure_coverage": 0.70,
    "minimum_modeled_primary_events": 50,
    "minimum_contributing_sites": 20,
    "exact_composite_required": True,
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible object with stable serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    """Read the NWICU one-space SHA-256 manifest."""
    rows: dict[str, str] = {}
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter=" "), start=1):
            fields = [field for field in row if field]
            if len(fields) != 2 or len(fields[0]) != 64:
                raise ValueError(f"Malformed checksum manifest line {line_number}")
            rows[fields[1]] = fields[0]
    if not rows:
        raise ValueError("Checksum manifest is empty")
    return rows


def verify_source_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    """Verify every source listed by the supplied NWICU checksum manifest."""
    manifest = read_checksum_manifest(dataset_root / "SHA256SUMS.txt")
    results: list[dict[str, Any]] = []
    for relative, expected in sorted(manifest.items()):
        path = dataset_root / relative
        actual = sha256_file(path) if path.is_file() else None
        results.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size if path.is_file() else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    if not all(row["match"] for row in results):
        raise RuntimeError("NWICU source checksum verification failed")
    return results


def compute_frozen_exposure(raw_spo2: pd.DataFrame) -> pd.DataFrame:
    """Compute the exact frozen feature and apply only the MIMIC transform."""
    feature = frozen_exposure_from_raw(raw_spo2)
    transformed = standardize_with_mimic(feature["raw_rms_residual"], FROZEN_EXPOSURE_PARAMETERS)
    return pd.concat(
        [feature.reset_index(drop=True), transformed.drop(columns="raw_rms_residual").reset_index(drop=True)],
        axis=1,
    )


def evaluate_frozen_feasibility(
    *,
    preliminary_eligible_stays: int,
    preliminary_valid_exposures: int,
    modified_endpoint_observable: bool,
    site_identifier_available: bool,
    contributing_sites: int | None,
    modeled_primary_events: int | None,
    waive_coverage_gate: bool = True,
    waive_site_gate: bool = True,
) -> dict[str, Any]:
    """Apply retained gates plus the two explicit user-authorized waivers."""
    preliminary_coverage = (
        preliminary_valid_exposures / preliminary_eligible_stays
        if preliminary_eligible_stays
        else 0.0
    )
    endpoint_gate = bool(modified_endpoint_observable)
    observed_site_gate = bool(
        site_identifier_available
        and contributing_sites is not None
        and contributing_sites >= FROZEN_GATES["minimum_contributing_sites"]
    )
    site_gate = bool(observed_site_gate or waive_site_gate)
    event_gate = bool(
        endpoint_gate
        and modeled_primary_events is not None
        and modeled_primary_events >= FROZEN_GATES["minimum_modeled_primary_events"]
    )
    final_coverage_evaluable = endpoint_gate
    observed_coverage_gate = bool(
        final_coverage_evaluable and preliminary_coverage >= FROZEN_GATES["minimum_exposure_coverage"]
    )
    coverage_gate = bool(observed_coverage_gate or waive_coverage_gate)
    failures: list[str] = []
    if not endpoint_gate:
        failures.append("user_authorized_modified_endpoint_not_observable")
    if not observed_site_gate and not waive_site_gate:
        failures.append("minimum_20_contributing_sites_not_demonstrable")
    if not observed_coverage_gate and not waive_coverage_gate:
        failures.append("preliminary_exposure_coverage_below_70_percent")
    if not event_gate:
        failures.append("minimum_50_modeled_events_not_demonstrable")
    association_authorized = bool(endpoint_gate and site_gate and event_gate and coverage_gate)
    return {
        "preliminary_eligible_stays": int(preliminary_eligible_stays),
        "preliminary_valid_exposures": int(preliminary_valid_exposures),
        "preliminary_exposure_coverage": float(preliminary_coverage),
        "final_at_risk_coverage_evaluable": final_coverage_evaluable,
        "original_frozen_composite_observable": False,
        "modified_endpoint_observable": endpoint_gate,
        "site_identifier_available": bool(site_identifier_available),
        "contributing_sites": contributing_sites,
        "modeled_primary_events": modeled_primary_events,
        "gates": {
            "coverage_observed": observed_coverage_gate,
            "coverage_effective_after_waiver": coverage_gate,
            "events": event_gate,
            "sites_observed": observed_site_gate,
            "sites_effective_after_waiver": site_gate,
            "modified_endpoint": endpoint_gate,
        },
        "waivers": {
            "minimum_70_percent_exposure_coverage": bool(waive_coverage_gate),
            "minimum_20_contributing_sites": bool(waive_site_gate),
        },
        "failures": failures,
        "association_authorized": association_authorized,
        "classification": (
            "READY_FOR_FROZEN_ASSOCIATION"
            if association_authorized
            else "FROZEN_THIRD_COHORT_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE"
        ),
    }


def assert_association_authorized(feasibility: dict[str, Any]) -> None:
    """Fail closed before any exposure-outcome association can be read."""
    if feasibility.get("association_authorized") is not True:
        failures = feasibility.get("failures", [])
        raise RuntimeError("NWICU association prohibited by frozen gates: " + ", ".join(failures))


def lock_payload(
    *,
    protocol_sha256: str,
    frozen_external_lock_sha256: str,
    code_hashes: dict[str, str],
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Return the immutable NWICU implementation specification."""
    specification = {
        "protocol_sha256": protocol_sha256,
        "frozen_external_validation_lock_sha256": frozen_external_lock_sha256,
        "feature_specification": FROZEN_FEATURE_SPEC,
        "exposure_parameters": FROZEN_EXPOSURE_PARAMETERS,
        "mapping": NWICU_MAPPING,
        "modified_endpoint": MODIFIED_ENDPOINT,
        "gates": FROZEN_GATES,
        "user_authorized_waivers": {
            "minimum_70_percent_exposure_coverage": True,
            "minimum_20_contributing_sites": True,
            "minimum_50_modeled_primary_events": False,
        },
        "source_manifest_sha256": source_manifest_sha256,
        "prohibited": [
            "alternate_variability_formula",
            "dataset_specific_scaling",
            "alternate_outcome",
            "treating_unobserved_oliguria_as_absent",
            "careunit_as_hospital_cluster",
            "subgroup_or_threshold_rescue",
        ],
        "interpretation_rule": {
            "directionally_supportive": "primary RR > 1, HC0 95% CI excludes 1, p < 0.05, and later-window RR > 1",
            "otherwise": "modified endpoint not directionally supportive",
            "never_claim": "external validation of the original frozen composite",
        },
        "association_inspected": False,
    }
    return {
        "status": "LOCKED_BEFORE_NWICU_ASSOCIATION",
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "code_hashes": code_hashes,
    }
