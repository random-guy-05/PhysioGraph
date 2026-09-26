"""Guardrails for the frozen INSPIRE external validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .spo2_variability_nwicu_validation import (
    FROZEN_EXPOSURE_PARAMETERS,
    compute_frozen_exposure,
)

INSPIRE_MAPPING = {
    "timeline_origin": "operations.icuin_time",
    "heart_failure": {"icd10_prefix": "I50"},
    "vitals": {
        "hr": "hr",
        "sbp_cuff": "nibp_sbp",
        "map_cuff": "nibp_mbp",
        "resp_rate": "rr",
        "spo2": "spo2",
        "temperature_c": "bt",
        "urine_output": "uo",
    },
    "labs": {
        "creatinine": "creatinine",
        "lactate": "lacate",
        "alt": "alt",
        "ph": "ph",
    },
    "continuous_support": {
        "source": "administered medications with route=iv",
        "drug_names": [
            "norepinephrine",
            "epinephrine",
            "dopamine",
            "dobutamine",
            "milrinone",
            "vasopressin",
            "phenylephrine",
        ],
    },
    "mcs": {"iabp": "iabp", "ecmo": "ecmo"},
    "site_identifier": None,
    "released_site_count": 1,
}

FROZEN_ENDPOINT = {
    "name": "pressure_support_plus_hypoperfusion",
    "primary_window_minutes": {"lower_exclusive": 360, "upper_inclusive": 960},
    "later_window_minutes": {"lower_exclusive": 480, "upper_inclusive": 960},
    "maximum_domain_separation_minutes": 360,
    "pressure_support_components": [
        "sustained_cuff_hypotension",
        "continuous_support",
        "mcs",
    ],
    "hypoperfusion_components": [
        "lactate",
        "creatinine",
        "alt",
        "ph",
        "oliguria",
    ],
}

FROZEN_GATES = {
    "minimum_exposure_coverage": 0.70,
    "minimum_modeled_primary_events": 50,
    "minimum_contributing_sites": 20,
}


def sha256_file(path: Path) -> str:
    """Hash a file without loading it as one in-memory object."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible content using stable serialization."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read_checksum_manifest(path: Path) -> dict[str, str]:
    """Read INSPIRE's supplied one-space SHA-256 manifest."""
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
    """Verify every INSPIRE source listed in its checksum manifest."""
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
        raise RuntimeError("INSPIRE source checksum verification failed")
    return results


def evaluate_frozen_feasibility(
    *,
    eligible_stays: int,
    valid_exposures: int,
    modeled_primary_events: int,
    exact_composite_observable: bool,
    waive_coverage_gate: bool = True,
    waive_site_gate: bool = True,
) -> dict[str, Any]:
    """Apply the exact-composite and event gates plus two explicit waivers."""
    coverage = valid_exposures / eligible_stays if eligible_stays else 0.0
    coverage_observed = coverage >= FROZEN_GATES["minimum_exposure_coverage"]
    sites_observed = False
    events_pass = modeled_primary_events >= FROZEN_GATES["minimum_modeled_primary_events"]
    association_authorized = bool(
        exact_composite_observable
        and events_pass
        and (coverage_observed or waive_coverage_gate)
        and (sites_observed or waive_site_gate)
    )
    failures = []
    if not exact_composite_observable:
        failures.append("exact_frozen_composite_not_observable")
    if not events_pass:
        failures.append("minimum_50_modeled_events_not_demonstrable")
    if not coverage_observed and not waive_coverage_gate:
        failures.append("minimum_70_percent_exposure_coverage_not_met")
    if not sites_observed and not waive_site_gate:
        failures.append("minimum_20_contributing_sites_not_met")
    return {
        "eligible_stays": int(eligible_stays),
        "valid_exposures": int(valid_exposures),
        "exposure_coverage": float(coverage),
        "modeled_primary_events": int(modeled_primary_events),
        "exact_composite_observable": bool(exact_composite_observable),
        "contributing_sites": 1,
        "gates": {
            "exact_composite": bool(exact_composite_observable),
            "coverage_observed": coverage_observed,
            "coverage_effective_after_waiver": bool(coverage_observed or waive_coverage_gate),
            "events": events_pass,
            "sites_observed": sites_observed,
            "sites_effective_after_waiver": bool(sites_observed or waive_site_gate),
        },
        "waivers": {
            "minimum_70_percent_exposure_coverage": bool(waive_coverage_gate),
            "minimum_20_contributing_sites": bool(waive_site_gate),
        },
        "failures": failures,
        "association_authorized": association_authorized,
        "classification": (
            "READY_FOR_FROZEN_INSPIRE_EXTERNAL_ASSOCIATION"
            if association_authorized
            else "INSPIRE_EXTERNAL_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE"
        ),
    }


def assert_association_authorized(feasibility: dict[str, Any]) -> None:
    """Fail closed before the exposure-outcome association can be fitted."""
    if feasibility.get("association_authorized") is not True:
        raise RuntimeError(
            "INSPIRE association prohibited by frozen gates: "
            + ", ".join(feasibility.get("failures", []))
        )


__all__ = [
    "FROZEN_ENDPOINT",
    "FROZEN_EXPOSURE_PARAMETERS",
    "FROZEN_GATES",
    "INSPIRE_MAPPING",
    "assert_association_authorized",
    "canonical_sha256",
    "compute_frozen_exposure",
    "evaluate_frozen_feasibility",
    "read_checksum_manifest",
    "sha256_file",
    "verify_source_manifest",
]
