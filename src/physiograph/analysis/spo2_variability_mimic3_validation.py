"""Guardrails for the frozen MIMIC-III SpO2-variability replication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .spo2_variability_inspire_validation import (
    FROZEN_ENDPOINT,
    FROZEN_EXPOSURE_PARAMETERS,
    FROZEN_GATES,
    canonical_sha256,
    compute_frozen_exposure,
    read_checksum_manifest,
    sha256_file,
)

MIMIC3_MAPPING = {
    "timeline_origin": "ICUSTAYS.intime",
    "heart_failure": {"diagnoses_icd.icd9_prefix": "428"},
    "vitals": {
        "hr": [211],
        "sbp_cuff": [455],
        "map_cuff": [456],
        "resp_rate": [614, 615, 618],
        "spo2": [646],
        "temperature_c": [676, 677],
        "temperature_f": [678, 679],
    },
    "labs": {"lactate": [50813], "ph": [50820], "alt": [50861], "creatinine": [50912]},
    "continuous_support": {
        "source": "INPUTEVENTS_CV positive recorded rate/originalrate",
        "itemids": [
            30042, 30306, 30043, 30307, 30044, 30309, 30119, 30047,
            30120, 30125, 30127, 30128, 30051, 42802, 42273,
        ],
    },
    "mcs": {
        "source": "CareVue CHARTEVENTS recorded IABP/ECMO/VAD support fields",
        "itemids": [
            224, 225, 429, 600, 2515, 2865, 2957, 3265, 5798, 5931,
            5937, 6424, 6758, 6931, 7015, 7449,
        ],
    },
    "urine_output": {
        "source": "OUTPUTEVENTS direct urine-volume fields",
        "itemids": [
            40055, 40056, 40057, 40069, 40085, 40094, 40096,
            40405, 40428, 40473, 40651, 40715, 43175,
        ],
    },
    "site_identifier": None,
    "released_site_count": 1,
}


def verify_source_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    """Verify every file in the MIMIC-III one-space SHA-256 manifest."""
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
        raise RuntimeError("MIMIC-III source checksum verification failed")
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
    """Apply the retained 50-event gate and the two user-authorized waivers."""
    coverage = valid_exposures / eligible_stays if eligible_stays else 0.0
    coverage_observed = coverage >= FROZEN_GATES["minimum_exposure_coverage"]
    sites_observed = False
    events_pass = modeled_primary_events >= FROZEN_GATES["minimum_modeled_primary_events"]
    authorized = bool(
        exact_composite_observable
        and events_pass
        and (coverage_observed or waive_coverage_gate)
        and (sites_observed or waive_site_gate)
    )
    failures: list[str] = []
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
            "coverage_observed": bool(coverage_observed),
            "coverage_effective_after_waiver": bool(coverage_observed or waive_coverage_gate),
            "events": bool(events_pass),
            "sites_observed": False,
            "sites_effective_after_waiver": bool(waive_site_gate),
        },
        "waivers": {
            "minimum_70_percent_exposure_coverage": bool(waive_coverage_gate),
            "minimum_20_contributing_sites": bool(waive_site_gate),
        },
        "failures": failures,
        "association_authorized": authorized,
        "classification": (
            "READY_FOR_FROZEN_MIMIC3_REPLICATION_ASSOCIATION"
            if authorized
            else "MIMIC3_REPLICATION_NOT_RUN_FEASIBILITY_FAILURE"
        ),
    }


def assert_association_authorized(feasibility: dict[str, Any]) -> None:
    """Fail closed before either exposure-outcome model is fitted."""
    if feasibility.get("association_authorized") is not True:
        raise RuntimeError(
            "MIMIC-III association prohibited by frozen gates: "
            + ", ".join(feasibility.get("failures", []))
        )


__all__ = [
    "FROZEN_ENDPOINT",
    "FROZEN_EXPOSURE_PARAMETERS",
    "FROZEN_GATES",
    "MIMIC3_MAPPING",
    "assert_association_authorized",
    "canonical_sha256",
    "compute_frozen_exposure",
    "evaluate_frozen_feasibility",
    "sha256_file",
    "verify_source_manifest",
]
