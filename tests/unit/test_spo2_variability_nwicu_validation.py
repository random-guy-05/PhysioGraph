"""Tests for the frozen NWICU feasibility gate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from physiograph.analysis.spo2_variability_nwicu_validation import (
    FROZEN_EXPOSURE_PARAMETERS,
    assert_association_authorized,
    compute_frozen_exposure,
    evaluate_frozen_feasibility,
    read_checksum_manifest,
)


def test_checksum_manifest_accepts_supplied_one_space_format(tmp_path: Path) -> None:
    digest = "a" * 64
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text(f"{digest} data/table.csv.gz\n")
    assert read_checksum_manifest(path) == {"data/table.csv.gz": digest}


def test_exact_frozen_exposure_uses_mimic_scale() -> None:
    raw = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "minute": [5, 65, 125, 185],
            "spo2": [95, 97, 94, 98],
        }
    )
    result = compute_frozen_exposure(raw).iloc[0]
    expected = (
        min(max(result.raw_rms_residual, FROZEN_EXPOSURE_PARAMETERS["lower"]), FROZEN_EXPOSURE_PARAMETERS["upper"])
        - FROZEN_EXPOSURE_PARAMETERS["center"]
    ) / FROZEN_EXPOSURE_PARAMETERS["scale"]
    assert result.standardized_exposure == pytest.approx(expected)


def test_modified_endpoint_still_requires_minimum_event_support() -> None:
    feasibility = evaluate_frozen_feasibility(
        preliminary_eligible_stays=1495,
        preliminary_valid_exposures=862,
        modified_endpoint_observable=True,
        site_identifier_available=False,
        contributing_sites=None,
        modeled_primary_events=None,
    )
    assert feasibility["association_authorized"] is False
    assert feasibility["classification"] == "FROZEN_THIRD_COHORT_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE"
    assert feasibility["preliminary_exposure_coverage"] == pytest.approx(862 / 1495)
    with pytest.raises(RuntimeError, match="association prohibited"):
        assert_association_authorized(feasibility)


def test_all_frozen_gates_are_required() -> None:
    ready = evaluate_frozen_feasibility(
        preliminary_eligible_stays=1000,
        preliminary_valid_exposures=800,
        modified_endpoint_observable=True,
        site_identifier_available=True,
        contributing_sites=20,
        modeled_primary_events=50,
    )
    assert ready["association_authorized"] is True
    assert_association_authorized(ready)


def test_user_waivers_apply_only_to_coverage_and_sites() -> None:
    feasibility = evaluate_frozen_feasibility(
        preliminary_eligible_stays=100,
        preliminary_valid_exposures=50,
        modified_endpoint_observable=True,
        site_identifier_available=False,
        contributing_sites=None,
        modeled_primary_events=50,
    )
    assert feasibility["association_authorized"] is True
    assert feasibility["gates"]["coverage_observed"] is False
    assert feasibility["gates"]["sites_observed"] is False


def test_event_boundary_remains_closed() -> None:
    feasibility = evaluate_frozen_feasibility(
        preliminary_eligible_stays=100,
        preliminary_valid_exposures=80,
        modified_endpoint_observable=True,
        site_identifier_available=False,
        contributing_sites=None,
        modeled_primary_events=49,
    )
    assert feasibility["association_authorized"] is False
