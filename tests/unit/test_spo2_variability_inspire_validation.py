"""Tests for the frozen INSPIRE external-validation guardrails."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from physiograph.analysis.spo2_variability_inspire_validation import (
    FROZEN_ENDPOINT,
    FROZEN_EXPOSURE_PARAMETERS,
    assert_association_authorized,
    compute_frozen_exposure,
    evaluate_frozen_feasibility,
    read_checksum_manifest,
)


def test_checksum_manifest_accepts_inspire_one_space_format(tmp_path: Path) -> None:
    digest = "a" * 64
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text(f"{digest} vitals.csv.gz\n")
    assert read_checksum_manifest(path) == {"vitals.csv.gz": digest}


def test_original_composite_components_remain_present() -> None:
    assert FROZEN_ENDPOINT["pressure_support_components"] == [
        "sustained_cuff_hypotension",
        "continuous_support",
        "mcs",
    ]
    assert FROZEN_ENDPOINT["hypoperfusion_components"] == [
        "lactate",
        "creatinine",
        "alt",
        "ph",
        "oliguria",
    ]


def test_exact_frozen_exposure_uses_mimic_scale() -> None:
    raw = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "minute": [5, 65, 125, 185],
            "spo2": [95, 97, 94, 98],
        }
    )
    row = compute_frozen_exposure(raw).iloc[0]
    expected = (
        min(max(row.raw_rms_residual, FROZEN_EXPOSURE_PARAMETERS["lower"]), FROZEN_EXPOSURE_PARAMETERS["upper"])
        - FROZEN_EXPOSURE_PARAMETERS["center"]
    ) / FROZEN_EXPOSURE_PARAMETERS["scale"]
    assert row.standardized_exposure == pytest.approx(expected)


def test_event_gate_remains_binding_at_49_events() -> None:
    result = evaluate_frozen_feasibility(
        eligible_stays=1000,
        valid_exposures=500,
        modeled_primary_events=49,
        exact_composite_observable=True,
    )
    assert result["association_authorized"] is False
    with pytest.raises(RuntimeError, match="association prohibited"):
        assert_association_authorized(result)


def test_user_waivers_apply_only_to_coverage_and_sites() -> None:
    result = evaluate_frozen_feasibility(
        eligible_stays=1000,
        valid_exposures=500,
        modeled_primary_events=50,
        exact_composite_observable=True,
    )
    assert result["gates"]["coverage_observed"] is False
    assert result["gates"]["sites_observed"] is False
    assert result["association_authorized"] is True
    assert_association_authorized(result)


def test_exact_composite_cannot_be_waived() -> None:
    result = evaluate_frozen_feasibility(
        eligible_stays=1000,
        valid_exposures=800,
        modeled_primary_events=100,
        exact_composite_observable=False,
    )
    assert result["association_authorized"] is False
