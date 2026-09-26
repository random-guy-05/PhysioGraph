import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.shock_signal_discovery import pair_domain_events
from physiograph.analysis.spo2_variability_mimic3_parity_corrected import (
    AuthorizedFitLedger,
    CORRECTED_OLIGURIA_START_HOUR,
    EXPOSURE_AVAILABILITY_BRANCH,
    FROZEN_EXPOSURE_PARAMETERS,
    assert_exposure_branch_matches_lock,
    assert_lock_precedes_association,
    assert_preserved_classifications,
    compute_locked_exposure,
    corrected_later_events,
    corrected_oliguria_events,
    corrected_primary_events,
    select_frozen_exposure_rows,
    verify_protected_hash_manifest,
)


def test_corrected_oliguria_start_is_frozen_at_hour_four() -> None:
    assert CORRECTED_OLIGURIA_START_HOUR == 4
    urine = pd.DataFrame(
        {"stay_id": [1] * 10, "hour": range(10), "value": [10.0] * 10}
    )
    result = corrected_oliguria_events(urine)
    assert result["event_minute"].tolist() == [600.0]


def test_later_endpoint_filters_primary_completion_without_repairing() -> None:
    pressure = pd.DataFrame(
        {"stay_id": [1, 2], "event_minute": [420.0, 500.0], "component": ["p", "p"]}
    )
    perfusion = pd.DataFrame(
        {"stay_id": [1, 2], "event_minute": [600.0, 470.0], "component": ["h", "h"]}
    )
    primary = corrected_primary_events(pressure, perfusion)
    later = corrected_later_events(primary)
    assert set(later["stay_id"]) == {1, 2}
    reparied = pair_domain_events(pressure, perfusion, lower_minute=480, upper_minute=960)
    assert set(reparied["stay_id"]) != set(later["stay_id"])


def test_exposure_branch_and_item_646_availability_are_locked() -> None:
    assert EXPOSURE_AVAILABILITY_BRANCH == "AVAILABILITY_PARITY_APPLIED"
    assert_exposure_branch_matches_lock("AVAILABILITY_PARITY_APPLIED")
    with pytest.raises(RuntimeError, match="differs from locked"):
        assert_exposure_branch_matches_lock("AVAILABILITY_PARITY_NOT_SUPPORTABLE")
    rows = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1, 2],
            "itemid": [646, 646, 646, 646, 220277],
            "event_minute": [0.0, 120.0, 240.0, 180.0, 60.0],
            "store_available_minute": [5.0, 125.0, 240.0, 241.0, 60.0],
            "raw_value": [95.0, 97.0, 96.0, 99.0, 98.0],
            "error_flag": [0, 0, 0, 0, 0],
        }
    )
    selected = select_frozen_exposure_rows(rows)
    assert selected["minute"].tolist() == [0.0, 120.0, 240.0]
    assert set(selected["stay_id"]) == {1}


def test_exposure_uses_identical_ols_rms_and_mimic4_scale() -> None:
    rows = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 1],
            "itemid": [646] * 4,
            "event_minute": [0.0, 60.0, 120.0, 180.0],
            "store_available_minute": [1.0, 61.0, 121.0, 181.0],
            "raw_value": [95.0, 98.0, 96.0, 99.0],
            "error_flag": [0] * 4,
        }
    )
    result = compute_locked_exposure(rows).iloc[0]
    x = np.column_stack([np.ones(4), np.arange(4, dtype=float)])
    y = np.array([95.0, 98.0, 96.0, 99.0])
    residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    expected = np.sqrt(np.mean(residual**2))
    assert np.isclose(result["raw_rms_residual"], expected)
    assert FROZEN_EXPOSURE_PARAMETERS == {
        "lower": 1.160311428702309e-14,
        "upper": 4.300141942983509,
        "center": 0.8776215724100497,
        "scale": 0.8227000137550243,
    }
    expected_z = (
        np.clip(expected, FROZEN_EXPOSURE_PARAMETERS["lower"], FROZEN_EXPOSURE_PARAMETERS["upper"])
        - FROZEN_EXPOSURE_PARAMETERS["center"]
    ) / FROZEN_EXPOSURE_PARAMETERS["scale"]
    assert np.isclose(result["standardized_exposure"], expected_z)


def test_fit_ledger_rejects_unauthorized_and_more_than_two_models() -> None:
    ledger = AuthorizedFitLedger()
    with pytest.raises(RuntimeError, match="Unauthorized"):
        ledger.register("renal_free_rescue")
    ledger.register("objective_shock_corrected")
    ledger.register("objective_shock_corrected_later")
    with pytest.raises(RuntimeError, match="already fitted|More than two"):
        ledger.register("objective_shock_corrected")


def test_association_timestamp_must_follow_lock() -> None:
    assert_lock_precedes_association(
        "2026-09-10T10:00:00+00:00", "2026-09-10T10:00:01+00:00"
    )
    with pytest.raises(RuntimeError, match="after lock"):
        assert_lock_precedes_association(
            "2026-09-10T10:00:00+00:00", "2026-09-10T09:59:59+00:00"
        )


def test_original_and_forensic_classifications_cannot_be_overwritten(tmp_path) -> None:
    original = {
        "classification": "MIMIC3_HISTORICAL_REPLICATION_DID_NOT_CONFIRM_POST_HOC_SIGNAL"
    }
    forensic = {"primary_classification": "IMPLEMENTATION_OR_MAPPING_CONCERN"}
    assert_preserved_classifications(original, forensic)
    (tmp_path / "original.json").write_text(json.dumps(original))
    changed = json.loads((tmp_path / "original.json").read_text())
    changed["classification"] = "CONFIRMED"
    with pytest.raises(RuntimeError, match="Original"):
        assert_preserved_classifications(changed, forensic)


def test_protected_artifact_hash_change_fails_closed(tmp_path) -> None:
    protected = tmp_path / "frozen.json"
    protected.write_text('{"classification":"preserved"}\n')
    digest = hashlib.sha256(protected.read_bytes()).hexdigest()
    manifest = tmp_path / "artifact_hashes.json"
    manifest.write_text(
        json.dumps({"protected_artifacts": {"frozen.json": digest}}) + "\n"
    )
    verify_protected_hash_manifest(tmp_path, manifest)
    protected.write_text('{"classification":"overwritten"}\n')
    with pytest.raises(RuntimeError, match="Protected artifact changed"):
        verify_protected_hash_manifest(tmp_path, manifest)
