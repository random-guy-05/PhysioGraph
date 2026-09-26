"""Focused guards for the post-hoc SpO2 transportability audit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.spo2_variability_transportability import (
    FROZEN_FEATURE_SPEC,
    apply_locked_timing_degradation,
    assert_eicu_hospital_clustering,
    frozen_exposure_from_raw,
    measurement_pattern_hash,
    require_degradation_lock_before_outcomes,
    standardize_with_mimic,
    validate_frozen_feature_spec,
    write_immutable_degradation_lock,
)


def raw_frame(stay: int, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"stay_id": stay, "minute": [0, 60, 120, 180][: len(values)], "spo2": values})


def test_exact_frozen_exposure_parity_and_no_alternative_substitution() -> None:
    raw = raw_frame(1, [90.0, 92.0, 91.0, 95.0])
    result = frozen_exposure_from_raw(raw).iloc[0]
    hours = np.arange(4.0)
    design = np.column_stack([np.ones(4), hours])
    values = np.array([90.0, 92.0, 91.0, 95.0])
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    assert result["raw_rms_residual"] == pytest.approx(np.sqrt(np.mean(residual**2)))
    changed = dict(FROZEN_FEATURE_SPEC); changed["summary"] = "standard_deviation"
    with pytest.raises(ValueError, match="Alternative"):
        validate_frozen_feature_spec(changed)


def test_post_landmark_spo2_is_rejected() -> None:
    raw = pd.concat([raw_frame(1, [90, 91, 92]), pd.DataFrame({"stay_id": [1], "minute": [241], "spo2": [93]})])
    with pytest.raises(ValueError, match="0 through 240"):
        frozen_exposure_from_raw(raw)


def test_degradation_lock_is_immutable_and_required_before_outcomes(tmp_path) -> None:
    path = tmp_path / "measurement_degradation_lock.json"
    with pytest.raises(RuntimeError, match="before measurement-degradation lock"):
        require_degradation_lock_before_outcomes(path, ["objective_shock"])
    digest = write_immutable_degradation_lock(path, {"specification": {"seed": 1}})
    path.with_suffix(".sha256").write_text(digest + "\n")
    require_degradation_lock_before_outcomes(path, ["objective_shock"])
    with pytest.raises(FileExistsError):
        write_immutable_degradation_lock(path, {"specification": {"seed": 2}})


def test_seed_locked_timing_degradation_is_deterministic() -> None:
    mimic = pd.concat([raw_frame(1, [90, 92, 91, 95]), raw_frame(2, [96, 95, 97, 94])], ignore_index=True)
    eicu = pd.concat([
        pd.DataFrame({"stay_id": 10, "minute": [0, 30, 60, 120, 180], "spo2": [90, 90, 91, 92, 93]}),
        pd.DataFrame({"stay_id": 11, "minute": [5, 65, 125, 185], "spo2": [95, 95, 96, 97]}),
    ], ignore_index=True)
    spec = {
        "operator": "eicu_empirical_timing_template_with_mimic_last_observation_carried_forward",
        "outcome_information_used": False,
        "feature_specification": FROZEN_FEATURE_SPEC,
        "seed": 42,
        "quantization": "none",
        "template_pattern_sha256": measurement_pattern_hash(eicu),
    }
    first = apply_locked_timing_degradation(mimic, eicu, pd.Series([1, 2]), spec)
    second = apply_locked_timing_degradation(mimic, eicu, pd.Series([1, 2]), spec)
    pd.testing.assert_frame_equal(first, second)


def test_mimic_scaling_is_identical_for_both_database_vectors() -> None:
    parameters = {"lower": 0.1, "upper": 4.0, "center": 1.0, "scale": 0.5}
    mimic = standardize_with_mimic(pd.Series([0.0, 1.5, 5.0]), parameters)
    eicu = standardize_with_mimic(pd.Series([0.0, 1.5, 5.0]), parameters)
    pd.testing.assert_series_equal(mimic["standardized_exposure"], eicu["standardized_exposure"])
    assert mimic["standardized_exposure"].tolist() == pytest.approx([-1.8, 1.0, 6.0])


def test_eicu_primary_requires_hospital_clustering() -> None:
    assert_eicu_hospital_clustering({"covariance": "cluster_hospital", "hospitals": 194})
    with pytest.raises(ValueError, match="hospital-clustered"):
        assert_eicu_hospital_clustering({"covariance": "HC0", "hospitals": None})


def test_lock_payload_is_json_serializable() -> None:
    json.dumps({"feature": FROZEN_FEATURE_SPEC, "alternative_features": []}, allow_nan=False)
