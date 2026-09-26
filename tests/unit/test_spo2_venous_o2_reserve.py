import numpy as np
import pandas as pd

from physiograph.analysis.spo2_venous_o2_reserve import (
    build_primary_spo2_anchors,
    choose_venous_strategy,
    classify_venous_measurement,
    normalize_venous_saturation,
    phase_a_decision_summary,
    pre_post_window_flags,
)


def test_venous_measurement_classification_preserves_compartment():
    assert classify_venous_measurement("Mixed Venous O2% Sat") == "mixed_venous"
    assert classify_venous_measurement("SvO2") == "mixed_venous"
    assert classify_venous_measurement("ScvO2 (Presep)") == "central_venous"
    assert classify_venous_measurement("Venous oxygen saturation") == "venous_unspecified"
    assert classify_venous_measurement("Oxygen Saturation") is None
    assert classify_venous_measurement("Central Venous Pressure") is None
    assert classify_venous_measurement("PO2 (Mixed Venous)") is None
    assert classify_venous_measurement("SvO2 SQI") is None
    assert classify_venous_measurement("Presep Catheter Dressing") is None


def test_saturation_unit_validation_is_explicit_and_bounded():
    assert normalize_venous_saturation(72, "%") == (72.0, True)
    assert normalize_venous_saturation(0.72, "fraction") == (72.0, True)
    value, valid = normalize_venous_saturation(72, "")
    assert np.isnan(value) and valid is False
    value, valid = normalize_venous_saturation(5, "%")
    assert np.isnan(value) and valid is False


def test_primary_anchor_reuses_first_episode_and_deterministic_controls():
    events = pd.DataFrame(
        {
            "dataset": ["x"] * 12,
            "stay_id": [1] * 4 + [2] * 4 + [3] * 4,
            "concept": ["spo2"] * 12,
            "offset_minutes": [0, 15, 30, 45] * 3,
            "value_numeric": [98, 93, 94, 94, 98, 97, 98, 97, 97, 96, 97, 96],
        }
    )
    first = build_primary_spo2_anchors(events).sort_values("stay_id")
    second = build_primary_spo2_anchors(events).sort_values("stay_id")
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    exposed = first.loc[first["stay_id"].eq(1)].iloc[0]
    assert exposed["episode_exposed"] == 1
    assert exposed["anchor_offset_minutes"] == 15
    assert set(first.loc[first["stay_id"].isin([2, 3]), "episode_exposed"]) == {0}


def test_pre_post_boundaries_exclude_anchor_and_thirty_minutes():
    values = [40, 99.999, 100, 130, 130.001, 220]
    pre, post = pre_post_window_flags(values, 100)
    assert pre.tolist() == [True, True, False, False, False, False]
    assert post.tolist() == [False, False, False, False, True, True]


def test_outcome_blind_gate_stops_when_one_database_is_inadequate():
    rows = pd.DataFrame(
        [
            ("eicu", "mixed_venous", 330, 110, 220),
            ("mimic", "mixed_venous", 90, 40, 50),
            ("eicu", "central_venous", 20, 10, 10),
            ("mimic", "central_venous", 70, 30, 40),
        ],
        columns=["dataset", "venous_type", "valid_pre_post_pair", "exposed", "controls"],
    )
    strategy = choose_venous_strategy(rows)
    assert strategy["venous_o2_feasible"] is False
    summary = phase_a_decision_summary(strategy)
    assert summary["recommended_action"] == "run_predefined_fallback"
    assert summary["effect_estimates_inspected"] is False
    assert summary["physiology_gate_passed"] is False


def test_source_specific_compartments_are_not_called_same_measure_support():
    rows = pd.DataFrame(
        [
            ("eicu", "mixed_venous", 320, 110, 210),
            ("mimic", "mixed_venous", 10, 5, 5),
            ("eicu", "central_venous", 10, 5, 5),
            ("mimic", "central_venous", 320, 110, 210),
        ],
        columns=["dataset", "venous_type", "valid_pre_post_pair", "exposed", "controls"],
    )
    strategy = choose_venous_strategy(rows)
    assert strategy["venous_o2_feasible"] is True
    assert strategy["strategy"] == "source_specific"
    assert strategy["same_measure_support"] is False
