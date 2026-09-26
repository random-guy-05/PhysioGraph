"""Adversarial tests for dynamics-first SpO2 instability analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from physiograph.analysis.spo2_instability import (
    _derive_ventilation_outcomes,
    build_multiscale_instability_features,
)


def _spo2(dataset: str, stay_id: int, minute: float, value: float) -> dict[str, object]:
    return {
        "dataset": dataset,
        "stay_id": stay_id,
        "concept": "spo2",
        "source_table": "vitalPeriodic.csv" if dataset == "eicu" else "chartevents.csv",
        "raw_name": "SpO2",
        "offset_minutes": minute,
        "value_numeric": value,
    }


def test_native_resolution_recovers_instability_smoothed_by_fifteen_minute_medians() -> None:
    events = pd.DataFrame(
        [
            _spo2("eicu", 1, 0, 96),
            _spo2("eicu", 1, 5, 90),
            _spo2("eicu", 1, 10, 96),
            _spo2("eicu", 1, 15, 96),
            _spo2("eicu", 1, 20, 96),
            _spo2("eicu", 1, 25, 96),
        ]
    )
    features, cadence = build_multiscale_instability_features(events)
    row = features.iloc[0]

    assert cadence.iloc[0]["adaptive_max_pair_gap_minutes"] == 10
    assert row["adaptive_jump_ge4_count"] == 2
    assert row["adaptive_return_to_baseline_spike_fraction"] > 0
    assert row["adaptive_confirmed_jump_ge4_rate_per_hr"] == 0
    assert row["bin15_jump_ge4_rate_per_hr"] == 0


def test_adaptive_gap_uses_sparse_mimic_trajectory_without_relaxing_raw30_sensitivity() -> None:
    events = pd.DataFrame(
        [
            _spo2("mimic", 1, 0, 98),
            _spo2("mimic", 1, 35, 96),
            _spo2("mimic", 1, 70, 92),
            _spo2("mimic", 1, 105, 95),
            _spo2("mimic", 1, 140, 90),
        ]
    )
    features, cadence = build_multiscale_instability_features(events)
    row = features.iloc[0]

    assert cadence.iloc[0]["adaptive_max_pair_gap_minutes"] == 70
    assert row["adaptive_eligible_flag"] == 1
    assert row["raw30_eligible_flag"] == 0
    assert row["raw90_eligible_flag"] == 1
    assert row["adaptive_drop_ge3_rate_per_hr"] > 0


def test_persistent_drop_is_separate_from_spike_and_absolute_level() -> None:
    events = pd.DataFrame(
        [
            *[_spo2("eicu", 1, minute, value) for minute, value in [(0, 98), (5, 93), (10, 92), (15, 92)]],
            *[_spo2("eicu", 2, minute, value) for minute, value in [(0, 94), (5, 89), (10, 88), (15, 88)]],
        ]
    )
    features, _ = build_multiscale_instability_features(events)
    first = features.loc[features["stay_id"].eq(1)].iloc[0]
    second = features.loc[features["stay_id"].eq(2)].iloc[0]

    assert first["adaptive_any_confirmed_drop_ge3"] == 1
    assert first["adaptive_persistent_shift_rate_per_hr"] > 0
    assert first["adaptive_return_to_baseline_spike_rate_per_hr"] == 0
    assert first["adaptive_rmssd_10min_equivalent"] == second[
        "adaptive_rmssd_10min_equivalent"
    ]
    assert first["adaptive_mean"] != second["adaptive_mean"]


def test_clean_ventilation_outcome_rejects_false_text_matches() -> None:
    events = pd.DataFrame(
        [
            {
                "dataset": "eicu",
                "stay_id": 1,
                "concept": "mechanical_ventilation",
                "source_table": "treatment.csv",
                "raw_name": "medications|simvastatin",
                "offset_minutes": 500,
            },
            {
                "dataset": "eicu",
                "stay_id": 2,
                "concept": "mechanical_ventilation",
                "source_table": "treatment.csv",
                "raw_name": "respiratory|non-invasive ventilation",
                "offset_minutes": 500,
            },
            {
                "dataset": "eicu",
                "stay_id": 3,
                "concept": "mechanical_ventilation",
                "source_table": "treatment.csv",
                "raw_name": "respiratory|mechanical ventilation|settings",
                "offset_minutes": 500,
            },
        ]
    )
    frame = pd.DataFrame(
        {
            "dataset": ["eicu"] * 3,
            "stay_id": [1, 2, 3],
            "followup_end_offset_minutes": [2000] * 3,
            "death_offset_minutes": [np.nan] * 3,
        }
    )
    result, audit = _derive_ventilation_outcomes(events, frame)

    assert result["invasive_ventilation_8h_flag"].tolist() == [0.0, 0.0, 1.0]
    assert audit.iloc[0]["removed_total_rows"] == 2
    assert audit.iloc[0]["retained_literal_mechanical_ventilation_rows"] == 1
