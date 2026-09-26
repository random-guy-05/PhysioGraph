import math

import numpy as np
import pandas as pd

from physiograph.analysis.spo2_lactate_mechanistic import (
    POST_HOC_STATUS,
    build_episode_lactate_controlled_summary,
    build_episode_lactate_paired_summary,
    build_episode_lactate_records,
    build_lactate_episode_measurement_weighted,
    build_lactate_episode_meta_analysis,
    build_lactate_observation_process,
)


def _synthetic_frames(n_pairs: int = 12):
    event_rows = []
    cohort_rows = []
    analysis_rows = []
    for index in range(n_pairs * 2):
        stay_id = index + 1
        exposed = index < n_pairs
        cohort_rows.append(
            {
                "dataset": "mimic",
                "stay_id": stay_id,
                "person_id": 1000 + stay_id,
                "age": 60 + index % 8,
                "shock_icd_flag": int(index % 3 == 0),
                "baseline_vasoactive_flag": int(index % 4 == 0),
                "followup_end_offset_minutes": 2000,
                "death_offset_minutes": np.nan,
            }
        )
        analysis_rows.append(
            {
                "dataset": "mimic",
                "stay_id": stay_id,
                "resp_support_any_flag": int(index % 2 == 0),
            }
        )
        values = (96.0, 96.0, 90.0) if exposed else (96.0, 95.0, 95.0)
        for time, value in zip((0.0, 30.0, 60.0), values):
            event_rows.append(
                {
                    "dataset": "mimic",
                    "stay_id": stay_id,
                    "concept": "spo2",
                    "offset_minutes": time,
                    "value_numeric": value,
                }
            )
        # The value exactly at the episode/control anchor must never become the
        # baseline; the strict prior value at minute 30 is the baseline.
        for time, value in (
            (30.0, 1.0),
            (60.0, 9.0),
            (150.0, 2.0 if exposed else 1.0),
            (270.0, 2.2 if exposed else 1.1),
            (420.0, 2.4 if exposed else 1.2),
        ):
            event_rows.append(
                {
                    "dataset": "mimic",
                    "stay_id": stay_id,
                    "concept": "lactate",
                    "offset_minutes": time,
                    "value_numeric": value,
                }
            )
    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(cohort_rows),
        pd.DataFrame(analysis_rows),
    )


def test_episode_records_use_actual_anchor_and_strictly_prior_lactate():
    events, cohort, analysis = _synthetic_frames()
    records = build_episode_lactate_records(events, cohort, analysis)
    row = records.loc[
        records["episode_definition"].eq("absolute_jump_ge4")
        & records["signal_resolution"].eq("15_minute_median_bins")
        & records["stay_id"].eq(1)
        & records["lag_window"].eq("1_to_2h")
    ].iloc[0]
    assert row["episode_exposed"] == 1
    assert row["anchor_offset_minutes"] == 60
    assert row["prior_lactate_offset_minutes"] == 30
    assert row["prior_lactate"] == 1.0
    assert row["first_lactate_offset_minutes"] == 150
    assert row["first_lactate_delta"] == 1.0
    assert row["analysis_status"] == POST_HOC_STATUS
    assert not bool(row["causal_claim_permitted"])


def test_directional_episode_definitions_do_not_mix_recovery_and_drop():
    events, cohort, analysis = _synthetic_frames(n_pairs=6)
    # Add one pure upward recovery episode.
    stay_id = 999
    cohort = pd.concat(
        [
            cohort,
            pd.DataFrame(
                {
                    "dataset": ["mimic"],
                    "stay_id": [stay_id],
                    "person_id": [stay_id],
                    "followup_end_offset_minutes": [2000],
                }
            ),
        ],
        ignore_index=True,
    )
    analysis = pd.concat(
        [analysis, pd.DataFrame({"dataset": ["mimic"], "stay_id": [stay_id]})],
        ignore_index=True,
    )
    extra = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": stay_id,
            "concept": ["spo2", "spo2", "spo2", "lactate", "lactate"],
            "offset_minutes": [0, 30, 60, 30, 150],
            "value_numeric": [90, 90, 96, 1.0, 1.1],
        }
    )
    records = build_episode_lactate_records(
        pd.concat([events, extra], ignore_index=True), cohort, analysis
    )
    recovery = records.loc[
        records["stay_id"].eq(stay_id)
        & records["episode_definition"].eq("recovery_rise_ge4")
        & records["signal_resolution"].eq("15_minute_median_bins")
    ]
    drop = records.loc[
        records["stay_id"].eq(stay_id)
        & records["episode_definition"].eq("desaturation_drop_ge3")
        & records["signal_resolution"].eq("15_minute_median_bins")
    ]
    assert recovery["episode_exposed"].eq(1).all()
    assert drop["episode_exposed"].eq(0).all()


def test_paired_and_controlled_focused_families_recover_known_signal():
    events, cohort, analysis = _synthetic_frames(n_pairs=24)
    records = build_episode_lactate_records(events, cohort, analysis)
    paired = build_episode_lactate_paired_summary(records, bootstrap_repetitions=0)
    controlled = build_episode_lactate_controlled_summary(records)
    paired_row = paired.loc[
        paired["dataset"].eq("mimic")
        & paired["episode_definition"].eq("absolute_jump_ge4")
        & paired["signal_resolution"].eq("15_minute_median_bins")
        & paired["eligibility_scope"].eq("episode_pair_observed")
        & paired["outcome_kind"].eq("first_next_lactate")
        & paired["lag_window"].eq("1_to_8h")
    ].iloc[0]
    controlled_row = controlled.loc[
        controlled["dataset"].eq("mimic")
        & controlled["episode_definition"].eq("absolute_jump_ge4")
        & controlled["signal_resolution"].eq("15_minute_median_bins")
        & controlled["eligibility_scope"].eq("episode_pair_observed")
        & controlled["outcome_kind"].eq("first_next_lactate")
        & controlled["rise_threshold_mmol_l"].eq(0.5)
        & controlled["lag_window"].eq("1_to_8h")
    ].iloc[0]
    assert paired_row["median_delta_mmol_l"] == 1.0
    assert paired_row["focused_q_value"] < 0.05
    assert controlled_row["risk_ratio"] > 1
    assert controlled_row["focused_q_value"] < 0.05
    assert controlled_row["adjusted_status"] == "underpowered_low_information"
    assert controlled_row["adjusted_events_per_parameter"] < 5
    assert controlled_row["adjusted_reason"] == "events_per_parameter_below_5_fail_closed"


def test_observation_process_and_meta_analysis_are_explicit():
    events, cohort, analysis = _synthetic_frames(n_pairs=12)
    records = build_episode_lactate_records(events, cohort, analysis)
    observation = build_lactate_observation_process(records)
    controlled = build_episode_lactate_controlled_summary(records)
    meta = build_lactate_episode_meta_analysis(controlled)
    complete = observation.loc[
        observation["observation_target"].eq("complete_strict_pair")
        & observation["lag_window"].eq("1_to_8h")
    ].iloc[0]
    assert complete["observation_rate_exposed"] == 1.0
    assert complete["observation_rate_unexposed"] == 1.0
    assert complete["interpretation"].startswith("measurement_process")
    # A one-dataset synthetic fixture is retained but cannot be mislabeled as
    # replicated cross-dataset evidence.
    assert not meta.empty
    assert meta["status"].eq("single_dataset_only").all()
    assert math.isfinite(meta.iloc[0]["fixed_effect_risk_ratio"])


def test_measurement_weighted_sensitivity_estimates_when_information_is_adequate():
    rows = []
    for index in range(240):
        exposed = int(index < 120)
        observed = int(index % 5 != 0)
        event = int((index % 2 == 0) if exposed else (index % 4 == 0))
        rows.append(
            {
                "dataset": "mimic",
                "stay_id": index + 1,
                "episode_definition": "absolute_jump_ge4",
                "signal_resolution": "15_minute_median_bins",
                "lag_window": "0_to_1h",
                "episode_exposed": exposed,
                "prior_lactate_strictly_before_episode": 1,
                "complete_lactate_pair": observed,
                "first_lactate_delta": (
                    1.0 if event else 0.0
                )
                if observed
                else np.nan,
            }
        )
    result = build_lactate_episode_measurement_weighted(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row["status"] == "estimated"
    assert row["measurement_weighted_risk_ratio"] > 1
    assert math.isfinite(row["measurement_weighted_ci95_low"])
    assert row["effective_sample_size"] > 100
