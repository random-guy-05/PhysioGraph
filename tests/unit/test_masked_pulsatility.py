import math

import numpy as np
import pandas as pd

from physiograph.analysis.masked_pulsatility import (
    EICU_SUPPORT_ALIASES,
    MIMIC_SUPPORT_DRUGS,
    assign_pseudo_anchors,
    audit_future_control_crossings,
    build_hourly_bp_series,
    build_source_hourly_bp_pairs,
    build_two_hour_landmark,
    canonical_eicu_support_drug,
    deduplicate_cuff_pairs,
    delayed_support_classification,
    derived_mean_arterial_pressure,
    detect_incident_crossings,
    detect_source_consistent_crossings,
    feasibility_gate,
    fit_modified_poisson,
    followup_observation_flags,
    hospital_contribution_audit,
    intervention_free,
    latest_preanchor_lactate,
    latest_preanchor_measurement,
    merge_continuous_infusion_intervals,
    preanchor_linear_slope,
    prepare_frozen_covariates,
    proportional_pulse_pressure,
    protocol_precedes_outcomes,
    random_effects_meta,
    sha256_file,
    source_repair_feasibility,
    support_lead_bin,
    valid_bp_pair,
    valid_bp_triplet,
)
from physiograph.etl.eicu_extractor import (
    _stream_eicu_aperiodic_vital_events,
    _stream_eicu_nurse_bp_events,
)


def test_prpp_formula_and_physiologic_ranges():
    assert proportional_pulse_pressure(100, 75) == 0.25
    assert valid_bp_triplet(100, 75, 83)
    assert not valid_bp_triplet(75, 100, 83)
    assert not valid_bp_triplet(400, 75, 83)
    assert math.isnan(proportional_pulse_pressure(None, 75))


def test_hourly_aggregation_is_source_consistent_and_uses_latest_time():
    raw = pd.DataFrame(
        [
            ("x", 1, 5, "noninvasive_cuff", 100, 70, 80),
            ("x", 1, 50, "noninvasive_cuff", 120, 80, 92),
            ("x", 1, 65, "noninvasive_cuff", 100, 76, 84),
            ("x", 1, 5, "invasive_arterial", 110, 70, 85),
        ],
        columns=["dataset", "stay_id", "event_minute", "source", "sbp", "dbp", "map"],
    )
    hourly = build_hourly_bp_series(raw)
    assert set(hourly.source) == {"noninvasive_cuff"}
    assert hourly.iloc[0].sbp == 110
    assert hourly.iloc[0].anchor_minute == 50
    assert hourly.iloc[0].measurement_count == 2


def test_incident_crossing_and_already_low_exclusion():
    hourly = pd.DataFrame(
        {
            "dataset": ["x"] * 5,
            "stay_id": [1, 1, 1, 2, 2],
            "hour": [0, 1, 2, 0, 1],
            "prpp": [0.30, 0.27, 0.24, 0.20, 0.30],
            "anchor_minute": [50, 110, 170, 50, 110],
        }
    )
    crossings, initially_low = detect_incident_crossings(hourly)
    assert crossings.stay_id.tolist() == [1]
    assert crossings.hour.tolist() == [2]
    assert initially_low == {2}


def test_lactate_selection_is_strictly_preanchor_and_available():
    candidates = pd.DataFrame(
        [("x", 1, "a", 100)],
        columns=["dataset", "stay_id", "candidate_id", "anchor_minute"],
    )
    labs = pd.DataFrame(
        [
            ("x", 1, 70, 80, 1.5),
            ("x", 1, 90, 105, 1.2),
            ("x", 1, 101, 101, 4.0),
        ],
        columns=["dataset", "stay_id", "event_minute", "available_minute", "value"],
    )
    selected = latest_preanchor_lactate(labs, candidates).iloc[0]
    assert selected.baseline_lactate == 1.5
    assert selected.lactate_event_minute == 70
    assert selected.lactate_to_anchor_lag == 30


def test_preanchor_pressor_or_mcs_excludes_candidate():
    assert intervention_free(100, None, None)
    assert not intervention_free(100, 100, None)
    assert not intervention_free(100, None, 80)
    assert intervention_free(100, 101, 120)


def test_pseudo_anchor_assignment_is_deterministic_and_outcome_free():
    exposed = pd.DataFrame({"dataset": ["x", "x"], "anchor_minute": [60, 180]})
    controls = pd.DataFrame(
        {
            "dataset": ["x"] * 4,
            "stay_id": [10, 10, 20, 20],
            "candidate_id": ["10a", "10b", "20a", "20b"],
            "anchor_minute": [55, 175, 65, 185],
        }
    )
    first = assign_pseudo_anchors(exposed, controls)
    second = assign_pseudo_anchors(exposed, controls)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.stay_id) == {10, 20}


def test_followup_windows_are_strict_after_anchor():
    assert followup_observation_flags([100, 100.01, 460, 820], 100) == (True, True)
    assert followup_observation_flags([100, 820.01], 100) == (False, False)


def test_automatic_stop_logic_uses_fixed_minimums():
    assert feasibility_gate(
        {
            "crossings": 100,
            "controls": 200,
            "crossings_followup_6h": 75,
            "controls_followup_6h": 150,
        }
    )
    assert not feasibility_gate(
        {
            "crossings": 99,
            "controls": 1000,
            "crossings_followup_6h": 99,
            "controls_followup_6h": 999,
        }
    )


def test_protocol_hash_provenance(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen before outcomes\n")
    assert sha256_file(protocol) == "6b0860a13d02f606001a4f88d0475ab3e91a333eececc0809e5940b2c7b7c132"


def test_repaired_aperiodic_mapping_includes_dbp(tmp_path):
    pd.DataFrame(
        {
            "patientunitstayid": [1],
            "observationoffset": [30],
            "noninvasivesystolic": [110],
            "noninvasivediastolic": [70],
            "noninvasivemean": [83],
        }
    ).to_csv(tmp_path / "vitalAperiodic.csv", index=False)
    events = _stream_eicu_aperiodic_vital_events(
        tmp_path, [1], chunk_size=1, max_chunks=None
    )
    assert set(events["concept"]) == {"sbp", "dbp", "map"}


def test_nurse_nibp_mapping_is_exact_and_chunked(tmp_path):
    rows = [
        (1, 10, "Vital Signs", "Non-Invasive BP", "Non-Invasive BP Systolic", "120"),
        (1, 10, "Vital Signs", "Non-Invasive BP", "Non-Invasive BP Diastolic", "75"),
        (1, 10, "Vital Signs", "Non-Invasive BP", "Non-Invasive BP Mean", "90"),
        (1, 10, "Vital Signs", "Invasive BP", "Invasive BP Systolic", "121"),
        (1, 10, "Vital Signs", "Non-Invasive BP", "Approximate Systolic", "119"),
    ]
    pd.DataFrame(
        rows,
        columns=[
            "patientunitstayid",
            "nursingchartoffset",
            "nursingchartcelltypecat",
            "nursingchartcelltypevallabel",
            "nursingchartcelltypevalname",
            "nursingchartvalue",
        ],
    ).to_csv(tmp_path / "nurseCharting.csv", index=False)
    events = _stream_eicu_nurse_bp_events(tmp_path, [1], chunk_size=2, max_chunks=None)
    assert set(events["concept"]) == {"sbp", "dbp", "map"}
    assert len(events) == 3


def test_cuff_deduplication_prefers_aperiodic_and_preserves_unique_nurse():
    columns = ["stay_id", "timestamp_min", "sbp", "dbp", "map", "bp_source"]
    aperiodic = pd.DataFrame(
        [(1, 10, 120, 75, None, "nibp_vitalAperiodic")], columns=columns
    )
    nurse = pd.DataFrame(
        [
            (1, 10, 120, 75, 90, "nibp_nurseCharting"),
            (1, 30, 118, 74, 88, "nibp_nurseCharting"),
        ],
        columns=columns,
    )
    combined, audit = deduplicate_cuff_pairs(aperiodic, nurse)
    assert len(combined) == 2
    assert audit["exact_duplicates"] == 0
    assert audit["near_time_duplicates"] == 1
    assert audit["nurse_rows_dropped"] == 1
    assert audit["map_enriched_from_nurse"] == 1


def test_repaired_hourly_crossing_is_source_consistent_and_map_is_separate():
    pairs = pd.DataFrame(
        [
            ("x", 1, 10, "cuff", 120, 80, 93),
            ("x", 1, 70, "cuff", 100, 76, None),
            ("x", 1, 70, "arterial", 120, 70, 87),
        ],
        columns=["dataset", "stay_id", "timestamp_min", "bp_source", "sbp", "dbp", "map"],
    )
    hourly = build_source_hourly_bp_pairs(pairs)
    crossing, _ = detect_source_consistent_crossings(hourly)
    assert crossing["bp_source"].tolist() == ["cuff"]
    assert pd.isna(crossing.iloc[0]["map"])
    assert crossing.iloc[0]["derived_map"] == 84
    assert valid_bp_pair(100, 76)
    assert derived_mean_arterial_pressure(100, 76) == 84


def test_hospital_audit_and_new_support_gate():
    crossings = pd.DataFrame({"hospital_id": [1] * 5 + [2] * 3})
    audit = hospital_contribution_audit(crossings)
    assert audit["hospital_n"] == 2
    assert audit["hospitals_ge5_crossings"] == 1
    assert source_repair_feasibility(300, 300, 500, 500, 10, severe_center_concentration=False) == (
        "strong",
        "proceed_to_protocol_freeze",
    )
    assert source_repair_feasibility(300, 74, 500, 500, 10, severe_center_concentration=False) == (
        "fail",
        "stop_insufficient_routine_bp_support",
    )


def test_support_timing_boundaries_are_exactly_locked():
    assert support_lead_bin(0) is None
    assert support_lead_bin(60) == "0-1h"
    assert support_lead_bin(120) == "1-2h"
    assert support_lead_bin(120.01) == "2-4h"
    assert support_lead_bin(720) == "6-12h"
    assert support_lead_bin(720.01) is None


def test_two_hour_landmark_excludes_concurrent_and_includes_delayed():
    anchors = pd.DataFrame(
        {
            "dataset": ["x"] * 4,
            "stay_id": [1, 2, 3, 4],
            "candidate_id": ["a", "b", "c", "d"],
            "anchor_minute": [100.0] * 4,
            "anchor_type": ["crossing", "crossing", "control", "control"],
            "followup_end_offset_minutes": [1000.0, 1000.0, 1000.0, 150.0],
            "death_offset_minutes": [np.nan] * 4,
        }
    )
    starts = pd.DataFrame(
        {
            "dataset": ["x", "x", "x"],
            "stay_id": [1, 2, 3],
            "start_minute": [220.0, 220.01, 820.0],
            "drug": ["norepinephrine"] * 3,
        }
    )
    classified, _ = build_two_hour_landmark(anchors, starts)
    result = classified.set_index("candidate_id")
    assert bool(result.loc["a", "blanking_support"])
    assert not bool(result.loc["a", "landmark_eligible"])
    assert bool(result.loc["b", "delayed_support_2_12h"])
    assert bool(result.loc["b", "landmark_eligible"])
    assert bool(result.loc["c", "delayed_support_2_12h"])
    assert not bool(result.loc["d", "landmark_eligible"])


def test_future_control_crossing_audit_never_uses_outcomes():
    controls = pd.DataFrame(
        [("x", 1, "a", 60.0), ("x", 2, "b", 120.0)],
        columns=["dataset", "stay_id", "candidate_id", "anchor_minute"],
    )
    crossings = pd.DataFrame(
        [("x", 1, 180.0), ("x", 2, 100.0)],
        columns=["dataset", "stay_id", "anchor_minute"],
    )
    audit = audit_future_control_crossings(controls, crossings)
    assert audit["candidate_id"].tolist() == ["a"]


def test_preanchor_slope_excludes_postanchor_values():
    anchors = pd.DataFrame(
        [("x", 1, "a", 120.0)],
        columns=["dataset", "stay_id", "candidate_id", "anchor_minute"],
    )
    events = pd.DataFrame(
        [("x", 1, 0.0, 100.0), ("x", 1, 120.0, 80.0), ("x", 1, 121.0, 500.0)],
        columns=["dataset", "stay_id", "event_minute", "value"],
    )
    slope = preanchor_linear_slope(events, anchors).iloc[0]
    assert math.isclose(slope.slope_per_hour, -10.0)
    assert slope.slope_points == 2


def test_latest_preanchor_measurement_obeys_recency_and_availability():
    anchors = pd.DataFrame(
        [("x", 1, "a", 300.0)],
        columns=["dataset", "stay_id", "candidate_id", "anchor_minute"],
    )
    events = pd.DataFrame(
        [
            ("x", 1, 50.0, 60.0, 1.0),
            ("x", 1, 250.0, 310.0, 9.0),
            ("x", 1, 240.0, 245.0, 2.0),
            ("x", 1, 301.0, 301.0, 8.0),
        ],
        columns=["dataset", "stay_id", "event_minute", "available_minute", "value"],
    )
    selected = latest_preanchor_measurement(
        events, anchors, lookback_minutes=240, availability_column="available_minute"
    )
    assert selected.iloc[0].value == 2.0


def test_continuous_infusion_rows_merge_with_five_minute_grace():
    rows = pd.DataFrame(
        [
            ("mimic", 1, "norepinephrine", 10.0, 20.0, 1.0),
            ("mimic", 1, "norepinephrine", 25.0, 30.0, 1.0),
            ("mimic", 1, "norepinephrine", 36.0, 40.0, 1.0),
            ("mimic", 1, "norepinephrine", 50.0, 60.0, 0.0),
        ],
        columns=["dataset", "stay_id", "drug", "start_minute", "end_minute", "rate"],
    )
    episodes = merge_continuous_infusion_intervals(rows)
    assert episodes[["start_minute", "end_minute", "rows_merged"]].values.tolist() == [
        [10.0, 30.0, 2.0],
        [36.0, 40.0, 1.0],
    ]


def test_frozen_drug_definition_is_harmonized_across_databases():
    assert set(MIMIC_SUPPORT_DRUGS.values()) == set(EICU_SUPPORT_ALIASES)
    assert canonical_eicu_support_drug("Levophed 0.05 mcg/kg/min") == "norepinephrine"
    assert canonical_eicu_support_drug("normal saline") is None


def test_modified_poisson_supports_eicu_hospital_clustering():
    frame = pd.DataFrame(
        {
            "exposed": np.tile([0, 1], 30),
            "outcome": np.tile([0, 0, 0, 1, 0, 1], 10),
            "age": np.linspace(40, 80, 60),
            "hospital_id": np.repeat(np.arange(10), 6),
        }
    )
    prepared, columns, _ = prepare_frozen_covariates(frame, ["age"])
    result, _ = fit_modified_poisson(
        prepared,
        outcome="outcome",
        covariates=columns,
        cluster_column="hospital_id",
    )
    assert result["covariance"] == "hospital_clustered"
    assert result["n"] == 60


def test_random_effects_meta_reports_heterogeneity_without_rescue_logic():
    result = random_effects_meta(
        pd.DataFrame({"log_rr": [math.log(1.5), math.log(1.6)], "log_rr_se": [0.1, 0.2]})
    )
    assert 1.5 < result["pooled_rr"] < 1.6
    assert result["i2_percent"] >= 0


def test_protocol_order_and_automatic_failure_classification():
    assert protocol_precedes_outcomes("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z")
    assert not protocol_precedes_outcomes("2026-01-01T00:00:01Z", "2026-01-01T00:00:01Z")
    assert (
        delayed_support_classification(
            {"eicu_concurrent_positive": True, "primary_delayed_support_positive": False}
        )
        == "CONCURRENT_DETERIORATION_ONLY"
    )
