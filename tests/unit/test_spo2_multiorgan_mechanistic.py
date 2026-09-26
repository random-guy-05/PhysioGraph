from __future__ import annotations

import numpy as np
import pandas as pd

import physiograph.analysis.spo2_multiorgan_mechanistic as multiorgan_module
from physiograph.analysis.spo2_multiorgan_mechanistic import (
    LAB_SPECS,
    EventLookup,
    _conservative_any_records,
    _lab_records,
    _selection_weighted_row,
    _urine_records,
    _vis_records,
    run_multiorgan_episode_analyses,
)


def _anchors(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "dataset": "synthetic",
        "person_id": "p",
        "episode_definition": "absolute_jump_ge4",
        "signal_resolution": "15_minute_median_bins",
        "definition_role": "revised_primary",
        "episode_exposed": 1,
        "anchor_offset_minutes": 240.0,
        "anchor_spo2": 90.0,
        "preanchor_sampling_density_per_hr": 4.0,
        "age": 60.0,
        "shock_icd_flag": 0,
        "baseline_vasoactive_flag": 0,
        "followup_end_offset_minutes": 2_000.0,
        "death_offset_minutes": np.nan,
        "vis_source_available": 1,
        "urine_output_source_available": 1,
        "baseline_weight_outcome": 100.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _events(rows: list[tuple[object, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "stay_id": stay_id,
                "concept": concept,
                "offset_minutes": offset,
                "value_numeric": value,
            }
            for stay_id, concept, offset, value in rows
        ]
    )


def test_worst_lab_negative_requires_complete_window_but_early_event_is_observed() -> None:
    anchors = _anchors(
        [
            {"stay_id": 1, "followup_end_offset_minutes": 500.0},
            {"stay_id": 2, "followup_end_offset_minutes": 500.0},
        ]
    )
    lookup = EventLookup(
        _events(
            [
                (1, "creatinine", 200.0, 1.0),
                (1, "creatinine", 300.0, 1.5),
                (2, "creatinine", 200.0, 1.0),
                (2, "creatinine", 300.0, 1.1),
            ]
        )
    )
    spec = next(spec for spec in LAB_SPECS if spec.endpoint == "aki_creatinine")
    records = _lab_records(anchors, lookup, spec, lag_window="0_to_8h")
    worst = records.loc[records["outcome_variant"].eq("worst_in_window")].set_index(
        "stay_id"
    )
    assert worst.loc[1, "outcome_observed"] == 1
    assert worst.loc[1, "event"] == 1
    assert worst.loc[2, "outcome_observed"] == 0
    assert np.isnan(worst.loc[2, "event"])


def test_delayed_vis_window_excludes_escalation_that_started_before_window() -> None:
    anchors = _anchors([{"stay_id": 1}])
    lookup = EventLookup(
        _events(
            [
                (1, "vis", 180.0, 0.0),
                (1, "vis", 300.0, 5.0),
                (1, "vis", 600.0, 8.0),
            ]
        )
    )
    acute = _vis_records(anchors, lookup, lag_window="0_to_4h").iloc[0]
    delayed = _vis_records(anchors, lookup, lag_window="4_to_12h").iloc[0]
    assert acute["at_risk"] == 1
    assert acute["event"] == 1
    assert delayed["at_risk"] == 0
    assert np.isnan(delayed["event"])


def test_urine_records_include_strict_relative_decline_and_absolute_oliguria() -> None:
    anchors = _anchors([{"stay_id": 1}])
    lookup = EventLookup(
        _events(
            [
                (1, "urine_output", 60.0, 100.0),
                (1, "urine_output", 180.0, 100.0),
                (1, "urine_output", 300.0, 20.0),
                (1, "urine_output", 420.0, 20.0),
            ]
        )
    )
    records = _urine_records(anchors, lookup, lag_window="0_to_4h").set_index(
        "endpoint"
    )
    assert set(records.index) == {
        "oliguria_kdigo_proxy",
        "urine_output_decline_proxy",
    }
    assert records.loc["urine_output_decline_proxy", "baseline_observed"] == 1
    assert records.loc["urine_output_decline_proxy", "event"] == 1
    assert records.loc["oliguria_kdigo_proxy", "event"] == 1


def test_conservative_composite_does_not_treat_missing_component_as_negative() -> None:
    anchors = _anchors([{"stay_id": 1}, {"stay_id": 2}])
    first = anchors.copy()
    first["endpoint"] = "first"
    first["endpoint_family"] = "laboratory_organ_injury"
    first["lag_window"] = "0_to_8h"
    first["lag_role"] = "cumulative_companion"
    first["outcome_variant"] = "worst_in_window"
    first["source_available"] = 1
    first["at_risk"] = 1
    first["baseline_observed"] = 1
    first["post_observed"] = [1, 0]
    first["outcome_observed"] = [1, 0]
    first["event"] = [0.0, np.nan]
    second = first.copy()
    second["endpoint"] = "second"
    second["post_observed"] = [1, 1]
    second["outcome_observed"] = [1, 1]
    second["event"] = [0.0, 1.0]
    composite = _conservative_any_records(
        {"first": first, "second": second},
        {
            "endpoint": "composite",
            "family": "organ_injury_composite",
            "components": ("first", "second"),
        },
    ).set_index("stay_id")
    assert composite.loc[1, "outcome_observed"] == 1
    assert composite.loc[1, "event"] == 0
    assert composite.loc[2, "outcome_observed"] == 1
    assert composite.loc[2, "event"] == 1


def test_public_runner_returns_all_registered_tables_on_small_cached_fixture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        multiorgan_module,
        "ANALYZED_DEFINITIONS",
        (multiorgan_module.EPISODE_DEFINITIONS[0],),
    )
    monkeypatch.setattr(
        multiorgan_module,
        "LAG_WINDOWS",
        (multiorgan_module.LAG_WINDOWS[0],),
    )
    event_rows: list[tuple[object, str, float, float]] = []
    cohort_rows: list[dict[str, object]] = []
    analysis_rows: list[dict[str, object]] = []
    for stay_id in range(1, 7):
        spo2_values = (95.0, 89.0, 92.0) if stay_id <= 3 else (95.0, 94.0, 93.0)
        for offset, value in zip((60.0, 75.0, 90.0), spo2_values):
            event_rows.append((stay_id, "spo2", offset, value))
        event_rows.extend(
            [
                (stay_id, "lactate", 50.0, 1.0),
                (stay_id, "lactate", 120.0, 2.0 if stay_id <= 3 else 1.1),
                (stay_id, "creatinine", 50.0, 1.0),
                (stay_id, "creatinine", 300.0, 1.4 if stay_id <= 3 else 1.0),
            ]
        )
        cohort_rows.append(
            {
                "dataset": "synthetic",
                "stay_id": stay_id,
                "person_id": f"p{stay_id}",
                "age": 60,
                "shock_icd_flag": 0,
                "baseline_vasoactive_flag": 0,
                "baseline_mcs_flag": 0,
                "followup_end_offset_minutes": 2_000,
                "death_offset_minutes": np.nan,
                "mcs_source_available": 1,
                "pressor_source_available": 1,
                "vis_source_available": 0,
                "urine_output_source_available": 0,
                "respiratory_source_available": 0,
                "rrt_source_available": 1,
            }
        )
        analysis_rows.append(
            {
                "dataset": "synthetic",
                "stay_id": stay_id,
                "baseline_weight_outcome": 80.0,
                "mechanical_ventilation_flag": 0,
            }
        )
    outputs = run_multiorgan_episode_analyses(
        _events(event_rows),
        pd.DataFrame(cohort_rows),
        pd.DataFrame(analysis_rows),
        bootstrap_repetitions=0,
    )
    assert set(outputs) == {
        "multiorgan_episode_records",
        "multiorgan_episode_effects",
        "multiorgan_episode_observation_process",
        "multiorgan_episode_measurement_weighted",
        "multiorgan_episode_meta_analysis",
        "multiorgan_episode_evidence_summary",
        "multiorgan_episode_key_results",
        "multiorgan_episode_endpoint_definitions",
        "multiorgan_episode_availability",
    }
    effects = outputs["multiorgan_episode_effects"]
    assert {"lactate_rise", "aki_creatinine", "vis_escalation"}.issubset(
        set(effects["endpoint"])
    )
    assert "cross_endpoint_focused_q_value" in effects


def test_weighted_complete_observation_uses_cluster_robust_gee() -> None:
    rows = []
    for index in range(120):
        exposed = int(index < 60)
        rows.append(
            {
                "dataset": "synthetic",
                "stay_id": index,
                "person_id": f"p{index}",
                "endpoint": "test_endpoint",
                "endpoint_family": "laboratory_organ_injury",
                "lag_window": "0_to_8h",
                "episode_exposed": exposed,
                "source_available": 1,
                "at_risk": 1,
                "baseline_observed": 1,
                "outcome_observed": 1,
                "event": int(index % (3 if exposed else 6) == 0),
                "baseline_value": 1.0 + index / 1000.0,
                "anchor_offset_minutes": 120.0,
                "anchor_spo2": 92.0,
                "preanchor_sampling_density_per_hr": 4.0,
                "age": 50 + index % 20,
                "shock_icd_flag": 0,
                "baseline_vasoactive_flag": 0,
            }
        )
    frame = pd.DataFrame(rows)
    frame.index = np.arange(1_000, 1_000 + len(frame)) * 3
    result = _selection_weighted_row(frame)
    assert result["status"] == "not_needed_complete_observation"
    assert result["weighted_inference_method"] == "cluster_robust_gee_poisson_log_link"
    assert result["selection_weighted_risk_ratio"] > 1


def test_key_result_requires_cross_endpoint_fdr_in_all_three_analyses_for_robust_label() -> None:
    effects = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "endpoint": endpoint,
                "endpoint_family": "laboratory_organ_injury",
                "lag_window": "0_to_8h",
                "focused_endpoint_window": True,
                "status": "estimated",
                "risk_ratio": 2.0,
                "risk_ratio_ci95_low": 1.2,
                "risk_ratio_ci95_high": 3.0,
                "cross_endpoint_focused_q_value": 0.01,
                "endpoint_focused_q_value": 0.01,
                "adjusted_status": "estimated",
                "adjusted_risk_ratio": 1.6,
                "adjusted_ci95_low": 1.1,
                "adjusted_ci95_high": 2.3,
                "adjusted_claim_ready_epv10": True,
                "cross_endpoint_focused_adjusted_q_value": adjusted_q,
            }
            for endpoint, adjusted_q in (("fully_robust", 0.01), ("ci_supported", 0.20))
        ]
    )
    weighted = pd.DataFrame(
        [
            {
                "dataset": "synthetic",
                "endpoint": endpoint,
                "lag_window": "0_to_8h",
                "status": "estimated",
                "selection_weighted_risk_ratio": 1.5,
                "selection_weighted_ci95_low": 1.05,
                "selection_weighted_ci95_high": 2.2,
                "endpoint_focused_q_value": 0.01,
                "cross_endpoint_focused_q_value": weighted_q,
                "outcome_events_per_parameter": 20.0,
                "outcome_claim_ready_epv10": True,
            }
            for endpoint, weighted_q in (("fully_robust", 0.01), ("ci_supported", 0.20))
        ]
    )
    result = multiorgan_module._build_key_results(
        effects, weighted, pd.DataFrame()
    ).set_index("endpoint")
    assert (
        result.loc["fully_robust", "evidence_label"]
        == "robust_to_adjustment_weighting_and_cross_endpoint_fdr"
    )
    assert (
        result.loc["ci_supported", "evidence_label"]
        == "unadjusted_cross_fdr_with_adjusted_weighted_ci_support"
    )
