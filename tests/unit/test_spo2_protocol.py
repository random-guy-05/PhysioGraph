"""Protocol-contract and adversarial leakage tests for the SpO2 hypothesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.spo2_protocol import (
    POST_LANDMARK_HORIZONS_HOURS,
    PRIMARY_ENDPOINTS,
    build_endpoint_completeness_audit,
    build_temporal_precedence,
    compute_early_decompensation_outcomes,
    fit_grouped_incremental_models,
)


def _event(
    stay_id: int,
    concept: str,
    minute: float,
    value: float | None = None,
    *,
    dataset: str = "mimic",
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "stay_id": stay_id,
        "concept": concept,
        "offset_minutes": minute,
        "value_numeric": value,
        "raw_name": concept,
        "value_text": "",
        "source_table": "synthetic",
    }


def test_protocol_clock_is_twelve_and_twenty_four_hours_after_landmark():
    assert POST_LANDMARK_HORIZONS_HOURS == (12, 24)
    assert "lactate_rise_12h_flag" in PRIMARY_ENDPOINTS
    assert "vis_rise_24h_flag" in PRIMARY_ENDPOINTS


def test_early_outcomes_cover_primary_secondary_and_boundary_rules():
    cohort = pd.DataFrame(
        {"dataset": ["mimic", "mimic"], "stay_id": [1, 2], "person_id": [10, 20]}
    )
    events = pd.DataFrame(
        [
            _event(1, "lactate", 200, 2.0),
            _event(1, "creatinine", 180, 1.0),
            _event(1, "bilirubin_total", 180, 1.0),
            _event(1, "urine_output", 60, 200),
            _event(1, "urine_output", 180, 200),
            _event(1, "weight", 0, 80),
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 60, 88),
            # Exactly at the landmark is excluded from outcomes.
            _event(1, "lactate", 240, 9.0),
            _event(1, "lactate", 500, 2.7),
            _event(1, "creatinine", 600, 1.4),
            _event(1, "bilirubin_total", 700, 1.6),
            _event(1, "urine_output", 500, 200),
            _event(1, "vis", 500, 12.0),
            _event(1, "pressor", 500),
            _event(1, "mcs", 900),
            # 12h post-landmark ends at minute 960; this is 24h-only.
            _event(1, "lactate", 1000, 4.0),
            _event(2, "lactate", 200, 2.0),
            _event(2, "lactate", 500, 2.1),
        ]
    )
    result = compute_early_decompensation_outcomes(events, cohort)
    row = result.loc[result["stay_id"].eq(1)].iloc[0]

    assert row["person_id"] == 10
    assert row["outcome_window_end_minutes_12h"] == 960
    assert row["outcome_window_end_minutes_24h"] == 1680
    assert row["lactate_delta_12h"] == pytest.approx(0.7)
    assert row["lactate_rise_12h_flag"] == 1
    assert row["vis_rise_12h_flag"] == 1
    assert row["aki_creatinine_12h_flag"] == 1
    assert row["hepatic_lab_worsening_12h_flag"] == 1
    assert row["urine_output_decline_proxy_12h_flag"] == 1
    assert row["oliguria_kdigo_proxy_12h_flag"] == 1
    assert row["mcs_12h_flag"] == 1
    assert row["early_decompensation_12h_flag"] == 1
    assert row["lactate_last_12h"] == 2.7
    assert row["lactate_last_24h"] == 4.0


def test_vis_and_urine_are_missing_not_negative_when_source_unavailable():
    cohort = pd.DataFrame({"dataset": ["eicu"], "stay_id": [1], "person_id": [5]})
    events = pd.DataFrame([_event(1, "lactate", 100, 1.0, dataset="eicu")])
    result = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert result["vis_observed_12h"] == 0
    assert np.isnan(result["vis_rise_12h_flag"])
    assert result["urine_output_12h_observed"] == 0
    assert np.isnan(result["urine_output_decline_proxy_12h_flag"])


def _model_frame(n_patients: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(991)
    n = n_patients * 2
    person = np.repeat(np.arange(n_patients), 2)
    instability = rng.normal(size=n_patients).repeat(2) + rng.normal(0, 0.1, n)
    probability = 1 / (1 + np.exp(-(-0.2 + 0.8 * instability)))
    outcome = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": person,
            "lactate_rise_12h_flag": outcome,
            "spo2_plausible_count": 4,
            "spo2_mean": rng.normal(95, 1, n),
            "spo2_min": rng.normal(91, 2, n),
            "spo2_below_90_fraction": rng.uniform(0, 0.2, n),
            "spo2_sampling_density_per_hr": rng.uniform(2, 8, n),
            "spo2_sd": instability + 2,
            "spo2_rmssd": instability + 2.2,
            "spo2_iqr": instability + 1.5,
            "spo2_range": instability + 5,
            "spo2_mad": instability + 1,
            "spo2_abrupt_jump_rate_per_hr": np.clip(instability + 1, 0, None),
            "spo2_instability_proxy_score": instability + 3,
            "age": rng.normal(66, 10, n),
            "is_male": rng.integers(0, 2, n),
        }
    )


def test_incremental_models_are_patient_grouped_and_fold_local():
    frame = _model_frame()
    result = fit_grouped_incremental_models(
        frame,
        min_rows=50,
        min_events=10,
        n_splits=4,
        bootstrap_repetitions=20,
    )
    fitted = result.loc[result["status"].eq("fit")]
    assert set(fitted["model"]) == {
        "clinical_only",
        "absolute_spo2",
        "clinical_plus_spo2_instability",
    }
    assert fitted["grouping"].eq("dataset_qualified_person_id").all()
    assert fitted["preprocessing_scope"].eq("fit_within_each_training_fold").all()
    assert fitted["n_patients"].eq(80).all()
    incremental = fitted.loc[fitted["model"].eq("clinical_plus_spo2_instability")].iloc[0]
    assert np.isfinite(incremental["delta_auroc_vs_absolute"])
    assert np.isfinite(incremental["delta_auroc_vs_absolute_ci95_low"])


def test_endpoint_audit_never_converts_unavailable_to_zero_events():
    frame = pd.DataFrame(
        {
            "dataset": ["mimic"] * 4,
            "lactate_rise_12h_flag": [0, 1, np.nan, np.nan],
        }
    )
    audit = build_endpoint_completeness_audit(frame)
    row = audit.loc[audit["endpoint"].eq("lactate_rise_12h_flag")].iloc[0]
    assert row["n_observed"] == 2
    assert row["events"] == 1
    vis = audit.loc[audit["endpoint"].eq("vis_rise_12h_flag")].iloc[0]
    assert vis["status"] == "unavailable"


def test_temporal_precedence_reports_ordering_not_causality():
    cohort = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "person_id": [10]})
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 60, 88),
            _event(1, "lactate", 200, 2.0),
            _event(1, "lactate", 500, 2.8),
        ]
    )
    records, summary = build_temporal_precedence(
        events, cohort, bootstrap_repetitions=20
    )
    assert records.iloc[0]["lead_time_minutes"] == 440
    assert summary.iloc[0]["claim_scope"] == "landmark_ordering_not_causal_precedence"
