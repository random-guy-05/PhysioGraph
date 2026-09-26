"""Protocol-contract and adversarial leakage tests for the SpO2 hypothesis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.spo2_protocol import (
    POST_LANDMARK_HORIZONS_HOURS,
    PRIMARY_ENDPOINTS,
    INSTABILITY_FEATURES,
    PARSIMONIOUS_INSTABILITY_FEATURES,
    SAMPLING_ADJUSTMENT_FEATURES,
    build_endpoint_completeness_audit,
    build_continuous_trajectory_associations,
    build_temporal_precedence,
    compute_early_decompensation_outcomes,
    fit_external_transportability,
    fit_grouped_incremental_models,
    _first_instability,
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
    assert "spo2_instability_proxy_score" not in INSTABILITY_FEATURES
    assert "spo2_instability_proxy_score" not in PARSIMONIOUS_INSTABILITY_FEATURES
    assert "spo2_drop_3_count" in PARSIMONIOUS_INSTABILITY_FEATURES


def test_early_outcomes_cover_primary_secondary_and_boundary_rules():
    cohort = pd.DataFrame(
        {"dataset": ["mimic", "mimic"], "stay_id": [1, 2], "person_id": [10, 20]}
    )
    events = pd.DataFrame(
        [
            _event(1, "lactate", 200, 2.0),
            _event(1, "creatinine", 180, 1.0),
            _event(1, "bilirubin_total", 180, 1.0),
            _event(1, "troponin_t", 180, 0.02),
            _event(1, "urine_output", 60, 200),
            _event(1, "urine_output", 180, 200),
            _event(1, "weight", 0, 80),
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 60, 88),
            # Exactly at the landmark belongs to baseline, never the outcome window.
            _event(1, "lactate", 240, 9.0),
            _event(1, "lactate", 500, 2.7),
            _event(1, "creatinine", 600, 1.4),
            _event(1, "bilirubin_total", 700, 1.6),
            _event(1, "troponin_t", 700, 0.04),
            _event(1, "urine_output", 300, 50),
            _event(1, "urine_output", 500, 200),
            _event(1, "urine_output", 700, 50),
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
    assert row["baseline_lactate_outcome"] == pytest.approx(9.0)
    assert row["lactate_delta_12h"] == pytest.approx(-6.3)
    assert row["lactate_rise_12h_flag"] == 0
    assert row["vis_rise_12h_flag"] == 1
    assert row["aki_creatinine_12h_flag"] == 1
    assert row["hepatic_lab_worsening_12h_flag"] == 1
    assert row["troponin_relative_rise_12h_flag"] == 1
    assert row["troponin_ratio_12h"] == pytest.approx(2.0)
    assert row["urine_output_decline_proxy_12h_flag"] == 1
    # Documented-bin rate is 300 mL / 6 covered hours / 80 kg = 0.625;
    # unobserved bins must not be treated as zero urine output.
    assert row["oliguria_kdigo_proxy_12h_flag"] == 0
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
    assert result["pressor_initiation_12h_observed"] == 0
    assert np.isnan(result["pressor_initiation_12h_flag"])
    assert result["urine_output_12h_observed"] == 0
    assert np.isnan(result["urine_output_decline_proxy_12h_flag"])


def test_incompatible_pressor_units_make_vis_unknown_not_zero():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "vis_source_available": [1],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "vis", 240, 0.0),
            _event(1, "vis", 500, 0.0),
            _event(1, "vis_rate_unstandardized", 500, 7.0),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["vis_rate_units_complete_12h_flag"] == 0
    assert row["vis_observed_12h"] == 0
    assert np.isnan(row["vis_rise_12h_flag"])


def test_source_complete_mcs_absence_is_negative_and_positive_overrides_missing_labs():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "person_id": [5, 6],
            "mcs_source_available": [1, 1],
            "pressor_source_available": [1, 1],
            "followup_end_offset_minutes": [2000, 2000],
        }
    )
    events = pd.DataFrame([_event(2, "mcs", 500)])
    result = compute_early_decompensation_outcomes(events, cohort).set_index("stay_id")
    assert result.loc[1, "mcs_12h_observed"] == 1
    assert result.loc[1, "mcs_12h_flag"] == 0
    assert result.loc[1, "pressor_initiation_12h_observed"] == 1
    assert result.loc[1, "pressor_initiation_12h_flag"] == 0
    assert np.isnan(result.loc[1, "early_decompensation_12h_flag"])
    assert result.loc[2, "mcs_12h_flag"] == 1
    assert result.loc[2, "early_decompensation_12h_flag"] == 1


def test_monotone_event_before_censoring_remains_positive_but_last_value_endpoint_is_unknown():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [600],
            "mcs_source_available": [1],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "lactate", 200, 1.0),
            _event(1, "lactate", 500, 4.0),
            _event(1, "mcs", 550),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["complete_followup_12h"] == 0
    assert row["lactate_rise_12h_observed"] == 0
    assert np.isnan(row["lactate_rise_12h_flag"])
    assert row["lactate_max_rise_12h_flag"] == 1
    assert row["mcs_12h_observed"] == 1
    assert row["mcs_12h_flag"] == 1


def test_censored_non_event_and_post_censor_measurement_remain_unknown():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [600],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "lactate", 200, 1.0),
            _event(1, "lactate", 500, 1.1),
            _event(1, "lactate", 700, 5.0),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["complete_followup_12h"] == 0
    assert row["lactate_last_12h"] == pytest.approx(1.1)
    assert row["lactate_rise_12h_observed"] == 0
    assert np.isnan(row["lactate_rise_12h_flag"])


def test_missing_followup_timestamp_cannot_establish_fixed_horizon_non_event():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [np.nan],
            "mcs_source_available": [1],
        }
    )
    row = compute_early_decompensation_outcomes(pd.DataFrame(), cohort).iloc[0]
    assert row["complete_followup_12h"] == 0
    assert row["mcs_12h_observed"] == 0
    assert np.isnan(row["mcs_12h_flag"])


def test_death_exactly_at_horizon_does_not_censor_complete_window():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [960],
            "death_offset_minutes": [960],
            "mcs_source_available": [1],
        }
    )
    row = compute_early_decompensation_outcomes(pd.DataFrame(), cohort).iloc[0]
    assert row["complete_followup_12h"] == 1
    assert row["death_12h_flag"] == 1
    assert row["mcs_12h_observed"] == 1
    assert row["mcs_12h_flag"] == 0


def test_prevalent_mcs_is_retained_but_excluded_from_incident_mcs_risk_set():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [2000],
            "mcs_source_available": [1],
            "baseline_mcs_flag": [1],
        }
    )
    events = pd.DataFrame(
        [_event(1, "mcs", 500, dataset="eicu")]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["mcs_incident_risk_set_12h_flag"] == 0
    assert row["mcs_12h_observed"] == 0
    assert np.isnan(row["mcs_12h_flag"])


def test_operational_mcs_chart_documentation_is_an_incident_proxy():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [2000],
            "mcs_source_available": [1],
            "baseline_mcs_flag": [0],
        }
    )
    events = pd.DataFrame([_event(1, "mcs_context_lvad", 500, 4.5)])
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["mcs_12h_flag"] == 1
    assert "operational_chart_documentation" in row["mcs_endpoint_method"]


def test_in_hospital_death_is_a_separate_observed_secondary_endpoint():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "person_id": [5, 6],
            "death_offset_minutes": [500, np.nan],
            "followup_end_offset_minutes": [500, 700],
        }
    )
    result = compute_early_decompensation_outcomes(pd.DataFrame(), cohort).set_index("stay_id")
    assert result.loc[1, "death_12h_observed"] == 1
    assert result.loc[1, "death_12h_flag"] == 1
    assert result.loc[2, "death_12h_observed"] == 1
    assert result.loc[2, "death_12h_flag"] == 0
    assert np.isnan(result.loc[2, "lactate_rise_12h_flag"])


def test_true_zero_urine_is_preserved_for_kdigo_but_not_called_a_decline_from_zero():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [5],
            "urine_output_source_available": [1],
            "admission_weight_kg": [80.0],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "urine_output", 60, 0.0, dataset="eicu"),
            _event(1, "urine_output", 300, 0.0, dataset="eicu"),
            _event(1, "urine_output", 500, 0.0, dataset="eicu"),
            _event(1, "urine_output", 700, 0.0, dataset="eicu"),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["urine_output_12h_ml"] == 0
    assert row["urine_output_12h_count"] == 3
    assert row["urine_output_decline_proxy_12h_observed"] == 0
    assert np.isnan(row["urine_output_decline_proxy_12h_flag"])
    assert row["oliguria_kdigo_proxy_12h_flag"] == 1


def test_urine_coverage_bins_are_right_closed_at_two_hour_boundaries():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [5],
            "urine_output_source_available": [1],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "urine_output", 60, 100.0, dataset="eicu"),
            _event(1, "urine_output", 120, 100.0, dataset="eicu"),
            _event(1, "urine_output", 241, 50.0, dataset="eicu"),
            _event(1, "urine_output", 360, 50.0, dataset="eicu"),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["baseline_urine_output_covered_2h_bins"] == 1
    assert row["urine_output_12h_covered_2h_bins"] == 1


def test_urine_rate_uses_covered_time_not_missing_bins_as_zero_output():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [5],
            "urine_output_source_available": [1],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "urine_output", 60, 200.0, dataset="eicu"),
            _event(1, "urine_output", 180, 200.0, dataset="eicu"),
            _event(1, "urine_output", 300, 120.0, dataset="eicu"),
            _event(1, "urine_output", 500, 120.0, dataset="eicu"),
            _event(1, "urine_output", 700, 120.0, dataset="eicu"),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["baseline_urine_output_rate_ml_h"] == pytest.approx(100.0)
    assert row["urine_output_rate_12h_ml_h"] == pytest.approx(60.0)
    assert row["urine_output_12h_coverage_fraction"] == pytest.approx(0.5)
    assert row["urine_output_decline_proxy_12h_flag"] == 0


def test_troponin_t_and_i_are_never_compared_across_assays():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "troponin_t", 200, 0.01),
            _event(1, "troponin_i", 500, 10.0),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["troponin_relative_rise_12h_observed"] == 0
    assert np.isnan(row["troponin_relative_rise_12h_flag"])
    assert np.isnan(row["troponin_ratio_12h"])


def test_prevalent_abnormal_ast_and_platelets_require_actual_worsening():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "ast", 200, 300.0),
            _event(1, "platelets", 200, 80.0),
            _event(1, "ast", 500, 300.0),
            _event(1, "platelets", 500, 80.0),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["ast_injury_12h_flag"] == 0
    assert row["platelet_injury_12h_flag"] == 0


def test_incident_ast_and_platelet_threshold_crossings_are_events():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [5],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "ast", 200, 100.0),
            _event(1, "platelets", 200, 120.0),
            _event(1, "ast", 500, 220.0),
            _event(1, "platelets", 500, 90.0),
        ]
    )
    row = compute_early_decompensation_outcomes(events, cohort).iloc[0]
    assert row["ast_injury_12h_flag"] == 1
    assert row["platelet_injury_12h_flag"] == 1


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
            "spo2_missing_bin_count": rng.integers(0, 12, n),
            "spo2_longest_gap_minutes": rng.uniform(5, 90, n),
            "spo2_dynamics_eligible_flag": 1,
            "spo2_sd": instability + 2,
            "spo2_rmssd": instability + 2.2,
            "spo2_iqr": instability + 1.5,
            "spo2_range": instability + 5,
            "spo2_mad": instability + 1,
            "spo2_abrupt_jump_rate_per_hr": np.clip(instability + 1, 0, None),
            "spo2_abrupt_jump_fraction": np.clip((instability + 2) / 5, 0, 1),
            "spo2_drop_3_count": np.clip(np.rint(instability + 2), 0, None),
            "spo2_dynamics_proxy_score": instability + 3,
            "spo2_instability_proxy_score": instability + 3,
            "age": rng.normal(66, 10, n),
            "is_male": rng.integers(0, 2, n),
            "icu_type": np.where(np.arange(n) % 2, "SICU", "MICU"),
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
        "parsimonious_spo2_instability",
        "full_spo2_instability",
    }
    assert fitted["grouping"].eq(
        "dataset_qualified_person_id_with_stay_fallback"
    ).all()
    assert fitted["preprocessing_scope"].eq("fit_within_each_training_fold").all()
    assert fitted["n_patients"].eq(80).all()
    incremental = fitted.loc[fitted["model"].eq("parsimonious_spo2_instability")].iloc[0]
    assert np.isfinite(incremental["delta_auroc_vs_absolute"])
    assert np.isfinite(incremental["delta_auroc_vs_absolute_ci95_low"])
    assert incremental["incremental_reference"] == "clinical_absolute_spo2_and_sampling"
    reference_columns = set(
        fitted.loc[fitted["model"].eq("absolute_spo2"), "feature_columns"]
        .iloc[0]
        .split(",")
    )
    assert set(SAMPLING_ADJUSTMENT_FEATURES).issubset(reference_columns)


def test_pooled_models_normalize_mixed_categories_and_skip_source_only_endpoints():
    mimic = _model_frame(60)
    eicu = _model_frame(60)
    eicu["dataset"] = "eicu"
    eicu["stay_id"] += 10_000
    eicu["person_id"] += 10_000
    eicu["icu_type"] = np.where(np.arange(len(eicu)) % 2, 1, 2)
    mimic["vis_rise_12h_flag"] = mimic["lactate_rise_12h_flag"]
    eicu["vis_rise_12h_flag"] = np.nan

    result = fit_grouped_incremental_models(
        pd.concat([mimic, eicu], ignore_index=True),
        min_rows=50,
        min_events=10,
        n_splits=3,
        bootstrap_repetitions=5,
    )
    pooled = result.loc[result["analysis_scope"].eq("pooled_secondary")]
    lactate = pooled.loc[pooled["outcome"].eq("lactate_rise_12h_flag")]
    assert lactate["status"].eq("fit").any()
    assert not lactate["status"].eq("failed").any()

    vis = pooled.loc[pooled["outcome"].eq("vis_rise_12h_flag")]
    assert len(vis) == 1
    assert vis.iloc[0]["status"] == "skipped"
    assert vis.iloc[0]["observed_dataset_count"] == 1
    assert vis.iloc[0]["reason"] == (
        "pooled endpoint not observed in at least two datasets"
    )


def test_external_transportability_excludes_nonharmonized_context():
    mimic = _model_frame(60)
    eicu = _model_frame(60)
    eicu["dataset"] = "eicu"
    eicu["stay_id"] += 10_000
    eicu["person_id"] += 10_000
    mimic["admit_year"] = 2150
    eicu["admit_year"] = 2015
    mimic["race_ethnicity"] = "MIMIC_CATEGORY"
    eicu["race_ethnicity"] = "EICU_CATEGORY"
    mimic["unit_admit_source"] = np.nan
    eicu["unit_admit_source"] = "Emergency Department"
    result = fit_external_transportability(
        pd.concat([mimic, eicu], ignore_index=True),
        min_rows=50,
        min_events=10,
    )
    fitted = result.loc[result["status"].eq("fit")]
    assert not fitted.empty
    assert "clinical_only" in set(fitted["model"])
    columns = ",".join(fitted["feature_columns"].astype(str))
    for forbidden in (
        "admit_year",
        "race_ethnicity",
        "unit_admit_source",
        "hospital_id",
        "icu_type",
    ):
        assert forbidden not in columns


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
    assert row["status"] == "fragile_or_underpowered"
    assert row["reason"] == "below_minimum_200_observed_or_20_events_and_20_non_events"
    vis = audit.loc[audit["endpoint"].eq("vis_rise_12h_flag")].iloc[0]
    assert vis["status"] == "unavailable"


def test_endpoint_audit_all_missing_column_is_unavailable():
    frame = pd.DataFrame(
        {
            "dataset": ["eicu"] * 4,
            "vis_rise_12h_flag": [np.nan] * 4,
        }
    )
    audit = build_endpoint_completeness_audit(frame)
    row = audit.loc[audit["endpoint"].eq("vis_rise_12h_flag")].iloc[0]
    assert row["status"] == "unavailable"
    assert row["n_observed"] == 0
    assert row["reason"] == "no_observed_endpoint_values"


def test_continuous_trajectory_analysis_tests_worsening_beyond_absolute_spo2():
    n = 120
    score = np.linspace(-2, 2, n)
    frame = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": np.arange(n),
            "spo2_dynamics_eligible_flag": 1,
            "spo2_dynamics_proxy_score": score,
            "spo2_mean": 95.0,
            "spo2_min": 90.0,
            "spo2_below_90_fraction": 0.0,
            "spo2_sampling_density_per_hr": 4.0,
            "spo2_missing_bin_count": np.resize(np.arange(8), n),
            "spo2_longest_gap_minutes": np.resize(np.arange(10, 70), n),
            "lactate_delta_12h": score * 0.8,
        }
    )
    result = build_continuous_trajectory_associations(
        frame,
        min_rows=50,
        min_groups=50,
        bootstrap_repetitions=100,
    )
    row = result.loc[result["trajectory"].eq("lactate_delta_12h")].iloc[0]
    assert row["status"] == "estimated"
    assert row["partial_spearman_rho_beyond_absolute_spo2_and_sampling"] > 0.99
    assert row["partial_rho_cluster_boot_ci95_low"] > 0.99
    assert {"spo2_missing_bin_count", "spo2_longest_gap_minutes"}.issubset(
        set(row["adjustment_columns"].split(","))
    )


def test_continuous_trajectory_constant_signal_is_non_estimable():
    n = 120
    frame = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": np.arange(n),
            "spo2_dynamics_eligible_flag": 1,
            "spo2_dynamics_proxy_score": 1.0,
            "lactate_delta_12h": np.linspace(-1, 1, n),
        }
    )
    result = build_continuous_trajectory_associations(
        frame,
        min_rows=50,
        min_groups=50,
        bootstrap_repetitions=100,
    )
    row = result.loc[result["trajectory"].eq("lactate_delta_12h")].iloc[0]
    assert row["status"] == "non_estimable"
    assert np.isnan(row["partial_rho_cluster_boot_ci95_low"])


def test_temporal_precedence_reports_ordering_not_causality():
    cohort = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "person_id": [10]})
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 45, 97),
            _event(1, "spo2", 60, 88),
            _event(1, "lactate", 200, 2.0),
            _event(1, "lactate", 400, 99.0),
            _event(1, "lactate", 500, 2.8),
        ]
    )
    records, summary = build_temporal_precedence(
        events, cohort, bootstrap_repetitions=20
    )
    assert records.iloc[0]["lead_time_minutes"] == 440
    assert summary.iloc[0]["claim_scope"] == "landmark_ordering_design_enforced_not_causal_precedence"


def test_two_bins_cannot_create_a_dynamics_onset():
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 60, 85),
        ]
    )
    assert _first_instability(events).empty
    combined = _first_instability(events, include_hypoxemia=True)
    assert combined.iloc[0]["instability_onset_minutes"] == 60


def test_temporal_mcs_onset_excludes_prevalent_mcs():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [10],
            "baseline_mcs_flag": [1],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 45, 96),
            _event(1, "spo2", 60, 88),
            _event(1, "lactate", 200, 1.0),
            _event(1, "lactate", 500, 2.0),
            _event(1, "mcs", 550),
        ]
    )
    records, _ = build_temporal_precedence(
        events, cohort, bootstrap_repetitions=0
    )
    assert "lactate_rise" in set(records["outcome"])
    assert "mcs_initiation" not in set(records["outcome"])


def test_temporal_precedence_excludes_vis_with_incompatible_rate_units():
    cohort = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "person_id": [10]})
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97),
            _event(1, "spo2", 60, 88),
            _event(1, "lactate", 200, 1.0),
            _event(1, "lactate", 500, 2.0),
            _event(1, "vis", 240, 0.0),
            _event(1, "vis", 500, 10.0),
            _event(1, "vis_rate_unstandardized", 500, 7.0),
        ]
    )
    records, _ = build_temporal_precedence(events, cohort, bootstrap_repetitions=0)
    assert "lactate_rise" in set(records["outcome"])
    assert "vis_rise" not in set(records["outcome"])


def test_temporal_urine_decline_requires_protocol_coverage():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [10],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97, dataset="eicu"),
            _event(1, "spo2", 45, 97, dataset="eicu"),
            _event(1, "spo2", 60, 88, dataset="eicu"),
            _event(1, "lactate", 200, 1.0, dataset="eicu"),
            _event(1, "lactate", 500, 2.0, dataset="eicu"),
            _event(1, "urine_output", 60, 200.0, dataset="eicu"),
            _event(1, "urine_output", 180, 200.0, dataset="eicu"),
            _event(1, "urine_output", 300, 1.0, dataset="eicu"),
        ]
    )
    records, _ = build_temporal_precedence(
        events, cohort, bootstrap_repetitions=0
    )
    assert "lactate_rise" in set(records["outcome"])
    assert "urine_output_decline_proxy" not in set(records["outcome"])


def test_temporal_urine_onset_requires_aggregate_decline_not_one_low_bin():
    cohort = pd.DataFrame(
        {
            "dataset": ["eicu"],
            "stay_id": [1],
            "person_id": [10],
            "followup_end_offset_minutes": [2000],
        }
    )
    events = pd.DataFrame(
        [
            _event(1, "spo2", 30, 97, dataset="eicu"),
            _event(1, "spo2", 45, 97, dataset="eicu"),
            _event(1, "spo2", 60, 88, dataset="eicu"),
            _event(1, "lactate", 200, 1.0, dataset="eicu"),
            _event(1, "lactate", 500, 2.0, dataset="eicu"),
            _event(1, "urine_output", 60, 200.0, dataset="eicu"),
            _event(1, "urine_output", 180, 200.0, dataset="eicu"),
            *[
                _event(
                    1,
                    "urine_output",
                    minute,
                    1.0 if minute == 300 else 200.0,
                    dataset="eicu",
                )
                for minute in (300, 420, 540, 660, 780, 900)
            ],
        ]
    )
    records, _ = build_temporal_precedence(
        events, cohort, bootstrap_repetitions=0
    )
    assert "lactate_rise" in set(records["outcome"])
    assert "urine_output_decline_proxy" not in set(records["outcome"])
