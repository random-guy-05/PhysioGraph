"""Tests for the minimal Colab runner support functions."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import physiograph_colab_core as colab_core

from physiograph_colab_core import (
    LANDMARK_MINUTES,
    assemble_spo2_analysis_frame,
    build_availability_audit,
    build_cohort_flow_table,
    build_endpoint_conclusions,
    build_feature_missingness_table,
    build_lactate_negative_summary,
    build_oof_performance_curves,
    build_respiratory_context_tables,
    build_spo2_raw_event_summary,
    build_spo2_trajectory_summary,
    compute_horizon_outcomes,
    compute_respiratory_support_features,
    compute_rrt_features,
    compute_spo2_features,
    fit_spo2_models,
    fit_spo2_or_pvalue_tables,
    fit_spo2_variability_or_tables,
    lint_claims_and_outputs,
    build_analysis_manifest,
)


def _stay_index() -> pd.DataFrame:
    return pd.DataFrame({"dataset": ["mimic", "mimic"], "stay_id": [1, 2]})


def test_compute_spo2_features_captures_variability_and_sampling():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 0, "time_bin": 0, "window": "observation", "value_numeric": 98},
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 15, "time_bin": 1, "window": "observation", "value_numeric": 88},
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 30, "time_bin": 2, "window": "observation", "value_numeric": 96},
            {"dataset": "mimic", "stay_id": 1, "concept": "hr", "offset_minutes": 30, "time_bin": 2, "window": "observation", "value_numeric": 110},
            {"dataset": "mimic", "stay_id": 2, "concept": "spo2", "offset_minutes": 0, "time_bin": 0, "window": "observation", "value_numeric": 101},
        ]
    )
    result = compute_spo2_features(events, _stay_index())
    stay1 = result.loc[result["stay_id"].eq(1)].iloc[0]
    stay2 = result.loc[result["stay_id"].eq(2)].iloc[0]

    assert stay1["spo2_plausible_count"] == 3
    assert stay1["spo2_missing_bin_count"] == 13
    assert stay1["spo2_below_90_fraction"] == 1 / 3
    assert stay1["spo2_rmssd"] > 0
    assert stay1["spo2_dynamics_proxy_score"] > 0
    assert stay1["spo2_abrupt_jump_count"] == 2
    assert stay1["spo2_sustained_abrupt_jump_episode_count"] == 1
    assert stay2["spo2_plausible_count"] == 0
    assert stay2["spo2_implausible_count"] == 1


def test_two_spo2_bins_do_not_create_an_eligible_dynamics_signal():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 0, "value_numeric": 98},
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 15, "value_numeric": 88},
        ]
    )
    row = compute_spo2_features(events, pd.DataFrame({"dataset": ["mimic"], "stay_id": [1]})).iloc[0]
    assert row["spo2_dynamics_eligible_flag"] == 0
    assert np.isnan(row["spo2_rmssd"])
    assert np.isnan(row["spo2_drop_3_count"])
    assert np.isnan(row["spo2_dynamics_proxy_score"])
    assert row["spo2_below_90_fraction"] == pytest.approx(0.5)


def test_respiratory_support_features_fallback_and_flags():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "fio2", "raw_name": "FiO2", "value_text": "", "source_table": "chartevents", "offset_minutes": 10, "window": "observation", "value_numeric": 60},
            {"dataset": "mimic", "stay_id": 1, "concept": "device", "raw_name": "BiPAP oxygen device", "value_text": "", "source_table": "chartevents", "offset_minutes": 20, "window": "observation", "value_numeric": np.nan},
            {"dataset": "mimic", "stay_id": 2, "concept": "spo2", "raw_name": "SpO2", "value_text": "", "source_table": "chartevents", "offset_minutes": 20, "window": "observation", "value_numeric": 95},
        ]
    )
    result = compute_respiratory_support_features(events, _stay_index())
    stay1 = result.loc[result["stay_id"].eq(1)].iloc[0]
    stay2 = result.loc[result["stay_id"].eq(2)].iloc[0]

    assert stay1["fio2_measurement_count"] == 1
    assert stay1["fio2_max"] == 60
    assert stay1["noninvasive_ventilation_flag"] == 1
    assert stay1["oxygen_device_flag"] == 1
    assert stay1["resp_support_any_flag"] == 1
    assert stay2["resp_support_any_flag"] == 0


def test_rrt_features_detect_baseline_dialysis():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "procedure", "raw_name": "CRRT dialysis", "value_text": "", "source_table": "procedureevents", "offset_minutes": 60, "window": "observation"},
            {"dataset": "mimic", "stay_id": 2, "concept": "procedure", "raw_name": "other", "value_text": "", "source_table": "procedureevents", "offset_minutes": 60, "window": "observation"},
        ]
    )
    result = compute_rrt_features(events, _stay_index())
    stay1 = result.loc[result["stay_id"].eq(1)].iloc[0]
    stay2 = result.loc[result["stay_id"].eq(2)].iloc[0]

    assert stay1["baseline_rrt_flag"] == 1
    assert stay1["baseline_dialysis_flag"] == 1
    assert stay1["rrt_or_dialysis_flag"] == 1
    assert stay2["rrt_or_dialysis_flag"] == 0


def test_horizon_outcomes_are_post_landmark_and_cumulative():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "death_offset_minutes": [3000, 120],
        }
    )
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "mcs", "raw_name": "iabp", "value_text": "", "source_table": "procedureevents", "offset_minutes": 300, "window": "outcome", "is_intervention": 1},
            {"dataset": "mimic", "stay_id": 2, "concept": "mcs", "raw_name": "iabp", "value_text": "", "source_table": "procedureevents", "offset_minutes": 200, "window": "observation", "is_intervention": 1},
        ]
    )
    result = compute_horizon_outcomes(events, cohort, _stay_index(), horizons=(48, 72))
    stay1 = result.loc[result["stay_id"].eq(1)].iloc[0]
    stay2 = result.loc[result["stay_id"].eq(2)].iloc[0]

    assert stay1["mcs_48h_flag"] == 1
    assert stay1["death_48h_flag"] == 0
    assert stay1["death_72h_flag"] == 1
    assert stay1["mcs_or_death_72h_flag"] == 1
    assert stay2["mcs_48h_flag"] == 0
    assert stay2["death_48h_flag"] == 0


def test_protocol_outcomes_override_legacy_24h_mcs_label():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic"],
            "stay_id": [1],
            "person_id": [10],
            "followup_end_offset_minutes": [2000],
            "mcs_source_available": [1],
            "baseline_mcs_flag": [0],
        }
    )
    features = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1]})
    labels = pd.DataFrame(
        {"dataset": ["mimic"], "stay_id": [1], "mcs_24h_flag": [1]}
    )
    frame, _ = assemble_spo2_analysis_frame(
        cohort=cohort,
        events=pd.DataFrame(),
        features=features,
        labels=labels,
    )
    assert frame.iloc[0]["mcs_24h_flag"] == 0
    assert frame.iloc[0]["endpoint_definition_version"] == "spo2_protocol_v2.2"


def test_fit_spo2_variability_or_tables_returns_or_ci():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(42)
    n = 500
    spo2_sd = rng.normal(2.0, 0.6, size=n)
    spo2_rmssd = rng.normal(2.1, 0.7, size=n)
    spo2_iqr = rng.normal(1.8, 0.5, size=n)
    spo2_range = rng.normal(5.0, 1.2, size=n)
    spo2_mad = rng.normal(1.3, 0.4, size=n)
    jump_rate = rng.normal(0.5, 0.2, size=n)
    instability = rng.normal(2.0, 0.9, size=n)
    spo2_mean = rng.normal(95.0, 2.0, size=n)
    spo2_min = rng.normal(90.0, 3.0, size=n)
    below90 = np.clip(rng.normal(0.08, 0.06, size=n), 0, 1)
    age = rng.normal(66, 12, size=n)
    lactate = np.clip(rng.normal(2.2, 1.0, size=n), 0.2, None)
    creat = np.clip(rng.normal(1.3, 0.5, size=n), 0.2, None)
    map_ = rng.normal(72, 10, size=n)
    rr = rng.normal(20, 5, size=n)
    fio2 = np.clip(rng.normal(45, 15, size=n), 21, 100)
    male = rng.integers(0, 2, size=n)
    resp_support = rng.integers(0, 2, size=n)
    mech = rng.integers(0, 2, size=n)
    niv = rng.integers(0, 2, size=n)
    rrt = rng.integers(0, 2, size=n)
    dataset = np.where(rng.random(n) > 0.5, "mimic", "eicu")

    logit = (
        -2.7
        + 0.75 * ((spo2_sd - spo2_sd.mean()) / spo2_sd.std())
        + 0.25 * ((below90 - below90.mean()) / below90.std())
        + 0.20 * ((lactate - lactate.mean()) / lactate.std())
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    death = rng.binomial(1, p, size=n)

    df = pd.DataFrame(
        {
            "death_168h_flag": death,
            "mcs_or_death_168h_flag": death,
            "target": rng.binomial(1, np.clip(p + 0.05, 0, 1), size=n),
            "spo2_sd": spo2_sd,
            "spo2_rmssd": spo2_rmssd,
            "spo2_iqr": spo2_iqr,
            "spo2_range": spo2_range,
            "spo2_mad": spo2_mad,
            "spo2_abrupt_jump_rate_per_hr": jump_rate,
            "spo2_abrupt_jump_count": np.maximum(0, np.round(jump_rate * 4)).astype(float),
            "spo2_instability_proxy_score": instability,
            "spo2_mean": spo2_mean,
            "spo2_min": spo2_min,
            "spo2_below_90_fraction": below90,
            "spo2_sampling_density_per_hr": np.clip(rng.normal(6, 2, size=n), 0.25, None),
            "age": age,
            "is_male": male,
            "baseline_lactate": lactate,
            "baseline_creatinine": creat,
            "baseline_map": map_,
            "baseline_resp_rate": rr,
            "fio2_max": fio2,
            "resp_support_any_flag": resp_support,
            "mechanical_ventilation_flag": mech,
            "noninvasive_ventilation_flag": niv,
            "rrt_or_dialysis_flag": rrt,
            "dataset": dataset,
        }
    )

    out = fit_spo2_variability_or_tables(df, min_rows=200, min_events=20)
    fitted = out.loc[out["status"].eq("fit")]
    assert not fitted.empty

    rows = fitted.loc[
        fitted["outcome"].eq("death_168h_flag")
        & fitted["feature"].eq("spo2_sd")
    ].copy()
    assert not rows.empty
    model_order = {"clinical_adjusted": 0, "oxygen_adjusted": 1, "unadjusted": 2}
    rows["model_rank"] = rows["model"].map(model_order).fillna(99)
    row = rows.sort_values("model_rank").iloc[0]
    assert row["or_per_1sd"] > 1.0
    assert row["ci95_high"] > row["ci95_low"]


def test_variability_or_table_is_schema_stable_when_all_models_skip():
    df = pd.DataFrame(
        {
            "dataset": ["mimic"] * 8,
            "stay_id": range(8),
            "death_168h_flag": [0] * 7 + [1],
            "spo2_dynamics_eligible_flag": [1] * 8,
            "spo2_sd": np.linspace(1, 2, 8),
        }
    )
    result = fit_spo2_variability_or_tables(df, min_rows=200, min_events=20)
    assert result["status"].eq("skipped").all()
    assert {"p_value", "p_value_adj", "or_per_1sd"}.issubset(result.columns)

def test_post_landmark_spo2_excluded_from_spo2_features():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 30, "time_bin": 2, "window": "observation", "value_numeric": 95},
            {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 300, "time_bin": 20, "window": "outcome", "value_numeric": 80},
        ]
    )
    result = compute_spo2_features(events, _stay_index().iloc[:1])
    row = result.iloc[0]
    assert row["spo2_plausible_count"] == 1
    assert row["spo2_min"] == 95


def test_post_landmark_fio2_vent_rrt_excluded_from_controls():
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "fio2", "raw_name": "FiO2", "value_text": "", "source_table": "chartevents", "offset_minutes": 10, "window": "observation", "value_numeric": 50},
            {"dataset": "mimic", "stay_id": 1, "concept": "device", "raw_name": "BiPAP", "value_text": "", "source_table": "chartevents", "offset_minutes": 300, "window": "outcome", "value_numeric": np.nan},
            {"dataset": "mimic", "stay_id": 1, "concept": "procedure", "raw_name": "CRRT dialysis", "value_text": "", "source_table": "procedureevents", "offset_minutes": 400, "window": "outcome"},
        ]
    )
    resp = compute_respiratory_support_features(events, _stay_index().iloc[:1])
    rrt = compute_rrt_features(events, _stay_index().iloc[:1])
    r = resp.iloc[0]
    d = rrt.iloc[0]
    assert r["fio2_max"] == 50
    assert r["noninvasive_ventilation_flag"] == 0
    assert d["rrt_or_dialysis_flag"] == 0


def test_analysis_frame_preserves_eligible_key_missing_from_feature_table():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "person_id": [10, 20],
            "age": [60, 70],
            "is_male": [1, 0],
            "cohort_hf_flag": [1, 1],
            "shock_icd_flag": [0, 0],
            "excluded_before_landmark_flag": [0, 0],
            "followup_end_offset_minutes": [2000, 2000],
        }
    )
    features = pd.DataFrame(
        {"dataset": ["mimic"], "stay_id": [1], "custom_feature": [1.2]}
    )
    labels = pd.DataFrame(
        {"dataset": ["mimic", "mimic"], "stay_id": [1, 2], "target": [0, 0]}
    )
    frame, _ = assemble_spo2_analysis_frame(
        cohort=cohort, events=pd.DataFrame(), features=features, labels=labels
    )
    assert set(frame["stay_id"]) == {1, 2}
    assert frame.loc[frame["stay_id"].eq(2), "age"].iloc[0] == 70
    assert np.isnan(frame.loc[frame["stay_id"].eq(2), "custom_feature"]).all()


def test_analysis_frame_cannot_reintroduce_cohort_exclusion_from_artifacts():
    cohort = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "excluded_before_landmark_flag": [0, 1],
        }
    )
    features = pd.DataFrame(
        {"dataset": ["mimic", "mimic"], "stay_id": [1, 2], "age": [60, 70]}
    )
    labels = pd.DataFrame(
        {"dataset": ["mimic", "mimic"], "stay_id": [1, 2], "target": [0, 1]}
    )
    frame, _ = assemble_spo2_analysis_frame(
        cohort=cohort, events=pd.DataFrame(), features=features, labels=labels
    )
    assert frame["stay_id"].tolist() == [1]


def test_horizon_boundary_excludes_leq_240_includes_gt_240():
    cohort = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "death_offset_minutes": [np.nan]})
    events = pd.DataFrame(
        [
            {"dataset": "mimic", "stay_id": 1, "concept": "mcs", "raw_name": "iabp", "value_text": "", "source_table": "p", "offset_minutes": 240, "window": "outcome"},
            {"dataset": "mimic", "stay_id": 1, "concept": "mcs", "raw_name": "iabp", "value_text": "", "source_table": "p", "offset_minutes": 241, "window": "outcome"},
        ]
    )
    result = compute_horizon_outcomes(events, cohort, pd.DataFrame({"dataset": ["mimic"], "stay_id": [1]}), horizons=(48,))
    row = result.iloc[0]
    assert row["mcs_48h_flag"] == 1
    assert LANDMARK_MINUTES == 240


def _synthetic_model_df(n: int = 120, *, zero_spo2_rows: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    stay_ids = np.arange(n)
    plausible = np.ones(n)
    plausible[:zero_spo2_rows] = 0
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "mcs_or_death_168h_flag": y,
            "spo2_plausible_count": plausible,
            "spo2_min": np.where(plausible > 0, rng.normal(92, 2, n), np.nan),
            "spo2_mean": np.where(plausible > 0, rng.normal(95, 1, n), np.nan),
            "spo2_sd": np.where(plausible > 0, rng.normal(1.5, 0.3, n), np.nan),
            "spo2_rmssd": np.where(plausible > 0, rng.normal(1.6, 0.4, n), np.nan),
            "spo2_sampling_density_per_hr": np.where(plausible > 0, rng.normal(4, 1, n), 0),
            "spo2_below_90_fraction": np.where(plausible > 0, rng.uniform(0, 0.2, n), np.nan),
            "spo2_instability_proxy_score": np.where(plausible > 0, rng.normal(1, 0.5, n), np.nan),
            "age": rng.normal(65, 10, n),
            "is_male": rng.integers(0, 2, n),
            "baseline_lactate": rng.normal(2, 0.5, n),
        }
    )
    return df


def test_fit_spo2_models_excludes_zero_plausible_and_reports_scope():
    pytest.importorskip("sklearn")
    df = _synthetic_model_df(n=120, zero_spo2_rows=15)
    out = fit_spo2_models(df, min_rows=20, min_events=10, n_splits=3)
    fit_rows = out.loc[out["status"].eq("fit")]
    assert not fit_rows.empty
    assert "auroc_apparent" not in out.columns
    assert "auroc_oof" in out.columns
    row = fit_rows.iloc[0]
    assert row["n_input_rows"] == 120
    assert row["n_measured_spo2_rows"] == 105
    assert "metric_scope" in row
    assert row["n"] == 105


def test_or_table_applies_fdr_to_fit_rows():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(11)
    n = 250
    feature = rng.normal(0, 1, n)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "mcs_or_death_168h_flag": y,
            "spo2_min": feature,
            "spo2_mean": feature + 3,
            "spo2_sd": rng.normal(1, 0.2, n),
            "spo2_rmssd": rng.normal(1, 0.2, n),
            "spo2_sampling_density_per_hr": rng.normal(5, 1, n),
            "spo2_missing_bin_count": rng.integers(0, 12, n),
            "spo2_longest_gap_minutes": rng.uniform(5, 90, n),
            "spo2_dynamics_eligible_flag": np.ones(n),
            "spo2_below_90_fraction": rng.uniform(0, 0.2, n),
            "spo2_abrupt_jump_fraction": rng.uniform(0, 0.4, n),
            "spo2_drop_3_count": rng.integers(0, 5, n),
            "spo2_dynamics_proxy_score": rng.normal(1, 0.3, n),
            "spo2_instability_proxy_score": rng.normal(1, 0.3, n),
            "spo2_plausible_count": np.ones(n),
            "age": rng.normal(65, 8, n),
        }
    )
    out = fit_spo2_or_pvalue_tables(df, min_rows=50, min_events=20)
    fitted = out.loc[out["status"].eq("fit")]
    assert not fitted.empty
    assert "p_value_adj" in fitted.columns
    assert "n_tests" in fitted.columns
    assert fitted["p_value_adj"].notna().all()
    dynamics = fitted.loc[fitted["feature"].eq("spo2_rmssd")].iloc[0]
    assert dynamics["adjustment_set"] == "clinical_absolute_spo2_and_sampling"
    assert {
        "spo2_sampling_density_per_hr",
        "spo2_missing_bin_count",
        "spo2_longest_gap_minutes",
    }.issubset(set(dynamics["adjustment_columns"].split(",")))
    assert dynamics["collinear_adjustment_columns_dropped"]


def test_or_table_skips_when_events_below_floor():
    pytest.importorskip("statsmodels")
    df = pd.DataFrame(
        {
            "mcs_or_death_168h_flag": [0] * 50 + [1] * 5,
            "spo2_min": np.linspace(85, 98, 55),
            "spo2_mean": np.linspace(90, 99, 55),
            "spo2_sd": np.ones(55),
            "spo2_rmssd": np.ones(55),
            "spo2_sampling_density_per_hr": np.ones(55) * 4,
            "spo2_below_90_fraction": np.zeros(55),
            "spo2_instability_proxy_score": np.ones(55),
            "spo2_plausible_count": np.ones(55),
        }
    )
    out = fit_spo2_or_pvalue_tables(df, min_rows=20, min_events=20)
    assert out["status"].eq("skipped").all()
    assert out["reason"].str.contains("insufficient", case=False).any()


def test_or_table_skips_when_non_events_below_floor():
    pytest.importorskip("statsmodels")
    df = pd.DataFrame(
        {
            "mcs_or_death_168h_flag": [0] * 5 + [1] * 50,
            "spo2_min": np.linspace(85, 98, 55),
            "spo2_plausible_count": np.ones(55),
        }
    )
    out = fit_spo2_or_pvalue_tables(df, min_rows=20, min_events=20)
    assert out["status"].eq("skipped").all()
    assert out["non_events"].eq(5).all()
    assert out["reason"].str.contains("non-events", case=False).all()


def test_or_table_classifies_perfect_separation_as_non_estimable():
    pytest.importorskip("statsmodels")
    y = np.repeat([0, 1], 40)
    df = pd.DataFrame(
        {
            "mcs_or_death_168h_flag": y,
            "spo2_min": y.astype(float),
            "spo2_plausible_count": np.ones(len(y)),
        }
    )
    out = fit_spo2_or_pvalue_tables(df, min_rows=40, min_events=20)
    assert not out.empty
    assert out["status"].eq("non_estimable").all()
    assert out["reason"].str.contains(
        "separation|non-finite|non-estimable|unstable", case=False, regex=True
    ).all()


def test_pooled_or_tables_skip_endpoints_observed_in_only_one_source():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(81)
    n = 80
    frame = pd.DataFrame(
        {
            "dataset": ["mimic"] * n + ["eicu"] * n,
            "stay_id": np.arange(2 * n),
            "lactate_rise_12h_flag": [*np.tile([0, 1], n // 2), *([np.nan] * n)],
            "spo2_min": rng.normal(92, 2, 2 * n),
            "spo2_plausible_count": np.ones(2 * n),
        }
    )
    out = fit_spo2_or_pvalue_tables(frame, min_rows=40, min_events=20)
    pooled = out.loc[
        out["analysis_scope"].eq("pooled_secondary")
        & out["outcome"].eq("lactate_rise_12h_flag")
    ]
    assert not pooled.empty
    assert pooled["status"].eq("skipped").all()
    assert pooled["reason"].eq(
        "pooled endpoint not observed in at least two datasets"
    ).all()


def test_respiratory_context_returns_strata_or_skip():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame(
        {
            "dataset": np.where(np.arange(n) % 2, "mimic", "eicu"),
            "stay_id": np.arange(n),
            "person_id": np.arange(n),
            "lactate_rise_24h_flag": rng.integers(0, 2, n),
            "spo2_below_90_fraction": rng.uniform(0, 0.3, n),
            "spo2_plausible_count": np.ones(n),
            "spo2_dynamics_proxy_score": rng.normal(1.0, 0.3, n),
            "spo2_dynamics_eligible_flag": np.ones(n),
            "resp_support_any_flag": rng.integers(0, 2, n),
            "mechanical_ventilation_flag": rng.integers(0, 2, n),
            "fio2_max": rng.uniform(21, 80, n),
            "age": rng.normal(65, 10, n),
        }
    )
    out = build_respiratory_context_tables(df)
    assert not out.empty
    interaction = out.loc[out["analysis"].notna()]
    assert len(interaction) == 4
    assert interaction["status"].eq("fit").all()
    assert interaction["nonfinite_nuisance_inference_count"].eq(0).all()


def test_binomial_inference_gate_keeps_finite_focal_term_with_sparse_nuisance(
    monkeypatch,
):
    sm = pytest.importorskip("statsmodels.api")

    class FakeResult:
        converged = True
        params = pd.Series({"const": 0.0, "focal": 0.2, "rare_level": -1.0})
        bse = pd.Series({"const": 0.1, "focal": 0.2, "rare_level": np.nan})
        pvalues = pd.Series({"const": 1.0, "focal": 0.3, "rare_level": np.nan})

        @staticmethod
        def conf_int():
            return pd.DataFrame(
                {
                    0: {"const": -0.2, "focal": -0.2, "rare_level": np.nan},
                    1: {"const": 0.2, "focal": 0.6, "rare_level": np.nan},
                }
            )

    class FakeGlm:
        @staticmethod
        def fit(**_kwargs):
            return FakeResult()

    monkeypatch.setattr(sm, "GLM", lambda *_args, **_kwargs: FakeGlm())
    design = pd.DataFrame(
        {
            "const": [1.0, 1.0, 1.0, 1.0],
            "focal": [0.0, 1.0, 0.0, 1.0],
            "rare_level": [0.0, 0.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    cluster = pd.Series([1, 2, 3, 4])

    fit = colab_core._fit_binomial_glm_inference(
        sm,
        y,
        design,
        cluster,
        required_terms=["focal"],
    )
    assert fit._physiograph_nonfinite_nuisance_terms == ("rare_level",)

    with pytest.raises(
        colab_core._NonEstimableInferenceError,
        match="non-finite required-term inference",
    ):
        colab_core._fit_binomial_glm_inference(sm, y, design, cluster)


def test_lactate_negative_includes_event_counts_and_fragility():
    df = pd.DataFrame(
        {
            "baseline_lactate": [1.0, 1.5, 3.0, 0.8],
            "mcs_or_death_168h_flag": [0, 1, 1, 0],
            "spo2_plausible_count": [2, 3, 1, 0],
            "spo2_min": [94, 90, 88, np.nan],
            "spo2_rmssd": [1.0, 1.2, 1.5, np.nan],
            "spo2_below_90_fraction": [0.1, 0.2, 0.3, np.nan],
        }
    )
    out = build_lactate_negative_summary(df)
    assert "n_events" in out.columns
    assert "fragility_label" in out.columns
    assert "analysis_note" in out.columns
    row = out.loc[out["outcome"].eq("mcs_or_death_168h_flag")].iloc[0]
    assert row["n_lactate_negative"] == 2
    assert row["n_events"] == 1


def test_raw_summaries_use_protocol_gaps_and_stay_bin_weighting():
    raw = pd.DataFrame(
        {
            "dataset": ["mimic"] * 7,
            "stay_id": [1, 1, 1, 1, 1, 1, 2],
            "offset_minutes": [0, 60, 61, 75, 76, 77, 60],
            "time_bin": [0, 4, 4, 5, 5, 5, 4],
            "value_numeric": [96, 80, 80, 81, 81, 81, 100],
        }
    )
    analysis = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "lactate_rise_12h_flag": [1, 1],
        }
    )
    summary = build_spo2_raw_event_summary(raw, analysis)
    dataset_row = summary.loc[summary["scope"].eq("dataset")].iloc[0]
    assert dataset_row["n_qualified_transitions"] == 1
    assert dataset_row["abrupt_jump_fraction"] == 0

    trajectory = build_spo2_trajectory_summary(
        raw, analysis, outcome="lactate_rise_12h_flag"
    )
    bin_four = trajectory.loc[
        trajectory["analysis_scope"].eq("mimic")
        & trajectory["time_bin"].eq(4)
        & trajectory["outcome_value"].eq(1)
    ].iloc[0]
    assert bin_four["n_stays"] == 2
    assert bin_four["mean"] == 90


def test_availability_audit_race_absent_explicit(monkeypatch):
    artifacts = {
        "mimic": {
            "cohort": pd.DataFrame({"dataset": ["mimic"], "stay_id": [1]}),
            "events": pd.DataFrame(
                [
                    {"dataset": "mimic", "stay_id": 1, "concept": "spo2", "offset_minutes": 10, "window": "observation", "value_numeric": 96},
                ]
            ),
            "features": pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "age": [70]}),
            "labels": pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "mcs_or_death_168h_flag": [0], "mcs_168h_flag": [0], "death_168h_flag": [0]}),
        }
    }
    analysis = pd.DataFrame(
        {"dataset": ["mimic"], "stay_id": [1], "spo2_plausible_count": [1]}
    )
    monkeypatch.setattr(
        colab_core,
        "assemble_spo2_analysis_frame",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("availability audit must reuse the authoritative frame")
        ),
    )
    out = build_availability_audit(artifacts, analysis)
    assert out.iloc[0]["race_ethnicity_status"] == "absent_explicit"
    assert out.iloc[0]["spo2_measured_rows"] == 1

    artifacts_race = {
        "mimic": {
            **artifacts["mimic"],
            "cohort": pd.DataFrame({"dataset": ["mimic"], "stay_id": [1], "race": ["white"]}),
        }
    }
    out2 = build_availability_audit(artifacts_race, analysis)
    assert out2.iloc[0]["race_ethnicity_status"] == "present"


def test_claims_linter_catches_apparent_metrics():
    bad_metrics = pd.DataFrame([{"status": "fit", "auroc_apparent": 0.9}])
    warnings = lint_claims_and_outputs(
        model_metrics=bad_metrics,
        manifest={"fresh_colab_execution": False},
        output_paths={},
    )
    assert warnings["severity"].eq("error").any()
    assert warnings["check"].str.contains("apparent").any()

    good_metrics = pd.DataFrame([{"status": "fit", "auroc_oof": 0.75}])
    clean = lint_claims_and_outputs(
        model_metrics=good_metrics,
        manifest={
            "fresh_colab_execution": False,
            "build_new": True,
            "execution_environment": "local",
        },
        output_paths={},
    )
    assert not clean["check"].eq("apparent_metrics_in_model_output").any()
    provenance = clean.loc[clean["check"].eq("execution_provenance"), "detail"]
    assert provenance.tolist() == ["fresh_local_execution_not_colab"]
    assert not clean["detail"].astype(str).str.contains("precomputed").any()


def test_manifest_propagates_fresh_execution_truthfully(tmp_path):
    manifest = build_analysis_manifest(
        build_new=True,
        fresh_colab_execution=True,
        paths={"model_metrics": tmp_path / "metrics.csv"},
        datasets=["mimic"],
        rows=100,
        claims_warnings=pd.DataFrame(),
        execution_environment="colab",
        run_comparator_requested=False,
        max_stays=None,
        max_chunks=None,
        chunk_size=500_000,
        require_all_requested_datasets=True,
    )
    assert manifest["fresh_colab_execution"] is True
    assert manifest["build_new"] is True
    assert manifest["schema_version"] == "physiograph_spo2_study_v2.3"
    assert len(manifest["run_id"]) == 32
    assert manifest["execution_environment"] == "colab"
    assert manifest["max_stays"] is None
    assert manifest["max_chunks"] is None
    assert manifest["chunk_size"] == 500_000
    assert manifest["require_all_requested_datasets"] is True
    assert manifest["outcome_clock"] == "12_and_24_hours_after_4h_landmark"
    assert manifest["output_fingerprint_integrity_basis"] == "sha256_and_size_bytes"
    assert manifest["output_mtime_ns_role"] == "informational_cloud_sync_may_change_it"


def test_colab_runner_commits_explicit_complete_status(tmp_path, monkeypatch):
    artifact = {
        "cohort": pd.DataFrame(),
        "events": pd.DataFrame(),
        "features": pd.DataFrame(),
        "labels": pd.DataFrame(),
    }
    monkeypatch.setattr(
        colab_core,
        "_run_or_load_dataset",
        lambda dataset, **kwargs: artifact,
    )

    def fake_drilldown(dataset_artifacts, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"run_id": "test-run"}))
        return {"paths": {"manifest": manifest_path}}

    def fake_archive(project_root, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "archive_candidates.json"
        path.write_text("{}")
        return path

    monkeypatch.setattr(colab_core, "run_spo2_drilldown", fake_drilldown)
    monkeypatch.setattr(colab_core, "write_archive_candidates", fake_archive)
    output_root = tmp_path / "output"
    colab_core.run_physiograph_colab(
        project_root=tmp_path,
        build_new=True,
        mimic_root=tmp_path / "mimic",
        eicu_root=tmp_path / "eicu",
        output_root=output_root,
        execution_environment="local",
        require_all_requested_datasets=True,
    )
    status = json.loads((output_root / "run_status.json").read_text())
    assert status["status"] == "complete"
    assert status["requested_datasets"] == ["eicu", "mimic"]
    assert status["started_at_utc"]
    assert status["completed_at_utc"]
    assert status["duration_seconds"] >= 0
    assert status["spo2_manifest_sha256"] == colab_core._file_sha256(
        output_root / "spo2_drilldown" / "manifest.json"
    )


def test_colab_runner_commits_failure_status_for_analysis_error(tmp_path, monkeypatch):
    artifact = {
        "cohort": pd.DataFrame(),
        "events": pd.DataFrame(),
        "features": pd.DataFrame(),
        "labels": pd.DataFrame(),
    }
    monkeypatch.setattr(
        colab_core,
        "_run_or_load_dataset",
        lambda dataset, **kwargs: artifact,
    )
    monkeypatch.setattr(
        colab_core,
        "run_spo2_drilldown",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("analysis boom")),
    )
    output_root = tmp_path / "output"
    with pytest.raises(RuntimeError, match="analysis boom"):
        colab_core.run_physiograph_colab(
            project_root=tmp_path,
            build_new=True,
            mimic_root=tmp_path / "mimic",
            eicu_root=tmp_path / "eicu",
            output_root=output_root,
            execution_environment="local",
            require_all_requested_datasets=True,
        )
    status = json.loads((output_root / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["reason"] == "analysis boom"
    assert status["completed_at_utc"]


def test_analysis_only_rejects_etl_rebuild(tmp_path):
    with pytest.raises(ValueError, match="incompatible"):
        colab_core.run_physiograph_colab(
            project_root=tmp_path,
            build_new=True,
            analysis_only=True,
            output_root=tmp_path / "output",
        )


def test_analysis_only_loader_allows_analysis_code_change(tmp_path, monkeypatch):
    artifact = {
        "cohort": pd.DataFrame(),
        "events": pd.DataFrame(),
        "features": pd.DataFrame(),
        "labels": pd.DataFrame(),
    }
    observed = {}

    def fake_ready(dataset_dir, **kwargs):
        observed.update(kwargs)
        return not kwargs["require_current_code_hash"]

    monkeypatch.setattr(colab_core, "_artifact_ready", fake_ready)
    monkeypatch.setattr(colab_core, "_load_dataset_artifacts", lambda path: artifact)
    loaded = colab_core._run_or_load_dataset(
        "mimic",
        dataset_dir=tmp_path,
        mimic_root=None,
        eicu_root=None,
        build_new=False,
        max_stays=None,
        max_chunks=None,
        chunk_size=500_000,
        analysis_only=True,
    )
    assert loaded is artifact
    assert observed["require_current_code_hash"] is False


def test_cohort_flow_does_not_treat_missing_columns_as_scalar_series():
    artifacts = {
        "mimic": {
            "cohort": pd.DataFrame({"dataset": ["mimic"] * 2, "stay_id": [1, 2]})
        }
    }
    analysis = pd.DataFrame({"dataset": ["mimic"] * 2, "stay_id": [1, 2]})
    result = build_cohort_flow_table(artifacts, analysis)
    assert result.loc[result["step"].eq("analysis_frame_assembled"), "n"].iloc[0] == 2
    assert result.loc[
        result["step"].eq("at_least_one_plausible_spo2_0_to_4h"), "n"
    ].iloc[0] == 0


def test_oof_performance_curves_include_all_three_curve_types():
    predictions = pd.DataFrame(
        {
            "analysis_scope": ["mimic"] * 6,
            "outcome": ["lactate_rise_12h_flag"] * 6,
            "model": ["absolute_spo2"] * 6,
            "y_true": [0, 0, 0, 1, 1, 1],
            "probability": [0.05, 0.1, 0.4, 0.55, 0.8, 0.95],
        }
    )
    curves = build_oof_performance_curves(predictions)
    assert set(curves["curve"]) == {"roc", "precision_recall", "calibration"}


def test_endpoint_conclusion_is_unavailable_when_endpoint_source_is_unavailable():
    audit = pd.DataFrame(
        [{"dataset": "mimic", "endpoint": "urine_output_decline_12h_flag", "tier": "primary", "status": "unavailable"}]
    )
    result = build_endpoint_conclusions(
        audit,
        model_metrics=pd.DataFrame(),
        risk_tables=pd.DataFrame(),
        sensitivity=pd.DataFrame(),
    )
    assert result.iloc[0]["classification"] == "unavailable"
    assert result.iloc[0]["sensitivity_estimates"] == 0


def test_adequate_endpoint_without_completed_analysis_is_not_called_null():
    audit = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "endpoint": "lactate_rise_12h_flag",
                "tier": "primary",
                "status": "adequate",
            }
        ]
    )
    result = build_endpoint_conclusions(
        audit,
        model_metrics=pd.DataFrame(
            [{"analysis_scope": "mimic", "status": "skipped"}]
        ),
        risk_tables=pd.DataFrame(
            [{"dataset": "mimic", "status": "underpowered"}]
        ),
        sensitivity=pd.DataFrame(),
    )
    assert result.iloc[0]["classification"] == (
        "unresolved_analysis_underpowered_or_failed"
    )


def test_robust_endpoint_conclusion_requires_auc_auprc_and_cluster_rr_support():
    audit = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "endpoint": "lactate_rise_12h_flag",
                "tier": "primary",
                "status": "adequate",
            }
        ]
    )
    model = pd.DataFrame(
        [
            {
                "analysis_scope": "mimic",
                "outcome": "lactate_rise_12h_flag",
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "epv_status": "adequate_ge_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": -0.01,
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "endpoint": "lactate_rise_12h_flag",
                "status": "estimated",
                "risk_ratio": 1.2,
                "risk_ratio_ci_method": "patient_cluster_bootstrap",
                "risk_ratio_ci95_low": 0.99,
                "fisher_p_bh_adjusted": 0.01,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "endpoint": "lactate_rise_12h_flag",
                "status": "estimated",
                "effect": value,
            }
            for value in (1.1, 1.2, 1.3, 1.4)
        ]
    )
    row = build_endpoint_conclusions(
        audit, model, risk, sensitivity
    ).iloc[0]
    assert row["classification"] == "suggestive_positive"
    assert not row["incremental_auprc_ci_excludes_zero"]
    assert not row["epidemiology_cluster_rr_ci_excludes_one"]


def test_robust_endpoint_conclusion_requires_valid_external_bootstrap():
    endpoint = "lactate_rise_12h_flag"
    audit = pd.DataFrame(
        [{"dataset": "mimic", "endpoint": endpoint, "tier": "primary", "status": "adequate"}]
    )
    model = pd.DataFrame(
        [
            {
                "analysis_scope": "mimic",
                "outcome": endpoint,
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "epv_status": "adequate_ge_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": 0.01,
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions_valid": 500,
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "endpoint": endpoint,
                "status": "estimated",
                "risk_ratio": 1.2,
                "risk_ratio_ci_method": "patient_cluster_bootstrap",
                "risk_ratio_ci95_low": 1.05,
                "fisher_p_bh_adjusted": 0.01,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {"dataset": "mimic", "endpoint": endpoint, "status": "estimated", "effect": value}
            for value in (1.1, 1.2, 1.3, 1.4)
        ]
    )
    external = pd.DataFrame(
        [
            {
                "outcome": endpoint,
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "train_dataset": "mimic",
                "test_dataset": "eicu",
                "epv_status": "adequate_ge_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": 0.01,
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions_valid": 399,
            }
        ]
    )
    fragile = build_endpoint_conclusions(
        audit, model, risk, sensitivity, external
    ).iloc[0]
    assert fragile["classification"] == "suggestive_positive"

    external.loc[0, "bootstrap_repetitions_valid"] = 500
    robust = build_endpoint_conclusions(
        audit, model, risk, sensitivity, external
    ).iloc[0]
    assert robust["classification"] == "robust_positive"


def test_endpoint_conclusion_does_not_promote_fragile_predictive_signal():
    endpoint = "lactate_rise_12h_flag"
    audit = pd.DataFrame(
        [{"dataset": "mimic", "endpoint": endpoint, "tier": "primary", "status": "adequate"}]
    )
    model = pd.DataFrame(
        [
            {
                "analysis_scope": "mimic",
                "outcome": endpoint,
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "epv_status": "fragile_lt_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": 0.01,
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions_valid": 500,
            }
        ]
    )
    row = build_endpoint_conclusions(
        audit, model, pd.DataFrame(), pd.DataFrame()
    ).iloc[0]
    assert row["classification"] == "null_or_no_incremental_value"
    assert row["incremental_auroc_ci_excludes_zero"]
    assert row["incremental_auprc_ci_excludes_zero"]
    assert not row["incremental_epv_adequate_for_claim"]


def test_external_transport_evidence_must_match_claimed_training_dataset():
    endpoint = "lactate_rise_12h_flag"
    audit = pd.DataFrame(
        [{"dataset": "eicu", "endpoint": endpoint, "tier": "primary", "status": "adequate"}]
    )
    model = pd.DataFrame(
        [
            {
                "analysis_scope": "eicu",
                "outcome": endpoint,
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "epv_status": "adequate_ge_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": 0.01,
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions_valid": 500,
            }
        ]
    )
    risk = pd.DataFrame(
        [
            {
                "dataset": "eicu",
                "endpoint": endpoint,
                "status": "estimated",
                "risk_ratio": 1.2,
                "risk_ratio_ci_method": "patient_cluster_bootstrap",
                "risk_ratio_ci95_low": 1.05,
                "fisher_p_bh_adjusted": 0.01,
            }
        ]
    )
    sensitivity = pd.DataFrame(
        [
            {"dataset": "eicu", "endpoint": endpoint, "status": "estimated", "effect": value}
            for value in (1.1, 1.2, 1.3, 1.4)
        ]
    )
    external = pd.DataFrame(
        [
            {
                "outcome": endpoint,
                "model": "parsimonious_spo2_instability",
                "status": "fit",
                "train_dataset": "mimic",
                "test_dataset": "eicu",
                "epv_status": "adequate_ge_10",
                "delta_auroc_vs_absolute_ci95_low": 0.01,
                "delta_auprc_vs_absolute_ci95_low": 0.01,
                "bootstrap_repetitions_requested": 500,
                "bootstrap_repetitions_valid": 500,
            }
        ]
    )
    row = build_endpoint_conclusions(
        audit, model, risk, sensitivity, external
    ).iloc[0]
    assert row["classification"] == "suggestive_positive"
    assert not row["external_incremental_auroc_ci_excludes_zero"]
    assert not row["external_incremental_auprc_ci_excludes_zero"]
    assert not row["external_training_epv_adequate_for_claim"]


def test_feature_missingness_is_dataset_specific():
    analysis = pd.DataFrame(
        {
            "dataset": ["mimic", "eicu"],
            "stay_id": [1, 2],
            "spo2_rmssd": [1.2, np.nan],
        }
    )
    result = build_feature_missingness_table(analysis)
    selected = result.loc[result["feature"].eq("spo2_rmssd")].set_index("dataset")
    assert selected.loc["mimic", "missing_fraction"] == 0
    assert selected.loc["eicu", "missing_fraction"] == 1
    assert selected.loc["mimic", "reason"] == ""
    assert selected.loc["eicu", "reason"] == "no_observed_values"
