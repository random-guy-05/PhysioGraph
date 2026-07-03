"""Tests for the minimal Colab runner support functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physiograph_colab_core import (
    LANDMARK_MINUTES,
    build_availability_audit,
    build_lactate_negative_summary,
    build_respiratory_context_tables,
    compute_horizon_outcomes,
    compute_respiratory_support_features,
    compute_rrt_features,
    compute_spo2_features,
    fit_spo2_models,
    fit_spo2_or_pvalue_tables,
    fit_spo2_variability_or_tables,
    lint_claims_and_outputs,
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
    assert stay1["spo2_abrupt_jump_count"] == 2
    assert stay2["spo2_plausible_count"] == 0
    assert stay2["spo2_implausible_count"] == 1


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
            "spo2_below_90_fraction": rng.uniform(0, 0.2, n),
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


def test_respiratory_context_returns_strata_or_skip():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    n = 250
    df = pd.DataFrame(
        {
            "mcs_or_death_168h_flag": rng.integers(0, 2, n),
            "spo2_below_90_fraction": rng.uniform(0, 0.3, n),
            "spo2_plausible_count": np.ones(n),
            "resp_support_any_flag": rng.integers(0, 2, n),
            "mechanical_ventilation_flag": rng.integers(0, 2, n),
            "fio2_max": rng.uniform(21, 80, n),
            "age": rng.normal(65, 10, n),
        }
    )
    out = build_respiratory_context_tables(df)
    assert not out.empty
    assert out["status"].isin(["descriptive", "fit", "skipped"]).any() or "stratum" in out.columns


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


def test_availability_audit_race_absent_explicit():
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
    analysis = pd.DataFrame({"dataset": ["mimic"], "stay_id": [1]})
    out = build_availability_audit(artifacts, analysis)
    assert out.iloc[0]["race_ethnicity_status"] == "absent_explicit"

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
        manifest={"fresh_colab_execution": False},
        output_paths={},
    )
    assert not clean["check"].eq("apparent_metrics_in_model_output").any()
