from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import physiograph.analysis.spo2_advanced_inference as advanced


def _preanchor_fixture(n: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260902)
    age = rng.normal(64.0, 12.0, n)
    pre_spo2 = rng.normal(94.0, 2.0, n)
    propensity = 1.0 / (1.0 + np.exp(-(-0.8 + 0.035 * (age - 64.0))))
    exposure = rng.binomial(1, propensity)
    preanchor = pd.DataFrame(
        {
            "dataset": "synthetic",
            "stay_id": np.arange(n),
            "person_id": [f"person-{index}" for index in range(n)],
            "episode_exposed": exposure,
            "anchor_offset_minutes": 90.0,
            "anchor_spo2": pre_spo2 - 2.0 * exposure,
            "previous_spo2": pre_spo2,
            "preanchor_sampling_density_per_hr": rng.uniform(2.0, 8.0, n),
            "pre_spo2_last": pre_spo2,
            "pre_spo2_mean": pre_spo2,
            "pre_spo2_min": pre_spo2 - 1.0,
            "pre_spo2_std": rng.uniform(0.2, 2.0, n),
            "pre_hr_last": rng.normal(90.0, 12.0, n),
            "pre_map_last": rng.normal(72.0, 8.0, n),
            "pre_resp_rate_last": rng.normal(20.0, 4.0, n),
            "preanchor_numeric_event_count": rng.integers(4, 20, n),
            "age": age,
            "is_male": rng.binomial(1, 0.6, n),
            "shock_icd_flag": rng.binomial(1, 0.2, n),
            "baseline_vasoactive_flag": rng.binomial(1, 0.2, n),
            "hospital_id": [f"site-{index % 6}" for index in range(n)],
            "icu_type": "cardiac",
            "unit_admit_source": "emergency",
            "analysis_status": advanced.ADVANCED_ANALYSIS_STATUS,
        }
    )
    outcome_probability = 1.0 / (
        1.0 + np.exp(-(-2.2 + 0.9 * exposure + 0.025 * (age - 64.0)))
    )
    event = rng.binomial(1, outcome_probability)
    observation = rng.binomial(1, 0.82 - 0.08 * exposure)
    change = 0.35 * exposure + 0.01 * (age - 64.0) + rng.normal(0.0, 0.45, n)
    records = pd.DataFrame(
        {
            "dataset": "synthetic",
            "stay_id": np.arange(n),
            "person_id": [f"person-{index}" for index in range(n)],
            "episode_exposed": exposure,
            "endpoint": "lactate_rise",
            "endpoint_family": "lactate",
            "lag_window": "1_to_8h",
            "source_available": 1,
            "at_risk": 1,
            "baseline_observed": 1,
            "outcome_observed": observation,
            "baseline_value": rng.normal(1.8, 0.4, n),
            "event": np.where(observation == 1, event, np.nan),
            "change_value": np.where(observation == 1, change, np.nan),
            "change_scale": "absolute_delta_mmol_l",
        }
    )
    return preanchor, records


def test_preanchor_covariates_are_strictly_before_anchor() -> None:
    events = pd.DataFrame(
        [
            ("synthetic", 1, "spo2", 0.0, 96.0),
            ("synthetic", 1, "spo2", 15.0, 95.0),
            ("synthetic", 1, "spo2", 30.0, 89.0),
            ("synthetic", 1, "hr", 20.0, 90.0),
            ("synthetic", 1, "hr", 30.0, 150.0),
            ("synthetic", 1, "hr", 45.0, 170.0),
        ],
        columns=["dataset", "stay_id", "concept", "offset_minutes", "value_numeric"],
    )
    records = pd.DataFrame(
        {
            "dataset": ["synthetic"],
            "stay_id": [1],
            "person_id": ["p1"],
            "episode_exposed": [1],
            "anchor_offset_minutes": [30.0],
            "anchor_spo2": [89.0],
            "preanchor_sampling_density_per_hr": [6.0],
            "age": [60.0],
            "shock_icd_flag": [0],
            "baseline_vasoactive_flag": [0],
        }
    )
    cohort = pd.DataFrame(
        {"dataset": ["synthetic"], "stay_id": [1], "hospital_id": ["A"]}
    )
    row = advanced.build_episode_preanchor_covariates(events, records, cohort).iloc[0]
    assert row["pre_hr_last"] == 90.0
    assert row["pre_spo2_last"] == 95.0
    assert row["previous_spo2"] == 95.0
    assert row["spo2_delta"] == -6.0
    assert row["preanchor_boundary_rule"].endswith("strictly_less_than_anchor")


def test_overlap_weights_improve_observed_covariate_balance() -> None:
    preanchor, _ = _preanchor_fixture()
    outputs = advanced.build_episode_overlap_weights(preanchor)
    balance = outputs["advanced_episode_covariate_balance"].set_index("covariate")
    assert (
        balance.loc["age", "absolute_smd_overlap_weighted"]
        < balance.loc["age", "absolute_smd_unweighted"]
    )
    summary = outputs["advanced_episode_overlap_summary"].iloc[0]
    assert summary["overlap_effective_sample_size"] > 0
    assert summary["max_absolute_smd_overlap_weighted"] < 0.15


def test_crossfit_aipw_recovers_positive_synthetic_signal(monkeypatch) -> None:
    monkeypatch.setattr(advanced, "N_FOLDS", 3)
    preanchor, records = _preanchor_fixture(600)
    row = advanced.build_crossfit_aipw_effects(records, preanchor).iloc[0]
    assert row["status"] == "estimated"
    assert row["aipw_risk_ratio"] > 1.2
    assert 0 < row["tmle_risk_exposed"] < 1
    assert 0 < row["tmle_risk_unexposed"] < 1
    assert row["tmle_risk_ratio"] > 1.2
    assert abs(row["tmle_efficient_influence_mean_exposed"]) < 1e-6
    assert abs(row["tmle_efficient_influence_mean_unexposed"]) < 1e-6
    assert row["n_outcome_observed"] < row["n_risk_set"]
    assert row["observation_probability_min"] < 1.0


def test_overlap_weighted_effect_is_estimable_with_informative_observation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(advanced, "N_FOLDS", 3)
    preanchor, records = _preanchor_fixture(600)
    weights = advanced.build_episode_overlap_weights(preanchor)[
        "advanced_episode_anchor_weights"
    ]
    row = advanced.build_overlap_weighted_effects(
        records, preanchor, weights
    ).iloc[0]
    assert row["status"] == "estimated"
    assert row["overlap_weighted_risk_ratio"] > 1.0
    assert row["effective_sample_size_exposed"] > 20
    assert row["effective_sample_size_unexposed"] > 20


def test_complete_observation_is_not_spuriously_clipped(monkeypatch) -> None:
    monkeypatch.setattr(advanced, "N_FOLDS", 3)
    preanchor, records = _preanchor_fixture(600)
    records["outcome_observed"] = 1
    records["event"] = records["event"].fillna(0)
    row = advanced.build_crossfit_aipw_effects(records, preanchor).iloc[0]
    assert row["status"] == "estimated"
    assert row["observation_probability_min"] == pytest.approx(1.0)
    assert row["observation_probability_max"] == pytest.approx(1.0)
    assert row["observation_probability_clipped_count"] == 0
    assert row["observation_probability_lower_clipped_count"] == 0
    assert row["claim_ready_observation_support"]


def test_weight_support_gate_rejects_severe_observation_positivity_failure() -> None:
    diagnostics = advanced._weight_support_diagnostics(
        np.full(100, 0.5),
        np.r_[np.full(80, 0.01), np.ones(20)],
        np.ones(100),
        np.ones(100),
    )
    assert diagnostics["claim_ready_propensity_support"]
    assert not diagnostics["claim_ready_observation_support"]
    assert not diagnostics["claim_ready_weight_diagnostics"]


def test_continuous_aipw_detects_graded_worsening_without_threshold(
    monkeypatch,
) -> None:
    monkeypatch.setattr(advanced, "N_FOLDS", 3)
    preanchor, records = _preanchor_fixture(600)
    row = advanced.build_crossfit_continuous_aipw_effects(
        records, preanchor
    ).iloc[0]
    assert row["status"] == "estimated"
    assert row["aipw_mean_change_difference"] > 0.2
    assert row["aipw_mean_change_difference_ci95_low"] > 0
    assert row["continuous_change_direction"] == "higher_is_worse"


def test_multicenter_empty_outcome_is_reported_not_raised() -> None:
    focused = pd.DataFrame(
        {
            "dataset": ["synthetic"] * 8,
            "stay_id": np.arange(8),
            "episode_exposed": [0, 1] * 4,
            "endpoint": "mcs_initiation",
            "endpoint_family": "incident_intervention",
            "lag_window": "0_to_4h",
            "source_available": 0,
            "at_risk": 0,
            "outcome_observed": 0,
            "event": np.nan,
        }
    )
    cohort = pd.DataFrame(
        {
            "dataset": ["synthetic"] * 8,
            "stay_id": np.arange(8),
            "hospital_id": ["A"] * 8,
        }
    )
    summary = advanced.build_multicenter_robustness(focused, cohort)[
        "advanced_episode_multicenter_summary"
    ].iloc[0]
    assert summary["status"] == "underpowered_multicenter"
    assert summary["hospitals_estimable"] == 0


def test_multicenter_random_effect_detects_consistent_site_signal() -> None:
    rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    stay = 0
    for site in range(6):
        for exposure in (0, 1):
            for index in range(40):
                stay += 1
                event = index < (16 if exposure else 4)
                rows.append(
                    {
                        "dataset": "synthetic",
                        "stay_id": stay,
                        "episode_exposed": exposure,
                        "endpoint": "pressor_initiation",
                        "endpoint_family": "incident_intervention",
                        "lag_window": "4_to_12h",
                        "source_available": 1,
                        "at_risk": 1,
                        "outcome_observed": 1,
                        "event": int(event),
                    }
                )
                cohort_rows.append(
                    {
                        "dataset": "synthetic",
                        "stay_id": stay,
                        "hospital_id": f"site-{site}",
                    }
                )
    summary = advanced.build_multicenter_robustness(
        pd.DataFrame(rows), pd.DataFrame(cohort_rows)
    )["advanced_episode_multicenter_summary"].iloc[0]
    assert summary["status"] == "estimated"
    assert summary["claim_ready_multicenter"]
    assert summary["random_effect_risk_ratio"] == pytest.approx(4.0)
    assert summary["random_effect_ci95_low_hksj"] > 1.0


def test_e_value_formula_and_null_crossing_limit() -> None:
    assert advanced._e_value(2.0) == pytest.approx(2.0 + np.sqrt(2.0))
    assert advanced._e_value_confidence_limit(2.0, 1.2, 3.0) > 1.0
    assert advanced._e_value_confidence_limit(2.0, 0.9, 3.0) == 1.0


def test_key_synthesis_does_not_hide_multicenter_cross_fdr_signal() -> None:
    keys = ["dataset", "endpoint", "lag_window"]
    base = pd.DataFrame(
        [
            {
                "dataset": "eicu",
                "endpoint": "death",
                "lag_window": "8_to_24h",
                "status": "estimated",
                "evidence_label": "directionally_positive_inconclusive",
            }
        ]
    )
    overlap = pd.DataFrame(
        [{**dict(zip(keys, base.loc[0, keys])), "status": "underpowered"}]
    )
    aipw = pd.DataFrame(
        [{**dict(zip(keys, base.loc[0, keys])), "status": "underpowered"}]
    )
    continuous = pd.DataFrame(
        [{**dict(zip(keys, base.loc[0, keys])), "status": "continuous_outcome_not_defined"}]
    )
    multicenter = pd.DataFrame(
        [
            {
                **dict(zip(keys, base.loc[0, keys])),
                "status": "estimated",
                "claim_ready_multicenter": True,
                "multicenter_cross_endpoint_fdr_positive": True,
                "random_effect_risk_ratio": 2.0,
                "random_effect_ci95_low_hksj": 1.2,
                "random_effect_ci95_high_hksj": 3.1,
            }
        ]
    )
    balance = pd.DataFrame([{"dataset": "eicu"}])
    row = advanced.build_advanced_key_results(
        base, overlap, aipw, continuous, multicenter, balance
    ).iloc[0]
    assert row["advanced_cross_fdr_methods"] == (
        "multicenter_unadjusted_random_effect"
    )
    assert row["advanced_evidence_label"] == (
        "multicenter_unadjusted_cross_endpoint_fdr_signal"
    )
