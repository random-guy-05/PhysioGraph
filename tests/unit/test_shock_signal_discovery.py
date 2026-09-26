"""Focused tests for the finite objective-shock discovery screen."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from physiograph.analysis.shock_signal_discovery import (
    assemble_candidate_frame,
    benjamini_hochberg,
    candidate_adjustment_set,
    classify_discovery,
    final_winner_gate,
    fit_candidate_model,
    pair_domain_events,
    rolling_oliguria_events,
    statistical_winner_gate,
    sustained_hypotension_events,
    trajectory_features,
)


def test_candidate_frame_reuses_level_without_merge_suffixes() -> None:
    model_base = pd.DataFrame({"stay_id": [1, 2], "hr_level": [70.0, 80.0]})
    feature_long = pd.DataFrame(
        {
            "stay_id": [1, 2, 1, 2],
            "feature": ["hr_level", "hr_level", "hr_slope", "hr_slope"],
            "value": [70.0, 80.0, 1.0, 2.0],
        }
    )

    level = assemble_candidate_frame(model_base, feature_long, "hr_level")
    slope = assemble_candidate_frame(model_base, feature_long, "hr_slope")

    assert level.columns.tolist() == ["stay_id", "hr_level"]
    assert level["hr_level"].tolist() == [70.0, 80.0]
    assert slope["hr_slope"].tolist() == [1.0, 2.0]


def test_modified_poisson_handles_complementary_missingness_covariate() -> None:
    rng = np.random.default_rng(7)
    n = 250
    frame = pd.DataFrame({"candidate": rng.normal(size=n), "lactate": rng.normal(size=n), "event": rng.binomial(1, .3, n)})
    frame.loc[:49, "lactate"] = np.nan
    frame["observed"] = frame.lactate.notna().astype(int)
    result, fit = fit_candidate_model(frame, feature="candidate", outcome="event", adjustment_covariates=["lactate", "observed"])
    assert np.isfinite(result["log_rr_se"])
    assert np.linalg.matrix_rank(fit.model.exog) == fit.model.exog.shape[1]


def test_empty_statistical_gate_completes_without_external_access(tmp_path) -> None:
    cells = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/shock_signal_discovery_cells.py"))
    decisions = {}
    namespace = {"pd": pd, "np": np, "OUT": tmp_path, "log": lambda _: None,
                 "write_json": lambda name, value: decisions.update({name: value}),
                 "classify_discovery": classify_discovery, "first_association_read": "synthetic",
                 "screen": pd.DataFrame({"feature": ["hr_level"], "statistical_gate": [False], "q_value": [.9], "effect_magnitude": [1.0], "coverage": [.9]})}
    exec(cells["STABILITY"], namespace)  # noqa: S102 - exercise the actual notebook cell with synthetic data
    assert decisions["decision_gate_summary.json"]["classification"] == "NO_DISCOVERY_SIGNAL"
    assert decisions["decision_gate_summary.json"]["eicu_accessed"] is False


def test_hypotension_waits_for_documentation() -> None:
    hourly = pd.DataFrame({"stay_id": [1, 1], "hour": [6, 7], "sbp": [80, 80], "map": [60, 60], "available_minute": [430, 550]})
    assert sustained_hypotension_events(hourly).event_minute.tolist() == [550.0]


def test_fixed_test_family_keeps_unestimable_tests_in_denominator() -> None:
    q = benjamini_hochberg(pd.Series([.001] + [np.nan] * 35))
    assert np.isclose(q.iloc[0], .036)


def test_trajectory_features_use_only_hours_zero_through_four() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1] * 5,
            "signal": ["hr"] * 5,
            "hour": [0, 1, 2, 4, 5],
            "value": [60, 62, 64, 68, 200],
        }
    )
    result = trajectory_features(hourly).iloc[0]
    assert result["level"] == 68
    assert result["change"] == 8
    assert np.isclose(result["slope"], 2)


def test_dynamic_features_require_three_bins_and_two_hour_span() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 2],
            "signal": ["sbp"] * 5,
            "hour": [0, 4, 1, 1.5, 2],
            "value": [110, 100, 110, 108, 106],
        }
    )
    result = trajectory_features(hourly).set_index("stay_id")
    assert np.isnan(result.loc[1, "slope"])
    assert np.isnan(result.loc[2, "slope"])


def test_variability_is_detrended() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "signal": ["map"] * 3,
            "hour": [0, 2, 4],
            "value": [70, 80, 90],
        }
    )
    assert np.isclose(trajectory_features(hourly).iloc[0]["variability"], 0)


def test_sustained_hypotension_requires_consecutive_low_bins() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2],
            "hour": [6, 7, 8, 6, 8],
            "sbp": [100, 88, 86, 85, 82],
            "map": [70, 64, 62, 60, 58],
        }
    )
    result = sustained_hypotension_events(hourly)
    assert result.loc[result["stay_id"].eq(1), "event_minute"].tolist() == [540.0]
    assert 2 not in set(result["stay_id"])


def test_oliguria_does_not_treat_missing_hours_as_zero() -> None:
    hourly = pd.DataFrame(
        {
            "stay_id": [1, 1, 1, 2, 2, 2, 2],
            "hour": [4, 5, 6, 4, 5, 6, 7],
            "value": [0, 0, 0, 20, 25, 15, 10],
        }
    )
    result = rolling_oliguria_events(hourly, start_hour=4, end_hour=10)
    assert 1 not in set(result["stay_id"])
    assert result.loc[result["stay_id"].eq(2), "event_minute"].tolist() == [600.0]


def test_composite_requires_both_domains_after_landmark() -> None:
    pressure = pd.DataFrame(
        {"stay_id": [1, 2, 3], "event_minute": [420, 420, 300], "component": ["p"] * 3}
    )
    perfusion = pd.DataFrame(
        {"stay_id": [1, 2, 3], "event_minute": [600, 900, 500], "component": ["h"] * 3}
    )
    result = pair_domain_events(pressure, perfusion, lower_minute=360, upper_minute=960)
    assert result[["stay_id", "event_minute"]].values.tolist() == [[1, 600.0]]


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    q = benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, np.nan]))
    assert np.allclose(q.iloc[:3], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3])
    assert np.isnan(q.iloc[3])


def test_adjustment_removes_only_exact_duplicate_level() -> None:
    base = ["age", "hr_level", "sbp_level", "map_level"]
    assert candidate_adjustment_set("hr", base) == ["age", "sbp_level", "map_level"]
    assert candidate_adjustment_set("pulse_pressure", base) == base


def test_statistical_gate_enforces_coverage_even_for_inverse_effect() -> None:
    row = pd.Series(
        {
            "coverage": 0.69,
            "cohort_events": 200,
            "events": 100,
            "q_value": 0.001,
            "ci_low": 0.5,
            "ci_high": 0.8,
            "effect_magnitude": 1.5,
            "lead_q_value": 0.001,
            "lead_ci_low": 0.6,
            "lead_ci_high": 0.9,
            "log_rr": -0.4,
            "lead_log_rr": -0.3,
            "lead_effect_magnitude": 1.35,
        }
    )
    assert not statistical_winner_gate(row)


def test_final_gate_and_classification_prevent_runner_up_rescue() -> None:
    row = pd.Series(
        {
            "coverage": 0.9,
            "cohort_events": 200,
            "events": 100,
            "q_value": 0.001,
            "ci_low": 1.2,
            "ci_high": 1.8,
            "effect_magnitude": 1.5,
            "lead_q_value": 0.002,
            "lead_ci_low": 1.1,
            "lead_ci_high": 1.7,
            "log_rr": 0.4,
            "lead_log_rr": 0.3,
            "lead_effect_magnitude": 1.35,
            "bootstrap_success_fraction": 1.0,
            "bootstrap_sign_consistency": 0.95,
            "bootstrap_ci_low": 1.2,
            "bootstrap_ci_high": 1.7,
            "leave_domain_log_rr": 0.25,
            "leave_domain_ci_low": 1.05,
            "leave_domain_ci_high": 1.6,
            "leave_domain_effect_magnitude": 1.28,
        }
    )
    assert final_winner_gate(row)
    assert classify_discovery(eligible_winners=1, novelty_passed=False) == "KNOWN_SIGNAL_NO_NOVEL_WINNER"
    assert classify_discovery(eligible_winners=0, novelty_passed=None) == "NO_DISCOVERY_SIGNAL"
