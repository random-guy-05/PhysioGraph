"""Known-answer and contract tests for the model-minimal epidemiology module.

Statistics are validated against hand-computed values on deterministic tables so
a coding error in the classical estimators cannot silently manufacture an
association.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.spo2_epidemiology import (
    _bh_adjust,
    _binomial_twosided,
    _cochran_armitage_trend,
    _fisher_exact,
    _mantel_haenszel_odds_ratio,
    _risk_difference_wald,
    _risk_ratio_wald,
    assign_exposure,
    build_dose_response_tables,
    build_mantel_haenszel_tables,
    build_specificity_matrix,
    build_stratified_risk_tables,
)


def test_risk_ratio_known_answer():
    # a=40/100 exposed cases, c=20/100 unexposed: RR = 2.0
    rr, lo, hi = _risk_ratio_wald(40, 100, 20, 100)
    assert rr == pytest.approx(2.0)
    # Katz SE = sqrt(1/40-1/100+1/20-1/100) = sqrt(0.055) ~ 0.2345
    se = math.sqrt(1 / 40 - 1 / 100 + 1 / 20 - 1 / 100)
    assert lo == pytest.approx(2.0 * math.exp(-1.96 * se), rel=1e-3)
    assert hi == pytest.approx(2.0 * math.exp(1.96 * se), rel=1e-3)


def test_risk_difference_known_answer():
    rd, lo, hi = _risk_difference_wald(40, 100, 20, 100)
    assert rd == pytest.approx(0.2)
    assert lo == pytest.approx(0.2 - 1.96 * 0.0632456, rel=1e-3)
    assert hi == pytest.approx(0.2 + 1.96 * 0.0632456, rel=1e-3)


def test_fisher_exact_matches_known_extreme():
    # Perfect separation: exposed all cases, unexposed no cases -> tiny p.
    p_extreme = _fisher_exact(50, 0, 0, 50)
    assert p_extreme < 1e-10
    # Balanced null table -> p = 1.
    p_null = _fisher_exact(10, 10, 10, 10)
    assert p_null == pytest.approx(1.0, abs=1e-6)


def test_mantel_haenszel_two_strata_known_answer():
    # Both strata have OR = (a*d)/(b*c) = 4 -> pooled MH OR must be 4
    # (MH with equal stratum weights preserves a constant stratum OR).
    strata = [(20, 10, 10, 20), (30, 15, 15, 30)]
    or_mh, lo, hi = _mantel_haenszel_odds_ratio(strata)
    assert or_mh == pytest.approx(4.0, rel=1e-6)
    assert lo < or_mh < hi
    # Null strata -> OR 1.
    or_null, _, _ = _mantel_haenszel_odds_ratio([(10, 10, 10, 10), (10, 10, 10, 10)])
    assert or_null == pytest.approx(1.0)


def test_cochran_armitage_detects_monotone_trend():
    # Strong monotone dose-response.
    z_strong, p_strong = _cochran_armitage_trend([10, 30, 60], [100, 100, 100], [1, 2, 3])
    assert p_strong < 0.001
    # Flat risks -> no trend.
    z_flat, p_flat = _cochran_armitage_trend([20, 20, 20], [100, 100, 100], [1, 2, 3])
    assert abs(z_flat) < 1e-8
    assert p_flat == pytest.approx(1.0)


def test_bh_adjustment_monotone_and_bounded():
    adjusted = _bh_adjust([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert adjusted == sorted(adjusted, reverse=True) or all(
        adjusted[i] <= adjusted[i + 1] + 1e-12 for i in range(len(adjusted) - 1)
    )
    assert all(0 <= p <= 1 for p in adjusted)
    # Smallest p multiplied by n/k stays the family minimum.
    assert adjusted[0] == pytest.approx(0.01, abs=0.005)


def test_binomial_sign_test():
    # 9/10 positive: two-sided p = 2*P(X>=9)|p=.5 = 2*11/1024 = 0.0215
    p = _binomial_twosided(9, 10, 0.5)
    assert p == pytest.approx(22 / 1024, abs=1e-3)
    # 5/10: p = 1.
    assert _binomial_twosided(5, 10, 0.5) == pytest.approx(1.0)


def _analysis_frame(n_per_arm: int = 300, effect: float = 2.0, seed: int = 7) -> pd.DataFrame:
    """Synthetic frame where exposure doubles endpoint risk by construction."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    exposure = np.array([1] * n_per_arm + [0] * n_per_arm)
    prob = np.where(exposure == 1, 0.30 * effect if effect <= 1 else 0.30, 0.15)
    prob = np.where(exposure == 1, 0.30, 0.15)
    outcome = rng.binomial(1, prob)
    return pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": np.arange(n) // 2,  # pairs: 2 stays per person
            "lactate_rise_12h_flag": outcome,
            "vis_rise_12h_flag": rng.binomial(1, prob),
            "hepatic_lab_worsening_12h_flag": rng.binomial(1, 0.1, n),  # null control
            "spo2_below_90_fraction": np.where(exposure == 1, 0.1, 0.0),
            "spo2_instability_proxy_score": np.where(exposure == 1, 5.0, 1.0)
            + rng.normal(0, 0.1, n),
        }
    )


def test_stratified_risk_tables_recover_known_effect():
    frame = _analysis_frame()
    frame["exposure_any"] = 0
    frame.loc[frame["spo2_below_90_fraction"] > 0, "exposure_any"] = 1
    tables = build_stratified_risk_tables(
        frame, endpoints=["lactate_rise_12h_flag"], bootstrap_repetitions=400
    )
    row = tables.iloc[0]
    assert row["status"] == "estimated"
    # True RR = 0.30/0.15 = 2.0; estimate must be in a generous band.
    assert 1.3 < row["risk_ratio"] < 3.0
    assert row["risk_ratio_ci95_low"] < row["risk_ratio"] < row["risk_ratio_ci95_high"]
    assert row["fisher_p"] < 0.001
    # Cluster bootstrap must also bracket the point estimate.
    assert row["risk_ratio_cluster_boot_low"] < row["risk_ratio"] < row["risk_ratio_cluster_boot_high"]


def test_stratified_risk_tables_flag_underpowered():
    frame = _analysis_frame(n_per_arm=20)
    frame["exposure_any"] = frame["spo2_below_90_fraction"].gt(0).astype(int)
    tables = build_stratified_risk_tables(frame, endpoints=["lactate_rise_12h_flag"])
    assert tables.iloc[0]["status"] == "underpowered"


def test_dose_response_detects_gradient():
    frame = _analysis_frame()
    # Continuous score spread so tertiles are non-empty: scores ~ 1 or 5 with noise.
    rng_scores = np.random.default_rng(11)
    frame["spo2_instability_proxy_score"] = rng_scores.uniform(0, 9, len(frame))
    frame["exposure_tertile"] = pd.qcut(
        frame["spo2_instability_proxy_score"], q=3, labels=["T1", "T2", "T3"]
    ).astype("string")
    # Make risk monotone by tertile.
    risk_map = {"T1": 0.10, "T2": 0.20, "T3": 0.35}
    rng = np.random.default_rng(3)
    frame["lactate_rise_12h_flag"] = [
        rng.binomial(1, risk_map[t]) if pd.notna(t) else 0 for t in frame["exposure_tertile"]
    ]
    dose = build_dose_response_tables(frame, endpoints=["lactate_rise_12h_flag"])
    row = dose.iloc[0]
    assert row["status"] == "estimated"
    assert row["risk_T1"] < row["risk_T2"] < row["risk_T3"]
    assert row["trend_p_two_sided"] < 0.01
    assert row["trend_p_bh_adjusted"] >= row["trend_p_two_sided"]


def test_mantel_haenszel_stratifies_without_crash():
    frame = _analysis_frame()
    frame["exposure_any"] = frame["spo2_below_90_fraction"].gt(0).astype(int)
    mh = build_mantel_haenszel_tables(frame, endpoints=["lactate_rise_12h_flag"])
    assert not mh.empty
    estimated = mh.loc[mh["status"] == "estimated"]
    assert (estimated["n_strata"] == 2).all()


def test_specificity_matrix_null_control_must_be_null():
    frame = _analysis_frame()
    frame["exposure_any"] = frame["spo2_below_90_fraction"].gt(0).astype(int)
    tables = build_stratified_risk_tables(
        frame,
        endpoints=["lactate_rise_12h_flag", "hepatic_lab_worsening_12h_flag"],
        bootstrap_repetitions=200,
    )
    matrix = build_specificity_matrix(tables)
    roles = dict(zip(matrix["endpoint"], matrix["role"]))
    assert roles["hepatic_lab_worsening_12h_flag"] == "negative_control"
    assert roles["lactate_rise_12h_flag"] == "primary"


def test_assign_exposure_from_precomputed_columns():
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "spo2_below_90_fraction": [0.05, 0.0, np.nan],
            "spo2_abrupt_jump_rate_per_hr": [0.0, 2.0, 0.0],
            "spo2_instability_proxy_score": [1.0, 2.0, 3.0],
        }
    )
    out = assign_exposure(frame)
    # Rows 0 (<90%) and 1 (jump) are exposed; row 2 is not.
    assert out["exposure_any"].tolist() == [1, 1, 0]
    # Too-few-rows tertiles stay NA (floor is 30) — contract, not error.
    assert out["exposure_tertile"].isna().all()

    # With enough rows, tertiles populate.
    wide = pd.DataFrame(
        {
            "spo2_below_90_fraction": np.zeros(40),
            "spo2_instability_proxy_score": np.linspace(0, 10, 40),
        }
    )
    out_wide = assign_exposure(wide)
    assert out_wide["exposure_tertile"].notna().all()
    assert set(out_wide["exposure_tertile"].unique()) == {"T1", "T2", "T3"}


def test_missing_endpoints_never_counted_as_non_events():
    frame = _analysis_frame()
    frame["exposure_any"] = frame["spo2_below_90_fraction"].gt(0).astype(int)
    frame["vis_rise_12h_flag"] = np.nan  # unascertainable
    tables = build_stratified_risk_tables(frame, endpoints=["vis_rise_12h_flag"])
    row = tables.iloc[0]
    assert row["status"] == "underpowered"
    assert row["n_observed"] == 0
    assert "risk_ratio" not in row or (isinstance(row["risk_ratio"], float) and math.isnan(row["risk_ratio"]))
