"""Tests for the targeted SpO2-variability replication helpers."""

import numpy as np
import pandas as pd

from physiograph.analysis.spo2_variability_validation import (
    fit_fixed_scale_model,
    fixed_exposure_parameters,
)


def test_existing_density_column_is_not_duplicated() -> None:
    base = pd.DataFrame({"stay_id": [1], "spo2_bins": [5]})
    feature = pd.DataFrame({"stay_id": [1], "spo2_variability": [.4]})
    merged = base.merge(feature, on="stay_id", validate="one_to_one")
    assert merged.columns.tolist() == ["stay_id", "spo2_bins", "spo2_variability"]


def test_fixed_parameters_winsorize_before_scaling() -> None:
    values = pd.Series([0.0] * 50 + [1.0] * 50 + [1000.0])
    result = fixed_exposure_parameters(values)
    assert result["upper"] < 1000
    assert result["scale"] > 0


def test_fixed_scale_model_uses_development_scale_and_clustered_se() -> None:
    rng = np.random.default_rng(17); n = 500
    feature = rng.normal(size=n); outcome = rng.binomial(1, 1/(1+np.exp(-(-1.5+.4*feature))))
    frame = pd.DataFrame({"feature": feature, "outcome": outcome, "age": rng.normal(70,10,n), "hospital": np.repeat(np.arange(25),20)})
    parameters = {"lower": -3.0, "upper": 3.0, "center": 0.0, "scale": 2.0}
    result, _ = fit_fixed_scale_model(frame, feature="feature", outcome="outcome", adjustment_covariates=["age"], exposure_parameters=parameters, cluster="hospital")
    assert result["hospitals"] == 25
    assert result["covariance"] == "cluster_hospital"
    assert result["rr_per_mimic_sd"] > 1
