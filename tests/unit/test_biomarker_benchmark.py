from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from physiograph.analysis.biomarker_benchmark import (
    _standardized_exposures,
    _weighted_binary_metric_samples,
    assign_kapur_scai_stage,
    build_biomarker_analysis_frame,
    fit_biomarker_associations,
    fit_biomarker_prediction_benchmark,
)
from physiograph.etl.eicu_extractor import _stream_eicu_aperiodic_vital_events


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "sbp": 110,
                "map_value": 75,
                "lactate": 1.2,
                "alt": 40,
                "ph": 7.4,
                "drug_count": 0,
                "device_count": 0,
                "ohca": 0,
            },
            "A",
        ),
        ({"sbp": 80, "lactate": 1.2, "drug_count": 0, "device_count": 0}, "B"),
        ({"sbp": 110, "lactate": 3, "drug_count": 0, "device_count": 0}, "B"),
        ({"sbp": 80, "lactate": 3, "drug_count": 0, "device_count": 0}, "C"),
        ({"sbp": 110, "lactate": 1.2, "drug_count": 1, "device_count": 0}, "C"),
        ({"sbp": 80, "lactate": 5, "drug_count": 0, "device_count": 0}, "D"),
        ({"sbp": 110, "lactate": 1.2, "drug_count": 2, "device_count": 0}, "D"),
        (
            {
                "sbp": 110,
                "lactate": 3,
                "drug_count": 1,
                "device_count": 0,
                "persistent_abnormality": True,
            },
            "D",
        ),
        ({"sbp": 59, "lactate": 1.2, "drug_count": 0, "device_count": 0}, "E"),
        ({"sbp": 110, "lactate": 10.01, "drug_count": 0, "device_count": 0}, "E"),
        ({"sbp": 110, "lactate": 1.2, "ph": 7.2, "drug_count": 0, "device_count": 0}, "E"),
        ({"sbp": 110, "lactate": 1.2, "drug_count": 4, "device_count": 0}, "E"),
        ({"sbp": 110, "lactate": 1.2, "drug_count": 0, "device_count": 3}, "E"),
        ({"sbp": 110, "lactate": 1.2, "drug_count": 0, "device_count": 0, "ohca": 1}, "E"),
    ],
)
def test_kapur_2022_stage_boundaries(kwargs: dict[str, float], expected: str) -> None:
    stage, number, _, _ = assign_kapur_scai_stage(**kwargs)
    assert stage == expected
    assert number == {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}[expected]


def test_kapur_stage_does_not_turn_missing_normal_data_into_stage_a() -> None:
    stage, number, evidence, lower_bound = assign_kapur_scai_stage(
        sbp=110,
        lactate=1.2,
        drug_count=0,
        device_count=0,
    )
    assert stage is None
    assert math.isnan(number)
    assert evidence == "insufficient_components_for_stage_A"
    assert lower_bound


def test_four_hour_features_exclude_landmark_and_count_active_therapy() -> None:
    analysis = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "person_id": ["p1", "p2"],
            "pressor_source_available": [1, 1],
            "mcs_source_available": [1, 1],
            "baseline_vasoactive_flag": [1, 0],
            "baseline_mcs_flag": [0, 0],
            "spo2_min": [90.0, 96.0],
            "spo2_dynamics_proxy_score": [2.0, 0.2],
        }
    )
    rows = [
        (1, "sbp", 0, 100.0, ""),
        (1, "sbp", 100, 80.0, ""),
        (1, "sbp", 240, 40.0, "excluded_at_landmark"),
        (1, "map", 0, 70.0, ""),
        (1, "map", 100, 60.0, ""),
        (1, "lactate", 0, 2.0, ""),
        (1, "lactate", 120, 6.0, ""),
        (1, "alt", 0, 100.0, ""),
        (1, "alt", 180, 600.0, ""),
        (1, "ph", 0, 7.35, ""),
        (1, "pressor_active", 240, 1.0, "norepinephrine"),
        (2, "sbp", 0, 120.0, ""),
        (2, "map", 0, 80.0, ""),
        (2, "lactate", 0, 1.0, ""),
        (2, "alt", 0, 40.0, ""),
        (2, "ph", 0, 7.4, ""),
    ]
    events = pd.DataFrame(
        [
            {
                "dataset": "mimic",
                "stay_id": stay,
                "concept": concept,
                "offset_minutes": offset,
                "value_numeric": value,
                "raw_name": raw,
                "value_text": raw,
            }
            for stay, concept, offset, value, raw in rows
        ]
    )
    result = build_biomarker_analysis_frame(events, analysis).set_index("stay_id")
    assert result.loc[1, "sbp_min_4h"] == 80.0
    assert result.loc[1, "sbp_mean_4h"] == 90.0
    assert result.loc[1, "sbp_below_90_fraction_4h"] == 0.5
    assert result.loc[1, "sbp_rmssd_4h"] == 20.0
    assert result.loc[1, "lactate_delta_4h"] == 4.0
    assert result.loc[1, "kapur_active_drug_count_4h"] == 1
    assert result.loc[1, "kapur_scai_stage_first4h"] == "D"
    assert pd.isna(result.loc[2, "kapur_scai_stage_first4h"])


def test_aperiodic_eicu_cuff_pressure_is_extracted(tmp_path) -> None:
    pd.DataFrame(
        {
            "patientunitstayid": [1, 1, 2],
            "observationoffset": [30, 240, 20],
            "noninvasivesystolic": [88, 50, 120],
            "noninvasivediastolic": [55, 30, 70],
            "noninvasivemean": [62, 40, 80],
        }
    ).to_csv(tmp_path / "vitalAperiodic.csv", index=False)
    events = _stream_eicu_aperiodic_vital_events(
        tmp_path,
        [1],
        chunk_size=10,
        max_chunks=None,
    )
    assert set(events["concept"]) == {"sbp", "dbp", "map"}
    assert set(events["source_table"]) == {"vitalAperiodic.csv"}
    assert events["offset_minutes"].max() == 30
    assert sorted(events["value_numeric"].tolist()) == [55, 62, 88]


def test_prediction_comparison_uses_identical_rows_and_folds() -> None:
    rng = np.random.default_rng(4)
    n = 40
    y = np.tile([0, 1], n // 2)
    frame = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": [f"p{i}" for i in range(n)],
            "age": rng.normal(70, 8, n),
            "is_male": rng.integers(0, 2, n),
            "shock_icd_flag": rng.integers(0, 2, n),
            "spo2_dynamics_eligible_flag": 1,
            "lactate_rise_12h_flag": y,
            "spo2_mean": rng.normal(95, 2, n),
            "spo2_min": rng.normal(91, 3, n),
            "spo2_below_90_fraction": rng.uniform(0, 0.4, n),
            "spo2_sampling_density_per_hr": rng.uniform(2, 4, n),
            "spo2_missing_bin_count": rng.integers(0, 4, n),
            "spo2_longest_gap_minutes": rng.uniform(15, 45, n),
            "spo2_rmssd": rng.uniform(0, 5, n),
            "spo2_abrupt_jump_fraction": rng.uniform(0, 0.5, n),
            "spo2_drop_3_count": rng.integers(0, 4, n),
            "sbp_first_4h": rng.normal(110, 15, n),
            "sbp_last_4h": rng.normal(105, 15, n),
            "sbp_mean_4h": rng.normal(108, 12, n),
            "sbp_min_4h": rng.normal(90, 12, n),
            "sbp_sd_4h": rng.uniform(2, 15, n),
            "sbp_rmssd_4h": rng.uniform(2, 15, n),
            "sbp_slope_per_hr_4h": rng.normal(0, 5, n),
            "sbp_below_90_fraction_4h": rng.uniform(0, 0.5, n),
            "sbp_count_4h": rng.integers(2, 12, n),
            "lactate_first_4h": rng.uniform(1, 4, n),
            "lactate_last_4h": rng.uniform(1, 5, n),
            "lactate_max_4h": rng.uniform(1, 6, n),
            "lactate_delta_4h": rng.normal(0, 1, n),
            "lactate_slope_per_hr_4h": rng.normal(0, 0.5, n),
            "lactate_ge_2_fraction_4h": rng.uniform(0, 1, n),
            "lactate_count_4h": rng.integers(1, 4, n),
            "kapur_scai_stage_num_first4h": rng.integers(1, 6, n),
            "kapur_scai_core_observed": 1,
            "kapur_scai_full_component_observed": 1,
        }
    )
    performance, comparisons, predictions = fit_biomarker_prediction_benchmark(
        frame,
        n_splits=2,
        min_rows=20,
        min_events=5,
        bootstrap_repetitions=5,
        random_state=10,
    )
    fitted = performance.loc[
        performance["status"].eq("fit")
        & performance["population"].eq("harmonized_hf")
        & performance["outcome"].eq("lactate_rise_12h_flag")
    ]
    assert fitted["n"].nunique() == 1
    assert fitted["cv_random_state"].nunique() == 1
    assert fitted["model"].nunique() == 11
    assert not comparisons.empty
    assert predictions["fold_id"].isin([0, 1]).all()


def test_association_standardization_retains_endpoints_and_raw_exposures() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["mimic", "mimic"],
            "stay_id": [1, 2],
            "person_id": ["p1", "p2"],
            "lactate_rise_12h_flag": [0, 1],
            "spo2_instability_index": [1.0, 3.0],
            "irrelevant_column": [99, 99],
        }
    )

    standardized, audit = _standardized_exposures(frame)

    assert standardized["lactate_rise_12h_flag"].tolist() == [0, 1]
    assert standardized["exposure__spo2_instability"].tolist() == [-1.0, 1.0]
    assert "irrelevant_column" not in standardized
    assert audit["exposure__spo2_instability"]["n_observed"] == 2


def test_vectorized_weighted_metrics_match_sklearn_with_ties() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.array([0, 1, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.4, 0.8, 0.9])
    weights = np.array([[1, 2, 3, 1, 2], [2, 1, 0, 3, 1]], dtype=float)
    observed = _weighted_binary_metric_samples(y, probability, weights)

    for row, sample_weight in enumerate(weights):
        assert observed["auroc"][row] == pytest.approx(
            roc_auc_score(y, probability, sample_weight=sample_weight)
        )
        assert observed["auprc"][row] == pytest.approx(
            average_precision_score(y, probability, sample_weight=sample_weight)
        )
        assert observed["brier"][row] == pytest.approx(
            np.average((probability - y) ** 2, weights=sample_weight)
        )


def test_association_fit_executes_with_declared_controls() -> None:
    rng = np.random.default_rng(12)
    n = 80
    frame = pd.DataFrame(
        {
            "dataset": "mimic",
            "stay_id": np.arange(n),
            "person_id": [f"p{i}" for i in range(n)],
            "age": rng.normal(70, 8, n),
            "is_male": rng.integers(0, 2, n),
            "sex_unknown_flag": 0,
            "shock_icd_flag": rng.integers(0, 2, n),
            "lactate_rise_12h_flag": np.tile([0, 1], n // 2),
            "spo2_instability_index": rng.normal(size=n),
        }
    )

    result = fit_biomarker_associations(frame, min_rows=20, min_events=5)
    estimate = result.loc[
        result["model"].eq("marker_specific_spo2_instability")
        & result["outcome"].eq("lactate_rise_12h_flag")
        & result["population"].eq("harmonized_hf")
    ]

    assert len(estimate) == 1
    assert estimate.iloc[0]["status"] == "estimated"
    assert np.isfinite(estimate.iloc[0]["risk_ratio_per_1sd_worse"])
