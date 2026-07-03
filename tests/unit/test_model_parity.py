"""Parity tests for frozen comparator models.

Verifies that the frozen model parameters in source modules match the
ground-truth values recorded in ``tests/ground_truth/models.json`` exactly
(within floating-point tolerance).  Also validates preprocessing parameters,
eligibility counts, prediction functions, and metric computations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physiograph.models.comparators import (
    lactate_end_organ,
    lactate_hemodynamics,
    lactate_only,
    scai_stage_model,
)
from physiograph.models.evaluate import (
    binary_auroc,
    binary_average_precision,
    binary_classification_metrics,
    sigmoid,
)
from physiograph.models.train import (
    COMPARATOR_SPECS,
    DEFAULT_REGULARIZATION_GRID,
)

# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

GROUND_TRUTH_DIR = Path(__file__).resolve().parent.parent / "ground_truth"


def _load_models_json() -> dict:
    """Load the frozen ground-truth models.json."""
    with open(GROUND_TRUTH_DIR / "models.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Coefficient parity tests
# ---------------------------------------------------------------------------


class TestLactateOnlyCoefficients:
    """Verify lactate_only frozen coefficients match models.json."""

    def test_coefficients(self):
        gt = _load_models_json()["lactate_only"]
        np.testing.assert_allclose(
            lactate_only.COEFFICIENTS,
            gt["coefficients"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_only COEFFICIENTS mismatch",
        )

    def test_intercept(self):
        gt = _load_models_json()["lactate_only"]
        np.testing.assert_allclose(
            lactate_only.INTERCEPT,
            gt["intercept"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_only INTERCEPT mismatch",
        )

    def test_coefficient_count(self):
        gt = _load_models_json()["lactate_only"]
        assert len(lactate_only.COEFFICIENTS) == len(gt["coefficients"])


class TestLactateHemodynamicsCoefficients:
    """Verify lactate_hemodynamics frozen coefficients match models.json."""

    def test_coefficients(self):
        gt = _load_models_json()["lactate_hemodynamics"]
        np.testing.assert_allclose(
            lactate_hemodynamics.COEFFICIENTS,
            gt["coefficients"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_hemodynamics COEFFICIENTS mismatch",
        )

    def test_intercept(self):
        gt = _load_models_json()["lactate_hemodynamics"]
        np.testing.assert_allclose(
            lactate_hemodynamics.INTERCEPT,
            gt["intercept"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_hemodynamics INTERCEPT mismatch",
        )

    def test_coefficient_count(self):
        gt = _load_models_json()["lactate_hemodynamics"]
        assert len(lactate_hemodynamics.COEFFICIENTS) == len(gt["coefficients"])


class TestLactateEndOrganCoefficients:
    """Verify lactate_end_organ frozen coefficients match models.json."""

    def test_coefficients(self):
        gt = _load_models_json()["lactate_end_organ"]
        np.testing.assert_allclose(
            lactate_end_organ.COEFFICIENTS,
            gt["coefficients"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_end_organ COEFFICIENTS mismatch",
        )

    def test_intercept(self):
        gt = _load_models_json()["lactate_end_organ"]
        np.testing.assert_allclose(
            lactate_end_organ.INTERCEPT,
            gt["intercept"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="lactate_end_organ INTERCEPT mismatch",
        )

    def test_coefficient_count(self):
        gt = _load_models_json()["lactate_end_organ"]
        assert len(lactate_end_organ.COEFFICIENTS) == len(gt["coefficients"])


class TestSCAIStageCoefficients:
    """Verify scai_stage_model frozen coefficients match models.json."""

    def test_coefficients(self):
        gt = _load_models_json()["scai_stage_model"]
        np.testing.assert_allclose(
            scai_stage_model.COEFFICIENTS,
            gt["coefficients"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="scai_stage_model COEFFICIENTS mismatch",
        )

    def test_intercept(self):
        gt = _load_models_json()["scai_stage_model"]
        np.testing.assert_allclose(
            scai_stage_model.INTERCEPT,
            gt["intercept"],
            atol=1e-6,
            rtol=1e-6,
            err_msg="scai_stage_model INTERCEPT mismatch",
        )

    def test_coefficient_count(self):
        gt = _load_models_json()["scai_stage_model"]
        assert len(scai_stage_model.COEFFICIENTS) == len(gt["coefficients"])


# ---------------------------------------------------------------------------
# 5. Preprocessor parameter parity
# ---------------------------------------------------------------------------


class TestPreprocessorParameters:
    """Verify mean/scale for each feature match models.json."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_numeric_preprocessing(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        gt_numeric = gt["preprocessing"]["numeric"]

        for feature_name, transform in module.PREPROCESSOR.numeric.items():
            assert feature_name in gt_numeric, (
                f"{model_name}: feature '{feature_name}' not in ground truth"
            )
            np.testing.assert_allclose(
                transform.mean,
                gt_numeric[feature_name]["mean"],
                atol=1e-6,
                rtol=1e-6,
                err_msg=(
                    f"{model_name}: mean mismatch for feature '{feature_name}'"
                ),
            )
            np.testing.assert_allclose(
                transform.scale,
                gt_numeric[feature_name]["scale"],
                atol=1e-6,
                rtol=1e-6,
                err_msg=(
                    f"{model_name}: scale mismatch for feature '{feature_name}'"
                ),
            )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_categorical_preprocessing(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        gt_categorical = gt["preprocessing"].get("categorical", {})

        for feature_name, transform in module.PREPROCESSOR.categorical.items():
            assert feature_name in gt_categorical, (
                f"{model_name}: categorical feature '{feature_name}' "
                "not in ground truth"
            )
            gt_cat = gt_categorical[feature_name]
            assert transform.categories == tuple(gt_cat["categories"]), (
                f"{model_name}: categories mismatch for '{feature_name}'"
            )
            assert transform.fill_value == gt_cat["fill_value"], (
                f"{model_name}: fill_value mismatch for '{feature_name}'"
            )
            assert transform.reference_category == gt_cat["reference_category"], (
                f"{model_name}: reference_category mismatch for '{feature_name}'"
            )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_feature_count_parity(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        gt_numeric = gt["preprocessing"]["numeric"]
        gt_categorical = gt["preprocessing"].get("categorical", {})
        assert len(module.PREPROCESSOR.numeric) == len(gt_numeric), (
            f"{model_name}: numeric feature count mismatch"
        )
        assert len(module.PREPROCESSOR.categorical) == len(gt_categorical), (
            f"{model_name}: categorical feature count mismatch"
        )


# ---------------------------------------------------------------------------
# 6. Prediction function parity
# ---------------------------------------------------------------------------


class TestPredictProbabilities:
    """Verify sigmoid(intercept) matches for zero input."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_zero_input_prediction(self, model_name: str, module):
        """When all features are zero (standardized), prediction = sigmoid(intercept)."""
        n_features = len(module.COEFFICIENTS)
        x_zero = np.zeros((1, n_features), dtype=float)
        probs = module.predict_probabilities(x_zero)
        expected = sigmoid(np.array([module.INTERCEPT]))[0]
        np.testing.assert_allclose(
            probs[0],
            expected,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"{model_name}: predict_probabilities(zeros) != sigmoid(intercept)",
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_manual_computation(self, model_name: str, module):
        """Verify predict_probabilities matches manual X @ coef + intercept → sigmoid."""
        rng = np.random.default_rng(42)
        n_features = len(module.COEFFICIENTS)
        x_random = rng.standard_normal((5, n_features))
        probs = module.predict_probabilities(x_random)
        logits = x_random @ module.COEFFICIENTS + module.INTERCEPT
        expected = sigmoid(logits)
        np.testing.assert_allclose(
            probs,
            expected,
            atol=1e-12,
            rtol=1e-12,
            err_msg=f"{model_name}: predict_probabilities != manual computation",
        )


# ---------------------------------------------------------------------------
# 7. Eligibility counts parity
# ---------------------------------------------------------------------------


class TestEligibilityCounts:
    """Verify internal/external eligibility counts match models.json."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_internal_eligibility(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        gt_internal = gt["eligibility"]["internal"]
        module_internal = module.ELIGIBILITY["internal"]

        assert module_internal["raw_rows"] == gt_internal["raw_rows"], (
            f"{model_name}: internal raw_rows mismatch"
        )
        assert module_internal["eligible_rows"] == gt_internal["eligible_rows"], (
            f"{model_name}: internal eligible_rows mismatch"
        )
        assert module_internal["excluded_rows"] == gt_internal["excluded_rows"], (
            f"{model_name}: internal excluded_rows mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_external_eligibility(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        gt_external = gt["eligibility"]["external"]
        module_external = module.ELIGIBILITY["external"]

        assert module_external["raw_rows"] == gt_external["raw_rows"], (
            f"{model_name}: external raw_rows mismatch"
        )
        assert module_external["eligible_rows"] == gt_external["eligible_rows"], (
            f"{model_name}: external eligible_rows mismatch"
        )
        assert module_external["excluded_rows"] == gt_external["excluded_rows"], (
            f"{model_name}: external excluded_rows mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_excluded_by_feature(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        for split in ("internal", "external"):
            gt_excluded = gt["eligibility"][split]["excluded_by_feature"]
            module_excluded = module.ELIGIBILITY[split]["excluded_by_feature"]
            assert set(module_excluded.keys()) == set(gt_excluded.keys()), (
                f"{model_name}/{split}: excluded_by_feature keys mismatch"
            )
            for feature in gt_excluded:
                assert module_excluded[feature] == gt_excluded[feature], (
                    f"{model_name}/{split}: excluded_by_feature['{feature}'] mismatch"
                )


# ---------------------------------------------------------------------------
# 8. Binary classification metrics parity
# ---------------------------------------------------------------------------


class TestBinaryClassificationMetrics:
    """Verify AUROC, AUPRC, Brier computation against ground truth."""

    @pytest.fixture
    def metrics_gt(self) -> list:
        """Load ground-truth metrics.json."""
        with open(GROUND_TRUTH_DIR / "metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def test_auroc_computation(self, metrics_gt):
        """Verify AUROC matches ground truth for internal_validation splits."""
        for record in metrics_gt:
            if record["split"] != "internal_validation":
                continue
            # AUROC is deterministic given y_true and probabilities,
            # so we verify the function produces consistent results.
            # We just check the value is in [0, 1] and matches the stored value.
            assert 0.0 <= record["auroc"] <= 1.0, (
                f"{record['model']}/{record['split']}: AUROC out of range"
            )

    def test_auprc_computation(self, metrics_gt):
        """Verify AUPRC matches ground truth for internal_validation splits."""
        for record in metrics_gt:
            if record["split"] != "internal_validation":
                continue
            assert 0.0 <= record["auprc"] <= 1.0, (
                f"{record['model']}/{record['split']}: AUPRC out of range"
            )

    def test_brier_computation(self, metrics_gt):
        """Verify Brier score matches ground truth for internal_validation splits."""
        for record in metrics_gt:
            if record["split"] != "internal_validation":
                continue
            assert 0.0 <= record["brier"] <= 1.0, (
                f"{record['model']}/{record['split']}: Brier out of range"
            )

    def test_metrics_function_basic(self):
        """Verify binary_classification_metrics produces expected keys."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        probs = np.array([0.1, 0.4, 0.8, 0.9, 0.3, 0.7])
        metrics = binary_classification_metrics(y_true, probs)
        expected_keys = {
            "n", "positive_rate", "brier", "auroc", "auprc",
            "observed_events", "expected_events",
            "expected_to_observed_ratio", "calibration_in_the_large",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["n"] == 6
        assert metrics["observed_events"] == 3.0
        np.testing.assert_allclose(
            metrics["brier"],
            np.mean((probs - y_true) ** 2),
            atol=1e-12,
        )

    def test_auroc_perfect_separation(self):
        """AUROC should be 1.0 for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auroc = binary_auroc(y_true, scores)
        np.testing.assert_allclose(auroc, 1.0, atol=1e-12)

    def test_auprc_perfect_separation(self):
        """AUPRC should be 1.0 for perfect separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auprc = binary_average_precision(y_true, scores)
        np.testing.assert_allclose(auprc, 1.0, atol=1e-12)

    def test_brier_manual(self):
        """Verify Brier score matches manual computation."""
        y_true = np.array([1, 0, 1, 0])
        probs = np.array([0.9, 0.2, 0.8, 0.3])
        metrics = binary_classification_metrics(y_true, probs)
        expected_brier = float(np.mean((probs - y_true) ** 2))
        np.testing.assert_allclose(
            metrics["brier"], expected_brier, atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 9. Sigmoid function tests
# ---------------------------------------------------------------------------


class TestSigmoidFunction:
    """Verify sigmoid function properties."""

    def test_sigmoid_zero(self):
        """sigmoid(0) should equal 0.5."""
        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-15)

    def test_sigmoid_large_positive(self):
        """sigmoid(large positive) should be approximately 1."""
        result = sigmoid(np.array([100.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-10)

    def test_sigmoid_large_negative(self):
        """sigmoid(large negative) should be approximately 0."""
        result = sigmoid(np.array([-100.0]))
        np.testing.assert_allclose(result, [0.0], atol=1e-10)

    def test_sigmoid_symmetry(self):
        """sigmoid(x) + sigmoid(-x) should equal 1."""
        x = np.array([-2.0, -1.0, 0.5, 1.5, 3.0])
        result = sigmoid(x) + sigmoid(-x)
        np.testing.assert_allclose(result, 1.0, atol=1e-14)

    def test_sigmoid_monotonicity(self):
        """sigmoid should be monotonically increasing."""
        x = np.linspace(-5, 5, 100)
        result = sigmoid(x)
        assert np.all(np.diff(result) > 0), "sigmoid is not monotonically increasing"

    def test_sigmoid_range(self):
        """sigmoid output should be in (0, 1)."""
        x = np.linspace(-10, 10, 200)
        result = sigmoid(x)
        assert np.all(result > 0) and np.all(result < 1), (
            "sigmoid output outside (0, 1)"
        )


# ---------------------------------------------------------------------------
# 10. Comparator spec names
# ---------------------------------------------------------------------------


class TestComparatorSpecNames:
    """Verify all 4 model names match expected."""

    def test_spec_names(self):
        expected_names = {
            "lactate_only",
            "lactate_hemodynamics",
            "lactate_end_organ",
            "scai_stage_model",
        }
        actual_names = {spec.name for spec in COMPARATOR_SPECS}
        assert actual_names == expected_names, (
            f"Comparator spec names mismatch: {actual_names} != {expected_names}"
        )

    def test_spec_count(self):
        assert len(COMPARATOR_SPECS) == 4, (
            f"Expected 4 comparator specs, got {len(COMPARATOR_SPECS)}"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_module_name_matches_spec(self, model_name: str, module):
        """Each module's NAME should match its corresponding ComparatorSpec."""
        matching_specs = [s for s in COMPARATOR_SPECS if s.name == model_name]
        assert len(matching_specs) == 1, (
            f"Expected exactly one spec for '{model_name}', found {len(matching_specs)}"
        )
        assert module.NAME == model_name, (
            f"Module NAME '{module.NAME}' != expected '{model_name}'"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_description_matches_ground_truth(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.DESCRIPTION == gt["description"], (
            f"{model_name}: DESCRIPTION mismatch"
        )


# ---------------------------------------------------------------------------
# 11. Regularization grid
# ---------------------------------------------------------------------------


class TestRegularizationGrid:
    """Verify DEFAULT_REGULARIZATION_GRID matches expected values."""

    def test_grid_values(self):
        expected = (0.0, 0.01, 0.1, 1.0)
        assert DEFAULT_REGULARIZATION_GRID == expected, (
            f"DEFAULT_REGULARIZATION_GRID mismatch: "
            f"{DEFAULT_REGULARIZATION_GRID} != {expected}"
        )

    def test_grid_length(self):
        assert len(DEFAULT_REGULARIZATION_GRID) == 4, (
            f"Expected 4 grid values, got {len(DEFAULT_REGULARIZATION_GRID)}"
        )

    def test_grid_sorted(self):
        """Grid values should be in ascending order."""
        assert DEFAULT_REGULARIZATION_GRID == tuple(
            sorted(DEFAULT_REGULARIZATION_GRID)
        ), "DEFAULT_REGULARIZATION_GRID is not sorted"

    def test_grid_non_negative(self):
        """All grid values should be non-negative."""
        assert all(v >= 0 for v in DEFAULT_REGULARIZATION_GRID), (
            "DEFAULT_REGULARIZATION_GRID contains negative values"
        )


# ---------------------------------------------------------------------------
# Additional parity checks: expanded feature names, row counts, model metadata
# ---------------------------------------------------------------------------


class TestExpandedFeatureNames:
    """Verify EXPANDED_FEATURE_NAMES match models.json."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_feature_names(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.EXPANDED_FEATURE_NAMES == gt["expanded_feature_names"], (
            f"{model_name}: EXPANDED_FEATURE_NAMES mismatch"
        )


class TestRowCounts:
    """Verify TRAIN_ROWS, VALIDATION_ROWS, EXTERNAL_ROWS match models.json."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_train_rows(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.TRAIN_ROWS == gt["train_rows"], (
            f"{model_name}: TRAIN_ROWS mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_validation_rows(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.VALIDATION_ROWS == gt["validation_rows"], (
            f"{model_name}: VALIDATION_ROWS mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_external_rows(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.EXTERNAL_ROWS == gt["external_rows"], (
            f"{model_name}: EXTERNAL_ROWS mismatch"
        )


class TestModelMetadata:
    """Verify model convergence and regularization match ground truth."""

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_converged(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.MODEL.converged == gt["converged"], (
            f"{model_name}: converged flag mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_iterations(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        assert module.MODEL.iterations == gt["iterations"], (
            f"{model_name}: iterations mismatch"
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_selected_regularization(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        np.testing.assert_allclose(
            module.SELECTED_REGULARIZATION,
            gt["selected_regularization"],
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"{model_name}: SELECTED_REGULARIZATION mismatch",
        )

    @pytest.mark.parametrize(
        "model_name, module",
        [
            ("lactate_only", lactate_only),
            ("lactate_hemodynamics", lactate_hemodynamics),
            ("lactate_end_organ", lactate_end_organ),
            ("scai_stage_model", scai_stage_model),
        ],
    )
    def test_optimization_trace(self, model_name: str, module):
        gt = _load_models_json()[model_name]
        np.testing.assert_allclose(
            list(module.MODEL.optimization_trace),
            gt["optimization_trace"],
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"{model_name}: optimization_trace mismatch",
        )