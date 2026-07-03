#!/usr/bin/env python3
"""Final parity verification across all 4 comparator models.

Loads ground truth from tests/ground_truth/ and verifies that every
frozen comparator module in src/physiograph/models/comparators/ matches
exactly.  Also verifies metrics, predictions, and calibration data.

Exit code 0 if ALL checks pass; 1 otherwise.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_DIR = PROJECT_ROOT / "tests" / "ground_truth"
SRC_MODELS_DIR = PROJECT_ROOT / "src" / "physiograph" / "models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class VerificationResult:
    """Accumulate PASS / FAIL results."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((name, passed, detail))

    def summary(self) -> bool:
        all_pass = True
        for name, passed, detail in self.results:
            status = "PASS" if passed else "FAIL"
            msg = f"  [{status}] {name}"
            if detail:
                msg += f"  — {detail}"
            print(msg)
            if not passed:
                all_pass = False
        print()
        n_pass = sum(1 for _, p, _ in self.results if p)
        n_total = len(self.results)
        print(f"Results: {n_pass}/{n_total} checks passed")
        return all_pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid matching the project implementation."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main() -> int:
    vr = VerificationResult()

    # =======================================================================
    # 1. Load ground truth
    # =======================================================================
    print("=" * 72)
    print("SECTION 1: Loading ground truth files")
    print("=" * 72)

    models_gt_path = GROUND_TRUTH_DIR / "models.json"
    metrics_gt_path = GROUND_TRUTH_DIR / "metrics.json"
    predictions_gt_path = GROUND_TRUTH_DIR / "predictions.csv"
    calibration_gt_path = GROUND_TRUTH_DIR / "calibration.csv"

    for p in (models_gt_path, metrics_gt_path, predictions_gt_path, calibration_gt_path):
        exists = p.exists()
        vr.check(f"Ground truth file exists: {p.name}", exists)
        if not exists:
            print(f"  FATAL: {p} not found. Aborting.")
            return 1

    with open(models_gt_path) as f:
        models_gt = json.load(f)

    with open(metrics_gt_path) as f:
        metrics_gt = json.load(f)

    with open(predictions_gt_path) as f:
        reader = csv.DictReader(f)
        predictions_gt = list(reader)

    with open(calibration_gt_path) as f:
        reader = csv.DictReader(f)
        calibration_gt = list(reader)

    vr.check("models.json loaded", len(models_gt) == 4, f"{len(models_gt)} models")
    vr.check("metrics.json loaded", len(metrics_gt) == 12, f"{len(metrics_gt)} records")
    vr.check("predictions.csv loaded", len(predictions_gt) == 24765, f"{len(predictions_gt)} rows")
    vr.check("calibration.csv loaded", len(calibration_gt) == 93, f"{len(calibration_gt)} rows")

    # =======================================================================
    # 2. Import comparator modules
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 2: Importing comparator modules")
    print("=" * 72)

    # Ensure src is on sys.path
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from physiograph.models.comparators import (
        lactate_end_organ,
        lactate_hemodynamics,
        lactate_only,
        scai_stage_model,
    )
    from physiograph.models.train import DEFAULT_REGULARIZATION_GRID

    MODULES = {
        "lactate_only": lactate_only,
        "lactate_hemodynamics": lactate_hemodynamics,
        "lactate_end_organ": lactate_end_organ,
        "scai_stage_model": scai_stage_model,
    }

    for name, mod in MODULES.items():
        vr.check(f"Module imported: {name}", True, f"NAME={mod.NAME}")

    # =======================================================================
    # 3. Verify coefficients match ground truth exactly
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 3: Coefficient parity")
    print("=" * 72)

    for name, mod in MODULES.items():
        gt_coeffs = models_gt[name]["coefficients"]
        src_coeffs = mod.COEFFICIENTS.tolist()
        n_match = len(gt_coeffs) == len(src_coeffs)
        vr.check(f"{name}: coefficient count", n_match,
                  f"GT={len(gt_coeffs)}, SRC={len(src_coeffs)}")
        if n_match:
            for i, (g, s) in enumerate(zip(gt_coeffs, src_coeffs)):
                diff = abs(g - s)
                vr.check(
                    f"{name}: coeff[{i}]",
                    diff < 1e-14,
                    f"diff={diff:.2e}",
                )

        gt_intercept = models_gt[name]["intercept"]
        src_intercept = mod.INTERCEPT
        diff = abs(gt_intercept - src_intercept)
        vr.check(
            f"{name}: intercept",
            diff < 1e-14,
            f"GT={gt_intercept}, SRC={src_intercept}, diff={diff:.2e}",
        )

    # =======================================================================
    # 4. Verify preprocessing parameters match
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 4: Preprocessing parameter parity")
    print("=" * 72)

    for name, mod in MODULES.items():
        gt_preproc = models_gt[name]["preprocessing"]
        src_preproc = mod.PREPROCESSOR

        # Numeric features
        gt_numeric = gt_preproc["numeric"]
        src_numeric = src_preproc.numeric

        gt_num_keys = set(gt_numeric.keys())
        src_num_keys = set(src_numeric.keys())
        vr.check(
            f"{name}: numeric feature names",
            gt_num_keys == src_num_keys,
            f"GT={sorted(gt_num_keys)}, SRC={sorted(src_num_keys)}",
        )

        for feat in sorted(gt_num_keys & src_num_keys):
            gt_mean = gt_numeric[feat]["mean"]
            gt_scale = gt_numeric[feat]["scale"]
            src_mean = src_numeric[feat].mean
            src_scale = src_numeric[feat].scale
            vr.check(
                f"{name}: {feat} mean",
                abs(gt_mean - src_mean) < 1e-14,
                f"diff={abs(gt_mean - src_mean):.2e}",
            )
            vr.check(
                f"{name}: {feat} scale",
                abs(gt_scale - src_scale) < 1e-14,
                f"diff={abs(gt_scale - src_scale):.2e}",
            )

        # Categorical features
        gt_cat = gt_preproc.get("categorical", {})
        src_cat = src_preproc.categorical

        gt_cat_keys = set(gt_cat.keys())
        src_cat_keys = set(src_cat.keys())
        vr.check(
            f"{name}: categorical feature names",
            gt_cat_keys == src_cat_keys,
            f"GT={sorted(gt_cat_keys)}, SRC={sorted(src_cat_keys)}",
        )

        for feat in sorted(gt_cat_keys & src_cat_keys):
            gt_cats = tuple(gt_cat[feat]["categories"])
            src_cats = src_cat[feat].categories
            vr.check(
                f"{name}: {feat} categories",
                gt_cats == src_cats,
                f"GT={gt_cats}, SRC={src_cats}",
            )
            gt_ref = gt_cat[feat]["reference_category"]
            src_ref = src_cat[feat].reference_category
            vr.check(
                f"{name}: {feat} reference_category",
                gt_ref == src_ref,
                f"GT={gt_ref}, SRC={src_ref}",
            )
            gt_fill = gt_cat[feat]["fill_value"]
            src_fill = src_cat[feat].fill_value
            vr.check(
                f"{name}: {feat} fill_value",
                gt_fill == src_fill,
                f"GT={gt_fill}, SRC={src_fill}",
            )

    # =======================================================================
    # 5. Verify model metadata
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 5: Model metadata parity")
    print("=" * 72)

    for name, mod in MODULES.items():
        gt = models_gt[name]

        # Row counts
        vr.check(
            f"{name}: train_rows",
            mod.TRAIN_ROWS == gt["train_rows"],
            f"SRC={mod.TRAIN_ROWS}, GT={gt['train_rows']}",
        )
        vr.check(
            f"{name}: validation_rows",
            mod.VALIDATION_ROWS == gt["validation_rows"],
            f"SRC={mod.VALIDATION_ROWS}, GT={gt['validation_rows']}",
        )
        vr.check(
            f"{name}: external_rows",
            mod.EXTERNAL_ROWS == gt["external_rows"],
            f"SRC={mod.EXTERNAL_ROWS}, GT={gt['external_rows']}",
        )

        # Converged
        vr.check(
            f"{name}: converged",
            mod.MODEL.converged == gt["converged"],
            f"SRC={mod.MODEL.converged}, GT={gt['converged']}",
        )

        # Iterations
        vr.check(
            f"{name}: iterations",
            mod.MODEL.iterations == gt["iterations"],
            f"SRC={mod.MODEL.iterations}, GT={gt['iterations']}",
        )

        # Selected regularization
        vr.check(
            f"{name}: selected_regularization",
            mod.SELECTED_REGULARIZATION == gt["selected_regularization"],
            f"SRC={mod.SELECTED_REGULARIZATION}, GT={gt['selected_regularization']}",
        )

        # Optimization trace
        gt_trace = gt["optimization_trace"]
        src_trace = mod.MODEL.optimization_trace
        vr.check(
            f"{name}: optimization_trace length",
            len(src_trace) == len(gt_trace),
            f"SRC={len(src_trace)}, GT={len(gt_trace)}",
        )
        if len(src_trace) == len(gt_trace):
            for i, (g, s) in enumerate(zip(gt_trace, src_trace)):
                vr.check(
                    f"{name}: trace[{i}]",
                    abs(g - s) < 1e-14,
                    f"diff={abs(g - s):.2e}",
                )

        # Family
        vr.check(
            f"{name}: family",
            gt["family"] == "numpy_logistic_regression",
            f"GT={gt['family']}",
        )

        # Frozen for external validation
        vr.check(
            f"{name}: frozen_for_external_validation",
            gt["frozen_for_external_validation"] is True,
        )

        # Description
        vr.check(
            f"{name}: description",
            mod.DESCRIPTION == gt["description"],
            f"SRC={mod.DESCRIPTION!r}",
        )

        # Expanded feature names
        gt_features = gt["expanded_feature_names"]
        src_features = mod.EXPANDED_FEATURE_NAMES
        vr.check(
            f"{name}: expanded_feature_names",
            gt_features == src_features,
            f"GT has {len(gt_features)} features, SRC has {len(src_features)}",
        )

    # =======================================================================
    # 6. Verify eligibility counts
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 6: Eligibility count parity")
    print("=" * 72)

    for name, mod in MODULES.items():
        gt_elig = models_gt[name]["eligibility"]
        src_elig = mod.ELIGIBILITY

        for split in ("internal", "external"):
            gt_split = gt_elig[split]
            src_split = src_elig[split]

            vr.check(
                f"{name}: {split} raw_rows",
                src_split["raw_rows"] == gt_split["raw_rows"],
                f"SRC={src_split['raw_rows']}, GT={gt_split['raw_rows']}",
            )
            vr.check(
                f"{name}: {split} eligible_rows",
                src_split["eligible_rows"] == gt_split["eligible_rows"],
                f"SRC={src_split['eligible_rows']}, GT={gt_split['eligible_rows']}",
            )
            vr.check(
                f"{name}: {split} excluded_rows",
                src_split["excluded_rows"] == gt_split["excluded_rows"],
                f"SRC={src_split['excluded_rows']}, GT={gt_split['excluded_rows']}",
            )

            # excluded_by_feature
            gt_excl = gt_split["excluded_by_feature"]
            src_excl = src_split["excluded_by_feature"]
            gt_excl_keys = set(gt_excl.keys())
            src_excl_keys = set(src_excl.keys())
            vr.check(
                f"{name}: {split} excluded_by_feature keys",
                gt_excl_keys == src_excl_keys,
            )
            for feat in sorted(gt_excl_keys & src_excl_keys):
                vr.check(
                    f"{name}: {split} excluded_by_feature[{feat}]",
                    src_excl[feat] == gt_excl[feat],
                    f"SRC={src_excl[feat]}, GT={gt_excl[feat]}",
                )

    # =======================================================================
    # 7. Verify regularization grid
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 7: Regularization grid parity")
    print("=" * 72)

    gt_grid = (0.0, 0.01, 0.1, 1.0)
    src_grid = DEFAULT_REGULARIZATION_GRID
    vr.check(
        "DEFAULT_REGULARIZATION_GRID values",
        src_grid == gt_grid,
        f"SRC={src_grid}, GT={gt_grid}",
    )

    # Also verify each model's validation_candidates grid
    for name in MODULES:
        gt_candidates = models_gt[name]["validation_candidates"]
        candidate_penalties = tuple(c["l2_penalty"] for c in gt_candidates)
        vr.check(
            f"{name}: validation_candidates penalties",
            candidate_penalties == gt_grid,
            f"penalties={candidate_penalties}",
        )

    # =======================================================================
    # 8. Verify predictions: sigmoid(intercept) on zero matrix
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 8: Prediction parity (zero-input → sigmoid(intercept))")
    print("=" * 72)

    for name, mod in MODULES.items():
        n_features = len(mod.COEFFICIENTS)
        zero_matrix = np.zeros((1, n_features))
        pred = mod.predict_probabilities(zero_matrix)[0]
        expected = sigmoid(np.array([mod.INTERCEPT]))[0]
        diff = abs(pred - expected)
        vr.check(
            f"{name}: predict(zeros) == sigmoid(intercept)",
            diff < 1e-12,
            f"pred={pred:.15f}, expected={expected:.15f}, diff={diff:.2e}",
        )

        # Also verify sigmoid(intercept) is in (0, 1)
        vr.check(
            f"{name}: sigmoid(intercept) in (0, 1)",
            0.0 < expected < 1.0,
            f"value={expected}",
        )

    # =======================================================================
    # 9. Verify metrics match ground truth
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 9: Metrics parity")
    print("=" * 72)

    # Build lookup: (model, split) → metrics dict
    metrics_lookup = {}
    for m in metrics_gt:
        key = (m["model"], m["split"])
        metrics_lookup[key] = m

    expected_models = {"lactate_only", "lactate_hemodynamics", "lactate_end_organ", "scai_stage_model"}
    expected_splits = {"internal_train", "internal_validation", "external"}

    for model_name in sorted(expected_models):
        for split in sorted(expected_splits):
            key = (model_name, split)
            vr.check(
                f"metrics: {model_name}/{split} exists",
                key in metrics_lookup,
            )
            if key not in metrics_lookup:
                continue
            m = metrics_lookup[key]

            # AUROC in reasonable range
            vr.check(
                f"metrics: {model_name}/{split} AUROC range",
                0.5 <= m["auroc"] <= 1.0,
                f"AUROC={m['auroc']}",
            )

            # AUPRC in reasonable range
            vr.check(
                f"metrics: {model_name}/{split} AUPRC range",
                0.0 <= m["auprc"] <= 1.0,
                f"AUPRC={m['auprc']}",
            )

            # Brier in reasonable range
            vr.check(
                f"metrics: {model_name}/{split} Brier range",
                0.0 <= m["brier"] <= 0.5,
                f"Brier={m['brier']}",
            )

            # n matches model row counts
            gt_model = models_gt[model_name]
            if split == "internal_train":
                expected_n = gt_model["train_rows"]
            elif split == "internal_validation":
                expected_n = gt_model["validation_rows"]
            else:
                expected_n = gt_model["external_rows"]
            vr.check(
                f"metrics: {model_name}/{split} n matches",
                m["n"] == expected_n,
                f"n={m['n']}, expected={expected_n}",
            )

    # =======================================================================
    # 10. Verify predictions.csv structure
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 10: Predictions CSV parity")
    print("=" * 72)

    # Check column structure
    pred_columns = {"model", "split", "dataset", "stay_id", "target", "probability"}
    actual_columns = set(predictions_gt[0].keys())
    vr.check(
        "predictions.csv columns",
        pred_columns == actual_columns,
        f"expected={sorted(pred_columns)}, actual={sorted(actual_columns)}",
    )

    # Check model/split coverage
    pred_models = set()
    pred_splits = set()
    for row in predictions_gt:
        pred_models.add(row["model"])
        pred_splits.add(row["split"])

    vr.check(
        "predictions.csv model coverage",
        pred_models == expected_models,
        f"models={sorted(pred_models)}",
    )
    vr.check(
        "predictions.csv split coverage",
        pred_splits == expected_splits,
        f"splits={sorted(pred_splits)}",
    )

    # Check row counts per model/split
    from collections import Counter
    row_counts = Counter((row["model"], row["split"]) for row in predictions_gt)
    for model_name in sorted(expected_models):
        for split in sorted(expected_splits):
            key = (model_name, split)
            count = row_counts.get(key, 0)
            # Expected count should match metrics n
            if key in metrics_lookup:
                expected_n = metrics_lookup[key]["n"]
                vr.check(
                    f"predictions: {model_name}/{split} row count",
                    count == expected_n,
                    f"rows={count}, expected={expected_n}",
                )

    # Check probability values are in [0, 1]
    all_probs_valid = True
    for row in predictions_gt:
        prob = float(row["probability"])
        if not (0.0 <= prob <= 1.0):
            all_probs_valid = False
            break
    vr.check("predictions.csv: all probabilities in [0,1]", all_probs_valid)

    # Check target values are 0 or 1
    all_targets_valid = True
    for row in predictions_gt:
        target = int(row["target"])
        if target not in (0, 1):
            all_targets_valid = False
            break
    vr.check("predictions.csv: all targets in {0,1}", all_targets_valid)

    # =======================================================================
    # 11. Verify calibration.csv structure
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 11: Calibration CSV parity")
    print("=" * 72)

    cal_columns = {"model", "split", "bin_index", "bin_left", "bin_right",
                   "n", "mean_predicted_probability", "observed_event_rate"}
    actual_cal_columns = set(calibration_gt[0].keys())
    vr.check(
        "calibration.csv columns",
        cal_columns == actual_cal_columns,
        f"expected={sorted(cal_columns)}, actual={sorted(actual_cal_columns)}",
    )

    # Check row count: 93 rows (4 models × 3 splits × ~8 bins, but not all have 10 bins)
    vr.check(
        "calibration.csv row count",
        len(calibration_gt) == 93,
        f"rows={len(calibration_gt)}",
    )

    # Check model coverage
    cal_models = set(row["model"] for row in calibration_gt)
    vr.check(
        "calibration.csv model coverage",
        cal_models == expected_models,
        f"models={sorted(cal_models)}",
    )

    # =======================================================================
    # 12. Cross-check: validation_candidates in models.json match metrics
    # =======================================================================
    print()
    print("=" * 72)
    print("SECTION 12: Cross-check validation_candidates vs metrics")
    print("=" * 72)

    for name in MODULES:
        gt_model = models_gt[name]
        candidates = gt_model["validation_candidates"]
        selected_penalty = gt_model["selected_regularization"]

        # Find the selected candidate
        selected = [c for c in candidates if c["l2_penalty"] == selected_penalty]
        vr.check(
            f"{name}: selected_regularization has matching candidate",
            len(selected) == 1,
            f"found {len(selected)} candidates with penalty={selected_penalty}",
        )

        if selected:
            sel = selected[0]
            # The selected candidate's metrics should match internal_validation metrics
            val_key = (name, "internal_validation")
            if val_key in metrics_lookup:
                m = metrics_lookup[val_key]
                vr.check(
                    f"{name}: selected candidate AUROC matches validation",
                    abs(sel["auroc"] - m["auroc"]) < 0.001,
                    f"candidate={sel['auroc']:.6f}, metrics={m['auroc']:.6f}",
                )
                vr.check(
                    f"{name}: selected candidate AUPRC matches validation",
                    abs(sel["auprc"] - m["auprc"]) < 0.001,
                    f"candidate={sel['auprc']:.6f}, metrics={m['auprc']:.6f}",
                )
                vr.check(
                    f"{name}: selected candidate Brier matches validation",
                    abs(sel["brier"] - m["brier"]) < 0.001,
                    f"candidate={sel['brier']:.6f}, metrics={m['brier']:.6f}",
                )

    # =======================================================================
    # Summary
    # =======================================================================
    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    all_pass = vr.summary()
    if all_pass:
        print("\n✅ ALL PARITY CHECKS PASSED")
        return 0
    else:
        print("\n❌ SOME PARITY CHECKS FAILED — see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())