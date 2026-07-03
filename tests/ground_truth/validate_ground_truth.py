#!/usr/bin/env python3
"""Validate ground-truth artifacts against source manifests.

Run from project root:
    python tests/ground_truth/validate_ground_truth.py
"""
import json
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_DIR = PROJECT_ROOT / "tests" / "ground_truth"

EXPECTED = {
    "mimic/cohort.csv": 17892,
    "mimic/labels.csv": 14553,
    "mimic/features.csv": 14553,
    "eicu/cohort.csv": 27695,
    "eicu/labels.csv": 24194,
    "eicu/features.csv": 24194,
}


def count_csv_rows(path: Path) -> int:
    """Count data rows in CSV (excludes header)."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def validate_source_cohorts():
    """Verify source CSV row counts match manifest declarations."""
    errors = []
    for rel_path, expected in EXPECTED.items():
        full_path = PROJECT_ROOT / "physiograph_outputs" / rel_path
        if not full_path.exists():
            errors.append(f"MISSING: {full_path}")
            continue
        actual = count_csv_rows(full_path)
        if actual != expected:
            errors.append(
                f"ROW COUNT MISMATCH: {rel_path} "
                f"expected={expected} actual={actual}"
            )
    return errors


def validate_ground_truth_files():
    """Verify ground_truth directory has all required files."""
    required = [
        "predictions.csv",
        "metrics.json",
        "models.json",
        "calibration.csv",
        "csv_manifest.json",
        "README.md",
        "validate_ground_truth.py",
    ]
    errors = []
    for name in required:
        path = GT_DIR / name
        if not path.exists():
            errors.append(f"MISSING: {path}")
    return errors


def validate_predictions_row_count():
    """Verify predictions.csv has expected row count from csv_manifest.json."""
    manifest_path = GT_DIR / "csv_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    expected = manifest["files"]["predictions.csv"]["row_count"]
    actual = count_csv_rows(GT_DIR / "predictions.csv")
    if actual != expected:
        return [
            f"predictions.csv row count: expected={expected} actual={actual}"
        ]
    return []


def main():
    all_errors = []
    all_errors.extend(validate_source_cohorts())
    all_errors.extend(validate_ground_truth_files())
    all_errors.extend(validate_predictions_row_count())

    if all_errors:
        print("VALIDATION FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED")
        print(f"  - Source cohort row counts match manifests")
        print(f"  - All ground-truth files present")
        print(f"  - Predictions row count matches manifest")
        sys.exit(0)


if __name__ == "__main__":
    main()
