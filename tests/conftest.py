"""
Shared pytest fixtures and helpers for PhysioGraph test suite.

Provides:
- Path fixtures for project directories and output files
- Ground-truth loading utilities
- Parity assertion helpers
- Custom pytest markers (slow, integration, mimic, eicu)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config):
    """Register custom markers to avoid unknown-marker warnings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "mimic: marks tests requiring MIMIC-IV data")
    config.addinivalue_line("markers", "eicu: marks tests requiring eICU data")


@pytest.fixture
def project_root() -> Path:
    """Absolute path to the PhysioGraph project root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def ground_truth_dir(project_root: Path) -> Path:
    """Path to tests/ground_truth/ directory."""
    return project_root / "tests" / "ground_truth"


@pytest.fixture
def mimic_cohort_path(project_root: Path) -> Path:
    """Path to MIMIC cohort CSV output."""
    return project_root / "physiograph_outputs" / "mimic" / "cohort.csv"


@pytest.fixture
def eicu_cohort_path(project_root: Path) -> Path:
    """Path to eICU cohort CSV output."""
    return project_root / "physiograph_outputs" / "eicu" / "cohort.csv"


@pytest.fixture
def metrics_path(project_root: Path) -> Path:
    """Path to locked comparator validation metrics JSON."""
    return project_root / "physiograph_outputs" / "locked_comparator_validation" / "metrics.json"


@pytest.fixture
def models_path(project_root: Path) -> Path:
    """Path to locked comparator validation models JSON."""
    return project_root / "physiograph_outputs" / "locked_comparator_validation" / "models.json"


def load_ground_truth(name: str, ground_truth_dir: Path | None = None) -> dict | list:
    """Load a ground-truth file by name (JSON or CSV).

    Parameters
    ----------
    name : str
        Filename within tests/ground_truth/ (e.g. "metrics.json", "predictions.csv").
    ground_truth_dir : Path, optional
        Override directory. Defaults to tests/ground_truth/ relative to this file.

    Returns
    -------
    dict or list
        Parsed JSON object, or list of dicts for CSV files.
    """
    if ground_truth_dir is None:
        ground_truth_dir = Path(__file__).resolve().parent / "ground_truth"

    filepath = ground_truth_dir / name
    if not filepath.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {filepath}")

    if filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    elif filepath.suffix == ".csv":
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported ground-truth format: {filepath.suffix}")


def assert_parity(
    actual,
    expected,
    tolerance: float = 1e-6,
    err_msg: str = "",
):
    """Assert numerical parity between actual and expected values.

    Handles scalars, arrays, and nested structures (dicts/lists of numbers).

    Parameters
    ----------
    actual : scalar, array-like, or dict
        Computed values.
    expected : scalar, array-like, or dict
        Reference ground-truth values.
    tolerance : float
        Absolute + relative tolerance for numpy.testing.assert_allclose.
    err_msg : str
        Optional message prepended to assertion errors.
    """
    prefix = f"{err_msg}: " if err_msg else ""

    if isinstance(actual, dict) and isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys()), (
            f"{prefix}Dict keys mismatch: {set(actual.keys())} vs {set(expected.keys())}"
        )
        for key in actual:
            assert_parity(actual[key], expected[key], tolerance, err_msg=f"{prefix}key={key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (
            f"{prefix}Length mismatch: {len(actual)} vs {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_parity(a, e, tolerance, err_msg=f"{prefix}index={i}")
    else:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            atol=tolerance,
            rtol=tolerance,
            err_msg=f"{prefix}Numerical parity failed",
        )
