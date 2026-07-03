"""Smoke tests verifying pytest infrastructure works correctly."""

import json
from pathlib import Path

import numpy as np
import pytest


def test_project_root_exists(project_root: Path):
    assert project_root.is_dir()


def test_ground_truth_dir_exists(ground_truth_dir: Path):
    assert ground_truth_dir.is_dir()


@pytest.mark.slow
def test_slow_marker_works():
    assert True


@pytest.mark.integration
def test_integration_marker_works():
    assert True


@pytest.mark.mimic
def test_mimic_marker_works():
    assert True


@pytest.mark.eicu
def test_eicu_marker_works():
    assert True


def test_load_ground_truth_json(ground_truth_dir: Path):
    filepath = ground_truth_dir / "metrics.json"
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, (dict, list))


def test_assert_parity_scalars():
    np.testing.assert_allclose(1.0, 1.0 + 1e-8, atol=1e-6, rtol=1e-6)


def test_assert_parity_arrays():
    np.testing.assert_allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
