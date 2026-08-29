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


def test_explicit_config_override_is_applied(tmp_path: Path):
    from physiograph.config import load_config

    override = tmp_path / "override.yaml"
    override.write_text("observation_hours: 6\n", encoding="utf-8")
    config = load_config(dataset="mimic", config_path=override)
    assert config["observation_hours"] == 6


def test_missing_explicit_config_fails_loudly(tmp_path: Path):
    from physiograph.config import load_config

    with pytest.raises(FileNotFoundError, match="Explicit config not found"):
        load_config(dataset="mimic", config_path=tmp_path / "missing.yaml")
