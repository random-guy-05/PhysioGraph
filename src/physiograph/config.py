from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR: Path | None = None


def _find_config_dir() -> Path:
    global _CONFIG_DIR
    if _CONFIG_DIR is not None:
        return _CONFIG_DIR

    candidates = [
        Path(__file__).resolve().parent.parent.parent / "configs",
        Path.cwd() / "configs",
    ]
    for candidate in candidates:
        if (candidate / "default.yaml").exists():
            _CONFIG_DIR = candidate
            return _CONFIG_DIR

    raise FileNotFoundError(
        "Cannot locate configs/ directory. Searched: "
        + ", ".join(str(c) for c in candidates)
    )


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_paths(cfg: dict) -> dict:
    paths = cfg.get("paths", {})
    colab_roots = paths.get("colab_drive_roots", [])
    local_root = paths.get("local_drive_root", "")

    drive_root: Path | None = None
    for candidate in colab_roots:
        p = Path(candidate)
        if p.exists():
            drive_root = p
            break

    if drive_root is None:
        expanded = Path(local_root).expanduser()
        if expanded.exists():
            drive_root = expanded

    if drive_root is not None:
        mimic_sub = paths.get("mimic_data_subdir", "Data/MIMIC/Full")
        eicu_sub = paths.get("eicu_data_subdir", "Data/eICU/Full")
        output_sub = paths.get("output_subdir", "physiograph_outputs")

        cfg.setdefault("resolved_paths", {})
        cfg["resolved_paths"]["drive_root"] = str(drive_root)
        cfg["resolved_paths"]["mimic_root"] = str(drive_root / mimic_sub)
        cfg["resolved_paths"]["eicu_root"] = str(drive_root / eicu_sub)
        cfg["resolved_paths"]["output_root"] = str(drive_root / output_sub)

    return cfg


_REQUIRED_TOP_LEVEL = [
    "observation_hours",
    "outcome_hours",
    "time_step_hours",
    "clinical_normals",
    "training",
    "model",
]


def _validate(cfg: dict) -> None:
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")


def load_config(dataset: str | None = None) -> dict[str, Any]:
    config_dir = _find_config_dir()

    with open(config_dir / "default.yaml") as f:
        cfg = yaml.safe_load(f)

    if dataset is not None:
        dataset_file = config_dir / f"{dataset}.yaml"
        if not dataset_file.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {dataset_file}. "
                f"Expected one of: mimic.yaml, eicu.yaml"
            )
        with open(dataset_file) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            cfg = _deep_merge(cfg, overrides)

    cfg = _resolve_paths(cfg)
    _validate(cfg)

    return cfg
