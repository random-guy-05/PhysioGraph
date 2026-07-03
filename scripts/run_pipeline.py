#!/usr/bin/env python3
"""Standalone pipeline runner with local-path defaults.

Adjust the paths and parameters below, then run the script locally.

Prerequisites::

    pip install -e /path/to/PhysioGraph

"""

from __future__ import annotations

import json
from pathlib import Path

from physiograph.config import load_config
from physiograph.pipeline import run_pipeline


# ──────────────────────────────────────────────────────────────────────
# Configuration — edit these values for your environment
# ──────────────────────────────────────────────────────────────────────

DATASET = "mimic"  # "mimic" or "eicu"

# Mounted Google Drive roots
MYDRIVE_ROOT = Path("/content/drive/MyDrive")
PROJECT_ROOT = MYDRIVE_ROOT / "Projects" / "PhysioGraph"
MIMIC_ROOT = str(MYDRIVE_ROOT / "Data" / "MIMIC" / "Full")
EICU_ROOT = str(MYDRIVE_ROOT / "Data" / "eICU" / "Full")

# Output directory — defaults to PROJECT_ROOT / "physiograph_outputs" / <dataset>
OUTPUT_DIR = None  # set to a string path to override

# Optional limits for testing (set to None for full run)
MAX_STAYS = None
MAX_CHUNKS = None
CHUNK_SIZE = 250_000


# ──────────────────────────────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────────────────────────────


def main() -> dict[str, str]:
    cfg = load_config(dataset=DATASET)
    resolved = cfg.get("resolved_paths", {})
    default_output_dir = PROJECT_ROOT / "physiograph_outputs" / DATASET

    if DATASET == "mimic":
        mimic_root = MIMIC_ROOT or resolved.get("mimic_root")
        if not mimic_root:
            raise ValueError(
                "MIMIC data root not found. Set MIMIC_ROOT or "
                "configure configs/mimic.yaml with a valid path."
            )
        artifacts = run_pipeline(
            dataset="mimic",
            mimic_root=mimic_root,
            output_dir=str(OUTPUT_DIR or default_output_dir),
            max_stays=MAX_STAYS,
            max_chunks=MAX_CHUNKS,
            chunk_size=CHUNK_SIZE,
        )
    elif DATASET == "eicu":
        eicu_root = EICU_ROOT or resolved.get("eicu_root")
        if not eicu_root:
            raise ValueError(
                "eICU data root not found. Set EICU_ROOT or "
                "configure configs/eicu.yaml with a valid path."
            )
        artifacts = run_pipeline(
            dataset="eicu",
            eicu_root=eicu_root,
            output_dir=str(OUTPUT_DIR or default_output_dir),
            max_stays=MAX_STAYS,
            max_chunks=MAX_CHUNKS,
            chunk_size=CHUNK_SIZE,
        )
    else:
        raise ValueError(f"Unsupported dataset: {DATASET!r}")

    print("Pipeline completed. Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    manifest_path = artifacts.get("manifest")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())
        counts = manifest.get("counts", {})
        print(f"\nRow counts: {counts}")

    return artifacts


if __name__ == "__main__":
    main()
