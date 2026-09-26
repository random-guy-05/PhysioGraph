#!/usr/bin/env python3
"""Refresh advanced episode inference from committed cached artifacts only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

from physiograph_colab_core import refresh_advanced_episode_inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root containing mimic/, eicu/, and spo2_drilldown/.",
    )
    parser.add_argument("--mimic-root", type=Path)
    parser.add_argument("--eicu-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = refresh_advanced_episode_inference(
        args.output_root,
        mimic_root=args.mimic_root,
        eicu_root=args.eicu_root,
        max_stays=None,
        max_chunks=None,
        chunk_size=args.chunk_size,
    )
    print(
        "Advanced episode inference refresh complete: "
        f"{result['output_dir'] / 'advanced_episode_refresh_status.json'}"
    )


if __name__ == "__main__":
    main()
