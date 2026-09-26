#!/usr/bin/env python3
"""Regenerate PhysioGraph analyses from committed ETL artifacts.

This intentionally cannot rebuild raw data. Artifact schema, parameters,
SHA-256 fingerprints, sizes, and (when roots are supplied) source fingerprints
are validated before the analysis layer starts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

from physiograph_colab_core import run_physiograph_colab


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root containing mimic/ and eicu/ artifact directories.",
    )
    parser.add_argument("--mimic-root", type=Path)
    parser.add_argument("--eicu-root", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--run-legacy-comparator",
        action="store_true",
        help="Also rerun the legacy locked comparator (off by default).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_physiograph_colab(
        project_root=args.project_root,
        build_new=False,
        analysis_only=True,
        mimic_root=args.mimic_root,
        eicu_root=args.eicu_root,
        output_root=args.output_root,
        run_comparator=args.run_legacy_comparator,
        max_stays=None,
        max_chunks=None,
        chunk_size=args.chunk_size,
        execution_environment="local",
        require_all_requested_datasets=True,
    )
    print(f"Analysis-only rerun complete: {result['status_path']}")


if __name__ == "__main__":
    main()
