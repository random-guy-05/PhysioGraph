#!/usr/bin/env python3
"""Refresh only the multiscale dynamics-first SpO2 instability analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physiograph_colab_core import refresh_multiscale_instability_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root containing fingerprint-validated cached artifacts.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    result = refresh_multiscale_instability_analysis(
        args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        n_splits=args.n_splits,
    )
    print(
        "Dynamics-first instability refresh complete: "
        f"{result['output_dir'] / 'instability_refresh_status.json'}"
    )


if __name__ == "__main__":
    main()
