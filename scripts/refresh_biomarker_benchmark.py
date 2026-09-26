#!/usr/bin/env python3
"""Refresh only the SpO2/SBP/lactate/Kapur-SCAI benchmark.

The command reads fingerprint-validated cached events and endpoints.  It never
invokes MIMIC or eICU extraction and therefore does not rebuild source data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physiograph_colab_core import refresh_biomarker_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root containing fingerprinted dataset artifacts.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=300,
        help="Patient-cluster bootstrap repetitions for paired performance CIs.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    result = refresh_biomarker_benchmark(
        args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        n_splits=args.n_splits,
    )
    print(
        "Biomarker benchmark refresh complete: "
        f"{result['output_dir'] / 'biomarker_benchmark_refresh_status.json'}"
    )


if __name__ == "__main__":
    main()
