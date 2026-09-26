#!/usr/bin/env python3
"""Run the locked eICU-to-MIMIC validation from cached analysis artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))

from physiograph_colab_core import refresh_locked_external_validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root with mimic/, eicu/, and spo2_drilldown/.",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        required=True,
        help="MIMIC CSV directory containing d_items.csv and procedureevents.csv.",
    )
    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Skip overlap-weighting, doubly robust, and multicenter sensitivities.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = refresh_locked_external_validation(
        args.output_root,
        mimic_root=args.mimic_root,
        run_advanced=not args.no_advanced,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    summary = result["frames"]["locked_external_validation_summary"]
    columns = [
        "endpoint",
        "locked_validation_tier",
        "eicu_adjusted_risk_ratio",
        "mimic_adjusted_risk_ratio",
        "conclusion",
    ]
    print(summary.reindex(columns=columns).to_string(index=False))
    print(f"Locked validation complete: {result['paths']['locked_external_validation_summary']}")


if __name__ == "__main__":
    main()
