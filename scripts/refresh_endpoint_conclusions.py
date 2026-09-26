#!/usr/bin/env python3
"""Refresh only the endpoint evidence-classification matrix.

The command reads fingerprint-validated analysis summaries and cannot invoke
raw MIMIC/eICU extraction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physiograph_colab_core import refresh_endpoint_conclusions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing run root containing spo2_drilldown/.",
    )
    args = parser.parse_args()
    result = refresh_endpoint_conclusions(args.output_root)
    print(
        "Endpoint-conclusion refresh complete: "
        f"{result['output_dir'] / 'endpoint_conclusions_refresh_status.json'}"
    )


if __name__ == "__main__":
    main()
