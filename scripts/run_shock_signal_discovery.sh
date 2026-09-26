#!/bin/bash
# Launch the complete notebook run with a durable local diagnostic log.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
discovery_log="/Users/admin/Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive/Data/PhysioGraph_Biological_Discovery_20260905/shock_discovery_attempt_004.log"
if [[ -e "$discovery_log" ]]; then
  echo "Refusing to overwrite an existing attempt log: $discovery_log" >&2
  exit 1
fi
uv run --with duckdb==1.4.3 --with numpy==2.2.6 \
  --with pandas==2.2.3 --with scipy==1.15.3 --with statsmodels==0.14.5 \
  --with matplotlib==3.10.6 python -u scripts/build_biological_notebook.py \
  --shock-signal-discovery-only 2>&1 | tee "$discovery_log"
