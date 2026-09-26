"""Build and locally execute the MIMIC-III frozen-replication report notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/spo2_variability_mimic3_validation"
NOTEBOOK = ROOT / "PhysioGraph_Biological_Discovery.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# MIMIC-III frozen SpO₂-variability replication\n\n"
            "**Execution label:** LOCAL standalone-runtime reporting execution (not Google Colab).\n\n"
            "This notebook reports the already locked and locally executed MIMIC-III v1.4 analysis. "
            "It verifies hashes and reads frozen results; it does not refit models. MIMIC-III and "
            "MIMIC-IV come from the same hospital and may overlap, so this is a historical "
            "database-version replication—not independent-site external validation."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import hashlib, json\n"
            "from IPython.display import Image, display\n\n"
            "ROOT = Path.cwd()\n"
            "OUT = ROOT / 'research/spo2_variability_mimic3_validation'\n"
            "def sha256(path):\n"
            "    h = hashlib.sha256()\n"
            "    with path.open('rb') as f:\n"
            "        for block in iter(lambda: f.read(8*1024*1024), b''):\n"
            "            h.update(block)\n"
            "    return h.hexdigest()\n"
            "lock = json.loads((OUT/'analysis_lock.json').read_text())\n"
            "assert sha256(OUT/'analysis_lock.json') == (OUT/'analysis_lock.sha256').read_text().strip()\n"
            "artifact_hashes = json.loads((OUT/'artifact_hashes.json').read_text())\n"
            "for relative, expected in artifact_hashes.items():\n"
            "    assert sha256(OUT/relative) == expected, relative\n"
            "status = json.loads((OUT/'run_status.json').read_text())\n"
            "support = json.loads((OUT/'support_counts.json').read_text())\n"
            "feasibility = json.loads((OUT/'feasibility.json').read_text())\n"
            "print('Lock and artifact hashes verified.')\n"
            "print('Execution: LOCAL_STANDALONE_RUNTIME (not Colab)')"
        ),
        nbformat.v4.new_code_cell(
            "print(status['classification'])\n"
            "print(f\"At risk: {support['at_risk']:,}\")\n"
            "print(f\"Frozen exposure: {support['valid_frozen_exposure']:,} ({support['exposure_coverage']:.1%})\")\n"
            "print(f\"Modeled primary events: {support['primary_events_modeled']} / 50 required\")\n"
            "print(f\"Association models fitted: {status['association_models_fitted']}\")\n"
            "print('Coverage gate waived:', feasibility['waivers']['minimum_70_percent_exposure_coverage'])\n"
            "print('20-site gate waived:', feasibility['waivers']['minimum_20_contributing_sites'])"
        ),
        nbformat.v4.new_code_cell(
            "result_path = OUT/'validation_results.json'\n"
            "if result_path.exists():\n"
            "    result = json.loads(result_path.read_text())\n"
            "    p, s = result['primary'], result['later_window']\n"
            "    print(f\"Primary adjusted RR per frozen MIMIC-IV SD: {p['rr_per_mimic_sd']:.3f} \"\n"
            "          f\"(95% CI {p['ci_low']:.3f}–{p['ci_high']:.3f}), p={p['p_value']:.4g}\")\n"
            "    print(f\"Later-window RR: {s['rr_per_mimic_sd']:.3f} \"\n"
            "          f\"(95% CI {s['ci_low']:.3f}–{s['ci_high']:.3f})\")\n"
            "else:\n"
            "    assert status['association_models_fitted'] == 0\n"
            "    print('Feasibility stop remained binding; no SpO₂-outcome association was fitted or inspected.')"
        ),
        nbformat.v4.new_code_cell(
            "figures = sorted((OUT/'figures').glob('*.png'))\n"
            "for figure in figures:\n"
            "    display(Image(filename=str(figure)))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation\n\n"
            "The endpoint is the frozen constructed pressure/support-plus-hypoperfusion composite, "
            "not adjudicated cardiogenic shock. The earlier exposure-coverage and site-count waivers "
            "are disclosed. No alternate feature, endpoint, threshold, subgroup, unit, or time-window "
            "rescue was used. This result does not change the frozen eICU classification and cannot "
            "establish independent external transportability because the MIMIC versions share a source institution."
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    return notebook


def main() -> None:
    notebook = build_notebook()
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    nbformat.write(executed, NOTEBOOK)
    print(json.dumps({"notebook": str(NOTEBOOK), "cells": len(executed.cells), "execution": "LOCAL_STANDALONE_RUNTIME"}, indent=2))


if __name__ == "__main__":
    main()
