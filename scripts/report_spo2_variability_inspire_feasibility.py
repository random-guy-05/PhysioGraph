"""Build the reporting-only notebook for the stopped INSPIRE validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/spo2_variability_inspire_validation"
NOTEBOOK = ROOT / "PhysioGraph_Biological_Discovery.ipynb"
CLASSIFICATION = "INSPIRE_EXTERNAL_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    support = json.loads((OUT / "support_counts.json").read_text())
    feasibility = json.loads((OUT / "feasibility.json").read_text())
    status = json.loads((OUT / "run_status.json").read_text())
    lock = OUT / "analysis_lock.json"
    if digest(lock) != lock.with_suffix(".sha256").read_text().strip():
        raise RuntimeError("INSPIRE analysis lock integrity failure")
    if (OUT / "association_read_receipt.json").exists() or (OUT / "validation_results.json").exists():
        raise RuntimeError("Stopped INSPIRE run must not have association artifacts")
    if feasibility["association_authorized"] is not False:
        raise RuntimeError("Reporting-only notebook requires a failed feasibility gate")
    if status["association_models_fitted"] != 0 or status["association_inspected"] is not False:
        raise RuntimeError("INSPIRE zero-association invariant failed")
    if support["primary_events_modeled"] != 0:
        raise RuntimeError("INSPIRE reporting closure expected zero modeled endpoint events")
    if status["classification"] != CLASSIFICATION:
        raise RuntimeError("Unexpected INSPIRE classification")
    cells = [
        nbformat.v4.new_markdown_cell(
            """# SpO2 variability: INSPIRE external-validation attempt

This notebook reports a **LOCAL standalone-runtime execution**, not a Colab run. INSPIRE was evaluated under the original frozen pressure/support-plus-hypoperfusion endpoint. The retained ≥50 modeled-event gate failed, so no INSPIRE SpO2–outcome association was fitted or inspected."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import hashlib, json
import pandas as pd
from IPython.display import Image, display

PROJECT = Path.cwd()
AUDIT = PROJECT / "research/spo2_variability_inspire_validation"
support = json.loads((AUDIT / "support_counts.json").read_text())
feasibility = json.loads((AUDIT / "feasibility.json").read_text())
status = json.loads((AUDIT / "run_status.json").read_text())
lock_path = AUDIT / "analysis_lock.json"
assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == lock_path.with_suffix(".sha256").read_text().strip()
assert status["classification"] == "INSPIRE_EXTERNAL_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE"
assert status["association_models_fitted"] == 0
assert status["association_inspected"] is False
assert feasibility["association_authorized"] is False
assert not (AUDIT / "association_read_receipt.json").exists()
assert not (AUDIT / "validation_results.json").exists()
print(status["classification"])
print("LOCAL standalone-runtime execution")
print("association_models_fitted = 0")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Frozen design

The exposure is the RMS residual around a linear trend through hourly median SpO2 in ICU minutes 0–240, transformed only with the MIMIC-frozen clipping and standardization constants. The original endpoint pairs sustained cuff hypotension, recorded vasoactive/inotropic support, or MCS with lactate, creatinine, ALT, pH, or oliguria hypoperfusion within six hours. Both domains must occur strictly after minute 360 through 960; the later sensitivity starts strictly after minute 480.

The previously authorized coverage and site-count waivers were retained and are reported. The ≥50 modeled-event gate and exact endpoint were not waived."""
        ),
        nbformat.v4.new_code_cell(
            """summary = pd.DataFrame([
    {"stage": "Preliminary eligible HF ICU cohort", "stays_or_events": support["preliminary_cohort_before_pre360_exclusions"]},
    {"stage": "At risk after pre-360 exclusions", "stays_or_events": support["at_risk"]},
    {"stage": "Exact frozen exposure", "stays_or_events": support["valid_frozen_exposure"]},
    {"stage": "Primary events in at-risk cohort", "stays_or_events": support["primary_events_at_risk"]},
    {"stage": "Primary events in exposure-observed frame", "stays_or_events": support["primary_events_modeled"]},
    {"stage": "Later-window events in exposure-observed frame", "stays_or_events": support["later_events_modeled"]},
])
display(summary)
print(f"Exposure coverage: {support['exposure_coverage']:.1%} — below the original 70% gate; waiver noted")
print(f"Released sites: {support['released_sites']} — below the original 20-site gate; waiver noted")
print(f"Retained modeled-event gate: {support['primary_events_modeled']} / 50 — FAILED")
display(Image(filename=str(AUDIT / "figures/inspire_support_funnel.png")))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Result

`INSPIRE_EXTERNAL_VALIDATION_NOT_RUN_FEASIBILITY_FAILURE`

INSPIRE did not supply an exposure-observed event frame capable of testing the frozen association: the at-risk cohort contained one endpoint event, and that stay lacked a valid frozen SpO2-variability exposure. This is a failed external-validation **attempt due to insufficient event support**, not a null association and not evidence that the biological hypothesis is false. No alternate endpoint, exposure, scaling, threshold, department, subgroup, or time window was used as a rescue."""
        ),
        nbformat.v4.new_markdown_cell(
            """## Prior results remain frozen

The original post-hoc signal remains MIMIC RR 1.251 (95% CI 1.104–1.418), while eICU remains `POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED` with RR 1.104 (95% CI 0.890–1.369). The separate NWICU modified-endpoint feasibility result also remains unchanged. INSPIRE does not reclassify any of them because no INSPIRE association was fitted."""
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"inspire-spo2-{index}"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "physiograph": {
                "scope": "read_only_local_inspire_external_validation_feasibility",
                "execution_environment": "LOCAL_STANDALONE_RUNTIME",
                "association_models_fitted": 0,
            },
        },
    )
    nbformat.write(notebook, NOTEBOOK)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
