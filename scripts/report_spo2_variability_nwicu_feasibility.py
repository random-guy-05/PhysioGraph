"""Create reporting-only artifacts for the stopped NWICU validation attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/spo2_variability_nwicu_validation"
NOTEBOOK = ROOT / "PhysioGraph_Biological_Discovery.ipynb"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> None:
    support = json.loads((OUT / "support_counts.json").read_text())
    feasibility = json.loads((OUT / "feasibility.json").read_text())
    lock_path = OUT / "analysis_lock.json"
    if lock_path.with_suffix(".sha256").read_text().strip() != digest(lock_path):
        raise RuntimeError("NWICU lock integrity failure")
    if (OUT / "association_read_receipt.json").exists() or (OUT / "validation_results.json").exists():
        raise RuntimeError("Reporting-only closure expected no NWICU association artifacts")
    if feasibility["association_authorized"] is not False:
        raise RuntimeError("Reporting-only closure requires a failed feasibility gate")

    classification = "NWICU_MODIFIED_ENDPOINT_ANALYSIS_NOT_RUN_INSUFFICIENT_EVENTS"
    report = f"""# NWICU SpO2-variability modified-endpoint feasibility result

**{classification}**

No NWICU exposure–outcome association was fitted or inspected. This is a **LOCAL standalone-runtime execution** of a reporting-only support audit under the prospectively locked modified-endpoint amendment. It is not external validation of the original frozen composite and does not change the frozen eICU classification `POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED`.

## Decisive support result

- Preliminary adult HF cohort alive and observable beyond minute 360: **{support['preliminary_cohort_before_available_pre360_exclusions']:,} stays**.
- Modified-endpoint at-risk cohort after available pre-360 exclusions: **{support['modified_endpoint_at_risk']:,} stays**.
- Exact frozen SpO2 exposure observed: **{support['valid_frozen_exposure']:,} stays ({support['exposure_coverage']:.1%})**.
- Modified primary-endpoint events in the at-risk cohort: **{support['modified_endpoint_events_at_risk']}**.
- Modified primary-endpoint events among exposure-observed stays: **{support['modified_endpoint_events_modeled']}**.
- Later-window events among exposure-observed stays: **{support['later_window_events_modeled']}**.

The user explicitly waived the original 70% exposure-coverage and 20-site requirements. Coverage nevertheless remained only **{support['exposure_coverage']:.1%}**, and NWICU exposes no hospital/site identifier. The retained minimum of 50 modeled events failed by a large margin: 4 were observed. Fitting a many-covariate modified-Poisson model with four events would not constitute credible validation, so the locked implementation stopped before writing an association-read receipt.

## Modified endpoint and structural limits

The prospectively amended endpoint paired sustained cuff-SBP hypotension or mapped MCS with lactate, creatinine, blood-pH, or ALT hypoperfusion within six hours, with both components strictly after minute 360 through 960. Continuous vasoactive-infusion support and oliguria were structurally unavailable. The NWICU dictionary exposed an ECMO pump-settings procedure mapping, but no mapped MCS event contributed to the primary window. The available paired events were driven by sustained hypotension plus creatinine criteria.

The modified endpoint is not the original frozen composite and is not adjudicated cardiogenic shock. No alternate feature, scaling, endpoint, threshold, subgroup, care-unit selection, or low-event association was used as a rescue.

## What this means

NWICU cannot adjudicate the SpO2-variability hypothesis under this amendment. The failure is a combination of low exposure coverage and, decisively, near-absence of observable modified endpoint events—not evidence that the biological association is absent.
"""
    (OUT / "FINAL_REPORT.md").write_text(report)
    status = {
        "status": "completed_without_association",
        "classification": classification,
        "environment": "LOCAL_STANDALONE_RUNTIME",
        "execution_label": "LOCAL standalone-runtime execution",
        "association_inspected": False,
        "association_models_fitted": 0,
        "outcome_fit_count": 0,
        "reason": "retained minimum 50 modeled primary events failed with 4 events",
    }
    write_json(OUT / "run_status.json", status)

    figure_dir = OUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    labels = ["At risk", "Frozen exposure", "Modified events"]
    values = [support["modified_endpoint_at_risk"], support["valid_frozen_exposure"], support["modified_endpoint_events_modeled"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#446E9B", "#729FCF", "#CC4C4C"])
    ax.bar_label(bars, fmt="%d")
    ax.set_ylabel("Stays")
    ax.set_title("NWICU modified-endpoint support")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(figure_dir / "nwicu_support_funnel.png", dpi=180)
    fig.savefig(figure_dir / "nwicu_support_funnel.pdf")
    plt.close(fig)

    cells = [
        nbformat.v4.new_markdown_cell(
            """# SpO2 variability: NWICU modified-endpoint feasibility

This notebook reports the completed **LOCAL standalone-runtime execution** of the NWICU support audit. The modified endpoint and two explicit gate waivers were locked before any exposure–outcome association. The retained ≥50-event gate failed, so **no association was fitted or inspected**. This does not change the frozen eICU classification."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import hashlib, json
import pandas as pd
from IPython.display import Image, display

PROJECT = Path.cwd()
AUDIT = PROJECT / "research/spo2_variability_nwicu_validation"
support = json.loads((AUDIT / "support_counts.json").read_text())
feasibility = json.loads((AUDIT / "feasibility.json").read_text())
status = json.loads((AUDIT / "run_status.json").read_text())
lock_path = AUDIT / "analysis_lock.json"
assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == lock_path.with_suffix(".sha256").read_text().strip()
assert not (AUDIT / "association_read_receipt.json").exists()
assert not (AUDIT / "validation_results.json").exists()
assert status["association_models_fitted"] == 0
assert status["outcome_fit_count"] == 0 and status["association_inspected"] is False
print(status["classification"])
print(status["execution_label"])
print("association_models_fitted = 0")
print("No NWICU association was fitted or inspected.")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Prospective amendment

Available pressure/support components were sustained cuff-SBP hypotension and recorded MCS. Available laboratory components were lactate, creatinine, blood pH, and ALT. Continuous vasoactive-infusion support and oliguria were structurally unavailable. The user waived the original ≥70% coverage and ≥20-site requirements; the ≥50 modeled-event requirement remained locked."""
        ),
        nbformat.v4.new_code_cell(
            """summary = pd.DataFrame([
    {"stage": "Preliminary adult HF cohort", "stays_or_events": support["preliminary_cohort_before_available_pre360_exclusions"]},
    {"stage": "Modified-endpoint at risk", "stays_or_events": support["modified_endpoint_at_risk"]},
    {"stage": "Exact frozen exposure", "stays_or_events": support["valid_frozen_exposure"]},
    {"stage": "Modeled modified events", "stays_or_events": support["modified_endpoint_events_modeled"]},
    {"stage": "Modeled later-window events", "stays_or_events": support["later_window_events_modeled"]},
])
display(summary)
print(f"Exposure coverage: {support['exposure_coverage']:.1%} (original 70% gate waived and explicitly noted)")
print(f"Retained event gate: {support['modified_endpoint_events_modeled']} / 50 — FAILED")
display(Image(filename=str(AUDIT / "figures/nwicu_support_funnel.png")))"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Interpretation

NWICU cannot adjudicate this hypothesis under the prospectively amended endpoint. Only four endpoint events remained among exposure-observed stays. This is a feasibility failure caused by insufficient observable endpoint support—not a null association and not evidence that the biological hypothesis failed."""
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"nwicu-spo2-{index}"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "physiograph": {
                "scope": "read_only_local_nwicu_modified_endpoint_feasibility",
                "execution_environment": "LOCAL_STANDALONE_RUNTIME",
                "association_models_fitted": 0,
            },
        },
    )
    nbformat.write(notebook, NOTEBOOK)
    artifacts = [path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_hashes.json"]
    write_json(OUT / "artifact_hashes.json", {str(path.relative_to(OUT)): digest(path) for path in sorted(artifacts)})
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
