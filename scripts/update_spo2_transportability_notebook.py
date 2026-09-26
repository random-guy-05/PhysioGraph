"""Build the concise, read-only SpO2 transportability result notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "PhysioGraph_Biological_Discovery.ipynb"


def main() -> None:
    cells = [
        nbformat.v4.new_markdown_cell(
            """# Frozen SpO2-variability validation and transportability audit

This notebook is a concise, read-only presentation of the completed **LOCAL** analysis. It does not refit the frozen eICU validation or the one-time MIMIC measurement-degradation experiment. The frozen external result remains **POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED**; the subsequent decision is **THIRD_COHORT_STRONGLY_JUSTIFIED**.

The endpoint is a constructed pressure/support-plus-hypoperfusion composite, **not adjudicated cardiogenic shock**. No Colab execution is claimed."""
        ),
        nbformat.v4.new_code_cell(
            """from pathlib import Path
import hashlib
import json
import pandas as pd
from IPython.display import Image, display

PROJECT = Path.cwd()
AUDIT = PROJECT / "research/spo2_variability_transportability"
LOCKED = PROJECT / "research/spo2_variability_validation"
if not AUDIT.exists() or not LOCKED.exists():
    raise FileNotFoundError("Run this notebook from the PhysioGraph repository root with completed LOCAL artifacts available.")

sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
external_lock = json.loads((LOCKED / "external_validation_lock.json").read_text())
external_config = json.loads((LOCKED / "external_validation_config.json").read_text())
degradation_lock = json.loads((AUDIT / "measurement_degradation_lock.json").read_text())
assert sha256(LOCKED / "external_validation_config.json") == external_lock["config_sha256"]
assert sha256(AUDIT / "measurement_degradation_lock.json") == (AUDIT / "measurement_degradation_lock.sha256").read_text().strip()
assert degradation_lock["frozen_external_validation_lock_sha256"] == external_lock["config_sha256"]

summary = json.loads((AUDIT / "transportability_summary.json").read_text())
precision = json.loads((AUDIT / "eicu_precision_analysis.json").read_text())
degradation = json.loads((AUDIT / "mimic_degradation_results.json").read_text())
decision = json.loads((AUDIT / "third_cohort_decision.json").read_text())
print("Frozen validation:", summary["frozen_validation_classification"])
print("Transportability decision:", decision["classification"])
print("Execution environment:", summary["environment"])"""
        ),
        nbformat.v4.new_markdown_cell(
            """## The two decisive findings

1. **eICU was not precise enough to adjudicate RR 1.20–1.25 well.** With 85 modeled events, the observed hospital-clustered SE implies approximate two-sided power of 38.2% for RR 1.20 and 52.9% for RR 1.25. Approximate event requirements are 243/325 for 80%/90% power at RR 1.20 and 162/217 at RR 1.25.

2. **The measurement process can plausibly attenuate this exact feature.** The outcome-blind eICU timing-template operator was hashed before outcome access. Applied once to MIMIC, it reduced the frozen RR from 1.251 to 1.061, corresponding to 73.5% attenuation of log(RR). This is a post-hoc mechanism audit, not validation and not proof that measurement differences fully caused the eICU result.

MIMIC had a median 4 raw SpO2 observations in minutes 0–240 versus 46 in eICU; median observation spacing was 38 versus 5 minutes. Both sources were integer-valued, so the locked operator added no rounding or artificial noise."""
        ),
        nbformat.v4.new_code_cell(
            """power = pd.DataFrame(precision["power"])[["true_rr", "power"]].copy()
power["power_percent"] = 100 * power.pop("power")
required = pd.DataFrame(precision["required_events"])[["true_rr", "target_power", "required_events"]]
effects = pd.DataFrame([
    {"analysis": "MIMIC frozen", **{k: degradation["original_locked_mimic"][k] for k in ["rr_per_mimic_sd", "ci_low", "ci_high", "p_value"]}},
    {"analysis": "MIMIC after locked degradation", **{k: degradation["degraded_mimic_single_fit"][k] for k in ["rr_per_mimic_sd", "ci_low", "ci_high", "p_value"]}},
    {"analysis": "eICU frozen external", "rr_per_mimic_sd": summary["precision"]["estimate"]["rr"], **{k: summary["precision"]["estimate"][k] for k in ["ci_low", "ci_high"]}},
])
display(power.round(2))
display(required)
display(effects.round(4))
print(f"Original/degraded Pearson r: {degradation['agreement']['pearson_correlation']:.3f}")
print(f"Same-quartile retention: {degradation['agreement']['same_quartile_fraction']:.1%}")
print(f"Log-RR attenuation: {degradation['percent_attenuation_of_log_rr']:.1f}%")"""
        ),
        nbformat.v4.new_markdown_cell(
            """## Decision boundary

One additional independent cohort is scientifically justified if it can contribute substantially more than 85 events and a measurement process capable of reproducing the frozen hourly-median construct with high fidelity. The exposure, chronology, adjustment philosophy, support gates, and confirmation rule are frozen in `research/spo2_variability_transportability/THIRD_COHORT_VALIDATION_PROTOCOL.md`.

Pooling remains descriptive: **pooling does not convert a failed external validation into successful external validation.**"""
        ),
        nbformat.v4.new_code_cell(
            """for name in [
    "measurement_density_comparison.png",
    "feature_distribution_comparison.png",
    "original_vs_degraded_mimic.png",
    "locked_mimic_eicu_forest.png",
]:
    path = AUDIT / "figures" / name
    print(name)
    display(Image(filename=str(path)))"""
        ),
    ]
    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"spo2-transport-{index}"
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "physiograph": {"scope": "read_only_local_spo2_transportability_audit"},
        },
    )
    nbformat.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK} with {len(cells)} concise cells")


if __name__ == "__main__":
    main()
