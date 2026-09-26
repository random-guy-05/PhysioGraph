"""Record the human-readable novelty adjudication without rerunning locked models."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/shock_signal_discovery"


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


config = read("analysis_config.json")
assert all(digest(ROOT / name) == expected for name, expected in config["code_hashes"].items())
screen = read("mimic_candidate_screen.json")
assert len(screen) == 36
winners = [row for row in screen if row["final_gate"]]
assert len(winners) == 1 and winners[0]["feature"] == "sbp_level"
assert read("run_status.json")["status"] == "completed"
decision = read("decision_gate_summary.json")
assert decision["eicu_accessed"] is False
now = datetime.now(timezone.utc).isoformat()
review = {
    "review_utc": now,
    "reviewed_candidate": "sbp_level",
    "novelty_passed": False,
    "classification": "KNOWN_SIGNAL_NO_NOVEL_WINNER",
    "reason": "Lower systolic pressure is an established hemodynamic warning marker. The exact fitted coefficient in this cohort is not a novel physiological signal.",
    "scope": "Only the sole eligible candidate was reviewed. No runner-up was substituted.",
    "sources": [
        {"title": "SCAI SHOCK Stage Classification Expert Consensus Update (2022)", "url": "https://www.jacc.org/doi/10.1016/j.jacc.2022.01.018", "finding": "Stage B includes relative hypotension without hypoperfusion; the framework already treats low pressure as a warning state."},
        {"title": "2025 ACC Concise Clinical Guidance on Cardiogenic Shock", "url": "https://www.jacc.org/doi/10.1016/j.jacc.2025.02.018", "finding": "Low systolic pressure and a substantial fall from baseline are established reasons to suspect cardiogenic shock."}
    ],
    "limitation": "These sources establish the marker's lack of conceptual novelty; they do not independently validate this exact four-hour coefficient or endpoint."
}
write("novelty_review.json", review)
write("statistical_decision_before_novelty.json", decision)
decision.update(classification=review["classification"], novelty_passed=False, novelty_review_utc=now, external_winner=None, runner_up_substituted=False, eicu_accessed=False, eicu_must_remain_unopened=True)
write("decision_gate_summary.json", decision)
status = read("run_status.json")
status.update(classification=review["classification"], stage="completed_including_novelty_review", novelty_review_utc=now)
write("run_status.json", status)
report = """# Completed objective-shock single-signal discovery experiment

Final decision: **KNOWN_SIGNAL_NO_NOVEL_WINNER**.

The finite MIMIC-only screen completed locally on 2026-09-07. Exactly 36 candidates
(nine signals × four dynamics) were evaluated. One candidate passed the statistical,
lead-window, bootstrap, and anti-circularity gates: lower systolic blood-pressure
level. It failed the prespecified literature-novelty gate. No candidate was substituted
and eICU was not opened for this experiment. The prior masked-pulsatility hypothesis
remains archived.

## Completed results

The corrected at-risk cohort contains 8,196 stays, with 238 primary composite
events (2.90%) and 221 events in the >8–16-hour sensitivity window. The SBP model
includes 7,429 stays and 196 events, giving 90.64% observed coverage.

| SBP level result | RR per 1 SD higher SBP | 95% interval | Additional evidence |
|---|---:|---:|---|
| Adjusted primary model | 0.551 | 0.434–0.699 | BH q = 0.0000160 |
| >8–16-hour outcome window | 0.533 | 0.418–0.681 | BH q = 0.00000406 |
| Bootstrap median | 0.546 | 0.421–0.685 | 200/200 successful; 100% sign consistency |
| Excluding hypotension as endpoint completion | 0.415 | 0.274–0.627 | Requires support/MCS plus hypoperfusion |

One SD was approximately 21.75 mmHg after winsorization. The primary model therefore
corresponds to RR 1.82 per SD lower SBP. This is an adjusted observational association,
not a causal effect or a validated clinical warning algorithm. The fixed temporal
separation is not proof that this signal changes earliest among all possible signals.

No slope, change, or variability feature passed all gates. SpO2 variability had
RR 1.251 and q=0.00539 but failed the required primary magnitude of 1.30.
Temperature variability had only 245 observed stays and six events (2.99% coverage),
so its large estimate could not advance. Urine-output level coverage was 43.73%,
and urine dynamics coverage was 15.93%; those candidates failed the coverage gate.

## Novelty adjudication

The [2022 SCAI consensus](https://www.jacc.org/doi/10.1016/j.jacc.2022.01.018)
already recognizes relative hypotension with preserved perfusion as a beginning
shock state. The [2025 ACC guidance](https://www.jacc.org/doi/10.1016/j.jacc.2025.02.018)
includes low SBP and a substantial fall from baseline among established warning signs.
These sources support a lack of conceptual novelty for the selected marker, rather
than independent validation of this exact model. Under the frozen rule, that ends
the experiment without an external test or runner-up review.

## Execution and provenance

The written protocol froze at 18:04:48 UTC. A failed initial association attempt at
19:32:30 UTC stopped before its first fit because a duplicated level-column merge
removed the expected column name. Two correction records preserve the implementation
history, including timing-boundary, eligibility, availability, multiplicity, and
algebraic-redundancy repairs. Scientific gates and the written protocol were unchanged.

The successful run locked configuration SHA-256
`5041d8085d3f5f7b5221b7fc2aeda194990950291f7a02cb8a7b9ddb9f00037d`
before association access at 20:17:55 UTC, then completed the statistical analysis at
20:19:55 UTC. The notebook contains all seven executed code cells, with no stored
errors. Fifteen focused tests and Ruff checks passed before this run. The novelty
adjudication was appended afterward without changing any locked analysis code.

The primary runnable deliverable is `PhysioGraph_Biological_Discovery.ipynb`, intended
for Google Colab using the configured Drive project and source files. This execution
was **local**, not a fresh Google Colab execution. Re-running the notebook regenerates
the statistical screen; this saved report records the separate literature adjudication.

## Limits

This is a constructed shock/hypoperfusion endpoint in a heart-failure cohort, not
adjudicated cardiogenic shock. Laboratory and urine documentation remain selective;
missing measurements are not evidence of normal physiology. The outcome requires
observed paired domains, and ordinary fixed-window regression does not establish
causality or resolve informative loss of observation. The result supports the
protocol's stop decision, not a universal claim that no useful precursor exists.
"""
(OUT / "FINAL_REPORT.md").write_text(report)
notebook_path = ROOT / "PhysioGraph_Biological_Discovery.ipynb"
notebook = json.loads(notebook_path.read_text())
assert len(notebook["cells"]) == 9
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
assert len(code_cells) == 7
assert all(cell["execution_count"] is not None for cell in code_cells)
assert not any(output["output_type"] == "error" for cell in code_cells for output in cell["outputs"])
for cell in code_cells:
    source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    compile(source, cell["id"], "exec")
note = "\n\n## Completed local run — final literature decision\n\n**KNOWN_SIGNAL_NO_NOVEL_WINNER.** All 36 candidates were screened in 8,196 at-risk MIMIC stays (238 events). Lower SBP level was the sole statistical/stability winner (adjusted RR 0.551 per SD higher SBP; 95% CI 0.434–0.699), but it is an established warning marker under the [2022 SCAI consensus](https://www.jacc.org/doi/10.1016/j.jacc.2022.01.018) and [2025 ACC guidance](https://www.jacc.org/doi/10.1016/j.jacc.2025.02.018). Novelty failed; no runner-up was substituted and eICU was not tested. Full results and correction provenance: `research/shock_signal_discovery/FINAL_REPORT.md`. The outputs below are a local execution. Re-running regenerates the statistical screen; literature adjudication is recorded separately in `novelty_review.json`.\n"
source = notebook["cells"][0]["source"]
source = source if isinstance(source, str) else "".join(source)
if "## Completed local run — final literature decision" not in source:
    notebook["cells"][0]["source"] = source + note
notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
write("completion_manifest.json", {
    "completed_utc": now,
    "classification": review["classification"],
    "locked_analysis_code_verified_unchanged": True,
    "candidate_count": 36,
    "notebook_code_cells_executed": 7,
    "notebook_error_outputs": 0,
    "execution_environment": "local",
    "files": {name: digest(OUT / name) for name in ["FINAL_REPORT.md", "novelty_review.json", "decision_gate_summary.json", "run_status.json", "mimic_candidate_screen.json", "analysis_lock.json", "analysis_config.json"]},
    "notebook_sha256": digest(notebook_path),
    "note": "Original artifact_hashes.json records the pre-novelty statistical output. This completion manifest records the final adjudicated state."
})
print(json.dumps({"classification": review["classification"], "code_cells": 7, "errors": 0, "eicu_accessed": False}, indent=2))
