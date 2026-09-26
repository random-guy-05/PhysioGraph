# Research decision — September 5, 2026

Continuation update: the stopping assessment below was reopened. Full raw
laboratory recovery has now tested an explicitly unresolved cache limitation:
additional records do not increase two-sided first-transition timing support.
See [raw recovery results](SPO2_ARTERIAL_RAW_RECOVERY_RESULTS.md). A distinct
[direct cardiac-output protocol](SPO2_DIRECT_FLOW_PLAN.md) is now investigating
whether the original SpO2 phenotype precedes circulatory deterioration. These
are concrete research actions, not a discovery or a reversal of prior negatives.

The requested biological discovery has **not** been obtained. The current
SpO2 research direction has reached an evidentiary impasse. This assessment
does not show that MIMIC/eICU cannot contain an important finding; it says
that the completed work does not presently justify another positive claim
or a particular next biological experiment.

## Evidence controlling the decision

| Completed experiment | Result and implication |
|---|---|
| First-transition oxygen/heart-rate direction and subsequent hospital mortality | During falling SpO2, HR-fall versus HR-rise mortality differences were +1.25 percentage points in eICU and +9.87 in MIMIC. Multiplicity-adjusted intervals were −6.65 to +8.84 and −3.63 to +23.62. The comparison against rising SpO2 also failed. This does not establish a replicated large-risk physiological phenotype. |
| Paired diagnostic-ECG repolarization | Exposure-associated JTcF changes were +0.011 ms in discovery and −0.680 ms in validation; all four primary Holm p-values were 1.0. Neither the large-effect nor specificity criterion passed. |
| Within-person oxygen–troponin coupling | eICU discovery/validation troponin fold-change ratios were 0.9503 and 0.9627, with both intervals crossing one. The original MIMIC repeated-oxygen support gate failed. No positive internal or external replication. |
| Troponin reporting-limit emergence | Descriptive 12-hour risk differences were +0.40 percentage points in eICU and −9.34 in MIMIC, both intervals crossing zero; adjusted models lacked support. |

The full experiment and candidate history remains in
[the triage register](SPO2_DISCOVERY_CANDIDATE_TRIAGE.md). Negative estimates
and failed support gates have different meanings: neither establishes the
absence of the underlying biology. The initial troponin association cannot
be promoted over the failed downstream checks.

## Measurement limits and what was checked

The existing arterial extraction contains no qualified two-sided PaO2 pairs
within five minutes of both sides of an original SpO2 transition, and only
two eICU/four MIMIC people at fifteen minutes. This is a limit of that cached
extraction and specification, **not** a census of all raw arterial samples,
not a claim that single ABG–SpO2 pairs are absent, and not evidence of artifact.

The linked continuous-waveform pilot contains five exposed original
encounters; none has a complete first-exposure block under the frozen
requirements. The [official public waveform resource](https://physionet.org/content/mimic4wdb/)
was checked again on September 5 and still resolves to the 200-record,
198-person v0.1.0 preview. Its announcement of a larger future collection
does not supply currently analyzed records. Diagnostic ECG metadata created
a separate measurement route, but the completed paired biological experiment
was negative and machine/procedure clock correspondence remains uncertain.

A fresh filename inventory of the authorized local raw sources confirms
that eICU `Full` contains `vitalPeriodic.csv`, `medication.csv`, and
`infusionDrug.csv`, among the other study tables. Neither `vitalAperiodic`
nor `nurseCharting` files, nor matching archive files, were found by the
targeted search under `Data/eICU`. This is a statement about the supplied
local files, not the complete eICU database. It does not imply that all
treatment records are unavailable.

## Continuation criterion

A defensible next experiment needs a materially new, independently supported
biological lead with adequate measurements in the original MIMIC/eICU
cohorts, or additional aligned measurements that resolve a specific current
measurement failure. A documented raw-source recovery may qualify if it
actually supplies such measurements; current cached ceilings do not rule
that out. Merely changing thresholds, endpoints, timing windows, or
adjustment sets until significance appears does not qualify.

The goal is an important biological finding with credible potential relevance
to HF mortality. A randomized trial is not imposed as a prerequisite for
that research finding, but these observational results do not establish a
survival benefit. The current impasse is the absence of a supported biological
lead and the limits of the inspected measurements, not a general objection
to retrospective research.

No new patient experiment was run during this decision assessment. Recording
the impasse is not biological progress. The prior turn did make progress by
finishing the HR-direction experiment, so this is the first recorded
consecutive impasse assessment; the goal initially remained active and unfulfilled.

Subsequent continuation assessments on September 5 reproduced the same
impasse. The third assessment rechecked every recorded evidence fingerprint
and the actual primary HR/ECG result files: no evidence changed and both
advancement criteria still failed. No supported next experiment or live
analysis awaiting completion was identified. The goal was therefore marked
**blocked, not complete**, after three consecutive impasse assessments.
This administrative status change is not research progress or a biological
conclusion. The continuation criteria above remain applicable.

The primary runnable artifact remains
[PhysioGraph_Biological_Discovery.ipynb](../PhysioGraph_Biological_Discovery.ipynb).
Its recorded analyses ran locally in incremental sections. The current
notebook has not been claimed to have run uninterrupted or in Google Colab.
