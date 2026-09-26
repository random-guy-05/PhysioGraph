# One-time locked MIMIC-III endpoint-parity correction

**MIMIC3_PARITY_CORRECTED_INVERSE**

## What was corrected

Only the two endpoint defects established by the zero-association forensic audit were corrected: MIMIC-III oliguria surveillance now begins at hour 4, and the later endpoint is the corrected primary composite whose completion time is after minute 480. The SpO₂ exposure uses CareVue item 646 only and the one pre-locked `AVAILABILITY_PARITY_APPLIED` branch. CareVue STORETIME was judged sufficiently comparable to impose availability by minute 240 based on the documented MIMIC-III and MIMIC-IV field semantics; no second branch was run.

## Feasibility and endpoint membership

- Preliminary cohort: **5,405**.
- At-risk cohort: **2,368**.
- Corrected primary events among all at-risk stays: **100**.
- Valid frozen exposures: **1,595** (67.4%).
- Modeled corrected primary events: **64**.
- Corrected later events among all at-risk stays: **96**.
- Modeled corrected later events: **63**.
- Primary endpoint IDs—original only: **4**; corrected only: **1**; shared: **99**; Jaccard: **0.952**.
- Shared endpoint completion times changed: **8**.
- Released sites: **1**. The prior coverage and site waivers are retained and explicitly noted; no new waiver was created.

## Locked effect estimates

- Original MIMIC-III primary: RR **0.969** (95% CI **0.761–1.234**), p=0.801.
- Original MIMIC-III later: RR **0.877** (95% CI **0.647–1.190**), p=0.400.
- Parity-corrected MIMIC-III primary: RR **0.925** (95% CI **0.721–1.186**), p=0.5385; n=1,595, events=64.
- Parity-corrected MIMIC-III later: RR **0.937** (95% CI **0.727–1.206**), p=0.611; n=1,595, events=63.
- MIMIC-IV frozen primary: RR **1.251** (95% CI **1.104–1.418**), p=0.00045.
- MIMIC-IV frozen later: RR **1.248** (95% CI **1.095–1.422**), p=0.00090.


## Formal comparison and precision

The corrected primary log RR was **-0.0780** with HC0 robust SE **0.1268**. Relative to MIMIC-IV, corrected-minus-MIMIC-IV log RR was **-0.3020** (SE **0.1419**), heterogeneity z=**-2.128**, p=**0.03337**, Q=**4.527**, and I²=**77.9%**. I² based on only two database-era estimates is unstable; no pooled estimate is used to claim replication.

The original-versus-corrected MIMIC-III log-RR change was **-0.0469**. This is descriptive only because the estimates use overlapping, dependent stays.

Approximate two-sided power from the corrected model's observed HC0 SE: RR 1.10: 11.7%, RR 1.15: 19.7%, RR 1.20: 30.1%, RR 1.25: 42.1%, RR 1.30: 54.4%. Approximate event requirements: 80% at RR 1.20: 243 events, 90% at RR 1.20: 326 events, 80% at RR 1.25: 163 events, 90% at RR 1.25: 218 events. Passing the 50-event gate is not evidence of adequate power.


## Interpretation

The parity-corrected analysis is post hoc with respect to the previously observed MIMIC-III non-confirmation. It was undertaken because a zero-association forensic audit identified objective implementation non-equivalence in the replication endpoint. It does not erase or replace the original locked replication result.

The historical original classification remains **MIMIC3_HISTORICAL_REPLICATION_DID_NOT_CONFIRM_POST_HOC_SIGNAL**. The forensic classification remains **IMPLEMENTATION_OR_MAPPING_CONCERN**. MIMIC-III and MIMIC-IV come from the same institution and may overlap, so this is a historical database-version replication—not independent external validation. Logical endpoint parity was restored as specified, but source-era endpoint observability is not fully identical. No alternative exposure, endpoint, availability branch, subgroup, threshold, covariate set, or time window was fitted.

Execution was **LOCAL using a standalone runtime**, not Google Colab.

## Required flags

- `oliguria_parity_corrected`: `true`
- `later_endpoint_parity_corrected`: `true`
- `exposure_mathematics_identical`: `true`
- `exposure_availability_parity_applied`: `true`
- `exposure_availability_parity_supportable`: `true`
- `endpoint_logically_equivalent_after_correction`: `true`
- `endpoint_observability_fully_equivalent`: `false`
- `corrected_primary_event_gate_passed`: `true`
- `corrected_later_event_gate_passed`: `true`
- `corrected_primary_association_fitted`: `true`
- `corrected_later_association_fitted`: `true`
- `original_replication_preserved`: `true`
- `forensic_classification_preserved`: `true`
- `outcome_peek_before_lock`: `false`
- `unauthorized_models_fitted`: `false`
