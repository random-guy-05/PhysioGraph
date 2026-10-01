# Finalization Analysis Lock

Locked: 2026-09-26 03:32:35 UTC  
Scope: post-result hardening of the existing corrected MIMIC-III analysis. This lock was written and SHA256-recorded before any new robustness estimates were examined or calculated.

This is a post-result hardening analysis. It does not convert any result into a preregistered finding. The previously fixed exposure, outcome definitions, threshold, horizons, and episode windows remain unchanged. No subgroup will be selected based on results. No covariate will be removed because it weakens an association. No estimator will replace the primary estimator because another estimate is more favorable. Every requested feasible robustness analysis, including null, unstable, unavailable, or unfavorable results, will be reported.

## Frozen scientific definitions

- Primary early SpO2 exposure: at least one absolute qualifying transition of 4 percentage points, among the already-defined dynamics-eligible first-four-hour SpO2 series.
- Landmark: ICU minute 240. Baseline lactate is the last valid linked lactate at or before minute 240; minute 240 remains baseline.
- Landmark primary outcome: last lactate minus baseline at least 0.5 mmol/L in (240,960] or (240,1680], respectively, and complete follow-up through the corresponding 12-hour or 24-hour horizon is required for events and non-events.
- Episode anchor: representative time of the second 15-minute bin completing the first qualifying absolute 4-point transition. Episode baseline is the last linked lactate strictly before the anchor.
- Fixed episode windows: acute (0,1], delayed (1,8], and cumulative (0,4], (0,8], (0,12] hours. Complete observable follow-up to each fixed window end is required for the last-value binary endpoint.
- Early events before censoring remain eligible only for the prespecified monotone maximum-rise and threshold-crossing companion endpoints.

## Locked hardening methods

1. Reproduce corrected episode and landmark results from saved tables; retain primary patient-cluster inference and save row-level before/after reconciliations.
2. Use Fisher exact p-values for the five fixed episode-window raw binary comparisons, applying Holm separately to the two primary windows, to the three cumulative windows, and across all five as the conservative sensitivity. Keep adjusted-model p-values separate.
3. Model episode-window outcome observation using only information known before each anchor, patient-grouped out-of-fold predictions, stabilized inverse-probability weights, and prespecified 1st/99th percentile truncation. Report untruncated and truncated weight diagnostics, ESS, positivity, and weighted RR/RD without causal interpretation.
4. Describe censoring by ICU departure, hospital discharge, death, any early end of observable follow-up, and complete-window probability. Use existing timestamps; do not create a replacement composite outcome.
5. Summarize pre-anchor lactate levels, serial changes, elapsed specimen interval, and trend. For the already-fixed delayed-window model retain continuous pre-anchor trend plus the trend-missing indicator; label low-EPV fits fragile and subset estimates descriptive.
6. Reassign control pseudo-anchors outcome-blind for seeds 1–100 while preserving the exposed empirical anchor-time distribution. Report the full assignment distribution; do not select a favorable seed. Label it design-assignment sensitivity, not a confidence interval.
7. Keep the existing no-episode control analysis primary. A post-hoc risk-set analysis may match at-risk controls at each exposed anchor time on the exact 15-minute bin, sampling regime, and care unit where supported; controls may have a later episode. Use patient-cluster uncertainty and report balance/support.
8. Keep the 4-point exposure fixed for absolute-oxygenation models, descriptive strata, and the fixed normoxemic subset (all first-four-hour bins at least 90%). Run only listed, already-prespecified signal sensitivities; no threshold search.
9. Audit ventilation item mapping before any re-analysis of respiratory-support adjustment. Preserve the prior zero result and only issue a documented source-mapping correction if objective codebook evidence demonstrates a defect. Do not claim respiratory independence where support is absent.
10. Use only pre-anchor variables for circulatory adjustment. Report hemodynamic balance and EPV; for synchronous MAP analysis use valid MAP 20–200 mmHg in both transition bins, no carry-forward, decline at least 10 mmHg, and require two jointly observed transitions before outcome modeling.
11. Retain drop and rise episode analyses separately; do not add them to the primary multiplicity family.
12. Report continuous lactate distributions without changing the binary primary endpoint. Quantify waveform-subset selection using pre-landmark variables and standardized mean differences; label the model outcome-blind.
13. Reconfirm corrected landmark IPW and the existing nested patient-grouped OOF predictive analysis. Preserve the negative incremental-prediction result.
14. Search for eICU data and validated cached artifacts. Run external replication only if adequate source tables exist and transfer definitions unchanged; otherwise create a precise source-requirements note. Do not use MIMIC-IV parity as substitute replication.
15. Produce claim grading, publication tables, figures, final report, and a self-audit. Every statistic must link to saved inputs/results; hash final outputs after completion.

No results will be used to change the definitions or methods above. Any branch lacking data or adequate support will be reported as unavailable or descriptive rather than rescued by relaxing its criteria.


## Pre-estimate specification clarification (2026-09-26 03:50:06 UTC)

The project’s frozen `docs/ANALYSIS_PLAN.md`, §10, specifies that the strict pre-episode lactate baseline must be within six hours before the anchor. The saved implementation required only a timestamp strictly before the anchor and therefore could retain older values. To honor the existing plan, all new episode re-estimates in this finalization will require the last plausible lactate in `(anchor−360 min, anchor)`. This is a correction to match the pre-existing episode definition; it does not change the SpO2 exposure, lactate threshold, outcome horizons/windows, or estimator hierarchy. The corrected follow-up rule remains in force.
