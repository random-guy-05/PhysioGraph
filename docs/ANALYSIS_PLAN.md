# Analysis Plan — spo2_protocol v2.3

Version 2.2 below remains the frozen prespecified analysis. Version 2.3 adds a
clearly labeled **post-result mechanistic amendment**; it does not retroactively
convert a result-informed analysis into a preregistered confirmatory test.

The original v1.0 plan was frozen 2026-08-29, **before** any full-data
MIMIC/eICU rebuild or model-performance inspection. Versions 2.0–2.2 record the
implementation corrections made 2026-08-30 after code review, bounded pilots,
and full-source schema/runtime discovery. The discovery model fits all failed
before producing valid performance estimates, so these amendments were frozen
before inspection of any valid uncapped model result.
Section 9 lists every amendment and its rationale.

Derived from PROJECT_GOAL.md and the implementation in
`src/physiograph/analysis/spo2_protocol.py` (protocol version `spo2_protocol_v2.2`).

## 1. Cohort eligibility, follow-up, and censoring policy

| Question | Frozen rule |
|---|---|
| Already on MCS at landmark (240 min) | Stay **retained** for other endpoints but excluded from the incident-MCS risk set; its MCS initiation flag is unavailable, not negative. The baseline MCS flag remains a clinical control. |
| Already on vasoactive support at landmark | Stay **retained**; VIS rise endpoint uses `vis_delta > 0` (post-landmark max VIS − baseline VIS), i.e., **escalation on top of baseline**, not initiation |
| Discharged or transferred out before outcome window ends | Endpoint **unavailable (NaN)** for that horizon, not a non-event — enforced by availability indicators (`*_observed` columns) |
| Death before outcome ascertainment | Non-death endpoints are **unavailable (NaN)** beyond death, never silently negative. Death is extracted separately and is not folded into the primary 12/24-hour organ-injury endpoints. |
| Repeat ICU stays per patient | One first ICU anchor per hospital encounter is retained (MIMIC admission; eICU health-system stay). Repeat hospitalizations remain eligible; `person_id`-qualified grouping keeps all encounters of a person inside one CV fold and bootstrap replicate. |
| Observation-window SpO2 data | Stays with zero plausible SpO2 values in [0, 240) are excluded from modeling (`spo2_plausible_count > 0` filter); count and fraction excluded reported in the cohort flow |

Endpoint windows are strictly `(240, 240 + horizon]` minutes; a measurement exactly
at the landmark belongs to baseline, never to the outcome.

## 2. Primary endpoint definitions (locked)

- **Lactate rise:** last lactate observed inside the fixed post-landmark window −
  last pre-landmark lactate ≥ **0.5 mmol/L**. This last-value primary endpoint
  requires complete 12/24-hour follow-up for both events and non-events. The
  monotone maximum-rise and threshold-crossing companions retain a confirmed
  event observed before death/discharge, while their negatives still require
  complete follow-up.
  Rationale: ≈ the upper bound of routine analytic/biologic variation for serial
  lactate in this range, so the flag represents a rise unlikely to be measurement
  noise while remaining sensitive in a 12–24 h window. The continuous delta
  (`lactate_delta_*h`), max-value delta, and the ≥2.0 mmol/L crossing flag are
  retained as prespecified companions, not post-hoc alternatives.
- **VIS rise:** any increase > 0 in the drug-weighted vasoactive-inotropic score
  (coefficients: dopamine 1, dobutamine 1, milrinone 10, phenylephrine 10,
  epinephrine 100, norepinephrine 100, vasopressin 10,000) over the pre-landmark
  baseline within the window.

## 3. Prespecified sensitivity analyses (run after the primary analysis, all listed before results)

| # | Axis | Levels |
|---|---|---|
| S1 | Outcome horizon | 12 h vs 24 h (already parallel in protocol) |
| S2 | Lactate threshold | Δ ≥ 0.5 (primary); Δ ≥ 1.0; max-delta variant; continuous delta (Spearman rank association with patient-cluster bootstrap 95% CI) |
| S3 | Sampling density | Above-median vs below-median `spo2_sampling_density_per_hr` strata |
| S4 | Mechanical ventilation documentation | any qualifying documentation in `[0, 240)` vs none (not exact instantaneous state at minute 240) |
| S5 | Respiratory support / FiO2 | any qualifying first-4-hour support documentation vs none; FiO2 strata where extractable |
| S6 | Baseline hypoxemia | below-90 fraction > 0 vs = 0 (normoxemic) strata |
| S7 | Baseline lactate | elevated (> 2.0 mmol/L) vs lower |
| S8 | Vasoactive status at landmark | on vasoactives vs off |
| S9 | Artifact filter | current plausibility (50–100%) vs stricter (>70%) vs raw |
| S10 | Instability definition | jump-threshold ≥3 vs ≥5 pp; sustained (≥2 consecutive) jump/low events; RMSSD-based signature |
| S11 | Repeated stays | all stays (primary) vs first stay per patient only |
| S12 | Measurement-intensity control | missingness/sampling-only model vs physiological model comparison |

Not prespecified (report only as exploratory, if at all): any subgroup not listed
above; any threshold not listed in S2/S10.

## 4. Model specification rules (locked before fitting)

- Instability feature block is collinear by construction (SD/RMSSD/IQR/MAD/range/
  jump-rate are correlated summaries of one signal). The prespecified primary
  instability model is the dynamics-only parsimonious signature (RMSSD +
  abrupt-jump fraction + count of ≥3-point drops). The full instability block is reported as a
  deliberately broader, fragility-flagged companion.
- The incremental reference is clinical context + absolute SpO2 (mean, minimum,
  below-90 fraction) + sampling/missingness (measurements per hour, missing
  15-minute bins, longest gap). Thus dynamics never receive credit for monitoring
  intensity or absolute hypoxemia already available to the reference model.
- Regularization: L2 (C = 1.0) fixed; `max_iter = 3000`.
- Events-per-variable: report events, feature count after one-hot expansion, and
  events-per-feature for every fitted model; flag any model below 10 EPV as fragile.
  A fragile model may be displayed diagnostically but cannot create a suggestive
  or robust endpoint classification.
- Calibration: report ECE, calibration intercept/slope with the same OOF predictions;
  calibration uncertainty via the existing patient-level bootstrap.
- MIMIC-to-eICU transportability uses only harmonized clinical controls. Raw
  site, year, unit-type/source, and race-category fields are excluded because
  their coding systems are not commensurate across databases; controls entirely
  missing in either MIMIC or eICU are also excluded.
- A transportability result supports only a claim for its training dataset.
  MIMIC-to-eICU validation cannot be reused as validation of an eICU-trained
  model; the reverse direction would require a separately fitted analysis.

## 5. Missing-data handling (locked)

- Fold-local median/mode imputation with missingness indicators (already enforced
  in code) — no global fitting.
- Report per-feature missingness tables for MIMIC and eICU separately.
- Confounder availability denominators (baseline lactate, creatinine, MAP, resp
  rate, FiO2, vent, RRT, demographics) are published before adjusted-model
  interpretation; an adjusted model run on a heavily selected complete subset is
  labeled as such.
- The measurement-intensity control (S12) is interpreted alongside every headline
  result: if sampling-only features approach physiological-model performance,
  the instability association is reported as likely confounded by measurement intensity.

## 6. Reporting package (frozen list)

Cohort flow diagram; Table 1 per dataset; endpoint availability/event-rate table
(the completeness audit); SpO2 feature distributions; BH-corrected association
table; nested model comparison; ΔAUROC/ΔAUPRC with CIs; calibration plots;
ROC/PR curves for adequate endpoints; lead-time distribution (full distribution,
not median only); missingness-control comparison; MIMIC→eICU transportability
table; sensitivity matrix S1–S12; per-endpoint conclusion classification
(robust / suggestive / fragile / null / unavailable). An inferential row is
never labeled estimated unless its point estimate and required confidence
interval are finite; zero-cell and degenerate designs are explicit
non-estimable/underpowered rows and are excluded from multiplicity correction.
Predictive claim grading additionally requires at least 10 events per
transformed feature and dataset-matched external-training provenance.

## 7. Claim scope

- Temporal-ordering outputs are **landmark ordering, not causal precedence**.
- No causal language about SpO2 instability independent of respiratory failure
  without the prespecified respiratory-support adjustments (S4–S6).
- eICU VIS: quantitative VIS is unavailable because infusion rates cannot be
  harmonized to validated dose units. External VIS validation is therefore
  excluded; pressor presence is never substituted for quantitative VIS.
- eICU MCS is the first post-landmark device-treatment documentation, not a
  verified device start time. MIMIC MCS uses a matching line/procedure start
  when available and otherwise the first positive operational device charting
  (flow, speed, sweep, or setting) for IABP/Impella/ECMO/LVAD/RVAD; both are
  source-documentation proxies rather than adjudicated clinical initiation.
- Urine-output decline is a proxy and requires at least 50% coverage of two-hour
  bins in both baseline and outcome windows. KDIGO-style oliguria is additionally reported
  only where weight and interval coverage are sufficient. True documented zeros
  are retained. The mounted MIMIC
  source has no `outputevents.csv`, so MIMIC urine endpoints are explicitly
  unavailable; eICU urine output is extracted from `intakeOutput.csv`.

## 8. Model-minimal (epidemiological) companion analyses — frozen 2026-08-29

Frozen before the full-data rebuild completed, so no full-data result was visible
when these definitions were locked. All estimates are model-free; the incremental
logistic models are NOT involved. Implemented in
`src/physiograph/analysis/spo2_epidemiology.py` (outputs under
`spo2_drilldown/`).

**Exposure (prespecified, matching the protocol temporal-precedence definition):**
- `exposure_any` = dynamics-only: >=1 adjacent >=4 percentage-point SpO2 change
  in [0, 240) minutes after same-time deduplication and 15-minute binning;
  transition pairs separated by more than 30 minutes are not treated as adjacent,
  and the signal must meet the >=3-bin/>=2-transition eligibility rule.
- `exposure_hypoxemia_or_dynamics` = SpO2 <90% OR the dynamics exposure. It is
  descriptive/sensitivity-only and cannot support a claim beyond absolute SpO2.
- `exposure_tertile` = within-dataset tertiles of the dynamics-only score
  (RMSSD + abrupt transitions + drops; T1<T2<T3), with identical scores kept in
  the same category even if this makes tertiles unavailable. The combined instability/
  hypoxemia score remains descriptive and is not used to claim incremental
  value beyond absolute SpO2.

**A. Stratified risk tables (cohort level).** Per dataset x endpoint:
risk in exposed vs unexposed, risk ratio (Katz log CI), risk difference (Wald CI),
Fisher exact p, and a person-level cluster bootstrap CI for the RR (2000 reps,
patient clusters — repeated stays never split). Floors from §1 audit apply:
endpoint rows with no jointly observed exposure/outcome values are unavailable;
observed rows below floors are labeled underpowered. Neither is interpreted.

**B. Confounder-stratified pooled estimates.** Mantel-Haenszel pooled odds ratio
(stratified by baseline hypoxemia: `spo2_below_90_fraction > 0` vs = 0; and by
respiratory support where available) with Greenland-Robins CI — tests whether the
exposure-outcome association survives within absolute-SpO2 strata, which is the
"beyond how low it went" question without a model.

**C. Dose-response.** Outcome risk by exposure tertile with a two-sided
Cochran-Armitage trend test — a graded monotone relationship is the strongest
model-free evidence of a real signal (Bradford Hill).

**D. Paired within-patient temporal precedence.** For every stay with both an
instability onset and an outcome onset: lead-time distribution, fraction of
outcome events **preceded** by instability, a descriptive sign test vs 0.5, and median lead
time with patient-cluster bootstrap CI. Complement reported: stays with an
outcome but NO prior instability (guards against instability being ubiquitous).
The 0.5 null is not scientifically justified, so its p-value is never used to
claim causal or inferential precedence; pre-landmark ordering is enforced by design.

**E. Specificity matrix.** Every endpoint classified
(positive_robust / positive_suggestive / null / underpowered / unavailable) using
BH-corrected Fisher p across the endpoint family, with
`hepatic_lab_worsening` prespecified as a **specificity comparator**, not a true
negative control: hypoxemia can plausibly accompany hepatic injury. Instability
is more endpoint-specific only if acute endpoints are robust while this comparator
is null; a non-null comparator does not establish or refute causality.

Claim scope: these are observational associations with temporal ordering; the
same non-causal language rules as §7 apply.

**Continuous trajectory companion.** In addition to binary thresholds, the
analysis reports patient-cluster-bootstrap partial Spearman associations between
the dynamics-only score and worsening-oriented lactate, VIS, urine-output,
creatinine, hepatic, platelet, and assay-matched troponin trajectories. Both the
exposure and trajectory are rank-residualized against absolute SpO2 summaries,
sampling density, missing-bin count, and longest monitoring gap. These estimates are descriptive, not causal effects or
validated prediction metrics.

## 9. Versions 2.0–2.2 amendments recorded before valid uncapped-result inspection

1. Enforced the exact landmark clock: predictors use `[0, 240)` and outcomes use
   `(240, 240 + horizon]`; the minute-240 value is baseline.
2. Required at least three valid 15-minute SpO2 bins and two gap-qualified
   transitions for dynamics models. Singleton measurements no longer imply zero
   variability.
3. Changed primary modeling from pooled-first to MIMIC/eICU separately first;
   pooled grouped-CV estimates are secondary, and MIMIC-to-eICU evaluation is a
   distinct transportability analysis.
4. Added death/discharge censoring, endpoint-source availability, conservative
   missing composites, and baseline-MCS/vasoactive retention.
5. Reclassified bilirubin from “negative control” to “specificity comparator,”
   and made temporal-precedence tests descriptive because landmark ordering is
   imposed by design.
6. Added all outcome events without prior instability to precedence denominators;
   the earlier paired-only implementation could not estimate this complement.
7. Expanded organ-injury outcomes to creatinine, bilirubin, AST/ALT, INR, and
   platelets; added lactate threshold/max/crossing variants and all S1–S12 axes.
8. Declared quantitative eICU VIS unavailable rather than treating
   unstandardized infusion rates as zero, and labeled eICU MCS timing as a
   documentation proxy.
9. Added source/cache fingerprints, fatal full-run SpO2 invariants, cohort flow,
   Table 1, missingness tables, OOF curves, EPV/fragility labels, patient-cluster
   bootstrap intervals, and endpoint conclusion grading.
10. Separated the dynamics-only score from the combined instability/hypoxemia
    proxy so the incremental model cannot reintroduce absolute low-SpO2 burden.
11. Expanded MIMIC-IV laboratory discovery through the mounted `d_labitems.csv`
    dictionary, including current-release parallel item IDs for lactate,
    creatinine, BUN, AST/ALT, bilirubin, troponin I, platelets, and INR; added
    Fahrenheit-to-Celsius and pound-to-kilogram normalization.
12. Enforced one first eICU unit per hospital encounter, while preserving repeat
    hospitalizations and robust person/encounter identifiers for grouping.
13. Split last-value fixed-window censoring from monotone event ascertainment:
    negatives require full follow-up, while a confirmed event before censoring is
    retained. Death is a separate 12/24-hour endpoint.
14. Prevented troponin T and troponin I values from ever being ratioed across
    assays; assay-specific rises are combined only at the final event-flag level.
15. Required adequate two-hour-bin coverage for urine-output decline/oliguria,
    retained true zeros, and introduced a five-minute MIMIC infusion restart
    grace so documentation fragments do not create false pressor initiations.
16. Treated every non-numeric pandas dtype (including the current string dtype)
    as categorical inside each training fold. The earlier schema-discovery model
    attempts failed at preprocessing and are explicitly excluded from results.
17. Required a known follow-up timestamp to establish any fixed-window negative,
    and removed prevalent MCS cases from the incident-MCS risk set while retaining
    those stays for all other eligible endpoints.
18. Separated the model-free dynamics exposure from absolute hypoxemia, preserved
    tied dynamics scores in dose-response categories, used patient-cluster
    bootstrap RR intervals without hybridizing them with Katz limits, and kept
    missing respiratory status missing in stratified analyses.
19. Restricted external validation to controls observed in MIMIC and harmonized
    with eICU, preventing deidentified calendar years or database-specific site,
    unit, and race categories from driving cross-database predictions.
20. Added worsening-oriented continuous trajectory associations so the study
    tests signal change, not only thresholded endpoints, with patient-cluster
    uncertainty and explicit adjustment for absolute SpO2 and sampling density.
21. Added mounted eICU aliases for `platelets x 1000`, `pt - inr`, hyphenated
    troponin I/T, and vasoactive trade names; these high-volume source rows had
    previously failed exact-name/token matching.
22. Corrected eICU hospital death/discharge timing: `hospitalDischargeOffset`
    is already relative to unit admission, so the negative hospital-admit offset
    must not be subtracted a second time.
23. Routed the public MIMIC/eICU cohort builders through the production cohort
    implementations, eliminating legacy admission-ID anchors, prevalent-support
    exclusion, pediatric thresholds, and encounter-duplication drift.
24. Made protocol early outcomes authoritative during analysis-frame
    assembly; the legacy uncensored `mcs_24h_flag` can no longer overwrite the
    incident, censoring-aware 24-hour MCS endpoint.
25. Excluded eICU device removal/explant/discontinuation strings from MCS
    initiation and corrected concatenated `intraaortic` device-family parsing.
26. Required an incident crossing for absolute AST/ALT and platelet thresholds;
    unchanged abnormalities already present at the landmark no longer count as
    new organ injury, while prespecified relative worsening remains positive.
27. Made urine coverage bins right-closed at exact two-hour boundaries and
    required protocol-level baseline/post coverage before assigning a urine
    decline onset in the temporal-ordering analysis.
28. Refit ranks and nuisance adjustment inside every patient-cluster bootstrap
    replicate for continuous trajectories instead of reusing full-sample
    residuals.
29. Added MIMIC whole-blood creatinine (`itemid=52024`, mg/dL) and operational
    LVAD/RVAD/IABP/Impella/ECMO chart documentation through the outcome window;
    inactive/removal and weak maintenance rows are excluded.
30. Strengthened robust endpoint grading: both incremental AUROC and AUPRC
    bootstrap intervals must exclude zero, and the patient-cluster RR interval
    (not an unclustered fallback) must exclude one.
31. Treated eICU `treatment.csv` as a respiratory-context source when structured
    respiratory charting is unavailable and made missing person IDs fall back
    to distinct stay-level bootstrap groups.
32. Excluded eICU `Urine Count`, `Urine Occurrence`, incontinence-event, and
    mixed urine/stool rows from quantitative urine volume; these are event
    counts or contaminated volumes despite living under an output-mL path.
33. Removed the combined hypoxemia/instability proxy from inferential per-feature
    and variability model lists. It remains descriptive only; every incremental
    dynamics claim uses features that do not contain absolute low-SpO2 burden.
34. Added sampling density, missing-bin count, and longest gap directly to the
    absolute-SpO2 reference in every internal and external incremental model;
    S12 remains a separate measurement-intensity-only diagnostic.
35. Replaced legacy 48–168-hour targets in descriptive, association, trajectory,
    and figure outputs with all prespecified 12/24-hour endpoints; all summaries
    are database-specific first, and trajectories use one median per stay/bin.
36. Aligned temporal urine onset with the fixed-window endpoint: adequate
    coverage and an aggregate rate decline are required before a low two-hour
    interval can define the onset.
37. Increased primary grouped-CV bootstrap repetitions to 500 and barred a
    robust classification unless at least 80% of requested internal and external
    patient-cluster resamples are valid.
38. Made the clinically eligible cohort index authoritative during frame
    assembly, preventing excluded stays from re-entering through stale feature
    or label keys; cohort identity, demographics, censoring, and source metadata
    now also fill any row-level artifact gaps.
39. After the uncapped-result status audit, corrected a reporting gate that
    rejected an otherwise finite focal OR/interaction whenever a sparse nuisance
    category alone had undefined robust inference. Required focal inference
    remains fail-closed; affected nuisance terms are now emitted as warnings.
    This correction changes estimability/status reporting for secondary OR and
    respiratory diagnostics, not endpoint definitions, exposure definitions,
    predictive models, or the prespecified robust-conclusion rule.

## 10. Version 2.3 post-result mechanistic lactate amendment

### Why this amendment is necessary

The fixed landmark asks whether a four-hour summary predicts a last or maximum
lactate over the next 12/24 hours. It does **not** directly test whether a
specific instability episode is followed by the next lactate rise. Four masks
are therefore separated explicitly:

1. A lactate near the end of the first four hours could be downstream of an
   earlier SpO2 episode yet be reused as “baseline.” The revised baseline is the
   last lactate **strictly before** the episode, within six hours. A lactate at
   the same timestamp as the episode is excluded from baseline.
2. A last-value endpoint can miss a treated transient peak. The first and
   maximum post-episode lactates are reported separately, as continuous deltas
   and ≥0.5/≥1.0 mmol/L rises.
3. A 12/24-hour window can dilute short-lag physiology. Acute `(0,1]` and
   delayed `(1,8]` hours are the focused temporal family. Non-overlapping
   `(1,2]`, `(2,4]`, and `(4,8]` bins localize delayed kinetics; `(8,12]` and
   cumulative `(0,8]` are companions.
4. Absolute change combines deterioration (downward desaturation) with recovery
   (upward reoxygenation). The original ≥4-point absolute jump remains the
   revised primary reconstruction, while ≥3/≥5-point drops and ≥4-point upward
   recovery are direction-specific sensitivity/comparator analyses.

### Episode construction and comparator

- SpO2 is restricted to 50–100% and `[0,240)` minutes. Same-time duplicates are
  median collapsed. The main reconstruction uses 15-minute medians and only
  adjacent transitions 0–30 minutes apart. A same-time-deduplicated raw-chart
  sensitivity quantifies masking by 15-minute binning.
- A directly observed episode requires one gap-qualified transition; the older
  ≥3-bin/≥2-transition rule remains a **strict signal-estimation sensitivity**.
  It is not imposed on the main episode analysis because an observed transition
  is the estimand, not a four-hour variability summary.
- Each exposed stay is anchored at its first qualifying episode. No-episode
  stays receive deterministic target times spanning quantiles of the exposed
  anchor-time distribution and use their nearest observed transition. This
  avoids collapsing all controls onto one median time while preserving a
  transparent no-episode definition.
- The within-episode paired change and the episode/no-episode risk contrast are
  reported separately. Neither is causal. The probability of having a strict
  prior and post-episode lactate is itself an output, because informative lab
  ordering can bias both directions.

### Focused family, safeguards, and evidence grades

- The focused family contains exactly two within-dataset tests: first-next
  lactate after a binned ≥4-point absolute episode in acute `(0,1]` and delayed
  `(1,8]` windows. BH correction is applied across those two tests. The three
  delayed localization bins have their own secondary BH family. Global
  exploratory q-values across every direction/resolution/companion analysis are
  retained so focused correction cannot hide search breadth.
- Parsimonious modified-Poisson adjustment uses only pre-episode information:
  strict prior lactate, episode time, absolute SpO2, pre-anchor sampling density,
  age, shock, and baseline vasoactive status where available. Adjusted inference
  fails closed below five events per fitted parameter and is labeled fragile
  from five to ten.
- Because lactate remeasurement is informative, a separate inverse-probability
  sensitivity models post-episode lactate observation among stays with a strict
  pre-episode lactate, uses stabilized 1st/99th-percentile-truncated weights,
  clips observation probabilities only at 0.02/0.98 with clipping counts and a
  positivity flag exposed, and refits the parsimonious RR. Selection and outcome models both fail closed
  below five events per parameter. This is a missing-outcome sensitivity, not a
  guarantee that measurement bias has been removed.
- Shock, vasoactive status, strict prior lactate, absolute hypoxemia, and
  sampling-density strata are heterogeneity analyses, not independent proof.
- Harmonized MIMIC/eICU risk ratios are combined with fixed and random effects,
  Cochran Q, I², and τ². Requiring each database to be individually significant
  is not the replication rule; direction, uncertainty, and heterogeneity are all
  shown.
- Evidence is graded separately as paired-change support, controlled-association
  support, and cross-dataset support. A single binary “robust” gate no longer
  turns a suggestive component into a null, but no grade can erase uncertainty.

These reporting choices follow the separation of model development, validation,
and bias assessment emphasized by [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378)
and [PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505), with observational
reporting mapped to [STROBE](https://www.strobe-statement.org/). The implemented tables are produced by
`src/physiograph/analysis/spo2_lactate_mechanistic.py`.

## 11. Version 2.3 post-result all-endpoint episode amendment

The episode estimand in §10 is applied to every endpoint rather than only
lactate. This is an explicitly exploratory, post-result amendment. Its complete
endpoint/window grid was frozen before the first all-endpoint fit; results are
reported whether positive, null, underpowered, or source-unavailable.

### Frozen timing families

- The exposure remains the first qualifying SpO2 transition during ICU hours
  0–4, with time-aligned no-episode controls. The primary binned absolute
  ≥4-point definition and all five directional/resolution definitions in §10
  are carried to every endpoint.
- The primary definition is evaluated in non-overlapping `(0,1]`, `(1,4]`,
  `(4,8]`, `(8,12]`, and `(12,24]` hour localization windows and cumulative
  `(1,8]`, `(0,4]`, `(0,8]`, `(0,12]`, `(0,24]`, `(8,24]`, and `(4,12]`
  companions. Sensitivity definitions use the two frozen focused windows for
  each endpoint family.
- Focused pairs are `(0,1]`/`(1,8]` for lactate; `(0,8]`/`(8,24]` for organ
  labs, troponin, and hepatic/organ composites; `(0,4]`/`(4,12]` for VIS,
  pressors, MCS, RRT, ventilation, urine, and early composites; and
  `(0,8]`/`(8,24]` for death and the broad clinical composite.

### Endpoint safeguards

- Lab baselines are the last plausible value strictly before the episode within
  six hours. First-next and worst-in-window variants are both retained; the
  protocol-selected variant is declared per endpoint. A confirmed early event
  remains observed, while a worst-in-window non-event requires follow-up through
  the end of that window.
- Creatinine, bilirubin, AST, ALT, INR, platelet, and lactate changes preserve
  their clinical direction. Troponin T and I are compared only within assay and
  are labeled myocardial injury rather than adjudicated infarction.
- Pressor, MCS, RRT, invasive-ventilation, and death outcomes use incident
  window-specific risk sets. A support event before a delayed window removes
  that stay from risk for a new onset in that window. VIS similarly tests a new
  maximum above the strict pre-episode maximum and excludes escalation that
  already started before a delayed window.
- Urine is tested both as a coverage-qualified `<0.5 mL/kg/h` KDIGO proxy and
  as a ≥50% decline from a coverage-qualified strict pre-episode rate. Source
  absence remains unavailable, never zero or negative.
- Composite non-events require all source-available components to be eligible
  and observed; any observed component event is positive. Event-saturated or
  poorly observed composites are labeled underpowered and cannot become a
  headline by construction.

### Avoiding both hidden signal loss and false-positive search

- Every focused endpoint has unadjusted continuous and threshold contrasts,
  parsimonious modified-Poisson adjustment, an observation-process audit, and
  inverse-probability-of-observation weighted modified-Poisson/GEE sensitivity
  when estimable. Covariate priority and information thresholds are fixed and
  never selected from an endpoint p-value.
- Adjusted and weighted claims fail closed below five events per parameter and
  are claim-ready only at ten or more. Patient identity is the clustering unit.
- BH q-values are retained at four scopes: within endpoint, within biological
  family, across all focused endpoints in a dataset, and globally across every
  exploratory definition/window. A within-endpoint result is therefore visible
  even when it does not survive the much broader cross-endpoint family.
- A result is labeled fully robust only when unadjusted, adjusted, and weighted
  estimates are positive, claim-ready, and each survives the cross-endpoint
  correction. Cross-dataset random-effects evidence is labeled separately and
  never turns one database row into an independently replicated result.
- Recovery-rise comparators diagnose whether a result is specific to downward
  desaturation. Positive recovery comparators are reported as a monitoring,
  treatment-response, or severity caution rather than hidden.

Implemented by
`src/physiograph/analysis/spo2_multiorgan_mechanistic.py`; the fast cached-data
entry point is `scripts/refresh_multiorgan_episode_analysis.py`.

## 12. Version 2.4 advanced episode-inference amendment

This is a post-result exploratory robustness amendment. It does not replace the
frozen v2.3 contrasts and cannot upgrade them to confirmatory or causal
evidence. Its purpose is to distinguish a repeatable episode association from
confounding, informative endpoint measurement, poor overlap, thresholding, and
single-center dominance without choosing a model from the observed p-values.

### Strict temporal design and nuisance covariates

- One row is retained per primary binned ≥4-point episode/control anchor. Every
  adjustment variable used in the exposure model occurs strictly before the
  anchor: prior SpO2 level and distribution, heart rate, MAP/SBP, respiratory
  rate, FiO2, lactate, pH, creatinine, temperature, support state, sampling
  density, demographics, anchor time, unit, and site.
- Concurrent anchor SpO2 is excluded from propensity estimation and balance
  assessment because it partly defines the exposure. It is included only in
  endpoint-observation and outcome regressions to ask whether a transition adds
  information beyond the absolute SpO2 at that instant.
- Exposure-overlap weights target the overlap population and are fixed before
  inspecting any endpoint. All encoded covariates and missingness indicators
  are reported before and after weighting using absolute standardized mean
  differences.

### Estimators and safeguards

- Binary outcomes use a five-fold patient-grouped, cross-fitted bounded TMLE as
  the primary doubly robust estimator. The one-step AIPW estimate is retained
  as a diagnostic even when it falls outside the probability bounds.
  Endpoint-observation probabilities are modeled cross-fitted and enter both
  estimators under a missing-at-random assumption.
- Graded lab and urine changes use patient-grouped cross-fitted AIPW mean
  differences in their clinical units, so a binary threshold cannot be the
  only route by which a biological change is detected.
- Observation probabilities are lower-truncated at 0.02 but never
  upper-truncated: a probability of 1 is valid for a completely observed
  endpoint. Exposure probabilities are truncated at 0.02/0.98.
- A positive estimate is claim-ready only when each exposure arm has at least
  ten events and ten non-events (or 30 continuous measurements), no more than
  10% of rows violate the exposure-propensity bounds, no more than 10% fall
  below the observation-probability floor, effective sample size is at least 50
  per arm, and TMLE did not hit its targeting boundary. Estimates failing a
  gate remain visible but cannot receive a positive evidence label.
- eICU center robustness uses hospital-specific risk ratios, Paule–Mandel
  heterogeneity, modified Hartung–Knapp–Sidik–Jonkman intervals, and
  leave-one-hospital-out estimates. These site estimates are deliberately
  labeled unadjusted and do not substitute for the overlap or doubly robust
  estimators. MIMIC is not treated as multicenter in this layer.
- BH correction is applied within endpoint and across every focused
  endpoint/window separately for each estimator family. E-values are reported
  with explicit information/weight-support eligibility; they quantify the
  strength of residual confounding needed to explain an association and do not
  prove exchangeability.

Implemented by
`src/physiograph/analysis/spo2_advanced_inference.py`; the cached-data-only entry
point is `scripts/refresh_advanced_episode_inference.py`.

## 13. Locked eICU-to-MIMIC endpoint validation

This post-discovery external-validation analysis freezes the eICU exposure and
lag before inspecting MIMIC outcomes: the first gap-qualified absolute SpO2
jump of at least four points in 15-minute median bins and incident ventilation
4–12 hours later. It is observational and not prospective preregistration.

- Both databases use one fail-closed phenotype: age ≥18, explicit heart
  failure, and ICU admission within 24 hours of hospital admission or
  cardiogenic shock. Shock, cardiomyopathy, and acute MI never substitute for
  HF. Diagnosis flags are propagated across the hospital encounter and one
  first ICU anchor is retained.
- The transferred primary MIMIC endpoint is direct `procedureevents` item
  225792, Invasive Ventilation, timed at interval start. Direct item 224385,
  Intubation, is a prespecified respiratory-specificity endpoint.
- The eICU endpoint is first treatment documentation containing a literal
  pipe-delimited `mechanical ventilation` segment. NIV, ventilator-weaning, and
  nonliteral substring matches—including `simvastatin` matching `simv`—are
  removed and counted in a dedicated audit. Its time is a documentation proxy,
  not a guaranteed support-initiation time.
- Intervention risk sets exclude any earlier qualifying documentation. Missing
  source/follow-up remains unavailable rather than a non-event.
- The direct ventilation endpoint is the single transferred primary. Secondary
  clinical-escalation, organ-injury, and exploratory composite tiers each use
  their own BH family; they cannot redefine primary replication success.
- Conventional modified-Poisson, observation-weighted, overlap-weighted,
  cross-fitted TMLE/AIPW, continuous AIPW, multicenter, and E-value outputs are
  regenerated for every dataset-endpoint pair under the same locked window.

Implemented by `harmonize_hf_cohort`,
`extract_mimic_respiratory_procedure_events`, and
`run_locked_external_replication`. The fast analysis-only entry point is
`scripts/refresh_locked_external_validation.py`.

## 14. Prespecified head-to-head biomarker benchmark

This comparator layer asks whether the complete early SpO2 signal is more
effective than, or adds prognostic value to, SBP, lactate, and SCAI shock
severity for every registered 12/24-hour endpoint. It is prognostic rather than
causal and does not replace the episode-anchored estimands.

- Every predictor is frozen in `[0,240)` minutes. Outcomes retain the existing
  post-landmark 12/24-hour clock; a value at minute 240 cannot enter a
  predictor.
- The harmonized explicit-HF cohort is primary. Shock-code-positive stays are
  a prespecified association sensitivity, not a second opportunity to select a
  favorable predictive result.
- All 11 models use one outcome-specific SpO2-dynamics-eligible patient set,
  identical `StratifiedGroupKFold` assignments, and fold-local preprocessing.
  Paired AUROC, AUPRC, and Brier differences use 300 joint patient-cluster
  bootstrap draws.
- The standalone SpO2-signal specification includes absolute level,
  sampling/missingness, and parsimonious instability. A nested contrast tests
  the instability block beyond level and observation process. Additional
  contrasts test SpO2 beyond SBP+lactate, Kapur-SCAI, and all conventional
  markers.
- SBP includes first/last/mean/minimum, SD, RMSSD, slope, hypotension fraction,
  and count. Lactate includes first/last/maximum, delta, slope, elevated
  fraction, and count. This prevents an artificially weak single-value
  comparator.
- Kapur-CSWG SCAI uses the first hemodynamic/perfusion values within four hours
  and active treatment at minute 240, with a worst-four-hour sensitivity.
  Thresholds follow Kapur et al., JACC 2022
  (doi:10.1016/j.jacc.2022.04.049). It is explicitly labeled an EHR
  operationalization. OHCA and physical examination are unavailable; missing
  normal components never become stage A, and all classified stages retain a
  lower-bound flag.
- Marker-specific and mutually adjusted modified-Poisson models report RRs per
  database-specific 1-SD worsening with patient-clustered inference and BH
  correction within dataset, population, model, and endpoint tier.
- Frozen models are transported in both MIMIC-to-eICU and eICU-to-MIMIC
  directions. Decision curves span thresholds 0.01–0.50. Coverage, component
  availability, events per feature, calibration, and eICU cuff-BP source
  absence are emitted as claim-limiting audits.

Implemented by
`src/physiograph/analysis/biomarker_benchmark.py`. The cached-data-only entry
point is `scripts/refresh_biomarker_benchmark.py`, and the clean notebook
renders every benchmark table.
