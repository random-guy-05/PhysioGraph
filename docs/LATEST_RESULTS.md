# Latest PhysioGraph Results

## Latest respiratory investigation: event-aligned capnography

Full eICU vitalPeriodic and MIMIC chartevents metadata extraction completed
locally. An interruption occurred during source hashing after both raw tables
were committed; the stopped process was verified absent, and hashing and
pairing resumed from those tables. Both native hashes match earlier independent
extractions. No EtCO2, minute-volume or mortality amplitudes were selected.

Among the original first episodes, five-minute two-sided EtCO2 timing support
was 43 eICU people (23 upward/20 downward SpO2 events, 25 hospitals) and five
MIMIC people (two upward/three downward events). Both fail the frozen minimum
of 50 people and 20 per direction per database. eICU respiratory-rate fields
were present at both selected times in 40; MIMIC minute-volume timestamps were
available on both sides in one. Neither establishes stable minute ventilation.

The 15-minute sensitivity has 30/three pairs, with fewer eligible events because
it requires larger complete windows inside minutes 0–240. It does not rescue
the primary gate. Independent SQL/Python episode signs, nearest timestamps,
forward ordering and context flags agree. See
`SPO2_CAPNOGRAPHY_FEASIBILITY_RESULTS.md` and the frozen plan. This is inadequate
temporal support for the specified contrast, not a negative biological result.
The mortality-relevant discovery goal remains active and unfulfilled.

## Latest acute biological experiment: coupled potassium fall and glucose rise

The frozen episode-anchored experiment completed locally. The first original
SpO2 episode and deterministic no-episode anchors were independently reproduced.
Full native laboratory scans and exact-decimal qualification yielded 152 eICU
people across 61 hospitals and 134 MIMIC people with paired potassium/glucose
panels. The last potassium draw in the preceding three hours and first draw
within two hours after the anchor were fixed before examining values; MIMIC
required potassium and glucose to share a specimen ID at each time.

The joint event required potassium decrease ≥0.5 mEq/L **and** glucose increase
≥20 mg/dL **and** ≥20%. These are investigator-selected substantial-change
thresholds, not a validated syndrome or a definition of hypokalemia.

| Dataset | Exposed joint events | Control joint events | Risk difference, pp |
|---|---|---|---:|
| eICU | 0/56 (0%) | 4/96 (4.17%) | −4.167 |
| MIMIC | 1/37 (2.70%) | 3/97 (3.09%) | −0.390 |

Both point estimates run against the proposed increase. The fixed event-count
floor fails, so no bootstrap intervals, p-values, adjusted models or mortality
associations were calculated. This is not evidence of protection or equivalence.
The eight component-threshold combinations and continuous changes are retained
descriptively; they cannot substitute for the failed joint-event criterion.

Independent SQL/Python exposure, anchors, unit/entry/decimal flags, selected
panels, exact event labels and risk-difference checks passed, including 634
missing-hospital candidate assignments. The source contains 174 MIMIC potassium
rows with a hemolysis-related text mention; the frozen conservative rule rejects
such latest entries. No mention in eICU does not establish absence of hemolysis.
One SQL alias error stopped event creation; the syntax-only correction and
unchanged cache-based rerun are recorded. See
`SPO2_ADRENERGIC_METABOLIC_RESULTS.md` and its frozen plan.

These are selected-patient, unadjusted recorded changes. Samples bracket the
anchor but do not locate biochemical onset or separate endogenous signaling
from treatment. A generic glucose/potassium prognostic association is already
published and cannot serve as this project's novel finding. The discovery goal
remains active and unfulfilled; execution was local and incremental, not Colab.

## Latest biochemical experiment: calcium and pH

The full source extraction completed locally with independently verified
hospital linkage, fixed sample times, assay units and specimen pairing.
Absolute-unit qualification retained 7 eICU and 526 MIMIC people, so the
original joint screen stopped before contrasts. A subsequent metadata check
showed 98 eICU people with pH at both fixed calcium times and matching unit
labels. A separate protocol, frozen before effects, tested native calcium
ratios with stable unit tuples and a common plausible conversion set. It
retained 98 eICU people in 17 hospitals and 526 MIMIC people. The conversion
factor cancels algebraically; this does not resolve absolute calcium units
or establish that calibration is stable.

| Primary contrast | eICU (simultaneous 98.75% CI) | MIMIC (simultaneous 98.75% CI) |
|---|---|---|
| Ratio of calcium geometric fold changes | 0.990985 (0.930496–1.014992) | 0.999429 (0.962104–1.038802) |
| Difference in mean pH changes | +0.008840 (−0.052021 to +0.140190) | +0.013343 (−0.017706 to +0.044733) |

The joint directional criterion failed. All conventional 95% intervals also
cross their null values. Independent SQL/Python quality flags, possible-unit
sets, selected sample pairs, log ratios and contrasts agree. Every applicable
shared-unit cancellation and 100 directly expanded bootstrap draws per source
validate; all 20,000 draws/source retain both exposure groups. These are
unadjusted biochemical associations, with no demonstrated calcium depletion,
binding mechanism, intracellular myocardial change or mortality benefit.
See `SPO2_IONIZED_CALCIUM_RESULTS.md`, `SPO2_CALCIUM_RELATIVE_RESULTS.md` and
their separate frozen plans. All execution was local and incremental in the
single primary notebook, not an uninterrupted full run or a Colab run.

## Latest biochemical experiment: methemoglobin change

Both full raw laboratory sources were scanned locally. The fixed original
pre-ICU/first-follow-up timing specification has 346 paired eICU people and
zero MIMIC people. The original joint gate failed before numeric qualification.
That result is retained; a separate eICU hospital protocol was locked before
values or biochemical contrasts were inspected. Its fixed hospital hash gave
96/250 raw paired people and 85/237 qualified people across 19/30 disjoint
hospitals. No hospital reassignment or sample-time replacement was used.

| eICU split | People | Exposed minus unexposed MetHb change, pp | Nominal 95% CI | Simultaneous 97.5% CI |
|---|---:|---:|---|---|
| Discovery | 85 | -0.054102 | -0.225357 to 0.130560 | -0.248151 to 0.153531 |
| Validation | 237 | -0.008073 | -0.085838 to 0.049974 | -0.099911 to 0.058345 |

The proposed oxidation signal did not pass the fixed directional criterion.
The independent hospital split, unit/entry/value qualification, selected pairs,
SQL mean contrasts and direct expansion of 100 of 20,000 hospital-bootstrap
draws per split all validate. The raw eICU cache contains two unit-conflicting
rows and 203 explicitly censored result rows; they are handled by the frozen
quality rules, not imputed as exact values. No mortality model was fitted.
These are unadjusted biochemical associations in selected tested people, with
assay, drug, transfusion and selection limitations. See
`SPO2_HEMOGLOBIN_OXIDATION_RESULTS.md`, `SPO2_OXIDATION_HOSPITAL_RESULTS.md`,
and the explicit separate protocols. The discovery goal remains active.

## Latest biological mortality screen: cardiovascular recovery

The frozen recovery/sham experiment completed locally in 3,613 eICU people
across 185 hospitals and 322 MIMIC people. Recovered desaturation cases numbered
547 (84 deaths) and 110 (19 deaths); both support floors passed. Persistent HR
elevation per 10 bpm had within-case odds ratios 1.1556 (95% CI 0.9276–1.4396)
and 1.3052 (0.8357–2.0385). The desaturation-specific interaction odds ratios
were 1.0254 (0.7785–1.3506) and 1.3110 (0.7762–2.2143). All primary Holm
p-values exceed 0.78; the fixed joint advancement criterion failed.

SQL/Python independently reproduce original episodes, sham hashes/anchors,
recovery flags and HR values. Both eight-parameter models converge, score
equations hold, and independently reconstructed sandwich covariance agrees
to within 1.8e−14. This is a minimally adjusted observational screen with
recovery and measurement selection; it does not establish equivalence,
an autonomic mechanism or mortality benefit. See
`SPO2_RECOVERY_HYSTERESIS_RESULTS.md` and its frozen plan.

## Latest completed source experiment: full raw arterial timing recovery

Both full supplied laboratory files were scanned locally under a frozen
protocol. Missing-hospital linkage and additional pO2 labels recovered eight
eICU and 40 MIMIC rows in the common time range, but added no two-sided
first-transition timing pairs. At ±5 minutes the permissive ceilings remain
zero eICU/two MIMIC people; at ±15 minutes, two/nine. These are unqualified
timing ceilings, not arterial oxygen changes. The 3,631 eICU/4,133 MIMIC old
common-window rows reconstruct exactly, and independent Python validates
all timing flags and 59 missing-hospital candidate assignments.
See `SPO2_ARTERIAL_RAW_RECOVERY_RESULTS.md`.

A distinct full raw cardiac-output metadata scan also completed. CCO paired
support was 73/95 people in the internal splits, including only nine/five
exposed. Thermodilution support was 73/75, including six/12 exposed. Neither
modality met the fixed floor; no flow values or outcomes were selected.
See `SPO2_DIRECT_FLOW_RESULTS.md`. Source SHA256 hashes from both raw
experiments match the earlier audited supplied files.

## Latest mortality experiment: HR direction during the first SpO2 transition

Completed locally on September 5, 2026, from existing audited first-four-hour
bins in the original cohorts. Person selection preceded HR matching and
mortality joining. HR was matched at both ends in 4,036 eICU and 1,171 MIMIC
people. The primary nonflat-HR sample contained 1,058 eICU people across 161
hospitals and 478 MIMIC people. Both fixed count gates passed.

| Primary crude estimand | eICU, pp (nominal 98.75% CI) | MIMIC, pp (nominal 98.75% CI) |
|---|---|---|
| HR fall versus HR rise during falling SpO2 | +1.25 (−6.65 to +8.84) | +9.87 (−3.63 to +23.62) |
| Above difference minus the corresponding difference during rising SpO2 | −2.75 (−12.98 to +7.65) | +5.11 (−13.15 to +23.54) |

This does not meet the frozen replicated large-risk criterion. Unadjusted
95% intervals also cross zero for all four contrasts. The larger MIMIC
point estimate is not a replicated biological finding. No adjusted models,
alternative HR thresholds or replacement episodes were fitted. Flat and
unavailable HR states and unknown mortality remain separately reported.

The intervals use 20,000 fixed-seed bootstrap draws, resampling hospitals in
eICU and people in MIMIC, with Bonferroni adjustment for four primary estimands;
coverage is approximate. All draws had nonempty group denominators. Independent
episode/person/HR-label reconstruction and direct expansion of 100 resamples
per source passed. Cache hashes and landmark/identity checks passed. These
coarse measurements do not identify a chemoreflex, cardiac output, or arterial
hypoxia; confounding by severity, treatment and rhythm remains. See
`docs/SPO2_HEART_RATE_POLARITY_RESULTS.md` and its frozen protocol. Execution
was local and incremental in the primary notebook, with no raw-source rescan
and no full uninterrupted or Google Colab run. The discovery goal is unfulfilled.

## Latest biological experiment: paired ECG repolarization

Completed locally on September 5, 2026, in the primary notebook. Original
metadata pairing retained 338 MIMIC people before measurement exclusions;
the fixed split yielded 124 usable discovery and 85 validation pairs.
Both sample and design gates passed, with no missing model covariates or
removed nuisance terms. The unchanged original SpO2 exposure was used.

| Adjusted contrast | Discovery estimate, ms (95% CI) | Validation estimate, ms (95% CI) |
|---|---|---|
| Change in JTcF | +0.011 (−12.759 to +12.782) | −0.680 (−17.074 to +15.714) |
| JTcF change minus QRS change | +3.120 (−11.339 to +17.578) | −7.371 (−27.019 to +12.277) |

All four primary Holm p-values are 1.0. This fails the frozen large-effect
screen in both splits. Under these models, the JTcF upper confidence limits
are below the investigator-selected +20-ms prioritization threshold. This
does not exclude transient effects, effects in excluded rhythms, or effects
obscured by measurement error and selection. The nominal timing-margin and
same-device sensitivities fail sample support (76/49 and 3/2 people), so
no sensitivity estimates were fitted. No mortality endpoint was accessed.

The publisher-verified machine file contained ten selected records whose
timestamps disagreed with the registry despite matching subject/study IDs.
The first run stopped before quality counts or models. An explicit
pre-outcome linkage amendment excluded those records without substituting
ECGs or changing analytical gates. Independent pairing, interval arithmetic,
coefficients and HC3 covariance checks passed. All results are machine-derived;
waveforms have not been adjudicated. Rate-correction and algebraic-component
limitations were recorded before outcomes and remain material. See
`docs/SPO2_ECG_REPOLARIZATION_RESULTS.md`, `docs/SPO2_ECG_TIMESTAMP_AMENDMENT.md`
and `docs/SPO2_ECG_INTERPRETATION_LIMITS.md`.

This specification is closed to positive promotion. It establishes neither
a new biological mechanism nor mortality benefit. Execution was local and
incremental, not a full uninterrupted notebook run or Google Colab.

## Diagnostic ECG access and temporal corroboration

Locally executed on September 5, 2026, in the primary notebook. The official
open MIMIC-IV ECG v1.0 registry contains 800,035 ECGs; 60,708 records match
subjects in the original 4,711-encounter cohort. Recorded timestamps place an
ECG within 2,921 hospital encounters, within the first four ICU hours in 1,251,
and in both the pre-ICU and later windows in 341 encounters (338 people;
95 exposed encounters, 246 unexposed). These are nominal time matches.
Publisher checksums and independent SQL/Python linkage checks pass.

A separately frozen comparison extracted 3,343 unique charted EKG procedures
in 1,980 original encounters. It found 711 isolated candidate pairs using a
fixed ±24-hour window: 269 agree within 5 minutes, 460 within 15 minutes,
618 within 60 minutes, and 650 within four hours. The median signed
ECG-minus-procedure difference is −4 minutes, with long tails. Another 743
valid-start procedures have no candidate and 1,784 have multiple candidates.
All candidate edges, degrees, isolated pair identities and offsets match an
independent Python calculation.

The source warns of unsynchronized ECG clocks; charted procedures are not
verified acquisition links, and the comparison window truncates possible
disagreement. No clocks were corrected. No ECG amplitudes, machine values,
interpretations, or mortality outcomes were examined in these two analyses.
This establishes a possible measurement resource, not a biological finding
or a validated acute sequence. Generic hypoxemia–electrical associations are
already known; see the novelty triage. The goal remains unachieved.
See `docs/SPO2_ECG_LINKAGE_RESULTS.md` and `docs/SPO2_ECG_CLOCK_RESULTS.md`.

## Latest biological test: within-patient coupling across eICU hospitals

The explicitly secondary, prospectively fixed hospital split completed locally
on September 5, 2026. It retained the original oxygen-window and consecutive
troponin rules, with patient fixed effects and hospital-clustered uncertainty.

| Split | Hospitals | Patients | Intervals | Troponin fold-change ratio | 95% CI | Primary Holm p |
|---|---:|---:|---:|---:|---|---:|
| Discovery | 68 | 608 | 1,363 | 0.9503 | 0.8916–1.0129 | 0.2313 |
| Validation | 70 | 460 | 1,034 | 0.9627 | 0.8557–1.0830 | 0.5217 |

No positive internal validation. The pre-estimate sensitivity omitting prior
troponin also fails: ratios 0.9591 (0.8513–1.0805) and 1.0225
(0.8586–1.2178), secondary Holm p=0.9726 in both splits. These ratios compare
measured troponin fold changes, not mortality risks. Short-panel bias and
time-varying confounding remain; MAP is missing in 84.7%/90.2% of the two
splits' windows. This is not a pressure-independent or causal finding.
No mortality endpoint was examined, and no biological breakthrough is established.
See `docs/SPO2_EICU_HOSPITAL_RESULTS.md` and its explicit amendment.

The original two-database within-person specification remains stopped at its
frozen oxygen gate: eICU supports 1,068 people/419 instability switchers;
MIMIC only 16/5, despite 293 people with adequate laboratory timing. New raw
extractions contain 718,723 valid eICU and 33,026 MIMIC vital records. Both
match the existing first-four-hour extraction; SQL/Python window features
agree. No model was fitted under the failed cross-database specification.
The eICU experiment above is a separate hospital-disjoint amendment, not MIMIC
replication. All execution was local and incremental in the primary notebook.
The current full notebook has not been run uninterrupted or in Google Colab.

## Latest biological extension: first-specimen reporting-limit emergence

Locally executed under a frozen protocol on September 5, 2026. Among people
whose last available four-hour troponin was explicitly below a reporting bound,
the first subsequent specimen gave 13 definite emergences in 337 classifiable
eICU people and 11 in 62 MIMIC people at the 12-hour horizon. Exposed versus
unexposed: eICU 6/147 versus 7/190; MIMIC 2/18 versus 9/44. Descriptive risk
differences were +0.40 pp (95% CI -3.93 to +5.31) and -9.34 pp (-25.51 to
+14.26). Neither supports a replicated positive association. All adjusted
models failed frozen support gates; no gate was relaxed. Unmeasured results
were retained separately, not counted negative. The 24-hour sensitivity also
has intervals crossing zero. No mortality outcome was accessed. Source hashes,
independent SQL/Python selection, and decimal interval labels validate.
See `docs/SPO2_TROPONIN_EMERGENCE_RESULTS.md`. This is local incremental
execution in the primary notebook, not a full uninterrupted run or Colab.
The requested biological discovery remains unestablished.

## Provenance

- Result date: 2026-09-01/02 UTC.
- Execution: local analysis-only rerun from fingerprint-validated, precomputed
  MIMIC/eICU ETL artifacts. This was **not** a fresh Google Colab execution and
  did **not** rebuild raw data.
- Full analysis-only run: complete, 29,786 analysis rows, 6,193.18 seconds.
- Episode-to-lactate component refresh: complete, 213.95 seconds, no raw rebuild.
- All-endpoint episode refresh: complete, 913,584 focused records, 1,116
  effect rows, 2,520 observation-process rows, and 4,253.46 seconds. The final
  weighted/evidence-only refresh took 145.87 seconds and rebuilt neither raw
  data nor focused records.
- Endpoint-conclusion refresh: complete, 3.74 seconds, no raw rebuild.
- Advanced episode-inference refresh: complete, 1,084.13 seconds, from cached
  events/cohorts and focused records only. It produced 21,752 strict-preanchor
  rows, 84 overlap-weighted rows, 84 binary TMLE/AIPW rows, 84 continuous rows,
  4,616 site-effect rows, 84 multicenter summaries, and 84 key rows.
- Locked eICU-to-MIMIC validation refresh: complete, 1,525.45 seconds, with
  473,792 endpoint records and all 44 dataset-endpoint advanced fits. It reused
  the cached ETL artifacts and read only MIMIC `d_items.csv` and
  `procedureevents.csv` to recover direct respiratory procedure endpoints.
- Integrity audit: passed for 85 output tables, 48 figures, both dataset
  artifacts, all output fingerprints, exact primary and sensitivity grids,
  GEE/TMLE status and targeting checks, overlap/positivity/effective-sample-size
  claim gates, evidence-label consistency, and zero claims-linter errors.
- Tests: 524 passed in 230.30 seconds. The only warning was a third-party
  Pandera import deprecation notice.

## What “instability” means in this project

All landmark signal summaries use charted SpO2 in ICU hours 0–4. Same-time
duplicates are collapsed by median, values outside 50–100% are excluded, and
measurements are summarized in 15-minute median bins. Transition metrics use
only forward gaps of at most 30 minutes. Dynamics require at least three valid
bins and two qualifying transitions.

Measured features include SD, IQR, MAD, range, RMSSD of adjacent changes,
linear slope, counts/rates/fractions of absolute jumps, counts of 3- and
5-point jumps and drops, sustained jump runs, time/fraction/deficit below SpO2
thresholds, desaturation episode count and duration, sampling/gap measures,
and transparent dynamics/instability scores.

The primary landmark epidemiology exposure is any absolute adjacent change of
at least 4 percentage points. The episode amendment uses the first such
transition as the exposed anchor. No-episode controls use the transition
nearest a deterministic quantile of the exposed anchor-time distribution.

## Cohort and signal coverage

| Dataset | Landmark rows | Any plausible SpO2 | Dynamics eligible | Charted SpO2 events | Median readings in first 4 h |
|---|---:|---:|---:|---:|---:|
| MIMIC | 17,758 | 17,052 | 4,708 | 96,480 | 5 |
| eICU | 12,028 | 11,572 | 11,452 | 493,594 | 46 |

MIMIC clearly contains SpO2. Its limitation is sparse chart resolution, not
absence: only 26.5% of landmark rows meet the strict dynamics requirement,
versus 95.2% in eICU. These are charted values, not continuous waveforms.

## Locked harmonized eICU-to-MIMIC validation (authoritative)

The external-validation analysis now applies exactly the same fail-closed
phenotype in both databases: age at least 18 years, an explicit heart-failure
diagnosis, and ICU admission within 24 hours of hospital admission or
cardiogenic shock. Shock, cardiomyopathy, and acute MI remain covariates and
cannot substitute for HF. After the four-hour exclusion, 17,758 MIMIC and
12,028 eICU hospital encounters remain.

The exposure and window are transferred unchanged from the eICU discovery
result: first gap-qualified absolute SpO2 jump of at least four points in
15-minute median bins, followed by incident invasive ventilation 4–12 hours
after the episode. MIMIC uses direct `procedureevents` item 225792 (Invasive
Ventilation interval start); item 224385 (Intubation task time) is a direct
specificity endpoint. The targeted scan recovered 6,153 ventilation rows in
5,391 stays and 1,806 intubation rows in 1,540 stays.

The prior eICU treatment regex was not valid: it included `simvastatin` through
the substring `simv`, NIV, ventilator-weaning, and tracheostomy documentation.
The authoritative analysis retains only a literal pipe-delimited `mechanical
ventilation` treatment segment and excludes NIV and weaning. In the harmonized
event set it retained 10,416 rows and removed 2,362: 471 NIV, 1,392 weaning,
and 499 nonliteral matches. eICU timing remains first treatment documentation,
not a guaranteed physiological start time.

| Dataset/endpoint | Adjusted RR (95% CI), p | Observation-weighted RR (95% CI), p | Advanced sensitivity |
|---|---|---|---|
| eICU invasive ventilation | 2.05 (1.49–2.84), 1.24e-5 | 2.03 (1.47–2.81), 1.59e-5 | overlap 1.77 (1.13–2.79); TMLE 2.22 (1.28–3.86); AIPW 2.43 (1.39–4.25); multicenter 2.02 (1.38–2.95) |
| MIMIC invasive ventilation | 1.06 (0.74–1.52), 0.767 | 1.06 (0.74–1.52), 0.763 | overlap 0.88 (0.61–1.27); TMLE 1.00 (0.63–1.59); AIPW 1.25 (0.60–2.59) |
| MIMIC intubation specificity | 1.11 (0.70–1.75), 0.656 | 1.11 (0.70–1.75), 0.651 | overlap 1.02 (0.64–1.61); TMLE 1.03 (0.60–1.78); AIPW 1.06 (0.61–1.84) |

Thus the cleaned eICU association is internally strong and estimator-robust,
but the transferred primary endpoint is **not externally replicated in
MIMIC**. The adjusted random-effects synthesis is RR 1.48 (0.77–2.84),
p=0.238, with I²=86.2%. This heterogeneity must not be reframed as a successful
two-database validation.

The strongest cross-database secondary signal is assay-matched troponin rise:
eICU adjusted RR 1.25 (0.99–1.58), MIMIC adjusted RR 1.39 (1.04–1.86), and
adjusted random-effects RR 1.30 (1.09–1.56), p=0.00439, tier-FDR q=0.0483,
I²=0%. It is not uniformly advanced-robust: MIMIC overlap weighting remains
positive, but TMLE is inverse and AIPW is null. INR is nominally positive in
MIMIC and in adjusted synthesis but misses tier FDR. Lactate is directionally
positive in both databases but null (pooled adjusted RR 1.13, 0.84–1.52,
p=0.417).

## Current all-endpoint episode analysis

This is the most biologically local analysis now in the project. It asks
whether the first actual gap-qualified SpO2 instability transition in ICU hours
0–4 precedes an outcome in a frozen post-episode lag, compared with no-episode
stays anchored to the same time distribution. It tests all 21 registered
endpoints, all 12 primary lag windows, all five directional/resolution
sensitivities in focused windows, strict pre-episode baselines, incident risk
sets, continuous changes, outcome-observation bias, parsimonious adjustment,
and observation-weighted clustered GEE. This is exploratory association, not a
causal or confirmatory analysis.

### Advanced v2.4 robustness layer

The cached rerun adds endpoint-independent exposure-overlap weighting, complete
pre/post balance diagnostics, patient-grouped cross-fitted observation and
outcome models, bounded TMLE, one-step AIPW diagnostics, continuous AIPW change
models, eICU center-specific random-effects analysis, leave-one-center-out
checks, and E-values. Propensity models use only information strictly before
the anchor; concurrent absolute SpO2 enters observation/outcome models but not
the exposure model. A positive claim now also requires no more than 10% of
propensities outside 0.02–0.98, no more than 10% of observation probabilities
below 0.02, effective sample size ≥50 in both arms, and no TMLE
targeting-boundary hit.

Overlap weighting removed measured pre-anchor imbalance: eICU maximum absolute
SMD fell from 1.186 to 0.021 and MIMIC from 0.848 to 0.015, with zero encoded
covariates above 0.10 afterward. This changes the target population: 2,569 of
11,667 eICU anchors and 300 of 10,085 MIMIC anchors were outside 0.05–0.95;
overlap effective sample sizes were 5,899 and 5,698, respectively.

The earlier broad-regex ventilation estimates are superseded by the cleaned
locked analysis above. Under that corrected definition, one endpoint-window
has positive confidence intervals across every advanced estimator, with
cross-endpoint FDR support for targeted doubly robust and multicenter methods:

| Dataset, endpoint, lag | Overlap + observation-weighted RR (95% CI), cross q | Bounded cross-fitted TMLE RR (95% CI), cross q | eICU center random-effects RR (modified-HKSJ 95% CI), cross q |
|---|---|---|---|
| eICU invasive ventilation, 4–12 h | 1.77 (1.13–2.79), 0.172 | 2.22 (1.28–3.86), 0.0160 | 2.02 (1.38–2.95), 0.00341; I²=0% |

The cleaned ventilation TMLE had effective sample sizes of 952 exposed and 944
unexposed, no low observation-probability violations, and 8.8% exposure
propensity truncation. Its E-value was 3.86 for the point estimate and 1.87 for
the confidence limit; these quantify, but do not eliminate, possible residual
confounding.

Other important advanced findings are deliberately one tier lower:

- eICU pressor initiation at 4–12 h remained positive in the original adjusted
  and observation-weighted analysis and across centers, RR 1.76 (1.37–2.28),
  cross q=0.00087, I²=0%. TMLE was nominally positive, RR 1.43
  (1.002–2.06), p=0.049, but cross q=0.286; overlap RR 1.29 crossed one.
- eICU death at 8–24 h had an unadjusted center random-effects RR 1.97
  (1.23–3.15), cross q=0.042, I²=0%, but overlap RR 1.16 and TMLE RR 1.18
  both crossed one. It is a multicenter descriptive signal, not adjusted
  doubly robust support.
- MIMIC RRT at 0–4 h was nominally positive by TMLE, RR 2.78
  (1.12–6.94), and overlap weighting, RR 2.61 (1.08–6.32), but neither
  survived cross-endpoint FDR (q=0.165 and 0.205).
- The original MIMIC troponin 0–8 h association was not reinforced by the
  advanced layer. Overlap RR remained positive at 1.59 (1.09–2.30), but TMLE
  was 0.68 (0.51–0.90); 43.2% of observation probabilities fell below the
  support floor and exposed effective sample size was 45, so the advanced fit
  is not claim-ready in either direction.
- No continuous AIPW endpoint survived the cross-endpoint correction or the
  complete support gate. This is a genuine negative result, not a missing
  analysis.

The initially striking MIMIC organ-lab composite TMLE, RR 1.11
(1.06–1.16), cross q=0.000039, is **not claim-ready**: only 390 of 10,085
risk-set anchors had the conservative composite observed, 76.2% of estimated
observation probabilities were below 0.02, and exposed effective sample size
was 25. The estimate remains in the table for auditability but is blocked from
positive evidence grading. This is why numerical significance alone is not
treated as a publishable finding.

### Findings meeting the original v2.3 evidence gate

These three rows have a positive point estimate and 95% interval, at least ten
events per fitted parameter, and cross-endpoint BH q<0.05 in the unadjusted,
adjusted, and observation-weighted analyses.

| Dataset, endpoint, lag after episode | Unadjusted RR (95% CI), cross-endpoint q | Adjusted RR (95% CI), cross-endpoint q | Observation-weighted RR (95% CI), cross-endpoint q |
|---|---|---|---|
| eICU invasive ventilation, 4–12 h (cleaned locked definition) | 2.12 (1.58–2.85), 6.44e-6 | 2.05 (1.49–2.84), 0.000210 | 2.03 (1.47–2.81), 0.000271 |
| eICU pressor initiation, 4–12 h | 1.71 (1.36–2.14), 4.57e-5 | 1.68 (1.32–2.15), 5.28e-4 | 1.68 (1.31–2.15), 5.72e-4 |
| MIMIC assay-matched troponin rise, 0–8 h | 1.87 (1.40–2.50), 0.00230 | 1.69 (1.23–2.31), 0.0344 | 1.75 (1.27–2.41), 0.0220 |

Troponin is myocardial injury evidence, not adjudicated myocardial infarction.
Direct MIMIC ventilation and intubation endpoints are now available; neither
supports the eICU effect in the transferred 4–12-hour analysis.

### Complete endpoint review

The table below reports each endpoint rather than selecting only positive ones.
“Endpoint-FDR” means the result survives correction across that endpoint's two
focused windows but not across every endpoint in the dataset. “CI support” means
adjusted and weighted 95% intervals remain above one but their cross-endpoint
q-values do not.

| Endpoint | MIMIC focused result | eICU focused result |
|---|---|---|
| Lactate rise | 0–1 h RR 2.32 (1.42–3.78), cross q=0.011; adjusted RR 2.17, but weighted RR 1.66 (0.88–3.13) | 0–1 h RR 3.50, very sparse and inconclusive; 1–8 h null |
| VIS escalation | 0–4 h RR 1.21 (1.08–1.36), cross q=0.010, adjusted/weighted CI support | Unavailable: infusion-rate units are not standardized |
| Pressor initiation | 0–4 h RR 1.45 (1.07–1.96), endpoint-FDR; adjustment crosses one | **Robust 4–12 h signal**, RR 1.71 |
| MCS initiation | 0–4 h RR 6.88 (1.26–37.56), but only 4 versus 2 events and multiplicity-sensitive | No positive focused association |
| RRT initiation | 0–4 h RR 2.28 (1.02–5.06), sparse and multiplicity-sensitive | No positive focused association |
| Invasive ventilation | Direct 4–12 h adjusted RR 1.06 (0.74–1.52), null across advanced estimators | **Cleaned robust within-database 4–12 h signal**, adjusted RR 2.05 (1.49–2.84) |
| Death | 0–8 h RR 2.41 (1.54–3.79), cross q=0.0036; adjustment attenuates | 0–8 h RR 5.00 and 8–24 h RR 2.27; adjusted/weighted CI support, with early EPV fragility |
| Troponin relative rise | **Robust 0–8 h signal**, RR 1.87 | Directionally positive but inconclusive |
| Creatinine AKI | Directionally positive, inconclusive in both focused windows | Null/inconclusive |
| Bilirubin worsening | Directionally positive, inconclusive | Directionally positive, inconclusive |
| AST injury | Directionally positive, inconclusive | 8–24 h RR 1.93 (1.01–3.68), multiplicity-sensitive and low-information |
| ALT injury | Directionally positive, inconclusive | 8–24 h RR 2.33 (1.06–5.12), multiplicity-sensitive and low-information |
| INR injury | 0–8 h RR 2.40 and 8–24 h RR 1.90; both cross-endpoint FDR-positive unadjusted, but adjustment is not cross-FDR robust | Directionally positive and underpowered |
| Platelet injury | Inconclusive | No positive focused association |
| KDIGO oliguria proxy | Unavailable: mounted source lacks urine output events | 4–12 h RR 1.18 (1.03–1.35), endpoint-FDR; weighted RR 1.21 |
| Relative urine-output decline | Unavailable | No positive association; exposed episodes also had lower outcome observability |
| Extended hepatic composite | 0–8 h RR 1.55 (1.17–2.06), cross q=0.028 with adjusted/weighted CI support | No positive association |
| Organ-lab composite | 0–8 h RR 1.19 (1.05–1.34), endpoint-FDR and weighted CI support | No positive association |
| Early-decompensation composite | Event-saturated, no useful positive contrast | Underpowered/null |
| Early-decompensation + VIS | Event-saturated, no useful positive contrast | VIS unavailable; remaining composite underpowered/null |
| Any clinical decompensation | Event-saturated/incompletely observed; not headline-eligible | Event-saturated and underpowered; not headline-eligible |

### Cross-dataset and sensitivity evidence

In the locked harmonized analysis, troponin is the only adjusted two-database
secondary synthesis surviving its prespecified tier FDR, RR 1.30 (1.09–1.56),
q=0.0483, I²=0%; its advanced estimator discordance keeps it secondary.

In the broader pre-lock all-window screen, random-effects estimates surviving cross-endpoint FDR were acute lactate rise
at 0–1 h, RR 2.37 (1.47–3.82), q=0.0077, I²=0%; INR injury at 0–8 h,
RR 2.23 (1.39–3.58), q=0.0078, I²=0%; death at 0–8 h, RR 3.41
(1.67–6.96), q=0.0078, I²=75%; and death at 8–24 h, RR 1.86
(1.24–2.78), q=0.0179, I²=61%. The latter two have substantial heterogeneity.
A meta-analysis signal does not mean each source was independently significant.

Across the 420 directional/resolution sensitivity rows, 60 positive contrasts
survived their within-definition endpoint correction. Recurrent patterns were
death, delayed eICU ventilation and pressor initiation, acute MIMIC lactate and
VIS escalation, MIMIC troponin/INR, eICU oliguria, and sparse MIMIC MCS/RRT.
Recovery-rise comparators also predicted several outcomes. That weakens a
direction-specific hypoxic mechanism and is consistent with a broader
instability, treatment-response, monitoring-intensity, or illness-severity
marker.

### Masks that were exposed rather than hidden

Outcome classification was differential for ten focused rows after
cross-endpoint correction, including acute MIMIC lactate (19.1% observed after
episodes versus 13.1% in controls), eICU urine decline, eICU ventilation, and
several composites. IPW/GEE was estimable in 65/84 focused rows: 61 required
weighting and four had complete observation; 11 lacked enough outcome
information and eight were source/risk-set unavailable. This is why the
weighted estimate, not only the complete-case p-value, is shown for every
claim-ready positive.

These historical models contain selected short-lag associations, chiefly
database-specific support-documentation outcomes and a qualified troponin-rise
proxy. They do not establish a biological mechanism or adjudicated myocardial
injury. The harmonized respiratory primary endpoint failed MIMIC validation;
troponin estimators disagree, and the later within-patient eICU hospital test
did not positively validate the hypothesis. Other endpoint families remain
exploratory, null, or too sparse. A collection of nominal associations must
not be presented as confirmed multiorgan biological progression.

## Lactate-specific episode details

The revised estimand asks whether the first qualifying SpO2 episode is followed
by a rise of at least 0.5 mmol/L in the **first next lactate**, relative to a
strictly pre-episode lactate obtained within six hours. Acute `(0, 1]` hour and
delayed `(1, 8]` hour windows form the focused two-test family. The 1–2, 2–4,
and 4–8 hour windows localize kinetics; 0–8 and 8–12 hours are companions.

### Focused results

| Dataset/window | Complete exposed/control pairs | Controlled RR (95% CI), focused q | Adjusted RR (95% CI), focused q | Remeasurement-weighted RR (95% CI), focused q |
|---|---:|---|---|---|
| eICU 0–1 h | 20 / 14 | 3.50 (0.46–26.80), 0.727 | Not estimable | Not estimable: selection EPP 3.78 |
| MIMIC 0–1 h | 78 / 196 | 2.32 (1.42–3.78), 0.0031 | 2.17 (1.22–3.86), 0.0166 | 1.66 (0.89–3.10), 0.218 |
| eICU 1–8 h | 143 / 192 | 1.03 (0.63–1.69), 1.000 | 0.81 (0.43–1.54), 0.518 | 0.93 (0.48–1.79), 0.817 |
| MIMIC 1–8 h | 237 / 830 | 1.27 (0.96–1.69), 0.105 | 1.14 (0.83–1.56), 0.431 | 1.12 (0.80–1.56), 0.523 |

In the advanced layer, acute MIMIC lactate remained directionally positive but
imprecise: overlap RR 1.84 (0.81–4.20), cross q=0.412, and TMLE RR 2.14
(0.81–5.63), cross q=0.539. The doubly robust mean lactate change difference
was +0.87 mmol/L (-0.73 to +2.47), cross q=0.778. Only 20 effective exposed
observations remained after exposure/measurement weighting, below the fixed
claim threshold of 50. Delayed MIMIC lactate was null by overlap weighting
(RR 1.00), TMLE (RR 1.01), and continuous change (+0.16 mmol/L; -0.30 to
+0.61). eICU acute lactate remained underpowered, and its delayed fit failed
weight-support requirements. The acute lactate association is nominally
positive in selected complete/adjusted models and their synthesis, but fails
the advanced support and robustness requirements. It is not an established
biological effect or a stable standalone finding in these strict pairs.

The fixed/random cross-dataset acute estimate is RR 2.37 (95% CI 1.47–3.82),
focused q=0.00074, I²=0%. The delayed estimate is RR 1.21 (0.95–1.55),
q=0.130, I²=0%. The short 1–2 hour localization estimate is RR 1.46
(0.95–2.25), localization q=0.252; later bins are null.

### What the positive acute result does and does not mean

The positive result is a **tail-risk contrast**, not a claim that the typical
lactate increased. Acute median paired change was 0.00 mmol/L in eICU
(95% CI -0.27 to 0.20) and -0.15 in MIMIC (-0.30 to 0.10). Delayed paired
medians declined in both datasets. In MIMIC acute pairs, 30.8% of exposed
versus 13.3% of control pairs rose by at least 0.5 mmol/L, producing the RR.

The controlled MIMIC acute association persisted after parsimonious adjustment
for prior lactate, anchor time, anchor SpO2, pre-anchor sampling, age, shock,
and baseline vasoactive use. It was directionally similar in both shock strata
(RR 2.31 with shock; 2.13 without), both sampling strata, and when anchor SpO2
was at least 90% (RR 1.86). These stratum results are descriptive and are not
separate significance claims.

Directional/resolution sensitivities also pointed upward in MIMIC acute pairs:
3-point drops RR 1.82, 5-point drops RR 2.39, raw-time absolute jumps RR 2.03,
and raw-time 3-point drops RR 1.71. Those companion p-values did not survive
the deliberately global exploratory correction across the full 672-row table;
they support pattern consistency, not independent confirmation. Recovery rises
also trended upward, so the study does not establish direction-specific hypoxic
causation.

### Informative lactate testing

Complete strict lactate pairs were more likely after exposed anchors in eICU
(0.46% vs 0.19%; observation RR 2.38, focused q=0.025) and modestly more likely
in MIMIC (3.40% vs 2.52%; RR 1.35, q=0.056). Inverse-probability weighting for
post-episode remeasurement attenuated MIMIC acute RR from 2.32 to 1.66 and made
it inconclusive. Therefore the acute association is promising and publishable
as exploratory evidence, but it is not robust to every missing-measurement
assumption and is not causal.

## Original first-4-hour landmark to 12/24-hour outcomes

The primary fixed-landmark question remains mostly null. Values below are
patient-cluster-bootstrap risk ratios for the primary ≥4-point jump exposure;
q-values are BH-adjusted within each dataset's prespecified endpoints.

### Lactate and vasoactive support

- Lactate rise: eICU 12 h RR 1.15 (0.77–1.73), q=0.614; 24 h RR 1.06
  (0.73–1.59), q=0.864. MIMIC 12 h RR 1.10 (0.79–1.47), q=0.795;
  24 h RR 1.00 (0.70–1.38), q=1.000.
- VIS rise: MIMIC 12 h RR 1.10 (0.97–1.23), q=0.386; 24 h RR 1.03
  (0.93–1.14), q=0.795. eICU quantitative VIS is unavailable because its
  infusion rates are not standardized enough for a harmonized VIS.
- New MCS: eICU 12 h RR 0.98 (0.37–2.25), q=1.000; 24 h RR 0.80
  (0.41–1.36), q=0.614. MIMIC 12 h RR 1.25 (0.51–2.56), q=0.795;
  24 h RR 1.09 (0.55–1.95), q=0.820.
- Pressor initiation was positive in eICU—12 h RR 1.44 (1.20–1.71),
  q=0.00059; 24 h RR 1.34 (1.15–1.56), q=0.00089—but not MIMIC.

### Renal, hepatic, platelet, and urine outcomes

- Creatinine AKI was null in both datasets at both horizons. RRs ranged from
  1.06 to 1.17 at 12 h and 1.10 to 1.13 at 24 h, all CIs crossing 1.
- Binary hepatic-lab and extended hepatic-injury endpoints were null. The only
  isolated continuous signal was eICU 12-hour AST ratio, partial Spearman
  rho=0.127 (cluster-bootstrap 95% CI 0.028–0.231); it was not replicated in
  MIMIC and is exploratory.
- Platelet injury was null in both datasets at both horizons.
- eICU urine-rate decline was null: 12 h RR 0.96 (0.85–1.07), 24 h RR 0.95
  (0.82–1.09). The eICU 12-hour KDIGO oliguria proxy was nominally positive,
  RR 1.19 (1.03–1.38), but q=0.074; 24 hours was null. MIMIC urine endpoints
  are unavailable because the mounted source lacks `outputevents`.

### Other decompensation signals

- Troponin relative rise was the strongest replicated secondary association:
  eICU 12 h RR 1.25 (1.09–1.44), q=0.010; 24 h RR 1.23 (1.08–1.40),
  q=0.010. MIMIC 12 h RR 1.55 (1.25–1.93), q=0.0058; 24 h RR 1.41
  (1.16–1.73), q=0.0157.
- Mortality was associated in eICU at 12 and 24 hours (RR 4.37 and 2.58;
  q<3.2e-9) and in MIMIC at 12 hours (RR 2.27, q=0.046), but not MIMIC
  24 hours.
- Broad early-decompensation composites are event-saturated or fragile in
  important subsets and are not suitable as headline endpoints.

## Head-to-head biomarker benchmark

The 2026-09-03 analysis-only refresh compared the complete 0–4-hour SpO2
signal block with 0–4-hour SBP, lactate, and an EHR operationalization of the
CSWG-refined SCAI stages in Kapur et al. (JACC 2022;
doi:10.1016/j.jacc.2022.04.049). All 11 candidate models used the same
outcome-observed, SpO2-dynamics-eligible patients, the same five
patient-grouped folds, fold-local preprocessing, and 300 paired patient-cluster
bootstrap draws. The SpO2-signal model contains absolute level,
sampling/missingness, and the instability block; the explicit instability
contrast asks what RMSSD, abrupt jumps, and >=3-point drops add beyond level
and observation process.

### Coverage and interpretation limits

| Dataset | Harmonized HF stays | SpO2 dynamics | SBP | Lactate | Kapur-SCAI classified | All-four shared |
|---|---:|---:|---:|---:|---:|---:|
| MIMIC | 17,758 | 26.5% | 95.3% | 43.3% | 42.7% | 9.0% |
| eICU | 12,028 | 95.2% | 9.2% | 16.3% | 17.0% | 2.1% |

The eICU cache lacks `vitalAperiodic.csv`, so its SBP comparison uses sparse
invasive systemic pressure and is source-limited. Out-of-hospital cardiac
arrest is unavailable in both cached databases. Normal incomplete records are
therefore left unclassified rather than assigned stage A, and every assigned
stage is marked as a lower bound. SCAI was unclassified in 57.3% of MIMIC and
83.0% of eICU harmonized-HF stays. This is a transparent four-hour EHR
operationalization, not clinician-adjudicated SCAI.

### Overall discrimination

Median patient-grouped out-of-fold AUROCs across all fitted endpoints were:

| Scope | Endpoints | SpO2 signal | SBP | Lactate | Kapur-SCAI | All conventional | All conventional + SpO2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIMIC | 30 | 0.577 | 0.576 | 0.659 | 0.626 | 0.690 | 0.691 |
| eICU | 27 | 0.526 | 0.552 | 0.584 | 0.595 | 0.590 | 0.577 |
| Pooled secondary | 34 | 0.562 | 0.579 | 0.640 | 0.641 | 0.667 | 0.663 |

SpO2 was therefore not the strongest general-purpose predictor. Lactate had
the highest point AUROC for 18/30 MIMIC endpoints and 11/27 eICU endpoints;
Kapur-SCAI led 8/30 and 14/27. The SpO2 signal led only incident MCS at 12 and
24 hours in each separate database. Those MCS results are sparse and
events-per-feature fragile: MIMIC had 32 and 47 events and eICU had 24 and 54.
They did not provide a stable external-database advantage.

For the four MIMIC primary endpoints, SpO2/SBP/lactate/SCAI AUROCs were:

- lactate rise 12 h: 0.566/0.553/0.552/0.569; no paired difference was
  significant;
- lactate rise 24 h: 0.480/0.498/0.600/0.545; SpO2 was worse than lactate and
  SCAI;
- VIS rise 12 h: 0.670/0.748/0.725/0.816; and
- VIS rise 24 h: 0.641/0.717/0.694/0.790.

eICU has no quantitative VIS endpoint. Its lactate-rise AUROCs were
0.433/0.478/0.454/0.491 at 12 hours and 0.526/0.565/0.628/0.613 at 24 hours.
No primary endpoint showed a significant positive instability increment beyond
absolute SpO2 and sampling.

### Qualified SpO2 signals

- In MIMIC, instability improved on absolute SpO2 for incident lactate crossing
  2 mmol/L: delta AUROC +0.046 (95% CI 0.005–0.088) at 12 hours and +0.064
  (0.026–0.102) at 24 hours. This did not replicate internally in eICU and did
  not significantly improve the full conventional model.
- The complete SpO2 signal had high internal MCS AUROC (MIMIC 0.885/0.863;
  eICU 0.876/0.835 at 12/24 hours), but the instability block itself added
  approximately zero, event information was fragile, and transportability was
  not confirmed.
- In eICU, adding SpO2 to all conventional markers increased mortality AUROC by
  +0.064 (0.018–0.106) at 12 hours and +0.021 (0.002–0.041) at 24 hours. This
  improvement did not reproduce in MIMIC or consistently transport between
  databases, and calibration/net-benefit gains were not general.
- Marker-specific modified-Poisson models associated SpO2 instability with
  eICU death, pressor initiation, oliguria, platelet injury, and troponin rise;
  MIMIC supported 12-hour death and both troponin windows. Troponin was the
  clearest two-database marker-specific association (RR per 1 SD worse
  instability 1.10/1.08 in eICU and 1.16/1.12 in MIMIC at 12/24 hours, all
  tier-adjusted q<0.013). No positive instability association remained
  FDR-significant after simultaneous adjustment for the other biomarkers.

The honest head-to-head conclusion is that early lactate and Kapur-SCAI are
stronger broad severity markers, SBP is most useful for MIMIC vasopressor/VIS
escalation, and charted SpO2 contributes selected early-warning associations
but not a consistent independent predictive increment. The current data
support further evaluation of selected charted associations, not a demonstrated
independent biological marker, continuous-waveform mechanism, or improvement
in HF mortality management.

## Prediction versus association

The instability block did not provide reliable incremental prediction beyond
clinical context, absolute SpO2, and sampling/missingness. For example, eICU
24-hour lactate had ΔAUROC 0.014 (95% CI -0.0038 to 0.0296) and ΔAUPRC 0.0137
(-0.0010 to 0.0360). Apparent improvements in sparse endpoints had inadequate
events per transformed feature and are barred from claim grading. MIMIC-to-eICU
transport validates only MIMIC-trained models; it is no longer eligible as
evidence for an eICU-trained model. No endpoint is classified `robust_positive`.
That label belongs to the fixed-landmark incremental-prediction matrix; it does
not negate the original v2.3 episode associations, but the v2.4 support and
double-robustness checks determine how strongly those associations can now be
presented.

## Defensible publication conclusion

The strongest result is a cleaned, internally robust eICU association between
an actual charted SpO2 instability episode and invasive-ventilation
documentation 4–12 hours later. It survives conventional adjustment,
observation weighting, overlap weighting, TMLE, AIPW, center random effects,
information/positivity gates, and targeted/multicenter multiplicity correction.
It does **not** replicate using either direct MIMIC invasive-ventilation starts
or direct intubation timestamps, and the pooled primary estimate is
heterogeneous and null. Accordingly, it cannot be presented as a validated
cross-database primary result.

Troponin is the strongest harmonized two-database secondary association and
survives adjusted tier-level meta-analysis multiplicity, but estimator
discordance—especially inverse MIMIC TMLE and null AIPW—prevents a robust claim.
INR, acute lactate, MIMIC VIS escalation, hepatic injury, eICU oliguria, RRT,
and mortality remain supportive or hypothesis-generating with explicit
missingness, multiplicity, or estimator-discordance caveats. Pressor initiation
is strong within eICU but not MIMIC. MCS remains too sparse; creatinine,
platelets, and relative urine decline do not show a consistent effect.

The publication claim is an observational association with eICU support
documentation and a qualified cross-database troponin-rise proxy. The later
eICU hospital-disjoint within-patient test failed positive validation and
must be included in the conclusion. Neither myocardial-injury causation,
transportable predictive benefit nor mortality reduction has been established.

## September 5 paired cardiorenal follow-up

A later local analysis of the raw-replay cohort compared the troponin-rise
proxy and creatinine worsening in the same patients with complete marker pairs
and follow-up. After the documented decimal-threshold correction, the 12-hour
adjusted differential association was +8.25 percentage points in eICU
(95% CI −0.81 to +17.30; Holm p=0.1486; 656 encounters) and −1.35 in
MIMIC (−11.23 to +8.54; Holm p=0.7890; 621
encounters). Both 24-hour sensitivity intervals also crossed zero. This does
not provide convincing cardiac-specific biomarker evidence, and it does not
establish that renal clearance caused the troponin changes. Shared measurement
selection, different marker kinetics and thresholds limit the comparison.

The assay-matched 1.5-fold troponin ratio does not require a documented
99th-percentile reference-limit exceedance and must not be treated as an
adjudicated myocardial-injury or MI diagnosis. See
`SPO2_CARDIORENAL_CONTRAST_RESULTS.md` for the locked comparison, validated
patient-clustered models and limitations. Preliminary floating-point results
are archived; see `SPO2_CARDIORENAL_DECIMAL_THRESHOLD_CORRECTION.md`.
This later experiment does not
overwrite the historical estimates above or establish a biological discovery.

## September 5 timing follow-ups

The later September 5 CVP timing experiment completed locally with 65 eICU
people (66 encounters) and six MIMIC people meeting its frozen before/after
measurement rules. Both fail its 100-person floor; eICU counts additionally
retain unresolved native units. No pressure change or outcome association was
computed. See `SPO2_VENOUS_PRESSURE_RESULTS.md`.

A separately documented descriptive amendment then examined all 133 eICU /
76 MIMIC complete serial-troponin trajectories, without reversing the failed
primary count gate. Patterns (neither / pre only / post only / both rises)
were 74/25/15/19 in eICU and 33/17/11/15 in MIMIC. Thus recorded pre-episode
rises occurred in 44/133 and 32/76 encounters. All are distinct people, and
independent SQL/Python concentration and label checks passed. No hypothesis
test or mortality model was fitted. These changes precede the first episode
captured in the four-hour ICU window; earlier hypoxemia and injury onset are
unresolved. See `SPO2_TROPONIN_DESCRIPTIVE_RESULTS.md` and its explicit
amendment. This selected-case observation is not a new biological mechanism
or a demonstrated mortality benefit.

## Remaining non-code limitations

The reporting-recovery extension identified additional interval information.
eICU's discarded bounds provide an optimistic ceiling of 431 extra timed
pairs at 12 hours and 337 at 24 hours; these are not valid rise counts.
Fourteen existing 12-hour pairs and 19 existing 24-hour pairs have discarded
follow-up bounds, and none has a later discarded baseline bound. MIMIC's
strict whole-comment parser remains negative, but a separate whitelist
amendment recovered 598 `<0.01` and three `>25` ng/mL results, all with
missing numeric values. It treats the accompanying 0.10 ng/mL interpretation
statement as metadata, not a measurement or diagnosis. Source identities,
SQL/Python classification, patient flags and unchanged comment hashes validate.
No original label was changed and no new association was fitted. See
`SPO2_TROPONIN_REPORTING_RECOVERY_RESULTS.md` and
`SPO2_TROPONIN_COMMENT_TEMPLATE_RESULTS.md`. These are actual local runs,
not a full uninterrupted notebook or Colab execution.

The later original-file troponin reporting audit reconstructed the current
numeric pipeline and labels from 13,682 eICU and 4,758 MIMIC source records.
eICU's retained values have exact matching result text, while 1,800 explicitly
bounded source records are omitted by the numeric pipeline. In MIMIC, all
4,137 retained numeric records have `___` as result text; the remaining 621
records lack that text and an accepted numeric value. Consequently the
original reporting qualifiers cannot be checked from this MIMIC field.
This is not proof of false rises, and eICU's retained-value agreement does
not resolve selection from omitted bounded measurements. No labels or
associations were changed. Independent source grouping, values/availability,
ratios, labels and used-sample flags passed validation. See
`SPO2_TROPONIN_ASSAY_AUDIT_RESULTS.md`. Execution was local, with raw scans
followed by validation-only reruns; it was not a complete fresh Colab run.

The subsequent local troponin sampling-opportunity sensitivity used 2,294
eICU and 774 MIMIC troponin-paired encounters in the raw-replay landmark
cohort, without requiring paired creatinine. At 12 hours, adjusted exposure
coefficients for maximum versus one-observed-sample expectation were
3.62 versus 2.86 pp in eICU and 7.18 versus 8.29 pp in MIMIC. Their paired
differences were +0.75 pp (95% CI -0.10 to +1.60) and -1.12 pp (-2.48 to
+0.24), both Holm p=0.1649. First-sample results and 24-hour sensitivities
are reported in full. This does not support a consistent maximum-selection
explanation, but it does not rule out informative testing or validate a
biological mechanism. In the same primary sample, 441/2,294 eICU and
259/774 MIMIC baseline results were unavailable at minute 240 under the
retained specimen-time endpoint rules. The analysis is retrospective.
Independent endpoint/model checks passed. See
`SPO2_TROPONIN_OPPORTUNITY_RESULTS.md`; execution was local incremental,
not an uninterrupted full notebook or Google Colab run.

The subsequent local arterial-pairing audit found zero qualified paired PaO2
measurements within five minutes of both original SpO2-bin times in either
database. At the prespecified 15-minute tolerance, only two eICU and four
MIMIC people qualify. Independent SQL/Python timing reconstruction and source
hash checks passed. These counts concern the existing hospital-linked cache,
not every possible blood-gas record in either database. No oxygen change or
mortality association was estimated; the cached-pairing specification is
closed. See `SPO2_ARTERIAL_PAIRING_RESULTS.md`. The primary notebook contains
this actual local incremental execution, not a fresh complete Colab run.

1. Charted SpO2 is not waveform SpO2; artifact morphology and sub-minute
   instability cannot be reconstructed from these cached tables.
2. MIMIC dynamics are sparse (median five readings in four hours), creating a
   selected dynamics-eligible cohort.
3. Complete strict pre/post lactate pairs are rare and clinically selected.
4. Cached events begin at ICU offset zero, so pre-ICU lactates cannot be added
   without a deliberate raw-data rebuild.
5. eICU quantitative VIS and MIMIC urine output remain unavailable for source
   reasons. MIMIC incident ventilation is now recovered directly from
   `procedureevents`; eICU ventilation timing remains first treatment
   documentation rather than a guaranteed start time.
6. The episode analysis is a transparent post-result protocol amendment, not a
   prospectively preregistered confirmatory analysis.
7. Broad composites are event-saturated or incompletely observed and should not
   be used as headline evidence; individual endpoints are more interpretable.
8. Independent temporal/site validation or a prospective waveform cohort is
   required before clinical deployment or causal interpretation.
9. Overlap-weighted estimates target patients with comparable exposure
   propensity, not the full ICU population; TMLE/AIPW still require no
   unmeasured confounding and missing-at-random endpoint observation.
