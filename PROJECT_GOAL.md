# PhysioGraph Project Goal

## Current user-directed research scope — September 5, 2026 UTC

Seek a genuinely novel biological finding with substantial potential relevance
to heart-failure mortality, comparable in clinical importance to lactate in
SCAI staging. The user explicitly rejected a measurement-method audit as a
substitute. Use the original databases under Google Drive `Data/MIMIC/Full`
and `Data/eICU/Full`. The primary runnable artifact is
`PhysioGraph_Biological_Discovery.ipynb`, containing only the SpO2 follow-up.
No biological breakthrough or mortality benefit is currently established.
The latest event-aligned capnography investigation completed after recovering
committed raw tables from an interrupted local run. Five-minute two-sided
timing support was 43 eICU people in 25 hospitals and five MIMIC people;
both fail the frozen minimum. Full native hashes match prior independent
extractions; original episode signs, nearest times and context flags validate
in SQL and Python. No numeric CO2 change or mortality association was calculated.
See `docs/SPO2_CAPNOGRAPHY_FEASIBILITY_RESULTS.md`. This is a measurement limit
of this proposed respiratory contrast, not evidence against the mechanism
or completion of the requested biological discovery. The goal remains active.
The latest acute joint potassium/glucose experiment completed locally in 152
eICU and 134 MIMIC people. The predeclared coupled event occurred in 0/56
exposed versus 4/96 controls in eICU and 1/37 versus 3/97 in MIMIC. Both risk
differences point against the proposed increase; event counts are below the
fixed inference floor. No bootstrap intervals, adjusted model, alternate
endpoint or mortality association was used to rescue the candidate. Native
source hashes, episode/control anchors, exact-decimal qualification, paired
specimens and joint labels validate independently. See
`docs/SPO2_ADRENERGIC_METABOLIC_RESULTS.md`. This is new research evidence,
not completion of the biological-discovery goal, which remains active.
The subsequent methemoglobin experiment completed locally. Original timing
support was 346 paired people in eICU and zero in MIMIC, preventing the
two-database screen. A separate protocol fixed disjoint eICU hospital groups
before numeric qualification: 85/237 people across 19/30 hospitals qualified.
Exposed-minus-unexposed methemoglobin-change contrasts were -0.054102 and
-0.008073 percentage points; both multiplicity-adjusted intervals cross zero.
The biochemical oxidation candidate failed its directional replication
criterion. See `docs/SPO2_OXIDATION_HOSPITAL_RESULTS.md`. This is actual new
biochemical research, not a fulfilled discovery goal. The goal remains active.
The paired ionized-calcium/pH experiment also completed. Absolute-unit support
failed in eICU (7 people; MIMIC 526). A separately frozen native-ratio study
used stable unit tuples so a common conversion cancels, yielding 98 eICU and
526 MIMIC people. Calcium geometric fold-change ratios were 0.990985 and
0.999429; all four calcium/pH simultaneous intervals cross their null values.
The biochemical criterion failed in both sources, with independent pairing,
possible-unit, algebraic-cancellation and bootstrap checks passed. See
`docs/SPO2_CALCIUM_RELATIVE_RESULTS.md`. The goal is active and unfulfilled.
The latest recovery-specific biological mortality screen completed in both
sources. Within-desaturation HR-displacement odds ratios were 1.1556 and
1.3052 per 10 bpm; neither the within-case effects nor the comparison with
sham times passed the frozen criterion (all Holm p>0.78). Both support
floors and independent phenotype/model checks passed. See
`docs/SPO2_RECOVERY_HYSTERESIS_RESULTS.md`. This is completed research work,
not a fulfilled discovery goal. The goal remains active.
Current continuation: active research has resumed with a completed full raw
arterial-laboratory census and a physiologically motivated direct
cardiac-output measurement census. The raw laboratory census recovered records
but did not increase two-sided timing support. The flow census found too few
exposed people for its fixed validation floor. No flow values were selected.
See `docs/SPO2_ARTERIAL_RAW_RECOVERY_RESULTS.md` and `docs/SPO2_DIRECT_FLOW_RESULTS.md`.
The impasse statements below are historical assessments, not evidence that
all available research routes were exhausted. No new biological effect is
established by either source availability or a protocol.
Earlier September 5 continuation assessments marked the goal blocked after
repeated impasse assessments; see `docs/SPO2_RESEARCH_DECISION.md` for that
history. Those assessments were superseded by actual raw-source, recovery,
and biochemical experiments. They do not describe the current active goal
or establish that MIMIC/eICU research routes are exhausted.
The latest frozen HR-direction mortality screen completed in both original
cohorts. During falling SpO2, crude mortality differences for HR fall versus
HR rise were +1.25 pp in eICU (nominal multiplicity-adjusted interval −6.65
to +8.84) and +9.87 pp in MIMIC (−3.63 to +23.62). Comparison with rising
SpO2 also failed in both sources. The primary samples included 1,058 eICU
people/161 hospitals and 478 MIMIC people; hospital/person bootstrap and
independent phenotype checks passed. No adjusted model or alternative
threshold was used to rescue the failed screen. These are coarse observational
states, not a demonstrated autonomic mechanism. See
`docs/SPO2_HEART_RATE_POLARITY_RESULTS.md`. The goal remains active.
The latest paired-ECG biological experiment completed locally in 124 discovery
and 85 validation people. Instability-associated JTcF changes were +0.011 ms
(95% CI −12.759 to +12.782) and −0.680 ms (−17.074 to +15.714); the paired
repolarization-versus-conduction contrast also failed. All four primary Holm
p-values are 1.0. The fixed large-effect criterion failed, and neither timing
margin nor same-device sensitivities had adequate support. Ten inconsistent
machine timestamps were excluded under an explicit pre-outcome linkage
amendment without replacement ECGs. Rate-correction, rhythm selection and
timing limitations prevent a broad biological absence claim. See
`docs/SPO2_ECG_REPOLARIZATION_RESULTS.md`. The discovery goal remains active.
The newly verified open MIMIC-IV diagnostic ECG resource covers 2,921 of the
original encounters by nominal hospital time, including 341 with nominal
pre-ICU/later ECG pairs. Local metadata linkage and the separately frozen
charted-EKG comparison completed in the same notebook. Among 711 isolated
candidate timestamp pairs, 460 agree within 15 minutes and 618 within an
hour; long tails and uncertain procedure-to-acquisition identity prevent
declaring clocks synchronized. Those initial metadata assessments examined
no ECG phenotype or mortality outcome. They created a direct cardiac measurement
route, not the requested biological result. Generic hypoxia–arrhythmia and
repolarization associations are already known. See
`docs/SPO2_ECG_LINKAGE_RESULTS.md` and `docs/SPO2_ECG_CLOCK_RESULTS.md`.
The latest separate eICU hospital validation weakens the within-patient
troponin hypothesis: 608 patients in 68 discovery hospitals and 460 in 70
validation hospitals yielded fold-change ratios 0.9503 (95% CI 0.8916–1.0129)
and 0.9627 (0.8557–1.0830), primary Holm p=0.2313/0.5217. The prior-troponin
omission sensitivity also fails positive validation. No mortality outcome was
examined. MAP is mostly missing and short-panel bias remains unresolved.
The original two-database specification stopped at MIMIC's oxygen support gate
(16 people/five switchers); the eICU analysis is a separate, explicitly frozen
hospital-disjoint amendment. See `docs/SPO2_EICU_HOSPITAL_RESULTS.md` and
`docs/SPO2_WITHIN_PERSON_TROPONIN_RESULTS.md`. All new work ran locally in the
single primary notebook; no new mechanism or completed goal is claimed.
The preceding frozen first-specimen reporting-limit emergence experiment completed
locally: 12-hour descriptive SpO2-exposure risk differences +0.40 pp in eICU
(95% CI -3.93 to +5.31) and -9.34 pp in MIMIC (-25.51 to +14.26). Events were
too sparse for the predeclared adjusted models. Missing tests remained missing,
and reported upper bounds were not imputed as exact baseline values. This
does not support a replicated positive biological association; the experiment
is documented in `docs/SPO2_TROPONIN_EMERGENCE_RESULTS.md` and the primary notebook.
The goal remains active. The initial phosphate experiment was underpowered
and is archived under `research/biological_discovery`; it is not rerun by the
primary notebook. The circulatory-context extension is documented in
`docs/SPO2_CIRCULATORY_CONTEXT_PLAN.md` and anchored to the existing
SpO2-myocardial injury/decompensation results.

The corrected circulatory-context experiment completed locally on September
5, 2026 UTC. All four primary timing comparisons failed the frozen count
gate; no association model was fitted. See `docs/SPO2_CIRCULATORY_RESULTS.md`
for actual counts, source corrections, numerical-representation validation,
and limitations. This is an unsupported candidate, not a completed discovery.

The myocardial-injury ordering feasibility experiment also completed locally
on September 5, 2026 UTC. After including explicitly linked MIMIC ED samples,
complete serial pre/post troponin trajectories numbered 133 in eICU and 23
in MIMIC. Independent SQL/Pandas patient-set checks agreed. The frozen minimum
of 100 in each database failed; trajectory directions and mortality associations
were not inspected. See `docs/SPO2_TROPONIN_DIRECTION_RESULTS.md`. This does not
determine whether injury precedes or follows SpO2 instability. Both completed
experiments are retained in the primary notebook; the discovery goal is unfulfilled.

A further candidate asks whether pre-ICU hemoglobin modifies susceptibility to
the existing SpO2-associated myocardial-injury signal. Its feasibility protocol
is frozen in `docs/SPO2_HEMOGLOBIN_FEASIBILITY_PLAN.md`. Its subsequent outcome
interaction experiment has now completed locally. Novelty, biological mechanism, and therapeutic
relevance remain unproven. It cannot be used to claim transfusion benefit.
Its linked-record feasibility run completed with 659 eICU and 133 MIMIC
observations, failing the frozen minimum of 200 in each database. No outcome
interaction was fitted. A separate outcome-blind MIMIC pre-ICU laboratory
linkage audit completed and passed its CBC/troponin validation criterion. The
documented linkage-only amendment completed using the original feasibility
algorithms with uniquely assigned missing-ID records. Hb paired support is now
659 eICU and 516 MIMIC, passing the same minimum screen. Injury-ordering support is 133 and 76,
still failing its gate. No biological window or gate changed. See
`docs/MIMIC_PREICU_LAB_LINKAGE_RESULTS.md` and
`docs/PREICU_LAB_LINKAGE_AMENDMENT.md`.
The frozen Hb interaction experiment completed with clinical-context adjustment,
missingness sensitivities and independent model validation. No endpoint meets
the positive replication criterion. Valid primary interaction odds ratios per
2 g/dL lower Hb: eICU troponin 0.840 (95% CI 0.585–1.205), MIMIC mortality
1.100 (0.913–1.325); both Holm p=1. The eICU mortality and MIMIC troponin
models have quasi-complete separation from rare missingness indicators and
are non-estimable. A post-fit validity correction is documented; no variable
was dropped to rescue a model. Sensitivities do not supply a convincing
positive signal. This candidate is unsupported, not a biological discovery.
See `docs/SPO2_HEMOGLOBIN_INTERACTION_RESULTS.md` and
`docs/SPO2_HEMOGLOBIN_MODEL_VALIDATION_CORRECTION.md`. The primary notebook
contains the actual locally executed extension; no Colab execution is claimed.

The subsequent oxygen-response feasibility experiment also completed locally.
Only 33 eICU and 3 MIMIC encounters have a qualifying documented oxygen
escalation, closely paired SpO2 observations and a corroborating oxygen setting.
Both fail the frozen support gate. No saturation-response magnitude or outcome
association was examined for that candidate. Direct ST-segment testing is not
supported by the inspected harmonized sources: the MIMIC ST item is a
monitoring-enabled checkbox. See `docs/SPO2_OXYGEN_RESPONSE_FEASIBILITY_RESULTS.md`.
These are reasons to reject proposed experiments, not biological discoveries.
The goal remains unfulfilled and active.

A subsequent mechanism review found that normoxemic stratification is already
specified/implemented (S6), and that saturation instability linked to periodic
breathing in HF was experimentally studied in 1996. Neither can be relabeled
as a new discovery. See `docs/SPO2_MECHANISM_REVIEW.md` for primary-source
evidence and the distinction between correcting a surrogate and improving survival.
The public MIMIC-IV waveform preview offers a small new measurement opportunity
within the same existing cohort: 48 records match exact HF hospital admissions,
28 overlap the first four ICU hours in non-gap segments, and 11 belong to the
original dynamics-eligible risk set. Those 11 list respiratory, ECG and Pleth
channels. The initial stage downloaded metadata only. Independent
plain-header parsing confirmed all 48 record–encounter identity/time comparisons.
The initial case-sensitive Resp/Pleth flags were corrected before any waveform
experiment. See `docs/MIMIC_WAVEFORM_OVERLAP_RESULTS.md`.

The subsequently frozen respiratory pilot completed locally in those 11 encounters
(five with the original exposure, six controls). All 487 selected source files
passed official checksums. None of the five first-episode blocks passed the
monitor-SpO2 completeness rules: three had less than 90% observed seconds and
two retained gaps. No episode coherence or phase-null test was computed; the
five p=1 slots are missing-test placeholders. Other times contained 18/60 usable
exposed blocks and 33/72 control blocks, but no later episode was substituted.
Independent numeric clocks and direct FLAC reconstruction passed across all 11
records, comparing 16,234,176 valid channel frames. An integer-frame averaging
assumption in the independent checker was corrected and documented; extracted
signals and biological thresholds did not change. See
`docs/SPO2_WAVEFORM_PILOT_RESULTS.md` and
`docs/SPO2_WAVEFORM_CALIBRATION_VALIDATION_CORRECTION.md`.
This is insufficient evidence to test the proposed mechanism, not evidence
that it is absent. No mortality benefit or novel biological finding is established.

The arterial acid–base follow-up also completed locally. Outcome-blind extraction
and independent SQL/Pandas reconstruction identified preceding complete gases
in 1,767 eICU and 757 MIMIC encounters. A separately frozen crude mortality
screen retained one encounter per person before joining outcomes. Its additive
interaction estimates for instability with respiratory alkalemia were −1.17
percentage points in eICU and +25.23 in MIMIC. Simultaneous conservative bounds
were −39.58 to +36.73 and −32.98 to +81.05 percentage points, respectively.
The MIMIC joint group contained five deaths among only 13 patients. Neither
database passed the screen; there is no convincing replicated interaction.
These unadjusted, imprecise contrasts establish neither an oxygen-unloading
mechanism nor benefit from changing ventilation, pH or oxygen treatment. See
`docs/SPO2_ACID_BASE_FEASIBILITY_RESULTS.md` and
`docs/SPO2_ACID_BASE_MORTALITY_SCREEN_RESULTS.md`. The requested discovery
remains unestablished and the goal remains active.

A paired cardiorenal falsification analysis then compared troponin rise with
creatinine worsening in the same fully observed patients. All four adjusted
models were estimable and their patient-clustered covariances independently
reconstructed. After the documented decimal-threshold correction, the primary
12-hour differential associations were +8.25 pp (95% CI −0.81 to +17.30;
Holm p=0.1486) in eICU and −1.35 pp (−11.23 to +8.54; Holm p=0.7890)
in MIMIC, using 656 and 621 encounters. Preliminary estimates remain archived.
Both 24-hour sensitivities also crossed zero. This does not establish cardiac
specificity or show that renal clearance explains the troponin signal.
The ratio endpoint does not itself require an assay reference-limit exceedance
and is not adjudicated myocardial injury or MI. See
`docs/SPO2_CARDIORENAL_CONTRAST_RESULTS.md` and the pre-estimate assay-encoding
clarification and decimal-threshold correction. No new biological discovery or
mortality benefit is established.

Latest scope correction: use **only MIMIC and eICU**, and stay anchored to
PhysioGraph's SpO2-instability/decompensation work. The outcome-blind CVP
timing experiment completed locally at 12:25:58 UTC. Only 65 eICU people
(66 encounters) and six MIMIC people meet its full timing rules, below the
100-person requirement in both sources. eICU units remain unresolved and its
counts are optimistic finite-value timing support. Independent reconstruction
matches first episodes and timing-qualified sets exactly. No CVP direction or
outcome association was estimated. See `docs/SPO2_VENOUS_PRESSURE_RESULTS.md`.

An explicitly documented descriptive amendment now examines the original
133 eICU / 76 MIMIC complete serial-troponin encounters. It departs from the
earlier stop rule while preserving the failed primary gate and all time/assay
rules. It reports every pre/post pattern without hypothesis tests or mortality
claims. The local description found a pre-episode 1.5-fold rise in 44/133
eICU and 32/76 MIMIC encounters. Patterns (neither / pre only / post only /
both) were 74/25/15/19 and 33/17/11/15. Every encounter is a distinct person;
the deterministic sensitivity is therefore unchanged. Independent SQL/Python
values and labels agree. These are recorded numeric changes before the first
captured ICU episode; earlier hypoxemia, injury onset and causal direction
remain unresolved. See `docs/SPO2_TROPONIN_DESCRIPTIVE_RESULTS.md` and its
explicit amendment. This observation does not establish the requested discovery.

The next local arterial-pairing audit completed at 16:36:22 UTC. No qualified
PaO2 pairs occur within five minutes of both original SpO2-bin times in either
database (4,302 eICU / 1,371 MIMIC full-window episodes). Prespecified
15-minute sensitivity support is only two eICU / four MIMIC people. Even the
permissive raw timing ceiling is sparse. SQL/Python flags and source hashes
validate independently. This cached-pairing specification is closed without
reading oxygen changes or mortality; the result does not resolve physiology
versus artifact. See `docs/SPO2_ARTERIAL_PAIRING_RESULTS.md`. The discovery
goal remains unmet.

The local sampling-opportunity falsification is also complete. Among 2,294
eICU and 774 MIMIC troponin-paired encounters at the four-hour landmark,
the adjusted maximum-minus-one-observed-sample exposure contrasts are
+0.75 pp (95% CI -0.10 to +1.60) and -1.12 pp (-2.48 to +0.24), both
Holm p=0.1649. This provides no consistent evidence that the maximum alone
explains the association; it does not establish equivalence or rule out
selective testing. The 24-hour sensitivities also cross zero. Native legacy
endpoints and the new decimal-label differences are audited, and independent
coefficient/covariance checks pass. This is a new landmark sensitivity, not
a reconciliation of the older locked episode-based estimators. See
`docs/SPO2_TROPONIN_OPPORTUNITY_RESULTS.md`. No novel biological finding or
mortality benefit is established.

Scope remains restricted to
PhysioGraph's SpO2-instability/decompensation work. The original-file troponin
reporting audit is complete locally: 13,682 eICU and 4,758 MIMIC records.
Retained eICU values have matching exact numeric text; 1,800 explicitly
bounded source results fall outside the accepted numeric pipeline. All
4,137 retained MIMIC numeric results have `___` as text, preventing a
reporting-limit check from that field. This is a reporting limitation, not
proof that the rises are false. Independent source/cached-value and
endpoint-label reconstruction passed. No clinical interpretation was rescued
by substituting a laboratory reference range or an Hb/Hct volume formula.
See `docs/SPO2_TROPONIN_ASSAY_AUDIT_RESULTS.md`. The goal remains unmet.

The reporting-recovery extension subsequently completed. eICU discarded
bounds allow an optimistic ceiling of 431 additional 12-hour timed pairs,
while affecting follow-up in 14 existing pairs and replacing no later
pre-landmark baseline in the existing pairs. MIMIC's whole comments are not
standalone results, but a separate whitelist amendment recovered 598 upper
bounds of 0.01 ng/mL and three lower bounds of 25 ng/mL, all with missing
numeric values. It explicitly excludes the neighboring 0.10 ng/mL assay
interpretation statement from the patient result. Independent SQL/Python
classifications and source identities agree. These intervals have not been
substituted as concentrations or used for a new outcome association. See
`docs/SPO2_TROPONIN_COMMENT_TEMPLATE_RESULTS.md`. No biological discovery or
mortality benefit is established.

Scope remains restricted to
PhysioGraph's SpO2-instability/decompensation work. Do not pursue molecular,
HLHS, Garcia, or unrelated datasets. New experiments must develop unresolved
biological questions arising from the existing results. The SpO2 protocol
below remains the study's scientific anchor.

## Research Question

**Can SpO2 signal instability (variability, abrupt changes, sampling behavior, and related signal-quality features) measured in the first 4 hours precede or predict early signs of cardiogenic decompensation in the subsequent 12–24 hours?**

Decompensation is operationalized as one or more of:

- Need for mechanical circulatory support (MCS)
- Rising lactate
- Laboratory markers of organ injury
- Falling urine output
- Rising vasopressor/inotropic score (VIS)

### What we mean by each term

| Term | Operational definition |
|------|------------------------|
| **SpO2 signal instability** | Primary dynamics-only features: RMSSD, abrupt gap-qualified changes, and directional drops. Absolute hypoxemia and sampling density are separate adjustment/sensitivity variables, not part of the primary dynamics claim. |
| **Precede** | SpO2 instability appears before other decompensation signals move (lead-time / temporal-ordering analysis) |
| **Predict** | SpO2 features at the 4-hour landmark forecast decompensation events or trajectories in the next 12–24 hours |
| **Decompensation outcomes** | MCS initiation, lactate rise, organ-injury labs, urine-output decline, VIS increase — assessed separately and as a composite where appropriate |

### Endpoint hierarchy

| Priority | Endpoint | Rationale |
|----------|----------|-----------|
| Primary | Lactate rise, VIS rise (12–24h) | Frequent, measurable, central to shock escalation |
| Secondary | Urine-output decline, organ-injury labs | End-organ signals with clear directionality |
| Exploratory | MCS need, composite "any decompensation" | Clinically critical but event-sparse |

### Scope guard

We test whether SpO2 **signal dynamics** add information beyond absolute SpO2 and respiratory support context. We do **not** claim SpO2 instability is causally independent of respiratory failure without explicit adjustment and sensitivity analyses.

---

## Mission

Build a rigorous, reproducible pipeline to answer the research question above using physiological time-series from ICU electronic health records (MIMIC, eICU). Every claim must survive leakage checks, external validation, and honest uncertainty reporting.

## How We Answer the Question

1. **Fixed 4-hour observation landmark** — All SpO2 instability features computed from pre-landmark data only.
2. **12–24 hour outcome window** — Decompensation signals assessed in the post-landmark horizon (with sensitivity at 12h vs 24h).
3. **SpO2 feature set** — Primary dynamics (RMSSD, SD/IQR companions, abrupt jumps, directional drops) are separated from absolute hypoxemia, sampling intensity, and signal-quality proxies; implausible values and missing bins are tracked explicitly.
4. **Outcome trajectories** — Not just binary labels: lactate delta, VIS delta, urine-output trend, lab shifts, MCS initiation.
5. **Confounder controls** — Respiratory support (FiO2, vent status), absolute SpO2 level, demographics, baseline hemodynamics; documented availability denominators per dataset.
6. **Comparator baselines** — SpO2 instability models are compared on the same
   eligible sample against a clinical baseline and clinical + absolute-SpO2
   + sampling/missingness reference. Frozen lactate comparators are
   legacy/supporting analyses only.
7. **Actual-episode timing** — In addition to the fixed landmark, anchor the
   first gap-qualified instability transition, compare it with time-aligned
   no-episode controls, and test every endpoint over non-overlapping and
   cumulative 0–24-hour lags with strict pre-episode baselines and incident
   risk sets.
8. **Advanced robustness** — For every focused episode endpoint, audit
   pre-anchor covariate overlap and outcome-observation positivity, estimate
   bounded patient-grouped cross-fitted TMLE plus continuous AIPW changes, and
   quantify center heterogeneity without promoting estimates that fail event,
   effective-sample-size, overlap, or targeting gates.

## Core Technical Goals

1. **Reproducible pipeline** — Modular ETL, cohort selection, feature engineering, labeling, and validation in the `physiograph` package.
2. **Leakage-safe modeling** — PROBAST+AI Domain 4 guards: no post-landmark features, no outcome contamination, deterministic patient-level splits, fit-on-train-only preprocessing.
3. **Patient-grouped internal validation** — Estimate MIMIC and eICU performance
   separately using fold-local preprocessing and patient-grouped OOF predictions.
4. **External validation** — Train the harmonized model in MIMIC and evaluate it
   untouched in eICU; report eligibility, discrimination, calibration, and uncertainty.
5. **Testable analysis modules** — SpO2 drilldown, horizon models, and subgroup analyses live in tested Python modules, not hidden notebook cells.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Research question answered | SpO2 instability tested against all decompensation outcomes at 12–24h with pre-specified endpoint hierarchy |
| Lead-time evidence | Report both fixed-landmark ordering and actual-episode/time-aligned-control contrasts for every endpoint; do not infer causal precedence |
| Incremental value | SpO2 dynamics improve prediction beyond absolute SpO2 and respiratory-support-adjusted baselines |
| Pipeline reproducibility | Notebook parity within tolerance; automated tests passing |
| Leakage prevention | All guard checks pass; no forbidden columns in feature sets |
| External validation | eICU harmonization with documented eligibility and missingness rates |
| Claim integrity | Apparent vs validated metrics clearly labeled; null and negative findings reported |
| Robust episode inference | Cross-FDR findings remain distinguishable across overlap-weighted, doubly robust, and unadjusted multicenter estimators, with failed support diagnostics shown rather than hidden |

## Non-Goals (Current Scope)

- Graph-based modeling or lactate-centered prediction as the primary research question (supporting methods only).
- CLIF integration until data access and schema mapping are confirmed.
- Causal claims about SpO2 independent of respiratory failure without explicit adjustment.
- Presenting in-sample model metrics as validation results.

## Guiding Principles

1. **One question** — Everything in this project serves the SpO2 instability → decompensation hypothesis.
2. **Truth over polish** — Report null, fragile, and negative findings alongside robust results.
3. **Provenance matters** — Label artifacts as cached, precomputed, local, or freshly executed.
4. **Per-dataset checks first** — Run MIMIC and eICU analyses separately before any pooled claims.
5. **Controls are explicit** — Respiratory support, renal replacement therapy, and baseline hemodynamics must have documented availability denominators.
6. **Single source of truth** — Canonical constants, schemas, and forbidden-column lists live in the package.
