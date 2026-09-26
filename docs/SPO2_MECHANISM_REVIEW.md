# Mechanistic review after the completed negative experiments

Scope remains MIMIC and eICU, anchored to the existing SpO2 study. None of the
findings below is a new biological discovery by this project.

A September 6 literature check also rules out treating venous oxygen saturation
or an oxygen-extraction proxy as a new generic shock marker. A [1998 acute
decompensated-HF study](https://pubmed.ncbi.nlm.nih.gov/9781972/) already examined
lactate and central venous saturation in otherwise clinically occult shock.
A [2025 post-hoc ECMO-CS analysis](https://pubmed.ncbi.nlm.nih.gov/40660350/)
examined cardiac index, venous saturation and the CO2 gap as possible treatment
effect modifiers. These are precedents, not additional patient datasets or
proof of benefit in this project. A [2025 conference abstract](https://esc365.escardio.org/journal/82372)
also discusses unexpectedly high central venous saturation and inflammatory
deterioration in cardiogenic shock; its small exploratory analysis does not
establish the mechanism. No venous-saturation extraction or outcome model was
initiated for these generic questions. Any narrower SpO2-linked extraction
question would still require a distinct biological contrast and reliably
identified venous specimen/site metadata.

The current respiratory candidate concerns EtCO2 timing at the original SpO2
transitions. The supplied eICU periodic vital table contains EtCO2, and MIMIC
item 228640 is numeric EtCO2; neither supplied dictionary establishes its
absolute unit. The frozen first stage selects timing and context metadata only,
before any CO2 amplitudes or outcomes. See `SPO2_CAPNOGRAPHY_FEASIBILITY_PLAN.md`.

Novelty precedents are substantial. A [2012 systolic-HF study](https://pubmed.ncbi.nlm.nih.gov/22537025/)
already assessed resting and exercise EtCO2 prognosis in 963 people. A
[2020 upright Cheyne–Stokes study](https://www.jacc.org/doi/10.1016/j.jacc.2020.04.033)
measured respiration, EtCO2 and saturation together. A [2022 hypoxia-induced
periodic-breathing experiment](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.912056/full)
reported joint hemodynamic, ventilation, CO2 and SpO2 oscillations in systolic
HF. These references were identified while the locked raw timing scans ran,
without inspecting new CO2 amplitudes. A generic coupling or prognosis result
would repeat established observations. Five-minute monitor medians also cannot
resolve the breath-level physiology measured in those experiments.

The proposed narrower distinction is a ventilatory disturbance versus a
perfusion-compatible change around the existing coarse ICU episode. EtCO2
alone cannot identify which occurred: ventilation, blood flow, metabolism,
oxygen delivery and sampling devices can alter the signal. Respiratory rate
alone does not supply minute ventilation. Source coverage and co-timing are
necessary prerequisites, with a separate value and contrast specification
required before interpreting a physiological change. No arterial–end-tidal
gradient will be calculated from unspecified units or called true Bohr dead
space, and no generic EtCO2 mortality association will be promoted as novel.

The locked timing study has completed: 43 eICU people in 25 hospitals and
five MIMIC people had two-sided CO2 timestamps at the five-minute tolerance.
Both source support floors fail. Source hashes and independent event/pair
reconstruction validate after recovery from an interrupted hashing step.
No CO2 amplitudes were selected or interpreted. This finding limits the
specific event-aligned comparison and leaves the biological mechanism
unresolved. See `SPO2_CAPNOGRAPHY_FEASIBILITY_RESULTS.md`.

The acute metabolic question tested a substantial potassium fall and glucose
rise in the same person soon after the original episode. [Human epinephrine
experiments](https://pubmed.ncbi.nlm.nih.gov/6314140/) establish that this paired
direction can occur through adrenergic physiology without tachycardia in one
protocol; [hypoxia experiments](https://pubmed.ncbi.nlm.nih.gov/15044204/) also
demonstrate impaired glucose tolerance. These studies motivate the question,
but the paired laboratory changes do not specifically measure endogenous
catecholamines or distinguish them from treatment effects.

Novelty is particularly constrained: a [2024 HFpEF analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11612494/)
already studied prognosis using the glucose/potassium ratio, and a [multicenter
coronary-care study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11460640/) examined
its association with hospital mortality. A generic ratio–mortality model cannot
be claimed as a new biological discovery here. The frozen current experiment
instead tests an acute paired-change event relative to fixed SpO2/sham anchors;
see `SPO2_ADRENERGIC_METABOLIC_PLAN.md`. The original review's acute lactate
association concerned a tail event despite little median change, providing a
reason to examine a coupled tail event rather than presuming a mean shift.
The laboratory pair brackets the anchor and does not locate the onset of a
biochemical change within that interval. Even a positive contrast would need
additional temporal and treatment evidence before being attributed to the
SpO2 episode or endogenous adrenergic signaling.

The frozen experiment has now completed in 152 eICU people across 61 hospitals
and 134 MIMIC people. Joint events occurred in 0/56 exposed versus 4/96 controls
and 1/37 versus 3/97, respectively. Both point differences are negative and
both event-support gates fail; no bootstrap interval or adjusted/mortality
model was fitted. Independent raw qualification, anchors, exact labels and
patient sets validate. This does not support the proposed acute metabolic
signal; it does not establish equivalence or absence of adrenergic physiology.
See `SPO2_ADRENERGIC_METABOLIC_RESULTS.md`. Descriptive component changes are
retained without promoting them to replacement positive endpoints.

The subsequent biochemical test asks whether SpO2 instability precedes an
increase in recorded methemoglobin. A [laboratory reoxygenation experiment](https://www.sciencedirect.com/science/article/pii/S0167488999001639)
used modified cell-free hemoglobin with endothelial cells, limiting its direct
relevance to intact adult red cells in HF. [Clinical sepsis research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11497947/)
already examines methemoglobin prognosis; generic MetHb-mortality association
would not establish novelty. Methemoglobin is not a unique ROS/NO flux marker;
[hydroxocobalamin assay interference](https://academic.oup.com/labmed/article/55/1/50/7179524)
is one concrete alternative explanation requiring attention if a signal appears.

The completed original timing specification has 346 eICU pairs and no MIMIC
pairs. A separate hospital-disjoint eICU protocol, locked before numeric
qualification and changes, yielded 85/237 people in 19/30 hospitals. Change
contrasts -0.054102/-0.008073 percentage points both have multiplicity-adjusted
intervals crossing zero. This does not support a replicated positive biochemical
signal. SQL/Python qualification, pairing and hospital-bootstrap checks passed.
See `SPO2_OXIDATION_HOSPITAL_RESULTS.md`; the original failed support test remains
preserved rather than relabeled as database replication.

Another completed candidate concerns paired blood ionized-calcium and pH change.
The [1997 human hyperventilation study](https://pubmed.ncbi.nlm.nih.gov/9264919/)
did not observe a significant calcium decrease despite alkalinization, and
[experimental myocardial responses](https://pubmed.ncbi.nlm.nih.gov/8238582/)
are biphasic. Thus neither a progressive contractile deficit nor a blood-calcium
decline follows automatically from textbook binding chemistry. The new protocol
requires the predicted biochemical changes in both existing databases and does
not equate blood calcium with intracellular myocardial calcium. This is distinct
from the earlier baseline alkalemia-by-SpO2 mortality interaction.

Absolute-unit support failed (7 eICU / 526 MIMIC). A subsequent protocol fixed
native calcium fold changes with stable unit tuples before exposure effects
were inspected, recovering 98 eICU / 526 MIMIC paired people. The common
conversion cancels in a ratio, but no absolute unit or stable analyzer is
thereby proven. Calcium geometric fold-change ratios 0.990985 and 0.999429
and pH change contrasts +0.008840 and +0.013343 all have intervals crossing
the null. The joint biological screen failed. See
`SPO2_CALCIUM_RELATIVE_RESULTS.md`; original failed absolute-unit support and
the explicit separate protocol remain available.

The subsequent cardiovascular-recovery experiment tested whether persistent
HR elevation after recorded saturation recovery has a different mortality
association from HR drift at fixed sham times without an original desaturation.
It completed in 3,613 eICU/322 MIMIC people with both support floors satisfied.
Within-desaturation odds ratios per 10 bpm were 1.1556 and 1.3052; the
desaturation-specific contrasts were 1.0254 and 1.3110. All primary intervals
cross one and Holm p-values exceed 0.78. This does not validate the candidate.
The phenotype is a coarse, selected recovery observation, not a measured
autonomic reflex or oxygen debt. See `SPO2_RECOVERY_HYSTERESIS_RESULTS.md`.

Two prerequisite raw-source experiments also completed. Missing-hospital
linkage and broader pO2 labels add lab rows but no two-sided first-transition
timing pairs. Direct cardiac-output metadata identify only 14 exposed people
with CCO pairs and 18 with thermodilution pairs, failing the predeclared
internal-validation floor without reading flow values. These narrow source
limits do not establish that MIMIC/eICU cannot yield another biological finding.

The latest direct cardiac follow-ups also failed their frozen criteria.
Paired diagnostic ECGs in 124 discovery/85 validation people showed no
replicated large JTcF change or repolarization-versus-conduction contrast;
see `SPO2_ECG_REPOLARIZATION_RESULTS.md`. A separate first-transition
HR-direction mortality screen in both original databases yielded +1.25 pp
(eICU) and +9.87 pp (MIMIC) crude HR-fall versus HR-rise differences during
falling SpO2, with intervals crossing zero and failed comparisons against
rising SpO2; see `SPO2_HEART_RATE_POLARITY_RESULTS.md`. Neither establishes
injury, autonomic failure or mortality benefit. The paired arterial-gas
failure concerns two samples around one transition; it does not establish
artifact or agreement of individual SpO2 readings with arterial oxygenation.

## Repeated-measure follow-up and a rejected alternative

The completed repeated-measure experiment compares consecutive troponin changes
across oxygen windows within the same patient. The frozen specification is in
`SPO2_WITHIN_PERSON_TROPONIN_PLAN.md`; it extends the existing cohort through the
already extracted 28-hour laboratory window without changing the original
admission-window estimates. Its first timing gate has passed in 1,090 eICU and
293 MIMIC people. Subsequent oxygen coverage reduced usable MIMIC support to
16 people and five instability switchers, so the frozen cross-database gate
failed without fitting models. See `SPO2_WITHIN_PERSON_TROPONIN_RESULTS.md`.
A separately frozen eICU hospital-disjoint amendment then tested 608 patients
in 68 discovery hospitals and 460 in 70 validation hospitals. Primary ratios
of troponin fold changes were 0.9503 and 0.9627, with both confidence intervals
crossing one; omission of the prior-troponin covariate also failed positive
validation. This weakens the hypothesis's current evidentiary basis. It does
not establish equivalence, an unbiased causal estimate or a mortality benefit.
Short-panel bias, informative testing and MAP missingness of 84.7%/90.2% remain.
See `SPO2_EICU_HOSPITAL_RESULTS.md` and the explicit amendments.

A competing acid-base/acetazolamide idea was rejected as a new-mechanism claim
after primary-literature review. The [2006 double-blind HF crossover study](https://pubmed.ncbi.nlm.nih.gov/16239622/)
already demonstrated reductions in central apnea and hypoxemic sleep time.
The [2014 physiological crossover study](https://pubmed.ncbi.nlm.nih.gov/24251826/)
found improved apnea despite increased hypercapnic ventilatory responsiveness.
These studies are small and do not establish survival benefit. They preclude
presenting the broad treatment or a simple reduction-in-chemosensitivity story
as this project's novel discovery. No additional patient dataset was analyzed.

## Troponin endpoint interpretation

The [2026 universal definition](https://www.jacc.org/doi/10.1016/j.jacc.2026.07.025)
defines acute myocardial injury using a rise/fall with a value above the
sex-specific 99th-percentile upper reference limit; infarction requires
additional evidence of ischemia. The current project's assay-matched
1.5-fold rise alone is therefore a troponin-rise proxy, not an adjudicated
injury or infarction diagnosis.

A direct header check of the supplied raw files on September 5 confirmed
`ref_range_lower` and `ref_range_upper` columns in MIMIC `labevents.csv`.
The supplied eICU `lab.csv` header contains no reference-limit column.
This is a schema finding only: actual completeness of MIMIC reference limits,
their equivalence to assay-specific 99th percentiles, and sex-specific
calibration have not been established. No concentrations, thresholds or
existing endpoint labels were changed by this check. A nominal laboratory
reference range must not automatically be equated to the required 99th percentile.

## Concentration changes and apparent biomarker rise

Hemoconcentration is an established HF research topic. It is not sufficient
evidence of successful decongestion: a [704-patient AHF study](https://doi.org/10.1016/j.cardfail.2016.04.005)
found poor agreement between hemoconcentration and clinical congestion.
More directly, [paired measured-volume validation](https://pubmed.ncbi.nlm.nih.gov/30098381/)
found poor correspondence between formula-estimated and measured changes in
plasma volume. An [acute-HF validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8788058/)
also reported poor estimation of individual volume changes from Hb/Hct.

These findings make it inappropriate to treat an Hb/Hct-derived correction
of troponin as a measured change in circulating biomarker mass in the current
project. This rejects that mechanistic interpretation, not every descriptive
analysis of serial blood concentrations. Transfusions, bleeding and changes
in red-cell volume further complicate interpretation. No new patient model
or volume-corrected troponin endpoint was calculated in this triage.

The next audit instead returns to original troponin reporting text and
compares it with the numeric values underlying the existing signal; see
`SPO2_TROPONIN_ASSAY_AUDIT_PLAN.md`. An inequality is not an exact
concentration, and a raw reference range is not automatically the required
assay-specific 99th percentile.

## Avoid duplicating the existing question

The normoxemic versus hypoxemic comparison is already sensitivity S6 in
docs/ANALYSIS_PLAN.md, and is implemented in spo2_epidemiology.py. Do not
rename that comparison as a new biological hypothesis or choose a different
normal-saturation threshold simply to create another significance opportunity.
The source code establishes that this comparison was specified and implemented;
this review does not independently certify every historical output row.

The broad saturation-instability mechanism also predates this project. In a
[1996 physiological study](https://pubmed.ncbi.nlm.nih.gov/8813833/), 40 stable
CHF patients and eight controls underwent breathing and saturation recordings.
Controlled breathing reduced saturation instability particularly with periodic
breathing/Cheyne–Stokes respiration; oxygen was compared in five patients.
This was not an ICU mortality study and is not evidence that our charted
four-point jumps represent the same phenomenon. It precludes claiming that
oxygen saturation instability in HF, or its link to periodic breathing, is new.

A more direct precedent is the [Light et al. study published online in 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708937/).
In 19 hospitalized acute-HF participants, respiratory polygraphy and paired
evening/morning high-sensitivity troponin measurements associated overnight
troponin increase with sleep-apnea severity and central-apnea patterns. The
study excluded ICU patients and advanced HF support, and its estimates were
very imprecise. It does not validate the present ICU phenotype or establish
causality, but it rules out claiming first discovery of the broad
sleep-disordered-breathing/troponin relationship in decompensated HF.
The [authors' response](https://pmc.ncbi.nlm.nih.gov/articles/PMC10256834/)
explicitly acknowledges confounding and the need for intervention studies.
These are literature observations only; their patient data are not included
in any MIMIC/eICU experiment here.

Improvement in respiratory surrogates must not be equated with survival benefit.
[SERVE-HF](https://www.nejm.org/doi/full/10.1056/NEJMoa1506459) effectively treated
central sleep apnea yet found increased all-cause and cardiovascular mortality
with its ASV intervention in the studied systolic-HF population. The later
[ADVENT-HF trial](https://pubmed.ncbi.nlm.nih.gov/38142697/) used a different
ASV strategy and found no significant effect on its primary outcome or mortality,
without identifying an ASV safety issue. These trials do not establish that
SpO2 instability is protective, or that all respiratory stabilization is harmful.
They prevent a shortcut from prognostic association to a mortality-saving
intervention. No trial patient data are added to this project's analyses.

## Metadata feasibility leading to the waveform pilot

The [public MIMIC-IV waveform preview](https://physionet.org/content/mimic4wdb/0.1.0/)
is part of MIMIC and uses matching subject/admission identifiers and date offsets.
Its documented release contains 200 records from 198 patients; this is not a
claim that any of them meets the current HF cohort or exposure requirements.
The checked release is a technical preview, not the promised larger release.

Fetch only its public RECORDS index and metadata headers for subjects already
in the current canonical MIMIC HF cohort. Join exact subject_id and hadm_id,
then calculate overlap with the existing first-four-hour ICU window. Distinguish
record-span overlap from explicitly non-gap segments. Inspect layout signal
names, but do not equate the union of available channels with continuous,
simultaneous, artifact-free measurements. Keep matched identifiers and headers
in private Drive Data; publish only aggregate overlap and provenance.

This metadata check does not download waveform amplitudes, add a non-MIMIC
patient cohort, change the SpO2 exposure, inspect mortality, or estimate a
mechanism. If overlap exists, a separate biological protocol and signal-quality
assessment are required before analyzing respiratory/circulatory coupling.
The supplied eICU five-minute medians cannot resolve breath-by-breath cycles;
they cannot be represented as waveform-level mechanistic replication.

All previous failed candidates and numerical corrections remain preserved.
The requested significant biological finding and paradigm-shifting completion
criterion remain unfulfilled.

## Venous-pressure candidate and existing biology

The [1979 coronary venous hypertension experiment](https://pubmed.ncbi.nlm.nih.gov/759726/)
already examined potentiation of ischemic injury. [Experimental peripheral
venous congestion in HFrEF](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884348/)
also altered inflammatory, endothelial and neurohormonal markers. These
findings do not establish what happens around the current SpO2 episodes.
The [2026 LVAD study](https://pubmed.ncbi.nlm.nih.gov/41655605/) linked CVP
reduction with increased hypoxic ventilatory sensitivity in a small cohort;
the direction cautions against a simple assumption that higher CVP always
causes more respiratory instability. General CVP prognosis or congestion
biology is not a new discovery. The separately frozen timing feasibility in
`SPO2_VENOUS_PRESSURE_PLAN.md` is a prerequisite for a narrower temporal
question, with unresolved eICU pressure units explicitly retained.

An additional alternative explanation was identified while the frozen raw scan
was running, before any CVP direction or association was inspected. A
[2005 perioperative waveform study](https://pubmed.ncbi.nlm.nih.gov/15728063/)
observed forehead waveforms consistent with venous contributions, while a
[2006 healthy-volunteer experiment](https://pubmed.ncbi.nlm.nih.gov/17122573/)
reduced false-low forehead saturation readings during head-down positioning
using a tensioning headband. These sensor/site-specific findings do not prove
artifact in our ICU data or generalize to every probe. They mean that even a
replicated CVP–charted-SpO2 association would need independent arterial
oxygenation evidence before being called congestion-induced hypoxemia. No
new pressure or oxygenation analysis was performed to select this explanation,
and the frozen timing criteria are unchanged.

The separately locked pilot has now completed locally in the 11 overlapping
original dynamics-eligible encounters. The five first-exposure blocks all failed
monitor-SpO2 completeness, so no episode coherence test was computed and no
later block was substituted. Checksum, numeric-clock and full overlapping
waveform calibration checks passed. See `SPO2_WAVEFORM_PILOT_RESULTS.md` and the
documented independent-checker correction. This leaves the respiratory
mechanism unresolved; it supplies no mortality-relevant discovery.
