# Venous-pressure antecedent: locked timing feasibility

Scope is the original MIMIC/eICU HF SpO2-instability cohort. This is a prerequisite
for a possible biological test, not a discovery or a substitute for mortality
validation. It asks whether the first original SpO2 episode has enough preceding
and subsequent CVP measurements to distinguish temporal sequences. No pressure
change, association with troponin, or mortality estimate will be calculated here.
Earlier experiments and marginal clinical outcomes have already been examined;
neither database is a pristine validation set.

## Biological rationale and novelty limits

Venous congestion and injury are established: [2009 invasive ADHF study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856960/),
[2021 congestion/troponin study](https://pubmed.ncbi.nlm.nih.gov/34534666/),
and [2024 experimental venous congestion study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884348/).
The [1979 coronary venous hypertension experiment](https://pubmed.ncbi.nlm.nih.gov/759726/)
already investigated potentiation of ischemic injury. A [2026 LVAD study](https://pubmed.ncbi.nlm.nih.gov/41655605/)
related decreasing CVP to increased hypoxic ventilatory sensitivity in 14 patients.
We cannot claim to discover these broad mechanisms. A temporally resolved link
between venous-pressure changes, the current charted SpO2 phenotype, and later
injury would be a distinct question, but CVP is not coronary venous pressure,
and neither causality nor treatment benefit follows from such an association.

## Frozen sources and timing support

- Preserve all 11,452 eICU and 4,711 MIMIC original dynamics encounters.
- Reconstruct the first original absolute SpO2 jump >=4 percentage points from
  the existing 15-minute median bins in [0,240). Require consecutive occupied
  bins' representative times to differ by >0 and <=30 minutes, as in the
  project. The first qualifying pair defines onset a and end b; retain the
  first pair even if its CVP timing is inadequate. No later episode substitution.
- Extract only first-four-hour CVP records. MIMIC uses item 220074, explicit
  matching stay, subject and hospital admission, warning not equal to one,
  numeric finite value, explicit mmHg unit, storetime no later than minute 240,
  and a broad measurement-quality range -5 to 50 mmHg inclusive. Do not treat
  CVP alarm settings or central venous saturation as pressure.
- eICU uses vitalPeriodic.cvp linked by patientunitstayid. The official
  [table documentation](https://eicu.mit.edu/eicutables/vitalperiodic/)
  describes unvalidated five-minute monitor medians; it does not provide a
  per-record pressure unit. Count finite native values without assigning units
  or assuming that apparently plausible magnitudes validate them. This is an
  optimistic timing upper bound, not a validated physiological sample. The
  local nurseCharting table is absent at initial source inspection.
- Deduplicate exact stay/time pairs. Timing support requires a>=60 and b<180,
  at least two distinct CVP times in [a-60,a), and at least two in (b,b+60].
  In each window, require a span >=15 minutes. Require the latest preceding
  measurement within 30 minutes before a and the earliest subsequent measurement
  within 30 minutes after b. All raw times remain in [0,240).
- Report sequential attrition: original exposed, full windows, any preceding
  and subsequent measurement, two per window, span requirement, proximity
  requirement. Count encounters and distinct people; do not inspect pressure
  directions, MAP changes, biomarker values or death labels.
- A subsequent biological analysis requires at least 100 distinct people with
  complete timing in each database AND established comparable pressure units.
  The count requirement is only a floor for a later power analysis, not a claim
  of adequate mortality power. Failure closes this specification; do not widen
  windows, choose later episodes or pool databases to rescue it.
- Independently reconstruct first SpO2 pairs and timing-qualified patient sets
  in Python from deduplicated times and compare exact sets with SQL. The code
  must reconcile first-episode presence to original exposure labels.

Keep individual records outside the repository under the user's /Data folder.
Record source sizes and modification times before and after scanning, hash the
plan and extraction code, and preserve real local execution outputs in the one
primary notebook. Do not describe local execution as Google Colab execution.
