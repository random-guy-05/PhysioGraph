# Biological discovery protocol: residual danger after lactate improvement

User-directed scope amendment, 2026-09-05 UTC. The objective is a substantial
biological finding relevant to heart-failure mortality, comparable in clinical
importance to lactate-informed shock staging. Measurement-method results do
not satisfy this objective. No result is currently claimed.

## First falsifiable candidate

Test whether an early rise in circulating phosphate identifies ongoing
physiological deterioration in HF patients whose lactate normalizes, beyond
renal dysfunction, initial phosphate, and the lactate trajectory. The
biological possibilities include impaired renal elimination, cellular injury,
acid-base shifts and administered phosphate; serum phosphate does not directly
measure ATP depletion. Distinguishing those explanations is part of the
research, not an inference to make from a regression coefficient.

Phosphate alone is NOT novel: a 2025 MIMIC-IV cardiogenic-shock study already
examined its nonlinear mortality association
(https://www.medrxiv.org/content/10.1101/2025.05.20.25327987v1.full), and a 2023
cardiac-arrest study compared phosphate with lactate
(https://pmc.ncbi.nlm.nih.gov/articles/PMC10177342/). Normal lactate despite low
cardiac output in advanced HF is also established
(https://pubmed.ncbi.nlm.nih.gov/33734459/). This candidate concerns the joint
early trajectory, conditional on renal trajectory, with cross-database
replication and a decision-relevant residual mortality risk. That specificity
does not by itself establish novelty or clinical importance.

## Frozen first experiment

- Population: adults with explicitly documented HF, first ICU unit/stay per
  hospital encounter, admitted to ICU within 24 hours of hospital admission.
  Alive and still in the ICU at 12 hours. This deliberately excludes late ICU
  shock cases rather than using a discharge shock diagnosis as an early
  predictor. Retrospective discharge-code HF ascertainment in MIMIC is a
  cohort limitation, not evidence of real-time HF recognition.
- Baseline: first valid lab in ICU hours [0,4). Follow-up: last valid lab in
  [8,12). No value at or after 12 hours enters predictors. Actual collection
  times and reporting/revision times are retained; delayed availability
  requires a sensitivity analysis before any bedside prediction claim.
- Primary risk set: observed baseline lactate >=2 mmol/L and follow-up
  lactate <2 mmol/L, with phosphate and creatinine measured in both windows.
  Exposure: phosphate increase >=0.5 mg/dL. Reference: increase <0.5 mg/dL.
  Thresholds are investigator-specified before reading these associations.
- Primary outcome: death before hospital discharge, after the 12-hour
  landmark. Unknown discharge status is unavailable, never a survivor.
  Hospital mortality is used because it is shared across both databases.
- Primary contrast: adjusted mortality association of phosphate rise within
  lactate normalizers. Fit modified Poisson with patient-clustered robust
  standard errors, adjusting for age, sex, log1p baseline lactate, log1p
  follow-up lactate, log1p baseline phosphate, log1p baseline creatinine and
  creatinine change. Explicit complete-case denominators; >=100 deaths and
  >=20 deaths in each exposure arm are required to grade a fitted result.
  No coefficient selection, threshold optimization or outcome-driven covariates.
- eICU is discovery; transfer identical definitions and model specification to
  MIMIC. Both databases have been explored in earlier project work; this is
  a newly specified analysis, not an untouched prospectively held-out cohort.
- Secondary falsification: repeat among those with baseline creatinine <2
  mg/dL and creatinine rise <0.3 mg/dL; analyze continuous phosphate change;
  report four joint lactate/phosphate trajectory groups in the baseline
  hyperlactatemia population. Secondary p-values do not rescue a failed primary.
- RRT, phosphate supplementation, cardiac arrest, infection, respiratory
  interventions, pH and baseline organ injury require source-grounded
  sensitivity controls before a biological or actionable claim. Missing
  treatment information is not absence of treatment.

## Sequential decision gates

1. Audit assay identity/units, patient grouping, chronology, denominators and
   missingness from the ORIGINAL databases. No ignored repository caches.
2. Execute the locked primary contrast and publish every result, including
   sparse/null/reversed estimates. Stop promoting this candidate if it fails
   replication or renal/treatment sensitivity. Do not select the best result
   from additional time windows.
3. Only if it survives, compare patient/site-held-out mortality prediction
   with lactate and a faithfully reconstructable SCAI benchmark on the same
   patients. A partial lab-only adjustment is not SCAI adjustment. Require
   calibration and decision-curve utility, not AUROC alone.
4. For prioritization, seek >=5 percentage points adjusted absolute residual
   risk and a replicated risk ratio whose lower 95% limit exceeds 1.25.
   These are research prioritization thresholds, not validated clinical
   decision cutoffs or proof of a practice-changing discovery.
5. An observational association cannot prove phosphate manipulation or any
   other response reduces mortality. Mechanistic discrimination, genuinely
   independent clinical replication and prospective intervention evidence
   remain necessary for the user's ultimate target. Never call that target
   achieved because one retrospective p-value is small.

## Reproducibility

One executable `PhysioGraph_Biological_Discovery.ipynb` will contain the raw
extraction, checks, analyses and outputs. Local execution is labeled local.
Patient-level intermediate files belong under the original Drive Data area,
outside the repository; repository deliverables contain aggregate results.
Preserve original study artifacts and pre-existing edits. Log failed runs.
Keep this protocol and its hash with each output and record further
amendments before their analyses. No fake, simulated or precomputed data may
stand in for these clinical experiments.
