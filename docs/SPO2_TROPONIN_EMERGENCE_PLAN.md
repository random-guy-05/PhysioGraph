# First-specimen troponin emergence after a reported upper bound

Frozen before this experiment's patient selection, event counts, or associations. Previous numeric troponin associations, reporting-template frequencies, and optimistic pairing counts are already known. This is an exploratory extension of the original HF SpO2 project, using only its MIMIC and eICU cohorts.

## Question and interpretation

Does original early SpO2 instability associate with the first subsequent troponin specimen clearly reaching 1.5 times the upper limit of a baseline result reported as `<U` or `<=U`? This is a conservative reporting-limit emergence endpoint. U is never imputed as an exact baseline. U is not assumed to be an assay's 99th percentile. A positive endpoint does not establish new injury onset, myocardial infarction, causation, or a mortality benefit. A negative endpoint does not exclude a 1.5-fold rise from the unknown true baseline.

## Frozen source and selection rules

Reuse the audited original source rows and the separately validated MIMIC comment whitelist. eICU: strict whole-text exact or inequality parser; numeric exact results must agree with the stored numeric field. Explicit inequalities are kept as intervals even when the numeric field is missing. MIMIC: accepted stored numeric values remain exact-as-recorded, with the known placeholder-text limitation; missing numeric values may receive only the previously whitelisted comment bounds. Reject nonfinite, negative, or greater-than-one-million magnitudes. Zero cannot qualify as a baseline reporting limit.

Group by encounter, specimen minute, assay, and original normalized unit. A group is interpretable only when every source row agrees on operator and decimal magnitude. Otherwise retain it as unknown, without taking an interval median. Availability is the maximum of specimen time and all reporting times; a missing reporting time makes the group unavailable. This conservatively uses the final recorded reporting/revision time, not an invented initial report time.

At the original four-hour landmark, choose the latest fully available group with specimen time in [0,240], across both troponin assays and all units, even if its value is uninterpretable. Break simultaneous ties by troponin T before I, then unit alphabetically. Qualify only an explicit positive upper bound in ng/ml. Thus a later exact, lower-bound, invalid-unit, or unknown available result blocks selection of an older upper bound. Retain the smallest stay ID among qualifying encounters per person, before inspecting follow-up; this deterministic selection is not claimed to be chronological. No requirement for survival or hospital presence through the endpoint horizon is added.

For the selected assay and unit, use the first specimen after minute 240 through 960 (primary: 12 hours after landmark); 1680 is a 24-hour sensitivity. Never skip an unknown or unavailable first group to use a later one. It must be reported by the respective horizon. Classify no specimen, unavailable report, and uninterpretable/overlapping interval separately; none is counted negative. Report full selection counts, assay/bound distributions, observation fractions, and first-specimen timing by exposure.

## Decimal endpoint rules

Let T = Decimal('1.5') * U. Exact P >= T is positive, otherwise negative. A lower-bound result `>L` or `>=L` with L >= T is positive; otherwise indeterminate. An upper bound `<V` with V <= T is negative; `<=V` requires V < T for a definite negative. Other upper bounds are indeterminate. These rules deliberately use the fixed threshold above U, not a reconstructed fold-change. Preserve source decimal strings to avoid binary floating-point boundary changes.

## Analysis and stopping rules

Show exposure-specific positive/negative/indeterminate/unavailable/not-remeasured counts for all baseline-eligible people. Among definite positive/negative results, report risks, Wilson 95% intervals, and the exposed-minus-unexposed risk difference with Newcombe 95% interval. These are descriptive estimates conditional on being measured and classifiable.

Adjusted inference requires, in each exposure arm, at least 50 definite observations, five positive and five negative outcomes. Fit the predeclared linear probability design: intercept, original exposure, age, male sex and sex-unknown flag, mean SpO2, hypoxic fraction, log(1 + SpO2 readings), mean HR, mean MAP, FiO2 maximum, ventilation and vasoactive documentation, log(U), and troponin-T indicator. Median-impute missing continuous values within each dataset with missing indicators; remove only constant nuisance columns. Require full rank and at least five observations in the smaller outcome class per nonconstant parameter. If any gate fails, do not fit or relax it. Otherwise use HC3 uncertainty and normal 95% intervals, with an independent least-squares/sandwich reconstruction. Holm correction covers the two primary dataset exposure p-values, treating an unfitted model as p=1. The 24-hour model is sensitivity only. No mortality outcome is accessed in this experiment.

Replication requires a positive adjusted association in both datasets with both primary Holm p-values <0.05; this would still be an exploratory association requiring observation-bias, temporal, and biological follow-up, not a paradigm-shifting discovery. Report failed gates and negative results equally. Do not change existing endpoint labels or original analysis caches.

## Validation and deliverable

Fingerprint source caches and this protocol before selection. Check decimal interval boundary counterexamples. Independently reproduce baseline and first-specimen choices in SQL, and endpoint labels with interval comparisons. Keep encounter-level records in the private Data directory; publish only aggregate JSON and a results document. Add the executed cells to the single primary Biological Discovery notebook. Distinguish local incremental execution from uninterrupted notebook execution or Colab.
