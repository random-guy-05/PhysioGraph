# Hemoglobin and susceptibility to SpO2-associated myocardial injury

Status: new biological candidate, feasibility protocol only. No hemoglobin
effect estimates, outcome interactions, or therapeutic benefits have been
computed. This is not a claim of novelty or a completed discovery.

## Question and prior evidence

Does hemoglobin measured before ICU SpO2 observation modify the association
between the established instability exposure and subsequent myocardial injury?
The biological rationale is reduced oxygen-carrying capacity: equal saturation
does not imply equal arterial oxygen content. A positive interaction could
motivate a susceptibility hypothesis, but would not measure oxygen delivery,
establish ischemia, or identify a transfusion-responsive population.

This extends the project's existing SpO2–troponin signal. The primary exposure
remains the first-four-hour absolute adjacent change of at least four percentage
points in 15-minute median bins, with positive gaps no longer than 30 minutes,
at least three bins, and two qualifying transitions. No MAP requirement is
added. First-drop and first-rise groups are retained for a later specificity
comparison; a recovery rise is not a valid negative control because it may
follow an unrecorded hypoxic episode.

The broad mechanism is not new. A [2022 COVID-19 cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447453/)
reported an interaction between hemoglobin and SpO2/FiO2 on mortality.
The [MINT HF analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999761)
already studied transfusion strategies in patients with MI, anemia and HF.
Neither is an additional patient dataset for this project. Current literature
searches did not identify the exact ICU SpO2-dynamics interaction, which is
insufficient to establish that it is novel. A replicated regression interaction
alone would not meet the user's paradigm-shifting completion requirement.

## Locked feasibility extraction

Use only the canonical MIMIC and eICU HF cohorts and raw sources in Drive Data.
Use the existing production-quality SpO2 bins, preserving their raw floating
point timestamps. Measurements at ICU minute 240 are excluded.

1. Extract explicitly identified hemoglobin laboratory measurements from ICU
   minutes -1440 through 0 inclusive. Measurements must belong to the same
   hospital encounter. MIMIC requires matching hadm_id and a charttime no earlier
   than the earlier of ED registration and inpatient admission. eICU requires
   matching patientunitstayid and a labresultoffset no earlier than
   hospitaladmitoffset. Do not assign unlinked MIMIC laboratories by proximity.
   The supplied MIMIC dictionary identifies item 51222 as Blood/Hematology
   Hemoglobin; use this item only, excluding blood-gas item 50811 and chemistry
   variants. For eICU, require labtypeid 3 and a trimmed case-insensitive exact
   name of Hgb, Hemoglobin or Haemoglobin. The [official table documentation](https://eicu.mit.edu/eicutables/lab/)
   identifies type 3 as hematology and type 7 as ABG. Audit all source names
   containing hgb or hemoglobin before applying this explicit whitelist.
2. Resolve hemoglobin source names and units from the source dictionary and
   aggregate laboratory names/units before deriving hemoglobin strata. Prefer
   hematology/CBC hemoglobin to a blood-gas estimate. Never substitute hematocrit
   divided by three. Preserve original values, units and source identifiers.
3. Normalize documented g/L to g/dL by division by ten and documented g/dL
   without conversion. Unknown or ambiguous units fail closed. Retain a broad
   plausibility range of 3–22 g/dL; report excluded counts. This is a quality
   rule, not an outcome-optimized threshold.
4. Select the latest pre-ICU CBC value, with same-time duplicates collapsed by
   median. If a database has no CBC-specific designation, document the source
   limitation before any outcome analysis. Do not select a value based on
   its magnitude or use post-ICU hemoglobin to improve coverage.
5. Produce aggregate coverage by database and established SpO2 exposure group,
   and availability of the existing assay-matched pre/post troponin endpoint.
   Do not expose mortality counts or troponin-rise frequencies across hemoglobin
   values or strata in this phase. Counts of observed endpoints are permitted;
   their outcomes are not used to tune eligibility.

## Feasibility decision

The candidate proceeds to an analysis specification only if each database has
at least 200 patients with pre-ICU hemoglobin and an observed 12-hour troponin
endpoint, including at least 50 with and 50 without the established SpO2
instability exposure. This is a minimal estimability screen, not a power claim.
An interaction power/precision assessment and a frozen confounder model are
still required before opening hemoglobin-by-outcome associations. Failure
closes this specification; do not substitute a post-ICU hemoglobin or enlarge
the exposure window to rescue it.

## Required next-stage protections

Before fitting an outcome model, freeze the estimand, continuous hemoglobin
functional form, interaction scale, multiplicity family, missing-data strategy,
support diagnostics and effect size needed to justify further investigation.
Explicitly address absolute saturation, hypoxemic burden, sampling, renal
function, circulatory state, respiratory support and vasoactive treatment.
Incomplete pressure/treatment data and transfusion/bleeding confounding must
not be hidden by a statistically significant coefficient. Discharge diagnosis
codes must not be represented as prospectively available covariates.

Use hospital mortality only as a separately specified clinical-relevance
endpoint after the model lock. No causal treatment or mortality-reduction
claim follows from this observational interaction. MIMIC and eICU have both
been explored already and are not pristine discovery/validation holdouts.

Patient-level outputs remain outside git in Drive Data. The eventual executable
workflow belongs in the single primary Colab notebook. All actual execution
must be labeled local unless it truly ran in Colab. The prior unsupported
circulatory-context and troponin-ordering experiments remain visible.
