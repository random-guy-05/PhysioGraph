# Arterial acid–base support for the existing SpO2 hypothesis

Scope: only the existing dynamics-eligible HF cohorts in MIMIC and eICU.
The preceding waveform pilot was a verified feasibility failure; it did not
test or refute a respiratory mechanism. This is a different, larger-cohort
question: does a preceding arterial acid–base state alter the association
between the unchanged first-four-hour SpO2 instability and subsequent injury?
This stage measures support without selecting mortality, troponin outcomes,
or outcome-observation flags. Association testing requires a separately frozen
model plan after support is known. No breakthrough is implied by extraction.

## Prior evidence and claim boundary

Hypocapnia prognosis is already studied in [MIMIC acute HF](https://pmc.ncbi.nlm.nih.gov/articles/PMC11287307/).
Alkalosis prognosis has conflicting observational evidence: a
[2012 ICU study](https://pubmed.ncbi.nlm.nih.gov/22819039/) reported higher risk,
whereas a [2025 multicenter ED study](https://pubmed.ncbi.nlm.nih.gov/39898943/)
did not find higher adjusted mortality. The
[ALCALOTIC study](https://pubmed.ncbi.nlm.nih.gov/38709335/) also did not establish
an all-cause mortality association for metabolic alkalosis. None of these
observations establishes treatment benefit. Changes in oxygen affinity and
coronary vascular tone are known physiological mechanisms, not our discovery.
A [2026 SpO2 entropy study](https://doi.org/10.1113/EP093235) already emphasizes
interpreting variability in physiological context; merely renaming contextual
variability would not be a biological advance. Its non-MIMIC reference data
are not used in this project.

The candidate is a replicated interaction with the current instability/injury
signal, not another standalone pH/PaCO2 mortality predictor. Even a replicated
interaction would require independent mechanistic and therapeutic evidence
before a mortality-improving discovery could be claimed. Arterial pH/PaCO2
do not directly measure myocardial oxygen unloading.

## Fixed specimen, timing and quality rules

Keep the original cohort, exposure, 15-minute bins, and four-hour landmark.
Primary modifier support uses the last complete arterial gas collected in
[-1440,0] minutes relative to ICU admission, inside the same hospital encounter,
with results available by minute 240. Separately report availability by minute
0. A descriptive first-four-hour [0,240) gas count is retained to characterize
measurement support; it is not a fallback for a preceding-state analysis.
Do not use later outcomes to choose a gas or timing window.

MIMIC: select explicit matching subject_id and hadm_id, require non-null
specimen_id, and group by that specimen. Use dictionary-confirmed blood items
50820 (pH), 50818 (pCO2), 50821 (pO2), and 52033 (specimen type). Require
unambiguous explicit arterial text (ART., ART, ARTERIAL); exclude missing,
venous, mixed or conflicting specimen types. Do not infer arterial status
from oxygen values or the SpO2 signal. Missing hadm_id samples are not assigned
by the earlier CBC/troponin linkage amendment. Record their absence as a limit.
These identifiers agree with the
[official MIMIC blood-gas query](https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iv/concepts/measurement/bg.sql).

eICU: require exact pH, paCO2 and paO2 lab names with labTypeID=7, documented as
ABG in the [source schema](https://eicu.mit.edu/eicutables/lab/). Pair only at
the identical patient-unit and collection minute. These are ABG-labelled,
co-timed panels, not independently verified specimen IDs. Retain that limitation.
For each component choose the latest revision available by minute 240; reject
conflicting values at that revision instead of averaging conflicting results.

Require all three components. Pressure units must be explicitly mmHg (including
mm Hg) or kPa; convert kPa by 7.50061683. A recognized unit in either eICU unit
field is sufficient only if the other field is blank or agrees. Reject unknown
or conflicting pressure units. pH is dimensionless: accept blank, pH or units.
Use the source-query plausibility ranges consistently in both datasets:
pH 6.5–8.5, PaCO2 5–250 mmHg, PaO2 15–720 mmHg. Keep raw provenance in private
Drive Data. Do not estimate P50 from these measurements or calculate SaO2.

For support tables only, the candidate respiratory-alkalemia state is fixed as
pH>7.45 AND PaCO2<35 mmHg; all other complete panels form its complement.
This is a joint measured state, not a definitive diagnosis of a primary acid–base
disorder. Report primary availability and its four cells with the original
SpO2 exposure, source/units exclusions, and contemporaneous availability.
No outcome association or p-value is permitted in this feasibility stage.

Implement the actual workflow in the single primary Colab notebook. Execution
is labelled local until actually run in Colab. Keep original raw files and
earlier experiment outputs unchanged. The discovery goal remains active.
