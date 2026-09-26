# Arterial oxygen measurement support around the original SpO2 change

Specified September 5, 2026 before computing this pairing coverage. Prior
SpO2, acid-base, CVP and troponin results are already known. This is a new
measurement-feasibility audit in the original MIMIC/eICU HF cohorts, not a
new association test or a revision of any failed earlier gate.

Question: can two independent oxygen-tension samples resolve the two
SpO2 observations defining the first original instability episode?
One nearby gas cannot validate a change. PaO2 is oxygen tension, not measured
arterial saturation; even a supported pairing would require a separate
measurement-validity and analysis specification before interpreting biology.

Use existing validated `spo2_context.duckdb` bins and
`spo2_acid_base.duckdb` components. Reconstruct the first gap-qualified
absolute SpO2 change >=4 percentage points from the original 15-minute bins,
with a positive gap <=30 minutes, in the unchanged dynamics-eligible cohort.
Let a and b be the representative times of its preceding and current bins.
Assert exact agreement with original exposed labels and independently
reconstruct the pairs in Python.

Read only PaO2 identifiers and timestamps from the accepted components;
do not read oxygen values, troponin values or mortality. Existing MIMIC
components require explicit arterial specimen text, hospital linkage, units,
plausibility and valid revisions available by minute 240. eICU uses exact
paO2 names with ABG lab type 7, with the existing unit/revision checks; it
lacks equivalent specimen identifiers. Neither source proves how SaO2 was
measured. The cache covers hospital-bounded [-1440,240] ICU minutes.

Primary tolerance is +/-5 minutes around each SpO2 observation. Prespecified
coverage sensitivities are +/-10 and +/-15 minutes. For each tolerance,
require the complete windows inside the hospital-bounded cache interval.
Assign a gas to the a side only when its time is strictly below (a+b)/2,
and to the b side when it is at or above that midpoint. This prevents the
same sample from validating both states. Count encounters and distinct people
with either side and both sides. Use no later episode and no imputation.

Also calculate a permissive timing ceiling from all already-extracted raw
PaO2 rows, ignoring specimen, units, validity and result-availability rules.
This is an upper bound within this extraction, not a count of usable arterial
samples and not an upper bound on all records in either database. It shows
whether reprocessing those exclusions could plausibly solve timing scarcity.
No new raw source scan is part of this audit.

Require at least 385 distinct people with both sides at the primary tolerance
in EACH database before designing a confirmation-fraction analysis. This is
a conservative initial size corresponding approximately to a 95% binomial
interval half-width of five percentage points at p=0.5; clustering and
selection could require more. This is not power for mortality or a mechanism
test. Sensitivity windows cannot rescue a failed primary gate. If the gate
fails, close this cached-pairing specification without examining oxygen
changes. Absence of supported pairs is not absence of physiological changes.

Record cache SHA-256 fingerprints, exact SQL/Python patient-set agreement,
aggregate results and local/Colab execution identity in the single primary
notebook. No novel biological finding or mortality benefit is presumed.
