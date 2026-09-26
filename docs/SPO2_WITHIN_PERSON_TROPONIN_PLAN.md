# Within-person oxygen instability and consecutive troponin changes

Exploratory protocol frozen before the new interval counts, longitudinal vital extraction, or within-person coefficients. All prior PhysioGraph results and failures are known. Patient scope remains the original dynamics-eligible MIMIC/eICU HF cohorts. The new question extends observation through the already extracted 28-hour laboratory window; it does not redefine or replace the original four-hour exposure or its published estimates.

## Biological question and limits

When the same patient's charted oxygenation becomes unstable, is the next consecutive troponin change larger, after accounting for that patient's stable characteristics, prior troponin, oxygenation level, and observed hemodynamics? A positive replicated within-person association would strengthen the original hypothesis beyond between-patient differences. It would not identify injury onset: troponin release is delayed, testing is informative, and treatment can change both processes. It cannot establish a mortality-saving intervention. Laboratory values are exact-as-recorded, not adjudicated injury or infarction.

## Sequential, outcome-blind support stage

Use the existing source-audited interval groups in `troponin_emergence.duckdb`, preserving unknown groups and explicit bounds. Construct consecutive specimen pairs within each encounter/assay/unit; do not skip an intervening unknown, zero, or bounded result. Require both results exact-as-recorded and strictly positive, unit ng/ml, t0 >=0, t1 <=1680 minutes, and a separation of 240–720 minutes. The baseline must be fully reported by t1−240; the follow-up must be reported by minute 1680. These rules permit a four-hour oxygen exposure window entirely after the baseline was available. They use final recorded availability, not an invented initial report.

Select the assay with the most qualifying intervals per encounter, ties T before I. Keep encounters with at least two qualifying intervals and the smallest stay ID per person; this is a deterministic choice, not an assertion about chronology. Select before seeing any oxygen exposure or troponin change. Count timing-eligible people and intervals, without calculating log changes or associating values. If either dataset has fewer than 100 people, stop this specification before new raw vital scans and outcome changes. Do not widen times or relax this gate after counts are known.

## Longitudinal oxygen extraction, conditional on timing support

For timing-selected encounters only, read original eICU `vitalPeriodic.csv` and MIMIC `chartevents.csv` through minute 1680. Reuse the original item mapping and source warning exclusions. Keep SpO2, MAP and HR; original physiologic ranges are 50–100%, 20–200 mmHg and 20–250/min. Deduplicate same-time/concept values by median. For each t1, summarize [t1−240,t1) in 15-minute bins relative to the window start. SpO2 instability is any adjacent absolute change >=4 points across a forward gap <=30 minutes, using median raw timestamps as representatives. Require at least three observed SpO2 bins and two qualifying transitions, as in the original project. Retain all missingness/coverage counts; do not interpolate. This cannot resolve breath-by-breath physiology.

Features: mean SpO2, fraction of observed bins below 90%, log(1+number of deduplicated SpO2 timestamps), mean MAP and HR. A missing MAP/HR mean is not normal physiology. Keep only people with at least two qualifying oxygen windows. Require at least 100 people and 50 whose binary instability changes between windows in each database before fitting either primary model. No gate may be rescued with a different assay, encounter, or window after observing exposure.

## Frozen association model, conditional on both support stages

Outcome: natural log(troponin at t1 / troponin at t0), retaining recorded decimal magnitudes until ratio calculation. Predictors: binary instability, mean SpO2, hypoxic fraction, log readings, mean MAP, mean HR, interval duration in hours, t1 in hours, its square, and log baseline troponin. Include patient fixed effects. Median-impute missing MAP/HR within dataset with missing indicators. Remove only nuisance columns with no within-person variation; require full rank and at least ten observations per retained slope. No simplified model after a failed design gate.

Use within-person least squares with patient-clustered sandwich covariance, accounting for absorbed patient effects in residual degrees of freedom, t uncertainty with G−1 degrees of freedom, and Holm correction across the two primary exposure tests. Independently validate centering, coefficients and covariance, including a synthetic full-dummy regression comparison. Report exposure coefficient as a multiplicative ratio of troponin fold changes with a 95% interval, alongside the log coefficient. A positive replication requires both coefficients >0 and both Holm p<0.05. The statistical criterion is necessary for further work, never sufficient for biological novelty or causality.

This protocol does not examine mortality or select subgroups by mortality. Important unresolved time-varying confounders include treatment, changing congestion, ischemia, renal clearance and selective testing. Fixed effects do not remove these. This analysis is not a treatment-effect estimate or a mechanistic proof.

## Provenance and deliverable

Fingerprint protocol and input caches before counting. Independently reproduce consecutive pair and person selection in SQL and Python. Keep patient rows in private Drive Data; report aggregate support, exclusions, execution state and honest conclusions. Append runnable cells to the single Biological Discovery notebook. Label local incremental execution explicitly; do not claim a fresh Colab or uninterrupted full-notebook run.
