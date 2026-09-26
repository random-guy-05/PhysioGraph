# SpO2 Variability Targeted Replication Protocol

Frozen UTC: `2026-09-07T21:09:05Z`

## Status

This is a user-authorized, explicitly post hoc follow-up to the completed 36-feature
MIMIC screen. SpO2 variability had adjusted RR 1.251 per SD (95% CI 1.104–1.418;
BH q=0.00539), but failed the original RR >=1.30 advancement gate. That original
decision remains unchanged. This follow-up tests whether the exact signal is stable
inside MIMIC and externally transports to eICU. It must not be described as a
preregistered winner or as confirmation of the original discovery protocol.

## Locked exposure and chronology

- Exposure: root-mean-square residual around an ordinary least-squares line fitted
  to hourly median SpO2 during ICU minutes 0 through 240.
- At least three hourly bins spanning at least two hours are required.
- Values are restricted to 50–100%. The exposure is clipped at the MIMIC 1st and
  99th percentiles and standardized with the MIMIC clipped mean and SD; these four
  constants are written before eICU association access.
- Landmark and eligibility time: ICU minute 360.
- Primary outcome: the same objective pressure/support plus hypoperfusion composite
  strictly after minute 360 through 960.
- Lead sensitivity: the same endpoint strictly after minute 480 through 960.

## MIMIC follow-up

The already-completed MIMIC cohort, feature, and outcome tables are reused. Report:
1. the original adjusted association;
2. 1,000 outcome-stratified bootstrap fits;
3. the same model additionally adjusted for the number of observed early SpO2 bins;
4. a complete-covariate-case sensitivity;
5. crude event risks by exposure quartile;
6. a density-stratified result for 3, 4, and 5 observed hourly bins when supported.

These are stability analyses, not a new discovery search. No alternative SpO2
variability formula, threshold, or outcome is permitted.

## eICU external validation

- Population: the existing validated adult, explicit-HF, first-ICU-per-hospital-
  encounter eICU cohort.
- Periodic `sao2` is the exposure source. Non-invasive aperiodic and nurse-charted
  cuff BP define hypotension. Positive-rate matched infusions define support; first
  non-removal IABP/Impella/ECMO documentation defines MCS.
- Lactate, creatinine, ALT, pH, and urine output use the same thresholds as MIMIC.
  Revised/entry offsets are used as availability times where supplied.
- The adjustment set is age, sex, latest HR, cuff SBP, cuff MAP, respiratory rate,
  SpO2 level, temperature, creatinine, lactate, lactate-observed status, BP density,
  and urine-output density, all measured by minute 240.
- Primary inference is modified Poisson regression with hospital-clustered robust
  variance, using the fixed MIMIC exposure transformation. HC0 is a sensitivity.
- External confirmation requires RR >1, a two-sided cluster-robust 95% CI excluding
  1, p<0.05, at least 50 exposed-frame events, at least 20 contributing hospitals,
  and at least 70% feature coverage. The lead-window estimate must have RR >1; its
  significance is supportive and is reported without changing the primary decision.
- Exactly one eICU exposure/outcome association is tested. No threshold, subgroup,
  hospital selection, or alternate outcome rescue is allowed after access.

## Interpretation

Successful external validation would establish transportable observational
association, not causality, cardiogenic-shock adjudication, clinical utility, or
incremental predictive value. Failure ends the follow-up as non-replicated.
