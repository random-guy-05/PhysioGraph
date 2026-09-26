# Masked Pulsatility Source Forensic Audit

## Purpose

This is an outcome-blind extraction audit, not a new biological experiment and
not an amendment to the frozen masked-pulsatility phenotype.  It determines why
eICU blood-pressure coverage was 1,103 of 12,028 HF stays and distinguishes a
source-availability failure from biological non-replication.

## Frozen audit rules

- Use the production HF cohorts and original raw files.
- Audit the first four ICU hours only.
- Keep invasive, cuff, and nurse-charted BP sources separate.
- Pair SBP and DBP only at the same timestamp and within the same source.
- Use hourly medians and require the first valid bin to have PrPP >= 0.25 for an
  incident crossing.
- Report pair-only coverage separately from the original analysis requirement
  for a simultaneous SBP/DBP/MAP triplet.
- Never substitute zero for an unavailable source.  Report unavailable counts
  as null with an explicit reason.
- Describe the original frozen-analysis attrition without changing its source
  selection, thresholds, cohort, windows, or eligibility rules.
- At compensated crossing anchors, partition latest eligible pre-anchor lactate
  into no measurement, <2 mmol/L, and >=2 mmol/L.
- Count subsequent vasoactive/inotropic initiation within 6 and 12 hours only
  as an endpoint-availability audit among crossings free of prior support.  Do
  not inspect lactate outcomes, mortality, treatment effects, or associations.

## Required conclusions

The audit must state whether `vitalAperiodic.csv` and `nurseCharting.csv` exist,
whether each is configured, mapped with SBP and DBP, and included in the prior
notebook.  It must provide source-specific attrition for eICU and MIMIC and
classify the prior result as one of:

- `invalid_for_intended_routine_bp_hypothesis_missing_sources`
- `routine_sources_present_but_low_crossing_prevalence`

No Phase B analysis or replacement hypothesis is authorized by this audit.
