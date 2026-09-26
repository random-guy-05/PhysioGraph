# Frozen third-cohort validation protocol template

Status: **FUTURE, NOT EXECUTED**  
Decision basis: `THIRD_COHORT_STRONGLY_JUSTIFIED`  
Created after the completed MIMIC/eICU audit. No third-cohort outcome was inspected.

## Eligibility

- Adults with explicit heart failure, using the first ICU stay per hospital encounter.
- Alive and observable beyond ICU minute 360.
- Exclude pressure/support or hypoperfusion events through minute 360 using the same frozen component definitions.
- Require at least three hourly SpO2 bins spanning at least two hours in minutes 0–240.

## Frozen exposure

- Restrict SpO2 to 50–100%.
- Compute hourly medians during minutes 0–240.
- Fit ordinary least squares SpO2 on hour within each stay.
- Exposure is the root-mean-square residual around that line.
- Clip at `1.160311428702309e-14` and `4.300141942983509`.
- Standardize with MIMIC center `0.8776215724100497` and SD `0.8227000137550243`.
- No dataset-specific centering, scaling, threshold optimization, or alternative SpO2 feature.

## Outcomes and chronology

- Primary: the constructed pressure/support-plus-hypoperfusion composite strictly after minute 360 through 960, with domains paired within six hours.
- Later-window sensitivity: the same endpoint strictly after minute 480 through 960.
- This is not adjudicated cardiogenic shock.

## Adjustment and missingness

- Preserve the clinical adjustment philosophy: age, sex, latest pre-240-minute HR, cuff SBP, cuff MAP, respiratory rate, SpO2 level, temperature, creatinine, lactate, lactate-observed status, BP density, and urine-output density.
- Missing continuous covariates use the same deterministic preparation as the frozen validation; missing outcomes remain missing rather than negative.
- Use the MIMIC exposure transform without refitting it.

## Inference and support gates

- Modified Poisson RR per frozen MIMIC SD.
- Cluster-robust variance by hospital/site; HC0 only as a labeled sensitivity.
- Minimum 70% exposure coverage, 50 modeled primary events, and 20 contributing hospitals.
- Confirmation requires RR >1, two-sided clustered 95% CI excluding 1, p<0.05, and later-window RR >1.
- Stop without rescue if feasibility gates fail or the primary confirmation rule fails.
- No subgroup, hospital, endpoint, time-window, feature, or threshold search.

## Future secondary benchmark

Run only if basic exposure/outcome feasibility passes, on identical patients for every comparison:

1. Clinical baseline
2. Clinical + baseline lactate
3. Clinical + frozen SpO2 variability
4. Clinical + baseline lactate + frozen SpO2 variability

Pre-specify a lactate-normal subgroup as baseline lactate <2.0 mmol/L. This benchmark is secondary, does not alter the primary RR decision, and must not be run in unavailable data.
