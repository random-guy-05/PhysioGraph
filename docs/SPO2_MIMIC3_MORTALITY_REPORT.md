# First-four-hour SpO₂ instability and in-hospital mortality

## Analysis summary

This report documents the CKD-corrected local re-analysis of the MIMIC-III v1.4 heart-failure ICU cohort and its mean-SpO₂ sensitivity. The full adjusted association with first-four-hour mean SpO₂ added was **RR 1.4125 (patient-cluster robust 95% CI 1.0472–1.9052; Wald p = 0.02369)**. The CKD-corrected model before adding mean SpO₂ was RR 1.5089 (95% CI 1.1470–1.9848; p = 0.00328); the previous 585.x-only CKD model was RR 1.5127 (95% CI 1.1497–1.9902; p = 0.00311). The mean-SpO₂ model retains the revised CKD flag and prior covariate definitions.

The analysis used 911 stays from 831 patients: 159 deaths and 752 survivors. In this model sample, mortality was 79/345 (22.9%) among exposed stays and 80/566 (14.1%) among unexposed stays. The mean-SpO₂ model had 11 parameters including the intercept and exposure, or 14.5 events per parameter. Expanded prior-coded CKD flags increased from 171 to 186 in this model sample; the other prior-code positive counts were unchanged.

These are local/precomputed results. The analysis was not run in Google Colab. The estimate is observational and should not be interpreted as a causal effect.

## Cohort and exposure

The source workflow selected adults with explicit heart-failure ICD-9 diagnosis (`428*`), the first ICU stay per admission, eligible ICU timing and shock criteria, and observation beyond ICU hour 4. The resulting clinical cohort contained 10,230 eligible stays; 1,654 had usable linked dense SpO₂ signal, and 1,597 were dynamics-eligible. Of those, 539 were exposed and 1,058 unexposed. This mortality model then used complete cases for the mortality flag, demographics, baseline lactate, mean SpO₂, exposure, and adjustment covariates, yielding 911 stays.

Exposure was at least one qualifying absolute SpO₂ transition of **≥4 percentage points** during the first four ICU hours, `[0, 240)` minutes. SpO₂ values were restricted to 50–100 and summarized in 15-minute median bins. A qualifying transition was between adjacent qualifying bins, with a gap no greater than 30 minutes. Dynamics eligibility required at least three valid bins and at least two qualifying forward transitions. The added level covariate was the mean of valid 15-minute SpO₂ medians during ICU minutes `[0, 240)`, from `02_signal_qc/signal_features_recomputed.csv` (`mean_binned_spo2`).

## Outcome and data sources

The outcome was in-hospital death for the index admission, taken from `HOSPITAL_EXPIRE_FLAG` in the raw MIMIC-III v1.4 `ADMISSIONS.csv.gz`. It is not ICU mortality or 30-day mortality. Because mortality is defined at the admission level, the lactate complete-window rule for lactate-rise endpoints does not apply here.

Age and sex were taken from `04_episode_lactate/episode_anchor_records.csv`. Baseline lactate was the last valid linked lactate at or before ICU minute 240, in mmol/L, from `03_landmark_lactate/lactate_outcomes.csv`. The mortality flag came from the raw admissions table. Prior-coded comorbidities were derived by linking the raw `DIAGNOSES_ICD.csv.gz` to prior admissions in the raw admissions table.

Raw input files used:

- `/Users/admin/Downloads/mimic-iii-clinical-database-1.4/ADMISSIONS.csv.gz`
- `/Users/admin/Downloads/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv.gz`

No patient-level rows are included in this report. The rerun and before/after results are saved under `/Users/admin/Downloads/physiograph_age_subgroup_extension/ckd_corrected_full_adjusted_models/`, with its executable analysis script in the parent directory.

## Adjustment set and model

The model adjusted for:

- Age in years
- Sex (`sex_male`)
- Baseline lactate in mmol/L
- Mean SpO₂ across valid 15-minute bins during ICU minutes `[0, 240)`
- Prior-coded chronic kidney disease (CKD)
- Prior-coded ESRD (display label; the retained code set includes dialysis-status codes)
- Prior-coded chronic pulmonary disease
- Prior-coded chronic liver disease
- Prior-coded diabetes

The modified Poisson model used a log link:

```text
log E(Yᵢ) = β₀ + β₁ instabilityᵢ + β₂ ageᵢ + β₃ sex_maleᵢ
            + β₄ baseline_lactateᵢ + β₅ mean_SpO2ᵢ + β₆ CKDᵢ
            + β₇ prior_ESRDᵢ + β₈ chronic_pulmonaryᵢ
            + β₉ chronic_liverᵢ + β₁₀ diabetesᵢ
```

The adjusted RR is `exp(β₁)`. Standard errors used a patient-clustered sandwich estimator by `SUBJECT_ID`, with finite-sample correction `G/(G−1) × (n−1)/(n−k)`. The reported confidence interval and p-value are Wald-based. The mean-SpO₂ model converged in 7 Newton/IRLS iterations and was full rank (11/11). The CKD-corrected model without mean SpO₂ also converged in 7 iterations and was full rank (10/10).

Prior-code positives in the 911-stay model sample under the revised definitions were CKD 186, ESRD 54, chronic pulmonary disease 206, chronic liver disease 24, and diabetes 222. Under the prior 585.x-only CKD flag, CKD positives were 171. Flags can overlap.

## Prior-code definitions

“Prior-coded” means a diagnosis in an earlier hospital admission for the same patient, with prior `ADMITTIME` strictly before the index admission. Diagnoses from the index admission were excluded. ICD-9 codes were normalized by removing decimal points.

| Covariate | Operational definition |
|---|---|
| CKD-related diagnosis proxy | Code families `585.x`, `403.x`, `404.x`, `582.x`, `250.4x`, and `753.12`, `753.13`, `753.14` |
| Prior-coded ESRD (display label) | Retained code prefix `5856`, or one of `V4511`, `V4512`, `V560`, `V5631`, `V5632`, `V568`; dialysis-status codes remain included by design |
| Chronic pulmonary disease | `4168`, `4169`; numeric prefixes 490–505; or exact codes `5064`, `5081`, `5088`, `515`, `51883`, `51884` |
| Chronic liver disease | Code prefix `571` |
| Diabetes | Code prefix `250` |

The expanded CKD-related proxy is a broader diagnosis-code flag, not a chart-validated CKD phenotype: this MIMIC adaptation uses any code in a strictly earlier admission and does not require repeated codes or confirm CKD with eGFR. In the corrected lactate models, prior-coded CKD increased from 69 to 75 of 388 stays at 12 hours and from 72 to 78 of 421 stays at 24 hours. Prior ESRD code-positive counts were unchanged at 22 and 20, respectively. The ESRD wording is a label change only; its operational code sets were not altered.

The selected families adapt code lists from an [EHR CKD registry](https://pmc.ncbi.nlm.nih.gov/articles/PMC3022247/) and an [eMERGE CKD phenotype](https://pmc.ncbi.nlm.nih.gov/articles/PMC4419875/). The source algorithms have their own repeated-code or clinical validation requirements; those requirements were not reproduced in this MIMIC prior-admission proxy.

## Interpretation and scope

In this complete-case sample, the in-hospital mortality association remained above the null after adjustment for first-four-hour mean SpO₂, baseline lactate, demographics, and prior-coded comorbidities. This supports the narrower statement that the mortality association persisted independently of measured SpO₂ level and baseline lactate in this adjusted model. The analysis does not establish causality. Selection into the model depends in part on baseline-lactate availability; there was no imputation or inverse-probability weighting in this mortality model.

This model did not adjust for VIS, MCS, SCAI, broader hemodynamics, or other organ-injury laboratory values, and it did not include an SpO₂-instability-by-lactate interaction. Those are separate analyses and should not be inferred from this estimate.
