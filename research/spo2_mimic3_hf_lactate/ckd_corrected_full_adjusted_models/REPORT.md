# CKD-corrected full adjusted lactate and mortality models with mean-SpO2 sensitivity

Local/precomputed MIMIC-III analysis; run completed 2026-09-26T19:26:18.423409+00:00.

## Results

| Outcome | CKD specification | N | Events | Patients | Exposed events / N | Unexposed events / N | Prior CKD positive | Crude RR | Crude RD | Adjusted RR (patient-cluster robust 95% CI) | p | EPV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary_lactate_12h | Prior 585.x only | 388 | 50 | 373 | 27 / 155 | 23 / 233 | 69 | 1.765 | 0.075 | 1.781 (1.044–3.040) | 0.0342 | 5.0 |
| primary_lactate_12h | Expanded prior-coded CKD | 388 | 50 | 373 | 27 / 155 | 23 / 233 | 75 | 1.765 | 0.075 | 1.772 (1.040–3.021) | 0.0355 | 5.0 |
| primary_lactate_12h | Expanded CKD + 0–4h mean SpO2 | 388 | 50 | 373 | 27 / 155 | 23 / 233 | 75 | 1.765 | 0.075 | 1.800 (1.026–3.161) | 0.0406 | 4.5 |
| primary_lactate_24h | Prior 585.x only | 421 | 49 | 404 | 25 / 174 | 24 / 247 | 72 | 1.479 | 0.047 | 1.497 (0.866–2.588) | 0.1480 | 4.9 |
| primary_lactate_24h | Expanded prior-coded CKD | 421 | 49 | 404 | 25 / 174 | 24 / 247 | 78 | 1.479 | 0.047 | 1.478 (0.859–2.541) | 0.1580 | 4.9 |
| primary_lactate_24h | Expanded CKD + 0–4h mean SpO2 | 421 | 49 | 404 | 25 / 174 | 24 / 247 | 78 | 1.479 | 0.047 | 1.428 (0.784–2.599) | 0.2444 | 4.5 |
| in_hospital_mortality | Prior 585.x only | 911 | 159 | 831 | 79 / 345 | 80 / 566 | 171 | 1.620 | 0.088 | 1.513 (1.150–1.990) | 0.0031 | 15.9 |
| in_hospital_mortality | Expanded prior-coded CKD | 911 | 159 | 831 | 79 / 345 | 80 / 566 | 186 | 1.620 | 0.088 | 1.509 (1.147–1.985) | 0.0033 | 15.9 |
| in_hospital_mortality | Expanded CKD + 0–4h mean SpO2 | 911 | 159 | 831 | 79 / 345 | 80 / 566 | 186 | 1.620 | 0.088 | 1.412 (1.047–1.905) | 0.0237 | 14.5 |

The lactate outcomes are the locked complete-window last-value rise endpoints: follow-up through the 12-hour or 24-hour horizon is required for both events and non-events. Mortality is the prior report’s index-admission in-hospital death endpoint (`HOSPITAL_EXPIRE_FLAG`), not a fixed 12-/24-hour death endpoint.

## Reconciliation

| Analysis | N | Events / non-events | Prior CKD 585.x only | Expanded prior CKD | Additional flags | Old RR | Revised RR | RR after mean SpO2 | Change after mean SpO2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary_lactate_12h | 388 | 50 / 338 | 69 | 75 | 6 | 1.781 | 1.772 | 1.800 | 0.0283 |
| primary_lactate_24h | 421 | 49 / 372 | 72 | 78 | 6 | 1.497 | 1.478 | 1.428 | -0.0501 |
| in_hospital_mortality | 911 | 159 / 752 | 171 | 186 | 15 | 1.513 | 1.509 | 1.412 | -0.0964 |

## Conclusion

After adding the 0–4h mean SpO2 covariate, the 12h lactate association (RR 1.800, 95% CI 1.026–3.161) and in-hospital mortality association (RR 1.412, 95% CI 1.047–1.905) remained above the null. These associations persisted independently of measured early mean SpO2 level and baseline lactate in these adjusted models. The 24h lactate interval (0.784–2.599) crosses 1, so this conclusion does not extend to the 24h endpoint.

## Fixed model and code definitions

- Exposure: saved first-four-hour indicator for any qualifying absolute SpO₂ transition ≥4 percentage points; same dynamics-eligible cohort (1,597 stays, 539 exposed).
- Lactate outcomes: the saved corrected primary endpoint, last lactate in the fixed post-landmark window minus the last pre-landmark lactate ≥0.5 mmol/L; no endpoint or window was rebuilt or changed.
- Added SpO2 level covariate: `mean_binned_spo2` is the mean of valid 15-minute SpO2 medians during ICU minutes [0, 240); it is available for all 1,597 dynamics-eligible stays.
- Lactate model adjustment: age, sex, baseline lactate, prior-coded CKD, prior-coded ESRD, prior-coded chronic pulmonary disease, prior-coded chronic liver disease, and prior-coded diabetes.
- Full adjusted models: age, sex, baseline lactate, revised prior-coded CKD, prior-coded ESRD, chronic pulmonary disease, chronic liver disease, and diabetes. Added-level sensitivity adds the saved 0–4h mean of valid 15-minute SpO2 medians (`mean_binned_spo2`) to each model.
- Revised prior-coded CKD proxy: ICD-9-CM 585.x, 403.x, 404.x, 582.x, 250.4x, and 753.12/.13/.14, in any strictly earlier admission. This expands the prior 585.x-only flag; no eGFR verification or repeated-code requirement is imposed, so it is a broader diagnosis-code proxy, not a chart-validated CKD phenotype.
- Code-family basis: the EHR CKD registry reports CKD, diabetic nephropathy, glomerulonephritis, and polycystic-kidney code families ([registry code table](https://pmc.ncbi.nlm.nih.gov/articles/PMC3022247/)); a separate electronic CKD phenotype includes hypertensive CKD codes 403.xx/404.xx ([eMERGE phenotype](https://pmc.ncbi.nlm.nih.gov/articles/PMC4419875/)). The downloaded MIMIC-III diagnosis dictionary confirmed the expected descriptions for all six families; the registry/phenotype inclusion algorithms are not reproduced here.
- ESRD is a label-only change. Existing ESRD-associated code sets were retained for each model; these include dialysis-status codes (V45.11 in lactate; V45.11, V45.12, V56.0, V56.31, V56.32, V56.8 in mortality) and were not redefined.
- Other lactate comorbidity code rules were unchanged: prior ICD-9 490–496 or 500–508 for chronic pulmonary disease, 571.x for liver disease, and 250.x for diabetes.
- Other mortality comorbidity code rules were unchanged from the mortality report: prior 416.8/416.9, 490–505, 506.4, 508.1, 508.8, 515, 518.83, or 518.84 for chronic pulmonary disease; 571.x for liver disease; 250.x for diabetes.
- Prior means diagnosis codes from strictly earlier hospital admission time(s); index-admission diagnoses are not used. Same-timestamp admissions are excluded from each other’s history.
- Modified Poisson log link; patient-cluster sandwich covariance by SUBJECT_ID with finite-sample correction; two-sided Wald p-values and 95% CIs. Convergence status and iteration counts are saved in the result table.
- Before-change validation: corrected 12-/24-hour lactate core and old full-adjusted estimates reproduce their saved results; old mortality reproduces the prior report (n=911, 159 deaths, 831 patients, RR 1.5127, 95% CI 1.1497–1.9902, p=0.00311). All rerun models converged and all design matrices were full rank; fit details are in the result table.
- The mean-SpO2 model is a supplemental covariate sensitivity; it does not replace the original full-adjusted estimate.

These are observational associations. “No prior code” is not confirmed absence of disease. The lactate extended models have low events per parameter, so their estimates are imprecise and vulnerable to overfitting; do not interpret p-values as proof of clinical utility or causality.

All outputs are local/precomputed; this run was not executed in Google Colab.
