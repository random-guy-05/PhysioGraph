# Primary lactate and mortality models adding early VIS

Local rerun using the fixed support-adjusted model inputs. Maximum concurrent VIS is measured only during ICU hours 0–4; hour-16 VIS is not used for these models.

| Outcome | Model | N | Events | Adjusted RR | Patient-cluster robust 95% CI | p |
|---|---|---:|---:|---:|---|---:|
| primary_lactate_12h | current_support_adjusted | 380 | 49 | 1.829631 | 1.042273–3.211779 | 0.035360 |
| primary_lactate_12h | current_support_adjusted_VIS_available_sample | 361 | 47 | 2.006536 | 1.142820–3.523030 | 0.015316 |
| primary_lactate_12h | plus_maximum_VIS_0_4h | 361 | 47 | 2.034661 | 1.157148–3.577628 | 0.013629 |
| primary_lactate_12h | sensitivity_plus_log1p_maximum_VIS_0_4h | 361 | 47 | 2.094332 | 1.197576–3.662586 | 0.009535 |
| in_hospital_mortality | current_support_adjusted | 900 | 156 | 1.430819 | 1.058789–1.933570 | 0.019710 |
| in_hospital_mortality | current_support_adjusted_VIS_available_sample | 828 | 140 | 1.374209 | 0.992754–1.902234 | 0.055345 |
| in_hospital_mortality | plus_maximum_VIS_0_4h | 828 | 140 | 1.351570 | 0.971272–1.880772 | 0.073923 |
| in_hospital_mortality | sensitivity_plus_log1p_maximum_VIS_0_4h | 828 | 140 | 1.380561 | 0.998102–1.909573 | 0.051353 |

## Covariates and score definition

- Existing covariates retained: age, sex, baseline lactate, expanded prior-coded CKD, model-specific prior-coded ESRD and chronic pulmonary disease, chronic liver disease, diabetes, 0–4-hour mean SpO₂, vasopressor infusion use, and documented mechanical ventilation.
- Addition: maximum concurrently summed six-drug VIS during [0,4) hours, as a continuous linear term. The log(1+VIS) model is a separate sensitivity for skewness; it does not replace the requested primary model.
- VIS = dopamine + dobutamine + 100×epinephrine + 100×norepinephrine + 10×milrinone + 10,000×vasopressin. All rates are µg/kg/min except vasopressin in units/kg/min; phenylephrine is excluded. Drug maxima are not summed across different times. Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC9891263/
- MV intervals active at each change point; CV latest numeric rate within the prior 60 minutes. Explicit stop records and unit conversions follow the existing VIS mapping. Explicit MV bolus/drug-push records are excluded. No rates or weights after hour4 are used. Dose outliers are retained.
- Documented absence of six-drug infusions with early infusion-table coverage gives zero. Missing dosing/units/required prior weight, a stale positive CV rate at an evaluated state, or no early infusion documentation gives unavailable VIS. The score is a maximum of documented EHR states, not an adjudicated continuous bedside maximum.
- Complete-case sample changes are shown separately. The prior model is refitted in the exact VIS-available subset before VIS is added. Both models use modified Poisson regression with SUBJECT_ID-clustered robust errors and the existing finite-sample correction; every fit also passes the existing independent likelihood/covariance calculation.
- All cohort, exposure, endpoint, comorbidity and prior support definitions remain fixed. The primary lactate endpoint still requires complete follow-up for events and non-events through its unchanged 12-hour post-landmark horizon.
- Support, VIS and instability share hours0–4; this estimates a conditional association and does not establish causation or independent predictive improvement. The lactate analysis has few events relative to its parameter count.
- Patient-level dose and VIS caches remain outside the public project.

## Denominator reconciliation

| Outcome | Prior N/events | VIS N/events | Removed events | Removed non-events | Removed exposed/unexposed |
|---|---:|---:|---:|---:|---:|
| primary_lactate_12h | 380/49 | 361/47 | 2 | 17 | 12/7 |
| in_hospital_mortality | 900/156 | 828/140 | 16 | 56 | 33/39 |
