# Primary models with early SpO2 level and organ support

Local analysis-only model rerun. Cohort, exposure, endpoint labels, windows, and prior-comorbidity definitions are fixed. New support covariates alone were reconstructed from the validated original clinical sources.

| Outcome | Model | N | Events | Adjusted RR | Patient-cluster robust 95% CI | p |
|---|---|---:|---:|---:|---|---:|
| primary_lactate_12h | original_full_adjusted | 388 | 50 | 1.772152 | 1.039593–3.020917 | 0.035493 |
| primary_lactate_12h | plus_mean_SpO2 | 388 | 50 | 1.800459 | 1.025509–3.161019 | 0.040588 |
| primary_lactate_12h | original_full_adjusted_matched_support_sample | 380 | 49 | 1.852454 | 1.085494–3.161312 | 0.023770 |
| primary_lactate_12h | plus_mean_SpO2_matched_support_sample | 380 | 49 | 1.860286 | 1.057917–3.271204 | 0.031122 |
| primary_lactate_12h | plus_mean_SpO2_vasopressors_ventilation | 380 | 49 | 1.829631 | 1.042273–3.211779 | 0.035360 |
| primary_lactate_12h | sensitivity_plus_mean_SpO2_any_vasoactive_ventilation | 380 | 49 | 1.836005 | 1.046031–3.222575 | 0.034278 |
| in_hospital_mortality | original_full_adjusted | 911 | 159 | 1.508854 | 1.147033–1.984809 | 0.003275 |
| in_hospital_mortality | plus_mean_SpO2 | 911 | 159 | 1.412479 | 1.047212–1.905152 | 0.023686 |
| in_hospital_mortality | original_full_adjusted_matched_support_sample | 900 | 156 | 1.500519 | 1.136899–1.980437 | 0.004154 |
| in_hospital_mortality | plus_mean_SpO2_matched_support_sample | 900 | 156 | 1.406128 | 1.038663–1.903597 | 0.027422 |
| in_hospital_mortality | plus_mean_SpO2_vasopressors_ventilation | 900 | 156 | 1.430819 | 1.058789–1.933570 | 0.019710 |
| in_hospital_mortality | sensitivity_plus_mean_SpO2_any_vasoactive_ventilation | 900 | 156 | 1.430265 | 1.058530–1.932547 | 0.019784 |

## Definitions and validation

- Original adjustment: age, sex, baseline lactate, expanded prior-coded CKD, prior-coded ESRD, chronic pulmonary disease, chronic liver disease, and diabetes. The existing model-specific ESRD and lung code sets are preserved.
- SpO2 level: mean of valid 15-minute SpO2 medians during ICU minutes [0,240), identical to the previously reported mean-SpO2 sensitivity; not a single admission reading.
- Vasopressor: any positive-rate norepinephrine/Levophed, epinephrine, dopamine, vasopressin, or phenylephrine/Neosynephrine infusion in [0,4) hours. CareVue uses chart times; MetaVision uses infusion-interval overlap, including infusions already running at ICU entry. Rewritten MetaVision rows are excluded. Pure dobutamine/milrinone are excluded from the main pressor indicator and included only in the any-vasoactive sensitivity.
- Mechanical ventilation: any first-four-hour evidence from the MIMIC-III ventilation-setting classification, explicit positive numeric/text mechanically-ventilated checkbox (226260), or an overlapping noncancelled invasive-ventilation procedure (225792). Error-marked chart records and rewritten/cancelled procedures are excluded. Noninvasive procedure (225794) alone does not set this flag. This measures documented ventilation evidence; no procedure record alone is not treated as a negative.
- Ventilation-setting mapping: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/ventilation_classification.sql
- Both original RRs/CIs and exact outcome/exposure counts were reproduced before adding support. Stays without valid early chart documentation or positive invasive-procedure evidence retain unavailable ventilation status and are excluded from the expanded complete-case model. Matched-sample original and mean-SpO2-only models distinguish sample changes from the added adjustment. Every model has a full-rank design, converged, and passed an independent likelihood and patient-cluster sandwich-CI calculation.
- Modified Poisson with log link; sandwich covariance clustered by SUBJECT_ID, same finite-sample correction and two-sided Wald intervals as the original models. The lactate outcome still requires complete follow-up through its fixed horizon for events and non-events. Mortality remains index-admission in-hospital mortality.
- The expanded lactate model has 49 events and 13 parameters (3.77 events/parameter). Its uncertainty and risk of overfitting require caution. Support and instability are both measured in hours 0–4; this adjustment establishes a conditional association, not a causal effect or demonstrated predictive improvement.
- Patient-level support caches remain outside the public project. All result tables here are aggregate.

## Complete-case denominator reconciliation

| Outcome | Original N/events | Expanded N/events | Removed events | Removed non-events | Removed exposed/unexposed |
|---|---:|---:|---:|---:|---:|
| primary_lactate_12h | 388/50 | 380/49 | 1 | 7 | 1/7 |
| in_hospital_mortality | 911/159 | 900/156 | 3 | 8 | 2/9 |
