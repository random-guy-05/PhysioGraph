# INSPIRE external validation of SpO2 variability

Status: **FROZEN BEFORE INSPIRE ENDPOINT SUPPORT COUNTS OR ASSOCIATION FITTING**

INSPIRE v1.4.2 is used as a new, single-center external cohort. The frozen MIMIC and eICU results, the original discovery decision, and the separate NWICU feasibility result are not modified.

## Cohort and eligibility

- Use adults with an explicit three-character ICD-10-CM heart-failure code beginning `I50`.
- Define ICU minute zero from `operations.icuin_time` and choose the first ICU-bearing operation per hospital admission.
- Require ICU and hospital observability beyond minute 360 and survival past minute 360.
- Exclude patients with a frozen pressure/support or hypoperfusion event through minute 360.

## Frozen exposure

- Use `ward_vitals.spo2` during ICU minutes 0–240, restricted to 50–100%.
- Compute hourly medians, requiring at least three occupied hourly bins spanning at least two hours.
- Fit ordinary least squares SpO2 on hour and use the RMS residual.
- Clip and standardize only with the frozen MIMIC constants: lower `1.160311428702309e-14`, upper `4.300141942983509`, center `0.8776215724100497`, and SD `0.8227000137550243`.

## Frozen outcome

The original objective composite is retained. A qualifying event requires a pressure/support component and a hypoperfusion component within six hours, with both components strictly after minute 360 and no later than minute 960. The later-window sensitivity requires both components strictly after minute 480.

Pressure/support components:

- Two consecutive hourly cuff-BP bins with SBP <90 mmHg or MAP <65 mmHg.
- An administered IV record for norepinephrine, epinephrine, dopamine, dobutamine, milrinone, vasopressin, or phenylephrine. INSPIRE documents medication administrations but does not release a rate field; an administration row is the source-specific positive-use record.
- Recorded IABP or ECMO use.

Hypoperfusion components:

- Lactate ≥4 mmol/L, or lactate ≥2 with an increase ≥0.5 from the latest baseline.
- Creatinine increase ≥0.3 mg/dL or ≥1.5-fold from the latest baseline.
- ALT crossing above 200 IU/L from a baseline ≤200, or increasing ≥3-fold.
- pH <7.2 from a baseline ≥7.2.
- Six-hour mean documented urine output <30 mL/h with at least four documented hours; missing hours are not zero.

## Adjustment, inference, and gates

- Preserve the frozen adjustment set: age, sex, latest pre-240-minute HR, cuff SBP, cuff MAP, respiratory rate, SpO2 level, temperature, creatinine, lactate, lactate-observed status, BP density, and urine-output density.
- Use modified Poisson regression with the fixed MIMIC exposure transformation.
- INSPIRE is single-center. The user-authorized site gate waiver is retained, so HC0 robust variance replaces unavailable site-clustered inference and is explicitly labeled.
- The user-authorized 70% exposure-coverage waiver is retained and any shortfall is reported.
- The minimum 50 modeled primary events and exact-composite observability requirements remain binding.
- Confirmation requires primary RR >1, two-sided HC0 95% CI excluding 1, p<0.05, and later-window RR >1.
- No feature, outcome, threshold, time-window, subgroup, diagnosis, procedure, department, or care-path rescue is permitted.

This is an external cohort validation of a constructed physiological endpoint, not adjudicated cardiogenic shock and not evidence of causality or clinical utility.
