# Prospective NWICU modified-endpoint amendment

Status: **FROZEN BEFORE ANY NWICU EXPOSURE–OUTCOME ASSOCIATION**

This amendment permits one NWICU analysis because the released NWICU schema does not contain continuous vasoactive-infusion rates or urine output. It does not change the frozen SpO2 exposure, its MIMIC transformation, eligibility chronology, laboratory thresholds, adjustment philosophy, or the prohibition on rescue analyses.

## Modified endpoint

The endpoint is the first qualifying pairing, within six hours, of an available pressure/support component and an available laboratory hypoperfusion component, with both components occurring strictly after ICU minute 360 and no later than minute 960.

Available pressure/support components:

- Sustained cuff-SBP hypotension using the existing frozen threshold, duration, and timing rules: cuff SBP below 90 mmHg in two consecutive hourly bins; event availability is the later of the second bin's completion and the two bins' recorded availability.
- Recorded mechanical circulatory support using the frozen procedure concept mappings to the extent exposed by NWICU. The NWICU dictionary exposes ECMO pump settings (`itemid=736876`) but no mapped IABP or Impella item.

Available laboratory hypoperfusion components:

- Lactate: value at least 4 mmol/L, or value at least 2 mmol/L with an increase of at least 0.5 mmol/L from the latest available baseline through minute 240.
- Creatinine: increase of at least 0.3 mg/dL or at least 1.5-fold from the latest available baseline through minute 240.
- Blood pH: baseline at least 7.2 followed by pH below 7.2.
- ALT: if baseline is at most 200 U/L, a later value above 200 U/L; otherwise at least three-fold the baseline.

Structurally unavailable components:

- Continuous vasoactive-infusion support.
- Oliguria.

The modified endpoint is not the original frozen composite and is not adjudicated cardiogenic shock.

## Exposure and cohort

- Adults with explicit ICD-9 `428*` or ICD-10 `I50*` heart failure.
- First ICU stay per hospital encounter, beginning within 24 hours of hospital admission.
- Alive and observable beyond ICU minute 360.
- Exclude available pressure/support or laboratory-hypoperfusion events through minute 360.
- SpO2 values 50–100%, charted and available within minutes 0–240.
- Hourly medians; at least three bins spanning at least two hours; RMS residual from OLS SpO2 on hour.
- Clip to `1.160311428702309e-14`–`4.300141942983509`; standardize with MIMIC center `0.8776215724100497` and SD `0.8227000137550243`.

## Inference and explicit waivers

- One modified-Poisson model for the primary modified endpoint and one later-window model requiring both components strictly after minute 480 through 960.
- HC0 robust covariance because NWICU is a single source with no released hospital/site identifier. Care unit is not a hospital and will not be used as a surrogate cluster.
- The user explicitly waived the original minimum 20-site gate and permitted analysis below 70% exposure coverage. Observed coverage must be reported prominently.
- The minimum 50 modeled-primary-event support requirement is retained. If it fails, no association is fitted.
- No alternate exposure, data-specific scaling, additional endpoint, threshold change, subgroup, care-unit selection, or rescue analysis.

## Interpretation rule

The modified endpoint is directionally supportive only if the primary RR is greater than 1, its two-sided HC0 95% CI excludes 1, p is below 0.05, and the later-window RR is greater than 1. Regardless of result, it cannot be called external validation of the original frozen composite.
