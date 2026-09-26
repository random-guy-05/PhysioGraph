# Frozen MIMIC-III SpO2-variability replication

This attempt applies the existing frozen SpO2-variability exposure, eligibility,
pressure/support-plus-hypoperfusion endpoint, chronology, adjustment set, and
confirmation rule to MIMIC-III v1.4. It is locked before endpoint support or any
SpO2-outcome association is read.

MIMIC-III and the MIMIC-IV development cohort originate from the same hospital
and may include overlapping source patients and admissions. This run is therefore
a historical database-version replication, not an independent-site external
validation. A positive result cannot establish independent transportability.

## Frozen population and exposure

- Adults with an explicit heart-failure ICD-9 diagnosis (`428*`).
- First ICU stay per hospital admission; alive and observable after ICU minute 360.
- Exclude qualifying pressure/support or hypoperfusion through minute 360.
- SpO2 item 646, restricted to 50–100%, during minutes 0–240.
- Hourly medians; at least three bins spanning at least two hours.
- OLS RMS residual, clipped and standardized with the frozen MIMIC-IV constants.

## Frozen endpoint

Pressure/support is sustained cuff hypotension (SBP item 455 or MAP item 456),
a positive CareVue vasoactive-infusion rate, or recorded CareVue IABP/ECMO/VAD
support. Hypoperfusion is qualifying lactate, creatinine, pH, ALT, or six-hour
oliguria. The first cross-domain pair no more than six hours apart must occur
strictly after minute 360 and no later than minute 960. The sensitivity window
starts strictly after minute 480.

The CareVue urine mapping excludes pigtail-drain output and irrigation fields.
The release's MetaVision input and procedure files contain headers only, so the
endpoint uses the prospectively locked CareVue mappings.

## Gates and inference

The user's prior waivers for 70% exposure coverage and 20 contributing sites are
carried forward and must be reported. The retained gate of at least 50 primary
events among exposure-observed model candidates is binding. If it fails, no
association may be fitted or inspected. If it passes, fit exactly the primary and
later-window modified-Poisson models with HC0 covariance (one released hospital),
using the frozen adjustment set and MIMIC-IV exposure scale. No rescue analysis is
allowed.

All execution is LOCAL in a standalone runtime outside the project `.venv`; it is
not Google Colab execution.
