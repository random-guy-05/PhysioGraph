# Prespecified Analysis Plan — spo2_protocol v1.0

Frozen 2026-08-29, **before** any full-data MIMIC/eICU rebuild or model performance
inspection. Sensitivity choices below are fixed in advance so that no analysis
decision can be made in response to observed results. Deviations, if any become
necessary, must be documented in a changelog with rationale at the point of change.

Derived from PROJECT_GOAL.md and the implementation in
`src/physiograph/analysis/spo2_protocol.py` (protocol version `spo2_protocol_v1.0`).

## 1. Cohort eligibility, follow-up, and censoring policy

| Question | Frozen rule |
|---|---|
| Already on MCS at landmark (240 min) | Stay **retained**; MCS endpoint is defined as post-landmark **initiation** only (pre-landmark MCS is not an event; baseline MCS flag enters as a clinical control when extractable) |
| Already on vasoactive support at landmark | Stay **retained**; VIS rise endpoint uses `vis_delta > 0` (post-landmark max VIS − baseline VIS), i.e., **escalation on top of baseline**, not initiation |
| Discharged or transferred out before outcome window ends | Endpoint **unavailable (NaN)** for that horizon, not a non-event — enforced by availability indicators (`*_observed` columns) |
| Death before outcome ascertainment | Same as discharge: **unavailable (NaN)**, never silently a negative. Death is not currently an extracted event; if death extraction is added it will be reported separately, not folded into the composite |
| Repeat ICU stays per patient | All stays retained; `person_id`-qualified grouping keeps every stay of a patient inside one CV fold and inside one bootstrap replicate |
| Observation-window SpO2 data | Stays with zero plausible SpO2 values in [0, 240) are excluded from modeling (`spo2_plausible_count > 0` filter); count and fraction excluded reported in the cohort flow |

Endpoint windows are strictly `(240, 240 + horizon]` minutes; a measurement exactly
at the landmark belongs to baseline, never to the outcome.

## 2. Primary endpoint definitions (locked)

- **Lactate rise:** last post-window lactate − last pre-landmark lactate ≥ **0.5 mmol/L**.
  Rationale: ≈ the upper bound of routine analytic/biologic variation for serial
  lactate in this range, so the flag represents a rise unlikely to be measurement
  noise while remaining sensitive in a 12–24 h window. The continuous delta
  (`lactate_delta_*h`), max-value delta, and the ≥2.0 mmol/L crossing flag are
  retained as prespecified companions, not post-hoc alternatives.
- **VIS rise:** any increase > 0 in the drug-weighted vasoactive-inotropic score
  (coefficients: dopamine 1, dobutamine 1, milrinone 10, phenylephrine 10,
  epinephrine 100, norepinephrine 100, vasopressin 10,000) over the pre-landmark
  baseline within the window.

## 3. Prespecified sensitivity analyses (run after the primary analysis, all listed before results)

| # | Axis | Levels |
|---|---|---|
| S1 | Outcome horizon | 12 h vs 24 h (already parallel in protocol) |
| S2 | Lactate threshold | Δ ≥ 0.5 (primary); Δ ≥ 1.0; max-delta variant; continuous delta (linear model) |
| S3 | Sampling density | Above-median vs below-median `spo2_sampling_density_per_hr` strata |
| S4 | Mechanical ventilation | ventilated vs non-ventilated at landmark |
| S5 | Respiratory support / FiO2 | any support vs none; FiO2 strata where extractable |
| S6 | Baseline hypoxemia | below-90 fraction > 0 vs = 0 (normoxemic) strata |
| S7 | Baseline lactate | elevated (> 2.0 mmol/L) vs lower |
| S8 | Vasoactive status at landmark | on vasoactives vs off |
| S9 | Artifact filter | current plausibility (50–100%) vs stricter (>70%) vs raw |
| S10 | Instability definition | jump-threshold ≥3 vs ≥5 pp; sustained (≥2 consecutive) jump/low events; RMSSD-based signature |
| S11 | Repeated stays | all stays (primary) vs first stay per patient only |
| S12 | Negative control | missingness/sampling-only model vs physiological model comparison |

Not prespecified (report only as exploratory, if at all): any subgroup not listed
above; any threshold not listed in S2/S10.

## 4. Model specification rules (locked before fitting)

- Instability feature block is collinear by construction (SD/RMSSD/IQR/MAD/range/
  jump-rate are correlated summaries of one signal). The prespecified primary
  model is the full block **plus** a prespecified parsimonious signature
  (RMSSD + abrupt-jump rate + instability proxy score) reported alongside it;
  if the full block shows unstable/multicollinear fits, the parsimonious
  signature is the reported specification and the full block is labeled exploratory.
- Regularization: L2 (C = 1.0) fixed; `max_iter = 3000`.
- Events-per-variable: report events, feature count after one-hot expansion, and
  events-per-feature for every fitted model; flag any model below 10 EPV as fragile.
- Calibration: report ECE, calibration intercept/slope with the same OOF predictions;
  calibration uncertainty via the existing patient-level bootstrap.

## 5. Missing-data handling (locked)

- Fold-local median/mode imputation with missingness indicators (already enforced
  in code) — no global fitting.
- Report per-feature missingness tables for MIMIC and eICU separately.
- Confounder availability denominators (baseline lactate, creatinine, MAP, resp
  rate, FiO2, vent, RRT, demographics) are published before adjusted-model
  interpretation; an adjusted model run on a heavily selected complete subset is
  labeled as such.
- The missingness negative control (S12) is interpreted alongside every headline
  result: if sampling-only features approach physiological-model performance,
  the instability association is reported as likely confounded by measurement intensity.

## 6. Reporting package (frozen list)

Cohort flow diagram; Table 1 per dataset; endpoint availability/event-rate table
(the completeness audit); SpO2 feature distributions; BH-corrected association
table; nested model comparison; ΔAUROC/ΔAUPRC with CIs; calibration plots;
ROC/PR curves for adequate endpoints; lead-time distribution (full distribution,
not median only); missingness-control comparison; MIMIC→eICU transportability
table; sensitivity matrix S1–S12; per-endpoint conclusion classification
(robust / fragile / null / unavailable).

## 7. Claim scope

- Temporal-ordering outputs are **landmark ordering, not causal precedence**.
- No causal language about SpO2 instability independent of respiratory failure
  without the prespecified respiratory-support adjustments (S4–S6).
- eICU VIS: if quantitative VIS cannot be validated in eICU source data, external
  VIS validation is **excluded** and declared unavailable — pressor-presence is
  never substituted for quantitative VIS.
- Urine-output outcomes are reported as a **decline proxy** unless weight-
  normalized data support the KDIGO oliguria definition.
