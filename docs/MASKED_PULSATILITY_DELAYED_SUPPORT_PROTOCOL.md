# Masked Pulsatility → Delayed Hemodynamic Support: Locked Outcome Protocol

## Freeze and scope

- Protocol freeze UTC: `2026-09-07T16:19:32Z`.
- Development database: MIMIC-IV.
- Locked external-confirmatory database: eICU-CRD.
- This protocol is frozen before reading any post-anchor routine-cuff support outcomes.
- The source-repair exposure and control artifacts are reused without changing the PrPP threshold, BP pairing, hourly aggregation, anchor, or pseudo-anchor assignment.
- No outcome, window, threshold, drug, covariate, or model may be added or selected after the MIMIC outcome is read.
- MIMIC is executed first. The identical hashed code/config is then executed in eICU. Any effect-independent bug fix requires a documented new lock and rerunning both databases from MIMIC onward.

## Scientific question and claim boundary

The primary question is whether an incident loss of proportional pulse pressure during otherwise preserved routine-cuff SBP and MAP predicts new continuous vasoactive/inotropic support more than two hours later, beyond current conventional pressure and its pre-anchor trajectory.

PrPP is `(SBP - DBP) / SBP`. It is not treated as a direct cardiac-output measurement. The study is associational/predictive, does not adjudicate cardiogenic shock, and does not estimate a treatment effect or survival benefit from acting on PrPP.

## Population and locked anchors

- Population: the existing harmonized adult heart-failure ICU cohorts: 17,758 MIMIC stays and 12,028 eICU stays.
- Exposure: the first eligible source-consistent routine-cuff transition from hourly PrPP `>=0.25` to `<0.25` during ICU minutes `[0,240)`.
- MIMIC BP source: `nibp_mimic_chartevents`.
- eICU BP source: the deduplicated `nibp_cuff_union` of `vitalAperiodic` and exact-mapped `nurseCharting` NIBP.
- Exposure-anchor requirements: SBP `>=90 mmHg`, direct MAP `>=65 mmHg`, no validated vasoactive/inotropic infusion at or before the anchor, no prior MCS, and alive/observed at the anchor.
- Controls: the already constructed outcome-blind eligible control pool with PrPP `>=0.25`, preserved SBP/direct MAP, no previous crossing in the locked first-four-hour exposure window, no prior vasoactive/inotropic infusion or MCS, and alive/observed at the deterministic pseudo-anchor.
- Primary pre-landmark counts: MIMIC 1,123 exposed and 9,878 controls; eICU 577 exposed and 9,147 controls.
- A stay contributes one anchor. The locked control pool already excludes every control with a known cuff crossing anywhere in the first-four-hour exposure-construction window; the corresponding contamination count is audited. Later cuff measurements were not part of the locked source-repair pipeline, so no outcome-dependent reclassification is permitted.

## Validated continuous-infusion outcome definition

The primary outcome is the first new start of any validated continuous vasoactive/inotropic infusion.

### MIMIC-IV

Source: actual `inputevents.csv` administration intervals, not orders. Retain positive-rate rows with a valid start and end and one of these fixed item IDs:

| Item ID | Canonical drug |
|---:|---|
| 221906 | norepinephrine |
| 221289 | epinephrine |
| 221662 | dopamine |
| 221653 | dobutamine |
| 221749 | phenylephrine |
| 222315 | vasopressin |
| 221986 | milrinone |

Within a stay and canonical drug, contiguous/restarted intervals separated by at most five minutes are one episode. The first episode start across drugs is the any-support initiation. A stay with a positive-rate validated infusion active at or initiated before the anchor is ineligible.

### eICU-CRD

Source: actual positive-rate `infusionDrug.csv` rows, using `drugrate` and then `infusionrate` as fallback. Fixed case-insensitive normalized aliases:

- norepinephrine: `norepinephrine`, `noradrenaline`, `levophed`
- epinephrine: `epinephrine`, `epinepherine`, `adrenalin`
- dopamine: `dopamine`
- dobutamine: `dobutamine`, `dobutrex`
- milrinone: `milrinone`, `primacor`, `primacore`
- vasopressin: `vasopressin`, `pitressin`
- phenylephrine: `phenylephrine`, `neo-synephrine`, `neosynephrine`, `neosynsprine`

Exact duplicate stay/drug/offset rows are collapsed. The first positive-rate charted offset across canonical drugs is the any-support initiation. A stay with any validated positive-rate row at or before the anchor is ineligible. No rate threshold beyond `rate > 0` is imposed.

## Temporal design and risk sets

- Concurrent/blanking window: `>0` through `<=2 h` after anchor.
- Primary delayed window: `>2` through `<=12 h` after anchor.
- Fixed shorter sensitivity: `>2` through `<=6 h`.
- Descriptive bins: `>0–1`, `>1–2`, `>2–4`, `>4–6`, and `>6–12 h`.
- Primary analysis is a two-hour landmark. Patients must be alive/observed and free of validated support through anchor `+120 min`. Starts in `(0,120]` are reported as concurrent deterioration and removed from the primary risk set in both groups.
- An event at exactly `120 min` is blanking, not primary. An event after `120 min` and at or before `720 min` is primary.
- ICU discharge or death after two-hour landmark without prior support is retained as no observed ICU support initiation; discharge/death is a competing end of ICU observation, not silently treated as missing.

## Frozen pre-anchor covariates

Every covariate is computed from values timestamped at or before the anchor. Post-anchor physiology is prohibited.

Primary full-cohort adjustment:

1. exposure indicator;
2. age and male sex, plus explicit missingness indicators;
3. current cuff SBP and direct MAP from the anchor hourly bin;
4. last HR at or before anchor within 60 minutes;
5. OLS slopes of SBP, MAP, and HR over `[anchor-120 min, anchor]`, requiring at least two distinct hourly points;
6. prior PrPP from the latest source-consistent cuff hourly bin strictly before the anchor;
7. last respiratory rate and SpO2 at or before anchor within 60 minutes;
8. last creatinine available within 24 hours before anchor; MIMIC availability requires `storetime <= anchor`, and eICU requires revised-result offset `<= anchor`;
9. cuff BP measurement count in `[anchor-240 min, anchor]`;
10. anchor time in hours from ICU admission.

Continuous covariates are winsorized at the pooled-within-database 1st/99th percentiles, median-imputed within database, accompanied by missingness indicators when any values are missing, and standardized within database. Sex is encoded as male, female reference, and unknown indicator. No outcome-driven variable selection is allowed. HR, SBP, and MAP are primary; shock index is reserved for the fixed comparator sensitivity and is not included simultaneously in the primary model.

Respiratory-support state is not included in the primary model because the currently validated sources do not provide an identical cross-database point-in-time three-state definition: MIMIC supplies administration/procedure intervals, whereas eICU treatment rows are documentation events without reliable active intervals. Current respiratory rate and SpO2 remain in the primary model. This omission was fixed before outcomes.

Multicollinearity is reported using the design-matrix condition number and variance-inflation factors. A VIF above 10 or condition number above 100 after standardization is flagged, not used to drop variables.

## Primary and falsification models

- Unadjusted risks, risk difference, risk ratio, and 95% confidence intervals are reported by database.
- Primary adjusted model: modified Poisson GLM with log link. MIMIC uses HC0 robust covariance; eICU uses hospital-clustered robust covariance with finite-sample correction.
- Adjusted RR is `exp(beta_exposure)`. Marginal adjusted risks, RD, and RR use model standardization over the observed covariate distribution and delta-method covariance from the fitted robust covariance matrix.
- Critical SBP/MAP falsification is the full primary adjusted model above. `effect_survives_sbp_map_adjustment` requires eICU adjusted RR `>1`, its 95% CI lower bound `>1`, and adjusted RR `>=1.25`.
- Conventional-warning comparator replaces HR/SBP/MAP level-and-slope terms with shock index and its two-hour slope; it is a sensitivity only.
- Static-versus-dynamic falsification fits: (A) the frozen primary covariates with continuous anchor PrPP but no crossing indicator, and (B) the same model plus the crossing indicator. `dynamic_beats_static` requires the crossing coefficient in model B to have RR `>1`, 95% CI lower bound `>1` in eICU, and model-B AIC at least 2 lower than model A. No threshold is optimized.

## Baseline lactate and observation process

- Lactate is not required for the primary cohort and missing lactate is never coded as normal.
- Measured-lactate analysis uses the latest valid lactate event in `[anchor-240 min, anchor]` that was available by the anchor, its continuous value, and lactate-to-anchor lag, in addition to the fixed primary covariates.
- The probability of having baseline lactate measured is modeled from all non-lactate pre-anchor primary covariates using logistic regression.
- Stabilized inverse-observation weights are truncated at the 1st/99th percentiles. Report exposure/control observation rates, predicted-probability quantiles, fraction below 0.05 or above 0.95, weight distribution, and Kish ESS.
- An observation-weighted measured-lactate estimate is reported only if both exposure groups contain measured observations, no more than 5% of predicted probabilities are below 0.05 or above 0.95, and weighted ESS is at least 50% of the measured sample.
- `measured_lactate_direction_consistent` requires adjusted point estimates above 1 in both databases; lack of statistical significance alone does not negate directional consistency.
- `observation_process_acceptable` requires the positivity and ESS conditions in both databases.

## Covariate-balance robustness

- Propensity for exposure is fit from the full fixed non-lactate pre-anchor covariate set.
- Overlap weights are `1-p` for exposed and `p` for controls, truncated only by numerical clipping of `p` to `[0.001,0.999]`.
- Report propensity quantiles by group, absolute SMD before/after, maximum weighted absolute SMD, Kish ESS, weighted risks, RD, RR, and robust 95% CIs.
- Weighting cannot rescue a failed primary adjusted model. Gross positivity failure is either group having fewer than 1% of observations within the other group’s propensity range, weighted ESS below 25% of the landmark cohort, or maximum weighted absolute SMD above 0.10.

## Temporal lead and figures

- Lead time is support-start minute minus anchor minute.
- Report median, IQR, and fixed-bin counts separately by exposure and database.
- Plot cause-specific event-free/cumulative-incidence curves through 12 hours, clearly marking `0–2 h` as concurrent/blanking and `>2–12 h` as predictive. Death/discharge without support ends observation without creating a support event.
- `meaningful_temporal_lead` requires an eICU positive delayed adjusted association, a positive eICU delayed absolute RD, and at least 50% of the nonnegative exposed-minus-control cumulative excess risk across `0–12 h` occurring after two hours.
- The pre-anchor trajectory figure uses unsmoothed hourly medians and pointwise bootstrap 95% intervals for PrPP, SBP, MAP, and HR over four hours preceding the anchor. No post-anchor values or backward-moving smoother is allowed.

## Cross-database replication and meta-analysis

- Report MIMIC and eICU separately before pooling.
- Pool source-specific adjusted log RRs by inverse-variance random-effects meta-analysis using DerSimonian–Laird tau-squared; report pooled RR, 95% CI, Cochran Q, tau-squared, and I-squared.
- `primary_delayed_support_positive` and `eicu_confirmatory_positive` require the eICU adjusted RR to exceed 1 with 95% CI excluding 1.
- `mimic_direction_consistent` requires MIMIC adjusted RR `>1`.
- `cross_database_replication` requires eICU confirmation, MIMIC directional consistency, and I-squared `<=60%`.
- A pooled estimate cannot rescue a null or reversed eICU result.

## Fixed secondary outcomes

Only after the primary analysis is complete:

1. new MCS initiation in `>2–12 h`, using existing MIMIC procedure-event item IDs and the existing eICU treatment-device mapping;
2. first available post-anchor lactate in `>2–12 h` among patients with baseline lactate, with worsening defined prospectively as an increase of at least `0.5 mmol/L`; values are loaded only in the secondary stage;
3. hospital/ICU-recorded mortality as available in the locked cohort.

These are corroborative and cannot rescue the primary result.

## Claim readiness and failure-state precedence

`high_impact_claim_ready` requires all of the following: eICU confirmatory CI excludes 1; MIMIC direction is consistent; eICU adjusted RR is at least 1.50; eICU adjusted RD is positive with CI excluding 0; the effect survives SBP/MAP level and trajectory adjustment; dynamic crossing beats static PrPP; acceptable overlap-weighting positivity/balance; meaningful lead beyond two hours; cross-database replication; and no secondary endpoint is used to rescue the result.

Final classification uses this fixed precedence:

1. `CLAIM_READY_REPLICATED_SIGNAL` if `high_impact_claim_ready` is true.
2. `DISCOVERY_ONLY_NO_EXTERNAL_REPLICATION` if MIMIC adjusted RR is `>1` with CI excluding 1 but eICU confirmation is false.
3. `CONCURRENT_DETERIORATION_ONLY` if the eICU concurrent-window unadjusted RR is `>1` with CI excluding 1 but the delayed adjusted eICU association is not positive.
4. `CONVENTIONAL_BP_EXPLAINS_SIGNAL` if the eICU unadjusted delayed RR is `>1` with CI excluding 1 but `effect_survives_sbp_map_adjustment` is false.
5. `STATIC_PRPP_EXPLAINS_SIGNAL` if the effect survives SBP/MAP adjustment but `dynamic_beats_static` is false.
6. `NO_SIGNAL` otherwise.

The final report uses the classification literally and makes only the claim supported by these locked criteria.
