# SpO2-variability transportability and precision audit

Execution environment: **LOCAL**. This is a post-hoc audit, not a rescue analysis. The frozen result remains **POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED**.

## Decision

**THIRD_COHORT_STRONGLY_JUSTIFIED**

eICU precision is demonstrably modest for RR 1.20–1.25, its interval contains the MIMIC effect, direction is concordant, and the MIMIC signal is bootstrap-stable; a higher-event, high-fidelity cohort can materially adjudicate transportability.

This decision does **not** reclassify the frozen eICU analysis. It answers only whether another locked validation is worth the access and engineering cost.

## Frozen analysis guard

- Feature: `spo2_variability` from hours 0–4 only, with hourly medians, a linear residual over observed bins, RMS residual, frozen 1st/99th-percentile winsorization, and MIMIC center/scale.
- Frozen MIMIC RR: 1.251 (1.104–1.418); frozen eICU RR: 1.104 (0.890–1.369).
- All frozen protocol, configuration, documentation, and code hashes passed. No alternate feature, threshold, subgroup, hospital exclusion, or outcome definition was fitted.

## Precision

The eICU log RR was 0.0991 with hospital-cluster-robust SE 0.1098. Its RR interval width was 0.479, equivalent to a multiplicative 95% CI factor of 1.240. With 85 modeled events, approximate two-sided power was 14.0%, 24.6%, 38.2%, 52.9%, and 66.6% for true RRs 1.10, 1.15, 1.20, 1.25, and 1.30, respectively. A true RR of 1.20 would require approximately 243 events for 80% power and 325 for 90%; RR 1.25 would require 162 and 217, respectively.

These are **normal approximations using the observed clustered SE and inverse event scaling**, not simulation-based guarantees. They show that the eICU result was not precise enough to separate a small effect near 1.10 from a clinically relevant effect near 1.20–1.25.

## Measurement process and feature transport

The databases did not differ by simple scarcity in eICU. Median raw SpO2 observations per stay were 4 in MIMIC and 46 in eICU; median spacing was 38 versus 5 minutes, and median observations within an occupied hour were 1 versus 12. However, consecutive identical readings were 24.6% versus 53.8%, and the median maximum identical run was 2 versus 7. Thus eICU was denser but substantially more repetitive, consistent with a different charting/device process rather than simply better temporal resolution.

On the frozen MIMIC scale, the MIMIC exposure had mean 0.000, SD 1.000, median -0.251, and IQR 1.131; eICU had mean -0.206, SD 0.895, median -0.459, and IQR 0.794. eICU therefore showed a lower, narrower frozen-feature distribution. Hospital-specific summaries are preserved in `feature_distribution_comparison.csv`; no hospital was excluded.

## Measurement degradation

The outcome-independent eICU timing-template operator was fully specified and hashed before outcome access. It sampled empirical eICU timing templates with replacement under seed 20260908 and mapped each scheduled time to the latest actual MIMIC value at or before that time (earliest later value only when no prior value existed); it added no Gaussian noise and introduced no arbitrary tuning constant. Across 5,957 paired MIMIC stays, original-versus-degraded Pearson correlation was 0.811, Spearman correlation was 0.754, reliability slope was 0.796, exact-quartile preservation was 63.4%, and 8.3% moved by at least two quartiles.

The locked original MIMIC RR was 1.251; the **single permitted** degraded-exposure fit gave RR 1.061 (0.909–1.238), p=0.451. Log-RR attenuation was 73.5%. This is strong evidence that the frozen statistic is sensitive to measurement-process transport, but it is a post-hoc diagnostic and does not prove that measurement alone caused the eICU attenuation.

## Outcome and case mix

The constructed endpoint is pressure/support plus hypoperfusion, **not adjudicated cardiogenic shock**. MIMIC had 238 events among 8,196 at-risk stays (2.90%); eICU had 95 among 8,743 (1.09%). Sustained hypotension completed the pressure side in 67.6% of MIMIC endpoints and 80.0% of eICU endpoints. Oliguria completed the perfusion side in 50.8% versus 32.6%; lactate did so in 22.7% versus 31.6%. These differences show meaningful outcome-composition and event-rate transport issues.

Age medians were 74 and 70; baseline lactate medians were 1.3 and 1.6, but availability differed (28.2% versus 12.6%). Ventilation, FiO2, and oxygen-support variables were not harmonized in either frozen frame, so respiratory context and heart-failure severity could **not** be directly adjudicated. No unsupported clinical equivalence claim is made.

Among the 85 modeled eICU events, the top 1%, 5%, and 10% of hospitals contributed 24.7%, 50.6%, and 68.2%, respectively. This concentration is descriptively important; an association refit after dropping hospitals was intentionally not performed because hospital exclusion was prohibited. Full component and hospital tables are retained in `outcome_composition_comparison.csv` and `eicu_hospital_heterogeneity.csv`.

## Cross-database synthesis

The fixed-effect descriptive summary RR was 1.212 (1.088–1.351); the two-study DerSimonian–Laird estimate was 1.212 (1.088–1.351). The database-by-exposure interaction test gave z=0.983, p=0.326. Q=0.966, I²=0.0%, and tau²=0.0000. These heterogeneity estimates are unstable and weakly informative with only two datasets.

## Third-cohort requirements

A third cohort should be pursued only with the frozen protocol in `THIRD_COHORT_VALIDATION_PROTOCOL.md`, high-fidelity timestamped SpO2 provenance, the same outcome construction, and site-clustered inference. The precision target should be at least 243 modeled endpoint events for approximately 80% power at RR 1.20 (preferably 325 for 90%), not merely a large patient count. Feasibility and support gates must be assessed without reading the exposure–outcome association; a failed confirmatory test ends the program without rescue.

**Pooling does not convert a failed external validation into successful external validation.** Neither this audit nor the frozen study establishes causality, clinical utility, or adjudicated cardiogenic shock.
