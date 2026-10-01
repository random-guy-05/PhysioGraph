# Final hardened analysis: dense early SpO₂ dynamics and later lactate deterioration in MIMIC-III

**Analysis status:** local/precomputed MIMIC-III reanalysis and post-result hardening; not executed in Google Colab. No cohort, waveform, or original full-table processing was rerun in this finalization. The corrected endpoint and all robustness estimates below are drawn from saved analysis artifacts; the ventilation update uses only the small MIMIC-III procedure-events table and D_ITEMS codebook.

## 1. Executive summary

Among 1,597 dynamics-eligible ICU stays in critically ill adults with heart failure, 539 had at least one first-four-hour absolute SpO₂ transition of at least 4 percentage points. With complete observation through the relevant horizon required for both events and non-events, the 12-hour last-value lactate rise association remained positive (27/155 vs 23/233; patient-cluster RR 1.77, 95% CI 1.05–3.04; RD +7.5 percentage points, 95% CI +0.6 to +14.7). The 24-hour estimate was positive but imprecise (25/174 vs 24/247; RR 1.48, 95% CI 0.85–2.53).

In episode-aligned analyses, the acute 0–1 hour contrast was imprecise and did not demonstrate an association. The delayed 1–8 hour contrast was 18/99 versus 17/170 (RR 1.82; patient-cluster 95% CI 0.98–3.59), but did **not** survive Holm correction across the two prespecified primary episode windows (p=0.123) or the conservative five-window family (p=0.309). This is suggestive and fragile, not a multiplicity-supported primary finding. Dynamics did not improve out-of-fold prediction. Lactate testing selection and waveform-subset selection remain material limitations; these observational results do not establish cause or mechanism.

## 2. Exact scientific question

Within the available MIMIC-III heart-failure ICU waveform subset, is a first-four-hour absolute SpO₂ transition of at least 4 percentage points associated with a subsequent complete-window last-value lactate increase of at least 0.5 mmol/L at the fixed 12-hour or 24-hour landmark? Separately, after a first qualifying transition, do acute (0,1] hour or delayed (1,8] hour complete-window episode outcomes differ from time-aligned no-episode controls?

## 3. Cohort

The saved clinical cohort contains 10,230 eligible adult first-ICU-per-admission HF stays; 1,667 were represented in the available HF waveform manifest, 1,654 had usable 15-minute SpO₂ signal, 1,597 met dynamics eligibility, and 539 had a qualifying ≥4-pp transition. The upstream all-waveform candidate list was unavailable. The study subset is therefore not a random sample of all MIMIC-III HF ICU stays.

## 4. Signal definition

The exposure remains fixed: 15-minute median SpO₂ values in [0,240) minutes, plausible range 50–100%, at least three valid bins and two forward transitions with gaps ≤30 minutes; exposure is at least one absolute transition ≥4 percentage points. Patients/stays, never waveform samples, are the analysis units. Independent reconstruction matched the saved feature file on all core signal fields.

## 5. Corrected lactate endpoint

At the landmark, baseline is the last valid linked lactate at or before minute 240. The 12-hour endpoint window is (240,960] minutes; the 24-hour window is (240,1680]. The primary binary event is last lactate in that window minus baseline ≥0.5 mmol/L. Episode baseline is the last lactate in (anchor−360 min, anchor); episode windows retain the frozen acute, delayed, and cumulative definitions in the analysis plan.

## 6. Complete-follow-up correction

Complete observable follow-up to the fixed horizon is required for both positive and negative primary last-value classifications. A rise seen before earlier death, ICU departure, discharge, or end of observation is unavailable for the primary last-value endpoint. Only prespecified monotone maximum-rise/threshold-crossing companions may retain an observed positive before censoring. A value exactly at minute 240 remains baseline; endpoint interval upper bounds are included.

The landmark correction removed the following previously classifiable observations in the dynamics sample:

| window | previously_observed_n | previously_events | previously_non_events | previously_positive_removed | previously_negative_removed | updated_observed_n | updated_events | updated_non_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12h | 391 | 53 | 338 | 3 | 0 | 388 | 50 | 338 |
| 24h | 454 | 54 | 400 | 5 | 28 | 421 | 49 | 372 |

The episode reconciliation below reports before-correction positive/negative observations removed at every fixed window and group. The strict six-hour episode baseline remains in force; no exposure, threshold, horizon, window, or estimator was changed.

| window | group | before_observed_n | before_events | before_non_events | followup_removed_positive | followup_removed_negative | final_observed_n | final_events | final_non_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acute_0_1h | episode | 40 | 6 | 34 | 2 | 5 | 33 | 4 | 29 |
| acute_0_1h | time_aligned_no_episode | 52 | 9 | 43 | 0 | 3 | 49 | 9 | 40 |
| delayed_1_8h | episode | 119 | 25 | 94 | 7 | 13 | 99 | 18 | 81 |
| delayed_1_8h | time_aligned_no_episode | 196 | 20 | 176 | 3 | 23 | 170 | 17 | 153 |
| cumulative_0_4h | episode | 98 | 19 | 79 | 5 | 11 | 82 | 14 | 68 |
| cumulative_0_4h | time_aligned_no_episode | 157 | 23 | 134 | 2 | 17 | 138 | 21 | 117 |
| cumulative_0_8h | episode | 139 | 26 | 113 | 8 | 18 | 113 | 18 | 95 |
| cumulative_0_8h | time_aligned_no_episode | 218 | 22 | 196 | 3 | 25 | 190 | 19 | 171 |
| cumulative_0_12h | episode | 158 | 29 | 129 | 8 | 21 | 129 | 21 | 108 |
| cumulative_0_12h | time_aligned_no_episode | 240 | 24 | 216 | 5 | 31 | 204 | 19 | 185 |

Direction-specific last-value analyses use separately anchored first-drop and first-rise episodes; their before-versus-after removals are shown here:

| window | group | previously_observed_n | previously_events | previously_non_events | previously_positive_removed | previously_negative_removed | updated_observed_n | updated_events | updated_non_events |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drop_acute_0_1h | drop_episode | 25 | 4 | 21 | 2 | 2 | 21 | 2 | 19 |
| drop_acute_0_1h | time_aligned_no_episode | 44 | 7 | 37 | 1 | 6 | 37 | 6 | 31 |
| drop_delayed_1_8h | drop_episode | 75 | 12 | 63 | 3 | 8 | 64 | 9 | 55 |
| drop_delayed_1_8h | time_aligned_no_episode | 195 | 24 | 171 | 3 | 14 | 178 | 21 | 157 |
| rise_acute_0_1h | rise_episode | 33 | 4 | 29 | 1 | 5 | 27 | 3 | 24 |
| rise_acute_0_1h | time_aligned_no_episode | 57 | 8 | 49 | 1 | 2 | 54 | 7 | 47 |
| rise_delayed_1_8h | rise_episode | 97 | 20 | 77 | 6 | 10 | 81 | 14 | 67 |
| rise_delayed_1_8h | time_aligned_no_episode | 200 | 24 | 176 | 0 | 17 | 183 | 24 | 159 |

The complete row-level reconciliation is saved in `03_landmark_lactate/complete_followup_before_after_reconciliation.csv` and `04_episode_lactate/final_corrected_episode_reconciliation.csv`.

## 7. Corrected 12-hour result

The complete-window denominator is N=388 with 50 events: exposed 27/155 (17.4%) and unexposed 23/233 (9.9%). RR=1.765; patient-cluster bootstrap 95% CI 1.050–3.042. RD=+0.075; patient-cluster 95% CI +0.006 to +0.147. The parsimonious clinical-core adjusted RR is 1.85 (1.10–3.10; EPV=10.0). The signal/context-adjusted estimate is 2.46 (1.24–4.85; EPV=5.0) and is labelled fragile. Corrected observation IPW gives RR=1.91, RD=0.086, ESS=360.2; the saved weighted output does not include a bootstrap interval.

## 8. Corrected 24-hour result

The complete-window denominator is N=421 with 49 events: exposed 25/174 (14.4%) and unexposed 24/247 (9.7%). RR=1.479; patient-cluster bootstrap 95% CI 0.854–2.532. RD=+0.047; patient-cluster 95% CI −0.018 to +0.110. The parsimonious clinical-core adjusted RR is 1.54 (0.90–2.62; EPV=9.8) and remains imprecise. The richer signal-adjusted 24-hour model did not meet convergence tolerance and is diagnostic only. Corrected observation IPW gives RR=1.61, RD=0.058, ESS=402.5; IPW is not causal and its saved result has no bootstrap interval.

## 9. Corrected episode analysis

The final episode table uses the strict six-hour pre-anchor baseline and complete follow-up through each window. It reports both groups' classifiable denominators, events, risks, analytic intervals, patient-cluster bootstrap intervals, Fisher tests, adjusted estimates, EPV, and convergence:

| window | episode_n | episode_events | time_aligned_no_episode_n | time_aligned_no_episode_events | RR | RR_cluster_boot_low | RR_cluster_boot_high | RD | RD_cluster_boot_low | RD_cluster_boot_high | Fisher_exact_p | holm_primary_two_p | holm_global_five_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acute_0_1h | 33 | 4 | 49 | 9 | 0.660 | 0.128 | 1.937 | -0.062 | -0.223 | 0.097 | 0.547 | 0.547 | 1.000 |
| delayed_1_8h | 99 | 18 | 170 | 17 | 1.818 | 0.979 | 3.588 | 0.082 | -0.002 | 0.170 | 0.062 | 0.123 | 0.309 |
| cumulative_0_4h | 82 | 14 | 138 | 21 | 1.122 | 0.554 | 2.071 | 0.019 | -0.081 | 0.120 | 0.708 | — | 1.000 |
| cumulative_0_8h | 113 | 18 | 190 | 19 | 1.593 | 0.837 | 3.043 | 0.059 | -0.021 | 0.141 | 0.148 | — | 0.443 |
| cumulative_0_12h | 129 | 21 | 204 | 19 | 1.748 | 0.956 | 3.300 | 0.070 | -0.005 | 0.147 | 0.082 | — | 0.330 |

## 10. Acute versus delayed temporal pattern

The acute 0–1 hour estimate is RR=0.66 (patient-cluster 95% CI 0.13–1.94); it is not evidence of an immediate lactate response. The delayed 1–8 hour estimate is RR=1.82 (patient-cluster 95% CI 0.98–3.59). The point-estimate pattern is delayed rather than acute, but classification depends on relatively few tested lactate outcomes and does not survive the planned multiplicity correction.

## 11. Multiplicity

The raw delayed-window Fisher p-value is 0.062; the Holm-adjusted primary-family p-value across acute and delayed windows is 0.123. The global five-window Holm sensitivity is 0.309. The adjusted-model p-value is reported separately and is not included in this raw binary endpoint family. The delayed finding does not survive either correction.

## 12. Pre-episode lactate trajectory

Only 84 episode stays and 137 controls had serial lactate specimens before their anchor. Median prior change per hour was -0.184 mmol/L/h in episodes and -0.025 in controls; the serial-subset comparison did not show greater prior worsening among episodes (descriptive p=0.023). These selected serial draws cannot rule out reverse timing. The delayed model retains continuous pre-anchor trend and a missing-trend indicator, with adjusted RR 1.88 (1.00–3.54), EPV 5.0; this is fragile and does not establish temporal causation.

## 13. Measurement-selection analysis

For the delayed episode window, strict pre-anchor baseline availability was 261/539 episodes and 444/1,058 time-aligned controls. Classifiable primary outcomes were 99/539 and 170/1,058. Patient-grouped OOF observation weighting used only pre-anchor predictors and 1st/99th percentile truncation. Delayed weighted RR=1.89 (patient-cluster 95% CI 0.96–4.06); weighted RD=0.081 (-0.005–0.181). Overall ESS=227.0 (episode 83.4; control 143.6); the predicted observation range approached 0 and 1, indicating near-positivity extremes. IPW addresses measured testing predictors only and does not remove informative testing.

## 14. Censoring analysis

Censoring is reported as separate ICU departure, hospital discharge, death, any early end of follow-up, and complete-window follow-up measures; no substitute composite endpoint was created. Landmark 24-hour complete follow-up was 174/539 exposed and 247/1,058 unexposed; early observable-end censoring was 92/539 and 201/1,058. The 12-hour any-censor proportions were 17/539 and 20/1,058. Episode-window completion and testing availability are shown separately in `final_table5_observation_censoring.csv`; group comparisons are in `censoring_comparison_tests.csv`. Censoring and specimen availability are distinct.

## 15. Control-anchor robustness

Across 100 outcome-blind assignments preserving the exposed empirical anchor-time distribution exactly, delayed RR median was 1.287; assignment 2.5th–97.5th percentiles were 1.055–1.703, minimum 0.963, maximum 1.859, and 98% of assignments exceeded 1. This is a design-assignment distribution, not a confidence interval, and no seed was selected for favorable results.

## 16. Risk-set control analysis

The post-hoc risk-set design matched exact 15-minute bins and sampling regime, used care unit when supported, and allowed controls to have a later episode. Among classifiable observations there were 93 cases/18 events and 210 controls/38 events across 232 sets. Weighted RR=1.07 (patient-cluster 95% CI 0.63–1.80); RD=0.013 (-0.079–0.105). Age and strict baseline lactate remained imbalanced in available records (SMDs about 0.30 and 0.23). This does not replace the no-episode-control primary design.

## 17. Normoxemia and absolute oxygenation

The fixed all-16-bin ≥90% subset had 34 classifiable 12-hour records and 2 events (1/10 exposed vs 1/24 unexposed; RR=2.40); at 24 hours it had 39 classifiable records and 4 events. This subset is underpowered. The all-16-bin normoxemic delayed episode subset had 14 classifiable observations and 1 events; the row is descriptive only. Absolute SpO₂ context remains in the adjusted model, which is fragile at EPV 5; therefore the association is not claimed to be independent of oxygenation.

## 18. Respiratory context

The original all-missing ventilation covariate was a source-mapping defect, not evidence of no support. D_ITEMS maps invasive and non-invasive ventilation to PROCEDUREEVENTS_MV and the mechan ventilation field to a CHARTEVENTS Checkbox. The small procedure-events audit found positive invasive/non-invasive procedure evidence in 1,192/10,230 eligible stays; among dynamics-eligible stays, the documented-positive lower bound was 97/539 exposed and 221/1,058 unexposed. No procedure record is unknown, not a negative. The numeric checkbox rows were not re-read because no row-level chart cache was present and a full-table re-scan was excluded by the user's analysis-only instruction. FiO₂ remains available in about 24% of clinical stays. A respiratory-adjusted outcome model is not supported; no respiratory-independence claim is made.

## 19. Hemodynamic context

Saved [0,240) HR, SBP, MAP, RR, FiO₂, and vasoactive summaries are suitable for landmark description but may include measurements after episode anchors. They were not used as episode-adjustment covariates. The existing MAP item cache omits additional D_ITEMS-coded mean-pressure items, and no exact-bin MAP medians are present in the validated cache. Thus the ≥2 jointly observed transition support rule cannot be assessed; synchronous MAP-decline classification and its outcome model are unavailable rather than relaxed. This analysis does not measure cardiac output.

## 20. Directionality

Both prespecified directions are retained, using first qualifying drop and rise anchors, strict pre-anchor lactate baselines, and complete follow-up through each fixed window. In the acute window, drops were 2/21 versus 6/37 controls (RR 0.59; patient-cluster CI 0.00–2.22); rises were 3/27 versus 7/54 (RR 0.86; CI 0.00–3.08). In the delayed window, drops were 9/64 versus 21/178 controls (RR 1.19; patient-cluster CI 0.48–2.44); rises were 14/81 versus 24/183 (RR 1.32; CI 0.69–2.32). Estimates are imprecise and do not identify a mechanism; no interaction test was promoted because event support is insufficient for a stable multivariable interaction.

## 21. Sampling and coverage

The fixed ≥4-pp association direction was similar in the prespecified 1-Hz and approximately 1/min strata; interaction tests did not demonstrate material heterogeneity. Coverage-restricted estimates generally remained positive but became imprecise as coverage and event counts declined. All requested rows are preserved in `sampling_regime_sensitivity.csv` and `coverage_sensitivity.csv` and summarized in Table 4.

## 22. Continuous lactate behavior

The continuous last-value deltas do not show a uniform upward shift. In the delayed episode window, the episode mean delta was −0.828 mmol/L versus −0.763 in controls; medians were −0.40 versus −0.50 mmol/L, median difference +0.10, rank-test p=0.457. Patient-cluster bootstrap mean/median-difference intervals are saved in `continuous_lactate_distribution.csv`. The binary ≥0.5 endpoint may reflect a higher probability of upper-tail upward excursions rather than a general shift; the continuous data do not provide a clear global-distribution support for that interpretation.

## 23. Waveform-subset selection

The waveform manifest covered 1,667/10,230 eligible stays (16.3%); 1,654 had usable signal. The outcome-blind, patient-grouped selection model had OOF AUC 0.693. Some care-unit differences were substantial (MICU SMD about −0.93; CCU about −0.44); measured age and baseline lactate differed little, while vasoactive use had SMD about 0.14. Creatinine was unavailable in the saved pre-landmark covariates. Generalization beyond the waveform subset is not established.

## 24. Prediction results

Five-fold SUBJECT_ID-grouped OOF predictions used identical complete-follow-up rows for all four nested models; preprocessing was fit within each training fold, and no outcome variable was a predictor. Audit: ['pass', 'pass']. At 12 hours, adding parsimonious dynamics to clinical+absolute-SpO₂/context changed AUROC by -0.033 (-0.070–-0.003) and AUPRC by -0.007 (-0.019–0.002); both were non-improving. At 24 hours the AUROC change was 0.021 (-0.010–0.050) and AUPRC change -0.005 (-0.051–0.019), neither establishing improvement. Calibration and Brier metrics remain in Table 6. The negative incremental-prediction result is retained.

## 25. eICU external replication

No eICU replication is included, at the user's direction. Partial interim eICU files have been moved to `eicu_partial_not_used/` and are excluded from all estimates and figures. MIMIC-III is not treated as independent validation of MIMIC-IV.

## 26. Strengths

The primary SpO₂ exposure and outcome windows were held fixed; saved waveform features were independently reconstructed; endpoint follow-up semantics were corrected for both events and non-events; cluster-aware inference and outcome-observation analyses were used; all fixed episode windows, both directions, negative prediction results, and unavailable branches are disclosed.

## 27. Limitations

The waveform subset is selected, lactate ordering is informative, complete-window requirements reduce the observed outcome sample, the episode analyses have low events per model parameter, and the delayed episode finding fails its prespecified multiplicity correction. Respiratory support and time-aligned MAP could not be fully reconstructed from the validated caches under the analysis-only scope. Residual confounding and measurement error remain plausible.

## 28. Exact supported claim

In the available MIMIC-III HF ICU waveform subset, a first-four-hour ≥4-pp SpO₂ transition was associated with a higher risk of complete-window last-value lactate rise at 12 hours. The 24-hour association was directionally positive but imprecise. The delayed episode pattern is suggestive but does not survive multiplicity correction.

## 29. Exact unsupported claims

These results do not show that SpO₂ instability causes lactate deterioration, measures cardiac output, predicts cardiogenic shock, is independent of respiratory or hemodynamic confounding, generalizes to all MIMIC HF patients, improves prediction, or is externally validated or ready for clinical use.

## 30. Publication interpretation

A manuscript focused on this association is defensible as an exploratory, observational signal study if the 12-hour result leads, the delayed episode result is explicitly labelled suggestive and multiplicity-negative, the 24-hour uncertainty and testing selection are prominent, and the negative prediction result remains visible. It is not ready to support mechanism, clinical prediction, or implementation claims; a skeptical reviewer would reasonably require independent replication and stronger time-aligned confounding measurement.

## 31. Reproducibility and provenance

The primary output is local/precomputed and was not executed in Google Colab. The analysis lock was written and SHA256-hashed before hardening; its recorded hash remains unchanged. The finalization used saved MIMIC-III cohort, waveform, lactate, covariate, and model artifacts, plus the targeted D_ITEMS/PROCEDUREEVENTS_MV mapping audit. `run_status.json`, `final_self_audit.json`, and `output_hashes.csv` record the scope, checks, and output provenance. Figures 1–9 are generated from the saved cohort and result tables by `scripts/build_final_package.py`.

### Final tables

- Table 1: `table1_waveform_by_exposure.csv`
- Table 2: `final_table2_corrected_landmark.csv`
- Table 3: `final_table3_corrected_episode.csv`
- Table 4: `final_table4_robustness_matrix.csv`
- Table 5: `final_table5_observation_censoring.csv`
- Table 6: `final_table6_predictive_metrics.csv`
