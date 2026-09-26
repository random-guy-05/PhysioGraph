# Separate eICU hospital validation after failed MIMIC temporal support

The primary two-database within-person specification is closed after the MIMIC oxygen gate failed (16 patients, five switches). Its joint stop rule is not retroactively changed. eICU has 1,068 patients, 2,397 usable intervals and 419 switchers, with independently validated features and source extraction. No troponin changes or association coefficients were calculated in that stopped specification.

This is a new, explicitly secondary discovery/validation experiment. It uses the unchanged eICU analysis intervals, one selected encounter per person, and the same two prespecified model forms. It is internally validated across disjoint hospitals, never called MIMIC replication or independent external-dataset validation. The source cohort and broad oxygen/troponin hypothesis have already been explored in this project.

## Hospital assignment fixed before outcomes

Join the original eICU patient table to obtain hospital IDs for selected stays; do not join outcomes. Missing hospital IDs are excluded and counted. No person can occur in both groups. Assign a hospital to split 0 or 1 using the parity of the first byte of SHA256 of `PhysioGraph within-person eICU hospital v1|` followed by its canonical integer hospital ID. Split 0 is discovery and split 1 is validation. Do not search for another seed, rebalance by outcomes, or move hospitals after seeing support. Save the assignment privately and publish only aggregate counts.

Each split must have at least 100 patients, 50 patients switching instability, and 20 distinct hospitals. If a split fails, stop this new specification without reducing the requirements. Retain the already frozen oxygen-window and exact positive consecutive-troponin rules; no alternate assay, encounter or window is substituted.

## Same-patient model and hospital uncertainty

Use log(troponin_t1/troponin_t0), patient fixed effects, binary instability, mean SpO2, hypoxic fraction, log(1+SpO2 readings), mean MAP, mean HR, interval hours, endpoint hours and its square, and log baseline troponin. Median-impute MAP/HR separately within each split with missing indicators; remove only nuisance columns having no within-person variation. Require full within-person rank, at least ten observations per retained slope, and positive residual degrees of freedom N−G−K. No simplified fit after a failed design gate.

Primary uncertainty clusters at the hospital level, allowing patients in the same hospital to share time-varying care patterns. Use the sandwich finite-sample factor H/(H−1) × (N−1)/(N−G−K), where H is hospitals, G is absorbed patient effects and K is retained slopes; use t uncertainty with H−1 degrees of freedom. Independently check the within-person and uncentered sufficient-statistic fits and sandwich covariance. Compare with a synthetic full-dummy model with patients nested in hospitals.

Report each split's exposure coefficient and ratio of troponin fold changes, 95% interval, unadjusted p-value and Holm p-value across the two primary split tests. A positive internal validation requires both coefficients positive and both Holm p<0.05. This is a necessary screening criterion, not a biological discovery standard. Do not pool a failed validation into a favorable overall result.

On exactly the same modeled rows, run the already declared sensitivity omitting only log baseline troponin; use the same imputation and hospital-clustered uncertainty. Treat its two p-values as a separate secondary Holm family. It cannot rescue failed primary validation. Neither inclusion nor omission of the lag yields a proven unbiased causal estimator in these short, selectively observed series.

## Interpretation and records

The analysis can reveal or weaken a within-patient association across hospitals. It cannot identify injury onset, infer infarction, remove treatment/clearance/testing confounding, establish a new mechanism, or demonstrate mortality benefit. No mortality outcome is accessed. Patient rows and hospital assignment remain in private Drive Data; aggregate JSON, limitations and executed cells remain in the single primary notebook. Keep the failed cross-database result visible. This amendment is frozen before hospital split counts or troponin changes are examined.
