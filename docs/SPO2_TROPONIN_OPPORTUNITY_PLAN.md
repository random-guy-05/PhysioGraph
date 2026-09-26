# Dependence of the troponin signal on sampling opportunities

Specified September 5, 2026 before inspecting the new contrast. Earlier
SpO2/troponin associations and their discordance across estimators are known.
This is an exploratory falsification in the current MIMIC/eICU HF raw-replay
cohorts. It does not redefine the older locked episode-based replication.

Existing models already examine outcome observation and hospital effects.
They do not automatically remove dependence of a maximum on the number and
timing of samples among observed patients. Informative observation is an
established methodological problem, not a new biological discovery:
[primary simulation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6919310/).
This experiment tests dependence on recorded sampling opportunities; it is
not an inverse-intensity correction or a causal test of testing practice.

## Fixed data and endpoints

Use read-only `spo2_cardiorenal.duckdb` cohort, labs, lab covariates and
existing troponin pair tables, plus `spo2_context.duckdb` only to verify the
legacy labels. Keep all original dynamics-eligible encounters, exposed and
unexposed, with an assay-matched positive baseline and a follow-up troponin
measurement. Do not require a creatinine pair. Keep the original requirement
that hospital follow-up reaches the window end; its survivor/retention
selection limits interpretation.

Primary: baseline is the last positive value per assay/unit in ICU minutes
[0,240]; follow-up is (240,960], the next 12 hours. Sensitivity: (240,1680],
the next 24 hours. Reuse normalized assay/unit values and timestamp medians.
Baseline specimen time, not result availability, defines the original pair;
report the number whose baseline result was unavailable at minute 240.
This is therefore a retrospective biomarker audit, not a deployable rule.

At every follow-up timestamp with a matched assay, label whether any matched
assay is >=1.5 times its own baseline. Same-time assay results count as one
sampling opportunity, avoiding an extra draw merely because two assays were
measured together. Let n be the number of distinct follow-up timestamps and
k the number meeting the rise rule.

- Maximum endpoint: I(k>0).
- One-observed-sample expectation: k/n. This is the exact expected indicator
  if one of that patient's recorded follow-up timestamps were selected
  uniformly. It is not a measured biomarker probability under scheduled care.
- First-sample endpoint: the indicator at the first follow-up timestamp;
  report descriptively because time to detection can change its meaning.
- Paired primary outcome: D = I(k>0) - k/n.

Use Decimal(str(value)) arithmetic for these new labels at the fixed 1.5
boundary. Independently reconstruct the native-floating-point maximum ratio
and assert agreement with the original cached ratio and labels. Report
decimal-versus-legacy label differences; never overwrite the prior results.

## Frozen analysis

Report all arm counts, people, mean/median sampling opportunities, fractions
with repeated testing, maximum/one-sample/first-sample means, and baseline
availability. Reuse the exact `design` function from `cardiorenal_cells.py`
for adjustment: demographics, baseline physiology, absolute SpO2, SpO2
sampling, respiratory/vasoactive context, and baseline lab covariates with
its existing median-imputation/missingness scheme. Do not adjust for the
number of follow-up tests, a post-exposure variable involved in the contrast.

Fit linear projections of maximum, one-sample expectation, first-sample
indicator and D on that same matrix. Report the SpO2 coefficient for each.
Primary inference is only the D coefficient: the difference between the
maximum and one-sample exposure associations, with patient-clustered
covariance, finite-sample correction, t intervals and Holm adjustment over
the two databases. The 24-hour result is sensitivity-only. Require >=50
distinct people in each exposure arm, nonconstant D and full-rank design;
otherwise report non-estimability without dropping covariates to rescue it.
Independently verify coefficients, cluster covariance and the exact linear
contrast identity. No threshold is a discovery gate.

A positive D association would show that the apparent association depends
partly on taking a maximum over recorded samples. It could reflect real
kinetics, clinician-triggered testing, or both. A null result would not
exclude selective testing or validate a causal mechanism. Neither result
establishes myocardial injury, mortality benefit, or novelty. No new mortality
analysis, hospital search, threshold optimization or missing-value outcomes.

Keep hashes, exclusions, all estimates and local/Colab execution labels in
the single primary notebook. Patient-level outputs remain outside the repo.
