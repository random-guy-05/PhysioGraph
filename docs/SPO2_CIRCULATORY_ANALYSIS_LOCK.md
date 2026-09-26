# Circulatory-context analysis specification

Written before new exposure–outcome cross-tabulations or model fits on
2026-09-05 UTC. The raw scans are running. Feature and total endpoint
availability may be inspected; no effect estimates have been inspected.
This supplements SPO2_CIRCULATORY_CONTEXT_PLAN.md without changing its
thresholds, cohort, or biological question.

## Estimand and support gate

The primary timing comparison is **synchronous versus asynchronous events
among encounters having both a qualifying SpO2 change and a qualifying MAP
fall during the first four hours**. It is not synchrony versus every other
patient. The latter would mostly compare different component abnormalities
and would require impossible counterfactual synchrony in patients without
either component. Report the four component states descriptively.

Primary outcomes are hospital mortality and the existing troponin-rise
endpoint during 12 hours after the four-hour landmark (ICU minutes 240–960).
The 24-hour endpoint is a sensitivity analysis; a negative or sparse 12-hour
result cannot be rescued by selecting 24 hours. Troponin requires a positive
baseline value and a subsequent value of the same assay and unit, with rise
ratio at least 1.5. Without a detected rise, follow-up must extend through
the horizon. Missing or incompletely observed outcomes are not negatives.

Evaluate the previously frozen event gate **within the primary timing
comparison**, separately in each database: at least 100 events overall and
at least 20 events and 20 non-events in each timing group. Otherwise produce
counts only, explicitly fail the primary fitting gate, and do not lower it.
Also report marginal event support in all jointly observed patients to make
the source of attrition clear. No group with fewer than five patients is
used to support a biological claim.

## Model sequence if support is adequate

First quantify a conditional association with a binomial logit model.
The base model contains age, sex, mean heart rate, mean and minimum SpO2,
mean and minimum MAP, counts of SpO2 jumps and MAP falls, count of jointly
observed transitions, and log(1 + raw SpO2 reading count). The extended
model adds the binary synchrony indicator. Numeric predictors are centered
and scaled within each source; sex is categorical. Missing heart rate uses
the source median and an explicit missing indicator. Other required
physiological values must be observed. No predictor selection by p-value.

Use person-clustered covariance within each source, and an eICU sensitivity
with hospital-clustered covariance when at least 30 hospitals contribute.
Check model convergence, design rank, complete separation, and adequate
events for the number of fitted coefficients (at least 10 events and 10
non-events per coefficient). Failure means no model claim. Report the
conditional odds ratio and 95% confidence interval, explicitly as an odds
ratio. Do not relabel it as a risk ratio or causal effect. Correct the two
primary endpoint tests by Holm within each database. Replication requires
the same direction and multiplicity-adjusted evidence in both sources.

This first model is a **screen**, not the final biological test. Advancement
requires baseline respiratory support, oxygen therapy and vasoactive
treatment controls from original sources, observed baseline lactate and
creatinine, pressure-source sensitivity, ordering/measurement selection
analysis for troponin, and evaluation of predictive improvement. No
unadjusted or screen-only association is promoted as a discovery.

## Fixed sensitivity checks

- Repeat MIMIC features using arterial MAP only. eICU's supplied periodic
  pressure source is arterial; do not infer unmeasured cuff pressure.
- Analyze downward and upward SpO2 changes separately without choosing a
  preferred direction after seeing results.
- Evaluate the predeclared all-SpO2-bins-at-least-90% and measured baseline
  lactate-below-2 mmol/L subgroups only if they pass the same support gate.
  Retrospective specimen-time and actual result-availability definitions
  must be distinguished. eICU revision time is not proven first availability.
- Compare within-bin measurement time differences across pressure sources.
  Same-bin synchrony is coarse charted synchrony, not beat-to-beat coupling.

Low perfusion can alter pulse-oximeter measurement. Neither synchrony nor
troponin elevation establishes myocardial hypoxia, reduced cardiac output,
or a treatable mechanism. Reject significance that depends on a changed
threshold, an unsupported subgroup, a missing-data convention, or a single
database. The paradigm-shifting discovery goal remains unfulfilled unless
substantially stronger evidence is actually obtained.

## Prior-art check

The joint prognostic relevance of low saturation and low blood pressure in
acute HF was already reported in a [2009 clinical cohort](https://pubmed.ncbi.nlm.nih.gov/19372679/).
Combining those two absolute levels is therefore not a new biological finding.
A [2021 registered protocol](https://cdn.clinicaltrials.gov/large-docs/01/NCT05188001/Prot_SAP_000.pdf)
also proposed vital-sign time series for predicting troponin and myocardial
injury after noncardiac surgery; this is adjacent prior art, not an HF result
or proof that its planned analyses succeeded. The narrower timing contrast
in this extension remains a hypothesis. A search finding no exact match
does not establish novelty.
