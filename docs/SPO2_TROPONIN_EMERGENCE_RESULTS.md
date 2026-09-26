# First-specimen reporting-limit emergence

Actual execution: local incremental notebook cells. No mortality outcome was accessed.

| Dataset | Horizon | SpO2 exposed | Baseline people | Positive | Negative | Indeterminate | Unavailable | Not remeasured |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eicu | 12h | 0 | 251 | 7 | 183 | 0 | 2 | 59 |
| eicu | 12h | 1 | 202 | 6 | 141 | 0 | 3 | 52 |
| eicu | 24h | 0 | 251 | 8 | 186 | 0 | 0 | 57 |
| eicu | 24h | 1 | 202 | 6 | 146 | 0 | 0 | 50 |
| mimic | 12h | 0 | 109 | 9 | 35 | 0 | 1 | 64 |
| mimic | 12h | 1 | 62 | 2 | 16 | 0 | 1 | 43 |
| mimic | 24h | 0 | 109 | 10 | 36 | 0 | 1 | 62 |
| mimic | 24h | 1 | 62 | 4 | 17 | 0 | 0 | 41 |

Risks and associations condition on having a definite first follow-up result; unmeasured and indeterminate results are not negative. Baseline is an explicit interval, not an imputed value.

- eicu 12h: descriptive risk difference 0.40 percentage points (95% Newcombe interval -3.93 to 5.31). Adjusted model not fitted: fewer than five minority outcomes per nonconstant parameter
- eicu 24h: descriptive risk difference -0.18 percentage points (95% Newcombe interval -4.53 to 4.66). Adjusted model not fitted: fewer than five minority outcomes per nonconstant parameter
- mimic 12h: descriptive risk difference -9.34 percentage points (95% Newcombe interval -25.51 to 14.26). Adjusted model not fitted: fewer than 50 definite observations per exposure arm; fewer than five events or non-events per exposure arm; fewer than five minority outcomes per nonconstant parameter
- mimic 24h: descriptive risk difference -2.69 percentage points (95% Newcombe interval -20.61 to 20.31). Adjusted model not fitted: fewer than 50 definite observations per exposure arm; fewer than five events or non-events per exposure arm; fewer than five minority outcomes per nonconstant parameter

SQL/Python selection and independent decimal interval labeling agree. Original caches are unchanged. Results do not establish injury onset, infarction, causal biology, mortality benefit, or a paradigm-shifting discovery. The research goal remains unmet.

Novelty check: conversion from initially negative troponin to detectable or positive values in acute HF was already studied in [PROTECT](https://pubmed.ncbi.nlm.nih.gov/21900185/) and a [198-patient serial-sampling study](https://pubmed.ncbi.nlm.nih.gov/22407461/). Sleep-disordered breathing and overnight troponin increases were also examined by [Light et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708937/). Our conservative interval endpoint and ICU exposure definition differ, but those differences alone are not a new biological mechanism. Literature supplies context only; all patient analyses here use MIMIC and eICU.
