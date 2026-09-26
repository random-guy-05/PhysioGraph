# Troponin sampling-opportunity sensitivity

Actual execution: local. Existing MIMIC/eICU HF raw-replay caches. This is a four-hour-landmark analysis, not the earlier locked episode-window replication.

| Source | Horizon | Exposure | Encounters | Mean samples | Repeated samples | Maximum rise | One-sample expectation | First-sample rise |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eicu | 12h | 0 | 1436 | 1.36 | 33.0% | 21.9% | 20.5% | 20.1% |
| eicu | 12h | 1 | 858 | 1.36 | 33.2% | 28.1% | 26.0% | 25.4% |
| eicu | 24h | 0 | 1269 | 1.77 | 53.8% | 24.0% | 20.6% | 20.3% |
| eicu | 24h | 1 | 773 | 1.78 | 52.1% | 31.6% | 28.0% | 27.7% |
| mimic | 12h | 0 | 516 | 1.31 | 29.7% | 22.9% | 20.9% | 20.0% |
| mimic | 12h | 1 | 258 | 1.30 | 27.9% | 35.7% | 34.2% | 33.7% |
| mimic | 24h | 0 | 475 | 1.86 | 54.9% | 27.4% | 23.1% | 20.6% |
| mimic | 24h | 1 | 243 | 1.86 | 55.6% | 41.6% | 37.6% | 36.6% |

| Source | Horizon | Status | Adjusted maximum-minus-one-sample association (pp) | 95% interval (pp) | Primary Holm p |
|---|---|---|---:|---|---:|
| eicu | 12h | estimable | 0.75 | -0.10 to 1.60 | 0.1649 |
| eicu | 24h | estimable | 0.28 | -0.98 to 1.55 | sensitivity only |
| mimic | 12h | estimable | -1.12 | -2.48 to 0.24 | 0.1649 |
| mimic | 24h | estimable | -0.62 | -2.99 to 1.75 | sensitivity only |

Decimal versus original floating-point maximum labels differ in 11 horizon-specific rows (12/24-hour patients overlap). Legacy ratios and labels reconstruct exactly. Independent sample joins, timestamp outcomes, regression coefficients, covariance and paired-coefficient identities passed.

One-sample expectation is the mean threshold indicator over the patient's observed timestamps. It is an exact resampling expectation, not a probability observed under a standardized clinical sampling schedule. Taking a maximum and taking an average define different summaries of real kinetics and clinician-selected observations.

The cohort is conditioned on paired troponin testing and hospital follow-up to the window end. Baselines use original specimen-time rules; result unavailability at the four-hour landmark is reported separately in arm_counts.json. Numeric ratios do not establish adjudicated myocardial injury or MI. These results do not identify causal testing bias, reconcile the older episode-based estimators, or demonstrate mortality benefit or novel biology.
