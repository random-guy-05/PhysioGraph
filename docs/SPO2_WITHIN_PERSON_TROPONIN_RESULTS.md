# Within-person oxygen instability and troponin change

Actual execution: local incremental notebook cells. Original MIMIC/eICU patients only.

| Dataset | Timing-eligible people | Timing intervals |
|---|---:|---:|
| eicu | 1090 | 2450 |
| mimic | 293 | 640 |

| Dataset | People with repeated usable oxygen windows | Intervals | People changing instability status | Gate |
|---|---:|---:|---:|---|
| eicu | 1068 | 2397 | 419 | pass |
| mimic | 16 | 34 | 5 | fail |

A frozen joint support gate failed. No troponin changes or association models were calculated; no thresholds were relaxed.

Independent SQL/Python specimen and oxygen-feature reconstructions agree. When models are fitted, absorbed-effect and full-dummy synthetic checks, independent sufficient-statistic coefficients and patient-clustered covariance are validated. Raw-source fingerprints and original first-four-hour extraction agreement are recorded separately.

The planned fixed-effects design addresses stable patient differences when it can be fitted. Short-panel dynamic bias from the prior-troponin covariate is an additional material limitation identified before estimates; see SPO2_WITHIN_PERSON_LAG_AMENDMENT.md. Correct standard errors do not correct that bias. Time-varying treatment, congestion, ischemia, renal clearance, selective testing, delayed troponin release and sparse chart resolution remain unresolved. No mortality outcome was accessed. Neither a causal mechanism nor a mortality benefit is established. The discovery goal remains unmet.

## Prespecified lag-adjustment sensitivity

This diagnostic removes only the previous-troponin covariate on the same primary rows. It is not a validated bias correction and cannot rescue the primary replication criterion.
No primary models were fitted because a joint support gate failed; the diagnostic is also skipped.
