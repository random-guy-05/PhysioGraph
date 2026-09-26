# First SpO2 transition, HR direction and mortality

Execution: local incremental notebook cells. Existing audited MIMIC/eICU bins; no fresh raw-source extraction. Crude observational screen.

| Source | SpO2 | HR | Deaths / known outcomes | Unknown | Risk |
|---|---|---|---|---:|---:|
| eicu | fall | fall | 35 / 199 | 2 | 17.59% |
| eicu | fall | flat | 262 / 1702 | 16 | 15.39% |
| eicu | fall | rise | 66 / 404 | 6 | 16.34% |
| eicu | fall | unavailable | 1 / 13 | 0 | 7.69% |
| eicu | rise | fall | 50 / 313 | 4 | 15.97% |
| eicu | rise | flat | 183 / 1225 | 22 | 14.94% |
| eicu | rise | rise | 17 / 142 | 1 | 11.97% |
| eicu | rise | unavailable | 2 / 8 | 0 | 25.00% |
| mimic | fall | fall | 33 / 119 | 0 | 27.73% |
| mimic | fall | flat | 52 / 359 | 0 | 14.48% |
| mimic | fall | rise | 20 / 112 | 0 | 17.86% |
| mimic | fall | unavailable | 27 / 89 | 0 | 30.34% |
| mimic | rise | fall | 32 / 161 | 0 | 19.88% |
| mimic | rise | flat | 55 / 334 | 0 | 16.47% |
| mimic | rise | rise | 13 / 86 | 0 | 15.12% |
| mimic | rise | unavailable | 13 / 84 | 0 | 15.48% |

Joint count gate passed: True.

| Source | Estimand | Estimate, pp | Nominal 98.75% bootstrap interval, pp | Advances |
|---|---|---:|---|---|
| eicu | falling_spo2_hr_fall_minus_rise_rd | 1.25 | -6.65 to 8.84 | False |
| eicu | direction_interaction_rd | -2.75 | -12.98 to 7.65 | False |
| mimic | falling_spo2_hr_fall_minus_rise_rd | 9.87 | -3.63 to 23.62 | False |
| mimic | direction_interaction_rd | 5.11 | -13.15 to 23.54 | False |

Both-database advancement: False. No adjusted or treatment-effect models were fitted.

Intervals use hospital resampling in eICU and person resampling in MIMIC with fixed seeds. Percentile bootstrap coverage is approximate. Missing outcomes and unmatched HR remain separate.

These coarse charted states do not measure an acute chemoreflex, cardiac output or confirmed arterial oxygen change. Severity, baseline vital levels, medications, rhythm, pacing, treatment and observation patterns remain uncontrolled. A positive crude association would not establish biological novelty, mechanism or benefit from changing heart rate. A failed screen cannot exclude smaller or confounded effects.

The original exposure, person selections, HR labels, source hashes and available bootstrap arithmetic validate. The discovery goal remains unfulfilled.
