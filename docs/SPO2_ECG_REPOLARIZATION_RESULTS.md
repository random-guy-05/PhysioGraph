# Paired ECG repolarization experiment

Execution: local incremental notebook cells. MIMIC original cohort only; no external replication or mortality endpoint.

| Split | Metadata people | Quality people | Exposed | Unexposed | Support gate |
|---|---:|---:|---:|---:|---|
| 0 | 182 | 124 | 30 | 94 | True |
| 1 | 156 | 85 | 26 | 59 | True |

| Analysis / split | Outcome | Estimate, ms | 95% CI | Primary Holm p |
|---|---|---:|---|---|
| primary / 0 | jtcf_change | 0.011 | -12.759 to 12.782 | 1.0 |
| primary / 0 | jtcf_minus_qrs_change | 3.120 | -11.339 to 17.578 | 1.0 |
| primary / 0 | qrs_change_descriptive | -3.108 | -9.140 to 2.923 | not confirmatory |
| primary / 1 | jtcf_change | -0.680 | -17.074 to 15.714 | 1.0 |
| primary / 1 | jtcf_minus_qrs_change | -7.371 | -27.019 to 12.277 | 1.0 |
| primary / 1 | qrs_change_descriptive | 6.691 | -2.747 to 16.129 | not confirmatory |

Fixed primary prioritization gate passed: False. This is not a novelty, mechanism, or clinical-benefit verdict.

The first execution stopped on timestamp inconsistency. Under the explicit pre-outcome linkage amendment, inconsistent machine records were rejected without replacing the original ECG pairs. See timestamp_consistency.json and SPO2_ECG_TIMESTAMP_AMENDMENT.md.

If the joint support gate failed, no outcome changes or effect models were computed. All denominators and model gate failures are saved. Rhythm exclusions are machine-text flags; absence of a flag is not confirmed sinus rhythm. No ECG waveform-quality validation has been performed.

Clock disagreement, selection for repeated ECGs, medication/electrolyte changes, lead placement and residual clinical confounding remain unresolved. Recorded-time margins do not establish true acquisition ordering. A positive machine-derived result would require the additional validation specified in the protocol; a failed primary cannot be rescued by sensitivities.

The biological discovery goal remains unfulfilled.
