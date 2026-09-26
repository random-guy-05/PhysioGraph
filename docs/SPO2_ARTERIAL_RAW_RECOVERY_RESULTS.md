# Full raw laboratory timing recovery

Actual execution: local. Full supplied lab scans; metadata only.

| Dataset | Tier | Tolerance | Full-window encounters | Both-side people |
|---|---|---:|---:|---:|
| eicu | canonical_explicit | 5 | 4302 | 0 |
| eicu | canonical_explicit | 10 | 4189 | 1 |
| eicu | canonical_explicit | 15 | 4101 | 2 |
| eicu | all_recovered_pao2_labels | 5 | 4302 | 0 |
| eicu | all_recovered_pao2_labels | 10 | 4189 | 1 |
| eicu | all_recovered_pao2_labels | 15 | 4101 | 2 |
| mimic | canonical_explicit | 5 | 1371 | 2 |
| mimic | canonical_explicit | 10 | 1322 | 4 |
| mimic | canonical_explicit | 15 | 1291 | 9 |
| mimic | all_recovered_pao2_labels | 5 | 1371 | 2 |
| mimic | all_recovered_pao2_labels | 10 | 1322 | 4 |
| mimic | all_recovered_pao2_labels | 15 | 1291 | 9 |

Legacy common-window rows reconstruct exactly; independent Python agrees with SQL on original episodes, all timing flags, and missing-hospital admission assignments.
These are permissive laboratory timestamp ceilings. Extra fluid-pO2 labels are not assumed arterial; values and mortality were not selected. This does not census charted gases or all physiological measurements.
