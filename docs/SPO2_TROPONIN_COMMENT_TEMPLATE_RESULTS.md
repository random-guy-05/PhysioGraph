# MIMIC reporting-bound template recovery

Actual execution: local. Separate amendment after the strict whole-comment audit; template frequencies were already known.

| Comment category | Existing numeric value accepted | Unit | Source records | Missing numeric value |
|---|---|---|---:|---:|
| interpretation_only | True | ng/ml | 3446 | 0 |
| lower_bound_25 | False | ng/ml | 3 | 3 |
| unrecognized | False | ng/ml | 20 | 20 |
| unrecognized | True | ng/ml | 691 | 0 |
| upper_bound_0.01 | False | ng/ml | 598 | 598 |

A leading bound is separated from the assay interpretation statement. The 0.10 ng/mL comparison inside that statement is never used as the patient value or an MI diagnosis. Other comment formats remain unrecognized.

All source-ID classifications agree between SQL and Python; counterexamples pass; the comments table and reference hashes are unchanged. Recovered bounds remain intervals. No new clinical endpoint, association, biological mechanism or mortality benefit has been established.
