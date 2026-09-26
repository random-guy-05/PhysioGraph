# Diagnostic ECG / charted EKG temporal corroboration

Execution: local incremental notebook cell. Outcome-blind metadata analysis; no ECG values.

3343 unique charted EKG procedures in 1980 original encounters. 6290 candidate edges within ±24 hours; 711 isolated pairs in 609 people.

Valid-start procedures without a candidate: 743; with multiple candidates: 1784.

| Maximum absolute recorded-time difference | Isolated pairs | Denominator |
|---|---:|---:|
| 5 min | 269 | 711 |
| 15 min | 460 | 711 |
| 30 min | 557 | 711 |
| 60 min | 618 | 711 |
| 240 min | 650 | 711 |

Signed ECG-minus-procedure difference: {"n": 711, "p01": -1084.5, "q25": -15.0, "median": -4.0, "q75": 3.0, "p99": 1143.5999999999979} minutes.

These are isolated candidate matches, not validated acquisition links. The matching window truncates possible disagreement; documentation delays and coincidental matches remain possible. No timestamps were corrected, and no acute biological sequence or mortality benefit is established.

Independent SQL/Python candidate edges, degrees, isolated identities and offsets agree. Original failed biological contrasts remain unchanged.
