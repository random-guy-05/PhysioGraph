# Venous-pressure timing feasibility: actual local results

| Database | Stage | Encounters | People |
|---|---|---:|---:|
| eicu | first_original_episode | 4309 | 4057 |
| eicu | full_windows | 1650 | 1600 |
| eicu | any_both_windows | 77 | 76 |
| eicu | two_both_windows | 74 | 73 |
| eicu | span_at_least_15 | 66 | 65 |
| eicu | complete_timing | 66 | 65 |
| mimic | first_original_episode | 1398 | 1344 |
| mimic | full_windows | 423 | 418 |
| mimic | any_both_windows | 23 | 23 |
| mimic | two_both_windows | 11 | 11 |
| mimic | span_at_least_15 | 7 | 7 |
| mimic | complete_timing | 6 | 6 |

Timing count floor passed: **False**. Comparable pressure units established: **False**.

eICU counts are optimistic timing support from finite native CVP values, without physiological range validation. MIMIC uses explicit mmHg records passing the frozen quality criteria. Neither the size of a timing sample nor its presence proves a biological mechanism.

Independent Python reconstruction agrees with SQL on every first-episode identity/time and every complete-timing encounter. First-episode presence reconciles to the original SpO2 exposure in both databases. No later episode substitution was used.

No CVP change, MAP response, biomarker association or mortality effect was calculated. The gate result governs this specification and must not be represented as absence of venous-congestion biology. No discovery or mortality benefit is established.
