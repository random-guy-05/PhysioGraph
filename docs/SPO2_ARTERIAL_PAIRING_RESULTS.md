# Arterial oxygen pairing feasibility

Actual execution: local. Existing caches; no new raw source scan.

| Source | Timing tier | Tolerance (min) | Full-window encounters | Both-side encounters | Both-side people |
|---|---|---:|---:|---:|---:|
| eicu | qualified_pao2 | 5 | 4302 | 0 | 0 |
| eicu | qualified_pao2 | 10 | 4189 | 1 | 1 |
| eicu | qualified_pao2 | 15 | 4101 | 2 | 2 |
| eicu | raw_pao2_timing_ceiling | 5 | 4302 | 0 | 0 |
| eicu | raw_pao2_timing_ceiling | 10 | 4189 | 1 | 1 |
| eicu | raw_pao2_timing_ceiling | 15 | 4101 | 2 | 2 |
| mimic | qualified_pao2 | 5 | 1371 | 0 | 0 |
| mimic | qualified_pao2 | 10 | 1322 | 1 | 1 |
| mimic | qualified_pao2 | 15 | 1291 | 4 | 4 |
| mimic | raw_pao2_timing_ceiling | 5 | 1371 | 2 | 2 |
| mimic | raw_pao2_timing_ceiling | 10 | 1322 | 4 | 4 |
| mimic | raw_pao2_timing_ceiling | 15 | 1291 | 9 | 9 |

Primary requirement: 385 people with paired qualified PaO2 at +/-5 minutes in each database. Gate: FAIL.

SQL and independent Python reconstruction agree on original episode times, exposed patient sets, and all timing flags. Source SHA-256 hashes were unchanged.

The raw timing ceiling includes unqualified values and nonarterial or ambiguous specimens. It is only a ceiling within the existing hospital-linked extraction. Qualified samples retain prior specimen, unit, range and revision checks; later-available results and samples beyond minute 240 are absent. eICU uses ABG labels without equivalent specimen identifiers. PaO2 does not directly measure SaO2.

No oxygen change, mortality association, mechanism or treatment benefit was estimated. This cached-pairing specification is closed; sensitivity windows do not rescue it. Sparse pairing cannot establish that the SpO2 change is artifact or that true arterial oxygen changes are absent.
