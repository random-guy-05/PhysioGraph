# Arterial acid–base feasibility

**Actual execution: local. No mortality or troponin outcome selected.**

| Source | Original dynamics cohort | Pre-ICU complete gas | Also available by ICU admission | First-four-hour complete gas |
|---|---:|---:|---:|---:|
| eicu | 11452 | 1767 | 1671 | 1997 |
| mimic | 4711 | 757 | 717 | 1580 |

| Source | SpO2 instability | Preceding respiratory alkalemia | Encounters |
|---|---|---|---:|
| eicu | 0 | False | 1016 |
| eicu | 0 | True | 66 |
| eicu | 1 | False | 634 |
| eicu | 1 | True | 51 |
| mimic | 0 | False | 559 |
| mimic | 0 | True | 52 |
| mimic | 1 | False | 133 |
| mimic | 1 | True | 13 |

Respiratory alkalemia here denotes pH>7.45 with PaCO2<35 mmHg, not a definitive primary-disorder diagnosis. MIMIC requires explicit arterial specimen text and explicit hospital linkage. eICU uses ABG-labelled co-timed panels and lacks equivalent specimen identifiers. Units, revision conflicts, availability and physiological ranges are checked before panel selection. Exclusion counts overlap and should not be summed.

The preceding-state analysis cannot substitute a first-four-hour gas when its pre-ICU measurement is missing. These counts do not establish an association, mechanism, novelty or treatment benefit. No P50 or myocardial oxygen-unloading estimate was calculated. Subsequent association work requires a separate frozen model specification. The discovery goal remains unmet.
