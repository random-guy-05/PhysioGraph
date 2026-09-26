# Acid–base mortality interaction screen

**Actual execution: local. Crude observational patient-level screen; no treatment-effect estimate.**

| Source | Instability | Respiratory alkalemia | Deaths / known mortality | Missing mortality | Risk | Simultaneous risk interval |
|---|---|---|---:|---:|---:|---|
| eicu | 0 | False | 105 / 957 | 10 | 10.97% | 8.38% to 14.01% |
| eicu | 0 | True | 9 / 58 | 3 | 15.52% | 5.25% to 32.32% |
| eicu | 1 | False | 119 / 594 | 10 | 20.03% | 15.75% to 24.87% |
| eicu | 1 | True | 11 / 47 | 2 | 23.40% | 9.23% to 43.72% |
| mimic | 0 | False | 38 / 550 | 0 | 6.91% | 4.29% to 10.39% |
| mimic | 0 | True | 5 / 52 | 0 | 9.62% | 1.91% to 25.94% |
| mimic | 1 | False | 14 / 133 | 0 | 10.53% | 4.56% to 19.79% |
| mimic | 1 | True | 5 / 13 | 0 | 38.46% | 8.46% to 77.12% |

| Source | Interaction in absolute mortality risk | Simultaneous conservative bounds | Pass |
|---|---:|---|---|
| eicu | -1.17% | -39.58% to 36.73% | False |
| mimic | 25.23% | -32.98% to 81.05% | False |

Both-database advancement: **False**. The fixed rule requires each simultaneous lower interaction bound to exceed five percentage points. The interaction subtracts the instability-associated risk difference without respiratory alkalemia from that with it. These are percentage-point contrasts, not relative risks.

Eight two-sided Clopper–Pearson intervals use alpha=0.05/8; the interaction bounds follow by interval arithmetic. Their simultaneous coverage is conditional on the independent-patient binomial model. A deterministic outcome-blind choice retains one encounter per person. Unknown mortality is reported and not imputed.

These estimates are unadjusted and cannot isolate oxygen unloading, coronary vascular effects, severity, treatment, or selection into blood-gas measurement. Small joint-state groups sharply limit precision. No non-advancement establishes biological absence, and no positive percentage establishes a mechanism or mortality-saving intervention. Both project databases have been explored previously. No novel biological discovery is established.
