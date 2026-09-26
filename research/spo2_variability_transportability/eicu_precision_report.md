# eICU precision audit

Execution environment: **LOCAL**. No model was refit for this phase; precision is derived from the frozen hospital-clustered estimate.

- Locked RR: 1.104
- log(RR): 0.099132
- implied cluster-robust SE: 0.109840
- 95% CI: 0.890–1.369
- multiplicative 95% CI factor: ×/1.240
- RR-scale CI width: 0.479

## Approximate power at 85 events

- True RR 1.10: 14.0% power
- True RR 1.15: 24.6% power
- True RR 1.20: 38.2% power
- True RR 1.25: 52.9% power
- True RR 1.30: 66.6% power

## Approximate event requirements

- RR 1.20, 80% power: approximately 243 events
- RR 1.20, 90% power: approximately 325 events
- RR 1.25, 80% power: approximately 162 events
- RR 1.25, 90% power: approximately 217 events

These are explicitly normal-approximation calculations holding the observed exposure distribution and hospital-clustering design effect constant through the observed robust SE. They are a precision audit, not an excuse for the failed validation and not a replacement for the frozen external decision.
