# Capnography at the original SpO2 events

Actual execution: local. Full eICU vitalPeriodic and MIMIC chartevents metadata scans; no numeric CO2 or ventilation values selected.

| Database | Tolerance (min) | Full-window people | Paired people | SpO2 up | SpO2 down | eICU hospitals | Context timestamps on both sides | Floor |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| eicu | 5 | 3772 | 43 | 23 | 20 | 25 | 40 | False |
| eicu | 15 | 2964 | 30 | 13 | 17 | 18 | 27 | False |
| mimic | 5 | 1207 | 5 | 2 | 3 | NA | 1 | False |
| mimic | 15 | 950 | 3 | 1 | 2 | NA | 0 | False |

Primary joint five-minute timing floor: False. The 15-minute rows are descriptive and do not replace it.
The larger tolerance also requires larger full windows within minutes 0–240 and therefore excludes more events near the boundaries. Its eligible set differs; paired counts need not increase with tolerance.
The original episode signs and timestamps, all nearest choices, forward order, full windows and context flags agree in independent SQL/Python reconstruction. Source hashes were recorded and input database hashes rechecked.
Context means respiratory-rate field presence in eICU and actual minute-volume item timestamps in MIMIC; neither establishes constant minute ventilation. Nonempty chart text, unqualified units and warning flags are still included in this optimistic timing ceiling.
No CO2 change, mortality association, biological mechanism or treatment benefit was estimated. The primary cross-database event-scale contrast is unsupported by these recordings; this does not reject the underlying biological mechanism.
