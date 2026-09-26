# Carbon-dioxide timing at the original SpO2 transitions

## Biological rationale and prior knowledge

The remaining respiratory question is whether the original rapid SpO2
transitions accompany a ventilatory disturbance or a change compatible with
pulmonary perfusion. End-tidal CO2 (EtCO2) could help distinguish candidate
patterns, but depends on ventilation, perfusion, metabolism and sampling.
A decrease alone is not evidence of reduced cardiac output. Respiratory rate
does not measure minute ventilation; set ventilation and alarm limits do not
measure delivered ventilation. An arterial–end-tidal difference is not a
measurement of true Bohr dead space.

Neither hypercapnia in acute HF nor EtCO2 in cardiogenic shock is new:
[acute-HF airway intervention study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5746960/),
[2004 cardiogenic-shock study](https://www.jstage.jst.go.jp/article/jnms/71/3/71_3_160/_article/-char/en),
and [2026 arterial–end-tidal gradient study](https://pmc.ncbi.nlm.nih.gov/articles/PMC13126784/).
These precedents rule out presenting a generic prognostic association as a
novel mechanism. The narrower candidate concerns CO2 dynamics at the existing
SpO2 episodes, conditional on enough temporal and ventilatory context.

## Frozen first stage: metadata only

This plan precedes patient-level EtCO2 coverage counts and amplitudes. Native
headers, MIMIC item labels and an eICU respiratory-label catalogue have already
been inspected. All earlier project findings, including the failed coupled
potassium/glucose event test, are known. This is a timing feasibility test,
not a biological result or a new mortality screen.

Use the existing lowest-stay-per-person selection from the original dynamics
cohorts: 10,167 eICU and 4,305 MIMIC people. Reconstruct the original first
absolute SpO2 jump of at least four percentage points in adjacent 15-minute
median bins with a forward representative-time gap in (0,30] minutes.
Retain its original sign; never substitute a later downward event. The exposed
sets must agree with the original 3,798 eICU and 1,276 MIMIC people.

Scan the supplied full raw files with explicit CSV double-quote/escape rules:

- eICU `vitalPeriodic.csv`: nonempty `etco2` at observationoffset in [0,240].
  Retain timestamps, row identifier and presence of a nonempty `respiration`
  field, without selecting either numeric amplitude. The official schema
  describes monitor values archived as five-minute medians; they are not
  bedside-validated observations. It does not specify EtCO2 units.
- MIMIC `chartevents.csv`: item 228640, EtCO2, with nonempty value or valuenum,
  in [0,240], using exact original subject, hospital and ICU stay identifiers.
  Retain chart/store times, unit and warning metadata, without selecting values.
  Separately retain timestamp metadata for item 224687, Minute Volume (L/min),
  as a possible ventilation-context source. Do not include alarm limits,
  percent minute volume, set minute ventilation, status or indication fields.

MIMIC's supplied EtCO2 dictionary unit is unspecified; do not infer mmHg from
its label. Neither missing units nor nonempty text qualify a physiological
measurement. Raw timing support is an optimistic ceiling pending value,
unit, duplicate, warning and sampling-device qualification.

## Pairing and fixed decision

For each exposed person's original episode, let a and b be its two SpO2
representative times and m=(a+b)/2. The primary tolerance is five minutes.
Require full observation windows a-5>=0 and b+5<=240. Select the closest EtCO2
timestamp to a within five minutes and strictly below m, and the closest to b
within five minutes and at least m. Resolve equal-distance ties toward the
earlier timestamp. This prevents one observation from being used twice and
keeps the two CO2 observations in forward order. Duplicate records at one
timestamp do not create repeated measurements. No alternative event is used.

Report original, full-window, pre-available, post-available and paired counts,
including upward/downward SpO2 events and eICU hospital counts. Describe MIMIC
minute-volume timestamp support within the same side-specific five-minute
windows; this is not proof of stable ventilation. Describe eICU respiratory
rate presence at the selected periodic timestamps with the same limitation.

The minimum support floor for a separately specified mechanistic contrast is
50 paired exposed people, at least 20 in each SpO2 direction, in each database,
and at least ten eICU hospitals among paired people. These are feasibility
floors, not a power calculation or a clinically meaningful effect threshold.
Repeat timing counts at 15 minutes only as a descriptive ceiling; they cannot
replace the primary gate or rescue its failure.

Reconstruct all episode times, signs, nearest-pair choices and coverage counts
independently in Python and SQL. Check source hashes and preserve all
patient-level metadata under private Drive Data. Include executed local cells
in the single primary notebook; do not claim Colab or uninterrupted execution.

## Subsequent analysis requires a separate specification

If both primary timing floors pass, freeze numerical qualification, instrument
and unit handling, ventilatory context, the biological contrast and validation
criterion before selecting EtCO2 amplitudes. Medication, ventilation and
sampling changes must be considered before a perfusion interpretation. If the
gate fails, report inadequate temporal support and do not calculate CO2
changes, mortality associations or treatment effects from these pairs.

Sources: [eICU schema](https://eicu.mit.edu/eicutables/vitalperiodic/) and
[eICU database paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6132188/).
