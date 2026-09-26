# Masked pulsatility: outcome-blind feasibility specification

Frozen before the first masked-pulsatility raw-source scan. This document
authorizes Phase A only. If and only if both databases pass the fixed support
gate, a separate analysis plan must be written and hashed before any
post-anchor lactate value is selected.

## Population and observation window

Reuse the production adult explicit-heart-failure cohort, first ICU stay per
hospital encounter, early ICU/cardiogenic-shock qualification, and the existing
exclusion of death or end of observable follow-up by four hours. The early BP
observation period is ICU minutes `[0, 240)`; it will not be changed based on
support or outcomes.

## BP construction

Use original MIMIC `chartevents.csv` and eICU `vitalPeriodic.csv`.

- MIMIC invasive arterial IDs: SBP 220050, DBP 220051, MAP 220052.
- MIMIC noninvasive cuff IDs: SBP 220179, DBP 220180, MAP 220181.
- eICU periodic columns: `systemicsystolic`, `systemicdiastolic`,
  `systemicmean`, classified as the supplied invasive arterial source.
- First form same-timestamp, same-source SBP/DBP/MAP triplets. Never pair
  components from different timestamps or sources.
- Valid triplets require SBP 50–300, DBP 20–200, MAP 20–200 mmHg, SBP > DBP,
  and MAP between DBP and SBP.
- Within each ICU hour and source, take the median of valid triplets. Anchor
  time is the latest contributing timestamp in that hour, preventing use of
  measurements after the anchor.
- Select one primary source per stay before trajectory construction: greatest
  number of valid hourly bins, then invasive arterial, then noninvasive, then
  unknown. Source cannot change across a transition.

Calculate `PP = SBP - DBP` and `PrPP = PP / SBP` without rounding.

## Incident crossing and compensated state

The first valid hourly PrPP bin must be at least 0.25; patients already below
0.25 in their first valid bin are not incident cases. The exposure is the first
later valid bin below 0.25, after at least one prior bin at or above 0.25.
Missing intervening hours are allowed; the transition is between consecutive
valid hourly bins from the same selected source.

At the crossing, require SBP ≥90 mmHg, MAP ≥65 mmHg, patient alive and observed,
most recent valid lactate <2 mmol/L, and no vasoactive/inotropic infusion or MCS
documented by the anchor. Baseline lactate must be collected and available in
`[anchor-240, anchor]` minutes. MIMIC uses chart time plus store time; eICU uses
result offset plus revised-result offset. Plausible lactate is `(0, 30]` mmol/L
with an explicit compatible unit.

## Controls

Controls come from stays with no incident crossing in `[0, 240)`. Candidate
hours require PrPP ≥0.25 and the same compensated-state rules. Sort controls by
stable stay ID, map them deterministically to quantiles of the exposed anchor
distribution, and choose the eligible control hour nearest its assigned target
(earlier hour breaks ties). No follow-up measurement or outcome information
enters assignment.

## Outcome-blind follow-up support

After final anchors are fixed, inspect only whether a lactate measurement
timestamp exists in `(0, 360]` and `(0, 720]` minutes after anchor. Do not select,
load, aggregate, compare, or print post-anchor lactate values or any clinical
endpoint.

## Gate

Each database must have at least 100 incident crossings, 200 controls, 75
crossings with a 0–6-hour follow-up lactate measurement, and 150 controls with
one. Preferred support is 200 crossings and 400 controls. Failure in either
database sets `masked_pulsatility_feasible=false` and stops the program without
changing the 0.25 threshold, cohort, bin width, or windows.

All patient-level tables remain in the existing private Drive/Data database.
Repository outputs are aggregate only and execution provenance must distinguish
local from genuine Google Colab execution.
