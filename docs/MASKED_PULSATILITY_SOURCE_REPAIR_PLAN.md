# Masked Pulsatility Source Repair and Final Outcome-Blind Feasibility Plan

This plan is frozen after nurse-label discovery and before repaired crossing
prevalence is computed. It does not alter or overwrite the prior feasibility
analysis.

## Sources and exact mappings

- `vitalPeriodic.csv`: same-row `systemicsystolic`, `systemicdiastolic`, and
  `systemicmean`; source `ibp_vitalPeriodic`.
- `vitalAperiodic.csv`: same-row `noninvasivesystolic`,
  `noninvasivediastolic`, and `noninvasivemean`; source
  `nibp_vitalAperiodic`.
- `nurseCharting.csv`: only category `Vital Signs`, label `Non-Invasive BP`,
  and exact names `Non-Invasive BP Systolic`, `Non-Invasive BP Diastolic`, or
  `Non-Invasive BP Mean`; source `nibp_nurseCharting`. SBP and DBP are paired
  only at an identical `nursingchartoffset`.

No fuzzy label is used for production mapping. Invasive nurse labels are
audited but excluded from the cuff stream.

## Cuff deduplication

All valid `vitalAperiodic` cuff pairs are retained. A nurse cuff pair is
considered a duplicate only when the same stay has a `vitalAperiodic` pair
within plus or minus one minute with numerically identical SBP and DBP and MAP
is identical or missing in either representation. The structured
`vitalAperiodic` row wins. A missing aperiodic MAP may be filled from its matched
nurse representation. All unmatched nurse cuff pairs remain. Deduplication is
performed before hourly aggregation or crossing counts.

## Pairing, validation, and hourly summaries

- Window: ICU minutes `[0, 240)`.
- Pair SBP/DBP only within the same row/offset and BP source.
- Valid pair: SBP 50–300, DBP 20–200, and SBP > DBP.
- Direct MAP, when present, must be 20–200 and between DBP and SBP to qualify
  for the direct-MAP compensation layer.
- Derived MAP is `(SBP + 2*DBP)/3` and is reported separately; it never silently
  replaces the prior direct-MAP definition.
- Hourly source-specific medians define PP and PrPP. Cuff and invasive values
  are never averaged together.

## Exposure and pre-anchor layers

The first source-consistent hourly transition from PrPP >=0.25 to PrPP <0.25 is
the incident crossing. Stays starting below 0.25 are excluded from incident
crossing counts. Report sequentially: pair-only; crossing SBP >=90; direct MAP
>=65; direct-or-derived MAP >=65; no prior pressor/inotrope; no prior MCS; and
pre-crossing lactate status (none, <2, >=2). Measured lactate is not required for
exposure feasibility.

Controls are non-crossing stays with valid routine cuff monitoring and a
deterministic anchor-time quantile match, SBP >=90, direct MAP >=65, and no
prior pressor/MCS. No future outcome is used for anchor assignment.

Post-anchor lactate is represented only by timestamp-existence flags for 0–6
and 0–12 hours. Its value is prohibited. All clinical outcomes, effect
estimates, and threshold alternatives are prohibited.

## Gate

- Strong: at least 300 compensated routine-cuff crossings per database, at
  least 500 controls per database, and at least 10 eICU hospitals with
  crossings.
- Acceptable: at least 150 crossings per database, at least 300 controls per
  database, and at least 10 eICU hospitals with crossings.
- Weak: 75–149 crossings in either database or failure of a non-fatal support
  criterion.
- Fail: fewer than 75 routine-cuff compensated crossings in either database or
  severe center concentration. Severe center concentration is prospectively
  defined as the top eICU hospital contributing more than 25% of crossings or
  the top five hospitals contributing more than 60%.

Passing stops with `proceed_to_protocol_freeze`; it does not authorize outcome
inspection or Phase B.
