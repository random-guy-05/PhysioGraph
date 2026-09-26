# PhysioGraph extension: circulatory context of SpO2 instability

Locked 2026-09-05 UTC before extraction of the new joint-signal features.
Scope: **MIMIC and eICU only**, original files under My Drive/Data. This develops
the current SpO2-instability project, especially its cross-database myocardial
injury association. No molecular or unrelated datasets are part of this work.

## Biological question

Does an abrupt SpO2 change accompanied by a contemporaneous fall in systemic
arterial pressure distinguish episodes followed by myocardial injury and
death from isolated SpO2 changes? This tests a circulatory-context hypothesis;
it does not assume that a peripheral oximeter measures cardiac output or
microcirculation. The prior analysis already found that recovery rises can
predict events, so absolute changes remain primary and direction is explicit.

## Unchanged anchors

Use the production PhysioGraph adult, explicit-HF phenotype, first ICU per
hospital encounter, early ICU admission or cardiogenic shock, and exclusion
of death or end of observable follow-up by four hours. Follow-up ends at
the earlier of ICU and hospital discharge, matching production code.
Use 0-4-hour SpO2 values in 50-100%,
same-time medians, 15-minute median bins, at least three bins and two
forward transitions no more than 30 minutes apart. The SpO2 exposure remains
an absolute adjacent change of at least four percentage points.

## New feature and comparators

For each qualifying SpO2 transition, obtain MAP medians in its two bins.
Never carry a pressure measurement across an empty bin. A circulatory decline
is a MAP decrease of at least 10 mmHg over that same transition. MAP values
outside 20-200 mmHg are excluded; the threshold is fixed before outcomes.

Classify eligible encounters into four states: neither event, SpO2 change
only, MAP decline only, both. Also record whether the events occur on the
SAME transition; co-occurrence anywhere within four hours is not synchrony.
Require at least two jointly observed transitions. Do not label missing MAP
as stable circulation. Preserve arterial and noninvasive pressure provenance;
eICU currently lacks the noninvasive-pressure source.

Primary question: does synchronous instability carry information beyond
separate SpO2-jump and MAP-decline indicators, their event burdens, absolute
levels, age, sex, heart rate and sampling? Secondary fixed checks: downward
versus upward SpO2 jumps; stable absolute oxygenation (all bins >=90%);
normal measured baseline lactate (<2 mmol/L); invasive-pressure-only MIMIC.
Normal lactate requires an actual measurement; missing is not normal.

## Endpoints and safeguards

Two primary endpoint families: subsequent hospital death and the existing
assay-matched troponin rise endpoint after the four-hour landmark. For the
first feasibility/feature build, extract raw labs through 28 hours and report
coverage before any effect estimates. Freeze the exact window and model
after verifying assay/time support, before new feature-outcome associations.
This document freezes the feature, not a fully specified final model.

An eventual mortality model must compare against separate component signals
and baseline physiology/respiratory treatment, with patient-clustered
uncertainty and eICU hospital heterogeneity. No future lactate, troponin,
ventilation or pressors enter baseline adjustment. Myocardial injury is not
adjudicated infarction. Troponin ordering requires an observation analysis.
Do not claim full SCAI equivalence when source fields are missing.

Minimum support for a primary fitted contrast: 100 endpoint events overall,
at least 20 events and 20 non-events in the synchronous group and comparator,
and adequate model/weight support. Sparse strata remain descriptive. Do not
choose a replacement threshold or endpoint to rescue a failed result. Report
both databases separately, transfer definitions without retuning, and retain
negative results. Previously explored databases are not pristine holdouts.

## Gates for advancing

A robust cross-database association is only the first gate. Useful early
warning also requires improvement over component signals, lactate and
faithfully available shock measures; calibration/net benefit; temporal and
treatment controls; and direct mechanistic corroboration. Mortality reduction
needs intervention evidence. Neither association nor a new score alone
completes the user's paradigm-shifting discovery goal.
