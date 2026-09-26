## Encounter-boundary amendment, before trajectory directions

The first feasibility pass used MIMIC `admittime` as encounter start and
found 17 complete trajectories. A source check then found that 987 of the
1,398 exposed encounters had an earlier `edregtime`, with a median 190
minutes from ED registration to inpatient admission. The [official MIMIC
documentation](https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html)
distinguishes ED registration from inpatient admission.

For the revised feasibility pass, define MIMIC encounter start as the earlier
of `edregtime` and `admittime` from the same `hadm_id` row. Continue to require
an explicit matching laboratory `hadm_id`; do not assign unlinked samples by
proximity or by patient identity alone. Retain the original 24-hour pre-ICU
extraction bound, 12-hour pre/post-episode windows, 60-minute pre-sample
separation, 120-minute post-sample delay, assay-selection rule and 100-complete-
trajectory gate in both databases. eICU's hospital boundary is unchanged.

This amendment follows inspection of feasibility counts and source metadata,
but precedes inspection of trajectory directions, ratios or mortality.
Preserve the original specification, lock and results. This is a corrected
feasibility assessment, not a claim that the ED rule was prospectively frozen
before the first support pass or that a failed significance test was rescued.
