# Diagnostic ECG / charted EKG temporal corroboration

Frozen before reading patient-level procedure timestamps or ECG overlap counts. This follows the metadata-only linkage protocol in the original MIMIC heart-failure cohort; prior negative troponin results remain unchanged. No mortality, ECG amplitudes, machine measurements, or interpretations enter this assessment.

The diagnostic ECG source warns that machine clocks may not be synchronized with clinical clocks. The local dictionary identifies item 225402, EKG, in procedureevents. According to the [official procedure documentation](https://mimic.mit.edu/docs/iv/modules/icu/procedureevents.html), starttime describes the event and storetime its documentation. Neither the label nor agreement with a nearby ECG proves an acquisition-time link.

## Extraction

Read item 225402 from local procedureevents.csv, retaining only exact subject_id/hadm_id/stay_id matches to the existing 4,711-encounter cohort. Record identifiers, start/end/store timestamps, orderid/linkorderid and statusdescription. Hash the source before and after extraction. Count missing/invalid times, statuses, exact duplicates, procedure durations and documentation delays; remove only exact selected-column duplicates. Do not infer exposure or outcomes from these rows.

## Fixed temporal comparison

Use the publisher-verified ECG registry's valid paths and parseable times for the same subjects. For each charted EKG with a valid starttime, enumerate every ECG within an inclusive 24-hour recorded-time distance. Require subject equality, but do not require nominal ECG admission equality, since that itself assumes clock alignment. Count all candidates before any choice. A candidate is isolated only if the procedure has exactly one candidate ECG and that ECG has exactly one candidate procedure across all selected encounters. Never choose the nearest among several candidates. Procedures with no candidates or multiple candidates remain in the denominator and are reported separately.

For isolated pairs, report ECG recorded time minus charted procedure start in minutes: median, quartiles, 1st/99th percentiles, and fractions with absolute difference at most 5, 15, 30, 60 and 240 minutes. Separately report available procedure start-to-store delays and procedure durations. Repeat descriptive offsets for isolated pairs with status FinishedRunning; this is a status restriction, not a new matching algorithm. Count paired ECGs outside the procedure's nominal hospital admission. Do not modify clocks or assign corrected ECG timestamps.

Independently reproduce all candidate edges, degree counts, isolated-pair identities and offsets with Python, comparing to SQL. Retain patient rows only under Drive Data; write aggregate outputs to the project. Empty support is a valid result; do not widen the matching interval after seeing counts.

## Interpretation and stopping rule

Isolation within 24 hours is a conservative ambiguity screen, not a validated record linkage: unrelated procedures can still match and real repeated ECGs can be excluded. The interval truncates possible offsets, so these distributions cannot estimate unconditional clock error. No fraction threshold will be called clock validation, and no acute before/after biological claim will be accepted from this assessment alone. Any later electrophysiology analysis requires its own frozen biological question, timing assumptions and sensitivity analyses. In particular, generic hypoxemia–arrhythmia or repolarization associations are already known ([HF study, 1991](https://pubmed.ncbi.nlm.nih.gov/1907836/); [repolarization study, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11380986/)) and cannot count as the requested novel discovery.
