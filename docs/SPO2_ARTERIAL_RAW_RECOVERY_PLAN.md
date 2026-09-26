# Raw-source arterial timing recovery

This is a prerequisite to the original SpO2 biological investigation, not a
new biological endpoint or a discovery. Frozen before the new raw counts.
Earlier negative analyses and cached timing counts are known.

Question: did explicit hospital-ID linkage or a restricted laboratory item
mapping conceal simultaneous independent oxygen measurements at the original
first SpO2 transition? A complete raw laboratory scan can answer this narrower
question; it cannot establish the absence of every physiological measurement.

Use the original 11,452 eICU and 4,711 MIMIC encounters and first absolute
SpO2 jump of at least four points, with positive interval at most 30 minutes.
No new episodes, thresholds, outcomes or cohort members. Retrieve timestamp
metadata within −15 to 255 minutes of ICU admission. Pairing still requires
both complete windows within the original hospital/240-minute boundary,
and disjoint sides of the midpoint, at the existing 5/10/15-minute tolerances.

Scan supplied raw eICU lab.csv and MIMIC labevents.csv. Do not select numeric
oxygen values, clinical outcomes, or interpret result changes. eICU label
normalization is trim/lower, matching pao2 or po2; retain native names and
lab types. MIMIC retains 50821 (blood pO2) and 52042 (fluid pO2) separately.
The latter is an optimistic unqualified ceiling only, never assumed arterial.
The source dictionary was read before this protocol and establishes both
labels; other body-fluid pO2 50832 is excluded.

Retain explicit MIMIC subject/hospital matches. For missing hospital IDs only,
assign a hospital if exactly one admission for that subject contains charttime
in [admittime, dischtime). Determine uniqueness using all supplied admissions,
before joining to the original ICU cohort. Do not reassign discordant explicit
hospital IDs. Record ambiguity. Keep lab timestamps even if storetime/revision
time is missing or after minute 240, because this is an optimistic timing
ceiling, not a bedside-available diagnostic test. The old raw cache already
retained late results; this alone is not claimed to add records to its ceiling.

Compare raw lab IDs/times with the old raw cache in the common range. Require
all old canonical records to reconstruct, independently validate first-event
times and all timing flags in Python, and distinguish additional rows from
additional independent specimen times. Validate recovered missing-hospital
assignments against admissions independently in Python. Missing outcomes
and oxygen values are never inferred.

Report both one-side and two-side timing support. The earlier 385-person
two-sided ±5-minute gate stays unchanged; a permissive ceiling cannot itself
pass the arterial qualification gate. If the ceiling is small, close this
exact two-sample mechanism test without calling all other questions impossible.
If adequate, freeze specimen/unit/value qualification before reading values.

Patient rows and the new DuckDB stay in private Drive Data. Publish aggregate
JSON/Markdown and actual incremental local execution in the single primary
notebook. Validate source fingerprints and leave earlier results unchanged.
