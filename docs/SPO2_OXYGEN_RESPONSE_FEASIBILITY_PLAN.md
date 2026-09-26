# Can the current records support an oxygen-response mechanism test?

Only MIMIC and eICU, the existing HF cohort, and ICU minutes [0,240).
This is a new outcome-blind feasibility test, not a new biological finding.
The previous Hb experiment is closed as unsupported. No thresholds, outcomes
or failed gates from that experiment are changed.

The unresolved question is whether apparent SpO2 recovery identifies different
pulmonary physiology when it requires escalating inspired oxygen. This could
help distinguish restoration of gas exchange from persistent oxygen dependence,
but charted oxygen changes are clinician decisions, not randomized challenges.
Delivered FiO2, interface changes, positive pressure, ventilatory demand and
clinical response are incompletely measured. Do not estimate a physiological
oxygen-response slope or treatment effect from undocumented exposure intervals.

Routine oxygen targeting is not a novel hypothesis: the
[REDOX-AHF protocol](https://pubmed.ncbi.nlm.nih.gov/42172256/) already compares
restrictive and liberal targets in acute HF. Prior
[HiLo-HF trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC6676301/) and
[eICU observational work](https://pmc.ncbi.nlm.nih.gov/articles/PMC8268364/)
also preclude a first-discovery claim for oxygen treatment in HF.

## Locked measurement requirements

Use the previously audited numeric FiO2 records in hemoglobin_interaction.duckdb
and production-quality SpO2 bins in spo2_context.duckdb. No new raw extraction,
imputation of room air, forward filling, or reuse of ignored caches.

Collapse same-time valid FiO2 values (21–100%) by median. Define an observed
escalation as an adjacent rise of at least ten percentage points with a
strictly positive interval no longer than 60 minutes. This is the time of
documentation, not a guaranteed time of oxygen administration. Count all such
transitions and select the first per encounter for a future independent-person
analysis. Keep the original SpO2 exposure unchanged; do not select transitions
based on subsequent saturation response, troponin or mortality.

For each transition t, require a pre-transition SpO2 representative timestamp
in [t−30,t) and a post-transition timestamp in [t+5,t+30], both inside [0,240).
Select the latest pre and earliest post timestamps. Do not compute their
value difference yet. Separately assess whether another FiO2 record at or
after that selected post-SpO2 timestamp and no later than t+30 corroborates
the increased setting within five percentage points. This does not prove
continuous administration; it merely makes sustained exposure less ambiguous.

Report available encounters, timestamps, eligible escalations, paired-SpO2
support and corroboration by database and original SpO2 exposure. Do not join
death, troponin concentrations, outcome flags or response magnitudes. A later
outcome protocol is justified only if both sources have at least 200 distinct
encounters with the full timing/corroboration requirements and at least 50
in each original exposure group. This is a minimal feasibility screen, not
an interaction-power calculation or a clinical-validity guarantee. If it
fails, close this specification without widening windows or thresholds.

## Direct ischemia measurements checked

The supplied MIMIC dictionary's ST-segment item 228305 is a monitoring-enabled
checkbox, not an ST voltage. eICU has st1/st2/st3 columns but the supplied
schema does not establish comparable lead identities and units. Targeted
checks found no ECG directory, waveform directory, record_list.csv or
machine_measurements.csv at the checked MIMIC locations. These checks do not
prove no such release exists elsewhere; they do establish that a direct
cross-database ST test cannot be inferred from the inspected sources. Never
reinterpret the checkbox as myocardial ischemia.

Keep patient-level timing records private. Embed the executable feasibility
test in the primary notebook and label its actual execution environment.
Feasibility counts cannot satisfy the requested biological-discovery goal.
