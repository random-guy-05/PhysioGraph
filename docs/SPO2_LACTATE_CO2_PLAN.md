# Acute coupled lactate, arterial CO2 and pH change

## Question, prior knowledge and limitations

Does the original SpO2 instability episode precede a coupled rise in lactate
and arterial CO2 with falling pH more often than fixed no-episode control
anchors? This is a candidate combined metabolic/respiratory deterioration
pattern, not a diagnosis of respiratory muscle fatigue, failure of a measured
chemoreflex, lactate-caused acidosis, or proof of tissue hypoxia.

The original exploratory lactate result concerns a tail event, with little
median rise and attenuation under observation weighting. It motivates testing
a coupled individual response rather than assuming a mean shift. The previous
potassium/glucose, calcium, methemoglobin and event-aligned capnography results
are known and remain unchanged.

Generic lactate/acid-base prognosis in HF is established in a
[2022 study](https://academic.oup.com/ehjacc/article/11/3/242/6529439).
A [2026 VA-ECMO study](https://doi.org/10.1161/JAHA.125.043615) also studied
combined respiratory and metabolic acidosis. Neither that generic association
nor a new name for a familiar blood-gas disorder constitutes novelty here.
This experiment asks a narrower event-anchored paired-change question. A
positive result would remain exploratory until treatment, observation,
temporal specificity and mortality relevance were independently evaluated.

## Frozen design before new joint counts or changes

MIMIC and eICU only. Reuse the previously independently validated, lowest
original stay per person: 10,167 eICU and 4,305 MIMIC people. Reuse the exact
original first-episode and deterministic no-episode timestamps from
`adrenergic_metabolic.{dataset}_anchors`, including its existing hash rule;
do not regenerate controls to improve matching or sample size. Independently
compare these anchors to original SpO2 bins again.

Scan full raw laboratory sources with explicit CSV quote and escape rules.
Window: [max(hospital start, anchor−180), min(anchor+120, follow-up end)].
Retain raw text, numerical strings, units, specimen IDs, collection and entry
times. MIMIC assays: blood-gas lactate 50813, pCO2 50818, pH 50820 and specimen
type 52033; dictionary labels verified before locking. Use exact hospital IDs;
recover missing IDs only when one unique all-admission interval contains the
collection time and matches the original encounter, using the prior audited
admission/ED intervals. Never reassign an explicit hospital ID.

eICU pH and paCO2 require native labtype 7. Lactate uses the exact native
`lactate` label with labtype 1 or 7. A co-timed lactate result does not prove
the same specimen or arterial lactate in eICU; retain this limitation. MIMIC
requires all three numerical components at the same collection time and
specimen ID, explicitly labelled ART., ART or ARTERIAL by a single consistent
normalized specimen-type label. Unknown, venous, mixed, conflicting or absent
specimen labels do not qualify as arterial.

Choose the last source-eligible pCO2 timestamp strictly before the anchor and
the first strictly after it. MIMIC specimen metadata must identify the timestamp
as arterial before this selection; if more than one arterial specimen exists
at that timestamp, reject it. Do not select another timestamp after a missing
component or numerical-quality rejection. No sample at the exact anchor is
used. No measurement outside the fixed window is substituted.

## Numerical and record qualification

Require ordinary decimal numerical strings without inequality or censored
text. Parse with exact decimal arithmetic. pH units may be blank, pH or units;
pCO2 requires mmHg or kPa, with all nonblank source/interface labels agreeing
in eICU (kPa×7.50061683). Lactate requires mmol/L (including MMOLL spelling);
eICU's two unit labels must both agree. Do not infer missing lactate units or
substitute mg/dL without a separately justified protocol.

Plausibility limits: pH 6.5–8.5, pCO2 5–250 mmHg, lactate 0–40 mmol/L.
For each component at the fixed timestamp, require nonmissing entry times;
select the latest entry first, reject conflicting latest values, invalid
latest entries or differing normalized unit tuples. MIMIC components must
share the chosen arterial specimen. Explicitly censored entries are not exact
values. No imputation or replacement panels. Keep patient-level data private.

## Primary joint event and analysis

The joint event is lactate increase ≥0.5 mmol/L AND pCO2 increase ≥5 mmHg AND
pH decrease ≥0.03 between the two fixed panels. These are predeclared substantial
change thresholds, not a validated syndrome or diagnostic compensation formula.
Lactate ≥0.5 follows the original project endpoint; the CO2/pH thresholds are
investigator choices fixed before new changes. Use exact decimal comparisons
and independently reproduce event labels in SQL.

Report the exposed/control risk difference in each database. The minimum
paired-panel support is 50 people and 20 per exposure group per database,
with at least ten eICU hospitals. Before uncertainty estimation, require at
least five joint events and five non-events in each group, with eICU events
represented in at least three hospitals per group. If the panel support fails,
do not calculate changes. If only the event support fails, report exact event
counts and descriptive risk differences without p-values or confidence limits.

If supported, use 20,000 hospital-cluster bootstrap draws in eICU and person
bootstrap draws in MIMIC, with fixed seeds 202609061 and 202609062. The primary
advancement criterion is a positive lower bound of the two-sided 95% interval
in BOTH databases. This conjunction tests replication, not pooled significance.
Do not promote a favorable component or change thresholds, windows or outcome
when the joint criterion fails. This screening contrast is unadjusted; no
mortality or treatment model is authorized by this protocol alone.

Independently verify original anchors, hospital assignment, fixed sample
selection, quality flags, exact event labels and aggregate counts. If bootstraps
run, directly expand 100 draws per source to check the implementation. Record
full source hashes, execution environment, failures and corrections in the
single primary notebook. Local execution is not Colab execution. Even a
replicated result is not by itself the requested paradigm-shifting discovery.
