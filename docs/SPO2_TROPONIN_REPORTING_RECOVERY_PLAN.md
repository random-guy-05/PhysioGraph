# Reporting-limit recovery and selection audit

Specified September 5, 2026 after the completed assay audit found 1,800
excluded explicitly bounded eICU results and placeholder text for 4,137
retained MIMIC troponin results. These counts and previous associations are
known. This follow-up is a source/selection audit, not a new outcome analysis.

## MIMIC comments

The [official table documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)
describes laboratory comments separately from value and valuenum. Rescan
only the same local labevents file, retaining comments for the exact source
IDs already present in `troponin_assay_audit.duckdb`. Preserve the original
source-file size/mtime and numeric/text/item identity. Store comments only
in a new private DuckDB file; print no free-text examples or patient IDs.

Apply the existing frozen strict parser to the entire comment. Accept only
a standalone numeric literal or standalone reporting bound, including its
already specified symbolic and less-than/greater-than forms. Do not mine
numbers or limits from narrative comments. Report classifications and the
number of existing numeric records with a matching exact comment or a
standalone bound. Neither a comment nor the printed reference range is
automatically an assay-specific 99th percentile. Do not replace labels.

## eICU selection from omitted reporting bounds

Use the already extracted eICU rows, existing numeric labs and unchanged
HF cohort. For each 12-/24-hour window require original full hospital
follow-up. Restrict assay/unit pairs to those present in the existing numeric
troponin cache. Preserve exact source times and original assay matching.
Bounded rows from unmatched units remain counted separately.

Independently reconstruct the existing last positive numeric baseline in
[0,240] and its available same-assay numeric follow-up in (240,960] or
(240,1680]. Verify the original paired patient set exactly. Then count:

1. Existing paired encounters with a discarded reporting bound later than
   their selected numeric baseline but still no later than minute 240, in
   that paired assay/unit. This signals a potentially different baseline,
   not proof the original baseline is wrong.
2. Existing paired encounters with any omitted same-assay reporting bound
   during follow-up, including non-peak times.
3. Encounters without an existing numeric pair that have a potential timed
   pair when positive-magnitude bounds are admitted as concentration
   information. A positive upper bound does not prove a positive underlying
   baseline; this is an optimistic timing ceiling, not a count of valid rise
   endpoints. Keep assay/unit matching and the same windows.

Report denominators, encounters and distinct people by exposure, separately
for each horizon. Do not assign a bounded value its reporting limit, impute a
rise, estimate an exposure-outcome association or inspect mortality. Verify
SQL and independent Python patient sets and source classification. Keep
reference hashes, original source fingerprints and exact execution labels.
The workflow belongs in the single primary notebook. This audit cannot
establish a biological discovery or complete the mortality-relevant goal.
