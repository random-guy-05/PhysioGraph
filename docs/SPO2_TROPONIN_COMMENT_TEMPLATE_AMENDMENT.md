# MIMIC troponin comment-template amendment

This is a new reporting-parser amendment after the strict whole-comment audit,
not a change to its frozen rules or its result. That audit found no standalone
values in 4,758 comments. Subsequent inspection of repeated, short assay
templates found an interpretive threshold statement, sometimes preceded by
an actual reporting bound. Template frequencies are already known; no new
patient outcomes or exposure associations have been examined.

Recognize only these formats after case-folding and whitespace normalization:

- The standard cTropnT interpretive statement alone: no patient value or bound.
- A leading `<0.01.` followed by that exact statement: an upper reporting bound
  of 0.01 in the original source unit.
- A leading `>25*.` followed by that exact statement: a lower reporting bound
  of 25 in the original source unit; preserve the source's asterisk annotation.

The exact interpretive statement is specified in the executable parser. It
contains a comparison with 0.10 ng/mL and mentions acute MI. That comparison
must NEVER be treated as the patient's result, an adjudicated MI diagnosis,
or proof that the threshold is the assay's sex-specific 99th percentile.
Reject every other narrative format, including arbitrary prefixes, suffixes,
other limit values and partially matching sentences. Do not strip arbitrary
asterisks or mine numeric substrings.

Independently implement the whitelist in Python and SQL, compare classifications
for every source ID, and verify representative counterexamples. Persist only
source-level classifications and exact bound strings in the private database.
Report aggregate counts by existing numeric availability, unit, and template
category. Report any coexisting numeric/limit magnitude disagreement separately
from missing numeric values. Keep a semantic hash of the comments table before
and after; do not rescan raw files or alter the earlier numeric endpoints.

This recovers interval information only. Additional usable baseline/follow-up
pairs, interval-consistent rises and their relation to SpO2 require a separate
frozen analysis. No reporting bound is an exact concentration. This amendment
does not establish a biological finding or mortality benefit.
