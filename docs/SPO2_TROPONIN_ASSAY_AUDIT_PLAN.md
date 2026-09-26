# Raw troponin reporting audit

Specified September 5, 2026 before extracting the original assay text for
this audit. All earlier numeric troponin results are known. Scope is the
same MIMIC/eICU HF dynamics-eligible cohorts, four-hour landmark, and
12-/24-hour follow-up used in the sampling-opportunity experiment.

The existing endpoint uses numeric concentrations. Original reporting text
such as `<0.01` describes a limit, not an exact concentration. This audit
asks which existing labels rely on such records or on unparsed source text.
It does not recode results or declare a new myocardial injury endpoint.

## Extraction

Use `spo2_cardiorenal.duckdb` cohort/labs and `spo2_context.duckdb` cohort,
assay mappings and endpoint labels as read-only references. Rescan only
eICU `lab.csv` and MIMIC `labevents.csv` under Drive Data. Preserve all
troponin records for the same hospital/ICU linkage and ICU minutes
[0,min(1680,followup_end)] used in the original extraction, including rows
with missing numeric values. Do not infer missing hospital links. Retain
original text, numeric strings, units, result times and lab identifiers.
MIMIC also supplies specimen IDs and reference-range fields; retain those
for provenance and report numeric upper-range availability only. eICU has
no reference-range field. A reported range is not a validated 99th percentile.

Reconstruct the original normalization exactly: native numeric values in
[0,1000000], original unit precedence and trimming, then median numeric value
at each stay/time/assay/unit. Require exact key/row-set equality with the
cached troponin labs, numerical equality within 1e-12, and matching maximal
result times. Audit MIMIC subject-ID agreement with the linked cohort.
Stop on any source reconstruction mismatch; do not fit associations.

## Text classification and impact

Trim whitespace; normalize Unicode <=/>= signs and case. Recognize a full
numeric literal (optional sign, decimal, scientific notation), or one such
literal preceded by `<`, `<=`, `>`, `>=`, or `=`. Also normalize an initial
`less than` or `greater than` phrase. Do not silently strip trailing units
or interpret other narrative text. Keep categories: exact numeric,
lower/upper-bound reporting by operator, missing text, and unparsed text.
Retain a separate flag if the parsed magnitude differs from the stored
numeric magnitude (comparison at 1e-12 tolerance). Missing or unparsed text
is an uncertainty flag, not proof of a censored or erroneous result.

For records entering the current numeric pipeline, aggregate source flags
over each exact stay/time/assay/unit group. A group is fully text-supported
only when every contributing record is exact numeric and agrees with the
stored numeric value. Report source-record counts by category and assay.
Do not print patient IDs, raw comments or free-text examples into the repo.

For the original last-positive-baseline and all assay-matched follow-up
groups, report encounter/person counts with a baseline flag, follow-up flag,
or either, by database, horizon, exposure and unchanged legacy rise label.
Require hospital follow-up through the window end, matching the opportunity
experiment. All same-time contributors matter because the original summary
uses their median. Count any qualifying or nonqualifying follow-up group,
not just the group attaining the peak. Separately count explicit inequalities;
do not combine them with missing/unparsed text as proven censoring.

Independently reconstruct the source medians and used-sample flags in
SQL/Python. Reconstruct legacy maximum ratios and binary labels exactly.
No oxygen response, mortality, regression or automatic label replacement.
A clean audit validates this reporting layer only. An affected label needs
interval-aware adjudication, not automatic deletion or a claim it is false.

## Execution

Keep patient records in a new private DuckDB file under Drive Data. Record
source size/mtime before and after each scan, SHA-256 of the cached reference
databases, protocol hash and source code identity. Use explicit stage/status
files and preserve completed raw extraction for a validation-only rerun.
The actual workflow belongs in the single primary notebook; label local
execution and incomplete stages honestly. The discovery goal remains unmet.
