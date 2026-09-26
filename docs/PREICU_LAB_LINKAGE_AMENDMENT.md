# Amendment: validated temporal linkage for pre-ICU laboratories

This amendment follows, and does not erase, the failed explicit-ID-only
feasibility results: MIMIC had 23 complete injury-ordering trajectories and
133 Hb-plus-troponin observations. Biological trajectory directions and
hemoglobin-by-outcome associations have not been inspected.

## Evidence supporting the amendment

The fixed, outcome-blind audit used subject_id and documented encounter bounds
against all 546,028 supplied MIMIC admissions. It found 2,681 additional CBC
and 2,009 troponin records uniquely assignable to eligible encounters. Most
preceded formal inpatient admission. It did not select lab concentrations.

Among uniquely assigned records with known IDs, all assignments agreed for
1,091 CBC patients and 555 troponin patients. The one-sided 95% exact lower
bounds for patient-level complete agreement were 99.73% and 99.46%, exceeding
the frozen 99% requirement. An independent Pandas implementation reproduced
all candidate and ambiguity assignments for all 9,175 audited records.
See `research/mimic_preicu_lab_linkage/` for the counts and provenance.

The [MIMIC documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)
explicitly warns that some temporally associated labs lack hadm_id and
describes time-based joins as an option. Validation on known IDs does not
prove that missing-ID records have the same error distribution; the latter
remain inferred links and must be labeled as such.

## Sole analytical change

Permit MIMIC CBC and troponin records with a genuinely missing hadm_id when
the already-frozen audit assigns exactly one hospital encounter for the same
subject within its documented ED/inpatient-to-discharge bounds. The record
must be in the audited 24-hour pre-ICU window through ICU minute zero and
assigned to the same canonical eligible hospital encounter. Use the saved
candidate lab IDs; do not optimize the matching rule or include additional
records found by a wider search.

Never overwrite a nonmissing recorded ID or include an ambiguous assignment.
Keep source_hadm_id and assigned_hadm_id distinct in the restored-record table.
Record linkage provenance and retain the original explicit-ID analyses.
Do not modify the raw source files or the prior patient tables.

No additional hemoglobin assay is introduced: CBC item 51222 remains the sole
MIMIC Hb source. Blood-gas and chemistry Hb remain excluded. Hemoglobin units,
3–22 g/dL quality range, pre-ICU latest-value selection, SpO2 definitions,
troponin assay matching and positivity bounds, outcome horizons, serial-sample
separation, and both feasibility thresholds are unchanged. eICU is unchanged.
The Hb experiment continues to use the existing ICU-baseline troponin endpoint;
this amendment does not expand that endpoint's baseline window.

## Reruns and decision

1. Retrieve actual concentrations only for the saved, eligible missing-ID CBC
   and troponin records. Verify source fingerprints and exact lab-ID coverage.
2. Recompute the Hb availability screen using the original normalization and
   latest-value algorithm, adding only the restored CBC records.
3. Recompute the original injury-ordering availability screen, adding only the
   restored troponin records to the original episode anchors and assay rules.
4. Report original and amended counts together. These are two analyses of
   overlapping data, not independent replications. An amended pass permits a
   separately frozen biological analysis; it is not evidence for the hypothesis.

Do not open Hb-by-outcome associations or troponin trajectory directions in
these reruns. If a feasibility gate still fails, report that failure without
lowering its threshold or enlarging its biological windows. Later analyses
must preserve explicit-ID-only sensitivity estimates and linkage uncertainty.

The primary deliverable remains one notebook using MIMIC and eICU. All current
execution is local. A linkage improvement is not the requested biological
discovery and cannot establish a mortality benefit.
