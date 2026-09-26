# Pre-ICU laboratory encounter-linkage audit

This is an outcome-blind source audit needed to determine whether the current
SpO2 biological experiments are losing relevant pre-ICU observations. It is
not a biological finding and does not fulfill the discovery goal.

The linked-record hemoglobin specification completed with 659 usable eICU and
133 MIMIC observations. MIMIC had 1,441 candidate CBC records; none was excluded
for units and only three for the frozen value range. Thus unit handling does
not explain its low coverage. The earlier troponin-ordering specification also
had sparse MIMIC pre-episode measurements. Neither specification is changed
by this audit, and their failed support results remain visible.

The [official MIMIC laboratory documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)
states that some temporally associated hospital laboratories lack hadm_id and
that time-based joins may be needed. The following fixed audit evaluates that
possibility; it does not assume all unlinked tests belong to an admission.

## Extraction

Use the supplied MIMIC files only. Restrict patient windows to the existing
SpO2-dynamics-eligible HF cohort, with laboratory charttime in the 24 hours
before ICU admission through ICU minute zero. Extract identifiers, specimen
identifier, item, timestamps, units, and whether a positive numeric result
exists. Do not select laboratory concentrations or any mortality outcomes.

Include CBC hemoglobin item 51222 and the existing production troponin item
mapping. Items 50811 and 51640 are included only to inventory alternative
hemoglobin sources; they are not accepted into the frozen CBC experiment.
No oxygen, troponin or hemoglobin threshold will be optimized in this audit.

## Fixed temporal rule and validation

For each candidate laboratory, use the same subject_id and the complete
admissions table, including admissions outside the HF cohort. A temporal
candidate requires charttime between the earlier of edregtime and admittime,
and dischtime, with both boundaries valid. No additional time tolerance,
nearest-admission assignment or outpatient proximity rule is allowed.

Count zero, one and multiple matching hospital encounters. A missing hadm_id
can only be a candidate for later use when exactly one encounter matches.
Never overwrite an explicit nonmissing hadm_id, including when it conflicts
with a temporal candidate. Preserve candidate assignments separately from
all prior patient tables and the raw sources.

Validate the fixed rule by masking known hadm_id values: compute the proportion
of uniquely temporally assigned records for which the candidate exactly equals
the recorded identifier, separately for CBC, troponin and alternative Hb.
Report coverage, errors, ambiguous assignments and confidence intervals rather
than accuracy alone. To avoid treating repeated tests from one patient as
independent validation, define patient success as all of that patient's
uniquely assigned known records agreeing with their recorded IDs. A candidate
procedure requires at least 200 distinct patients for each analyte family it
would support and a one-sided 95% exact lower confidence bound above 99% for
this patient-level agreement. Also report the record-level agreement descriptively.
This is a necessary validation condition, not proof that missing links have
the same error distribution as known links. Do not tune the rule to pass it.

Report the number of eligible encounters with additional uniquely assignable
missing-ID observations, distinguishing records before formal inpatient
admission from those after it. Do not recompute troponin trajectories, Hb
interaction effects, mortality associations or biological support gates.

## Decision boundary

Any use of newly assigned records requires a separately documented amendment
that acknowledges the earlier feasibility results and the original explicit-ID
restriction. The original plans and results must remain unchanged. Passing
this audit would justify considering an amendment, not silently reclassifying
the original experiments as successful or preregistered under the new rule.

Patient-level records remain in a separate private database in Drive Data.
Aggregate audit results and provenance are saved with the primary notebook.
Execution is local unless actual Colab execution is documented.
