# Assay-encoding clarification before the paired contrast was estimated

The initial local run stopped on a SQL alias parsing error in baseline-covariate
construction. It produced no model-results file. Its log and launch provenance
are preserved. No new paired differential-association estimate was inspected.

Review identified a separate categorical-encoding issue: coding unavailable
troponin subtype as `not troponin T` would incorrectly assign missing results
to the reference category. In the MIMIC cohort, where the inspected inventory
contains only troponin T, this would also make the subtype column the exact
complement of the baseline-troponin missingness indicator.

Clarify the previously unspecified categorical imputation: impute missing
troponin subtype by the mode of the observed pre-landmark subtype indicator,
retaining the baseline-troponin missingness indicator already in the design.
If the imputed subtype is constant, omit it under the original constant-column
rule. Do not drop a nonconstant covariate based on model results. Record the
imputation in the design manifest. Endpoints, risk sets, horizons, continuous
imputation, effect contrast, multiplicity and advancement rules are unchanged.
This is a pre-estimate implementation clarification, not a rescue of a
favorable or unfavorable association.
