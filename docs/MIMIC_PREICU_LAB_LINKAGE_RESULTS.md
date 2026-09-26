# Pre-ICU laboratory linkage audit results

The local audit completed on 2026-09-05 at 06:35 UTC. It identified a substantial
loss of pre-ICU observations under the explicit-admission-ID restriction.
This is a source-linkage result, not a biological discovery.

| Analyte | Audited records | Recorded ID present | Recorded ID missing | Uniquely matched known records / correct | Validation patients, all correct | One-sided 95% lower bound for patient agreement |
|---|---:|---:|---:|---:|---:|---:|
| CBC hemoglobin | 4,385 | 1,479 | 2,906 | 1,445 / 1,445 | 1,091 | 99.73% |
| Troponin | 2,834 | 741 | 2,093 | 735 / 735 | 555 | 99.46% |

The fixed rule required the same subject and a laboratory timestamp within
exactly one documented ED/inpatient-to-discharge encounter, considering all
546,028 admissions. Known records without a unique temporal match were not
counted as successful validations or reassigned. Across known and missing-ID
records, 250 CBC and 86 troponin records had no temporal match; one CBC and
four troponin records had multiple matches. These cannot be restored by the
fixed unique-assignment rule.

Among missing-ID records uniquely assigned to the canonical eligible
encounters, the audit identified 2,681 additional CBC records in 2,506 encounters
and 2,009 troponin records in 1,821 encounters. Of these, 2,532 CBC and 1,891
troponin records preceded formal inpatient admission. Blood-gas Hb was also
inventoried but is not accepted into the frozen CBC experiment.

Independent Pandas interval matching exactly reproduced the SQL candidate
counts, unique assignments and ambiguities across all 9,175 audited records.
The audit selected identifiers, timing, units and numeric-availability flags;
it did not select concentrations or mortality outcomes.

The CBC and troponin families passed the prespecified patient-level validation
criterion. Agreement on known IDs does not establish ground truth for missing
IDs, whose missingness mechanism may differ. Restored records must retain
their inferred-link provenance and explicit-ID-only sensitivity analyses.

The [documented amendment](PREICU_LAB_LINKAGE_AMENDMENT.md) permits only these
saved CBC and troponin lab IDs and reruns the original feasibility algorithms.
It changes no biological window, assay choice, hemoglobin range or support
threshold. Original failed results remain available. Passing a linkage audit
or an amended feasibility screen is not evidence of myocardial susceptibility,
biological ordering, or mortality benefit.
