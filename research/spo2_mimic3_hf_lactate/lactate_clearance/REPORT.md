# Failure of lactate clearance

Local analysis of a new endpoint requested after the previous lactate-rise results. Original endpoint reports remain preserved.

Primary follow-up policy: **complete_window**. Both follow-up populations are reported; results are not used to select the definition.

Baseline is the last validated lactate in [0,4] ICU hours; follow-up is the last in (4,16]. Their actual separation must be at least2 hours. Failure is clearance<10% AND not both values≤2 mmol/L. Exact decimal comparisons preserve the10% boundary.

The unchanged exposure permits transition gaps≤30 minutes. Complete-window results require observation through hour16 for both failures and successes. All-pair results can include earlier ICU departures/deaths and describe last observed paired values rather than an assured complete16-hour trajectory.

## Counts and unadjusted results

| Population | N | Failures | Instability failures/N | Stable failures/N | Raw RR | RD (percentage points) |
|---|---:|---:|---:|---:|---:|---:|
| all_qualifying_pairs | 243 | 66 | 24/99 | 42/144 | 0.831169 | -4.924242 |
| all_qualifying_pairs_full_model_sample | 227 | 64 | 23/92 | 41/135 | 0.823171 | -5.370370 |
| complete_followup_through_hour16 | 239 | 63 | 23/97 | 40/142 | 0.841753 | -4.457674 |
| complete_followup_through_hour16_full_model_sample | 223 | 61 | 22/90 | 39/133 | 0.833618 | -4.878864 |

## Modified Poisson models

| Population | Model | N | Failures | Adjusted RR | Patient-cluster robust95% CI | p |
|---|---|---:|---:|---:|---|---:|
| all_qualifying_pairs | support_and_interval_without_VIS | 239 | 65 | 0.815622 | 0.489640–1.358630 | 0.433735 |
| all_qualifying_pairs | same_VIS_available_sample_without_VIS | 227 | 64 | 0.801664 | 0.476204–1.349560 | 0.405467 |
| all_qualifying_pairs | full_requested_model | 227 | 64 | 0.798787 | 0.474734–1.344039 | 0.397417 |
| complete_followup_through_hour16 | support_and_interval_without_VIS | 235 | 62 | 0.810604 | 0.481607–1.364349 | 0.429263 |
| complete_followup_through_hour16 | same_VIS_available_sample_without_VIS | 223 | 61 | 0.794480 | 0.467064–1.351418 | 0.395960 |
| complete_followup_through_hour16 | full_requested_model | 223 | 61 | 0.788782 | 0.463749–1.341625 | 0.381280 |

The full model includes age, sex, baseline lactate, mean0–4-hour SpO₂, vasopressor infusion use, mechanical ventilation, CKD, ESRD, chronic pulmonary disease, chronic liver disease, diabetes, maximum0–4-hour VIS, and the actual lactate interval in hours. Existing coding and support definitions are unchanged. SUBJECT_ID-clustered robust errors, finite-sample correction and independent likelihood/covariance calculations follow the existing model code. All requested columns must be retained and the design must have full rank.

Covariate exclusions are detailed in covariate_exclusions.csv; raw denominators include all qualifying pairs for each policy. Endpoint reclassification against the original≥0.5 mmol/L rise is in endpoint_reconciliation.csv; normal/clearance components are in clearance_classification.csv. Patient-level records remain private.

The interval is measured after the landmark and may reflect evolving illness and clinician sampling. Adjusting for it does not eliminate informative measurement or establish causation/predictive improvement. This endpoint adapts a sepsis definition to HF and a different sampling window; it requires validation. See CLEARANCE_ENDPOINT_ADDENDUM.md for the locked requested definition and source qualifications.

## Cohort and endpoint reconciliation

| Qualification | Stays |
|---|---:|
| waveform_dynamics_eligible | 1597 |
| baseline_in_0_4h | 426 |
| also_followup_in_4_16h | 244 |
| also_baseline_to_followup_at_least2h | 243 |
| also_complete_followup_through_hour16 | 239 |

The previous baseline could precede ICU admission; this new baseline must fall in ICU hours0–4. The paired denominator is therefore different from the earlier388-stay primary endpoint cohort. Original baseline selection, original outcomes and mortality results are retained in their existing artifacts.

| Population | Old≥0.5 rise endpoint | New clearance failure | Stays | Instability |
|---|---|---:|---:|---:|
| all_qualifying_pairs | 0 | 0 | 174 | 73 |
| all_qualifying_pairs | 0 | 1 | 28 | 4 |
| all_qualifying_pairs | 1 | 0 | 2 | 1 |
| all_qualifying_pairs | 1 | 1 | 35 | 19 |
| all_qualifying_pairs | unavailable | 0 | 1 | 1 |
| all_qualifying_pairs | unavailable | 1 | 3 | 1 |
| complete_followup_through_hour16 | 0 | 0 | 174 | 73 |
| complete_followup_through_hour16 | 0 | 1 | 28 | 4 |
| complete_followup_through_hour16 | 1 | 0 | 2 | 1 |
| complete_followup_through_hour16 | 1 | 1 | 35 | 19 |

### Sample reconciliation with the prior endpoint

Old events refer to the≥0.5 mmol/L rise; new events refer to failure of clearance. Removed-old-event columns describe population/complete-case exclusions, separately from endpoint reclassification among retained stays.

| Sample | Old N/rise events | New complete-window N | Old rise events retained | New failures | Removed old rise events | Removed old rise non-events |
|---|---:|---:|---:|---:|---:|---:|
| paired_cohort | 388/50 | 239 | 37 | 63 | 13 | 136 |
| support_model | 380/49 | 235 | 36 | 62 | 13 | 132 |
| VIS_model | 361/47 | 223 | 34 | 61 | 13 | 125 |

## Missing covariates

| Population | Missing covariates | Removed stays | Removed failures | Removed instability |
|---|---|---:|---:|---:|
| all_qualifying_pairs | VIS_max_0_4h | 12 | 1 | 6 |
| all_qualifying_pairs | mechanical_ventilation_0_4h, VIS_max_0_4h | 4 | 1 | 1 |
| complete_followup_through_hour16 | VIS_max_0_4h | 12 | 1 | 6 |
| complete_followup_through_hour16 | mechanical_ventilation_0_4h, VIS_max_0_4h | 4 | 1 | 1 |

The full requested primary model does not establish an association: RR 0.788782, 95% CI 0.463749–1.341625. Its confidence interval and all sensitivity results are reported without changing the specification. It has 61 events and 15 parameters (4.07 events per parameter); estimates should be interpreted with that limited support.

The≥10%/both-normal definition is documented in the [Jones et al. multicenter sepsis trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC2918907/). That is methodological precedent rather than validation of this HF sampling window.
