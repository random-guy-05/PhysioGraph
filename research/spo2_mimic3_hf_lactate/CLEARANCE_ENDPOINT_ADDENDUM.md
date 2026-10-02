# User-requested failure-of-lactate-clearance endpoint

Specified on 2026-10-01 after results for the previously locked lactate-rise and mortality endpoints were available. This is a new post-result endpoint specification, not a retrospectively prespecified endpoint. The existing reports and models are preserved.

## Fixed specification before this endpoint's model results

- Use the saved adult HF, waveform-dynamics eligible cohort and exposure. The user explicitly confirmed retaining transition gaps **≤30 minutes**.
- Baseline: last valid lactate during ICU minutes **[0,240]**, excluding pre-ICU measurements.
- Follow-up: last valid lactate during **(240,960]**. A draw at minute240 belongs only to baseline. Select the last draw first, then require at least120 minutes between it and baseline; do not substitute an earlier baseline or follow-up to meet the interval rule.
- Clearance percentage:100×(baseline−follow-up)/baseline. Success means clearance≥10% **or both values≤2.0 mmol/L**. Failure is the complement, so a patient can have an absolute rise but remain a success when both values are normal.
- Classification at the exact10% boundary uses decimal arithmetic, equivalent to follow-up≤0.9×baseline, to avoid floating-point misclassification. Invalid or missing lactates do not become negatives; the original validated lactate-draw rules remain fixed.
- The user explicitly confirmed **complete follow-up through hour16** for the new primary endpoint. The existing earliest ICU departure, hospital discharge, death, or observation end must reach minute960 for successes and failures alike. Report all qualifying pairs without this requirement only as a separate sensitivity analysis.
- Model: modified Poisson, patient-clustered robust standard errors, same finite-sample correction. Covariates: age, sex, baseline lactate, mean0–4-hour SpO₂, vasopressor infusion use and mechanical ventilation in hours0–4, prior-coded expanded CKD, the existing lactate-model ESRD and chronic pulmonary disease flags, chronic liver disease, diabetes, maximum concurrent six-drug VIS in hours0–4, and baseline-to-follow-up lactate interval in hours. VIS and elapsed interval are continuous linear terms.
- Report raw counts/RR/RD, complete-case exclusions, original-versus-new endpoint reconciliation, and a no-VIS model on the same VIS-available sample. No alternative thresholds, windows, covariate selection, or favorable-result selection.
- Reuse saved processed lactate draws, endpoints, support covariates and VIS scores. No waveform or large clinical-table processing is needed. Patient-level output remains private; aggregate results and runnable notebook support are public.

## Source context

The combined ≥10%/both-normal rule is explicitly documented in the multicenter [Jones et al. randomized sepsis trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC2918907/). [Arnold et al.](https://pubmed.ncbi.nlm.nih.gov/19533847/) studied early clearance in severe sepsis; the supplied PubMed page could not be fetched completely, so the combined rule is attributed to the verified Jones trial rather than assumed from the Arnold citation. The [review](https://pubmed.ncbi.nlm.nih.gov/23879729/) supports a10% target at a minimum of2 hours in severe sepsis. These support precedent for the definition; they do not validate a last-draw4–16-hour HF endpoint. [DOREMI](https://pmc.ncbi.nlm.nih.gov/articles/PMC9075306/) excluded baseline lactate<2.0 mmol/L and investigated different clearance definitions and timings.

Elapsed-time adjustment accounts for measured interval differences within the fitted model. It does not eliminate informative repeat sampling, selection into the paired cohort, variable sampling times, or confounding from evolving treatment. The binary outcome measures observed lactate concentration change, not directly measured physiological clearance.
