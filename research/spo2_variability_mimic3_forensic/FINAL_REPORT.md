# Forensic audit of the MIMIC-III SpO2-variability replication

## Primary classification

**IMPLEMENTATION_OR_MAPPING_CONCERN**

The frozen numerical record is internally reproducible: an independent reconstruction recovered the exact MIMIC-III counts (5,405 preliminary; 2,368 at risk; 1,799 exposed; 103 at-risk primary events; 72 modeled primary events; 53 modeled later events) and the exact MIMIC-IV support counts (17,758 preliminary; 8,196 at risk; 5,957 exposed; 238 at-risk primary events; 158 modeled events). Lock, source, and code hashes pass, no source item-mapping error was found, and no frozen result was changed.

The dominant forensic finding is a material implementation-parity concern in the endpoint. Frozen MIMIC-IV calls the six-hour oliguria helper with its default `start_hour=4`, whereas frozen MIMIC-III explicitly calls it with `start_hour=0`; therefore MIMIC-III admits urine windows that overlap more of the exposure/landmark period. The later endpoints are also not identical: MIMIC-IV labels a primary composite whose event time is after minute 480, while MIMIC-III re-pairs the domains with both components required strictly after minute 480. These findings are documented only; the locked MIMIC-III outcomes and two fitted models were not corrected or rerun.

Independent of that defect, endpoint observability is materially non-equivalent. MIMIC-IV uses availability-aware chart/store timestamps, interval starts for infusions, and procedure starts for MCS. MIMIC-III uses CareVue charttime alone, repeated positive-rate infusion rows, and device-chart documentation. Both at-risk cohorts require observation only beyond minute 360 for an endpoint extending through minute 960, so incomplete late follow-up is a shared limitation rather than a cross-era difference. Preliminary selection still differs: MIMIC-IV requires ICU entry within 24 hours unless a shock code is present, whereas MIMIC-III has no ICU-entry-time restriction.

The exposure computation is mathematically identical after extraction and uses the exact MIMIC-IV clipping/scale. However, extraction is not byte-equivalent because MIMIC-IV requires storetime <=240 while MIMIC-III uses charttime only. In MIMIC-III, 2,726/30,193 early SpO2 rows (9.03%) were stored after minute 240; this is documented but not used to alter or refit the frozen exposure.

MIMIC-III was independently reconstructed from its original source files. After Google Drive timed out on a raw MIMIC-IV admissions read, the MIMIC-IV reference side used the already validated read-only frozen discovery and SpO2-context databases. Those tables replayed the exact 8,196-stay at-risk set and all 238 primary endpoint IDs; no MIMIC-IV estimate was refitted.

## Cohort transportability

The harmonized modeled cohorts do not look radically different by age or sex: mean age is 72.48 years in MIMIC-IV versus 72.56 in MIMIC-III (SMD 0.005), and male prevalence is 53.6% versus 54.4% (SMD 0.016). Differences are more visible for baseline SpO2 (96.38 versus 97.12; SMD 0.245) and MAP (83.57 versus 80.80; SMD -0.196). A coarse cross-era ICU grouping shows more medical-ICU representation in MIMIC-III (47.1% versus 27.2%), while MIMIC-IV includes a 15.5% mixed medical/surgical group and broader neuro/intermediate units. Ventilation, FiO2, MIMIC-III respiratory-support status, and cross-era HF coding density/comorbidity burden were not harmonized in the frozen frames. Thus the available data do not show a substantially different CareVue HF population on age/sex, but ICU mix and unavailable context prevent declaring the cohorts fully comparable.

## Precision and cross-era compatibility

MIMIC-III log RR was -0.0311 with HC0 SE 0.1232; its RR-scale CI width was 0.473 and its multiplicative 95% factor was ×/1.273. Approximate power was 31.6% for a true RR of 1.20 and 44.1% for 1.25. The MIMIC-IV RR 1.251 lies just outside the MIMIC-III 95% CI upper bound 1.234. The cross-era difference test gives z=1.839, p=0.0660, Q=3.380, and I²=70.4%; with two studies, I² is unstable. The effects are not formally heterogeneous at alpha .05, but the divergence is suggestive and the MIMIC-III null is not precise enough to call a true negative.

## Measurement and sanity checks

The source audit confirms CareVue item 646 is labeled SpO2 with percent units and does not mix arterial SaO2, venous saturation, or FiO2. Every mapped itemid is listed with counts, units, missingness, distributions, common values, and implausibility counts. Frozen endpoint pairs all satisfy the >360, <=960, and <=360-minute separation rules. Descriptive checks of SBP, MAP, lactate, and absolute SpO2 are recorded only as pipeline sanity checks, not discoveries.

No degradation experiment was triggered: the dominant limitations are endpoint and cohort observability, not a sufficiently dominant outcome-blind measurement-process shift that would justify another one-off MIMIC-IV model. No association model was fitted in this forensic audit.

## Evidentiary conclusion

The MIMIC-III result is numerically reproducible, but it is not an exact endpoint replication. It still argues against casually assuming the MIMIC-IV association is era-invariant; however, its evidentiary weight as biological non-confirmation is limited first by the material endpoint-parity defect, and additionally by weaker CareVue observability and a different ICU-entry phenotype. The correct conclusion remains **historical non-confirmation with an implementation concern**, not rescue, not proof of no effect, and not successful external validation.
