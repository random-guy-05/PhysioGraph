# Hour-16 minimum evidenced Kapur/CSWG SCAI distribution

Local descriptive calculation on the fixed 1,597-stay SpO2 cohort; separate rows also show the 380-stay lactate and 900-stay mortality samples. Stage at hour 16 uses latest valid physiology during hours 12–16 and time-specific treatment evidence. Measurements after hour 16 are excluded.

**This is an EHR-derived lower-bound classification. Stage A cannot be certified because OHCA and complete examination information are unavailable. Zero assigned A observations does not mean zero clinically stage-A patients.**

| Population | Category | N | Cohort N | % of cohort |
|---|---|---:|---:|---:|
| all_spo2_eligible | A | 0 | 1597 | 0.00% |
| all_spo2_eligible | B | 323 | 1597 | 20.23% |
| all_spo2_eligible | C | 190 | 1597 | 11.90% |
| all_spo2_eligible | D | 109 | 1597 | 6.83% |
| all_spo2_eligible | E | 52 | 1597 | 3.26% |
| all_spo2_eligible | unclassified | 886 | 1597 | 55.48% |
| all_spo2_eligible | died_by_hour16 | 23 | 1597 | 1.44% |
| all_spo2_eligible | discharged_by_hour16 | 2 | 1597 | 0.13% |
| all_spo2_eligible | left_icu_by_hour16 | 12 | 1597 | 0.75% |
| primary_lactate_model | A | 0 | 380 | 0.00% |
| primary_lactate_model | B | 72 | 380 | 18.95% |
| primary_lactate_model | C | 65 | 380 | 17.11% |
| primary_lactate_model | D | 69 | 380 | 18.16% |
| primary_lactate_model | E | 28 | 380 | 7.37% |
| primary_lactate_model | unclassified | 146 | 380 | 38.42% |
| primary_lactate_model | died_by_hour16 | 0 | 380 | 0.00% |
| primary_lactate_model | discharged_by_hour16 | 0 | 380 | 0.00% |
| primary_lactate_model | left_icu_by_hour16 | 0 | 380 | 0.00% |
| mortality_model | A | 0 | 900 | 0.00% |
| mortality_model | B | 182 | 900 | 20.22% |
| mortality_model | C | 109 | 900 | 12.11% |
| mortality_model | D | 82 | 900 | 9.11% |
| mortality_model | E | 39 | 900 | 4.33% |
| mortality_model | unclassified | 470 | 900 | 52.22% |
| mortality_model | died_by_hour16 | 15 | 900 | 1.67% |
| mortality_model | discharged_by_hour16 | 0 | 900 | 0.00% |
| mortality_model | left_icu_by_hour16 | 3 | 900 | 0.33% |

## Methods and limitations

- Reference: https://www.jacc.org/doi/10.1016/j.jacc.2022.04.049
- Highest evidenced stage uses the existing project Kapur/CSWG thresholds; treatment count includes vasopressors and inotropes plus distinct documented cardiac-device types. Stage E uses the published >3 drugs or >3 devices threshold.
- Stage D persistence requires abnormal early and current physiology with early treatment; this remains a retrospective proxy for failure to stabilize.
- Same-time numeric duplicates are collapsed by their median. Blood pH uses item 50820; urine and other-fluid pH are excluded. Lactate uses item 50813, and ALT 50861.
- SBP plausible range 20–300 mmHg, MAP 10–250, lactate 0–50 mmol/L, ALT 0–100,000 U/L, blood pH 6.5–8.0. These filters and lookbacks are explicit operational choices.
- Missing components never become normal. Patients dead, discharged, or outside the index ICU by hour 16 appear separately.
- Stage A is unavailable under the strict completeness rule. B–E categories indicate minimum evidenced severity; missing components could imply a higher stage.
- Non-cardiac causes of hypotension, lactate elevation, or vasoactive use are not adjudicated. This does not prove cardiogenic shock in every classified patient.
- Full stage, exposure-stratified distributions and 1-/6-hour physiology-lookback sensitivities are in stage_distribution.csv. Component coverage is in component_availability.csv.
- Patient-linked results remain in the private cache; repository tables contain aggregate results only.
