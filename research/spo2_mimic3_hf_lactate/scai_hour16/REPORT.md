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

## Graphs

- [Overall distributions](figures/scai_hour16_distributions.png), with [vector PDF](figures/scai_hour16_distributions.pdf) and [SVG](figures/scai_hour16_distributions.svg).
- [Distributions by early SpO₂ instability](figures/scai_hour16_by_instability.png), with [vector PDF](figures/scai_hour16_by_instability.pdf) and [SVG](figures/scai_hour16_by_instability.svg).
- [Mortality and VIS by hour-16 stage](figures/scai_hour16_mortality_by_stage.png), with [vector PDF](figures/scai_hour16_mortality_by_stage.pdf) and [SVG](figures/scai_hour16_mortality_by_stage.svg). All figures have categories on the x-axis and counts, percentages, or VIS on the y-axis in a wide, side-by-side layout. The mortality graph retains the deaths/survivors panel and uses VIS distributions in the right panel.

The graphs use the saved aggregate results with the prespecified 240-minute physiology lookback. Counts and percentages include every stay in the respective sample/exposure group. Stage A is marked unavailable rather than plotted as zero. Discharged and other ICU-departure counts are combined for display; deaths remain separate. The exposure comparison is descriptive and unadjusted. Figure generation does not rerun clinical processing or models. The notebook includes the plotting step; `figures/figure_audit.json` records the source-table hash and display conventions.

### Mortality within hour-16 categories

The fixed 900-stay mortality model sample has 156 in-hospital deaths. The stage-specific graph includes 882 stays alive and in the ICU at hour 16, with 141 subsequent in-hospital deaths. Fifteen deaths by hour 16 and three earlier ICU departures (zero in-hospital deaths) are shown separately and reconcile the full sample. Stage A remains unavailable. These are descriptive, unadjusted rates conditional on reaching the hour-16 ICU landmark; no mortality model is refit.

| Hour-16 category | Stays | In-hospital deaths | Hospital survivors | Mortality |
|---|---:|---:|---:|---:|
| B | 182 | 25 | 157 | 13.7% |
| C | 109 | 29 | 80 | 26.6% |
| D | 82 | 33 | 49 | 40.2% |
| E | 39 | 13 | 26 | 33.3% |
| Unclassified | 470 | 41 | 429 | 8.7% |

`mortality_by_stage.csv` contains aggregate counts; `mortality_by_stage_audit.json` records reconciliation with the saved stage distribution and index-admission hospital death flag. Patient-level data remain outside the repository.

### VIS at hour 16

The right panel uses the standard six-drug score: dopamine + dobutamine + 100 × epinephrine + 100 × norepinephrine + 10 × milrinone + 10,000 × vasopressin. Doses are in µg/kg/min except vasopressin, which is in units/kg/min. Phenylephrine is excluded from this version of VIS. [Published six-drug VIS method](https://pmc.ncbi.nlm.nih.gov/articles/PMC9891263/).

- MetaVision: latest active infusion dose at minute 960; rewritten/cancelled records excluded. CareVue: latest numeric drug rate or explicit stop in minutes 900–960. Null volume-only rows do not erase a numeric rate recorded during that hour; an explicit stop at the latest timestamp overrides a positive rate.
- Compact CareVue units (`mcgkgmin`, `mcgmin`, `Umin`, `Uhr`) and slash-delimited MetaVision units are converted explicitly. The two documented CareVue vasopressin item exceptions (42273/42802) use the rate stored in the amount field. [MIMIC medication documentation conventions](https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iii/concepts/durations/vasopressor_durations.sql).
- Doses requiring weight use the infusion's documented MetaVision weight when 20–300 kg; otherwise the latest valid charted weight during hours 0–16. Weight items and lb/oz conversion follow the [MIMIC weight mapping](https://raw.githubusercontent.com/MIT-LCP/mimic-code/main/mimic-iii/concepts/durations/weight_durations.sql); no later weight is backfilled and no population weight is imputed.
- No active documented six-drug infusion with contemporaneous infusion-table coverage produces documented VIS zero; this is not adjudicated proof of clinical medication absence. No recent infusion documentation, a stale positive/unknown CareVue dose, or unresolvable dose/units/weight makes VIS unavailable.
- Boxes show the 25th–75th percentiles, the line shows the median, and whiskers show the 5th–95th percentiles. Exact minimum/maximum and available/zero/missing counts are preserved in `vis_by_stage.csv`; missingness reasons are in `vis_availability.csv`. Dose outliers are retained in the calculations.
- This is a local descriptive extraction and aggregation. Prior processing and models are unchanged. Vasoactive treatment is also a component of the SCAI proxy, so this distribution is partly related to the stage definition and does not establish independent predictive performance.

`vis_by_stage_audit.json` records source hashes, drug/item/unit mappings and score rules. Patient-level medication, weight and VIS caches remain private.

VIS was calculable for 695/882 stays (78.8%); 187 were unavailable. Missingness comprised 175 stays without contemporaneous infusion documentation, eight with stale CareVue dose information, and four with unresolvable dosing information or required weight. The 18 stays outside the hour-16 ICU risk set are excluded from both displayed panels.

| Category | VIS available / all stays | VIS zero | Median VIS | 25th–75th percentiles |
|---|---:|---:|---:|---:|
| B | 138/182 | 138 | 0.00 | 0.00–0.00 |
| C | 104/109 | 33 | 3.75 | 0.00–7.00 |
| D | 80/82 | 5 | 10.01 | 4.33–23.00 |
| E | 30/39 | 13 | 2.90 | 0.00–32.66 |
| Unclassified | 343/470 | 342 | 0.00 | 0.00–0.00 |

Values outside the 5th–95th percentile whiskers are not plotted. They remain in the summaries, including the stage-D maximum of 457.50 and stage-E maximum of 449.78. Zero quartiles for the unclassified group do not mean every score is zero (342/343 are zero; the remaining value is 4.00).
