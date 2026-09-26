# Masked pulsatility source forensic audit

Classification: `invalid_for_intended_routine_bp_hypothesis_missing_sources`

Protocol: `5e6d7e099ffda52ce763fbf926a833dceb2f855b70d2957bf5f079dead307146`

The frozen analysis was not changed. No clinical outcomes or post-anchor lactate values were inspected.

## Source attrition

| dataset   | source                          | source_available   | unavailable_reason                               |   any_sbp_or_dbp_stays |   paired_sbp_dbp_stays |   ge3_hourly_pair_bins |   initially_low |   incident_pair_crossings |
|:----------|:--------------------------------|:-------------------|:-------------------------------------------------|-----------------------:|-----------------------:|-----------------------:|----------------:|--------------------------:|
| eicu      | invasive_arterial               | True               | nan                                              |                   1116 |                   1106 |                    873 |              48 |                        25 |
| eicu      | noninvasive_cuff_vitalAperiodic | False              | vitalAperiodic.csv absent from local eICU export |                    nan |                    nan |                    nan |             nan |                       nan |
| eicu      | noninvasive_nurseCharting       | False              | nurseCharting.csv absent from local eICU export  |                    nan |                    nan |                    nan |             nan |                       nan |
| mimic     | invasive_arterial               | True               | nan                                              |                   3062 |                   3058 |                   1898 |              91 |                        80 |
| mimic     | noninvasive_cuff                | True               | nan                                              |                  15187 |                  15180 |                  12869 |            1150 |                      1645 |
| eicu      | source_consistent_union         | True               | nan                                              |                   1116 |                   1106 |                    873 |              48 |                        25 |
| mimic     | source_consistent_union         | True               | nan                                              |                  16917 |                  16913 |                  14446 |            1138 |                      1674 |

## Frozen-analysis attrition

| dataset   |   hf_cohort |   valid_source_selected_triplet_stays |   ge3_hourly_triplet_bins |   incident_crossings_before_compensation |   compensated_crossings |   crossings_no_preanchor_lactate |   crossings_lactate_lt2 |   crossings_lactate_ge2 |   crossings_without_prior_pressor_mcs |   crossings_lactate_lt2_without_prior_support |   subsequent_pressor_6h |   subsequent_pressor_12h |
|:----------|------------:|--------------------------------------:|--------------------------:|-----------------------------------------:|------------------------:|---------------------------------:|------------------------:|------------------------:|--------------------------------------:|----------------------------------------------:|------------------------:|-------------------------:|
| eicu      |       12028 |                                  1103 |                       872 |                                       24 |                      16 |                               13 |                       2 |                       1 |                                    14 |                                             2 |                       0 |                        2 |
| mimic     |       17758 |                                 16902 |                     14258 |                                     1644 |                    1382 |                              943 |                     211 |                     228 |                                  1132 |                                           160 |                      43 |                       73 |

## Triplet concentration

| dataset   | source            |   complete_triplets |   instrumented_stays |   median_triplets_per_instrumented_stay |   maximum_triplets_per_stay |
|:----------|:------------------|--------------------:|---------------------:|----------------------------------------:|----------------------------:|
| eicu      | invasive_arterial |               39625 |                 1116 |                                      42 |                          48 |
| mimic     | invasive_arterial |               16850 |                 3031 |                                       5 |                          58 |
| mimic     | noninvasive_cuff  |               71895 |                15174 |                                       4 |                          55 |

## Lactate availability

| dataset   |   compensated_stays |   earliest_candidate_no_lactate |   earliest_candidate_lactate_lt2 |   earliest_candidate_lactate_ge2 |   any_candidate_anchor_no_lactate |   any_candidate_anchor_lactate_lt2 |   any_candidate_anchor_lactate_ge2 | any_anchor_counts_are_nonexclusive   |
|:----------|--------------------:|--------------------------------:|---------------------------------:|---------------------------------:|----------------------------------:|-----------------------------------:|-----------------------------------:|:-------------------------------------|
| eicu      |                 966 |                             765 |                               68 |                              133 |                               812 |                                136 |                                175 | True                                 |
| mimic     |               14500 |                           11313 |                             1504 |                             1683 |                             11683 |                               3153 |                               2640 | True                                 |

Earliest-candidate categories are mutually exclusive. Any-candidate-anchor counts reproduce the prior coverage metric but are nonexclusive because a stay can contribute several candidate anchors.

## Mapping audit

| dataset   | source                      | file_present   | configured   | sbp_mapped   | dbp_mapped   | included_in_prior_notebook   |
|:----------|:----------------------------|:---------------|:-------------|:-------------|:-------------|:-----------------------------|
| eicu      | vitalPeriodic.systemic*     | True           | True         | True         | True         | True                         |
| eicu      | vitalAperiodic.noninvasive* | False          | True         | True         | False        | False                        |
| eicu      | nurseCharting NIBP          | False          | False        | False        | False        | False                        |
| mimic     | chartevents invasive        | True           | True         | True         | True         | True                         |
| mimic     | chartevents cuff            | True           | True         | True         | True         | True                         |

The 39,625 eICU complete triplets were repeated invasive measurements concentrated in the arterial-line subset; they did not represent routine cuff coverage.
