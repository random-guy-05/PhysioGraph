# MIMIC diagnostic ECG metadata overlap

Actual execution: local incremental notebook cells. No ECG values, signal amplitudes, injury labels or mortality outcomes were examined.

| Criterion | Encounters | People | Eligible encounters |
|---|---:|---:|---:|
| any_subject_ecg | 3880 | 3512 | 4711 |
| nominal_hospital_ecg | 2921 | 2713 | 4711 |
| pre_icu_ecg | 623 | 609 | 4711 |
| first4h_ecg | 1251 | 1216 | 4711 |
| later_ecg | 1595 | 1526 | 4711 |
| nominal_pre_later_pair | 341 | 338 | 4711 |

Subject-matched registry records: 60708. Invalid matched paths: 0; invalid matched times: 0.

Ambiguous nominal hospital assignments: 0 records; ambiguous first-four-hour ICU assignments: 0 records.

These are nominal matches to recorded timestamps. The [source documentation](https://physionet.org/content/mimic-iv-ecg/1.0/) explicitly warns that ECG clocks may be unsynchronized. These counts do not prove ECG acquisition preceded or followed an SpO2 episode, and do not validate an acute injury sequence.

The local dictionary contains 1 ECG/EKG-related entries; their labels and source tables are saved separately for clock-reference assessment. No item has yet been accepted as an independent acquisition timestamp.

Publisher hashes, local source fingerprints, SQL/Python hospital links, all encounter-window flags and nominal offsets validate. The original troponin failures remain unchanged. This creates a possible measurement route for a separately specified electrophysiology study, not a new biological finding or an achieved goal.
