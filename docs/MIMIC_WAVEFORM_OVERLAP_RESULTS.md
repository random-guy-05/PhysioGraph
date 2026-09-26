# MIMIC waveform-preview overlap

**Actual execution: local. Metadata-only feasibility; no waveform experiment or biological discovery.**

| Quantity | Count |
|---|---:|
| public_records | 200 |
| public_subjects | 198 |
| current_hf_encounters | 17758 |
| subject_matched_records | 62 |
| subject_matched_subjects | 60 |
| header_identity_or_time_failures | 0 |
| exact_hospital_matched_records | 48 |
| first4h_non_gap_overlap_records | 28 |
| first4h_non_gap_overlap_encounters | 28 |
| original_dynamics_eligible_overlap_encounters | 11 |
| dynamics_overlap_with_resp_pleth_ecg_layout | 11 |
| downloaded_metadata_files | 126 |
| downloaded_metadata_bytes | 1709972 |

The source is the [public MIMIC-IV waveform preview v0.1.0](https://physionet.org/content/mimic4wdb/0.1.0/), joined only to the existing MIMIC HF cohort. No additional non-MIMIC patient dataset was introduced.

Positive overlap means the metadata contains non-gap segments inside the existing ICU window. Layout channels do not prove simultaneous usable respiratory, ECG and plethysmographic signals; amplitudes, artifact, and physiological validity have not been assessed. No mortality, troponin outcome or SpO2 response was examined.

These counts determine whether a direct physiological pilot is possible; they do not establish adequate power, a new mechanism, or a mortality-improving intervention. Matched identifiers and downloaded headers remain in private Drive Data.
