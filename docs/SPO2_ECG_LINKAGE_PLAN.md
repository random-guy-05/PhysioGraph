# MIMIC diagnostic-ECG linkage and clock observability

Frozen before downloading the ECG index or inspecting cohort overlap. This is a prerequisite for an independent cardiac-physiology experiment, not a biological discovery or a replacement for the failed troponin models. Only original PhysioGraph MIMIC HF encounters are eligible. No non-MIMIC/eICU patient dataset is added.

The [MIMIC-IV-ECG v1.0 documentation](https://physionet.org/content/mimic-iv-ecg/1.0/) describes an open diagnostic-ECG module with linked subject identifiers. It also warns that the recording machines' clocks were often unsynchronized. That warning prevents treating recorded ECG order relative to SpO2 as verified biological order. The public bedside-waveform release remains v0.1.0; the completed 11-encounter pilot is unchanged.

## Metadata retrieval and provenance

Download only `record_list.csv`, `SHA256SUMS.txt`, `machine_measurements_data_dictionary.csv` and `LICENSE.txt` from the official versioned ECG source. Store them under private Drive Data. Validate downloaded files against the publisher's checksum list and preserve response metadata and local hashes. Do not download machine measurements, reports, or signal amplitudes in this stage. The dictionary supplies schema only.

Fingerprint the existing `spo2_context.duckdb` and `spo2_cardiorenal.duckdb`. The eligible set is the original `mimic_cohort` in the latter, linked to subject, hospital-admission and ICU-admission identifiers/times in the former. Read only subject/admission identifiers and admission/discharge times from the local MIMIC admissions table; do not read outcomes.

## Linkage and nominal timing counts

Validate record IDs and paths, count missing/unparseable times, and require unique study IDs and paths. Join exact subject IDs to the unchanged eligible cohort. Report subject overlap separately from recorded-time overlap with the indexed hospital admission. Hospital membership inferred only from the ECG timestamp is nominal because the external clock has not been calibrated. Flag records nominally matching multiple hospital admissions or multiple ICU windows; never silently pick a preferred admission.

For nominally matched current hospital admissions, count ECGs in three fixed windows relative to ICU admission: [-1440,0), [0,240), [240,1680] minutes. Report cohort people/encounters with any ECG in each window, and with a pre-ICU ECG plus a later ECG in [240,1680]. Report encounter counts by original SpO2 exposure without analyzing ECG values or outcomes. Independently reproduce all encounter-level window flags and pairing flags in SQL and Python. These counts do not establish synchronized pre/post pairs or justify a window-specific injury claim.

Inspect only the local MIMIC item dictionary for ECG/EKG procedure or acquisition-documentation fields that could potentially provide an independent clock reference. A monitoring-enabled checkbox, lead selection, rhythm label or equipment state is not an independent diagnostic ECG acquisition time. Do not inspect patient values until a separately stated clock-validation approach is justified by metadata.

## Advancement rule

No ECG phenotype, injury direction, mortality model or new prognostic threshold is estimated in this stage. If overlap exists, the next protocol must specify a substantive electrophysiological question, signal-quality/calibration checks and how unsynchronized clocks constrain interpretation. A metadata match is not a novel biological finding. Retain old failures and correct any review language that incorrectly equates a troponin ratio with confirmed injury.

Publish aggregate linkage/clock-feasibility results and append the executed cells to the single primary notebook. Keep patient identifiers, record paths and timestamps private. Distinguish actual local execution from a Colab or uninterrupted notebook run.
