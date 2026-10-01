# MIMIC-III HF SpO2 and lactate analysis

## Contents

- RUN_DENSE_WAVEFORM_ANALYSIS.ipynb is the runnable end-to-end workflow.
- scripts/run_dense_waveform_analysis.py is the notebook's standalone implementation.
- FINAL_HARDENED_REPORT.md records the completed MIMIC-III analysis, including the complete-follow-up correction for 12-hour and 24-hour last-value lactate endpoints.
- 03_landmark_lactate/ and 09_final_report/ contain aggregate result and reconciliation tables supporting the report.
- 01_cohort/, 02_signal_qc/, 04_episode_lactate/, 05_predictive_models/, 06_mimic4_parity/, and 07_sensitivity/ contain the remaining aggregate diagnostics and sensitivity results; 08_figures/ contains the saved plots from the run.
- ckd_corrected_full_adjusted_models/ contains the CKD-corrected full lactate and mortality model results, including the 0–4-hour mean-SpO2 sensitivity, its audit, and reconciliation.
- support_adjusted_models/ contains the primary lactate and mortality rerun adding 0–4-hour mean SpO2, vasopressor infusion use, and documented mechanical ventilation. The notebook includes this model extension after the main workflow; scripts/rerun_support_adjusted_models.py can also run it against saved analysis outputs.
- scai_hour16/ contains the descriptive hour-16 Kapur/CSWG minimum evidenced SCAI distribution, cohort and component-availability denominators, and physiology-freshness sensitivities. The notebook includes this calculation; stage A cannot be confirmed from the available OHCA/examination data, and assigned stages are lower bounds.
- FINALIZATION_ANALYSIS_LOCK.md, provenance/, and SOURCE_README.md preserve the protocol and run provenance.

## Reproduction and provenance

The full analysis recorded here was executed locally on MIMIC-III v1.4 inputs; it was not executed in Google Colab. The notebook is configured for a start-to-finish run when the user supplies the required MIMIC-III clinical tables and matched 15-minute SpO2 artifacts in the configured input folder. The source data are not included in this repository.

The CKD-adjusted model rerun is analysis-only: it reuses the saved cohort, exposure, and corrected lactate endpoints, plus local MIMIC-III admission and diagnosis tables. It does not rebuild waveform, cohort, or lactate processing. Its source script is scripts/rerun_ckd_corrected_full_adjusted_models.py.

This package contains code, reports, provenance, and aggregate outputs. It omits MIMIC-III source data and row-level patient records/intermediate tables. The original local output-hash manifest is retained as provenance; hashes for omitted local-only files cannot be recomputed from this repository. No eICU replication is included.

The corrected primary lactate endpoint requires complete observable follow-up through the relevant horizon for both events and non-events. Death, ICU departure, discharge, or an earlier end of observation makes the primary last-value label unavailable. The report retains the prespecified monotone maximum-rise/threshold-crossing companion semantics separately.
