# PhysioGraph dense waveform analysis

## Reproduce the run

The primary runnable deliverable is RUN_DENSE_WAVEFORM_ANALYSIS.ipynb. It embeds the complete end-to-end pipeline and writes the standalone implementation to scripts/run_dense_waveform_analysis.py.

Local run:

    python3 -m pip install -r requirements.txt
    python3 scripts/run_dense_waveform_analysis.py --input-root ~/Downloads --output-root ~/Downloads/physiograph_dense_waveform_analysis

The script searches the input folder recursively, selects full MIMIC-III v1.4 clinical tables by schema, and records alternatives in data_inventory.csv. It requires the supplied mimic3_waveform_spo2_15min.csv.gz file. The full MIMIC-III CHARTEVENTS and LABEVENTS inputs are streamed to reconstruct pre-landmark measurements and lactates.

To rerun only outcome-dependent analyses from a completed output folder after an endpoint-rule correction, use:

    python3 scripts/run_dense_waveform_analysis.py --input-root ~/Downloads --output-root ~/Downloads/physiograph_dense_waveform_analysis --analysis-only-complete-followup

This analysis-only mode reuses linked lactate draws, saved signal features, covariates, and model inputs. It does not reread large event or waveform tables.

## Results

09_final_report/FINAL_REPORT.md contains all 20 requested report sections. Output folders separate cohort, signal QC, landmark lactate, episode lactate, predictive models, MIMIC-IV parity status, sensitivity analyses, figures, and the final report. execution.log, run_status.json, input inventory hashes, and output_hashes.csv preserve execution provenance.

The notebook embeds the entire pipeline, so a Colab copy only needs the input data in its configured Drive folder. The completed analysis in this directory ran locally; it was not executed in Google Colab. The report records unavailable analyses when required source files were not found and does not infer an upstream waveform denominator.
