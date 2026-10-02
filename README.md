# PhysioGraph

## Start here

Open [START_HERE.md](START_HERE.md) for the current study, runnable notebook, results, and consolidated files from other locations. The dense MIMIC-III HF lactate/mortality study lives in [research/spo2_mimic3_hf_lactate](research/spo2_mimic3_hf_lactate/README.md).

The sections below describe the broader PhysioGraph framework. Study-specific definitions and execution provenance are recorded with each study; the consolidation archive contains preserved source snapshots.

Leakage-safe landmark analysis of **SpO2 signal instability as an early warning signal for cardiogenic decompensation** in ICU patients. PhysioGraph extracts structured events from MIMIC and eICU electronic health records, computes pre-landmark SpO2 instability features from the first 4 ICU hours, and tests whether they predict decompensation (lactate rise, vasopressor/inotropic-score rise, urine-output decline, organ-injury labs, MCS initiation) in the subsequent 12–24 hours — beyond clinical context, absolute SpO2, and monitoring intensity/missingness.

The research question and endpoint hierarchy are defined in [PROJECT_GOAL.md](PROJECT_GOAL.md).

## Study design

```
ICU admission ──▶ [0, 240) min observation window ──▶ fixed landmark at 240 min ──▶ (240, 960] / (240, 1680] min
                   SpO2 instability features            all predictors frozen          12 h / 24 h outcome windows
```

- **Predictors (pre-landmark only):** SpO2 variability (SD, RMSSD, IQR, range, MAD), abrupt jumps/drops, and a dynamics-only score; absolute SpO2 summaries, below-90 burden, sampling density, missing-bin count, and longest gap are included in the reference block with clinical controls.
- **Primary outcomes:** lactate rise (Δ ≥ 0.5 mmol/L) and VIS rise at 12 h and 24 h post-landmark.
- **Secondary/exploratory outcomes:** death, vasoactive initiation, urine-output decline and KDIGO oliguria proxies, creatinine/AKI, hepatic, platelet, assay-matched troponin worsening, MCS initiation, and conservative composite flags.
- **Missing ≠ negative:** endpoints that cannot be ascertained (e.g., no VIS events extracted for a dataset) remain unavailable/NaN and are never silently counted as non-events.
- **Incident MCS risk set:** stays already on MCS at the landmark remain in the study for other outcomes but have no incident-MCS label.
- **Endpoint adequacy audit:** every endpoint is classified adequate vs fragile/underpowered using observed-sample and event-count floors (≥200 observed, ≥20 events, ≥20 non-events).

## Analysis methods

| Analysis | Function (`physiograph.analysis.spo2_protocol`) | Purpose |
|---|---|---|
| Incremental nested models | `fit_grouped_incremental_models` | clinical baseline → +absolute SpO2 + sampling/missingness → +SpO2 instability, with patient-grouped `StratifiedGroupKFold`, fold-local preprocessing, out-of-fold AUROC/AUPRC/Brier/ECE/calibration, and patient-level bootstrap CIs incl. ΔAUROC/ΔAUPRC |
| Measurement-intensity control | `fit_grouped_missingness_control` | sampling density / missingness-only model to detect measurement-intensity confounding |
| Temporal ordering | `build_temporal_precedence` | lead time from first pre-landmark dynamics-only instability to each outcome onset (landmark ordering, explicitly **not** causal precedence) |
| External transportability | `fit_external_transportability` | train on MIMIC → freeze preprocessing/model → evaluate untouched eICU |
| Endpoint audit | `build_endpoint_completeness_audit` | coverage, event prevalence, and adequacy status per dataset per endpoint |
| Continuous trajectories | `build_continuous_trajectory_associations` | patient-cluster-bootstrap partial Spearman associations for lactate, VIS, urine, renal, hepatic, platelet, and troponin worsening beyond absolute SpO2, sampling density, missing bins, and longest gap |
| Episode-anchored multiorgan analysis | `run_multiorgan_episode_analyses` | first instability episode versus time-aligned no-episode controls across lactate, organ labs, troponin, VIS/support, MCS/RRT/ventilation, urine, death, and composites; includes strict pre-episode baselines, onset risk sets, observation-process/IPW sensitivity, hierarchical multiplicity, and cross-dataset meta-analysis |
| Advanced episode inference | `run_advanced_episode_inference` | strictly pre-anchor physiology, exposure-overlap balance, cross-fitted observation models, bounded TMLE and one-step AIPW, continuous AIPW changes, center heterogeneity/leave-one-site-out checks, E-values, and fail-closed positivity/ESS gates |
| Locked eICU-to-MIMIC validation | `run_locked_external_replication` | identical explicit-HF phenotype, unchanged ≥4-point binned episode and 4–12 h lag, direct MIMIC ventilation/intubation procedures, audited eICU ventilation-text cleanup, tiered multiplicity, and the complete advanced estimator suite |
| Head-to-head biomarker benchmark | `run_biomarker_benchmark` | identical patient-grouped folds and paired patient-cluster bootstraps for absolute/dynamic SpO2, SBP, baseline lactate trajectory, and a four-hour EHR operationalization of Kapur-CSWG SCAI; includes adjusted RRs, decision curves, and bidirectional MIMIC/eICU transportability |

Quantitative VIS is computed only when infusion units can be standardized. The
current eICU rates are explicitly marked unstandardized, so eICU VIS endpoints
remain unavailable rather than being converted to zero. The mounted MIMIC source
does not include `outputevents.csv`, so MIMIC urine-output endpoints are likewise
explicitly unavailable; eICU urine output comes from `intakeOutput.csv`.
MCS timing is source-documentation based: eICU uses first matching treatment
documentation, while MIMIC prefers device-line/procedure starts and supplements
them with positive operational device charting (including LVAD/RVAD) when a
dedicated procedure row is absent. Removal/inactive rows are excluded.

## Status

The corrected landmark protocol is `spo2_protocol_v2.2`; the hardened artifact
schema is `physiograph_spo2_study_v2.3`, with the exploratory advanced episode
inference amendment documented as v2.4. Its amendments are recorded in
`docs/ANALYSIS_PLAN.md`; no failed or schema-discovery model attempt is treated
as a scientific result. Full-data local results and Colab execution
provenance are reported separately: a local run is never labeled as a Colab run. Legacy
graph-based / long-horizon comparator analyses are supporting/exploratory and do
not drive the primary scientific story.

The final manifest records the execution environment, uncapped/capped run
parameters, input-manifest hashes, source-code hashes, every tabular output
fingerprint, and every figure fingerprint. Fresh-local execution is labeled
fresh-local—not cached/precomputed and never fresh-Colab.
The top-level `run_status.json` is written as `running`, then atomically finalized
as `complete` or `failed` with start/end times, duration, and an explicit failure
reason. SHA-256 plus byte size is authoritative for generated-output integrity;
modification time is informational because cloud synchronization may rewrite it.

To avoid correlated-unit duplication, MIMIC contributes the first ICU stay per
hospital admission and eICU contributes the first ICU unit per hospital encounter;
repeat hospitalizations by the same person remain eligible and are kept together
within patient-grouped resampling.

## Installation

```bash
pip install -e .
```

For analysis-code updates, use `scripts/run_analysis_only.py` to regenerate the
complete statistical layer without rebuilding cohort artifacts. For changes
limited to the episode-to-lactate timing analysis, use
`scripts/refresh_lactate_episode_analysis.py`; it validates the same immutable
artifacts and refreshes only that component and its manifest fingerprints.
For claim-grading changes, `scripts/refresh_endpoint_conclusions.py` rebuilds
only the fingerprint-validated conclusion matrix in seconds and cannot invoke
raw extraction.
For an all-endpoint episode update, run
`scripts/refresh_multiorgan_episode_analysis.py`; it reuses validated cached
events/cohorts and cannot invoke ETL. Once its focused record table exists,
`--weighted-only` refreshes only the observation-weighted GEE, evidence, and
key-result tables in minutes. For the advanced overlap/TMLE/AIPW and
multicenter layer, run `scripts/refresh_advanced_episode_inference.py`; it also
uses only fingerprint-validated cached artifacts. To repeat the locked
eICU-to-MIMIC endpoint validation without rebuilding ETL, run
`scripts/refresh_locked_external_validation.py`; it reads only the cached
artifacts plus MIMIC `d_items.csv` and `procedureevents.csv` for direct
respiratory endpoints. See
[docs/LATEST_RESULTS.md](docs/LATEST_RESULTS.md)
for the current local/precomputed findings and claim limits.

For development with test tooling:

```bash
pip install -e ".[dev]"
```

Requires Python 3.10 or later. Core dependencies: numpy, pandas, scikit-learn, statsmodels, pyyaml, pandera, matplotlib.

## Quick Start

```python
from physiograph.config import load_config
from physiograph.cohort import build_cohort
from physiograph.analysis.spo2_protocol import (
    compute_early_decompensation_outcomes,
    fit_grouped_incremental_models,
    build_endpoint_completeness_audit,
)

config = load_config("mimic")

# Build cohort and events from raw EHR data (see physiograph.pipeline for orchestration)
result = build_cohort("mimic", data_root="/path/to/mimic/csvs")
cohort_df = result.cohort_df

# 12/24 h post-landmark outcomes with availability indicators
outcomes = compute_early_decompensation_outcomes(events_df, cohort_df)

# Endpoint adequacy audit — classify every endpoint before modeling
audit = build_endpoint_completeness_audit(analysis_df)

# Patient-grouped incremental models (clinical → +absolute SpO2/sampling → +instability)
model_results = fit_grouped_incremental_models(analysis_df)
```

The authoritative end-to-end workflow is `PhysioGraph_Final_Clean.ipynb`, which drives `physiograph_colab_core.py`.

## Module Overview

| Module | Purpose |
|--------|---------|
| `physiograph.analysis` | SpO2 protocol v2.2 plus episode-anchored lactate and all-endpoint mechanistic analyses: landmark outcomes, incremental models, measurement-intensity control, transportability, temporal ordering, IPW/GEE, and meta-analysis |
| `physiograph.cohort` | Patient cohort selection for MIMIC and eICU |
| `physiograph.features` | Feature extraction: SpO2 dynamics, lactate dynamics, hemodynamics, missingness |
| `physiograph.models` | Legacy frozen comparator models (supporting/exploratory) |
| `physiograph.validation` | Metrics, calibration, transportability |
| `physiograph.etl` | Raw data extraction with chunked streaming: vitals, labs, pressors/VIS, urine output, and source-qualified MCS procedure/documentation proxies |
| `physiograph.guards` | Data leakage prevention (PROBAST+AI Domain 4 compliant) |
| `physiograph.config` | YAML-based configuration with dataset-specific overrides and explicit config layering |
| `physiograph.constants` | Canonical constants sourced from config |
| `physiograph.schema` | Pandera schemas for cohort, feature, label, and event DataFrames |
| `physiograph.pipeline` | End-to-end orchestration: ETL, labels, features, validation |

## Testing

```bash
# Run all tests
pytest tests/

# Run the SpO2 protocol contract tests
pytest tests/unit/test_spo2_protocol.py

# Run with coverage
pytest tests/ --cov=physiograph --cov-report=term-missing
```

The suite covers protocol clock/boundary contracts, missing-not-negative endpoint
behavior, patient-grouped fold-local modeling, endpoint audits, temporal ordering,
raw-CSV chunking, parity with original notebooks, schema contracts, leakage guards,
and integration flows. Use the command above for the current verified count.

## Configuration

All parameters are centralized in `configs/default.yaml` with dataset-specific overrides:

- `configs/mimic.yaml` for MIMIC paths and item IDs
- `configs/eicu.yaml` for eICU paths and token mappings

Explicit override files can be layered on top: `load_config(dataset="mimic", config_path="my_overrides.yaml")`.

## Data Leakage Prevention

PhysioGraph implements 12 guard mechanisms verified against PROBAST+AI Domain 4 criteria:

- Temporal leakage: post-landmark features blocked by `assert_no_post_landmark_features`
- Patient overlap: grouped CV and deterministic splits with `assert_no_patient_overlap`
- Outcome contamination: `assert_no_outcome_in_features` with forbidden-column lists
- Preprocessing leakage: fold-local / fit-on-train-only preprocessing throughout

See [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md) for the analysis protocol and [PROJECT_GOAL.md](PROJECT_GOAL.md) for claim-scope rules.

## License

MIT
