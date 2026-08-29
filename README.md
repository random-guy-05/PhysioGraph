# PhysioGraph

Leakage-safe landmark analysis of **SpO2 signal instability as an early warning signal for cardiogenic decompensation** in ICU patients. PhysioGraph extracts structured events from MIMIC and eICU electronic health records, computes pre-landmark SpO2 instability features from the first 4 ICU hours, and tests whether they predict decompensation (lactate rise, vasopressor/inotropic-score rise, urine-output decline, organ-injury labs, MCS initiation) in the subsequent 12–24 hours — beyond what absolute SpO2 level alone provides.

The research question and endpoint hierarchy are defined in [PROJECT_GOAL.md](PROJECT_GOAL.md).

## Study design

```
ICU admission ──▶ [0, 240) min observation window ──▶ fixed landmark at 240 min ──▶ (240, 960] / (240, 1680] min
                   SpO2 instability features            all predictors frozen          12 h / 24 h outcome windows
```

- **Predictors (pre-landmark only):** SpO2 variability (SD, RMSSD, IQR, range, MAD), abrupt-jump rate, below-90 fraction, instability proxy score, plus absolute SpO2 summaries, sampling density, and clinical controls (demographics, baseline lactate/creatinine, MAP, respiratory support, ventilation, RRT).
- **Primary outcomes:** lactate rise (Δ ≥ 0.5 mmol/L) and VIS rise at 12 h and 24 h post-landmark.
- **Secondary/exploratory outcomes:** urine-output decline proxy, KDIGO-style creatinine worsening, hepatic laboratory worsening, MCS initiation, and a composite early-decompensation flag.
- **Missing ≠ negative:** endpoints that cannot be ascertained (e.g., no VIS events extracted for a dataset) remain unavailable/NaN and are never silently counted as non-events.
- **Endpoint adequacy audit:** every endpoint is classified adequate vs fragile/underpowered using observed-sample and event-count floors (≥200 observed, ≥20 events, ≥20 non-events).

## Analysis methods

| Analysis | Function (`physiograph.analysis.spo2_protocol`) | Purpose |
|---|---|---|
| Incremental nested models | `fit_grouped_incremental_models` | clinical baseline → +absolute SpO2 → +SpO2 instability, with patient-grouped `StratifiedGroupKFold`, fold-local preprocessing, out-of-fold AUROC/AUPRC/Brier/ECE/calibration, and patient-level bootstrap CIs incl. ΔAUROC/ΔAUPRC |
| Negative control | `fit_grouped_missingness_control` | sampling density / missingness-only model to detect measurement-intensity confounding |
| Temporal ordering | `build_temporal_precedence` | lead time from first pre-landmark instability to each outcome onset (landmark ordering, explicitly **not** causal precedence) |
| External transportability | `fit_external_transportability` | train on MIMIC → freeze preprocessing/model → evaluate untouched eICU |
| Endpoint audit | `build_endpoint_completeness_audit` | coverage, event prevalence, and adequacy status per dataset per endpoint |

VIS is computed quantitatively from drug-specific infusion rates (norepinephrine, epinephrine, dopamine, dobutamine, phenylephrine, vasopressin, milrinone) with weight/unit normalization; urine output is streamed from source records in both datasets.

## Status

The analysis protocol (`spo2_protocol_v1.0`) and its implementation are complete and tested (full suite green). **Scientific conclusions are not yet frozen**: full-data MIMIC/eICU endpoint QA, fresh rebuilds, prespecified sensitivity analyses, and clean-Colab end-to-end execution are still being finalized. Legacy graph-based / long-horizon comparator analyses are retained as supporting/exploratory material and no longer drive the primary scientific story.

## Installation

```bash
pip install -e .
```

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

# Patient-grouped incremental models (clinical → +absolute SpO2 → +instability)
model_results = fit_grouped_incremental_models(analysis_df)
```

The authoritative end-to-end workflow is `PhysioGraph_Final_Clean.ipynb`, which drives `physiograph_colab_core.py`.

## Module Overview

| Module | Purpose |
|--------|---------|
| `physiograph.analysis` | SpO2 protocol v1.0: landmark outcomes, incremental models, negative controls, transportability, temporal ordering |
| `physiograph.cohort` | Patient cohort selection for MIMIC and eICU |
| `physiograph.features` | Feature extraction: SpO2 dynamics, lactate dynamics, hemodynamics, missingness |
| `physiograph.models` | Legacy frozen comparator models (supporting/exploratory) |
| `physiograph.validation` | Metrics, calibration, transportability |
| `physiograph.etl` | Raw data extraction with chunked streaming: vitals, labs, pressors/VIS, urine output, MCS procedures |
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

The suite includes 426 passing tests covering protocol clock/boundary contracts, missing-not-negative endpoint behavior, patient-grouped fold-local modeling, endpoint audits, temporal ordering, parity with original notebooks, schema contracts, leakage guards, and integration flows.

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

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the leakage risk audit and [PROJECT_GOAL.md](PROJECT_GOAL.md) for claim-scope rules.

## License

MIT