# PhysioGraph

Graph-based physiological time-series analysis for clinical prediction of cardiogenic shock progression in heart failure patients. PhysioGraph extracts structured features from MIMIC-III and eICU electronic health records, engineers clinical variables (lactate dynamics, hemodynamic thresholds, SCAI staging), and validates predictions against four frozen comparator models with locked coefficients.

## Installation

```bash
pip install -e .
```

For development with test tooling:

```bash
pip install -e ".[dev]"
```

For PyTorch-based GNN models:

```bash
pip install -e ".[torch]"
```

Requires Python 3.10 or later. Core dependencies: numpy, pandas, scikit-learn, pyyaml, pandera.

## Quick Start

```python
from physiograph.config import load_config
from physiograph.cohort import build_cohort
from physiograph.features import build_feature_table
from physiograph.guards import LeakageGuard

# Load dataset configuration
config = load_config("mimic")

# Build cohort from raw EHR data
result = build_cohort("mimic", data_root="/path/to/mimic/csvs")
cohort_df = result.cohort_df

# Extract features from observation-window events
features_df = build_feature_table(events_df, cohort_df, config=config)

# Run leakage guards before training
guard = LeakageGuard(landmark_hours=config["observation_hours"])
guard.assert_no_post_landmark_features(features_df)
guard.assert_no_outcome_in_features(features_df)
guard.assert_no_patient_overlap(train_ids, test_ids)
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `physiograph.cohort` | Patient cohort selection for MIMIC-III and eICU |
| `physiograph.features` | Feature extraction: lactate dynamics, hemodynamics, SCAI staging, missingness |
| `physiograph.models` | Comparator model training, evaluation, and frozen inference |
| `physiograph.validation` | Metrics, calibration, transportability, locked comparator validation |
| `physiograph.etl` | Raw data extraction with chunked streaming for large CSVs |
| `physiograph.guards` | Data leakage prevention (PROBAST+AI Domain 4 compliant) |
| `physiograph.config` | YAML-based configuration with dataset-specific overrides |
| `physiograph.constants` | Canonical constants sourced from config |
| `physiograph.schema` | Pandera schemas for cohort, feature, label, and event DataFrames |
| `physiograph.pipeline` | End-to-end orchestration: ETL, labels, features, validation |

## Testing

```bash
# Run all tests
pytest tests/

# Run unit tests only (skip slow/integration)
pytest tests/ -m "not slow and not integration"

# Run with coverage
pytest tests/ --cov=physiograph --cov-report=term-missing
```

The test suite includes 402 tests covering parity with original notebooks, schema contracts, leakage guards, model coefficients, and integration flows.

## Comparator Models

Four logistic regression comparators with frozen coefficients from the original study:

| Model | Features | MIMIC AUROC | eICU AUROC | eICU Eligible |
|-------|----------|-------------|------------|----------------|
| `lactate_only` | 1 (baseline lactate) | 0.613 | 0.640 | 12.4% |
| `lactate_hemodynamics` | 14 (lactate + hemodynamics) | 0.625 | 0.750 | 2.4% |
| `lactate_end_organ` | 10 (lactate + end-organ markers) | 0.842 | 0.843 | 2.5% |
| `scai_stage_model` | 5 (lactate + SCAI stage) | 0.669 | 0.697 | 12.4% |

## Configuration

All parameters are centralized in `configs/default.yaml` with dataset-specific overrides:

- `configs/mimic.yaml` for MIMIC-III paths and item IDs
- `configs/eicu.yaml` for eICU paths and token mappings

```python
from physiograph.config import load_config

# Load default config
config = load_config()

# Load MIMIC-specific config (merges with default)
config = load_config("mimic")
```

## Data Leakage Prevention

PhysioGraph implements 12 guard mechanisms verified against PROBAST+AI Domain 4 criteria:

- Temporal leakage: post-landmark features blocked by `assert_no_post_landmark_features`
- Patient overlap: deterministic splits with `assert_no_patient_overlap`
- Outcome contamination: `assert_no_outcome_in_features` with 15 forbidden columns
- Preprocessing leakage: YAIB-style fit-on-train-only `Preprocessor` class

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the full leakage risk audit, calibration analysis, and transportability findings.

## Citation

If you use PhysioGraph in your research, please cite:

```
PhysioGraph: Graph-based physiological time-series analysis for
cardiogenic shock prediction in heart failure patients.
```

## License

MIT