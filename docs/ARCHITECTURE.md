# Architecture

PhysioGraph processes clinical data through a linear pipeline: raw EHR tables are extracted into cohort and event DataFrames, labels are derived from outcome-window interventions, features are engineered from observation-window events, and predictions are validated against frozen comparator models.

## Module Dependency Diagram

```
                          configs/default.yaml
                                  |
                                  v
                          physiograph.config
                         (load_config, merge)
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
          physiograph.      physiograph.     physiograph.
           constants          schema          guards
                 |                |                |
                 |                v                |
                 |    validate_schema()            |
                 |                |                |
                 +--------+-------+--------+-------+
                          |                |
                          v                v
                    physiograph.etl    physiograph.cohort
                   (extract_mimic,    (build_cohort,
                    extract_eicu)      MIMICCohortBuilder,
                          |            EICUCohortBuilder)
                          |                |
                          v                v
                    physiograph.pipeline
                   (derive_labels,
                    build_feature_table,
                    run_pipeline)
                          |
                 +--------+--------+
                 |                 |
                 v                 v
        physiograph.features   physiograph.validation
        (build_feature_table,  (evaluate, locked_inference,
         lactate,              transportability)
         hemodynamics,
         missingness)
                 |
                 v
        physiograph.models
        (train, evaluate,
         comparators/*)
```

## Data Flow

```
Raw EHR Tables (MIMIC-III / eICU)
        |
        v
  [ETL Extraction]  physiograph.etl
   extract_mimic() / extract_eicu()
        |
        v
  Cohort DataFrame + Events DataFrame
        |
        v
  [Cohort Selection]  physiograph.cohort
   build_cohort("mimic" | "eicu")
        |
        v
  CohortResult (cohort_df, valid_stay_ids, anchors)
        |
        v
  [Label Derivation]  physiograph.pipeline
   derive_labels()
        |
        v
  Labels DataFrame (target, outcome flags, lactate clearance)
        |
        v
  [Feature Engineering]  physiograph.features
   build_feature_table()
        |
        v
  Features DataFrame (50 columns per stay)
        |
        v
  [Leakage Guards]  physiograph.guards
   LeakageGuard.check_probast_domain4()
        |
        v
  [Preprocessing]  physiograph.guards.Preprocessor
   fit(train) -> transform(train, validation, external)
        |
        v
  [Model Training]  physiograph.models.train
   run_locked_comparator_validation()
        |
        v
  Predictions, Metrics, Calibration
        |
        v
  [Transportability]  physiograph.validation.transportability
   cross_dataset_evaluation()
```

## Module Responsibilities

### `physiograph.config`

Loads and merges YAML configuration files. `load_config(dataset)` reads `configs/default.yaml` and optionally merges dataset-specific overrides from `configs/mimic.yaml` or `configs/eicu.yaml`. Resolves data paths for both Colab and local environments. Validates that required top-level keys are present.

### `physiograph.constants`

Canonical constants sourced from `configs/default.yaml` via the config loader. Provides module-level access to time windows, clinical normals, ICD patterns, MIMIC item IDs, eICU mappings, column sets, training parameters, and model architecture values. Notebook-specific aliases included for backward compatibility.

### `physiograph.schema`

Pandera schemas for five data types: MIMICCohortSchema, EICUCohortSchema, FeatureSchema, LabelSchema, and EventSchema. Each schema enforces column presence, types, and value constraints. All schemas use `coerce=True` to handle type mismatches (e.g., int age in MIMIC vs float age in eICU).

### `physiograph.guards`

Data leakage prevention implementing PROBAST+AI Domain 4 compliance. Two main classes:

- **LeakageGuard**: Checks for temporal leakage, patient overlap, outcome contamination, and observation-window violations. `check_probast_domain4()` runs all checks at once.
- **Preprocessor**: YAIB-style fit/transform with `RuntimeError` guard if `transform()` is called before `fit()`. Uses median fill for continuous columns, mode fill for binary columns.

Standalone functions: `assert_no_feature_leakage_columns()`, `assert_observation_only()`, `require_columns()`.

### `physiograph.cohort`

Cohort selection for MIMIC-III and eICU datasets. `build_cohort(dataset, data_root, config)` dispatches to the appropriate builder. MIMIC uses ICD-9/10 prefix matching for HF and cardiogenic shock phenotypes. eICU uses token-based substring matching on free-text diagnosis fields. Both produce `CohortResult` dataclasses with `cohort_df`, `valid_stay_ids`, and `anchors`.

Validators enforce required columns, no duplicate stays, minimum cohort size, and flag value constraints.

### `physiograph.etl`

Extract-transform-load pipelines for raw EHR data. `extract_mimic()` and `extract_eicu()` stream large CSVs in chunks (250K rows) to handle multi-gigabyte tables. Produces `SourceExtraction` containers with cohort and events DataFrames. Shared utilities include `classify_offset_minutes()` for time window assignment, `build_event_frame()` for event construction, and `sanitize_events()` for unit conversion.

`AuditLogger` tracks pipeline steps with timestamps, row counts, and stay counts.

### `physiograph.features`

Feature engineering from observation-window events. `build_feature_table()` aggregates per-stay events into 50 feature columns including baseline vitals, lactate dynamics (slope, clearance, persistence, thresholds), hemodynamic flags (hypotension, tachycardia, time-below fractions), SCAI staging, interaction terms, and composite scores.

Submodules:
- `lactate.py`: 12 functions for lactate binning, clearance, slope, threshold flags, SCAI modifier, and interaction terms
- `hemodynamics.py`: 12 functions for hemodynamic flags, time-below fractions, ratios, HR banding, and perfusion burden
- `missingness.py`: Missingness taxonomy (structural/informative/random), forward-fill with decay, clinical normals fill

### `physiograph.pipeline`

End-to-end orchestration. `run_pipeline()` chains ETL extraction, label derivation, feature engineering, and validation. `derive_labels()` computes outcome flags from intervention events in the outcome window (4-28 hours post-landmark). `build_feature_table()` filters to observation-window events and computes all features.

### `physiograph.models`

Comparator model training and evaluation. `train.py` provides the full frozen comparator pipeline: `ComparatorSpec`, `Preprocessor`, `LogisticModel` dataclasses, `fit_logistic_regression()` with Newton-Raphson optimization, `deterministic_internal_split()`, and `run_locked_comparator_validation()` orchestration.

Four comparator modules under `comparators/` contain frozen coefficients, intercepts, and preprocessing parameters matching the original study exactly.

`evaluate.py` provides pure-NumPy metric computation: AUROC (Wilcoxon-Mann-Whitney), AUPRC (step-function interpolation), Brier score, calibration-in-the-large, ECE, and bootstrap confidence intervals.

### `physiograph.validation`

Model validation and transportability assessment:

- `evaluate.py`: Metric computation, bootstrap CIs, calibration tables, temperature scaling (softplus + LBFGS with guard conditions)
- `locked_inference.py`: Frozen comparator pipeline with locked preprocessing parameters. Reproduces the original study's validation exactly.
- `transportability.py`: Cross-dataset evaluation, calibration drift analysis, split comparison

## Configuration Hierarchy

```
configs/default.yaml          # Shared constants (time windows, clinical normals, column sets)
  |
  +-- configs/mimic.yaml      # MIMIC-III overrides (paths, item IDs, cohort definition)
  |
  +-- configs/eicu.yaml       # eICU overrides (paths, token mappings, cohort definition)
```

`load_config("mimic")` deep-merges `default.yaml` with `mimic.yaml`. `load_config()` returns defaults only.

## Key Design Decisions

1. **Config-driven constants**: All magic numbers live in YAML, not in code. Constants module reads from config to avoid duplication.
2. **YAIB-style preprocessing**: `Preprocessor.fit()` on training data only, `transform()` on all splits. `RuntimeError` if called before fit.
3. **Deterministic splits**: `deterministic_internal_split()` uses evenly-spaced positional selection within sorted target-class groups, not hashing. Non-overlapping by construction.
4. **Chunked streaming**: ETL processes multi-GB CSVs in 250K-row chunks to bound memory usage.
5. **Frozen comparators**: Coefficients, intercepts, and preprocessing parameters are locked to the original study values. No retraining.
6. **Leakage guards at every boundary**: Features, labels, and events are checked for temporal leakage, patient overlap, and outcome contamination at build time and load time.
7. **Separate artifacts**: `features.csv` and `labels.csv` are produced independently. Only the `target` column is merged at training time, preventing outcome leakage by architecture.