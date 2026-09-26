# Architecture

PhysioGraph's primary workflow is a 4-hour SpO2 landmark study over MIMIC-IV and
eICU. Raw EHR tables are harmonized into cohort/event artifacts, source-aware
12/24-hour outcomes are derived after the landmark, and SpO2 instability is
evaluated per dataset before pooled or cross-dataset analyses. Frozen comparator
models remain legacy/supporting analyses.

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

## Primary data flow

```
PhysioGraph_Final_Clean.ipynb
        |
        v
physiograph_colab_core.run_physiograph_colab()
        |
        +-----------------------+
        |                       |
        v                       v
 MIMIC-IV raw CSVs          eICU raw CSVs
        |                       |
        v                       v
 extract_mimic()            extract_eicu()
        +-----------+-----------+
                    |
                    v
 cohort.csv + events.csv + source/config/code manifest
                    |
                    v
 derive_labels() + build_feature_table()
                    |
                    v
 assemble_spo2_analysis_frame()
   [0, 240) SpO2 features + source-aware (240, 960]/(240, 1680] outcomes
                    |
         +----------+----------------+------------------+
         |          |                |                  |
         v          v                v                  v
 endpoint audit   epidemiology   episode-anchored   grouped OOF models
 & cohort flow    + S1-S12       lactate + all      + transportability
                                  endpoints + IPW/
                                  GEE + meta +
                                  overlap/TMLE/AIPW/
                                  site robustness
         +----------+----------------+------------------+
                    |
                    v
 CSV/JSON/PNG outputs + claims/provenance lint
```

Analysis updates have validated cache-only paths. `analysis_only=True`
regenerates the entire statistical layer without raw extraction. For
episode-timing changes, `refresh_lactate_episode_analysis()` refreshes only the
lactate component. `refresh_multiorgan_episode_analysis()` applies the same
validated-cache boundary to all registered endpoints; after focused records
are committed, `refresh_multiorgan_weighted_analysis()` can rebuild just the
selection-weighted GEE, evidence, and key-results tables. Every path commits
component-specific fingerprints to the existing manifest and none can invoke
raw extraction. `refresh_advanced_episode_inference()` reuses those focused
records for strictly pre-anchor overlap balance, bounded cross-fitted TMLE,
continuous AIPW, center heterogeneity, and E-values; its claim gates also audit
propensity support, outcome-observation support, effective sample size, and
targeting boundaries.

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

Cohort selection for MIMIC-IV and eICU datasets. `build_cohort(dataset,
data_root, config)` dispatches to the package cohort builders; the production
raw-data path performs the equivalent selection inside each streaming extractor.
MIMIC uses ICD-9/10 prefix matching for HF and cardiogenic shock phenotypes.
eICU uses token-based diagnosis matching. Production extraction retains adults
with HF and either ICU admission within 24 hours or cardiogenic shock.

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

### `physiograph.analysis.spo2_protocol`, `spo2_epidemiology`, `spo2_lactate_mechanistic`, and `spo2_multiorgan_mechanistic`

The primary scientific layer defines exact landmark outcomes, censoring,
availability, patient-grouped fold-local models, MIMIC-to-eICU transportability,
measurement-intensity controls, temporal-ordering summaries, risk tables,
dose-response, specificity comparisons, and the S1-S12 sensitivity matrix.
The v2.3 mechanistic modules anchor actual gap-qualified SpO2 transitions,
require strictly prior outcome baselines, test first/maximum lab changes and
incident clinical events at short lags, audit informative observation, separate
desaturation from recovery, apply selection-weighted cluster-robust GEE, and
perform harmonized MIMIC/eICU meta-analysis. The multiorgan layer covers every
registered endpoint with explicit unavailable and underpowered states. It is
an explicitly post-result exploratory amendment.

`run_physiograph_colab(analysis_only=True)` reloads committed ETL artifacts and
starts at this scientific layer. Artifact SHA-256/schema/build checks remain
active; only the requirement that extraction-time and current analysis code
hashes be identical is relaxed.

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
  +-- configs/mimic.yaml      # MIMIC-IV overrides (paths, item IDs, cohort definition)
  |
  +-- configs/eicu.yaml       # eICU overrides (paths, token mappings, cohort definition)
```

`load_config("mimic")` deep-merges `default.yaml` with `mimic.yaml`. `load_config()` returns defaults only.

## Key Design Decisions

1. **Frozen, traceable constants**: Shared extraction mappings and reusable defaults live in YAML; protocol thresholds frozen in `ANALYSIS_PLAN.md` live beside the analysis implementation and are fingerprinted with the code.
2. **YAIB-style preprocessing**: `Preprocessor.fit()` on training data only, `transform()` on all splits. `RuntimeError` if called before fit.
3. **Deterministic splits**: `deterministic_internal_split()` uses evenly-spaced positional selection within sorted target-class groups, not hashing. Non-overlapping by construction.
4. **Chunked streaming**: ETL processes multi-GB CSVs in 250K-row chunks to bound memory usage.
5. **Frozen comparators**: Coefficients, intercepts, and preprocessing parameters are locked to the original study values. No retraining.
6. **Leakage guards at every boundary**: Features, labels, and events are checked for temporal leakage, patient overlap, and outcome contamination at build time and load time.
7. **Separate artifacts**: `features.csv` and `labels.csv` are produced independently. Only the `target` column is merged at training time, preventing outcome leakage by architecture.
