# Changelog

All notable changes to PhysioGraph, organized by development wave.

---

## Wave 1: Ground Truth, Scaffolding, Config, Tests, Schemas

### Ground Truth Extraction (Task 1)

- Extracted reference CSVs, metrics, and model definitions from original notebooks
- Created `tests/ground_truth/` with `predictions.csv`, `metrics.json`, `models.json`, `calibration.csv`
- Validated all row counts against manifests: MIMIC (17,892 cohort, 14,553 labels/features), eICU (27,695 cohort, 24,194 labels/features)
- Confirmed 4 comparator models with frozen coefficients in `models.json`
- Confirmed 97.5% eICU exclusion rate for `lactate_end_organ` model

### Package Scaffolding (Task 2)

- Created `src/physiograph/` package layout with subpackages: `cohort/`, `features/`, `models/`, `models/comparators/`, `validation/`, `etl/`
- Created `pyproject.toml` with build system, dependencies, and pytest configuration
- Package installs and imports: `physiograph.__version__` = "0.1.0"
- Created `conftest.py` with project root and ground truth fixtures

### YAML Configuration (Task 3)

- Created `configs/default.yaml` (650+ lines) with all shared constants
- Created `configs/mimic.yaml` and `configs/eicu.yaml` for dataset-specific overrides
- Implemented `config.py` with `load_config(dataset)` that deep-merges defaults with overrides
- Path resolution: tries Colab drive roots first, falls back to local path
- Validation: checks 6 required top-level fields

### Pytest Infrastructure (Task 4)

- Expanded `conftest.py` with 6 fixtures, 2 helper functions, and marker registration
- Fixed broken `from conftest import ...` in `test_infrastructure.py`
- All 9 infrastructure tests pass
- Markers registered: `slow`, `integration`, `mimic`, `eicu`

### Schema Contracts (Task 5)

- Created `schema.py` with pandera schemas for 5 data types
- MIMICCohortSchema and EICUCohortSchema: 13 columns each
- FeatureSchema: 50 columns with type and range constraints
- LabelSchema: 21 columns with binary flag constraints
- EventSchema: 14 columns for long-format events
- All schemas use `coerce=True` for type coercion

### Data Leakage Guards (Task 6)

- Created `guards.py` (470 lines) with `LeakageGuard` class (7 methods) and `Preprocessor` class (YAIB-style)
- Module-level constants: `ANALYSIS_ONLY_CONTEXT_COLUMNS` (6), `OUTCOME_FLAG_COLUMNS` (9), `FORBIDDEN_FEATURE_COLUMNS` (15, frozenset)
- Standalone functions: `assert_no_feature_leakage_columns()`, `assert_observation_only()`, `require_columns()`
- `Preprocessor.fit()` stores stats from training data only; `transform()` raises `RuntimeError` if called before fit

---

## Wave 2: Core Module Extraction

### Cohort Module (Task 6 continued)

- Created `src/physiograph/cohort/` with `mimic_cohort.py`, `eicu_cohort.py`, `validators.py`
- MIMIC: ICD-9/10 prefix matching, HF admissions intersected with (early ICU or shock ICD)
- eICU: Token-based diagnosis matching, `parse_eicu_age()` handles ">89" de-identification
- Both produce `CohortResult` dataclass with `cohort_df`, `valid_stay_ids`, `anchors`
- `build_cohort()` dispatch function routes by dataset name

### Constants Module (Task 6 continued)

- Created `constants.py` with 100+ constants sourced from `configs/default.yaml`
- Time windows, clinical normals, ICD patterns, MIMIC item IDs, eICU mappings
- Column sets: `PRIMARY_FEATURE_COLUMNS` (50), `COHORT_REQUIRED_COLUMNS` (13), etc.
- SCAI stage assignment logic: E > D > C > B > A based on clinical thresholds

### Feature Engineering Module (Task 6 continued)

- Created `src/physiograph/features/` with `extraction.py`, `lactate.py`, `hemodynamics.py`, `missingness.py`
- `build_feature_table()`: 50 features per stay from observation-window events
- Lactate: binning, clearance, slope, persistence, threshold flags, SCAI modifier, interactions
- Hemodynamics: hypotension/tachycardia flags, time-below fractions, ratios, HR banding, perfusion burden
- Missingness: taxonomy (structural/informative/random), forward-fill with decay, clinical normals fill

### ETL Pipeline Module (Task 8)

- Created `src/physiograph/etl/` with `mimic_extractor.py`, `eicu_extractor.py`, `shared.py`, `audit.py`
- MIMIC: ICD-based cohort, ItemID-based event extraction, chunked streaming
- eICU: Token-based cohort, chunked streaming for 2.2GB+ tables
- `AuditLogger` tracks pipeline steps with timestamps and row counts
- `classify_offset_minutes()` assigns time windows at ETL time

### Validation Module (Task 6 continued)

- Created `src/physiograph/validation/` with `evaluate.py`, `locked_inference.py`, `transportability.py`
- Pure-NumPy metrics: AUROC (Wilcoxon-Mann-Whitney), AUPRC (step-function), Brier, CITL
- Temperature scaling with softplus parameterization and LBFGS optimizer
- Locked comparator pipeline: `ComparatorSpec`, `Preprocessor`, `LogisticModel` dataclasses
- `run_locked_comparator_validation()`: full orchestration with artifact writing

### Comparator Models (Task 7)

- Created 4 frozen comparator modules under `models/comparators/`
- `lactate_only`: 1 feature, intercept=0.127, AUROC 0.613
- `lactate_hemodynamics`: 14 features, intercept=0.046, AUROC 0.625
- `lactate_end_organ`: 10 features, intercept=0.995, AUROC 0.842
- `scai_stage_model`: 5 coefficients, intercept=-0.557, AUROC 0.669
- All coefficients verified against `models.json` to within 1e-14

### Pipeline Module (Task 8 continued)

- Created `pipeline.py` (~900 lines) with `derive_labels()`, `build_feature_table()`, `run_pipeline()`
- `derive_labels()`: computes outcome flags from intervention events in outcome window
- `build_feature_table()`: filters to observation window, computes 50 features, runs leakage guards
- `run_pipeline()`: end-to-end orchestration of ETL, labels, features, validation

---

## Wave 3: Parity Tests (402 tests)

### ETL Parity Tests (Task 15)

- Created `tests/unit/test_etl_parity.py` with 56 tests across 14 test classes
- Covers: text normalization, ICD matching, time window classification, event frame construction, audit logging, column requirements, time window constants
- Key finding: `OUTCOME_WINDOW_END_MINUTES` = 1680 (28h), not 1440

### Cohort Parity Tests (Task 13)

- Created `tests/unit/test_cohort_parity.py` with 59 tests across 10 test classes
- Covers: ICD pattern matching, eICU diagnosis tokens, exclusion criteria (age, LOS, pre-landmark death), validators, dispatch logic

### Leakage Guard Tests (Task 17)

- Created `tests/unit/test_leakage_guards.py` with 46 tests across 10 test classes
- Covers: patient overlap, post-landmark features, outcome contamination, observation-only events, PROBAST+AI Domain 4, Preprocessor fit/transform, constants consistency, standalone functions, edge cases

### Model Parity Tests (Task 16)

- Created `tests/unit/test_model_parity.py` with 103 tests across 13 test classes
- Covers: all 4 model coefficients, intercepts, preprocessing parameters, eligibility counts, prediction functions, metric computations, sigmoid, regularization grid, expanded feature names, row counts, model metadata

### Schema Contract Tests (Task 18)

- Created `tests/unit/test_schema_contracts.py` with 27 tests across 6 test classes
- Covers: all 5 schemas, type coercion, missing columns, value constraints, helper function

### Integration Tests (Task 19)

- Created `tests/integration/test_pipeline_integration.py` with 10 tests
- Covers: config loading, constants from config, schema validation, guard integration, comparator specs, full import chain, preprocessor fit/transform, audit logger, cohort dispatch, feature extraction

---

## Wave 4: Hidden Error Audit

### Missingness Audit (Task 22)

- MIMIC vital signs: 94.7-95.7% coverage. Lactate: 37.4%. End-organ markers: 27.2-56.6%
- eICU: Heart rate 96.8%. All other baseline features missing at 70-87%
- Critical finding: eICU SBP/MAP missing at ~87% due to data extraction gap, not clinical pattern
- Feature overlap in eICU: only 19.6% vs 89.8% in MIMIC
- Complete-case attrition: 97.5-97.6% eICU exclusion for end-organ models

### eICU 97.5% Exclusion Rate Audit (Task 20)

- Confirmed `lactate_end_organ` excludes 23,582 of 24,194 eICU patients (97.47%)
- Root cause: data availability, not code bug. eICU has 1.2-1.6x higher lab missingness than MIMIC
- Conditional attrition: among 3,008 patients with lactate, adding pH + creatinine + bilirubin removes 79.65%
- MIMIC eligibility: 12.19% (5x better than eICU's 2.53%)

### Data Leakage Risk Audit (Task 21)

- Systematic review of 10 potential leakage paths
- No CRITICAL risks found. 8 LOW, 2 MEDIUM
- MEDIUM: Cohort definition bias (MIMIC ICD vs eICU tokens), duplicate forbidden-column definitions
- 12 distinct guard mechanisms, 46 dedicated tests
- PROBAST+AI Domain 4 compliance: PASS on all 5 criteria

### Calibration and Transportability Audit (Task 24)

- Internal calibration: all models near-perfect CITL on MIMIC validation
- External calibration: 3/4 models over-predict on eICU (E/O 1.17-1.21)
- Calibration drift: consistent positive direction (CITL +0.079 to +0.094)
- Root cause: prevalence shift (MIMIC 49-58% positive vs eICU 43-53%)
- Temperature scaling: implemented but never applied in pipeline
- AUROC paradox: all models appear better externally due to selection bias
- AUPRC reveals true degradation: 3/4 models lose 1.9-3.6 pp externally
- Primary transportability barrier: eligibility, not calibration

### Duplicate and Dead Code Audit (Task 23)

- `validation/locked_inference.py` is 95%+ identical to `models/train.py` (~1,200 lines duplicated)
- `pipeline.py:build_feature_table()` duplicates `features/extraction.py:build_feature_table()`
- Column lists defined in 4+ locations across modules
- Preprocessor class name collision between `guards.py` and `models/train.py`
- Estimated ~3,650 duplicate lines; ~2,500 removable by Priority 1 actions

---

## Wave 5: Integration, Colab, Documentation

### CLI Interface

- Created `cli.py` with `run-pipeline`, `validate`, and `audit` subcommands
- `__main__.py` entry point for `python -m physiograph`

### Documentation (Task 28)

- `README.md`: Package overview, installation, quick start, module overview, testing, comparator models, configuration, leakage prevention, citation
- `docs/ARCHITECTURE.md`: Module dependency diagram, data flow, module responsibilities, configuration hierarchy, design decisions
- `docs/API.md`: Full public API reference for all modules with signatures, parameters, and usage examples
- `docs/CHANGELOG.md`: This file, documenting Waves 1-5