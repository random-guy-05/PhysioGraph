# API Reference

Public functions and classes for each module in `physiograph`.

---

## `physiograph.config`

### `load_config(dataset=None)`

Load and merge YAML configuration files.

```python
from physiograph.config import load_config

# Default config only
config = load_config()

# MIMIC-specific config (deep-merged with default)
config = load_config("mimic")

# eICU-specific config
config = load_config("eicu")
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dataset` | `str \| None` | `None` | `"mimic"` or `"eicu"` for dataset-specific overrides |

**Returns:** `dict[str, Any]` with all configuration values, paths resolved.

**Raises:** `FileNotFoundError` if config files are missing. `ValueError` if required keys are absent.

---

## `physiograph.cohort`

### `build_cohort(dataset, data_root=None, config=None, max_stays=None)`

Dispatch to the appropriate cohort builder.

```python
from physiograph.cohort import build_cohort

result = build_cohort("mimic", data_root="/path/to/mimic/csvs")
print(result.cohort_df.shape)       # (N, 13)
print(len(result.valid_stay_ids))   # number of valid stays
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dataset` | `str` | required | `"mimic"` or `"eicu"` |
| `data_root` | `str \| None` | `None` | Path to raw CSV directory |
| `config` | `dict \| None` | `None` | Pre-loaded config dict |
| `max_stays` | `int \| None` | `None` | Cap cohort size (for testing) |

**Returns:** `CohortResult` with `cohort_df`, `valid_stay_ids`, `anchors`.

### `MIMICCohortBuilder`

```python
from physiograph.cohort import MIMICCohortBuilder

builder = MIMICCohortBuilder(data_root=Path("/mimic"), config=config)
result = builder.build_cohort()
```

ICD-9/10 prefix matching for HF (`^428`, `^I50`) and cardiogenic shock (`^78551`, `^R570`). Cohort = HF admissions intersected with (early ICU or shock ICD) intersected with has-ICU-anchor.

### `EICUCohortBuilder`

```python
from physiograph.cohort import EICUCohortBuilder

builder = EICUCohortBuilder(data_root=Path("/eicu"), config=config)
result = builder.build_cohort()
```

Token-based substring matching on free-text diagnosis fields. Inclusion uses the
same rule as MIMIC: adult admissions with an explicit HF match and either ICU
admission within 24 hours of hospital admission or cardiogenic shock. Shock,
cardiomyopathy, and acute MI matches are retained as covariates and never
substitute for HF.

### `harmonize_hf_cohort(cohort_df)`

Apply the same adult explicit-HF plus early-ICU-or-cardiogenic-shock rule to a
combined MIMIC/eICU cohort, propagate diagnosis flags across hospital-unit
transfers, retain one first ICU anchor per encounter, and return both the
harmonized cohort and a dataset-level inclusion/exclusion audit.

### `match_icd_prefix(codes, patterns, versions)`

```python
from physiograph.cohort import match_icd_prefix

matches = match_icd_prefix(
    codes=pd.Series(["4280", "I509"]),
    patterns={9: ["^428"], 10: ["^I50"]},
    versions=pd.Series([9, 10])
)
# matches = pd.Series([True, True])
```

### Validators

```python
from physiograph.cohort import (
    assert_required_columns,
    assert_no_duplicate_stays,
    assert_cohort_size,
    assert_flag_values,
    assert_no_nulls_in_required,
)

assert_required_columns(df, ["stay_id", "age"], "cohort")
assert_no_duplicate_stays(df, "stay_id")
assert_cohort_size(df, min_rows=100)
assert_flag_values(df, ["is_male", "cohort_hf_flag"])
assert_no_nulls_in_required(df, ["stay_id", "age"])
```

---

## `physiograph.etl`

### `extract_mimic(root, audit, *, max_stays=None, max_chunks=None, chunk_size=250_000)`

Extract the MIMIC-IV cohort and events from raw CSVs.

```python
from physiograph.etl import AuditLogger, extract_mimic

audit = AuditLogger(dataset="mimic")
result = extract_mimic(Path("/mimic"), audit, max_chunks=None)
cohort_df = result.cohort_df
events_df = result.events_df
```

**Returns:** `SourceExtraction(cohort_df, events_df)`.

### `extract_eicu(root, audit, *, max_stays=None, max_chunks=None, chunk_size=250_000)`

Extract eICU cohort and events with chunked streaming for large tables.

```python
from physiograph.etl import AuditLogger, extract_eicu

audit = AuditLogger(dataset="eicu")
result = extract_eicu(Path("/eicu"), audit)
```

### `extract_mimic_respiratory_procedure_events(root, cohort_df)`

Targeted analysis-only extraction of direct MIMIC Invasive Ventilation
(item 225792) and Intubation (item 224385) procedure starts. It reads only
`d_items.csv` and `procedureevents.csv` and does not mutate cached ETL outputs.

### `classify_offset_minutes(offset_minutes)`

Classify a time offset into observation window categories.

```python
from physiograph.etl import classify_offset_minutes

classify_offset_minutes(120)   # "observation"  (0 <= x < 240)
classify_offset_minutes(240)   # "landmark"
classify_offset_minutes(500)   # "outcome"      (240 < x <= 1680)
classify_offset_minutes(2000)  # "outside"
```

### `AuditLogger`

```python
from physiograph.etl import AuditLogger

audit = AuditLogger(dataset="mimic")
audit.log(step="cohort_extraction", row_count=17892, stay_count=17892)
audit.log(step="event_extraction", row_count=500000, details="chartevents chunked")
```

---

## `physiograph.analysis`

### `compute_early_decompensation_outcomes(events_df, cohort_df, horizons=(12, 24))`

Derive source-aware post-landmark lactate, VIS, pressor-initiation, MCS, death,
creatinine/KDIGO, hepatic, platelet, assay-matched troponin, and urine-output outcomes. Predictor values
use `[0, 240)` minutes; minute 240 is baseline; outcomes use
`(240, 240 + horizon]`. Last-value negatives require complete fixed-window
follow-up; directly observed pre-censor events remain positive. Death/discharge
and missing source tables are represented through availability columns rather
than negative labels.

### `fit_grouped_incremental_models(analysis_df, ...)`

Fit clinical-only, absolute-SpO2, parsimonious-instability, and full-instability
models separately in MIMIC and eICU, followed by a secondary pooled analysis.
Uses patient-grouped stratified folds, fold-local preprocessing, OOF predictions,
EPV labels, and patient-cluster bootstrap uncertainty.

### `fit_external_transportability(analysis_df, ...)`

Train the nested specifications in MIMIC and evaluate untouched eICU rows using
the training-fitted preprocessing/model pipeline.

### `build_endpoint_completeness_audit(analysis_df)`

Report source availability, observed denominators, event/non-event counts, and
adequacy/fragility for every endpoint and dataset.

### `build_continuous_trajectory_associations(analysis_df, ...)`

Report patient-cluster-bootstrap partial Spearman associations between the
dynamics-only SpO2 score and worsening-oriented continuous lactate, VIS, urine,
renal, hepatic, platelet, and assay-matched troponin trajectories after
rank-adjustment for absolute SpO2, sampling density, missing bins, and longest gap.

### `build_temporal_precedence(events_df, cohort_df, ...)`

Report design-enforced pre-landmark instability ordering for every observed
outcome event, including the complement with no prior instability. These outputs
are descriptive and noncausal.

### `run_epidemiology_analyses(analysis_df, events_df, cohort_df, ...)`

Run stratified risk tables, Mantel-Haenszel estimates, dose-response,
paired-ordering summaries, the hepatic specificity comparator, and the complete
S1-S12 sensitivity matrix.

### `run_lactate_episode_analyses(events_df, cohort_df, analysis_df=None, ...)`

Run the v2.3 post-result mechanistic amendment. It anchors the first qualifying
SpO2 transition, requires the lactate baseline to be strictly before that
episode, evaluates acute 0–1 and delayed 1–8 hour windows plus localization and
cumulative companions, audits informative lactate remeasurement, applies a
remeasurement-weighted sensitivity, tests downward/upward and raw/binned
definitions, and synthesizes harmonized MIMIC/eICU risk ratios. Returned tables
include row-level primary records, paired and controlled summaries,
observation-process and measurement-weighted diagnostics, phenotype strata,
meta-analysis, and graded evidence. All outputs are exploratory and noncausal.

### `refresh_lactate_episode_analysis(output_root, ...)`

Validate immutable cached MIMIC/eICU artifacts and the fingerprinted SpO2
analysis frame, regenerate only episode-anchored lactate tables, and atomically
refresh their manifest fingerprints. This function cannot call raw extractors.
The CLI wrapper is `scripts/refresh_lactate_episode_analysis.py`.

### `run_multiorgan_episode_analyses(events_df, cohort_df, analysis_df=None, ...)`

Run the post-result episode-anchored analysis for every registered
decompensation endpoint. The primary binned ≥4-point absolute transition is
tested over the complete frozen lag grid; five directional/resolution
definitions are tested in endpoint-family focused windows. Outputs contain
strict pre-episode baselines, first-next and worst-window labs, assay-matched
troponin, incident support risk sets, VIS escalation, absolute and relative
urine endpoints, conservative composites, observation-process diagnostics,
IPW/GEE sensitivity, hierarchical q-values, meta-analysis, definitions,
availability, and a one-row-per-focused-estimand key-results table.

### `refresh_multiorgan_episode_analysis(output_root, ...)`

Validate the fingerprinted MIMIC/eICU artifacts, reuse cached cohorts/events,
and rebuild only the all-endpoint episode layer. It cannot call raw extractors.
The CLI wrapper is `scripts/refresh_multiorgan_episode_analysis.py`.

### `refresh_multiorgan_weighted_analysis(output_root)`

Validate the committed focused records/effects/meta-analysis and recompute only
the selection-weighted GEE, evidence summary, and key-results table. The CLI
equivalent is `scripts/refresh_multiorgan_episode_analysis.py --weighted-only`.
This path does not rebuild focused records or raw data.

### `run_advanced_episode_inference(events_df, focused_records, key_results, cohort_df)`

Run the v2.4 exploratory robustness layer from the already frozen primary
episode anchors and focused endpoint grid. It constructs strictly pre-anchor
multivariable physiology, exposure-overlap weights and full balance tables,
cross-fitted endpoint-observation models, bounded binary TMLE with one-step
AIPW diagnostics, continuous AIPW mean changes, eICU hospital-specific and
Paule–Mandel/modified-HKSJ summaries, leave-one-hospital-out estimates, and
support-qualified E-values. Concurrent anchor SpO2 is excluded from exposure
propensity models and used only in observation/outcome models. Positive evidence
flags require minimum event/non-event information, propensity and observation
support, effective sample size, converged non-boundary targeting, and the
relevant multiplicity threshold.

### `refresh_advanced_episode_inference(output_root, ...)`

Validate the fingerprinted cohort/event and all-endpoint focused-record
artifacts, rerun only the advanced episode layer, atomically write its 12 output
tables and refresh receipt, and synchronize the manifest. It cannot invoke raw
extractors. The CLI wrapper is
`scripts/refresh_advanced_episode_inference.py`.

### `run_locked_external_replication(events_df, cohort_df, analysis_df=None, ...)`

Run the unchanged ≥4-point binned SpO2 episode and 4–12-hour ventilation test
in the harmonized MIMIC/eICU HF cohort. The runner cleans the eICU treatment
proxy, recomputes baseline ventilation risk, analyzes every registered endpoint
under the same lag, and returns design, cohort, endpoint-cleaning, conventional,
weighted, meta-analysis, and summary audit tables.

### `refresh_locked_external_validation(output_root, mimic_root, ...)`

Validate immutable cached ETL artifacts, scan only MIMIC `d_items.csv` and
`procedureevents.csv` for direct ventilation/intubation starts, run the locked
analysis and optional complete advanced suite, atomically replace its outputs,
and refresh the manifest. The CLI wrapper is
`scripts/refresh_locked_external_validation.py`.

### `refresh_endpoint_conclusions(output_root)`

Validate the committed completeness, model, epidemiology, sensitivity, and
transportability summaries; rebuild only `endpoint_conclusion_matrix.csv`; and
atomically refresh its manifest fingerprint. Predictive evidence is ineligible
for a suggestive or robust classification below 10 events per transformed
feature, and external evidence must have been trained in the dataset whose
claim it is validating. The CLI wrapper is
`scripts/refresh_endpoint_conclusions.py`; it cannot invoke raw extractors.

### `run_physiograph_colab(..., analysis_only=False)`

Set `analysis_only=True` with `build_new=False` to reuse schema- and
fingerprint-verified `cohort.csv`, `events.csv`, `features.csv`, and `labels.csv`
after analysis-code changes. This bypasses only the ETL code-hash equality gate;
artifact SHA-256, size, schema, build parameters, and source-fingerprint checks
remain enforced. It regenerates the complete analysis/output layer without
rescanning raw MIMIC/eICU source tables.

---

## `physiograph.features`

### `build_feature_table(events_df, cohort_df, config=None)`

Aggregate observation-window events into a per-stay feature matrix with 50 columns.

```python
from physiograph.features import build_feature_table

features_df = build_feature_table(events_df, cohort_df, config=config)
print(features_df.columns.tolist())  # 50 feature columns
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `events_df` | `pd.DataFrame` | required | Long-format events with `stay_id`, `concept`, `offset_minutes`, `value_numeric` |
| `cohort_df` | `pd.DataFrame` | required | Cohort metadata with demographics and flags |
| `config` | `dict \| None` | `None` | Config dict (defaults to notebook values if None) |

**Returns:** `pd.DataFrame` with one row per stay, 50 feature columns plus `stay_id` and `dataset`.

### `extract_features(events_df, cohort_df, config=None)`

Alias for `build_feature_table`. Same signature and behavior.

### Lactate Functions

```python
from physiograph.features.lactate import (
    bin_lactate,                    # Bin lactate into clinical categories
    bin_lactate_focus,              # Focus binning (<2, 2-3, 3-5, >=5)
    compute_lactate_clearance_4h_pct,  # % clearance over 4h
    compute_lactate_slope_per_hr,  # Linear slope per hour
    compute_persistent_lactate_flag,    # Flag persistent elevation
    compute_lactate_threshold_flags,    # Flags for >=2, >=3.1, >=5
    compute_lactate_scai_modifier,     # SCAI stage modifier from lactate
    compute_lactate_acidemia_interaction,    # Lactate x acidemia
    compute_lactate_hypotension_interaction, # Lactate x hypotension
    compute_lactate_tachycardia_interaction, # Lactate x tachycardia
    compute_occult_hypoperfusion_flag,  # Hidden perfusion deficit
)
```

### Hemodynamics Functions

```python
from physiograph.features.hemodynamics import (
    compute_hypotension_flag,          # SBP < 90 or MAP < 65
    compute_severe_hypotension_flag,   # SBP < 80 or MAP < 55
    compute_tachycardia_flag,          # HR > 100
    compute_time_below_65_fraction,   # Fraction of time MAP < 65
    compute_time_below_90_fraction,   # Fraction of time SBP < 90
    compute_time_above_100_fraction,  # Fraction of time HR > 100
    compute_measurement_fraction_series,  # Per-variable measurement completeness
    compute_lactate_map_ratio,         # Lactate / MAP ratio
    compute_lactate_sbp_ratio,         # Lactate / SBP ratio
    bin_heart_rate,                    # Coarse HR binning
    bin_heart_rate_fine,              # Fine HR banding (<100, 100-109, etc.)
    compute_perfusion_burden_score,    # Composite perfusion deficit score
)
```

### Missingness Functions

```python
from physiograph.features.missingness import (
    classify_missingness,          # Taxonomy: structural/informative/random
    apply_forward_fill_and_decay,  # Forward-fill with exponential decay
    fill_clinical_normals,         # Fill NaN with clinical normal values
    compute_missingness_flags_df,  # Per-variable missingness indicators
)
```

---

## `physiograph.guards`

### `LeakageGuard`

```python
from physiograph.guards import LeakageGuard

guard = LeakageGuard(
    landmark_hours=4.0,
    outcome_column="target",
    patient_id_column="stay_id",
    timestamp_column="offset_minutes",
)

# Individual checks
guard.assert_no_post_landmark_features(features_df)
guard.assert_no_outcome_in_features(features_df)
guard.assert_no_patient_overlap(train_ids, test_ids)
guard.assert_observation_only(events_df)

# Comprehensive PROBAST+AI Domain 4 check
violations = guard.check_probast_domain4(
    features_df, train_ids, test_ids, events_df
)
```

### `Preprocessor`

```python
from physiograph.guards import Preprocessor

preprocessor = Preprocessor(
    numeric_columns=["baseline_lactate", "baseline_hr"],
    binary_columns=["is_male", "cohort_hf_flag"],
    categorical_columns=["scai_stage"],
)

preprocessor.fit(train_df)           # Compute stats from training data only
transformed = preprocessor.transform(test_df)  # Apply stored stats
# Raises RuntimeError if transform() called before fit()

print(preprocessor.fitted)           # True
print(preprocessor.feature_names_out)  # List of output column names
```

### Standalone Functions

```python
from physiograph.guards import (
    assert_no_feature_leakage_columns,  # Check columns against forbidden set
    assert_observation_only,            # Check events are observation-window only
    require_columns,                    # Assert required columns exist
)
```

---

## `physiograph.pipeline`

### `derive_labels(cohort_df, events_df, config=None)`

Compute outcome labels from intervention events in the outcome window.

```python
from physiograph.pipeline import derive_labels

labels_df = derive_labels(cohort_df, events_df, config=config)
# Columns: target, pressor_24h_flag, mcs_24h_flag, escalation_24h_flag,
#           renal_injury_24h_flag, hypoperfusion_24h_flag, etc.
```

### `build_feature_table(events_df, cohort_df, config=None)`

Feature engineering pipeline. Filters events to observation window, computes 50 features per stay. Includes leakage guard assertions.

### `run_pipeline(dataset, data_root=None, config=None, output_dir=None)`

End-to-end orchestration: ETL, labels, features, validation.

```python
from physiograph.pipeline import run_pipeline

result = run_pipeline("mimic", data_root="/path/to/mimic")
```

### Validation Functions

```python
from physiograph.pipeline import (
    validate_cohort,    # Validate cohort DataFrame against schema
    validate_labels,    # Validate labels DataFrame against schema
    validate_events,    # Validate events DataFrame against schema
    validate_features,  # Validate features DataFrame against schema
)
```

---

## `physiograph.models`

### `train.py`

#### `fit_logistic_regression(x, y, spec)`

Newton-Raphson logistic regression with backtracking line search.

```python
from physiograph.models.train import fit_logistic_regression

model = fit_logistic_regression(x_train, y_train, spec)
print(model.coefficients)  # np.ndarray
print(model.intercept)     # float
```

**Returns:** `LogisticModel` dataclass with `coefficients`, `intercept`, `converged`, `iterations`, `optimization_trace`.

#### `deterministic_internal_split(df, validation_fraction=0.25, seed=42)`

Deterministic train/validation split with no patient overlap.

```python
from physiograph.models.train import deterministic_internal_split

split = deterministic_internal_split(df, validation_fraction=0.25)
train_mask = split["train"]
val_mask = split["validation"]
```

#### `run_locked_comparator_validation(internal_df, external_df, specs=None)`

Full comparator validation pipeline: split, preprocess, train, evaluate, serialize.

```python
from physiograph.models.train import run_locked_comparator_validation

results = run_locked_comparator_validation(mimic_df, eicu_df)
```

#### `ComparatorSpec`, `Preprocessor`, `LogisticModel`

```python
from physiograph.models.train import ComparatorSpec, Preprocessor, LogisticModel

spec = ComparatorSpec(
    name="lactate_only",
    description="Baseline landmark lactate only.",
    numeric_features=["baseline_lactate"],
    categorical_features=[],
)
```

### `evaluate.py`

```python
from physiograph.models.evaluate import (
    binary_classification_metrics,  # AUROC, AUPRC, Brier, E/O ratio, CITL
    binary_auroc,                   # Wilcoxon-Mann-Whitney AUROC
    binary_average_precision,       # Step-function AUPRC
    calibration_table_rows,         # 10-bin reliability diagram
    compute_calibration,            # Calibration metrics
    average_ranks,                 # Average rank across metrics
    logistic_objective,             # Logistic loss function
    sigmoid,                        # Sigmoid function
)
```

### Comparator Modules

Four frozen comparator models with locked coefficients:

```python
from physiograph.models.comparators.lactate_only import (
    NAME, COEFFICIENTS, INTERCEPT, PREPROCESSOR, MODEL,
    predict_probabilities,
)

from physiograph.models.comparators.lactate_hemodynamics import ...
from physiograph.models.comparators.lactate_end_organ import ...
from physiograph.models.comparators.scai_stage_model import ...
```

Each module exports: `NAME`, `COEFFICIENTS`, `INTERCEPT`, `PREPROCESSOR`, `MODEL`, `ELIGIBLE_FEATURES`, `predict_probabilities(x_matrix)`.

---

## `physiograph.validation`

### `evaluate.py`

```python
from physiograph.validation.evaluate import (
    binary_classification_metrics,  # Full metrics dict
    binary_auroc,                  # AUROC via Wilcoxon-Mann-Whitney
    binary_average_precision,      # AUPRC via step-function
    calibration_table_rows,        # 10-bin calibration
    expected_calibration_error,     # ECE
    bootstrap_metrics,             # Bootstrap confidence intervals
    probability_spread_stats,      # Spread statistics
    temperature_scale,             # Temperature scaling with LBFGS
)
```

### `locked_inference.py`

Frozen comparator pipeline reproducing the original study validation. Key classes and functions mirror `models/train.py` with locked parameters.

### `transportability.py`

```python
from physiograph.validation.transportability import (
    compare_splits,                 # Compare train/validation/external splits
    calibration_drift,              # CITL and ECE drift between datasets
    cross_dataset_evaluation,       # Full internal vs external evaluation
)
```

---

## `physiograph.schema`

```python
from physiograph.schema import (
    MIMICCohortSchema,   # 13 columns, validates MIMIC cohort
    EICUCohortSchema,    # 13 columns, validates eICU cohort
    FeatureSchema,       # 50 columns, validates feature matrix
    LabelSchema,         # 21 columns, validates labels
    EventSchema,         # 14 columns, validates long-format events
    validate_schema,     # Helper: validate DataFrame against schema
)
```

All schemas use `coerce=True` for type coercion. `validate_schema(df, schema, context="label")` catches `SchemaErrors` and re-raises with context.

---

## `physiograph.constants`

Module-level constants sourced from `configs/default.yaml`:

```python
from physiograph.constants import (
    # Time windows
    OBSERVATION_HOURS,       # 4.0
    OUTCOME_HOURS,            # 24.0
    TIME_STEP_HOURS,          # 0.25
    NUM_STEPS,                # 16
    LANDMARK_MINUTES,         # 240

    # Clinical normals (28 variables)
    CLINICAL_NORMALS,         # dict: hr=75.0, sbp=115.0, ...

    # Column sets
    PRIMARY_FEATURE_COLUMNS,  # 50 columns
    COHORT_REQUIRED_COLUMNS,  # 13 columns
    LABEL_REQUIRED_COLUMNS,  # 17 columns
    EVENT_REQUIRED_COLUMNS,   # 14 columns

    # Training parameters
    SEED,                     # 42
    MAX_EPOCHS,               # 60
    BATCH_SIZE,               # 64
    FOCAL_GAMMA,              # 1.5

    # Model architecture
    PHYSIOGRAPH_HIDDEN_DIM,   # 128
    PHYSIOGRAPH_HEADS,         # 8
)
```

---

## Head-to-head biomarker benchmark

### `run_biomarker_benchmark(events, analysis_frame, n_splits=5, bootstrap_repetitions=300)`

Builds leakage-safe `[0,240)`-minute SpO2, SBP, lactate, and Kapur-CSWG SCAI
comparators, then evaluates 11 nested specifications on identical
outcome-observed rows and patient-grouped folds. Outputs include internal OOF
performance, paired patient-cluster-bootstrap contrasts, modified-Poisson
associations, decision curves, marker coverage, SCAI-stage audits, and
bidirectional MIMIC/eICU transportability.

The SCAI variable is a documented EHR operationalization of Kapur et al., JACC
2022 (doi:10.1016/j.jacc.2022.04.049), not clinician adjudication. Missing
normal components are unclassified rather than imputed to stage A; stages are
flagged as lower bounds when OHCA is unavailable.

```python
from physiograph.analysis import run_biomarker_benchmark

tables = run_biomarker_benchmark(
    events,
    analysis_frame,
    n_splits=5,
    bootstrap_repetitions=300,
)
```

### `refresh_biomarker_benchmark(output_root, bootstrap_repetitions=300, n_splits=5)`

Validates the committed analysis frame and MIMIC/eICU manifests, reuses cached
events, and atomically refreshes only the comparator-analysis layer. It never
rebuilds raw data and fails if project code changes during execution.

```bash
python scripts/refresh_biomarker_benchmark.py \
  --output-root /path/to/committed/run \
  --bootstrap-repetitions 300 \
  --n-splits 5
```
