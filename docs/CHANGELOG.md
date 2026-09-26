# Changelog

All notable changes to PhysioGraph, organized by development wave.

---

## Wave 7: Episode-Anchored Lactate Amendment and Fast Analysis Resume

- Added a transparent post-result mechanistic analysis that asks whether an
  actual SpO2 instability transition precedes the next lactate rise in acute
  0–1 and delayed 1–8 hour windows, with 1–2/2–4/4–8 localization and 8–12 and
  cumulative 0–8 hour companions.
- Enforced a last lactate strictly before the episode; a same-time or later
  lactate can never be reused as baseline.
- Separated downward desaturations from upward recovery, tested 3/4/5-point
  episode thresholds, and added raw-chart versus 15-minute-bin sensitivity.
- Added quantile time-aligned no-episode controls, continuous and thresholded
  changes, informative-remeasurement diagnostics, inverse-probability
  remeasurement-weighted sensitivity, clinical/measurement strata, and
  fixed/random MIMIC-eICU meta-analysis with heterogeneity.
- Added focused within-question BH q-values while retaining global exploratory
  q-values; all new result-informed analyses are labeled post hoc.
- Replaced overparameterized focal GLM adjustment with a parsimonious candidate
  set and a hard fail-closed gate below five events per fitted parameter.
- Added `analysis_only=True`, which preserves artifact/schema/fingerprint
  validation while allowing analysis-code changes to reuse committed ETL
  artifacts. The clean notebook defaults to this fast resume mode.
- Added `refresh_lactate_episode_analysis()` and its CLI wrapper for a
  component-only, manifest-committed refresh that never invokes raw extraction.
- Made endpoint claim grading fail closed when an internal or external model
  has fewer than 10 events per transformed feature, and prevented MIMIC-to-eICU
  transport results from being misapplied to an eICU-trained model claim.
- Added `refresh_endpoint_conclusions()` and its CLI wrapper for a seconds-long,
  fingerprint-validated conclusion-only refresh with no raw-data access.
- Component refreshes now synchronize the top-level `run_status.json` manifest
  SHA; the integrity audit fails closed on any stale provenance pointer.

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

---

## Wave 6: SpO2 Protocol v2.0 Correctness and Source-Backed Validation

### Root-cause correction

- Confirmed directly in raw MIMIC-IV that item `220277` is pulse-oximetry SpO2
  and occurs millions of times; the historical all-zero result was not a source-
  data fact.
- Fixed pilot cohort/chunk misalignment: truncated pilots now select eligible
  stays from the same physical vital-sign chunks being scanned.
- Replaced hospital-admission IDs with true ICU `stay_id` anchors and retained
  one first ICU stay per admission to prevent cross-stay event mixing.
- Added source/config/code fingerprints and a fatal full-run invariant when raw
  SpO2 rows exist but no plausible cohort-window SpO2 survives.

### Cohort, exposure, and outcome protocol

- Enforced `[0, 240)` predictor and `(240, 240 + horizon]` outcome windows.
- Added same-timestamp deduplication, 15-minute bins, gap-qualified transitions,
  minimum dynamics support, alternate jump thresholds, desaturation duration,
  deficit AUC, and sustained-episode features.
- Retained patients already receiving MCS/vasoactives at the landmark and modeled
  post-landmark initiation/escalation with explicit baseline flags.
- Added death/discharge censoring and source-aware missingness so unavailable
  endpoints remain `NaN`, never negative.
- Expanded lactate, creatinine/KDIGO, bilirubin, AST/ALT, INR, platelet, MCS,
  pressor-initiation, VIS, and urine-output outcomes at 12 and 24 hours.
- Marked unstandardized eICU infusion rates as quantitatively unavailable for VIS;
  labeled eICU MCS time as first device-treatment documentation rather than a
  verified start time.
- Preserved true zero urine measurements; declared MIMIC urine outcomes
  unavailable because the mounted raw source lacks `outputevents.csv`.

### Inference and reporting

- Made MIMIC and eICU per-dataset grouped-CV analyses primary; pooled estimates
  are secondary and MIMIC-to-eICU transportability is reported separately.
- Added a common dynamics-eligible sample, fold-local preprocessing, patient-
  grouped folds, patient-cluster bootstrap intervals, OOF calibration/ROC/PR,
  EPV and fragility labels, and parsimonious/full instability specifications.
- Implemented all sensitivity axes S1-S12 with unavailable rows retained in the
  matrix and BH adjustment across prespecified families.
- Corrected exposure assignment so measured stable stays are unexposed rather
  than missing, and added balanced within-dataset instability tertiles.
- Added all outcome events to temporal-ordering denominators, including events
  with no prior instability; precedence is labeled design-enforced and noncausal.
- Reclassified hepatic worsening as a specificity comparator rather than a true
  negative control.
- Added cohort flow, Table 1, dataset-specific feature missingness, endpoint
  completeness/conclusion matrices, OOF prediction files, performance curves,
  and output/claim linting.

### Runtime defects found only by raw-data pilots

- Disabled pandas nested low-memory inference in the shared chunk reader; later
  mixed-type eICU respiratory chunks previously triggered an internal parser
  `IndexError`.
- Stabilized sparse odds-ratio table schemas when every candidate fit is skipped.
- Guarded temporal-precedence bootstrap logic when outcomes exist but no eligible
  prior-instability pairs exist.

---

## Wave 7: SpO2 Protocol v2.1 Full-Source Hardening

- Added current-release MIMIC-IV laboratory item discovery from `d_labitems.csv`,
  vital-sign unit normalization, milrinone VIS extraction, and a five-minute
  pressor restart grace.
- Restricted eICU to the first ICU unit per hospital encounter while preserving
  repeat hospitalizations and patient-grouped validation.
- Corrected fixed-window censoring, retained confirmed pre-censor events, added
  separate early-death endpoints, and required urine-output interval coverage.
- Kept troponin T and I assay-specific through baseline/post comparisons and
  labeled both MIMIC and eICU MCS timings as source-specific proxies.
- Fixed pandas string/categorical preprocessing that invalidated the first
  full-source discovery model attempt; that attempt is not a scientific result.
- Added atomic output commits plus source, configuration, code, input-manifest,
  and output fingerprints so a mixed-version or partial run cannot be accepted.
- Added mounted eICU aliases for current platelet/INR/troponin names and
  vasoactive trade names; corrected eICU discharge-offset arithmetic.
- Excluded IABP-removal documentation from incident MCS and broadened MIMIC MCS
  capture to positive operational LVAD/RVAD/IABP/Impella/ECMO documentation.
- Added MIMIC whole-blood creatinine, incident-only absolute organ-injury
  crossings, right-closed urine-coverage bins, and coverage-qualified urine
  temporal onsets.
- Refit continuous-trajectory nuisance adjustment inside clustered bootstrap
  replicates and tightened robust conclusions to require AUROC, AUPRC, and
  patient-cluster epidemiologic interval support.
- Removed eICU urine occurrence/count, incontinence-event, and mixed-stool rows
  from quantitative urine volume before decline and oliguria derivation.

## Wave 8: SpO2 Protocol v2.2 Hypothesis-Complete Reference and Reporting

- Added sampling density, missing-bin count, and longest gap directly to the
  clinical + absolute-SpO2 reference for internal and MIMIC-to-eICU incremental
  prediction; per-feature dynamics ORs use the same adjustment set.
- Removed residual legacy-target drift from descriptive tables, raw summaries,
  trajectories, and plots; prespecified 12/24-hour outcomes are now reported
  per database, with one median per stay/bin rather than measurement weighting.
- Required aggregate urine-rate decline before assigning a temporal urine onset,
  raised primary model bootstrap precision to 500 repetitions, and required at
  least 80% valid internal/external resamples for a robust endpoint conclusion.
- Made the eligible cohort index and cohort metadata authoritative during
  analysis-frame assembly so stale feature/label keys cannot restore exclusions
  and omitted feature rows cannot erase known demographics or censoring data.

## Wave 9: Artifact Schema v2.3 Inference and Cross-Source Hardening

- Normalized heterogeneous categorical values inside every training fold so
  pooled secondary models cannot fail when equivalent source fields use mixed
  integer and string representations.
- Required a pooled endpoint to be observed in at least two datasets; endpoints
  unique to MIMIC or eICU remain dataset-specific and are explicitly skipped in
  pooled prediction, inference, summaries, and trajectories.
- Required minimum event and non-event counts for binomial analyses, classified
  separation/non-convergence/non-finite odds-ratio inference as non-estimable,
  and excluded such rows from multiplicity correction.
- Reused the already-authoritative analysis frame for availability reporting,
  eliminating a redundant multi-million-row reassembly and its risk of drift.

## Wave 10: Post-run Inference-Status and Provenance Audit

- Required finite point estimates and confidence intervals before risk-ratio,
  Mantel-Haenszel, dose-response, or continuous-trajectory rows can be labeled
  estimated; degenerate and zero-cell designs are explicit non-estimable rows.
- Added patient-cluster bootstrap 95% intervals and valid-replicate counts to
  continuous-lactate S2 Spearman sensitivities, with an 80% validity floor.
- Classified present-but-entirely-missing endpoint columns as unavailable,
  correcting eICU VIS and source-absent MIMIC urine reporting semantics.
- Made fresh-local versus cached/precomputed versus fresh-Colab provenance
  explicit and added standalone run parameters plus figure fingerprints to the
  final manifest.
- Distinguished source-unavailable endpoints from merely underpowered endpoints
  in risk, Mantel-Haenszel, dose-response, and S1-S12 sensitivity tables; every
  non-success inferential row now carries an explicit reason.
- Made `run_status.json` an atomic running/complete/failed state record with
  timestamps, duration, and failure details, and documented SHA-256/byte size
  as authoritative output integrity when cloud sync rewrites file mtimes.

## Wave 11: Focal-Inference Gate Correction

- Corrected the binomial inference gate so a finite prespecified focal
  coefficient, robust standard error, p-value, and confidence interval are not
  discarded solely because a sparse nuisance-category coefficient has undefined
  robust inference.
- Retained the sparse nuisance terms as explicit warning columns in association,
  variability, and respiratory-interaction outputs; required focal terms still
  fail closed on separation, non-convergence, missing coefficients, or any
  non-finite focal inference.
- Added adversarial coverage for finite focal inference with non-finite nuisance
  categories and strengthened the respiratory-interaction regression test to
  require all four prespecified diagnostics to fit on an estimable design.

## Wave 12: Episode-Anchored Multiorgan Analysis

- Generalized the actual-episode SpO2 design from lactate to all 21 registered
  endpoints, using a frozen 12-window primary grid and all five
  direction/resolution sensitivities in endpoint-family focused windows.
- Added strict pre-episode lab baselines, first-next and worst-window variants,
  full-follow-up requirements for worst-window negatives, assay-matched
  troponin, incident support risk sets, delayed-onset VIS logic, and both
  absolute KDIGO and strict-preanchor relative urine outcomes.
- Added explicit observation-process tables and inverse-probability-of-
  observation sensitivities. Weighted outcome inference now uses
  patient-clustered Poisson-log-link GEE; a regression test covers filtered
  frames with non-contiguous source indices.
- Added endpoint-, family-, cross-endpoint-, and global BH corrections plus
  information gates. The compact key table reserves its strongest label for
  results that survive cross-endpoint FDR in unadjusted, adjusted, and weighted
  analyses; cross-dataset random-effects evidence is labeled separately.
- Added analysis-only and weighted-only refresh paths, nine fingerprinted output
  tables, notebook displays, exact primary/sensitivity-grid integrity checks,
  endpoint availability reasons, and refresh receipts that attest no raw-data
  rebuild.

## Wave 13: Advanced Episode Inference and Positivity Gates

- Added one strictly pre-anchor covariate row per primary episode/control
  anchor, with multivariable physiology, support state, measurement intensity,
  demographics, unit, and site. Concurrent anchor SpO2 is excluded from the
  exposure model because it partly defines the transition.
- Added endpoint-independent exposure-overlap weights with complete encoded
  covariate-balance output; both datasets reach maximum weighted absolute
  standardized mean difference below 0.022.
- Added patient-grouped cross-fitted outcome-observation and outcome nuisance
  models, bounded binary TMLE, retained one-step AIPW diagnostics, and
  continuous AIPW mean worsening in clinical units for every defined graded
  endpoint.
- Added eICU hospital-specific risk ratios, Paule–Mandel heterogeneity,
  modified HKSJ intervals, leave-one-hospital-out estimates, and support-aware
  E-values. Multicenter estimates remain explicitly unadjusted.
- Corrected observation-probability truncation so complete observation at
  probability 1 is never spuriously clipped. Positive labels now fail closed
  when more than 10% of rows violate exposure or observation support, either
  weighted arm has effective sample size below 50, or TMLE hits its targeting
  boundary.
- Added a cached-data-only advanced refresh, 12 fingerprinted tables, notebook
  displays, exact-grid and targeting audits, and regression tests preventing
  support failures or multicenter cross-FDR signals from being mislabeled.

## Wave 14: Head-to-Head SpO2, SBP, Lactate, and Kapur-SCAI Benchmark

- Added full four-hour SBP and lactate level/trajectory feature blocks and a
  boundary-tested EHR operationalization of the Kapur-CSWG 2022 SCAI stages.
  Missing normal components remain unclassified, OHCA absence is explicit, and
  every assigned stage is labeled a lower bound.
- Added 11 fair nested specifications on identical outcome rows and
  patient-grouped folds, 300 paired patient-cluster bootstrap contrasts,
  fold-local preprocessing, calibration, decision curves, and events-per-
  feature gates across every registered endpoint.
- Added marker-specific and mutually adjusted patient-clustered modified-
  Poisson associations with tier-specific BH correction plus bidirectional
  MIMIC/eICU frozen-model transportability.
- Added explicit source coverage audits and optional streaming extraction of
  eICU cuff SBP/MAP from `vitalAperiodic.csv`; current cached eICU SBP results
  remain source-limited because that mounted file is absent.
- Added an atomic cached-data-only benchmark refresh, 14 fingerprinted output
  tables, clean-notebook displays, vectorized exact weighted bootstrap metrics,
  and regression tests for every SCAI boundary, leakage at minute 240, cuff-BP
  extraction, common folds, weighted metrics, and association execution.
