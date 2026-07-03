# Ground-Truth Parity Artifacts

## Provenance

- **Source notebooks**: `PhysioGraph_HF_Shock.ipynb`, `PhysioGraph_External_Pipeline.ipynb`
- **Extraction date**: 2026-04-25
- **Extraction method**: Direct copy from `physiograph_outputs/locked_comparator_validation/`
- **No transformations applied** — these are canonical frozen outputs.

## Tolerance Thresholds

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| AUROC | ±0.001 | Floating-point rounding across platforms |
| Predictions (probability) | ±1e-6 | Numeric precision in logistic regression |
| Cohort row counts | Exact match | Deterministic pipeline — no stochasticity |
| Calibration bins | Exact match | Derived from fixed predictions |

## File Inventory

| File | Description |
|------|-------------|
| `predictions.csv` | 24,765 rows — model predictions for 4 comparators × 3 splits |
| `metrics.json` | 12 metric records — AUROC/AUPRC/Brier/calibration per model×split |
| `models.json` | 4 frozen model definitions — coefficients, preprocessing, eligibility |
| `calibration.csv` | 93 rows — calibration bin data for reliability diagrams |
| `csv_manifest.json` | Row counts and column headers for all CSV files |
| `validate_ground_truth.py` | Validation script — run from project root |

## Models

| Model | Features | External AUROC | External N |
|-------|----------|----------------|------------|
| `lactate_only` | 1 | 0.6395 | 3,008 |
| `lactate_hemodynamics` | 14 | 0.7501 | 589 |
| `lactate_end_organ` | 10 | 0.8428 | 612 |
| `scai_stage_model` | 5 | 0.6968 | 3,008 |

## Source Cohort Sizes (from manifests)

| Dataset | Cohort Rows | Label Rows | Feature Rows | Event Rows |
|---------|-------------|------------|--------------|------------|
| MIMIC | 17,892 | 14,553 | 14,553 | 838,835 |
| eICU | 27,695 | 24,194 | 24,194 | 4,176,446 |

## Validation

```bash
python tests/ground_truth/validate_ground_truth.py
```

This script verifies:
1. Source CSV row counts match manifest declarations
2. All required ground-truth files are present
3. Predictions row count matches csv_manifest.json
