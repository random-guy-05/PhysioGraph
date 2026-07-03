# PhysioGraph Project Goal

## Research Question

**Can SpO2 signal instability (variability, abrupt changes, sampling behavior, and related signal-quality features) measured in the first 4 hours precede or predict early signs of cardiogenic decompensation in the subsequent 12–24 hours?**

Decompensation is operationalized as one or more of:

- Need for mechanical circulatory support (MCS)
- Rising lactate
- Laboratory markers of organ injury
- Falling urine output
- Rising vasopressor/inotropic score (VIS)

### What we mean by each term

| Term | Operational definition |
|------|------------------------|
| **SpO2 signal instability** | Pre-landmark features such as RMSSD, abrupt jumps, below-90 fraction, sampling density, and instability proxy — not absolute SpO2 alone |
| **Precede** | SpO2 instability appears before other decompensation signals move (lead-time / temporal-ordering analysis) |
| **Predict** | SpO2 features at the 4-hour landmark forecast decompensation events or trajectories in the next 12–24 hours |
| **Decompensation outcomes** | MCS initiation, lactate rise, organ-injury labs, urine-output decline, VIS increase — assessed separately and as a composite where appropriate |

### Endpoint hierarchy

| Priority | Endpoint | Rationale |
|----------|----------|-----------|
| Primary | Lactate rise, VIS rise (12–24h) | Frequent, measurable, central to shock escalation |
| Secondary | Urine-output decline, organ-injury labs | End-organ signals with clear directionality |
| Exploratory | MCS need, composite "any decompensation" | Clinically critical but event-sparse |

### Scope guard

We test whether SpO2 **signal dynamics** add information beyond absolute SpO2 and respiratory support context. We do **not** claim SpO2 instability is causally independent of respiratory failure without explicit adjustment and sensitivity analyses.

---

## Mission

Build a rigorous, reproducible pipeline to answer the research question above using physiological time-series from ICU electronic health records (MIMIC, eICU). Every claim must survive leakage checks, external validation, and honest uncertainty reporting.

## How We Answer the Question

1. **Fixed 4-hour observation landmark** — All SpO2 instability features computed from pre-landmark data only.
2. **12–24 hour outcome window** — Decompensation signals assessed in the post-landmark horizon (with sensitivity at 12h vs 24h).
3. **SpO2 feature set** — Variability (RMSSD, SD, IQR), abrupt jumps, below-90 fraction, sampling density, instability proxy; implausible values and missing bins tracked explicitly.
4. **Outcome trajectories** — Not just binary labels: lactate delta, VIS delta, urine-output trend, lab shifts, MCS initiation.
5. **Confounder controls** — Respiratory support (FiO2, vent status), absolute SpO2 level, demographics, baseline hemodynamics; documented availability denominators per dataset.
6. **Comparator baselines** — SpO2 instability models compared against absolute SpO2, lactate-only, and lactate + hemodynamics comparators to test *incremental* value.

## Core Technical Goals

1. **Reproducible pipeline** — Modular ETL, cohort selection, feature engineering, labeling, and validation in the `physiograph` package.
2. **Leakage-safe modeling** — PROBAST+AI Domain 4 guards: no post-landmark features, no outcome contamination, deterministic patient-level splits, fit-on-train-only preprocessing.
3. **Locked comparator validation** — Benchmark against frozen logistic regression comparators on MIMIC and eICU.
4. **External validation** — Harmonize MIMIC and eICU into a shared concept schema; report transportability and calibration per dataset.
5. **Testable analysis modules** — SpO2 drilldown, horizon models, and subgroup analyses live in tested Python modules, not hidden notebook cells.

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Research question answered | SpO2 instability tested against all decompensation outcomes at 12–24h with pre-specified endpoint hierarchy |
| Lead-time evidence | Temporal precedence analysis: does instability precede lactate/VIS/UO shifts? |
| Incremental value | SpO2 dynamics improve prediction beyond absolute SpO2 and respiratory-support-adjusted baselines |
| Pipeline reproducibility | Notebook parity within tolerance; automated tests passing |
| Leakage prevention | All guard checks pass; no forbidden columns in feature sets |
| External validation | eICU harmonization with documented eligibility and missingness rates |
| Claim integrity | Apparent vs validated metrics clearly labeled; null and negative findings reported |

## Non-Goals (Current Scope)

- Graph-based modeling or lactate-centered prediction as the primary research question (supporting methods only).
- CLIF integration until data access and schema mapping are confirmed.
- Causal claims about SpO2 independent of respiratory failure without explicit adjustment.
- Presenting in-sample model metrics as validation results.

## Guiding Principles

1. **One question** — Everything in this project serves the SpO2 instability → decompensation hypothesis.
2. **Truth over polish** — Report null, fragile, and negative findings alongside robust results.
3. **Provenance matters** — Label artifacts as cached, precomputed, local, or freshly executed.
4. **Per-dataset checks first** — Run MIMIC and eICU analyses separately before any pooled claims.
5. **Controls are explicit** — Respiratory support, renal replacement therapy, and baseline hemodynamics must have documented availability denominators.
6. **Single source of truth** — Canonical constants, schemas, and forbidden-column lists live in the package.
