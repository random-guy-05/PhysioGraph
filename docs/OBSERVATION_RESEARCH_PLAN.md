# Observation policy and the meaning of SpO2 instability

Protocol frozen before downloading the numeric recordings: 2026-09-05 UTC.
Exploratory, retrospective methods research; not external preregistration.

## Review and research decision

The current review is `docs/LATEST_RESULTS.md`, including the September 3
biomarker benchmark. It reports no consistent primary-endpoint incremental
prediction, no externally replicated primary ventilation association, and
estimator discordance for troponin. Searching additional endpoints until one
is significant would not repair these limitations. The existing multiscale
module already considers cadence, so merely adding another resolution is not
a new contribution.

The unresolved measurement question is narrower: does the diffusion-style
normalization in `src/physiograph/analysis/spo2_instability.py` preserve the
meaning and ordering of instability when the same underlying recordings are
observed on different schedules? This is a prerequisite for interpreting
between-database biomarker transport. It does not by itself explain prior nulls.

## Prior art and novelty boundary

- Eytan et al., 2019, studied vital-sign variability across time scales,
  including oxygen saturation: https://pubmed.ncbi.nlm.nih.gov/31162373/ .
- van Zanten et al., 2022, compared one-per-second and one-per-minute neonatal
  oxygen summaries and found descriptive summaries comparable in their data:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9133439/ .
- Oxygen intermittency measures and sampling sensitivity already exist:
  https://www.nature.com/articles/s41598-022-20493-0 .
- SpO2 critical-slowing-down analyses also exist; recovery rate is not a new
  biological concept: https://pmc.ncbi.nlm.nih.gov/articles/PMC11760018/ .

Candidate contribution: a paired observation-policy intervention audit of the
specific normalized PhysioGraph statistic, including record-rank changes,
phase uncertainty, a fixed-physical-lag comparator, and explicit support
failure. No first-in-literature or paradigm-shifting claim is established.

## Frozen experiments

1. **Analytic and simulated controls.** Compare Brownian increments,
   stationary Ornstein–Uhlenbeck processes with 5/30/120-minute recovery
   constants, and OU plus independent measurement noise. Use 1,000 independent
   four-hour trajectories, observation intervals 1/5/15/60 minutes, and a
   60-minute fixed-lag comparator. Brownian behavior is a positive control for
   the normalization, not a realistic bounded SpO2 model. Report theoretical
   expectations beside Monte Carlo estimates; no hypothesis-test selection.
2. **Real-record paired intervention.** Download all 53 eight-minute BIDMC
   numeric records (1 Hz monitor-derived SpO2, not raw optical saturation or
   arterial ground truth). Use native WFDB `.hea`/`.dat` files; verify every
   downloaded input against the publisher SHA256SUMS manifest. Exclude values
   outside 50–100, retain missingness and require >=90% plausible samples.
   Use only the first 480 samples, retaining 8 minutes on [0,480) seconds.
   Thin the same recording at 1/5/15/30/60-second cadence and enumerate every
   integer-second phase. Do not concatenate records or extrapolate to four
   hours. No heart-failure phenotype or clinical endpoint is available here.
3. **Comparator and uncertainty.** Evaluate normalized RMSSD (sqrt(600 times
   sum of squared adjacent changes / sum of retained gaps)), raw RMSSD,
   standard deviation, mean SpO2, and fixed-60-second-lag RMSSD. The latter
   only uses observed pairs exactly 60 seconds apart, requires >=3 pairs and
   never fills gaps. For each measure compare each thinned phase with its
   native reference. Primary contrast is each record's phase-averaged
   absolute relative drift at 60-second cadence, normalized versus fixed-lag
   RMSSD, restricted to a common positive-reference sample. Bootstrap records
   2,000 times for the paired mean difference. Phases are not independent
   subjects. Report exclusions and zero-reference support separately. Report
   rank correlations and >=4-point adjacent-jump relabeling descriptively.
4. **Policy-only negative control.** Over 500 seeded balanced random label
   assignments, compare AUROC for normalized RMSSD when both label groups use
   1-second recording, versus when artificial label 1 uses 60-second recording
   and label 0 uses 1-second recording. Repeat for the fixed-lag comparator.
   These are artificial labels, not clinical predictions or significance tests.
   AUROC orientation remains fixed; do not flip low AUROCs to claim performance.

## Interpretation and decision gates

A method-level concern is supported if the paired bootstrap interval for the
primary drift difference excludes zero, and analytic controls corroborate the
failure mechanism. A null result is retained. Any benefit for fixed-lag RMSSD
is limited to measurement invariance under these schedules, not prediction.
Report native reference zeros, exclusion rates, phase variation and the
loss of usable pairs. An 8-minute experiment cannot validate the project's
15-minute bins, 30-minute gap limits or 4-hour risk landmark.

For stationary OU plus independent noise, the population normalized squared
increment is 600*[2*V*(1-exp(-h/tau))+2*noise_variance]/h, with h and tau in
seconds. Neither its value nor a single lag autocorrelation identifies a
cardiogenic mechanism. Fixed-lag differences estimate a structure function;
that estimator and its idea are established, not newly invented here.

## Subsequent clinical research (not executed by this protocol)

1. Freeze this measurement audit before revisiting outcomes. In sufficiently
   long, high-resolution HF recordings, thin each record using independently
   estimated MIMIC/eICU schedules; never estimate schedules using outcomes.
2. Estimate common-lag structure functions only on overlapping lag support.
   Fit a noise-nugget plus recovery model only when a >=3-lag identifiability
   profile supports separate parameters; otherwise abstain. Control for
   respiratory interventions, FiO2, perfusion and monitor model where observed.
3. Lock one endpoint (24-hour lactate rise) and one incremental contrast beyond
   clinical + oxygen level + observation history. Train in one system; freeze
   preprocessing, feature support and calibration before testing another.
   Use patient/site-level resampling and observation-weighting sensitivity.
4. Require replicated discrimination, calibration, net benefit and reliable
   measurement support before claiming clinical value. Size a prospective
   standardized-sampling study using pilot event rates and a stated minimum
   useful improvement, rather than inventing a sample-size target.

## Deliverables and boundaries

One standalone Colab-compatible notebook downloads public data and executes
all experiments, tables, checks and figures. Execution here is local and will
be labeled as such. Existing gitignored patient caches will not be accessed.
Preserve the original study notebook and all pre-existing working-tree edits.
Save seeds, software versions, protocol hash, per-record data provenance,
record-level results, bootstrap summaries and figures. No external publication,
communication, clinical intervention or outcome claims are part of this run.
