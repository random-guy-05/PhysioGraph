# Independent measurement replication and adversarial checks

Amendment specified 2026-09-05 UTC, before downloading or inspecting OSV
recordings. This is a local protocol freeze, not external preregistration.
It does not change the original BIDMC primary contrast. No clinical endpoint
will be searched in either dataset.

## Question and locked analysis

Does the same recording-policy effect appear in the independent UCL OSV
dataset (36 healthy adults, approximately one hour of 1 Hz oxygen saturation),
and does it persist when the recordings are longer? Use all records listed in
the publisher RECORDS file. Verify WFDB headers and data against publisher
SHA256SUMS. Retain each record's first min(3600, available) seconds, require at
least 3500 seconds and at least 90% plausible SpO2 samples in 50–100%. No
imputation, retrospective record selection, clinical labeling or age analysis.
Save exclusions and exact durations. This is independent measurement
replication, not external validation of a heart-failure biomarker.

Apply the original 1/5/15/30/60-second cadence interventions and all integer
phases, the same normalized RMSSD, fixed-60-second RMSSD, mean and SD. Lock the
primary replication contrast to phase-averaged absolute relative drift at
60 seconds, normalized minus fixed-lag, on the same positive-reference
records. Bootstrap records 2,000 times with seed 20260915. A 95% interval
excluding zero in the expected direction supports only this measurement
concern. Keep both datasets separate; never pool them to rescue a null.
Repeat the 500 artificial-label policy controls with seed 20260916. These
AUROCs measure artificial recording confounding, not disease prediction.

## Adversarial estimator checks

1. At exactly 60-second cadence, normalized RMSSD and fixed-60-second RMSSD
   are proportional by sqrt(10) on the same valid pairs. Verify this numerical
   identity. They cannot have different patient rankings on that schedule.
2. For complete 1 Hz records, the union of fixed-60-second pairs over all
   phases is exactly the set of native 60-second pairs. Verify the pair-count
   weighted mean of squared phase estimates equals the squared native
   estimate. This explains a comparator advantage; it is an established
   finite-sample identity, not newly discovered physiology.
3. Explicitly distinguish differences between estimands from sampling error:
   normalized adjacent differences estimate 600*D(h)/h, while the fixed-lag
   comparator estimates D(60). Do not say the comparator recovers all
   information, cures irregular observation, or improves prediction.
4. Retain rank, phase, zero-reference and missing-pair diagnostics. A common
   lag preserves a specified estimand but can have large sampling variance.

## Verified literature and novelty decisions

- Eytan et al. (2019), 747 critically ill children, already studied temporal
  sampling and variability including oxygen saturation:
  https://pubmed.ncbi.nlm.nih.gov/31162373/ . Sampling sensitivity is not novel.
- Salverda et al. (2022), not van Zanten as incorrectly named in the original
  plan, found comparable descriptive oxygen summaries at 1/min and 1/sec:
  https://pubmed.ncbi.nlm.nih.gov/35633953/ . Mean/SD controls matter.
- Galuzio et al. (2022) already analyzed oxygen intermittency:
  https://doi.org/10.1038/s41598-022-20493-0 .
- Khalil et al. (2025) already investigated critical slowing down around
  extubation, including an SpO2 supplement:
  https://doi.org/10.1371/journal.pone.0317211 . Recovery-time terminology
  cannot turn this project into a new biological theory.
- OSV source: https://physionet.org/content/osv/1.0.0/ ; original article:
  https://doi.org/10.3389/fphys.2017.00555 . Dataset license: ODC Attribution
  v1.0, as stated by the publisher.

Even replicated positive results would establish a targeted measurement
critique, not a paradigm-shifting discovery. The proposed clinical next step
remains long HF monitor recordings, independently established clinical
outcomes, respiratory/treatment context, and locked cross-site evaluation.
The current repository forbids reading ignored paths, which include its
patient caches; this amendment does not bypass that restriction.
