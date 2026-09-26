# Respiratory modulation pilot in the original SpO2 cohort

Frozen before waveform amplitudes or monitor numerics are inspected. The
metadata-selected population is all 11 original dynamics-eligible MIMIC HF
encounters with first-four-hour waveform overlap: five have the original
charted SpO2 exposure and six do not. Do not select patients using outcomes,
waveform appearance, or a favorable coupling estimate. eICU remains the other
project database but its five-minute medians cannot replicate this waveform
experiment. No non-MIMIC/eICU patient data are introduced.

## Biological question and limits

Does the first charted SpO2-instability episode occur during coherent slow
modulation of respiratory impedance amplitude and monitor SpO2? A positive
pilot would support studying a respiratory contribution to the current signal.
It would not diagnose central apnea, establish causal direction, demonstrate
myocardial ischemia, or show mortality benefit. Respiratory impedance is not
calibrated airflow or tidal volume, and shared motion can produce coupling.
The association of periodic breathing with HF saturation oscillations is
[prior knowledge](https://pubmed.ncbi.nlm.nih.gov/8813833/), and direct
[hemodynamic/oxygenation timing studies](https://journals.physiology.org/doi/full/10.1152/jappl.1997.83.4.1184)
also predate this project. No claim of discovering that broad mechanism is allowed.

## Inputs and timing

Retrieve checksum-verified public waveform/numeric files only for these 11
metadata-selected records. Decode Resp and Pleth channels only for the existing
ICU interval [0,240 minutes), preserving gaps. Keep native data and decoded
patient arrays in private Drive Data. Numeric timestamps use header counter
frequency, while waveform frames use header sampling frequency; do not confuse
these clocks. Retain monitor SpO2 values 50–100%, matching original plausibility.
Do not read mortality or troponin outcome columns in this pilot.

Reconstruct each first charted jump from the unchanged production-quality
15-minute bins: absolute adjacent change >=4 percentage points, positive
representative timestamp gap <=30 minutes. Use its ending timestamp. Assign
that episode to the fixed, non-overlapping 20-minute ICU block containing it.
Do not move the block or choose a later episode if the first lacks usable data.

## Fixed processing and signal checks

Analyze all twelve potential 20-minute ICU blocks for coverage. A block needs
full recorded waveform extent, >=95% finite Resp and Pleth samples, and no
remaining gap after internal linear interpolation limited to 0.5 seconds.
Reject constant Resp/Pleth signals. Resample monitor SpO2 to one-second medians,
require >=90% directly observed seconds, and interpolate only internal missing
runs of at most ten seconds. Any remaining missing second makes the block
unusable. Require SpO2 range >=1 percentage point for a spectral estimate;
otherwise report an uninformative constant-saturation block.

Band-pass Resp at 0.1–0.7 Hz (fourth-order Butterworth, zero-phase) and derive
a trailing five-second RMS envelope, aggregated to one-second means. This is
relative amplitude modulation, not ventilation volume. Remove ten seconds at
each block edge. Evaluate the prespecified slow-modulation band 1/120–1/40 Hz;
it is a candidate physiological band, not a clinical diagnostic definition.
Compute detrended Hann-window spectral estimates in non-overlapping 256-second
chunks and their magnitude-squared coherence. Use no overlapping chunks in
the phase-null test. Report maximum coherence in the fixed band.

As a limited pulsatility screen, estimate Pleth power in 0.5–4 Hz divided by
power in 0.05–8 Hz, using 30-second Welch segments. A ratio <0.5 prevents a
positive mechanistic flag. This is an explicit heuristic, not a validated
clinical signal-quality index and not proof that motion is absent.

## Falsification and reporting

For each prespecified episode block, generate 499 signal surrogates by
independently randomizing SpO2 Fourier phase across non-overlapping chunks and
frequencies, retaining its chunk-wise spectral magnitudes and the respiratory
Fourier coefficients. Compare the observed maximum band coherence with the
surrogate maximum over the same frequencies. Seed 20260905; Monte Carlo
p=(1+surrogates>=observed)/500. These are mathematical null signals, never
fabricated patients. They test phase alignment under that signal null, not
clinical causality or all possible common causes.

Holm correction has five slots, one for each exposed encounter. Missing or
uninformative episode blocks occupy p=1 placeholders. A positive pilot flag
requires corrected p<0.05, coherence >=0.5 and the pulsatility screen. Report
all five episode statuses, the number with usable signals and aggregate
coverage for the six controls. Do not test mortality, pool windows as if they
were patients, or search other frequency bands/lags after seeing results.
Advancement requires at least four usable exposed episode blocks and at least
three independently exposed patients with a positive pilot flag. Even passing
would justify follow-up only; eleven encounters cannot establish the requested
major mortality-relevant discovery. Keep failed support visible and do not
relax the above rules to rescue the pilot.

The runnable workflow and aggregate outputs belong in the primary notebook.
Any individual traces, arrays or identifiers remain private. Actual execution
is local unless the notebook truly runs in Colab. The discovery goal is not
complete upon successful extraction, a coherent example, or a small p-value.
