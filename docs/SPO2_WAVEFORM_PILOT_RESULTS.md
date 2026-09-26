# MIMIC respiratory modulation pilot

**Actual execution: local. No mortality or troponin outcome analysis.**

The fixed pilot included 11 existing MIMIC HF encounters: five exposed to the original charted SpO2 instability and six controls. 0/5 first-episode blocks were usable; 0/5 met the corrected phase-coherence and Pleth screen. Advancement gate: **FAIL**.

| Episode | Status | Coherence | Pleth ratio | Raw p | Holm p | Positive |
|---|---|---:|---:|---:|---:|---|
| E1 | spo2_observed_below_90pct | — | — | 1.0000 | 1.0000 | False |
| E2 | spo2_gap_exceeds_limit | — | — | 1.0000 | 1.0000 | False |
| E3 | spo2_observed_below_90pct | — | — | 1.0000 | 1.0000 | False |
| E4 | spo2_observed_below_90pct | — | — | 1.0000 | 1.0000 | False |
| E5 | spo2_gap_exceeds_limit | — | — | 1.0000 | 1.0000 | False |

Missing or uninformative episodes occupy p=1 placeholders in five-slot Holm correction. Episode labels identify analysis slots without patient identifiers. The gate requires at least four usable exposed episodes and at least three positive exposed patients.

| Group | Encounters | Usable blocks / 20-minute blocks | Encounters with usable blocks |
|---|---:|---:|---:|
| Control | 6 | 33 / 72 | 6 |
| Exposed | 5 | 18 / 60 | 5 |

The frequency band, first-episode selection, quality checks, planned 499 phase-null replicates per estimable episode and gate were locked before amplitudes were inspected. Actual episode coherence tests computed: 0. A failed completeness screen is insufficient evidence, not evidence that respiratory coupling is absent. Respiratory impedance amplitude is not calibrated ventilation; the Pleth screen does not exclude shared motion. Phase coherence does not establish causality, central apnea, ischemia, or mortality benefit. HF periodic breathing and oxygen saturation oscillations have substantial prior literature. This pilot does not establish a novel biological discovery.

All downloaded source files passed the official MIMIC waveform release checksums. Numeric time uses header counter frequency; waveform time uses frame sampling frequency. Patient arrays and detailed source membership remain in private Drive Data.
