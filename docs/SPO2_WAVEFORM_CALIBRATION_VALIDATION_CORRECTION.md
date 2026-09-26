# Independent waveform calibration check correction

The first local pilot run completed its frozen episode screen, but its independent
FLAC validation stopped on a Pleth calibration difference. The initial run and
notebook are preserved in `research/spo2_waveform_pilot/provisional_validation_run`.
No episode coherence test was estimable: all five first-episode blocks failed
the monitor-SpO2 completeness criteria before spectral testing.

The diagnostic found a maximum difference of 0.0001220703125 in Pleth units,
exactly half a digital count at gain 4096. The independent check had averaged
two calibrated physical samples per frame. Inspection of the installed WFDB
4.3.0 `rdrecord` and `SignalMixin.smooth_frames` implementations established that
the selected `smooth_frames=True` path averages digital samples into an integer
array before digital-to-physical calibration. Reconstructing that integer
truncation matched the diagnosed samples exactly. This is a correction to the
independent checker, not to the extracted waveform or biological protocol.

The revised checker independently reads the FLAC integer samples, reconstructs
frame averaging with integer truncation, and then applies the plain-header
gain and baseline. It checks all downloaded, overlapping Resp/Pleth frames,
not only a short sample. It also explicitly rejects partial missing-sentinel
frames, because averaging a missing sentinel with a valid subframe could hide
a gap. Any such failure requires investigation; it must not be called a pass.
Numeric-clock validation remains unchanged. No thresholds, episode choices,
frequency bands, or advancement gates change. No new result is claimed from
this implementation correction.
