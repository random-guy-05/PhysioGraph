# Post-fit numerical validity correction

The initial local frozen-model run completed its association calculations on
2026-09-05, but the independent validation cell failed before the final report
was produced. A separately fitted GLM disagreed with GEE by approximately two
log-odds units on the eICU mortality missing-HR indicator. Inspection also
identified a very large negative missing-MAP coefficient in the MIMIC troponin
model. Both optimizers had reported convergence despite a likelihood boundary.
The initial primary interaction estimates were non-significant; outcomes have
now been examined. This correction is explicitly post-fit, not preregistration.

Add an explicit complete/quasi-complete separation check before reporting any
GEE inference. For signed design Q=(2y−1)X, use linear programming to test
whether Qb>=0 and sum(Qb)>=1 is feasible, with unrestricted coefficients.
A feasible direction indicates a likelihood recession direction. Mark that
model non-estimable, preserve the provisional run, and assign p=1 in its
unchanged four-test multiplicity slot. Never drop rare missingness indicators,
replace them with zero, switch to penalized outcome regression, or change
eligibility to rescue the fit. Solver failure also fails numerical validation.

This adds a necessary validity check to the frozen model; it changes no
biological hypothesis, covariate, exposure, outcome, time window, or primary
contrast. Repeat the same model code for unaffected fits and independently
reconstruct their patient-clustered covariance. Report unsupported estimates
as unavailable and retain the full original output under an explicitly
provisional archive. There is no biological discovery from this correction.
