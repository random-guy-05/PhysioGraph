# PhysioGraph Objective-Shock Single-Signal Discovery Protocol

Frozen UTC: `2026-09-07T18:04:48Z`

## Status and purpose

This is a finite, hypothesis-generating MIMIC discovery screen. The prior
masked-pulsatility hypothesis is closed as `NO_SIGNAL`; PrPP thresholds,
alternative BP sources, narrower pressor classes, and masked-pulsatility
subgroups are prohibited here. eICU must not be inspected for any candidate
association unless exactly one MIMIC signal passes every gate below and is
subsequently frozen before a single external test.

The objective is to identify, or fail to identify, one routinely measured
physiological trajectory that precedes objective progression from initially
non-shocked heart-failure ICU care to overt circulatory shock/hypoperfusion.

## Development population and chronology

- Database: MIMIC-IV only during discovery.
- Population: the validated adult, first-ICU-stay-per-admission heart-failure
  cohort used by PhysioGraph.
- Fixed feature anchor: ICU hour 4.
- Candidate measurements: ICU hours 0 through 4 only.
- Blanking/landmark interval: strictly after hour 4 through hour 6.
- Primary outcome window: strictly after ICU hour 6 through hour 16, equivalent
  to >2 through <=12 hours after the feature anchor.
- Several-hour-lead sensitivity: strictly after ICU hour 8 through hour 16.
- At-risk requirement at hour 6: alive, still observed, and without prior
  qualifying pressure/support abnormality, hypoperfusion abnormality, overt
  composite endpoint, continuous vasoactive/inotropic support, or MCS.
- No post-hour-4 measurement may enter a candidate feature or adjustment
  covariate.

## Objective progression endpoint

The primary endpoint is the first time that both of these distinct domains are
newly satisfied within six hours of each other during the outcome window. The
endpoint time is the later domain-completion time.

### Domain 1: pressure or hemodynamic support

At least one of:

1. sustained routine-cuff hypotension: two consecutive hourly bins with
   median SBP <90 mmHg or direct MAP <65 mmHg;
2. new actually administered continuous vasoactive/inotropic infusion with a
   positive rate, using the existing harmonized seven-drug definition;
3. new MCS initiation using the existing validated IABP, Impella, or ECMO
   procedure definition.

### Domain 2: objective hypoperfusion or acute organ injury

At least one of:

1. lactate >=4 mmol/L, or lactate >=2 mmol/L with an increase >=0.5 mmol/L
   from the latest valid pre-anchor lactate;
2. urine output <30 mL/h averaged across a fully post-anchor rolling six-hour
   window with at least four documented hourly bins;
3. creatinine increase >=0.3 mg/dL or >=1.5-fold from the latest valid
   pre-anchor creatinine;
4. ALT >200 U/L when no pre-anchor ALT already exceeded 200 U/L, or a >=3-fold
   rise from the latest pre-anchor ALT;
5. arterial or venous pH <7.20 when no pre-anchor pH was already <7.20.

Treatment alone, isolated hypotension, or isolated laboratory abnormality is
not the endpoint. Measurements must be available by the analysis time; MIMIC
laboratory `storetime` cannot exceed the event time used.

## Anti-circularity endpoint

Every screened candidate uses the full composite for discovery. The
provisional winner must also pass a prespecified leave-domain-out analysis:

- SBP, DBP, MAP, or pulse-pressure candidates: Domain 1 must be completed by
  vasoactive/inotropic support or MCS, not hypotension alone.
- Urine-output candidates: Domain 2 must be completed by lactate, creatinine,
  ALT, or pH, not urine output.
- Other candidates use the full composite unchanged.

Failure of this analysis prevents winner declaration and eICU access.

## Candidate signals

Exactly nine routine signals are screened:

1. heart rate;
2. routine-cuff SBP;
3. routine-cuff DBP;
4. direct routine-cuff MAP;
5. pulse pressure, SBP minus DBP from the same hourly bin;
6. respiratory rate;
7. SpO2;
8. temperature in degrees Celsius;
9. urine output in mL/h.

PrPP and other ratios are excluded. For each signal, exactly four features are
computed, yielding 36 primary tests:

- `level`: latest hourly value through hour 4;
- `slope`: ordinary least-squares slope across hours 0 through 4;
- `change`: latest minus earliest hourly value;
- `variability`: root-mean-square residual around the fitted linear trend.

Level requires one valid hourly bin. Slope, change, and variability require at
least three valid hourly bins spanning at least two hours. Hourly vitals use
medians; hourly urine output uses the sum of valid nonnegative output volumes.
Missing urine documentation is not zero output. Temperature is harmonized to
Celsius before aggregation. Physiological plausibility bounds are frozen in
the executable protocol.

## Primary adjustment and models

Each feature is winsorized at its MIMIC 1st/99th percentiles and standardized.
The primary model is modified Poisson regression with log link and HC0 robust
variance. Effect sizes are per one standard deviation of the candidate.

The fixed pre-anchor adjustment set is age, sex, latest HR, routine-cuff SBP,
direct routine-cuff MAP, respiratory rate, SpO2, temperature, creatinine,
lactate, baseline-lactate observation, BP measurement density, and urine-output
measurement density. Continuous adjustment covariates are winsorized,
median-imputed, standardized, and paired with missingness indicators. When the
candidate is the same level variable as an adjustment covariate, that exact
level is omitted from adjustment to prevent algebraic duplication; all other
covariates remain fixed. No outcome-driven variable selection is allowed.

Benjamini-Hochberg correction is applied across all 36 primary tests and,
separately, all 36 several-hour-lead tests. No uncorrected p-value can advance a
candidate.

## Internal stability

For every candidate that passes the initial coverage, event-support, effect,
and multiplicity gates, perform 200 outcome-stratified stay-level bootstrap
replicates with the identical model. Report bootstrap percentile intervals,
median RR, sign consistency, and successful-fit fraction. No alternative
resampling scheme may be substituted after results are seen.

## Winner gates

Exactly one provisional MIMIC winner may advance. A candidate must satisfy all
of:

1. observed coverage >=70% of the at-risk cohort;
2. >=100 endpoint events and >=50 observed endpoint events in its model frame;
3. primary BH q <0.05 and robust 95% CI excluding 1;
4. adverse-direction magnitude `max(RR, 1/RR) >=1.30` per SD;
5. several-hour-lead BH q <0.05, CI excluding 1, same coefficient direction,
   and `max(RR, 1/RR) >=1.20`;
6. bootstrap successful-fit fraction >=95%, sign consistency >=90%, and the
   bootstrap percentile interval excludes 1;
7. leave-domain-out direction is unchanged, CI excludes 1, and
   `max(RR, 1/RR) >=1.20`;
8. no extraction, availability-time, or endpoint-integrity failure.

Eligible candidates are ranked by lower primary BH q, then larger absolute
log-RR, then higher coverage, using exact deterministic ordering. Only the
top-ranked candidate receives a focused literature-novelty review. If it is
already an established warning marker, the experiment ends as
`KNOWN_SIGNAL_NO_NOVEL_WINNER`; no runner-up is substituted. If no candidate
passes, the result is `NO_DISCOVERY_SIGNAL` and eICU remains unopened.

## External confirmation rule

Only a provisional MIMIC winner that passes the statistical, stability,
anti-circularity, integrity, and novelty gates may be frozen with its sign,
transform, model, endpoint, lead window, and claim language. The complete eICU
implementation and hashes must then be frozen before one eICU association is
read. No second candidate and no threshold repair is allowed after eICU.

## Interpretation limits

This is observational discovery. It does not adjudicate cardiogenic shock,
establish causality, prove treatment benefit, or justify clinical action. A
negative screen does not prove that no physiological precursor exists; it
means none of these 36 frozen routine candidates met the advancement gates.

Endpoint concepts were grounded before analysis in the 2025 ACC cardiogenic
shock guidance and the 2022 SCAI shock-stage update, which use sustained
hypotension/support plus urine-output and biochemical hypoperfusion domains.

