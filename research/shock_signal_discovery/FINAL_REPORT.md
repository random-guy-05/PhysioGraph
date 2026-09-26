# Completed objective-shock single-signal discovery experiment

Final decision: **KNOWN_SIGNAL_NO_NOVEL_WINNER**.

The finite MIMIC-only screen completed locally on 2026-09-07. Exactly 36 candidates
(nine signals × four dynamics) were evaluated. One candidate passed the statistical,
lead-window, bootstrap, and anti-circularity gates: lower systolic blood-pressure
level. It failed the prespecified literature-novelty gate. No candidate was substituted
and eICU was not opened for this experiment. The prior masked-pulsatility hypothesis
remains archived.

## Completed results

The corrected at-risk cohort contains 8,196 stays, with 238 primary composite
events (2.90%) and 221 events in the >8–16-hour sensitivity window. The SBP model
includes 7,429 stays and 196 events, giving 90.64% observed coverage.

| SBP level result | RR per 1 SD higher SBP | 95% interval | Additional evidence |
|---|---:|---:|---|
| Adjusted primary model | 0.551 | 0.434–0.699 | BH q = 0.0000160 |
| >8–16-hour outcome window | 0.533 | 0.418–0.681 | BH q = 0.00000406 |
| Bootstrap median | 0.546 | 0.421–0.685 | 200/200 successful; 100% sign consistency |
| Excluding hypotension as endpoint completion | 0.415 | 0.274–0.627 | Requires support/MCS plus hypoperfusion |

One SD was approximately 21.75 mmHg after winsorization. The primary model therefore
corresponds to RR 1.82 per SD lower SBP. This is an adjusted observational association,
not a causal effect or a validated clinical warning algorithm. The fixed temporal
separation is not proof that this signal changes earliest among all possible signals.

No slope, change, or variability feature passed all gates. SpO2 variability had
RR 1.251 and q=0.00539 but failed the required primary magnitude of 1.30.
Temperature variability had only 245 observed stays and six events (2.99% coverage),
so its large estimate could not advance. Urine-output level coverage was 43.73%,
and urine dynamics coverage was 15.93%; those candidates failed the coverage gate.

## Novelty adjudication

The [2022 SCAI consensus](https://www.jacc.org/doi/10.1016/j.jacc.2022.01.018)
already recognizes relative hypotension with preserved perfusion as a beginning
shock state. The [2025 ACC guidance](https://www.jacc.org/doi/10.1016/j.jacc.2025.02.018)
includes low SBP and a substantial fall from baseline among established warning signs.
These sources support a lack of conceptual novelty for the selected marker, rather
than independent validation of this exact model. Under the frozen rule, that ends
the experiment without an external test or runner-up review.

## Execution and provenance

The written protocol froze at 18:04:48 UTC. A failed initial association attempt at
19:32:30 UTC stopped before its first fit because a duplicated level-column merge
removed the expected column name. Two correction records preserve the implementation
history, including timing-boundary, eligibility, availability, multiplicity, and
algebraic-redundancy repairs. Scientific gates and the written protocol were unchanged.

The successful run locked configuration SHA-256
`5041d8085d3f5f7b5221b7fc2aeda194990950291f7a02cb8a7b9ddb9f00037d`
before association access at 20:17:55 UTC, then completed the statistical analysis at
20:19:55 UTC. The notebook contains all seven executed code cells, with no stored
errors. Fifteen focused tests and Ruff checks passed before this run. The novelty
adjudication was appended afterward without changing any locked analysis code.

The primary runnable deliverable is `PhysioGraph_Biological_Discovery.ipynb`, intended
for Google Colab using the configured Drive project and source files. This execution
was **local**, not a fresh Google Colab execution. Re-running the notebook regenerates
the statistical screen; this saved report records the separate literature adjudication.

## Limits

This is a constructed shock/hypoperfusion endpoint in a heart-failure cohort, not
adjudicated cardiogenic shock. Laboratory and urine documentation remain selective;
missing measurements are not evidence of normal physiology. The outcome requires
observed paired domains, and ordinary fixed-window regression does not establish
causality or resolve informative loss of observation. The result supports the
protocol's stop decision, not a universal claim that no useful precursor exists.
