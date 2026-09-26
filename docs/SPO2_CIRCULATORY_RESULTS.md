# Circulatory-context experiment: no established biological finding

Actual local computation completed on 2026-09-05 UTC using only the supplied
MIMIC and eICU files. **This candidate did not meet the frozen evidence gates.
No primary association model was fitted. The discovery goal remains open.**

The experiment asked whether SpO2 changes accompanied by a MAP fall on the
same transition distinguish subsequent myocardial injury and death. The
primary timing comparison includes only encounters experiencing both
component abnormalities: synchronous versus asynchronous occurrence.
Thresholds remained an absolute SpO2 change of at least four percentage
points and a MAP decline of at least 10 mmHg, using 15-minute bins during
ICU minutes [0,240).

## Actual primary comparison counts

| Database | Endpoint | Synchronous: events / encounters | Asynchronous: events / encounters | Total events | Frozen count gate |
|---|---|---:|---:|---:|---|
| eICU | Hospital death | 24 / 76 | 33 / 108 | 57 | Failed |
| MIMIC | Hospital death | 47 / 223 | 46 / 254 | 93 | Failed |
| eICU | Troponin rise within 12 h after landmark | 14 / 19 | 15 / 24 | 29 | Failed |
| MIMIC | Troponin rise within 12 h after landmark | 17 / 36 | 24 / 47 | 41 | Failed |

The gate required at least 100 events in the timing comparison and at least
20 events and 20 non-events in each group. A further model-complexity gate
was specified, but the corrected primary analyses all failed the earlier
count gate. No p-values, fitted odds ratios, causal effects, or mortality
benefits are claimed. Insufficient evidence is not proof that an effect is
absent.

Troponin rise means a ratio of at least 1.5 to a positive baseline result of
the same assay and unit. The primary subsequent window is ICU minutes
(240,960]. Incompletely observed negatives are not classified as event-free.
Troponin elevation is myocardial injury, not adjudicated myocardial
infarction. Laboratory ordering and follow-up selection remain limitations.

## Coverage and sensitivity

The reconstructed HF cohorts contain 12,028 eICU and 17,758 MIMIC encounters,
matching the original cohort counts. Adequate joint SpO2/MAP observations
exist in 1,022 and 2,846 encounters, respectively. Among these, both component
abnormalities occur in 185 eICU and 477 MIMIC encounters. One eICU encounter
in that comparison has unknown hospital mortality.

Restricting MIMIC to arterial MAP leaves 105 encounters in the timing
comparison and 18 deaths. This also fails the event gate. eICU's supplied
periodic pressure source is arterial; the supplied files do not contain its
noninvasive-pressure table. Therefore these data do not establish a broadly
applicable HF phenotype or a robust result across pressure sources.

The 24-hour troponin sensitivity and the prespecified normal-lactate and
preserved-saturation subgroups did not supply a qualified primary finding.
Missing lactate was never classified as normal. No threshold, endpoint, or
direction was selected to rescue significance.

## Reproduction and corrections

The primary runnable artifact is **PhysioGraph_Biological_Discovery.ipynb**.
It contains only the current SpO2 follow-up; the superseded phosphate
notebook is retained under research/biological_discovery as audit history.
The computations reported here were local extraction and corrected replay,
not Google Colab execution. A fresh start-to-finish Colab execution has not
been performed.

Three implementation differences were corrected: the earlier of ICU and
hospital discharge defines follow-up; the dynamic SpO2 window excludes the
exact four-hour timestamp; and MIMIC measurements with warning = 1 are
excluded as in the production extractor. The warning-filter omission was
identified after provisional count tables had been seen. No association
model had been fitted. Provisional outputs remain explicitly archived.

The rebuilt exposed-patient sets match the production `_first_instability`
helper applied to the current raw replay. Eligible-patient sets also match
the production transition helper. The current MIMIC dynamics count is 4,711.
An actual serialization check on the same 95,997 SpO2 records reproduced the
historical 4,708 after a default pandas CSV write/read round trip. Reading
with `float_precision="round_trip"` preserved all timestamps and the 4,711
count. This demonstrates sensitivity to floating-point timestamp parsing;
it does not establish that every historical cached value was identical.
The analysis retained the direct raw replay. eICU's dynamics count matches
the historical 11,452. An independent SQL reconstruction also verified the
new transition features.

Raw-source SHA-256 fingerprints, the locally frozen analysis specification,
software versions, all count gates, and correction history are recorded in
research/spo2_circulatory_context. The analysis specification is a local
prospective lock for this extension, not an externally registered protocol.

## Research decision

Do not advance this candidate as a biological discovery or a mortality
intervention. The broad combination of low saturation and low blood pressure
already has [HF prognostic precedent](https://pubmed.ncbi.nlm.nih.gov/19372679/).
The narrower timing hypothesis tested here remains unsupported by the
prespecified analysis. A new name, a relaxed event gate, or a selected
subgroup would not meet the requested standard.
