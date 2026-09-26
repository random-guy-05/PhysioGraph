# Myocardial-injury ordering: cross-database support is insufficient

The local feasibility experiment completed on 2026-09-05 UTC. It used only
MIMIC and eICU and the existing PhysioGraph first-SpO2-episode definition.
**The cross-database support gate failed. Troponin trajectory directions,
ratios and mortality associations were not inspected. No biological finding
or mortality benefit has been established.**

The question was whether myocardial injury was already progressing before
the first SpO2 episode. Answering it requires serial measurements preceding
the episode; one baseline value cannot establish a stable prior trajectory.

| Database | SpO2 episodes | Any pre-episode assay | Serial pre-episode trajectory | Complete pre/post trajectory |
|---|---:|---:|---:|---:|
| eICU | 4,309 | 1,326 | 224 | 133 |
| MIMIC | 1,398 | 288 | 40 | 23 |

The frozen gate required at least 100 complete trajectories in **each**
database. The selected assay had to have two positive pre-episode values
at least 60 minutes apart and a delayed post-episode measurement between
two and twelve hours afterward. The preceding window was twelve hours.
Assay selection used only pre-episode timing, with the predefined tie rule.
No MAP requirement was imposed.

An independent Pandas reconstruction matched the SQL-defined complete patient
sets in both databases (133 and 23). This verification used sample times and
assay identifiers only; it did not compare concentrations or join mortality.
The aggregate check is saved in `independent_validation.json`.

## Encounter-boundary correction

The first pass used formal inpatient admission as the MIMIC encounter start
and found 17 complete trajectories. The admissions source then showed 987
episode encounters with earlier ED registration, typically 190 minutes
earlier. The revised pass used the earlier of ED registration and inpatient
admission from the same hospitalization row. This reflects the distinction
in the [MIMIC admissions documentation](https://mimic.mit.edu/docs/iv/modules/hosp/admissions.html).

The correction increased MIMIC's complete trajectories to 23. Laboratory
records still required an explicit matching `hadm_id`; unlinked records were
not assigned by proximity. Windows, sample separation, assay selection and
the support threshold were unchanged. The correction followed feasibility
counts, before concentrations were compared over time. The original pass,
protocol lock and amendment are preserved under
research/spo2_troponin_direction.

## Interpretation

The available paired observations do not support the planned two-source
test of biological ordering. This does not show that injury occurs before
or after SpO2 instability. A later measured troponin rise also cannot, on
its own, locate injury onset because release and clearance affect its time
course. The [human elimination study](https://pubmed.ncbi.nlm.nih.gov/39253802/)
illustrates why observed concentration decay should not be equated with
injury cessation; its findings are prior literature, not a discovery here.

Timing fields were checked against source documentation: eICU describes
`labresultoffset` as draw time and its revised offset as entry of a revised
value; MIMIC describes `charttime` as usually specimen acquisition and
`storetime` as result availability. These fields were kept separate.
[eICU lab documentation](https://eicu.mit.edu/eicutables/lab/),
[MIMIC lab documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html).

The primary notebook contains the complete workflow and saved local outputs.
No Google Colab execution is claimed. Private patient-level data remain in
Data; the repository contains aggregate support, provenance and validation.
The paradigm-shifting discovery goal remains unfulfilled.

## Fresh notebook replay

The complete 14-cell notebook version finished a fresh local run on
2026-09-05 at 06:10 UTC. All eleven code cells executed without error outputs;
schema and source checks passed, and the final circulatory-context tables and
troponin trajectory support exactly matched the archived corrected analyses.
The verified notebook, builder and validation record are preserved under
`research/spo2_troponin_direction/completed_full_notebook_replay/`.
Later hemoglobin feasibility cells extend the primary notebook and have their
own execution record; they are not part of this 14-cell replay claim.

The subsequent documented linkage-only amendment increased MIMIC complete
trajectories from 23 to 76; eICU remained at 133. The frozen two-database gate
still fails, and directions remain unexamined. See
PREICU_LINKAGE_AMENDED_RESULTS.md for the amended counts. The original results
above retain their explicit-ID-only scope.

## Later descriptive departure

The primary gate above remains failed. A separately documented amendment,
written after these feasibility counts but before directional inspection,
subsequently described all 133 eICU / 76 MIMIC complete trajectories without
hypothesis tests or mortality analysis. It explicitly departs from the earlier
stop rule; it is not a successful continuation of the original primary test.
Pre-episode numeric rises occurred in 44 and 32 encounters, respectively.
See `SPO2_TROPONIN_DESCRIPTIVE_AMENDMENT.md` and
`SPO2_TROPONIN_DESCRIPTIVE_RESULTS.md` for all four pre/post patterns,
independent validation, and the limits of first-captured-episode ordering.
