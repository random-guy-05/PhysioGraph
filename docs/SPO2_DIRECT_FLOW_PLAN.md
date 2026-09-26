# Direct cardiac-flow follow-up of the original SpO2 phenotype

## Biological question and limits

Does the original first-four-hour saturation-instability phenotype precede
a subsequent fall in directly charted cardiac output, including when arterial
pressure is maintained? This asks about circulatory deterioration rather than
another concentration-based injury proxy. It is a candidate research question,
not an established new mechanism or a claim that oxygen treatment causes harm.

The direction is biologically uncertain. Prior HF experiments reported
adverse flow responses to oxygen ([Haque et al.](https://pubmed.ncbi.nlm.nih.gov/8557905/)),
whereas a small low-dose oxygen substudy found a small increase in cardiac
output and lower pulmonary resistance ([HOT substudy](https://www.ncbi.nlm.nih.gov/books/NBK316455/)).
These studies motivate measuring flow, not assuming that saturation recovery
means better perfusion. Recovering their broad findings is not a discovery.
Our observational exposure is SpO2 instability, not randomized oxygen delivery.

The prior arterial timing gate addresses two samples around one rapid jump.
It does not answer this distinct subsequent-circulatory-course question.
Earlier negative troponin, HR and ECG results remain negative. This protocol
does not change their gates or replace their endpoints.

## First stage: outcome-blind direct-measurement support

Use the same 4,711 original MIMIC encounters. The eICU supplied periodic
vital file does not contain cardiac output; no eICU surrogate is to be
represented as direct flow replication. This stage is MIMIC-only within
the user's permitted databases.

The supplied d_items dictionary was inspected before this plan. Retrieve
chart metadata for item 220088, Cardiac Output (thermodilution), and 224842,
Cardiac Output (CCO), from the full chartevents source at 0–1,680 minutes.
Keep measurement modalities separate. Do not substitute NICOM, Impella,
cardiac index, SvO2, or an alarm/calibration flag. Do not select valuenum/value,
mortality, or post-exposure injury results in this stage.

Use exact original subject_id, hadm_id and stay_id; charttime determines the
event time. Retain storetime and unit metadata without interpreting values.
Select the lowest original stay_id for each person before examining flow
availability. For each modality, select the last timestamp in [0,240] and
first in (240,1680]. No alternative pair if later quality checks reject it.
Report person/encounter coverage and postmeasurement delays, and require
matching IDs without conflating duplicate rows at one timestamp with repeats.

Before flow counts, assign people to two internal splits by parity of the
first SHA256 byte of `PhysioGraph direct flow v1|` plus the original canonical
person ID. The primary modality is CCO. A minimally viable internal biological
analysis requires at least 100 paired people, including at least 25 exposed
and 25 unexposed, in each split. This is a support floor, not a power calculation
or a definition of a clinically meaningful finding. Report thermodilution
support separately without promoting it when CCO fails. A failed floor is
not proof that a cardiac effect is absent.

Independently reconstruct selected encounters, timestamp pairs and split
counts in Python and SQL. Verify source dictionary labels and raw file hashes.
Keep patient-level data in private Drive Data. Append executed cells to the
single primary notebook and label local execution correctly.

## Required next stage if support exists

Freeze value/unit and duplicate/revision qualification, a clinically justified
effect criterion, covariates, missingness/selection analyses and the independent
validation contrast before reading cardiac-output values. Preserve source
modality and inspect surgical/treatment context; a new catheter or preferential
repeat testing can itself select deteriorating patients. Confirm time-aligned
pressure support before claiming a flow/pressure dissociation. Do not fit a
mortality model merely to search for significance. Internal replication alone
would not establish treatment benefit or the requested paradigm shift.
