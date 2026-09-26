# Experiment 2: cross-etiology macrophage failure screen

**Cancelled before expression analysis by explicit user scope correction:
MIMIC/eICU only, anchored to the current PhysioGraph project.** This document
is an abandoned proposal, not an active protocol or an executed experiment.

Frozen 2026-09-05, after metadata inspection and before expression comparisons.
This follows an inconclusive clinical phosphate/lactate experiment; it is not
a retrospective change to that experiment's endpoints or support thresholds.

## Question and scope

Which macrophage transcripts change consistently in failing HLHS and dilated
cardiomyopathy, compared with donor hearts, after treating the person as the
experimental unit? The objective is to find credible molecular candidates,
then test their novelty and causal relevance. This is an exploratory genome-wide
screen, not a prespecified validation of a selected gene or mortality model.

Use the GSE203274 immune raw counts and metadata in My Drive/Data/HLHS Data.
If Drive cannot hydrate a file within a bounded read, retrieve the corresponding
file from the official NCBI supplementary archive, keep a separate copy under
the private analysis directory, and record its URL and SHA256. Do not claim
complete byte identity with an unreadable Drive original.

The Garcia files represent a separate potential comparison. Their workbook
lists eight failing single-ventricle subjects, five pre-Fontan subjects and
five post-Fontan subjects. Expression data, tissue identity, donor matching,
assay scale, and independence must be verified before analyzing them. The
workbook alone is not an expression experiment or evidence of TCR sequences.

## Locked analysis

- Check the R-exported metadata format explicitly: an unnamed row index makes
  each data row one column wider than the header. Restore the cell-ID column;
  assert widths, unique cell IDs and exact count/metadata barcode matching.
- Select the author's MainCellType = Mac; aggregate integer raw counts by
  patientID, merging technical libraries and chambers from the same person.
- Require at least 50 macrophages per donor. Report chamber-specific sensitivity
  separately; matched chambers from one person are never independent controls.
- Primary contrast: HF_HLHS and DCM versus Donor. Neo_HLHS, TOF and HCM are
  excluded from the primary failure contrast. Diagnose support by unique people.
- Retain genes with at least 20 total counts and CPM >= 1 in at least three
  primary donors. Normalize each donor by total macrophage library size;
  use log2(CPM + 0.5). This is a simple exploratory screen; it is not DESeq2.
- Compute a two-sided exact donor-label permutation test for the difference of
  mean logCPM, enumerating all allocations when feasible. Correct the entire
  retained gene family with Benjamini-Hochberg FDR. Report the permutation
  resolution, including its implications for claims with very few donors.
- Candidate gates: FDR < 0.05, absolute log2 expression difference >= 1,
  concordant mean effects in both HF etiologies, and stable direction after
  deleting each donor in turn. Report ALL tested genes, not just favorable ones.
- Report age/sex and chamber distributions. Require the direction to survive
  a linear age/sex-adjusted diagnostic model, but do not interpret a model
  with near-saturated degrees of freedom as adequate confounding control.
- For robustness, repeat the main contrast using only LV macrophages; require
  adequate cell support, and report when this drops the HLHS component.

## Interpretation and next gates

The published study already described CHD immune deficiency, macrophage
states, insulin resistance, FOXO/CRIM1 and YAP biology. A 2024 reanalysis
already described HLHS macrophage GAS6/AXL and FOXO3/ELF2 findings. Neither
rediscovery nor a low unadjusted p value is a novel mechanism.

No progression to a biological discovery claim without independent biological
replication, donor-confounding assessment, orthogonal protein/function evidence,
and targeted review of prior publications. These tissue snapshots do not
establish survival benefit, reversibility, a treatment target, or prospective
HF mortality prediction. A paradigm-shifting goal remains incomplete even
if the exploratory screen yields statistically supported candidates.

Primary sources:
- https://www.nature.com/articles/s41586-022-04989-3
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203274
- https://www.sciencedirect.com/org/science/article/pii/S1747079X2400008X
