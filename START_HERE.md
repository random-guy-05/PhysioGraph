# PhysioGraph — start here

## Main project folder

The repository root is the central project home. Local material gathered from Downloads and other Drive locations is organized under `consolidated/`; the original analysis paths continue to serve the runnable workflows.

## HF SpO₂, lactate, and mortality study

| What you need | Where to go |
|---|---|
| Study overview and model extensions | [Study README](research/spo2_mimic3_hf_lactate/README.md) |
| Native abstract and presentations | Local links in `consolidated/references/` |
| Runnable notebook | [RUN_DENSE_WAVEFORM_ANALYSIS.ipynb](research/spo2_mimic3_hf_lactate/RUN_DENSE_WAVEFORM_ANALYSIS.ipynb) |
| Complete-follow-up analysis report | [FINAL_HARDENED_REPORT.md](research/spo2_mimic3_hf_lactate/FINAL_HARDENED_REPORT.md) |
| Analysis lock and provenance | [FINALIZATION_ANALYSIS_LOCK.md](research/spo2_mimic3_hf_lactate/FINALIZATION_ANALYSIS_LOCK.md), [source provenance](research/spo2_mimic3_hf_lactate/SOURCE_README.md) |
| CKD-corrected model extension | `research/spo2_mimic3_hf_lactate/ckd_corrected_full_adjusted_models/` |
| Early SpO₂, vasopressor, and ventilation model extension | `research/spo2_mimic3_hf_lactate/support_adjusted_models/` |
| Early VIS model extension | `research/spo2_mimic3_hf_lactate/vis_adjusted_models/` |
| Failure-of-lactate-clearance analysis (separate endpoint) | [Endpoint addendum](research/spo2_mimic3_hf_lactate/CLEARANCE_ENDPOINT_ADDENDUM.md), [report](research/spo2_mimic3_hf_lactate/lactate_clearance/REPORT.md) |
| Hour-16 SCAI/VIS summaries and graphs | `research/spo2_mimic3_hf_lactate/scai_hour16/` |

The saved analysis was executed locally. The notebook is the reproduction entry point; saved outputs do not establish a fresh Google Colab execution. Model-specific reports retain their own adjustment sets and provenance.

## Consolidated material

| Folder | Contents |
|---|---|
| `consolidated/archives/` | Six older full-run folders and their original backup manifest |
| `consolidated/analysis_snapshots/` | Downloads analysis and age/comorbidity extension folders, including their cached artifacts |
| `consolidated/notebooks/` | Additional notebooks from Downloads and other Drive locations |
| `consolidated/manuscripts/` | Abstract document export, manuscript exports, and notebook PDFs |
| `consolidated/figures/` | Downloaded and Drive poster/notebook figures, separated by origin |
| `consolidated/data/` | Downloaded MIMIC source archive, supplementary clinical tables, and waveform artifacts |
| `consolidated/supporting_code/` | Downloaded standalone pipeline code |
| `consolidated/logs/` | Run log found alongside the main project |
| `consolidated/references/` | Links to native Google Docs/Slides |
| `consolidated/archived_originals/` | Original notebooks, logs, and figure folders relocated from other Drive locations |

The local `docs/CONSOLIDATION.md` record lists the transfer scope and verification. Imported files retain their source filenames and folder structure. Historical snapshots have not been merged into the active study or scientifically revalidated. The imported files and local transfer records are excluded from Git and are not included in a GitHub checkout.

The local `docs/CLEANUP_2026-10-01.md` record documents the removal of duplicate Downloads items and the relocation of original Drive figure folders into the main archive. Those folders include the seventeen previously unavailable figures; full local download availability remains unverified. Native Google Docs/Slides are linked from the central folder and retain their original locations.

## Broader project

| Area | Purpose |
|---|---|
| `research/` | Study-specific reports, notebooks, and saved results |
| `docs/` | Protocols, amendments, and result narratives |
| `src/physiograph/` | Shared analysis package |
| `scripts/` | Shared execution and refresh tools |
| `configs/` | Framework configuration |
| `tests/` | Existing validation code and ground-truth fixtures |
| `graphify-out/` | Generated code knowledge graph |

[README.md](README.md) documents the broader framework. [PROJECT_GOAL.md](PROJECT_GOAL.md) and [docs/LATEST_RESULTS.md](docs/LATEST_RESULTS.md) contain dated research records across multiple investigations; consult the relevant study directory for its specific results. The other root notebooks serve separate or older workflows and are preserved under their existing names.

Use the [research topic index](research/README.md) and [documentation index](docs/README.md) to navigate those investigations.
