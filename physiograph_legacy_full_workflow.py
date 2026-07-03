"""Legacy full-workflow representation for PhysioGraph.

This module preserves a script-first representation of the old
``PhysioGraph_Final_Full.ipynb`` workflow in a Colab-friendly, reproducible
form. It keeps the canonical clean notebook untouched and exposes a separate
runner for:

1) Comparator model training + external validation
2) Lactate-in-context analysis tables
3) Subgroup SpO2 inferential models (OR/CI/p-values)
4) Legacy GNN stage representation audit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physiograph_colab_core import (
    CONTROL_FEATURE_CANDIDATES,
    run_physiograph_colab,
)


LEGACY_NOTEBOOK_PATH = (
    "_archive/2026-05-08_legacy_cleanup/notebooks/PhysioGraph_Final_Full.ipynb"
)


def _ensure_project_imports(project_root: Path) -> None:
    for candidate in (project_root, project_root / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _prepare_design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    x = frame[columns].copy()
    categorical = {
        col
        for col in columns
        if col in {"dataset", "race", "ethnicity", "race_ethnicity"}
        or str(x[col].dtype) in {"object", "category", "string"}
    }
    for col in columns:
        if col in categorical:
            x[col] = x[col].fillna("missing").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            median = float(x[col].median()) if x[col].notna().any() else 0.0
            x[col] = x[col].fillna(median)
    x = pd.get_dummies(x, columns=sorted(categorical), dummy_na=False, drop_first=False, dtype=float)
    if not x.empty:
        nunique = x.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            x = x.drop(columns=constant_cols)
    return x


def build_lactate_context_tables(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_cols = [col for col in ("baseline_lactate_focus_bin", "scai_stage_collapsed") if col in analysis_df]
    outcome_cols = [
        col
        for col in (
            "target",
            "mcs_24h_flag",
            "mortality_24h_flag",
            "death_168h_flag",
            "mcs_or_death_168h_flag",
        )
        if col in analysis_df
    ]
    signal_cols = [
        col
        for col in (
            "spo2_min",
            "spo2_rmssd",
            "spo2_below_90_fraction",
            "spo2_instability_proxy_score",
            "spo2_sampling_density_per_hr",
        )
        if col in analysis_df
    ]
    if len(context_cols) < 2:
        return (
            pd.DataFrame([{"status": "skipped", "reason": "missing lactate/scai context columns"}]),
            pd.DataFrame([{"status": "skipped", "reason": "missing lactate/scai context columns"}]),
        )

    group_cols = context_cols
    rates_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    grouped = analysis_df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        base["n"] = int(len(group))
        for outcome in outcome_cols:
            y = pd.to_numeric(group[outcome], errors="coerce")
            base[f"{outcome}_event_rate"] = float(y.mean()) if y.notna().any() else math.nan
        rates_rows.append(base.copy())

        signal_row = base.copy()
        for signal in signal_cols:
            s = pd.to_numeric(group[signal], errors="coerce")
            signal_row[f"{signal}_mean"] = float(s.mean()) if s.notna().any() else math.nan
            signal_row[f"{signal}_median"] = float(s.median()) if s.notna().any() else math.nan
        signal_rows.append(signal_row)
    return pd.DataFrame(rates_rows), pd.DataFrame(signal_rows)


def fit_subgroup_spo2_models(analysis_df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame([{"status": "skipped", "reason": f"statsmodels unavailable: {exc}"}])

    feature_cols = [
        col
        for col in (
            "spo2_below_90_fraction",
            "spo2_rmssd",
            "spo2_instability_proxy_score",
            "spo2_sampling_density_per_hr",
            "spo2_min",
        )
        if col in analysis_df.columns
    ]
    outcome_cols = [col for col in ("target", "death_168h_flag", "mcs_or_death_168h_flag") if col in analysis_df]
    controls = [col for col in CONTROL_FEATURE_CANDIDATES if col in analysis_df.columns]
    if not feature_cols or not outcome_cols:
        return pd.DataFrame([{"status": "skipped", "reason": "required columns unavailable"}])

    masks: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=analysis_df.index))]
    if "dataset" in analysis_df.columns:
        for ds in sorted(analysis_df["dataset"].dropna().astype(str).unique()):
            masks.append((f"dataset={ds}", analysis_df["dataset"].astype(str).eq(ds)))
    if "baseline_lactate" in analysis_df.columns:
        lact = pd.to_numeric(analysis_df["baseline_lactate"], errors="coerce")
        masks.append(("baseline_lactate<2", lact < 2))
        masks.append(("baseline_lactate>=2", lact >= 2))
    if "resp_support_any_flag" in analysis_df.columns:
        rsp = pd.to_numeric(analysis_df["resp_support_any_flag"], errors="coerce")
        masks.append(("resp_support=0", rsp.eq(0)))
        masks.append(("resp_support=1", rsp.eq(1)))

    rows: list[dict[str, Any]] = []
    for subgroup, mask in masks:
        subgroup_df = analysis_df.loc[mask.fillna(False)].copy()
        for outcome in outcome_cols:
            for feature in feature_cols:
                frame = subgroup_df[[outcome, feature, *controls]].copy()
                frame[outcome] = pd.to_numeric(frame[outcome], errors="coerce")
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
                frame = frame.loc[frame[outcome].notna() & frame[feature].notna()].copy()
                n = int(len(frame))
                events = int(frame[outcome].sum()) if n else 0
                if n < 300 or events < 20 or frame[outcome].nunique() < 2:
                    rows.append(
                        {
                            "subgroup": subgroup,
                            "outcome": outcome,
                            "feature": feature,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "insufficient rows/events",
                        }
                    )
                    continue
                sd = float(frame[feature].std(ddof=0))
                if not np.isfinite(sd) or sd <= 0:
                    rows.append(
                        {
                            "subgroup": subgroup,
                            "outcome": outcome,
                            "feature": feature,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "feature variance is zero",
                        }
                    )
                    continue
                z_feature = f"{feature}_z"
                mean = float(frame[feature].mean())
                frame[z_feature] = (frame[feature] - mean) / sd
                x = _prepare_design_matrix(frame, [z_feature, *controls])
                if z_feature not in x.columns or x.empty:
                    rows.append(
                        {
                            "subgroup": subgroup,
                            "outcome": outcome,
                            "feature": feature,
                            "status": "skipped",
                            "n": n,
                            "events": events,
                            "reason": "design matrix unavailable",
                        }
                    )
                    continue
                y = frame[outcome].astype(int)
                try:
                    fit = sm.Logit(y, sm.add_constant(x, has_constant="add")).fit(
                        disp=0, method="lbfgs", maxiter=500
                    )
                    coef = float(fit.params[z_feature])
                    se = float(fit.bse[z_feature])
                    p = float(fit.pvalues[z_feature])
                except Exception as exc:
                    rows.append(
                        {
                            "subgroup": subgroup,
                            "outcome": outcome,
                            "feature": feature,
                            "status": "failed",
                            "n": n,
                            "events": events,
                            "reason": str(exc),
                        }
                    )
                    continue
                rows.append(
                    {
                        "subgroup": subgroup,
                        "outcome": outcome,
                        "feature": feature,
                        "status": "fit",
                        "n": n,
                        "events": events,
                        "or_per_1sd": float(np.exp(coef)),
                        "ci95_low": float(np.exp(coef - 1.96 * se)),
                        "ci95_high": float(np.exp(coef + 1.96 * se)),
                        "p_value": p,
                        "mean_feature": mean,
                        "sd_feature": sd,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["subgroup", "outcome", "p_value", "feature"], na_position="last"
    )


def summarize_external_validation(output_root: Path) -> pd.DataFrame:
    metrics_path = output_root / "locked_comparator_validation" / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame([{"status": "skipped", "reason": f"missing: {metrics_path}"}])
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "split" not in metrics.columns or "model" not in metrics.columns:
        return pd.DataFrame([{"status": "skipped", "reason": "invalid comparator metrics schema"}])

    use_cols = [col for col in ("model", "split", "auroc", "auprc", "brier", "n", "positive_rate") if col in metrics]
    subset = metrics[use_cols].copy()
    internal = subset.loc[subset["split"].eq("internal_validation")].copy()
    external = subset.loc[subset["split"].eq("external")].copy()
    if internal.empty or external.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "required internal/external splits unavailable"}])

    merged = internal.merge(external, on="model", how="inner", suffixes=("_internal_val", "_external"))
    for metric in ("auroc", "auprc", "brier"):
        i = f"{metric}_internal_val"
        e = f"{metric}_external"
        if i in merged.columns and e in merged.columns:
            merged[f"{metric}_gap_external_minus_internal_val"] = merged[e] - merged[i]
    return merged.sort_values("model").reset_index(drop=True)


def build_legacy_gnn_representation(project_root: Path, output_root: Path) -> pd.DataFrame:
    archived_notebook = project_root / LEGACY_NOTEBOOK_PATH
    candidate_artifacts = [
        "processed_data/hf_shock/gnn_artifacts/gnn_tensors.npz",
        "processed_data/hf_shock/gnn_only_weights.pth",
        "processed_data/hf_shock/gru_baseline_weights.pth",
        "processed_data/hf_shock/physiograph_weights.pth",
    ]
    rows: list[dict[str, Any]] = [
        {
            "item": "archived_full_notebook",
            "path": str(archived_notebook),
            "exists": bool(archived_notebook.exists()),
            "note": "Legacy source for full GNN/evaluation notebook sections.",
        }
    ]
    for rel in candidate_artifacts:
        path = project_root / rel
        rows.append(
            {
                "item": "legacy_gnn_artifact",
                "path": rel,
                "exists": bool(path.exists()),
                "note": "Artifact-presence audit only; clean workflow does not require this.",
            }
        )
    rows.append(
        {
            "item": "representation_script",
            "path": str(output_root / "legacy_full_workflow_manifest.json"),
            "exists": True,
            "note": "This script is the .py representation of the old final-full workflow surface.",
        }
    )
    return pd.DataFrame(rows)


def run_legacy_full_workflow(
    *,
    project_root: str | Path,
    build_new: bool = False,
    mimic_root: str | Path | None = None,
    eicu_root: str | Path | None = None,
    output_root: str | Path | None = None,
    run_comparator: bool = True,
) -> dict[str, Any]:
    """Run legacy-full representation workflow and export analysis outputs."""
    project_root = Path(project_root).expanduser().resolve()
    _ensure_project_imports(project_root)
    output_root = Path(output_root) if output_root is not None else project_root / "physiograph_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    legacy_output = output_root / "legacy_full_workflow"
    legacy_output.mkdir(parents=True, exist_ok=True)

    run_status = run_physiograph_colab(
        project_root=project_root,
        build_new=build_new,
        mimic_root=mimic_root,
        eicu_root=eicu_root,
        output_root=output_root,
        run_comparator=run_comparator,
    )

    analysis_path = output_root / "spo2_drilldown" / "spo2_analysis_frame.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"SpO2 analysis frame not found: {analysis_path}")
    analysis_df = pd.read_csv(analysis_path, low_memory=False)

    lactate_outcomes, lactate_signal = build_lactate_context_tables(analysis_df)
    subgroup_results = fit_subgroup_spo2_models(analysis_df)
    comparator_summary = summarize_external_validation(output_root)
    gnn_representation = build_legacy_gnn_representation(project_root, legacy_output)

    output_paths = {
        "lactate_context_outcome_rates": legacy_output / "lactate_context_outcome_rates.csv",
        "lactate_context_signal_summary": legacy_output / "lactate_context_signal_summary.csv",
        "subgroup_spo2_or_pvalues": legacy_output / "subgroup_spo2_or_pvalues.csv",
        "comparator_external_summary": legacy_output / "comparator_external_summary.csv",
        "gnn_stage_representation": legacy_output / "gnn_stage_representation.csv",
        "manifest": legacy_output / "legacy_full_workflow_manifest.json",
    }
    lactate_outcomes.to_csv(output_paths["lactate_context_outcome_rates"], index=False)
    lactate_signal.to_csv(output_paths["lactate_context_signal_summary"], index=False)
    subgroup_results.to_csv(output_paths["subgroup_spo2_or_pvalues"], index=False)
    comparator_summary.to_csv(output_paths["comparator_external_summary"], index=False)
    gnn_representation.to_csv(output_paths["gnn_stage_representation"], index=False)

    manifest = {
        "analysis": "Legacy final-full representation workflow",
        "fresh_colab_execution": False,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "legacy_output_dir": str(legacy_output),
        "sections_represented": [
            "MIMIC/eICU ETL pipeline",
            "Comparator model training + internal/external validation",
            "Lactate-in-context analysis",
            "Subgroup SpO2 inferential analysis",
            "Legacy GNN stage representation audit",
        ],
        "legacy_source_notebook": str(project_root / LEGACY_NOTEBOOK_PATH),
        "outputs": {k: str(v) for k, v in output_paths.items() if k != "manifest"},
        "upstream_run_status_path": str(run_status["status_path"]),
        "notes": [
            "This script is a representation of old final-full workflow scope and reuses package/core logic.",
            "GNN stage is represented as an audited legacy section unless dedicated GNN training artifacts/code are re-enabled.",
            "No Colab execution is claimed by this manifest unless run inside Colab.",
        ],
    }
    _write_json(output_paths["manifest"], manifest)

    return {
        "status": "ok",
        "output_root": output_root,
        "legacy_output_dir": legacy_output,
        "paths": output_paths,
        "manifest": manifest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PhysioGraph legacy full-workflow representation.")
    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--mimic-root", type=str, default=None)
    parser.add_argument("--eicu-root", type=str, default=None)
    parser.add_argument("--build-new", action="store_true")
    parser.add_argument("--no-comparator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = run_legacy_full_workflow(
        project_root=args.project_root,
        build_new=args.build_new,
        mimic_root=args.mimic_root,
        eicu_root=args.eicu_root,
        output_root=args.output_root,
        run_comparator=not args.no_comparator,
    )
    print(json.dumps({"status": results["status"], "manifest": str(results["paths"]["manifest"])}, indent=2))


if __name__ == "__main__":
    main()
