"""Command-line interface for PhysioGraph.

Usage::

    # Run the full ETL pipeline
    physiograph run-pipeline --dataset mimic --config configs/mimic.yaml

    # Validate predictions against ground truth
    physiograph validate --predictions path/to/predictions.csv

    # Generate an audit report
    physiograph audit --output-dir ./audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


# ──────────────────────────────────────────────────────────────────────
# Sub-command handlers
# ──────────────────────────────────────────────────────────────────────


def _cmd_run_pipeline(args: argparse.Namespace) -> None:
    """Execute the ETL pipeline for the requested dataset."""
    cfg = load_config(dataset=args.dataset)

    resolved = cfg.get("resolved_paths", {})
    if args.dataset == "mimic":
        data_root = args.mimic_root or resolved.get("mimic_root")
        if not data_root:
            sys.exit(
                "Error: --mimic-root is required when no resolved path "
                "is available in config.  Pass --mimic-root explicitly."
            )
        eicu_root = None
    else:
        data_root = args.eicu_root or resolved.get("eicu_root")
        if not data_root:
            sys.exit(
                "Error: --eicu-root is required when no resolved path "
                "is available in config.  Pass --eicu-root explicitly."
            )
        eicu_root = data_root
        data_root = None  # mimic_root not needed for eicu

    output_dir = args.output_dir or resolved.get("output_root")

    kwargs: dict = {}
    if args.max_stays is not None:
        kwargs["max_stays"] = args.max_stays
    if args.max_chunks is not None:
        kwargs["max_chunks"] = args.max_chunks
    if args.chunk_size is not None:
        kwargs["chunk_size"] = args.chunk_size

    if args.dataset == "mimic":
        artifacts = run_pipeline(
            dataset="mimic",
            mimic_root=data_root,
            output_dir=output_dir,
            **kwargs,
        )
    else:
        artifacts = run_pipeline(
            dataset="eicu",
            eicu_root=eicu_root,
            output_dir=output_dir,
            **kwargs,
        )

    print("Pipeline completed. Artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate predictions CSV against ground-truth schema."""
    import pandas as pd

    from .schema import validate_schema, FeatureSchema

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        sys.exit(f"Error: predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    try:
        validate_schema(df, FeatureSchema, context="predictions")
        print(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
    except Exception as exc:
        sys.exit(f"Validation failed: {exc}")


def _cmd_audit(args: argparse.Namespace) -> None:
    """Generate an audit report from pipeline outputs."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = output_dir / "audit.json"
    if not audit_path.exists():
        sys.exit(
            f"Error: no audit.json found in {output_dir}. "
            "Run the pipeline first."
        )

    with open(audit_path) as f:
        entries = json.load(f)

    report_path = output_dir / "audit_report.txt"
    with open(report_path, "w") as f:
        f.write(f"PhysioGraph Audit Report\n")
        f.write(f"========================\n\n")
        f.write(f"Source: {audit_path}\n")
        f.write(f"Entries: {len(entries)}\n\n")
        for entry in entries:
            f.write(f"- [{entry.get('step', '?')}] ")
            if 'row_count' in entry:
                f.write(f"rows={entry['row_count']} ")
            if 'stay_count' in entry:
                f.write(f"stays={entry['stay_count']} ")
            if 'details' in entry:
                f.write(f"details={entry['details']} ")
            f.write("\n")

    print(f"Audit report written to {report_path}")


# ──────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physiograph",
        description="PhysioGraph: Graph-based physiological time-series "
        "analysis for clinical prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run-pipeline ──────────────────────────────────────────────────
    run_p = subparsers.add_parser(
        "run-pipeline",
        help="Run the full ETL pipeline for MIMIC-IV or eICU-CRD.",
    )
    run_p.add_argument(
        "--dataset",
        required=True,
        choices=["mimic", "eicu"],
        help="Dataset to process (mimic or eicu).",
    )
    run_p.add_argument(
        "--config",
        default=None,
        help="Path to dataset-specific YAML config (optional; auto-detected).",
    )
    run_p.add_argument(
        "--mimic-root",
        default=None,
        help="Path to MIMIC-IV data directory.",
    )
    run_p.add_argument(
        "--eicu-root",
        default=None,
        help="Path to eICU-CRD data directory.",
    )
    run_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to ./physiograph_outputs/<dataset>).",
    )
    run_p.add_argument(
        "--max-stays",
        type=int,
        default=None,
        help="Cap on cohort size (for testing).",
    )
    run_p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Cap on chunks per streaming table.",
    )
    run_p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Rows per chunk for CSV streaming (default: 250000).",
    )

    # ── validate ───────────────────────────────────────────────────────
    val_p = subparsers.add_parser(
        "validate",
        help="Validate predictions CSV against ground-truth schema.",
    )
    val_p.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions CSV file.",
    )

    # ── audit ──────────────────────────────────────────────────────────
    aud_p = subparsers.add_parser(
        "audit",
        help="Generate an audit report from pipeline outputs.",
    )
    aud_p.add_argument(
        "--output-dir",
        default="./physiograph_outputs",
        help="Directory containing pipeline outputs (default: ./physiograph_outputs).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "run-pipeline": _cmd_run_pipeline,
        "validate": _cmd_validate,
        "audit": _cmd_audit,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)