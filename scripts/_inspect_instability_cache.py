#!/usr/bin/env python3
"""Temporary schema/density audit for the cached instability analysis."""

from pathlib import Path
import sys

import pandas as pd

from physiograph.analysis.spo2_instability import build_multiscale_instability_features


root = Path(sys.argv[1]).expanduser().resolve()
frames = []
for dataset in ("mimic", "eicu"):
    path = root / dataset / "events.csv"
    frame = pd.read_csv(
        path,
        usecols=[
            "dataset",
            "stay_id",
            "concept",
            "source_table",
            "offset_minutes",
            "value_numeric",
            "value_text",
            "raw_name",
        ],
        low_memory=False,
    )
    frames.append(frame)
events = pd.concat(frames, ignore_index=True)
events["offset_minutes"] = pd.to_numeric(events["offset_minutes"], errors="coerce")
events["value_numeric"] = pd.to_numeric(events["value_numeric"], errors="coerce")

interesting = events.loc[
    events["concept"].isin(
        [
            "spo2",
            "lactate",
            "vis",
            "urine_output",
            "creatinine",
            "bilirubin_total",
            "ast",
            "alt",
            "inr",
            "platelets",
            "troponin_t",
            "troponin_i",
            "pressor",
            "pressor_initiation",
            "mcs",
            "mechanical_ventilation",
        ]
    )
]
summary = (
    interesting.groupby(["dataset", "concept"], dropna=False)
    .agg(
        rows=("stay_id", "size"),
        stays=("stay_id", "nunique"),
        min_offset=("offset_minutes", "min"),
        median_offset=("offset_minutes", "median"),
        max_offset=("offset_minutes", "max"),
        numeric_fraction=("value_numeric", lambda x: x.notna().mean()),
    )
    .reset_index()
)
print(summary.to_string(index=False))

spo2 = interesting.loc[
    interesting["concept"].eq("spo2")
    & interesting["offset_minutes"].between(0, 240, inclusive="left")
    & interesting["value_numeric"].between(50, 100, inclusive="both")
].copy()
spo2 = spo2.drop_duplicates(
    ["dataset", "stay_id", "offset_minutes", "value_numeric"]
).sort_values(["dataset", "stay_id", "offset_minutes"])
spo2["gap"] = spo2.groupby(["dataset", "stay_id"])["offset_minutes"].diff()
spo2["delta"] = spo2.groupby(["dataset", "stay_id"])["value_numeric"].diff()
for dataset, group in spo2.groupby("dataset"):
    valid = group["gap"].gt(0)
    print(f"\n{dataset} first-four-hour SpO2")
    print(
        {
            "rows": len(group),
            "stays": group["stay_id"].nunique(),
            "median_rows_per_stay": group.groupby("stay_id").size().median(),
            "gap_quantiles": group.loc[valid, "gap"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
            "abs_delta_quantiles": group.loc[valid, "delta"].abs().quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict(),
        }
    )
    print(
        group.groupby("source_table").agg(rows=("stay_id", "size"), stays=("stay_id", "nunique")).sort_values("rows", ascending=False).head(15).to_string()
    )

features, cadence = build_multiscale_instability_features(events)
print("\nCadence\n", cadence.to_string(index=False))
print("\nFeature coverage")
for dataset, group in features.groupby("dataset"):
    print(dataset, {
        "n": len(group),
        "adaptive_eligible": int(group["adaptive_eligible_flag"].sum()),
        "raw30_eligible": int(group["raw30_eligible_flag"].sum()),
        "bin15_eligible": int(group["bin15_eligible_flag"].sum()),
        "bin60_eligible": int(group["bin60_eligible_flag"].sum()),
        "any_jump": float(group["adaptive_any_jump_ge4"].mean()),
        "recurrent_jump": float(group["adaptive_recurrent_jump_ge4"].mean()),
        "median_score": float(group["adaptive_instability_burden_score"].median()),
    })
