"""Export hour-16 SCAI proxy figures from aggregate results only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


POPULATIONS = {
    "all_spo2_eligible": "Full SpO₂-eligible cohort",
    "primary_lactate_model": "Primary lactate model sample",
    "mortality_model": "Mortality model sample",
}
CATEGORIES = ["B", "C", "D", "E", "unclassified", "died_by_hour16", "departure"]
LABELS = ["Stage B", "Stage C", "Stage D", "Stage E", "Unclassified", "Died by hour 16", "Left ICU / discharged"]
COLORS = ["#2676AD", "#CF942B", "#C65448", "#8064A2", "#9AA5AF", "#374451", "#CAD2D9"]


def counts_for(frame, population, group):
    rows = frame[(frame.population == population) & (frame.exposure_group == group)]
    expected = {"A", "B", "C", "D", "E", "unclassified", "died_by_hour16", "discharged_by_hour16", "left_icu_by_hour16"}
    if len(rows) != len(expected) or set(rows.category) != expected or rows.cohort_n.nunique() != 1:
        raise ValueError(f"Incomplete or duplicated categories: {population}/{group}")
    n = int(rows.cohort_n.iloc[0])
    counts = rows.set_index("category")["n"].to_dict()
    if any(v < 0 or int(v) != v for v in counts.values()) or sum(counts.values()) != n:
        raise ValueError(f"Category counts do not reconcile: {population}/{group}")
    if counts["A"] != 0:
        raise ValueError("Stage A availability has changed; revise the figure labels.")
    if not np.allclose(rows.percent_of_cohort, rows.n / n * 100, rtol=0, atol=1e-8):
        raise ValueError("Stored percentages disagree with category counts.")
    counts["departure"] = counts["discharged_by_hour16"] + counts["left_icu_by_hour16"]
    return n, np.array([counts[c] for c in CATEGORIES], dtype=int)


def style_axis(ax, index):
    ax.set_xlim(0, 74)
    ax.set_xticks([0, 20, 40, 60])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color="#E5E9ED", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(len(CATEGORIES)), LABELS if index == 0 else [""] * len(CATEGORIES))
    ax.set_ylim(len(CATEGORIES) - 0.5, -0.7)
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Percentage of sample", labelpad=12)


def footer(fig, stratified=False):
    denominator = "Percentages use the full sample in each panel." if not stratified else "Percentages use each exposure group's full sample; labels show count (percentage)."
    fig.text(0.035, 0.092, "Stage A cannot be determined from the available data; it is not a zero-prevalence category.", fontsize=11, weight="bold", color="#263542")
    fig.text(0.035, 0.061, "B–E are minimum evidenced SCAI proxy stages. Unclassified stays may include stage A or higher stages.", fontsize=10, color="#52606D")
    fig.text(0.035, 0.034, denominator + " Counts are ICU stays; the model samples overlap and are subsets of the full cohort.", fontsize=9, color="#52606D")
    fig.text(0.035, 0.011, "Physiology: latest valid value in hours 12–16; treatment evidence at hour 16. ICU departure/discharge categories combined for display.", fontsize=9, color="#52606D")


def export(fig, root, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"{stem}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.results_dir / "stage_distribution.csv"
    frame = pd.read_csv(source)
    frame = frame[frame.measurement_lookback_minutes == 240].copy()
    output = args.results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.145, right=0.99, top=0.79, bottom=0.21, wspace=0.11)
    fig.text(0.035, 0.94, "Hour-16 SCAI proxy distribution", fontsize=23, weight="bold", color="#263542")
    fig.text(0.035, 0.891, "All stays remain in the denominator, including unclassified stays and those no longer in the ICU.", fontsize=12, color="#52606D")
    for i, (population, title) in enumerate(POPULATIONS.items()):
        n, counts = counts_for(frame, population, "all")
        rates = counts / n * 100
        ax = axes[i]
        style_axis(ax, i)
        ax.set_title(f"{title}\nn = {n:,} stays", loc="left", fontsize=12, weight="bold", pad=21)
        ax.barh(np.arange(len(counts)), rates, height=0.6, color=COLORS)
        for y, count, rate in zip(range(len(counts)), counts, rates):
            ax.text(rate + 1.0, y, f"{count:,} ({rate:.1f}%)", va="center", fontsize=10, color="#263542")
    footer(fig)
    export(fig, output, "scai_hour16_distributions")

    fig, axes = plt.subplots(1, 3, figsize=(17, 9.5))
    fig.subplots_adjust(left=0.145, right=0.99, top=0.74, bottom=0.20, wspace=0.11)
    fig.text(0.035, 0.95, "Hour-16 SCAI proxy by early SpO₂ instability", fontsize=22, weight="bold", color="#263542")
    fig.text(0.035, 0.905, "Descriptive, unadjusted comparison within each sample; no inference of causation or predictive performance.", fontsize=12, color="#52606D")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2676AD"), plt.Rectangle((0, 0), 1, 1, color="#8897A4")]
    fig.legend(handles, ["Early instability", "No early instability"], loc="upper left", bbox_to_anchor=(0.03, 0.88), frameon=False, ncol=2, fontsize=11)
    for i, (population, title) in enumerate(POPULATIONS.items()):
        ax = axes[i]
        style_axis(ax, i)
        total_n, total_counts = counts_for(frame, population, "all")
        n1, c1 = counts_for(frame, population, "instability")
        n0, c0 = counts_for(frame, population, "no_instability")
        if n1 + n0 != total_n or not np.array_equal(c1 + c0, total_counts):
            raise ValueError(f"Exposure groups do not reconcile: {population}")
        ax.set_title(f"{title}\nInstability n = {n1:,}; no instability n = {n0:,}", loc="left", fontsize=11, weight="bold", pad=20)
        for n, counts, offset, color in [(n1, c1, -0.18, "#2676AD"), (n0, c0, 0.18, "#8897A4")]:
            rates = counts / n * 100
            ys = np.arange(len(counts)) + offset
            ax.barh(ys, rates, height=0.31, color=color)
            for y, count, rate in zip(ys, counts, rates):
                ax.text(rate + 1.0, y, f"{count:,} ({rate:.1f}%)", va="center", fontsize=9, color="#263542")
    footer(fig, stratified=True)
    export(fig, output, "scai_hour16_by_instability")
    audit = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "measurement_lookback_minutes": 240, "percent_denominator": "entire population/exposure group, including unclassified and not-in-ICU stays", "stage_A": "unavailable; not plotted as zero", "departure_display": "sum of mutually exclusive discharged_by_hour16 and left_icu_by_hour16", "execution": "local aggregate plotting; no clinical data processing or model rerun", "figures": ["scai_hour16_distributions", "scai_hour16_by_instability"], "formats": ["png", "pdf", "svg"]}
    (output / "figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved two figures in PNG, PDF, and SVG: {output}")


if __name__ == "__main__":
    main()
