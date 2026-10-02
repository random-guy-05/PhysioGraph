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
    ax.set_ylim(0, 74)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="#E5E9ED", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(len(CATEGORIES)), ["B", "C", "D", "E", "Unclassified", "Died ≤16 h", "Departed"])
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlim(-0.65, len(CATEGORIES) - 0.35)
    ax.tick_params(axis="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Hour-16 category", labelpad=10)
    if index == 0:
        ax.set_ylabel("Percentage of sample", labelpad=10)


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


def aggregate_mortality(patient_cache, results_dir):
    columns = ["ICUSTAY_ID", "mortality_sample", "measurement_lookback_minutes", "stage_minimum_evidenced", "in_hospital_death", "HOSPITAL_EXPIRE_FLAG", "status_at16"]
    patient = pd.read_csv(patient_cache, usecols=columns)
    current = patient.loc[patient.measurement_lookback_minutes.eq(240)].copy()
    if len(current) != 1597 or not current.ICUSTAY_ID.is_unique or not current.mortality_sample.isin([True, False]).all():
        raise ValueError("Cached cohort/lookback or sample flags have changed.")
    sample = current.loc[current.mortality_sample.eq(True)]
    if len(sample) != 900 or not sample.in_hospital_death.isin([0, 1]).all() or sample.in_hospital_death.sum() != 156:
        raise ValueError("Mortality cache does not reproduce the fixed 900-stay/156-death model sample.")
    if not sample.in_hospital_death.eq(sample.HOSPITAL_EXPIRE_FLAG).all():
        raise ValueError("Mortality outcome disagrees with the admission death flag.")
    expected = pd.read_csv(results_dir / "stage_distribution.csv")
    expected = expected.loc[expected.population.eq("mortality_model") & expected.exposure_group.eq("all") & expected.measurement_lookback_minutes.eq(240)]
    if not sample.stage_minimum_evidenced.isin(expected.category).all():
        raise ValueError("Unknown stage category in cached mortality sample.")
    rows = []
    for record in expected.itertuples():
        group = sample.loc[sample.stage_minimum_evidenced.eq(record.category)]
        n, deaths = len(group), int(group.in_hospital_death.sum())
        if n != record.n:
            raise ValueError("Mortality categories do not match the original stage distribution.")
        if record.category in list("BCDE") + ["unclassified"] and not group.status_at16.eq("in_icu_at_hour16").all():
            raise ValueError("Stage category includes a stay outside the hour-16 ICU risk set.")
        if record.category == "died_by_hour16" and deaths != n:
            raise ValueError("Early death category contains a survivor.")
        rows.append({"category": record.category, "n": n, "in_hospital_deaths": deaths, "hospital_survivors": n - deaths, "mortality_percent": 100 * deaths / n if n else np.nan})
    table = pd.DataFrame(rows)
    if table.n.sum() != 900 or table.in_hospital_deaths.sum() != 156:
        raise ValueError("Mortality table fails sample/event reconciliation.")
    table.to_csv(results_dir / "mortality_by_stage.csv", index=False)
    audit = {"execution": "local aggregation of saved hour-16 patient cache; no clinical processing or model rerun", "private_source_sha256": hashlib.sha256(patient_cache.read_bytes()).hexdigest(), "private_source_name": patient_cache.name, "outcome": "index-admission HOSPITAL_EXPIRE_FLAG (in-hospital death)", "population": "fixed organ-support-adjusted mortality model sample", "sample_n": 900, "in_hospital_deaths": 156, "measurement_lookback_minutes": 240, "reconciliation": "category counts equal saved stage_distribution.csv; mortality equals admission flag; each ICU stay counted once", "stage_A": "unavailable"}
    (results_dir / "mortality_by_stage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def plot_mortality(results_dir, output):
    source = results_dir / "mortality_by_stage.csv"
    if not source.exists():
        return None
    table = pd.read_csv(source).set_index("category")
    order = ["B", "C", "D", "E", "unclassified"]
    displayed = table.loc[order]
    vis_source = results_dir / "vis_by_stage.csv"
    vis = pd.read_csv(vis_source).set_index("category").loc[order]
    if not np.array_equal(vis.stage_n, displayed.n) or not np.array_equal(vis.vis_available_n + vis.vis_unavailable_n, vis.stage_n):
        raise ValueError("VIS denominators disagree with the mortality stage categories.")
    n = displayed.n.to_numpy(dtype=int)
    deaths = displayed.in_hospital_deaths.to_numpy(dtype=int)
    survivors = displayed.hospital_survivors.to_numpy(dtype=int)
    if not np.array_equal(n, deaths + survivors) or not np.allclose(displayed.mortality_percent, deaths / n * 100):
        raise ValueError("Mortality graph rates/counts disagree.")
    total_n, total_deaths = int(table.n.sum()), int(table.in_hospital_deaths.sum())
    early = table.loc["died_by_hour16"]
    departed = table.loc[["left_icu_by_hour16", "discharged_by_hour16"]].sum()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.36, wspace=0.20)
    fig.text(0.035, 0.945, "Mortality and VIS by hour-16 SCAI proxy stage", fontsize=22, weight="bold", color="#263542")
    fig.text(0.035, 0.891, f"Mortality model sample: {total_deaths:,}/{total_n:,} deaths ({100 * total_deaths / total_n:.1f}%). Stage bars include {int(n.sum()):,} stays alive and in the ICU at hour 16.", fontsize=12, color="#52606D")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#C65448"), plt.Rectangle((0, 0), 1, 1, color="#CAD2D9")]
    fig.legend(handles, ["In-hospital death", "Survived to hospital discharge"], loc="upper left", bbox_to_anchor=(0.03, 0.86), frameon=False, ncol=2, fontsize=11)
    ys = np.arange(len(order))
    for i, ax in enumerate(axes):
        ax.set_xticks(ys, ["B", "C", "D", "E", "Unclassified"])
        ax.set_xlim(-0.6, 4.6)
        ax.tick_params(length=0, pad=8)
        ax.grid(axis="y", color="#E5E9ED", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Deaths and survivors within each category", loc="left", fontsize=12, weight="bold", pad=15)
    axes[0].bar(ys, deaths, width=0.58, color="#C65448")
    axes[0].bar(ys, survivors, bottom=deaths, width=0.58, color="#CAD2D9")
    for y, d, s, count in zip(ys, deaths, survivors, n):
        axes[0].text(y if d >= 20 else y + 0.36, d / 2, str(d), color="white" if d >= 20 else "#C65448", va="center", ha="center" if d >= 20 else "left", fontsize=10, weight="bold")
        axes[0].text(y, d + s / 2, str(s), color="#263542", va="center", ha="center", fontsize=10)
        axes[0].text(y, count + 8, f"n = {count}", ha="center", fontsize=10, color="#263542")
    axes[0].set_ylim(0, max(n) * 1.2)
    axes[0].set_ylabel("Number of ICU stays", labelpad=12)
    axes[0].set_xlabel("Hour-16 SCAI proxy category", labelpad=10)
    axes[1].set_title("VIS distribution at hour 16", loc="left", fontsize=12, weight="bold", pad=15)
    boxes, positions = [], []
    for i, row in enumerate(vis.itertuples()):
        if row.vis_available_n:
            boxes.append({"med": row.median, "q1": row.q25, "q3": row.q75, "whislo": row.p05, "whishi": row.p95, "fliers": []})
            positions.append(i)
    artists = axes[1].bxp(boxes, positions=positions, widths=0.5, showfliers=False, patch_artist=True, manage_ticks=False)
    for box in artists["boxes"]:
        box.set(facecolor="#BCD5E5", edgecolor="#2676AD", linewidth=1.5)
    for line in artists["medians"]:
        line.set(color="#263542", linewidth=2)
    max_vis = float(vis.p95.max())
    if max_vis > 60:
        axes[1].set_yscale("symlog", linthresh=10)
        axes[1].set_ylabel("VIS (linear 0–10; log scale above 10)", labelpad=12)
    else:
        axes[1].set_ylabel("Vasoactive–inotropic score (VIS)", labelpad=12)
    axes[1].set_ylim(0, max(10, max_vis * 1.35))
    axes[1].set_xticks(ys, [f"{label}\nn = {int(row.vis_available_n)}/{int(row.stage_n)}" for label, row in zip(["B", "C", "D", "E", "Unclassified"], vis.itertuples())])
    axes[1].set_xlabel("Category; VIS available / all stays", labelpad=10)
    fig.text(0.035, 0.212, "VIS: boxes = IQR; line = median; whiskers = 5th–95th percentiles (outside values not plotted). Missing VIS is excluded; available / total shown.", fontsize=10, color="#52606D")
    fig.text(0.035, 0.170, f"Outside both panels: {int(early.n)} deaths by hour 16; {int(departed.n)} earlier ICU departures/discharges ({int(departed.in_hospital_deaths)} in-hospital deaths).", fontsize=11, color="#263542")
    fig.text(0.035, 0.125, "Stage A is unavailable. B–E are minimum evidenced proxy stages; unclassified does not mean normal.", fontsize=11, weight="bold", color="#263542")
    fig.text(0.035, 0.080, "Standard six-drug VIS uses documented dose rates at hour 16. No recent documentation or an unresolvable dose makes VIS unavailable.", fontsize=10, color="#52606D")
    fig.text(0.035, 0.039, "Descriptive comparison among stays alive/in ICU at hour 16. Vasoactive treatment contributes to SCAI staging, so these measures are not independent.", fontsize=10, color="#52606D")
    export(fig, output, "scai_hour16_mortality_by_stage")
    return {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "stage_bar_n": int(n.sum()), "stage_bar_deaths": int(deaths.sum()), "right_panel": "hour-16 six-drug VIS distribution", "vis_source_sha256": hashlib.sha256(vis_source.read_bytes()).hexdigest(), "vis_available_n": int(vis.vis_available_n.sum()), "orientation": "categories on x; counts and VIS on y"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--patient-cache", type=Path, help="Optionally aggregate mortality from the saved private hour-16 cache.")
    args = parser.parse_args()
    if args.patient_cache is not None:
        aggregate_mortality(args.patient_cache, args.results_dir)
    source = args.results_dir / "stage_distribution.csv"
    frame = pd.read_csv(source)
    frame = frame[frame.measurement_lookback_minutes == 240].copy()
    output = args.results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})

    fig, axes = plt.subplots(1, 3, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.79, bottom=0.27, wspace=0.18)
    fig.text(0.035, 0.94, "Hour-16 SCAI proxy distribution", fontsize=23, weight="bold", color="#263542")
    fig.text(0.035, 0.891, "All stays remain in the denominator, including unclassified stays and those no longer in the ICU.", fontsize=12, color="#52606D")
    for i, (population, title) in enumerate(POPULATIONS.items()):
        n, counts = counts_for(frame, population, "all")
        rates = counts / n * 100
        ax = axes[i]
        style_axis(ax, i)
        ax.set_title(f"{title}\nn = {n:,} stays", loc="left", fontsize=12, weight="bold", pad=21)
        ax.bar(np.arange(len(counts)), rates, width=0.6, color=COLORS)
        for y, count, rate in zip(range(len(counts)), counts, rates):
            ax.text(y, rate + 1.0, f"{count:,}\n({rate:.1f}%)", ha="center", va="bottom", fontsize=9, color="#263542")
    footer(fig)
    export(fig, output, "scai_hour16_distributions")

    fig, axes = plt.subplots(1, 3, figsize=(17, 9.5))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.74, bottom=0.26, wspace=0.18)
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
            ax.bar(ys, rates, width=0.31, color=color)
            for y, count, rate in zip(ys, counts, rates):
                ax.text(y, rate + 1.0, f"{count:,} ({rate:.1f}%)", ha="center", va="bottom", rotation=90, fontsize=8, color="#263542")
    footer(fig, stratified=True)
    export(fig, output, "scai_hour16_by_instability")
    audit = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "measurement_lookback_minutes": 240, "percent_denominator": "entire population/exposure group, including unclassified and not-in-ICU stays", "stage_A": "unavailable; not plotted as zero", "departure_display": "sum of mutually exclusive discharged_by_hour16 and left_icu_by_hour16", "execution": "local aggregate plotting; no clinical data processing or model rerun", "figures": ["scai_hour16_distributions", "scai_hour16_by_instability"], "formats": ["png", "pdf", "svg"]}
    mortality = plot_mortality(args.results_dir, output)
    audit["axis_orientation"] = "categories on x; counts, percentages or VIS on y"
    if mortality is not None:
        audit["mortality_by_stage"] = mortality
        audit["figures"].append("scai_hour16_mortality_by_stage")
    (output / "figure_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Saved {len(audit['figures'])} figures in PNG, PDF, and SVG: {output}")


if __name__ == "__main__":
    main()
