"""Dose-based, six-drug VIS at hour 16 in the fixed mortality model sample."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import duckdb
import numpy as np
import pandas as pd

FACTORS = {"dopamine": 1, "dobutamine": 1, "epinephrine": 100, "norepinephrine": 100, "milrinone": 10, "vasopressin": 10000}
PATTERNS = {"dopamine": "dopamine", "dobutamine": "dobutamine|dobutrex", "epinephrine": r"(?<!nor)epinephrine|adrenalin", "norepinephrine": "norepinephrine|noradrenaline|levophed", "milrinone": "milrinone|primacor", "vasopressin": "vasopressin"}
REFERENCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9891263/"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sql(path):
    return "'" + str(path).replace("'", "''") + "'"


def source_file(args, name, hashes):
    if args.clinical_root:
        path = args.clinical_root / (name + ".csv.gz")
        if sha(path) != hashes[name + ".csv.gz"]:
            raise ValueError(f"Source hash mismatch: {name}")
        return path, False
    path = args.private_cache_root / (name + ".csv.gz")
    with zipfile.ZipFile(args.clinical_archive) as archive:
        members = [x for x in archive.namelist() if Path(x).name == name + ".csv.gz"]
        if len(members) != 1:
            raise ValueError(f"Ambiguous archive member: {name}")
        print(f"Reading validated archive member {name}...", flush=True)
        h = hashlib.sha256()
        with archive.open(members[0]) as src, path.open("wb") as dest:
            for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                h.update(block); dest.write(block)
        if h.hexdigest() != hashes[name + ".csv.gz"]:
            raise ValueError(f"Archive source hash mismatch: {name}")
    return path, True


def query_cache(args, con, name, statement, key, hashes):
    dest = args.private_cache_root / (name.lower() + "_vis.parquet")
    metadata = dest.with_suffix(".json")
    expected = {**key, "source_sha256": hashes[name + ".csv.gz"], "query_template": statement}
    if dest.exists() and metadata.exists() and json.loads(metadata.read_text()) == expected:
        return con.execute(f"SELECT * FROM read_parquet({sql(dest)})").df()
    source, temporary = source_file(args, name, hashes)
    try:
        query = statement.replace("SOURCE_PATH", sql(source))
        print(f"Selecting hour-16 dose/weight records from {name}...", flush=True)
        con.execute(f"COPY ({query}) TO {sql(dest)} (FORMAT PARQUET)")
        metadata.write_text(json.dumps(expected, indent=2))
    finally:
        if temporary:
            source.unlink(missing_ok=True)
    return con.execute(f"SELECT * FROM read_parquet({sql(dest)})").df()


def unit_name(value):
    value = str(value).lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    value = value.replace("micrograms", "mcg").replace("microgram", "mcg").replace("ug/", "mcg/").replace("minutes", "min").replace("minute", "min").replace("hours", "hr").replace("hour", "hr").replace("units", "unit").replace("iu/", "unit/")
    aliases = {"mcgkgmin": "mcg/kg/min", "mcgmin": "mcg/min", "mcgkghr": "mcg/kg/hr", "mcghr": "mcg/hr", "mghr": "mg/hr", "mgmin": "mg/min", "umin": "unit/min", "uhr": "unit/hr", "ukgmin": "unit/kg/min"}
    return aliases.get(value, value)


def normalized_rate(row, chart_weight):
    stopped = str(row.stopped).lower()
    if stopped == "stopped" or stopped.startswith("d/c"):
        return 0.0
    if pd.isna(row.rate) or row.rate < 0:
        return np.nan
    if row.rate == 0:
        return 0.0
    unit = unit_name(row.rate_unit)
    weight = row.patient_weight if pd.notna(row.patient_weight) and 20 <= row.patient_weight <= 300 else chart_weight
    if row.drug == "vasopressin":
        units = {"unit/kg/min": (1, False), "unit/kg/hr": (1 / 60, False), "unit/min": (1, True), "u/min": (1, True), "unit/hr": (1 / 60, True), "u/hr": (1 / 60, True), "milliunit/min": (0.001, True), "mu/min": (0.001, True)}
    else:
        units = {"mcg/kg/min": (1, False), "mcg/kg/hr": (1 / 60, False), "mg/kg/min": (1000, False), "mg/kg/hr": (1000 / 60, False), "ng/kg/min": (0.001, False), "mcg/min": (1, True), "mcg/hr": (1 / 60, True), "mg/min": (1000, True), "mg/hr": (1000 / 60, True), "ng/min": (0.001, True)}
    if unit not in units:
        return np.nan
    factor, needs_weight = units[unit]
    if needs_weight and (pd.isna(weight) or not 20 <= weight <= 300):
        return np.nan
    return row.rate * factor / (weight if needs_weight else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--clinical-root", type=Path)
    sources.add_argument("--clinical-archive", type=Path)
    for field in ["patient-cache", "private-cache-root", "results-dir", "source-manifest"]:
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    args.private_cache_root.mkdir(parents=True, exist_ok=True)
    hashes = pd.read_csv(args.source_manifest).set_index("filename").sha256.to_dict()
    patient = pd.read_csv(args.patient_cache, usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "DBSOURCE", "measurement_lookback_minutes", "mortality_sample", "status_at16", "stage_minimum_evidenced"], parse_dates=["INTIME"])
    sample = patient.loc[patient.measurement_lookback_minutes.eq(240) & patient.mortality_sample.eq(True)].copy()
    if len(sample) != 900 or not sample.ICUSTAY_ID.is_unique:
        raise ValueError("Fixed mortality sample changed.")
    stays = sample.loc[sample.status_at16.eq("in_icu_at_hour16")].copy()
    if len(stays) != 882:
        raise ValueError("Hour-16 ICU risk set changed.")
    item_source, temporary = source_file(args, "D_ITEMS", hashes)
    items = pd.read_csv(item_source)
    if temporary:
        item_source.unlink()
    mapping = {}
    for drug, pattern in PATTERNS.items():
        selected = items.loc[items.LINKSTO.fillna("").str.startswith("inputevents") & items.LABEL.fillna("").str.contains(pattern, case=False, regex=True)]
        for item in selected.ITEMID:
            if int(item) in mapping:
                raise ValueError("Drug item mapped twice.")
            mapping[int(item)] = drug
    if mapping.get(221906) != "norepinephrine" or mapping.get(221289) != "epinephrine":
        raise ValueError("Drug mapping failed.")
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='2GB'; SET preserve_insertion_order=false")
    con.register("stays", stays)
    key = {"cohort_sha256": hashlib.sha256(stays.to_csv(index=False).encode()).hexdigest(), "algorithm": "six-drug-hour16-v1"}
    item_sql = ",".join(map(str, sorted(mapping)))
    linkage = "try_cast(p.SUBJECT_ID AS BIGINT)=s.SUBJECT_ID AND try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID"
    parts = []
    for system in ["CV", "MV"]:
        start = "STARTTIME" if system == "MV" else "CHARTTIME"
        end = "date_diff('second',s.INTIME,try_cast(p.ENDTIME AS TIMESTAMP))/60.0" if system == "MV" else "NULL::DOUBLE"
        weight = "try_cast(p.PATIENTWEIGHT AS DOUBLE)" if system == "MV" else "NULL::DOUBLE"
        timing = "try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME" if system == "MV" else "try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME"
        extras = "AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten' AND coalesce(try_cast(p.CANCELREASON AS INTEGER),0)=0" if system == "MV" else ""
        rate_expr, unit_expr = "try_cast(p.RATE AS DOUBLE)", "p.RATEUOM"
        stop_expr = ""
        if system == "CV":
            rate_expr = "try_cast(CASE WHEN try_cast(p.ITEMID AS BIGINT) IN (42273,42802) THEN p.AMOUNT ELSE p.RATE END AS DOUBLE)"
            unit_expr = "CASE WHEN try_cast(p.ITEMID AS BIGINT) IN (42273,42802) THEN p.AMOUNTUOM ELSE p.RATEUOM END"
            stop_expr = ",p.STOPPED AS stopped"
        selected = f"try_cast(p.ITEMID AS BIGINT) IN ({item_sql}) OR try_cast(p.{start} AS TIMESTAMP)>=s.INTIME+INTERVAL '15 hours'"
        if system == "MV":
            selected += " OR try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME+INTERVAL '16 hours'"
        query = f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
          date_diff('second',s.INTIME,try_cast(p.{start} AS TIMESTAMP))/60.0 AS start_minute,
          {end} AS end_minute,{rate_expr} AS rate,{unit_expr} AS rate_unit,
          {weight} AS patient_weight{stop_expr}
          FROM read_csv(SOURCE_PATH,header=true,all_varchar=true) p
          JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE {linkage} AND try_cast(p.{start} AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'
          AND {timing} AND ({selected}) {extras}"""
        part = query_cache(args, con, "INPUTEVENTS_" + system, query, key, hashes)
        if system == "MV":
            part["stopped"] = ""
        part["system"] = system; part["drug"] = part.itemid.map(mapping); parts.append(part)
    query = f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
       date_diff('second',s.INTIME,try_cast(p.CHARTTIME AS TIMESTAMP))/60.0 AS minute,
       try_cast(p.VALUENUM AS DOUBLE) AS value,p.VALUEUOM AS unit
       FROM read_csv(SOURCE_PATH,header=true,all_varchar=true) p
       JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
       WHERE {linkage} AND try_cast(p.ITEMID AS BIGINT) IN (762,763,3580,3581,3582,224639,226512)
       AND try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME
       AND try_cast(p.CHARTTIME AS TIMESTAMP)<=s.INTIME+INTERVAL '16 hours'
       AND try_cast(p.ERROR AS DOUBLE) IS DISTINCT FROM 1"""
    weight = query_cache(args, con, "CHARTEVENTS", query, key, hashes)
    con.close()
    weight["kg"] = weight.value
    lb = weight.itemid.eq(3581) | weight.unit.fillna("").str.lower().isin(["lb", "lbs", "pounds"])
    oz = weight.itemid.eq(3582) | weight.unit.fillna("").str.lower().isin(["oz", "ounces"])
    weight.loc[lb, "kg"] *= 0.45359237; weight.loc[oz, "kg"] *= 0.028349523125
    # kg records take priority; paired lb/oz records combine at the same time.
    kg_rows = weight.loc[~lb & ~oz].groupby(["ICUSTAY_ID", "minute"]).kg.median()
    imperial = weight.loc[lb | oz].groupby(["ICUSTAY_ID", "minute", "itemid"]).kg.median().groupby(level=[0, 1]).sum()
    weights = kg_rows.combine_first(imperial).reset_index().rename(columns={0: "kg"})
    weights = weights.loc[weights.kg.between(20, 300)].sort_values("minute").drop_duplicates("ICUSTAY_ID", keep="last").set_index("ICUSTAY_ID").kg.to_dict()
    drugs = pd.concat(parts, ignore_index=True)
    rows = []
    for stay in stays.itertuples():
        records = drugs.loc[drugs.ICUSTAY_ID.eq(stay.ICUSTAY_ID)].copy()
        mv = records.loc[records.system.eq("MV") & records.start_minute.le(960) & records.end_minute.gt(960)]
        cv = records.loc[records.system.eq("CV") & records.start_minute.between(900, 960)]
        coverage = not mv.empty or not cv.empty
        status = "available" if coverage else "no_recent_infusion_documentation"
        cv_drug = cv.loc[cv.drug.notna()].copy()
        stopped = cv_drug.stopped.fillna("").str.lower()
        cv_effective = cv_drug.loc[cv_drug.rate.notna() | stopped.eq("stopped") | stopped.str.startswith("d/c")]
        # Null volume-only chart entries do not erase a numeric rate documented in the same hour.
        unresolved = cv_drug.loc[~cv_drug.drug.isin(cv_effective.drug)]
        selected = pd.concat([mv.loc[mv.drug.notna()], cv_effective, unresolved])
        # A stale positive CV rate cannot become an observed zero at hour 16.
        cv_history = records.loc[records.system.eq("CV") & records.drug.notna()].sort_values("start_minute")
        stopped = cv_history.stopped.fillna("").str.lower()
        cv_history["off"] = stopped.eq("stopped") | stopped.str.startswith("d/c")
        cv_history = cv_history.loc[cv_history.rate.notna() | cv_history.off]
        cv_latest = cv_history.groupby(["drug", "start_minute"], as_index=False).agg(rate=("rate", "max"), off=("off", "max")).sort_values("start_minute").drop_duplicates("drug", keep="last")
        if ((cv_latest.start_minute < 900) & (cv_latest.rate.gt(0) | cv_latest.rate.isna()) & ~cv_latest.off).any():
            status = "stale_CareVue_dose"
        value = 0.0
        for drug, group in selected.groupby("drug"):
            latest = group.loc[group.start_minute.eq(group.start_minute.max())]
            stopped = latest.stopped.fillna("").str.lower()
            if (stopped.eq("stopped") | stopped.str.startswith("d/c")).any():
                continue
            doses = [normalized_rate(row, weights.get(stay.ICUSTAY_ID, np.nan)) for row in latest.itertuples()]
            if not doses or any(pd.isna(dose) for dose in doses):
                status = "unavailable_dose_unit_or_weight"
            else:
                value += FACTORS[drug] * max(doses)
        rows.append({"ICUSTAY_ID": stay.ICUSTAY_ID, "category": stay.stage_minimum_evidenced, "VIS": value if status == "available" else np.nan, "availability": status})
    result = pd.DataFrame(rows)
    result.to_csv(args.private_cache_root / "patient_vis_hour16.csv", index=False)
    summaries = []
    for category in ["B", "C", "D", "E", "unclassified"]:
        group = result.loc[result.category.eq(category)]; vals = group.VIS.dropna()
        row = {"category": category, "stage_n": len(group), "vis_available_n": len(vals), "vis_unavailable_n": int(group.VIS.isna().sum()), "vis_zero_n": int(vals.eq(0).sum())}
        row.update({label: float(vals.quantile(q)) if len(vals) else np.nan for label, q in [("minimum", 0), ("p05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("p95", .95), ("maximum", 1)]})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(args.results_dir / "vis_by_stage.csv", index=False)
    result.groupby(["category", "availability"]).size().rename("n").reset_index().to_csv(args.results_dir / "vis_availability.csv", index=False)
    unit_counts = drugs.loc[drugs.drug.notna()].groupby(["system", "drug", "rate_unit"], dropna=False).size().rename("n").reset_index()
    audit = {"execution": "local dose extraction and descriptive aggregation; existing cohort, SCAI stage and mortality unchanged", "reference": REFERENCE, "formula": "dopamine + dobutamine + 100*epinephrine + 100*norepinephrine + 10*milrinone + 10000*vasopressin", "units": "mcg/kg/min except vasopressin in units/kg/min", "phenylephrine": "excluded from standard six-drug VIS", "timing": "MV interval active at minute 960; CV latest documented drug rate in [900,960]; no future data", "weight": "MV infusion PATIENTWEIGHT when 20–300 kg; otherwise latest valid charted weight in [0,960], no future backfill or population imputation", "duplicate_rates": "latest active start/chart time per drug; tied rows use maximum dose after unit conversion", "zero_rule": "no documented active six-drug infusion plus contemporaneous infusion-table coverage; EHR absence is not adjudicated clinical absence", "missing_rule": "no contemporaneous infusion documentation, stale positive/unknown CV rate, unrecognized dose units or missing required weight => VIS unavailable", "risk_set_n": len(result), "available_n": int(result.VIS.notna().sum()), "source_hashes": {name: hashes[name] for name in ["D_ITEMS.csv.gz", "INPUTEVENTS_CV.csv.gz", "INPUTEVENTS_MV.csv.gz", "CHARTEVENTS.csv.gz"]}, "patient_cache_sha256": sha(args.patient_cache), "code_sha256": sha(__file__), "item_mapping": mapping, "record_units": unit_counts.fillna("").to_dict("records"), "limitations": ["SCAI proxy stages themselves use vasoactive treatment; the VIS comparison is partly definition-related", "infusion records may be incomplete; no undocumented medication assumed clinically absent", "dose outliers retained; box whiskers will show 5th–95th percentiles", "descriptive distribution, no adjusted association or predictive test"]}
    (args.results_dir / "vis_by_stage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
