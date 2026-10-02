"""Add maximum concurrently summed 0–4-hour VIS to the fixed primary models."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import calculate_vis_hour16 as doses
import rerun_support_adjusted_models as support


def load_frame(args):
    frame = pd.read_csv(args.patient_cache, parse_dates=["INTIME", "OUTTIME"])
    frame = frame.loc[frame.measurement_lookback_minutes.eq(240)].copy()
    if len(frame) != 1597 or not frame.ICUSTAY_ID.is_unique or frame.exposure.sum() != 539:
        raise ValueError("Locked cohort/exposure changed.")
    early = pd.read_csv(args.support_cache)
    check = frame.merge(early[["ICUSTAY_ID", "mechanical_ventilation_0_4h"]], on="ICUSTAY_ID", suffixes=("", "_reference"), validate="one_to_one")
    if not check.mechanical_ventilation_0_4h.fillna(-1).eq(check.mechanical_ventilation_0_4h_reference.fillna(-1)).all():
        raise ValueError("Ventilation cache disagrees with the fixed model frame.")
    frame = frame.merge(early[["ICUSTAY_ID", "vasopressor_0_4h", "vasoactive_0_4h"]], on="ICUSTAY_ID", validate="one_to_one")
    if frame.mortality_sample.sum() != 900 or frame.primary_lactate_sample.sum() != 380:
        raise ValueError("Fixed model samples changed.")
    if (frame.primary_lactate_sample & ~frame.mortality_sample).any():
        raise ValueError("Primary lactate sample is not contained in the mortality sample.")
    return frame


def extract(args, frame):
    hashes = pd.read_csv(args.source_manifest).set_index("filename").sha256.to_dict()
    reference = json.loads(args.dose_reference.read_text())
    if any(doses.sha(p) != reference["patient_cache_sha256"] for p in [args.patient_cache, args.weight_patient_cache]):
        raise ValueError("Patient cache changed from the validated VIS dose/weight reference.")
    for name, h in reference["source_hashes"].items():
        if hashes[name] != h:
            raise ValueError("VIS mapping/source reference changed.")
    mapping = {int(k): v for k, v in reference["item_mapping"].items()}
    if set(mapping.values()) != set(doses.FACTORS) or mapping.get(221906) != "norepinephrine":
        raise ValueError("Six-drug VIS mapping invalid.")
    stays = frame.loc[frame.mortality_sample, ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]].copy()
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='2GB'; SET preserve_insertion_order=false")
    con.register("stays", stays)
    key = {"cohort_sha256": hashlib.sha256(stays.to_csv(index=False).encode()).hexdigest(), "algorithm": "maximum-concurrent-vis-0-4-v1"}
    linkage = "try_cast(p.SUBJECT_ID AS BIGINT)=s.SUBJECT_ID AND try_cast(p.HADM_ID AS BIGINT)=s.HADM_ID"
    parts = []
    for system in ["CV", "MV"]:
        start = "CHARTTIME" if system == "CV" else "STARTTIME"
        end = "NULL::DOUBLE" if system == "CV" else "date_diff('second',s.INTIME,try_cast(p.ENDTIME AS TIMESTAMP))/60.0"
        rate = "try_cast(p.RATE AS DOUBLE)" if system == "MV" else "try_cast(CASE WHEN try_cast(p.ITEMID AS BIGINT) IN (42273,42802) THEN p.AMOUNT ELSE p.RATE END AS DOUBLE)"
        unit = "p.RATEUOM" if system == "MV" else "CASE WHEN try_cast(p.ITEMID AS BIGINT) IN (42273,42802) THEN p.AMOUNTUOM ELSE p.RATEUOM END"
        weight = "NULL::DOUBLE" if system == "CV" else "try_cast(p.PATIENTWEIGHT AS DOUBLE)"
        stopped = "p.STOPPED" if system == "CV" else "''"
        category = "''" if system == "CV" else "coalesce(p.ORDERCATEGORYNAME,'') || ' ' || coalesce(p.ORDERCATEGORYDESCRIPTION,'')"
        timing = "try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME" if system == "CV" else "try_cast(p.ENDTIME AS TIMESTAMP)>s.INTIME"
        extra = "" if system == "CV" else "AND coalesce(p.STATUSDESCRIPTION,'')!='Rewritten' AND coalesce(try_cast(p.CANCELREASON AS INTEGER),0)=0"
        query = f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
          date_diff('second',s.INTIME,try_cast(p.{start} AS TIMESTAMP))/60.0 AS start_minute,
          {end} AS end_minute,{rate} AS rate,{unit} AS rate_unit,
          {weight} AS patient_weight,{stopped} AS stopped,{category} AS category_text
          FROM read_csv(SOURCE_PATH,header=true,all_varchar=true) p
          JOIN stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE {linkage} AND try_cast(p.{start} AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours'
          AND {timing} {extra}"""
        part = doses.query_cache(args, con, "INPUTEVENTS_" + system, query, key, hashes)
        part["system"] = system; part["drug"] = part.itemid.map(mapping); parts.append(part)
    records = pd.concat(parts, ignore_index=True)
    meta = json.loads((args.weight_cache_root / "chartevents_vis.json").read_text())
    if meta["source_sha256"] != hashes["CHARTEVENTS.csv.gz"]:
        raise ValueError("Cached weight source hash changed.")
    weight = con.execute(f"SELECT * FROM read_parquet({doses.sql(args.weight_cache_root / 'chartevents_vis.parquet')}) WHERE minute>=0 AND minute<240").df()
    old_patient = pd.read_csv(args.weight_patient_cache, usecols=["ICUSTAY_ID", "measurement_lookback_minutes", "mortality_sample", "status_at16"])
    covered = set(old_patient.loc[old_patient.measurement_lookback_minutes.eq(240) & old_patient.mortality_sample & old_patient.status_at16.eq("in_icu_at_hour16"), "ICUSTAY_ID"])
    # Only patients outside the prior 882-stay weight extraction can need another source scan.
    needs = set()
    for row in records.loc[records.drug.notna() & records.rate.gt(0) & ~records.ICUSTAY_ID.isin(covered)].itertuples():
        if pd.isna(doses.normalized_rate(row, np.nan)) and pd.notna(doses.normalized_rate(row, 80.0)):
            needs.add(row.ICUSTAY_ID)
    if needs:
        extra_stays = stays.loc[stays.ICUSTAY_ID.isin(needs)].copy(); con.register("extra_stays", extra_stays)
        query = f"""SELECT s.ICUSTAY_ID,try_cast(p.ITEMID AS BIGINT) AS itemid,
          date_diff('second',s.INTIME,try_cast(p.CHARTTIME AS TIMESTAMP))/60.0 AS minute,
          try_cast(p.VALUENUM AS DOUBLE) AS value,p.VALUEUOM AS unit
          FROM read_csv(SOURCE_PATH,header=true,all_varchar=true) p
          JOIN extra_stays s ON try_cast(p.ICUSTAY_ID AS BIGINT)=s.ICUSTAY_ID
          WHERE {linkage} AND try_cast(p.ITEMID AS BIGINT) IN (762,763,3580,3581,3582,224639,226512)
          AND try_cast(p.CHARTTIME AS TIMESTAMP)>=s.INTIME
          AND try_cast(p.CHARTTIME AS TIMESTAMP)<s.INTIME+INTERVAL '4 hours'
          AND try_cast(p.ERROR AS DOUBLE) IS DISTINCT FROM 1"""
        extra = doses.query_cache(args, con, "CHARTEVENTS", query, {**key, "extra_stays_sha256": hashlib.sha256(extra_stays.to_csv(index=False).encode()).hexdigest()}, hashes)
        weight = pd.concat([weight, extra], ignore_index=True)
    con.close()
    return records, weight, mapping, hashes, len(needs)


def weight_series(weight):
    weight = weight.copy(); weight["kg"] = weight.value
    lb = weight.itemid.eq(3581) | weight.unit.fillna("").str.lower().isin(["lb", "lbs", "pounds"])
    oz = weight.itemid.eq(3582) | weight.unit.fillna("").str.lower().isin(["oz", "ounces"])
    weight.loc[lb, "kg"] *= 0.45359237; weight.loc[oz, "kg"] *= 0.028349523125
    kg = weight.loc[~lb & ~oz].groupby(["ICUSTAY_ID", "minute"]).kg.median()
    imperial = weight.loc[lb | oz].groupby(["ICUSTAY_ID", "minute", "itemid"]).kg.median().groupby(level=[0, 1]).sum()
    result = kg.combine_first(imperial).reset_index()
    return result.loc[result.kg.between(20, 300)].sort_values("minute")


def peak_score(stay, records, weight):
    end = min(240.0, (stay.OUTTIME - stay.INTIME).total_seconds() / 60)
    if end <= 0:
        return np.nan, "no_observable_early_window", 0
    records = records.loc[records.start_minute.lt(end)].copy()
    if records.empty:
        return np.nan, "no_early_infusion_documentation", 0
    # The conventional score describes continuous infusions, not resuscitation boluses.
    bolus = records.system.eq("MV") & records.category_text.fillna("").str.contains("bolus|drug push", case=False, regex=True)
    meds = records.loc[records.drug.notna() & ~bolus].copy()
    if meds.empty:
        return 0.0, "available_documented_zero", 1
    mv = meds.loc[meds.system.eq("MV")]
    cv = meds.loc[meds.system.eq("CV")].copy()
    stop = cv.stopped.fillna("").str.lower(); cv["off"] = stop.eq("stopped") | stop.str.startswith("d/c")
    effective = cv.loc[cv.rate.notna() | cv.off]
    # Evaluate concurrently active doses at change points. Expiry alone can only decrease a calculable score.
    points = {0.0}
    points.update(meds.loc[meds.start_minute.ge(0), "start_minute"])
    points.update(mv.loc[mv.end_minute.ge(0) & mv.end_minute.lt(end), "end_minute"])
    points.update(weight.loc[weight.minute.lt(end), "minute"])
    peak, observed = 0.0, 0
    for minute in sorted(p for p in points if 0 <= p < end):
        active_mv = mv.loc[mv.start_minute.le(minute) & mv.end_minute.gt(minute)]
        prior = effective.loc[effective.start_minute.le(minute)]
        latest = prior.groupby(["drug", "start_minute"], as_index=False).agg(rate=("rate", "max"), off=("off", "max")).sort_values("start_minute").drop_duplicates("drug", keep="last")
        stale = latest.start_minute.lt(minute - 60) & (latest.rate.gt(0) | latest.rate.isna()) & ~latest.off
        if stale.any():
            return np.nan, "stale_CareVue_dose_at_observed_state", observed
        active_cv = prior.loc[prior.start_minute.ge(minute - 60)]
        unknown = cv.loc[cv.start_minute.le(minute) & cv.start_minute.ge(minute - 60) & ~cv.drug.isin(prior.drug)]
        if not unknown.empty:
            return np.nan, "unavailable_CareVue_rate", observed
        active = pd.concat([active_mv, active_cv])
        prior_weight = weight.loc[weight.minute.le(minute)]
        kg = float(prior_weight.kg.iloc[-1]) if len(prior_weight) else np.nan
        total = 0.0
        for drug, group in active.groupby("drug"):
            group = group.loc[group.start_minute.eq(group.start_minute.max())]
            stopped = group.stopped.fillna("").str.lower()
            if (stopped.eq("stopped") | stopped.str.startswith("d/c")).any():
                continue
            normalized = [doses.normalized_rate(row, kg) for row in group.itertuples()]
            if not normalized or any(pd.isna(x) for x in normalized):
                return np.nan, "unresolvable_dose_unit_or_prior_weight", observed
            total += doses.FACTORS[drug] * max(normalized)
        peak = max(peak, total); observed += 1
    return peak, "available", observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clinical-root", type=Path); source.add_argument("--clinical-archive", type=Path)
    for name in ["patient-cache", "support-cache", "weight-cache-root", "weight-patient-cache", "dose-reference", "source-manifest", "private-cache-root", "output-root", "reference-models"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); args.private_cache_root.mkdir(parents=True, exist_ok=True); args.output_root.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args)
    base = ["age_years", "sex_male", "baseline_lactate_mmol_l", "ckd_expanded", "liver_prior", "diabetes_prior", "mean_binned_spo2", "vasopressor_0_4h", "mechanical_ventilation_0_4h"]
    specs = [("primary_lactate_12h", "lactate_rise_ge0_5_12h", "primary_lactate_sample", ["esrd_lactate", "lung_lactate"]), ("in_hospital_mortality", "in_hospital_death", "mortality_sample", ["esrd_mortality", "lung_mortality"])]
    references = pd.read_csv(args.reference_models); baseline = {}
    for label, y, mask, extra in specs:
        fit, sample = support.checked_fit(frame.loc[frame[mask]], y, base + extra)
        old = references.loc[references.analysis.eq(label) & references.specification.eq("plus_mean_SpO2_vasopressors_ventilation")].iloc[0]
        if fit["n"] != old.n or fit["events"] != old.events or not np.allclose([fit["adjusted_RR"], fit["adjusted_RR_95CI_low"], fit["adjusted_RR_95CI_high"]], old[["adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high"]].to_numpy(float), rtol=1e-8):
            raise ValueError("Existing support-adjusted model not reproduced.")
        baseline[label] = fit
    print("Existing lactate and mortality models reproduced from saved inputs.", flush=True)
    records, raw_weight, mapping, hashes, additional_weights = extract(args, frame)
    weights = weight_series(raw_weight); vis = []
    for stay in frame.loc[frame.mortality_sample].itertuples():
        score, status, states = peak_score(stay, records.loc[records.ICUSTAY_ID.eq(stay.ICUSTAY_ID)], weights.loc[weights.ICUSTAY_ID.eq(stay.ICUSTAY_ID)])
        vis.append({"ICUSTAY_ID": stay.ICUSTAY_ID, "VIS_max_0_4h": score, "VIS_status": status, "evaluated_states": states})
    vis = pd.DataFrame(vis); vis.to_csv(args.private_cache_root / "vis_by_stay.csv", index=False)
    frame = frame.merge(vis, on="ICUSTAY_ID", how="left", validate="one_to_one"); frame["log1p_VIS_max_0_4h"] = np.log1p(frame.VIS_max_0_4h)
    rows, reconcile, availability = [], [], []
    for label, y, mask, extra in specs:
        full = frame.loc[frame[mask]].copy(); matched = full.loc[full.VIS_max_0_4h.notna()].copy()
        for name, use, covariates in [("current_support_adjusted", full, base + extra), ("current_support_adjusted_VIS_available_sample", matched, base + extra), ("plus_maximum_VIS_0_4h", matched, base + extra + ["VIS_max_0_4h"]), ("sensitivity_plus_log1p_maximum_VIS_0_4h", matched, base + extra + ["log1p_VIS_max_0_4h"])]:
            fit, sample = support.checked_fit(use, y, covariates)
            exposed = sample.loc[sample.exposure.eq(1)]; unexposed = sample.loc[sample.exposure.eq(0)]
            fit.update(analysis=label, specification=name, requested_covariates="|".join(covariates), exposed_n=len(exposed), exposed_events=int(exposed[y].sum()), unexposed_n=len(unexposed), unexposed_events=int(unexposed[y].sum()), raw_RR=float(exposed[y].mean() / unexposed[y].mean()), raw_RD=float(exposed[y].mean() - unexposed[y].mean()))
            rows.append(fit)
        removed = full.loc[full.VIS_max_0_4h.isna()]
        reconcile.append({"analysis": label, "original_n": len(full), "original_events": int(full[y].sum()), "VIS_sample_n": len(matched), "VIS_sample_events": int(matched[y].sum()), "removed_n": len(removed), "removed_events": int(removed[y].sum()), "removed_non_events": int(len(removed) - removed[y].sum()), "removed_exposed_n": int(removed.exposure.eq(1).sum()), "removed_unexposed_n": int(removed.exposure.eq(0).sum())})
        availability.extend({"analysis": label, "VIS_status": status, "n": len(group), "events": int(group[y].sum())} for status, group in full.groupby("VIS_status"))
    results = pd.DataFrame(rows); results.to_csv(args.output_root / "adjusted_models.csv", index=False)
    pd.DataFrame(reconcile).to_csv(args.output_root / "denominator_reconciliation.csv", index=False)
    pd.DataFrame(availability).to_csv(args.output_root / "VIS_availability.csv", index=False)
    audit = {"execution": "local analysis-only model rerun; no waveform, cohort, endpoint or existing-covariate reconstruction", "VIS": "maximum of concurrently summed documented six-drug infusion VIS during [0,240) ICU minutes", "primary_VIS_form": "continuous linear raw VIS", "sensitivity_VIS_form": "log(1+VIS), continuous", "boluses": "explicit MV bolus/drug-push categories excluded from VIS", "missing_rule": "no early infusion documentation, unresolved rate/units/weight, or stale positive CV dose at an evaluated state => unavailable; never replaced with zero", "zero_rule": "no documented six-drug continuous infusion with early infusion-table coverage => documented zero", "weight": "infusion PATIENTWEIGHT when valid; otherwise latest valid charted weight at or before the evaluated time; no future backfill", "carevue": "numeric rate carried up to 60 minutes; null volume entries do not erase known rate; explicit latest stop overrides rate", "same_time_duplicates": "latest active start/chart time per drug; tied normalized rates use maximum, no double counting", "existing_models_reproduced": baseline, "denominator_reconciliation": reconcile, "source_hashes": {name: hashes[name] for name in ["D_ITEMS.csv.gz", "INPUTEVENTS_CV.csv.gz", "INPUTEVENTS_MV.csv.gz", "CHARTEVENTS.csv.gz"]}, "additional_weight_extraction_stays": additional_weights, "cached_input_sha256": {str(p.name): doses.sha(p) for p in [args.patient_cache, args.support_cache, args.weight_cache_root / "chartevents_vis.parquet"]}, "code_sha256": doses.sha(__file__), "item_mapping": mapping, "patient_cluster_robust_CI": "unchanged modified Poisson fit and finite-sample sandwich correction; independently checked by existing checked_fit routine", "unchanged_primary_followup": "complete follow-up through hour16 required for primary last-value lactate events and non-events", "no_original_analysis_artifact_overwrite": True}
    audit["support_code_sha256"] = doses.sha(support.__file__)
    audit["model_code_sha256"] = doses.sha(support.core.__file__)
    audit["dose_conversion_code_sha256"] = doses.sha(doses.__file__)
    audit["dose_reference_sha256"] = doses.sha(args.dose_reference)
    audit["reference_models_sha256"] = doses.sha(args.reference_models)
    (args.output_root / "analysis_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    report = ["# Primary lactate and mortality models adding early VIS", "", "Local rerun using the fixed support-adjusted model inputs. Maximum concurrent VIS is measured only during ICU hours 0–4; hour-16 VIS is not used for these models.", "", "| Outcome | Model | N | Events | Adjusted RR | Patient-cluster robust 95% CI | p |", "|---|---|---:|---:|---:|---|---:|"]
    for row in rows:
        report.append(f"| {row['analysis']} | {row['specification']} | {row['n']} | {row['events']} | {row['adjusted_RR']:.6f} | {row['adjusted_RR_95CI_low']:.6f}–{row['adjusted_RR_95CI_high']:.6f} | {row['p_value']:.6f} |")
    report += ["", "## Covariates and score definition", "", "- Existing covariates retained: age, sex, baseline lactate, expanded prior-coded CKD, model-specific prior-coded ESRD and chronic pulmonary disease, chronic liver disease, diabetes, 0–4-hour mean SpO₂, vasopressor infusion use, and documented mechanical ventilation.", "- Addition: maximum concurrently summed six-drug VIS during [0,4) hours, as a continuous linear term. The log(1+VIS) model is a separate sensitivity for skewness; it does not replace the requested primary model.", "- VIS = dopamine + dobutamine + 100×epinephrine + 100×norepinephrine + 10×milrinone + 10,000×vasopressin. All rates are µg/kg/min except vasopressin in units/kg/min; phenylephrine is excluded. Drug maxima are not summed across different times. Reference: " + doses.REFERENCE, "- MV intervals active at each change point; CV latest numeric rate within the prior 60 minutes. Explicit stop records and unit conversions follow the existing VIS mapping. Explicit MV bolus/drug-push records are excluded. No rates or weights after hour4 are used. Dose outliers are retained.", "- Documented absence of six-drug infusions with early infusion-table coverage gives zero. Missing dosing/units/required prior weight, a stale positive CV rate at an evaluated state, or no early infusion documentation gives unavailable VIS. The score is a maximum of documented EHR states, not an adjudicated continuous bedside maximum.", "- Complete-case sample changes are shown separately. The prior model is refitted in the exact VIS-available subset before VIS is added. Both models use modified Poisson regression with SUBJECT_ID-clustered robust errors and the existing finite-sample correction; every fit also passes the existing independent likelihood/covariance calculation.", "- All cohort, exposure, endpoint, comorbidity and prior support definitions remain fixed. The primary lactate endpoint still requires complete follow-up for events and non-events through its unchanged 12-hour post-landmark horizon.", "- Support, VIS and instability share hours0–4; this estimates a conditional association and does not establish causation or independent predictive improvement. The lactate analysis has few events relative to its parameter count.", "- Patient-level dose and VIS caches remain outside the public project.", "", "## Denominator reconciliation", "", "| Outcome | Prior N/events | VIS N/events | Removed events | Removed non-events | Removed exposed/unexposed |", "|---|---:|---:|---:|---:|---:|"]
    for r in reconcile:
        report.append(f"| {r['analysis']} | {r['original_n']}/{r['original_events']} | {r['VIS_sample_n']}/{r['VIS_sample_events']} | {r['removed_events']} | {r['removed_non_events']} | {r['removed_exposed_n']}/{r['removed_unexposed_n']} |")
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n")
    print(results[["analysis", "specification", "n", "events", "adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high", "p_value"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
