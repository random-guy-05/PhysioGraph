"""Evaluate the user-specified lactate-clearance endpoint from saved private inputs."""
import argparse
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

import calculate_vis_hour16 as doses
import rerun_support_adjusted_models as support
import rerun_vis_adjusted_models as vis_models


COVARIATES = ["age_years", "sex_male", "baseline_lactate_mmol_l", "mean_binned_spo2",
              "vasopressor_0_4h", "mechanical_ventilation_0_4h", "ckd_expanded",
              "esrd_lactate", "lung_lactate", "liver_prior", "diabetes_prior",
              "VIS_max_0_4h", "lactate_interval_hours"]


def prepare(args):
    manifest = pd.read_csv(args.source_output_manifest).set_index("relative_path")
    for rel in ["03_landmark_lactate/lactate_outcomes.csv", "03_landmark_lactate/linked_lactate_draws.csv.gz"]:
        if doses.sha(args.source_output_root / rel) != manifest.loc[rel, "sha256"]:
            raise ValueError("Saved lactate input differs from validated manifest: " + rel)
    audit = json.loads(args.vis_audit.read_text())
    if doses.sha(args.patient_cache) != audit["cached_input_sha256"][args.patient_cache.name]:
        raise ValueError("Saved model frame changed since the VIS analysis.")
    if doses.sha(args.support_cache) != audit["cached_input_sha256"][args.support_cache.name]:
        raise ValueError("Saved support covariates changed since the VIS analysis.")
    frame = vis_models.load_frame(args)
    keys = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
    old = pd.read_csv(args.source_output_root / "03_landmark_lactate/lactate_outcomes.csv",
                      parse_dates=["baseline_lactate_time", "last_lactate_time_12h"])
    old = old[keys + ["baseline_lactate_mmol_l", "baseline_lactate_time", "last_lactate_12h",
                      "last_lactate_time_12h", "observable_end_minute", "lactate_rise_ge0_5_12h"]]
    old = old.rename(columns={"baseline_lactate_mmol_l": "old_baseline", "lactate_rise_ge0_5_12h": "old_rise_endpoint"})
    frame = frame.merge(old, on=keys, validate="one_to_one")
    assert np.allclose(frame.baseline_lactate_mmol_l, frame.old_baseline, equal_nan=True, rtol=0, atol=0)
    draws = pd.read_csv(args.source_output_root / "03_landmark_lactate/linked_lactate_draws.csv.gz",
                        parse_dates=["CHARTTIME", "INTIME"])
    draws = draws.merge(frame[keys], on=keys, validate="many_to_one")
    assert draws.lactate_mmol_l.gt(0).all()
    assert not draws.duplicated(keys + ["CHARTTIME"]).any()
    minute = (draws.CHARTTIME - draws.INTIME).dt.total_seconds() / 60
    assert np.allclose(minute, draws.minute_from_icu_intime, rtol=0, atol=1e-8)
    draws = draws.sort_values(["ICUSTAY_ID", "CHARTTIME"])
    before = draws.loc[minute.reindex(draws.index).between(0, 240)].drop_duplicates("ICUSTAY_ID", keep="last")
    after = draws.loc[minute.reindex(draws.index).gt(240) & minute.reindex(draws.index).le(960)].drop_duplicates("ICUSTAY_ID", keep="last")
    for selected, prefix in [(before, "selected_baseline"), (after, "selected_followup")]:
        part = selected[["ICUSTAY_ID", "lactate_mmol_l", "CHARTTIME"]].rename(
            columns={"lactate_mmol_l": prefix + "_value", "CHARTTIME": prefix + "_time"})
        frame = frame.merge(part, on="ICUSTAY_ID", how="left", validate="one_to_one")
    has_baseline = frame.selected_baseline_value.notna()
    has_followup = frame.selected_followup_value.notna()
    assert frame.loc[has_baseline, "selected_baseline_time"].eq(frame.loc[has_baseline, "baseline_lactate_time"]).all()
    assert np.allclose(frame.loc[has_baseline, "selected_baseline_value"], frame.loc[has_baseline, "old_baseline"], rtol=0, atol=0)
    assert frame.loc[has_followup, "selected_followup_time"].eq(frame.loc[has_followup, "last_lactate_time_12h"]).all()
    assert np.allclose(frame.loc[has_followup, "selected_followup_value"], frame.loc[has_followup, "last_lactate_12h"], rtol=0, atol=0)
    frame["baseline_lactate_mmol_l"] = frame.selected_baseline_value
    frame["lactate_interval_hours"] = (frame.selected_followup_time - frame.selected_baseline_time).dt.total_seconds() / 3600
    frame["paired"] = has_baseline & has_followup & frame.lactate_interval_hours.ge(2)
    frame["complete_followup"] = frame.observable_end_minute.ge(960)
    frame["failure_clearance"] = np.nan
    frame["clearance_percent"] = np.nan
    frame["both_normal"] = False
    frame["clearance_ge10"] = False
    for idx, row in frame.loc[frame.paired].iterrows():
        baseline, followup = Decimal(str(row.selected_baseline_value)), Decimal(str(row.selected_followup_value))
        clearance = Decimal(100) * (baseline - followup) / baseline
        both_normal = baseline <= 2 and followup <= 2
        cleared = followup <= Decimal("0.9") * baseline
        frame.loc[idx, ["failure_clearance", "clearance_percent", "both_normal", "clearance_ge10"]] = [float(not (both_normal or cleared)), float(clearance), both_normal, cleared]
    scores = pd.read_csv(args.vis_cache)
    assert len(scores) == 900 and scores.ICUSTAY_ID.is_unique
    assert set(scores.ICUSTAY_ID) == set(frame.loc[frame.mortality_sample, "ICUSTAY_ID"])
    frame = frame.merge(scores, on="ICUSTAY_ID", how="left", validate="one_to_one")
    previous = frame.copy()
    previous["baseline_lactate_mmol_l"] = previous.old_baseline
    references = pd.read_csv(args.vis_audit.parent / "adjusted_models.csv")
    prior_covariates = [c for c in COVARIATES if c != "lactate_interval_hours"]
    for label, outcome, mask in [("primary_lactate_12h", "lactate_rise_ge0_5_12h", "primary_lactate_sample"),
                                 ("in_hospital_mortality", "in_hospital_death", "mortality_sample")]:
        covariates = prior_covariates if label == "primary_lactate_12h" else [c.replace("esrd_lactate", "esrd_mortality").replace("lung_lactate", "lung_mortality") for c in prior_covariates]
        fit, _ = support.checked_fit(previous.loc[previous[mask]], outcome, covariates)
        reference = references.loc[references.analysis.eq(label) & references.specification.eq("plus_maximum_VIS_0_4h")].iloc[0]
        assert fit["n"] == reference.n and fit["events"] == reference.events
        assert np.allclose([fit["adjusted_RR"], fit["adjusted_RR_95CI_low"], fit["adjusted_RR_95CI_high"]], reference[["adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high"]].to_numpy(float), rtol=1e-8)
    assert frame.loc[frame.paired, "failure_clearance"].isin([0, 1]).all()
    return frame, has_baseline, has_followup


def counts(use, label):
    exposed, unexposed = [use.loc[use.exposure.eq(x)] for x in [1, 0]]
    risk_e, risk_u = exposed.failure_clearance.mean(), unexposed.failure_clearance.mean()
    return {"analysis": label, "n": len(use), "patients": use.SUBJECT_ID.nunique(),
            "events": int(use.failure_clearance.sum()), "exposed_n": len(exposed),
            "exposed_events": int(exposed.failure_clearance.sum()), "unexposed_n": len(unexposed),
            "unexposed_events": int(unexposed.failure_clearance.sum()),
            "exposed_risk": risk_e, "unexposed_risk": risk_u,
            "raw_RR": risk_e / risk_u if risk_u > 0 else np.nan, "raw_RD": risk_e - risk_u}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-output-root", "source-output-manifest", "patient-cache", "support-cache", "vis-cache", "vis-audit", "private-cache-root", "output-root"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--primary-followup-policy", choices=["complete_window", "all_pairs"], required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True); args.private_cache_root.mkdir(parents=True, exist_ok=True)
    frame, has_baseline, has_followup = prepare(args)
    frame["primary_clearance_available"] = frame.paired & (frame.complete_followup if args.primary_followup_policy == "complete_window" else True)
    frame["primary_failure_clearance"] = frame.failure_clearance.where(frame.primary_clearance_available)
    assert frame.loc[~frame.primary_clearance_available, "primary_failure_clearance"].isna().all()
    summary, fits, exclusions, transitions, classes = [], [], [], [], []
    no_vis = [c for c in COVARIATES if c != "VIS_max_0_4h"]
    for label, mask in [("all_qualifying_pairs", frame.paired), ("complete_followup_through_hour16", frame.paired & frame.complete_followup)]:
        use = frame.loc[mask].copy(); matched = use.dropna(subset=COVARIATES)
        summary.append(counts(use, label)); summary.append(counts(matched, label + "_full_model_sample"))
        for name, data, covariates in [("support_and_interval_without_VIS", use, no_vis),
                                       ("same_VIS_available_sample_without_VIS", matched, no_vis),
                                       ("full_requested_model", matched, COVARIATES)]:
            fit, sample = support.checked_fit(data, "failure_clearance", covariates)
            fit.update(counts(sample, label)); fit.update(specification=name, requested_covariates="|".join(covariates))
            fits.append(fit)
        removed = use.loc[~use.ICUSTAY_ID.isin(matched.ICUSTAY_ID)].copy()
        for row in removed.itertuples():
            missing = [c for c in COVARIATES if pd.isna(getattr(row, c))]
            reason = "|".join(missing)
            exclusions.append({"analysis": label, "missing_covariates": reason, "failure": int(row.failure_clearance), "exposure": int(row.exposure)})
        old_label = use.old_rise_endpoint.map(lambda x: "unavailable" if pd.isna(x) else str(int(x)))
        for (old, new), group in use.assign(old_label=old_label).groupby(["old_label", "failure_clearance"]):
            transitions.append({"analysis": label, "old_rise_ge0_5": old, "new_failure_clearance": int(new), "n": len(group), "exposed_n": int(group.exposure.sum())})
        for (normal, clear), group in use.groupby(["both_normal", "clearance_ge10"]):
            classes.append({"analysis": label, "both_lactates_le2": bool(normal), "clearance_ge10_percent": bool(clear), "n": len(group), "failures": int(group.failure_clearance.sum())})
    pd.DataFrame(summary).to_csv(args.output_root / "raw_results.csv", index=False)
    pd.DataFrame(fits).to_csv(args.output_root / "adjusted_models.csv", index=False)
    pd.DataFrame(transitions).to_csv(args.output_root / "endpoint_reconciliation.csv", index=False)
    pd.DataFrame(classes).to_csv(args.output_root / "clearance_classification.csv", index=False)
    new_complete = frame.paired & frame.complete_followup
    no_vis_complete = frame[no_vis].notna().all(axis=1)
    all_covariates_complete = frame[COVARIATES].notna().all(axis=1)
    denominator_comparison = []
    for name, original, current in [
        ("paired_cohort", frame.outcome_available_12h.fillna(False).astype(bool), new_complete),
        ("support_model", frame.primary_lactate_sample, new_complete & no_vis_complete),
        ("VIS_model", frame.primary_lactate_sample & frame.VIS_max_0_4h.notna(), new_complete & all_covariates_complete)]:
        assert not (current & ~original).any()
        removed = original & ~current
        denominator_comparison.append({"comparison": name, "old_n": int(original.sum()),
            "old_rise_events": int(frame.loc[original, "old_rise_endpoint"].sum()),
            "new_complete_window_n": int(current.sum()),
            "old_rise_events_in_new_sample": int(frame.loc[current, "old_rise_endpoint"].sum()),
            "new_clearance_failures": int(frame.loc[current, "failure_clearance"].sum()),
            "removed_n": int(removed.sum()),
            "removed_old_rise_events": int(frame.loc[removed, "old_rise_endpoint"].sum()),
            "removed_old_rise_non_events": int(removed.sum() - frame.loc[removed, "old_rise_endpoint"].sum())})
    pd.DataFrame(denominator_comparison).to_csv(args.output_root / "before_after_denominators.csv", index=False)
    excluded = pd.DataFrame(exclusions)
    exclusion_table = excluded.groupby(["analysis", "missing_covariates"], as_index=False).agg(n=("failure", "size"), failures=("failure", "sum"), exposed_n=("exposure", "sum"))
    exclusion_table.to_csv(args.output_root / "covariate_exclusions.csv", index=False)
    paired_any_gap = has_baseline & has_followup
    funnel = [{"step": "waveform_dynamics_eligible", "n": len(frame)},
              {"step": "baseline_in_0_4h", "n": int(has_baseline.sum())},
              {"step": "also_followup_in_4_16h", "n": int(paired_any_gap.sum())},
              {"step": "also_baseline_to_followup_at_least2h", "n": int(frame.paired.sum())},
              {"step": "also_complete_followup_through_hour16", "n": int((frame.paired & frame.complete_followup).sum())}]
    pd.DataFrame(funnel).to_csv(args.output_root / "cohort_flow.csv", index=False)
    frame.to_csv(args.private_cache_root / "clearance_patient_frame.csv", index=False)
    paths = [args.patient_cache, args.support_cache, args.vis_cache, args.vis_audit,
             args.source_output_root / "03_landmark_lactate/lactate_outcomes.csv",
             args.source_output_root / "03_landmark_lactate/linked_lactate_draws.csv.gz",
             args.source_output_manifest, args.vis_audit.parent / "adjusted_models.csv"]
    audit = {"execution": "local analysis from validated saved inputs; no raw clinical or waveform processing",
             "primary_followup_policy": args.primary_followup_policy, "endpoint_status": "new user-requested post-result endpoint",
             "baseline_window_minutes": "[0,240]", "followup_window_minutes": "(240,960]", "minimum_pair_interval_hours": 2,
             "exposure": "unchanged saved exposure; absolute jump>=4; transition gaps<=30 minutes",
             "failure": "followup>0.9*baseline AND not both<=2.0; exact decimal comparison",
             "complete_window_semantics": "observable_end_minute>=960 for failures and successes alike",
             "private_primary_endpoint": "primary_failure_clearance; unavailable pairs/censored stays remain NaN; failure_clearance retains observed-pair classification only for the separately labeled sensitivity",
             "covariates": COVARIATES, "VIS": "unchanged maximum concurrent six-drug VIS in [0,240), continuous linear",
             "interval": "actual baseline-to-last-followup interval in hours, continuous linear",
             "original_covariates": "saved prior-coded flags; lactate-model ESRD and pulmonary definitions retained",
             "input_sha256": {str(p): doses.sha(p) for p in paths}, "code_sha256": doses.sha(__file__),
             "model_code_sha256": doses.sha(support.core.__file__), "fit_check_code_sha256": doses.sha(support.__file__),
             "baseline_and_followup_verified_against_saved_draws": True, "prior_VIS_adjusted_models_reproduced": True, "cohort_flow": funnel,
             "raw_results": summary, "before_after_denominators": denominator_comparison, "original_reports_preserved": True}
    (args.output_root / "analysis_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    report = ["# Failure of lactate clearance", "", "Local analysis of a new endpoint requested after the previous lactate-rise results. Original endpoint reports remain preserved.",
              "", "Primary follow-up policy: **" + args.primary_followup_policy + "**. Both follow-up populations are reported; results are not used to select the definition.",
              "", "Baseline is the last validated lactate in [0,4] ICU hours; follow-up is the last in (4,16]. Their actual separation must be at least2 hours. Failure is clearance<10% AND not both values≤2 mmol/L. Exact decimal comparisons preserve the10% boundary.",
              "", "The unchanged exposure permits transition gaps≤30 minutes. Complete-window results require observation through hour16 for both failures and successes. All-pair results can include earlier ICU departures/deaths and describe last observed paired values rather than an assured complete16-hour trajectory.",
              "", "## Counts and unadjusted results", "", "| Population | N | Failures | Instability failures/N | Stable failures/N | Raw RR | RD (percentage points) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        report.append(f"| {row['analysis']} | {row['n']} | {row['events']} | {row['exposed_events']}/{row['exposed_n']} | {row['unexposed_events']}/{row['unexposed_n']} | {row['raw_RR']:.6f} | {100*row['raw_RD']:.6f} |")
    report += ["", "## Modified Poisson models", "", "| Population | Model | N | Failures | Adjusted RR | Patient-cluster robust95% CI | p |", "|---|---|---:|---:|---:|---|---:|"]
    for row in fits:
        report.append(f"| {row['analysis']} | {row['specification']} | {row['n']} | {row['events']} | {row['adjusted_RR']:.6f} | {row['adjusted_RR_95CI_low']:.6f}–{row['adjusted_RR_95CI_high']:.6f} | {row['p_value']:.6f} |")
    report += ["", "The full model includes age, sex, baseline lactate, mean0–4-hour SpO₂, vasopressor infusion use, mechanical ventilation, CKD, ESRD, chronic pulmonary disease, chronic liver disease, diabetes, maximum0–4-hour VIS, and the actual lactate interval in hours. Existing coding and support definitions are unchanged. SUBJECT_ID-clustered robust errors, finite-sample correction and independent likelihood/covariance calculations follow the existing model code. All requested columns must be retained and the design must have full rank.",
               "", "Covariate exclusions are detailed in covariate_exclusions.csv; raw denominators include all qualifying pairs for each policy. Endpoint reclassification against the original≥0.5 mmol/L rise is in endpoint_reconciliation.csv; normal/clearance components are in clearance_classification.csv. Patient-level records remain private.",
               "", "The interval is measured after the landmark and may reflect evolving illness and clinician sampling. Adjusting for it does not eliminate informative measurement or establish causation/predictive improvement. This endpoint adapts a sepsis definition to HF and a different sampling window; it requires validation. See CLEARANCE_ENDPOINT_ADDENDUM.md for the locked requested definition and source qualifications.",
               "", "## Cohort and endpoint reconciliation", "", "| Qualification | Stays |", "|---|---:|"]
    report.extend(f"| {r['step']} | {r['n']} |" for r in funnel)
    report += ["", "The previous baseline could precede ICU admission; this new baseline must fall in ICU hours0–4. The paired denominator is therefore different from the earlier388-stay primary endpoint cohort. Original baseline selection, original outcomes and mortality results are retained in their existing artifacts.",
               "", "| Population | Old≥0.5 rise endpoint | New clearance failure | Stays | Instability |", "|---|---|---:|---:|---:|"]
    report.extend(f"| {r['analysis']} | {r['old_rise_ge0_5']} | {r['new_failure_clearance']} | {r['n']} | {r['exposed_n']} |" for r in transitions)
    report += ["", "### Sample reconciliation with the prior endpoint", "", "Old events refer to the≥0.5 mmol/L rise; new events refer to failure of clearance. Removed-old-event columns describe population/complete-case exclusions, separately from endpoint reclassification among retained stays.",
               "", "| Sample | Old N/rise events | New complete-window N | Old rise events retained | New failures | Removed old rise events | Removed old rise non-events |", "|---|---:|---:|---:|---:|---:|---:|"]
    report.extend(f"| {r['comparison']} | {r['old_n']}/{r['old_rise_events']} | {r['new_complete_window_n']} | {r['old_rise_events_in_new_sample']} | {r['new_clearance_failures']} | {r['removed_old_rise_events']} | {r['removed_old_rise_non_events']} |" for r in denominator_comparison)
    report += ["", "## Missing covariates", "", "| Population | Missing covariates | Removed stays | Removed failures | Removed instability |", "|---|---|---:|---:|---:|"]
    report.extend(f"| {r.analysis} | {r.missing_covariates.replace('|', ', ')} | {r.n} | {r.failures} | {r.exposed_n} |" for r in exclusion_table.itertuples())
    main_row = next(r for r in fits if r['analysis'] == ('complete_followup_through_hour16' if args.primary_followup_policy == 'complete_window' else 'all_qualifying_pairs') and r['specification'] == 'full_requested_model')
    conclusion = "does not establish an association" if main_row['adjusted_RR_95CI_low'] <= 1 <= main_row['adjusted_RR_95CI_high'] else "supports an observational association"
    report += ["", f"The full requested primary model {conclusion}: RR {main_row['adjusted_RR']:.6f}, 95% CI {main_row['adjusted_RR_95CI_low']:.6f}–{main_row['adjusted_RR_95CI_high']:.6f}. Its confidence interval and all sensitivity results are reported without changing the specification. It has {main_row['events']} events and {main_row['parameters']} parameters ({main_row['events_per_parameter']:.2f} events per parameter); estimates should be interpreted with that limited support.",
               "", "The≥10%/both-normal definition is documented in the [Jones et al. multicenter sepsis trial](https://pmc.ncbi.nlm.nih.gov/articles/PMC2918907/). That is methodological precedent rather than validation of this HF sampling window."]
    (args.output_root / "REPORT.md").write_text("\n".join(report) + "\n")
    print(pd.DataFrame(fits)[["analysis", "specification", "n", "events", "adjusted_RR", "adjusted_RR_95CI_low", "adjusted_RR_95CI_high", "p_value"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
