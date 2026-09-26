"""Run the frozen post-hoc MIMIC follow-up and one eICU external validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physiograph.analysis.shock_signal_discovery import (
    pair_domain_events,
    rolling_oliguria_events,
    sustained_hypotension_events,
    trajectory_features,
)
from physiograph.analysis.spo2_variability_validation import (
    fit_fixed_scale_model,
    fixed_exposure_parameters,
    stratified_bootstrap_fixed,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT.parents[1]
PRIVATE = DRIVE / "Data/PhysioGraph_Biological_Discovery_20260905"
EICU = DRIVE / "Data/eICU/Full"
OUT = ROOT / "research/spo2_variability_validation"
PROTOCOL = OUT / "protocol.json"
DOC = ROOT / "docs/SPO2_VARIABILITY_TARGETED_VALIDATION_PROTOCOL.md"
DISCOVERY_DB = PRIVATE / "objective_shock_single_signal_discovery.duckdb"
PHASE_DB = PRIVATE / "masked_pulsatility_phase_a.duckdb"
DB = PRIVATE / "spo2_variability_validation.duckdb"
CODE_FILES = [
    "scripts/build_biological_notebook.py",
    "scripts/run_spo2_variability_validation.py",
    "scripts/spo2_variability_validation_cells.py",
    "src/physiograph/analysis/spo2_variability_validation.py",
    "src/physiograph/analysis/shock_signal_discovery.py",
    "tests/unit/test_spo2_variability_validation.py",
]
BASE_COVARIATES = ["age","male_sex","hr_level","sbp_level","map_level","resp_rate_level","spo2_level","temperature_c_level","baseline_creatinine","baseline_lactate","lactate_observed","bp_measurement_density","urine_measurement_density"]


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def write_json(name: str, value: object) -> None:
    OUT.mkdir(parents=True,exist_ok=True); (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+"\n")


def q(path: Path) -> str:
    return "'"+str(path).replace("'","''")+"'"


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}",flush=True)


def load_mimic() -> tuple[pd.DataFrame,pd.DataFrame,dict[str,float]]:
    con=duckdb.connect(str(DISCOVERY_DB),read_only=True)
    base=con.execute("SELECT * FROM model_base").df(); features=con.execute("SELECT stay_id,value,bins FROM feature_long WHERE feature='spo2_variability'").df(); con.close()
    feature_columns={"value":"spo2_variability"}
    if "spo2_bins" not in base.columns:
        feature_columns["bins"]="spo2_bins"
    else:
        features=features.drop(columns="bins")
    frame=base.merge(features.rename(columns=feature_columns),on="stay_id",how="left",validate="one_to_one")
    parameters=fixed_exposure_parameters(frame.spo2_variability)
    return frame,features,parameters


def freeze() -> None:
    frame,_,parameters=load_mimic()
    code_hashes={name:digest(ROOT/name) for name in CODE_FILES}
    config={"freeze_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(PROTOCOL),"documentation_sha256":digest(DOC),"code_hashes":code_hashes,"exposure_parameters":parameters,"mimic_at_risk_n":len(frame),"mimic_exposure_observed_n":int(frame.spo2_variability.notna().sum()),"eicu_association_inspected":False}
    write_json("external_validation_config.json",config)
    write_json("external_validation_lock.json",{"lock_utc":config["freeze_utc"],"config_sha256":digest(OUT/"external_validation_config.json"),"first_eicu_association_read_utc":None,"eicu_association_inspected":False})
    log(f"external specification locked before eICU association read: {digest(OUT/'external_validation_config.json')}")


def mimic_followup(frame: pd.DataFrame, parameters: dict[str,float]) -> dict[str,object]:
    primary,_=fit_fixed_scale_model(frame,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters)
    lead,_=fit_fixed_scale_model(frame,feature="spo2_variability",outcome="objective_shock_lead4",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters)
    bootstrap=stratified_bootstrap_fixed(frame,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters,replicates=1000)
    density,_=fit_fixed_scale_model(frame,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=[*BASE_COVARIATES,"spo2_bins"],exposure_parameters=parameters)
    complete=frame.dropna(subset=["spo2_variability",*BASE_COVARIATES]); complete_result,_=fit_fixed_scale_model(complete,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters)
    observed=frame.dropna(subset=["spo2_variability"]).copy(); observed["quartile"]=pd.qcut(observed.spo2_variability,4,labels=False,duplicates="drop")+1
    quartiles=observed.groupby("quartile",observed=True).objective_shock.agg(["size","sum","mean"]).reset_index().rename(columns={"size":"n","sum":"events","mean":"risk"}).to_dict("records")
    strata=[]
    for bins,group in observed.groupby("spo2_bins"):
        if len(group)>=300 and group.objective_shock.sum()>=20:
            try:
                result,_=fit_fixed_scale_model(group,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters); result["bins"]=int(bins); strata.append(result)
            except (ValueError,np.linalg.LinAlgError): pass
    result={"primary":primary,"lead":lead,"bootstrap":bootstrap,"density_adjusted":density,"complete_case":complete_result,"quartiles":quartiles,"density_strata":strata}
    write_json("mimic_targeted_followup.json",result); return result


def extract_eicu() -> tuple[pd.DataFrame,dict[str,int]]:
    con=duckdb.connect(str(DB)); con.execute("SET threads=4"); con.execute("SET memory_limit='5GB'"); con.execute(f"ATTACH {q(PHASE_DB)} AS phase (READ_ONLY)")
    con.execute(f"""CREATE OR REPLACE TABLE cohort AS SELECT c.*,try_cast(p.age AS DOUBLE) age,CASE upper(trim(p.gender)) WHEN 'MALE' THEN 1 WHEN 'M' THEN 1 WHEN 'FEMALE' THEN 0 WHEN 'F' THEN 0 END male_sex FROM phase.eicu_cohort c JOIN read_csv({q(EICU/'patient.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.patientunitstayid AS BIGINT)=c.stay_id""")
    con.execute(f"""CREATE OR REPLACE TABLE periodic AS SELECT c.stay_id,try_cast(v.observationoffset AS DOUBLE) AS "minute",try_cast(v.sao2 AS DOUBLE) AS spo2,try_cast(v.heartrate AS DOUBLE) AS hr,try_cast(v.respiration AS DOUBLE) AS resp_rate,try_cast(v.temperature AS DOUBLE) AS temperature_c FROM read_csv({q(EICU/'vitalPeriodic.csv')},header=true,all_varchar=true,parallel=true) v JOIN cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id WHERE try_cast(v.observationoffset AS DOUBLE) BETWEEN 0 AND 960""")
    con.execute(f"""CREATE OR REPLACE TABLE cuff_raw AS SELECT c.stay_id,try_cast(v.observationoffset AS DOUBLE) AS "minute",try_cast(v.observationoffset AS DOUBLE) AS available_minute,try_cast(v.noninvasivesystolic AS DOUBLE) AS sbp,try_cast(v.noninvasivemean AS DOUBLE) AS map FROM read_csv({q(EICU/'vitalAperiodic.csv')},header=true,all_varchar=true,parallel=true) v JOIN cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id WHERE try_cast(v.observationoffset AS DOUBLE) BETWEEN 0 AND 960 UNION ALL SELECT c.stay_id,try_cast(n.nursingchartoffset AS DOUBLE),greatest(try_cast(n.nursingchartoffset AS DOUBLE),try_cast(n.nursingchartentryoffset AS DOUBLE)),CASE WHEN lower(n.nursingchartcelltypevalname) LIKE '%systolic%' THEN try_cast(n.nursingchartvalue AS DOUBLE) END,CASE WHEN lower(n.nursingchartcelltypevalname) LIKE '%mean%' THEN try_cast(n.nursingchartvalue AS DOUBLE) END FROM read_csv({q(EICU/'nurseCharting.csv')},header=true,all_varchar=true,parallel=true) n JOIN cohort c ON try_cast(n.patientunitstayid AS BIGINT)=c.stay_id WHERE lower(n.nursingchartcelltypevallabel)='non-invasive bp' AND try_cast(n.nursingchartoffset AS DOUBLE) BETWEEN 0 AND 960""")
    con.execute("""CREATE OR REPLACE TABLE cuff_hourly AS SELECT stay_id,floor("minute"/60) AS hour,median(sbp) FILTER(WHERE sbp BETWEEN 40 AND 260) AS sbp,median(map) FILTER(WHERE map BETWEEN 30 AND 200) AS map,max(available_minute) AS available_minute FROM cuff_raw GROUP BY 1,2""")
    lab_case="CASE WHEN lower(trim(l.labname))='lactate' THEN 'lactate' WHEN lower(trim(l.labname))='creatinine' THEN 'creatinine' WHEN lower(trim(l.labname)) IN ('alt','alt (sgpt)') THEN 'alt' WHEN lower(trim(l.labname))='ph' THEN 'ph' END"
    con.execute(f"""CREATE OR REPLACE TABLE labs AS SELECT c.stay_id,{lab_case} AS marker,greatest(try_cast(l.labresultoffset AS DOUBLE),coalesce(try_cast(l.labresultrevisedoffset AS DOUBLE),try_cast(l.labresultoffset AS DOUBLE))) AS "minute",try_cast(l.labresult AS DOUBLE) AS value FROM read_csv({q(EICU/'lab.csv')},header=true,all_varchar=true,parallel=true) l JOIN cohort c ON try_cast(l.patientunitstayid AS BIGINT)=c.stay_id WHERE {lab_case} IS NOT NULL AND greatest(try_cast(l.labresultoffset AS DOUBLE),coalesce(try_cast(l.labresultrevisedoffset AS DOUBLE),try_cast(l.labresultoffset AS DOUBLE))) BETWEEN 0 AND 960""")
    con.execute(f"""CREATE OR REPLACE TABLE urine AS SELECT c.stay_id,floor(greatest(try_cast(i.intakeoutputoffset AS DOUBLE),try_cast(i.intakeoutputentryoffset AS DOUBLE))/60) AS hour,greatest(try_cast(i.intakeoutputoffset AS DOUBLE),try_cast(i.intakeoutputentryoffset AS DOUBLE)) AS available_minute,sum(try_cast(i.cellvaluenumeric AS DOUBLE)) AS value FROM read_csv({q(EICU/'intakeOutput.csv')},header=true,all_varchar=true,parallel=true) i JOIN cohort c ON try_cast(i.patientunitstayid AS BIGINT)=c.stay_id WHERE regexp_matches(lower(coalesce(i.celllabel,'')||' '||coalesce(i.cellpath,'')),'urine|foley|voided|nephrostomy|urostomy') AND NOT regexp_matches(lower(coalesce(i.celllabel,'')||' '||coalesce(i.cellpath,'')),'culture|specimen|appearance|color|count|occurrence|incontin|mixed|total|subtotal|cumulative|daily|net') AND try_cast(i.cellvaluenumeric AS DOUBLE) BETWEEN 0 AND 5000 AND greatest(try_cast(i.intakeoutputoffset AS DOUBLE),try_cast(i.intakeoutputentryoffset AS DOUBLE)) BETWEEN 0 AND 960 GROUP BY 1,2,3""")
    drug="norepinephrine|noradrenaline|levophed|epinephrine|adrenalin|dopamine|dobutamine|dobutrex|milrinone|primacor|vasopressin|pitressin|phenylephrine|neo-synephrine|neosynephrine"
    con.execute(f"""CREATE OR REPLACE TABLE support AS SELECT c.stay_id,try_cast(i.infusionoffset AS DOUBLE) event_minute,'continuous_support' component FROM read_csv({q(EICU/'infusionDrug.csv')},header=true,all_varchar=true,parallel=true) i JOIN cohort c ON try_cast(i.patientunitstayid AS BIGINT)=c.stay_id WHERE regexp_matches(lower(i.drugname),'{drug}') AND coalesce(try_cast(i.drugrate AS DOUBLE),try_cast(i.infusionrate AS DOUBLE),try_cast(regexp_extract(coalesce(i.drugrate,i.infusionrate),'[0-9]+[.]?[0-9]*') AS DOUBLE))>0 AND try_cast(i.infusionoffset AS DOUBLE) BETWEEN 0 AND 960""")
    mcs="iabp|intra.?aortic balloon|impella|ecmo|extracorporeal membrane oxygenation"
    con.execute(f"""CREATE OR REPLACE TABLE mcs AS SELECT stay_id,event_minute,'mcs' component FROM (SELECT c.stay_id,try_cast(t.treatmentoffset AS DOUBLE) event_minute,row_number() OVER(PARTITION BY c.stay_id ORDER BY try_cast(t.treatmentoffset AS DOUBLE)) rn FROM read_csv({q(EICU/'treatment.csv')},header=true,all_varchar=true,parallel=true) t JOIN cohort c ON try_cast(t.patientunitstayid AS BIGINT)=c.stay_id WHERE regexp_matches(lower(t.treatmentstring),'{mcs}') AND NOT regexp_matches(lower(t.treatmentstring),'remov|explant|discontinu') AND try_cast(t.treatmentoffset AS DOUBLE) BETWEEN 0 AND 960) WHERE rn=1""")
    cohort=con.execute("SELECT * FROM cohort").df(); periodic=con.execute("SELECT * FROM periodic").df(); cuff=con.execute("SELECT * FROM cuff_hourly").df(); labs=con.execute("SELECT * FROM labs").df(); urine=con.execute("SELECT * FROM urine").df(); urine=urine.groupby(["stay_id","hour"],as_index=False).agg(available_minute=("available_minute","max"),value=("value","sum")); pressure=pd.concat([sustained_hypotension_events(cuff),con.execute("SELECT * FROM support UNION ALL SELECT * FROM mcs").df()],ignore_index=True)
    hourly=[]
    for signal,bounds in {"spo2":(50,100),"hr":(20,250),"resp_rate":(4,80),"temperature_c":(30,43)}.items():
        part=periodic.loc[periodic.minute.between(0,240)&periodic[signal].between(*bounds),["stay_id","minute",signal]].copy(); part["hour"]=(part.minute//60).astype(int); part=part.groupby(["stay_id","hour"],as_index=False)[signal].median().rename(columns={signal:"value"}); part["signal"]=signal; hourly.append(part[["stay_id","signal","hour","value"]])
    for signal in ["sbp","map"]:
        part=cuff.loc[cuff.hour.between(0,4)&cuff.available_minute.le(240)&cuff[signal].notna(),["stay_id","hour",signal]].rename(columns={signal:"value"}); part["signal"]=signal; hourly.append(part[["stay_id","signal","hour","value"]])
    feature=trajectory_features(pd.concat(hourly,ignore_index=True)); levels=feature.pivot(index="stay_id",columns="signal",values="level").add_suffix("_level").reset_index(); spo2=feature.loc[:,["stay_id","signal","variability","bins"]].query("signal=='spo2'").rename(columns={"variability":"spo2_variability","bins":"spo2_bins"}).drop(columns="signal")
    baseline=labs.loc[labs.minute.le(240)].sort_values("minute").groupby(["stay_id","marker"]).tail(1).pivot(index="stay_id",columns="marker",values="value").add_prefix("baseline_").reset_index(); labs=labs.merge(baseline,on="stay_id",how="left"); perf=[]
    for marker,group in labs.loc[labs.minute.gt(240)].groupby("marker"):
        if marker=="lactate": mask=group.value.ge(4)|(group.baseline_lactate.notna()&group.value.ge(2)&group.value.sub(group.baseline_lactate).ge(.5))
        elif marker=="creatinine": mask=group.baseline_creatinine.notna()&(group.value.sub(group.baseline_creatinine).ge(.3)|group.value.div(group.baseline_creatinine).ge(1.5))
        elif marker=="alt": mask=group.baseline_alt.notna()&((group.baseline_alt.le(200)&group.value.gt(200))|group.value.div(group.baseline_alt).ge(3))
        else: mask=group.baseline_ph.notna()&group.baseline_ph.ge(7.2)&group.value.lt(7.2)
        current=group.loc[mask,["stay_id","minute"]].rename(columns={"minute":"event_minute"}); current["component"]=marker; perf.append(current)
    oliguria=rolling_oliguria_events(urine.rename(columns={"hour":"hour","value":"value"}),start_hour=0,end_hour=16); perfusion=pd.concat([*perf,oliguria],ignore_index=True)
    pre_pressure=set(pressure.loc[pressure.event_minute.le(360),"stay_id"]); pre_perfusion=set(perfusion.loc[perfusion.event_minute.le(360),"stay_id"]); at_risk=cohort.loc[cohort.followup_end_offset_minutes.gt(360)&~cohort.death_offset_minutes.between(0,360).fillna(False)&~cohort.stay_id.isin(pre_pressure|pre_perfusion)].copy()
    full=pair_domain_events(pressure,perfusion,lower_minute=360,upper_minute=960); outcomes=at_risk[["stay_id"]].merge(full[["stay_id","event_minute"]],on="stay_id",how="left"); outcomes["objective_shock"]=outcomes.event_minute.notna().astype(int); outcomes["objective_shock_lead4"]=outcomes.event_minute.gt(480).astype(int)
    urine_density=urine.loc[urine.hour.between(0,4)&urine.available_minute.le(240)].groupby("stay_id").hour.nunique().rename("urine_measurement_density"); bp_density=cuff.loc[cuff.hour.between(0,4)&cuff.available_minute.le(240)].groupby("stay_id").hour.nunique().rename("bp_measurement_density")
    model=at_risk.merge(outcomes,on="stay_id").merge(levels,on="stay_id",how="left").merge(spo2,on="stay_id",how="left").merge(baseline,on="stay_id",how="left").merge(bp_density,on="stay_id",how="left").merge(urine_density,on="stay_id",how="left"); model["lactate_observed"]=model.get("baseline_lactate",pd.Series(index=model.index,dtype=float)).notna().astype(int); model["hospital_id"]=model.hospital_id.astype(str)
    counts={"cohort":len(cohort),"at_risk":len(model),"events":int(model.objective_shock.sum()),"lead_events":int(model.objective_shock_lead4.sum()),"spo2_observed":int(model.spo2_variability.notna().sum()),"hospitals":int(model.hospital_id.nunique())}; con.close(); return model,counts


def validate() -> None:
    config=json.loads((OUT/"external_validation_config.json").read_text()); lock=json.loads((OUT/"external_validation_lock.json").read_text())
    assert digest(OUT/"external_validation_config.json")==lock["config_sha256"]
    assert all(digest(ROOT/name)==expected for name,expected in config["code_hashes"].items())
    frame,_,parameters=load_mimic(); mimic=mimic_followup(frame,parameters); model,counts=extract_eicu(); write_json("eicu_support_counts.json",counts)
    read_utc=datetime.now(timezone.utc).isoformat(); lock.update(first_eicu_association_read_utc=read_utc,eicu_association_inspected=True); write_json("external_validation_lock.json",lock); log("FIRST AND ONLY eICU SpO2-variability association read")
    primary,_=fit_fixed_scale_model(model,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters,cluster="hospital_id")
    lead,_=fit_fixed_scale_model(model,feature="spo2_variability",outcome="objective_shock_lead4",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters,cluster="hospital_id")
    hc0,_=fit_fixed_scale_model(model,feature="spo2_variability",outcome="objective_shock",adjustment_covariates=BASE_COVARIATES,exposure_parameters=parameters)
    confirmed=primary["rr_per_mimic_sd"]>1 and primary["ci_low"]>1 and primary["p_value"]<.05 and primary["events"]>=50 and primary["hospitals"]>=20 and primary["coverage"]>=.70
    result={"classification":"EXTERNALLY_VALIDATED_POST_HOC_SIGNAL" if confirmed else "POST_HOC_SIGNAL_NOT_EXTERNALLY_VALIDATED","primary":primary,"lead":lead,"hc0_sensitivity":hc0,"support":counts,"mimic":mimic,"first_eicu_association_read_utc":read_utc,"original_discovery_gate_remains_failed":True}; write_json("validation_results.json",result)
    fig,ax=plt.subplots(figsize=(7,3.5)); rr=[mimic["primary"]["rr_per_mimic_sd"],primary["rr_per_mimic_sd"]]; lo=[mimic["primary"]["ci_low"],primary["ci_low"]]; hi=[mimic["primary"]["ci_high"],primary["ci_high"]]; ax.errorbar(rr,[1,0],xerr=[np.array(rr)-lo,np.array(hi)-rr],fmt="o");ax.axvline(1,color="black",lw=.8);ax.set_yticks([1,0],["MIMIC post-hoc follow-up","eICU external validation"]);ax.set_xlabel("Adjusted RR per MIMIC SD greater SpO2 variability");fig.tight_layout();fig.savefig(OUT/"cross_database_forest.png",dpi=180);plt.close(fig)
    report=f"""# Targeted SpO2-variability replication\n\n**{result['classification']}**\n\nThis user-authorized follow-up is post hoc because the original discovery RR of 1.251 missed the frozen 1.30 advancement gate. The original decision is unchanged.\n\nMIMIC: adjusted RR {mimic['primary']['rr_per_mimic_sd']:.3f} ({mimic['primary']['ci_low']:.3f}–{mimic['primary']['ci_high']:.3f}); 1,000-bootstrap median {mimic['bootstrap']['median_rr']:.3f} ({mimic['bootstrap']['ci_low']:.3f}–{mimic['bootstrap']['ci_high']:.3f}).\n\neICU: {primary['n']:,} observed stays, {primary['events']} events, {primary['hospitals']} hospitals, coverage {primary['coverage']:.1%}; hospital-clustered adjusted RR {primary['rr_per_mimic_sd']:.3f} ({primary['ci_low']:.3f}–{primary['ci_high']:.3f}), p={primary['p_value']:.4g}. Lead-window RR {lead['rr_per_mimic_sd']:.3f} ({lead['ci_low']:.3f}–{lead['ci_high']:.3f}).\n\nThe outcome is a constructed pressure/support-plus-hypoperfusion composite, not adjudicated cardiogenic shock. Association does not establish causality or clinical utility. Execution environment: local.\n"""; (OUT/"FINAL_REPORT.md").write_text(report); write_json("run_status.json",{"status":"completed","classification":result["classification"],"environment":"local","end_utc":datetime.now(timezone.utc).isoformat()}); print(json.dumps(result,indent=2,default=str))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--freeze-only",action="store_true"); args=parser.parse_args()
    if args.freeze_only: freeze()
    else: validate()


if __name__=="__main__": main()
