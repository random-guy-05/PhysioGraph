"""Outcome-blind Phase A masked-pulsatility notebook cells."""

SETUP = r'''
import hashlib,importlib.util,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
EICU=DRIVE/'Data/eICU/Full';MIMIC=DRIVE/'Data/MIMIC/Full'
OUT=PROJECT/'research/masked_pulsatility';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.masked_pulsatility import assign_pseudo_anchors,build_hourly_bp_series,detect_incident_crossings,followup_observation_flags,intervention_free,latest_preanchor_lactate,phase_a_decision,sha256_file
def mp_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def mp_status(stage,state='running',**extra):
 row=dict(status=state,stage=stage,environment='google_colab' if IN_COLAB else 'local',updated_utc=datetime.now(timezone.utc).isoformat(),phase='A',post_anchor_lactate_values_inspected=False,clinical_outcomes_inspected=False,effect_estimates_computed=False,goal_complete=False);row.update(extra);mp_json('run_status.json',row);print(stage,state,flush=True)
def mp_register(name,frame):
 frame=frame.copy()
 for col in frame.select_dtypes(include=['string']).columns:frame[col]=frame[col].astype(object)
 con.register(name,frame)
def mp_exists(name):return bool(con.execute('SELECT count(*) FROM information_schema.tables WHERE table_schema=\'main\' AND table_name=?',[name]).fetchone()[0])
def mp_manifest(path):
 stat=path.stat();return dict(path=str(path),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
plan=PROJECT/'docs/MASKED_PULSATILITY_FEASIBILITY_PLAN.md'
assert sha256_file(plan)==MASKED_PULSATILITY_FEASIBILITY_SHA256
lock=OUT/'protocol.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==MASKED_PULSATILITY_FEASIBILITY_SHA256
else:mp_json('protocol.json',dict(sha256=MASKED_PULSATILITY_FEASIBILITY_SHA256,frozen_utc=datetime.now(timezone.utc).isoformat(),phase_a_protocol_frozen_before_scan=True,outcomes_prohibited=True))
for required in [EICU/'vitalPeriodic.csv',EICU/'lab.csv',EICU/'infusionDrug.csv',EICU/'treatment.csv',MIMIC/'chartevents.csv',MIMIC/'labevents.csv',MIMIC/'d_labitems.csv',MIMIC/'inputevents.csv',MIMIC/'procedureevents.csv',PRIVATE/'spo2_context.duckdb']:
 if not required.exists():raise FileNotFoundError(required)
mp_json('source_manifest.json',{'phase_a_sources':[mp_manifest(p) for p in [EICU/'vitalPeriodic.csv',EICU/'lab.csv',EICU/'infusionDrug.csv',EICU/'treatment.csv',MIMIC/'chartevents.csv',MIMIC/'labevents.csv',MIMIC/'d_labitems.csv',MIMIC/'inputevents.csv',MIMIC/'procedureevents.csv',PRIVATE/'spo2_context.duckdb']]})
try:con.close()
except (NameError,AttributeError):pass
db=PRIVATE/'masked_pulsatility_phase_a.duckdb';con=duckdb.connect(str(db));con.execute('SET threads=2');con.execute("SET memory_limit='3GB'");con.execute('SET enable_progress_bar=false')
if not mp_exists('eicu_cohort'):
 con.execute(f"ATTACH '{str(PRIVATE/'spo2_context.duckdb').replace(chr(39),chr(39)*2)}' AS prior (READ_ONLY)")
 con.execute('CREATE TABLE eicu_cohort AS SELECT stay_id,person_id,hospital_id,death_offset_minutes,followup_end_offset_minutes FROM prior.eicu_cohort')
 con.execute('CREATE TABLE mimic_cohort AS SELECT stay_id,person_id,hadm_id,admit_time,death_offset_minutes,followup_end_offset_minutes FROM prior.mimic_cohort')
 con.execute('DETACH prior')
assert con.execute('SELECT count(*) FROM eicu_cohort').fetchone()[0]==12028
assert con.execute('SELECT count(*) FROM mimic_cohort').fetchone()[0]==17758
mp_status('setup_complete')
'''

BP_EXTRACT = r'''
mp_status('blood_pressure_extraction')
if not mp_exists('eicu_bp_triplets'):
 con.execute(f"""
 CREATE TABLE eicu_bp_triplets AS
 SELECT 'eicu' dataset,c.stay_id,try_cast(v.observationoffset AS DOUBLE) event_minute,
        'invasive_arterial' AS "source",try_cast(v.systemicsystolic AS DOUBLE) sbp,
        try_cast(v.systemicdiastolic AS DOUBLE) dbp,try_cast(v.systemicmean AS DOUBLE) AS "map"
 FROM read_csv('{str(EICU/'vitalPeriodic.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) v
 JOIN eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240
   AND try_cast(v.systemicsystolic AS DOUBLE) IS NOT NULL
   AND try_cast(v.systemicdiastolic AS DOUBLE) IS NOT NULL
   AND try_cast(v.systemicmean AS DOUBLE) IS NOT NULL
 """)
 print('eICU raw BP triplets',con.execute('SELECT count(*) FROM eicu_bp_triplets').fetchone()[0],flush=True)
if not mp_exists('mimic_bp_triplets'):
 con.execute(f"""
 CREATE TABLE mimic_bp_triplets AS
 WITH selected AS (
  SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
   CASE WHEN try_cast(v.itemid AS BIGINT) IN (220050,220051,220052) THEN 'invasive_arterial' ELSE 'noninvasive_cuff' END AS "source",
   try_cast(v.itemid AS BIGINT) itemid,try_cast(v.valuenum AS DOUBLE) val
  FROM read_csv('{str(MIMIC/'chartevents.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) v
  JOIN mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
  WHERE try_cast(v.itemid AS BIGINT) IN (220050,220051,220052,220179,220180,220181)
    AND coalesce(try_cast(v.warning AS BIGINT),0)<>1
    AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0>=0
    AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0<240
 ), pivoted AS (
  SELECT stay_id,event_minute,"source",
   median(val) FILTER (WHERE itemid IN (220050,220179)) sbp,
   median(val) FILTER (WHERE itemid IN (220051,220180)) dbp,
   median(val) FILTER (WHERE itemid IN (220052,220181)) AS "map"
  FROM selected GROUP BY stay_id,event_minute,"source"
 )
 SELECT 'mimic' dataset,* FROM pivoted WHERE sbp IS NOT NULL AND dbp IS NOT NULL AND "map" IS NOT NULL
 """)
 print('MIMIC raw BP triplets',con.execute('SELECT count(*) FROM mimic_bp_triplets').fetchone()[0],flush=True)
if not mp_exists('hourly_bp'):
 triplets=con.execute('SELECT * FROM eicu_bp_triplets UNION ALL SELECT * FROM mimic_bp_triplets').df()
 hourly=build_hourly_bp_series(triplets)
 mp_register('_hourly',hourly);con.execute('CREATE TABLE hourly_bp AS SELECT * FROM _hourly');con.unregister('_hourly')
 print('valid source-consistent hourly BP bins',len(hourly),flush=True)
if not mp_exists('bp_candidates'):
 hourly=con.execute('SELECT * FROM hourly_bp ORDER BY dataset,stay_id,hour').df()
 crossings,initially_low=detect_incident_crossings(hourly)
 first_low=set((str(r.dataset),int(r.stay_id)) for r in crossings.itertuples())
 initial_by_dataset=set()
 for dataset,group in hourly.groupby('dataset'):
  _,low=detect_incident_crossings(group);initial_by_dataset.update((str(dataset),int(x)) for x in low)
 crossing_keys=set((str(r.dataset),int(r.stay_id)) for r in crossings.itertuples())
 control=hourly.loc[[((str(r.dataset),int(r.stay_id)) not in crossing_keys and (str(r.dataset),int(r.stay_id)) not in initial_by_dataset and float(r.prpp)>=0.25) for r in hourly.itertuples()]].copy()
 crossings=crossings.copy();crossings['candidate_type']='crossing';control['candidate_type']='control'
 candidates=pd.concat([crossings,control],ignore_index=True)
 candidates['candidate_id']=candidates.apply(lambda r:f"{r.dataset}|{int(r.stay_id)}|{int(r.hour)}|{r.candidate_type}",axis=1)
 eicu_meta=con.execute('SELECT stay_id,person_id,hospital_id,death_offset_minutes,followup_end_offset_minutes FROM eicu_cohort').df();eicu_meta['dataset']='eicu'
 mimic_meta=con.execute("SELECT stay_id,cast(person_id AS VARCHAR) person_id,'MIMIC' hospital_id,death_offset_minutes,followup_end_offset_minutes FROM mimic_cohort").df();mimic_meta['dataset']='mimic'
 meta=pd.concat([eicu_meta,mimic_meta],ignore_index=True)
 candidates=candidates.merge(meta,on=['dataset','stay_id'],how='left',validate='many_to_one')
 candidates=candidates.loc[candidates['sbp'].ge(90)&candidates['map'].ge(65)&candidates['followup_end_offset_minutes'].ge(candidates['anchor_minute'])&(candidates['death_offset_minutes'].isna()|candidates['death_offset_minutes'].gt(candidates['anchor_minute']))].copy()
 mp_register('_candidates',candidates);con.execute('CREATE TABLE bp_candidates AS SELECT * FROM _candidates');con.unregister('_candidates')
 print('compensated BP candidates',len(candidates),flush=True)
source=con.execute('SELECT dataset,"source",count(DISTINCT stay_id) stays,count(*) hourly_bins FROM hourly_bp GROUP BY dataset,"source" ORDER BY dataset,"source"').df()
source.to_csv(OUT/'source_mapping.csv',index=False)
mp_status('blood_pressure_complete')
'''

BASELINE_SCREEN = r'''
mp_status('preanchor_screening')
if not mp_exists('first_intervention'):
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS mimic_first_pressor AS
 SELECT c.stay_id,min(date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0) first_pressor
 FROM read_csv('{str(MIMIC/'inputevents.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) i
 JOIN mimic_cohort c ON try_cast(i.stay_id AS BIGINT)=c.stay_id
 WHERE try_cast(i.itemid AS BIGINT) IN (221906,221289,221662,221653,221749,222315,221986)
  AND coalesce(try_cast(i.rate AS DOUBLE),try_cast(i.amount AS DOUBLE),0)>0
  AND date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0<240
 GROUP BY c.stay_id
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS mimic_first_mcs AS
 SELECT c.stay_id,min(date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0) first_mcs
 FROM read_csv('{str(MIMIC/'procedureevents.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) p
 JOIN mimic_cohort c ON try_cast(p.stay_id AS BIGINT)=c.stay_id
 WHERE try_cast(p.itemid AS BIGINT) IN (224272,228169,229529,229530)
  AND date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0<240
 GROUP BY c.stay_id
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS eicu_first_pressor AS
 SELECT c.stay_id,min(try_cast(i.infusionoffset AS DOUBLE)) first_pressor
 FROM read_csv('{str(EICU/'infusionDrug.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) i
 JOIN eicu_cohort c ON try_cast(i.patientunitstayid AS BIGINT)=c.stay_id
 WHERE regexp_matches(lower(coalesce(i.drugname,'')),'norepinephrine|levophed|epinephrine|adrenalin|phenylephrine|neosynephrine|vasopressin|dopamine|dobutamine|milrinone')
  AND coalesce(try_cast(i.drugrate AS DOUBLE),try_cast(i.infusionrate AS DOUBLE),try_cast(regexp_extract(coalesce(i.drugrate,i.infusionrate,''),'[-+]?[0-9]*\\.?[0-9]+',0) AS DOUBLE),0)>0
  AND try_cast(i.infusionoffset AS DOUBLE)<240
 GROUP BY c.stay_id
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS eicu_first_mcs AS
 SELECT c.stay_id,min(try_cast(t.treatmentoffset AS DOUBLE)) first_mcs
 FROM read_csv('{str(EICU/'treatment.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) t
 JOIN eicu_cohort c ON try_cast(t.patientunitstayid AS BIGINT)=c.stay_id
 WHERE regexp_matches(lower(coalesce(t.treatmentstring,'')),'intra.?aortic balloon|iabp|impella|ventricular assist|ecmo|extracorporeal membrane')
  AND NOT regexp_matches(lower(coalesce(t.treatmentstring,'')),'remove|removal|explant|discontinu')
  AND try_cast(t.treatmentoffset AS DOUBLE)<240
 GROUP BY c.stay_id
 """)
 con.execute("""
 CREATE TABLE IF NOT EXISTS first_intervention AS
 SELECT 'eicu' dataset,c.stay_id,p.first_pressor,m.first_mcs FROM eicu_cohort c LEFT JOIN eicu_first_pressor p USING(stay_id) LEFT JOIN eicu_first_mcs m USING(stay_id)
 UNION ALL
 SELECT 'mimic' dataset,c.stay_id,p.first_pressor,m.first_mcs FROM mimic_cohort c LEFT JOIN mimic_first_pressor p USING(stay_id) LEFT JOIN mimic_first_mcs m USING(stay_id)
 """)
if not mp_exists('preanchor_lactate'):
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS eicu_preanchor_lactate AS
 SELECT 'eicu' dataset,b.stay_id,b.candidate_id,b.anchor_minute,
  try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
  try_cast(l.labresult AS DOUBLE) AS "value",coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,'') unit
 FROM bp_candidates b
 JOIN read_csv('{str(EICU/'lab.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=b.stay_id
 WHERE b.dataset='eicu' AND lower(trim(l.labname))='lactate'
  AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN b.anchor_minute-240 AND b.anchor_minute
  AND try_cast(l.labresultrevisedoffset AS DOUBLE)<=b.anchor_minute
  AND regexp_matches(lower(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,'')),'mmol')
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS mimic_lactate_items AS
 SELECT DISTINCT try_cast(itemid AS BIGINT) itemid
 FROM read_csv('{str(MIMIC/'d_labitems.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true)
 WHERE lower(trim(label))='lactate' AND regexp_matches(lower(coalesce(fluid,'')),'blood')
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS mimic_preanchor_lactate AS
 SELECT 'mimic' dataset,b.stay_id,b.candidate_id,b.anchor_minute,
  date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
  date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
  try_cast(l.valuenum AS DOUBLE) AS "value",coalesce(l.valueuom,'') unit
 FROM bp_candidates b JOIN mimic_cohort c USING(stay_id)
 JOIN read_csv('{str(MIMIC/'labevents.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id
 JOIN mimic_lactate_items d ON try_cast(l.itemid AS BIGINT)=d.itemid
 WHERE b.dataset='mimic'
  AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN b.anchor_minute-240 AND b.anchor_minute
  AND date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0<=b.anchor_minute
  AND regexp_matches(lower(coalesce(l.valueuom,'')),'mmol')
 """)
 con.execute('CREATE TABLE IF NOT EXISTS preanchor_lactate AS SELECT * FROM eicu_preanchor_lactate UNION ALL SELECT * FROM mimic_preanchor_lactate')
if not mp_exists('final_anchors'):
 candidates=con.execute('SELECT * FROM bp_candidates').df();measurements=con.execute('SELECT dataset,stay_id,candidate_id,anchor_minute,event_minute,available_minute,"value" FROM preanchor_lactate').df()
 baseline=latest_preanchor_lactate(measurements,candidates[['dataset','stay_id','candidate_id','anchor_minute']])
 candidates=candidates.merge(baseline.drop(columns=['anchor_minute']),on=['dataset','stay_id','candidate_id'],how='left',validate='one_to_one')
 interventions=con.execute('SELECT * FROM first_intervention').df();candidates=candidates.merge(interventions,on=['dataset','stay_id'],how='left',validate='many_to_one')
 candidates['intervention_free']=candidates.apply(lambda r:intervention_free(r.anchor_minute,r.first_pressor,r.first_mcs),axis=1)
 mp_register('_screen',candidates);con.execute('CREATE TABLE candidate_screen AS SELECT * FROM _screen');con.unregister('_screen')
 eligible=candidates.loc[candidates['baseline_lactate'].lt(2)&candidates['intervention_free']].copy()
 exposed=eligible.loc[eligible['candidate_type'].eq('crossing')].copy()
 controls=assign_pseudo_anchors(exposed,eligible.loc[eligible['candidate_type'].eq('control')].copy())
 final=pd.concat([exposed,controls],ignore_index=True);final['exposed']=final['candidate_type'].eq('crossing').astype(int)
 mp_register('_final',final);con.execute('CREATE TABLE final_anchors AS SELECT * FROM _final');con.unregister('_final')
 print('eligible crossings and controls',len(exposed),len(controls),flush=True)
mp_status('preanchor_screening_complete')
'''

FOLLOWUP_GATE = r'''
mp_status('followup_existence_and_gate')
if not mp_exists('followup_lactate_times'):
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS eicu_followup_lactate_times AS
 SELECT 'eicu' dataset,a.stay_id,a.candidate_id,a.anchor_minute,try_cast(l.labresultoffset AS DOUBLE) event_minute
 FROM final_anchors a
 JOIN read_csv('{str(EICU/'lab.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id
 WHERE a.dataset='eicu' AND lower(trim(l.labname))='lactate'
  AND try_cast(l.labresultoffset AS DOUBLE)>a.anchor_minute
  AND try_cast(l.labresultoffset AS DOUBLE)<=a.anchor_minute+720
  AND try_cast(l.labresultrevisedoffset AS DOUBLE)<=a.anchor_minute+720
 """)
 con.execute(f"""
 CREATE TABLE IF NOT EXISTS mimic_followup_lactate_times AS
 SELECT 'mimic' dataset,a.stay_id,a.candidate_id,a.anchor_minute,
  date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute
 FROM final_anchors a JOIN mimic_cohort c USING(stay_id)
 JOIN read_csv('{str(MIMIC/'labevents.csv').replace(chr(39),chr(39)*2)}',header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id
 JOIN mimic_lactate_items d ON try_cast(l.itemid AS BIGINT)=d.itemid
 WHERE a.dataset='mimic'
  AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0>a.anchor_minute
  AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0<=a.anchor_minute+720
  AND date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0<=a.anchor_minute+720
 """)
 con.execute('CREATE TABLE IF NOT EXISTS followup_lactate_times AS SELECT * FROM eicu_followup_lactate_times UNION ALL SELECT * FROM mimic_followup_lactate_times')
anchors=con.execute('SELECT * FROM final_anchors').df();times=con.execute('SELECT dataset,stay_id,candidate_id,event_minute FROM followup_lactate_times').df()
flags=[]
for row in anchors.itertuples():
 values=times.loc[times['candidate_id'].eq(row.candidate_id),'event_minute'].to_numpy(float);f6,f12=followup_observation_flags(values,row.anchor_minute);flags.append((row.candidate_id,f6,f12))
flags=pd.DataFrame(flags,columns=['candidate_id','followup_6h','followup_12h']);anchors=anchors.merge(flags,on='candidate_id',how='left',validate='one_to_one')
hourly=con.execute('SELECT dataset,stay_id,"source",hour FROM hourly_bp').df();candidates=con.execute('SELECT * FROM candidate_screen').df()
audit=[]
for dataset in ['mimic','eicu']:
 h=hourly.loc[hourly.dataset.eq(dataset)];a=anchors.loc[anchors.dataset.eq(dataset)];c=candidates.loc[candidates.dataset.eq(dataset)]
 bin_counts=h.groupby('stay_id').size()
 crossings=a.loc[a.exposed.eq(1)];controls=a.loc[a.exposed.eq(0)]
 source_counts={str(k):int(v) for k,v in h.groupby('source').stay_id.nunique().items()}
 row=dict(dataset=dataset,production_hf_cohort=int(con.execute(f'SELECT count(*) FROM {dataset}_cohort').fetchone()[0]),valid_sbp_dbp=int(h.stay_id.nunique()),at_least_2_hourly_prpp_bins=int((bin_counts>=2).sum()),at_least_3_hourly_prpp_bins=int((bin_counts>=3).sum()),patients_with_preanchor_lactate_lt2=int(c.loc[c.baseline_lactate.lt(2),'stay_id'].nunique()),patients_satisfying_compensated_state=int(c.stay_id.nunique()),crossings=int(len(crossings)),controls=int(len(controls)),crossings_followup_6h=int(crossings.followup_6h.sum()),crossings_followup_12h=int(crossings.followup_12h.sum()),controls_followup_6h=int(controls.followup_6h.sum()),controls_followup_12h=int(controls.followup_12h.sum()),bp_source_distribution=source_counts,median_bp_bins_per_patient=float(bin_counts.median()) if len(bin_counts) else 0.0,median_lactate_to_anchor_lag=float(a.lactate_to_anchor_lag.median()) if len(a) else None,eicu_hospitals=int(a.hospital_id.nunique()) if dataset=='eicu' else None)
 audit.append(row)
audit_frame=pd.json_normalize(audit);audit_frame.to_csv(OUT/'feasibility_audit.csv',index=False);audit_frame.to_csv(OUT/'support_audit.csv',index=False)
decision=phase_a_decision(audit,MASKED_PULSATILITY_FEASIBILITY_SHA256,'google_colab' if IN_COLAB else 'local')
decision['phase_a_protocol_frozen_before_scan']=True;decision['candidate_table_rows']=int(len(anchors));decision['candidate_table_sha256']=hashlib.sha256(anchors.sort_values(['dataset','candidate_id']).to_json(orient='records',double_precision=15).encode()).hexdigest()
mp_json('decision_gate_summary.json',decision)
if not decision['masked_pulsatility_feasible']:
 stopped=pd.DataFrame([dict(status='not_computed',reason='phase_a_fixed_gate_failed',post_anchor_lactate_values_inspected=False,clinical_outcomes_inspected=False)])
 for name in ['primary_results.csv','observation_weighted_results.csv','sbp_map_falsification.csv','static_vs_dynamic.csv','bp_source_sensitivity.csv','lead_time_results.csv','secondary_results.csv']:
  stopped.to_csv(OUT/name,index=False)
(OUT/'execution.log').write_text(f"{datetime.now(timezone.utc).isoformat()} phase_a_complete environment={'google_colab' if IN_COLAB else 'local'} masked_pulsatility_feasible={decision['masked_pulsatility_feasible']} post_anchor_lactate_values_inspected=false clinical_outcomes_inspected=false recommended_action={decision['recommended_action']}\n")
mp_status('phase_a_complete',state='completed',goal_complete=not decision['masked_pulsatility_feasible'],recommended_action=decision['recommended_action'],masked_pulsatility_feasible=decision['masked_pulsatility_feasible'])
print(json.dumps(decision,indent=2,allow_nan=False),flush=True)
if decision['masked_pulsatility_feasible']:
 print('PHASE_A_PASS: freeze and hash docs/MASKED_PULSATILITY_ANALYSIS_PLAN.md before any outcome values.',flush=True)
else:
 print('NO_VIABLE_SIGNAL: fixed per-database feasibility gate failed; no outcomes inspected and no fallback executed.',flush=True)
'''

__all__ = ['BASELINE_SCREEN', 'BP_EXTRACT', 'FOLLOWUP_GATE', 'SETUP']
