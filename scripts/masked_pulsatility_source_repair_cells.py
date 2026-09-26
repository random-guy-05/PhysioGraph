"""Final outcome-blind source repair and exposure-feasibility notebook cells."""

SETUP = r'''
import csv,hashlib,importlib.util,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3','tabulate':'tabulate==0.9.0'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905';EICU=DRIVE/'Data/eICU/Full';MIMIC=DRIVE/'Data/MIMIC/Full';OUT=PROJECT/'research/masked_pulsatility';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.masked_pulsatility import assign_pseudo_anchors,build_source_hourly_bp_pairs,deduplicate_cuff_pairs,detect_source_consistent_crossings,followup_observation_flags,hospital_contribution_audit,intervention_free,latest_preanchor_lactate,source_repair_feasibility,valid_bp_pair
START_UTC=datetime.now(timezone.utc).isoformat()
def sr_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def sr_q(value):return "'"+str(value).replace("'","''")+"'"
def sr_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def sr_register(name,frame):
 frame=frame.copy()
 for col in frame.select_dtypes(include=['string']).columns:frame[col]=frame[col].astype(object)
 sr_con.register(name,frame)
def sr_exists(name):return bool(sr_con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",[name]).fetchone()[0])
plan=PROJECT/'docs/MASKED_PULSATILITY_SOURCE_REPAIR_PLAN.md';plan_sha=hashlib.sha256(plan.read_bytes()).hexdigest();assert plan_sha==MASKED_PULSATILITY_SOURCE_REPAIR_SHA256
CODE_COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
code_files=['configs/eicu.yaml','docs/MASKED_PULSATILITY_SOURCE_REPAIR_PLAN.md','scripts/build_biological_notebook.py','scripts/masked_pulsatility_source_repair_cells.py','src/physiograph/analysis/masked_pulsatility.py','src/physiograph/etl/eicu_extractor.py','tests/unit/test_masked_pulsatility.py','tests/unit/test_biomarker_benchmark.py']
CODE_FILE_HASHES={name:sr_hash(PROJECT/name) for name in code_files};CODE_STATE_SHA256=hashlib.sha256(json.dumps(CODE_FILE_HASHES,sort_keys=True,separators=(',',':')).encode()).hexdigest()
roles={'patient.csv':'cohort','hospital.csv':'hospital metadata','diagnosis.csv':'HF cohort','admissionDx.csv':'HF cohort','vitalPeriodic.csv':'invasive arterial BP','vitalAperiodic.csv':'structured cuff BP','nurseCharting.csv':'nurse-charted cuff BP','lab.csv':'pre-anchor lactate and follow-up timestamp availability','infusionDrug.csv':'pre-anchor support','treatment.csv':'pre-anchor MCS','respiratoryCharting.csv':'inventory only','respiratoryCare.csv':'inventory only','carePlanGeneral.csv':'inventory only','medication.csv':'inventory only','intakeOutput.csv':'inventory only'}
inventory=[]
for filename,role in roles.items():
 path=EICU/filename;exists=path.is_file();size=path.stat().st_size if exists else None
 header=None
 if exists:
  with path.open('r',newline='',errors='replace') as handle:header=next(csv.reader(handle))
 digest=sr_hash(path) if exists and size<=500_000_000 else None
 inventory.append(dict(filename=filename,exists=exists,byte_size=size,sha256=digest,sha256_note=None if digest else ('missing' if not exists else 'omitted_for_files_over_500MB_to_avoid_extra_full_pass'),header=header,source_role=role))
pd.DataFrame([{**r,'header':json.dumps(r['header'])} for r in inventory]).to_csv(OUT/'source_repair_inventory.csv',index=False);sr_json('source_repair_inventory.json',inventory)
required_bp=['vitalPeriodic.csv','vitalAperiodic.csv','nurseCharting.csv'];assert all((EICU/f).is_file() for f in required_bp)
try:sr_con.close()
except (NameError,AttributeError):pass
db=PRIVATE/'masked_pulsatility_source_repair.duckdb';sr_con=duckdb.connect(str(db));sr_con.execute('SET threads=2');sr_con.execute("SET memory_limit='3GB'");sr_con.execute('SET enable_progress_bar=false')
phase=PRIVATE/'masked_pulsatility_phase_a.duckdb';old_audit=PRIVATE/'masked_pulsatility_source_audit.duckdb';sr_con.execute(f"ATTACH {sr_q(phase)} AS phase (READ_ONLY)");sr_con.execute(f"ATTACH {sr_q(old_audit)} AS prior_audit (READ_ONLY)")
sr_json('source_repair_run_status.json',dict(status='running',stage='inventory_complete',start_utc=START_UTC,environment='google_colab' if IN_COLAB else 'local',outcomes_inspected=False,post_anchor_lactate_values_inspected=False,phase_b_executed=False,protocol_sha256=plan_sha))
print(pd.DataFrame(inventory)[['filename','exists','byte_size','source_role']].to_string(index=False),flush=True)
'''

LABEL_DISCOVERY = r'''
if not sr_exists('nurse_bp_label_candidates'):
 sr_con.execute(f"""
 CREATE TABLE nurse_bp_label_candidates AS
 SELECT try_cast(n.patientunitstayid AS BIGINT) stay_id,try_cast(n.nursingchartoffset AS DOUBLE) timestamp_min,
  n.nursingchartcelltypecat,n.nursingchartcelltypevallabel,n.nursingchartcelltypevalname,n.nursingchartvalue
 FROM read_csv({sr_q(EICU/'nurseCharting.csv')},header=true,all_varchar=true,parallel=true) n
 JOIN phase.eicu_cohort c ON try_cast(n.patientunitstayid AS BIGINT)=c.stay_id
 WHERE try_cast(n.nursingchartoffset AS DOUBLE)>=0 AND try_cast(n.nursingchartoffset AS DOUBLE)<240
  AND (regexp_matches(lower(coalesce(n.nursingchartcelltypevallabel,'')),'blood pressure|non.?invasive bp|nibp|(^| )bp($| )')
    OR regexp_matches(lower(coalesce(n.nursingchartcelltypevalname,'')),'blood pressure|non.?invasive bp|nibp|(^| )bp($| )'))
 """)
exact_names={'Non-Invasive BP Systolic':'SBP','Non-Invasive BP Diastolic':'DBP','Non-Invasive BP Mean':'MAP'}
labels=sr_con.execute("""SELECT nursingchartcelltypecat,nursingchartcelltypevallabel,nursingchartcelltypevalname,count(*) n_rows,count(DISTINCT stay_id) unique_stays FROM nurse_bp_label_candidates GROUP BY 1,2,3 ORDER BY n_rows DESC""").df()
labels['mapped_concept']=labels.apply(lambda r:exact_names.get(r.nursingchartcelltypevalname) if r.nursingchartcelltypecat=='Vital Signs' and r.nursingchartcelltypevallabel=='Non-Invasive BP' else None,axis=1)
labels['source_classification']=np.where(labels['mapped_concept'].notna(),'nibp_nurseCharting',np.where(labels.nursingchartcelltypevallabel.eq('Invasive BP'),'excluded_invasive_nurse_label','excluded_unvalidated'))
labels['reason_included_excluded']=np.where(labels['mapped_concept'].notna(),'exact frozen Vital Signs / Non-Invasive BP tuple','not an exact validated non-invasive tuple')
labels.to_csv(OUT/'nurse_bp_mapping_audit.csv',index=False)
mapping_lock=dict(protocol_sha256=plan_sha,frozen_utc=datetime.now(timezone.utc).isoformat(),frozen_before_crossing_prevalence=True,exact_nurse_mapping=[dict(category='Vital Signs',label='Non-Invasive BP',name=name,concept=concept) for name,concept in exact_names.items()],aperiodic_mapping={'noninvasivesystolic':'SBP','noninvasivediastolic':'DBP','noninvasivemean':'MAP'},deduplication_rule='retain vitalAperiodic; drop nurse pair only for same stay within +/-1 minute with identical SBP/DBP and compatible MAP; fill missing aperiodic MAP from matched nurse row',threshold=0.25,outcomes_prohibited=True)
mapping_lock['lock_sha256']=hashlib.sha256(json.dumps(mapping_lock,sort_keys=True,separators=(',',':')).encode()).hexdigest();sr_json('source_repair_mapping_lock.json',mapping_lock)
print(labels.to_string(index=False),flush=True);print(json.dumps(mapping_lock,indent=2),flush=True)
'''

EXTRACT_AND_DEDUP = r'''
if not sr_exists('eicu_aperiodic_pairs_raw'):
 sr_con.execute(f"""
 CREATE TABLE eicu_aperiodic_pairs_raw AS
 SELECT 'eicu' dataset,cast(c.person_id AS VARCHAR) patient_id,c.stay_id,try_cast(v.observationoffset AS DOUBLE) timestamp_min,
  try_cast(v.noninvasivesystolic AS DOUBLE) sbp,try_cast(v.noninvasivediastolic AS DOUBLE) dbp,try_cast(v.noninvasivemean AS DOUBLE) AS "map",
  'nibp_vitalAperiodic' bp_source,'vitalAperiodic.csv' source_table,'Non-Invasive BP' source_label,cast(v.vitalaperiodicid AS VARCHAR) raw_row_provenance
 FROM read_csv({sr_q(EICU/'vitalAperiodic.csv')},header=true,all_varchar=true,parallel=true) v JOIN phase.eicu_cohort c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
 WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240
 """)
if not sr_exists('eicu_nurse_pairs_raw'):
 sr_con.execute("""
 CREATE TABLE eicu_nurse_pairs_raw AS
 SELECT 'eicu' dataset,cast(c.person_id AS VARCHAR) patient_id,c.stay_id,n.timestamp_min,
  median(try_cast(n.nursingchartvalue AS DOUBLE)) FILTER(WHERE n.nursingchartcelltypevalname='Non-Invasive BP Systolic') sbp,
  median(try_cast(n.nursingchartvalue AS DOUBLE)) FILTER(WHERE n.nursingchartcelltypevalname='Non-Invasive BP Diastolic') dbp,
  median(try_cast(n.nursingchartvalue AS DOUBLE)) FILTER(WHERE n.nursingchartcelltypevalname='Non-Invasive BP Mean') AS "map",
  'nibp_nurseCharting' bp_source,'nurseCharting.csv' source_table,'Vital Signs | Non-Invasive BP' source_label,
  concat('stay=',cast(c.stay_id AS VARCHAR),';offset=',cast(n.timestamp_min AS VARCHAR)) raw_row_provenance
 FROM nurse_bp_label_candidates n JOIN phase.eicu_cohort c USING(stay_id)
 WHERE n.nursingchartcelltypecat='Vital Signs' AND n.nursingchartcelltypevallabel='Non-Invasive BP'
  AND n.nursingchartcelltypevalname IN ('Non-Invasive BP Systolic','Non-Invasive BP Diastolic','Non-Invasive BP Mean')
 GROUP BY c.person_id,c.stay_id,n.timestamp_min
 """)
if not sr_exists('eicu_invasive_pairs_raw'):
 sr_con.execute("""CREATE TABLE eicu_invasive_pairs_raw AS SELECT 'eicu' dataset,cast(c.person_id AS VARCHAR) patient_id,r.stay_id,r.event_minute timestamp_min,r.sbp,r.dbp,r."map",'ibp_vitalPeriodic' bp_source,'vitalPeriodic.csv' source_table,'systemic arterial pressure' source_label,concat('stay=',cast(r.stay_id AS VARCHAR),';offset=',cast(r.event_minute AS VARCHAR)) raw_row_provenance FROM prior_audit.eicu_bp_rows r JOIN phase.eicu_cohort c USING(stay_id)""")
if not sr_exists('mimic_pairs_raw'):
 sr_con.execute("""CREATE TABLE mimic_pairs_raw AS SELECT 'mimic' dataset,cast(c.person_id AS VARCHAR) patient_id,r.stay_id,r.event_minute timestamp_min,r.sbp,r.dbp,r."map",CASE WHEN r."source"='noninvasive_cuff' THEN 'nibp_mimic_chartevents' ELSE 'ibp_mimic_chartevents' END bp_source,'chartevents.csv' source_table,r."source" source_label,concat('stay=',cast(r.stay_id AS VARCHAR),';minute=',cast(r.event_minute AS VARCHAR)) raw_row_provenance FROM prior_audit.mimic_bp_rows r JOIN phase.mimic_cohort c USING(stay_id)""")
def validate_source(frame,source):
 local=frame.copy();numeric=['timestamp_min','sbp','dbp','map']
 for col in numeric:local[col]=pd.to_numeric(local[col],errors='coerce')
 paired=local.sbp.notna()&local.dbp.notna();nonpositive=paired&(local.sbp.le(0)|local.dbp.le(0));gross=paired&~nonpositive&(~local.sbp.between(50,300)|~local.dbp.between(20,200));order=paired&~nonpositive&~gross&local.sbp.le(local.dbp);valid=paired&~nonpositive&~gross&~order
 audit=dict(bp_source=source,raw_rows=len(local),missing_sbp_or_dbp=int((~paired).sum()),nonpositive=int(nonpositive.sum()),gross_range=int(gross.sum()),sbp_le_dbp=int(order.sum()),valid_pairs=int(valid.sum()),valid_stays=int(local.loc[valid,'stay_id'].nunique()))
 return local.loc[valid].copy(),audit
aperiodic,va=validate_source(sr_con.execute('SELECT * FROM eicu_aperiodic_pairs_raw').df(),'nibp_vitalAperiodic')
nurse,nv=validate_source(sr_con.execute('SELECT * FROM eicu_nurse_pairs_raw').df(),'nibp_nurseCharting')
invasive,iv=validate_source(sr_con.execute('SELECT * FROM eicu_invasive_pairs_raw').df(),'ibp_vitalPeriodic')
mimic=sr_con.execute('SELECT * FROM mimic_pairs_raw').df();mimic_cuff,mcv=validate_source(mimic.loc[mimic.bp_source.eq('nibp_mimic_chartevents')],'nibp_mimic_chartevents');mimic_ibp,miv=validate_source(mimic.loc[mimic.bp_source.eq('ibp_mimic_chartevents')],'ibp_mimic_chartevents')
cuff_union,overlap=deduplicate_cuff_pairs(aperiodic,nurse,near_minutes=1.0);cuff_union['bp_source']='nibp_cuff_union';cuff_union['source_table']='deduplicated vitalAperiodic+nurseCharting';cuff_union['source_label']='deduplicated cuff union'
for name,frame in [('eicu_aperiodic_pairs',aperiodic),('eicu_nurse_pairs',nurse),('eicu_cuff_union_pairs',cuff_union),('eicu_invasive_pairs',invasive),('mimic_cuff_pairs',mimic_cuff),('mimic_invasive_pairs',mimic_ibp)]:
 sr_con.execute(f'DROP TABLE IF EXISTS {name}');sr_register('_frame',frame);sr_con.execute(f'CREATE TABLE {name} AS SELECT * FROM _frame');sr_con.unregister('_frame')
validation=pd.DataFrame([va,nv,iv,mcv,miv]);validation.to_csv(OUT/'bp_physiologic_validation_audit.csv',index=False)
overlap.update(dict(dataset='eicu',near_time_rule_minutes=1.0,deduplicated_union_rows=len(cuff_union),deduplicated_union_stays=int(cuff_union.stay_id.nunique())))
pd.DataFrame([overlap]).to_csv(OUT/'bp_cross_source_overlap_audit.csv',index=False);sr_json('bp_cross_source_overlap_audit.json',overlap)
print(validation.to_string(index=False),flush=True);print(json.dumps(overlap,indent=2),flush=True)
'''

HARMONIZE = r'''
pair_tables=[('eicu_aperiodic_pairs','nibp_vitalAperiodic'),('eicu_nurse_pairs','nibp_nurseCharting'),('eicu_cuff_union_pairs','nibp_cuff_union'),('eicu_invasive_pairs','ibp_vitalPeriodic'),('mimic_cuff_pairs','nibp_mimic_chartevents'),('mimic_invasive_pairs','ibp_mimic_chartevents')]
hourly_frames=[]
for table,source in pair_tables:
 frame=sr_con.execute(f'SELECT * FROM {table}').df();frame['bp_source']=source;hourly_frames.append(build_source_hourly_bp_pairs(frame))
hourly=pd.concat(hourly_frames,ignore_index=True)
for dataset,cuff_source,ibp_source in [('eicu','nibp_cuff_union','ibp_vitalPeriodic'),('mimic','nibp_mimic_chartevents','ibp_mimic_chartevents')]:
 cuff_stays=set(hourly.loc[hourly.dataset.eq(dataset)&hourly.bp_source.eq(cuff_source),'stay_id'])
 total=hourly.loc[hourly.dataset.eq(dataset)&((hourly.bp_source.eq(cuff_source))|(hourly.bp_source.eq(ibp_source)&~hourly.stay_id.isin(cuff_stays)))].copy();total['component_source']=total['bp_source'];total['bp_source']='source_consistent_total';hourly=pd.concat([hourly,total],ignore_index=True)
crossings,initially_low=detect_source_consistent_crossings(hourly)
eicu_meta=sr_con.execute('SELECT stay_id,person_id,hospital_id,death_offset_minutes,followup_end_offset_minutes FROM phase.eicu_cohort').df();eicu_meta['dataset']='eicu';mimic_meta=sr_con.execute("SELECT stay_id,person_id,'MIMIC' hospital_id,death_offset_minutes,followup_end_offset_minutes FROM phase.mimic_cohort").df();mimic_meta['dataset']='mimic';meta=pd.concat([eicu_meta,mimic_meta],ignore_index=True)
crossings=crossings.merge(meta,on=['dataset','stay_id'],how='left',validate='many_to_one');interventions=sr_con.execute('SELECT * FROM phase.first_intervention').df();crossings=crossings.merge(interventions,on=['dataset','stay_id'],how='left',validate='many_to_one')
crossings['sbp_preserved']=crossings.sbp.ge(90);crossings['direct_map_valid']=crossings['map'].between(20,200)&crossings['map'].ge(crossings.dbp)&crossings['map'].le(crossings.sbp);crossings['direct_map_preserved']=crossings.direct_map_valid&crossings['map'].ge(65);crossings['derived_map_preserved']=crossings.derived_map.ge(65)
crossings['no_prior_pressor']=[pd.isna(p) or float(p)>float(a) for p,a in zip(crossings.first_pressor,crossings.anchor_minute)];crossings['no_prior_mcs']=[pd.isna(m) or float(m)>float(a) for m,a in zip(crossings.first_mcs,crossings.anchor_minute)];crossings['no_prior_support']=crossings.no_prior_pressor&crossings.no_prior_mcs
crossings['alive_observed']=crossings.followup_end_offset_minutes.ge(crossings.anchor_minute)&(crossings.death_offset_minutes.isna()|crossings.death_offset_minutes.gt(crossings.anchor_minute));crossings['candidate_id']=crossings.apply(lambda r:f"{r.dataset}|{int(r.stay_id)}|{r.bp_source}|{int(r.hour)}|crossing",axis=1)
primary_sources={'eicu':'nibp_cuff_union','mimic':'nibp_mimic_chartevents'};primary_exposed=crossings.loc[[r.bp_source==primary_sources[r.dataset] and r.sbp_preserved and r.direct_map_preserved and r.no_prior_support and r.alive_observed for r in crossings.itertuples()]].copy()
control_bins=[]
for dataset,source in primary_sources.items():
 h=hourly.loc[hourly.dataset.eq(dataset)&hourly.bp_source.eq(source)].copy();crossed=set(crossings.loc[crossings.dataset.eq(dataset)&crossings.bp_source.eq(source),'stay_id']);low_stays={key[1] for key in initially_low if key[0]==dataset and key[2]==source};h=h.loc[~h.stay_id.isin(crossed|low_stays)&h.prpp.ge(0.25)&h.sbp.ge(90)&h['map'].between(65,200)&h['map'].ge(h.dbp)&h['map'].le(h.sbp)].copy();h=h.merge(meta.loc[meta.dataset.eq(dataset)],on=['dataset','stay_id'],how='left',validate='many_to_one').merge(interventions,on=['dataset','stay_id'],how='left',validate='many_to_one');h['intervention_free']=[intervention_free(a,p,m) for a,p,m in zip(h.anchor_minute,h.first_pressor,h.first_mcs)];h=h.loc[h.intervention_free&h.followup_end_offset_minutes.ge(h.anchor_minute)&(h.death_offset_minutes.isna()|h.death_offset_minutes.gt(h.anchor_minute))].copy();h['candidate_id']=h.apply(lambda r:f"{r.dataset}|{int(r.stay_id)}|{r.bp_source}|{int(r.hour)}|control",axis=1);control_bins.append(h)
controls=assign_pseudo_anchors(primary_exposed,pd.concat(control_bins,ignore_index=True));controls['anchor_type']='control';primary_exposed['anchor_type']='crossing'
for name,frame in [('source_repair_hourly_bp',hourly),('source_repair_crossings',crossings),('source_repair_primary_exposed',primary_exposed),('source_repair_controls',controls)]:
 sr_con.execute(f'DROP TABLE IF EXISTS {name}');sr_register('_frame',frame);sr_con.execute(f'CREATE TABLE {name} AS SELECT * FROM _frame');sr_con.unregister('_frame')
print('hourly bins',len(hourly),'crossings',len(crossings),'primary compensated no-support crossings',len(primary_exposed),'controls',len(controls),flush=True)
'''

LACTATE_AVAILABILITY = r'''
crossings=sr_con.execute('SELECT * FROM source_repair_crossings').df();controls=sr_con.execute('SELECT * FROM source_repair_controls').df();eligible_crossings=crossings.loc[crossings.sbp_preserved&crossings.direct_map_preserved&crossings.no_prior_support&crossings.alive_observed].copy();eligible_crossings['anchor_type']='crossing';anchors=pd.concat([eligible_crossings,controls],ignore_index=True,sort=False)
anchor_cols=['dataset','stay_id','candidate_id','anchor_minute','bp_source','anchor_type'];sr_con.execute('DROP TABLE IF EXISTS source_repair_analysis_anchors');sr_register('_anchors',anchors[anchor_cols]);sr_con.execute('CREATE TABLE source_repair_analysis_anchors AS SELECT * FROM _anchors');sr_con.unregister('_anchors')
if not sr_exists('source_repair_preanchor_lactate'):
 sr_con.execute(f"""CREATE TABLE eicu_repair_preanchor_lactate AS SELECT 'eicu' dataset,a.stay_id,a.candidate_id,a.anchor_minute,try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,try_cast(l.labresult AS DOUBLE) AS "value" FROM source_repair_analysis_anchors a JOIN read_csv({sr_q(EICU/'lab.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND lower(trim(l.labname))='lactate' AND regexp_matches(lower(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,'')),'mmol') AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN a.anchor_minute-240 AND a.anchor_minute AND try_cast(l.labresultrevisedoffset AS DOUBLE)<=a.anchor_minute""")
 sr_con.execute(f"""CREATE TABLE mimic_repair_preanchor_lactate AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,a.anchor_minute,date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,try_cast(l.valuenum AS DOUBLE) AS "value" FROM source_repair_analysis_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({sr_q(MIMIC/'labevents.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id JOIN phase.mimic_lactate_items d ON try_cast(l.itemid AS BIGINT)=d.itemid WHERE a.dataset='mimic' AND regexp_matches(lower(coalesce(l.valueuom,'')),'mmol') AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN a.anchor_minute-240 AND a.anchor_minute AND date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0<=a.anchor_minute""")
 sr_con.execute('CREATE TABLE source_repair_preanchor_lactate AS SELECT * FROM eicu_repair_preanchor_lactate UNION ALL SELECT * FROM mimic_repair_preanchor_lactate')
if not sr_exists('source_repair_followup_lactate_times'):
 sr_con.execute(f"""CREATE TABLE eicu_repair_followup_lactate_times AS SELECT 'eicu' dataset,a.stay_id,a.candidate_id,a.anchor_minute,try_cast(l.labresultoffset AS DOUBLE) event_minute FROM source_repair_analysis_anchors a JOIN read_csv({sr_q(EICU/'lab.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND lower(trim(l.labname))='lactate' AND try_cast(l.labresultoffset AS DOUBLE)>a.anchor_minute AND try_cast(l.labresultoffset AS DOUBLE)<=a.anchor_minute+720""")
 sr_con.execute(f"""CREATE TABLE mimic_repair_followup_lactate_times AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,a.anchor_minute,date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute FROM source_repair_analysis_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({sr_q(MIMIC/'labevents.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id JOIN phase.mimic_lactate_items d ON try_cast(l.itemid AS BIGINT)=d.itemid WHERE a.dataset='mimic' AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0>a.anchor_minute AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0<=a.anchor_minute+720""")
 sr_con.execute('CREATE TABLE source_repair_followup_lactate_times AS SELECT * FROM eicu_repair_followup_lactate_times UNION ALL SELECT * FROM mimic_repair_followup_lactate_times')
measurements=sr_con.execute('SELECT dataset,stay_id,candidate_id,event_minute,available_minute,"value" FROM source_repair_preanchor_lactate').df();baseline=latest_preanchor_lactate(measurements,anchors[['dataset','stay_id','candidate_id','anchor_minute']]);times=sr_con.execute('SELECT candidate_id,event_minute FROM source_repair_followup_lactate_times').df();flags=[]
for row in anchors.itertuples():
 values=times.loc[times.candidate_id.eq(row.candidate_id),'event_minute'].to_numpy(float);f6,f12=followup_observation_flags(values,row.anchor_minute);flags.append((row.candidate_id,f6,f12))
anchor_audit=anchors.merge(baseline.drop(columns=['dataset','stay_id','anchor_minute']),on='candidate_id',how='left',validate='one_to_one').merge(pd.DataFrame(flags,columns=['candidate_id','followup_lactate_6h','followup_lactate_12h']),on='candidate_id',how='left',validate='one_to_one')
sr_con.execute('DROP TABLE IF EXISTS source_repair_anchor_audit');sr_register('_anchor_audit',anchor_audit);sr_con.execute('CREATE TABLE source_repair_anchor_audit AS SELECT * FROM _anchor_audit');sr_con.unregister('_anchor_audit')
print('lactate status and timestamp-only follow-up availability complete; no post-anchor values selected',flush=True)
'''

REPORT = r'''
hourly=sr_con.execute('SELECT * FROM source_repair_hourly_bp').df();crossings=sr_con.execute('SELECT * FROM source_repair_crossings').df();anchors=sr_con.execute('SELECT * FROM source_repair_anchor_audit').df();controls=sr_con.execute('SELECT * FROM source_repair_controls').df();primary={'eicu':'nibp_cuff_union','mimic':'nibp_mimic_chartevents'}
comparison=[]
for dataset,source in sorted(hourly[['dataset','bp_source']].drop_duplicates().itertuples(index=False,name=None)):
 h=hourly.loc[hourly.dataset.eq(dataset)&hourly.bp_source.eq(source)];c=crossings.loc[crossings.dataset.eq(dataset)&crossings.bp_source.eq(source)];a=anchors.loc[anchors.dataset.eq(dataset)&anchors.bp_source.eq(source)&anchors.anchor_type.eq('crossing')];bins=h.groupby('stay_id').size();eligible=c.loc[c.sbp_preserved&c.direct_map_preserved&c.no_prior_support&c.alive_observed];hospital_n=np.nan
 if dataset=='eicu':
  meta=sr_con.execute('SELECT stay_id,hospital_id FROM phase.eicu_cohort').df();hospital_n=int(h[['stay_id']].drop_duplicates().merge(meta,on='stay_id').hospital_id.nunique())
 comparison.append(dict(dataset=dataset,bp_source=source,hf_n=int(sr_con.execute(f'SELECT count(*) FROM phase.{dataset}_cohort').fetchone()[0]),any_bp_n=int(h.stay_id.nunique()),two_bins_n=int((bins>=2).sum()),three_bins_n=int((bins>=3).sum()),crossing_n=len(c),crossing_pct=float(100*len(c)/max(1,h.stay_id.nunique())),sbp_preserved_crossing_n=int(c.sbp_preserved.sum()),map_preserved_crossing_n=int((c.sbp_preserved&c.direct_map_preserved).sum()),derived_map_preserved_crossing_n=int((c.sbp_preserved&c.derived_map_preserved).sum()),no_prior_support_crossing_n=len(eligible),no_lactate_n=int(a.baseline_lactate.isna().sum()),lactate_lt2_n=int(a.baseline_lactate.lt(2).sum()),lactate_ge2_n=int(a.baseline_lactate.ge(2).sum()),followup_lactate_6h_available_n=int(a.followup_lactate_6h.sum()),followup_lactate_12h_available_n=int(a.followup_lactate_12h.sum()),hospital_n=hospital_n,notes='outcome-blind; post-anchor lactate values not selected'))
comparison=pd.DataFrame(comparison);comparison.to_csv(OUT/'source_feasibility_comparison.csv',index=False)
attrition=[]
for dataset in ['eicu','mimic']:
 cuff=primary[dataset];row=comparison.loc[comparison.dataset.eq(dataset)&comparison.bp_source.eq(cuff)].iloc[0];a=anchors.loc[anchors.dataset.eq(dataset)&anchors.bp_source.eq(cuff)&anchors.anchor_type.eq('crossing')]
 def add(stage,count,source=cuff,notes=''):attrition.append(dict(dataset=dataset,stage=stage,bp_source=source,count=int(count),notes=notes))
 add('HF cohort',row.hf_n,'all')
 if dataset=='eicu':
  for source,label in [('nibp_vitalAperiodic','any cuff BP: vitalAperiodic'),('nibp_nurseCharting','any cuff BP: nurseCharting'),('nibp_cuff_union','any cuff BP: deduplicated union')]:add(label,comparison.loc[comparison.dataset.eq(dataset)&comparison.bp_source.eq(source),'any_bp_n'].iloc[0],source)
 else:
  add('any cuff BP: vitalAperiodic',0,'not_applicable','not an eICU source');add('any cuff BP: nurseCharting',0,'not_applicable','not an eICU source');add('any cuff BP: deduplicated union',row.any_bp_n,cuff)
 ibp='ibp_vitalPeriodic' if dataset=='eicu' else 'ibp_mimic_chartevents';add('any invasive BP',comparison.loc[comparison.dataset.eq(dataset)&comparison.bp_source.eq(ibp),'any_bp_n'].iloc[0],ibp);add('>=2 hourly cuff PrPP bins',row.two_bins_n);add('>=3 hourly cuff PrPP bins',row.three_bins_n);add('pair-only cuff crossings',row.crossing_n);add('SBP-preserved cuff crossings',row.sbp_preserved_crossing_n);add('direct-MAP-preserved cuff crossings',row.map_preserved_crossing_n);add('direct-or-derived-MAP-preserved cuff crossings',row.derived_map_preserved_crossing_n);add('no-prior-support cuff crossings',row.no_prior_support_crossing_n);add('lactate status: no measured lactate',a.baseline_lactate.isna().sum());add('lactate status: lactate <2',a.baseline_lactate.lt(2).sum());add('lactate status: lactate >=2',a.baseline_lactate.ge(2).sum());add('follow-up lactate available 6h',a.followup_lactate_6h.sum());add('follow-up lactate available 12h',a.followup_lactate_12h.sum());add('eligible compensated controls',controls.loc[controls.dataset.eq(dataset)].stay_id.nunique())
pd.DataFrame(attrition).to_csv(OUT/'source_repair_attrition.csv',index=False)
hospital_rows=[]
eicu_meta=sr_con.execute('SELECT stay_id,hospital_id FROM phase.eicu_cohort').df()
for source in comparison.loc[comparison.dataset.eq('eicu'),'bp_source']:
 h=hourly.loc[hourly.dataset.eq('eicu')&hourly.bp_source.eq(source),['stay_id']].drop_duplicates().merge(eicu_meta,on='stay_id');c=crossings.loc[crossings.dataset.eq('eicu')&crossings.bp_source.eq(source)&crossings.sbp_preserved&crossings.direct_map_preserved&crossings.no_prior_support&crossings.alive_observed].merge(eicu_meta,on='stay_id',suffixes=('','_meta'));audit=hospital_contribution_audit(c);audit.update(dict(dataset='eicu',bp_source=source,hospitals_contributing_bp=int(h.hospital_id.nunique()),hospitals_ge10_hf_stays=int(h.groupby('hospital_id').stay_id.nunique().ge(10).sum())));hospital_rows.append(audit)
hospital=pd.DataFrame(hospital_rows);hospital.to_csv(OUT/'source_repair_hospital_coverage.csv',index=False);primary_hospital=hospital.loc[hospital.bp_source.eq('nibp_cuff_union')].iloc[0]
mrow=comparison.loc[comparison.dataset.eq('mimic')&comparison.bp_source.eq(primary['mimic'])].iloc[0];erow=comparison.loc[comparison.dataset.eq('eicu')&comparison.bp_source.eq(primary['eicu'])].iloc[0];mcontrols=int(controls.loc[controls.dataset.eq('mimic')].stay_id.nunique());econtrols=int(controls.loc[controls.dataset.eq('eicu')].stay_id.nunique());severe=bool(primary_hospital.top1_crossing_pct>25 or primary_hospital.top5_crossing_pct>60);classification,action=source_repair_feasibility(int(mrow.no_prior_support_crossing_n),int(erow.no_prior_support_crossing_n),mcontrols,econtrols,int(primary_hospital.hospital_n),severe_center_concentration=severe)
mapping_lock_record=json.loads((OUT/'source_repair_mapping_lock.json').read_text())
decision=dict(all_required_eicu_sources_present=all(r['exists'] for r in inventory),vitalaperiodic_mapped=True,nursecharting_mapped=True,nursecharting_chunked=True,cross_source_duplicates_audited=True,cuff_union_validated=True,mimic_routine_bp_supported=bool(mrow.no_prior_support_crossing_n>=150 and mcontrols>=300),eicu_routine_bp_supported=bool(erow.no_prior_support_crossing_n>=150 and econtrols>=300),eicu_multicenter_supported=bool(primary_hospital.hospital_n>=10 and not severe),exposure_feasibility=classification,outcomes_inspected=False,post_anchor_lactate_values_inspected=False,phase_b_executed=False,mimic_compensated_cuff_crossings=int(mrow.no_prior_support_crossing_n),eicu_compensated_cuff_crossings=int(erow.no_prior_support_crossing_n),mimic_controls=mcontrols,eicu_controls=econtrols,eicu_crossing_hospitals=int(primary_hospital.hospital_n),severe_center_concentration=severe,recommended_action=action,protocol_sha256=plan_sha,mapping_lock_sha256=mapping_lock_record['lock_sha256'],code_commit=CODE_COMMIT,code_state_sha256=CODE_STATE_SHA256,code_file_hashes=CODE_FILE_HASHES,execution_environment='google_colab' if IN_COLAB else 'local',start_utc=mapping_lock_record['frozen_utc'],end_utc=datetime.now(timezone.utc).isoformat())
sr_json('source_repair_decision.json',decision)
artifact_names=['source_repair_inventory.csv','nurse_bp_mapping_audit.csv','bp_cross_source_overlap_audit.csv','source_feasibility_comparison.csv','source_repair_attrition.csv','source_repair_hospital_coverage.csv','source_repair_decision.json','source_repair_mapping_lock.json'];sr_json('source_repair_artifact_hashes.json',{name:sr_hash(OUT/name) for name in artifact_names})
sr_json('source_repair_run_status.json',dict(status='completed',stage='outcome_blind_feasibility_complete',environment=decision['execution_environment'],start_utc=decision['start_utc'],end_utc=decision['end_utc'],outcomes_inspected=False,post_anchor_lactate_values_inspected=False,phase_b_executed=False,recommended_action=action,protocol_sha256=plan_sha))
print(comparison.to_string(index=False),flush=True);print(hospital.to_string(index=False),flush=True);print(json.dumps(decision,indent=2),flush=True)
'''

__all__ = [
    'EXTRACT_AND_DEDUP',
    'HARMONIZE',
    'LABEL_DISCOVERY',
    'LACTATE_AVAILABILITY',
    'REPORT',
    'SETUP',
]
