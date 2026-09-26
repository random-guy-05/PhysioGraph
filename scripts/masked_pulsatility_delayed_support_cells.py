"""Locked masked-pulsatility delayed-support notebook cells."""

SETUP = r'''
import hashlib,importlib.util,json,math,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3','scipy':'scipy==1.15.3','statsmodels':'statsmodels==0.14.5','matplotlib':'matplotlib==3.10.6','tabulate':'tabulate==0.9.0'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,matplotlib.pyplot as plt,numpy as np,pandas as pd,statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905';MIMIC=DRIVE/'Data/MIMIC/Full';EICU=DRIVE/'Data/eICU/Full';OUT=PROJECT/'research/masked_pulsatility/delayed_support';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.masked_pulsatility import EICU_SUPPORT_ALIASES,MIMIC_SUPPORT_DRUGS,SUPPORT_BIN_LABELS,audit_future_control_crossings,build_two_hour_landmark,canonical_eicu_support_drug,delayed_support_classification,fit_modified_poisson,kish_effective_sample_size,merge_continuous_infusion_intervals,overlap_weights,prepare_frozen_covariates,protocol_precedes_outcomes,random_effects_meta,sha256_file,unadjusted_binary_risk
def ds_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ds_q(value):return "'"+str(value).replace("'","''")+"'"
def ds_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def ds_register(name,frame):
 frame=frame.copy()
 for col in frame.select_dtypes(include=['string']).columns:frame[col]=frame[col].astype(object)
 con.register(name,frame)
def ds_log(message):
 stamp=datetime.now(timezone.utc).isoformat();print(f'[{stamp}] {message}',flush=True)
 with (OUT/'execution.log').open('a') as handle:handle.write(f'[{stamp}] {message}\n')
DOC=PROJECT/'docs/MASKED_PULSATILITY_DELAYED_SUPPORT_PROTOCOL.md';ROOT_PROTOCOL=PROJECT/'research/masked_pulsatility/delayed_support_protocol.json';PROTOCOL=OUT/'protocol.json';FREEZE_RECEIPT=OUT/'protocol_freeze_receipt.json'
expected={'documentation_sha256':'c0289bb8f75df767b53c6e94ff7be479d0a6afbc28125a4125499269d362050b','root_protocol_json_sha256':'a21b9dedb937df26f385365a1049a1c943af1b4d7ca582d455ec521dc78ac8b6','execution_protocol_json_sha256':'469bc70fe6b8d4e755ecf49e9b84caa381ed0ad8e82f25ba6887d03d51283a8a'}
actual={'documentation_sha256':ds_hash(DOC),'root_protocol_json_sha256':ds_hash(ROOT_PROTOCOL),'execution_protocol_json_sha256':ds_hash(PROTOCOL)};assert actual==expected,(actual,expected)
protocol=json.loads(PROTOCOL.read_text());freeze=json.loads(FREEZE_RECEIPT.read_text());assert freeze['outcomes_inspected'] is False and freeze['first_outcome_read_utc'] is None
try:con.close()
except (NameError,AttributeError):pass
DB=PRIVATE/'masked_pulsatility_delayed_support.duckdb';con=duckdb.connect(str(DB));con.execute('SET threads=3');con.execute("SET memory_limit='4GB'");con.execute('SET enable_progress_bar=false')
phase=PRIVATE/'masked_pulsatility_phase_a.duckdb';repair=PRIVATE/'masked_pulsatility_source_repair.duckdb';con.execute(f"ATTACH {ds_q(phase)} AS phase (READ_ONLY)");con.execute(f"ATTACH {ds_q(repair)} AS repair (READ_ONLY)")
con.execute('DROP TABLE IF EXISTS delayed_anchors');con.execute("""CREATE TABLE delayed_anchors AS SELECT *,1 exposed FROM repair.source_repair_primary_exposed UNION ALL BY NAME SELECT *,0 exposed FROM repair.source_repair_controls""")
assert con.execute("SELECT count(*) FROM delayed_anchors WHERE dataset='mimic' AND exposed=1").fetchone()[0]==1123
assert con.execute("SELECT count(*) FROM delayed_anchors WHERE dataset='eicu' AND exposed=1").fetchone()[0]==577
run_start=datetime.now(timezone.utc).isoformat();ds_json('run_status.json',{'status':'running','stage':'protocol_verified_preoutcome','protocol_freeze_utc':protocol['freeze_utc'],'first_outcome_read_utc':None,'outcomes_inspected':False,'execution_environment':'google_colab' if IN_COLAB else 'local','start_utc':run_start})
ds_log('protocol hashes verified; no post-anchor outcome rows loaded')
'''

BASELINE = r'''
con.execute('DROP TABLE IF EXISTS demographic_features')
con.execute(f"""CREATE TABLE demographic_features AS
SELECT 'mimic' dataset,a.stay_id,try_cast(p.anchor_age AS DOUBLE)+(year(c.admit_time)-try_cast(p.anchor_year AS DOUBLE)) age,CASE WHEN upper(trim(p.gender))='M' THEN 1 WHEN upper(trim(p.gender))='F' THEN 0 ELSE NULL END male_sex
FROM delayed_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'patients.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.subject_id AS BIGINT)=try_cast(c.person_id AS BIGINT) WHERE a.dataset='mimic'
UNION ALL
SELECT 'eicu',a.stay_id,try_cast(replace(trim(p.age),'>','') AS DOUBLE),CASE WHEN lower(trim(p.gender))='male' THEN 1 WHEN lower(trim(p.gender))='female' THEN 0 ELSE NULL END
FROM delayed_anchors a JOIN read_csv({ds_q(EICU/'patient.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu'""")
if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='preanchor_vitals'").fetchone()[0]:
 con.execute(f"""CREATE TABLE mimic_preanchor_vitals AS
 SELECT 'mimic' dataset,a.stay_id,a.candidate_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
 CASE WHEN try_cast(v.itemid AS BIGINT)=220045 THEN 'hr' WHEN try_cast(v.itemid AS BIGINT) IN (220210,224689,224690) THEN 'resp_rate' WHEN try_cast(v.itemid AS BIGINT)=220277 THEN 'spo2' END concept,try_cast(v.valuenum AS DOUBLE) AS "value"
 FROM delayed_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'chartevents.csv')},header=true,all_varchar=true,parallel=true) v ON try_cast(v.stay_id AS BIGINT)=a.stay_id
 WHERE a.dataset='mimic' AND try_cast(v.itemid AS BIGINT) IN (220045,220210,224689,224690,220277)
 AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 BETWEEN greatest(0,a.anchor_minute-240) AND a.anchor_minute
 AND date_diff('second',c.admit_time,try_cast(v.storetime AS TIMESTAMP))/60.0<=a.anchor_minute""")
 con.execute(f"""CREATE TABLE eicu_preanchor_vitals AS
 SELECT 'eicu' dataset,a.stay_id,a.candidate_id,try_cast(v.observationoffset AS DOUBLE) event_minute,u.concept,u."value"
 FROM delayed_anchors a JOIN read_csv({ds_q(EICU/'vitalPeriodic.csv')},header=true,all_varchar=true,parallel=true) v ON try_cast(v.patientunitstayid AS BIGINT)=a.stay_id
 CROSS JOIN LATERAL (VALUES ('hr',try_cast(v.heartrate AS DOUBLE)),('resp_rate',try_cast(v.respiration AS DOUBLE)),('spo2',try_cast(v.sao2 AS DOUBLE))) u(concept,"value")
 WHERE a.dataset='eicu' AND try_cast(v.observationoffset AS DOUBLE) BETWEEN greatest(0,a.anchor_minute-240) AND a.anchor_minute""")
 con.execute("""CREATE TABLE preanchor_vitals AS SELECT * FROM mimic_preanchor_vitals WHERE (concept='hr' AND "value" BETWEEN 20 AND 250) OR (concept='resp_rate' AND "value" BETWEEN 4 AND 80) OR (concept='spo2' AND "value" BETWEEN 50 AND 100) UNION ALL SELECT * FROM eicu_preanchor_vitals WHERE (concept='hr' AND "value" BETWEEN 20 AND 250) OR (concept='resp_rate' AND "value" BETWEEN 4 AND 80) OR (concept='spo2' AND "value" BETWEEN 50 AND 100)""")
if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='preanchor_creatinine'").fetchone()[0]:
 con.execute(f"""CREATE TABLE mimic_preanchor_creatinine AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,try_cast(l.valuenum AS DOUBLE) AS "value" FROM delayed_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'labevents.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id WHERE a.dataset='mimic' AND try_cast(l.itemid AS BIGINT) IN (50912,52024,52546) AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 BETWEEN a.anchor_minute-1440 AND a.anchor_minute AND date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0<=a.anchor_minute AND try_cast(l.valuenum AS DOUBLE) BETWEEN 0.1 AND 30""")
 con.execute(f"""CREATE TABLE eicu_preanchor_creatinine AS SELECT 'eicu' dataset,a.stay_id,a.candidate_id,try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,try_cast(l.labresult AS DOUBLE) AS "value" FROM delayed_anchors a JOIN read_csv({ds_q(EICU/'lab.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND lower(trim(l.labname))='creatinine' AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN a.anchor_minute-1440 AND a.anchor_minute AND try_cast(l.labresultrevisedoffset AS DOUBLE)<=a.anchor_minute AND try_cast(l.labresult AS DOUBLE) BETWEEN 0.1 AND 30""")
 con.execute('CREATE TABLE preanchor_creatinine AS SELECT * FROM mimic_preanchor_creatinine UNION ALL SELECT * FROM eicu_preanchor_creatinine')
con.execute('DROP TABLE IF EXISTS bp_preanchor_features');con.execute("""CREATE TABLE bp_preanchor_features AS SELECT a.candidate_id,arg_max(h.prpp,h.anchor_minute) FILTER(WHERE h.anchor_minute<a.anchor_minute) prior_prpp,regr_slope(h.sbp,h.anchor_minute/60.0) FILTER(WHERE h.anchor_minute BETWEEN a.anchor_minute-120 AND a.anchor_minute) sbp_slope_2h,regr_slope(h."map",h.anchor_minute/60.0) FILTER(WHERE h.anchor_minute BETWEEN a.anchor_minute-120 AND a.anchor_minute) map_slope_2h,sum(h.measurement_count) FILTER(WHERE h.anchor_minute BETWEEN a.anchor_minute-240 AND a.anchor_minute) bp_measurement_count_4h FROM delayed_anchors a LEFT JOIN repair.source_repair_hourly_bp h ON a.dataset=h.dataset AND a.stay_id=h.stay_id AND a.bp_source=h.bp_source GROUP BY a.candidate_id""")
con.execute('DROP TABLE IF EXISTS vital_preanchor_features');con.execute("""CREATE TABLE vital_preanchor_features AS SELECT a.candidate_id,arg_max(v."value",v.event_minute) FILTER(WHERE v.concept='hr' AND v.event_minute>=a.anchor_minute-60) hr_anchor,arg_max(v."value",v.event_minute) FILTER(WHERE v.concept='resp_rate' AND v.event_minute>=a.anchor_minute-60) resp_rate_anchor,arg_max(v."value",v.event_minute) FILTER(WHERE v.concept='spo2' AND v.event_minute>=a.anchor_minute-60) spo2_anchor,regr_slope(v."value",v.event_minute/60.0) FILTER(WHERE v.concept='hr' AND v.event_minute BETWEEN a.anchor_minute-120 AND a.anchor_minute) hr_slope_2h FROM delayed_anchors a LEFT JOIN preanchor_vitals v ON a.candidate_id=v.candidate_id GROUP BY a.candidate_id""")
con.execute('DROP TABLE IF EXISTS creatinine_preanchor_features');con.execute("""CREATE TABLE creatinine_preanchor_features AS SELECT a.candidate_id,arg_max(c."value",c.event_minute) creatinine_preanchor_24h FROM delayed_anchors a LEFT JOIN preanchor_creatinine c ON a.candidate_id=c.candidate_id GROUP BY a.candidate_id""")
con.execute('DROP TABLE IF EXISTS shock_preanchor_features');con.execute("""CREATE TABLE shock_preanchor_features AS WITH vh AS (SELECT dataset,stay_id,candidate_id,floor(event_minute/60) vital_hour,median("value") hr FROM preanchor_vitals WHERE concept='hr' GROUP BY 1,2,3,4), joined AS (SELECT a.candidate_id,h."hour" bp_hour,vh.hr/h.sbp shock_index FROM delayed_anchors a JOIN repair.source_repair_hourly_bp h ON a.dataset=h.dataset AND a.stay_id=h.stay_id AND a.bp_source=h.bp_source LEFT JOIN vh ON a.candidate_id=vh.candidate_id AND h."hour"=vh.vital_hour WHERE h.anchor_minute BETWEEN a.anchor_minute-120 AND a.anchor_minute) SELECT candidate_id,arg_max(shock_index,bp_hour) shock_index,regr_slope(shock_index,bp_hour) shock_index_slope_2h FROM joined GROUP BY candidate_id""")
con.execute('DROP TABLE IF EXISTS delayed_baseline');con.execute("""CREATE TABLE delayed_baseline AS SELECT a.*,d.age,d.male_sex,b.prior_prpp,b.sbp_slope_2h,b.map_slope_2h,coalesce(b.bp_measurement_count_4h,a.measurement_count) bp_measurement_count_4h,v.hr_anchor,v.resp_rate_anchor,v.spo2_anchor,v.hr_slope_2h,c.creatinine_preanchor_24h,s.shock_index,s.shock_index_slope_2h,a.sbp sbp_anchor,a."map" map_anchor,a.prpp anchor_prpp,a.anchor_minute/60.0 anchor_hour,l.baseline_lactate,l.lactate_to_anchor_lag FROM delayed_anchors a LEFT JOIN demographic_features d USING(dataset,stay_id) LEFT JOIN bp_preanchor_features b USING(candidate_id) LEFT JOIN vital_preanchor_features v USING(candidate_id) LEFT JOIN creatinine_preanchor_features c USING(candidate_id) LEFT JOIN shock_preanchor_features s USING(candidate_id) LEFT JOIN repair.source_repair_anchor_audit l USING(candidate_id)""")
baseline=con.execute('SELECT * FROM delayed_baseline').df();assert baseline.candidate_id.is_unique
future=audit_future_control_crossings(con.execute("SELECT dataset,stay_id,candidate_id,anchor_minute FROM delayed_anchors WHERE exposed=0").df(),con.execute("SELECT dataset,stay_id,anchor_minute FROM repair.source_repair_crossings WHERE bp_source IN ('nibp_mimic_chartevents','nibp_cuff_union')").df())
pd.DataFrame([{'known_future_control_crossings_first4h':len(future),'controls':int((baseline.exposed==0).sum()),'note':'locked control pool excludes known crossings in the full first-four-hour exposure window'}]).to_csv(OUT/'control_crossing_audit.csv',index=False)
assert future.empty
ds_log(f'outcome-blind baseline complete: {len(baseline)} anchors; known early-window control crossings={len(future)}')
'''

LOCK_ANALYSIS = r'''
code_files=['scripts/build_biological_notebook.py','scripts/masked_pulsatility_delayed_support_cells.py','src/physiograph/analysis/masked_pulsatility.py','tests/unit/test_masked_pulsatility.py']
code_hashes={name:ds_hash(PROJECT/name) for name in code_files};dirty_code_sha=hashlib.sha256(json.dumps(code_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest();git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
source_paths={'mimic_inputevents':MIMIC/'inputevents.csv','mimic_chartevents':MIMIC/'chartevents.csv','mimic_labevents':MIMIC/'labevents.csv','mimic_procedureevents':MIMIC/'procedureevents.csv','mimic_patients':MIMIC/'patients.csv','mimic_admissions':MIMIC/'admissions.csv','eicu_infusionDrug':EICU/'infusionDrug.csv','eicu_vitalPeriodic':EICU/'vitalPeriodic.csv','eicu_lab':EICU/'lab.csv','eicu_treatment':EICU/'treatment.csv','eicu_patient':EICU/'patient.csv'}
source_manifest={name:{'path':str(path),'byte_size':path.stat().st_size,'sha256':ds_hash(path) if path.stat().st_size<=500_000_000 else None,'hash_note':'omitted_over_500MB' if path.stat().st_size>500_000_000 else None} for name,path in source_paths.items()}
analysis_config={'protocol_sha256':expected['execution_protocol_json_sha256'],'documentation_sha256':expected['documentation_sha256'],'freeze_utc':protocol['freeze_utc'],'code_hashes':code_hashes,'dirty_code_sha256':dirty_code_sha,'git_commit':git_commit,'source_manifest':source_manifest,'execution_order':['mimic','eicu'],'primary_covariates':protocol['primary_covariates'],'outcome':protocol['outcome'],'drugs':{'mimic':protocol['mimic_drugs'],'eicu':protocol['eicu_drug_aliases']}}
ds_json('analysis_config.json',analysis_config);analysis_config_sha=ds_hash(OUT/'analysis_config.json')
analysis_lock={'lock_utc':datetime.now(timezone.utc).isoformat(),'protocol_freeze_utc':protocol['freeze_utc'],'protocol_hashes':expected,'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'git_commit':git_commit,'first_outcome_read_utc':None,'outcomes_inspected':False}
ds_json('analysis_lock.json',analysis_lock)
mapping=[]
for itemid,drug in MIMIC_SUPPORT_DRUGS.items():mapping.append({'dataset':'mimic','source_table':'inputevents.csv','source_identifier':str(itemid),'canonical_drug':drug,'definition':'positive-rate valid administration interval'})
for drug,aliases in EICU_SUPPORT_ALIASES.items():
 for alias in aliases:mapping.append({'dataset':'eicu','source_table':'infusionDrug.csv','source_identifier':alias,'canonical_drug':drug,'definition':'positive numeric drugrate or infusionrate'})
pd.DataFrame(mapping).to_csv(OUT/'support_drug_mapping.csv',index=False)
ds_json('run_status.json',{'status':'running','stage':'analysis_code_and_config_locked_preoutcome','protocol_freeze_utc':protocol['freeze_utc'],'analysis_lock_utc':analysis_lock['lock_utc'],'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'first_outcome_read_utc':None,'outcomes_inspected':False,'execution_environment':'google_colab' if IN_COLAB else 'local','start_utc':run_start})
ds_log(f'analysis locked before outcomes: config={analysis_config_sha} code={dirty_code_sha}')
'''

MIMIC_OUTCOME = r'''
assert ds_hash(PROTOCOL)==expected['execution_protocol_json_sha256'];assert ds_hash(OUT/'analysis_config.json')==analysis_config_sha;assert {name:ds_hash(PROJECT/name) for name in code_files}==code_hashes
FIRST_OUTCOME_READ_UTC=datetime.now(timezone.utc).isoformat();assert protocol_precedes_outcomes(protocol['freeze_utc'],FIRST_OUTCOME_READ_UTC)
outcome_receipt={'protocol_freeze_utc':protocol['freeze_utc'],'analysis_lock_utc':analysis_lock['lock_utc'],'first_outcome_read_utc':FIRST_OUTCOME_READ_UTC,'mimic_execution_utc':FIRST_OUTCOME_READ_UTC,'eicu_execution_utc':None,'protocol_hashes':expected,'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'protocol_preceded_outcomes':True,'execution_environment':'google_colab' if IN_COLAB else 'local'};ds_json('outcome_read_receipt.json',outcome_receipt)
ds_log('FIRST OUTCOME READ: MIMIC inputevents positive-rate support rows')
con.execute('DROP TABLE IF EXISTS mimic_support_rows')
con.execute(f"""CREATE TABLE mimic_support_rows AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,try_cast(i.itemid AS BIGINT) itemid,date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0 start_minute,date_diff('second',c.admit_time,try_cast(i.endtime AS TIMESTAMP))/60.0 end_minute,try_cast(i.rate AS DOUBLE) rate FROM delayed_anchors a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'inputevents.csv')},header=true,all_varchar=true,parallel=true) i ON try_cast(i.stay_id AS BIGINT)=a.stay_id WHERE a.dataset='mimic' AND try_cast(i.itemid AS BIGINT) IN (221906,221289,221662,221653,221749,222315,221986) AND try_cast(i.rate AS DOUBLE)>0 AND try_cast(i.endtime AS TIMESTAMP)>try_cast(i.starttime AS TIMESTAMP) AND date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0<=a.anchor_minute+720""")
mimic_rows=con.execute('SELECT dataset,stay_id,itemid,start_minute,end_minute,rate FROM mimic_support_rows').df();mimic_rows['drug']=mimic_rows.itemid.map(MIMIC_SUPPORT_DRUGS);mimic_episodes=merge_continuous_infusion_intervals(mimic_rows[['dataset','stay_id','drug','start_minute','end_minute','rate']]);con.execute('DROP TABLE IF EXISTS mimic_support_episodes');ds_register('_mimic_episodes',mimic_episodes);con.execute('CREATE TABLE mimic_support_episodes AS SELECT * FROM _mimic_episodes');con.unregister('_mimic_episodes')
prior=con.execute("SELECT count(DISTINCT a.candidate_id) FROM delayed_anchors a JOIN mimic_support_episodes e USING(dataset,stay_id) WHERE a.dataset='mimic' AND e.start_minute<=a.anchor_minute").fetchone()[0]
ds_log(f'MIMIC outcome extraction complete: {len(mimic_episodes)} episodes; prior-support revalidation exclusions={prior}')
'''

EICU_OUTCOME = r'''
assert ds_hash(PROTOCOL)==expected['execution_protocol_json_sha256'];assert ds_hash(OUT/'analysis_config.json')==analysis_config_sha;assert {name:ds_hash(PROJECT/name) for name in code_files}==code_hashes;assert dirty_code_sha==analysis_lock['dirty_code_sha256']
EICU_EXECUTION_UTC=datetime.now(timezone.utc).isoformat();outcome_receipt['eicu_execution_utc']=EICU_EXECUTION_UTC;ds_json('outcome_read_receipt.json',outcome_receipt);ds_log('LOCKED EXTERNAL CONFIRMATION READ: eICU infusionDrug positive-rate support rows')
con.execute('DROP TABLE IF EXISTS eicu_support_rows')
con.execute(f"""CREATE TABLE eicu_support_rows AS SELECT 'eicu' dataset,a.stay_id,a.candidate_id,try_cast(i.infusionoffset AS DOUBLE) start_minute,i.drugname,coalesce(try_cast(i.drugrate AS DOUBLE),try_cast(i.infusionrate AS DOUBLE)) rate FROM delayed_anchors a JOIN read_csv({ds_q(EICU/'infusionDrug.csv')},header=true,all_varchar=true,parallel=true) i ON try_cast(i.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND coalesce(try_cast(i.drugrate AS DOUBLE),try_cast(i.infusionrate AS DOUBLE))>0 AND try_cast(i.infusionoffset AS DOUBLE)<=a.anchor_minute+720""")
eicu_rows=con.execute('SELECT dataset,stay_id,start_minute,drugname,rate FROM eicu_support_rows').df();eicu_rows['drug']=eicu_rows.drugname.map(canonical_eicu_support_drug);eicu_rows=eicu_rows.dropna(subset=['drug']).drop_duplicates(['dataset','stay_id','drug','start_minute']);eicu_starts=eicu_rows[['dataset','stay_id','start_minute','drug']].copy();con.execute('DROP TABLE IF EXISTS eicu_support_starts');ds_register('_eicu_starts',eicu_starts);con.execute('CREATE TABLE eicu_support_starts AS SELECT * FROM _eicu_starts');con.unregister('_eicu_starts')
prior=con.execute("SELECT count(DISTINCT a.candidate_id) FROM delayed_anchors a JOIN eicu_support_starts e USING(dataset,stay_id) WHERE a.dataset='eicu' AND e.start_minute<=a.anchor_minute").fetchone()[0]
ds_log(f'eICU locked extraction complete: {len(eicu_starts)} positive mapped rows; prior-support revalidation exclusions={prior}')
'''

PRIMARY = r'''
mimic_starts=mimic_episodes[['dataset','stay_id','start_minute','drug']].copy();starts=pd.concat([mimic_starts,eicu_starts],ignore_index=True)
anchors=con.execute('SELECT * FROM delayed_baseline').df();prior_candidates=set()
for dataset,group in starts.groupby('dataset'):
 merged=anchors.loc[anchors.dataset.eq(dataset),['dataset','stay_id','candidate_id','anchor_minute']].merge(group,on=['dataset','stay_id'],how='inner');prior_candidates.update(merged.loc[merged.start_minute.le(merged.anchor_minute),'candidate_id'])
anchors=anchors.loc[~anchors.candidate_id.isin(prior_candidates)].copy();classified,flow=build_two_hour_landmark(anchors,starts);flow['prior_support_revalidation_exclusions']=flow.apply(lambda r:sum(1 for c in prior_candidates if c.startswith(str(r.dataset)+'|')),axis=1);flow.to_csv(OUT/'cohort_flow.csv',index=False)
con.execute('DROP TABLE IF EXISTS delayed_landmark');ds_register('_classified',classified);con.execute('CREATE TABLE delayed_landmark AS SELECT * FROM _classified');con.unregister('_classified')
landmark=classified.loc[classified.landmark_eligible].copy();landmark['exposed']=landmark.exposed.astype(int)
covariates=['age','male_sex','sbp_anchor','map_anchor','hr_anchor','sbp_slope_2h','map_slope_2h','hr_slope_2h','prior_prpp','resp_rate_anchor','spo2_anchor','creatinine_preanchor_24h','bp_measurement_count_4h','anchor_hour']
primary_results={};prepared_by_dataset={};columns_by_dataset={};fits={};diagnostics=[]
for dataset in ['mimic','eicu']:
 frame=landmark.loc[landmark.dataset.eq(dataset)].copy();prepared,columns,diag=prepare_frozen_covariates(frame,covariates);prepared_by_dataset[dataset]=prepared;columns_by_dataset[dataset]=columns;diag.insert(0,'dataset',dataset);diagnostics.append(diag)
 unadjusted=unadjusted_binary_risk(prepared,'delayed_support_2_12h');adjusted,fit=fit_modified_poisson(prepared,outcome='delayed_support_2_12h',covariates=columns,cluster_column='hospital_id' if dataset=='eicu' else None);fits[dataset]=fit;primary_results[dataset]={'unadjusted':unadjusted,'adjusted':adjusted}
 pd.DataFrame([{'dataset':dataset,'model':'unadjusted',**unadjusted},{'dataset':dataset,'model':'adjusted',**adjusted}]).to_csv(OUT/f'primary_{dataset}.csv',index=False)
con.execute('DROP TABLE IF EXISTS delayed_model_frame');all_prepared=pd.concat(prepared_by_dataset.values(),ignore_index=True,sort=False);ds_register('_prepared',all_prepared);con.execute('CREATE TABLE delayed_model_frame AS SELECT * FROM _prepared');con.unregister('_prepared')
pd.concat(diagnostics,ignore_index=True).to_csv(OUT/'covariate_processing.csv',index=False)
balance=[]
for dataset,frame in prepared_by_dataset.items():
 for raw in covariates:
  x=pd.to_numeric(frame[raw],errors='coerce');g=frame.exposed.eq(1);den=math.sqrt((x[g].var()+x[~g].var())/2);balance.append({'dataset':dataset,'covariate':raw,'exposed_mean':x[g].mean(),'control_mean':x[~g].mean(),'smd':(x[g].mean()-x[~g].mean())/den if den and np.isfinite(den) else 0,'exposed_missing_pct':100*x[g].isna().mean(),'control_missing_pct':100*x[~g].isna().mean()})
pd.DataFrame(balance).to_csv(OUT/'anchor_balance.csv',index=False)
vif=[]
for dataset,frame in prepared_by_dataset.items():
 cols=columns_by_dataset[dataset];x=frame[cols].astype(float)
 for j,col in enumerate(cols):vif.append({'dataset':dataset,'covariate':col,'vif':variance_inflation_factor(x.to_numpy(),j) if x[col].std()>0 else np.nan,'condition_number':float(np.linalg.cond(sm.add_constant(x,has_constant='add').to_numpy()))})
pd.DataFrame(vif).to_csv(OUT/'multicollinearity_diagnostics.csv',index=False)
meta_input=pd.DataFrame([{'dataset':d,'log_rr':primary_results[d]['adjusted']['log_rr'],'log_rr_se':primary_results[d]['adjusted']['log_rr_se']} for d in ['mimic','eicu']]);meta=random_effects_meta(meta_input);pd.DataFrame([{**meta,**{f'{d}_adjusted_rr':primary_results[d]['adjusted']['adjusted_rr'] for d in ['mimic','eicu']}}]).to_csv(OUT/'primary_meta.csv',index=False)
blank=[]
for dataset in ['mimic','eicu']:
 for exposed in [0,1]:
  s=classified.loc[classified.dataset.eq(dataset)&classified.exposed.eq(exposed)];blank.append({'dataset':dataset,'exposed':exposed,'n':len(s),'blanking_events':int(s.blanking_support.sum()),'blanking_risk':float(s.blanking_support.mean())})
pd.DataFrame(blank).to_csv(OUT/'blanking_period_results.csv',index=False)
lead=classified.loc[classified.lead_minutes.gt(0)&classified.lead_minutes.le(720)].copy();lead.groupby(['dataset','exposed']).lead_minutes.agg(n='size',median_minutes='median',q1_minutes=lambda x:x.quantile(.25),q3_minutes=lambda x:x.quantile(.75)).reset_index().to_csv(OUT/'lead_time_distribution.csv',index=False);lead.groupby(['dataset','exposed','support_lead_bin'],observed=False).size().rename('events').reset_index().to_csv(OUT/'lead_time_bins.csv',index=False)
falsification=[];static_dynamic=[]
for dataset,frame in prepared_by_dataset.items():
 primary=primary_results[dataset]['adjusted'];falsification.append({'dataset':dataset,'model':'sbp_map_level_and_trajectory','adjusted_rr':primary['adjusted_rr'],'ci_low':primary['adjusted_rr_ci_low'],'ci_high':primary['adjusted_rr_ci_high'],'adjusted_rd':primary['adjusted_rd'],'rd_ci_low':primary['adjusted_rd_ci_low'],'rd_ci_high':primary['adjusted_rd_ci_high']})
 comparator_raw=[c for c in covariates if c not in {'hr_anchor','sbp_anchor','map_anchor','hr_slope_2h','sbp_slope_2h','map_slope_2h'}]+['shock_index','shock_index_slope_2h'];comp,comp_cols,_=prepare_frozen_covariates(frame,comparator_raw);comp_result,_=fit_modified_poisson(comp,outcome='delayed_support_2_12h',covariates=comp_cols,cluster_column='hospital_id' if dataset=='eicu' else None);falsification.append({'dataset':dataset,'model':'shock_index_comparator','adjusted_rr':comp_result['adjusted_rr'],'ci_low':comp_result['adjusted_rr_ci_low'],'ci_high':comp_result['adjusted_rr_ci_high'],'adjusted_rd':comp_result['adjusted_rd'],'rd_ci_low':comp_result['adjusted_rd_ci_low'],'rd_ci_high':comp_result['adjusted_rd_ci_high']})
 static_frame,static_cols,_=prepare_frozen_covariates(frame,covariates+['anchor_prpp']);static_only=[c for c in static_cols];static_design=sm.add_constant(static_frame[static_only].astype(float),has_constant='add');static_fit=sm.GLM(static_frame.delayed_support_2_12h.astype(int),static_design,family=sm.families.Poisson()).fit();dynamic_result,dynamic_fit=fit_modified_poisson(static_frame,outcome='delayed_support_2_12h',covariates=static_only,cluster_column='hospital_id' if dataset=='eicu' else None);static_dynamic.append({'dataset':dataset,'static_aic':float(static_fit.aic),'dynamic_aic':dynamic_result['aic'],'aic_improvement':float(static_fit.aic-dynamic_result['aic']),'dynamic_rr':dynamic_result['adjusted_rr'],'dynamic_ci_low':dynamic_result['adjusted_rr_ci_low'],'dynamic_ci_high':dynamic_result['adjusted_rr_ci_high']})
pd.DataFrame(falsification).to_csv(OUT/'sbp_map_falsification.csv',index=False);pd.DataFrame(static_dynamic).to_csv(OUT/'static_vs_dynamic.csv',index=False)
ds_json('run_status.json',{'status':'running','stage':'primary_analysis_complete','protocol_freeze_utc':protocol['freeze_utc'],'first_outcome_read_utc':FIRST_OUTCOME_READ_UTC,'mimic_execution_utc':outcome_receipt['mimic_execution_utc'],'eicu_execution_utc':outcome_receipt['eicu_execution_utc'],'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'outcomes_inspected':True,'secondary_outcomes_inspected':False,'execution_environment':'google_colab' if IN_COLAB else 'local','start_utc':run_start})
ds_log('primary delayed-support analysis, falsification, and meta-analysis complete')
'''

ROBUSTNESS = r'''
weighted_rows=[];smd_rows=[];lactate_rows=[];observation_rows=[]
for dataset,frame in prepared_by_dataset.items():
 columns=columns_by_dataset[dataset];propensity,weights,smd=overlap_weights(frame,columns);frame=frame.copy();frame['propensity']=propensity;frame['overlap_weight']=weights;smd.insert(0,'dataset',dataset);smd_rows.append(smd)
 weighted,fit=fit_modified_poisson(frame,outcome='delayed_support_2_12h',covariates=[],cluster_column='hospital_id' if dataset=='eicu' else None,weights='overlap_weight');common1=((propensity[frame.exposed.eq(1)]>=propensity[frame.exposed.eq(0)].min())&(propensity[frame.exposed.eq(1)]<=propensity[frame.exposed.eq(0)].max())).mean();common0=((propensity[frame.exposed.eq(0)]>=propensity[frame.exposed.eq(1)].min())&(propensity[frame.exposed.eq(0)]<=propensity[frame.exposed.eq(1)].max())).mean();weighted_rows.append({'dataset':dataset,**weighted,'ess':kish_effective_sample_size(weights),'ess_fraction':kish_effective_sample_size(weights)/len(frame),'max_abs_smd_after':float(smd.smd_after.abs().max()),'exposed_common_support_fraction':common1,'control_common_support_fraction':common0,'propensity_exposed_q01':propensity[frame.exposed.eq(1)].quantile(.01),'propensity_exposed_q50':propensity[frame.exposed.eq(1)].quantile(.5),'propensity_exposed_q99':propensity[frame.exposed.eq(1)].quantile(.99),'propensity_control_q01':propensity[frame.exposed.eq(0)].quantile(.01),'propensity_control_q50':propensity[frame.exposed.eq(0)].quantile(.5),'propensity_control_q99':propensity[frame.exposed.eq(0)].quantile(.99)})
 measured=frame.loc[frame.baseline_lactate.notna()].copy();obs=frame.baseline_lactate.notna().astype(int);obs_design=sm.add_constant(frame[columns].astype(float),has_constant='add');obs_fit=sm.GLM(obs,obs_design,family=sm.families.Binomial()).fit();obs_p=pd.Series(np.clip(obs_fit.predict(obs_design),.001,.999),index=frame.index);raw_iow=pd.Series(float(obs.mean())/obs_p.loc[measured.index],index=measured.index);lo,hi=raw_iow.quantile([.01,.99]);measured['observation_weight']=raw_iow.clip(lo,hi);lactate_prepared,lactate_cols,_=prepare_frozen_covariates(measured,['baseline_lactate','lactate_to_anchor_lag']);lactate_cols=[*columns,*[c for c in lactate_cols if c not in columns]];unweighted,_=fit_modified_poisson(lactate_prepared,outcome='delayed_support_2_12h',covariates=lactate_cols,cluster_column='hospital_id' if dataset=='eicu' else None);weighted_lactate,_=fit_modified_poisson(lactate_prepared,outcome='delayed_support_2_12h',covariates=lactate_cols,cluster_column='hospital_id' if dataset=='eicu' else None,weights='observation_weight');lactate_rows.extend([{'dataset':dataset,'model':'measured_lactate_adjusted',**unweighted},{'dataset':dataset,'model':'observation_weighted_lactate_adjusted',**weighted_lactate}]);extreme=float(((obs_p<.05)|(obs_p>.95)).mean());ess=kish_effective_sample_size(measured.observation_weight);observation_rows.append({'dataset':dataset,'overall_observation_rate':float(obs.mean()),'exposed_observation_rate':float(obs[frame.exposed.eq(1)].mean()),'control_observation_rate':float(obs[frame.exposed.eq(0)].mean()),'probability_min':float(obs_p.min()),'probability_q01':float(obs_p.quantile(.01)),'probability_median':float(obs_p.median()),'probability_q99':float(obs_p.quantile(.99)),'probability_max':float(obs_p.max()),'extreme_probability_fraction':extreme,'measured_n':len(measured),'weighted_ess':ess,'ess_fraction_measured':ess/len(measured),'acceptable':bool(extreme<=.05 and ess/len(measured)>=.5 and measured.exposed.nunique()==2)})
pd.DataFrame(weighted_rows).to_csv(OUT/'overlap_weighting_results.csv',index=False);pd.concat(smd_rows,ignore_index=True).to_csv(OUT/'overlap_balance.csv',index=False);pd.DataFrame(lactate_rows).to_csv(OUT/'measured_lactate_analysis.csv',index=False);pd.DataFrame(observation_rows).to_csv(OUT/'lactate_observation_audit.csv',index=False)
hospital=[]
eicu=landmark.loc[landmark.dataset.eq('eicu')]
for hospital_id,group in eicu.groupby('hospital_id'):
 if group.exposed.nunique()==2:
  result=unadjusted_binary_risk(group,'delayed_support_2_12h');hospital.append({'hospital_id':hospital_id,**result})
pd.DataFrame(hospital).to_csv(OUT/'hospital_heterogeneity.csv',index=False)
ds_log('overlap weighting, measured-lactate, observation-process, and hospital audits complete')
'''

SECONDARY = r'''
assert json.loads((OUT/'run_status.json').read_text())['stage']=='primary_analysis_complete'
ds_log('secondary stage begins after primary completion: MCS, post-anchor lactate values, mortality')
con.execute('DROP TABLE IF EXISTS secondary_mcs')
con.execute(f"""CREATE TABLE secondary_mcs AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0 event_minute FROM delayed_landmark a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'procedureevents.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.stay_id AS BIGINT)=a.stay_id WHERE a.dataset='mimic' AND try_cast(p.itemid AS BIGINT) IN (224272,228169,229529,229530) AND date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0>a.anchor_minute+120 AND date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0<=a.anchor_minute+720 UNION ALL SELECT 'eicu',a.stay_id,a.candidate_id,try_cast(t.treatmentoffset AS DOUBLE) FROM delayed_landmark a JOIN read_csv({ds_q(EICU/'treatment.csv')},header=true,all_varchar=true,parallel=true) t ON try_cast(t.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND try_cast(t.treatmentoffset AS DOUBLE)>a.anchor_minute+120 AND try_cast(t.treatmentoffset AS DOUBLE)<=a.anchor_minute+720 AND regexp_matches(lower(t.treatmentstring),'iabp|intra.?aortic balloon|impella|ecmo|extracorporeal membrane|(^|[^a-z])lvad([^a-z]|$)|(^|[^a-z])rvad([^a-z]|$)|(^|[^a-z])bivad([^a-z]|$)') AND NOT regexp_matches(lower(t.treatmentstring),'remov|explant|discontinu')""")
con.execute('DROP TABLE IF EXISTS secondary_lactate')
con.execute(f"""CREATE TABLE secondary_lactate AS SELECT 'mimic' dataset,a.stay_id,a.candidate_id,date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,try_cast(l.valuenum AS DOUBLE) AS "value" FROM delayed_landmark a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'labevents.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id JOIN phase.mimic_lactate_items d ON try_cast(l.itemid AS BIGINT)=d.itemid WHERE a.dataset='mimic' AND a.baseline_lactate IS NOT NULL AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0>a.anchor_minute+120 AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0<=a.anchor_minute+720 AND date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0<=a.anchor_minute+720 AND regexp_matches(lower(coalesce(l.valueuom,'')),'mmol') UNION ALL SELECT 'eicu',a.stay_id,a.candidate_id,try_cast(l.labresultoffset AS DOUBLE),try_cast(l.labresult AS DOUBLE) FROM delayed_landmark a JOIN read_csv({ds_q(EICU/'lab.csv')},header=true,all_varchar=true,parallel=true) l ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu' AND a.baseline_lactate IS NOT NULL AND lower(trim(l.labname))='lactate' AND try_cast(l.labresultoffset AS DOUBLE)>a.anchor_minute+120 AND try_cast(l.labresultoffset AS DOUBLE)<=a.anchor_minute+720 AND try_cast(l.labresultrevisedoffset AS DOUBLE)<=a.anchor_minute+720""")
secondary=landmark[['dataset','stay_id','candidate_id','exposed','baseline_lactate']].copy();mcs=con.execute('SELECT DISTINCT candidate_id,1 mcs_event FROM secondary_mcs').df();post=con.execute('SELECT candidate_id,arg_min("value",event_minute) post_lactate FROM secondary_lactate WHERE "value" BETWEEN 0.1 AND 30 GROUP BY candidate_id').df();secondary=secondary.merge(mcs,on='candidate_id',how='left').merge(post,on='candidate_id',how='left');secondary['mcs_event']=secondary.mcs_event.fillna(0).astype(int);secondary['lactate_worsening']=np.where(secondary.baseline_lactate.notna()&secondary.post_lactate.notna(),(secondary.post_lactate-secondary.baseline_lactate>=.5).astype(float),np.nan)
mortality=[]
mimic_mort=con.execute(f"""SELECT a.candidate_id,try_cast(ad.hospital_expire_flag AS INTEGER) mortality FROM delayed_landmark a JOIN phase.mimic_cohort c USING(stay_id) JOIN read_csv({ds_q(MIMIC/'admissions.csv')},header=true,all_varchar=true,parallel=true) ad ON try_cast(ad.hadm_id AS BIGINT)=c.hadm_id WHERE a.dataset='mimic'""").df();eicu_mort=con.execute(f"""SELECT a.candidate_id,CASE WHEN lower(p.hospitaldischargestatus)='expired' THEN 1 WHEN p.hospitaldischargestatus IS NOT NULL THEN 0 ELSE NULL END mortality FROM delayed_landmark a JOIN read_csv({ds_q(EICU/'patient.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.patientunitstayid AS BIGINT)=a.stay_id WHERE a.dataset='eicu'""").df();mortality=pd.concat([mimic_mort,eicu_mort],ignore_index=True);secondary=secondary.merge(mortality,on='candidate_id',how='left')
secondary_rows=[]
for dataset in ['mimic','eicu']:
 for outcome in ['mcs_event','lactate_worsening','mortality']:
  s=secondary.loc[secondary.dataset.eq(dataset)&secondary[outcome].notna()].copy();result=unadjusted_binary_risk(s,outcome);secondary_rows.append({'dataset':dataset,'outcome':outcome,**result,'corroborative_only':True})
pd.DataFrame(secondary_rows).to_csv(OUT/'secondary_results.csv',index=False)
ds_log('fixed corroborative secondary analyses complete')
'''

REPORT = r'''
weighted=pd.DataFrame(weighted_rows);lactate_result=pd.DataFrame(lactate_rows);observation=pd.DataFrame(observation_rows);static_result=pd.DataFrame(static_dynamic);blanking=pd.DataFrame(blank)
def row(dataset,model='adjusted'):return primary_results[dataset][model]
eicu=row('eicu');mimic=row('mimic');eicu_un=row('eicu','unadjusted');mimic_sig=bool(mimic['adjusted_rr']>1 and mimic['adjusted_rr_ci_low']>1);eicu_confirm=bool(eicu['adjusted_rr']>1 and eicu['adjusted_rr_ci_low']>1);survives=bool(eicu['adjusted_rr']>=1.25 and eicu['adjusted_rr_ci_low']>1);dynamic_e=static_result.loc[static_result.dataset.eq('eicu')].iloc[0];dynamic=bool(dynamic_e.dynamic_rr>1 and dynamic_e.dynamic_ci_low>1 and dynamic_e.aic_improvement>=2);measured=bool(all(lactate_result.loc[(lactate_result.dataset.eq(d))&(lactate_result.model.eq('measured_lactate_adjusted')),'adjusted_rr'].iloc[0]>1 for d in ['mimic','eicu']));obs_ok=bool(observation.acceptable.all());weight_ok=bool((weighted.ess_fraction.ge(.25)&weighted.max_abs_smd_after.le(.10)&weighted.exposed_common_support_fraction.ge(.01)&weighted.control_common_support_fraction.ge(.01)).all());replication=bool(eicu_confirm and mimic['adjusted_rr']>1 and meta['i2_percent']<=60)
def crude_rd(dataset,window):
 s=classified.loc[classified.dataset.eq(dataset)].copy();return float(s.loc[s.exposed.eq(1),window].mean()-s.loc[s.exposed.eq(0),window].mean())
eicu_concurrent=unadjusted_binary_risk(classified.loc[classified.dataset.eq('eicu')],'blanking_support');concurrent_excess=max(0,crude_rd('eicu','blanking_support'));delayed_excess=max(0,eicu_un['risk_difference']);excess_fraction=delayed_excess/(delayed_excess+concurrent_excess) if delayed_excess+concurrent_excess>0 else 0;lead=bool(eicu_confirm and eicu['adjusted_rd']>0 and excess_fraction>=.5)
flags={'primary_delayed_support_positive':eicu_confirm,'eicu_confirmatory_positive':eicu_confirm,'mimic_direction_consistent':bool(mimic['adjusted_rr']>1),'mimic_significant_positive':mimic_sig,'effect_survives_sbp_map_adjustment':survives,'dynamic_beats_static':dynamic,'measured_lactate_direction_consistent':measured,'observation_process_acceptable':obs_ok,'meaningful_temporal_lead':lead,'cross_database_replication':replication,'eicu_concurrent_positive':bool(eicu_concurrent['risk_ratio']>1 and eicu_concurrent['rr_ci_low']>1),'eicu_unadjusted_delayed_positive':bool(eicu_un['risk_ratio']>1 and eicu_un['rr_ci_low']>1),'overlap_weighting_acceptable':weight_ok}
flags['high_impact_claim_ready']=bool(eicu_confirm and mimic['adjusted_rr']>1 and eicu['adjusted_rr']>=1.5 and eicu['adjusted_rd_ci_low']>0 and survives and dynamic and weight_ok and lead and replication)
classification=delayed_support_classification(flags)
claims={'CLAIM_READY_REPLICATED_SIGNAL':'Among heart-failure ICU patients with preserved cuff SBP and direct MAP and no ongoing hemodynamic support, incident proportional pulse-pressure collapse identified later initiation of vasoactive/inotropic support beyond two hours; the association survived conventional-pressure adjustment and externally replicated in eICU.','DISCOVERY_ONLY_NO_EXTERNAL_REPLICATION':'MIMIC showed a delayed-support association, but the locked eICU analysis did not confirm it; the data do not support a cross-database masked-pulsatility claim.','CONCURRENT_DETERIORATION_ONLY':'Any excess support initiation was confined to the prespecified first-two-hour concurrent-deterioration interval; the data do not support an early-warning claim.','CONVENTIONAL_BP_EXPLAINS_SIGNAL':'The unadjusted delayed-support association did not survive adjustment for current SBP/MAP and their pre-anchor trajectories; the data do not support independent pulsatility information.','STATIC_PRPP_EXPLAINS_SIGNAL':'The adjusted association was not shown to be specific to an incident dynamic crossing beyond static anchor PrPP; the data do not support a unique transition phenomenon.','NO_SIGNAL':'The locked analysis did not show a claim-ready delayed-support signal that replicated in eICU.'}
decision={**flags,'classification':classification,'supported_claim':claims[classification],'eicu_delayed_excess_fraction':excess_fraction,'protocol_freeze_utc':protocol['freeze_utc'],'first_outcome_read_utc':FIRST_OUTCOME_READ_UTC,'mimic_execution_utc':outcome_receipt['mimic_execution_utc'],'eicu_execution_utc':outcome_receipt['eicu_execution_utc'],'protocol_hashes':expected,'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'git_commit':git_commit,'execution_environment':'google_colab' if IN_COLAB else 'local','secondary_cannot_rescue_primary':True};ds_json('decision_gate_summary.json',decision)
trajectory=[]
bp=con.execute("""SELECT a.dataset,a.candidate_id,a.exposed,h.anchor_minute-a.anchor_minute relative_minute,h.prpp,h.sbp,h."map" FROM delayed_anchors a JOIN repair.source_repair_hourly_bp h ON a.dataset=h.dataset AND a.stay_id=h.stay_id AND a.bp_source=h.bp_source WHERE h.anchor_minute BETWEEN a.anchor_minute-240 AND a.anchor_minute""").df();hr=con.execute("""SELECT a.dataset,a.candidate_id,a.exposed,v.event_minute-a.anchor_minute relative_minute,v."value" hr FROM delayed_anchors a JOIN preanchor_vitals v USING(candidate_id) WHERE v.concept='hr' AND v.event_minute BETWEEN a.anchor_minute-240 AND a.anchor_minute""").df()
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for ax,metric,label,data in [(axes[0,0],'prpp','PrPP',bp),(axes[0,1],'sbp','SBP',bp),(axes[1,0],'map','MAP',bp),(axes[1,1],'hr','HR',hr)]:
 data=data.copy();data['relative_hour']=np.floor(data.relative_minute/60).astype(int);summary_rows=[];rng=np.random.default_rng(20260907)
 for (relative_hour,exposed),group in data.groupby(['relative_hour','exposed']):
  values=group[metric].dropna().to_numpy(float);boot=np.array([np.median(rng.choice(values,size=len(values),replace=True)) for _ in range(200)]) if len(values) else np.array([np.nan]);summary_rows.append({'relative_hour':relative_hour,'exposed':exposed,'median':np.median(values) if len(values) else np.nan,'ci_low':np.nanquantile(boot,.025),'ci_high':np.nanquantile(boot,.975),'count':len(values)})
 summary=pd.DataFrame(summary_rows)
 for exposed,color,name in [(0,'#4C78A8','Control'),(1,'#E45756','Crossing')]:
  s=summary.loc[summary.exposed.eq(exposed)].sort_values('relative_hour');ax.plot(s.relative_hour,s['median'],marker='o',color=color,label=name);ax.fill_between(s.relative_hour,s.ci_low,s.ci_high,color=color,alpha=.15)
 ax.axvline(0,color='black',linewidth=.8);ax.set_title(label);ax.grid(alpha=.2)
axes[1,0].set_xlabel('Hours before anchor');axes[1,1].set_xlabel('Hours before anchor');axes[0,0].legend();fig.suptitle('Unsmoothed pre-anchor physiology');fig.tight_layout();fig.savefig(OUT/'preanchor_trajectory.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for dataset,color in [('mimic','#4C78A8'),('eicu','#E45756')]:
 for exposed,style in [(0,'--'),(1,'-')]:
  s=classified.loc[classified.dataset.eq(dataset)&classified.exposed.eq(exposed)];times=np.arange(0,721,15);risk=[float((s.lead_minutes.gt(0)&s.lead_minutes.le(t)).mean()) for t in times];ax.step(times/60,risk,where='post',label=f'{dataset} '+('crossing' if exposed else 'control'),color=color,linestyle=style)
ax.axvspan(0,2,color='grey',alpha=.2,label='Concurrent / blanking');ax.axvline(2,color='black');ax.set(xlabel='Hours after anchor',ylabel='Cumulative support incidence',xlim=(0,12));ax.legend(ncol=2);fig.tight_layout();fig.savefig(OUT/'delayed_support_cumulative_incidence.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ys=[1,0];rr=[mimic['adjusted_rr'],eicu['adjusted_rr']];lo=[mimic['adjusted_rr_ci_low'],eicu['adjusted_rr_ci_low']];hi=[mimic['adjusted_rr_ci_high'],eicu['adjusted_rr_ci_high']];ax.errorbar(rr,ys,xerr=[np.array(rr)-np.array(lo),np.array(hi)-np.array(rr)],fmt='o');ax.axvline(1,color='black',linewidth=.8);ax.set_yticks(ys,['MIMIC development','eICU confirmation']);ax.set_xlabel('Adjusted risk ratio (95% CI)');fig.tight_layout();fig.savefig(OUT/'database_effect_forest.png',dpi=180);plt.close(fig)
artifact_names=['protocol.json','protocol_hash.txt','protocol_freeze_receipt.json','analysis_config.json','analysis_lock.json','outcome_read_receipt.json','cohort_flow.csv','anchor_balance.csv','support_drug_mapping.csv','primary_mimic.csv','primary_eicu.csv','primary_meta.csv','blanking_period_results.csv','lead_time_distribution.csv','lead_time_bins.csv','sbp_map_falsification.csv','static_vs_dynamic.csv','measured_lactate_analysis.csv','lactate_observation_audit.csv','overlap_weighting_results.csv','hospital_heterogeneity.csv','secondary_results.csv','decision_gate_summary.json','preanchor_trajectory.png','delayed_support_cumulative_incidence.png','database_effect_forest.png'];ds_json('artifact_hashes.json',{name:ds_hash(OUT/name) for name in artifact_names})
end_utc=datetime.now(timezone.utc).isoformat();ds_json('run_status.json',{'status':'completed','stage':'locked_outcome_analysis_complete','classification':classification,'protocol_freeze_utc':protocol['freeze_utc'],'first_outcome_read_utc':FIRST_OUTCOME_READ_UTC,'mimic_execution_utc':outcome_receipt['mimic_execution_utc'],'eicu_execution_utc':outcome_receipt['eicu_execution_utc'],'analysis_config_sha256':analysis_config_sha,'dirty_code_sha256':dirty_code_sha,'outcomes_inspected':True,'secondary_outcomes_inspected':True,'execution_environment':'google_colab' if IN_COLAB else 'local','start_utc':run_start,'end_utc':end_utc})
ds_log(f'locked outcome analysis complete: {classification}');print(json.dumps(decision,indent=2),flush=True)
'''

__all__ = [
    "BASELINE",
    "EICU_OUTCOME",
    "LOCK_ANALYSIS",
    "MIMIC_OUTCOME",
    "PRIMARY",
    "REPORT",
    "ROBUSTNESS",
    "SECONDARY",
    "SETUP",
]
