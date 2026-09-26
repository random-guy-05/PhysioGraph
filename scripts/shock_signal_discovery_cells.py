"""Notebook cells for the frozen MIMIC-only objective-shock signal screen."""

SETUP = r'''
import hashlib,importlib.util,importlib.metadata,json,math,subprocess,sys,shutil
from datetime import datetime,timezone
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3','scipy':'scipy==1.15.3','statsmodels':'statsmodels==0.14.5','matplotlib':'matplotlib==3.10.6'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None or importlib.metadata.version(module)!=package.split('==')[1]]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,matplotlib.pyplot as plt,numpy as np,pandas as pd,statsmodels.api as sm
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:
 from google.colab import drive
 drive.mount('/content/drive');DRIVE=Path('/content/drive/MyDrive')
else:DRIVE=Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905';MIMIC=DRIVE/'Data/MIMIC/Full';OUT=PROJECT/'research/shock_signal_discovery';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.shock_signal_discovery import BP_SIGNALS,CANDIDATE_DYNAMICS,CANDIDATE_SIGNALS,assemble_candidate_frame,benjamini_hochberg,bootstrap_candidate,candidate_adjustment_set,classify_discovery,final_winner_gate,fit_candidate_model,pair_domain_events,rolling_oliguria_events,statistical_winner_gate,sustained_hypotension_events,trajectory_features
DOC=PROJECT/'docs/OBJECTIVE_SHOCK_SINGLE_SIGNAL_DISCOVERY_PROTOCOL.md';PROTOCOL=OUT/'protocol.json';RECEIPT=OUT/'protocol_freeze_receipt.json'
expected_doc='164d6091b44657d4674857a83cf063388546dc3f4b6b95b822a98f1f74f2c03b';expected_protocol='ac46b95fc53530ee2c6a9bd7034cdd1761d2b7670f8deb9073f161bd40e0dabc'
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def q(value):return "'"+str(value).replace("'","''")+"'"
def write_json(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
def log(message):print(f'[{datetime.now(timezone.utc).isoformat()}] {message}',flush=True)
assert digest(DOC)==expected_doc and digest(PROTOCOL)==expected_protocol
protocol=json.loads(PROTOCOL.read_text());receipt=json.loads(RECEIPT.read_text())
assert receipt['mimic_candidate_outcome_associations_inspected'] is False
assert receipt['eicu_candidate_outcome_associations_inspected'] is False
prior_receipt=OUT/'association_read_receipt.json'
if prior_receipt.exists():
 prior=json.loads(prior_receipt.read_text());archive=OUT/'attempt_history'/prior['first_mimic_candidate_outcome_read_utc'].replace(':','-');archive.mkdir(parents=True,exist_ok=True)
 for name in ['analysis_lock.json','analysis_config.json','association_read_receipt.json','run_status.json','endpoint_support.json']:
  if (OUT/name).exists() and not (archive/name).exists():shutil.copy2(OUT/name,archive/name)
historical_first_association_attempt='2026-09-07T19:32:30.447739+00:00'
try:con.close()
except (NameError,AttributeError):pass
DB=PRIVATE/'objective_shock_single_signal_discovery.duckdb';con=duckdb.connect(str(DB));con.execute('SET threads=4');con.execute("SET memory_limit='4GB'");con.execute('SET enable_progress_bar=false')
phase=PRIVATE/'masked_pulsatility_phase_a.duckdb';con.execute(f'ATTACH {q(phase)} AS phase (READ_ONLY)')
run_start=datetime.now(timezone.utc).isoformat();write_json('run_status.json',{'status':'running','stage':'protocol_verified','environment':'google_colab' if IN_COLAB else 'local','protocol_freeze_utc':protocol['freeze_utc'],'mimic_associations_inspected':False,'eicu_associations_inspected':False,'start_utc':run_start})
log('protocol verified; eICU remains unopened')
'''

EXTRACT = r'''
# This source is mandatory because urine output is a frozen candidate and
# hypoperfusion component; fail before scanning other multi-gigabyte tables.
required_sources=[MIMIC/'chartevents.csv',MIMIC/'outputevents.csv',MIMIC/'inputevents.csv',MIMIC/'procedureevents.csv',MIMIC/'labevents.csv',MIMIC/'d_items.csv',MIMIC/'patients.csv']
missing_sources=[str(path) for path in required_sources if not path.is_file()]
assert not missing_sources,'Frozen discovery sources missing: '+', '.join(missing_sources)
con.execute('CREATE OR REPLACE TABLE cohort AS SELECT * FROM phase.mimic_cohort')
if not con.execute("SELECT count(*) FROM information_schema.columns WHERE table_name='vital_rows' AND column_name='available_minute'").fetchone()[0]:
 con.execute(f"""CREATE OR REPLACE TABLE vital_rows AS
SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
 greatest(date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0,date_diff('second',c.admit_time,try_cast(v.storetime AS TIMESTAMP))/60.0) available_minute,
 CASE try_cast(v.itemid AS BIGINT) WHEN 220045 THEN 'hr' WHEN 220179 THEN 'sbp' WHEN 220180 THEN 'dbp' WHEN 220181 THEN 'map'
 WHEN 220210 THEN 'resp_rate' WHEN 224689 THEN 'resp_rate' WHEN 224690 THEN 'resp_rate' WHEN 220277 THEN 'spo2'
 WHEN 223761 THEN 'temperature_c' WHEN 223762 THEN 'temperature_c' END signal,
 CASE WHEN try_cast(v.itemid AS BIGINT)=223761 THEN (try_cast(v.valuenum AS DOUBLE)-32)*5/9 ELSE try_cast(v.valuenum AS DOUBLE) END AS "value"
FROM read_csv({q(MIMIC/'chartevents.csv')},header=true,all_varchar=true,parallel=true) v JOIN cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
WHERE try_cast(v.itemid AS BIGINT) IN (220045,220179,220180,220181,220210,224689,224690,220277,223761,223762)
 AND date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 960
 AND CASE WHEN try_cast(v.itemid AS BIGINT)=220045 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 20 AND 250
 WHEN try_cast(v.itemid AS BIGINT)=220179 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 40 AND 260
 WHEN try_cast(v.itemid AS BIGINT)=220180 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 20 AND 180
 WHEN try_cast(v.itemid AS BIGINT)=220181 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 30 AND 200
 WHEN try_cast(v.itemid AS BIGINT) IN (220210,224689,224690) THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 4 AND 80
 WHEN try_cast(v.itemid AS BIGINT)=220277 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 50 AND 100
 WHEN try_cast(v.itemid AS BIGINT)=223761 THEN (try_cast(v.valuenum AS DOUBLE)-32)*5/9 BETWEEN 30 AND 43
 WHEN try_cast(v.itemid AS BIGINT)=223762 THEN try_cast(v.valuenum AS DOUBLE) BETWEEN 30 AND 43 ELSE false END""")
con.execute("""CREATE OR REPLACE TABLE vital_hourly AS SELECT stay_id,signal,floor(event_minute/60) AS "hour",median("value") AS "value",max(available_minute) available_minute,count(*) measurements FROM vital_rows GROUP BY 1,2,3""")
bp=con.execute("""SELECT stay_id,"hour",max("value") FILTER(WHERE signal='sbp') sbp,max("value") FILTER(WHERE signal='dbp') dbp,max("value") FILTER(WHERE signal='map') "map",max(available_minute) available_minute FROM vital_hourly WHERE signal IN ('sbp','dbp','map') GROUP BY 1,2""").df();bp['pulse_pressure']=bp.sbp-bp.dbp
vital_candidates=con.execute('SELECT stay_id,signal,floor(event_minute/60) AS "hour",median("value") AS "value" FROM vital_rows WHERE event_minute BETWEEN 0 AND 240 AND available_minute<=240 GROUP BY 1,2,3').df()
candidate_bp=vital_candidates.loc[vital_candidates.signal.isin(['sbp','dbp'])].pivot(index=['stay_id','hour'],columns='signal',values='value').reset_index();candidate_bp['pulse_pressure']=candidate_bp.sbp-candidate_bp.dbp;pp=candidate_bp.loc[candidate_bp.pulse_pressure.between(5,150),['stay_id','hour','pulse_pressure']].rename(columns={'pulse_pressure':'value'});pp['signal']='pulse_pressure';vital_candidates=pd.concat([vital_candidates,pp],ignore_index=True)
dictionary=con.execute(f"SELECT try_cast(itemid AS BIGINT) itemid,label FROM read_csv({q(MIMIC/'d_items.csv')},header=true,all_varchar=true,parallel=true)").df();label=dictionary.label.fillna('').astype(str);urine_map=dictionary.loc[label.str.contains(r'urine|foley|voided|nephrostomy|urostomy',case=False,regex=True)&~label.str.contains(r'culture|specimen|appearance|color|irrigant|irrigation|stool|subtotal|total',case=False,regex=True),['itemid','label']].drop_duplicates();con.register('_urine_map',urine_map);con.execute('CREATE OR REPLACE TABLE urine_map AS SELECT * FROM _urine_map');con.unregister('_urine_map')
con.execute(f"""CREATE OR REPLACE TABLE urine_hourly AS SELECT c.stay_id,floor(date_diff('second',c.admit_time,try_cast(o.charttime AS TIMESTAMP))/3600) AS "hour",sum(try_cast(o.value AS DOUBLE)) AS "value",max(greatest(date_diff('second',c.admit_time,try_cast(o.charttime AS TIMESTAMP))/60.0,date_diff('second',c.admit_time,try_cast(o.storetime AS TIMESTAMP))/60.0)) available_minute,count(*) measurements FROM read_csv({q(MIMIC/'outputevents.csv')},header=true,all_varchar=true,parallel=true) o JOIN urine_map u ON try_cast(o.itemid AS BIGINT)=u.itemid JOIN cohort c ON try_cast(o.stay_id AS BIGINT)=c.stay_id WHERE date_diff('second',c.admit_time,try_cast(o.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 960 AND try_cast(o.value AS DOUBLE) BETWEEN 0 AND 5000 GROUP BY 1,2""")
urine_candidates=con.execute(f"""SELECT c.stay_id,'urine_output_ml_h' signal,floor(date_diff('second',c.admit_time,try_cast(o.charttime AS TIMESTAMP))/3600) AS "hour",sum(try_cast(o.value AS DOUBLE)) AS "value" FROM read_csv({q(MIMIC/'outputevents.csv')},header=true,all_varchar=true,parallel=true) o JOIN urine_map u ON try_cast(o.itemid AS BIGINT)=u.itemid JOIN cohort c ON try_cast(o.stay_id AS BIGINT)=c.stay_id WHERE date_diff('second',c.admit_time,try_cast(o.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 240 AND greatest(try_cast(o.charttime AS TIMESTAMP),try_cast(o.storetime AS TIMESTAMP))<=c.admit_time+INTERVAL '4 hours' AND try_cast(o.value AS DOUBLE) BETWEEN 0 AND 5000 GROUP BY 1,2,3""").df();candidate_hourly=pd.concat([vital_candidates,urine_candidates],ignore_index=True);features_long=trajectory_features(candidate_hourly)
feature_rows=[]
for _,r in features_long.iterrows():
 for dynamic in CANDIDATE_DYNAMICS:feature_rows.append({'stay_id':r.stay_id,'signal':r.signal,'dynamic':dynamic,'feature':f'{r.signal}_{dynamic}','value':r[dynamic],'bins':r.bins,'span_hours':r.span_hours})
feature_long=pd.DataFrame(feature_rows);con.register('_feature_long',feature_long);con.execute('CREATE OR REPLACE TABLE feature_long AS SELECT * FROM _feature_long');con.unregister('_feature_long')
con.execute(f"""CREATE OR REPLACE TABLE support_events AS SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0 event_minute,'continuous_support' component FROM read_csv({q(MIMIC/'inputevents.csv')},header=true,all_varchar=true,parallel=true) i JOIN cohort c ON try_cast(i.stay_id AS BIGINT)=c.stay_id WHERE try_cast(i.itemid AS BIGINT) IN (221906,221289,221662,221653,221749,222315,221986) AND coalesce(i.statusdescription,'')!='Rewritten' AND try_cast(i.rate AS DOUBLE)>0 AND try_cast(i.endtime AS TIMESTAMP)>try_cast(i.starttime AS TIMESTAMP) AND date_diff('second',c.admit_time,try_cast(i.starttime AS TIMESTAMP))/60.0 <=960 AND try_cast(i.endtime AS TIMESTAMP)>c.admit_time""")
con.execute(f"""CREATE OR REPLACE TABLE mcs_events AS SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0 event_minute,'mcs' component FROM read_csv({q(MIMIC/'procedureevents.csv')},header=true,all_varchar=true,parallel=true) p JOIN cohort c ON try_cast(p.stay_id AS BIGINT)=c.stay_id WHERE try_cast(p.itemid AS BIGINT) IN (224272,228169,229529,229530) AND date_diff('second',c.admit_time,try_cast(p.starttime AS TIMESTAMP))/60.0 <=960 AND try_cast(p.endtime AS TIMESTAMP)>c.admit_time""")
con.execute(f"""CREATE OR REPLACE TABLE lab_rows AS SELECT c.stay_id,CASE WHEN try_cast(l.itemid AS BIGINT) IN (50813,52442,53154) THEN 'lactate' WHEN try_cast(l.itemid AS BIGINT) IN (50912,52024,52546) THEN 'creatinine' WHEN try_cast(l.itemid AS BIGINT) IN (50861,53084) THEN 'alt' WHEN try_cast(l.itemid AS BIGINT)=50820 THEN 'ph' END marker,greatest(date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0,date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0) event_minute,try_cast(l.valuenum AS DOUBLE) AS "value" FROM read_csv({q(MIMIC/'labevents.csv')},header=true,all_varchar=true,parallel=true) l JOIN cohort c ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id WHERE try_cast(l.itemid AS BIGINT) IN (50813,52442,53154,50912,52024,52546,50861,53084,50820) AND greatest(date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0,date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0) BETWEEN -1440 AND 960 AND CASE WHEN try_cast(l.itemid AS BIGINT) IN (50813,52442,53154) THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 0.1 AND 30 WHEN try_cast(l.itemid AS BIGINT) IN (50912,52024,52546) THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 0.1 AND 30 WHEN try_cast(l.itemid AS BIGINT) IN (50861,53084) THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 1 AND 10000 WHEN try_cast(l.itemid AS BIGINT)=50820 THEN try_cast(l.valuenum AS DOUBLE) BETWEEN 6.5 AND 8.0 ELSE false END""")
write_json('extraction_counts.json',{'cohort':con.execute('SELECT count(*) FROM cohort').fetchone()[0],'vital_rows':con.execute('SELECT count(*) FROM vital_rows').fetchone()[0],'urine_hours':con.execute('SELECT count(*) FROM urine_hourly').fetchone()[0],'support_rows':con.execute('SELECT count(*) FROM support_events').fetchone()[0],'mcs_rows':con.execute('SELECT count(*) FROM mcs_events').fetchone()[0],'lab_rows':con.execute('SELECT count(*) FROM lab_rows').fetchone()[0]})
log('MIMIC source extraction complete; no candidate-outcome associations computed')
'''

ENDPOINT = r'''
bp_events=sustained_hypotension_events(bp.dropna(subset=['sbp','map'],how='all'))
support=con.execute('SELECT * FROM support_events UNION ALL SELECT * FROM mcs_events').df();pressure=pd.concat([bp_events,support],ignore_index=True)
baseline=con.execute("""SELECT c.*,arg_max(l."value",l.event_minute) FILTER(WHERE l.marker='lactate' AND l.event_minute<=240) baseline_lactate,arg_max(l."value",l.event_minute) FILTER(WHERE l.marker='creatinine' AND l.event_minute<=240) baseline_creatinine,arg_max(l."value",l.event_minute) FILTER(WHERE l.marker='alt' AND l.event_minute<=240) baseline_alt,arg_max(l."value",l.event_minute) FILTER(WHERE l.marker='ph' AND l.event_minute<=240) baseline_ph FROM cohort c LEFT JOIN lab_rows l USING(stay_id) GROUP BY ALL""").df()
labs=con.execute('SELECT * FROM lab_rows WHERE event_minute>240 AND event_minute<=960 ORDER BY stay_id,event_minute').df();labs=labs.merge(baseline[['stay_id','baseline_lactate','baseline_creatinine','baseline_alt','baseline_ph']],on='stay_id',how='left');perf=[]
for marker,group in labs.groupby('marker'):
 if marker=='lactate':mask=group.value.ge(4)|(group.baseline_lactate.notna()&group.value.ge(2)&group.value.sub(group.baseline_lactate).ge(.5))
 elif marker=='creatinine':mask=group.baseline_creatinine.notna()&(group.value.sub(group.baseline_creatinine).ge(.3)|group.value.div(group.baseline_creatinine).ge(1.5))
 elif marker=='alt':mask=group.baseline_alt.notna()&((group.baseline_alt.le(200)&group.value.gt(200))|group.value.div(group.baseline_alt).ge(3))
 else:mask=group.baseline_ph.notna()&group.baseline_ph.ge(7.2)&group.value.lt(7.2)
 current=group.loc[mask,['stay_id','event_minute']].copy();current['component']=marker;perf.append(current)
oliguria=rolling_oliguria_events(con.execute('SELECT stay_id,"hour","value",available_minute FROM urine_hourly').df());perfusion=pd.concat([*perf,oliguria],ignore_index=True)
pre_pressure=set(pressure.loc[pressure.event_minute.le(360),'stay_id'])
pre_labs=con.execute('SELECT DISTINCT stay_id FROM lab_rows WHERE event_minute<=360 AND ((marker=\'lactate\' AND "value">=2) OR (marker=\'alt\' AND "value">200) OR (marker=\'ph\' AND "value"<7.2))').df()
pre_urine=rolling_oliguria_events(con.execute('SELECT stay_id,"hour","value" FROM urine_hourly WHERE "hour"<6').df(),start_hour=0,end_hour=6)
pre_perfusion=set(pre_labs.stay_id)|set(perfusion.loc[perfusion.event_minute.le(360),'stay_id'])|set(pre_urine.stay_id)
at_risk=baseline.loc[baseline.followup_end_offset_minutes.gt(360)&(~baseline.death_offset_minutes.between(0,360).fillna(False))&~baseline.stay_id.isin(pre_pressure|pre_perfusion)].copy()
full=pair_domain_events(pressure,perfusion,lower_minute=360,upper_minute=960);support_pressure=pressure.loc[pressure.component.isin(['continuous_support','mcs'])];no_hypotension=pair_domain_events(support_pressure,perfusion,lower_minute=360,upper_minute=960);nonurine=perfusion.loc[~perfusion.component.eq('oliguria')];no_urine=pair_domain_events(pressure,nonurine,lower_minute=360,upper_minute=960)
outcomes=at_risk[['stay_id']].copy()
for name,events in [('objective_shock',full),('objective_shock_no_hypotension',no_hypotension),('objective_shock_no_urine',no_urine)]:
 timing=events[['stay_id','event_minute']].rename(columns={'event_minute':name+'_minute'});outcomes=outcomes.merge(timing,on='stay_id',how='left');outcomes[name]=outcomes[name+'_minute'].notna().astype(int);outcomes[name+'_lead4']=outcomes[name+'_minute'].gt(480).astype(int)
levels=feature_long.loc[feature_long.dynamic.eq('level'),['stay_id','feature','value']].pivot(index='stay_id',columns='feature',values='value').reset_index();dens=feature_long.groupby(['stay_id','signal']).bins.max().unstack(fill_value=0).add_suffix('_bins').reset_index();model_base=at_risk.merge(outcomes,on='stay_id').merge(levels,on='stay_id',how='left').merge(dens,on='stay_id',how='left');model_base['age']=pd.to_numeric(model_base.get('age',np.nan),errors='coerce');model_base['male_sex']=pd.to_numeric(model_base.get('male_sex',np.nan),errors='coerce')
if model_base.age.isna().all() or model_base.male_sex.isna().all():
 demo=con.execute(f"""SELECT c.stay_id,try_cast(p.anchor_age AS DOUBLE)+(year(c.admit_time)-try_cast(p.anchor_year AS DOUBLE)) age,CASE upper(trim(p.gender)) WHEN 'M' THEN 1 WHEN 'F' THEN 0 END male_sex FROM cohort c JOIN read_csv({q(MIMIC/'patients.csv')},header=true,all_varchar=true,parallel=true) p ON try_cast(p.subject_id AS BIGINT)=c.person_id""").df();model_base=model_base.drop(columns=['age','male_sex'],errors='ignore').merge(demo,on='stay_id',how='left')
model_base['lactate_observed']=model_base.baseline_lactate.notna().astype(int);model_base['bp_measurement_density']=model_base.get('sbp_bins',0)+model_base.get('map_bins',0);model_base['urine_measurement_density']=model_base.get('urine_output_ml_h_bins',0)
con.register('_model_base',model_base);con.execute('CREATE OR REPLACE TABLE model_base AS SELECT * FROM _model_base');con.unregister('_model_base')
write_json('endpoint_support.json',{'at_risk_n':len(at_risk),'objective_shock_events':int(model_base.objective_shock.sum()),'lead4_events':int(model_base.objective_shock_lead4.sum()),'no_hypotension_events':int(model_base.objective_shock_no_hypotension.sum()),'no_urine_events':int(model_base.objective_shock_no_urine.sum()),'pressure_component_counts':pressure.component.value_counts().to_dict(),'perfusion_component_counts':perfusion.component.value_counts().to_dict()})
log('objective endpoint constructed; candidate effects remain unread')
'''

LOCK = r'''
code_files=['scripts/build_biological_notebook.py','scripts/shock_signal_discovery_cells.py','src/physiograph/analysis/shock_signal_discovery.py','tests/unit/test_shock_signal_discovery.py'];code_hashes={name:digest(PROJECT/name) for name in code_files};code_sha=hashlib.sha256(json.dumps(code_hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest();git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip();analysis_config={'protocol_sha256':expected_protocol,'documentation_sha256':expected_doc,'code_hashes':code_hashes,'code_aggregate_sha256':code_sha,'git_commit':git_commit,'historical_first_association_attempt_utc':historical_first_association_attempt,'implementation_corrections':{p.name:digest(p) for p in sorted(OUT.glob('implementation_correction_*.json'))},'candidate_signals':list(CANDIDATE_SIGNALS),'candidate_dynamics':list(CANDIDATE_DYNAMICS),'candidate_tests':36,'primary_outcome':'objective_shock','lead_outcome':'objective_shock_lead4','bootstrap_replicates':200,'eicu_accessed':False};write_json('analysis_config.json',analysis_config);config_sha=digest(OUT/'analysis_config.json');lock_utc=datetime.now(timezone.utc).isoformat();write_json('analysis_lock.json',{'lock_utc':lock_utc,'protocol_freeze_utc':protocol['freeze_utc'],'protocol_sha256':expected_protocol,'analysis_config_sha256':config_sha,'code_aggregate_sha256':code_sha,'first_mimic_candidate_outcome_read_utc':None,'mimic_candidate_outcome_associations_inspected':False,'eicu_candidate_outcome_associations_inspected':False});log(f'36-test MIMIC screen locked before association read: {config_sha}')
'''

SCREEN = r'''
assert digest(PROTOCOL)==expected_protocol and digest(OUT/'analysis_config.json')==config_sha and {name:digest(PROJECT/name) for name in code_files}==code_hashes
first_association_read=datetime.now(timezone.utc).isoformat();receipt_live={**receipt,'first_mimic_candidate_outcome_read_utc':first_association_read,'mimic_candidate_outcome_associations_inspected':True,'eicu_candidate_outcome_associations_inspected':False,'historical_first_association_attempt_utc':historical_first_association_attempt,'analysis_lock_utc':lock_utc,'analysis_config_sha256':config_sha};write_json('association_read_receipt.json',receipt_live);log('FIRST MIMIC CANDIDATE-OUTCOME ASSOCIATION READ; eICU remains unopened')
base_covariates=['age','male_sex','hr_level','sbp_level','map_level','resp_rate_level','spo2_level','temperature_c_level','baseline_creatinine','baseline_lactate','lactate_observed','bp_measurement_density','urine_measurement_density'];screen=[]
for signal in CANDIDATE_SIGNALS:
 for dynamic in CANDIDATE_DYNAMICS:
  feature=f'{signal}_{dynamic}';frame=assemble_candidate_frame(model_base,feature_long,feature);adjust=candidate_adjustment_set(signal,base_covariates) if dynamic=='level' else base_covariates
  try:
   primary,_=fit_candidate_model(frame,feature=feature,outcome='objective_shock',adjustment_covariates=adjust);lead,_=fit_candidate_model(frame,feature=feature,outcome='objective_shock_lead4',adjustment_covariates=adjust);screen.append({'signal':signal,'dynamic':dynamic,'feature':feature,'cohort_events':int(frame.objective_shock.sum()),**primary,'lead_rr_per_sd':lead['rr_per_sd'],'lead_ci_low':lead['ci_low'],'lead_ci_high':lead['ci_high'],'lead_p_value':lead['p_value'],'lead_log_rr':lead['log_rr'],'lead_effect_magnitude':lead['effect_magnitude']})
  except (ValueError,np.linalg.LinAlgError):screen.append({'signal':signal,'dynamic':dynamic,'feature':feature,'cohort_events':int(frame.objective_shock.sum()),'n':int(frame[feature].notna().sum()),'events':int(frame.loc[frame[feature].notna(),'objective_shock'].sum()),'coverage':float(frame[feature].notna().mean()),'p_value':np.nan,'lead_p_value':np.nan})
screen=pd.DataFrame(screen);screen['q_value']=benjamini_hochberg(screen.p_value);screen['lead_q_value']=benjamini_hochberg(screen.lead_p_value);screen['statistical_gate']=screen.apply(lambda r:statistical_winner_gate(r) if pd.notna(r.get('rr_per_sd')) else False,axis=1);screen.to_csv(OUT/'mimic_candidate_screen.csv',index=False);log(f'MIMIC screen complete: {int(screen.statistical_gate.sum())} candidates pass initial statistical gate')
'''

STABILITY = r'''
stability=[]
for _,candidate in screen.loc[screen.statistical_gate].sort_values(['q_value','effect_magnitude','coverage'],ascending=[True,False,False]).iterrows():
 signal=candidate.signal;feature=candidate.feature;frame=assemble_candidate_frame(model_base,feature_long,feature);adjust=candidate_adjustment_set(signal,base_covariates) if candidate.dynamic=='level' else base_covariates;boot=bootstrap_candidate(frame,feature=feature,outcome='objective_shock',adjustment_covariates=adjust,replicates=200)
 leave_outcome='objective_shock_no_hypotension' if signal in BP_SIGNALS else 'objective_shock_no_urine' if signal=='urine_output_ml_h' else 'objective_shock';leave,_=fit_candidate_model(frame,feature=feature,outcome=leave_outcome,adjustment_covariates=adjust);stability.append({'feature':feature,**boot,'leave_domain_outcome':leave_outcome,'leave_domain_rr':leave['rr_per_sd'],'leave_domain_ci_low':leave['ci_low'],'leave_domain_ci_high':leave['ci_high'],'leave_domain_log_rr':leave['log_rr'],'leave_domain_effect_magnitude':leave['effect_magnitude']})
stability=pd.DataFrame(stability) if stability else pd.DataFrame(columns=['feature','bootstrap_success_fraction']);screen=screen.merge(stability,on='feature',how='left');screen['final_gate']=screen.apply(lambda r:final_winner_gate(r) if r.statistical_gate and pd.notna(r.get('bootstrap_success_fraction')) else False,axis=1);screen.to_csv(OUT/'mimic_candidate_screen.csv',index=False);eligible=screen.loc[screen.final_gate].sort_values(['q_value','effect_magnitude','coverage'],ascending=[True,False,False]);provisional=eligible.head(1);classification=classify_discovery(eligible_winners=len(eligible),novelty_passed=None);decision={'classification':classification,'statistical_gate_count':int(screen.statistical_gate.sum()),'final_gate_count':int(screen.final_gate.sum()),'provisional_winner':None if provisional.empty else provisional.iloc[0].feature,'eicu_accessed':False,'eicu_must_remain_unopened':classification!='ONE_SIGNAL_READY_FOR_EXTERNAL_FREEZE','first_mimic_candidate_outcome_read_utc':first_association_read};write_json('decision_gate_summary.json',decision);log(f'discovery gate complete: {classification}')
'''

REPORT = r'''
write_json('mimic_candidate_screen.json',json.loads(screen.to_json(orient='records')))
ranked=screen.sort_values(['q_value','effect_magnitude','coverage'],ascending=[True,False,False],na_position='last');ranked.head(10).to_csv(OUT/'top_candidates.csv',index=False)
coverage=screen.groupby('signal').coverage.max().reset_index();coverage.to_csv(OUT/'measurement_coverage.csv',index=False)
fig,ax=plt.subplots(figsize=(9,8));plot=ranked.dropna(subset=['rr_per_sd']).copy();plot['label']=plot.signal+' · '+plot.dynamic;plot=plot.sort_values('rr_per_sd');y=np.arange(len(plot));ax.errorbar(plot.rr_per_sd,y,xerr=[plot.rr_per_sd-plot.ci_low,plot.ci_high-plot.rr_per_sd],fmt='o',color='#2457C5',ecolor='#92A9DA',capsize=2);ax.axvline(1,color='black',lw=1);ax.set_yticks(y,plot.label);ax.set_xscale('log');ax.set_xlabel('Adjusted risk ratio per 1 SD (log scale)');ax.set_title('Frozen MIMIC objective-shock candidate screen');fig.tight_layout();fig.savefig(OUT/'candidate_forest.png',dpi=180);plt.close(fig)
write_json('run_status.json',{'status':'completed','stage':'mimic_single_signal_discovery_complete','environment':'google_colab' if IN_COLAB else 'local','classification':classification,'protocol_freeze_utc':protocol['freeze_utc'],'first_mimic_candidate_outcome_read_utc':first_association_read,'eicu_candidate_outcome_associations_inspected':False,'end_utc':datetime.now(timezone.utc).isoformat()});artifacts=['protocol.json','protocol_freeze_receipt.json','analysis_config.json','analysis_lock.json','association_read_receipt.json','extraction_counts.json','endpoint_support.json','mimic_candidate_screen.json','mimic_candidate_screen.csv','top_candidates.csv','measurement_coverage.csv','decision_gate_summary.json','candidate_forest.png','run_status.json'];write_json('artifact_hashes.json',{name:digest(OUT/name) for name in artifacts});print(json.dumps(decision,indent=2));print(ranked[['feature','n','events','coverage','rr_per_sd','ci_low','ci_high','p_value','q_value','lead_rr_per_sd','lead_q_value','statistical_gate','final_gate']].head(10).to_string(index=False));con.close()
'''
