"""Phase-A-only, outcome-blind venous oxygen reserve notebook cells."""

SETUP = r"""
import hashlib,importlib.util,json,re,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3'}
missing=[package for module,package in requirements.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/venous_o2_reserve';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(PROJECT/'src'))
from physiograph.analysis.spo2_venous_o2_reserve import build_primary_spo2_anchors,classify_venous_measurement,normalize_venous_saturation,choose_venous_strategy,phase_a_decision_summary,VENOUS_TYPES,support_tier
def vo2_q(x):return "'"+str(x).replace("'","''")+"'"
def vo2_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def vo2_status(stage,state='running',**extra):
    row=dict(status=state,stage=stage,environment='google_colab' if IN_COLAB else 'local',updated_utc=datetime.now(timezone.utc).isoformat(),phase='A',clinical_outcomes_inspected=False,effect_estimates_computed=False,goal_complete=False);row.update(extra);vo2_json('run_status.json',row);print(stage,state,flush=True)
plan=PROJECT/'docs/SPO2_VENOUS_O2_FEASIBILITY_PLAN.md'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==VENOUS_O2_FEASIBILITY_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==VENOUS_O2_FEASIBILITY_SHA256
else:vo2_json('protocol_lock.json',dict(sha256=VENOUS_O2_FEASIBILITY_SHA256,frozen_utc=datetime.now(timezone.utc).isoformat(),before_new_raw_source_scan=True,phase_a_only=True,outcomes_prohibited=True))
try:con.close()
except (NameError,AttributeError):pass
db=PRIVATE/'venous_o2_reserve_phase_a.duckdb';con=duckdb.connect(str(db));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'");con.execute('SET enable_progress_bar=false')
def vo2_register(name,frame):
    frame=frame.copy()
    for column in frame:
        if str(frame[column].dtype) in {'str','string'}:frame[column]=frame[column].astype(object)
    con.register(name,frame);return frame
def vo2_table_exists(name):
    return bool(con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",[name]).fetchone()[0])
prior=PRIVATE/'spo2_context.duckdb';assert prior.is_file(),str(prior);con.execute(f'ATTACH {vo2_q(prior)} AS prior (READ_ONLY)')
RAW={'eicu':DRIVE/'Data/eICU/Full','mimic':DRIVE/'Data/MIMIC/Full'}
source_manifest={}
def vo2_source(dataset,name,required=True):
    p=RAW[dataset]/(name+'.csv')
    if not p.is_file():
        if required:raise FileNotFoundError(p)
        source_manifest[dataset+'/'+name]=dict(path=str(p),present=False);return None
    s=p.stat();source_manifest[dataset+'/'+name]=dict(path=str(p),present=True,bytes=s.st_size,mtime_ns=s.st_mtime_ns);return p
def vo2_csv(dataset,name,required=True):
    p=vo2_source(dataset,name,required)
    return None if p is None else f'read_csv({vo2_q(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
vo2_status('reconstructing established episode and deterministic control anchors')
anchor_validation=[]
for d in ['eicu','mimic']:
    b=con.execute(f"SELECT {vo2_q(d)} dataset,stay_id,'spo2' concept,representative_minute offset_minutes,value value_numeric FROM prior.{d}_bins WHERE concept='spo2' ORDER BY stay_id,bin").df()
    anchors=build_primary_spo2_anchors(b)
    anchors=anchors.loc[anchors.strict_dynamics_eligible_flag.eq(1)].copy()
    features=con.execute(f'SELECT stay_id,spo2_jump_any,spo2_bins,spo2_transitions FROM prior.{d}_features WHERE spo2_bins>=3 AND spo2_transitions>=2 ORDER BY stay_id').df()
    assert set(anchors.stay_id)==set(features.stay_id)
    check=anchors.merge(features,on='stay_id',validate='one_to_one')
    assert (check.episode_exposed.astype(int)==check.spo2_jump_any.astype(int)).all()
    if d=='eicu':cohort=con.execute('SELECT stay_id,cast(person_id AS VARCHAR) patient_id,cast(hospital_id AS VARCHAR) hospital_id FROM prior.eicu_cohort').df()
    else:cohort=con.execute('SELECT stay_id,cast(person_id AS VARCHAR) patient_id,cast(hadm_id AS BIGINT) hadm_id,admit_time,NULL::VARCHAR hospital_id FROM prior.mimic_cohort').df()
    anchors=anchors.merge(cohort,on='stay_id',validate='one_to_one')
    anchors=vo2_register('anchor_frame',anchors);con.execute(f'CREATE OR REPLACE TABLE {d}_anchors AS SELECT * FROM anchor_frame');con.unregister('anchor_frame')
    anchor_validation.append(dict(dataset=d,eligible_encounters=len(anchors),exposed=int(anchors.episode_exposed.sum()),controls=int(anchors.episode_exposed.eq(0).sum()),first_episode_matches_existing_exposure=True,deterministic_control_strategy=str(anchors.loc[anchors.episode_exposed.eq(0),'control_anchor_strategy'].drop_duplicates().iloc[0])))
vo2_json('anchor_validation.json',anchor_validation)
vo2_status('anchors verified; original-source mapping and scans pending')
"""

EXTRACT = r"""
def mimic_chart_variable(label,itemid):
    z=' '.join(str(label or '').casefold().split());vtype=classify_venous_measurement(z)
    if vtype:return 'venous_o2',vtype
    fixed={220277:'spo2',220045:'heart_rate',220050:'sbp',220179:'sbp',220052:'map',220181:'map',220074:'cvp'}
    if int(itemid) in fixed:return fixed[int(itemid)],None
    if 'alarm' in z or 'goal' in z:return None,None
    if ('arterial o2 saturation' in z or 'o2 saturation (arterial)' in z):return 'arterial_sao2',None
    if 'arterial o2 pressure' in z or 'po2 (arterial)' in z:return 'pao2',None
    if 'inspired o2 fraction' in z or re.search(r'\bfio2\b',z):return 'fio2',None
    if re.search(r'oxygen delivery|o2 delivery|oxygen device|ventilator mode|ventilation mode|airway type|endotracheal tube|ett placement',z):return 'respiratory_support',None
    if 'pulmonary artery pressure' in z:return 'pulmonary_artery_pressure',None
    if 'cardiac power output' in z:return 'cardiac_power_output',None
    if 'cardiac index' in z or re.match(r'^ci\s*\(',z):return 'cardiac_index',None
    if 'cardiac output' in z or re.match(r'^(?:co|cco)\s*\(',z):return 'cardiac_output',None
    return None,None
def mimic_lab_variable(label,fluid):
    z=' '.join(str(label or '').casefold().split());f=str(fluid or '').casefold()
    if f!='blood':return None
    if z in {'hemoglobin','hemoglobin, calculated'}:return 'hemoglobin'
    if z in {'hematocrit','hematocrit, calculated'}:return 'hematocrit'
    if z=='lactate':return 'lactate'
    if z in {'creatinine','creatinine, whole blood'}:return 'creatinine'
    if z=='po2':return 'po2_unspecified'
    if z=='oxygen saturation':return 'oxygen_saturation_unspecified'
    return None
items=pd.read_csv(vo2_source('mimic','d_items'),dtype=str)
chart_rows=[]
for r in items.itertuples(index=False):
    variable,vtype=mimic_chart_variable(r.label,r.itemid)
    if variable and str(r.linksto).casefold()=='chartevents':chart_rows.append(dict(itemid=int(r.itemid),variable=variable,venous_type=vtype,raw_label=r.label,dictionary_unit=r.unitname,category=r.category,documentation_mode='charted_numeric; interface_vs_manual_not_encoded'))
chart_map=pd.DataFrame(chart_rows).drop_duplicates('itemid');chart_map=vo2_register('mimic_chart_map',chart_map)
labitems=pd.read_csv(vo2_source('mimic','d_labitems'),dtype=str)
lab_rows=[]
for r in labitems.itertuples(index=False):
    variable=mimic_lab_variable(r.label,r.fluid)
    if variable:lab_rows.append(dict(itemid=int(r.itemid),variable=variable,raw_label=r.label,dictionary_unit='',category=r.category,documentation_mode='laboratory_record; arterial_vs_venous_not_encoded'))
lab_map=pd.DataFrame(lab_rows).drop_duplicates('itemid');lab_map=vo2_register('mimic_lab_map',lab_map)
vo2_status('scanning original MIMIC chartevents')
ce=vo2_csv('mimic','chartevents')
if not vo2_table_exists('mimic_chart_raw'):con.execute(f'''CREATE TABLE mimic_chart_raw AS SELECT a.stay_id,a.patient_id,a.hospital_id,a.episode_exposed,a.anchor_offset_minutes,a.admit_time,m.itemid,m.variable,m.venous_type,m.raw_label,m.dictionary_unit,m.category,m.documentation_mode,
 try_cast(v.charttime AS TIMESTAMP) event_timestamp,date_diff('second',a.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,try_cast(v.valuenum AS DOUBLE) numeric_value,v.valueuom raw_unit,coalesce(try_cast(v.warning AS DOUBLE),0) warning
 FROM {ce} v JOIN mimic_anchors a ON try_cast(v.stay_id AS BIGINT)=a.stay_id AND try_cast(v.hadm_id AS BIGINT)=a.hadm_id JOIN mimic_chart_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
 WHERE try_cast(v.charttime AS TIMESTAMP)>=a.admit_time AND try_cast(v.charttime AS TIMESTAMP)<a.admit_time+INTERVAL '6 hours' ''')
con.execute('DELETE FROM mimic_chart_raw WHERE itemid NOT IN (SELECT itemid FROM mimic_chart_map)')
con.execute('''UPDATE mimic_chart_raw SET variable=m.variable,venous_type=m.venous_type,raw_label=m.raw_label,dictionary_unit=m.dictionary_unit,category=m.category,documentation_mode=m.documentation_mode FROM mimic_chart_map m WHERE mimic_chart_raw.itemid=m.itemid''')
vo2_status('scanning original MIMIC labevents and inputevents')
le=vo2_csv('mimic','labevents')
if not vo2_table_exists('mimic_lab_raw'):con.execute(f'''CREATE TABLE mimic_lab_raw AS SELECT a.stay_id,a.patient_id,a.hospital_id,m.itemid,m.variable,NULL::VARCHAR venous_type,m.raw_label,m.dictionary_unit,m.category,m.documentation_mode,
 try_cast(v.charttime AS TIMESTAMP) event_timestamp,date_diff('second',a.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,try_cast(v.valuenum AS DOUBLE) numeric_value,v.valueuom raw_unit
 FROM {le} v JOIN mimic_anchors a ON try_cast(v.hadm_id AS BIGINT)=a.hadm_id JOIN mimic_lab_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
 WHERE try_cast(v.charttime AS TIMESTAMP)>=a.admit_time AND try_cast(v.charttime AS TIMESTAMP)<a.admit_time+INTERVAL '4 hours' ''')
pressors={221906:'norepinephrine',221289:'epinephrine',221662:'dopamine',221653:'dobutamine',221749:'phenylephrine',222315:'vasopressin',221986:'milrinone'}
pressor_map=pd.DataFrame([dict(itemid=k,raw_label=v,variable='vasoactive_support',documentation_mode='medication_administration_record') for k,v in pressors.items()]);pressor_map=vo2_register('mimic_pressor_map',pressor_map)
ie=vo2_csv('mimic','inputevents')
if not vo2_table_exists('mimic_pressor_raw'):con.execute(f'''CREATE TABLE mimic_pressor_raw AS SELECT a.stay_id,a.patient_id,a.hospital_id,m.itemid,m.variable,NULL::VARCHAR venous_type,m.raw_label,'' dictionary_unit,'Input' category,m.documentation_mode,
 try_cast(v.starttime AS TIMESTAMP) event_timestamp,date_diff('second',a.admit_time,try_cast(v.starttime AS TIMESTAMP))/60.0 event_minute,try_cast(v.rate AS DOUBLE) numeric_value,v.rateuom raw_unit
 FROM {ie} v JOIN mimic_anchors a ON try_cast(v.stay_id AS BIGINT)=a.stay_id JOIN mimic_pressor_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
 WHERE try_cast(v.starttime AS TIMESTAMP)<a.admit_time+INTERVAL '4 hours' AND try_cast(v.endtime AS TIMESTAMP)>a.admit_time''')
vo2_status('scanning original eICU lab source')
elab=vo2_csv('eicu','lab')
if not vo2_table_exists('eicu_lab_candidates'):con.execute(f'''CREATE TABLE eicu_lab_candidates AS SELECT a.stay_id,a.patient_id,a.hospital_id,a.episode_exposed,a.anchor_offset_minutes,try_cast(l.labid AS BIGINT) itemid,l.labname raw_label,try_cast(l.labtypeid AS BIGINT) category,
 try_cast(l.labresultoffset AS DOUBLE) event_minute,try_cast(l.labresult AS DOUBLE) numeric_value,l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface,
 'laboratory_record; bedside_vs_manual_not_encoded' documentation_mode
 FROM {elab} l JOIN eicu_anchors a ON try_cast(l.patientunitstayid AS BIGINT)=a.stay_id
 WHERE try_cast(l.labresultoffset AS DOUBLE)>=0 AND try_cast(l.labresultoffset AS DOUBLE)<360 AND regexp_matches(lower(l.labname),'svo2|scvo2|venous|oxygen|o2|saturation|lactate|creatinine|hemoglobin|haemoglobin|hematocrit|(^|[^a-z])hgb([^a-z]|$)|(^|[^a-z])hct([^a-z]|$)')''')
eicu_lab=con.execute('SELECT * FROM eicu_lab_candidates ORDER BY stay_id,event_minute,itemid').df()
def eicu_lab_variable(label):
    z=' '.join(str(label or '').casefold().split());vtype=classify_venous_measurement(z)
    if vtype:return 'venous_o2',vtype
    if z in {'hgb','hemoglobin','haemoglobin'}:return 'hemoglobin',None
    if z in {'hct','hematocrit'}:return 'hematocrit',None
    if z=='lactate':return 'lactate',None
    if z=='creatinine':return 'creatinine',None
    if 'arterial' in z and ('o2 sat' in z or 'oxygen saturation' in z):return 'arterial_sao2',None
    if 'arterial' in z and ('po2' in z or 'o2 pressure' in z):return 'pao2',None
    if z in {'po2','pao2'}:return 'po2_unspecified',None
    if 'oxygen saturation' in z or 'o2 sat' in z:return 'oxygen_saturation_unspecified',None
    return None,None
mapped=eicu_lab.raw_label.map(eicu_lab_variable);eicu_lab['variable']=[x[0] if x else None for x in mapped];eicu_lab['venous_type']=[x[1] if x else None for x in mapped]
eicu_lab=vo2_register('eicu_lab_frame',eicu_lab);con.execute('CREATE OR REPLACE TABLE eicu_lab_raw AS SELECT * FROM eicu_lab_frame WHERE variable IS NOT NULL');con.unregister('eicu_lab_frame')
vo2_status('scanning original eICU periodic vitals and respiratory support')
vp=vo2_csv('eicu','vitalPeriodic')
if not vo2_table_exists('eicu_vital_wide'):con.execute(f'''CREATE TABLE eicu_vital_wide AS SELECT a.stay_id,a.patient_id,a.hospital_id,try_cast(v.observationoffset AS DOUBLE) event_minute,
 try_cast(v.sao2 AS DOUBLE) spo2,try_cast(v.heartrate AS DOUBLE) heart_rate,try_cast(v.systemicsystolic AS DOUBLE) sbp,try_cast(v.systemicmean AS DOUBLE) mean_arterial_pressure,try_cast(v.cvp AS DOUBLE) cvp,
 try_cast(v.pasystolic AS DOUBLE) pa_systolic,try_cast(v.padiastolic AS DOUBLE) pa_diastolic,try_cast(v.pamean AS DOUBLE) pa_mean
 FROM {vp} v JOIN eicu_anchors a ON try_cast(v.patientunitstayid AS BIGINT)=a.stay_id WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<240''')
rc_path=vo2_source('eicu','respiratoryCharting');rc=f'read_csv({vo2_q(rc_path)},header=true,all_varchar=true,sample_size=10000,strict_mode=true,quote={vo2_q(chr(34))},escape={vo2_q(chr(34))})'
if not vo2_table_exists('eicu_resp_raw'):con.execute(f'''CREATE TABLE eicu_resp_raw AS SELECT a.stay_id,a.patient_id,a.hospital_id,try_cast(r.respchartoffset AS DOUBLE) event_minute,
 concat_ws(' | ',r.respcharttypecat,r.respchartvaluelabel) raw_label,r.respchartvalue raw_value,
 CASE WHEN regexp_matches(lower(concat_ws(' ',r.respcharttypecat,r.respchartvaluelabel)),'fio2|fraction inspired oxygen|inspired o2|oxygen concentration') THEN 'fio2' ELSE 'respiratory_support' END AS "variable"
 FROM {rc} r JOIN eicu_anchors a ON try_cast(r.patientunitstayid AS BIGINT)=a.stay_id WHERE try_cast(r.respchartoffset AS DOUBLE)>=0 AND try_cast(r.respchartoffset AS DOUBLE)<240
 AND regexp_matches(lower(concat_ws(' ',r.respcharttypecat,r.respchartvaluelabel,r.respchartvalue)),'fio2|fraction inspired oxygen|inspired o2|oxygen concentration|ventilat|bipap|cpap|nasal cannula|high flow|nonrebreather|oxygen mask')''')
inf=vo2_csv('eicu','infusionDrug')
if not vo2_table_exists('eicu_pressor_raw'):con.execute(f'''CREATE TABLE eicu_pressor_raw AS SELECT a.stay_id,a.patient_id,a.hospital_id,try_cast(i.infusionoffset AS DOUBLE) event_minute,i.drugname raw_label,i.drugrate raw_value,'vasoactive_support' AS "variable"
 FROM {inf} i JOIN eicu_anchors a ON try_cast(i.patientunitstayid AS BIGINT)=a.stay_id WHERE try_cast(i.infusionoffset AS DOUBLE)>=0 AND try_cast(i.infusionoffset AS DOUBLE)<240
 AND regexp_matches(lower(i.drugname),'norepinephrine|epinephrine|dopamine|dobutamine|phenylephrine|vasopressin|milrinone')''')
vo2_source('eicu','vitalAperiodic',required=False)
for key,info in source_manifest.items():
    if info.get('present'):
        s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns']),key
vo2_json('input_manifest.json',dict(protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256,sources=source_manifest,prior_database=str(prior),prior_database_bytes=prior.stat().st_size,phase_a_only=True))
vo2_status('original-source scans complete; outcome-blind support aggregation pending')
"""

SUPPORT = r"""
vo2_status('validating venous units and building aggregate support')
venous_frames=[];quality=[]
m=con.execute("SELECT stay_id,patient_id,hospital_id,episode_exposed,anchor_offset_minutes,event_timestamp,event_minute,numeric_value,coalesce(nullif(trim(raw_unit),''),dictionary_unit) unit,itemid,raw_label,venous_type,documentation_mode FROM mimic_chart_raw WHERE variable='venous_o2' AND warning<>1 ORDER BY stay_id,event_minute,itemid").df()
m['dataset']='mimic';venous_frames.append(m)
e=con.execute("SELECT stay_id,patient_id,hospital_id,episode_exposed,anchor_offset_minutes,NULL::TIMESTAMP event_timestamp,event_minute,numeric_value,coalesce(nullif(trim(unit_system),''),unit_interface) unit,itemid,raw_label,venous_type,documentation_mode,unit_system,unit_interface FROM eicu_lab_raw WHERE variable='venous_o2' ORDER BY stay_id,event_minute,itemid").df()
e['dataset']='eicu';venous_frames.append(e)
accepted=[]
for expected_dataset,frame in zip(['mimic','eicu'],venous_frames):
    values=[];ok=[]
    for r in frame.itertuples(index=False):
        value,valid=normalize_venous_saturation(r.numeric_value,r.unit)
        if r.dataset=='eicu':
            us='' if pd.isna(getattr(r,'unit_system',None)) else re.sub(r'\s+','',str(getattr(r,'unit_system')).casefold())
            ui='' if pd.isna(getattr(r,'unit_interface',None)) else re.sub(r'\s+','',str(getattr(r,'unit_interface')).casefold())
            if us and ui and us!=ui:valid=False;value=np.nan
        values.append(value);ok.append(valid)
    frame=frame.copy();frame['value']=values;frame['unit_valid']=ok
    quality.append(dict(dataset=expected_dataset,candidate_records=len(frame),accepted_records=int(frame.unit_valid.sum()),unknown_conflicting_or_out_of_range=int((~frame.unit_valid).sum())))
    accepted.append(frame.loc[frame.unit_valid].copy())
venous=pd.concat(accepted,ignore_index=True) if accepted else pd.DataFrame()
if len(venous):
    venous=venous.groupby(['dataset','stay_id','patient_id','hospital_id','episode_exposed','anchor_offset_minutes','event_minute','venous_type'],dropna=False,as_index=False).agg(value=('value','median'),unit=('unit','first'),raw_label=('raw_label',lambda x:' | '.join(sorted(set(map(str,x))))),itemid=('itemid',lambda x:' | '.join(sorted(set(map(str,x))))),documentation_mode=('documentation_mode','first'))
venous=vo2_register('venous_frame',venous);con.execute('CREATE OR REPLACE TABLE venous_measurements_private AS SELECT * FROM venous_frame');con.unregister('venous_frame')
mapping=[]
def add_mapping(frame,dataset,table,variable_col='variable',label_col='raw_label',unit_col='raw_unit',item_col='itemid',mode_col='documentation_mode'):
    if frame.empty:return
    frame=frame.copy();frame['dataset']=dataset;frame['source_table']=table
    for keys,g in frame.groupby([variable_col,label_col,item_col],dropna=False,sort=True):
        variable,label,item=keys;units=sorted({str(x) for x in g[unit_col].dropna() if str(x).strip()}) if unit_col in g else []
        mapping.append(dict(dataset=dataset,variable=variable,venous_type=(g.venous_type.dropna().iloc[0] if 'venous_type' in g and g.venous_type.notna().any() else None),source_table=table,raw_item_or_column=item,raw_label=label,units_observed=' | '.join(units),documentation_mode=(g[mode_col].iloc[0] if mode_col in g else 'not_encoded'),raw_records_0_4h=int(g.event_minute.between(0,240,inclusive='left').sum()),encounters_0_4h=int(g.loc[g.event_minute.between(0,240,inclusive='left'),'stay_id'].nunique()),mapping_status='observed'))
add_mapping(con.execute("SELECT * FROM mimic_chart_raw WHERE event_minute>=0 AND event_minute<240").df(),'mimic','chartevents')
add_mapping(con.execute("SELECT * FROM mimic_lab_raw").df(),'mimic','labevents')
add_mapping(con.execute("SELECT * FROM mimic_pressor_raw").df(),'mimic','inputevents')
add_mapping(con.execute("SELECT *,coalesce(nullif(trim(unit_system),''),unit_interface) raw_unit FROM eicu_lab_raw WHERE event_minute>=0 AND event_minute<240").df(),'eicu','lab',item_col='category')
wide=con.execute('SELECT * FROM eicu_vital_wide').df()
for variable,column,unit in [('spo2','spo2','percent_schema_convention'),('heart_rate','heart_rate','beats/min'),('sbp','sbp','mmHg_schema_convention'),('map','mean_arterial_pressure','mmHg_schema_convention'),('cvp','cvp','native_unit_not_encoded'),('pulmonary_artery_pressure','pa_systolic','mmHg_schema_convention'),('pulmonary_artery_pressure','pa_diastolic','mmHg_schema_convention'),('pulmonary_artery_pressure','pa_mean','mmHg_schema_convention')]:
    mask=wide[column].notna();mapping.append(dict(dataset='eicu',variable=variable,venous_type=None,source_table='vitalPeriodic',raw_item_or_column=column,raw_label=column,units_observed=unit,documentation_mode='periodic_monitor_interface',raw_records_0_4h=int(mask.sum()),encounters_0_4h=int(wide.loc[mask,'stay_id'].nunique()),mapping_status='observed'))
resp=con.execute('SELECT *,NULL::VARCHAR itemid,raw_value raw_unit FROM eicu_resp_raw').df();add_mapping(resp,'eicu','respiratoryCharting',item_col='raw_label')
press=con.execute("SELECT *,NULL::VARCHAR itemid,'' raw_unit FROM eicu_pressor_raw").df();add_mapping(press,'eicu','infusionDrug',item_col='raw_label',mode_col='variable')
mapping.extend([
 dict(dataset='eicu',variable='arterial_sao2',venous_type=None,source_table='vitalAperiodic',raw_item_or_column='saO2',raw_label='arterial saturation source absent from mounted extract',units_observed='',documentation_mode='unavailable',raw_records_0_4h=0,encounters_0_4h=0,mapping_status='source_file_absent'),
 dict(dataset='eicu',variable='cardiac_output',venous_type=None,source_table='nurseCharting',raw_item_or_column='',raw_label='source absent from mounted extract',units_observed='',documentation_mode='unavailable',raw_records_0_4h=0,encounters_0_4h=0,mapping_status='source_file_absent'),
 dict(dataset='eicu',variable='cardiac_index',venous_type=None,source_table='nurseCharting',raw_item_or_column='',raw_label='source absent from mounted extract',units_observed='',documentation_mode='unavailable',raw_records_0_4h=0,encounters_0_4h=0,mapping_status='source_file_absent'),
 dict(dataset='eicu',variable='cardiac_power_output',venous_type=None,source_table='nurseCharting',raw_item_or_column='',raw_label='source absent from mounted extract',units_observed='',documentation_mode='unavailable',raw_records_0_4h=0,encounters_0_4h=0,mapping_status='source_file_absent')])
source_mapping=pd.DataFrame(mapping);source_mapping['protocol_sha256']=VENOUS_O2_FEASIBILITY_SHA256;source_mapping.to_csv(OUT/'source_mapping.csv',index=False)
feasibility=[];anchor_support=[];frequency=[];sites=[]
for d in ['eicu','mimic']:
    anchors=con.execute(f'SELECT stay_id,patient_id,hospital_id,episode_exposed,anchor_offset_minutes FROM {d}_anchors ORDER BY stay_id').df()
    for vtype in VENOUS_TYPES:
        vm=venous.loc[venous.dataset.eq(d)&venous.venous_type.eq(vtype)].copy() if len(venous) else pd.DataFrame(columns=['stay_id','event_minute'])
        counts=vm.loc[vm.event_minute.between(0,240,inclusive='left')].groupby('stay_id').size()
        panel=anchors.copy();panel['n_0_4h']=panel.stay_id.map(counts).fillna(0).astype(int);panel['n_pre']=0;panel['n_post']=0
        bystay={k:g.event_minute.to_numpy(float) for k,g in vm.groupby('stay_id')} if len(vm) else {}
        gaps=[]
        for i,r in panel.iterrows():
            t=np.sort(np.unique(bystay.get(r.stay_id,np.array([],dtype=float))));rel=t-float(r.anchor_offset_minutes);panel.at[i,'n_pre']=int(((rel>=-60)&(rel<0)).sum());panel.at[i,'n_post']=int(((rel>30)&(rel<=120)).sum())
            x=t[(t>=0)&(t<240)]
            if len(x)>1:gaps.extend(np.diff(x).tolist())
        panel['valid_pair']=panel.n_pre.ge(1)&panel.n_post.ge(1)
        pair=panel.loc[panel.valid_pair];total=len(pair);exposed=int(pair.episode_exposed.sum());controls=int(pair.episode_exposed.eq(0).sum())
        feasibility.append(dict(dataset=d,venous_type=vtype,cohort=len(panel),any_0_4h=int(panel.n_0_4h.ge(1).sum()),at_least_2_measurements=int(panel.n_0_4h.ge(2).sum()),at_least_3_measurements=int(panel.n_0_4h.ge(3).sum()),valid_pre_post_pair=total,exposed=exposed,controls=controls,support_tier=support_tier(total,exposed,controls),contributing_eicu_hospitals=(int(pair.hospital_id.nunique()) if d=='eicu' else None),protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256))
        measured_counts=panel.loc[panel.n_0_4h.gt(0),'n_0_4h']
        frequency.append(dict(dataset=d,venous_type=vtype,accepted_records_0_4h=int(panel.n_0_4h.sum()),encounters_measured_0_4h=int(len(measured_counts)),median_measurements_per_measured_encounter=(float(measured_counts.median()) if len(measured_counts) else None),median_intermeasurement_interval_minutes=(float(np.median(gaps)) if gaps else None),protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256))
        for exposure in [0,1]:
            g=panel.loc[panel.episode_exposed.eq(exposure)];anchor_support.append(dict(dataset=d,venous_type=vtype,exposure='exposed' if exposure else 'control',eligible_anchors=len(g),any_0_4h=int(g.n_0_4h.ge(1).sum()),valid_pre_post_pair=int(g.valid_pair.sum()),protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256))
        if d=='eicu':
            for hospital,g in panel.groupby('hospital_id',dropna=False):sites.append(dict(hospital_id=hospital,venous_type=vtype,eligible_anchors=len(g),measured_0_4h=int(g.n_0_4h.ge(1).sum()),valid_pre_post_pair=int(g.valid_pair.sum()),protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256))
feasibility=pd.DataFrame(feasibility);frequency=pd.DataFrame(frequency);anchor_support=pd.DataFrame(anchor_support);site_distribution=pd.DataFrame(sites)
feasibility.to_csv(OUT/'feasibility_audit.csv',index=False);frequency.to_csv(OUT/'measurement_frequency.csv',index=False);anchor_support.to_csv(OUT/'anchor_support.csv',index=False);site_distribution.to_csv(OUT/'eicu_site_distribution.csv',index=False)
strategy=choose_venous_strategy(feasibility);summary=phase_a_decision_summary(strategy);summary.update(protocol_sha256=VENOUS_O2_FEASIBILITY_SHA256,execution_environment='google_colab' if IN_COLAB else 'local',phase_b_protocol_frozen=False,predefined_fallback_executed=False)
vo2_json('decision_gate_summary.json',summary);vo2_json('quality_audit.json',quality)
log='\n'.join([f"{datetime.now(timezone.utc).isoformat()} phase=A environment={summary['execution_environment']}",f"venous_o2_feasible={summary['venous_o2_feasible']} strategy={strategy['strategy']} tier={strategy['analysis_tier']}",'clinical_outcomes_inspected=false','effect_estimates_computed=false','endpoint_rates_printed=false'])+'\n';(OUT/'execution.log').write_text(log)
for key,info in source_manifest.items():
    if info.get('present'):
        s=Path(info['path']).stat();assert (s.st_size,s.st_mtime_ns)==(info['bytes'],info['mtime_ns']),key
con.close();vo2_status('Phase A outcome-blind feasibility audit','complete',venous_o2_feasible=summary['venous_o2_feasible'],recommended_action=summary['recommended_action'])
print(feasibility.to_string(index=False));print(json.dumps(summary,indent=2))
"""
