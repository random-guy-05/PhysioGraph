"""Cell sources embedded verbatim in the primary runnable notebook."""

SETUP = r"""
import sys,json,hashlib,importlib.util,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph'
PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_hemoglobin_interaction';OUT.mkdir(exist_ok=True)
def hj(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def hs(stage,status='running'):
    hj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),
        environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
    print(stage,status,flush=True)
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==HB_INTERACTION_PROTOCOL_SHA256
else:hj('protocol_lock.json',dict(sha256=HB_INTERACTION_PROTOCOL_SHA256,
    utc=datetime.now(timezone.utc).isoformat(),hemoglobin_outcome_associations_not_examined=True,
    environment='google_colab' if IN_COLAB else 'local'))
def qs(v):return "'"+str(v).replace("'","''")+"'"
hs('care-context extraction setup')
"""

CONTEXT = r"""
# Extract treatment and oxygen context without joining Hb concentrations or outcomes.
con=duckdb.connect(str(PRIVATE/'hemoglobin_interaction.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
previous=json.loads((PROJECT/'research/spo2_circulatory_context/input_manifest.json').read_text())
used={}
def source(dataset,name):
    key=dataset+'/'+name;p=DRIVE/'Data'/dataset/'Full'/(name+'.csv');s=p.stat()
    old=previous.get(key)
    if old:
        assert s.st_size==old['bytes'] and s.st_mtime_ns==old['mtime_ns'],key
        info={**old,'fingerprint_reused_from':'spo2_circulatory_context/input_manifest.json'}
    else:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        info=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest())
    used[key]=info
    return f'read_csv({qs(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true,delim={qs(",")},quote={qs(chr(34))},escape={qs(chr(34))})'
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_context_eligible AS
      SELECT c.stay_id,c.admit_time FROM prior.{d}_cohort c
      JOIN prior.{d}_features f USING(stay_id)
      WHERE f.spo2_bins>=3 AND f.spo2_transitions>=2''')
hs('eICU respiratory oxygen extraction')
resp=source('eICU','respiratoryCharting')
con.execute(rf'''CREATE OR REPLACE TABLE eicu_fio2_raw AS
  SELECT e.stay_id,try_cast(r.respchartoffset AS DOUBLE) event_minute,
    concat_ws(' ',r.respcharttypecat,r.respchartvaluelabel) raw_label,
    r.respchartvalue raw_value,
    try_cast(regexp_extract(r.respchartvalue,'([-+]?[0-9]*[.]?[0-9]+)',1) AS DOUBLE) numeric_value
  FROM {resp} r JOIN eicu_context_eligible e ON try_cast(r.patientunitstayid AS BIGINT)=e.stay_id
  WHERE try_cast(r.respchartoffset AS DOUBLE)>=0 AND try_cast(r.respchartoffset AS DOUBLE)<240
    AND regexp_matches(lower(concat_ws(' ',r.respcharttypecat,r.respchartvaluelabel)),
        '\bfio2\b|fraction inspired oxygen|inspired o2|oxygen concentration')''')
hs('eICU documented circulatory support extraction')
inf=source('eICU','infusionDrug')
con.execute(rf'''CREATE OR REPLACE TABLE eicu_pressor_raw AS
  SELECT e.stay_id,l.drugname raw_label,try_cast(l.infusionoffset AS DOUBLE) event_minute,
    coalesce(try_cast(l.drugrate AS DOUBLE),try_cast(l.infusionrate AS DOUBLE)) positive_rate
  FROM {inf} l JOIN eicu_context_eligible e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
  WHERE try_cast(l.infusionoffset AS DOUBLE)>=0 AND try_cast(l.infusionoffset AS DOUBLE)<240
  AND regexp_matches(lower(l.drugname),
    '\b(norepinephrine|levophed|epinephrine|adrenaline|dopamine|dobutamine|dobutrex|phenylephrine|neo[ -]?synephrine|vasopressin|milrinone|primacor)\b')''')
hs('eICU invasive ventilation documentation extraction')
treat=source('eICU','treatment')
con.execute(rf'''CREATE OR REPLACE TABLE eicu_vent_raw AS
  SELECT e.stay_id,l.treatmentstring raw_label,try_cast(l.treatmentoffset AS DOUBLE) event_minute,
    regexp_matches(lower(l.treatmentstring),'(^|\|)\s*mechanical ventilation\s*(\||$)')
      AND NOT regexp_matches(lower(l.treatmentstring),'non[ -]?invasive ventilation|ventilator weaning') accepted
  FROM {treat} l JOIN eicu_context_eligible e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
  WHERE try_cast(l.treatmentoffset AS DOUBLE)>=0 AND try_cast(l.treatmentoffset AS DOUBLE)<240
    AND regexp_matches(lower(l.treatmentstring),'ventilat')''')
hs('MIMIC vasoactive infusion extraction')
inf=source('MIMIC','inputevents')
con.execute(f'''CREATE OR REPLACE TABLE mimic_pressor_raw AS
  SELECT e.stay_id,l.itemid raw_label,l.statusdescription,
    date_diff('second',e.admit_time,try_cast(l.starttime AS TIMESTAMP))/60.0 event_minute,
    date_diff('second',e.admit_time,try_cast(l.endtime AS TIMESTAMP))/60.0 end_minute,
    try_cast(l.rate AS DOUBLE) positive_rate
  FROM {inf} l JOIN mimic_context_eligible e ON try_cast(l.stay_id AS BIGINT)=e.stay_id
  WHERE try_cast(l.itemid AS BIGINT) IN (221906,221289,221662,221653,221749,222315,221986)
    AND try_cast(l.starttime AS TIMESTAMP)<e.admit_time+INTERVAL '240 minutes'
    AND try_cast(l.endtime AS TIMESTAMP)>e.admit_time
    AND try_cast(l.endtime AS TIMESTAMP)>=try_cast(l.starttime AS TIMESTAMP)
    AND lower(coalesce(l.statusdescription,''))<>'rewritten' ''')
hs('MIMIC invasive ventilation extraction')
proc=source('MIMIC','procedureevents')
con.execute(f'''CREATE OR REPLACE TABLE mimic_vent_raw AS
  SELECT e.stay_id,l.itemid raw_label,
    date_diff('second',e.admit_time,try_cast(l.starttime AS TIMESTAMP))/60.0 event_minute,
    date_diff('second',e.admit_time,try_cast(l.endtime AS TIMESTAMP))/60.0 end_minute,true accepted
  FROM {proc} l JOIN mimic_context_eligible e ON try_cast(l.stay_id AS BIGINT)=e.stay_id
  WHERE try_cast(l.itemid AS BIGINT)=225792
    AND try_cast(l.starttime AS TIMESTAMP)<e.admit_time+INTERVAL '240 minutes'
    AND try_cast(l.endtime AS TIMESTAMP)>e.admit_time
    AND try_cast(l.endtime AS TIMESTAMP)>=try_cast(l.starttime AS TIMESTAMP)''')
hs('MIMIC FiO2 dictionary and chart extraction')
items=source('MIMIC','d_items')
con.execute(rf'''CREATE OR REPLACE TABLE mimic_fio2_items AS
  SELECT try_cast(itemid AS BIGINT) itemid,label,category FROM {items}
  WHERE lower(linksto)='chartevents'
    AND regexp_matches(lower(label),'inspired o2 fraction|\bfio2\b')
    AND NOT regexp_matches(lower(coalesce(category,'')||' '||label),'ecmo|tandemheart|\(ch\)')''')
hj('mimic_fio2_dictionary.json',con.execute('SELECT * FROM mimic_fio2_items').df().to_dict('records'))
charts=source('MIMIC','chartevents')
con.execute(f'''CREATE OR REPLACE TABLE mimic_fio2_raw AS
  SELECT e.stay_id,date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
    m.label raw_label,l.value raw_value,try_cast(l.valuenum AS DOUBLE) numeric_value
  FROM {charts} l JOIN mimic_context_eligible e ON try_cast(l.stay_id AS BIGINT)=e.stay_id
  JOIN mimic_fio2_items m ON try_cast(l.itemid AS BIGINT)=m.itemid
  WHERE try_cast(l.charttime AS TIMESTAMP)>=e.admit_time
    AND try_cast(l.charttime AS TIMESTAMP)<e.admit_time+INTERVAL '240 minutes'
    AND coalesce(try_cast(l.warning AS INTEGER),0)<>1''')
audit={};coverage=[]
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_fio2 AS
      SELECT *,CASE WHEN numeric_value BETWEEN 0.20 AND 1 THEN numeric_value*100
        ELSE numeric_value END fio2_percent FROM {d}_fio2_raw''')
    audit[d+'_fio2']=con.execute(f'''SELECT raw_label,count(*) records,
      count(*) FILTER(WHERE fio2_percent BETWEEN 21 AND 100) accepted,
      min(numeric_value) numeric_min,max(numeric_value) numeric_max
      FROM {d}_fio2 GROUP BY raw_label ORDER BY records DESC''').df().replace({np.nan:None}).to_dict('records')
    audit[d+'_pressor']=con.execute(f'''SELECT raw_label,count(*) records,
      count(*) FILTER(WHERE positive_rate>0) positive_records FROM {d}_pressor_raw
      GROUP BY raw_label ORDER BY records DESC''').df().to_dict('records')
    audit[d+'_vent']=con.execute(f'''SELECT raw_label,count(*) records,
      count(*) FILTER(WHERE accepted) accepted FROM {d}_vent_raw
      GROUP BY raw_label ORDER BY records DESC''').df().to_dict('records')
    con.execute(f'''CREATE OR REPLACE TABLE {d}_care AS
      SELECT e.stay_id,f.fio2_max,coalesce(v.vent_documented,0) vent_documented,
        coalesce(p.vasoactive_documented,0) vasoactive_documented
      FROM {d}_context_eligible e
      LEFT JOIN (SELECT stay_id,max(fio2_percent) fio2_max FROM {d}_fio2
        WHERE fio2_percent BETWEEN 21 AND 100 GROUP BY stay_id) f USING(stay_id)
      LEFT JOIN (SELECT stay_id,1 vent_documented FROM {d}_vent_raw WHERE accepted GROUP BY stay_id) v USING(stay_id)
      LEFT JOIN (SELECT stay_id,1 vasoactive_documented FROM {d}_pressor_raw WHERE positive_rate>0 GROUP BY stay_id) p USING(stay_id)''')
    coverage.append(dict(dataset=d,**con.execute(f'''SELECT count(*) eligible,count(fio2_max) fio2_observed,
      sum(vent_documented) documented_invasive_ventilation,sum(vasoactive_documented) documented_vasoactive
      FROM {d}_care''').df().iloc[0].to_dict()))
for key,info in used.items():
    s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],key
hj('source_manifest.json',used);hj('care_label_audit.json',audit);hj('care_coverage.json',coverage)
con.close()
print(json.dumps(coverage,indent=2,default=str))
hs('care-context extraction','complete')
"""

MODELS = r"""
# Open Hb-by-outcome associations only under the already frozen specification.
import warnings
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from scipy.optimize import linprog
from statsmodels.stats.multitest import multipletests
if importlib.util.find_spec('sklearn') is None:
    subprocess.check_call([sys.executable,'-m','pip','install','-q','scikit-learn==1.6.1'])
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
hs('frozen hemoglobin interaction analysis')
con=duckdb.connect(str(PRIVATE/'hemoglobin_interaction.duckdb'))
import re
care_checks=[]
for d in ['eicu','mimic']:
    f=con.execute(f'SELECT * FROM {d}_fio2_raw').df()
    if d=='mimic':assert set(f.raw_label)=={'Inspired O2 Fraction'}
    else:assert set(f.raw_label)<= {'respFlowSettings FiO2','respFlowCareData FIO2 (%)',
        'respFlowCareData FiO2','respFlowCareData Set Fraction of Inspired Oxygen (FIO2)'}
    f['normalized']=f.numeric_value.where(~f.numeric_value.between(.2,1),f.numeric_value*100)
    maxima=f.loc[f.normalized.between(21,100)].groupby('stay_id').normalized.max().sort_index()
    care=con.execute(f'SELECT * FROM {d}_care ORDER BY stay_id').df().set_index('stay_id')
    assert np.allclose(care.fio2_max.dropna(),maxima)
    assert care.fio2_max.dropna().index.equals(maxima.index)
    v=con.execute(f'SELECT * FROM {d}_vent_raw').df()
    if d=='eicu':
        accepted=v.raw_label.map(lambda s:'mechanical ventilation' in [x.strip() for x in s.lower().split('|')]
            and not re.search(r'non[ -]?invasive ventilation|ventilator weaning',s.lower()))
        assert (accepted.astype(bool)==v.accepted).all()
    else:accepted=v.accepted
    expected_vent=set(v.loc[accepted,'stay_id'])
    assert expected_vent==set(care.index[care.vent_documented==1])
    p=con.execute(f'SELECT * FROM {d}_pressor_raw').df()
    assert set(p.loc[p.positive_rate>0,'stay_id'])==set(care.index[care.vasoactive_documented==1])
    care_checks.append(dict(dataset=d,independent_normalization_and_aggregation_match=True,
        fio2_challenge_fields_contribute_no_rows=True if d=='mimic' else None,
        invasive_ventilation_patient_set_matches=True,positive_rate_patient_set_matches=True))
hj('independent_care_validation.json',care_checks)
for alias,name in [('prior','spo2_context'),('amend','preicu_linkage_amendment'),('original','spo2_hemoglobin')]:
    con.execute(f"ATTACH {qs(PRIVATE/(name+'.duckdb'))} AS {alias} (READ_ONLY)")
fingerprints={}
for name in ['spo2_context','preicu_linkage_amendment','spo2_hemoglobin']:
    p=PRIVATE/(name+'.duckdb');s=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    fingerprints[name]=dict(path=str(p),sha256=h.hexdigest(),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
hj('model_input_fingerprints.json',fingerprints)
frames={};availability=[]
for d in ['eicu','mimic']:
    con.execute(f'''CREATE OR REPLACE TABLE {d}_analysis AS
      SELECT e.stay_id,e.exposed,c.person_id,c.age,c.is_male,c.sex_unknown_flag,
        c.hospital_mortality,c.troponin_rise_12h,c.baseline_creatinine,c.baseline_lactate,
        h.hemoglobin,h.event_minute hb_event_minute,h.available_minute hb_available_minute,
        o.hemoglobin original_hemoglobin,o.available_minute original_hb_available_minute,
        f.spo2_mean,f.hr_mean,f.map_mean,s.hypoxic_fraction,s.spo2_readings,
        k.fio2_max,k.vent_documented,k.vasoactive_documented
      FROM amend.{d}_eligible e JOIN prior.{d}_context_endpoints c USING(stay_id)
      JOIN amend.{d}_baseline_hb h USING(stay_id)
      LEFT JOIN original.{d}_baseline_hb o USING(stay_id)
      JOIN prior.{d}_features f USING(stay_id) JOIN {d}_care k USING(stay_id)
      JOIN (SELECT stay_id,avg(CASE WHEN value<90 THEN 1.0 ELSE 0.0 END) hypoxic_fraction,
        sum(n_readings) spo2_readings FROM prior.{d}_bins WHERE concept='spo2' GROUP BY stay_id) s USING(stay_id)''')
    df=con.execute(f'SELECT * FROM {d}_analysis ORDER BY stay_id').df()
    assert df.stay_id.is_unique and df.person_id.notna().all()
    assert len(df)==({'mimic':3450,'eicu':3984}[d])
    assert int(df.troponin_rise_12h.notna().sum())==({'mimic':516,'eicu':659}[d])
    assert df.exposed.isin([0,1]).all()
    frames[d]=df
    availability.append(dict(dataset=d,n=len(df),unique_persons=df.person_id.nunique(),
        hb_entered_after_icu=int((df.hb_available_minute>0).sum()),
        missing={k:int(df[k].isna().sum()) for k in ['age','spo2_mean','hypoxic_fraction','hr_mean','map_mean',
          'baseline_creatinine','baseline_lactate','fio2_max','troponin_rise_12h','hospital_mortality']}))
hj('analysis_availability.json',availability)

def design(df,care=True,medians=None):
    z=pd.DataFrame(index=df.index)
    z['intercept']=1.0;z['X']=df.exposed.astype(float);z['H']=(12-df.hemoglobin)/2
    z['X_H']=z.X*z.H
    vals=dict(age=df.age/10,mean_spo2=(df.spo2_mean-95)/5,
        hypoxic_fraction=df.hypoxic_fraction,log_readings=np.log1p(df.spo2_readings),
        mean_hr=df.hr_mean/20,mean_map=df.map_mean/20,
        log_creatinine=np.log(df.baseline_creatinine.where(df.baseline_creatinine>0)),
        log_lactate=np.log(df.baseline_lactate.where(df.baseline_lactate>0)))
    if care:vals['fio2']=df.fio2_max/20
    fit_medians={} if medians is None else dict(medians)
    for key,v in vals.items():
        v=v.replace([np.inf,-np.inf],np.nan)
        if medians is None:fit_medians[key]=float(v.median()) if v.notna().any() else 0.0
        z[key]=v.fillna(fit_medians[key]);z[key+'_missing']=v.isna().astype(float)
    z['male']=df.is_male.fillna(0).astype(float)
    z['sex_unknown']=df.sex_unknown_flag.fillna(1).astype(float)
    if care:
        z['vent_documented']=df.vent_documented.astype(float)
        z['vasoactive_documented']=df.vasoactive_documented.astype(float)
    z['H_mean_spo2']=z.H*z.mean_spo2
    z['H_hypoxic_fraction']=z.H*z.hypoxic_fraction
    assert np.isfinite(z.to_numpy()).all()
    return z,fit_medians

def standardized(df,z,fit):
    neighborhoods={f'hb{hb}_X{x}':int(((df.hemoglobin.between(hb-1,hb+1))&(df.exposed==x)).sum())
        for hb in [8,12] for x in [0,1]}
    result=dict(neighborhoods=neighborhoods,support_pass=min(neighborhoods.values())>=20)
    if not result['support_pass']:return result
    estimates={};grads={};cov=np.asarray(fit.cov_params())
    for hb in [8,12]:
        for x in [0,1]:
            a=z.copy();a['X']=x;a['H']=(12-hb)/2;a['X_H']=a.X*a.H
            for name,parent in [('H_mean_spo2','mean_spo2'),('H_hypoxic_fraction','hypoxic_fraction')]:
                if name in a:a[name]=a.H*a[parent]
            av=a.to_numpy();p=expit(av@np.asarray(fit.params));g=(av*(p*(1-p))[:,None]).mean(axis=0)
            key=f'hb{hb}_X{x}';v=float(p.mean());se=float(np.sqrt(max(0,g@cov@g)))
            estimates[key]=dict(risk=v,ci95=[v-1.96*se,v+1.96*se]);grads[key]=g
    contrasts={}
    for hb in [8,12]:
        a,b=f'hb{hb}_X1',f'hb{hb}_X0';v=estimates[a]['risk']-estimates[b]['risk'];g=grads[a]-grads[b]
        se=float(np.sqrt(max(0,g@cov@g)));contrasts[f'RD_hb{hb}']=dict(value=v,ci95=[v-1.96*se,v+1.96*se])
    v=contrasts['RD_hb8']['value']-contrasts['RD_hb12']['value']
    g=grads['hb8_X1']-grads['hb8_X0']-grads['hb12_X1']+grads['hb12_X0']
    se=float(np.sqrt(max(0,g@cov@g)))
    result.update(risks=estimates,contrasts=contrasts,difference_in_differences=dict(value=v,ci95=[v-1.96*se,v+1.96*se]))
    return result

fits={};results=[];design_information=[]
def fit_one(d,df,endpoint,variant,care=True,weights=None):
    a=df.loc[df[endpoint].notna()].copy()
    y=a[endpoint].astype(float);assert y.isin([0,1]).all()
    z,med=design(a,care=care)
    constants=[k for k in z if k!='intercept' and z[k].nunique()==1]
    invalid_required=bool(set(constants)&{'X','H','X_H'})
    z=z.drop(columns=[k for k in constants if k not in ['X','H','X_H']])
    rank=int(np.linalg.matrix_rank(z));n=len(a);events=int(y.sum());clusters=int(a.person_id.nunique())
    info=dict(dataset=d,endpoint=endpoint,variant=variant,n=n,events=events,non_events=n-events,
        persons=clusters,parameters=z.shape[1],events_per_parameter=events/z.shape[1],
        hb_sd=float(a.hemoglobin.std()),columns=list(z),constant_columns_omitted=constants,
        rank=rank,condition_number=float(np.linalg.cond(z)),medians=med,estimable=False)
    if min(events,n-events)<100 or clusters<100 or a.hemoglobin.std()<0.5 or invalid_required or rank<z.shape[1]:
        info['reason']='Prespecified information, exposure variation or full-rank gate failed'
        results.append(info);return
    signed=z.to_numpy()*(2*y.to_numpy()-1)[:,None]
    separation=linprog(np.zeros(z.shape[1]),A_ub=np.vstack([-signed,-signed.sum(axis=0)]),
        b_ub=np.r_[np.zeros(n),-1.0],bounds=[(None,None)]*z.shape[1],method='highs')
    info['separation_solver_status']=int(separation.status)
    if separation.status!=2:
        info['reason']=('Complete/quasi-complete separation: no finite unpenalized model estimate'
            if separation.status==0 else 'Separation validity check could not establish a finite model')
        if separation.status==0:
            direction=separation.x;margin=signed@direction
            info['separation_certificate']=dict(min_signed_margin=float(margin.min()),
                summed_signed_margin=float(margin.sum()),
                direction={k:float(v) for k,v in zip(z.columns,direction) if abs(v)>1e-8})
            assert margin.min()>-1e-6 and margin.sum()>.999
        results.append(info);return
    if variant=='primary':
        # Planning approximation using a hypothetical constant risk; no observed y enters this calculation.
        inv=np.linalg.inv(z.to_numpy().T@z.to_numpy());j=list(z).index('X_H')
        planning=[]
        for risk in [.1,.3,.5]:
            se=float(np.sqrt(inv[j,j]/(risk*(1-risk))))
            planning.append(dict(assumed_constant_outcome_risk=risk,independent_fisher_se=se,
                interaction_OR_for_80_percent_power_bonferroni4=float(np.exp((norm.ppf(1-.05/8)+norm.ppf(.8))*se))))
        design_information.append(dict(dataset=d,endpoint=endpoint,planning=planning,
            limitation='Independent-observation null-risk approximation, not empirical clustered power'))
        hj('design_information.json',design_information)
    w=None if weights is None else np.asarray(weights.loc[a.index],float)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit=sm.GEE(y,z,groups=a.person_id.astype(str),family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Independence(),weights=w).fit(maxiter=200,ctol=1e-8)
        cov=np.asarray(fit.cov_params());eig=np.linalg.eigvalsh((cov+cov.T)/2)
        info.update(converged=bool(fit.converged),warnings=[str(x.message) for x in caught],
            robust_covariance_min_eigenvalue=float(eig.min()))
        if not fit.converged or not np.isfinite(fit.params).all() or not np.isfinite(cov).all() or eig.min()<=0:
            info['reason']='Convergence or finite positive-definite robust covariance requirement failed'
            results.append(info);return
        beta=float(fit.params['X_H']);se=float(fit.bse['X_H'])
        info.update(estimable=True,interaction_log_OR=beta,robust_se=se,
            interaction_OR=float(np.exp(beta)),interaction_OR_ci95=[float(np.exp(beta-1.96*se)),float(np.exp(beta+1.96*se))],
            p=float(2*norm.sf(abs(beta/se))),standardized=standardized(a,z,fit),
            coefficients={k:dict(beta=float(fit.params[k]),robust_se=float(fit.bse[k])) for k in z})
        fits[(d,endpoint,variant)]=(a,z,fit)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
        info['reason']=type(exc).__name__+': '+str(exc)
    results.append(info)

# Four primary tests are computed and adjusted as one immutable family.
for d,df in frames.items():
    for endpoint in ['troponin_rise_12h','hospital_mortality']:
        fit_one(d,df,endpoint,'primary')
primary=[r for r in results if r['variant']=='primary'];assert len(primary)==4
adjusted=multipletests([r.get('p',1.0) for r in primary],alpha=.05,method='holm')[1]
for r,p in zip(primary,adjusted):r['holm_p']=float(p)
hj('primary_results.json',primary)
print(json.dumps([{k:r.get(k) for k in ['dataset','endpoint','n','events','estimable','interaction_OR','interaction_OR_ci95','holm_p','reason']} for r in primary],indent=2))

for d,df in frames.items():
    hs(d+' prespecified sensitivities')
    original=df.loc[df.original_hemoglobin.notna()].copy()
    original['hemoglobin']=original.original_hemoglobin
    original['hb_available_minute']=original.original_hb_available_minute
    timely=df.loc[df.hb_available_minute.notna()&(df.hb_available_minute<=0)].copy()
    for endpoint in ['troponin_rise_12h','hospital_mortality']:
        fit_one(d,original,endpoint,'explicit_id_only')
        fit_one(d,timely,endpoint,'entered_by_icu_admission')
        fit_one(d,df,endpoint,'without_care_covariates',care=False)

observation_results=[]
for d,df in frames.items():
    obs=df.troponin_rise_12h.notna().astype(int)
    folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20260905)
    probabilities=pd.Series(np.nan,index=df.index);fold_audit=[]
    for fold,(train,test) in enumerate(folds.split(df,obs,groups=df.person_id)):
        a,b=df.iloc[train],df.iloc[test]
        assert not set(a.person_id)&set(b.person_id)
        z,med=design(a);v,_=design(b,medians=med)
        cols=[k for k in z if k!='intercept' and z[k].nunique()>1]
        scale=StandardScaler().fit(z[cols]);model=LogisticRegression(C=1,max_iter=2000,random_state=20260905)
        model.fit(scale.transform(z[cols]),obs.iloc[train]);probabilities.iloc[test]=model.predict_proba(scale.transform(v[cols]))[:,1]
        fold_audit.append(dict(fold=fold,train=len(train),test=len(test),person_disjoint=True,iterations=int(model.n_iter_.max())))
    assert probabilities.notna().all() and probabilities.between(0,1).all()
    weights=1/probabilities.clip(lower=.02)
    ess={}
    for x in [0,1]:
        w=weights.loc[(obs==1)&(df.exposed==x)];ess[str(x)]=float(w.sum()**2/(w*w).sum())
    bins=pd.qcut(probabilities,10,duplicates='drop')
    cal=pd.DataFrame(dict(predicted=probabilities,observed=obs,bin=bins)).groupby('bin',observed=True).agg(
        n=('observed','size'),predicted=('predicted','mean'),observed=('observed','mean')).reset_index(drop=True).to_dict('records')
    support=float((probabilities<.02).mean())<=.10 and min(ess.values())>=50
    w=weights.loc[obs==1]
    observation_results.append(dict(dataset=d,observed=int(obs.sum()),target=len(df),
        brier=float(brier_score_loss(obs,probabilities)),auc=float(roc_auc_score(obs,probabilities)),
        fraction_target_below_floor=float((probabilities<.02).mean()),ess_by_exposure=ess,
        ess_total=float(w.sum()**2/(w*w).sum()),weight_max=float(w.max()),support_pass=support,
        calibration=cal,folds=fold_audit,
        uncertainty='Robust outcome-model covariance conditional on fitted cross-fold observation weights; does not include weight estimation uncertainty'))
    saved=pd.DataFrame(dict(stay_id=df.stay_id,observation_probability=probabilities,observation_weight=weights))
    con.register('observation_frame',saved)
    con.execute(f'CREATE OR REPLACE TABLE {d}_observation_weights AS SELECT * FROM observation_frame')
    fit_one(d,df,'troponin_rise_12h','observation_weighted',weights=weights)
hj('observation_weighting.json',observation_results)
hj('all_model_results.json',results)
replicated=[]
for endpoint in ['troponin_rise_12h','hospital_mortality']:
    family=[r for r in primary if r['endpoint']==endpoint]
    if all(r.get('estimable') and r.get('interaction_OR',0)>1 and r.get('holm_p',1)<.05 for r in family):
        replicated.append(endpoint)
hj('analysis_manifest.json',dict(protocol_sha256=HB_INTERACTION_PROTOCOL_SHA256,
    numerical_validity_correction_sha256=hashlib.sha256((PROJECT/'docs/SPO2_HEMOGLOBIN_MODEL_VALIDATION_CORRECTION.md').read_bytes()).hexdigest(),
    environment='google_colab' if IN_COLAB else 'local',completed_utc=datetime.now(timezone.utc).isoformat(),
    primary_tests=4,positive_statistically_replicated_endpoints=replicated,
    biological_discovery=False,goal_complete=False,
    model_code_sha256=hashlib.sha256((PROJECT/'scripts/hemoglobin_interaction_cells.py').read_bytes()).hexdigest()))
for name,info in fingerprints.items():
    s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'],name
con.close()
hs('frozen hemoglobin interaction analysis','complete')
"""

VALIDATION = r"""
# Independent matrix reconstruction of the primary patient-cluster covariance.
# This checks the real fitted data, not fabricated patient examples.
checks=[]
for (d,endpoint,variant),(a,z,fit) in fits.items():
    if variant!='primary':continue
    av=z.to_numpy();y=a[endpoint].to_numpy();p=expit(av@np.asarray(fit.params))
    independent=sm.GLM(y,av,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
    score=pd.DataFrame(av*(y-p)[:,None],index=a.index)
    score['person']=a.person_id.astype(str).to_numpy()
    cluster_scores=score.groupby('person').sum().to_numpy()
    bread=np.linalg.inv(av.T@(av*(p*(1-p))[:,None]))
    covariance=bread@(cluster_scores.T@cluster_scores)@bread
    beta_error=float(np.max(np.abs(np.asarray(independent.params)-np.asarray(fit.params))))
    cov_error=float(np.max(np.abs(covariance-np.asarray(fit.cov_params()))))
    assert np.allclose(independent.params,fit.params,atol=1e-5,rtol=1e-5),(d,endpoint,beta_error)
    assert np.allclose(covariance,fit.cov_params(),atol=1e-8,rtol=1e-5),(d,endpoint,cov_error)
    checks.append(dict(dataset=d,endpoint=endpoint,independent_GLM_max_beta_error=beta_error,
        manually_reconstructed_cluster_covariance_max_error=cov_error,passed=True))
hj('independent_model_validation.json',dict(primary_model_checks=checks,
    all_estimable_primary_models_checked=len(checks)==sum(r['estimable'] for r in primary),
    patient_level_outputs_private=True,environment='google_colab' if IN_COLAB else 'local'))
print(json.dumps(checks,indent=2))
"""

REPORT = r"""
primary=json.loads((OUT/'primary_results.json').read_text())
all_results=json.loads((OUT/'all_model_results.json').read_text())
weights=json.loads((OUT/'observation_weighting.json').read_text())
manifest=json.loads((OUT/'analysis_manifest.json').read_text())
lines=['# Prespecified hemoglobin interaction results','',
    '**Actual execution: '+manifest['environment']+'. No established biological discovery or mortality-reduction finding.**','',
    'Only the existing MIMIC/eICU HF cohorts and SpO2 exposure were used. The protocol was frozen before opening Hb-by-outcome associations. Hb came from the pre-ICU CBC selection with the validated linkage-only amendment.','',
    'The interaction is the ratio of SpO2-exposure **odds ratios per 2 g/dL lower hemoglobin**, adjusted for absolute oxygenation, hypoxemic burden, sampling, physiology and documented treatment. It is not a risk ratio or a transfusion effect.','',
    '| Database | Endpoint | N | Events | Interaction OR (95% CI) | Holm p |','|---|---|---:|---:|---|---:|']
for r in primary:
    effect=(f"{r['interaction_OR']:.3f} ({r['interaction_OR_ci95'][0]:.3f}–{r['interaction_OR_ci95'][1]:.3f})" if r['estimable'] else 'Not estimable')
    lines.append(f"| {r['dataset']} | {r['endpoint']} | {r['n']} | {r['events']} | {effect} | {r['holm_p']:.4g} |")
lines+=['','Holm correction covers all four source/endpoint primary tests. Non-estimable models use p=1 only as conservative family placeholders, not as valid hypothesis tests. Both databases were previously explored and are not untouched validation samples.','',
    'An independent optimizer check exposed quasi-complete separation in provisional fits. A documented post-fit numerical correction now tests for likelihood recession directions by linear programming. Such models are non-estimable; no covariate was removed to rescue them. Provisional output is preserved in research/spo2_hemoglobin_interaction/provisional_numerical_run.','']
for r in primary:
    if not r['estimable']:lines.append(f"- {r['dataset']} {r['endpoint']}: {r.get('reason')}")
lines+=['',
    '## Prespecified sensitivities','',
    '| Database | Endpoint | Analysis | N | Interaction OR (95% CI) |','|---|---|---|---:|---|']
for r in all_results:
    if r['variant']=='primary':continue
    effect=(f"{r['interaction_OR']:.3f} ({r['interaction_OR_ci95'][0]:.3f}–{r['interaction_OR_ci95'][1]:.3f})" if r['estimable'] else r.get('reason','Not estimable'))
    lines.append(f"| {r['dataset']} | {r['endpoint']} | {r['variant']} | {r['n']} | {effect} |")
lines+=['','These are sensitivity analyses, not additional confirmatory tests. The weighted analysis assumes troponin ascertainment is explainable by measured covariates. Its reported covariance conditions on estimated weights.','']
for w in weights:
    lines.append(f"- {w['dataset']}: {w['fraction_target_below_floor']:.1%} of observation probabilities below 0.02; weighted ESS {w['ess_total']:.1f}, unexposed/exposed {w['ess_by_exposure']['0']:.1f}/{w['ess_by_exposure']['1']:.1f}; frozen weight-support gate {'passed' if w['support_pass'] else 'failed'}.")
lines+=['','## Absolute mortality contrasts','',
    'These standardize conditional model predictions at Hb 8 and 12 g/dL over the same observed covariate distribution. They are associative contrasts, not the effect of changing Hb.','']
for r in primary:
    if r['endpoint']!='hospital_mortality' or not r['estimable']:continue
    s=r['standardized']
    if not s['support_pass']:
        lines.append(f"- {r['dataset']}: frozen Hb-neighborhood support gate failed; no standardized contrast reported.")
        continue
    z=s['difference_in_differences'];rd=s['contrasts']
    lines.append(f"- {r['dataset']}: exposure risk difference at Hb 8 = {100*rd['RD_hb8']['value']:.2f} percentage points; at Hb 12 = {100*rd['RD_hb12']['value']:.2f}. Difference in differences {100*z['value']:.2f} points (95% CI {100*z['ci95'][0]:.2f} to {100*z['ci95'][1]:.2f}).")
lines+=['','## Decision and limits','']
if manifest['positive_statistically_replicated_endpoints']:
    lines.append('A positive interaction passed the four-test statistical replication screen for: '+', '.join(manifest['positive_statistically_replicated_endpoints'])+'. This remains a candidate requiring the prespecified effect-size and sensitivity assessment; it is not a completed discovery.')
else:
    lines.append('No endpoint passed the prespecified positive, multiplicity-adjusted replication requirement across both databases. This candidate does not establish that low hemoglobin amplifies the existing SpO2 association. Wide intervals remain inconclusive rather than proving the absence of an effect.')
lines+=['',
    'Incomplete eICU pressure/treatment documentation, selective Hb/troponin testing, inferred MIMIC admission links, bleeding/transfusion confounding, and care variables measured during the exposure window limit mechanistic interpretation. Revision/store times do not uniformly identify first clinical availability. The 12-hour troponin endpoint indicates assay-matched rise, not adjudicated MI; hospital mortality does not measure a fixed-time treatment response.','',
    'The broad anemia–hypoxemia hypothesis is prior knowledge: [COVID-19 interaction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447453/) and [MINT HF analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11999761/). Neither supplies patient data to this experiment.','',
    'Every estimable primary GEE model was checked against a separately fitted GLM, and its robust covariance was independently reconstructed from patient-level scores. Aggregate evidence is in research/spo2_hemoglobin_interaction; patient records remain in private Drive Data. The runnable workflow is PhysioGraph_Biological_Discovery.ipynb. The overall biological-discovery goal remains incomplete.','']
report='\n'.join(lines)
(PROJECT/'docs/SPO2_HEMOGLOBIN_INTERACTION_RESULTS.md').write_text(report)
print(report)
"""
