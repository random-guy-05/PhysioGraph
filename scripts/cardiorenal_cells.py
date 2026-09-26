"""Paired biomarker association contrast in the original SpO2 cohorts."""

ANALYSIS = r'''
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_cardiorenal';OUT.mkdir(exist_ok=True)
def cj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def status(stage,state='running'):
    cj('run_status.json',dict(stage=stage,status=state,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert hashlib.sha256((PROJECT/'docs/SPO2_CARDIORENAL_CONTRAST_PLAN.md').read_bytes()).hexdigest()==CARDIORENAL_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==CARDIORENAL_PROTOCOL_SHA256
else:cj('protocol_lock.json',dict(sha256=CARDIORENAL_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_paired_contrast=True,original_marginal_results_already_known=True))
def qs(x):return "'"+str(x).replace("'","''")+"'"
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
fingerprints={}
for name in ['spo2_context','spo2_hemoglobin','hemoglobin_interaction']:
    p=PRIVATE/(name+'.duckdb');s=p.stat()
    fingerprints[name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha_file(p))
cj('input_fingerprints.json',fingerprints)
con=duckdb.connect(str(PRIVATE/'spo2_cardiorenal.duckdb'));con.execute('SET threads=2')
for alias,name in [('prior','spo2_context'),('elig','spo2_hemoglobin'),('care','hemoglobin_interaction')]:
    con.execute(f"ATTACH {qs(PRIVATE/(name+'.duckdb'))} AS {alias} (READ_ONLY)")
status('paired marker construction and frozen covariates')
frames={};coverage=[];joint=[];observation=[];endpoint_validation=[]
for d in ['eicu','mimic']:
    con.execute(f"""CREATE OR REPLACE TABLE {d}_cohort AS
      SELECT e.stay_id,e.exposed,cast(c.person_id AS VARCHAR) person_id,c.age,c.is_male,c.sex_unknown_flag,
        c.followup_end_offset_minutes,f.spo2_mean,f.hr_mean,f.map_mean,
        s.hypoxic_fraction,s.spo2_readings,k.fio2_max,k.vent_documented,k.vasoactive_documented
      FROM elig.{d}_eligible e JOIN prior.{d}_cohort c USING(stay_id)
      JOIN prior.{d}_features f USING(stay_id) JOIN care.{d}_care k USING(stay_id)
      JOIN (SELECT stay_id,avg(CASE WHEN value<90 THEN 1.0 ELSE 0.0 END) hypoxic_fraction,
        sum(n_readings) spo2_readings FROM prior.{d}_bins WHERE concept='spo2' GROUP BY stay_id) s USING(stay_id)
      WHERE c.excluded_before_landmark_flag=0""")
    assert con.execute(f'SELECT count(*) FROM {d}_cohort').fetchone()[0]=={'eicu':11452,'mimic':4711}[d]
    con.execute(f"""CREATE OR REPLACE TABLE {d}_labs AS
      SELECT l.stay_id,l.event_minute,l.concept,l.unit,median(l.value) AS value,
        max(l.available_minute) available_minute
      FROM prior.{d}_context_labs l JOIN {d}_cohort c USING(stay_id)
      WHERE (l.concept IN ('troponin_i','troponin_t') AND l.value BETWEEN 0 AND 1000000)
        OR (l.concept='creatinine' AND l.unit='mg/dl' AND l.value BETWEEN .1 AND 30)
        OR (l.concept='lactate' AND l.unit='mmol/l' AND l.value BETWEEN .1 AND 40)
      GROUP BY l.stay_id,l.event_minute,l.concept,l.unit""")
    assert con.execute(f"SELECT count(*) FROM {d}_labs WHERE concept LIKE 'troponin%' AND unit<>'ng/ml'").fetchone()[0]==0
    con.execute(f"""CREATE OR REPLACE TABLE {d}_lab_covariates AS
      WITH known AS (
        SELECT * FROM {d}_labs WHERE event_minute BETWEEN 0 AND 240 AND available_minute<=240
          AND value>0
      ), selected AS (
        SELECT *,CASE WHEN concept LIKE 'troponin%' THEN 'troponin' ELSE concept END AS marker_family
        FROM known QUALIFY row_number() OVER(PARTITION BY stay_id,
          CASE WHEN concept LIKE 'troponin%' THEN 'troponin' ELSE concept END
          ORDER BY event_minute DESC,available_minute DESC,concept DESC)=1
      ) SELECT stay_id,max(value) FILTER(WHERE marker_family='creatinine') cov_creatinine,
        max(value) FILTER(WHERE marker_family='lactate') cov_lactate,
        max(value) FILTER(WHERE marker_family='troponin') cov_troponin,
        max(concept) FILTER(WHERE marker_family='troponin') cov_troponin_concept
        FROM selected GROUP BY stay_id""")
    for end,label in [(960,'12h'),(1680,'24h')]:
        con.execute(f"""CREATE OR REPLACE TABLE {d}_troponin_{label} AS
          WITH base AS (
            SELECT * FROM {d}_labs WHERE concept LIKE 'troponin%' AND value>0 AND event_minute<=240
            QUALIFY row_number() OVER(PARTITION BY stay_id,concept,unit ORDER BY event_minute DESC)=1
          ), post AS (
            SELECT stay_id,concept,unit,max(value) peak,count(*) post_count,
              min(event_minute) post_first,max(event_minute) post_last
            FROM {d}_labs WHERE concept LIKE 'troponin%' AND event_minute>240 AND event_minute<={end}
            GROUP BY stay_id,concept,unit
          ) SELECT b.stay_id,max(p.peak/b.value) troponin_ratio,count(*) paired_assays,
            sum(p.post_count) troponin_post_count,min(p.post_first) troponin_post_first,
            max(p.post_last) troponin_post_last,max(b.event_minute) troponin_baseline_last
          FROM base b JOIN post p USING(stay_id,concept,unit) GROUP BY b.stay_id""")
        con.execute(f"""CREATE OR REPLACE TABLE {d}_creatinine_{label} AS
          WITH base AS (
            SELECT * FROM {d}_labs WHERE concept='creatinine' AND event_minute<=240
            QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY event_minute DESC)=1
          ), post AS (
            SELECT stay_id,max(value) peak,count(*) creatinine_post_count,
              min(event_minute) creatinine_post_first,max(event_minute) creatinine_post_last
            FROM {d}_labs WHERE concept='creatinine' AND event_minute>240 AND event_minute<={end}
            GROUP BY stay_id
          ) SELECT b.stay_id,b.value creatinine_baseline,p.peak creatinine_peak,
            p.peak/b.value creatinine_ratio,p.peak-b.value creatinine_delta,
            b.event_minute creatinine_baseline_last,p.creatinine_post_count,
            p.creatinine_post_first,p.creatinine_post_last
          FROM base b JOIN post p USING(stay_id)""")
        con.execute(f"""CREATE OR REPLACE TABLE {d}_analysis_{label} AS
          SELECT c.*,t.* EXCLUDE(stay_id),k.* EXCLUDE(stay_id),v.* EXCLUDE(stay_id),
            cast(t.troponin_ratio>=1.5 AS INTEGER) T,
            cast(CAST(k.creatinine_peak AS DECIMAL(24,10))>=1.5*CAST(k.creatinine_baseline AS DECIMAL(24,10))
              OR CAST(k.creatinine_peak AS DECIMAL(24,10))-CAST(k.creatinine_baseline AS DECIMAL(24,10))>=.3 AS INTEGER) K
          FROM {d}_cohort c JOIN {d}_troponin_{label} t USING(stay_id)
          JOIN {d}_creatinine_{label} k USING(stay_id)
          LEFT JOIN {d}_lab_covariates v USING(stay_id)
          WHERE c.followup_end_offset_minutes>={end}""")
        frame=con.execute(f'SELECT *,T-K AS D FROM {d}_analysis_{label} ORDER BY stay_id').df()
        assert frame.stay_id.is_unique and frame.person_id.notna().all()
        check=con.execute(f"""SELECT a.stay_id,a.T,e.troponin_rise_{label} existing,
          a.troponin_ratio,e.troponin_ratio_{label} existing_ratio
          FROM {d}_analysis_{label} a JOIN prior.{d}_context_endpoints e USING(stay_id)""").df()
        assert check.existing.notna().all() and np.array_equal(check['T'],check.existing)
        assert np.allclose(check.troponin_ratio,check.existing_ratio,rtol=1e-12,atol=1e-12)
        for column in ['creatinine_peak','creatinine_baseline']:
            assert np.max(abs(frame[column]-frame[column].round(10)).to_numpy(),initial=0)<1e-12
        decimal_labels=[]
        for peak,base in zip(frame.creatinine_peak,frame.creatinine_baseline):
            peak=Decimal(str(round(peak,10)));base=Decimal(str(round(base,10)))
            decimal_labels.append(int(peak-base>=Decimal('0.3') or peak>=Decimal('1.5')*base))
        assert np.array_equal(frame['K'],decimal_labels)
        for marker in ['troponin','creatinine']:
            assert frame[marker+'_baseline_last'].between(0,240).all()
            assert (frame[marker+'_post_first']>240).all() and (frame[marker+'_post_last']<=end).all()
        endpoint_validation.append(dict(dataset=d,horizon=label,paired_encounters=len(check),existing_troponin_labels_match=True,
            independent_decimal_creatinine_labels_match=True,window_checks_pass=True))
        frames[(d,label)]=frame
        coverage.append(dict(dataset=d,horizon=label,shared_pair_encounters=len(frame),unique_patients=frame.person_id.nunique(),
            exposure_arm_patients={str(x):frame.loc[frame.exposed.eq(x),'person_id'].nunique() for x in [0,1]},
            multiple_paired_assay_encounters=int(frame.paired_assays.gt(1).sum())))
        for x,g in frame.groupby('exposed'):
            for t in [0,1]:
                for k in [0,1]:joint.append(dict(dataset=d,horizon=label,exposed=int(x),troponin_rise=t,creatinine_worsening=k,encounters=int((g['T'].eq(t)&g['K'].eq(k)).sum())))
            observation.append(dict(dataset=d,horizon=label,exposed=int(x),encounters=len(g),
                median_troponin_post_tests=float(g.troponin_post_count.median()),median_creatinine_post_tests=float(g.creatinine_post_count.median())))
con.close()
cj('coverage.json',coverage);cj('joint_distribution.json',joint);cj('observation_counts.json',observation)
cj('endpoint_validation.json',endpoint_validation)
status('paired contrast models with patient-clustered covariance')
def design(df):
    z=pd.DataFrame({'intercept':np.ones(len(df)),'X':df.exposed.to_numpy(dtype=float)},index=df.index)
    continuous=dict(age=df.age,mean_spo2=df.spo2_mean,hypoxic_fraction=df.hypoxic_fraction,
        log_readings=np.log1p(df.spo2_readings),mean_hr=df.hr_mean,mean_map=df.map_mean,
        log_creatinine=np.log(df.cov_creatinine),log_lactate=np.log(df.cov_lactate),
        log_troponin=np.log(df.cov_troponin),fio2=df.fio2_max)
    omitted=[];imputation={}
    for name,s in continuous.items():
        s=s.replace([np.inf,-np.inf],np.nan)
        if not s.notna().any():omitted.append(name+': all missing');continue
        median=float(s.median());filled=s.fillna(median);sd=float(filled.std(ddof=1))
        imputation[name]=dict(median=median,sd=sd,missing=int(s.isna().sum()))
        if sd>0:z[name]=(filled-median)/sd
        else:omitted.append(name+': constant')
        if s.isna().any():z[name+'_missing']=s.isna().astype(float)
    z['male']=df.is_male.fillna(0).astype(float);z['sex_unknown']=df.sex_unknown_flag.fillna(1).astype(float)
    subtype=df.cov_troponin_concept.map({'troponin_t':1.,'troponin_i':0.})
    subtype_mode=float(subtype.mode().iloc[0]) if subtype.notna().any() else 0.
    z['troponin_t']=subtype.fillna(subtype_mode)
    imputation['troponin_t']=dict(mode=subtype_mode,missing=int(subtype.isna().sum()),
        shared_missingness_indicator='log_troponin_missing')
    z['vent_documented']=df.vent_documented.astype(float);z['vasoactive_documented']=df.vasoactive_documented.astype(float)
    for name in list(z.columns):
        if name not in ['intercept','X'] and z[name].nunique()==1:omitted.append(name+': constant');z=z.drop(columns=name)
    return z,omitted,imputation
models=[];covariance_checks=[];design_details=[]
for (d,label),df in frames.items():
    z,omitted,imputation=design(df)
    row=dict(dataset=d,horizon=label,n=len(df),patients=df.person_id.nunique(),parameters=len(z.columns),
        status='pending',crude_differential_risk_difference=None,adjusted_differential_risk_difference=None,
        ci_lower=None,ci_upper=None,p_value=1.,holm_primary_p=None)
    if all(df.exposed.eq(x).any() for x in [0,1]):
        row['crude_differential_risk_difference']=float(df.loc[df.exposed.eq(1),'D'].mean()-df.loc[df.exposed.eq(0),'D'].mean())
        direct=(df.loc[df.exposed.eq(1),'T'].mean()-df.loc[df.exposed.eq(0),'T'].mean())-(df.loc[df.exposed.eq(1),'K'].mean()-df.loc[df.exposed.eq(0),'K'].mean())
        assert abs(direct-row['crude_differential_risk_difference'])<1e-12
    design_details.append(dict(dataset=d,horizon=label,columns=list(z.columns),omitted=omitted,imputation=imputation))
    if any(df.loc[df.exposed.eq(x),'person_id'].nunique()<50 for x in [0,1]):row['status']='insufficient_patient_groups'
    elif df.D.nunique()<2:row['status']='constant_paired_contrast'
    elif not np.isfinite(z.to_numpy()).all() or np.linalg.matrix_rank(z.to_numpy())<z.shape[1]:row['status']='invalid_design'
    else:
        fit=sm.OLS(df.D.astype(float),z).fit(cov_type='cluster',cov_kwds={'groups':df.person_id,'use_correction':True},use_t=True)
        cov=np.asarray(fit.cov_params())
        if not np.isfinite(cov).all() or np.any(np.diag(cov)<0):row['status']='invalid_covariance'
        else:
            x=z.to_numpy();y=df.D.to_numpy(dtype=float);coefficient=np.linalg.lstsq(x,y,rcond=None)[0]
            residual=y-x@coefficient;scores=pd.DataFrame(x*residual[:,None]).groupby(df.person_id.to_numpy()).sum().to_numpy()
            n,k=x.shape;g=len(scores);bread=np.linalg.inv(x.T@x)
            independent=bread@(scores.T@scores)@bread*(g/(g-1))*((n-1)/(n-k))
            coefficient_error=float(np.max(abs(coefficient-np.asarray(fit.params))));cov_error=float(np.max(abs(independent-cov)))
            assert coefficient_error<1e-8 and cov_error<1e-8
            bounds=fit.conf_int().loc['X']
            row.update(status='estimable',adjusted_differential_risk_difference=float(fit.params['X']),
                ci_lower=float(bounds.iloc[0]),ci_upper=float(bounds.iloc[1]),p_value=float(fit.pvalues['X']))
            covariance_checks.append(dict(dataset=d,horizon=label,independent_coefficient_max_error=coefficient_error,
                independent_cluster_covariance_max_error=cov_error,condition_number=float(np.linalg.cond(x))))
    models.append(row)
primary=[r for r in models if r['horizon']=='12h'];assert len(primary)==2
adjusted=multipletests([r['p_value'] for r in primary],method='holm')[1]
for r,p in zip(primary,adjusted):r['holm_primary_p']=float(p)
advance=all(r['status']=='estimable' and r['ci_lower']>0 and r['holm_primary_p']<.05 and r['adjusted_differential_risk_difference']>=.05 for r in primary)
cj('model_results.json',models);cj('design_details.json',design_details);cj('independent_model_validation.json',covariance_checks)
for name,info in fingerprints.items():
    s=(PRIVATE/(name+'.duckdb')).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns']
cj('analysis_manifest.json',dict(protocol_sha256=CARDIORENAL_PROTOCOL_SHA256,
    assay_encoding_clarification_sha256=sha_file(PROJECT/'docs/SPO2_CARDIORENAL_ASSAY_ENCODING_CLARIFICATION.md'),environment='google_colab' if IN_COLAB else 'local',
    decimal_threshold_correction_sha256=sha_file(PROJECT/'docs/SPO2_CARDIORENAL_DECIMAL_THRESHOLD_CORRECTION.md'),
    completed_utc=datetime.now(timezone.utc).isoformat(),primary_tests=2,primary_candidate_advance=advance,
    mortality_selected=False,troponin_reference_limit_required=False,causal_effect_estimated=False,biological_discovery=False,goal_complete=False))
def pp(v):return '—' if v is None else f'{100*v:.2f}'
report=['# Paired cardiorenal biomarker contrast','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Exploratory adjusted association, not a causal or diagnostic specificity estimate.**','',
    '| Source | Horizon | Paired encounters | Patients | Status | Adjusted differential association (pp) | 95% CI (pp) | Primary Holm p |',
    '|---|---|---:|---:|---|---:|---|---:|']
for r in models:report.append(f"| {r['dataset']} | {r['horizon']} | {r['n']} | {r['patients']} | {r['status']} | {pp(r['adjusted_differential_risk_difference'])} | {pp(r['ci_lower'])} to {pp(r['ci_upper'])} | {'—' if r['holm_primary_p'] is None else format(r['holm_primary_p'],'.4g')} |")
report+=['',f'Primary candidate advancement: **{advance}**. A positive coefficient means the instability-associated increase in the troponin-rise proxy is larger than the increase in creatinine worsening in this shared sample. The 24-hour analysis cannot rescue a failed primary result.','',
    'Both biomarkers require actual baseline/post pairs and complete follow-up through the stated endpoint. This selects patients remaining observed through 16 or 28 ICU hours and does not represent early deaths/discharges. The analysis controls for the frozen pre-landmark covariates, with patient-clustered uncertainty. Test ordering and non-random joint measurement remain limitations.','',
    'A 1.5-fold troponin increase is not necessarily above the assay reference limit and is not adjudicated myocardial injury or infarction. Creatinine may lag acute renal dysfunction. Different marker thresholds and kinetics prevent a positive differential association from proving organ-specific tissue injury or excluding clearance effects. No mortality outcome, intervention effect, or novel biological discovery is established here.','',
    'Final creatinine labels use decimal arithmetic at the unchanged 0.3 mg/dL and 1.5-fold boundaries. A post-estimate numerical correction and the archived preliminary run are documented in SPO2_CARDIORENAL_DECIMAL_THRESHOLD_CORRECTION.md.','']
(PROJECT/'docs/SPO2_CARDIORENAL_CONTRAST_RESULTS.md').write_text('\n'.join(report))
status('paired cardiorenal contrast and independent validation','complete');print('\n'.join(report))
'''
