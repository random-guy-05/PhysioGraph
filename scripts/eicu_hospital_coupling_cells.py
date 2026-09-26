"""Explicit secondary eICU hospital split after failed cross-database support."""

ANALYSIS = r'''
import ast,hashlib,importlib.util,json,runpy
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from statsmodels.regression.linear_model import OLS
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_eicu_hospital_coupling';OUT.mkdir(parents=True,exist_ok=True)
def hj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def hh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def hq(value):return "'"+str(value).replace("'","''")+"'"
def hs(stage,status='running'):
    hj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert hh(PROJECT/'docs/SPO2_EICU_HOSPITAL_AMENDMENT.md')==EICU_HOSPITAL_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==EICU_HOSPITAL_PROTOCOL_SHA256
else:hj('protocol_lock.json',dict(sha256=EICU_HOSPITAL_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_hospital_counts_and_troponin_changes=True,
  failed_cross_database_support_known=True,prior_experiments_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'eicu_hospital_coupling.duckdb'));con.execute('SET threads=2')
input_path=PRIVATE/'within_person_troponin.duckdb';patient_path=DRIVE/'Data/eICU/Full/patient.csv'
refs={str(p):dict(sha256=hh(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in [input_path,patient_path]}
manifest=dict(protocol_sha256=EICU_HOSPITAL_PROTOCOL_SHA256,references=refs)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
hj('input_manifest.json',manifest);hs('fixed hospital split before outcome changes')
con.execute(f'ATTACH {hq(input_path)} AS original (READ_ONLY)')
a=con.execute('SELECT * FROM original.eicu_analysis_intervals ORDER BY person_id,event_minute').df()
assert a.groupby('person_id').stay_id.nunique().eq(1).all()
mapping=con.execute(f"""SELECT try_cast(p.patientunitstayid AS BIGINT) stay_id,try_cast(p.hospitalid AS BIGINT) hospital_id
  FROM read_csv({hq(patient_path)},header=true,all_varchar=true,strict_mode=true) p
  JOIN (SELECT DISTINCT stay_id FROM original.eicu_analysis_intervals) s ON try_cast(p.patientunitstayid AS BIGINT)=s.stay_id
  ORDER BY stay_id""").df()
assert mapping.stay_id.is_unique
p=pd.read_csv(patient_path,usecols=['patientunitstayid','hospitalid'])
p=p.rename(columns={'patientunitstayid':'stay_id','hospitalid':'hospital_id'})
p=p.loc[p.stay_id.isin(a.stay_id)].sort_values('stay_id').reset_index(drop=True)
for frame in (p,mapping):frame['hospital_id']=frame.hospital_id.astype('Int64')
pd.testing.assert_frame_equal(mapping,p[mapping.columns],check_dtype=False)
assert set(mapping.stay_id)==set(a.stay_id)
merged=a.merge(mapping,on='stay_id',validate='many_to_one')
missing=dict(intervals=int(merged.hospital_id.isna().sum()),people=int(merged.loc[merged.hospital_id.isna(),'person_id'].nunique()))
merged=merged.loc[merged.hospital_id.notna()].copy();merged['hospital_id']=merged.hospital_id.astype(int)
def hospital_split(h):return hashlib.sha256(('PhysioGraph within-person eICU hospital v1|'+str(int(h))).encode()).digest()[0]%2
merged['split']=merged.hospital_id.map(hospital_split)
assert merged.groupby('person_id').split.nunique().eq(1).all()
assert merged.groupby('hospital_id').split.nunique().eq(1).all()
con.register('assignment',merged[['stay_id','person_id','hospital_id','split']].drop_duplicates())
con.execute('CREATE OR REPLACE TABLE hospital_assignment AS SELECT * FROM assignment');con.unregister('assignment')
sql_assignment=con.execute("""SELECT hospital_id,split,
  ('0x'||substr(sha256('PhysioGraph within-person eICU hospital v1|'||cast(hospital_id AS VARCHAR)),1,2))::INTEGER%2 oracle_split
  FROM hospital_assignment""").df()
assert np.array_equal(sql_assignment.split,sql_assignment.oracle_split)
support=[]
for split in [0,1]:
    s=merged.loc[merged.split.eq(split)];g=s.person_id.nunique();h=s.hospital_id.nunique();switches=s.groupby('person_id').unstable.nunique().eq(2).sum()
    support.append(dict(split=split,role='discovery' if split==0 else 'validation',people=int(g),hospitals=int(h),intervals=len(s),switchers=int(switches),
      minimum_people=100,minimum_hospitals=20,minimum_switchers=50,passes=g>=100 and h>=20 and switches>=50))
hj('hospital_support.json',support);hj('missing_hospital.json',missing)
hj('assignment_validation.json',dict(sql_python_patient_hospital_join_matches=True,independent_sql_sha_assignment_matches=True,patients_and_hospitals_disjoint=True))
support_pass=all(r['passes'] for r in support);models=[];checks=[];diagnostics=[]
def hospital_fit(x,y,people,hospitals):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    pu,pi=np.unique(people,return_inverse=True);hu,hi=np.unique(hospitals,return_inverse=True)
    n,k=x.shape;g=len(pu);h=len(hu);sizes=np.bincount(pi)
    sx=np.zeros((g,k));sy=np.zeros(g);np.add.at(sx,pi,x);np.add.at(sy,pi,y)
    xd=x-sx[pi]/sizes[pi,None];yd=y-sy[pi]/sizes[pi]
    fit=OLS(yd,xd).fit(cov_type='cluster',cov_kwds={'groups':hospitals,'use_correction':True},use_t=True)
    adjusted_cov=np.asarray(fit.cov_params())*((n-k)/(n-g-k))
    gram=x.T@x-(sx.T/sizes)@sx;cross=x.T@y-(sx.T/sizes)@sy
    beta=np.linalg.solve(gram,cross);bread=np.linalg.inv(gram)
    effect=(sy-sx@beta)/sizes;residual=y-x@beta-effect[pi]
    scores=np.zeros((h,k));np.add.at(scores,hi,x*residual[:,None])
    covariance=bread@(scores.T@scores)@bread*(h/(h-1))*((n-1)/(n-g-k))
    np.testing.assert_allclose(beta,fit.params,rtol=1e-6,atol=1e-8);np.testing.assert_allclose(covariance,adjusted_cov,rtol=1e-5,atol=1e-8)
    assert np.isfinite(covariance).all() and np.all(np.diag(covariance)>=0)
    check=dict(people=g,hospitals=h,intervals=n,slopes=k,residual_df=n-g-k,max_coefficient_error=float(np.max(np.abs(beta-fit.params))),max_covariance_error=float(np.max(np.abs(covariance-adjusted_cov))))
    return beta,covariance,check
if support_pass:
    rng=np.random.default_rng(241);sp=np.repeat(np.arange(25),np.tile([2,3,4,5,6],5));sh=sp//5;sn=len(sp)
    sx=rng.normal(size=(sn,3));sy=sx@np.array([.2,-.3,.1])+sp/10+rng.normal(size=sn)
    sb,sc,sv=hospital_fit(sx,sy,sp,sh)
    dummy=pd.get_dummies(sp,dtype=float).to_numpy();full=OLS(sy,np.column_stack([dummy,sx])).fit(cov_type='cluster',cov_kwds={'groups':sh,'use_correction':True},use_t=True)
    np.testing.assert_allclose(sb,full.params[-3:],rtol=1e-9,atol=1e-10);np.testing.assert_allclose(sc,np.asarray(full.cov_params())[-3:,-3:],rtol=1e-9,atol=1e-10)
    hj('synthetic_nested_validation.json',dict(full_dummy_nested_hospital_covariance_matches=True,clinical_records=False,**sv))
    for split in [0,1]:
        hs(('discovery' if split==0 else 'validation')+' frozen hospital model')
        s=merged.loc[merged.split.eq(split)].reset_index(drop=True)
        x=pd.DataFrame({name:s[col].to_numpy(dtype=float) for name,col in [('instability','unstable'),('spo2_mean','spo2_mean'),('hypoxic_fraction','hypoxic_fraction'),('map_mean','map_mean'),('hr_mean','hr_mean')]})
        x['log_readings']=np.log1p(s.spo2_readings.to_numpy());x['interval_hours']=(s.event_minute-s.prior_event_minute).to_numpy()/60
        x['end_hours']=s.event_minute.to_numpy()/60;x['end_hours_squared']=x.end_hours**2;x['log_baseline']=np.log(s.prior_magnitude.astype(float).to_numpy())
        imputation={}
        for c in ['map_mean','hr_mean']:
            mask=x[c].isna()
            if mask.any():
                median=float(x.loc[~mask,c].median()) if (~mask).any() else 0.
                x[c+'_missing']=mask.astype(float);x[c]=x[c].fillna(median);imputation[c]=dict(median=median,missing=int(mask.sum()))
        xd=x-x.groupby(s.person_id.to_numpy()).transform('mean')
        drop=[c for c in x if c!='instability' and np.max(np.abs(xd[c]))<=1e-12];x=x.drop(columns=drop);xd=xd.drop(columns=drop)
        n,k=x.shape;g=s.person_id.nunique();h=s.hospital_id.nunique();rank=int(np.linalg.matrix_rank(xd))
        info=dict(split=split,role='discovery' if split==0 else 'validation',model_fitted=False,people=int(g),hospitals=int(h),intervals=n,
          columns=list(x),imputation=imputation,removed_no_within_variation=drop,rank=rank,gate_failures=[])
        if not np.isfinite(x).all().all() or rank<k:info['gate_failures'].append('nonfinite or rank-deficient within-person design')
        if n<10*k or n-g-k<=0:info['gate_failures'].append('insufficient observations or residual degrees of freedom')
        if not info['gate_failures']:
            y=np.array([np.log(float(Decimal(v)/Decimal(b))) for v,b in zip(s.magnitude,s.prior_magnitude)]);assert np.isfinite(y).all()
            for variant,design in [('primary',x),('omit_lag',x.drop(columns=['log_baseline'],errors='ignore'))]:
                beta,cov,check=hospital_fit(design.to_numpy(),y,s.person_id.to_numpy(),s.hospital_id.to_numpy())
                j=list(design).index('instability');b=float(beta[j]);se=float(np.sqrt(cov[j,j]));crit=float(student_t.ppf(.975,h-1))
                result=dict(split=split,role=info['role'],variant=variant,model_fitted=True,log_coefficient=b,log_ci95=[b-crit*se,b+crit*se],
                  fold_change_ratio=float(np.exp(b)),ratio_ci95=[float(np.exp(b-crit*se)),float(np.exp(b+crit*se))],p=float(2*student_t.sf(abs(b/se),h-1)),hospital_df=int(h-1))
                checks.append(dict(split=split,variant=variant,**check))
                if variant=='primary':info.update(result)
                else:diagnostics.append(dict(**result,same_primary_rows_and_outcomes=True,primary_log_coefficient=info['log_coefficient']))
            s['log_troponin_change']=y;s['row_index']=np.arange(len(s))
            con.register('modeled_intervals',s);con.execute(f'CREATE OR REPLACE TABLE split_{split}_modeled AS SELECT * FROM modeled_intervals');con.unregister('modeled_intervals')
            design=x.assign(row_index=np.arange(len(x)))
            con.register('design',design);con.execute(f'CREATE OR REPLACE TABLE split_{split}_design AS SELECT * FROM design');con.unregister('design')
        models.append(info)
    for family in [models,diagnostics]:
        last=0
        for i,m in enumerate(sorted(family,key=lambda m:m.get('p',1))):last=max(last,min(1,(2-i)*m.get('p',1)));m['holm_p']=last
for path,ref in refs.items():
    p=Path(path);s=p.stat();assert s.st_size==ref['bytes'] and s.st_mtime_ns==ref['mtime_ns'] and hh(p)==ref['sha256']
hj('primary_models.json',models);hj('lag_diagnostic_models.json',diagnostics);hj('model_validation.json',checks)
replicated=len(models)==2 and all(m['model_fitted'] and m['log_coefficient']>0 and m['holm_p']<.05 for m in models)
hj('manifest.json',dict(protocol_sha256=EICU_HOSPITAL_PROTOCOL_SHA256,script_sha256=hh(PROJECT/'scripts/eicu_hospital_coupling_cells.py'),support_pass=support_pass,
  models_fitted=sum(m['model_fitted'] for m in models),positive_internal_validation=replicated,mimic_replication=False,mortality_accessed=False,causal_effect_identified=False,
  environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# Secondary eICU validation across disjoint hospitals','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. The failed MIMIC temporal-support experiment is preserved; this is not cross-database replication.',
  '', '| Split | Hospitals | Patients | Intervals | Instability switchers | Support gate |','|---|---:|---:|---:|---:|---|']
for r in support:report.append(f"| {r['role']} | {r['hospitals']} | {r['people']} | {r['intervals']} | {r['switchers']} | {'pass' if r['passes'] else 'fail'} |")
report+=['', '| Model | Split | Troponin fold-change ratio | 95% CI | Holm p |','|---|---|---:|---|---:|']
for m in models+diagnostics:
    if m['model_fitted']:report.append(f"| {m['variant']} | {m['role']} | {m['fold_change_ratio']:.4f} | {m['ratio_ci95'][0]:.4f}–{m['ratio_ci95'][1]:.4f} | {m['holm_p']:.6g} |")
    else:report.append(f"| primary | {m['role']} | Not fitted | {'; '.join(m['gate_failures'])} | — |")
if not support_pass:report+=['','A fixed split-support gate failed; no troponin changes or association models were calculated.']
elif replicated:report+=['','The prespecified positive internal-validation criterion is met. This is a secondary observational signal requiring further falsification, not an established biological mechanism.']
else:report+=['','The prespecified positive internal-validation criterion is not met. A favorable discovery estimate or sensitivity cannot rescue failed primary validation. This is not equivalence or proof of no relationship.']
report+=['','The ratio compares modeled troponin fold changes during unstable versus stable windows within patients; it is not a mortality risk ratio or an infarction diagnosis. Short-panel dynamic bias, treatment, renal clearance, selective testing and delayed release remain unresolved. Omitting the lag is a diagnostic, not a proven bias correction.',
  '', 'Original patient-to-hospital linkage and SHA256 assignment match independent implementations. Numerical validation uses uncentered sufficient statistics and a synthetic full-dummy model with patients nested in hospitals. Hospital-clustered uncertainty accounts for absorbed patient degrees of freedom. No mortality endpoint was accessed. No mortality benefit or paradigm-shifting biological finding is established. The goal remains unmet.']
for m in models:
    missing=m.get('imputation',{}).get('map_mean',{}).get('missing',0)
    report.append(f"\nHemodynamic limitation ({m['role']}): MAP was missing in {missing}/{m['intervals']} windows ({100*missing/m['intervals']:.1f}%) and median-imputed with a missingness indicator. This does not establish independence from changing blood pressure or perfusion.")
(PROJECT/'docs/SPO2_EICU_HOSPITAL_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');hs('secondary hospital experiment completed','complete');print('\n'.join(report));con.close()
'''
