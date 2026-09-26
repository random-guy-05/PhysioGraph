"""Paired sampling-opportunity falsification in the existing HF cohorts."""

ANALYSIS = r'''
import ast,json,hashlib,importlib.util,runpy
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import numpy as np
import pandas as pd
import duckdb
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_troponin_opportunity';OUT.mkdir(parents=True,exist_ok=True)
def oj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def oh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def oq(value):return "'"+str(value).replace("'","''")+"'"
def os_status(stage,state='running'):
    oj('run_status.json',dict(stage=stage,status=state,utc=datetime.now(timezone.utc).isoformat(),
      environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert oh(PROJECT/'docs/SPO2_TROPONIN_OPPORTUNITY_PLAN.md')==TROPONIN_OPPORTUNITY_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_OPPORTUNITY_PROTOCOL_SHA256
else:oj('protocol_lock.json',dict(sha256=TROPONIN_OPPORTUNITY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_contrast=True,earlier_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(':memory:');con.execute('SET threads=2')
inputs={}
for alias,name in [('care','spo2_cardiorenal'),('prior','spo2_context')]:
    p=PRIVATE/(name+'.duckdb');s=p.stat();inputs[name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=oh(p))
    con.execute(f'ATTACH {oq(p)} AS {alias} (READ_ONLY)')
# Reuse only the frozen design function, without executing its original analysis.
support_script=PROJECT/'scripts/cardiorenal_cells.py'
tree=ast.parse(runpy.run_path(str(support_script))['ANALYSIS'])
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='design')
design_namespace={'pd':pd,'np':np}
exec(compile(ast.Module(body=[function],type_ignores=[]),'reused-cardiorenal-design','exec'),design_namespace)
design=design_namespace['design'];design_hash=oh(support_script)
os_status('reconstructing troponin sampling opportunities')
counts=[];audits=[];models=[];checks=[];designs=[]
for d in ['eicu','mimic']:
    cohort=con.execute(f'SELECT c.*,v.* EXCLUDE(stay_id) FROM care.{d}_cohort c LEFT JOIN care.{d}_lab_covariates v USING(stay_id)').df()
    labs=con.execute(f"SELECT * FROM care.{d}_labs WHERE concept IN ('troponin_i','troponin_t') ORDER BY stay_id,concept,unit,event_minute").df()
    assert not labs.duplicated(['stay_id','concept','unit','event_minute']).any()
    keys=['stay_id','concept','unit']
    for end,label in [(960,'12h'),(1680,'24h')]:
        sql=con.execute(f"""WITH base AS (
          SELECT * FROM care.{d}_labs WHERE concept LIKE 'troponin%' AND value>0 AND event_minute BETWEEN 0 AND 240
          QUALIFY row_number() OVER(PARTITION BY stay_id,concept,unit ORDER BY event_minute DESC)=1)
          SELECT p.stay_id,p.concept,p.unit,p.event_minute,p.value,b.value base_value,
            b.event_minute base_time,b.available_minute base_available
          FROM care.{d}_labs p JOIN base b USING(stay_id,concept,unit)
          JOIN care.{d}_cohort c USING(stay_id)
          WHERE p.event_minute>240 AND p.event_minute<={end} AND c.followup_end_offset_minutes>={end}
          ORDER BY p.stay_id,p.concept,p.unit,p.event_minute""").df()
        base=labs.loc[labs.value.gt(0)&labs.event_minute.between(0,240)].drop_duplicates(keys,keep='last')
        base=base[keys+['value','event_minute','available_minute']].rename(columns={'value':'base_value','event_minute':'base_time','available_minute':'base_available'})
        python=labs.loc[labs.event_minute.gt(240)&labs.event_minute.le(end),keys+['event_minute','value']].merge(base,on=keys,validate='many_to_one')
        python=python.loc[python.stay_id.isin(cohort.loc[cohort.followup_end_offset_minutes.ge(end),'stay_id'])]
        python=python.sort_values(keys+['event_minute']).reset_index(drop=True)
        pd.testing.assert_frame_equal(sql,python[sql.columns],check_dtype=False)
        sql['native_ratio']=sql.value/sql.base_value
        sql['rise']=[int(Decimal(str(v))>=Decimal('1.5')*Decimal(str(b))) for v,b in zip(sql.value,sql.base_value)]
        con.register('opportunity_samples',sql)
        con.execute("""CREATE OR REPLACE TEMP TABLE opportunity_times AS
          SELECT stay_id,event_minute,max(rise) rise FROM opportunity_samples GROUP BY stay_id,event_minute""")
        f=con.execute("""SELECT stay_id,count(*) n_samples,sum(rise) n_rises,max(rise) maximum,
          avg(rise) single_expectation,arg_min(rise,event_minute) first_sample,
          min(event_minute) first_time,max(event_minute) last_time
          FROM opportunity_times GROUP BY stay_id ORDER BY stay_id""").df()
        # Separate reconstruction of timestamp-level opportunities and all outcomes.
        pytime=sql.groupby(['stay_id','event_minute'],sort=True).rise.max().reset_index()
        pyrows=[]
        for sid,g in pytime.groupby('stay_id',sort=True):
            values=g.rise.to_numpy()
            pyrows.append(dict(stay_id=sid,n_samples=len(g),n_rises=int(values.sum()),maximum=int(values.max()),
              single_expectation=float(values.mean()),first_sample=int(values[0]),first_time=g.event_minute.iloc[0],last_time=g.event_minute.iloc[-1]))
        pd.testing.assert_frame_equal(f,pd.DataFrame(pyrows)[f.columns],check_dtype=False,atol=1e-12,rtol=0)
        legacy=con.execute(f"""SELECT t.stay_id,t.troponin_ratio,e.troponin_rise_{label} AS legacy_label
          FROM care.{d}_troponin_{label} t JOIN care.{d}_cohort c USING(stay_id)
          JOIN prior.{d}_context_endpoints e USING(stay_id) WHERE c.followup_end_offset_minutes>={end}
          ORDER BY t.stay_id""").df()
        native=sql.groupby('stay_id',sort=True).native_ratio.max().reset_index()
        assert np.array_equal(legacy.stay_id,native.stay_id) and np.array_equal(f.stay_id,legacy.stay_id)
        assert np.allclose(legacy.troponin_ratio,native.native_ratio,rtol=1e-12,atol=1e-12)
        assert np.array_equal(legacy.legacy_label,native.native_ratio.ge(1.5).astype(int))
        unavailable=sql.assign(unavailable=sql.base_available.isna()|sql.base_available.gt(240)).groupby('stay_id').unavailable.max()
        f['baseline_unavailable']=f.stay_id.map(unavailable)
        f['D']=f.maximum-f.single_expectation
        f=f.merge(cohort,on='stay_id',validate='one_to_one')
        assert f.D.between(0,1).all() and (f.maximum>=f.first_sample).all()
        assert (f.loc[f.n_samples.eq(1),'D']==0).all()
        for arm,g in f.groupby('exposed'):
            counts.append(dict(dataset=d,horizon=label,exposed=int(arm),encounters=len(g),people=int(g.person_id.nunique()),
              mean_samples=float(g.n_samples.mean()),median_samples=float(g.n_samples.median()),repeated_sampling_fraction=float(g.n_samples.gt(1).mean()),
              maximum_mean=float(g.maximum.mean()),single_expectation_mean=float(g.single_expectation.mean()),first_sample_mean=float(g.first_sample.mean()),
              mean_first_sample_minute=float(g.first_time.mean()),baseline_unavailable=int(g.baseline_unavailable.sum())))
        audits.append(dict(dataset=d,horizon=label,assay_timestamp_samples=len(sql),distinct_timestamp_samples=int(f.n_samples.sum()),
          native_labels_and_ratios_match=True,decimal_legacy_label_differences=int(np.sum(f.maximum.to_numpy()!=legacy.legacy_label.to_numpy())),
          independent_sample_join_and_timestamp_outcomes_match=True))
        z,omitted,imputation=design(f)
        designs.append(dict(dataset=d,horizon=label,columns=list(z.columns),omitted=omitted,imputation=imputation))
        result=dict(dataset=d,horizon=label,n=len(f),people=int(f.person_id.nunique()),parameters=len(z.columns),status='pending',
          primary_contrast=None,ci_lower=None,ci_upper=None,p_value=1.,holm_p=None)
        if any(f.loc[f.exposed.eq(a),'person_id'].nunique()<50 for a in [0,1]):result['status']='insufficient_people'
        elif f.D.nunique()<2:result['status']='constant_contrast'
        elif not np.isfinite(z.to_numpy()).all() or np.linalg.matrix_rank(z)<z.shape[1]:result['status']='invalid_design'
        else:
            coefficients={}
            for outcome in ['maximum','single_expectation','first_sample','D']:
                fit=OLS(f[outcome].astype(float),z).fit(cov_type='cluster',cov_kwds={'groups':f.person_id,'use_correction':True},use_t=True)
                coefficients[outcome]=float(fit.params['X'])
                if outcome=='D':
                    x=z.to_numpy();y=f.D.to_numpy(dtype=float);beta=np.linalg.lstsq(x,y,rcond=None)[0]
                    residual=y-x@beta;scores=pd.DataFrame(x*residual[:,None]).groupby(f.person_id.to_numpy()).sum().to_numpy()
                    n,k=x.shape;groups=len(scores);bread=np.linalg.inv(x.T@x)
                    cov=bread@(scores.T@scores)@bread*(groups/(groups-1))*((n-1)/(n-k))
                    assert np.isfinite(cov).all() and np.all(np.diag(cov)>=0)
                    be=float(np.max(abs(beta-fit.params.to_numpy())));ce=float(np.max(abs(cov-np.asarray(fit.cov_params()))))
                    assert be<1e-8 and ce<1e-8
                    bounds=fit.conf_int().loc['X']
                    result.update(status='estimable',primary_contrast=float(fit.params['X']),ci_lower=float(bounds.iloc[0]),ci_upper=float(bounds.iloc[1]),p_value=float(fit.pvalues['X']))
                    checks.append(dict(dataset=d,horizon=label,coefficient_error=be,cluster_covariance_error=ce))
            assert abs(coefficients['maximum']-coefficients['single_expectation']-coefficients['D'])<1e-12
            result['adjusted_coefficients']=coefficients
        models.append(result)
        con.unregister('opportunity_samples')
primary=[r for r in models if r['horizon']=='12h'];assert len(primary)==2
for r,p in zip(primary,multipletests([r['p_value'] for r in primary],method='holm')[1]):r['holm_p']=float(p)
for info in inputs.values():
    p=Path(info['path']);s=p.stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'] and oh(p)==info['sha256']
assert oh(support_script)==design_hash
oj('arm_counts.json',counts);oj('endpoint_validation.json',audits);oj('model_results.json',models)
oj('design_details.json',designs);oj('independent_model_validation.json',checks)
oj('manifest.json',dict(inputs=inputs,protocol_sha256=TROPONIN_OPPORTUNITY_PROTOCOL_SHA256,design_source_sha256=design_hash,
  software={m:__import__(m).__version__ for m in ['numpy','pandas','duckdb','statsmodels','scipy']},
  source_hashes_unchanged=True,environment='google_colab' if IN_COLAB else 'local',mortality_read=False,biological_discovery=False,goal_complete=False))
report=['# Troponin sampling-opportunity sensitivity','',
  'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Existing MIMIC/eICU HF raw-replay caches. This is a four-hour-landmark analysis, not the earlier locked episode-window replication.',
  '', '| Source | Horizon | Exposure | Encounters | Mean samples | Repeated samples | Maximum rise | One-sample expectation | First-sample rise |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in counts:report.append(f"| {r['dataset']} | {r['horizon']} | {r['exposed']} | {r['encounters']} | {r['mean_samples']:.2f} | {r['repeated_sampling_fraction']:.1%} | {r['maximum_mean']:.1%} | {r['single_expectation_mean']:.1%} | {r['first_sample_mean']:.1%} |")
report+=['','| Source | Horizon | Status | Adjusted maximum-minus-one-sample association (pp) | 95% interval (pp) | Primary Holm p |','|---|---|---|---:|---|---:|']
for r in models:
    if r['status']=='estimable':
        hp=f"{r['holm_p']:.4g}" if r['holm_p'] is not None else 'sensitivity only'
        report.append(f"| {r['dataset']} | {r['horizon']} | {r['status']} | {100*r['primary_contrast']:.2f} | {100*r['ci_lower']:.2f} to {100*r['ci_upper']:.2f} | {hp} |")
    else:report.append(f"| {r['dataset']} | {r['horizon']} | {r['status']} | NA | NA | NA |")
report+=['',f"Decimal versus original floating-point maximum labels differ in {sum(a['decimal_legacy_label_differences'] for a in audits)} horizon-specific rows (12/24-hour patients overlap). Legacy ratios and labels reconstruct exactly. Independent sample joins, timestamp outcomes, regression coefficients, covariance and paired-coefficient identities passed.",
  '', 'One-sample expectation is the mean threshold indicator over the patient\'s observed timestamps. It is an exact resampling expectation, not a probability observed under a standardized clinical sampling schedule. Taking a maximum and taking an average define different summaries of real kinetics and clinician-selected observations.',
  '', 'The cohort is conditioned on paired troponin testing and hospital follow-up to the window end. Baselines use original specimen-time rules; result unavailability at the four-hour landmark is reported separately in arm_counts.json. Numeric ratios do not establish adjudicated myocardial injury or MI. These results do not identify causal testing bias, reconcile the older episode-based estimators, or demonstrate mortality benefit or novel biology.','']
(PROJECT/'docs/SPO2_TROPONIN_OPPORTUNITY_RESULTS.md').write_text('\n'.join(report))
con.close();os_status('sampling-opportunity contrast and independent validation','complete')
print('\n'.join(report))
'''
