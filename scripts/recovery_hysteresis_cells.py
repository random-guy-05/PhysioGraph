"""Fixed-time cardiovascular recovery screen in original MIMIC/eICU cohorts."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_recovery_hysteresis';OUT.mkdir(parents=True,exist_ok=True)
def rh_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def rh_q(x):return "'"+str(x).replace("'","''")+"'"
def rh_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert rh_hash(PROJECT/'docs/SPO2_RECOVERY_HYSTERESIS_PLAN.md')==RECOVERY_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==RECOVERY_PROTOCOL_SHA256
else:rh_json('protocol_lock.json',dict(sha256=RECOVERY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_recovery_classes=True,earlier_project_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'recovery_hysteresis.duckdb'));con.execute('SET threads=2')
inputs={}
for alias,name in [('prior','spo2_context'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');inputs[alias]=dict(path=str(p),sha256=rh_hash(p))
    con.execute(f'ATTACH {rh_q(p)} AS {alias} (READ_ONLY)')
rh_json('source_manifest.json',dict(inputs=inputs,environment='google_colab' if IN_COLAB else 'local'))
rh_json('run_status.json',dict(status='running',stage='fixed protocol; recovery reconstruction pending'))
print('Recovery/sham protocol frozen before recovery classes and their mortality.')
"""

PHENOTYPE = r"""
selection=[];validation=[]
for d in ['eicu','mimic']:
    original=con.execute(f'SELECT stay_id,cast(person_id AS VARCHAR) person_id,exposed,age,is_male,sex_unknown_flag FROM care.{d}_cohort ORDER BY stay_id').df()
    assert len(original)=={'eicu':11452,'mimic':4711}[d]
    con.execute(f'''CREATE OR REPLACE TABLE {d}_selected AS SELECT stay_id,cast(person_id AS VARCHAR) person_id,
      exposed,age,is_male,sex_unknown_flag FROM care.{d}_cohort
      QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1''')
    selected=con.execute(f'SELECT * FROM {d}_selected ORDER BY stay_id').df()
    assert np.array_equal(selected.stay_id,original.groupby('person_id',sort=False).head(1).stay_id)
    con.execute(f'''CREATE OR REPLACE TABLE {d}_first AS WITH p AS (
      SELECT b.stay_id,b.bin post_bin,lag(b.bin) OVER w pre_bin,b.value,
      lag(b.value) OVER w previous_value,b.representative_minute b,lag(b.representative_minute) OVER w a
      FROM prior.{d}_bins b JOIN {d}_selected e USING(stay_id) WHERE concept='spo2'
      WINDOW w AS (PARTITION BY b.stay_id ORDER BY bin))
      SELECT stay_id,pre_bin,post_bin,value-previous_value delta FROM p
      WHERE abs(value-previous_value)>=4 AND b-a>0 AND b-a<=30
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY post_bin)=1''')
    first=con.execute(f'SELECT * FROM {d}_first ORDER BY stay_id').df()
    assert set(first.stay_id)==set(selected.loc[selected.exposed.eq(1),'stay_id'])
    bins=con.execute(f'SELECT b.* FROM prior.{d}_bins b JOIN {d}_selected e USING(stay_id) ORDER BY stay_id,bin,concept').df()
    spo=bins.loc[bins.concept.eq('spo2')].copy()
    spo['pre_bin']=spo.groupby('stay_id').bin.shift();spo['delta']=spo.value-spo.groupby('stay_id').value.shift()
    spo['dt']=spo.representative_minute-spo.groupby('stay_id').representative_minute.shift()
    pyfirst=spo.loc[spo.delta.abs().ge(4)&spo.dt.gt(0)&spo.dt.le(30)].groupby('stay_id',sort=False).head(1)
    assert np.array_equal(first.stay_id,pyfirst.stay_id) and np.array_equal(first.post_bin,pyfirst.bin)
    assert np.array_equal(first.pre_bin,pyfirst.pre_bin) and np.allclose(first.delta,pyfirst.delta,rtol=0,atol=1e-12)
    cases=first.loc[first.delta.lt(0),['stay_id','pre_bin','post_bin']].copy();cases['desaturation']=1
    anchors=cases[['pre_bin','post_bin']].to_numpy();assert len(anchors)>0
    controls=selected.loc[selected.exposed.eq(0),['stay_id','person_id']].copy()
    indices=[int(hashlib.sha256(('PhysioGraph recovery sham v1|'+d+'|'+p).encode()).hexdigest(),16)%len(anchors) for p in controls.person_id]
    controls['pre_bin']=[int(anchors[i,0]) for i in indices];controls['post_bin']=[int(anchors[i,1]) for i in indices];controls['desaturation']=0
    candidate=pd.concat([cases,controls[cases.columns]],ignore_index=True)
    con.register('rh_candidate',candidate);con.execute(f'CREATE OR REPLACE TABLE {d}_candidate AS SELECT * FROM rh_candidate');con.unregister('rh_candidate')
    sql_sham=con.execute(f'''WITH hashes AS (
      SELECT stay_id,sha256('PhysioGraph recovery sham v1|{d}|'||person_id) h FROM {d}_selected WHERE exposed=0),
      idx AS (SELECT stay_id,list_reduce(list_transform(range(1,65), i -> strpos('0123456789abcdef',substr(h,i,1))-1),
        (acc,digit) -> (acc*16+digit)%{len(anchors)},0) anchor_index FROM hashes),
      anchors AS (SELECT pre_bin,post_bin,row_number() OVER(ORDER BY stay_id)-1 anchor_index FROM {d}_first WHERE delta<0)
      SELECT idx.stay_id,a.pre_bin,a.post_bin FROM idx JOIN anchors a USING(anchor_index) ORDER BY idx.stay_id''').df()
    expected_sham=controls.sort_values('stay_id')[['stay_id','pre_bin','post_bin']]
    assert np.array_equal(sql_sham.to_numpy(),expected_sham.to_numpy())
    con.execute(f'''CREATE OR REPLACE TABLE {d}_phenotype AS SELECT c.*,e.person_id,e.age,e.is_male,e.sex_unknown_flag,
      a.value spo2_pre,b.value spo2_post,r.value spo2_recovery,
      a.representative_minute time_pre,b.representative_minute time_post,r.representative_minute time_recovery,
      h.value hr_pre,k.value hr_recovery,h.representative_minute hr_time_pre,k.representative_minute hr_time_recovery,
      coalesce(b.representative_minute-a.representative_minute>0 AND b.representative_minute-a.representative_minute<=30
        AND r.representative_minute-b.representative_minute BETWEEN 45 AND 75 AND r.representative_minute<=240,false) timing_ok,
      coalesce(abs(r.value-a.value)<=1 AND (c.desaturation=1 OR abs(b.value-a.value)<=1),false) saturation_ok,
      coalesce(abs(h.representative_minute-a.representative_minute)<=5 AND abs(k.representative_minute-r.representative_minute)<=5,false) hr_ok
      FROM {d}_candidate c JOIN {d}_selected e USING(stay_id)
      LEFT JOIN prior.{d}_bins a ON a.stay_id=c.stay_id AND a.concept='spo2' AND a.bin=c.pre_bin
      LEFT JOIN prior.{d}_bins b ON b.stay_id=c.stay_id AND b.concept='spo2' AND b.bin=c.post_bin
      LEFT JOIN prior.{d}_bins r ON r.stay_id=c.stay_id AND r.concept='spo2' AND r.bin=c.post_bin+4
      LEFT JOIN prior.{d}_bins h ON h.stay_id=c.stay_id AND h.concept='hr' AND h.bin=c.pre_bin
      LEFT JOIN prior.{d}_bins k ON k.stay_id=c.stay_id AND k.concept='hr' AND k.bin=c.post_bin+4''')
    ph=con.execute(f'SELECT * FROM {d}_phenotype ORDER BY stay_id').df()
    lookup={(int(r.stay_id),int(r.bin),r.concept):(float(r.value),float(r.representative_minute)) for r in bins.itertuples()}
    flags=[]
    for row in ph.itertuples():
        a,ta=lookup.get((row.stay_id,int(row.pre_bin),'spo2'),(np.nan,np.nan))
        b,tb=lookup.get((row.stay_id,int(row.post_bin),'spo2'),(np.nan,np.nan))
        r,tr=lookup.get((row.stay_id,int(row.post_bin)+4,'spo2'),(np.nan,np.nan))
        h,th=lookup.get((row.stay_id,int(row.pre_bin),'hr'),(np.nan,np.nan))
        k,tk=lookup.get((row.stay_id,int(row.post_bin)+4,'hr'),(np.nan,np.nan))
        flags.append([0<tb-ta<=30 and 45<=tr-tb<=75 and tr<=240,abs(r-a)<=1 and (row.desaturation==1 or abs(b-a)<=1),abs(th-ta)<=5 and abs(tk-tr)<=5])
        for observed,expected in [(row.hr_pre,h),(row.hr_recovery,k),(row.spo2_pre,a),(row.spo2_recovery,r)]:
            assert (pd.isna(observed) and pd.isna(expected)) or abs(observed-expected)<=1e-12
    assert np.array_equal(ph[['timing_ok','saturation_ok','hr_ok']],np.asarray(flags))
    for group in [0,1]:
        p=ph.loc[ph.desaturation.eq(group)]
        for stage,mask in [('candidate',np.ones(len(p),dtype=bool)),('timing',p.timing_ok),('saturation_recovered',p.timing_ok&p.saturation_ok),('hr_matched',p.timing_ok&p.saturation_ok&p.hr_ok)]:
            selection.append(dict(dataset=d,desaturation=group,stage=stage,people=int(np.sum(mask))))
    validation.append(dict(dataset=d,lowest_stay_and_original_first_event_reconstruct=True,python_sql_timing_flags_and_values_agree=True,sham_hash_algorithm='SHA256 full integer modulo case anchor count',independent_sql_sha256_digitwise_modulo_and_sham_pairs_agree=True))
rh_json('selection.json',selection);rh_json('phenotype_validation.json',validation)
rh_json('run_status.json',dict(status='running',stage='phenotypes frozen and validated; mortality not yet joined'))
print('Recovery phenotype selected and independently validated in both original datasets.')
"""

SCREEN = r"""
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
results=[];support=[];model_checks=[]
for d in ['eicu','mimic']:
    p=con.execute(f'''SELECT p.*,c.hospital_id,c.hospital_mortality,c.excluded_before_landmark_flag
      FROM {d}_phenotype p JOIN prior.{d}_cohort c USING(stay_id)
      WHERE p.timing_ok AND p.saturation_ok AND p.hr_ok''').df()
    assert p.excluded_before_landmark_flag.eq(0).all()
    known=p.hospital_mortality.isin([0,1]);demo=p.age.notna()&np.isfinite(p.age)&p.sex_unknown_flag.eq(0)&p.is_male.isin([0,1])
    a=p.loc[known&demo].copy();a['hr_displacement']=(a.hr_recovery-a.hr_pre)/10
    rows=[]
    for g in [0,1]:
        v=a.loc[a.desaturation.eq(g)];n=len(v);deaths=int(v.hospital_mortality.sum())
        rows.append(dict(dataset=d,desaturation=g,people=n,deaths=deaths,survivors=n-deaths,
          unknown_outcomes=int((p.desaturation.eq(g)&~known).sum()),demographic_exclusions=int((p.desaturation.eq(g)&known&~demo).sum()),
          gate=n>=({0:100,1:50}[g]) and deaths>=10 and n-deaths>=10))
    hospitals=int(a.hospital_id.nunique()) if d=='eicu' else None
    gate=all(r['gate'] for r in rows) and (d!='eicu' or (a.hospital_id.notna().all() and hospitals>=20))
    support+=rows
    if not gate:
        for endpoint in ['within_desaturation_hr_or','desaturation_specificity_or']:
            results.append(dict(dataset=d,endpoint=endpoint,estimable=False,reason='fixed support floor failed',p=1.0))
        model_checks.append(dict(dataset=d,fit=False,hospitals=hospitals));continue
    X=np.column_stack([np.ones(len(a)),a.desaturation,a.hr_displacement,a.desaturation*a.hr_displacement,
      a.hr_pre/10,(a.spo2_pre-95)/5,(a.age-65)/10,a.is_male]).astype(float)
    y=a.hospital_mortality.to_numpy(float);n,k=X.shape
    valid=np.linalg.matrix_rank(X)==k and np.linalg.cond(X)<1e6
    if valid:
        kwargs=dict(cov_type='cluster',cov_kwds={'groups':a.hospital_id.to_numpy(),'use_correction':True}) if d=='eicu' else dict(cov_type='HC0')
        fit=sm.GLM(y,X,family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10,**kwargs)
        prob=np.asarray(fit.fittedvalues);beta=np.asarray(fit.params);H=X.T@((prob*(1-prob))[:,None]*X)
        valid=bool(fit.converged and np.isfinite(beta).all() and np.max(np.abs(beta))<25 and np.linalg.eigvalsh(H).min()>1e-10 and np.max(np.abs(X.T@(y-prob)))<1e-6)
    if not valid:
        for endpoint in ['within_desaturation_hr_or','desaturation_specificity_or']:
            results.append(dict(dataset=d,endpoint=endpoint,estimable=False,reason='fixed model validity check failed',p=1.0))
        model_checks.append(dict(dataset=d,fit=False,valid=False));continue
    bread=np.linalg.inv(H);scores=X*(y-prob)[:,None]
    if d=='eicu':
        summed=pd.DataFrame(scores).groupby(a.hospital_id.to_numpy()).sum().to_numpy();G=len(summed)
        meat=summed.T@summed*(G/(G-1))*((n-1)/(n-k))
    else:meat=scores.T@scores
    independent_cov=bread@meat@bread;cov=np.asarray(fit.cov_params())
    assert np.allclose(cov,independent_cov,rtol=1e-7,atol=1e-9)
    model_checks.append(dict(dataset=d,fit=True,n=n,k=k,hospitals=hospitals,converged=True,score_max_abs=float(np.max(np.abs(X.T@(y-prob)))),independent_sandwich_max_abs=float(np.max(np.abs(cov-independent_cov)))))
    for endpoint,L in [('within_desaturation_hr_or',np.array([0,0,1,1,0,0,0,0.])),('desaturation_specificity_or',np.array([0,0,0,1,0,0,0,0.]))]:
        estimate=float(L@beta);se=float(np.sqrt(L@cov@L))
        results.append(dict(dataset=d,endpoint=endpoint,estimable=True,or_per_10_bpm=float(np.exp(estimate)),lower95=float(np.exp(estimate-1.95996398454*se)),upper95=float(np.exp(estimate+1.95996398454*se)),p=float(2*norm.sf(abs(estimate/se)))))
    con.register('rh_analysis',a);con.execute(f'CREATE OR REPLACE TABLE {d}_analysis AS SELECT * FROM rh_analysis');con.unregister('rh_analysis')
assert len(results)==4
for r,padj in zip(results,multipletests([r['p'] for r in results],method='holm')[1]):r['holm_p']=float(padj)
advance=all(r['estimable'] and r.get('or_per_10_bpm',0)>1 and r['holm_p']<.05 for r in results)
advance=advance and all(r.get('or_per_10_bpm',0)>=1.5 for r in results if r['endpoint']=='within_desaturation_hr_or')
rh_json('support.json',support);rh_json('model_validation.json',model_checks)
rh_json('results.json',dict(results=results,advancement_gate_pass=advance,biological_discovery=False,mortality_benefit_established=False))
for info in inputs.values():assert rh_hash(info['path'])==info['sha256']
report=['# Cardiovascular recovery mortality screen','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Original cohorts, fixed recovery and sham anchors.','',
 '| Source | Contrast | OR per 10 bpm | Nominal 95% CI | Holm p |','|---|---|---:|---|---:|']
for r in results:
    if r['estimable']:report.append(f"| {r['dataset']} | {r['endpoint']} | {r['or_per_10_bpm']:.4f} | {r['lower95']:.4f} to {r['upper95']:.4f} | {r['holm_p']:.6g} |")
    else:report.append(f"| {r['dataset']} | {r['endpoint']} | Not estimated | {r['reason']} | 1 |")
report+=['','Prespecified joint advancement criterion: '+str(advance)+'.',
 'This is a minimally adjusted observational screen with selected recovered episodes, not a treatment effect or measured autonomic mechanism. Prior studies already establish broad HR-recovery prognosis. No alternative time window, threshold or subgroup was used to rescue a failed screen.','']
(PROJECT/'docs/SPO2_RECOVERY_HYSTERESIS_RESULTS.md').write_text('\n'.join(report))
rh_json('run_status.json',dict(status='complete',stage='fixed recovery screen and independent model validation',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('\n'.join(report));print('Known-outcome support:',support)
"""
