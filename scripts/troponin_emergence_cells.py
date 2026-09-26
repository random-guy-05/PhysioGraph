"""Conservative first-specimen emergence from explicitly bounded troponin."""

SETUP = r'''
import ast, hashlib, importlib.util, json, math, re, runpy
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_troponin_emergence';OUT.mkdir(parents=True,exist_ok=True)
def ej(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def eh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def eq(value):return "'"+str(value).replace("'","''")+"'"
def es(stage,status='running'):
    ej('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert eh(PROJECT/'docs/SPO2_TROPONIN_EMERGENCE_PLAN.md')==TROPONIN_EMERGENCE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_EMERGENCE_PROTOCOL_SHA256
else:ej('protocol_lock.json',dict(sha256=TROPONIN_EMERGENCE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_selection_counts=True,prior_reporting_audits_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'troponin_emergence.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
refs={}
for alias,name in [('audit','troponin_assay_audit'),('care','spo2_cardiorenal'),('recovery','troponin_reporting_recovery')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=eh(p));con.execute(f'ATTACH {eq(p)} AS {alias} (READ_ONLY)')
manifest=dict(references=refs,protocol_sha256=TROPONIN_EMERGENCE_PROTOCOL_SHA256)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
ej('input_manifest.json',manifest)
es('protocol locked; source classification starting')
'''

ANALYSIS = r'''
number=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
epattern=re.compile(r'^\s*(<=|>=|<|>|=)?\s*('+number+r')\s*$')
def decimal_or_none(x):
    try:
        v=Decimal(str(x))
        return v if v.is_finite() and 0<=v<=1000000 else None
    except InvalidOperation:return None
def strict_decimal(x):
    s=str(x).strip().lower().replace('≤','<=').replace('≥','>=')
    s=re.sub(r'^less\s+than\s*','<',s);s=re.sub(r'^greater\s+than\s*','>',s)
    m=epattern.fullmatch(s)
    if not m:return '?',None
    v=decimal_or_none(m.group(2))
    return (m.group(1) or '=',v) if v is not None else ('?',None)
def emergence(op,v,u):
    if op=='?' or v is None:return 'indeterminate'
    t=Decimal('1.5')*u
    if op=='=':return 'positive' if v>=t else 'negative'
    if op in ('>','>=') and v>=t:return 'positive'
    if op=='<' and v<=t or op=='<=' and v<t:return 'negative'
    return 'indeterminate'
fixtures=[('=', '0.015','positive'),('=','0.014999999999999999','negative'),
 ('<','0.015','negative'),('<=','0.015','indeterminate'),('<=','0.014','negative'),
 ('>','0.015','positive'),('>=','0.015','positive'),('>','0.014','indeterminate'),
 ('<','0.016','indeterminate'),('?','0.015','indeterminate'),('=','0','negative')]
for op,v,want in fixtures:assert emergence(op,Decimal(v),Decimal('.01'))==want
for text,want in [('<0.01',('<',Decimal('.01'))),('0.015 ng/ml',('?',None)),
 ('CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.',('?',None)),('≤1e-2',('<=',Decimal('.01'))),
 ('>25*. cTropnT > 0.10 ng/mL suggests Acute MI.',('?',None))]:assert strict_decimal(text)==want
validation=[];selection=[];counts=[];models=[];source_counts=[];bound_counts=[]
def wilson(k,n):
    if not n:return [None,None]
    z=norm.ppf(.975);p=k/n;den=1+z*z/n
    return [(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den,
            (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den]
for d in ('eicu','mimic'):
    es(d+' interval classification and first-specimen selection')
    raw=con.execute(f'SELECT * FROM audit.{d}_raw').df()
    if d=='mimic':
        template=con.execute('SELECT source_id,operator,bound_literal FROM recovery.mimic_comment_template_results').df()
        assert template.source_id.is_unique
        raw=raw.merge(template,on='source_id',how='left',validate='many_to_one')
    classified=[]
    for r in raw.itertuples():
        numeric=decimal_or_none(r.numeric_text)
        if d=='eicu':
            op,v=strict_decimal(r.result_text)
            if op=='=' and (numeric is None or v!=numeric):op,v='?',None
        elif numeric is not None:op,v='=',numeric
        else:
            v=decimal_or_none(r.bound_literal);op=str(r.operator)
            if op not in ('<','>') or v is None:op,v='?',None
        classified.append((op,str(v.normalize()) if v is not None else None))
    raw['op']=[x[0] for x in classified];raw['magnitude']=[x[1] for x in classified]
    for (op,),g in raw.groupby(['op']):source_counts.append(dict(dataset=d,operator=op,rows=len(g)))
    records=[]
    for key,g in raw.groupby(['stay_id','event_minute','concept','unit'],dropna=False):
        forms=set(zip(g.op,g.magnitude))
        op,v=next(iter(forms)) if len(forms)==1 else ('?',None)
        if op=='?':v=None
        available=float('inf') if g.available_minute.isna().any() else max(float(key[1]),float(g.available_minute.max()))
        records.append(dict(zip(['stay_id','event_minute','concept','unit'],key),op=op,magnitude=v,available_minute=available,n_source=len(g)))
    groups=pd.DataFrame(records);groups['assay_order']=groups.concept.map({'troponin_t':0,'troponin_i':1})
    assert groups.assay_order.notna().all()
    con.register('interval_groups',groups);con.execute(f'CREATE OR REPLACE TABLE {d}_groups AS SELECT * FROM interval_groups');con.unregister('interval_groups')
    cohort=con.execute(f'SELECT * FROM care.{d}_cohort').df();assert cohort.stay_id.is_unique
    candidates=groups.loc[groups.event_minute.between(0,240)&groups.available_minute.le(240)]
    latest=candidates.sort_values(['stay_id','event_minute','assay_order','unit'],ascending=[True,False,True,True],na_position='last').drop_duplicates('stay_id')
    valid=latest.op.isin(['<','<='])&latest.magnitude.map(lambda x:decimal_or_none(x) is not None and decimal_or_none(x)>0)&latest.unit.eq('ng/ml')
    eligible=latest.loc[valid].merge(cohort,on='stay_id',validate='one_to_one')
    base=eligible.sort_values('stay_id').drop_duplicates('person_id').reset_index(drop=True)
    sql=con.execute(f"""WITH latest AS (SELECT * FROM {d}_groups WHERE event_minute BETWEEN 0 AND 240 AND available_minute<=240
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY event_minute DESC,assay_order,unit NULLS LAST)=1)
      SELECT g.*,c.person_id FROM latest g JOIN care.{d}_cohort c USING(stay_id)
      WHERE op IN ('<','<=') AND try_cast(magnitude AS DOUBLE)>0 AND unit='ng/ml'
      QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1 ORDER BY stay_id""").df()
    cols=['stay_id','event_minute','concept','unit','op','magnitude','available_minute','person_id']
    pd.testing.assert_frame_equal(base[cols],sql[cols],check_dtype=False)
    con.register('baseline',base);con.execute(f'CREATE OR REPLACE TABLE {d}_baseline AS SELECT * FROM baseline');con.unregister('baseline')
    for arm in (0,1):
        selection.append(dict(dataset=d,exposed=arm,original_encounters=int(cohort.exposed.eq(arm).sum()),
          any_available_baseline=int(latest.stay_id.isin(cohort.loc[cohort.exposed.eq(arm),'stay_id']).sum()),
          bound_eligible_encounters=int(eligible.exposed.eq(arm).sum()),selected_people=int(base.exposed.eq(arm).sum())))
    for (arm,concept,u),g in base.groupby(['exposed','concept','magnitude']):bound_counts.append(dict(dataset=d,exposed=int(arm),concept=concept,upper_bound=u,people=len(g)))
    for end,label in [(960,'12h'),(1680,'24h')]:
        posts=groups.loc[groups.event_minute.gt(240)&groups.event_minute.le(end)].sort_values('event_minute').drop_duplicates(['stay_id','concept','unit'])
        chosen=base.merge(posts,on=['stay_id','concept','unit'],how='left',suffixes=('_baseline','_post'),validate='one_to_one')
        sqlpost=con.execute(f"""SELECT b.stay_id,p.event_minute,p.op,p.magnitude,p.available_minute
          FROM {d}_baseline b LEFT JOIN (SELECT * FROM {d}_groups WHERE event_minute>240 AND event_minute<={end}
            QUALIFY row_number() OVER(PARTITION BY stay_id,concept,unit ORDER BY event_minute)=1) p
          USING(stay_id,concept,unit) ORDER BY stay_id""").df()
        pp=chosen[['stay_id','event_minute_post','op_post','magnitude_post','available_minute_post']].copy();pp.columns=sqlpost.columns
        for frame in (pp,sqlpost):
            for col in ('op','magnitude'):frame[col]=frame[col].astype('string')
        pd.testing.assert_frame_equal(pp,sqlpost,check_dtype=False)
        labels=[];oracle=[]
        for r in chosen.itertuples():
            if pd.isna(r.event_minute_post):result=other='not_remeasured'
            elif r.available_minute_post>end:result=other='unavailable'
            else:
                v=decimal_or_none(r.magnitude_post);u=Decimal(r.magnitude_baseline)
                result=emergence(r.op_post,v,u)
                # Independent endpoint oracle: endpoints of the possible-value interval.
                lower=v if r.op_post in ('=','>','>=') else Decimal(0)
                upper=v if r.op_post in ('=','<','<=') else Decimal('Infinity')
                threshold=u+u/2
                if r.op_post=='?' or v is None:other='indeterminate'
                elif lower>=threshold:other='positive'
                elif upper<threshold or upper==threshold and r.op_post=='<':other='negative'
                else:other='indeterminate'
            labels.append(result);oracle.append(other)
        assert labels==oracle
        chosen['endpoint']=labels
        con.register('chosen',chosen);con.execute(f'CREATE OR REPLACE TABLE {d}_first_{label} AS SELECT * FROM chosen');con.unregister('chosen')
        armstats={}
        for arm in (0,1):
            g=chosen.loc[chosen.exposed.eq(arm)];dist=g.endpoint.value_counts().to_dict()
            k=int(dist.get('positive',0));n=k+int(dist.get('negative',0));interval=wilson(k,n);armstats[arm]=(k,n,interval)
            timing=g.event_minute_post.dropna()
            row=dict(dataset=d,horizon=label,exposed=arm,baseline_people=len(g),definite_people=n,positive=k,
              risk=k/n if n else None,wilson95=interval,first_specimen_minute_quantiles=[float(x) for x in timing.quantile([.25,.5,.75])] if len(timing) else None)
            for status in ('negative','indeterminate','unavailable','not_remeasured'):row[status]=int(dist.get(status,0))
            assert sum(row[s] for s in ('positive','negative','indeterminate','unavailable','not_remeasured'))==len(g)
            counts.append(row)
        k0,n0,ci0=armstats[0];k1,n1,ci1=armstats[1]
        info=dict(dataset=d,horizon=label,model_fitted=False)
        if n0 and n1:
            p0=k0/n0;p1=k1/n1;rd=p1-p0
            info.update(crude_rd=rd,crude_rd95=[rd-math.sqrt((p1-ci1[0])**2+(ci0[1]-p0)**2),rd+math.sqrt((ci1[1]-p1)**2+(p0-ci0[0])**2)])
        reasons=[]
        if min(n0,n1)<50:reasons.append('fewer than 50 definite observations per exposure arm')
        if min(k0,k1,n0-k0,n1-k1)<5:reasons.append('fewer than five events or non-events per exposure arm')
        a=chosen.loc[chosen.endpoint.isin(['positive','negative'])].copy()
        design=pd.DataFrame({'intercept':np.ones(len(a)),'exposed':a.exposed.to_numpy()})
        features={c:pd.to_numeric(a[c],errors='coerce').to_numpy() for c in ['age','is_male','sex_unknown_flag','spo2_mean','hypoxic_fraction','hr_mean','map_mean','fio2_max','vent_documented','vasoactive_documented']}
        features['log_readings']=np.log1p(a.spo2_readings.to_numpy());features['log_bound']=np.log(a.magnitude_baseline.astype(float).to_numpy());features['troponin_t']=a.concept.eq('troponin_t').astype(float).to_numpy()
        for name,values in features.items():
            values=np.asarray(values,dtype=float);missing=~np.isfinite(values)
            if missing.any():design[name+'_missing']=missing.astype(float)
            design[name]=np.where(missing,np.median(values[~missing]) if (~missing).any() else 0,values)
        nuisance=[c for c in design if c not in ('intercept','exposed') and design[c].nunique()<=1]
        design=design.drop(columns=nuisance);x=design.to_numpy(dtype=float);y=a.endpoint.eq('positive').to_numpy(dtype=float)
        rank=int(np.linalg.matrix_rank(x)) if len(x) else 0
        info.update(n_definite=len(a),design_columns=list(design),removed_constant_columns=nuisance,rank=rank)
        if rank!=x.shape[1]:reasons.append('design is not full rank')
        if min(k0+k1,n0+n1-k0-k1)<5*(x.shape[1]-1):reasons.append('fewer than five minority outcomes per nonconstant parameter')
        info['gate_failures']=reasons
        if not reasons:
            fit=OLS(y,x).fit(cov_type='HC3');j=list(design).index('exposed')
            beta=np.linalg.lstsq(x,y,rcond=None)[0];bread=np.linalg.inv(x.T@x)
            leverage=np.einsum('ij,jk,ik->i',x,bread,x);resid=y-x@beta
            covariance=bread@(x.T@(((resid/(1-leverage))**2)[:,None]*x))@bread
            np.testing.assert_allclose(beta,fit.params,rtol=1e-7,atol=1e-8);np.testing.assert_allclose(covariance,fit.cov_params(),rtol=1e-6,atol=1e-8)
            se=float(np.sqrt(covariance[j,j]));b=float(beta[j]);p=float(2*norm.sf(abs(b/se)))
            info.update(model_fitted=True,adjusted_rd=b,adjusted_rd95=[b-1.959963984540054*se,b+1.959963984540054*se],p=p,independent_hc3_matches=True)
        models.append(info)
        validation.append(dict(dataset=d,horizon=label,baseline_sql_python_match=True,first_specimen_sql_python_match=True,decimal_interval_oracle_matches=True,selected_people=len(base),endpoint_rows=len(chosen)))
primary=[m for m in models if m['horizon']=='12h'];ordered=sorted(primary,key=lambda m:m.get('p',1));last=0
for i,m in enumerate(ordered):last=max(last,min(1,(2-i)*m.get('p',1)));m['primary_holm_p']=last
for ref in refs.values():assert eh(Path(ref['path']))==ref['sha256']
for name,obj in [('source_class_counts.json',source_counts),('baseline_selection.json',selection),('baseline_bounds.json',bound_counts),('endpoint_counts.json',counts),('associations.json',models),('validation.json',validation)]:ej(name,obj)
ej('manifest.json',dict(protocol_sha256=TROPONIN_EMERGENCE_PROTOCOL_SHA256,script_sha256=eh(PROJECT/'scripts/troponin_emergence_cells.py'),source_fingerprints_unchanged=True,
  synthetic_interval_tests=len(fixtures),synthetic_parser_tests=5,mortality_accessed=False,existing_labels_replaced=False,environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# First-specimen reporting-limit emergence','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. No mortality outcome was accessed.',
 '', '| Dataset | Horizon | SpO2 exposed | Baseline people | Positive | Negative | Indeterminate | Unavailable | Not remeasured |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in counts:report.append('| '+' | '.join(str(r[k]) for k in ['dataset','horizon','exposed','baseline_people','positive','negative','indeterminate','unavailable','not_remeasured'])+' |')
report+=['','Risks and associations condition on having a definite first follow-up result; unmeasured and indeterminate results are not negative. Baseline is an explicit interval, not an imputed value.','']
for m in models:
    text=f"- {m['dataset']} {m['horizon']}: "
    if 'crude_rd' in m:text+=f"descriptive risk difference {100*m['crude_rd']:.2f} percentage points (95% Newcombe interval {100*m['crude_rd95'][0]:.2f} to {100*m['crude_rd95'][1]:.2f}). "
    text+=('Adjusted model: '+str({k:m[k] for k in ['adjusted_rd','adjusted_rd95','p']}) if m['model_fitted'] else 'Adjusted model not fitted: '+'; '.join(m['gate_failures']))
    report.append(text)
report+=['','SQL/Python selection and independent decimal interval labeling agree. Original caches are unchanged. Results do not establish injury onset, infarction, causal biology, mortality benefit, or a paradigm-shifting discovery. The research goal remains unmet.']
report+=['','Novelty check: conversion from initially negative troponin to detectable or positive values in acute HF was already studied in [PROTECT](https://pubmed.ncbi.nlm.nih.gov/21900185/) and a [198-patient serial-sampling study](https://pubmed.ncbi.nlm.nih.gov/22407461/). Sleep-disordered breathing and overnight troponin increases were also examined by [Light et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708937/). Our conservative interval endpoint and ICU exposure definition differ, but those differences alone are not a new biological mechanism. Literature supplies context only; all patient analyses here use MIMIC and eICU.']
(PROJECT/'docs/SPO2_TROPONIN_EMERGENCE_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');es('first-specimen emergence completed','complete')
print('\n'.join(report));con.close()
'''
