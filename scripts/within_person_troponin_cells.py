"""Sequential within-person support and analysis in the original HF cohort."""

SETUP = r'''
import hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,InvalidOperation
import duckdb
import numpy as np
import pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_within_person_troponin';OUT.mkdir(parents=True,exist_ok=True)
def wj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def wh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def wq(value):return "'"+str(value).replace("'","''")+"'"
def ws(stage,status='running'):
    wj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert wh(PROJECT/'docs/SPO2_WITHIN_PERSON_TROPONIN_PLAN.md')==WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256
else:wj('protocol_lock.json',dict(sha256=WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_new_interval_counts=True,prior_experiments_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'within_person_troponin.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
refs={}
for alias,name in [('intervals','troponin_emergence'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=wh(p));con.execute(f'ATTACH {wq(p)} AS {alias} (READ_ONLY)')
manifest=dict(references=refs,protocol_sha256=WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
wj('input_manifest.json',manifest);ws('timing gate initialized')
'''

TIMING = r'''
support=[];validation=[];attrition=[]
def positive_magnitude(value):
    try:
        v=Decimal(str(value));return v.is_finite() and v>0
    except InvalidOperation:return False
for d in ('eicu','mimic'):
    ws(d+' consecutive specimen support')
    groups=con.execute(f'SELECT * FROM intervals.{d}_groups ORDER BY stay_id,concept,unit,event_minute').df()
    cohort=con.execute(f'SELECT stay_id,person_id FROM care.{d}_cohort').df()
    keys=['stay_id','concept','unit']
    lagcols=['event_minute','available_minute','op','magnitude']
    shifted=groups.groupby(keys,dropna=False)[lagcols].shift(1).add_prefix('prior_')
    pairs=pd.concat([groups,shifted],axis=1)
    stages=[('has_previous',pairs.prior_event_minute.notna()),
      ('supported_unit',pairs.unit.eq('ng/ml')),
      ('window_times',pairs.prior_event_minute.ge(0)&pairs.event_minute.le(1680)&(pairs.event_minute-pairs.prior_event_minute).between(240,720)),
      ('baseline_available_before_window',pairs.prior_available_minute.le(pairs.event_minute-240)),
      ('followup_available',pairs.available_minute.le(1680)),
      ('both_exact_positive',pairs.op.eq('=')&pairs.prior_op.eq('=')&pairs.magnitude.map(positive_magnitude)&pairs.prior_magnitude.map(positive_magnitude))]
    keep=pd.Series(True,index=pairs.index)
    for stage,mask in stages:
        keep&=mask
        attrition.append(dict(dataset=d,stage=stage,intervals=int(keep.sum()),encounters=int(pairs.loc[keep,'stay_id'].nunique())))
    pairs=pairs.loc[keep].copy()
    assay=pairs.groupby(keys,as_index=False).size().rename(columns={'size':'n_intervals'})
    assay['assay_order']=assay.concept.map({'troponin_t':0,'troponin_i':1})
    assay=assay.sort_values(['stay_id','n_intervals','assay_order'],ascending=[True,False,True]).drop_duplicates('stay_id')
    selected=assay.loc[assay.n_intervals.ge(2)].merge(cohort,on='stay_id',validate='one_to_one').sort_values('stay_id').drop_duplicates('person_id')
    final=pairs.merge(selected[keys+['person_id']],on=keys,validate='many_to_one').sort_values(keys+['event_minute']).reset_index(drop=True)
    sql=con.execute(f"""WITH lagged AS (
      SELECT *,lag(event_minute) OVER w prior_event_minute,lag(available_minute) OVER w prior_available_minute,
        lag(op) OVER w prior_op,lag(magnitude) OVER w prior_magnitude
      FROM intervals.{d}_groups WINDOW w AS (PARTITION BY stay_id,concept,unit ORDER BY event_minute)),
      pairs AS (SELECT * FROM lagged WHERE unit='ng/ml' AND prior_event_minute>=0 AND event_minute<=1680
        AND event_minute-prior_event_minute BETWEEN 240 AND 720 AND prior_available_minute<=event_minute-240
        AND available_minute<=1680 AND op='=' AND prior_op='='
        AND try_cast(magnitude AS DOUBLE)>0 AND try_cast(prior_magnitude AS DOUBLE)>0),
      assays AS (SELECT stay_id,concept,unit,count(*) n_intervals FROM pairs GROUP BY stay_id,concept,unit
        QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY n_intervals DESC,CASE WHEN concept='troponin_t' THEN 0 ELSE 1 END)=1),
      chosen AS (SELECT a.*,c.person_id FROM assays a JOIN care.{d}_cohort c USING(stay_id) WHERE n_intervals>=2
        QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1)
      SELECT p.*,c.person_id FROM pairs p JOIN chosen c USING(stay_id,concept,unit)
      ORDER BY stay_id,concept,unit,event_minute""").df()
    cols=keys+['event_minute','available_minute','op','magnitude','prior_event_minute','prior_available_minute','prior_op','prior_magnitude','person_id']
    pd.testing.assert_frame_equal(final[cols],sql[cols],check_dtype=False)
    assert final.groupby('person_id').size().ge(2).all()
    assert final.groupby('person_id').stay_id.nunique().eq(1).all()
    con.register('selected_intervals',final);con.execute(f'CREATE OR REPLACE TABLE {d}_timing_intervals AS SELECT * FROM selected_intervals');con.unregister('selected_intervals')
    people=int(final.person_id.nunique())
    support.append(dict(dataset=d,people=people,intervals=len(final),minimum_people=100,passes=people>=100))
    validation.append(dict(dataset=d,sql_python_consecutive_pairs_and_person_selection_match=True,all_people_have_multiple_intervals=True,troponin_changes_calculated=False))
timing_pass=all(r['passes'] for r in support)
for ref in refs.values():assert wh(Path(ref['path']))==ref['sha256']
wj('timing_support.json',support);wj('timing_attrition.json',attrition);wj('timing_validation.json',validation)
wj('timing_manifest.json',dict(protocol_sha256=WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256,script_sha256=wh(PROJECT/'scripts/within_person_troponin_cells.py'),
  source_fingerprints_unchanged=True,both_timing_gates_pass=timing_pass,troponin_changes_calculated=False,raw_vitals_scanned=False,mortality_accessed=False,goal_complete=False))
report=['# Within-person troponin timing support','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'.',
  '', '| Dataset | People with at least two eligible consecutive intervals | Intervals | Required people | Gate |',
  '|---|---:|---:|---:|---|']
for r in support:report.append(f"| {r['dataset']} | {r['people']} | {r['intervals']} | {r['minimum_people']} | {'pass' if r['passes'] else 'fail'} |")
report+=['', 'Independent SQL/Python consecutive pairing, assay selection and person selection agree. Existing source caches are unchanged.']
if timing_pass:
    report+=['','Timing support passes. This does not establish usable oxygen windows, within-person exposure changes, or a biological association. The conditional vital extraction and exposure gate remain to be executed.']
    ws('timing gate passed; longitudinal oxygen extraction pending','timing_complete')
else:
    report+=['','The joint timing gate fails. This specification stops before new raw vital extraction, troponin-change calculations, exposure association or mortality analysis. No time window or patient-count gate was relaxed. This is insufficient support, not evidence that the biological relationship is absent. The discovery goal remains unmet.']
    ws('timing gate failed; specification stopped','complete')
(PROJECT/'docs/SPO2_WITHIN_PERSON_TROPONIN_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');print('\n'.join(report))
'''

VITALS = r'''
if not timing_pass:
    print('Longitudinal vital extraction skipped: the joint timing gate failed.')
else:
    import ast,yaml
    prior_path=PRIVATE/'spo2_context.duckdb'
    prior_ref=dict(path=str(prior_path),sha256=wh(prior_path))
    con.execute(f'ATTACH {wq(prior_path)} AS prior (READ_ONLY)')
    raw_refs={};extracted=[]
    def csv_source(dataset,name):
        p=DRIVE/'Data'/dataset/'Full'/(name+'.csv');s=p.stat()
        raw_refs[dataset+'/'+name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
        wj('vital_sources_pending.json',dict(prior=prior_ref,raw=raw_refs))
        return f'read_csv({wq(p)},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
    config=yaml.safe_load((PROJECT/'configs/default.yaml').read_text())
    tree=ast.parse((PROJECT/'src/physiograph/etl/mimic_extractor.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mimic_chartevent_item_map')
    ns=dict(Path=Path,pd=pd,MIMIC_VITAL_IDS=config['mimic_vital_ids'])
    exec(compile(ast.Module(body=[function],type_ignores=[]),'original-mimic-item-map','exec'),ns)
    csv_source('MIMIC','d_items')
    mapping={k:v for k,v in ns['_mimic_chartevent_item_map'](DRIVE/'Data/MIMIC/Full').items() if v in ('spo2','map','hr')}
    map_frame=pd.DataFrame([dict(itemid=k,concept=v) for k,v in mapping.items()]);con.register('vital_map',map_frame)
    for d,dataset in [('eicu','eICU'),('mimic','MIMIC')]:
        ws(d+' longitudinal raw vital scan')
        if d=='eicu':
            source=csv_source(dataset,'vitalPeriodic')
            con.execute(f"""CREATE OR REPLACE TEMP TABLE longitudinal_wide AS
              SELECT try_cast(v.patientunitstayid AS BIGINT) stay_id,try_cast(v.observationoffset AS DOUBLE) event_minute,
                try_cast(v.sao2 AS DOUBLE) spo2,try_cast(v.systemicmean AS DOUBLE) map_value,try_cast(v.heartrate AS DOUBLE) hr
              FROM {source} v JOIN (SELECT DISTINCT stay_id FROM eicu_timing_intervals) c ON try_cast(v.patientunitstayid AS BIGINT)=c.stay_id
              WHERE try_cast(v.observationoffset AS DOUBLE)>=0 AND try_cast(v.observationoffset AS DOUBLE)<1680""")
            con.execute('CREATE OR REPLACE TABLE eicu_raw_vitals AS '+' UNION ALL '.join(
                f"SELECT stay_id,event_minute,'{concept}' concept,{column} AS value FROM longitudinal_wide WHERE {column} IS NOT NULL" for concept,column in [('spo2','spo2'),('map','map_value'),('hr','hr')]))
        else:
            source=csv_source(dataset,'chartevents')
            con.execute(f"""CREATE OR REPLACE TABLE mimic_raw_vitals AS
              SELECT c.stay_id,date_diff('second',c.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
                m.concept,try_cast(v.valuenum AS DOUBLE) AS value
              FROM {source} v JOIN prior.mimic_cohort c ON try_cast(v.stay_id AS BIGINT)=c.stay_id
              JOIN (SELECT DISTINCT stay_id FROM mimic_timing_intervals) selected ON c.stay_id=selected.stay_id
              JOIN vital_map m ON try_cast(v.itemid AS BIGINT)=m.itemid
              WHERE try_cast(v.charttime AS TIMESTAMP)>=c.admit_time AND try_cast(v.charttime AS TIMESTAMP)<c.admit_time+INTERVAL '28 hours'
                AND coalesce(try_cast(v.warning AS DOUBLE),0)<>1 AND try_cast(v.valuenum AS DOUBLE) IS NOT NULL""")
        con.execute(f"""CREATE OR REPLACE TABLE {d}_vitals AS SELECT stay_id,event_minute,concept,median(value) AS value
          FROM {d}_raw_vitals WHERE (concept='spo2' AND value BETWEEN 50 AND 100)
            OR (concept='map' AND value BETWEEN 20 AND 200) OR (concept='hr' AND value BETWEEN 20 AND 250)
          GROUP BY stay_id,event_minute,concept""")
        # The existing first-four-hour extraction provides an independently materialized source check.
        actual=con.execute(f'SELECT * FROM {d}_vitals WHERE event_minute<240 ORDER BY stay_id,event_minute,concept').df()
        expected=con.execute(f"""SELECT stay_id,event_minute,concept,median(value) AS value FROM prior.{d}_vitals
          WHERE stay_id IN (SELECT DISTINCT stay_id FROM {d}_timing_intervals) AND event_minute>=0 AND event_minute<240
            AND ((concept='spo2' AND value BETWEEN 50 AND 100) OR (concept='map' AND value BETWEEN 20 AND 200)
                 OR (concept='hr' AND value BETWEEN 20 AND 250))
          GROUP BY stay_id,event_minute,concept ORDER BY stay_id,event_minute,concept""").df()
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=0,atol=1e-12)
        extracted.append(dict(dataset=d,raw_records=con.execute(f'SELECT count(*) FROM {d}_raw_vitals').fetchone()[0],
          deduplicated_valid_records=con.execute(f'SELECT count(*) FROM {d}_vitals').fetchone()[0],first_four_hour_source_rows=len(actual),first_four_hour_matches_existing_extraction=True))
        con.execute('CHECKPOINT');wj('vital_extraction_counts.json',extracted)
    for item in raw_refs.values():
        p=Path(item['path']);s=p.stat();assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns']
        item['sha256']=wh(p)
        s=p.stat();assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns']
    assert wh(prior_path)==prior_ref['sha256']
    wj('vital_extraction_manifest.json',dict(complete=True,prior=prior_ref,raw=raw_refs,mapping=mapping,
      mapping_source_sha256=wh(PROJECT/'src/physiograph/etl/mimic_extractor.py'),config_sha256=wh(PROJECT/'configs/default.yaml'),counts=extracted))
    ws('longitudinal vital extraction complete; exposure gate pending','extraction_complete')
    print(json.dumps(extracted,indent=2))
'''

FEATURES = r'''
exposure_support=[];feature_validation=[]
exposure_pass=False
if not timing_pass:
    print('Exposure features skipped: the joint timing gate failed.')
else:
    ws('longitudinal oxygen features and within-person exposure gate')
    extraction=json.loads((OUT/'vital_extraction_manifest.json').read_text());assert extraction['complete']
    for item in extraction['raw'].values():
        s=Path(item['path']).stat();assert s.st_size==item['bytes'] and s.st_mtime_ns==item['mtime_ns']
    for d in ('eicu','mimic'):
        con.execute(f"""CREATE OR REPLACE TABLE {d}_window_bins AS
          SELECT p.stay_id,p.event_minute window_end,v.concept,
            floor((v.event_minute-(p.event_minute-240))/15)::INTEGER bin,
            median(v.value) AS value,median(v.event_minute) representative_minute,count(*) n_readings
          FROM {d}_timing_intervals p JOIN {d}_vitals v ON p.stay_id=v.stay_id
            AND v.event_minute>=p.event_minute-240 AND v.event_minute<p.event_minute
          GROUP BY p.stay_id,p.event_minute,v.concept,bin""")
        sql=con.execute(f"""WITH lagged AS (SELECT *,lag(value) OVER w prior_value,lag(representative_minute) OVER w prior_minute
            FROM {d}_window_bins WINDOW w AS(PARTITION BY stay_id,window_end,concept ORDER BY bin)),
          features AS (SELECT stay_id,window_end,
            count(*) FILTER(WHERE concept='spo2') spo2_bins,
            sum(n_readings) FILTER(WHERE concept='spo2') spo2_readings,
            avg(value) FILTER(WHERE concept='spo2') spo2_mean,
            avg(CASE WHEN value<90 THEN 1.0 ELSE 0.0 END) FILTER(WHERE concept='spo2') hypoxic_fraction,
            avg(value) FILTER(WHERE concept='map') map_mean,avg(value) FILTER(WHERE concept='hr') hr_mean,
            count(*) FILTER(WHERE concept='spo2' AND representative_minute>prior_minute AND representative_minute-prior_minute<=30) transitions,
            max(CASE WHEN concept='spo2' AND representative_minute>prior_minute AND representative_minute-prior_minute<=30
              AND abs(value-prior_value)>=4 THEN 1 ELSE 0 END) unstable
            FROM lagged GROUP BY stay_id,window_end)
          SELECT p.stay_id,p.event_minute window_end,coalesce(f.spo2_bins,0) spo2_bins,coalesce(f.spo2_readings,0) spo2_readings,
            f.spo2_mean,f.hypoxic_fraction,f.map_mean,f.hr_mean,coalesce(f.transitions,0) transitions,coalesce(f.unstable,0) unstable
          FROM {d}_timing_intervals p LEFT JOIN features f ON p.stay_id=f.stay_id AND p.event_minute=f.window_end
          ORDER BY p.stay_id,p.event_minute""").df()
        vitals=con.execute(f'SELECT * FROM {d}_vitals ORDER BY stay_id,event_minute,concept').df()
        pairs=con.execute(f'SELECT * FROM {d}_timing_intervals ORDER BY stay_id,event_minute').df()
        by_stay={sid:g for sid,g in vitals.groupby('stay_id')};pyrows=[];pybins=[]
        for r in pairs.itertuples():
            g=by_stay.get(r.stay_id,vitals.iloc[:0]);g=g.loc[g.event_minute.ge(r.event_minute-240)&g.event_minute.lt(r.event_minute)].copy()
            g['bin']=np.floor((g.event_minute-(r.event_minute-240))/15).astype(int)
            b=g.groupby(['concept','bin'],as_index=False).agg(value=('value','median'),representative_minute=('event_minute','median'),n_readings=('event_minute','size'))
            b['stay_id']=r.stay_id;b['window_end']=r.event_minute;pybins.append(b)
            s=b.loc[b.concept.eq('spo2')].sort_values('bin');dt=np.diff(s.representative_minute);dv=np.diff(s.value)
            transitions=(dt>0)&(dt<=30)
            pyrows.append(dict(stay_id=r.stay_id,window_end=r.event_minute,spo2_bins=len(s),spo2_readings=int(s.n_readings.sum()),
              spo2_mean=float(s.value.mean()),hypoxic_fraction=float(s.value.lt(90).mean()) if len(s) else np.nan,
              map_mean=float(b.loc[b.concept.eq('map'),'value'].mean()),hr_mean=float(b.loc[b.concept.eq('hr'),'value'].mean()),
              transitions=int(transitions.sum()),unstable=int(((np.abs(dv)>=4)&transitions).any())))
        py=pd.DataFrame(pyrows)
        pd.testing.assert_frame_equal(sql,py[sql.columns],check_dtype=False,rtol=0,atol=1e-12)
        bins=con.execute(f'SELECT * FROM {d}_window_bins ORDER BY stay_id,window_end,concept,bin').df()
        pyb=pd.concat(pybins,ignore_index=True).sort_values(['stay_id','window_end','concept','bin']).reset_index(drop=True)
        pd.testing.assert_frame_equal(bins,pyb[bins.columns],check_dtype=False,rtol=0,atol=1e-12)
        f=pairs.merge(sql,left_on=['stay_id','event_minute'],right_on=['stay_id','window_end'],validate='one_to_one')
        f['oxygen_qualified']=f.spo2_bins.ge(3)&f.transitions.ge(2)
        eligible=f.loc[f.oxygen_qualified].copy();n=eligible.groupby('person_id').size()
        analysis=eligible.loc[eligible.person_id.isin(n[n.ge(2)].index)].copy()
        people=int(analysis.person_id.nunique());switchers=int(analysis.groupby('person_id').unstable.nunique().eq(2).sum())
        exposure_support.append(dict(dataset=d,timing_intervals=len(pairs),oxygen_qualified_intervals=len(eligible),analysis_intervals=len(analysis),
          analysis_people=people,instability_switchers=switchers,unstable_intervals=int(analysis.unstable.sum()),stable_intervals=int(analysis.unstable.eq(0).sum()),
          spo2_bins_quantiles=[float(x) for x in analysis.spo2_bins.quantile([.25,.5,.75])] if len(analysis) else None,
          minimum_people=100,minimum_switchers=50,passes=people>=100 and switchers>=50))
        con.register('window_features',f);con.execute(f'CREATE OR REPLACE TABLE {d}_window_features AS SELECT * FROM window_features');con.unregister('window_features')
        con.register('analysis_intervals',analysis);con.execute(f'CREATE OR REPLACE TABLE {d}_analysis_intervals AS SELECT * FROM analysis_intervals');con.unregister('analysis_intervals')
        feature_validation.append(dict(dataset=d,all_sql_python_bins_and_features_match=True,intervals=len(pairs),bins=len(bins),troponin_changes_calculated=False))
    exposure_pass=all(r['passes'] for r in exposure_support)
    wj('exposure_support.json',exposure_support);wj('feature_validation.json',feature_validation)
    con.execute('CHECKPOINT');ws('exposure gate evaluated','exposure_complete')
    print(json.dumps(exposure_support,indent=2))
'''

MODELS = r'''
models=[];model_validation=[]
if timing_pass and exposure_pass:
    from scipy.stats import t as student_t
    from statsmodels.regression.linear_model import OLS
    def within_fit(x,y,groups):
        x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);groups=np.asarray(groups)
        unique,inv=np.unique(groups,return_inverse=True);g=len(unique);n,k=x.shape
        sizes=np.bincount(inv);sx=np.zeros((g,k));sy=np.zeros(g)
        np.add.at(sx,inv,x);np.add.at(sy,inv,y)
        xd=x-sx[inv]/sizes[inv,None];yd=y-sy[inv]/sizes[inv]
        fit=OLS(yd,xd).fit(cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True},use_t=True)
        adjusted_cov=np.asarray(fit.cov_params())*((n-k)/(n-g-k))
        # Independent Schur-complement solve from uncentered sufficient statistics.
        gram=x.T@x-(sx.T/sizes)@sx;cross=x.T@y-(sx.T/sizes)@sy
        beta=np.linalg.solve(gram,cross);bread=np.linalg.inv(gram)
        effects=(sy-sx@beta)/sizes;residual=y-x@beta-effects[inv]
        scores=np.zeros((g,k));np.add.at(scores,inv,x*residual[:,None])
        covariance=bread@(scores.T@scores)@bread*(g/(g-1))*((n-1)/(n-g-k))
        np.testing.assert_allclose(beta,fit.params,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(covariance,adjusted_cov,rtol=1e-5,atol=1e-8)
        assert np.all(np.diag(covariance)>=0) and np.isfinite(covariance).all()
        return beta,covariance,dict(people=g,intervals=n,slopes=k,residual_df=n-g-k,
          max_coefficient_error=float(np.max(np.abs(beta-fit.params))),max_covariance_error=float(np.max(np.abs(covariance-adjusted_cov))))
    rng=np.random.default_rng(5906);sg=np.repeat(np.arange(8),5);sx=rng.normal(size=(40,3));sy=sx@np.array([.3,-.2,.1])+sg/10+rng.normal(size=40)
    sb,sc,sv=within_fit(sx,sy,sg)
    dummy=pd.get_dummies(sg,dtype=float).to_numpy();full=OLS(sy,np.column_stack([dummy,sx])).fit(cov_type='cluster',cov_kwds={'groups':sg,'use_correction':True},use_t=True)
    np.testing.assert_allclose(sb,full.params[-3:],rtol=1e-9,atol=1e-10);np.testing.assert_allclose(sc,np.asarray(full.cov_params())[-3:,-3:],rtol=1e-9,atol=1e-10)
    wj('synthetic_fixed_effect_validation.json',dict(full_dummy_and_within_coefficients_and_covariance_match=True,clinical_records=False))
    for d in ('eicu','mimic'):
        ws(d+' frozen within-person association')
        a=con.execute(f'SELECT * FROM {d}_analysis_intervals ORDER BY person_id,event_minute').df()
        x=pd.DataFrame({name:a[col].to_numpy(dtype=float) for name,col in [('instability','unstable'),('spo2_mean','spo2_mean'),('hypoxic_fraction','hypoxic_fraction'),('map_mean','map_mean'),('hr_mean','hr_mean')]})
        x['log_readings']=np.log1p(a.spo2_readings.to_numpy());x['interval_hours']=(a.event_minute-a.prior_event_minute).to_numpy()/60
        x['end_hours']=a.event_minute.to_numpy()/60;x['end_hours_squared']=x.end_hours**2
        x['log_baseline']=np.log(a.prior_magnitude.astype(float).to_numpy())
        imputation={}
        for c in ['map_mean','hr_mean']:
            miss=x[c].isna()
            if miss.any():
                median=float(x.loc[~miss,c].median()) if (~miss).any() else 0.
                x[c+'_missing']=miss.astype(float);x[c]=x[c].fillna(median);imputation[c]=dict(median=median,missing=int(miss.sum()))
        means=x.groupby(a.person_id.to_numpy()).transform('mean');xd=x-means
        drop=[c for c in x if c!='instability' and np.max(np.abs(xd[c]))<=1e-12];x=x.drop(columns=drop);xd=xd.drop(columns=drop)
        n,k=x.shape;g=a.person_id.nunique();rank=int(np.linalg.matrix_rank(xd))
        info=dict(dataset=d,model_fitted=False,people=int(g),intervals=n,slopes=k,columns=list(x),removed_no_within_variation=drop,imputation=imputation,rank=rank,gate_failures=[])
        if not np.isfinite(x).all().all() or rank<k:info['gate_failures'].append('nonfinite or rank-deficient within-person design')
        if n<10*k or n-g-k<=0:info['gate_failures'].append('insufficient observations or residual degrees of freedom')
        if not info['gate_failures']:
            y=np.array([np.log(float(Decimal(v)/Decimal(b))) for v,b in zip(a.magnitude,a.prior_magnitude)])
            assert np.isfinite(y).all()
            beta,cov,check=within_fit(x.to_numpy(),y,a.person_id.to_numpy());j=list(x).index('instability')
            b=float(beta[j]);se=float(np.sqrt(cov[j,j]));critical=float(student_t.ppf(.975,g-1));lo=b-critical*se;hi=b+critical*se
            info.update(model_fitted=True,log_coefficient=b,log_ci95=[lo,hi],fold_change_ratio=float(np.exp(b)),ratio_ci95=[float(np.exp(lo)),float(np.exp(hi))],
              p=float(2*student_t.sf(abs(b/se),g-1)),cluster_df=int(g-1),residual_df=int(n-g-k))
            model_validation.append(dict(dataset=d,**check))
            a['log_troponin_change']=y;a['row_index']=np.arange(len(a))
            con.register('modeled_intervals',a);con.execute(f'CREATE OR REPLACE TABLE {d}_modeled_intervals AS SELECT * FROM modeled_intervals');con.unregister('modeled_intervals')
            saved_design=x.assign(row_index=np.arange(len(x)))
            con.register('saved_design',saved_design);con.execute(f'CREATE OR REPLACE TABLE {d}_model_design AS SELECT * FROM saved_design');con.unregister('saved_design')
        models.append(info)
    last=0
    for i,m in enumerate(sorted(models,key=lambda m:m.get('p',1))):last=max(last,min(1,(2-i)*m.get('p',1)));m['holm_p']=last
else:
    print('Troponin changes and association models skipped: a joint support gate failed.')
wj('within_person_models.json',models);wj('model_validation.json',model_validation)
for ref in refs.values():assert wh(Path(ref['path']))==ref['sha256']
wj('analysis_manifest.json',dict(protocol_sha256=WITHIN_PERSON_TROPONIN_PROTOCOL_SHA256,script_sha256=wh(PROJECT/'scripts/within_person_troponin_cells.py'),
  timing_gate_pass=timing_pass,exposure_gate_pass=exposure_pass,models_fitted=sum(m['model_fitted'] for m in models),
  mortality_accessed=False,existing_endpoints_replaced=False,environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# Within-person oxygen instability and troponin change','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local incremental notebook cells')+'. Original MIMIC/eICU patients only.',
  '', '| Dataset | Timing-eligible people | Timing intervals |','|---|---:|---:|']
for r in support:report.append(f"| {r['dataset']} | {r['people']} | {r['intervals']} |")
if timing_pass:
    report+=['','| Dataset | People with repeated usable oxygen windows | Intervals | People changing instability status | Gate |','|---|---:|---:|---:|---|']
    for r in exposure_support:report.append(f"| {r['dataset']} | {r['analysis_people']} | {r['analysis_intervals']} | {r['instability_switchers']} | {'pass' if r['passes'] else 'fail'} |")
report+=['']
for m in models:
    if m['model_fitted']:report.append(f"- {m['dataset']}: adjusted within-person ratio of troponin fold changes {m['fold_change_ratio']:.4f} (95% CI {m['ratio_ci95'][0]:.4f}–{m['ratio_ci95'][1]:.4f}); Holm p={m['holm_p']:.6g}. {m['people']} patients, {m['intervals']} intervals. Ratio >1 indicates larger observed troponin change during unstable oxygen windows.")
    else:report.append(f"- {m['dataset']}: model not fitted: {'; '.join(m['gate_failures'])}.")
replicated=len(models)==2 and all(m['model_fitted'] and m['log_coefficient']>0 and m['holm_p']<.05 for m in models)
if not timing_pass or not exposure_pass:report.append('A frozen joint support gate failed. No troponin changes or association models were calculated; no thresholds were relaxed.')
elif replicated:report.append('The exploratory positive statistical replication criterion is met. This remains an observational within-person association, not a new mechanism or established therapeutic benefit.')
else:report.append('The frozen positive replication criterion is not met. This does not establish equivalence or absence of an association.')
report+=['','Independent SQL/Python specimen and oxygen-feature reconstructions agree. When models are fitted, absorbed-effect and full-dummy synthetic checks, independent sufficient-statistic coefficients and patient-clustered covariance are validated. Raw-source fingerprints and original first-four-hour extraction agreement are recorded separately.',
  '', 'The planned fixed-effects design addresses stable patient differences when it can be fitted. Short-panel dynamic bias from the prior-troponin covariate is an additional material limitation identified before estimates; see SPO2_WITHIN_PERSON_LAG_AMENDMENT.md. Correct standard errors do not correct that bias. Time-varying treatment, congestion, ischemia, renal clearance, selective testing, delayed troponin release and sparse chart resolution remain unresolved. No mortality outcome was accessed. Neither a causal mechanism nor a mortality benefit is established. The discovery goal remains unmet.']
(PROJECT/'docs/SPO2_WITHIN_PERSON_TROPONIN_RESULTS.md').write_text('\n'.join(report)+'\n')
con.execute('CHECKPOINT');ws('within-person specification completed','complete');print('\n'.join(report));con.close()
'''

LAG_SENSITIVITY = r'''
assert wh(PROJECT/'docs/SPO2_WITHIN_PERSON_LAG_AMENDMENT.md')==WITHIN_PERSON_LAG_AMENDMENT_SHA256
lock=OUT/'lag_amendment_lock.json'
assert lock.exists() and json.loads(lock.read_text())['sha256']==WITHIN_PERSON_LAG_AMENDMENT_SHA256
ws('pre-estimate lag sensitivity')
con=duckdb.connect(str(PRIVATE/'within_person_troponin.duckdb'))
lag_models=[];lag_checks=[]
for primary in models:
    d=primary['dataset'];info=dict(dataset=d,model_fitted=False,primary_model_fitted=primary['model_fitted'])
    if primary['model_fitted']:
        a=con.execute(f'SELECT row_index,person_id,log_troponin_change FROM {d}_modeled_intervals ORDER BY row_index').df()
        saved=con.execute(f'SELECT * FROM {d}_model_design ORDER BY row_index').df()
        assert np.array_equal(saved.row_index,a.row_index)
        x=saved.drop(columns=['row_index','log_baseline'],errors='ignore')
        assert list(x)==[c for c in primary['columns'] if c!='log_baseline']
        beta,cov,check=within_fit(x.to_numpy(),a.log_troponin_change.to_numpy(),a.person_id.to_numpy())
        j=list(x).index('instability');b=float(beta[j]);se=float(np.sqrt(cov[j,j]));g=a.person_id.nunique();crit=float(student_t.ppf(.975,g-1))
        info.update(model_fitted=True,people=int(g),intervals=len(a),columns=list(x),log_coefficient=b,
          fold_change_ratio=float(np.exp(b)),ratio_ci95=[float(np.exp(b-crit*se)),float(np.exp(b+crit*se))],
          p=float(2*student_t.sf(abs(b/se),g-1)),primary_log_coefficient=primary['log_coefficient'],
          secondary_minus_primary_log_coefficient=b-primary['log_coefficient'],same_primary_rows_and_outcomes=True)
        lag_checks.append(dict(dataset=d,**check))
    else:info['reason']='Primary model did not fit; sensitivity cannot rescue it.'
    lag_models.append(info)
last=0
for i,m in enumerate(sorted(lag_models,key=lambda m:m.get('p',1))):last=max(last,min(1,(2-i)*m.get('p',1)));m['secondary_holm_p']=last
wj('lag_sensitivity_models.json',lag_models);wj('lag_sensitivity_validation.json',lag_checks)
wj('lag_sensitivity_manifest.json',dict(amendment_sha256=WITHIN_PERSON_LAG_AMENDMENT_SHA256,models_fitted=sum(m['model_fitted'] for m in lag_models),
  primary_results_replaced=False,sensitivity_is_validated_bias_correction=False,goal_complete=False))
extra=['','## Prespecified lag-adjustment sensitivity','', 'This diagnostic removes only the previous-troponin covariate on the same primary rows. It is not a validated bias correction and cannot rescue the primary replication criterion.']
for m in lag_models:
    if m['model_fitted']:extra.append(f"- {m['dataset']}: ratio {m['fold_change_ratio']:.4f} (95% CI {m['ratio_ci95'][0]:.4f}–{m['ratio_ci95'][1]:.4f}); secondary Holm p={m['secondary_holm_p']:.6g}.")
    else:extra.append(f"- {m['dataset']}: skipped because its primary model was not fitted.")
if not lag_models:extra.append('No primary models were fitted because a joint support gate failed; the diagnostic is also skipped.')
path=PROJECT/'docs/SPO2_WITHIN_PERSON_TROPONIN_RESULTS.md'
path.write_text(path.read_text().split('\n## Prespecified lag-adjustment sensitivity')[0].rstrip()+'\n'+'\n'.join(extra)+'\n')
ws('within-person primary and lag sensitivity completed','complete');print('\n'.join(extra));con.close()
'''
