"""Native calcium ratios with stable unit metadata; separate frozen study."""

SETUP = r"""
import json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_calcium_relative';OUT.mkdir(parents=True,exist_ok=True)
def cr_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def cr_q(x):return "'"+str(x).replace("'","''")+"'"
def cr_hash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert cr_hash(PROJECT/'docs/SPO2_CALCIUM_RELATIVE_PLAN.md')==CALCIUM_RELATIVE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==CALCIUM_RELATIVE_PROTOCOL_SHA256
else:cr_json('protocol_lock.json',dict(sha256=CALCIUM_RELATIVE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),original_support_and_attrition_known=True,calcium_ph_contrasts_not_known=True))
original=json.loads((PROJECT/'research/spo2_ionized_calcium/results.json').read_text());assert not original['changes_computed']
try:con.close()
except (NameError,AttributeError):pass
origin=PRIVATE/'ionized_calcium.duckdb';origin_hash=cr_hash(origin)
con=duckdb.connect(str(PRIVATE/'calcium_relative.duckdb'));con.execute('SET threads=2');con.execute('SET enable_progress_bar=false')
con.execute(f'ATTACH {cr_q(origin)} AS origin (READ_ONLY)')
cr_json('source_manifest.json',dict(database=str(origin),sha256=origin_hash,original_results=original,environment='google_colab' if IN_COLAB else 'local',original_manifest=json.loads((PROJECT/'research/spo2_ionized_calcium/source_manifest.json').read_text())))
cr_json('run_status.json',dict(status='running',stage='native-ratio protocol locked before effects'))
print('Separate unit-invariant calcium protocol locked.')
"""

QUALIFY = r"""
def cr_token(x):return '' if pd.isna(x) else ''.join(str(x).casefold().split())
factors={'mg/dl':4.0078,'mmol/l':1.,'meq/l':2.};bits={'mg/dl':1,'mmol/l':2,'meq/l':4}
support=[];checks=[]
for d in ['eicu','mimic']:
    raw=con.execute(f"SELECT * FROM origin.{d}_raw WHERE assay='calcium'").df();raw['us']=raw.unit_system.map(cr_token);raw['ui']=raw.unit_interface.map(cr_token)
    raw['censored']=raw.raw_text.fillna('').str.contains(r'^\s*(?:[<>≤≥]|less\s+than|greater\s+than)',case=False,regex=True)
    rows=[]
    for (sid,t),g in raw.groupby(['stay_id','event_minute'],sort=True):
        latest=g.loc[g.available_minute.eq(g.available_minute.max())];unit_unique=len(g[['us','ui']].drop_duplicates())==1
        us=g.us.iloc[0];ui=g.ui.iloc[0];allowed=(us=='mg/dl' and ui in factors) if d=='eicu' else (us=='mmol/l' and ui=='')
        v=float(latest.raw_value.iloc[0]) if len(latest) else np.nan
        mask=sum(bits[u] for u in {us,ui} if u in factors and np.isfinite(v) and .1<=v/factors[u]<=3.) if allowed else 0
        specimen_ok=d=='eicu' or (g.specimen_id.notna().all() and g.specimen_id.nunique()==1)
        ok=bool(unit_unique and allowed and mask>0 and specimen_ok and g.available_minute.notna().all() and len(latest)>0 and np.isfinite(latest.raw_value).all() and not latest.censored.any() and latest.raw_value.nunique()==1)
        rows.append(dict(stay_id=sid,event_minute=t,us=us,ui=ui,qualified=ok,possible_units=mask,value=v if ok else np.nan,specimen_id=int(g.specimen_id.iloc[0]) if d=='mimic' and specimen_ok else np.nan))
    quality=pd.DataFrame(rows);con.register('cr_quality',quality);con.execute(f'CREATE OR REPLACE TABLE {d}_quality AS SELECT * FROM cr_quality');con.unregister('cr_quality')
    allowed_sql="us='mg/dl' AND ui IN ('mg/dl','mmol/l','meq/l')" if d=='eicu' else "us='mmol/l' AND ui=''"
    sql=con.execute(f'''WITH r AS (SELECT *,lower(regexp_replace(coalesce(unit_system,''),'\\s','','g')) us,lower(regexp_replace(coalesce(unit_interface,''),'\\s','','g')) ui,
      max(available_minute) OVER(PARTITION BY stay_id,event_minute) latest FROM origin.{d}_raw WHERE assay='calcium'),
      g AS (SELECT stay_id,event_minute,min(us) us,min(ui) ui,count(DISTINCT (us,ui))=1 unit_unique,
      count(*)=count(available_minute) times_ok,count(*)=count(specimen_id) AND count(DISTINCT specimen_id)=1 specimen_ok,
      bool_and(coalesce(isfinite(raw_value) AND NOT regexp_matches(coalesce(raw_text,''),'^\\s*([<>≤≥]|less\\s+than|greater\\s+than)','i'),false)) FILTER(WHERE available_minute=latest) latest_ok,
      count(DISTINCT raw_value) FILTER(WHERE available_minute=latest)=1 one_value,min(raw_value) FILTER(WHERE available_minute=latest) AS value
      FROM r GROUP BY stay_id,event_minute),
      a AS (SELECT *,CASE WHEN ({allowed_sql}) THEN
      CASE WHEN (us='mg/dl' OR ui='mg/dl') AND value/4.0078 BETWEEN .1 AND 3. THEN 1 ELSE 0 END+
      CASE WHEN (us='mmol/l' OR ui='mmol/l') AND value BETWEEN .1 AND 3. THEN 2 ELSE 0 END+
      CASE WHEN (us='meq/l' OR ui='meq/l') AND value/2. BETWEEN .1 AND 3. THEN 4 ELSE 0 END ELSE 0 END possible_units FROM g)
      SELECT stay_id,event_minute,coalesce(unit_unique AND ({allowed_sql}) AND possible_units>0 AND times_ok AND {'true' if d=='eicu' else 'specimen_ok'} AND latest_ok AND one_value,false) qualified,possible_units,value
      FROM a ORDER BY stay_id,event_minute''').df()
    assert np.array_equal(quality[['stay_id','event_minute','qualified']],sql[['stay_id','event_minute','qualified']])
    m=quality.qualified;assert np.array_equal(quality.loc[m,'possible_units'],sql.loc[m,'possible_units']) and np.allclose(quality.loc[m,'value'],sql.loc[m,'value'],rtol=0,atol=1e-12)
    con.execute(f'''CREATE OR REPLACE TABLE {d}_analysis AS SELECT p.*,a.value baseline_calcium,b.value followup_calcium,pa.value baseline_ph,pb.value followup_ph,
      a.us,a.ui,(a.possible_units&b.possible_units) common_units FROM origin.{d}_pairs p
      JOIN {d}_quality a ON a.stay_id=p.stay_id AND a.event_minute=p.baseline_minute AND a.qualified
      JOIN {d}_quality b ON b.stay_id=p.stay_id AND b.event_minute=p.followup_minute AND b.qualified AND a.us=b.us AND a.ui=b.ui AND (a.possible_units&b.possible_units)>0
      JOIN origin.{d}_quality pa ON pa.stay_id=p.stay_id AND pa.event_minute=p.baseline_minute AND pa.assay='ph' AND pa.qualified
      JOIN origin.{d}_quality pb ON pb.stay_id=p.stay_id AND pb.event_minute=p.followup_minute AND pb.assay='ph' AND pb.qualified
      {'WHERE a.specimen_id=pa.specimen_id AND b.specimen_id=pb.specimen_id' if d=='mimic' else ''}''')
    p=con.execute(f'SELECT * FROM {d}_analysis ORDER BY stay_id').df()
    ph=con.execute(f"SELECT * FROM origin.{d}_quality WHERE assay='ph' AND qualified").df()
    ca=quality.loc[quality.qualified].merge(ph,on=['stay_id','event_minute'],suffixes=('_ca','_ph'))
    if d=='mimic':ca=ca.loc[ca.specimen_id_ca.eq(ca.specimen_id_ph)]
    ca=ca[['stay_id','event_minute','us','ui','possible_units','value_ca','value_ph']]
    pairs=con.execute(f'SELECT * FROM origin.{d}_pairs').df()
    py=pairs.merge(ca.rename(columns={'event_minute':'baseline_minute'}),on=['stay_id','baseline_minute']).merge(ca.rename(columns={'event_minute':'followup_minute'}),on=['stay_id','followup_minute'],suffixes=('_pre','_post'))
    py=py.loc[py.us_pre.eq(py.us_post)&py.ui_pre.eq(py.ui_post)&(py.possible_units_pre&py.possible_units_post).gt(0)].sort_values('stay_id')
    assert np.array_equal(p.stay_id,py.stay_id) and np.allclose(p[['baseline_calcium','followup_calcium','baseline_ph','followup_ph']],py[['value_ca_pre','value_ca_post','value_ph_pre','value_ph_post']],rtol=0,atol=1e-12)
    ne=int(p.exposed.sum());nu=len(p)-ne;h=int(p.hospital_id.nunique());gate=len(p)>=50 and ne>=20 and nu>=20 and (d!='eicu' or (h>=10 and p.hospital_id.notna().all()))
    support.append(dict(dataset=d,people=len(p),exposed=ne,unexposed=nu,hospitals=h if d=='eicu' else None,gate=bool(gate)))
    checks.append(dict(dataset=d,sql_python_quality_and_possible_units=True,sql_python_fixed_pairs=True))
joint_gate=all(s['gate'] for s in support);cr_json('support.json',support);cr_json('qualification_validation.json',checks)
print('Native-ratio paired support:',support)
"""

SCREEN = r"""
def cr_screen():
    if not joint_gate:
        result=dict(joint_gate=False,changes_computed=False,biological_discovery=False);cr_json('results.json',result)
        (PROJECT/'docs/SPO2_CALCIUM_RELATIVE_RESULTS.md').write_text('# Unit-invariant calcium follow-up\n\nFixed support failed; no exposure contrasts computed.\n'+json.dumps(support,indent=2)+'\n')
        return result
    estimates=[];checks=[];means=[]
    for d in ['eicu','mimic']:
        p=con.execute(f'SELECT *,ln(followup_calcium/baseline_calcium) change_calcium,followup_ph-baseline_ph change_ph FROM {d}_analysis ORDER BY stay_id').df()
        v=p[['change_calcium','change_ph']].to_numpy();exp=p.exposed.to_numpy(int)
        assert np.allclose(v[:,0],np.log(p.followup_calcium/p.baseline_calcium),rtol=0,atol=1e-12)
        cancellations=0
        for unit,factor in factors.items():
            hit=(p.common_units.to_numpy(int)&bits[unit])>0
            assert np.allclose(v[hit,0],np.log((p.loc[hit,'followup_calcium']/factor)/(p.loc[hit,'baseline_calcium']/factor)),rtol=0,atol=1e-12);cancellations+=int(hit.sum())
        codes=pd.Categorical(p.hospital_id,categories=sorted(p.hospital_id.unique())).codes if d=='eicu' else np.arange(len(p));G=int(codes.max())+1;totals=np.zeros((G,6))
        for arm in [0,1]:
            for j in [0,1]:np.add.at(totals[:,arm*3+j],codes[exp==arm],v[exp==arm,j])
            np.add.at(totals[:,arm*3+2],codes[exp==arm],1);means.append(dict(dataset=d,exposed=arm,people=int((exp==arm).sum()),mean_log_calcium_change=float(v[exp==arm,0].mean()),mean_ph_change=float(v[exp==arm,1].mean())))
        point=v[exp==1].mean(axis=0)-v[exp==0].mean(axis=0)
        rng=np.random.default_rng({'eicu':2026090567,'mimic':2026090568}[d]);draws=[];checked=0;max_diff=0.
        for start in range(0,20000,500):
            w=rng.multinomial(G,np.repeat(1/G,G),size=500);s=w@totals
            with np.errstate(divide='ignore',invalid='ignore'):b=s[:,3:5]/s[:,5:6]-s[:,:2]/s[:,2:3]
            for weights,expected in zip(w[:max(0,100-checked)],b[:max(0,100-checked)]):
                idx=np.repeat(np.arange(len(p)),weights[codes]);a=exp[idx];z=v[idx];direct=z[a==1].mean(axis=0)-z[a==0].mean(axis=0) if (a==0).any() and (a==1).any() else np.array([np.nan,np.nan])
                assert np.allclose(direct,expected,rtol=0,atol=1e-10,equal_nan=True)
                if np.isfinite(direct).all():max_diff=max(max_diff,float(np.max(abs(direct-expected))))
                checked+=1
            draws.append(b)
        draws=np.concatenate(draws);valid=bool(np.isfinite(draws).all());assert checked==100
        for j,(name,expr) in enumerate([('calcium_log_ratio','ln(followup_calcium/baseline_calcium)'),('ph_change','followup_ph-baseline_ph')]):
            sqlpoint=con.execute(f'SELECT avg({expr}) FILTER(WHERE exposed=1)-avg({expr}) FILTER(WHERE exposed=0) FROM {d}_analysis').fetchone()[0];assert abs(sqlpoint-point[j])<1e-12
            r=dict(dataset=d,analyte=name,contrast=float(point[j]),bootstrap_valid=valid)
            if valid:
                lo,hi,lc,hc=np.quantile(draws[:,j],[.025,.975,.00625,.99375]);r.update(lower95=float(lo),upper95=float(hi),lower_simultaneous=float(lc),upper_simultaneous=float(hc),advance=bool(hc<0 if j==0 else lc>0))
                if j==0:r.update(geometric_fold_change_ratio=float(np.exp(point[j])),ratio_lower_simultaneous=float(np.exp(lc)),ratio_upper_simultaneous=float(np.exp(hc)))
            else:r['advance']=False
            estimates.append(r)
        checks.append(dict(dataset=d,sql_python_log_ratios_and_contrasts=True,shared_unit_cancellations_verified=cancellations,draws=20000,directly_expanded_draws=checked,max_absolute_direct_difference=max_diff,resampling='hospital' if d=='eicu' else 'person'))
    result=dict(joint_gate=True,changes_computed=True,estimates=estimates,group_means=means,advancement_gate_pass=all(r['advance'] for r in estimates),biological_discovery=False,mortality_benefit_established=False)
    cr_json('results.json',result);cr_json('bootstrap_validation.json',checks)
    report=['# Unit-invariant paired calcium and pH follow-up','','Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Separate protocol after absolute-unit support failed; original results retained.','',
      '| Dataset | Analyte | Change contrast | Nominal 95% CI | Simultaneous 98.75% CI |','|---|---|---:|---|---|']
    for r in estimates:
        if r['bootstrap_valid']:report.append(f"| {r['dataset']} | {r['analyte']} | {r['contrast']:.6f} | {r['lower95']:.6f} to {r['upper95']:.6f} | {r['lower_simultaneous']:.6f} to {r['upper_simultaneous']:.6f} |")
        else:report.append(f"| {r['dataset']} | {r['analyte']} | {r['contrast']:.6f} | Invalid bootstrap | Not promoted |")
    report+=['','Joint directional criterion: '+str(result['advancement_gate_pass'])+'.',
      'Native calcium ratios cancel a common multiplicative conversion; stable unit labels do not prove stable calibration. Unadjusted selected-patient associations do not establish causal calcium depletion/binding, respiratory alkalosis, cardiac dysfunction, novelty or mortality benefit.','']
    (PROJECT/'docs/SPO2_CALCIUM_RELATIVE_RESULTS.md').write_text('\n'.join(report));print('\n'.join(report))
    return result
result=cr_screen();assert cr_hash(origin)==origin_hash
cr_json('run_status.json',dict(status='complete',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False));con.close();print('Native calcium/pH result:',result)
"""
