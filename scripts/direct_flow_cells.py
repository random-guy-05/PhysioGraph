"""Direct cardiac-output metadata support in the original MIMIC cohort."""

SETUP = r"""
import json,hashlib,importlib.util,csv
from pathlib import Path
from datetime import datetime,timezone
import duckdb,numpy as np,pandas as pd
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_direct_flow';OUT.mkdir(parents=True,exist_ok=True)
def df_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def df_q(x):return "'"+str(x).replace("'","''")+"'"
def df_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert df_hash(PROJECT/'docs/SPO2_DIRECT_FLOW_PLAN.md')==DIRECT_FLOW_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==DIRECT_FLOW_PROTOCOL_SHA256
else:df_json('protocol_lock.json',dict(sha256=DIRECT_FLOW_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_flow_counts=True,earlier_project_results_known=True))
dictionary=DRIVE/'Data/MIMIC/Full/d_items.csv'
with dictionary.open() as f:lookup={int(r['itemid']):r for r in csv.DictReader(f)}
assert lookup[220088]['label']=='Cardiac Output (thermodilution)' and lookup[224842]['label']=='Cardiac Output (CCO)'
for i in [220088,224842]:assert lookup[i]['linksto']=='chartevents' and lookup[i]['unitname']=='L/min'
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'direct_flow.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f'ATTACH {df_q(PRIVATE/"spo2_acid_base.duckdb")} AS gas (READ_ONLY)')
con.execute('CREATE OR REPLACE TABLE original AS SELECT stay_id,person_id,hadm_id,admit_time,exposed FROM gas.mimic_eligible')
assert con.execute('SELECT count(*) FROM original').fetchone()[0]==4711
selected=con.execute('SELECT * FROM original QUALIFY row_number() OVER(PARTITION BY person_id ORDER BY stay_id)=1 ORDER BY stay_id').df()
selected['split']=[hashlib.sha256(('PhysioGraph direct flow v1|'+str(int(p))).encode()).digest()[0]%2 for p in selected.person_id]
con.register('df_selected',selected);con.execute('CREATE OR REPLACE TABLE selected AS SELECT * FROM df_selected');con.unregister('df_selected')
source=DRIVE/'Data/MIMIC/Full/chartevents.csv';stat=source.stat()
source_info=dict(path=str(source),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,dictionary_sha256=df_hash(dictionary))
df_json('run_status.json',dict(status='running',stage='protocol locked; full direct-flow metadata scan pending'))
print('Direct flow protocol and patient splits frozen before measurement counts.')
"""

EXTRACT = r"""
con.execute(f'''CREATE OR REPLACE TABLE raw AS SELECT e.stay_id,try_cast(v.itemid AS INTEGER) itemid,
 date_diff('second',e.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 event_minute,
 date_diff('second',e.admit_time,try_cast(v.storetime AS TIMESTAMP))/60.0 available_minute,
 v.valueuom unit,try_cast(v.warning AS INTEGER) warning
 FROM read_csv({df_q(source)},header=true,all_varchar=true,sample_size=10000,strict_mode=true) v
 JOIN original e ON try_cast(v.stay_id AS BIGINT)=e.stay_id AND try_cast(v.hadm_id AS BIGINT)=e.hadm_id
 AND try_cast(v.subject_id AS BIGINT)=e.person_id
 WHERE try_cast(v.itemid AS INTEGER) IN (220088,224842)
 AND date_diff('second',e.admit_time,try_cast(v.charttime AS TIMESTAMP))/60.0 BETWEEN 0 AND 1680''')
s=source.stat();assert (s.st_size,s.st_mtime_ns)==(source_info['bytes'],source_info['mtime_ns'])
source_info['sha256']=df_hash(source)
df_json('source_manifest.json',dict(source=source_info,full_raw_scan=True,cardiac_output_values_selected=False,mortality_selected=False))
df_json('run_status.json',dict(status='running',stage='full raw scan complete; independent support validation'))
print('Raw direct-flow metadata scan completed. No cardiac-output values selected.')
"""

SUPPORT = r"""
original=con.execute('SELECT * FROM original ORDER BY stay_id').df()
selected=con.execute('SELECT * FROM selected ORDER BY stay_id').df()
expected=original.sort_values('stay_id').groupby('person_id',sort=False).head(1)
assert np.array_equal(selected.stay_id,expected.stay_id)
assert np.array_equal(selected.split,[hashlib.sha256(('PhysioGraph direct flow v1|'+str(int(p))).encode()).digest()[0]%2 for p in selected.person_id])
con.execute('''CREATE OR REPLACE TABLE pairs AS SELECT e.stay_id,e.person_id,e.exposed,e.split,m.itemid,
 max(r.event_minute) FILTER(WHERE r.event_minute<=240) baseline_minute,
 min(r.event_minute) FILTER(WHERE r.event_minute>240) followup_minute
 FROM selected e CROSS JOIN (VALUES(220088),(224842)) m(itemid)
 LEFT JOIN raw r ON r.stay_id=e.stay_id AND r.itemid=m.itemid
 GROUP BY e.stay_id,e.person_id,e.exposed,e.split,m.itemid''')
pairs=con.execute('SELECT * FROM pairs ORDER BY stay_id,itemid').df()
raw=con.execute('SELECT * FROM raw').df()
times=raw[['stay_id','itemid','event_minute']].drop_duplicates()
pre=times.loc[times.event_minute.le(240)].groupby(['stay_id','itemid']).event_minute.max()
post=times.loc[times.event_minute.gt(240)].groupby(['stay_id','itemid']).event_minute.min()
for row in pairs.itertuples():
    key=(row.stay_id,row.itemid)
    for observed,lookup in [(row.baseline_minute,pre),(row.followup_minute,post)]:
        expected=lookup.get(key,np.nan)
        assert (pd.isna(observed) and pd.isna(expected)) or abs(observed-expected)<=1e-12
support=[];gates=[]
for item in [224842,220088]:
    for split in [0,1]:
        p=pairs.loc[pairs.itemid.eq(item)&pairs.split.eq(split)]
        for stage,mask in [('selected',np.ones(len(p),dtype=bool)),('baseline',p.baseline_minute.notna()),('followup',p.followup_minute.notna()),('paired',p.baseline_minute.notna()&p.followup_minute.notna())]:
            s=p.loc[mask];n=len(s);ne=int(s.exposed.sum());nu=n-ne
            row=dict(itemid=item,split=split,stage=stage,people=n,exposed=ne,unexposed=nu)
            if stage=='paired':
                row['followup_minutes_quantiles']={str(q):float(s.followup_minute.quantile(q)) for q in [.1,.5,.9]} if n else {}
                row['support_floor_pass']=n>=100 and ne>=25 and nu>=25
                if item==224842:gates.append(row['support_floor_pass'])
            support.append(row)
assert len(gates)==2
df_json('support.json',support)
df_json('validation.json',dict(lowest_stay_per_person_sql_python_agree=True,hash_splits_reconstruct=True,all_timestamp_pairs_sql_python_agree=True,
  original_encounters=len(original),selected_people=len(selected),metadata_rows=len(raw),distinct_measurement_times=len(times)))
df_json('results.json',dict(primary_cco_support_floor_pass=all(gates),cardiac_output_values_selected=False,mortality_selected=False,biological_discovery=False,
  interpretation='A metadata support test in MIMIC only; no external mechanistic replication or clinical effect estimated.'))
report=['# Direct cardiac-flow measurement support','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Full raw MIMIC chartevents metadata scan.','',
 '| Modality | Split | Paired people | Exposed | Unexposed | Median follow-up minute | Support floor |','|---|---|---:|---:|---:|---:|---|']
for r in support:
    if r['stage']=='paired':report.append(f"| {r['itemid']} | {r['split']} | {r['people']} | {r['exposed']} | {r['unexposed']} | {r['followup_minutes_quantiles'].get('0.5','NA')} | {r['support_floor_pass']} |")
report+=['','Primary CCO joint support floor: '+str(all(gates))+'. Values have not been read or qualified. SQL/Python patient selection, splitting and pair reconstruction agree.',
 'No treatment effect, mortality association, novel mechanism, or eICU direct-flow replication is established. Missing measurements are not normal cardiac output.','']
(PROJECT/'docs/SPO2_DIRECT_FLOW_RESULTS.md').write_text('\n'.join(report))
df_json('run_status.json',dict(status='complete',stage='direct-flow metadata support and independent validation',utc=datetime.now(timezone.utc).isoformat(),goal_complete=False))
con.close();print('\n'.join(report))
"""
