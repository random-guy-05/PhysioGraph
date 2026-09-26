"""Original assay text and numeric-pipeline reconciliation; no new association."""

SETUP = r'''
import re,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_troponin_assay_audit';OUT.mkdir(parents=True,exist_ok=True)
def aj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def ah(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def aq(x):return "'"+str(x).replace("'","''")+"'"
def astage(stage,status='running'):
    aj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),
      environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert ah(PROJECT/'docs/SPO2_TROPONIN_ASSAY_AUDIT_PLAN.md')==TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256
else:aj('protocol_lock.json',dict(sha256=TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_raw_text_extraction=True,prior_numeric_results_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'troponin_assay_audit.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
references={}
for alias,name in [('care','spo2_cardiorenal'),('prior','spo2_context')]:
    p=PRIVATE/(name+'.duckdb');s=p.stat();references[name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=ah(p))
    con.execute(f'ATTACH {aq(p)} AS {alias} (READ_ONLY)')
for d in ['eicu','mimic']:
    extra=',p.hadm_id,p.admit_time' if d=='mimic' else ''
    con.execute(f'CREATE OR REPLACE TEMP TABLE {d}_audit_cohort AS SELECT c.*{extra} FROM care.{d}_cohort c JOIN prior.{d}_cohort p USING(stay_id)')
    assert con.execute(f'SELECT count(*) FROM {d}_audit_cohort').fetchone()[0]=={'eicu':11452,'mimic':4711}[d]
    con.execute(f"CREATE OR REPLACE TEMP TABLE {d}_map AS SELECT DISTINCT lower(trim(cast(raw_name AS VARCHAR))) raw_name,concept FROM prior.{d}_context_labs WHERE concept IN ('troponin_i','troponin_t')")
    assert con.execute(f'SELECT count(*) FROM (SELECT raw_name FROM {d}_map GROUP BY raw_name HAVING count(*)>1)').fetchone()[0]==0
sources={}
for dataset,table in [('eICU','lab'),('MIMIC','labevents')]:
    p=DRIVE/'Data'/dataset/'Full'/(table+'.csv');s=p.stat();sources[dataset]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def acsv(dataset):return f'read_csv({aq(sources[dataset]["path"])},header=true,all_varchar=true,sample_size=10000,strict_mode=true)'
input_manifest=dict(references=references,sources=sources,protocol_sha256=TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256)
if (OUT/'input_manifest.json').exists():
    assert json.loads((OUT/'input_manifest.json').read_text())==input_manifest, 'Audit source/reference fingerprints changed'
aj('input_manifest.json',input_manifest)
astage('assay audit initialized')
'''

EXTRACT = r'''
astage('scanning original eICU troponin result text')
con.execute(f"""CREATE OR REPLACE TABLE eicu_raw AS
 SELECT c.stay_id,l.labid source_id,m.concept,try_cast(l.labresultoffset AS DOUBLE) event_minute,
   try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,l.labresult numeric_text,
   try_cast(l.labresult AS DOUBLE) numeric_value,l.labresulttext result_text,
   lower(trim(coalesce(l.labmeasurenamesystem,l.labmeasurenameinterface,''))) unit,
   l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface,l.labname raw_name
 FROM {acsv('eICU')} l JOIN eicu_audit_cohort c ON try_cast(l.patientunitstayid AS BIGINT)=c.stay_id
 JOIN eicu_map m ON lower(trim(l.labname))=m.raw_name
 WHERE try_cast(l.labresultoffset AS DOUBLE) BETWEEN 0 AND least(1680,c.followup_end_offset_minutes)""")
s=Path(sources['eICU']['path']).stat();assert s.st_size==sources['eICU']['bytes'] and s.st_mtime_ns==sources['eICU']['mtime_ns']
astage('scanning original MIMIC troponin result text')
con.execute(f"""CREATE OR REPLACE TABLE mimic_raw AS
 SELECT c.stay_id,l.labevent_id source_id,m.concept,
   date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
   date_diff('second',c.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
   l.valuenum numeric_text,try_cast(l.valuenum AS DOUBLE) numeric_value,l.value result_text,
   lower(trim(coalesce(l.valueuom,''))) unit,l.valueuom unit_system,'' unit_interface,
   l.itemid raw_name,l.specimen_id,try_cast(l.ref_range_upper AS DOUBLE) reference_upper,
   try_cast(l.subject_id AS BIGINT)=try_cast(c.person_id AS BIGINT) subject_matches
 FROM {acsv('MIMIC')} l JOIN mimic_audit_cohort c ON try_cast(l.hadm_id AS BIGINT)=c.hadm_id
 JOIN mimic_map m ON lower(trim(l.itemid))=m.raw_name
 WHERE try_cast(l.charttime AS TIMESTAMP)>=c.admit_time
   AND date_diff('second',c.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0<=least(1680,c.followup_end_offset_minutes)""")
for info in sources.values():
    s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns']
aj('raw_extraction.json',dict(complete=True,rows={d:con.execute(f'SELECT count(*) FROM {d}_raw').fetchone()[0] for d in ['eicu','mimic']},sources=sources))
con.execute('CHECKPOINT');astage('raw assay extraction complete; reconciliation pending')
'''

SUPPORT = r'''
astage('reconciling source values and reporting flags')
extraction=json.loads((OUT/'raw_extraction.json').read_text());assert extraction['complete'] and extraction['sources']==sources
numeric=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
pattern=re.compile(r'^\s*(<=|>=|<|>|=)?\s*('+numeric+r')\s*$')
def parse_text(value):
    if pd.isna(value) or not str(value).strip():return 'missing',np.nan
    value=str(value).strip().lower().replace('≤','<=').replace('≥','>=')
    value=re.sub(r'^less\s+than\s*','<',value);value=re.sub(r'^greater\s+than\s*','>',value)
    m=pattern.fullmatch(value)
    if m is None:return 'unparsed',np.nan
    operator=m.group(1) or '='
    return {'=':'exact','<':'less_than','<=':'less_equal','>':'greater_than','>=':'greater_equal'}[operator],float(m.group(2))
inventory=[];impact=[];validation=[]
for d in ['eicu','mimic']:
    raw=con.execute(f'SELECT * FROM {d}_raw').df()
    if d=='mimic':assert raw.subject_matches.fillna(False).all()
    parsed=[parse_text(v) for v in raw.result_text]
    raw['text_class']=[v[0] for v in parsed];raw['text_magnitude']=[v[1] for v in parsed]
    raw['numeric_accepted']=raw.numeric_value.between(0,1000000)
    raw['magnitude_matches']=np.isclose(raw.text_magnitude,raw.numeric_value,rtol=0,atol=1e-12,equal_nan=False)
    raw['text_uncertain']=~(raw.text_class.eq('exact')&raw.magnitude_matches)
    raw['explicit_bound']=raw.text_class.isin(['less_than','less_equal','greater_than','greater_equal'])
    for (concept,cls,accepted),g in raw.groupby(['concept','text_class','numeric_accepted'],dropna=False):
        inventory.append(dict(dataset=d,concept=concept,text_class=cls,numeric_accepted=bool(accepted),records=len(g),
          three_underscore_placeholder_records=int(g.result_text.eq('___').sum()),
          stored_numeric_missing=int(g.numeric_value.isna().sum()),
          magnitude_disagreement=int((g.numeric_value.notna()&g.text_magnitude.notna()&~g.magnitude_matches).sum()),
          numeric_reference_upper_records=int(g.reference_upper.notna().sum()) if d=='mimic' else None))
    accepted=raw.loc[raw.numeric_accepted].copy();keys=['stay_id','event_minute','concept','unit']
    rebuilt=accepted.groupby(keys,as_index=False).agg(value=('numeric_value','median'),available_minute=('available_minute','max'),
      text_uncertain=('text_uncertain','max'),explicit_bound=('explicit_bound','max')).sort_values(keys).reset_index(drop=True)
    cached=con.execute(f"SELECT * FROM care.{d}_labs WHERE concept IN ('troponin_i','troponin_t') ORDER BY stay_id,event_minute,concept,unit").df()
    pd.testing.assert_frame_equal(rebuilt[cached.columns],cached,check_dtype=False,rtol=0,atol=1e-12)
    con.register('audit_accepted',accepted)
    oracle=con.execute("""SELECT stay_id,event_minute,concept,unit,median(numeric_value) AS value,max(available_minute) AS available_minute,
      bool_or(text_uncertain) text_uncertain,bool_or(explicit_bound) explicit_bound
      FROM audit_accepted GROUP BY stay_id,event_minute,concept,unit ORDER BY stay_id,event_minute,concept,unit""").df()
    pd.testing.assert_frame_equal(rebuilt[oracle.columns],oracle,check_dtype=False,rtol=0,atol=1e-12)
    con.register('audit_groups',rebuilt)
    con.execute(f'CREATE OR REPLACE TABLE {d}_reporting_groups AS SELECT * FROM audit_groups')
    for end,label in [(960,'12h'),(1680,'24h')]:
        selected=con.execute(f"""WITH base AS (
          SELECT * FROM audit_groups WHERE value>0 AND event_minute<=240
          QUALIFY row_number() OVER(PARTITION BY stay_id,concept,unit ORDER BY event_minute DESC)=1)
          SELECT p.stay_id,c.person_id,c.exposed,p.event_minute,p.concept,p.unit,
            p.value/b.value ratio,b.text_uncertain base_uncertain,p.text_uncertain post_uncertain,
            b.explicit_bound base_bound,p.explicit_bound post_bound
          FROM audit_groups p JOIN base b USING(stay_id,concept,unit)
          JOIN {d}_audit_cohort c USING(stay_id)
          WHERE p.event_minute>240 AND p.event_minute<={end} AND c.followup_end_offset_minutes>={end}
          ORDER BY p.stay_id,p.concept,p.unit,p.event_minute""").df()
        con.register('audit_selected',selected)
        agg=con.execute("""SELECT stay_id,person_id,exposed,max(ratio) ratio,
          bool_or(base_uncertain) base_uncertain,bool_or(post_uncertain) post_uncertain,
          bool_or(base_bound) base_bound,bool_or(post_bound) post_bound
          FROM audit_selected GROUP BY stay_id,person_id,exposed ORDER BY stay_id""").df()
        # Independent pandas baseline selection, joins and patient flag aggregation.
        bk=['stay_id','concept','unit']
        base=rebuilt.loc[rebuilt.value.gt(0)&rebuilt.event_minute.le(240)].sort_values('event_minute').drop_duplicates(bk,keep='last')
        post=rebuilt.loc[rebuilt.event_minute.gt(240)&rebuilt.event_minute.le(end)]
        py=post.merge(base[bk+['value','text_uncertain','explicit_bound']],on=bk,suffixes=('_post','_base'),validate='many_to_one')
        py=py.loc[py.stay_id.isin(agg.stay_id)]
        py['ratio']=py.value_post/py.value_base
        check=py.groupby('stay_id').agg(ratio=('ratio','max'),base_uncertain=('text_uncertain_base','max'),
          post_uncertain=('text_uncertain_post','max'),base_bound=('explicit_bound_base','max'),post_bound=('explicit_bound_post','max')).reset_index()
        pd.testing.assert_frame_equal(agg[check.columns],check,check_dtype=False,rtol=0,atol=1e-12)
        legacy=con.execute(f"""SELECT t.stay_id,t.troponin_ratio,e.troponin_rise_{label} legacy_label
          FROM care.{d}_troponin_{label} t JOIN {d}_audit_cohort c USING(stay_id)
          JOIN prior.{d}_context_endpoints e USING(stay_id) WHERE c.followup_end_offset_minutes>={end}
          ORDER BY t.stay_id""").df()
        assert np.array_equal(agg.stay_id,legacy.stay_id) and np.allclose(agg.ratio,legacy.troponin_ratio,rtol=0,atol=1e-12)
        agg['legacy_rise']=agg.ratio.ge(1.5).astype(int);assert np.array_equal(agg.legacy_rise,legacy.legacy_label)
        for (exposed,rise),g in agg.groupby(['exposed','legacy_rise']):
            row=dict(dataset=d,horizon=label,exposed=int(exposed),legacy_rise=int(rise),encounters=len(g),people=int(g.person_id.nunique()))
            for flag in ['base_uncertain','post_uncertain','base_bound','post_bound']:
                row[flag+'_encounters']=int(g[flag].sum());row[flag+'_people']=int(g.loc[g[flag],'person_id'].nunique())
            for name,mask in [('any_uncertain',g.base_uncertain|g.post_uncertain),('any_bound',g.base_bound|g.post_bound)]:
                row[name+'_encounters']=int(mask.sum());row[name+'_people']=int(g.loc[mask,'person_id'].nunique())
            impact.append(row)
        validation.append(dict(dataset=d,horizon=label,original_keys_values_and_availability_match=True,
          sql_python_source_medians_and_used_flags_match=True,legacy_labels_and_ratios_match=True,encounters=len(agg)))
        con.unregister('audit_selected')
    con.unregister('audit_groups');con.unregister('audit_accepted')
for info in references.values():
    p=Path(info['path']);s=p.stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns'] and ah(p)==info['sha256']
for info in sources.values():
    s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns']
aj('text_inventory.json',inventory);aj('label_impact.json',impact);aj('validation.json',validation)
aj('manifest.json',dict(protocol_sha256=TROPONIN_ASSAY_AUDIT_PROTOCOL_SHA256,script_sha256=ah(PROJECT/'scripts/troponin_assay_audit_cells.py'),
  raw_sources_stable=True,reference_hashes_unchanged=True,existing_labels_changed=False,mortality_read=False,association_models=0,
  environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# Raw troponin reporting audit','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Original MIMIC/eICU source files rescanned; existing endpoint labels preserved.',
  '', '| Source | Horizon | Exposure | Legacy rise | Encounters | Any text uncertainty | Explicit bound in used samples |',
  '|---|---|---:|---:|---:|---:|---:|']
for r in impact:report.append(f"| {r['dataset']} | {r['horizon']} | {r['exposed']} | {r['legacy_rise']} | {r['encounters']} | {r['any_uncertain_encounters']} | {r['any_bound_encounters']} |")
report+=['', 'Source grouping, cached numeric values/result times, original ratios and labels, and SQL/Python used-sample flags all matched. Source inventories are in text_inventory.json; baseline/follow-up flags and person counts are in label_impact.json.',
  '', 'Post-audit text-shape clarification: MIMIC has '+str(sum(r['three_underscore_placeholder_records'] for r in inventory if r['dataset']=='mimic'))+' literal three-underscore placeholder records among '+str(sum(r['records'] for r in inventory if r['dataset']=='mimic' and r['text_class']=='unparsed'))+' unparsed records. This does not establish why that field was replaced or recover its original qualifiers. Classification and label-impact rules were unchanged by this clarification.',
  '', 'eICU contains '+str(sum(r['records'] for r in inventory if r['dataset']=='eicu' and r['text_class'] in ['less_than','less_equal','greater_than','greater_equal']))+' explicitly bounded source records; '+str(sum(r['records'] for r in inventory if r['dataset']=='eicu' and r['numeric_accepted'] and r['text_class'] in ['less_than','less_equal','greater_than','greater_equal']))+' enter the accepted numeric pipeline. Missing numeric values are reported separately from text/numeric disagreements. Clean text among retained values does not resolve selection caused by omitted bounded values.',
  '', 'An explicit bound describes an interval. Missing or unparsed text is an uncertainty flag, not proven censoring. A flagged encounter is not automatically a false rise: baseline and follow-up intervals require separate adjudication. The audit includes all matched follow-up samples, not only the maximum. A numeric upper reference range in MIMIC does not establish its equivalence to an assay-specific 99th percentile; eICU lacks that field.',
  '', 'These results validate or qualify reporting provenance only. No endpoint was replaced, no new association was fitted, and no myocardial injury diagnosis, biological discovery or mortality benefit is established.','']
(PROJECT/'docs/SPO2_TROPONIN_ASSAY_AUDIT_RESULTS.md').write_text('\n'.join(report))
con.close();astage('original assay reporting audit and independent validation','complete');print('\n'.join(report))
'''
