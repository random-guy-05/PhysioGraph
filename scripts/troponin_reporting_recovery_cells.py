"""Source-comment recovery and bounded-result selection in the existing cohort."""

SETUP = r'''
import ast,json,hashlib,importlib.util,runpy,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_troponin_reporting_recovery';OUT.mkdir(parents=True,exist_ok=True)
def rj(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def rh(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def rq(value):return "'"+str(value).replace("'","''")+"'"
def rs(stage,status='running'):
    rj('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),
      environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert rh(PROJECT/'docs/SPO2_TROPONIN_REPORTING_RECOVERY_PLAN.md')==TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256
else:rj('protocol_lock.json',dict(sha256=TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_comment_extraction_and_selection_counts=True,prior_audit_known=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'troponin_reporting_recovery.duckdb'));con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
refs={}
for alias,name in [('audit','troponin_assay_audit'),('care','spo2_cardiorenal')]:
    p=PRIVATE/(name+'.duckdb');refs[name]=dict(path=str(p),sha256=rh(p));con.execute(f'ATTACH {rq(p)} AS {alias} (READ_ONLY)')
source=json.loads((PROJECT/'research/spo2_troponin_assay_audit/input_manifest.json').read_text())['sources']['MIMIC']
s=Path(source['path']).stat();assert s.st_size==source['bytes'] and s.st_mtime_ns==source['mtime_ns']
manifest=dict(references=refs,source=source,protocol_sha256=TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256)
if (OUT/'input_manifest.json').exists():assert json.loads((OUT/'input_manifest.json').read_text())==manifest
rj('input_manifest.json',manifest)
tree=ast.parse(runpy.run_path(str(PROJECT/'scripts/troponin_assay_audit_cells.py'))['SUPPORT'])
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_text' or isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ['numeric','pattern'] for x in n.targets)]
parser_namespace=dict(re=re,np=np,pd=pd)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen-reporting-parser','exec'),parser_namespace)
parse_text=parser_namespace['parse_text']
if not globals().get('COMMENT_TEMPLATE_ONLY',False):rs('reporting recovery initialized')
'''

EXTRACT = r'''
rs('extracting comments for the exact audited MIMIC source IDs')
con.execute(f"""CREATE OR REPLACE TABLE mimic_comments AS
 SELECT l.labevent_id source_id,l.valuenum numeric_text,l.value result_text,l.itemid raw_name,l.comments
 FROM read_csv({rq(source['path'])},header=true,all_varchar=true,sample_size=10000,strict_mode=true) l
 JOIN (SELECT DISTINCT source_id FROM audit.mimic_raw) r ON l.labevent_id=r.source_id""")
actual=con.execute('SELECT source_id,numeric_text,result_text,raw_name FROM mimic_comments ORDER BY source_id').df()
expected=con.execute('SELECT DISTINCT source_id,numeric_text,result_text,raw_name FROM audit.mimic_raw ORDER BY source_id').df()
assert actual.source_id.is_unique
pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
s=Path(source['path']).stat();assert s.st_size==source['bytes'] and s.st_mtime_ns==source['mtime_ns']
rj('raw_extraction.json',dict(complete=True,source_identity_matches=True,source_rows=len(actual),source=source))
con.execute('CHECKPOINT');rs('MIMIC comment extraction complete')
'''

SUPPORT = r'''
rs('strict comment classification and eICU selection counts')
extraction=json.loads((OUT/'raw_extraction.json').read_text());assert extraction['complete'] and extraction['source']==source
comments=con.execute('SELECT * FROM mimic_comments').df()
parsed=[parse_text(x) for x in comments.comments]
comments['comment_class']=[x[0] for x in parsed];comments['comment_number']=[x[1] for x in parsed]
comments['numeric_value']=pd.to_numeric(comments.numeric_text,errors='coerce')
comments['numeric_accepted']=comments.numeric_value.between(0,1000000)
comments['exact_match']=comments.comment_class.eq('exact')&np.isclose(comments.comment_number,comments.numeric_value,atol=1e-12,rtol=0,equal_nan=False)
comment_inventory=[]
for (cls,accepted),g in comments.groupby(['comment_class','numeric_accepted']):
    comment_inventory.append(dict(comment_class=cls,numeric_accepted=bool(accepted),source_rows=len(g),
      exact_numeric_matches=int(g.exact_match.sum()),placeholder_comments=int(g.comments.eq('___').sum())))
con.register('classified_comments',comments)
comment_oracle=con.execute(r"""WITH normalized AS (
  SELECT source_id,regexp_replace(regexp_replace(replace(replace(lower(trim(coalesce(comments,''))),'≤','<='),'≥','>='),
    '^less\s+than\s*','<'),'^greater\s+than\s*','>') txt FROM mimic_comments),
  classified AS (SELECT *,CASE WHEN txt='' THEN 'missing'
    WHEN NOT regexp_full_match(txt,'(<=|>=|<|>|=)?\s*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?') THEN 'unparsed'
    WHEN starts_with(txt,'<=') THEN 'less_equal' WHEN starts_with(txt,'>=') THEN 'greater_equal'
    WHEN starts_with(txt,'<') THEN 'less_than' WHEN starts_with(txt,'>') THEN 'greater_than' ELSE 'exact' END AS comment_class
    FROM normalized)
  SELECT source_id,comment_class,CASE WHEN comment_class NOT IN ('missing','unparsed') THEN
    try_cast(regexp_extract(txt,'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?',0) AS DOUBLE) END AS comment_number
    FROM classified ORDER BY source_id""").df()
pd.testing.assert_frame_equal(comments.sort_values('source_id').reset_index(drop=True)[comment_oracle.columns],comment_oracle,check_dtype=False,rtol=0,atol=1e-12)
con.execute('CREATE OR REPLACE TABLE mimic_classified_comments AS SELECT * FROM classified_comments')
con.unregister('classified_comments')
raw=con.execute('SELECT stay_id,event_minute,concept,unit,result_text FROM audit.eicu_raw').df()
labs=con.execute("SELECT stay_id,event_minute,concept,unit,value FROM care.eicu_labs WHERE concept IN ('troponin_i','troponin_t')").df()
cohort=con.execute('SELECT stay_id,person_id,exposed,followup_end_offset_minutes FROM care.eicu_cohort').df()
classes=[parse_text(x) for x in raw.result_text]
raw['text_class']=[x[0] for x in classes];raw['bound_magnitude']=[x[1] for x in classes]
bounded=raw.loc[raw.text_class.isin(['less_than','less_equal','greater_than','greater_equal'])].copy()
allowed=set(labs[['concept','unit']].itertuples(index=False,name=None))
bounded['unit_supported']=[(c,u) in allowed for c,u in zip(bounded.concept,bounded.unit)]
bound_audit=dict(all_explicit_bound_rows=len(bounded),unmatched_unit_rows=int((~bounded.unit_supported).sum()),
  nonpositive_magnitude_rows=int(bounded.bound_magnitude.le(0).sum()))
bounded=bounded.loc[bounded.unit_supported&bounded.bound_magnitude.gt(0)].drop_duplicates(['stay_id','event_minute','concept','unit'])
con.register('recovery_bounds',bounded);con.register('recovery_labs',labs);con.register('recovery_cohort',cohort)
counts=[];validations=[]
for end,label in [(960,'12h'),(1680,'24h')]:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE recovery_baselines AS
      SELECT * FROM recovery_labs WHERE value>0 AND event_minute BETWEEN 0 AND 240
      QUALIFY row_number() OVER(PARTITION BY stay_id,concept,unit ORDER BY event_minute DESC)=1""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE recovery_pairs AS
      SELECT b.* FROM recovery_baselines b WHERE EXISTS (
        SELECT 1 FROM recovery_labs p WHERE p.stay_id=b.stay_id AND p.concept=b.concept AND p.unit=b.unit
        AND p.event_minute>240 AND p.event_minute<={end})""")
    sql=con.execute(f"""WITH flagged AS (
      SELECT c.stay_id,c.person_id,c.exposed,
        EXISTS(SELECT 1 FROM recovery_pairs p WHERE p.stay_id=c.stay_id) existing_pair,
        EXISTS(SELECT 1 FROM recovery_pairs p JOIN recovery_bounds b USING(stay_id,concept,unit)
          WHERE p.stay_id=c.stay_id AND b.event_minute>p.event_minute AND b.event_minute<=240) later_baseline_bound,
        EXISTS(SELECT 1 FROM recovery_pairs p JOIN recovery_bounds b USING(stay_id,concept,unit)
          WHERE p.stay_id=c.stay_id AND b.event_minute>240 AND b.event_minute<={end}) followup_bound
      FROM recovery_cohort c WHERE c.followup_end_offset_minutes>={end}),
      reports AS (SELECT stay_id,event_minute,concept,unit,value>0 baseline_possible FROM recovery_labs
        UNION ALL SELECT stay_id,event_minute,concept,unit,true baseline_possible FROM recovery_bounds)
      SELECT f.*,EXISTS(SELECT 1 FROM reports b JOIN reports p USING(stay_id,concept,unit)
        WHERE b.stay_id=f.stay_id AND b.baseline_possible AND b.event_minute BETWEEN 0 AND 240
        AND p.event_minute>240 AND p.event_minute<={end}) potential_pair
      FROM flagged f ORDER BY stay_id""").df()
    keys=['stay_id','concept','unit']
    base=labs.loc[labs.value.gt(0)&labs.event_minute.between(0,240)].sort_values('event_minute').drop_duplicates(keys,keep='last')
    postkeys=set(labs.loc[labs.event_minute.gt(240)&labs.event_minute.le(end),keys].itertuples(index=False,name=None))
    paired=base.loc[[k in postkeys for k in base[keys].itertuples(index=False,name=None)]]
    existing=set(paired.stay_id);later=set();follow=set()
    boundgroups={k:g.event_minute.to_numpy() for k,g in bounded.groupby(keys)}
    for p in paired.itertuples():
        t=boundgroups.get((p.stay_id,p.concept,p.unit),np.array([]))
        if ((t>p.event_minute)&(t<=240)).any():later.add(p.stay_id)
        if ((t>240)&(t<=end)).any():follow.add(p.stay_id)
    bkeys=set(base[keys].itertuples(index=False,name=None))|set(bounded.loc[bounded.event_minute.between(0,240),keys].itertuples(index=False,name=None))
    pkeys=postkeys|set(bounded.loc[bounded.event_minute.gt(240)&bounded.event_minute.le(end),keys].itertuples(index=False,name=None))
    potential={k[0] for k in bkeys&pkeys}
    python=cohort.loc[cohort.followup_end_offset_minutes.ge(end),['stay_id','person_id','exposed']].sort_values('stay_id').reset_index(drop=True)
    for name,ids in [('existing_pair',existing),('later_baseline_bound',later),('followup_bound',follow),('potential_pair',potential)]:python[name]=python.stay_id.isin(ids)
    pd.testing.assert_frame_equal(sql,python[sql.columns],check_dtype=False)
    legacy=con.execute(f'SELECT t.stay_id FROM care.eicu_troponin_{label} t JOIN recovery_cohort c USING(stay_id) WHERE c.followup_end_offset_minutes>={end}').df()
    assert set(sql.loc[sql.existing_pair,'stay_id'])==set(legacy.stay_id)
    assert (sql.potential_pair|~sql.existing_pair).all()
    for arm,g in sql.groupby('exposed'):
        row=dict(horizon=label,exposed=int(arm),full_followup_encounters=len(g),full_followup_people=int(g.person_id.nunique()))
        for name,mask in [('existing_pair',g.existing_pair),('later_baseline_bound',g.later_baseline_bound),
          ('followup_bound',g.followup_bound),('either_bound_in_existing_pair',g.later_baseline_bound|g.followup_bound),
          ('new_potential_pair',g.potential_pair&~g.existing_pair),('all_potential_pairs',g.potential_pair)]:
            row[name+'_encounters']=int(mask.sum());row[name+'_people']=int(g.loc[mask,'person_id'].nunique())
        counts.append(row)
    validations.append(dict(horizon=label,all_sql_python_patient_flags_match=True,original_numeric_pair_set_matches=True,independent_comment_classes_and_numbers_match=True))
for info in refs.values():assert rh(Path(info['path']))==info['sha256']
s=Path(source['path']).stat();assert s.st_size==source['bytes'] and s.st_mtime_ns==source['mtime_ns']
rj('comment_inventory.json',comment_inventory);rj('eicu_bound_audit.json',bound_audit);rj('eicu_selection_counts.json',counts);rj('validation.json',validations)
rj('manifest.json',dict(protocol_sha256=TROPONIN_REPORTING_RECOVERY_PROTOCOL_SHA256,script_sha256=rh(PROJECT/'scripts/troponin_reporting_recovery_cells.py'),
  source_and_reference_fingerprints_unchanged=True,labels_replaced=False,mortality_read=False,association_models=0,
  environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
report=['# Reporting-limit recovery and selection audit','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. MIMIC comments rescanned for the exact previously audited source IDs; eICU source rows reused.',
  '', '| MIMIC whole-comment class | Numeric accepted | Source records | Matching exact numeric comment |','|---|---|---:|---:|']
for r in comment_inventory:report.append(f"| {r['comment_class']} | {r['numeric_accepted']} | {r['source_rows']} | {r['exact_numeric_matches']} |")
report+=['','| eICU horizon | Exposure | Full follow-up encounters | Existing numeric pairs | Later omitted baseline bound | Omitted follow-up bound | Additional potential timed pairs |','|---|---:|---:|---:|---:|---:|---:|']
for r in counts:report.append(f"| {r['horizon']} | {r['exposed']} | {r['full_followup_encounters']} | {r['existing_pair_encounters']} | {r['later_baseline_bound_encounters']} | {r['followup_bound_encounters']} | {r['new_potential_pair_encounters']} |")
report+=['','All source IDs and original numeric/text/item identities matched. SQL/Python patient flags and original numeric-pair sets matched. Baseline and follow-up bound counts overlap and must not be added. Person counts and unit exclusions are in the aggregate JSON outputs.',
  '', 'Whole-comment parsing is deliberately strict; narrative numbers are not interpreted as assay limits. A recoverable bound still requires interval-aware adjudication. Additional potential pairs are an optimistic timing ceiling: a positive upper bound does not prove a positive baseline or a valid rise. No bound was substituted by its limit, no numeric label was changed, and no new association, biological discovery or mortality benefit was estimated.','']
(PROJECT/'docs/SPO2_TROPONIN_REPORTING_RECOVERY_RESULTS.md').write_text('\n'.join(report))
con.close();rs('comment recovery and bounded-result selection audit','complete');print('\n'.join(report))
'''

TEMPLATE_AUDIT = r'''
# Uses the reporting-recovery setup and its verified source/reference identities.
OUT=PROJECT/'research/spo2_troponin_reporting_recovery/template_amendment';OUT.mkdir(parents=True,exist_ok=True)
assert rh(PROJECT/'docs/SPO2_TROPONIN_COMMENT_TEMPLATE_AMENDMENT.md')==TROPONIN_COMMENT_TEMPLATE_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==TROPONIN_COMMENT_TEMPLATE_SHA256
else:rj('protocol_lock.json',dict(sha256=TROPONIN_COMMENT_TEMPLATE_SHA256,utc=datetime.now(timezone.utc).isoformat(),template_frequencies_already_known=True,no_new_exposure_association=True))
try:con.close()
except (NameError,AttributeError):pass
con=duckdb.connect(str(PRIVATE/'troponin_reporting_recovery.duckdb'))
comments=con.execute('SELECT * FROM mimic_comments ORDER BY source_id').df()
def comment_hash(frame):return hashlib.sha256(pd.util.hash_pandas_object(frame,index=False).values.tobytes()).hexdigest()
before_hash=comment_hash(comments)
statement='CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.'
def template_class(text):
    if pd.isna(text):return 'unrecognized'
    text=' '.join(str(text).upper().split())
    if text==statement:return 'interpretation_only'
    if text=='<0.01. '+statement:return 'upper_bound_0.01'
    if text=='>25*. '+statement:return 'lower_bound_25'
    return 'unrecognized'
fixtures=[(statement,'interpretation_only'),('<0.01.  '+statement,'upper_bound_0.01'),
  ('<0.01.   '+statement.lower(),'upper_bound_0.01'),('>25*. '+statement,'lower_bound_25'),
  ('<0.02. '+statement,'unrecognized'),('>25. '+statement,'unrecognized'),
  ('NOTE '+statement,'unrecognized'),(statement+' additional text','unrecognized'),
  ('0.10','unrecognized'),('___','unrecognized'),(None,'unrecognized')]
assert all(template_class(text)==expected for text,expected in fixtures)
comments['template_class']=comments.comments.map(template_class)
oracle=con.execute(r"""WITH normalized AS (
  SELECT source_id,upper(regexp_replace(trim(coalesce(comments,'')),'\s+',' ','g')) txt FROM mimic_comments)
  SELECT source_id,CASE
    WHEN txt='CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.' THEN 'interpretation_only'
    WHEN txt='<0.01. CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.' THEN 'upper_bound_0.01'
    WHEN txt='>25*. CTROPNT > 0.10 NG/ML SUGGESTS ACUTE MI.' THEN 'lower_bound_25'
    ELSE 'unrecognized' END AS template_class FROM normalized ORDER BY source_id""").df()
pd.testing.assert_frame_equal(comments[oracle.columns],oracle,check_dtype=False)
comments['bound_literal']=comments.template_class.map({'upper_bound_0.01':'0.01','lower_bound_25':'25'})
comments['operator']=comments.template_class.map({'upper_bound_0.01':'<','lower_bound_25':'>'})
comments['numeric_value']=pd.to_numeric(comments.numeric_text,errors='coerce')
comments['numeric_accepted']=comments.numeric_value.between(0,1000000)
con.execute(f"ATTACH {rq(PRIVATE/'troponin_assay_audit.duckdb')} AS audit (READ_ONLY)")
units=con.execute('SELECT DISTINCT source_id,unit FROM audit.mimic_raw').df()
assert units.source_id.is_unique
comments=comments.merge(units,on='source_id',validate='one_to_one')
rows=[]
for (kind,accepted,unit),g in comments.groupby(['template_class','numeric_accepted','unit'],dropna=False):
    limits=pd.to_numeric(g.bound_literal,errors='coerce')
    disagreement=g.numeric_value.notna()&limits.notna()&~np.isclose(g.numeric_value,limits,atol=1e-12,rtol=0,equal_nan=False)
    rows.append(dict(template_class=kind,numeric_accepted=bool(accepted),unit=unit,source_rows=len(g),
      missing_numeric_rows=int(g.numeric_value.isna().sum()),coexisting_magnitude_disagreements=int(disagreement.sum())))
classified=comments.drop(columns='comments')
con.register('template_rows',classified)
con.execute('CREATE OR REPLACE TABLE mimic_comment_template_results AS SELECT * FROM template_rows')
con.unregister('template_rows')
after=con.execute('SELECT * FROM mimic_comments ORDER BY source_id').df()
assert comment_hash(after)==before_hash
for info in refs.values():assert rh(Path(info['path']))==info['sha256']
rj('classification.json',rows)
rj('validation.json',dict(all_source_ids_sql_python_match=True,python_counterexamples_passed=len(fixtures),
  comment_table_hash=before_hash,comment_table_unchanged=True,original_references_unchanged=True,
  interpretation_threshold_never_assigned_as_patient_value=True,earlier_strict_parser_unchanged=True))
rj('manifest.json',dict(protocol_sha256=TROPONIN_COMMENT_TEMPLATE_SHA256,script_sha256=rh(PROJECT/'scripts/troponin_reporting_recovery_cells.py'),
  environment='google_colab' if IN_COLAB else 'local',source_rows=len(comments),bound_substituted_as_exact_value=False,
  original_labels_changed=False,association_models=0,goal_complete=False))
report=['# MIMIC reporting-bound template recovery','', 'Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Separate amendment after the strict whole-comment audit; template frequencies were already known.',
  '', '| Comment category | Existing numeric value accepted | Unit | Source records | Missing numeric value |','|---|---|---|---:|---:|']
for r in rows:report.append(f"| {r['template_class']} | {r['numeric_accepted']} | {r['unit']} | {r['source_rows']} | {r['missing_numeric_rows']} |")
report+=['','A leading bound is separated from the assay interpretation statement. The 0.10 ng/mL comparison inside that statement is never used as the patient value or an MI diagnosis. Other comment formats remain unrecognized.',
  '', 'All source-ID classifications agree between SQL and Python; counterexamples pass; the comments table and reference hashes are unchanged. Recovered bounds remain intervals. No new clinical endpoint, association, biological mechanism or mortality benefit has been established.','']
(PROJECT/'docs/SPO2_TROPONIN_COMMENT_TEMPLATE_RESULTS.md').write_text('\n'.join(report))
con.close();rs('whitelisted comment-template recovery','complete');print('\n'.join(report))
'''
