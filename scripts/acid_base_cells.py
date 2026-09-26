"""Outcome-blind arterial acid–base support in the existing HF cohorts."""

SETUP = r'''
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_acid_base';OUT.mkdir(exist_ok=True)
def acid_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def acid_status(stage,status='running'):
    acid_json('run_status.json',dict(stage=stage,status=status,utc=datetime.now(timezone.utc).isoformat(),environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
assert hashlib.sha256((PROJECT/'docs/SPO2_ACID_BASE_FEASIBILITY_PLAN.md').read_bytes()).hexdigest()==ACID_BASE_PROTOCOL_SHA256
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ACID_BASE_PROTOCOL_SHA256
else:acid_json('protocol_lock.json',dict(sha256=ACID_BASE_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_acid_base_outcome_analysis=True))
sources=json.loads((PROJECT/'research/spo2_hemoglobin/source_manifest.json').read_text())['sources']
for key in ['eICU/lab','MIMIC/labevents','MIMIC/d_labitems']:
    info=sources[key];s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns']
def qs(x):return "'"+str(x).replace("'","''")+"'"
con=duckdb.connect(str(PRIVATE/'spo2_acid_base.duckdb'))
con.execute('SET threads=2');con.execute("SET memory_limit='2GB'")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
con.execute(f"ATTACH {qs(PRIVATE/'spo2_hemoglobin.duckdb')} AS hb (READ_ONLY)")
con.execute("""CREATE OR REPLACE TABLE eicu_eligible AS
 SELECT stay_id,exposed,hospital_start_minute FROM hb.eicu_eligible""")
con.execute("""CREATE OR REPLACE TABLE mimic_eligible AS
 SELECT e.stay_id,e.exposed,e.hospital_start_minute,e.admit_time,e.hadm_id,c.person_id
 FROM hb.mimic_eligible e JOIN prior.mimic_cohort c USING(stay_id)""")
assert con.execute('SELECT count(*) FROM eicu_eligible').fetchone()[0]==11452
assert con.execute('SELECT count(*) FROM mimic_eligible').fetchone()[0]==4711
acid_status('arterial acid-base protocol locked; raw extraction pending')
print('Locked outcome-blind acid-base feasibility for the original MIMIC/eICU cohorts.')
'''

EXTRACT = r'''
acid_status('extracting eICU ABG-labelled panels')
con.execute(f"""CREATE OR REPLACE TABLE eicu_raw AS
 SELECT e.stay_id,try_cast(l.labid AS BIGINT) lab_id,
   try_cast(l.labresultoffset AS DOUBLE) event_minute,
   try_cast(l.labresultrevisedoffset AS DOUBLE) available_minute,
   l.labname raw_name,try_cast(l.labtypeid AS INTEGER) lab_type,
   try_cast(l.labresult AS DOUBLE) raw_value,
   l.labmeasurenamesystem unit_system,l.labmeasurenameinterface unit_interface
 FROM read_csv({qs(sources['eICU/lab']['path'])},header=true,all_varchar=true,
   sample_size=10000,strict_mode=true,quote='"',escape='"') l
 JOIN eicu_eligible e ON try_cast(l.patientunitstayid AS BIGINT)=e.stay_id
 WHERE l.labname IN ('pH','paCO2','paO2')
   AND try_cast(l.labresultoffset AS DOUBLE) BETWEEN greatest(-1440,e.hospital_start_minute) AND 240""")
acid_status('extracting explicitly linked MIMIC blood-gas specimens')
con.execute(f"""CREATE OR REPLACE TABLE mimic_raw AS
 SELECT e.stay_id,try_cast(l.labevent_id AS BIGINT) lab_id,
   try_cast(l.specimen_id AS BIGINT) specimen_id,
   date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0 event_minute,
   date_diff('second',e.admit_time,try_cast(l.storetime AS TIMESTAMP))/60.0 available_minute,
   try_cast(l.itemid AS INTEGER) itemid,l.value raw_text,
   try_cast(l.valuenum AS DOUBLE) raw_value,l.valueuom unit_system,'' unit_interface
 FROM read_csv({qs(sources['MIMIC/labevents']['path'])},header=true,all_varchar=true,
   sample_size=10000,strict_mode=true,quote='"',escape='"') l
 JOIN mimic_eligible e ON try_cast(l.subject_id AS BIGINT)=e.person_id
   AND try_cast(l.hadm_id AS BIGINT)=e.hadm_id
 WHERE try_cast(l.itemid AS INTEGER) IN (50820,50818,50821,52033)
   AND date_diff('second',e.admit_time,try_cast(l.charttime AS TIMESTAMP))/60.0
      BETWEEN greatest(-1440,e.hospital_start_minute) AND 240""")
for key in ['eICU/lab','MIMIC/labevents','MIMIC/d_labitems']:
    info=sources[key];s=Path(info['path']).stat();assert s.st_size==info['bytes'] and s.st_mtime_ns==info['mtime_ns']
acid_json('source_manifest.json',dict(fingerprints_reused_from='research/spo2_hemoglobin/source_manifest.json',
    same_size_and_mtime_before_and_after=True,sources={k:sources[k] for k in ['eICU/lab','MIMIC/labevents','MIMIC/d_labitems']},
    outcomes_selected=False,missing_mimic_hadm_id_not_assigned=True))
counts={d:con.execute(f'SELECT count(*) FROM {d}_raw').fetchone()[0] for d in ['eicu','mimic']}
acid_json('raw_counts.json',counts);print('Raw selected laboratory rows:',counts)
con.close();acid_status('raw extraction complete; specimen and unit checks pending')
'''

SUPPORT = r'''
acid_status('specimen, revision, unit and timing support')
con=duckdb.connect(str(PRIVATE/'spo2_acid_base.duckdb'))
raw={d:con.execute(f'SELECT * FROM {d}_raw').df() for d in ['eicu','mimic']}
eligible={d:con.execute(f'SELECT stay_id,exposed FROM {d}_eligible').df() for d in ['eicu','mimic']}
def unit_token(s):return '' if pd.isna(s) else ''.join(str(s).casefold().split())
def normalize_value(row):
    units=[unit_token(row.unit_system),unit_token(row.unit_interface)]
    if row.concept=='ph':
        valid=all(u in {'','ph','units'} for u in units);v=row.raw_value
    else:
        used=[u for u in units if u]
        mapped=[{'mmhg':'mmhg','kpa':'kpa'}.get(u) for u in used]
        valid=bool(mapped) and all(v is not None for v in mapped) and len(set(mapped))==1
        v=row.raw_value*(7.50061683 if valid and mapped[0]=='kpa' else 1)
    ranges={'ph':(6.5,8.5),'pco2':(5,250),'po2':(15,720)}
    if not valid:return np.nan,'unit_rejected'
    if not np.isfinite(v) or not ranges[row.concept][0]<=v<=ranges[row.concept][1]:return np.nan,'range_rejected'
    return float(v),'accepted'
all_support=[];exclusions=[];unit_tables=[]
for d in ['eicu','mimic']:
    r=raw[d].copy()
    if d=='eicu':
        r['concept']=r.raw_name.map({'pH':'ph','paCO2':'pco2','paO2':'po2'})
        r['panel_id']=r.event_minute.map(lambda v:f'{v:.9f}')
        r['source_ok']=r.lab_type.eq(7)
    else:
        r['concept']=r.itemid.map({50820:'ph',50818:'pco2',50821:'po2',52033:'specimen'})
        r['panel_id']=r.specimen_id.astype('string')
        # Specimen provenance uses every available type record; conflicting text is rejected.
        types=r.loc[r.concept.eq('specimen')&r.specimen_id.notna()&r.available_minute.le(240)].copy()
        types['normalized']=types.raw_text.fillna('').str.strip().str.upper()
        type_sets=types.groupby(['stay_id','panel_id']).normalized.agg(lambda x:set(x))
        accepted={k for k,v in type_sets.items() if len(v)==1 and next(iter(v)) in {'ART.','ART','ARTERIAL'}}
        r['source_ok']=[(sid,panel) in accepted for sid,panel in zip(r.stay_id,r.panel_id)]
        source_counts=types.groupby('normalized').agg(records=('lab_id','size'),encounters=('stay_id','nunique')).reset_index()
        acid_json('mimic_specimen_types.json',source_counts.to_dict('records'))
        r=r.loc[r.concept.ne('specimen')].copy()
    r['eligible_revision']=r.available_minute.notna()&r.available_minute.le(240)
    result=[normalize_value(row) for row in r.itertuples(index=False)]
    r['value']=[v[0] for v in result];r['value_status']=[v[1] for v in result]
    u=r.groupby(['concept','unit_system','unit_interface','value_status'],dropna=False).size().reset_index(name='records')
    u['dataset']=d;unit_tables+=u.fillna('').to_dict('records')
    # Choose the latest recorded revision first, then reject invalid/conflicting components.
    candidates=r.loc[r.source_ok&r.eligible_revision&r.panel_id.notna()].copy()
    keys=['stay_id','panel_id','concept']
    latest=candidates.groupby(keys).available_minute.transform('max')
    candidates=candidates.loc[candidates.available_minute.eq(latest)]
    component=[];conflicts=0
    for key,g in candidates.groupby(keys):
        vals=g.value.dropna().unique()
        if len(vals)!=1 or g.value.isna().any():
            conflicts+=1;continue
        component.append(dict(stay_id=key[0],panel_id=key[1],concept=key[2],value=float(vals[0]),
            event_minute=float(g.event_minute.max()),available_minute=float(g.available_minute.max())))
    comp=pd.DataFrame(component,columns=['stay_id','panel_id','concept','value','event_minute','available_minute'])
    if len(comp):
        p=comp.pivot(index=['stay_id','panel_id'],columns='concept',values='value').reindex(columns=['ph','pco2','po2'])
        times=comp.groupby(['stay_id','panel_id'])[['event_minute','available_minute']].max()
        panels=p.join(times).dropna(subset=['ph','pco2','po2']).reset_index()
    else:panels=pd.DataFrame(columns=['stay_id','panel_id','ph','pco2','po2','event_minute','available_minute'])
    panels['respiratory_alkalemia']=(panels.ph>7.45)&(panels.pco2<35)
    pre=panels.loc[panels.event_minute.between(-1440,0)].sort_values(['stay_id','event_minute','available_minute','panel_id']).drop_duplicates('stay_id',keep='last')
    early=panels.loc[panels.event_minute.ge(0)&panels.event_minute.lt(240)]
    for name,frame in [('normalized',r),('components',comp),('panels',panels),('baseline',pre)]:
        con.register('frame',frame);con.execute(f'CREATE OR REPLACE TABLE {d}_{name} AS SELECT * FROM frame');con.unregister('frame')
    cells=pre.merge(eligible[d],on='stay_id',validate='one_to_one').groupby(['exposed','respiratory_alkalemia']).size()
    support=dict(dataset=d,original_dynamics_encounters=len(eligible[d]),complete_panels=len(panels),
        preceding_complete_gas_encounters=len(pre),preceding_gas_available_by_icu_admission=int(pre.available_minute.le(0).sum()),
        first_four_hour_gas_encounters=int(early.stay_id.nunique()),
        preceding_cells=[dict(exposed=x,respiratory_alkalemia=a,encounters=int(cells.get((x,a),0))) for x in [0,1] for a in [False,True]])
    all_support.append(support)
    exclusions.append(dict(dataset=d,component_records=len(r),source_rejected=int((~r.source_ok).sum()),
        unavailable_by_landmark=int((~r.eligible_revision).sum()),unit_rejected=int(r.value_status.eq('unit_rejected').sum()),
        range_rejected=int(r.value_status.eq('range_rejected').sum()),invalid_or_conflicting_latest_components=conflicts))
acid_json('support.json',all_support);acid_json('exclusions.json',exclusions);acid_json('units.json',unit_tables)
acid_json('analysis_manifest.json',dict(protocol_sha256=ACID_BASE_PROTOCOL_SHA256,environment='google_colab' if IN_COLAB else 'local',
    completed_utc=datetime.now(timezone.utc).isoformat(),outcomes_read=False,association_models_fitted=0,
    biological_discovery=False,goal_complete=False,eicu_source_is_abg_label_not_specimen_identifier=True))
con.close()
report=['# Arterial acid–base feasibility','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. No mortality or troponin outcome selected.**','',
    '| Source | Original dynamics cohort | Pre-ICU complete gas | Also available by ICU admission | First-four-hour complete gas |','|---|---:|---:|---:|---:|']
for r in all_support:report.append(f"| {r['dataset']} | {r['original_dynamics_encounters']} | {r['preceding_complete_gas_encounters']} | {r['preceding_gas_available_by_icu_admission']} | {r['first_four_hour_gas_encounters']} |")
report+=['','| Source | SpO2 instability | Preceding respiratory alkalemia | Encounters |','|---|---|---|---:|']
for r in all_support:
    for s in r['preceding_cells']:report.append(f"| {r['dataset']} | {s['exposed']} | {s['respiratory_alkalemia']} | {s['encounters']} |")
report+=['','Respiratory alkalemia here denotes pH>7.45 with PaCO2<35 mmHg, not a definitive primary-disorder diagnosis. MIMIC requires explicit arterial specimen text and explicit hospital linkage. eICU uses ABG-labelled co-timed panels and lacks equivalent specimen identifiers. Units, revision conflicts, availability and physiological ranges are checked before panel selection. Exclusion counts overlap and should not be summed.','',
    'The preceding-state analysis cannot substitute a first-four-hour gas when its pre-ICU measurement is missing. These counts do not establish an association, mechanism, novelty or treatment benefit. No P50 or myocardial oxygen-unloading estimate was calculated. Subsequent association work requires a separate frozen model specification. The discovery goal remains unmet.','']
(PROJECT/'docs/SPO2_ACID_BASE_FEASIBILITY_RESULTS.md').write_text('\n'.join(report))
acid_status('arterial acid-base measurement support','complete');print('\n'.join(report))
'''

VALIDATION = r'''
# Independently reconstruct latest components, complete panels and baseline selection in SQL.
con.close()
con=duckdb.connect(str(PRIVATE/'spo2_acid_base.duckdb'),read_only=True)
validated=[]
for d in ['eicu','mimic']:
    independently=con.execute(f"""WITH latest AS (
      SELECT *,max(available_minute) OVER(PARTITION BY stay_id,panel_id,concept) last_revision
      FROM {d}_normalized WHERE source_ok AND eligible_revision AND panel_id IS NOT NULL
    ), components AS (
      SELECT stay_id,panel_id,concept,max(value) AS "value",max(event_minute) event_minute,
        max(available_minute) available_minute
      FROM latest WHERE available_minute=last_revision GROUP BY stay_id,panel_id,concept
      HAVING count(DISTINCT value)=1 AND count(*)=count(value)
    ), panels AS (
      SELECT stay_id,panel_id,max(event_minute) event_minute,max(available_minute) available_minute,
        max(value) FILTER(WHERE concept='ph') ph,max(value) FILTER(WHERE concept='pco2') pco2,
        max(value) FILTER(WHERE concept='po2') po2
      FROM components GROUP BY stay_id,panel_id HAVING count(DISTINCT concept)=3
    ) SELECT *,ph>7.45 AND pco2<35 respiratory_alkalemia FROM panels
      WHERE event_minute BETWEEN -1440 AND 0
      QUALIFY row_number() OVER(PARTITION BY stay_id ORDER BY event_minute DESC,available_minute DESC,panel_id DESC)=1
      ORDER BY stay_id""").df()
    actual=con.execute(f'SELECT * FROM {d}_baseline ORDER BY stay_id').df()
    assert list(actual.stay_id)==list(independently.stay_id)
    assert list(actual.panel_id.astype(str))==list(independently.panel_id.astype(str))
    for column in ['ph','pco2','po2','event_minute','available_minute']:
        assert np.allclose(actual[column],independently[column],rtol=0,atol=1e-12)
    assert list(actual.respiratory_alkalemia)==list(independently.respiratory_alkalemia)
    assert con.execute(f"""SELECT count(*) FROM {d}_baseline b JOIN {d}_eligible e USING(stay_id)
      WHERE b.event_minute<greatest(-1440,e.hospital_start_minute) OR b.event_minute>0
         OR b.available_minute>240 OR b.ph NOT BETWEEN 6.5 AND 8.5
         OR b.pco2 NOT BETWEEN 5 AND 250 OR b.po2 NOT BETWEEN 15 AND 720""").fetchone()[0]==0
    if d=='eicu':
        assert con.execute("""SELECT count(*) FROM eicu_normalized WHERE source_ok AND lab_type<>7""").fetchone()[0]==0
    else:
        assert con.execute("""SELECT count(*) FROM mimic_baseline b WHERE NOT EXISTS (
          SELECT 1 FROM mimic_raw r WHERE r.stay_id=b.stay_id
            AND cast(r.specimen_id AS VARCHAR)=b.panel_id AND r.itemid=52033
            AND upper(trim(r.raw_text)) IN ('ART.','ART','ARTERIAL') AND r.available_minute<=240)""").fetchone()[0]==0
    validated.append(dict(dataset=d,baseline_encounters_checked=len(actual),sql_pandas_panels_values_times_and_states_match=True))
con.close()
acid_json('independent_validation.json',dict(rows=validated,source_timing_and_ranges_pass=True,outcomes_read=False))
manifest=json.loads((OUT/'analysis_manifest.json').read_text());manifest['independent_validation_pass']=True
acid_json('analysis_manifest.json',manifest)
acid_status('arterial acid-base support and independent validation','complete')
print('Independent SQL/Pandas baseline reconstruction, specimen-source and timing checks passed.')
'''

MORTALITY = r'''
import sys,json,hashlib,importlib.util
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import beta,binomtest
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
DRIVE=Path('/content/drive/MyDrive') if IN_COLAB else Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph';PRIVATE=DRIVE/'Data/PhysioGraph_Biological_Discovery_20260905'
OUT=PROJECT/'research/spo2_acid_base/mortality_screen';OUT.mkdir(exist_ok=True)
def screen_json(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
assert hashlib.sha256((PROJECT/'docs/SPO2_ACID_BASE_MORTALITY_SCREEN_PLAN.md').read_bytes()).hexdigest()==ACID_BASE_MORTALITY_PROTOCOL_SHA256
assert json.loads((PROJECT/'research/spo2_acid_base/analysis_manifest.json').read_text())['independent_validation_pass']
lock=OUT/'protocol_lock.json'
if lock.exists():assert json.loads(lock.read_text())['sha256']==ACID_BASE_MORTALITY_PROTOCOL_SHA256
else:screen_json('protocol_lock.json',dict(sha256=ACID_BASE_MORTALITY_PROTOCOL_SHA256,utc=datetime.now(timezone.utc).isoformat(),before_candidate_mortality_selected=True))
screen_json('run_status.json',dict(status='running',stage='frozen crude mortality screen',environment='google_colab' if IN_COLAB else 'local'))
con=duckdb.connect(str(PRIVATE/'spo2_acid_base.duckdb'),read_only=True)
def qs(x):return "'"+str(x).replace("'","''")+"'"
con.execute(f"ATTACH {qs(PRIVATE/'spo2_context.duckdb')} AS prior (READ_ONLY)")
selected={}
for d in ['eicu','mimic']:
    frame=con.execute(f"""SELECT b.stay_id,b.respiratory_alkalemia,e.exposed,
      cast(c.person_id AS VARCHAR) person_id
      FROM {d}_baseline b JOIN {d}_eligible e USING(stay_id)
      JOIN prior.{d}_cohort c USING(stay_id)""").df()
    assert frame.person_id.notna().all() and frame.stay_id.is_unique
    frame['selection_hash']=[hashlib.sha256(f'20260905:{d}:{p}:{s}'.encode()).hexdigest() for p,s in zip(frame.person_id,frame.stay_id)]
    selected[d]=frame.sort_values(['person_id','selection_hash']).drop_duplicates('person_id').copy()
    selected[d].to_csv(PRIVATE/f'acid_base_mortality_selected_{d}.csv',index=False)
screen_json('selection_manifest.json',dict(utc=datetime.now(timezone.utc).isoformat(),selected_before_mortality_join=True,
    selected_people={d:len(f) for d,f in selected.items()},
    private_selection_sha256={d:hashlib.sha256((PRIVATE/f'acid_base_mortality_selected_{d}.csv').read_bytes()).hexdigest() for d in selected}))
alpha=.05/8;cells=[];interactions=[];validation=[]
for d,f in selected.items():
    con.register('selected_ids',f)
    result=con.execute(f"""SELECT s.*,c.hospital_mortality,c.excluded_before_landmark_flag
      FROM selected_ids s JOIN prior.{d}_cohort c USING(stay_id)""").df()
    con.unregister('selected_ids')
    assert len(result)==len(f) and result.person_id.is_unique
    assert result.excluded_before_landmark_flag.eq(0).all()
    assert set(result.hospital_mortality.dropna().unique())<={0.,1.}
    result.to_csv(PRIVATE/f'acid_base_mortality_analysis_{d}.csv',index=False)
    for x in [0,1]:
        for a in [False,True]:
            g=result.loc[result.exposed.eq(x)&result.respiratory_alkalemia.eq(a)]
            y=g.hospital_mortality.dropna();n=len(y);k=int(y.sum())
            lo=float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.
            hi=float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.
            if n:
                independent=binomtest(k,n).proportion_ci(confidence_level=1-alpha,method='exact')
                assert abs(independent.low-lo)<1e-10 and abs(independent.high-hi)<1e-10
                validation.append(dict(dataset=d,exposed=x,respiratory_alkalemia=a,
                    maximum_interval_difference=max(abs(independent.low-lo),abs(independent.high-hi))))
            cells.append(dict(dataset=d,exposed=x,respiratory_alkalemia=a,selected_patients=len(g),
                known_mortality=n,missing_mortality=len(g)-n,deaths=k,risk=k/n if n else None,
                simultaneous_lower=lo,simultaneous_upper=hi))
    lookup={(r['exposed'],r['respiratory_alkalemia']):r for r in cells if r['dataset']==d}
    terms=[((1,True),1),((0,True),-1),((1,False),-1),((0,False),1)]
    estimate=sum(sign*lookup[key]['risk'] for key,sign in terms) if all(lookup[key]['risk'] is not None for key,sign in terms) else None
    lower=sum(sign*lookup[key]['simultaneous_lower' if sign>0 else 'simultaneous_upper'] for key,sign in terms)
    upper=sum(sign*lookup[key]['simultaneous_upper' if sign>0 else 'simultaneous_lower'] for key,sign in terms)
    interactions.append(dict(dataset=d,interaction_risk_difference=estimate,simultaneous_lower=lower,
        simultaneous_upper=upper,screen_pass=bool(estimate is not None and lower>.05)))
con.close();advance=all(r['screen_pass'] for r in interactions)
screen_json('cell_risks.json',cells);screen_json('interaction_results.json',interactions)
screen_json('independent_interval_validation.json',dict(beta_quantiles_match_binomial_test_inversion=True,rows=validation))
screen_json('analysis_manifest.json',dict(protocol_sha256=ACID_BASE_MORTALITY_PROTOCOL_SHA256,
    environment='google_colab' if IN_COLAB else 'local',completed_utc=datetime.now(timezone.utc).isoformat(),
    all_eight_intervals_familywise_coverage_target=.95,confidence_method='Bonferroni Clopper-Pearson risk bounds propagated through the linear interaction',
    one_encounter_per_person_selected_before_outcomes=True,crude_unadjusted=True,association_screen_advance=advance,
    troponin_outcomes_read=False,causal_effect_estimated=False,biological_discovery=False,goal_complete=False))
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
report=['# Acid–base mortality interaction screen','',
    '**Actual execution: '+('Google Colab' if IN_COLAB else 'local')+'. Crude observational patient-level screen; no treatment-effect estimate.**','',
    '| Source | Instability | Respiratory alkalemia | Deaths / known mortality | Missing mortality | Risk | Simultaneous risk interval |',
    '|---|---|---|---:|---:|---:|---|']
for r in cells:report.append(f"| {r['dataset']} | {r['exposed']} | {r['respiratory_alkalemia']} | {r['deaths']} / {r['known_mortality']} | {r['missing_mortality']} | {pct(r['risk'])} | {pct(r['simultaneous_lower'])} to {pct(r['simultaneous_upper'])} |")
report+=['','| Source | Interaction in absolute mortality risk | Simultaneous conservative bounds | Pass |','|---|---:|---|---|']
for r in interactions:report.append(f"| {r['dataset']} | {pct(r['interaction_risk_difference'])} | {pct(r['simultaneous_lower'])} to {pct(r['simultaneous_upper'])} | {r['screen_pass']} |")
report+=['',f'Both-database advancement: **{advance}**. The fixed rule requires each simultaneous lower interaction bound to exceed five percentage points. The interaction subtracts the instability-associated risk difference without respiratory alkalemia from that with it. These are percentage-point contrasts, not relative risks.','',
    'Eight two-sided Clopper–Pearson intervals use alpha=0.05/8; the interaction bounds follow by interval arithmetic. Their simultaneous coverage is conditional on the independent-patient binomial model. A deterministic outcome-blind choice retains one encounter per person. Unknown mortality is reported and not imputed.','',
    'These estimates are unadjusted and cannot isolate oxygen unloading, coronary vascular effects, severity, treatment, or selection into blood-gas measurement. Small joint-state groups sharply limit precision. No non-advancement establishes biological absence, and no positive percentage establishes a mechanism or mortality-saving intervention. Both project databases have been explored previously. No novel biological discovery is established.','']
(PROJECT/'docs/SPO2_ACID_BASE_MORTALITY_SCREEN_RESULTS.md').write_text('\n'.join(report))
screen_json('run_status.json',dict(status='complete',stage='frozen crude mortality screen',environment='google_colab' if IN_COLAB else 'local',goal_complete=False))
print('\n'.join(report))
'''
