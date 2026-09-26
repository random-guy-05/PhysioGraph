"""Notebook cells for the targeted SpO2-variability replication."""

SETUP = r'''
import importlib.metadata,importlib.util,subprocess,sys
from pathlib import Path
requirements={'duckdb':'duckdb==1.4.3','numpy':'numpy==2.2.6','pandas':'pandas==2.2.3','scipy':'scipy==1.15.3','statsmodels':'statsmodels==0.14.5','matplotlib':'matplotlib==3.10.6'}
missing=[pin for module,pin in requirements.items() if importlib.util.find_spec(module) is None or importlib.metadata.version(module)!=pin.split('==')[1]]
if missing:subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])
IN_COLAB=importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
if IN_COLAB:
 from google.colab import drive
 drive.mount('/content/drive');DRIVE=Path('/content/drive/MyDrive')
else:DRIVE=Path.home()/'Library/CloudStorage/GoogleDrive-2arnavmana@gmail.com/My Drive'
PROJECT=DRIVE/'Projects/PhysioGraph'
'''

FREEZE = r'''
subprocess.check_call([sys.executable,str(PROJECT/'scripts/run_spo2_variability_validation.py'),'--freeze-only'],cwd=PROJECT)
'''

VALIDATE = r'''
subprocess.check_call([sys.executable,str(PROJECT/'scripts/run_spo2_variability_validation.py')],cwd=PROJECT)
'''
