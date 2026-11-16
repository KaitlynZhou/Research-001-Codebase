"""Run from repository root after downloading PTB-XL 1.0.3 into data/raw.

Usage: .venv/bin/python -u experiment4/run.py
Archive: data/raw/ptb-xl-1.0.3.zip (official PhysioNet download).
Existing team dataset builders are reused unchanged except Dataset 2 output path.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/ptb-xl'
RESULTS=ROOT/'results/experiment4'

def main():
    archive=ROOT/'data/raw/ptb-xl-1.0.3.zip'
    RAW.mkdir(parents=True,exist_ok=True)
    RESULTS.mkdir(parents=True,exist_ok=True)
    checked=0
    with zipfile.ZipFile(archive) as z:
        manifests=[n for n in z.namelist() if n.endswith('/SHA256SUMS.txt')]
        assert len(manifests)==1
        checksums={}
        for line in z.read(manifests[0]).decode().splitlines():
            digest,name=line.split(maxsplit=1)
            checksums[name.lstrip('*./')]=digest
        for name in z.namelist():
            relative='/'.join(name.split('/')[1:])
            if not relative or name.endswith('/'):
                continue
            if not (relative.startswith('records500/') or relative in ['scp_statements.csv','ptbxl_database.csv']):
                continue
            content=z.read(name)
            assert hashlib.sha256(content).hexdigest()==checksums[relative],relative
            destination=RAW/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            if relative=='ptbxl_database.csv' and destination.exists():
                assert destination.read_bytes()==content,'Local metadata differs from official archive'
            else:
                destination.write_bytes(content)
            checked+=1
    (RESULTS/'raw_verification.json').write_text(json.dumps({'dataset_version':'1.0.3','source':'https://physionet.org/content/ptb-xl/1.0.3/','native_500hz_files_and_metadata_verified':checked,'sha256_match':True},indent=2))
    for script in ['00_build_dataset1.py','01_build_dataset2.py','02_prepare_waveforms.py','08_verify.py','05_train_resnet.py','06_evaluate_validation.py','07_evaluate_test.py','09_summarize.py']:
        print('Running:',script,flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'experiment4'/script)],cwd=ROOT,check=True)
    print('Experiment 4 training and both evaluations complete.',flush=True)

if __name__=='__main__':main()
