"""Build exact Prisha cohorts and six physical-unit leads from PTB-XL 1.0.3."""
import argparse, ast, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

LEADS = ['I','AVL','V1','V2','V5','V6']
OUTCOMES = ['CLBBB','ILBBB']

def cohorts(raw):
    m = pd.read_csv(raw/'ptbxl_database.csv')
    scp = pd.read_csv(raw/'scp_statements.csv',index_col=0)
    codes = m.scp_codes.map(ast.literal_eval)
    for label in OUTCOMES:
        m[label] = codes.map(lambda c:int(label in c))
    abnormal = set(scp.index[scp.diagnostic.eq(1)]) - {'NORM'}
    m['dataset1'] = [bool(a or b or ('NORM' in c and not (set(c)&abnormal)))
                     for c,a,b in zip(codes,m.CLBBB,m.ILBBB)]
    assert m.ecg_id.is_unique and m.patient_id.notna().all()
    assert m.strat_fold.between(1,10).all()
    assert m.groupby('patient_id').strat_fold.nunique().eq(1).all(), 'Patient leakage'
    return m

def prepare(raw, destination, archive=None):
    destination.mkdir(parents=True,exist_ok=True)
    m = cohorts(raw)
    groups = {'train': m[m.dataset1 & m.strat_fold.le(8)],
              'val': m[m.dataset1 & m.strat_fold.eq(9)],
              'dataset2_test': m[m.strat_fold.eq(10)]}
    # Read test waveforms only for preparation; never used by training/model selection.
    selected = pd.concat(list(groups.values()))
    if archive:
        with zipfile.ZipFile(archive) as z:
            prefix = next(n for n in z.namelist() if n.endswith('/ptbxl_database.csv')).rsplit('/',1)[0]+'/'
            for f in selected.filename_lr:
                for suffix in ['.hea','.dat']:
                    p = raw/(f+suffix); p.parent.mkdir(parents=True,exist_ok=True)
                    if not p.exists(): p.write_bytes(z.read(prefix+f+suffix))
    audit = {'version':'PTB-XL 1.0.3','leads':LEADS,'outcomes':OUTCOMES,
             'patient_disjoint': True, 'splits':{},
             'metadata_sha256':hashlib.sha256((raw/'ptbxl_database.csv').read_bytes()).hexdigest()}
    m.to_csv(destination/'all_metadata.csv',index=False)
    for name, frame in groups.items():
        frame.to_csv(destination/f'{name}.csv',index=False)
        x = np.lib.format.open_memmap(destination/f'X_{name}.npy',mode='w+',dtype='float32',shape=(len(frame),6,1000))
        for i, r in enumerate(frame.itertuples()):
            signal, fields = wfdb.rdsamp(str(raw/r.filename_lr))
            names = [n.upper() for n in fields['sig_name']]
            assert fields['fs']==100 and signal.shape==(1000,12)
            assert len(names)==len(set(names))==12
            assert np.isfinite(signal).all()
            x[i] = signal[:,[names.index(n) for n in LEADS]].T.astype('float32')
        x.flush(); del x
        np.save(destination/f'y_{name}.npy',frame[OUTCOMES].to_numpy(dtype='float32'))
        audit['splits'][name] = {'n':len(frame), 'patients':int(frame.patient_id.nunique()),
                               **{c:int(frame[c].sum()) for c in OUTCOMES}}
        print(name,audit['splits'][name],flush=True)
    t = groups['dataset2_test']; t[t.dataset1].to_csv(destination/'dataset1_test.csv',index=False)
    audit['dataset1_test_subset_n'] = int(t.dataset1.sum())
    audit['test_cohorts_overlap'] = True
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--zip',type=Path)
    a=p.parse_args();prepare(a.raw,a.out,a.zip)
