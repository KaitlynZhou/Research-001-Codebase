"""Audit patient separation and cross-dataset membership before training."""
from pathlib import Path
import json
import hashlib
import subprocess
import pandas as pd
import numpy as np
from common import BASE, RESULTS

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    d2={s:pd.read_csv(BASE/f'{s}.csv') for s in ['train','val','test']}
    d1={s:pd.read_csv(f'data/processed/dataset1/{s}.csv') for s in ['train','val','test']}
    audit={}
    for name,parts in [('dataset1',d1),('dataset2',d2)]:
        audit[name]={}
        for a,b in [('train','val'),('train','test'),('val','test')]:
            overlap=set(parts[a].patient_id)&set(parts[b].patient_id)
            assert not overlap, f'{name} {a}/{b} patient leakage'
            assert not set(parts[a].ecg_id)&set(parts[b].ecg_id)
            audit[name][a+'_'+b+'_patient_overlap']=len(overlap)
        for split,df in parts.items():
            assert df.ecg_id.is_unique
            assert df.strat_fold.isin({'train':range(1,9),'val':[9],'test':[10]}[split]).all()
            audit[name][split]={'records':len(df),'patients':df.patient_id.nunique(),
                              'positives':{c:int(df[c].sum()) for c in ['CLBBB','ILBBB']}}
    for s in d1:
        reference=d2[s].set_index('ecg_id').loc[d1[s].ecg_id].reset_index()
        assert np.array_equal(d1[s].to_numpy(),reference[d1[s].columns].to_numpy())
    for s in ['train','val']:
        assert not set(d1['test'].patient_id)&set(d2[s].patient_id)
    audit['split_metadata_sha256']={}
    for path in [*BASE.glob('*.csv'),*Path('data/processed/dataset1').glob('*.csv')]:
        audit['split_metadata_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    audit['dataset1_test_subset_of_dataset2_test']=True
    audit['cross_dataset_test_vs_training_validation_patient_overlap']=0
    for split in ['train','val','test','test_dataset1']:
        x=np.load(BASE/f'X_{split}.npy',mmap_mode='r')
        y=np.load(BASE/f'y_{split}.npy',mmap_mode='r')
        assert x.shape==(len(y),6,5000) and y.shape[1]==2
        assert np.isin(y,[0,1]).all()
        for start in range(0,len(x),128):
            assert np.isfinite(x[start:start+128]).all()
        expected=(d1['test'] if split=='test_dataset1' else d2[split])[['CLBBB','ILBBB']].to_numpy()
        assert np.array_equal(y,expected)
    (RESULTS/'leakage_checks.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
