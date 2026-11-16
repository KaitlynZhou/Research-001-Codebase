"""Check waveform array alignment and patient-level splits before training."""
from pathlib import Path
import pandas as pd
import numpy as np

BASE=Path('data/processed/dataset2')


def main():
    all_parts={}
    for part in ['train','val','test']:
        metadata=pd.read_csv(BASE/f'{part}.csv')
        X=np.load(BASE/f'X_{part}.npy',mmap_mode='r')
        y=np.load(BASE/f'y_{part}.npy',mmap_mode='r')
        assert X.shape==(len(metadata),12,1000)
        assert y.shape==(len(metadata),2)
        assert np.array_equal(y,metadata[['CLBBB','ILBBB']].to_numpy())
        assert np.isfinite(X).all()
        required={'train':list(range(1,9)),'val':[9],'test':[10]}[part]
        assert metadata.strat_fold.isin(required).all()
        all_parts[part]=metadata
        print(f'{part}: N={len(metadata)}, X={X.shape}, positives={y.sum(axis=0)}')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert set(all_parts[a].patient_id).isdisjoint(all_parts[b].patient_id),f'Patient leakage: {a}/{b}'
    d1=pd.read_csv('data/processed/dataset1/test.csv')
    d2=all_parts['test'].set_index('ecg_id')
    assert d1.ecg_id.isin(d2.index).all()
    for col in ['patient_id','CLBBB','ILBBB','strat_fold','filename_lr']:
        assert np.array_equal(d1[col],d2.loc[d1.ecg_id,col]),f'Dataset1/2 mismatch: {col}'
    X1=np.load(BASE/'X_test_dataset1.npy',mmap_mode='r')
    y1=np.load(BASE/'y_test_dataset1.npy',mmap_mode='r')
    assert X1.shape==(len(d1),12,1000) and y1.shape==(len(d1),2)
    assert np.array_equal(y1,d1[['CLBBB','ILBBB']].to_numpy())
    print('PASS: no patient leakage; aligned labels; Dataset 1 test subset matches Dataset 2.')

if __name__=='__main__':main()
