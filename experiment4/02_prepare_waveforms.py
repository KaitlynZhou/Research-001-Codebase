"""Experiment 1 waveform preprocessing (500 Hz, 6 x 5000), saved as NPY arrays.

Use disk-backed NPY files to avoid holding Dataset 2 in RAM.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from numpy.lib.format import open_memmap

RAW_DIR = Path('data/raw/ptb-xl')
D2 = Path('data/processed/experiment4/dataset2')
D1 = Path('data/processed/dataset1')
# PTB-XL usually stores limb leads as AVR/AVL/AVF in WFDB metadata.
# Case-insensitive normalization permits aVR/aVL/aVF but preserves channel order.
EXPECTED_LEADS = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']


def prepare(metadata_file, out_dir, suffix):
    metadata = pd.read_csv(metadata_file)
    if metadata.empty:
        raise ValueError(f'Empty metadata: {metadata_file}')
    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f'X_{suffix}.npy'
    y_path = out_dir / f'y_{suffix}.npy'
    X = open_memmap(x_path, mode='w+', dtype='float32', shape=(len(metadata),6,5000))
    y = open_memmap(y_path, mode='w+', dtype='float32', shape=(len(metadata),2))
    source = pd.read_csv(RAW_DIR / 'ptbxl_database.csv').set_index('ecg_id')
    for i,row in enumerate(metadata.itertuples(index=False)):
        path = RAW_DIR / source.loc[row.ecg_id, 'filename_hr']
        signal,fields = wfdb.rdsamp(str(path))
        if fields['fs'] != 500 or signal.shape != (5000,12):
            raise ValueError(f'Unexpected frequency/shape at {path}')
        if [s.upper() for s in fields['sig_name']] != EXPECTED_LEADS:
            raise ValueError(f'Unexpected lead order at {path}: {fields["sig_name"]}')
        if not np.isfinite(signal).all():
            raise ValueError(f'Non-finite waveform: {path}')
        X[i] = signal[:,[0,4,6,7,10,11]].T.astype(np.float32)
        y[i] = (row.CLBBB,row.ILBBB)
        if (i+1) % 1000 == 0: print(f'{suffix}: {i+1}/{len(metadata)}')
    X.flush();y.flush()
    print(f'{suffix}: X={X.shape}, y={y.shape} saved under {out_dir}')


def main():
    for part in ['train','val','test']:
        prepare(D2 / f'{part}.csv',D2,part)
    # Required cross-dataset test: use Dataset 1 fold-10, never training data.
    if not (D1 / 'test.csv').exists():
        raise FileNotFoundError('Create Dataset 1 via Experiment 1 step 02 first: data/processed/dataset1/test.csv')
    prepare(D1 / 'test.csv', D2, 'test_dataset1')

if __name__=='__main__':
    main()
