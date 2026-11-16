"""Construct Dataset 2 from exactly the PTB-XL recordings used by Experiment 1."""
from pathlib import Path
import ast
import pandas as pd

RAW_DIR = Path('data/raw/ptb-xl')
OUT_DIR = Path('data/processed/dataset2')


def main():
    df = pd.read_csv(RAW_DIR / 'ptbxl_database.csv')
    assert df.ecg_id.is_unique, 'Duplicate ECG IDs'
    codes = df.scp_codes.apply(ast.literal_eval)
    df['CLBBB'] = codes.map(lambda x: int('CLBBB' in x))
    df['ILBBB'] = codes.map(lambda x: int('ILBBB' in x))
    columns = ['ecg_id','patient_id','CLBBB','ILBBB','strat_fold','filename_lr']
    dataset = df[columns].copy()
    assert dataset.strat_fold.between(1, 10).all()
    assert dataset.groupby('patient_id').strat_fold.nunique().eq(1).all(), 'Patient split leakage'
    assert not ((dataset["CLBBB"] == 1) & (dataset.ILBBB == 1)).any(), "Overlapping labels"
    out = {'metadata': dataset,
           'train': dataset[dataset.strat_fold <= 8],
           'val': dataset[dataset.strat_fold == 9],
           'test': dataset[dataset.strat_fold == 10]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in out.items():
        part.to_csv(OUT_DIR / f'{name}.csv',index=False)
        print(f'{name}: N={len(part)} CLBBB={part["CLBBB"].sum()} ILBBB={part.ILBBB.sum()}')
    # Cross-check patient/fold/labels for overlapping ECGs with Experiment 1.
    d1_path = Path('data/processed/dataset1/metadata.csv')
    if d1_path.exists():
        d1 = pd.read_csv(d1_path).set_index('ecg_id')
        d2 = dataset.set_index('ecg_id')
        assert d1.index.isin(d2.index).all(), 'Dataset 1 not contained in Dataset 2'
        for col in ['patient_id','strat_fold','CLBBB','ILBBB','filename_lr']:
            assert d1[col].equals(d2.loc[d1.index,col]), f'Dataset mismatch: {col}'
        print('Cross-dataset ECG IDs, folds, patients and labels match.')
    else:
        print('Dataset 1 metadata not found; cross-dataset checks deferred to step 07.')

if __name__ == '__main__':
    main()
