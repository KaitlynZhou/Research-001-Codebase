"""Build a conservative, auditable CLBBB/normal cohort; never download signals."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from .inspect_metadata import inspect, parse_codes

POLICY = 'strict-v1: target=100; only target and optional SR; all other code presence excludes'


def classify(codes):
    if 'ILBBB' in codes:
        return None, 'incomplete_LBBB'
    targets = set(codes) & {'CLBBB', 'NORM'}
    if len(targets) != 1:
        return None, 'conflicting_or_no_target'
    target = next(iter(targets))
    if codes[target] != 100:
        return None, 'uncertain_target'
    if set(codes) - {target, 'SR'}:
        return None, 'additional_statement'
    return int(target == 'CLBBB'), 'selected'


def verify_splits(df):
    if df.patient_id.isna().any() or df.ecg_id.isna().any():
        raise ValueError('Missing recording or patient IDs')
    if df.ecg_id.duplicated().any() or df.filename_lr.duplicated().any():
        raise ValueError('Duplicate recording ID or waveform path')
    if df.groupby('patient_id').strat_fold.nunique().gt(1).any():
        raise ValueError('Patient leakage between folds')
    if not df.strat_fold.isin(range(1, 11)).all():
        raise ValueError('Invalid folds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data/raw/ptb-xl'))
    p.add_argument('--output', type=Path, default=Path('data/processed'))
    args = p.parse_args()
    report = inspect(args.data_root)  # Existing independent full-source validation.
    df = pd.read_csv(args.data_root / 'ptbxl_database.csv')
    statements = pd.read_csv(args.data_root / 'scp_statements.csv', index_col=0)
    if statements.index.has_duplicates or statements.loc['SR', 'description'] != 'sinus rhythm':
        raise ValueError('Unexpected statement table or SR definition')
    required = ['ecg_id', 'patient_id', 'scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']
    if df[required].isna().any().any():
        raise ValueError('Missing required metadata')
    for col in ['ecg_id', 'patient_id', 'strat_fold']:
        values = pd.to_numeric(df[col], errors='raise')
        if not (values % 1 == 0).all():
            raise ValueError(f'Noninteger {col}')
        df[col] = values.astype('int64')
    verify_splits(df)
    decisions = df.scp_codes.map(lambda v: classify(parse_codes(v)))
    df['label'] = pd.array([v[0] for v in decisions], dtype='Int64')
    df['selection_reason'] = [v[1] for v in decisions]
    df['split'] = df.strat_fold.map(lambda f: 'train' if f <= 8 else 'val' if f == 9 else 'test')
    selected = df.loc[df.label.notna(), required + ['label', 'split']].copy()
    verify_splits(selected)
    for split in ['train', 'val', 'test']:
        if set(selected.loc[selected.split == split, 'label']) != {0, 1}:
            raise ValueError(f'{split} is missing a class')
    counts = selected.groupby(['strat_fold', 'split', 'label']).agg(
        recordings=('ecg_id', 'size'), patients=('patient_id', 'nunique')).reset_index()
    report.update({
        'policy': POLICY,
        'source_sha256': {name: hashlib.sha256((args.data_root / name).read_bytes()).hexdigest()
                          for name in ['ptbxl_database.csv', 'scp_statements.csv']},
        'selected_by_class': json.loads(selected.groupby('label').agg(
            recordings=('ecg_id', 'size'), patients=('patient_id', 'nunique')).to_json(orient='index')),
        'selected_by_split_class': json.loads(selected.groupby(['split', 'label']).agg(
            recordings=('ecg_id', 'size'), patients=('patient_id', 'nunique')).reset_index().to_json(orient='records')),
        'exclusion_counts': df.selection_reason.value_counts().to_dict(),
        'checks': {'duplicate_recordings': 0, 'missing_required_fields': 0,
                   'unknown_codes': 0, 'patient_overlap_between_folds': 0,
                   'patient_overlap_between_splits': 0},
        'waveform_headers_present': sum((args.data_root / f'{v}.hea').is_file() for v in selected.filename_lr),
        'dataset_version': 'Not established from filenames; match waveform release to metadata before extraction',
    })
    args.output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output / 'selected_recordings.csv', index=False)
    df[['ecg_id', 'patient_id', 'scp_codes', 'label', 'selection_reason']].to_csv(
        args.output / 'selection_audit.csv', index=False)
    counts.to_csv(args.output / 'fold_counts.csv', index=False)
    (args.output / 'metadata_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['selected_by_class', 'selected_by_split_class', 'exclusion_counts', 'checks', 'waveform_headers_present']}, indent=2))
    print(counts.to_string(index=False))


if __name__ == '__main__':
    main()
