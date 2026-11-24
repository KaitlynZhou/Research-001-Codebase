"""Build V1-V4 arrays for full 10-s ECGs and PTB-XL+ median beats on identical record sets.

Cohorts (labels, folds, Dataset 1 rule) come unchanged from diagnostic_leads/prepare.py.
Only records with a published PTB-XL+ 12SL median beat are kept, in every split and for both
input types, so the full-vs-median comparison is made on exactly the same recordings.
"""
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from common import cohorts, read_median, median_relpath, LEADS, OUTCOMES, INPUTS


def splits(m):
    return {'train': m[m.dataset1 & m.strat_fold.le(8)],
            'val': m[m.dataset1 & m.strat_fold.eq(9)],
            'dataset2_test': m[m.strat_fold.eq(10)]}


def extract_hr(raw, archive, files):
    with zipfile.ZipFile(archive) as z:
        prefix = next(n for n in z.namelist() if n.endswith('/ptbxl_database.csv')).rsplit('/', 1)[0] + '/'
        for f in files:
            for suffix in ('.hea', '.dat'):
                p = raw / (f + suffix)
                if not p.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(z.read(prefix + f + suffix))


def prepare(raw, medians, out, archive=None):
    out.mkdir(parents=True, exist_ok=True)
    m = cohorts(raw)
    m['has_median'] = [(medians / (median_relpath(i) + '.dat')).exists() for i in m.ecg_id]
    groups = splits(m)
    audit = {'source': 'PTB-XL 1.0.3 records500 + PTB-XL+ 1.0.1 median_beats/12sl', 'leads': LEADS,
             'outcomes': OUTCOMES, 'inputs': INPUTS, 'units': 'mV, no filtering or normalization',
             'metadata_sha256': hashlib.sha256((raw / 'ptbxl_database.csv').read_bytes()).hexdigest(),
             'excluded_without_median': {}, 'splits': {}}
    for name, g in groups.items():
        gone = g[~g.has_median]
        audit['excluded_without_median'][name] = {'n': len(gone), **{c: int(gone[c].sum()) for c in OUTCOMES}}
    groups = {k: g[g.has_median].reset_index(drop=True) for k, g in groups.items()}
    if archive:
        extract_hr(raw, archive, pd.concat(groups.values()).filename_hr)
    amplitude = []
    for name, g in groups.items():
        g.to_csv(out / f'{name}.csv', index=False)
        xf = np.lib.format.open_memmap(out / f'X_full_{name}.npy', mode='w+', dtype='float32', shape=(len(g), 4, 5000))
        xm = np.lib.format.open_memmap(out / f'X_median_{name}.npy', mode='w+', dtype='float32', shape=(len(g), 4, 600))
        for i, r in enumerate(g.itertuples()):
            signal, fields = wfdb.rdsamp(str(raw / r.filename_hr))
            names = [s.upper() for s in fields['sig_name']]
            assert fields['fs'] == 500 and signal.shape == (5000, 12) and len(set(names)) == 12
            assert all(u.lower() == 'mv' for u in fields['units']) and np.isfinite(signal).all()
            xf[i] = signal[:, [names.index(l) for l in LEADS]].T
            beat = read_median(medians / median_relpath(r.ecg_id))
            xm[i] = np.stack([beat[l] for l in LEADS])
            if i % 25 == 0:  # unit check: median-beat amplitude should match the raw ECG's beats
                full_ptp = np.percentile(xf[i], 99.5, axis=1) - np.percentile(xf[i], 0.5, axis=1)
                amplitude.append(np.ptp(xm[i], axis=1) / np.maximum(full_ptp, 1e-3))
        assert np.isfinite(xm).all()
        xf.flush(); xm.flush(); del xf, xm
        np.save(out / f'y_{name}.npy', g[OUTCOMES].to_numpy(dtype='float32'))
        audit['splits'][name] = {'n': len(g), 'patients': int(g.patient_id.nunique()),
                                 **{c: int(g[c].sum()) for c in OUTCOMES}}
        print(name, audit['splits'][name], flush=True)
    t = groups['dataset2_test']
    audit['splits']['dataset1_test'] = {'n': int(t.dataset1.sum()), **{c: int(t.loc[t.dataset1, c].sum()) for c in OUTCOMES}}
    ratio = np.concatenate(amplitude)
    audit['median_to_full_amplitude_ratio'] = {'median': float(np.median(ratio)), 'p5': float(np.percentile(ratio, 5)),
                                               'p95': float(np.percentile(ratio, 95)), 'records_sampled': len(amplitude)}
    assert 0.5 < audit['median_to_full_amplitude_ratio']['median'] < 1.5, 'Median-beat units do not match the raw ECG'
    train_p, eval_p = set(groups['train'].patient_id) | set(groups['val'].patient_id), set(t.patient_id)
    assert not set(groups['train'].patient_id) & set(groups['val'].patient_id) and not train_p & eval_p
    audit['patient_disjoint'] = True
    (out / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit['excluded_without_median']), json.dumps(audit['median_to_full_amplitude_ratio']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True, help='PTB-XL 1.0.3 directory (metadata CSVs; records500 extracted here)')
    p.add_argument('--medians', type=Path, required=True, help='PTB-XL+ root containing median_beats/12sl')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--zip', type=Path, help='PTB-XL 1.0.3 ZIP to extract the needed 500 Hz records from')
    a = p.parse_args(); prepare(a.raw, a.medians, a.out, a.zip)
