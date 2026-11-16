"""Write a compact report and checksums after both held-out evaluations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import RESULTS

def main():
    metrics=pd.read_csv(RESULTS/'both_test_metrics.csv')
    checkpoint=torch.load(RESULTS/'best_model.pt',map_location='cpu',weights_only=True)
    d1=pd.read_csv(RESULTS/'dataset1_test_predictions.csv').set_index('ecg_id')
    d2=pd.read_csv(RESULTS/'dataset2_test_predictions.csv').set_index('ecg_id').loc[d1.index]
    cols=['CLBBB_score','ILBBB_score']
    assert np.allclose(d1[cols],d2[cols],atol=1e-5), 'Cross-dataset scores differ for shared ECGs'
    summary={'best_epoch':checkpoint['epoch'],'best_validation_loss':checkpoint['val_loss'],
             'threshold':0.5,'confusion_matrix_order':[['TN','FP'],['FN','TP']],
             'shared_test_score_max_difference':float(np.max(np.abs(d1[cols].to_numpy()-d2[cols].to_numpy()))),
             'metrics':metrics.to_dict(orient='records')}
    history=json.loads((RESULTS/'history.json').read_text())
    summary['completed_epochs']=len(history)
    summary['training_seconds']=history[-1]['elapsed_seconds']
    summary['device']='mps' if torch.backends.mps.is_available() else 'cpu'
    summary['training_positive_weights']=checkpoint['pos_weight'].tolist()
    files=[p for p in RESULTS.iterdir() if p.suffix in ['.pt','.npy','.csv','.json'] and p.name!='summary.json']
    summary['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Experiment 4: Dataset 2, ResNet, conventional diagnostic leads','',
           'Trained one ResNet from scratch on Dataset 2 using I, aVL, V1, V2, V5, V6 at native 500 Hz. '+
           'Separate sigmoid outputs predict CLBBB and ILBBB. Physical WFDB signals are used without normalization, matching the team preprocessing.', '',
           'Patient folds: 1–8 training, 9 validation, 10 testing. '+
           'The best checkpoint uses validation loss only; both held-out evaluations use threshold 0.5. '+
           'Dataset 1 test is a subset of Dataset 2 test, so these evaluations are not independent cohorts.', '',
           f"Best epoch: {checkpoint['epoch']}; validation loss: {checkpoint['val_loss']:.6f}.",'',
           '| Test set | Label | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics.itertuples(index=False):
        lines.append(f'| {r.test_dataset} | {r.label} | {r.accuracy:.6f} | {r.precision:.6f} | {r.recall:.6f} | {r.f1:.6f} | {r.TN} | {r.FP} | {r.FN} | {r.TP} |')
    lines+=['','Confusion matrices use rows = actual [negative, positive], columns = predicted [negative, positive]: [[TN, FP], [FN, TP]].', '',
            'Dataset 1 uses the team’s diagnostic-only Normal filter plus all CLBBB/ILBBB records. '+
            'Dataset 2 uses Kevin’s all-record code-presence labels. Each label is evaluated separately against all other test records.', '',
            'Patient separation, cross-dataset membership, labels, array shapes, and finite values passed the checks in leakage_checks.json. '+
            'raw_verification.json records SHA-256 verification of the native signals. '+
            'training_config.json records exact source commits and settings; history.json contains all completed epochs.', '',
            'Reproduce from the repository root with `.venv/bin/python -u experiment4/run.py` after placing the official PTB-XL 1.0.3 ZIP at `data/raw/ptb-xl-1.0.3.zip`. '+
            'Install `experiment4/requirements-lock.txt` in a Python 3.13 environment.']
    (RESULTS/'experiment4_report.md').write_text('\n'.join(lines)+'\n')
    print(metrics.to_string(index=False))

if __name__=='__main__':main()
