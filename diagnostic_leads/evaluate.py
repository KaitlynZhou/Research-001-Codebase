"""Evaluate one frozen checkpoint on both held-out fold-10 populations."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_score,
                             recall_score,f1_score,confusion_matrix)
from model import ECGResNet

LABELS=['CLBBB','ILBBB']
def metrics(y,p):
    rows=[]
    for i,label in enumerate(LABELS):
        true=y[:,i];score=p[:,i];pred=score>=.5
        tn,fp,fn,tp=confusion_matrix(true,pred,labels=[0,1]).ravel()
        rows.append(dict(outcome=label,n=len(y),positives=int(true.sum()),prevalence=float(true.mean()),threshold=.5,
             roc_auc=float(roc_auc_score(true,score)) if len(np.unique(true))==2 else None,
             average_precision=float(average_precision_score(true,score)) if true.sum() else None,
             precision=float(precision_score(true,pred,zero_division=0)),
             sensitivity=float(recall_score(true,pred,zero_division=0)) if true.sum() else None,
             specificity=float(tn/(tn+fp)) if tn+fp else None,
             f1=float(f1_score(true,pred,zero_division=0)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
    return rows

def evaluate(data,out):
    assert (out/'complete.json').exists(), 'Training must complete before test evaluation'
    torch.set_num_threads(3)
    checkpoint=torch.load(out/'best_model.pt',map_location='cpu',weights_only=True)
    model=ECGResNet();model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    rows=[]
    for split in ['val','dataset2_test']:
        x=np.load(data/f'X_{split}.npy',mmap_mode='r');y=np.load(data/f'y_{split}.npy')
        scores=[]
        with torch.no_grad():
            for start in range(0,len(x),32):
                scores.append(torch.sigmoid(model(torch.tensor(x[start:start+32]))).numpy())
        scores=np.concatenate(scores);m=pd.read_csv(data/f'{split}.csv')
        predictions=m[['ecg_id','patient_id','strat_fold','dataset1','CLBBB','ILBBB']].copy()
        predictions[['p_CLBBB','p_ILBBB']]=scores
        predictions.to_csv(out/f'{split}_predictions.csv',index=False)
        for name,mask in ([(split,np.ones(len(y),dtype=bool))] if split=='val' else
                          [('dataset2_test',np.ones(len(y),dtype=bool)),('dataset1_test',m.dataset1.to_numpy()),('dataset2_test_additional_diagnoses',~m.dataset1.to_numpy())]):
            for row in metrics(y[mask],scores[mask]):rows.append({'population':name,**row})
            if name=='dataset1_test':predictions.loc[mask].to_csv(out/'dataset1_test_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(pd.DataFrame(rows).to_string(index=False))
    # Descriptive validation-only threshold sweep; no test tuning.
    v=pd.read_csv(out/'val_predictions.csv');threshold_rows=[]
    for label in LABELS:
        for t in [.5,.6,.7,.8,.9]:
            y=v[label];pred=v['p_'+label]>=t
            threshold_rows.append({'outcome':label,'threshold':t,'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0)})
    pd.DataFrame(threshold_rows).to_csv(out/'validation_thresholds.csv',index=False)
    (out/'evaluation_provenance.json').write_text(json.dumps({'checkpoint_epoch':checkpoint['epoch'], 'checkpoint_sha256':hashlib.sha256((out/'best_model.pt').read_bytes()).hexdigest(),'thresholds':[.5,.5],'test_fold':10,'dataset1_is_subset_of_dataset2':True},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.data,a.out)
