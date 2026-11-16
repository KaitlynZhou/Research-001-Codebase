"""Evaluate ONE Experiment 2 checkpoint on BOTH held-out Dataset 1 and Dataset 2.

Never select the model or classification threshold using test results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import BASE,RESULTS,load_best,predict,scores_table

def main():
    # Test membership and labels must match for shared ECGs.
    d1=pd.read_csv('data/processed/dataset1/test.csv')
    d2=pd.read_csv(BASE/'test.csv').set_index('ecg_id')
    assert d1.ecg_id.isin(d2.index).all()
    for col in ['CLBBB','ILBBB','patient_id','strat_fold','filename_lr']:
        assert np.array_equal(d1[col],d2.loc[d1.ecg_id,col]),f'Mismatch: {col}'
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model,checkpoint=load_best(device)
    print('Best epoch:',checkpoint['epoch'])
    RESULTs=[]
    for dataset_name,part in [('dataset1','test_dataset1'),('dataset2','test')]:
        true,scores=predict(model,part,device)
        np.save(RESULTS/f'{dataset_name}_test_scores.npy',scores)
        metrics=pd.DataFrame(scores_table(true,scores,threshold=0.5))
        metrics.insert(0,'test_dataset',dataset_name)
        metrics.to_csv(RESULTS/f'{dataset_name}_test_metrics.csv',index=False)
        RESULTs.append(metrics)
        print(dataset_name, 'test results:', metrics.to_string(index=False))
    pd.concat(RESULTs,ignore_index=True).to_csv(RESULTS/'both_test_metrics.csv',index=False)
    print('Saved all results under:',RESULTS)

if __name__=='__main__':main()
