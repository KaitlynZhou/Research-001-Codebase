"""Validation metrics and ILBBB threshold sensitivity, as in Experiment 1."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from common import RESULTS,load_best,predict,scores_table

def main():
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model,checkpoint=load_best(device)
    print('Best epoch:',checkpoint['epoch'],'Validation loss:',checkpoint['val_loss'])
    truth,scores=predict(model,'val',device)
    results=pd.DataFrame(scores_table(truth,scores,0.5))
    RESULTS.mkdir(parents=True,exist_ok=True)
    results.to_csv(RESULTS/'validation_metrics.csv',index=False)
    results[['label','TN','FP','FN','TP']].to_csv(RESULTS/'validation_confusion_matrices.csv',index=False)
    print(results.to_string(index=False))
    rows=[]; ilbbb_truth=truth[:,1]
    for t in [0.5,0.6,0.7,0.8,0.9]:
        pred=(scores[:,1]>=t).astype(int)
        tn,fp,fn,tp=confusion_matrix(ilbbb_truth,pred,labels=[0,1]).ravel()
        rows.append({'threshold':t,'precision':precision_score(ilbbb_truth,pred,zero_division=0),'recall':recall_score(ilbbb_truth,pred,zero_division=0),'f1':f1_score(ilbbb_truth,pred,zero_division=0),'true_positives':int(tp),'false_positives':int(fp)})
    table=pd.DataFrame(rows)
    table.to_csv(RESULTS/'ilbbb_threshold_comparison.csv',index=False)
    print('ILBBB validation sensitivity (report primary results at 0.50):', table.to_string(index=False))

if __name__=='__main__':main()
