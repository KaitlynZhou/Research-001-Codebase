"""Produce shareable plots for the completed six-lead run."""
import json,argparse
from pathlib import Path
import os, tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ecg-baseline-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'ecg-baseline-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve,precision_recall_curve

def plot(out):
    h=pd.DataFrame(json.loads((out/'history.json').read_text()))
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(h.epoch,h.train_loss,label='Training');ax.plot(h.epoch,h.val_loss,label='Validation')
    ax.set(xlabel='Epoch',ylabel='Weighted binary cross-entropy',title='Six diagnostic leads — Dataset 1');ax.legend();fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for col,label in enumerate(['CLBBB','ILBBB']):
        for name,text in [('dataset1_test','Dataset 1: LBBB + normal'),('dataset2_test','Dataset 2: all diagnoses')]:
            d=pd.read_csv(out/f'{name}_predictions.csv');y=d[label];p=d['p_'+label]
            fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
            axs[0,col].plot(fpr,tpr,label=text);axs[1,col].plot(recall,precision,label=text)
        axs[0,col].plot([0,1],[0,1],color='grey',ls='--');axs[0,col].set(title=f'{label}: ROC',xlabel='False-positive rate',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1.02))
        axs[1,col].set(title=f'{label}: precision–recall',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02));axs[1,col].legend(fontsize=8)
    fig.suptitle('Held-out fold 10; Dataset 1 is a subset of Dataset 2',fontsize=11);fig.tight_layout();fig.savefig(out/'test_curves.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plot(a.out)
