"""Small shared helpers used only by Experiment 2 training and evaluation scripts."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,accuracy_score

BASE=Path('data/processed/dataset2')
RESULTS=Path('results/experiment2')
LABELS=['CLBBB','ILBBB']

def get_model():
    path=Path(__file__).resolve().parent/'04_resnet_model.py'
    spec=importlib.util.spec_from_file_location('experiment2_model',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ECGResNet()

class ArrayDataset(Dataset):
    def __init__(self,xpath,ypath):
        self.X=np.load(xpath,mmap_mode='r')
        self.y=np.load(ypath,mmap_mode='r')
        assert len(self.X)==len(self.y)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        # Copy from read-only mmap to avoid PyTorch read-only ndarray warnings.
        return torch.from_numpy(np.array(self.X[i],copy=True)),torch.from_numpy(np.array(self.y[i],copy=True))

def loader(part,batch_size=32,shuffle=False):
    suffix=part
    return DataLoader(ArrayDataset(BASE/f'X_{suffix}.npy',BASE/f'y_{suffix}.npy'),batch_size=batch_size,shuffle=shuffle,num_workers=0)

def predict(model,part,device):
    model.eval();ys=[];scores=[]
    with torch.no_grad():
        for x,y in loader(part):
            pred=torch.sigmoid(model(x.to(device))).cpu().numpy()
            ys.append(y.numpy());scores.append(pred)
    return np.concatenate(ys),np.concatenate(scores)

def scores_table(y,scores,threshold=0.5):
    rows=[]
    for i,name in enumerate(LABELS):
        truth=y[:,i].astype(int);score=scores[:,i];pred=(score>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel()
        rows.append(dict(label=name,threshold=threshold,accuracy=accuracy_score(truth,pred),precision=precision_score(truth,pred,zero_division=0),recall=recall_score(truth,pred,zero_division=0),f1=f1_score(truth,pred,zero_division=0),roc_auc=roc_auc_score(truth,score) if len(np.unique(truth))>1 else float('nan'),average_precision=average_precision_score(truth,score) if truth.sum()>0 else float('nan'),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)))
    return rows

def load_best(device):
    checkpoint=torch.load(RESULTS/'best_model.pt',map_location=device,weights_only=True)
    model=get_model().to(device)
    model.load_state_dict(checkpoint['model_state_dict']);model.eval()
    return model,checkpoint
