"""Shared single-lead training and minimal matched-cohort evaluation."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .models import ARCHITECTURES
from .beats import prepare

LABELS = ("CLBBB", "ILBBB")
LEADS = {"II": 1, "III": 2, "aVR": 3, "aVF": 5}
SPLITS = ("train", "val", "test")
LENGTHS = {"full_ecg": 1000, "median_beat": 80}
SEED = 42
BATCH_SIZE = 32
LR = .001
EPOCHS = 30
PATIENCE = 7

class SingleLeadDataset(Dataset):
    def __init__(self, root, split, lead, representation):
        self.X = np.load(root / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(root / f"y_{split}.npy", mmap_mode="r")
        self.idx = LEADS[lead]
        if self.X.shape != (len(self.y), 12, LENGTHS[representation]):
            raise ValueError("Waveform shape does not match representation")
        if self.y.shape != (len(self.X), 2) or not np.isin(self.y, [0, 1]).all():
            raise ValueError("Invalid label array")
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(np.array(self.X[i,self.idx:self.idx+1,:],dtype=np.float32,copy=True)),
                torch.from_numpy(np.array(self.y[i],dtype=np.float32,copy=True)))

def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

def loader(root, split, lead, representation):
    return DataLoader(SingleLeadDataset(root, split, lead, representation),
                      batch_size=BATCH_SIZE, shuffle=split=="train", num_workers=0,
                      generator=torch.Generator().manual_seed(SEED) if split=="train" else None)

def metrics(y, scores):
    out = {}
    for i, label in enumerate(LABELS):
        truth = y[:,i]
        pred = scores[:,i]
        out[label+"_AUROC"] = float(roc_auc_score(truth,pred)) if len(np.unique(truth))==2 else float("nan")
        out[label+"_AUPRC"] = float(average_precision_score(truth,pred)) if truth.sum()>0 else float("nan")
    return out

def sync(device):
    if device.type=="mps":
        torch.mps.synchronize()
    elif device.type=="cuda":
        torch.cuda.synchronize()

def run_epoch(model, batches, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = torch.zeros((),device=device)
    count = 0
    targets, scores = [], []
    with torch.set_grad_enabled(training):
        for x,y in batches:
            x,y = x.to(device),y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits,y)
            if training:
                loss.backward()
                optimizer.step()
            else:
                targets.append(y.detach())
                scores.append(torch.sigmoid(logits).detach())
            total += loss.detach()*len(y)
            count += len(y)
    value = (total/count).item()
    if not np.isfinite(value):
        raise RuntimeError("Non-finite loss")
    if training:
        return value
    y = torch.cat(targets).cpu().numpy()
    s = torch.cat(scores).cpu().numpy()
    if not np.isfinite(s).all():
        raise RuntimeError("Non-finite scores")
    return value,metrics(y,s),y,s

def save_evaluation(directory,split,result,meta):
    loss,m,y,s = result
    row = {**meta,split+"_loss":loss,**m}
    pd.DataFrame([row]).to_csv(directory/f"{split}_metrics.csv",index=False)
    np.save(directory/f"{split}_targets.npy",y)
    np.save(directory/f"{split}_scores.npy",s)
    return row

def train(root,output,architecture,lead,representation,device):
    seed_all()
    directory = output/representation/architecture/lead
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Run already has files: {directory}; use a fresh output or move the run")
    train_batches = loader(root,"train",lead,representation)
    val_batches = loader(root,"val",lead,representation)
    positives = train_batches.dataset.y.sum(axis=0)
    if np.any(positives==0):
        raise ValueError("A training label has no positives")
    weights = torch.tensor((len(train_batches.dataset)-positives)/positives,dtype=torch.float32)
    model = ARCHITECTURES[architecture]().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    meta = dict(architecture=architecture,lead=lead,representation=representation,input_length=LENGTHS[representation],seed=SEED)
    config = dict(**meta,labels=list(LABELS),device=str(device),batch_size=BATCH_SIZE,learning_rate=LR,max_epochs=EPOCHS,patience=PATIENCE,optimizer="Adam",weight_decay=0,loss="weighted BCE",positive_weights=weights.tolist(),checkpoint_selection="validation_loss",parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (directory/"config.json").write_text(json.dumps(config,indent=2))
    best = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        sync(device)
        t = time.perf_counter()
        loss = run_epoch(model,train_batches,criterion,device,optimizer)
        val,m,_,_ = run_epoch(model,val_batches,criterion,device)
        sync(device)
        history.append(dict(epoch=epoch,train_loss=loss,val_loss=val,**m,epoch_time_seconds=time.perf_counter()-t))
        if val < best:
            best = val
            stale = 0
            torch.save(dict(**meta,epoch=epoch,val_loss=val,val_metrics=m,pos_weight=weights,model_state_dict=model.state_dict()),directory/"best_model.pt")
        else:
            stale += 1
        pd.DataFrame(history).to_csv(directory/"training_history.csv",index=False)
        print(architecture,lead,representation,history[-1],flush=True)
        if stale>=PATIENCE:
            break
    ckpt = torch.load(directory/"best_model.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    result = run_epoch(model,val_batches,criterion,device)
    row = save_evaluation(directory,"validation",result,dict(**meta,best_epoch=ckpt["epoch"],epochs_trained=len(history),training_time_seconds=time.perf_counter()-start))
    print(row)

def evaluate(root,output,architecture,lead,representation,device):
    directory = output/representation/architecture/lead
    ckpt = torch.load(directory/"best_model.pt",map_location="cpu",weights_only=True)
    for k,v in dict(architecture=architecture,lead=lead,representation=representation,input_length=LENGTHS[representation]).items():
        if ckpt.get(k)!=v:
            raise ValueError(f"Checkpoint provenance mismatch: {directory}: {k}")
    model = ARCHITECTURES[architecture]().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=ckpt["pos_weight"].to(device))
    batches = loader(root,"test",lead,representation)
    result = run_epoch(model,batches,criterion,device)
    np.testing.assert_array_equal(result[2],batches.dataset.y)
    return save_evaluation(directory,"test",result,dict(architecture=architecture,lead=lead,representation=representation,best_epoch=ckpt["epoch"]))

def compare(full,median,output):
    rows = []
    for split,filename in (("val","validation"),("test","test")):
        idx = np.load(median/f"indices_{split}.npy")
        full_y = np.load(full/f"y_{split}.npy")
        median_y = np.load(median/f"y_{split}.npy")
        if idx.ndim!=1 or len(np.unique(idx))!=len(idx) or np.any(idx<0) or np.any(idx>=len(full_y)):
            raise ValueError("Invalid accepted indices")
        np.testing.assert_array_equal(full_y[idx],median_y)
        for architecture in ARCHITECTURES:
            for lead in LEADS:
                f = output/"full_ecg"/architecture/lead
                m = output/"median_beat"/architecture/lead
                np.testing.assert_array_equal(np.load(f/f"{filename}_targets.npy"),full_y)
                np.testing.assert_array_equal(np.load(m/f"{filename}_targets.npy"),median_y)
                fm = metrics(median_y,np.load(f/f"{filename}_scores.npy")[idx])
                mm = metrics(median_y,np.load(m/f"{filename}_scores.npy"))
                for metric in fm:
                    rows.append(dict(split=split,architecture=architecture,lead=lead,metric=metric,n=len(idx),positive=int(median_y[:,LABELS.index(metric.split("_")[0])].sum()),full_ecg=fm[metric],median_beat=mm[metric],difference=mm[metric]-fm[metric]))
    table=pd.DataFrame(rows)
    table.to_csv(output/"matched_comparison.csv",index=False)
    print(table.round(4).to_string(index=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","train","evaluate","compare"])
    p.add_argument("--representation",choices=list(LENGTHS),default="full_ecg")
    p.add_argument("--architecture",choices=list(ARCHITECTURES))
    p.add_argument("--lead",choices=list(LEADS))
    p.add_argument("--full-data",type=Path,default=Path("data/processed/dataset2"))
    p.add_argument("--median-data",type=Path,default=Path("data/processed/median_beat"))
    p.add_argument("--output",type=Path,default=Path("results/single_lead"))
    p.add_argument("--device",choices=["cpu","cuda","mps"])
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.full_data,args.median_data)
        return
    if args.action=="compare":
        compare(args.full_data,args.median_data,args.output)
        return
    device=torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    root=args.full_data if args.representation=="full_ecg" else args.median_data
    rows=[]
    for a in ([args.architecture] if args.architecture else ARCHITECTURES):
        for lead in ([args.lead] if args.lead else LEADS):
            if args.action=="train":
                train(root,args.output,a,lead,args.representation,device)
            else:
                rows.append(evaluate(root,args.output,a,lead,args.representation,device))
    if rows:
        table=pd.DataFrame(rows)
        if not args.architecture and not args.lead:
            table.to_csv(args.output/args.representation/"all_test_results.csv",index=False)
        print(table.round(4).to_string(index=False))

if __name__=="__main__":
    main()
