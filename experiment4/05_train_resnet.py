"""Same optimizer, seed, loss, batch size and early stopping as Experiment 1."""
from pathlib import Path
import random
import json
import time
import numpy as np
import torch
from torch import nn
from common import BASE,RESULTS,get_model,loader

SEED=42
BATCH_SIZE=32
MAX_EPOCHS=30
LEARNING_RATE=0.001
PATIENCE=7

def run_epoch(model,data,criterion,device,optimizer=None):
    training=optimizer is not None
    model.train(training)
    total_loss=0.0; total_samples=0
    for batch,(X,y) in enumerate(data,1):
        X=X.to(device);y=y.to(device)
        if training:optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits=model(X);loss=criterion(logits,y)
            if training:loss.backward();optimizer.step()
        total_loss+=loss.item()*len(X);total_samples+=len(X)
        if training and batch%100==0:print(f'  Training batch {batch}/{len(data)}',flush=True)
    return total_loss/total_samples

def main():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:',device)
    model=get_model().to(device)
    print('Parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_loader=loader('train',BATCH_SIZE,shuffle=True)
    val_loader=loader('val',BATCH_SIZE)
    y_train=np.load(BASE/'y_train.npy',mmap_mode='r')
    positives=y_train.sum(axis=0)
    if (positives==0).any():raise ValueError('Missing positive labels in training')
    pos_weight=torch.as_tensor((len(y_train)-positives)/positives,dtype=torch.float32,device=device)
    print('Positive weights:',pos_weight.cpu().numpy())
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    RESULTS.mkdir(parents=True,exist_ok=True)
    history=[]
    started=time.time()
    best_val_loss=float('inf');no_improvement=0
    for epoch in range(1,MAX_EPOCHS+1):
        tr=run_epoch(model,train_loader,criterion,device,optimizer)
        va=run_epoch(model,val_loader,criterion,device)
        history.append({'epoch':epoch,'train_loss':tr,'val_loss':va,'elapsed_seconds':time.time()-started})
        (RESULTS/'history.json').write_text(json.dumps(history,indent=2))
        print(f'Epoch {epoch:02d} | Train loss: {tr:.4f} | Val loss: {va:.4f}')
        if va<best_val_loss:
            best_val_loss=va;no_improvement=0
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'val_loss':va,'pos_weight':pos_weight.cpu(),'seed':SEED},RESULTS/'best_model.pt')
            print('  Saved new best model')
        else:
            no_improvement+=1
        if no_improvement>=PATIENCE:
            print('Early stopping triggered');break
    print('Training complete. Best validation loss:',best_val_loss)

if __name__=='__main__':main()
