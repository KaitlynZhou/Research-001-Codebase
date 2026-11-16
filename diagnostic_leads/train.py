"""Train on Dataset 1 only. No test arrays are opened by this script."""
import argparse, json, random, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from model import ECGResNet


def train(data,out,epochs=30,threads=3):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'best_model.pt').exists(): raise FileExistsError('Use a fresh output directory to preserve the existing run.')
    torch.set_num_threads(threads)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model=ECGResNet().to(device)
    arrays={n:np.load(data/f'{n}.npy') for n in ['X_train','y_train','X_val','y_val']}
    loaders={s:DataLoader(TensorDataset(torch.from_numpy(arrays['X_'+s]),torch.from_numpy(arrays['y_'+s])),batch_size=32,shuffle=s=='train') for s in ['train','val']}
    y=arrays['y_train'];pos=y.sum(0);neg=len(y)-pos
    assert ((pos>0)&(neg>0)).all()
    weights=torch.tensor(neg/pos,dtype=torch.float32,device=device)
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    config={'seed':42,'batch_size':32,'max_epochs':epochs,'learning_rate':.001,'patience':7,
            'optimizer':'Adam','weight_decay':0,'pos_weight':weights.cpu().tolist(),
            'leads':['I','aVL','V1','V2','V5','V6'],'outcomes':['CLBBB','ILBBB'],
            'selection':'minimum validation weighted BCE','thresholds':[.5,.5],
            'sampling_hz':100,'samples':1000,'normalization':'none; physical mV',
            'device':str(device),'threads':threads,'torch':str(torch.__version__),'numpy':np.__version__,
            'parameters':sum(p.numel() for p in model.parameters()),
            'reference_commit':'4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e'}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    best=float('inf');stale=0;history=[];start=time.time()
    for epoch in range(1,epochs+1):
        row={'epoch':epoch}
        for split,loader in loaders.items():
            training=split=='train';model.train(training);total=0
            for x,y in loader:
                x=x.to(device);y=y.to(device)
                if training:optimizer.zero_grad()
                with torch.set_grad_enabled(training):
                    loss=loss_fn(model(x),y)
                    if training:loss.backward();optimizer.step()
                total+=loss.item()*len(x)
            row[split+'_loss']=total/len(loader.dataset)
        assert np.isfinite([row['train_loss'],row['val_loss']]).all()
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
        print(row,'elapsed_s',round(time.time()-start),flush=True)
        if row['val_loss']<best:
            best=row['val_loss'];stale=0
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'val_loss':best,'config':config},out/'best_model.pt')
        else:stale+=1
        if stale>=7:break
    (out/'complete.json').write_text(json.dumps({'epochs_run':epoch,'seconds':time.time()-start,'best_validation_loss':best,'checkpoint_sha256':hashlib.sha256((out/'best_model.pt').read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();train(a.data,a.out,a.epochs,a.threads)
