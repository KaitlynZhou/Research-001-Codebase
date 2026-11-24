"""100 Hz adaptation of the local notebook's multilead median-beat extractor."""
import numpy as np
import pandas as pd
from scipy import signal

BANDPASS = signal.butter(3,(8,30),fs=100,btype="bandpass",output="sos")
BASELINE = signal.butter(2,.5,fs=100,btype="highpass",output="sos")

def extract_median_beat(wave):
    wave = np.asarray(wave,dtype=np.float64)
    if wave.shape != (12,1000) or not np.isfinite(wave).all():
        raise ValueError("invalid_waveform")
    power = np.sqrt(np.mean(signal.sosfiltfilt(BANDPASS,wave,axis=-1)**2,axis=0))
    envelope = np.convolve(power,np.ones(7)/7,mode="same")
    ref = np.percentile(envelope,97)
    if ref < .015:
        raise ValueError("weak_qrs")
    peaks,_ = signal.find_peaks(envelope,distance=25,height=.22*ref,prominence=.16*ref)
    peaks = peaks[(peaks>=34)&(peaks<946)]
    anchors = np.array([p-4+np.argmax(power[p-4:p+5]) for p in peaks],dtype=int)
    if len(anchors)<3:
        raise ValueError("too_few_complete_beats")
    cleaned = signal.sosfiltfilt(BASELINE,wave,axis=-1)
    windows = np.stack([cleaned[:,p-30:p+50] for p in anchors])
    keep = np.max(np.ptp(windows,axis=-1),axis=1)<12
    windows,anchors = windows[keep],anchors[keep]
    if len(windows)<3:
        raise ValueError("too_few_valid_beats")
    core = windows[:,:,19:41]
    residual = np.sqrt(np.mean((core-np.median(core,axis=0))**2,axis=(1,2)))
    med = np.median(residual)
    keep = residual<=max(.08,med+3*np.median(np.abs(residual-med)))
    windows,anchors = windows[keep],anchors[keep]
    if len(windows)<3:
        raise ValueError("too_few_consistent_beats")
    return np.median(windows,axis=0).astype(np.float32),anchors,windows

def prepare(source,dest):
    dest.mkdir(parents=True,exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"Use an empty median data directory: {dest}")
    summaries=[]
    for split in ("train","val","test"):
        X=np.load(source/f"X_{split}.npy",mmap_mode="r")
        y=np.load(source/f"y_{split}.npy",mmap_mode="r")
        if X.shape!=(len(y),12,1000) or y.shape!=(len(X),2) or not np.isin(y,[0,1]).all():
            raise ValueError("Invalid source arrays")
        beats=[]; indices=[]; rows=[]
        for i,wave in enumerate(X):
            row=dict(original_index=i,CLBBB=int(y[i,0]),ILBBB=int(y[i,1]),median_index=-1,accepted_beats=0)
            try:
                median,anchors,_=extract_median_beat(wave)
            except ValueError as exc:
                row["status"]=str(exc)
            else:
                row.update(status="accepted",median_index=len(beats),accepted_beats=len(anchors))
                beats.append(median); indices.append(i)
            rows.append(row)
        idx=np.asarray(indices,dtype=np.int64)
        np.save(dest/f"X_{split}.npy",np.stack(beats))
        np.save(dest/f"y_{split}.npy",y[idx])
        np.save(dest/f"indices_{split}.npy",idx)
        pd.DataFrame(rows).to_csv(dest/f"{split}_metadata.csv",index=False)
        summaries.append(dict(split=split,original=len(y),accepted=len(idx),rejected=len(y)-len(idx),lost_CLBBB=int(y[:,0].sum()-y[idx,0].sum()),lost_ILBBB=int(y[:,1].sum()-y[idx,1].sum())))
    pd.DataFrame(summaries).to_csv(dest/"preprocessing_summary.csv",index=False)
    print(pd.DataFrame(summaries))
