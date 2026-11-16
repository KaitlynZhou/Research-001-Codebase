"""Regression checks for labels, patient separation and six-lead architecture."""
import importlib.util, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from prepare import cohorts
from model import ECGResNet

def main():
    # Mixed diagnoses remain positive, independent labels, code presence includes likelihood 0,
    # non-diagnostic annotations do not exclude otherwise normal controls.
    with tempfile.TemporaryDirectory() as d:
        d=Path(d)
        pd.DataFrame({'diagnostic':[1,1,1,1,0]},index=['NORM','CLBBB','ILBBB','MI','SR']).to_csv(d/'scp_statements.csv')
        codes=[{'CLBBB':100,'MI':100},{'ILBBB':0,'MI':100},{'NORM':100,'SR':100},{'NORM':100,'MI':100},{'CLBBB':100,'ILBBB':100}]
        pd.DataFrame({'ecg_id':range(5),'patient_id':range(5),'scp_codes':[str(c) for c in codes],'strat_fold':[1,2,9,10,10]}).to_csv(d/'ptbxl_database.csv',index=False)
        m=cohorts(d)
        assert m.dataset1.tolist()==[True,True,True,False,True]
        assert m[['CLBBB','ILBBB']].values.tolist()==[[1,0],[0,1],[0,0],[0,0],[1,1]]
        pd.DataFrame({'ecg_id':[1,2],'patient_id':[1,1],'scp_codes':["{'NORM':100}"]*2,'strat_fold':[1,10]}).to_csv(d/'ptbxl_database.csv',index=False)
        try: cohorts(d)
        except AssertionError: pass
        else: raise AssertionError('Patient leakage not rejected')
    # The only architecture change is input-channel count; all other state shapes match.
    ref=Path(__file__).parent.parent/'learning_pipeline/07_resnet_model.py'
    if not ref.exists():
        ref=Path(__file__).parent.parent/'reference/07_resnet_model.py'
    assert ref.exists(), 'Prisha reference model is required for architecture parity validation'
    if ref.exists():
        spec=importlib.util.spec_from_file_location('reference',ref);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
        original=r.ECGResNet();six=ECGResNet()
        s=original.state_dict();selected=[0,4,6,7,10,11]
        s['stem.0.weight']=s['stem.0.weight'][:,selected,:];six.load_state_dict(s)
        original.eval();six.eval();x=torch.randn(3,6,1000);full=torch.zeros(3,12,1000);full[:,selected,:]=x
        with torch.no_grad(): torch.testing.assert_close(original(full),six(x))
    print('PASS: mixed-diagnosis labels, separate outcomes, normal controls, patient leakage rejection, architecture parity')

if __name__=='__main__':main()
