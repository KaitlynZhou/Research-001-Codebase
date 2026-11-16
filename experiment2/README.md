# Experiment 2 — 12-lead ResNet (Dataset 2 vs all others)

Recreated from the team's **Experiment 1** pipeline on branch `feature/ptbxl-data-processing`, `learning_pipeline/`.

## Controlled comparison

Keep identical to Experiment 1: 100 Hz 12-lead waveforms `(12,1000)`, three-block ECGResNet, PTB-XL folds 1–8 train / 9 validation / 10 test, seed 42, batch size 32, Adam LR 0.001 (no weight decay), weighted `BCEWithLogitsLoss`, max 30 epochs, early stopping patience 7, best validation-loss checkpoint and default classification threshold 0.50. Labels are separate binary `CLBBB` and `ILBBB` outputs. **Do not reuse Experiment 1 weights**. Class weights differ because training population differs.

Only intentional training-data difference: Experiment 2 trains on **all PTB-XL recordings**, not Dataset 1's LBBB and clean-normal subset. Both test populations are evaluated with the *same* Experiment 2 checkpoint.

## Prerequisites

From repository root, install `python -m pip install -r experiment2/requirements.txt` and download PTB-XL 100 Hz WFDB files and metadata to:

```
data/raw/ptb-xl/ptbxl_database.csv
data/raw/ptb-xl/scp_statements.csv
data/raw/ptb-xl/records100/.../*.hea and *.dat
```

Dataset 1 split metadata (`data/processed/dataset1/test.csv`) must exist. Produce it with your teammate's `learning_pipeline/02_build_dataset1.py` if not available. The numeric `.npy` waveform arrays, data and model checkpoints remain local and should not be committed.

## Execute in order (from repository root)

```
python experiment2/01_build_dataset2.py
python experiment2/02_prepare_waveforms.py
python experiment2/03_validate_arrays.py
python experiment2/04_resnet_model.py
python experiment2/05_train_resnet.py
python experiment2/06_evaluate_validation.py
python experiment2/07_evaluate_test.py
```

Step 02 writes disk-backed NumPy arrays to avoid temporarily doubling memory consumption. It may require approximately 1.1 GB free disk space for Dataset 2 inputs (plus Dataset 1 test arrays), and all WFDB files must be fully downloaded. It validates sampling rate and channel order; accepts alternate *capitalization* of limb-lead names, never an alternate order.

Outputs: `data/processed/dataset2/{metadata,train,val,test}.csv`, `X_*.npy`, `y_*.npy`; `results/experiment2/best_model.pt`, validation metrics, ILBBB validation thresholds, and test scores/metrics for both datasets.

**Key methodological notes:** Test fold 10 is held out until training/model selection is complete; validation threshold sweep is descriptive (default reported threshold 0.50). Dataset 1 test is a subset of Dataset 2 test, not an independent cohort. The recorded seed makes comparisons more controlled but does not guarantee bitwise reproducibility across devices.

## Differences from Experiment 1 implementation

- New Dataset 2 cohort builder; two test populations instead of only Dataset 1.
- Disk-backed `.npy` writing/reading rather than building the entire 21,799-record array in RAM; no change in waveform values or order.
- Additional assertions for patient/fold/label consistency across datasets.
- The model implementation is functionally identical to the Experiment 1 model, including convolution bias and stride behavior; unused imports removed.
- The training method, model-selection criterion and hyperparameters match Experiment 1.

This code has been syntax-checked, but it has **not** been trained or evaluated on the actual PTB-XL data in this environment.
