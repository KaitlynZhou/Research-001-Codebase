# Single-lead V1–V4 models: full 10-second ECG vs median beat

Eight models: one per lead (**V1, V2, V3, V4**) × two input types (**full 10-second ECG**, **median beat**). Each has two independent sigmoid outputs, **CLBBB** and **ILBBB**, trained on Dataset 1 and evaluated on the fold-10 tests of Dataset 1 (LBBB vs normal) and Dataset 2 (LBBB vs all others). Results and limitations: [REPORT.md](REPORT.md).

## What is held fixed

- **Cohorts, labels, folds:** imported from [`diagnostic_leads/prepare.py`](../diagnostic_leads/prepare.py) (`cohorts`), so Dataset 1 / Dataset 2 match the six-lead branch exactly. Train on Dataset 1 folds 1–8, select on fold 9, test on fold 10.
- **Model:** the six-lead branch's three-block 1D ResNet; only the stem's input-channel count changes (6 → 1). `test_single_leads.py` checks parity.
- **Training:** seed 42, Adam lr 1e-3, batch 32, ≤30 epochs, patience 7, training-derived `pos_weight`, checkpoint at minimum validation weighted BCE, physical mV, no filtering/normalization/augmentation.

## What differs from the six-lead branch

- **Sampling rate:** 500 Hz for both input types. Median beats only exist at 500 Hz, so this keeps the full-vs-median comparison free of a resolution difference. (The six-lead branch used 100 Hz.)
- **Median beats:** PTB-XL+ 1.0.1 `median_beats/12sl` (GE 12SL), 600 samples (1.2 s). The released headers can't be read by `wfdb` (the record name contains a directory), so `common.read_median` parses them, verifies every channel's checksum, and converts the stored integer microvolts to mV (the headers say mV; the values are µV). `prepare.py` also checks median-beat amplitudes against the raw ECG.
- **Common record set:** recordings with no published median beat are removed from every split for **both** input types; `audit.json` counts what was removed, by label.

## Run

```bash
python -m pip install torch numpy pandas scikit-learn wfdb
```

Put `ptbxl_database.csv` and `scp_statements.csv` (PTB-XL 1.0.3) in a raw directory, then from this folder:

```bash
python test_single_leads.py
```

```bash
python fetch_medians.py --raw /path/to/ptb-xl --medians /path/to/ptb-xl-plus
```

```bash
python prepare.py --raw /path/to/ptb-xl --medians /path/to/ptb-xl-plus --out /path/to/processed --zip /path/to/ptb-xl-1.0.3.zip
```

```bash
python run_all.py --data /path/to/processed --results results
```

`fetch_medians.py` downloads only the median beats the cohorts need and checks each file against PTB-XL+'s `SHA256SUMS.txt`. `--zip` extracts only the needed 500 Hz records; omit it if `records500/` is already extracted. `run_all.py` is resumable (finished models are skipped), then runs `evaluate.py` and `report.py`. Training picks CUDA, MPS or CPU; evaluation uses CPU.

## Outputs

`results/metrics.csv` (AUROC, AUPRC with bootstrap 95% CIs, and thresholded counts at 0.50), `results/median_minus_full.csv` (paired bootstrap differences), `results/evaluation_provenance.json`, and per model `results/<input>_<lead>/{config,history,complete}.json`. `data_audit.json` is the preparation audit. Checkpoints, predictions and arrays are gitignored and stay local.

Data: PTB-XL 1.0.3 (https://physionet.org/content/ptb-xl/1.0.3/) and PTB-XL+ 1.0.1 (https://physionet.org/content/ptb-xl-plus/1.0.1/). Cite both and follow their licenses.
