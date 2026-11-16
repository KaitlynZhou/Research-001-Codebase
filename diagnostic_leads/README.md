# Local six-diagnostic-lead Dataset 1 baseline

Train two independent outputs, in order **CLBBB, ILBBB**, using leads **I, aVL, V1, V2, V5, V6**. Prepared and trained locally in the attached cloud workspace. Published on branch `feature/dataset1-diagnostic-leads`, based on Prisha’s reference branch.

## Reference and controlled comparison

Prisha's `feature/dataset1-12lead-clean` branch at `4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e`, particularly `learning_pipeline/02_build_dataset1.py`, `05_prepare_waveforms.py`, `07_resnet_model.py`, and `08_train_resnet.py`.

Same 1D residual network: stem 32 channels, three blocks (32,64,128), kernel 7, strided downsampling in the last two blocks, average pooling, two logits. Only the stem changes from 12 input channels to 6 (and its corresponding parameter count). Same Adam LR .001, batch 32, max 30 epochs, patience 7, seed 42, training-derived negative/positive class weights, minimum validation weighted BCE checkpoint, and fixed .50 decision threshold for each outcome. No additional waveform scaling, normalization, augmentation, or filtering. WFDB physical mV, 100 Hz, 10 seconds. Both outputs are trained jointly; CLBBB is the primary outcome and ILBBB is secondary.

Additional implementation safeguards do not change the experiment: patient/fold validation, channel selection by name (case insensitive), batched inference, a training-complete guard before test scoring, saved configuration, record-level predictions, and checkpoint hashes. CPU deterministic algorithms are enabled. Exact floating-point reproducibility across PyTorch versions and hardware is not guaranteed.

Optimizer hyperparameters are held fixed to isolate the input-lead change. The model has fewer input-layer parameters than Prisha's baseline.

## Data definitions

PTB-XL **1.0.3**, https://physionet.org/content/ptb-xl/1.0.3/ . Cite the dataset and follow its published license when reusing data. Metadata and raw waveforms are kept in the workspace, outside this code folder.

* A CLBBB or ILBBB code's **presence** sets the corresponding label to 1, regardless of other diagnoses or code likelihood (including 0), matching Prisha's implementation. The two labels are never merged, and co-occurrence is supported.
* Dataset 1 includes every CLBBB or ILBBB recording plus NORM recordings without another diagnostic code. Non-diagnostic rhythm/form annotations do not disqualify normal controls.
* Dataset 2 includes all PTB-XL recordings. This is an expanded evaluation population, not a separate external dataset.
* Train on Dataset 1 folds 1–8 only; use Dataset 1 fold 9 for model selection. The same frozen checkpoint scores Dataset 1 fold 10 and Dataset 2 fold 10.
* Dataset 1 test is a subset of Dataset 2 test. Their metrics are correlated and do not demonstrate external generalization. Neither population shares patients with training/validation. Predictions for shared records are exactly shared because Dataset 1 test is indexed from Dataset 2 test scores.
* There are only 62 ILBBB training positives, 7 validation positives, and 8 test positives. Report individual counts and average precision alongside ROC-AUC. One missed ILBBB changes test sensitivity by 12.5 percentage points. Conclusions about ILBBB are exploratory.

## Run on your computer

Use Python 3.12 or a compatible recent Python. Install PyTorch for your platform, then:

```bash
python -m pip install torch numpy pandas scikit-learn wfdb matplotlib
```

Place `ptbxl_database.csv`, `scp_statements.csv` and the WFDB `records100/` directory under a raw data directory. Or download the PTB-XL 1.0.3 ZIP from PhysioNet and supply `--zip`; preparation extracts only required 100 Hz recordings. Metadata CSV files must already be under `--raw`. The full official ZIP includes 500 Hz waveforms too (~1.84 GB compressed), but only 100 Hz data is used.

Run from the directory containing these scripts:

```bash
python test_methodology.py
python prepare.py --raw /path/to/ptb-xl --out /path/to/processed --zip /path/to/ptbxl.zip
python train.py --data /path/to/processed --out /path/to/new_results
python evaluate.py --data /path/to/processed --out /path/to/new_results
```

Omit `--zip` if WFDB files are already extracted. Use a fresh results directory for every training run; the script refuses to overwrite an existing checkpoint. `--epochs` supports a short smoke run but defaults to the actual 30-epoch baseline maximum. Training supports CUDA, MPS, or CPU automatically; evaluation uses CPU. Do not compare smoke-run metrics to the full baseline.

## Artifacts

`results/config.json`, `history.json`, `complete.json`, `best_model.pt`, `metrics.csv`, `metrics.json`, `val_predictions.csv`, `dataset1_test_predictions.csv`, `dataset2_test_predictions.csv`, `validation_thresholds.csv`, `evaluation_provenance.json`. Preparation writes `audit.json`, cohort metadata, and six-lead arrays in the separate processed directory. `provenance.json` records branch/commit references and methodology differences. `requirements-lock.txt` records the execution package versions.

Weighted-BCE sigmoid outputs are model scores and are not calibrated clinical probabilities.

Metrics use fixed thresholds .50 and include ROC-AUC, average precision, precision, sensitivity, specificity, F1 and full confusion counts. Validation-only threshold sweeps are descriptive; no test threshold optimization. The Dataset 2 additional-diagnoses subset contains no target positives by construction, so AUROC/AP/sensitivity are not estimable there; false-positive counts and specificity describe that subset.

The repository includes source code, the inspection notebook, the experiment report, aggregate metrics, and audit summaries. Raw/processed ECG data, trained weights, record-level predictions, and generated plots remain local. After training and evaluation, run `python plot_results.py --out /path/to/new_results` and `python make_report.py --out /path/to/new_results` to regenerate plots and the report. Open `Explore_baseline.ipynb` from this directory to inspect available results.
