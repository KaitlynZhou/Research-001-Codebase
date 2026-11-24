# Predictive Experiment 2 — Individual non-diagnostic leads

## Objective
Compare II, III, aVR and aVF independently, using 100 Hz full 10-second ECGs and aligned 800 ms median beats. Each model has independent CLBBB and ILBBB logits. Dataset 2 includes all other ECGs as negatives for each label, not only normal ECGs.

## Reproduction
From repository root, install `pip install -r single_lead/requirements.txt`. Supply the existing Dataset 2 arrays (X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy) under data/processed/dataset2. The original data preparation is on [Experiment 2 branch](https://github.com/KaitlynZhou/Research-001-Codebase/tree/kevin/experiment2-dataset2-all12). This branch does not redistribute ECGs or checkpoints.

```sh
python -m single_lead.pipeline prepare
python -m single_lead.pipeline train --representation full_ecg
python -m single_lead.pipeline train --representation median_beat
python -m single_lead.pipeline evaluate --representation full_ecg
python -m single_lead.pipeline evaluate --representation median_beat
python -m single_lead.pipeline compare
```

Use --device cpu, mps or cuda to override automatic selection; --output selects a fresh output root (default results/single_lead). Training runs are independent and existing run directories with files are refused to avoid accidental overwriting. Use --lead and --architecture to run a single configuration. The scripts are a reconstruction of the notebook procedure, not a byte-for-byte copy of the local notebook; they have not been executed on PTB-XL here.

## Dataset and methods
Official patient folds 1–8 train, 9 validation, 10 test. Canonical channel order I, II, III, aVR, aVL, aVF, V1–V6. Full ECG shape (N,12,1000); labels (N,2), order CLBBB, ILBBB.

| Split | N | CLBBB+ | ILBBB+ |
|---|---:|---:|---:|
| Train | 17418 | 428 | 62 |
| Validation | 2183 | 54 | 7 |
| Test | 2198 | 54 | 8 |

Baseline: kernel-7 stem, 32 channels, three two-convolution residual blocks (32,64,128; strides 1,2,2), global mean pooling, linear 128→2 (242114 parameters). Modified architecture: parallel kernels 5/11/21 with 8/12/12 channels, same residual blocks, mean+max pooling, dropout 0.2, linear 256→2 (242570 parameters). No ablation. Adam 0.001, batch 32, maximum 30 epochs, patience 7 consecutive epochs without a new minimum validation weighted BCE. Positive weights are training negatives/positives independently per label. Seed 42, no scheduler or oversampling. AUROC and AUPRC (scikit-learn average precision) computed separately per target.

Median extraction adapts generative_ecg/beats.py to 100 Hz: multilead 8–30 Hz filtered RMS QRS envelope, 70 ms smoothing, 250 ms peak separation, ±40 ms refinement, zero-phase 0.5 Hz high-pass, 300 ms before/500 ms after anchor, amplitude and QRS consistency filtering, at least 3 beats, pointwise median. Output (12,80). Original split indices retained. All 12 leads inform alignment and rejection, so this is single-lead classification, not strictly single-lead acquisition. Extraction examples accepted 6,9,11 beats for non-LBBB, CLBBB, ILBBB respectively. Full cohort rejection audit and verified median metrics are pending.

## Observed full-ECG results
Values transcribed from local notebook screenshots, rounded to four decimals. See [CSV](full_ecg_results.csv).

| Architecture | Lead | Test CLBBB AUROC | AP | Test ILBBB AUROC | AP |
|---|---|---:|---:|---:|---:|
| Baseline | II | .9817 | .6043 | .8725 | .0575 |
| Baseline | III | .9855 | .6778 | .9584 | .0947 |
| Baseline | aVR | .9961 | .8730 | .9322 | .1222 |
| Baseline | aVF | .9763 | .5216 | .9335 | .0416 |
| Modified | II | .9784 | .6743 | .8513 | .0507 |
| Modified | III | .9618 | .4539 | .8012 | .0179 |
| Modified | aVR | .9945 | .8541 | .8918 | .1030 |
| Modified | aVF | .9706 | .6273 | .8204 | .1027 |

Baseline won 11/16 validation and 13/16 test metric comparisons. aVR led CLBBB metrics in both architectures and splits. Baseline aVR was selected using validation CLBBB AP (.8358), with validation AUROC .9878 and ILBBB AUROC/AP .9489/.1716. Selection criterion was chosen after initial validation comparison, before test evaluation. Lead III baseline had highest test ILBBB AUROC. Baseline is preferred for controlled lead comparisons. aVR's larger absolute horizontal projection among the four leads is a physiological hypothesis, not proof of a learned mechanism.

## Held-out test ROC and precision–recall curves

Full 10-second ECG, PTB-XL fold 10 (2,198 recordings; 54 CLBBB and 8 ILBBB positives). Each curve compares II, III, aVR and aVF. The PR legends report average precision (AP).

Figure files are cropped from the original notebook output, not reconstructed from aggregate metrics. They must be placed in the `single_lead/figures/` directory to render.

### CLBBB

| Baseline ResNet | Improved ResNet |
|---|---|
| ![Baseline CLBBB ROC](figures/test_baseline_CLBBB_ROC.png) | ![Improved CLBBB ROC](figures/test_improved_CLBBB_ROC.png) |
| ![Baseline CLBBB precision-recall](figures/test_baseline_CLBBB_PR.png) | ![Improved CLBBB precision-recall](figures/test_improved_CLBBB_PR.png) |

aVR provides the highest CLBBB AUROC and AP for both architectures.

### ILBBB

| Baseline ResNet | Improved ResNet |
|---|---|
| ![Baseline ILBBB ROC](figures/test_baseline_ILBBB_ROC.png) | ![Improved ILBBB ROC](figures/test_improved_ILBBB_ROC.png) |
| ![Baseline ILBBB precision-recall](figures/test_baseline_ILBBB_PR.png) | ![Improved ILBBB precision-recall](figures/test_improved_ILBBB_PR.png) |

ILBBB curves are particularly sensitive to individual cases because the test split contains only eight positive recordings. The prevalence references are 0.0246 for CLBBB and 0.0036 for ILBBB.

## Pending and limitations
Median notebook aggregation failed because at least one saved summary representation was not median_beat. Do not relabel unverified checkpoints; verify original loader/config or retrain affected runs. New scripts explicitly record representation and input length. Compare full and median scores on the same accepted validation/test indices; report exclusions by label. Training cohorts may differ due to rejection. Filtering, alignment, and aggregation jointly differ between representations. Single seed, rare ILBBB (7 validation and 8 test positives), descriptive comparisons only, previously used PTB-XL test fold, no external cohort or clinical validation. Weighted sigmoid scores are not calibrated probabilities. No claim of physiological causation.
