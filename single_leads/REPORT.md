# Single-lead models: V1, V2, V3, V4 — full 10-second ECG vs median beat

Eight models: one per lead (V1, V2, V3, V4) for each input type (full 10-second ECG, PTB-XL+ median beat). Each model has two independent sigmoid outputs, CLBBB and ILBBB, as on `feature/dataset1-diagnostic-leads`. All models are trained on Dataset 1 and evaluated with one frozen checkpoint on the fold-10 tests of Dataset 1 and Dataset 2.

## Method

- **Inputs.** Full ECG: PTB-XL 1.0.3 `records500`, 10 s at 500 Hz (5,000 samples). Median beat: PTB-XL+ 1.0.1 `median_beats/12sl` (GE 12SL), 1.2 s at 500 Hz (600 samples). One lead per model, physical mV, no filtering, normalization or augmentation. Median-beat amplitudes were checked against the raw ECG (median peak-to-peak ratio 1.02).
- **Cohorts.** Labels, folds and the Dataset 1 rule are imported unchanged from `diagnostic_leads/prepare.py`. Recordings without a published PTB-XL+ median beat are dropped from every split for **both** input types, so each full-vs-median pair is compared on identical recordings.
- **Splits.** Train: Dataset 1 folds 1–8 (7,467 recordings; 406 CLBBB, 59 ILBBB). Validation: Dataset 1 fold 9 (961; 54, 7). Test: fold 10 — Dataset 1 (960; 48, 8) and Dataset 2 (2,158; 48, 8). Patient-disjoint.
- **Excluded for lack of a median beat:** train 266 (22 CLBBB, 3 ILBBB), validation 14 (0, 0), Dataset 2 test 40 (6, 0).
- **Model and training.** The six-lead branch's three-block 1D ResNet with a 1-channel stem (242,114 parameters). Adam lr 0.001, batch 32, seed 42, at most 30 epochs, early-stopping patience 7, training-derived positive-class weights, checkpoint at minimum validation weighted BCE. Device: mps.
- **Metrics.** AUROC and AUPRC (average precision) with record-level bootstrap 95% CIs (2,000 resamples). Full-vs-median differences use paired bootstrap on the same resampled recordings. No threshold tuning on test data.

## Dataset 1 test (LBBB vs normal)

**CLBBB** — 960 recordings, 48 positives (prevalence 5.0%).

| Lead | AUROC, full ECG | AUROC, median beat | AUPRC, full ECG | AUPRC, median beat |
|---|---:|---:|---:|---:|
| V1 | 0.997 (0.994–0.999) | 0.997 (0.993–0.999) | 0.933 (0.859–0.983) | 0.883 (0.767–0.983) |
| V2 | 0.998 (0.996–1.000) | 0.997 (0.994–0.999) | 0.954 (0.898–0.994) | 0.914 (0.810–0.985) |
| V3 | 0.999 (0.998–1.000) | 0.995 (0.990–0.998) | 0.981 (0.957–0.997) | 0.880 (0.767–0.970) |
| V4 | 0.998 (0.995–0.999) | 0.996 (0.992–0.999) | 0.949 (0.888–0.988) | 0.886 (0.775–0.971) |

**ILBBB** — 960 recordings, 8 positives (prevalence 0.8%).

| Lead | AUROC, full ECG | AUROC, median beat | AUPRC, full ECG | AUPRC, median beat |
|---|---:|---:|---:|---:|
| V1 | 0.920 (0.777–0.992) | 0.947 (0.901–0.980) | 0.195 (0.065–0.487) | 0.105 (0.041–0.237) |
| V2 | 0.946 (0.885–0.985) | 0.945 (0.906–0.981) | 0.143 (0.052–0.370) | 0.168 (0.040–0.480) |
| V3 | 0.901 (0.774–0.976) | 0.968 (0.936–0.990) | 0.099 (0.030–0.289) | 0.262 (0.067–0.558) |
| V4 | 0.923 (0.845–0.983) | 0.935 (0.867–0.987) | 0.144 (0.037–0.430) | 0.135 (0.047–0.302) |

## Dataset 2 test (LBBB vs all others)

**CLBBB** — 2,158 recordings, 48 positives (prevalence 2.2%).

| Lead | AUROC, full ECG | AUROC, median beat | AUPRC, full ECG | AUPRC, median beat |
|---|---:|---:|---:|---:|
| V1 | 0.989 (0.981–0.995) | 0.990 (0.986–0.994) | 0.720 (0.583–0.846) | 0.587 (0.464–0.749) |
| V2 | 0.991 (0.986–0.995) | 0.978 (0.971–0.984) | 0.706 (0.576–0.820) | 0.329 (0.243–0.436) |
| V3 | 0.982 (0.974–0.988) | 0.965 (0.951–0.975) | 0.528 (0.393–0.661) | 0.257 (0.190–0.339) |
| V4 | 0.975 (0.966–0.982) | 0.970 (0.961–0.979) | 0.408 (0.292–0.567) | 0.316 (0.230–0.444) |

**ILBBB** — 2,158 recordings, 8 positives (prevalence 0.4%).

| Lead | AUROC, full ECG | AUROC, median beat | AUPRC, full ECG | AUPRC, median beat |
|---|---:|---:|---:|---:|
| V1 | 0.902 (0.757–0.982) | 0.929 (0.859–0.980) | 0.051 (0.016–0.133) | 0.050 (0.017–0.130) |
| V2 | 0.919 (0.823–0.979) | 0.891 (0.820–0.952) | 0.055 (0.015–0.161) | 0.031 (0.008–0.094) |
| V3 | 0.842 (0.703–0.939) | 0.906 (0.844–0.952) | 0.022 (0.007–0.064) | 0.026 (0.010–0.081) |
| V4 | 0.864 (0.742–0.963) | 0.878 (0.748–0.977) | 0.048 (0.008–0.214) | 0.041 (0.013–0.105) |

## Median beat minus full ECG (paired bootstrap 95% CI)

| Test population | Outcome | Lead | ΔAUROC | ΔAUPRC |
|---|---|---|---:|---:|
| dataset1_test | CLBBB | V1 | -0.000 (-0.003 to +0.002) | -0.051 (-0.161 to +0.049) |
| dataset1_test | CLBBB | V2 | -0.001 (-0.004 to +0.001) | -0.040 (-0.135 to +0.025) |
| dataset1_test | CLBBB | V3 | -0.004 (-0.008 to -0.001) | -0.101 (-0.211 to -0.014) |
| dataset1_test | CLBBB | V4 | -0.002 (-0.005 to -0.000) | -0.064 (-0.162 to +0.002) |
| dataset1_test | ILBBB | V1 | +0.027 (-0.066 to +0.182) | -0.090 (-0.358 to +0.048) |
| dataset1_test | ILBBB | V2 | -0.001 (-0.052 to +0.070) | +0.025 (-0.106 to +0.257) |
| dataset1_test | ILBBB | V3 | +0.067 (-0.013 to +0.199) | +0.163 (-0.065 to +0.444) |
| dataset1_test | ILBBB | V4 | +0.013 (-0.012 to +0.040) | -0.009 (-0.250 to +0.115) |
| dataset2_test | CLBBB | V1 | +0.001 (-0.004 to +0.007) | -0.134 (-0.261 to +0.004) |
| dataset2_test | CLBBB | V2 | -0.013 (-0.019 to -0.008) | -0.376 (-0.469 to -0.268) |
| dataset2_test | CLBBB | V3 | -0.017 (-0.028 to -0.008) | -0.270 (-0.371 to -0.168) |
| dataset2_test | CLBBB | V4 | -0.005 (-0.010 to +0.001) | -0.092 (-0.189 to -0.009) |
| dataset2_test | ILBBB | V1 | +0.027 (-0.085 to +0.190) | -0.001 (-0.072 to +0.065) |
| dataset2_test | ILBBB | V2 | -0.028 (-0.113 to +0.092) | -0.024 (-0.091 to +0.010) |
| dataset2_test | ILBBB | V3 | +0.064 (-0.049 to +0.211) | +0.005 (-0.028 to +0.053) |
| dataset2_test | ILBBB | V4 | +0.015 (-0.029 to +0.063) | -0.007 (-0.157 to +0.057) |

## Training runs

| Input | Lead | Epochs run | Selected epoch | Validation weighted BCE | Minutes |
|---|---|---:|---:|---:|---:|
| full | V1 | 13 | 6 | 0.5089 | 17.2 |
| full | V2 | 15 | 8 | 0.4777 | 18.6 |
| full | V3 | 30 | 29 | 0.4675 | 36.0 |
| full | V4 | 17 | 10 | 0.2717 | 20.3 |
| median | V1 | 13 | 6 | 0.5170 | 1.9 |
| median | V2 | 21 | 14 | 0.3975 | 3.1 |
| median | V3 | 26 | 19 | 0.4007 | 3.8 |
| median | V4 | 22 | 15 | 0.3611 | 3.2 |

## Limitations

- full V3 selected an epoch at the 30-epoch cap, so validation loss was still falling; a longer run might score differently. The cap was kept at the six-lead branch's value for comparability.
- Dataset 1 test is a subset of Dataset 2 test (same PTB-XL fold 10), so the two test results are correlated and are not external validation. Their positives are identical.
- ILBBB is exploratory: 8 test positives. Its AUPRC confidence intervals are wide; treat lead or input-type differences for ILBBB with caution.
- V1 and V2 belong to the conventional diagnostic lead set (I, aVL, V1, V2, V5, V6); V3 and V4 do not. The four leads were trained as requested.
- Median beats are GE 12SL outputs; recordings where 12SL produced none are excluded from both input types. Exclusions are not random: 22 of 266 excluded training recordings and 6 of 40 excluded test recordings are CLBBB, so 6 of 54 CLBBB test positives are missing. Results describe recordings for which 12SL produced a median beat, and the CLBBB test set differs from the six-lead branch's, so numbers are not directly comparable with it.
- Each configuration is a single seed-42 run; seed-to-seed variation is not estimated. Sigmoid outputs from weighted BCE are scores, not calibrated probabilities.
- Bootstrap CIs resample recordings, not patients; a few patients have several fold-10 recordings.

## Artifacts

In `results/`: `metrics.csv` (all metrics, including thresholded counts at 0.50), `median_minus_full.csv`, `evaluation_provenance.json` (checkpoint hashes), and per model `config.json`, `history.json`, `complete.json`. `data_audit.json` holds cohort counts, exclusions and the unit check. Checkpoints, record-level predictions and ECG arrays stay local.
