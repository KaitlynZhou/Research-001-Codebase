# Validated waveform mixture

The selected generator learns a joint distribution of aligned 500 Hz median beats directly in waveform space. It uses the original PTB-XL extraction and patient folds. Two orthonormal coordinates span the six limb leads, and the six chest leads complete the eight independent channels. Reconstruction and generation both map back to the canonical 12 leads, enforcing four limb-lead identities by construction.

The model weights the P and T regions six and three times relative to QRS during PCA, uses 64 whitened components, and fits a regularized 12-component Gaussian mixture with full covariances. It samples scores jointly and inverts the transform. The PCA model retained 96.19% of weighted training variance. The previous VAE architectures, outputs, and original baseline report were retained.

## Experiment design

Fit candidate models on PTB-XL folds 1–7 and select settings using fold 8, a patient-disjoint subset of the official training data. The compact 32-component variant lost QRS detail. The unweighted 64-component variant had lower QRS reconstruction error than the selected weighted variant (0.0391 vs. 0.0480 mV) but higher P-region error (0.0209 vs. 0.0159 mV) and T-region error (0.0247 vs. 0.0206 mV). Synthetic pairwise distances remained close to 0.18 mV in all three variants. The selected variant was refit on all 17,417 accepted training beats, checked on fold 9, and evaluated on the unchanged 2,198 fold 10 beats. Each generator contributed 1,000 synthetic samples. [Training-fold ablations](development_metrics.json) · [Configuration](config.json) · [Validation metrics](validation_metrics.json)

## Detector validation

The annotation-trained detector uses smoothed lead-II and multilead waveform features to label P, QRS, and T segments. It requires a local waveform deflection for P and T detections, with the amplitude floors estimated from the lower decile of annotated training waves. It was fitted only on odd-numbered records 1–79 from the public LUDB dataset. On the 358 annotated windows from even-numbered records 2–80, conditional PR/QRS/QT MAE fell from **18.1 / 26.4 / 50.0 ms** with the original detector to **7.0 / 11.9 / 15.6 ms**. It detected 201 of 266 reference PR intervals, 358 of 358 QRS durations, and 259 of 320 reference QT intervals. It still returned 28 PR estimates and 32 QT estimates without an in-window reference. On 197 windows from independent records 81–100, conditional MAE was 6.8 / 11.4 / 15.0 ms versus 16.7 / 19.9 / 46.3 ms for the original detector. This is a lead-II annotation audit on real beat windows. Performance may differ on PTB-XL medians and generated waveforms. [Detector audit](detector_audit.json) · [LUDB reference dataset](https://physionet.org/content/ludb/1.0.1/)

## Held-out comparison

Interval coverage and distances below use the same annotation-trained detector on every set. Coverage counts estimates, and the interval distances compare only the estimates that detector returned. Regional deflection distance compares each lead's distribution of peak deflection from a local straight-line baseline, then averages the 12 distances.

| Metric | Original VAE | Frozen latent mixture | Weighted VAE | Waveform mixture | Real |
|---|---:|---:|---:|---:|---:|
| PR coverage | 91.9% | 68.6% | 56.3% | **65.4%** | 67.7% |
| PR distribution distance (ms) | 17.0 | 11.9 | 14.0 | **2.4** | 0 |
| QRS coverage | 100.0% | 97.9% | 98.2% | 96.4% | 98.5% |
| QRS distribution distance (ms) | 16.3 | 6.6 | 7.9 | 7.8 | 0 |
| QT coverage | 100.0% | 95.4% | 91.8% | **85.4%** | 84.5% |
| QT distribution distance (ms) | 24.0 | 17.8 | 17.5 | **5.5** | 0 |
| P-region deflection distance (mV) | 0.0311 | 0.0259 | 0.0286 | **0.0077** | 0 |
| T-region deflection distance (mV) | 0.1043 | 0.0742 | 0.0641 | **0.0153** | 0 |
| Limb identity error p95 (mV) | 0.0216 | 0.0624 | 0.0113 | **<0.000001** | 0.0103 |
| Median pairwise waveform RMSE (mV) | 0.0217 | 0.0837 | 0.1042 | **0.1789** | 0.1822 |
| Pairs below 0.01 mV RMSE | 3.30% | 0.86% | 0.22% | **0.00%** | 0.00% |
| Median nearest-training RMSE (mV) | 0.0422 | 0.0446 | 0.0450 | 0.0716 | 0.0648 |
| Paired reconstruction RMSE (mV) | 0.1273 | 0.1273 | 0.1187 | **0.0308** | — |

The original interval detector also found more measurable P and T morphology in the new samples. PR coverage was 9.9% and QT coverage was 53.0%, versus 0% and 36.1% for the weighted VAE. The lead-II examples show more varied, recognizable P-like deflections. The largest further gains are in morphology and diversity. The synthetic median QRS estimate was 110 ms versus 102 ms on real beats, and QRS coverage fell compared with the VAE. The closest synthetic-to-training match in the 500-sample screen was 0.0420 mV, versus 0.0281 mV for real held-out beats. The fast waveform residual screen and paired errors are saved in the full metrics. [Held-out metrics](test_metrics.json) · [Lead-II P-region examples](p_wave_examples.png) · [12-lead waveform example](example_1.png) · [Reconstruction overlay](reconstruction.png) · [Waveform variation](waveform_spread.png)

## Limitations

The generator models a single median beat. It inherits the extraction window, alignment jitter, and occasional adjacent-beat contamination. It does not model rhythm, RR intervals, or beat-to-beat variation. Rare diagnostic morphologies and clinical interpretation were not independently adjudicated. Exact limb-lead relationships alone do not establish precordial or clinical validity. The annotated detector can still overcall or miss waves under domain shift, so the report includes waveform evidence, a second detector, and the real reference alongside coverage.

## Reproduce

Install the repository requirements and extract PTB-XL beats using the root README. Download LUDB records 1–80 and, for the independent check, records 81–100. Each record uses `hea`, `dat`, and lead-II `ii` files. Keep those files and generated arrays under gitignored `artifacts/`.

```sh
python -m generative_ecg.annotated_detector artifacts/ludb/data artifacts/detector_audit.json \
  --confirmation artifacts/ludb/confirmation
python -m generative_ecg.waveform_model artifacts/beats artifacts/waveform_final --reconstruct val
python -m generative_ecg.waveform_audit artifacts artifacts/waveform_final artifacts/ludb/data \
  artifacts/waveform_val_audit --split val
python -m generative_ecg.waveform_audit artifacts artifacts/waveform_final artifacts/ludb/data \
  artifacts/waveform_test_audit --split test
```

Use `waveform_model --development` with fresh output directories and the settings in the ablation JSON to reproduce training-fold comparisons. The audit automatically includes earlier results when their arrays are locally available at `artifacts/generated/`, `artifacts/mixture_generated/`, and `artifacts/physiology_p_test/`. Both audits reuse the same fitted sampler and synthetic draws. The fitted parameters are saved as a NumPy archive that can be loaded with `allow_pickle=False`.
