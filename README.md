# Research 001 — Single-lead LBBB prediction

## Predictive Experiment 2: non-diagnostic leads

This branch extends the 12-lead Dataset 2 experiment to independent classifiers for II, III, aVR and aVF, with two binary outputs (CLBBB, ILBBB). See [experiment documentation](single_lead/README.md) for methods, results, limitations and reproduction commands.

**Status:** eight full-ECG models trained and evaluated locally; median-beat extraction tested on three examples. Eight median training summaries were collected locally, but a representation metadata assertion failed. Median training provenance and final evaluation remain unverified.

The original generative experiment documentation is preserved below.

---

# Generative Experiment 1

## Final Results: Synthetic vs Real ECGs

The final **waveform mixture** generates 12-lead median beats at 500 Hz. It fits a joint distribution in waveform space and enforces the four limb-lead identities when decoding. The ECG-SCD VAE documented below is the original baseline.

The final evaluation compares **1,000 synthetic beats** with **2,198 real held-out PTB-XL fold 10 median beats**. Brackets show the 5th–95th percentiles around the median. Displayed values are rounded from the saved final evaluation.

| Metric | Final synthetic (n = 1,000) | Real held-out (n = 2,198) |
|---|---:|---:|
| Pairwise waveform RMSE (mV) | 0.1789 [0.1057–0.3582] | 0.1822 [0.1004–0.3946] |
| Pairs below 0.01 mV RMSE | 0.00% | 0.00% |
| Median nearest-training RMSE (mV) | 0.0716 | 0.0648 |
| PR interval (ms) | 164.0 [124.0–200.0] | 162.0 [130.0–200.0] |
| PR measurable | 654/1,000 (65.4%) | 1,487/2,198 (67.7%) |
| QRS duration (ms) | 110.0 [86.0–168.0] | 102.0 [78.0–158.0] |
| QRS measurable | 964/1,000 (96.4%) | 2,164/2,198 (98.5%) |
| QT interval (ms) | 392.0 [336.0–444.7] | 382.0 [330.0–440.0] |
| QT measurable | 854/1,000 (85.4%) | 1,858/2,198 (84.5%) |
| Lead II P-region peak deflection (mV) | 0.0869 [0.0314–0.1622] | 0.0909 [0.0180–0.1798] |
| Lead II QRS-region peak deflection (mV) | 0.6149 [0.2263–1.2519] | 0.5856 [0.2051–1.2143] |
| Lead II T-region peak deflection (mV) | 0.1796 [0.0557–0.3672] | 0.1782 [0.0474–0.3817] |
| Limb-lead identity RMSE, median (mV) | 5.325e-09 | 0.004974 |
| Limb-lead identity RMSE, 95th percentile (mV) | 9.513e-09 | 0.010261 |
| Median fast waveform residual RMSE (mV) | 0.0257 | 0.0263 |

- **Diversity and similarity.** Pairwise RMSE uses 124,750 pairs from 500 sampled beats per group. The nearest-training comparison uses those same 500 beats per group and all 17,417 accepted training beats. Neither full set contained an exact duplicate.
- **Intervals.** The same annotation-trained detector estimated PR, QRS duration, and QT in both groups. Interval summaries use only beats with an estimate. Coverage is the fraction of beats with an estimate. The synthetic median QRS and QT were 8 ms and 10 ms longer than the corresponding real medians.
- **Morphology.** Lead II peak deflection measures the largest absolute amplitude relative to the line joining the endpoints of each fixed P, QRS, or T region. The mean across-lead distances between synthetic and real deflection distributions were **0.0077, 0.0309, and 0.0153 mV** for P, QRS, and T, respectively. The fast waveform residual is the RMSE from a smoothed copy of the same beat.
- **Lead consistency.** The synthetic limb identity errors are at numerical precision because the model enforces the four identities. This check covers the six limb leads. The experiment evaluates single median beats, so it provides no measurement of rhythm or beat-to-beat variation.

[Full final evaluation and reproduction commands](reports/generative_experiment1_waveform/README.md) · [Exact evaluation values](reports/generative_experiment1_waveform/test_metrics.json) · [12-lead waveform comparison](reports/generative_experiment1_waveform/example_1.png) · [Lead II P-wave examples](reports/generative_experiment1_waveform/p_wave_examples.png) · [Waveform diversity figure](reports/generative_experiment1_waveform/waveform_spread.png) · [Reconstruction overlay](reports/generative_experiment1_waveform/reconstruction.png)

## Original VAE baseline

This experiment extracts one representative 12-lead median beat from each eligible PTB-XL recording and trains a variational autoencoder to reconstruct and generate these beats.

### Setup and execution

Use Python 3.11 and a local PTB-XL directory with `ptbxl_database.csv` and the complete `records500/` tree. Follow the Code availability section of the linked paper to obtain the ECG-SCD source, then check out revision `f7b58768e2bb760a9a665aeb82191269fdf1d39b` in a separate directory. This project imports the original VAE modules at runtime. The upstream `LICENSE` file contains the source license. [Research paper](https://www.nature.com/articles/s41586-026-10674-6)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ECG_SCD_ROOT=/path/to/ECG-SCD
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="${TMPDIR:-/tmp}/generative-ecg-mpl"

python -m pytest -q
python -m generative_ecg.beats "$HOME/Desktop/PTB-XL" artifacts/beats
python -m generative_ecg.model train "$ECG_SCD_ROOT" artifacts/beats artifacts/model
python -m generative_ecg.model infer "$ECG_SCD_ROOT" artifacts/beats artifacts/model --out artifacts/generated
python -m generative_ecg.evaluate artifacts/beats artifacts/generated artifacts/evaluation
```

Extraction, training, and inference each require a fresh output directory. Use the split CSV's zero-based `array_index` to join an accepted ECG to its stored beat. The output also records rejected ECGs, accepted beat counts, and the waveform contract. Fold 10 reconstructions preserve the row order of `test.npy`. The artifacts directory is gitignored and stays local.

## Median beats

The loader follows the existing native-rate experiment's WFDB and patient-fold checks. It resolves all 12 channels into canonical order, checks the original rate, dimensions and mV units, and uses `filename_hr` exclusively. Official PTB-XL folds 1–8 are training, 9 is validation, and 10 is test. A multilead band-passed QRS envelope provides shared peak candidates. Local refinement gives one common anchor per beat. A zero-phase high-pass filter removes baseline wander before cutting 300 ms pre-anchor and 500 ms from the anchor onward at native resolution. Extraction rejects incomplete windows, excessive amplitudes, and QRS morphology outliers. At least three valid windows are required. The pointwise median produces a float32 array of shape `(12, 400)` per recording.

The extraction plot overlays an accepted beat and its representative median. The peak finder targets the dominant QRS deflection across leads, so the anchor may fall on an S wave in an individual lead. Fixed windows can capture adjacent activity during fast rhythms or truncate late repolarization. Median aggregation emphasizes the dominant morphology and may suppress intermittent findings.

## Source and adaptation

- Upstream `00_Data_Preprocessing/x04_segment_beats.py` and `x05_create_morphing_beats.py` produce individual filtered NTUH beats with the `(-100, 200)` sample window and additional cohort criteria. This experiment supplies PTB-XL median beats and uses the official patient folds.
- Upstream `02_Morphing/s05_utils_vae.py` defines the encoder, decoder, Gaussian reconstruction loss, KL divergence, and reparameterized sampler. The imported CNN encoder has two 12-filter convolutions with average pooling, a 128-unit dense layer, and 512-dimensional mean and softplus variance heads. The imported decoder has four 100-unit ReLU layers and a linear projection to 4,800 values. The checkpoint records the upstream source file's SHA-256 and inference checks it.
- Upstream `s08_train_generator.py` trains a discriminatively regularized VAE with a separate SCD predictor and samples using training posterior moment statistics. The PTB-XL adapter uses Gaussian reconstruction and a fixed KL weight of 0.01 with Adam and cosine learning-rate decay. It omits the SCD regularizer because PTB-XL lacks the corresponding outcome/model for this experiment. Training uses folds 1–8 and selects the checkpoint by posterior-mean validation MSE on fold 9.
- The upstream loss and `s10_morph_ecgs.py` reshape decoder outputs into ECGs. Here, reconstruction decodes the posterior mean deterministically. Generation follows the upstream training-posterior moment sampling scheme, with sampled variances floored at a small positive value before decoding. All sampling statistics are fitted on the training split.

## Evaluation

The evaluation writes per-lead peak-to-peak amplitude summaries, limb-lead identity error, held-out reconstruction errors, and PR, QRS, and QT timing proxies with measurement coverage. The timing proxies use a threshold-based multilead activity envelope and are exploratory. A single median beat provides no RR interval or rhythm-level evidence.

Variation is measured with pointwise standard deviation and pairwise waveform RMSE. The near-duplicate threshold is 0.01 mV waveform RMSE. A reproducible 500-beat subset from each of the held-out and generated groups is compared to the entire training split using nearest-neighbor waveform RMSE. This is a similarity screen at fixed alignment.

Files include `artifacts/beats/extraction_example.png`, `artifacts/model/best.msgpack`, `artifacts/model/history.json`, and `artifacts/evaluation/metrics.json` with waveform and timing figures.

## Completed run

The Desktop source contained all expected WFDB files for 21,799 recordings, with zero patients crossing official folds. Extraction retained 17,417 training, 2,183 validation, and 2,198 test beats. One training recording had too few valid windows after amplitude filtering. Training ran for 20 epochs on CPU with seed 42 and batch size 128. Epoch 20 had the lowest validation MSE, 0.01660 mV². The run reconstructed all 2,198 held-out beats and generated 1,000 synthetic beats.

| Metric | Held-out real | Reconstructed | Synthetic |
|---|---:|---:|---:|
| Median limb-lead identity RMSE (mV) | 0.0050 | 0.0118 | 0.0130 |
| Median QRS timing proxy (ms) | 72 | 72 | 76 |
| PR proxy measured | 332 / 2,198 | 30 / 2,198 | 0 / 1,000 |
| QT proxy measured | 1,280 / 2,198 | 629 / 2,198 | 2 / 1,000 |
| Median pairwise waveform RMSE (mV) | 0.1822 | — | 0.0217 |
| Pairs below 0.01 mV RMSE | 0% | — | 3.30% |
| Median nearest-training waveform RMSE (mV) | 0.0648 | — | 0.0422 |

Held-out reconstruction RMSE was 0.1273 mV, and MAE was 0.0616 mV. The synthetic set had zero exact internal duplicates, but its median pairwise distance was roughly 12% of the real set's. Generated beats were also closer to training examples than held-out beats were under the same nearest-neighbor test. These findings indicate a narrow range of generated morphologies and warrant caution when interpreting similarity as memorization. The timing proxies recovered QRS duration in most beats, but the near-total failure to measure synthetic PR and QT, limited waveform variation, and higher lead-identity error mean this run did not establish physiological realism.

The saved baseline includes [evaluation metrics](reports/generative_experiment1/metrics.json), [training history](reports/generative_experiment1/history.json), [extraction example](reports/generative_experiment1/extraction_example.png), [waveform comparison](reports/generative_experiment1/comparison_1.png), [waveform variation](reports/generative_experiment1/waveform_variation.png), and [interval distributions](reports/generative_experiment1/intervals.png). The reproduction commands create the checkpoint and arrays locally under `artifacts/`.

## Validated waveform generator

The [waveform mixture report](reports/generative_experiment1_waveform/README.md) documents the selected generator, annotated interval audit, and reproduction commands. The model learns a joint distribution in waveform space and enforces all four limb-lead identities. The original ECG-SCD experiment above remains the baseline.
