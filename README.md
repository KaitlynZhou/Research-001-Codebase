# Generative Experiment 1

This experiment extracts one representative 12-lead median beat from each eligible PTB-XL recording and trains a variational autoencoder to reconstruct and generate these beats.

## Setup and execution

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
