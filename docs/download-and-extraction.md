# Completed PTB-XL download and V6 extraction

Only PTB-XL version 1.0.3 `records100/` was downloaded, from
[PhysioNet](https://physionet.org/files/ptb-xl/1.0.3/records100/).
The checksum manifest was fetched separately for integrity verification.
No `records500/` directory was downloaded. About 60 GiB was free before download;
the downloader required a conservative 1.9 GB reserve. Python requests was already
installed, so no additional dependency installation was needed.

All 43,598 files (21,799 `.hea`/`.dat` pairs) passed the official SHA-256 checks.
Every low-resolution path in `ptbxl_database.csv` is present, including all
6,532 conservative-cohort recordings. Original subfolders are preserved under
`data/raw/ptb-xl/records100/`. Disk usage is approximately 596 MiB for the waveforms
and 26 MiB for the processed V6 output folder.

## Saved arrays

All outputs are in `data/processed/v6/`:

| Split | X filename | X shape | y filename | y shape | Normal / CLBBB |
| --- | --- | --- | --- | --- | --- |
| Train, folds 1–8 | `X_train.npy` | `(5278, 1, 1000)` | `y_train.npy` | `(5278,)` | 5095 / 183 |
| Validation, fold 9 | `X_val.npy` | `(614, 1, 1000)` | `y_val.npy` | `(614,)` | 595 / 19 |
| Test, fold 10 | `X_test.npy` | `(640, 1, 1000)` | `y_test.npy` | `(640,)` | 615 / 25 |

X arrays are float32 V6 signals in mV at 100 Hz, 1000 samples per recording.
No filtering, resampling, padding or normalization was applied. Labels are int64:
0 = normal, 1 = complete LBBB. The original conservative policy is unchanged.
Each split has a row-aligned CSV manifest; preprocessing details and the selection
manifest SHA-256 are saved in `preprocessing.json`.

## Verification performed

- All metadata paths have both waveform files; every downloaded file matched the
  official SHA-256 manifest before receiving its final name.
- Every selected source recording passed 100 Hz, 1000-sample, unique-V6, mV-unit,
  and finite-sample checks. No selected records were silently omitted.
- Every saved array row was reloaded and compared with its source V6 waveform.
- Labels and selected ECG IDs were checked against the existing conservative
  classifier and original metadata; saved manifests preserve row order.
- Shapes, dtypes and every saved label were checked; no missing/nonfinite signals
  were found.
- Patient sets are disjoint across splits: 4994 train, 593 validation, 593 test.
  Full-source patient-to-fold consistency also passed.
- SHA-256 hashes confirm original CSVs and existing processed metadata outputs
  are unchanged. `metadata_report.json` remains the historical pre-download audit;
  consult `results/v6_verification.json` for current waveform verification.
- `.hea`, `.dat`, processed arrays, raw data and generated results are Git-ignored;
  no data files are tracked. Nothing was committed or pushed. No model was trained.

## Resume or reproduce

From the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.download_records100
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.extract_v6
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.verify_v6
```

Run these in order and proceed only after each succeeds. The downloader skips
existing checksum-valid files and resumes `.part` files using byte ranges. It
retries failed transfers and restarts only a partial file if the server does not
support resume. Extraction overwrites generated arrays, never raw recordings.
The verifier uses the original hashes in `results/pre_download_hashes.json`.

Local evidence:

- `data/raw/ptb-xl/records100-checksums.txt`: official records100 hashes.
- `data/raw/ptb-xl/records100-download-report.json`: successful verified transfer counts.
- `results/v6_verification.json`: independent signal, label, shape and split checks.

Your partner can now load arrays with `numpy.load` and convert them to PyTorch
tensors. Any learned normalization should be fit on training data only.
