# PTB-XL V6 data processing

The metadata pipeline is complete and has run on the two supplied CSVs. The PTB-XL 1.0.3 records100 download and real V6 extraction are also complete; see [verification results](download-and-extraction.md). No model
is built or trained. The metadata and extraction scripts perform no downloads; the separate downloader fetches only records100. No script commits or pushes changes.

## Environment and commands

Run commands from the repository root. The existing Python 3.11 `.venv` was reused
and NumPy, Pandas, WFDB and their dependencies were installed. In VS Code select
`.venv/bin/python` as the Python interpreter.

```sh
source .venv/bin/activate
# For a new environment: python3.11 -m venv .venv
python -m pip install -r requirements.txt
# To reproduce the resolved versions on a compatible machine instead:
# python -m pip install -r requirements-lock.txt
PYTHONPATH=src python -m ecg_lbbb.process_metadata
python -m unittest discover -s tests -v
python -m pip check
```

The metadata command overwrites its generated reports on rerun, never raw files.
To inspect the full code table independently:

```sh
PYTHONPATH=src python -m ecg_lbbb.inspect_metadata
```

## Files and folders

| File or folder | Purpose |
| --- | --- |
| `data/raw/ptb-xl/` | Unmodified copies of the downloaded CSVs; future `records100/` goes here |
| `data/processed/selected_recordings.csv` | One selected recording per row: ECG ID, patient ID, original SCP codes, low/high resolution paths, original fold, binary label, split |
| `data/processed/selection_audit.csv` | Every source recording and its selection/exclusion reason |
| `data/processed/fold_counts.csv` | Selected recording and unique patient counts per fold and class |
| `data/processed/metadata_report.json` | Full source inspection, code definitions, confidence distributions, counts, checks, source SHA-256 fingerprints and policy |
| `data/processed/v6/` | Future signal arrays and row-aligned manifests |
| `src/ecg_lbbb/inspect_metadata.py` | Existing independent source validator and label inspector, preserved |
| `src/ecg_lbbb/process_metadata.py` | Reads both CSVs with Pandas, applies selection policy, checks folds, writes reports |
| `src/ecg_lbbb/extract_v6.py` | Loads local WFDB recordings, finds V6 by name, checks signals, saves arrays |
| `tests/test_metadata.py` | Existing metadata validation tests, preserved |
| `tests/test_processing.py` | Tests ambiguous-label exclusions and synthetic WFDB V6 extraction |
| `requirements.txt` | Dependencies for data processing only |
| `requirements-lock.txt` | Exact installed dependency versions for this run |
| `requirements-original.txt` | Preserved original requirements, including partner/model dependencies |
| `docs/previous-setup.md` | Preserved earlier project guide; this guide supersedes its staged plans |
| `.gitignore` | Excludes all local data, generated arrays, environments, results and model files |

The originals remain in Downloads. The root README's original content is retained.
Existing source, tests, editor settings and placeholder folders were preserved.
Data is ignored by Git; no data files were already tracked at verification time.

## Diagnosis labels and baseline policy — supervisor review needed

`scp_codes` is a string containing a Python dictionary, for example
`{'CLBBB': 100.0, 'SR': 0.0}`. It is parsed with `ast.literal_eval`, never `eval`.
Keys are statements; values describe annotation confidence. Dictionary order does
not identify a primary diagnosis. Zero means unknown confidence, not absence.
`scp_statements.csv` maps each key to a description and diagnostic/form/rhythm flags.
The supplied file verifies these exact diagnostic labels:

| Code | Description | Use |
| --- | --- | --- |
| `CLBBB` | complete left bundle branch block | class 1 |
| `ILBBB` | incomplete left bundle branch block | always exclude |
| `NORM` | normal ECG | class 0 |

Applied policy `strict-v1`: exactly one target (`CLBBB` or `NORM`), confidence
exactly 100, and no other statement except optional `SR` (verified sinus rhythm).
Any additional diagnostic, rhythm or form code excludes the recording, even at
confidence zero. This deliberately restrictive choice also excludes sinus
bradycardia/arrhythmia/tachycardia and descriptive form statements. Confirm this
policy and confidence cutoff with your supervisor before using the cohort in a
research result. It may exclude clinically acceptable LBBB morphology. It defines
a clean initial annotation cohort, not a clinical diagnosis or proof of health.
No extra exclusion based on signal-quality flags, age, or free-text report is
applied; these remain possible later review criteria.

There are 536 source recordings mentioning CLBBB, 77 mentioning ILBBB, and 9,514
mentioning NORM before filtering. The source contains 21,799 recordings from
18,869 patients. Do not replace these counts with counts from another release.
The exact release cannot be established from the downloaded filenames alone;
match future waveform files to the metadata release. Source hashes are saved.

## Actual results

| Class | Selected recordings | Unique patients |
| --- | ---: | ---: |
| Normal (0) | 6,305 | 5,982 |
| Complete LBBB (1) | 227 | 198 |

Unique patient counts are calculated separately per class; they are not a claim
that patients cannot have both labels on different visits. Patients remain in one
fold regardless of their recording labels. Repeated visits are retained.

| Fold | Split | Normal recordings | LBBB recordings |
| --- | --- | ---: | ---: |
| 1 | train | 613 | 28 |
| 2 | train | 639 | 21 |
| 3 | train | 661 | 19 |
| 4 | train | 621 | 21 |
| 5 | train | 633 | 19 |
| 6 | train | 640 | 23 |
| 7 | train | 656 | 27 |
| 8 | train | 632 | 25 |
| 9 | validation | 595 | 19 |
| 10 | test | 615 | 25 |

Train: 5,095 normal / 183 LBBB; validation: 595 / 19; test: 615 / 25.
Normal patient counts by split: 4,834 / 576 / 572.
LBBB patient counts by split: 160 / 17 / 21.
All 77 ILBBB recordings were excluded. Other mutually exclusive exclusion reasons
(first matching reason wins): no/conflicting target 11,672; uncertain target 2,343;
additional statement 1,175. Selected plus excluded accounts for all 21,799 rows.

Checks passed across the full source: unique recording IDs and low-resolution
paths, required fields present, valid nonempty label dictionaries and confidence
ranges, known SCP codes, valid integer patient IDs/folds, and no patient in more
than one fold. Selected splits have both classes and zero patient overlap.
Seven automated tests pass; `pip check` reports no broken dependencies.

## Waveform extraction

The full version 1.0.3 `records100/` directory is now local and SHA-256 verified.
Real extraction and all-row verification passed. The original metadata report
remains an unchanged pre-download snapshot; its zero waveform-header count is
historical. The current verification is in `results/v6_verification.json`.
Files preserve their original paths:

```text
data/raw/ptb-xl/records100/00000/00001_lr.hea
data/raw/ptb-xl/records100/00000/00001_lr.dat
```

Then run:

```sh
PYTHONPATH=src python -m ecg_lbbb.extract_v6
```

The script preflights all selected `.hea`/`.dat` pairs before loading and never
fetches remote files. It checks 100 Hz, exactly 1,000 samples, a unique V6 lead,
mV units, and finite physical signal values. It fails instead of padding,
truncating, resampling or silently skipping records. All signals are validated
before outputs are written. It writes `X_train.npy`, `y_train.npy`, `X_val.npy`,
`y_val.npy`, `X_test.npy`, `y_test.npy`, split manifests and preprocessing details
into `data/processed/v6/`. Rerunning overwrites generated outputs.

Each X is float32 with shape `(N, 1, 1000)`; each y is int64 with shape `(N,)`.
Rows follow the corresponding split manifest. Your partner can use `np.load`
and `torch.from_numpy` to read them. No filtering or normalization is applied;
fit any subsequent normalization using training data only. For a binary loss,
your partner may need to cast labels to float. Do not tune on the test set.

The folds and confidence interpretation follow the
[official PTB-XL documentation](https://physionet.org/content/ptb-xl/1.0.3/).
Recommended evaluation uses folds 1–8 for training, 9 for validation, 10 for test,
with patient assignments respected. The supplied folds are preserved exactly.

## Resumable records100 download and independent array verification

The version-pinned downloader uses Python requests (installed with WFDB), checks
free disk space, reads the official release SHA-256 list, and downloads only
`records100/` `.hea` and `.dat` files. It never requests `records500/`.
Run from the repository root:

```sh
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.download_records100
```

Existing files are verified and skipped; incomplete files use `.part` names and
HTTP byte-range requests, with retries. A server that ignores ranges restarts
only that partial file. A file is promoted to its final name only after its
SHA-256 matches the official manifest. The checksum list and completion report
are stored locally under `data/raw/ptb-xl/` and ignored by Git.

After the download succeeds:

```sh
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.extract_v6
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.verify_v6
```

`verify_v6.py` checks original metadata and selection hashes recorded for this
run in `results/pre_download_hashes.json`, full metadata waveform coverage,
conservative labels, array shapes/dtypes/finiteness, row order, every saved V6
waveform against its raw source, and patient separation. It writes
`results/v6_verification.json`. The download, extraction and verification commands
are separate so an interrupted download cannot trigger premature extraction.

## Reproduce the conservative dataset from a fresh clone

Use Python 3.11 and run from the repository root. The repository intentionally
contains no patient-level metadata or waveforms. Download the two metadata CSVs
from the same 1.0.3 release, then run the committed processing code:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
mkdir -p data/raw/ptb-xl
curl --fail --location --output data/raw/ptb-xl/ptbxl_database.csv https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
curl --fail --location --output data/raw/ptb-xl/scp_statements.csv https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
python -m unittest discover -s tests -v
PYTHONPATH=src python -m ecg_lbbb.process_metadata
PYTHONPATH=src python -m ecg_lbbb.download_records100
PYTHONPATH=src python -m ecg_lbbb.extract_v6
PYTHONPATH=src python -m ecg_lbbb.verify_v6
```

Proceed only after each command succeeds. These curl commands are for a fresh
clone; do not overwrite existing raw metadata from another release. The download
requires internet access and sufficient disk space and may take around 25 minutes.
It creates the ignored pre-download hash snapshot if absent, allowing verification
on a fresh clone. If pinned dependencies are unavailable for your platform,
install `requirements.txt` instead and record the resolved versions.

Expected output is 6,305 normal and 227 complete LBBB recordings. The six arrays
are saved locally in `data/processed/v6/`; shapes and split counts are in
[the extraction report](download-and-extraction.md). No model training is included.
