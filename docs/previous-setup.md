# PTB-XL complete LBBB research setup

The original root README is preserved. This guide documents the staged project.
Stage 1 implements project setup and metadata inspection. Signal processing,
cohort assignment, the ResNet, training, and evaluation are planned stages below;
they are not implemented yet. No dataset has been downloaded.

## Install and use in VS Code

A Python 3.11 virtual environment has been created in `.venv`. Open this repository
in VS Code, install Microsoft's Python extension if needed, and use **Python:
Select Interpreter** to select `.venv/bin/python`. Open a new terminal:

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

Dependency installation is a separate step; the environment currently contains
only the virtual environment's bootstrap tools. The metadata inspector and its
tests use the Python standard library and work before installing dependencies.
The requirements have bounded version ranges, not exact pins. After resolving
and testing dependencies, save `python -m pip freeze` in a new versioned lockfile
for the experiment. Use CPU first; choose GPU installation for the target machine
when training is ready.

## Files and directories

| Path | Purpose |
| --- | --- |
| `requirements.txt` | PyTorch, NumPy, Pandas, WFDB, scikit-learn, Matplotlib |
| `.gitignore` | Excludes environments, data, results, checkpoints, caches |
| `.vscode/settings.json` | Default interpreter, source lookup, unittest discovery |
| `src/ecg_lbbb/__init__.py` | Research package |
| `src/ecg_lbbb/inspect_metadata.py` | Validates CSVs and prints diagnostic/fold audit as JSON |
| `tests/test_metadata.py` | Synthetic tests for counts, conflicts, invalid labels, patient leakage |
| `data/raw/` | Local PTB-XL metadata and original WFDB signals |
| `data/processed/` | Future cohort manifests and signal arrays |
| `models/` | Future trained checkpoints |
| `results/` | Local inspection reports and future metrics/plots |
| `.gitkeep` files | Keep empty directories in Git |

## Obtain metadata first

Use [PTB-XL version 1.0.3 on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
Review its license and citations. Download only `ptbxl_database.csv` and
`scp_statements.csv` initially, from the **same version**, into:

```text
data/raw/ptb-xl/ptbxl_database.csv
data/raw/ptb-xl/scp_statements.csv
```

The program does not fetch anything. Run from the repository root:

```sh
PYTHONPATH=src python -m ecg_lbbb.inspect_metadata --data-root data/raw/ptb-xl
```

It validates statement definitions, dictionary syntax, likelihood ranges,
unknown codes, ECG identifiers, patient identifiers, and fold assignments.
It reports all diagnostic labels and descriptions, all code counts, target
likelihood distributions, CLBBB co-occurrences, conflicting code presence, and
record counts by fold. It fails on patients appearing in multiple folds.
No report is written automatically, avoiding accidental replacement of files.
No cohort labels or claims of clinical health are inferred from this report.

## Verify before stage 2: cohort and signal processing

1. Confirm local definitions: `CLBBB` means complete left bundle branch block,
   `ILBBB` means incomplete left bundle branch block, `NORM` means normal ECG.
2. Review target likelihood distributions and conflicts with your supervisor.
   SCP likelihood values are annotation confidence, not calibrated probabilities;
   zero must not silently be interpreted as absence.
3. Agree a confidence threshold (proposed conservative starting point: 100).
   Proposed positive: confident CLBBB, no ILBBB or NORM code present. Decide
   whether other co-occurring pathologies are acceptable in positives.
4. Proposed negative: confident NORM and no other diagnostic statement present,
   regardless of other statements' confidence. Review rhythm/form statements too:
   normal diagnostic labeling does not establish that a person is healthy.
5. Review exclusions for uncertainty, incompatible statements, pacemakers and
   signal quality. Keep a reason for each excluded record; compare counts and
   patient counts per class and split. These policies are proposals, not applied labels.
6. Verify each patient belongs to one official fold, including excluded records.

Stage 2 should use folds 1–8 for training, 9 for validation, 10 for test,
following the [dataset's recommended split](https://physionet.org/content/ptb-xl/1.0.3/).
Never split records randomly across patients. Check disjoint patient sets again
after cohort selection and require both classes in every split.

After the metadata review, obtain `records100/` (100 Hz) for the first baseline;
preserve the original relative directory structure and `.hea`/`.dat` pairs.
Avoid downloading `records500/` unless the experiment needs 500 Hz.
The future processor will read `filename_lr` with WFDB, check sample rate,
finite values, 10-second duration, lead names and units, and produce float32
arrays `[records, channels, samples]`: initially `[N, 12, 1000]` at 100 Hz.
Select leads by header name later, preserving the chosen order. Fit channel
normalization on training records only and save those statistics for all splits.
Retain ECG IDs, patient IDs, folds, labels, rate, leads, version and policy in manifests.

## Stage 3: model and training (pending)

Implement a small 1D ResNet with residual Conv1d blocks, projection shortcuts,
global average pooling and one binary logit. Accept the lead count as an input.
Test tensor shapes for 12 leads and a single lead. Use BCEWithLogitsLoss, Adam,
fixed seeds and explicit device selection. Select checkpoints by validation
metrics only; store configuration, preprocessing, code version and random seed.
Decide class weighting using training labels only.

Training and evaluation commands will be added when those scripts exist.
Do not start training until cohort counts, exclusions, leakage checks and signal
shape tests have passed. Start with a small synthetic forward/backward smoke test.

## Stage 4: evaluation (pending)

Compute accuracy, precision, recall, F1, AUROC and PR-AUC, plus a confusion
matrix and class counts. Define PR-AUC explicitly (proposed: average precision).
Use sigmoid probabilities for AUROC/PR metrics, class 1 = CLBBB, and record the
classification threshold. Choose thresholds on validation only; evaluate the
test set once after decisions are fixed. Report undefined metrics explicitly
for single-class inputs. Repeated records from a patient are correlated; any
confidence intervals should resample patients. This is a research baseline.

## Preservation and scope

Existing files were preserved. No commits or pushes are performed by setup.
Data, generated results and checkpoints stay local. Inspect `git status --short`
before committing. Nothing here starts a download or a training job.
