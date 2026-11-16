# Experiment 5: six-lead 500 Hz results

The completed run uses II, III, aVR, aVF, V3, V4 from native PTB-XL records500
via filename_hr, at 500 Hz with 5,000 samples. Training completed 10 epochs;
the best validation checkpoint was epoch 3, loss 0.4496812247.

| Label | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLBBB | 97.54% | 70.27% | 96.30% | 81.25% | 898 | 22 | 2 | 52 |
| ILBBB | 86.55% | 5.11% | 87.50% | 9.66% | 836 | 130 | 1 | 7 |

Threshold is 0.5. Confusion matrices use true rows / predicted columns,
negative then positive. Metrics are recording-level; ILBBB has eight test
positives. The source test fold has prior development exposure, so these are
internal held-out results, not external validation.

- `metrics.json`: exact Dataset 1 per-label test results.
- `history.json` and `training_config.json`: training and checkpoint selection.
- `provenance.json`: contracts, split counts and export fingerprints.
- `raw_verification.json`: exact comparisons against all 9,682 native waveforms.
- `summary.json`: source/checkpoint hashes, preservation checks and pending work.
- `cleanup_verification.json`: 11 passing tests, unchanged tensors/arrays, and exact
  full-test probability reproduction with optimizer steps disabled.

The local checkpoint and predictions are in `outputs/experiment5_500hz/` and are
Git-ignored. Cleanup changed paths and embedded provenance only: model tensors,
input arrays, labels, saved predictions and numerical results are unchanged.
No model was retrained. Dataset 2 evaluation and Experiment 6 remain pending.
See [the project README](../../README.md) and
[contract/model details](../../docs/non_diagnostic_leads.md).
