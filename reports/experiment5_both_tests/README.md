# Experiment 5: both held-out test evaluations

Training dataset: dataset1; native 500 Hz, six ordered channels,
5,000 samples, independent CLBBB/ILBBB outputs and threshold 0.5.
Selected epoch: 3; completed training epochs: 10.

`metrics.json` reports per-label accuracy, precision, recall, F1 and confusion
matrices for both test cohorts. `leakage_checks.json` verifies zero patient overlap
with training/validation and matching same-fold cross-dataset records.
`provenance.json` records export contracts, class counts and fingerprints.
`summary.json` records checkpoint identity, training settings and verification.
The final model checkpoints are included under `outputs/`; per-record scores
remain Git-ignored. Dataset 1 test
is contained in Dataset 2 test; these are correlated internal held-out results.
See [the README](../../README.md) and [all results](../assignment_results.json).
