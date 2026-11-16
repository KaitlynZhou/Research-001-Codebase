# Historical 100 Hz results — superseded

This entire bundle describes the previous **100 Hz, 1,000-sample** run.
It is preserved for provenance and must never be reported as 500 Hz results.
Its source hashes refer to the previous runner version.

# Experiments 5 and 6: aggregate review bundle

See [Lida's report](../../docs/lida_non_diagnostic_leads.md) for methods,
architecture, counts, reproduction commands and limitations.

Experiment 5 completed 12 epochs on MPS; the best validation checkpoint was
at epoch 5. `metrics.json` contains separate CLBBB and ILBBB results on Dataset 1
at threshold 0.5, including confusion matrices. ILBBB precision was only 5.74%
(115 false positives); this is an important performance limitation.

`history.json` and `training_config.json` document selection and configuration.
`provenance.json` contains split counts, export contracts and file hashes.
`summary.json` records checkpoint/source hashes, final checks and pending tasks.
The fold-10 source cohort has historical development exposure; results are
internal held-out evaluation, not untouched external validation.

Dataset 2 evaluation and Experiment 6 remain pending because no approved team
Dataset 2 implementation/export is available. No results are implied for them.

Weights and patient-level data/predictions are excluded from this bundle and
remain in ignored local directories. The final checkpoint is
`outputs/lida/experiment5_mps/best_model.pt`. An interrupted CPU attempt in
`outputs/lida/experiment5/` is preserved and is not used for reported results.
