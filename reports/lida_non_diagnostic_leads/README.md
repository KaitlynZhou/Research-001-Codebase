# Experiments 5 and 6: native 500 Hz aggregate review bundle

See [Lida's report](../../docs/lida_non_diagnostic_leads.md) for methods,
architecture, split counts, reproduction commands and limitations.

Experiment 5 was freshly retrained from original PTB-XL `records500` waveforms
using `filename_hr`: **500 Hz, 5,000 samples, six leads** in order
II, III, aVR, aVF, V3, V4. It completed 10 epochs on MPS and restored
the best validation checkpoint at epoch 3. The architecture,
labels and patient-separated official folds are unchanged. No old 100 Hz weights
or metrics were reused as 500 Hz results.

`metrics.json` contains per-label Dataset 1 test accuracy, precision, recall,
F1 and confusion matrices at threshold 0.5. `history.json` and
`training_config.json` document selection and configuration. `provenance.json`
records counts, contracts and export fingerprints. `raw_verification.json`
records independent all-row exact comparisons with native WFDB samples.
`summary.json` records checks, source/weight hashes and pending tasks.

Weights and patient-level predictions remain outside Git at
`outputs/lida/experiment5_500hz_mps/`. Historical 100 Hz aggregates are preserved
separately in `reports/lida_non_diagnostic_leads_100hz_historical/`; their model
is still in `outputs/lida/experiment5_mps/` and is superseded.

Experiment 6 and Dataset 2 test evaluation remain pending until the team's
Dataset 2 policy and exports are available. The pipeline requires the same
500 Hz input contract for Dataset 2. These scores are internal held-out metrics;
the source fold 10 has historical development exposure, and external validation
has not been performed. ILBBB has only eight test positives; consult its false
positive count and precision as well as accuracy and recall.
