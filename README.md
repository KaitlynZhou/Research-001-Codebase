# Dataset 1 baseline with six diagnostic ECG leads

This branch contains the experiment using **I, aVL, V1, V2, V5, and V6**, trained on Dataset 1 with separate **CLBBB** and **ILBBB** outputs.

- [Experiment summary](results/experiment1/experiment_summary.md)
- [Full report and limitations](diagnostic_leads/REPORT.md)
- [Setup and reproduction instructions](diagnostic_leads/README.md)
- [Source code and inspection notebook](diagnostic_leads/)
- [Six-lead validation and test metrics](diagnostic_leads/results/metrics.csv)

Training followed Prisha's baseline settings with six input channels instead of twelve. The model was evaluated on Dataset 1 and Dataset 2 test populations using the same selected checkpoint.

This branch was based on `feature/dataset1-12lead-clean`. The files under `learning_pipeline/` and the neighboring CSVs under `results/experiment1/` are inherited 12-lead reference artifacts. The updated experiment summary links to the six-lead results, which are stored under `diagnostic_leads/`. The [original 12-lead report](https://github.com/KaitlynZhou/Research-001-Codebase/blob/4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e/results/experiment1/experiment_summary.md) remains available at its pinned reference commit.
