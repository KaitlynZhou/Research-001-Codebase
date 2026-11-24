# Research-001-Codebase

## Experiment 3 — Six-lead ResNet trained on Dataset 1

Experiment 3 evaluated a six-lead ECG model trained on left bundle branch block cases and normal controls. It used leads I, aVL, V1, V2, V5, and V6 at 100 Hz, with separate outputs for complete (CLBBB) and incomplete (ILBBB) left bundle branch block.

The same checkpoint was evaluated at a fixed 0.50 threshold on both held-out test cohorts. Dataset 2 contained the entire Dataset 1 test cohort plus 1,224 ECGs with other diagnoses and neither target label. [Methods and cohort definitions](https://github.com/KaitlynZhou/Research-001-Codebase/blob/b919cba356b20aea1aacb3635e1e176c3526a237/diagnostic_leads/README.md)

| Outcome | Test cohort | Cases detected | False positives | Precision |
|---|---|---:|---:|---:|
| CLBBB | Dataset 1 (974 ECGs) | 52 of 54 | 8 | 86.7% |
| CLBBB | Dataset 2 (2,198 ECGs) | 52 of 54 | 215 | 19.5% |
| ILBBB | Dataset 1 (974 ECGs) | 7 of 8 | 79 | 8.1% |
| ILBBB | Dataset 2 (2,198 ECGs) | 7 of 8 | 460 | 1.5% |

The model detected the same 52 CLBBB cases in both cohorts, but the additional ECGs increased false positives and reduced precision. ILBBB results are exploratory given the eight positive test cases and low precision in both cohorts. [Full held-out results](https://github.com/KaitlynZhou/Research-001-Codebase/blob/b919cba356b20aea1aacb3635e1e176c3526a237/diagnostic_leads/REPORT.md)

Experiment 4 kept the same six leads and changed both the training cohort (Dataset 2) and sampling rate (500 Hz). [Experiment 4 report](results/experiment4/experiment4_report.md)
