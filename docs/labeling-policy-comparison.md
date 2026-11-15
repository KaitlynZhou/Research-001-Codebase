# CLBBB labeling-policy comparison

Computed from the supplied local CSVs. This analysis did not change the selected
recordings, labels, original fold assignments, raw files, or existing reports.

## What the current filter excludes

- Total recordings containing CLBBB, at any confidence: **536**
  (430 unique patients).
- Selected CLBBB: **227**; excluded CLBBB: **309**.
- Excluded by the current rule order for confidence below 100: **1**.
- Excluded by the current rule order for additional statements: **308**.
- CLBBB with ILBBB present: **0**.
- CLBBB with NORM present: **0**.

The current rule order checks ILBBB, conflicting/missing targets, confidence,
then additional statements. Independently of that order, 1 CLBBB
recordings have confidence below 100 and 309 have an additional
non-SR statement; 1 have both. These are overlapping
properties, so do not add them as mutually exclusive exclusion counts.

## Exact policies compared

**Conservative (current):** CLBBB confidence exactly 100, only CLBBB plus optional
SR. Any other code excludes the recording, regardless of its confidence.

**Broader (candidate only):** CLBBB confidence exactly 100, no ILBBB and no NORM
code at any confidence; allow other diagnostic, form, and rhythm statements.
The confidence requirement is unchanged. The broader rule treats comorbidities as
compatible with a CLBBB-presence target, not inherently ambiguous. It does not
resolve potential conflicts hidden in free text or clinically adjudicate labels.
Mentor review is needed, particularly for pacing, other conduction disturbances,
and rhythm disorders. Zero confidence on a co-code is never treated as absence.

**Normal class unchanged in both:** NORM confidence 100, only NORM plus optional SR.
NORM is a normal ECG annotation, not proof of a healthy patient.

## Cohort sizes

| Policy | Positive recordings | Positive patients | Normal recordings | Normal patients | All unique patients | Patients with both classes |
| --- | --- | --- | --- | --- | --- | --- |
| Conservative | 227 | 198 | 6305 | 5982 | 6180 | 0 |
| Broader | 535 | 429 | 6305 | 5982 | 6411 | 0 |

Patient counts are deduplicated within each displayed group. The all-patient
column is the union, not the sum of per-class counts. Repeat visits are retained.

## Recommended splits

| Policy | Split | Positive recordings | Positive patients | Normal recordings | Normal patients | All unique patients |
| --- | --- | --- | --- | --- | --- | --- |
| Conservative | train | 183 | 160 | 5095 | 4834 | 4994 |
| Conservative | validation | 19 | 17 | 595 | 576 | 593 |
| Conservative | test | 25 | 21 | 615 | 572 | 593 |
| Broader | train | 427 | 340 | 5095 | 4834 | 5174 |
| Broader | validation | 54 | 47 | 595 | 576 | 623 |
| Broader | test | 54 | 42 | 615 | 572 | 614 |

Folds 1–8 train, 9 validation, 10 test. Both candidate cohorts pass the patient-fold
separation check; there is zero patient overlap between splits. The test-positive
patient count matters as well as the recording count because repeat ECGs from
one patient are correlated.

## Positive counts by fold

| Policy | Fold | Positive recordings | Positive patients | Normal recordings |
| --- | --- | --- | --- | --- |
| Conservative | 1 | 28 | 25 | 613 |
| Conservative | 2 | 21 | 19 | 639 |
| Conservative | 3 | 19 | 15 | 661 |
| Conservative | 4 | 21 | 20 | 621 |
| Conservative | 5 | 19 | 15 | 633 |
| Conservative | 6 | 23 | 22 | 640 |
| Conservative | 7 | 27 | 21 | 656 |
| Conservative | 8 | 25 | 23 | 632 |
| Conservative | 9 | 19 | 17 | 595 |
| Conservative | 10 | 25 | 21 | 615 |
| Broader | 1 | 52 | 41 | 613 |
| Broader | 2 | 54 | 45 | 639 |
| Broader | 3 | 53 | 40 | 661 |
| Broader | 4 | 54 | 43 | 621 |
| Broader | 5 | 53 | 42 | 633 |
| Broader | 6 | 53 | 42 | 640 |
| Broader | 7 | 54 | 41 | 656 |
| Broader | 8 | 54 | 46 | 632 |
| Broader | 9 | 54 | 47 | 595 |
| Broader | 10 | 54 | 42 | 615 |

## Most common additional statements among excluded CLBBB recordings

| Code | Description | Metadata category | Excluded CLBBB recordings |
| --- | --- | --- | --- |
| 1AVB | first degree AV block | diagnostic | 93 |
| AFIB | atrial fibrillation | rhythm | 71 |
| PVC | ventricular premature complex | form | 55 |
| LAO/LAE | left atrial overload/enlargement | diagnostic | 54 |
| STACH | sinus tachycardia | rhythm | 31 |
| ASMI | anteroseptal myocardial infarction | diagnostic | 26 |
| LVH | left ventricular hypertrophy | diagnostic | 24 |
| LPR | prolonged PR interval | form | 16 |
| SARRH | sinus arrhythmia | rhythm | 12 |
| ABQRS | abnormal QRS | form | 12 |
| SBRAD | sinus bradycardia | rhythm | 11 |
| SVARR | supraventricular arrhythmia | rhythm | 9 |
| ISC_ | non-specific ischemic | diagnostic | 8 |
| ILMI | inferolateral myocardial infarction | diagnostic | 7 |
| ANEUR | ST-T changes compatible with ventricular aneurysm | diagnostic | 7 |
| PAC | atrial premature complex | form | 7 |
| LMI | lateral myocardial infarction | diagnostic | 6 |
| IMI | inferior myocardial infarction | diagnostic | 4 |
| ALMI | anterolateral myocardial infarction | diagnostic | 4 |
| NST_ | non-specific ST changes | diagnostic, form | 4 |
| 3AVB | third degree AV block | diagnostic | 3 |
| QWAVE | Q waves present | form | 2 |
| DIG | digitalis-effect | diagnostic, form | 2 |
| BIGU | bigeminal pattern (unknown origin, SV or Ventricular) | rhythm | 2 |
| RAO/RAE | right atrial overload/enlargement | diagnostic | 2 |
| RVH | right ventricular hypertrophy | diagnostic | 1 |
| LPFB | left posterior fascicular block | diagnostic | 1 |
| ISCAL | ischemic in anterolateral leads | diagnostic | 1 |
| AMI | anterior myocardial infarction | diagnostic | 1 |
| SVTAC | supraventricular tachycardia | rhythm | 1 |
| IVCD | non-specific intraventricular conduction disturbance (block) | diagnostic | 1 |
| PACE | normal functioning artificial pacemaker | rhythm | 1 |
| INJAS | subendocardial injury in anteroseptal leads | diagnostic | 1 |
| IPLMI | inferoposterolateral myocardial infarction | diagnostic | 1 |

Counts mean recordings with each code, not mutually exclusive diagnoses. A record
can contribute to several rows. SR is excluded from this table because it is
already allowed; it is not an exclusion trigger. Diagnostic/form/rhythm categories
and descriptions come directly from the supplied statement table. The current
filter excludes all non-SR statements, not just codes flagged diagnostic. Full
row-level evidence is in `results/policy_comparison/clbbb_audit.csv`.

## Trade-offs to discuss with the mentor

The conservative cohort targets isolated, high-confidence CLBBB versus isolated
normal annotations. It minimizes explicitly annotated comorbidities but produces
a smaller and more selected sample. Excluding form statements can remove ECGs
whose morphology is relevant to CLBBB; whether these exclusions are appropriate
requires clinical review. Its results would describe this restricted cohort.

The broader cohort targets high-confidence CLBBB presence, including comorbidities.
It offers more positive patients and test examples, but changes the research
question. With normal-only negatives, other conditions and rhythm differences
could become predictive shortcuts; performance would not show discrimination of
CLBBB from other abnormal ECGs. This risk cannot be quantified from metadata alone.

For discussion: choose the intended target before training; consider a reviewed
allowlist or subgroup analysis rather than accepting every co-code automatically.
Use one prespecified primary cohort and the other as a sensitivity analysis.
Keep patient folds fixed, avoid choosing policy based on test performance, and
report both recording and patient counts. Do not infer waveform validity from
this metadata comparison. No model, extraction, or selection update was run.

## Reproduce and verify

Run from the project root:

```sh
PYTHONPATH=src .venv/bin/python -m ecg_lbbb.compare_policies
```

This writes a separate Markdown report in `docs/` and audit/count CSVs in the
Git-ignored `results/policy_comparison/` folder. Assertions confirm the current
selection matches the conservative policy; all existing processed files and raw
CSVs are checked by SHA-256 before and after analysis. Counts are calculated
from the local files; no external dataset counts are substituted.
