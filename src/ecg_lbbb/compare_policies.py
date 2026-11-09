"""Read-only cohort comparison; writes separate reports, never changes selection."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import pandas as pd
from .inspect_metadata import inspect, parse_codes
from .process_metadata import classify, verify_splits


def table(df):
    return '\n'.join(['| ' + ' | '.join(map(str, df.columns)) + ' |',
                       '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in df.itertuples(index=False, name=None)])


def main():
    root = Path('data/raw/ptb-xl')
    protected = list(root.glob('*.csv')) + list(Path('data/processed').glob('*.*'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    inspect(root)
    df = pd.read_csv(root / 'ptbxl_database.csv')
    scp = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    verify_splits(df)
    codes = df.scp_codes.map(parse_codes)
    decisions = codes.map(classify)
    cl = codes.map(lambda c: 'CLBBB' in c)
    conservative = decisions.map(lambda d: d[0] == 1)
    negatives = decisions.map(lambda d: d[0] == 0)
    # Ambiguity here means target confidence <100 or conflicting target codes.
    broader = codes.map(lambda c: c.get('CLBBB') == 100 and not ({'ILBBB', 'NORM'} & c.keys()))
    existing = pd.read_csv('data/processed/selected_recordings.csv')
    assert set(existing.ecg_id) == set(df.loc[conservative | negatives, 'ecg_id'])
    assert set(existing.loc[existing.label == 1, 'ecg_id']) == set(df.loc[conservative, 'ecg_id'])
    excluded = cl & ~conservative
    reason_counts = Counter(decisions.loc[excluded].map(lambda d: d[1]))
    low = cl & codes.map(lambda c: c.get('CLBBB', 100) < 100)
    additional = cl & codes.map(lambda c: bool(set(c) - {'CLBBB', 'SR'}))
    counts = Counter()
    for c in codes.loc[excluded]:
        counts.update(set(c) - {'CLBBB', 'SR'})
    code_rows = []
    for code, n in counts.most_common():
        row = scp.loc[code]
        categories = ', '.join(flag for flag in ['diagnostic', 'form', 'rhythm'] if row[flag] == 1)
        code_rows.append({'Code': code, 'Description': row.description,
                          'Metadata category': categories, 'Excluded CLBBB recordings': n})
    code_df = pd.DataFrame(code_rows)
    summaries, splits, folds = [], [], []
    for policy, positive in [('Conservative', conservative), ('Broader', broader)]:
        cohort = df.loc[positive | negatives]
        verify_splits(cohort)
        summaries.append({'Policy': policy, 'Positive recordings': int(positive.sum()),
                          'Positive patients': df.loc[positive, 'patient_id'].nunique(),
                          'Normal recordings': int(negatives.sum()),
                          'Normal patients': df.loc[negatives, 'patient_id'].nunique(),
                          'All unique patients': cohort.patient_id.nunique(),
                          'Patients with both classes': len(set(df.loc[positive, 'patient_id']) & set(df.loc[negatives, 'patient_id']))})
        for split, fs in [('train', range(1, 9)), ('validation', [9]), ('test', [10])]:
            mask = df.strat_fold.isin(fs)
            splits.append({'Policy': policy, 'Split': split,
                           'Positive recordings': int((positive & mask).sum()),
                           'Positive patients': df.loc[positive & mask, 'patient_id'].nunique(),
                           'Normal recordings': int((negatives & mask).sum()),
                           'Normal patients': df.loc[negatives & mask, 'patient_id'].nunique(),
                           'All unique patients': df.loc[(positive | negatives) & mask, 'patient_id'].nunique()})
        for f in range(1, 11):
            mask = df.strat_fold.eq(f)
            folds.append({'Policy': policy, 'Fold': f, 'Positive recordings': int((positive & mask).sum()),
                          'Positive patients': df.loc[positive & mask, 'patient_id'].nunique(),
                          'Normal recordings': int((negatives & mask).sum())})
    summary, split_df, fold_df = map(pd.DataFrame, [summaries, splits, folds])
    out = Path('results/policy_comparison')
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('policy_counts', summary), ('split_counts', split_df), ('fold_counts', fold_df), ('additional_codes', code_df)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    audit = df.loc[cl, ['ecg_id', 'patient_id', 'strat_fold', 'scp_codes']].copy()
    audit['current_reason'] = decisions.loc[cl].map(lambda d: d[1])
    audit['broader_positive'] = broader.loc[cl]
    audit.to_csv(out / 'clbbb_audit.csv', index=False)
    text = f'''# CLBBB labeling-policy comparison

Computed from the supplied local CSVs. This analysis did not change the selected
recordings, labels, original fold assignments, raw files, or existing reports.

## What the current filter excludes

- Total recordings containing CLBBB, at any confidence: **{int(cl.sum())}**
  ({df.loc[cl, 'patient_id'].nunique()} unique patients).
- Selected CLBBB: **{int(conservative.sum())}**; excluded CLBBB: **{int(excluded.sum())}**.
- Excluded by the current rule order for confidence below 100: **{reason_counts['uncertain_target']}**.
- Excluded by the current rule order for additional statements: **{reason_counts['additional_statement']}**.
- CLBBB with ILBBB present: **{int((cl & codes.map(lambda c: 'ILBBB' in c)).sum())}**.
- CLBBB with NORM present: **{int((cl & codes.map(lambda c: 'NORM' in c)).sum())}**.

The current rule order checks ILBBB, conflicting/missing targets, confidence,
then additional statements. Independently of that order, {int(low.sum())} CLBBB
recordings have confidence below 100 and {int(additional.sum())} have an additional
non-SR statement; {int((low & additional).sum())} have both. These are overlapping
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

{table(summary)}

Patient counts are deduplicated within each displayed group. The all-patient
column is the union, not the sum of per-class counts. Repeat visits are retained.

## Recommended splits

{table(split_df)}

Folds 1–8 train, 9 validation, 10 test. Both candidate cohorts pass the patient-fold
separation check; there is zero patient overlap between splits. The test-positive
patient count matters as well as the recording count because repeat ECGs from
one patient are correlated.

## Positive counts by fold

{table(fold_df)}

## Most common additional statements among excluded CLBBB recordings

{table(code_df)}

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
'''
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    (Path('docs') / 'labeling-policy-comparison.md').write_text(text)
    (out / 'source_and_selection_hashes.json').write_text(json.dumps(before, indent=2) + '\n')
    print(summary.to_string(index=False))
    print(split_df.to_string(index=False))
    print('Reasons:', dict(reason_counts), 'low:', int(low.sum()), 'additional:', int(additional.sum()), 'both:', int((low & additional).sum()))
    print(code_df.head(12).to_string(index=False))
    print('Existing raw CSVs and processed outputs unchanged (SHA-256 verified).')


if __name__ == '__main__':
    main()
