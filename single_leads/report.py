"""Write REPORT.md from the evaluated artifacts. Every number is read from the run's files."""
import argparse, json
from pathlib import Path
import pandas as pd
from common import LEADS, OUTCOMES, INPUTS
from evaluate import run_dir

LABEL = {'dataset1_test': 'Dataset 1 test (LBBB vs normal)', 'dataset2_test': 'Dataset 2 test (LBBB vs all others)'}


def cell(r, metric):
    return f"{r[metric]:.3f} ({r[metric + '_ci_low']:.3f}–{r[metric + '_ci_high']:.3f})"


def report(data, root, audit_path=None):
    m = pd.read_csv(root / 'metrics.csv'); d = pd.read_csv(root / 'median_minus_full.csv')
    audit = json.loads((audit_path or data / 'audit.json').read_text())
    ev = json.loads((root / 'evaluation_provenance.json').read_text())
    sp, ex = audit['splits'], audit['excluded_without_median']
    runs = {(k, l): (json.loads((run_dir(root, k, l) / 'config.json').read_text()),
                     json.loads((run_dir(root, k, l) / 'complete.json').read_text())) for k in INPUTS for l in LEADS}
    cfg = runs['full', 'V1'][0]
    L = ['# Single-lead models: V1, V2, V3, V4 — full 10-second ECG vs median beat', '',
         'Eight models: one per lead (V1, V2, V3, V4) for each input type (full 10-second ECG, PTB-XL+ median beat). '
         'Each model has two independent sigmoid outputs, CLBBB and ILBBB, as on `feature/dataset1-diagnostic-leads`. '
         'All models are trained on Dataset 1 and evaluated with one frozen checkpoint on the fold-10 tests of Dataset 1 and Dataset 2.', '',
         '## Method', '',
         '- **Inputs.** Full ECG: PTB-XL 1.0.3 `records500`, 10 s at 500 Hz (5,000 samples). Median beat: PTB-XL+ 1.0.1 '
         '`median_beats/12sl` (GE 12SL), 1.2 s at 500 Hz (600 samples). One lead per model, physical mV, no filtering, normalization or augmentation. '
         f"Median-beat amplitudes were checked against the raw ECG (median peak-to-peak ratio {audit['median_to_full_amplitude_ratio']['median']:.2f}).",
         '- **Cohorts.** Labels, folds and the Dataset 1 rule are imported unchanged from `diagnostic_leads/prepare.py`. '
         'Recordings without a published PTB-XL+ median beat are dropped from every split for **both** input types, so each full-vs-median pair is compared on identical recordings.',
         f"- **Splits.** Train: Dataset 1 folds 1–8 ({sp['train']['n']:,} recordings; {sp['train']['CLBBB']} CLBBB, {sp['train']['ILBBB']} ILBBB). "
         f"Validation: Dataset 1 fold 9 ({sp['val']['n']:,}; {sp['val']['CLBBB']}, {sp['val']['ILBBB']}). "
         f"Test: fold 10 — Dataset 1 ({sp['dataset1_test']['n']:,}; {sp['dataset1_test']['CLBBB']}, {sp['dataset1_test']['ILBBB']}) and "
         f"Dataset 2 ({sp['dataset2_test']['n']:,}; {sp['dataset2_test']['CLBBB']}, {sp['dataset2_test']['ILBBB']}). Patient-disjoint.",
         f"- **Excluded for lack of a median beat:** train {ex['train']['n']} ({ex['train']['CLBBB']} CLBBB, {ex['train']['ILBBB']} ILBBB), "
         f"validation {ex['val']['n']} ({ex['val']['CLBBB']}, {ex['val']['ILBBB']}), Dataset 2 test {ex['dataset2_test']['n']} "
         f"({ex['dataset2_test']['CLBBB']}, {ex['dataset2_test']['ILBBB']}).",
         f"- **Model and training.** The six-lead branch's three-block 1D ResNet with a 1-channel stem ({cfg['parameters']:,} parameters). "
         f"Adam lr {cfg['learning_rate']}, batch {cfg['batch_size']}, seed {cfg['seed']}, at most {cfg['max_epochs']} epochs, early-stopping patience {cfg['patience']}, "
         'training-derived positive-class weights, checkpoint at minimum validation weighted BCE. Device: ' + cfg['device'] + '.',
         f"- **Metrics.** AUROC and AUPRC (average precision) with record-level bootstrap 95% CIs ({ev['bootstrap_resamples']:,} resamples). "
         'Full-vs-median differences use paired bootstrap on the same resampled recordings. No threshold tuning on test data.', '']
    for pop in ['dataset1_test', 'dataset2_test']:
        L += [f'## {LABEL[pop]}', '']
        for outcome in OUTCOMES:
            s = m[(m.population == pop) & (m.outcome == outcome)]
            n, pos = int(s.n.iloc[0]), int(s.positives.iloc[0])
            L += [f'**{outcome}** — {n:,} recordings, {pos} positives (prevalence {pos / n:.1%}).', '',
                  '| Lead | AUROC, full ECG | AUROC, median beat | AUPRC, full ECG | AUPRC, median beat |', '|---|---:|---:|---:|---:|']
            for lead in LEADS:
                f = s[(s.input == 'full') & (s.lead == lead)].iloc[0]; md = s[(s.input == 'median') & (s.lead == lead)].iloc[0]
                L.append(f'| {lead} | {cell(f, "auroc")} | {cell(md, "auroc")} | {cell(f, "auprc")} | {cell(md, "auprc")} |')
            L.append('')
    L += ['## Median beat minus full ECG (paired bootstrap 95% CI)', '',
          '| Test population | Outcome | Lead | ΔAUROC | ΔAUPRC |', '|---|---|---|---:|---:|']
    for r in d.itertuples():
        L.append(f'| {r.population} | {r.outcome} | {r.lead} | {r.delta_auroc:+.3f} ({r.delta_auroc_ci_low:+.3f} to {r.delta_auroc_ci_high:+.3f}) | '
                 f'{r.delta_auprc:+.3f} ({r.delta_auprc_ci_low:+.3f} to {r.delta_auprc_ci_high:+.3f}) |')
    L += ['', '## Training runs', '', '| Input | Lead | Epochs run | Selected epoch | Validation weighted BCE | Minutes |', '|---|---|---:|---:|---:|---:|']
    for (k, l), (_, c) in runs.items():
        L.append(f"| {k} | {l} | {c['epochs_run']} | {c['best_epoch']} | {c['best_validation_loss']:.4f} | {c['seconds'] / 60:.1f} |")
    capped = [f'{k} {l}' for (k, l), (cf, c) in runs.items() if c['epochs_run'] == cf['max_epochs'] and c['best_epoch'] >= cf['max_epochs'] - 1]
    L += ['', '## Limitations', '']
    if capped:
        L.append(f"- {', '.join(capped)} selected an epoch at the {cfg['max_epochs']}-epoch cap, so validation loss was still falling; "
                 'a longer run might score differently. The cap was kept at the six-lead branch\'s value for comparability.')
    L += [
          '- Dataset 1 test is a subset of Dataset 2 test (same PTB-XL fold 10), so the two test results are correlated and are not external validation. Their positives are identical.',
          f"- ILBBB is exploratory: {sp['dataset1_test']['ILBBB']} test positives. Its AUPRC confidence intervals are wide; treat lead or input-type differences for ILBBB with caution.",
          '- V1 and V2 belong to the conventional diagnostic lead set (I, aVL, V1, V2, V5, V6); V3 and V4 do not. The four leads were trained as requested.',
          '- Median beats are GE 12SL outputs; recordings where 12SL produced none are excluded from both input types. '
          f"Exclusions are not random: {ex['train']['CLBBB']} of {ex['train']['n']} excluded training recordings and "
          f"{ex['dataset2_test']['CLBBB']} of {ex['dataset2_test']['n']} excluded test recordings are CLBBB, so "
          f"{ex['dataset2_test']['CLBBB']} of {sp['dataset2_test']['CLBBB'] + ex['dataset2_test']['CLBBB']} CLBBB test positives are missing. "
          'Results describe recordings for which 12SL produced a median beat, and the CLBBB test set differs from the six-lead branch\'s, so numbers are not directly comparable with it.',
          '- Each configuration is a single seed-42 run; seed-to-seed variation is not estimated. Sigmoid outputs from weighted BCE are scores, not calibrated probabilities.',
          '- Bootstrap CIs resample recordings, not patients; a few patients have several fold-10 recordings.', '',
          '## Artifacts', '',
          'In `results/`: `metrics.csv` (all metrics, including thresholded counts at 0.50), `median_minus_full.csv`, `evaluation_provenance.json` (checkpoint hashes), '
          'and per model `config.json`, `history.json`, `complete.json`. `data_audit.json` holds cohort counts, exclusions and the unit check. '
          'Checkpoints, record-level predictions and ECG arrays stay local.', '']
    (root.parent / 'REPORT.md').write_text('\n'.join(L))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--results', type=Path, required=True)
    p.add_argument('--audit', type=Path)
    a = p.parse_args(); report(a.data, a.results, a.audit)
