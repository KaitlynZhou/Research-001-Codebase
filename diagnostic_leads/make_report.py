"""Write a compact report from the completed experiment artifacts.

All counts are read from the run's artifacts (metrics.csv, config.json, complete.json,
evaluation_provenance.json and the preparation audit), so the report stays in step with
whatever run is passed via --out.
"""
import argparse,json
from pathlib import Path
import pandas as pd

# The 12-lead reference differs only in the stem convolution: 32 filters x kernel 7 per input lead.
STEM_FILTERS,STEM_KERNEL,REFERENCE_LEADS=32,7,12

def fmt(value):
    return '' if pd.isna(value) else f'{value:.4f}'

def report(out,audit_path=None):
    metrics=pd.read_csv(out/'metrics.csv');config=json.loads((out/'config.json').read_text())
    done=json.loads((out/'complete.json').read_text());ev=json.loads((out/'evaluation_provenance.json').read_text())
    audit_path=audit_path or out.parent/'data_audit.json'
    splits=json.loads(audit_path.read_text())['splits']
    tr,va=splits['train'],splits['val']
    get=lambda pop,outcome:metrics[metrics.population.eq(pop)&metrics.outcome.eq(outcome)].iloc[0]
    d1,d2=get('dataset1_test','CLBBB'),get('dataset2_test','CLBBB')
    il=get('dataset1_test','ILBBB')
    extra=int(d2.n-d1.n)
    fewer=(REFERENCE_LEADS-len(config['leads']))*STEM_FILTERS*STEM_KERNEL
    selected=pd.concat([metrics[metrics.population.eq(name)] for name in ['dataset1_test','dataset2_test']])
    lines=['# Six diagnostic leads: Dataset 1 baseline','',
      'Training and evaluation were performed locally on PTB-XL 1.0.3. Source code and aggregate results are published on `feature/dataset1-diagnostic-leads`.','',
      '## Method','',
      f"Reference: Prisha’s `feature/dataset1-12lead-clean` branch at commit `{config['reference_commit']}`. Leads: {', '.join(config['leads'])}. Separate CLBBB and ILBBB outputs; any recording containing a target code is positive for that output, including recordings with additional diagnoses. CLBBB is the primary outcome; ILBBB is secondary.",'',
      f"Same three-block 1D ResNet and two-output weighted BCE as Prisha; change input channels from {REFERENCE_LEADS} to {len(config['leads'])}. Adam LR {config['learning_rate']}, batch {config['batch_size']}, seed {config['seed']}, maximum {config['max_epochs']} epochs, early stopping patience {config['patience']}, no waveform normalization or augmentation. Training class weights: {config['pos_weight']}.",'',
      f"Train: Dataset 1 folds 1–8 ({tr['n']:,} recordings, {tr['CLBBB']} CLBBB / {tr['ILBBB']} ILBBB). Validation: Dataset 1 fold 9 ({va['n']:,} recordings, {va['CLBBB']} / {va['ILBBB']}). Test: fold 10 only. Patient disjointness and exact waveform duplicate checks passed.",'',
      f"Completed {done['epochs_run']} epochs in {done['seconds']/60:.1f} minutes; selected epoch {ev['checkpoint_epoch']} by minimum validation weighted BCE ({done['best_validation_loss']:.6f}). Parameters: {config['parameters']:,}. Device: {config['device']}.",'',
      '## Held-out results','',
      'One frozen checkpoint, fixed threshold 0.50 for both outcomes. No test-set threshold tuning.','',
      '| Test population | Outcome | N | Positives | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected.itertuples():
      cells=[r.population,r.outcome,str(r.n),str(r.positives)]+[fmt(getattr(r,c)) for c in ['roc_auc','average_precision','precision','sensitivity','specificity','f1']]+[str(getattr(r,c)) for c in ['TN','FP','FN','TP']]
      lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Interpretation and limitations','',
      f"Dataset 1 test ({d1.n:,} recordings) is a subset of Dataset 2 test ({d2.n:,} recordings). These are overlapping cohorts from the same dataset, not independent external validation. All {d1.positives} CLBBB and {il.positives} ILBBB test positives occur in both cohorts; therefore sensitivity is identical. The {extra:,} additional test recordings have other diagnoses and no target labels. Their false-positive counts are reported separately in metrics.csv.",'',
      f"ILBBB is exploratory: only {tr['ILBBB']} training, {va['ILBBB']} validation, and {il.positives} test positives. Each missed test positive changes sensitivity by {100/max(il.positives,1):.1f} percentage points. ROC-AUC alone is insufficient for this rare outcome; inspect average precision, prevalence, and false positives. Zero precision/F1 when there are no predicted positives follows the stated zero_division=0 convention. AUROC, average precision and sensitivity are undefined for the additional-diagnoses subset because it contains no positives.",'',
      'Sigmoid outputs from class-weighted BCE are model scores, not calibrated clinical probabilities. This diagnostic experiment does not estimate sudden cardiac death risk or establish clinical utility. No interpretability/VAE experiments were run.','',
      f"The six-lead network has {fewer:,} fewer parameters than the original model. Training hyperparameters were held fixed to isolate the requested lead change. Device/version differences can prevent bitwise reproduction of Prisha’s run. A single run cannot establish feature stability across seeds.",'',
      '## Local artifacts','',
      'See README.md for execution commands and Explore_baseline.ipynb for local inspection. The repository’s results/ contains configuration, epoch history and aggregate metrics. Checkpoints, record-level predictions, ECG arrays and generated plots remain local. provenance.json records GitHub references. waveform_audit.json records duplicate checks. data_audit.json records cohort counts and metadata checksum. Raw/processed ECG arrays are retained separately in the workspace and omitted from the portable package.']
    (out.parent/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit',type=Path,help='Preparation audit.json (default: data_audit.json beside the results directory)')
    a=p.parse_args();report(a.out,a.audit)
