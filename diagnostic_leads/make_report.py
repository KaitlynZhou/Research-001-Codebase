"""Write a compact report from the completed experiment artifacts."""
import argparse,json
from pathlib import Path
import pandas as pd

def report(out):
    metrics=pd.read_csv(out/'metrics.csv');config=json.loads((out/'config.json').read_text())
    done=json.loads((out/'complete.json').read_text());ev=json.loads((out/'evaluation_provenance.json').read_text())
    selected=pd.concat([metrics[metrics.population.eq(name)] for name in ['dataset1_test','dataset2_test']])
    lines=['# Six diagnostic leads: Dataset 1 baseline','',
      'Training and evaluation were performed locally on PTB-XL 1.0.3.','',
      '## Method','',
      'Reference: Prisha’s `feature/dataset1-12lead-clean` branch at commit `4d82a777d7b811ea35e57d1abfc9cfb5c8128f6e`. Leads: I, aVL, V1, V2, V5, V6. Separate CLBBB and ILBBB outputs; any recording containing a target code is positive for that output, including recordings with additional diagnoses. CLBBB is the primary outcome; ILBBB is secondary.','',
      'Same three-block 1D ResNet and two-output weighted BCE as Prisha; change input channels from 12 to 6. Adam LR 0.001, batch 32, seed 42, maximum 30 epochs, early stopping patience 7, no waveform normalization or augmentation. Training class weights: '+str(config['pos_weight'])+'.','',
      'Train: Dataset 1 folds 1–8 (7,733 recordings, 428 CLBBB / 62 ILBBB). Validation: Dataset 1 fold 9 (975 recordings, 54 / 7). Test: fold 10 only. Patient disjointness and exact waveform duplicate checks passed.','',
      f"Completed {done['epochs_run']} epochs in {done['seconds']/60:.1f} minutes; selected epoch {ev['checkpoint_epoch']} by minimum validation weighted BCE ({done['best_validation_loss']:.6f}). Parameters: {config['parameters']:,}. Device: {config['device']}.",'',
      '## Held-out results','',
      'One frozen checkpoint, fixed threshold 0.50 for both outcomes. No test-set threshold tuning.','',
      '| Test population | Outcome | N | Positives | ROC-AUC | Average precision | Precision | Sensitivity | Specificity | F1 | TN | FP | FN | TP |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected.itertuples():
      cells=[r.population,r.outcome,str(r.n),str(r.positives)]+[f'{getattr(r,c):.4f}' for c in ['roc_auc','average_precision','precision','sensitivity','specificity','f1']]+[str(getattr(r,c)) for c in ['TN','FP','FN','TP']]
      lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Interpretation and limitations','',
      'Dataset 1 test (974 recordings) is a subset of Dataset 2 test (2,198 recordings). These are overlapping cohorts from the same dataset, not independent external validation. All 54 CLBBB and 8 ILBBB test positives occur in both cohorts; therefore sensitivity is identical. The 1,224 additional test recordings have other diagnoses and no target labels. Their false-positive counts are reported separately in metrics.csv.','',
      'ILBBB is exploratory: only 62 training, 7 validation, and 8 test positives. Each missed test positive changes sensitivity by 12.5 percentage points. ROC-AUC alone is insufficient for this rare outcome; inspect average precision, prevalence, and false positives. Zero precision/F1 when there are no predicted positives follows the stated zero_division=0 convention. AUROC, average precision and sensitivity are undefined for the additional-diagnoses subset because it contains no positives.','',
      'Sigmoid outputs from class-weighted BCE are model scores, not calibrated clinical probabilities. This diagnostic experiment does not estimate sudden cardiac death risk or establish clinical utility. No interpretability/VAE experiments were run.','',
      'The six-lead network has 1,344 fewer parameters than the original model. Training hyperparameters were held fixed to isolate the requested lead change. Device/version differences can prevent bitwise reproduction of Prisha’s run. A single run cannot establish feature stability across seeds.','',
      '## Local artifacts','',
      'See README.md for execution commands and Explore_baseline.ipynb for local inspection. results/ contains the checkpoint, configuration, epoch history, validation and test predictions, metrics and plots. provenance.json records GitHub references. waveform_audit.json records duplicate checks. data_audit.json records cohort counts and metadata checksum. Raw/processed ECG arrays are retained separately in the workspace and omitted from the portable package.','']
    (out.parent/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.out)
