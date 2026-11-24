"""Train the eight single-lead models (2 inputs x 4 leads), then evaluate and write the report.

Resumable: a model whose directory already has complete.json is skipped. A directory with a
checkpoint but no complete.json is an interrupted run and must be removed by hand.
"""
import argparse
from pathlib import Path
from common import LEADS, INPUTS
from train import train
from evaluate import evaluate, run_dir
from report import report

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True); p.add_argument('--results', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    a = p.parse_args()
    for kind in INPUTS:
        for lead in LEADS:
            d = run_dir(a.results, kind, lead)
            if (d / 'complete.json').exists():
                print('skip', d.name, '(complete)'); continue
            train(a.data, d, kind, lead, a.epochs)
    evaluate(a.data, a.results)
    report(a.data, a.results)
