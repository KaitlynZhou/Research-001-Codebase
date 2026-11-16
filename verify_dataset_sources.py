"""Verify prepared cohorts against pinned team builders without running their writers."""
from pathlib import Path
import argparse
import ast
import contextlib
import io
import json
import subprocess

import numpy as np
import pandas as pd

from experiment_utils import sha256
from non_diagnostic_leads import DATASET_SOURCES, LABELS, SPLITS, load, verify_cross_dataset


def source_text(dataset):
    source = DATASET_SOURCES[dataset]
    return subprocess.check_output(
        ['git', 'show', f"{source['commit']}:{source['builder']}"], text=True)


def reference_metadata(root, dataset):
    """Execute only original selection/split statements in memory, never export code."""
    tree = ast.parse(source_text(dataset))
    if dataset == 'dataset1':
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'pathlib':
                break  # Original file-writing section starts here.
            body.append(node)
        class RawPaths(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value.startswith('data/raw/ptb-xl/'):
                    return ast.copy_location(ast.Constant(str(root / Path(node.value).name)), node)
                return node
        module = RawPaths().visit(ast.Module(body=body, type_ignores=[]))
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(module), '<dataset1 selection only>', 'exec'), namespace)
        return namespace['dataset1']
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    body = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'out' for t in node.targets):
            break  # Exclude output construction and all file-writing statements.
        body.append(node)
    namespace = dict(pd=pd, ast=ast, RAW_DIR=root)
    exec(compile(ast.Module(body=body, type_ignores=[]), '<dataset2 selection only>', 'exec'), namespace)
    return namespace['dataset']


def verify(root, directories):
    raw = pd.read_csv(root / 'ptbxl_database.csv').set_index('ecg_id')
    result, frames = {}, {}
    for dataset, directory in directories.items():
        _, provenance, actual = load(directory, dataset)
        frames[dataset] = actual
        reference = reference_metadata(root, dataset)
        contract = provenance['contract']
        for name, key in [('ptbxl_database.csv', 'metadata_sha256'), ('scp_statements.csv', 'statements_sha256')]:
            if contract[key] != sha256(root / name):
                raise ValueError(f'{dataset}: raw metadata fingerprint mismatch')
        checks = {}
        for split, actual_rows in zip(SPLITS, actual):
            folds = list(range(1, 9)) if split == 'train' else [9 if split == 'val' else 10]
            expected = reference[reference.strat_fold.isin(folds)].reset_index(drop=True)
            columns = ['ecg_id', 'patient_id', 'CLBBB', 'ILBBB', 'strat_fold']
            pd.testing.assert_frame_equal(actual_rows[columns], expected[columns], check_dtype=False)
            np.testing.assert_array_equal(actual_rows.filename_hr,
                                          raw.loc[expected.ecg_id, 'filename_hr'])
            if 'filename_lr' in actual_rows:
                np.testing.assert_array_equal(actual_rows.filename_lr, expected.filename_lr)
            y = np.load(directory / f'y_{split}.npy', allow_pickle=False)
            np.testing.assert_array_equal(y, expected[LABELS].to_numpy(dtype=np.float32))
            checks[split] = dict(recordings=len(expected), record_ids_match=True,
                patient_ids_match=True, separate_labels_match=True, folds_match=True,
                row_order_matches=True, native_hr_paths_match=True)
        result[dataset] = dict(source=DATASET_SOURCES[dataset],
            source_builder_sha256=__import__('hashlib').sha256(source_text(dataset).encode()).hexdigest(),
            checks=checks, counts=provenance['counts'], export_sha256=provenance['sha256'],
            cross_split_patient_overlap=0)
    if set(frames) == {'dataset1', 'dataset2'}:
        result['cross_dataset_checks'] = verify_cross_dataset(frames['dataset1'], frames['dataset2'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--dataset1', type=Path, required=True)
    parser.add_argument('--dataset2', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directories = dict(dataset1=args.dataset1)
    if args.dataset2:
        directories['dataset2'] = args.dataset2
    result = verify(args.raw_root, directories)
    if args.output.exists():
        parser.error('Choose a new verification report path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
