"""Adapt a pinned copy of Prisha's preprocessing on existing local data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PREPROCESSING_COMMIT = '77873b1309ec049f204fbbec3d4af7bdb7976911'
MODULES = ('__init__.py', 'inspect_metadata.py', 'process_metadata.py', 'extract_v6.py')


def export_pipeline(destination):
    """Read Git objects without checkout, merge, branch changes, or network access."""
    repository = Path(__file__).resolve().parent
    package = Path(destination) / 'src' / 'ecg_lbbb'
    package.mkdir(parents=True)
    for name in MODULES:
        content = subprocess.check_output(
            ['git', 'show', f'{PREPROCESSING_COMMIT}:src/ecg_lbbb/{name}'], cwd=repository)
        (package / name).write_bytes(content)
    # Only extraction changes; cohort selection and patient folds stay unchanged.
    extractor = package / 'extract_v6.py'
    text = extractor.read_text()
    substitutions = {
        '100 Hz': '500 Hz', '1000': '5000', '!= 100 ': '!= 500 ',
        "'fs': 100": "'fs': 500", 'filename_lr': 'filename_hr',
        'records100': 'records500',
    }
    for before, after in substitutions.items():
        if before not in text:
            raise RuntimeError(f'Pinned extraction interface changed: missing {before!r}')
        text = text.replace(before, after)
    extractor.write_text(text)
    return package.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New output directory outside the Git repository')
    args = parser.parse_args()
    raw_root, output = args.raw_root.expanduser().resolve(), args.output.expanduser().resolve()
    repository = Path(__file__).resolve().parent
    if output.is_relative_to(repository):
        parser.error('Keep processed ECG data outside the Git repository')
    if output.exists():
        parser.error('Choose a new output directory to preserve existing processed files')
    for filename in ('ptbxl_database.csv', 'scp_statements.csv'):
        if not (raw_root / filename).is_file():
            parser.error(f'Missing {raw_root / filename}')
    if not (raw_root / 'records500').is_dir():
        parser.error('Missing local records500 directory; no download attempted')
    with tempfile.TemporaryDirectory(prefix='ptbxl-pipeline-') as temporary:
        source = export_pipeline(temporary)
        environment = dict(os.environ, PYTHONPATH=str(source))
        subprocess.run([sys.executable, '-m', 'ecg_lbbb.process_metadata',
                        '--data-root', str(raw_root), '--output', str(output / 'metadata')],
                       env=environment, check=True)
        subprocess.run([sys.executable, '-m', 'ecg_lbbb.extract_v6',
                        '--data-root', str(raw_root),
                        '--manifest', str(output / 'metadata' / 'selected_recordings.csv'),
                        '--output', str(output / 'v6')], env=environment, check=True)
    (output / 'pipeline_source.json').write_text(json.dumps({
        'preprocessing_commit': PREPROCESSING_COMMIT, 'raw_root': str(raw_root),
        'modules': list(MODULES), 'downloads': False,
        'extraction_adaptation': 'filename_hr; records500; 500 Hz; 5000 samples; no resampling',
        'label_and_split_logic': 'unchanged'}, indent=2))
    print(f'Completed local extraction: {output / "v6"}')


if __name__ == '__main__':
    main()
