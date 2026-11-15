"""Resumable PTB-XL 1.0.3 records100-only download with official SHA-256 verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import requests

BASE = 'https://physionet.org/files/ptb-xl/1.0.3/'
ROOT = Path('data/raw/ptb-xl')
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
    return LOCAL.session


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(item):
    relative, expected = item
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == expected:
        return 'verified_existing'
    part = target.with_suffix(target.suffix + '.part')
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with session().get(BASE + relative, headers=headers, stream=True, timeout=(30, 90)) as r:
                if r.status_code == 416:
                    if digest(part) == expected:
                        part.replace(target)
                        return 'downloaded'
                    part.unlink()
                    continue
                r.raise_for_status()
                append = offset and r.status_code == 206
                if append and not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected byte-range response')
                with part.open('ab' if append else 'wb') as f:
                    for block in r.iter_content(128 * 1024):
                        f.write(block)
            if digest(part) != expected:
                part.unlink()
                raise ValueError(f'SHA-256 mismatch: {relative}')
            part.replace(target)
            return 'downloaded'
        except (requests.RequestException, OSError, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f'Could not download {relative}')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Preserve a baseline for verification on a fresh clone, without replacing one.
    snapshot = Path('results/pre_download_hashes.json')
    if not snapshot.exists() and Path('data/processed/selected_recordings.csv').is_file():
        files = list(ROOT.glob('*.csv')) + list(Path('data/processed').glob('*.*'))
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({str(p): digest(p) for p in files}, indent=2) + '\n')
    r = session().get(BASE + 'SHA256SUMS.txt', timeout=(30, 90))
    r.raise_for_status()
    entries = []
    for line in r.text.splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if name.startswith('./'):
            name = name[2:]
        p = PurePosixPath(name)
        if p.parts and p.parts[0] == 'records100' and p.suffix in {'.hea', '.dat'}:
            if p.is_absolute() or '..' in p.parts or len(checksum) != 64:
                raise ValueError('Unsafe checksum entry')
            entries.append((str(p), checksum))
    if not entries or len(dict(entries)) != len(entries):
        raise ValueError('Missing or duplicate records100 checksum entries')
    # Each 12-lead, 1000-sample int16 record is 24 KB; reserve generously for headers/arrays.
    required = len(entries) * 32000 + 500_000_000
    free = shutil.disk_usage(ROOT).free
    print(f'Official records100 entries: {len(entries)}; free bytes: {free}; required reserve: {required}', flush=True)
    if free < required:
        raise OSError('Insufficient free disk space')
    (ROOT / 'records100-checksums.txt').write_text(''.join(f'{h}  {p}\n' for p, h in entries))
    counts = {}
    failures = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(fetch, item): item[0] for item in entries}
        for n, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            try:
                result = future.result()
                counts[result] = counts.get(result, 0) + 1
            except Exception as exc:
                failures.append({'path': jobs[future], 'error': str(exc)})
            if n % 500 == 0 or n == len(entries):
                print(f'{n}/{len(entries)} checked; {counts}; failures={len(failures)}; elapsed={time.monotonic()-started:.0f}s', flush=True)
    report = {'version': '1.0.3', 'source': BASE + 'records100/', 'files': len(entries),
              'counts': counts, 'failures': failures, 'sha256_verified': len(entries) - len(failures)}
    (ROOT / 'records100-download-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if failures:
        raise RuntimeError(f'{len(failures)} failures; rerun same command to resume')


if __name__ == '__main__':
    main()
