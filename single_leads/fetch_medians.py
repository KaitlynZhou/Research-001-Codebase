"""Download the PTB-XL+ 1.0.1 12SL median beats needed by the cohorts, verifying SHA-256 sums."""
import argparse, hashlib, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import cohorts, median_relpath, MEDIAN_SUBDIR

BASE = 'https://physionet-open.s3.amazonaws.com/ptb-xl-plus/1.0.1/'


def fetch(url, dest, attempts=6):
    """Download to a .part file, then rename; retry transient network errors with backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + '.part')
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                tmp.write_bytes(r.read())
            tmp.replace(dest)
            return
        except (OSError, urllib.error.URLError):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def main(raw, root):
    root.mkdir(parents=True, exist_ok=True)
    sums_path = root / 'SHA256SUMS.txt'
    if not sums_path.exists():
        fetch(BASE + 'SHA256SUMS.txt', sums_path)
    sums = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.startswith(MEDIAN_SUBDIR + '/'):
            sums[name] = digest
    m = cohorts(raw)
    needed = m[m.dataset1 | m.strat_fold.eq(10)].ecg_id
    wanted = [median_relpath(i) + s for i in needed for s in ('.hea', '.dat')]
    available = [w for w in wanted if w in sums]
    print(f'{len(needed)} cohort records; {len(available) // 2} have a published median beat', flush=True)

    def one(name):
        dest = root / name
        for attempt in range(3):
            if not dest.exists():
                fetch(BASE + name, dest)
            if hashlib.sha256(dest.read_bytes()).hexdigest() == sums[name]:
                return
            dest.unlink()
        raise RuntimeError(f'SHA-256 mismatch after retries: {name}')

    with ThreadPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(one, available), 1):
            if k % 2000 == 0:
                print(k, '/', len(available), flush=True)
    print('All median-beat files present and SHA-256 verified.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, required=True, help='PTB-XL 1.0.3 directory with ptbxl_database.csv')
    p.add_argument('--medians', type=Path, required=True, help='Destination PTB-XL+ root')
    a = p.parse_args(); sys.exit(main(a.raw, a.medians))
