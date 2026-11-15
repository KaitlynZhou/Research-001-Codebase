"""Read local PTB-XL CSVs without downloading data or assigning final targets."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path


def parse_codes(value):
    codes = ast.literal_eval(value)
    if not isinstance(codes, dict) or not codes:
        raise ValueError("scp_codes must be a nonempty dictionary")
    for code, likelihood in codes.items():
        if (not isinstance(code, str) or isinstance(likelihood, bool)
                or not isinstance(likelihood, (int, float))
                or not math.isfinite(likelihood) or not 0 <= likelihood <= 100):
            raise ValueError("Invalid SCP code or likelihood")
    return codes


def read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def inspect(root):
    statements = read_csv(root / "scp_statements.csv", ["description", "diagnostic"])
    if not statements:
        raise ValueError("Empty SCP statement table")
    # Official CSV has an unnamed first column containing the SCP code.
    code_column = next(iter(statements[0]))
    lookup = {row[code_column]: row for row in statements}
    if len(lookup) != len(statements):
        raise ValueError("Duplicate statement codes")
    expected = {"CLBBB": "complete left bundle branch block",
                "ILBBB": "incomplete left bundle branch block", "NORM": "normal ecg"}
    for code, description in expected.items():
        if code not in lookup or lookup[code]["description"].strip().lower() != description:
            raise ValueError(f"Verify statement definition for {code}; unexpected or missing description")
        if lookup[code]["diagnostic"] not in ("1", "1.0"):
            raise ValueError(f"{code} is not marked diagnostic")
    rows = read_csv(root / "ptbxl_database.csv",
                    ["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_lr", "filename_hr"])
    if not rows:
        raise ValueError("Empty ECG metadata")
    counts, likelihoods, cooccurrence = Counter(), defaultdict(Counter), Counter()
    folds, patients, ids = Counter(), defaultdict(set), set()
    conflicts = Counter()
    diagnostic = {code for code, row in lookup.items() if row["diagnostic"] in ("1", "1.0")}
    for row in rows:
        ecg_id = row["ecg_id"]
        if not ecg_id or ecg_id in ids:
            raise ValueError(f"Missing or duplicate ecg_id: {ecg_id}")
        ids.add(ecg_id)
        if not row["patient_id"].strip():
            raise ValueError(f"Missing patient_id at ECG {ecg_id}")
        # Canonicalize numeric patient IDs, e.g. 12 and 12.0.
        patient = float(row["patient_id"])
        if not math.isfinite(patient) or not patient.is_integer():
            raise ValueError(f"Invalid patient_id at ECG {ecg_id}")
        fold_value = float(row["strat_fold"])
        if not fold_value.is_integer() or not 1 <= fold_value <= 10:
            raise ValueError(f"Invalid fold at ECG {ecg_id}")
        fold = int(fold_value)
        patients[int(patient)].add(fold)
        folds[fold] += 1
        codes = parse_codes(row["scp_codes"])
        unknown = set(codes) - lookup.keys()
        if unknown:
            raise ValueError(f"Unknown codes at ECG {ecg_id}: {sorted(unknown)}")
        counts.update(codes.keys())
        for code, value in codes.items():
            likelihoods[code][str(value)] += 1
        # Presence, including likelihood 0, is retained for conservative ambiguity review.
        if "CLBBB" in codes:
            cooccurrence.update(set(codes) - {"CLBBB"})
        if {"CLBBB", "ILBBB"} <= codes.keys():
            conflicts["CLBBB_with_ILBBB"] += 1
        if {"CLBBB", "NORM"} <= codes.keys():
            conflicts["CLBBB_with_NORM"] += 1
        if "NORM" in codes and (set(codes) & diagnostic) - {"NORM"}:
            conflicts["NORM_with_other_diagnostic"] += 1
    overlap = sum(len(values) > 1 for values in patients.values())
    if overlap:
        raise ValueError(f"Patient leakage: {overlap} patients occur in multiple folds")
    return {
        "records": len(rows), "patients": len(patients),
        "records_per_fold": dict(sorted(folds.items())),
        "patients_in_multiple_folds": overlap,
        "target_definitions": {code: lookup[code] for code in expected},
        "diagnostic_labels": {code: {"description": lookup[code]["description"],
                                    "records": counts[code]} for code in sorted(diagnostic)},
        "all_code_counts": dict(sorted(counts.items())),
        "target_likelihood_counts": {code: dict(likelihoods[code]) for code in expected},
        "CLBBB_cooccurring_codes": dict(cooccurrence.most_common()),
        "conflicts_by_code_presence": dict(conflicts),
        "note": "Inspection only; no labels assigned. Likelihood 0 is not treated as absence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ptb-xl"))
    args = parser.parse_args()
    try:
        report = inspect(args.data_root)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"Metadata inspection failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
