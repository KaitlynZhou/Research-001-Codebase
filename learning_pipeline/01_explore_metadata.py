import pandas as pd 
import ast
from collections import Counter

df = pd.read_csv("data/raw/ptb-xl/ptbxl_database.csv")


clbbb_count = 0
ilbbb_count = 0
total_lbbb_count = 0


scp = pd.read_csv("data/raw/ptb-xl/scp_statements.csv", index_col=0)
normal_records = []

for _, row in df.iterrows():
    codes = ast.literal_eval(row["scp_codes"])

    if "NORM" in codes and "CLBBB" not in codes and "ILBBB" not in codes:
        normal_records.append(row["ecg_id"])

print("Preliminary normal recordings:", len(normal_records))

conflicting_count = 0

for diagnosis in df["scp_codes"]:
    codes = ast.literal_eval(diagnosis)

    if "NORM" in codes and "CLBBB" not in codes and "ILBBB" not in codes:

        diagnostic_codes = [
            code for code in codes
            if code in scp.index and scp.loc[code, "diagnostic"] == 1
        ]

        if any(code != "NORM" for code in diagnostic_codes):
            conflicting_count += 1

print("Normal recordings with conflicting diagnostic codes:", conflicting_count)
conflicting_codes = Counter()

for diagnosis in df["scp_codes"]:
    codes = ast.literal_eval(diagnosis)

    if "NORM" in codes and "CLBBB" not in codes and "ILBBB" not in codes:
        for code in codes:
            if (
                code != "NORM"
                and code in scp.index
                and scp.loc[code, "diagnostic"] == 1
            ):
                conflicting_codes[code] += 1

print(conflicting_codes.most_common(10))