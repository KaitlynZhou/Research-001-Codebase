import pandas as pd
import ast


df = pd.read_csv("data/raw/ptb-xl/ptbxl_database.csv")
scp = pd.read_csv("data/raw/ptb-xl/scp_statements.csv", index_col=0)

selected_records = []

for _, row in df.iterrows():
    codes = ast.literal_eval(row["scp_codes"])

    clbbb = int("CLBBB" in codes)
    ilbbb = int("ILBBB" in codes)

    
    other_diagnoses = [
        code for code in codes
        if code in scp.index
        and scp.loc[code, "diagnostic"] == 1
        and code != "NORM"
    ]

    
    is_normal = "NORM" in codes and len(other_diagnoses) == 0

    
    if clbbb or ilbbb or is_normal:
        selected_records.append({
            "ecg_id": row["ecg_id"],
            "patient_id": row["patient_id"],
            "CLBBB": clbbb,
            "ILBBB": ilbbb,
            "strat_fold": row["strat_fold"],
            "filename_lr": row["filename_lr"]
        })

dataset1 = pd.DataFrame(selected_records)

print("Total recordings:", len(dataset1))
print("CLBBB:", dataset1["CLBBB"].sum())
print("ILBBB:", dataset1["ILBBB"].sum())
print("Normal:", ((dataset1["CLBBB"] == 0) &
                  (dataset1["ILBBB"] == 0)).sum())

# print("\nFirst five rows:")
# print(dataset1.head())
#################

train_df = dataset1[dataset1["strat_fold"].isin(range(1, 9))]
val_df = dataset1[dataset1["strat_fold"] == 9]
test_df = dataset1[dataset1["strat_fold"] == 10]

for name, split in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(f"\n{name}:")
    print("Total recordings:", len(split))
    print("CLBBB:", split["CLBBB"].sum())
    print("ILBBB:", split["ILBBB"].sum())
    print("Normal:", ((split["CLBBB"] == 0) &
                      (split["ILBBB"] == 0)).sum())

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
test_patients = set(test_df["patient_id"])

print("Train/validation overlap:", len(train_patients & val_patients))
print("Train/test overlap:", len(train_patients & test_patients))
print("Validation/test overlap:", len(val_patients & test_patients))


from pathlib import Path

output_dir = Path("data/processed/dataset1")
output_dir.mkdir(parents=True, exist_ok=True)

dataset1.to_csv(output_dir / "metadata.csv", index=False)
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("Dataset 1 metadata saved successfully!")