import numpy as np
import pandas as pd

base = "data/processed/dataset1"

for split in ["train", "val", "test"]:
    X = np.load(f"{base}/X_{split}.npy")
    y = np.load(f"{base}/y_{split}.npy")
    metadata = pd.read_csv(f"{base}/{split}.csv")

    expected_labels = metadata[["CLBBB", "ILBBB"]].to_numpy()

    assert len(X) == len(y) == len(metadata)
    assert np.array_equal(y, expected_labels)
    assert np.isfinite(X).all()

    print(f"{split}: All validation checks passed!")