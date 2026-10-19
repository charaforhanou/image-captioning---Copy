import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"data\processed\concept_labels")
CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

def binarize_present(v):
    # present = positif (1) ou incertain (-1) ; absent/non-mentionne = 0
    if pd.isna(v):
        return 0
    try:
        val = float(v)
    except (ValueError, TypeError):
        return 0
    return 1 if (val == 1.0 or val == -1.0) else 0

for split in ["train", "val", "test"]:
    labels = pd.read_csv(BASE / f"{split}_labeled" / "labeled_reports.csv")
    uids = pd.read_csv(BASE / f"{split}_uids.csv")

    if len(labels) != len(uids):
        raise SystemExit(f"{split}: desalignement labels={len(labels)} vs uids={len(uids)}")

    out = pd.DataFrame()
    out["uid"] = uids["uid"].values
    for c in CONDITIONS:
        if c in labels.columns:
            out[c] = labels[c].map(binarize_present).values
        else:
            out[c] = 0

    out.to_csv(BASE / f"{split}_concept_targets.csv", index=False)
    n_pos = out[CONDITIONS].sum().sum()
    print(f"{split:<6}: {len(out)} etudes, {int(n_pos)} labels positifs au total")
    # distribution par pathologie
    print("   " + "  ".join(f"{c.split()[0][:4]}={int(out[c].sum())}" for c in CONDITIONS))

print("\nOK -> data\\processed\\concept_labels\\{split}_concept_targets.csv")
