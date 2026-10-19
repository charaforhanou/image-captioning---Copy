import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    "No Finding",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-labels", required=True)
    p.add_argument("--ref-labels", required=True)
    p.add_argument("--positive-set", default="pos+unc", choices=["pos", "pos+unc"])
    p.add_argument("--exclude-no-finding", action="store_true")
    return p.parse_args()


def load_labels(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in CHEXPERT_LABELS]
    if not cols:
        raise SystemExit(f"Aucune colonne de label trouvee dans {path}. Colonnes: {list(df.columns)}")
    return df[cols]


def binarize(df, positive_set):
    M = np.zeros((len(df), len(CHEXPERT_LABELS)), dtype=int)
    for j, lab in enumerate(CHEXPERT_LABELS):
        if lab not in df.columns:
            continue
        for i, v in enumerate(df[lab]):
            if pd.isna(v):
                present = False
            else:
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    M[i, j] = 0
                    continue
                if positive_set == "pos":
                    present = (val == 1.0)
                else:
                    present = (val == 1.0) or (val == -1.0) or (val == 2.0)
            M[i, j] = 1 if present else 0
    return M


def main():
    args = parse_args()
    dfp = load_labels(args.pred_labels)
    dfr = load_labels(args.ref_labels)
    if len(dfp) != len(dfr):
        raise SystemExit(f"Lignes differentes: preds={len(dfp)}, refs={len(dfr)}")

    y_pred = binarize(dfp, args.positive_set)
    y_true = binarize(dfr, args.positive_set)

    label_idx = list(range(len(CHEXPERT_LABELS)))
    if args.exclude_no_finding:
        nf = CHEXPERT_LABELS.index("No Finding")
        label_idx = [j for j in label_idx if j != nf]
        y_pred = y_pred[:, label_idx]
        y_true = y_true[:, label_idx]
    used = [CHEXPERT_LABELS[j] for j in label_idx]

    from sklearn.metrics import precision_recall_fscore_support
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_l, r_l, f_l, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(used))))

    print(f"Cas evalues : {len(y_true)}")
    print(f"Present = {args.positive_set}" + (" | No Finding exclu" if args.exclude_no_finding else ""))
    print("\n=== Clinical Efficacy (CheXbert) ===")
    print(f"{'':<28}{'P':>8}{'R':>8}{'F1':>8}")
    print(f"{'MICRO':<28}{p_mi:>8.4f}{r_mi:>8.4f}{f_mi:>8.4f}")
    print(f"{'MACRO':<28}{p_ma:>8.4f}{r_ma:>8.4f}{f_ma:>8.4f}")
    print("\n--- Par pathologie ---")
    print(f"{'Label':<28}{'P':>8}{'R':>8}{'F1':>8}{'n_ref':>7}")
    for j, lab in enumerate(used):
        print(f"{lab:<28}{p_l[j]:>8.4f}{r_l[j]:>8.4f}{f_l[j]:>8.4f}{int(sup[j]):>7}")

    out = Path(args.pred_labels).parent / "clinical_efficacy.csv"
    rows = [{"label": "MICRO", "P": p_mi, "R": r_mi, "F1": f_mi, "n_ref": ""},
            {"label": "MACRO", "P": p_ma, "R": r_ma, "F1": f_ma, "n_ref": ""}]
    for j, lab in enumerate(used):
        rows.append({"label": lab, "P": p_l[j], "R": r_l[j], "F1": f_l[j], "n_ref": int(sup[j])})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nEcrit : {out}")


if __name__ == "__main__":
    main()
