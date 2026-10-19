import argparse
import numpy as np
import pandas as pd

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

def load(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c in CHEXPERT_LABELS]
    return df[cols]

def binarize(df):
    M = np.zeros((len(df), len(CHEXPERT_LABELS)), dtype=int)
    for j, lab in enumerate(CHEXPERT_LABELS):
        if lab in df.columns:
            for i, v in enumerate(df[lab]):
                if pd.isna(v):
                    M[i, j] = 0
                else:
                    try:
                        M[i, j] = 1 if float(v) in (1.0, -1.0) else 0
                    except (ValueError, TypeError):
                        M[i, j] = 0
    return M

p = argparse.ArgumentParser()
p.add_argument("--pred-labels", required=True)
p.add_argument("--ref-labels", required=True)
a = p.parse_args()

yp = binarize(load(a.pred_labels))
yt = binarize(load(a.ref_labels))

# F1 sample-wise : un F1 par rapport, puis moyenne
f1s = []
for i in range(len(yt)):
    tp = int((yp[i] & yt[i]).sum())
    fp = int((yp[i] & (1 - yt[i])).sum())
    fn = int(((1 - yp[i]) & yt[i]).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    # convention : si ref et pred sont tous deux "tout absent", F1 = 1 (accord parfait)
    if yt[i].sum() == 0 and yp[i].sum() == 0:
        f1 = 1.0
    f1s.append(f1)

print(f"Cas : {len(f1s)}")
print(f"F1 sample-wise (moyenne) : {np.mean(f1s):.4f}")
print(f"  -- variante sans le cas 'tout absent = 1' :")
f1s_strict = []
for i in range(len(yt)):
    tp = int((yp[i] & yt[i]).sum()); fp = int((yp[i] & (1-yt[i])).sum()); fn = int(((1-yp[i]) & yt[i]).sum())
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1s_strict.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
print(f"F1 sample-wise strict : {np.mean(f1s_strict):.4f}")
