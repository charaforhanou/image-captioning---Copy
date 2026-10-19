import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.evaluation.metrics_pycoco import compute_all_metrics

meta = pd.read_csv("data/processed/metadata_r2gen.csv")
def norm(s): return str(s).lower().strip()
train_reports = set(meta[meta["split"]=="train"]["report"].map(norm))

df = pd.read_csv("results/test_predictions_r2gen.csv").drop_duplicates(subset="sample_id", keep="first")
df = df.dropna(subset=["prediction","reference"])
df["ref_in_train"] = df["reference"].map(norm).apply(lambda r: r in train_reports)

seen = df[df["ref_in_train"]]
unseen = df[~df["ref_in_train"]]

print(f"=== Etudes dont le rapport EST dans le train (memorisation possible): {len(seen)} ===")
m = compute_all_metrics(seen["prediction"].tolist(), seen["reference"].tolist())
print(f"BLEU-4={m['bleu4']:.4f}  ROUGE-L={m['rougeL_f1']:.4f}  CIDEr={m['cider']:.4f}")

print(f"\n=== Etudes dont le rapport N'EST PAS dans le train (vraie generalisation): {len(unseen)} ===")
m = compute_all_metrics(unseen["prediction"].tolist(), unseen["reference"].tolist())
print(f"BLEU-4={m['bleu4']:.4f}  ROUGE-L={m['rougeL_f1']:.4f}  CIDEr={m['cider']:.4f}")