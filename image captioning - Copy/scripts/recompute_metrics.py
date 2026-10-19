import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.evaluation.metrics_pycoco import compute_all_metrics_with_ci, count_empty

df = pd.read_csv("results/test_predictions_r2gen.csv")
df = df.sort_values("beam_score", ascending=False).drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)

preds = df["prediction"].tolist()
refs = df["reference"].tolist()
print(f"Etudes evaluees : {len(df)}")
print(f"Predictions vides/NaN : {count_empty(preds)}")

m = compute_all_metrics_with_ci(preds, refs, n_bootstrap=1000, use_meteor=False)

label = {"bleu1":"BLEU-1","bleu2":"BLEU-2","bleu3":"BLEU-3","bleu4":"BLEU-4","rougeL_f1":"ROUGE-L","cider":"CIDEr"}
print(f"\n{'Metric':<10}{'Mean':>10}{'95% CI':>26}")
print("-"*46)
lines = []
for k in ["bleu1","bleu2","bleu3","bleu4","rougeL_f1","cider"]:
    line = f"{label[k]:<10}{m[k]['mean']:>10.4f}     [{m[k]['ci_low']:.4f}, {m[k]['ci_high']:.4f}]"
    print(line); lines.append(line)

Path("results/metrics_r2gen.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\nEcrit dans results/metrics_r2gen.txt")