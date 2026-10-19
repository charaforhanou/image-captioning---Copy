"""Evaluation stratifiee (316 cas non vus) avec METEOR, sans relancer Java 1000x.

METEOR (lent, via Java) est calcule UNE seule fois en point estimate.
BLEU/ROUGE/CIDEr (rapides) ont leur IC par bootstrap.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import io

import numpy as np
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

ROOT = Path(__file__).resolve().parents[1]


def simple(s):
    return str(s).lower().strip()


def main():
    meta = pd.read_csv(ROOT / "data/processed/metadata_r2gen.csv")
    train_reports = set(meta[meta["split"] == "train"]["report"].map(simple))

    df = pd.read_csv(ROOT / "results/test_predictions_r2gen.csv").drop_duplicates(subset="sample_id", keep="first")
    df = df.dropna(subset=["prediction", "reference"])
    df["ref_in_train"] = df["reference"].map(simple).apply(lambda r: r in train_reports)
    unseen = df[~df["ref_in_train"]].reset_index(drop=True)

    preds = unseen["prediction"].astype(str).tolist()
    refs = unseen["reference"].astype(str).tolist()
    n = len(preds)
    print(f"Etudes non vues (generalisation) : {n}")

    gts = {i: [simple(refs[i])] for i in range(n)}
    res = {i: [simple(preds[i])] for i in range(n)}

    # --- METEOR : UNE seule fois (Java, lent) ---
    print("Calcul METEOR (une fois)...")
    from pycocoevalcap.meteor.meteor import Meteor
    with contextlib.redirect_stdout(io.StringIO()):
        meteor_score, _ = Meteor().compute_score(gts, res)
    meteor_score = float(meteor_score)
    print(f"METEOR = {meteor_score:.4f}")

    # --- Fonctions rapides pour le bootstrap (pas de Java) ---
    def fast_scores(p, r):
        g = {i: [simple(r[i])] for i in range(len(r))}
        s = {i: [simple(p[i])] for i in range(len(p))}
        out = {}
        with contextlib.redirect_stdout(io.StringIO()):
            bleu, _ = Bleu(4).compute_score(g, s)
            rouge, _ = Rouge().compute_score(g, s)
            cider, _ = Cider().compute_score(g, s)
        out["BLEU-1"], out["BLEU-2"], out["BLEU-3"], out["BLEU-4"] = [float(b) for b in bleu]
        out["ROUGE-L"] = float(rouge)
        out["CIDEr"] = float(cider)
        return out

    point = fast_scores(preds, refs)
    point["METEOR"] = meteor_score

    # Bootstrap sur les metriques rapides uniquement
    print("Bootstrap (BLEU/ROUGE/CIDEr, 1000x)...")
    rng = np.random.default_rng(42)
    keys_fast = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"]
    samples = {k: [] for k in keys_fast}
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        p = [preds[i] for i in idx]
        r = [refs[i] for i in idx]
        s = fast_scores(p, r)
        for k in keys_fast:
            samples[k].append(s[k])

    print(f"\n{'Metric':<10}{'Mean':>10}{'95% CI':>26}")
    print("-" * 46)
    lines = [f"Evaluation stratifiee - {n} etudes non vues (generalisation)", ""]
    order = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    for k in order:
        if k == "METEOR":
            line = f"{k:<10}{point[k]:>10.4f}     (point estimate, pas d'IC)"
        else:
            arr = np.array(samples[k])
            lo, hi = np.quantile(arr, 0.025), np.quantile(arr, 0.975)
            line = f"{k:<10}{point[k]:>10.4f}     [{lo:.4f}, {hi:.4f}]"
        print(line)
        lines.append(line)

    (ROOT / "results/metrics_stratified.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nEcrit dans results/metrics_stratified.txt")


if __name__ == "__main__":
    main()