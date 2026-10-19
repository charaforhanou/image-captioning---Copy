"""Analyse un fichier test_predictions_r2gen.csv pour comprendre d'où viennent
les scores : taux de doublons, répartition normal/anormal, et métriques NLG
calculées séparément sur chaque sous-ensemble.

Ne modifie rien. Diagnostic uniquement.

Usage :
    python scripts/analyze_predictions.py --pred results/ablations/full/eval/test_predictions_r2gen.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# --- Heuristique "cas normal" -------------------------------------------------
# Un rapport est considéré "normal" s'il ne contient AUCUN de ces marqueurs
# d'anomalie. Volontairement conservateur : mieux vaut classer un cas limite
# comme anormal que l'inverse.
ABNORMAL_PATTERNS = [
    r"opacit", r"consolidat\w*\b(?<!no consolidat)", r"effusion", r"pneumothorax",
    r"pneumonia", r"edema", r"atelectasis", r"nodule", r"mass\b", r"infiltrat",
    r"fracture", r"cardiomegaly", r"enlarge", r"emphysema", r"fibrosis",
    r"granuloma", r"calcif", r"hernia", r"scarring", r"thickening",
]
# marqueurs de négation qui précèdent souvent une anomalie ("no effusion")
NEGATIONS = [r"no ", r"without ", r"free of ", r"negative for ", r"clear of ",
             r"absence of ", r"resolution of ", r"no evidence of "]


def is_normal(text: str) -> bool:
    """True si le rapport ne mentionne aucune anomalie NON niée."""
    t = str(text).lower()
    for pat in ABNORMAL_PATTERNS:
        for m in re.finditer(pat, t):
            start = m.start()
            # fenêtre de 25 caractères avant le marqueur pour chercher une négation
            window = t[max(0, start - 25):start]
            if not any(re.search(neg, window) for neg in NEGATIONS):
                return False
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="Chemin vers test_predictions_r2gen.csv")
    p.add_argument("--dedup-by", default="reference",
                   choices=["reference", "sample_id", "none"],
                   help="Clé de déduplication avant analyse.")
    return p.parse_args()


def norm_text(s: str) -> str:
    """Normalisation légère pour comparer les références (espaces, casse, ponctuation finale)."""
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


def bleu_rouge_cider(preds, refs):
    """Calcule BLEU-1..4, ROUGE-L, CIDEr via pycocoevalcap. Retourne un dict."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    gts = {i: [norm_text(r)] for i, r in enumerate(refs)}
    res = {i: [norm_text(p)] for i, p in enumerate(preds)}

    out = {}
    bleu, _ = Bleu(4).compute_score(gts, res)
    out.update({f"BLEU-{i+1}": round(b, 4) for i, b in enumerate(bleu)})
    out["ROUGE-L"] = round(Rouge().compute_score(gts, res)[0], 4)
    out["CIDEr"] = round(Cider().compute_score(gts, res)[0], 4)
    return out


def report_block(title, df):
    print(f"\n{'='*60}\n{title}  (n={len(df)})\n{'='*60}")
    if len(df) == 0:
        print("  (vide)")
        return
    try:
        scores = bleu_rouge_cider(df["prediction"].tolist(), df["reference"].tolist())
        for k, v in scores.items():
            print(f"  {k:<10} {v}")
    except ImportError:
        print("  pycocoevalcap non installé — scores omis (pip install pycocoevalcap)")


def main():
    args = parse_args()
    df = pd.read_csv(args.pred)
    print(f"Fichier : {args.pred}")
    print(f"Lignes brutes : {len(df)}")

    # --- 1. Doublons de référence -------------------------------------------
    df["_ref_norm"] = df["reference"].map(norm_text)
    n_unique_ref = df["_ref_norm"].nunique()
    print(f"\n--- Doublons ---")
    print(f"Références uniques : {n_unique_ref} / {len(df)} "
          f"({100*n_unique_ref/len(df):.1f}% uniques)")
    print(f"Taux de duplication : {100*(1 - n_unique_ref/len(df)):.1f}%")

    top_dupes = df["_ref_norm"].value_counts().head(5)
    print("Références les plus répétées :")
    for txt, cnt in top_dupes.items():
        if cnt > 1:
            print(f"  {cnt}×  {txt[:70]}...")

    # prédictions identiques à la référence (mémorisation du rapport standard)
    df["_pred_norm"] = df["prediction"].map(norm_text)
    n_exact = (df["_pred_norm"] == df["_ref_norm"]).sum()
    print(f"\nPrédictions == référence (exact après normalisation) : "
          f"{n_exact} ({100*n_exact/len(df):.1f}%)")

    # --- 2. Déduplication ----------------------------------------------------
    if args.dedup_by == "reference":
        df_eval = df.drop_duplicates(subset="_ref_norm", keep="first").reset_index(drop=True)
    elif args.dedup_by == "sample_id":
        df_eval = df.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
    else:
        df_eval = df
    print(f"\nAprès déduplication par '{args.dedup_by}' : {len(df_eval)} cas évalués")

    # --- 3. Normal vs anormal (sur la RÉFÉRENCE) ----------------------------
    df_eval["_is_normal"] = df_eval["reference"].map(is_normal)
    n_norm = df_eval["_is_normal"].sum()
    print(f"\n--- Répartition (basée sur la référence) ---")
    print(f"Normaux  : {n_norm} ({100*n_norm/len(df_eval):.1f}%)")
    print(f"Anormaux : {len(df_eval)-n_norm} ({100*(len(df_eval)-n_norm)/len(df_eval):.1f}%)")

    # --- 4. Scores par sous-ensemble ----------------------------------------
    report_block("SCORES — tous les cas (dédupliqués)", df_eval)
    report_block("SCORES — cas NORMAUX", df_eval[df_eval["_is_normal"]])
    report_block("SCORES — cas ANORMAUX", df_eval[~df_eval["_is_normal"]])

    print("\n" + "="*60)
    print("Lecture : si les cas ANORMAUX ont des scores nettement plus bas que")
    print("les NORMAUX, le modèle réussit surtout en reproduisant le rapport")
    print("normal standard. C'est le point à traiter avant publication.")


if __name__ == "__main__":
    main()