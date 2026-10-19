import argparse
import re
from pathlib import Path

import pandas as pd

ABNORMAL = [
    "opacit", "consolidat", "effusion", "pneumothorax", "pneumonia", "edema",
    "atelectasis", "atelectat", "nodule", "mass", "infiltrat", "fracture",
    "cardiomegaly", "enlarge", "emphysema", "fibrosis", "granuloma", "calcif",
    "hernia", "scarring", "thickening", "lesion",
]
NEG = ["no ", "without ", "free of ", "negative for ", "clear of ",
       "absence of ", "no evidence of ", "resolution of "]


def mentions_abnormality(text):
    t = " " + str(text).lower() + " "
    for pat in ABNORMAL:
        for m in re.finditer(pat, t):
            window = t[max(0, m.start() - 25):m.start()]
            if not any(neg in window for neg in NEG):
                return True
    return False


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--n-examples", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pred)
    if "beam_score" in df.columns:
        df = df.sort_values("beam_score", ascending=False)
    df = df.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
    df["_r"] = df["reference"].map(norm)
    df = df.drop_duplicates(subset="_r", keep="first").reset_index(drop=True)
    df = df.drop(columns=["_r"])
    print(f"Cas evalues : {len(df)}\n")

    df["_pred_abn"] = df["prediction"].map(mentions_abnormality)
    df["_ref_abn"] = df["reference"].map(mentions_abnormality)
    n = len(df)

    print("=== Mentions d'anomalie ===")
    print(f"References anormales   : {df['_ref_abn'].sum()} ({100*df['_ref_abn'].mean():.1f}%)")
    print(f"Predictions anormales  : {df['_pred_abn'].sum()} ({100*df['_pred_abn'].mean():.1f}%)")

    tp = ((df["_ref_abn"]) & (df["_pred_abn"])).sum()
    fn = ((df["_ref_abn"]) & (~df["_pred_abn"])).sum()
    fp = ((~df["_ref_abn"]) & (df["_pred_abn"])).sum()
    tn = ((~df["_ref_abn"]) & (~df["_pred_abn"])).sum()
    print("\n=== Detection binaire cas anormal (rapport entier) ===")
    print(f"  Ref anormale + Pred anormale (bon)   : {tp}")
    print(f"  Ref anormale + Pred normale  (RATE)  : {fn}")
    print(f"  Ref normale  + Pred anormale (faux+) : {fp}")
    print(f"  Ref normale  + Pred normale  (bon)   : {tn}")
    if (tp + fn) > 0:
        print(f"  Recall sur cas anormaux              : {tp/(tp+fn):.3f}")

    print("\n=== Diversite / longueur ===")
    print(f"Predictions distinctes : {df['prediction'].map(norm).nunique()} / {n}")
    print(f"Longueur moy pred (mots) : {df['prediction'].map(lambda s: len(str(s).split())).mean():.1f}")
    print(f"Longueur moy ref  (mots) : {df['reference'].map(lambda s: len(str(s).split())).mean():.1f}")

    top = df["prediction"].map(norm).value_counts().head(3)
    print("\nPredictions les plus frequentes :")
    for txt, cnt in top.items():
        print(f"  {cnt}x  {txt[:80]}...")

    print("\n" + "=" * 70)
    print("EXEMPLES - reference contient une anomalie")
    print("=" * 70)
    abn = df[df["_ref_abn"]].head(args.n_examples)
    for _, row in abn.iterrows():
        flag = "PRED DECRIT ANOMALIE" if row["_pred_abn"] else ">>> PRED = NORMAL (rate)"
        print(f"\n[{row['sample_id']}]  {flag}")
        print(f"  REF  : {str(row['reference'])[:180]}")
        print(f"  PRED : {str(row['prediction'])[:180]}")


if __name__ == "__main__":
    main()
