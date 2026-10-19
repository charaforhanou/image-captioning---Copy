
import argparse
import re
from pathlib import Path

import pandas as pd


def norm_ref(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


def clean_for_chexbert(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-dedup-ref", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pred)
    print(f"Lignes brutes : {len(df)}")

    if "beam_score" in df.columns:
        df = df.sort_values("beam_score", ascending=False)
    df = df.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)

    if not args.no_dedup_ref:
        df["_r"] = df["reference"].map(norm_ref)
        df = df.drop_duplicates(subset="_r", keep="first").reset_index(drop=True)
        df = df.drop(columns=["_r"])
    print(f"Cas retenus : {len(df)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.sort_values("sample_id").reset_index(drop=True)

    preds = pd.DataFrame({"Report Impression": df["prediction"].map(clean_for_chexbert)})
    refs = pd.DataFrame({"Report Impression": df["reference"].map(clean_for_chexbert)})

    df["sample_id"].reset_index(drop=True).to_csv(
        out_dir / "sample_ids.csv", index=False, header=["sample_id"])

    preds_path = out_dir / "preds_for_chexbert.csv"
    refs_path = out_dir / "refs_for_chexbert.csv"
    preds.to_csv(preds_path, index=False)
    refs.to_csv(refs_path, index=False)

    print(f"Ecrit : {preds_path}  ({len(preds)} lignes)")
    print(f"Ecrit : {refs_path}   ({len(refs)} lignes)")


if __name__ == "__main__":
    main()
