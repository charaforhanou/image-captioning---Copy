import pandas as pd
import re
from pathlib import Path

META = r"data\processed\metadata_r2gen.csv"
OUT = Path(r"data\processed\concept_labels")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META)
print(f"Total etudes : {len(df)}")

def clean(s):
    return re.sub(r"\s+", " ", str(s).strip())

for split in ["train", "val", "test"]:
    sub = df[df["split"] == split].copy().reset_index(drop=True)
    # une ligne par etude (uid), dans l'ordre du metadata
    reports = pd.DataFrame({"Report Impression": sub["report"].map(clean)})
    reports.to_csv(OUT / f"{split}_reports.csv", index=False)
    # table uid alignee (meme ordre) pour rattacher les labels ensuite
    sub[["uid"]].to_csv(OUT / f"{split}_uids.csv", index=False)
    print(f"{split:<6}: {len(sub)} etudes -> {split}_reports.csv")

print("\nOK. Fichiers dans data\\processed\\concept_labels\\")
