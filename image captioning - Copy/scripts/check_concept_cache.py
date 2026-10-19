import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/processed/metadata_r2gen.csv"
CACHE_DIR = ROOT / "data/processed/concept_cache"

df = pd.read_csv(META, dtype={"uid": str})
uids_needed = set(df["uid"])
print(f"uid requis par le split R2Gen : {len(uids_needed)}")

# 1. Combien de fichiers cache existent au total ?
cache_files = list(CACHE_DIR.glob("*.json"))
print(f"Fichiers .json dans le cache : {len(cache_files)}")

# 2. Combien de uid R2Gen ont leur cache ?
present = {uid for uid in uids_needed if (CACHE_DIR / f"{uid}.json").is_file()}
missing = uids_needed - present
print(f"uid R2Gen AVEC cache : {len(present)} / {len(uids_needed)}")
print(f"uid R2Gen SANS cache : {len(missing)}")
if missing:
    print("  exemples manquants :", list(missing)[:10])

# 3. Inspecter le contenu d'un cache pour voir s'il est bien forme
print("\n=== Contenu d'un fichier cache (exemple) ===")
sample_uid = next(iter(present), None)
if sample_uid:
    with open(CACHE_DIR / f"{sample_uid}.json", encoding="utf-8") as f:
        c = json.load(f)
    print(f"uid: {sample_uid}")
    print(f"  cles: {list(c.keys())}")
    concepts = c.get("concepts", [])
    print(f"  nb concepts: {len(concepts)}")
    if concepts:
        print(f"  1er concept: {concepts[0]}")
    ew = c.get("edge_weights")
    if ew:
        print(f"  edge_weights: {len(ew)}x{len(ew[0]) if ew else 0}")

# 4. Date de modif des fichiers cache (pour voir s'ils sont recents ou anciens)
import datetime
if cache_files:
    times = sorted((f.stat().st_mtime for f in cache_files))
    oldest = datetime.datetime.fromtimestamp(times[0])
    newest = datetime.datetime.fromtimestamp(times[-1])
    print(f"\n=== Dates des fichiers cache ===")
    print(f"  plus ancien : {oldest}")
    print(f"  plus recent : {newest}")