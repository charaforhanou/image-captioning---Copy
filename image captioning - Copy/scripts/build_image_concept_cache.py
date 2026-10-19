"""Genere un concept cache HONNETE : concepts predits depuis l image (DenseNet),
pas extraits du texte du rapport. Elimine la fuite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.concept_utils import ConceptEmbedder

PREDS = ROOT / "data" / "processed" / "concept_preds"
OUT = ROOT / "data" / "processed" / "concept_cache_image"

CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']

NAMES = {c: c.lower() for c in CONDITIONS}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-prob", type=float, default=0.3)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--include-no-finding", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ConceptEmbedder(device=device)

    labels_for_embed = [c for c in CONDITIONS if c != "No Finding" or args.include_no_finding]
    emb = embedder.embed([NAMES[c] for c in labels_for_embed])
    name_to_emb = {c: emb[i].tolist() for i, c in enumerate(labels_for_embed)}

    total_concepts = 0
    total_studies = 0
    for split in ["train", "val", "test"]:
        df = pd.read_csv(PREDS / f"{split}_concept_probs.csv")
        total_studies += len(df)
        for _, row in df.iterrows():
            uid = row["uid"]
            probs = {c: float(row[c]) for c in labels_for_embed}
            selected = [c for c in labels_for_embed if probs[c] >= args.min_prob]
            if not selected:
                selected = sorted(labels_for_embed, key=lambda c: -probs[c])[:args.top_n]
            selected = sorted(selected, key=lambda c: -probs[c])[:args.top_n]

            concepts = []
            for c in selected:
                concepts.append({
                    "text": NAMES[c],
                    "canonical_name": c,
                    "confidence": probs[c],
                    "embedding": name_to_emb[c],
                })
            total_concepts += len(concepts)

            k = len(concepts)
            edges = np.zeros((k, k), dtype=np.float32)
            for i in range(k):
                for j in range(i + 1, k):
                    w = (concepts[i]["confidence"] + concepts[j]["confidence"]) / 2.0
                    edges[i, j] = edges[j, i] = w

            with open(OUT / f"{uid}.json", "w", encoding="utf-8") as f:
                json.dump({"concepts": concepts, "edge_weights": edges.tolist()}, f)

        print(f"{split}: {len(df)} etudes traitees")

    print(f"\nCache honnete ecrit dans {OUT}")
    print(f"Moyenne concepts/etude : {total_concepts / total_studies:.2f}")


if __name__ == "__main__":
    main()
