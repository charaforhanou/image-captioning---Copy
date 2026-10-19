"""Réentraîne le tokenizer BPE sur les rapports OFFICIELS R2Gen.

IMPORTANT : le tokenizer est entraîné UNIQUEMENT sur le split TRAIN de R2Gen.
Inclure val/test ferait fuiter du vocabulaire de test dans le modèle, ce qui
invaliderait la comparaison. Le seuil min_frequency=3 reproduit le filtrage des
mots rares de R2Gen (Chen et al. 2020).

Le texte cible est le `report` brut R2Gen (sans balises de section), conforme à
l'option A (alignement total sur le protocole des baselines).

Usage:
    python scripts/rebuild_tokenizer_r2gen.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path

import pandas as pd

from src.data.tokenizer import MedicalTokenizer

META = Path("data/processed/metadata_r2gen.csv")
OUT_DIR = Path("data/processed/tokenizer_r2gen")
VOCAB_SIZE = 8000      # ajuste si besoin ; R2Gen a un petit vocabulaire effectif
MIN_FREQUENCY = 3      # aligné sur le filtrage des mots rares de R2Gen
MAX_LENGTH = 128        # garde la valeur de ton config si différente


def main() -> None:
    df = pd.read_csv(META)
    train_reports = df[df["split"] == "train"]["report"].astype(str).tolist()
    print(f"Rapports d'entraînement (train R2Gen) : {len(train_reports)}")
    if not train_reports:
        raise SystemExit("Aucun rapport train trouvé — vérifie metadata_r2gen.csv.")

    # Aperçu pour vérifier le format du texte
    print("Exemple de rapport :", train_reports[0][:120], "...")

    tok = MedicalTokenizer.train(
        texts=train_reports,
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        max_length=MAX_LENGTH,
    )
    tok.save(OUT_DIR)
    print(f"\nTokenizer sauvegardé dans {OUT_DIR}")
    print(f"Taille du vocabulaire : {tok.vocab_size}")

    # Sanity check : longueur en tokens des rapports (pour calibrer max_length)
    lengths = [len(tok.encode(r)["input_ids"].nonzero()) for r in train_reports[:200]]
    if lengths:
        import numpy as np
        arr = np.array(lengths)
        print(f"Longueur tokens (échantillon 200) : "
              f"moy={arr.mean():.0f}  p95={np.percentile(arr,95):.0f}  max={arr.max()}")
        print("Si p95 << max_length, tu peux réduire max_length pour accélérer.")


if __name__ == "__main__":
    main()