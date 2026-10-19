"""Clinical Efficacy (CE) — évalue les prédictions au niveau des PATHOLOGIES,
pas des n-grammes. C'est la métrique attendue par les venues RRG sérieuses
(le tableau SOTA R2Gen/XPRONET/KiUT la rapporte).

Étiquette chaque prédiction et chaque référence avec les 14 labels CheXpert,
puis calcule precision / recall / F1 (micro et macro) entre les deux.

Deux backends possibles pour l'étiquetage :
  1. CheXbert (le standard, mais nécessite le modèle + torch)   --labeler chexbert
  2. Un labeler à base de règles/négation (rapide, approximatif)  --labeler rules

Le backend "rules" est un repli honnête quand CheXbert n'est pas installé :
il te donne un ordre de grandeur, pas le chiffre officiel. Pour un papier,
utilise CheXbert.

Usage :
    python scripts/clinical_efficacy.py --pred results/ablations/full/eval/test_predictions_r2gen.csv --labeler rules
    python scripts/clinical_efficacy.py --pred ... --labeler chexbert --chexbert-checkpoint path/to/chexbert.pth
"""

import argparse
from pathlib import Path

import pandas as pd

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    "No Finding",
]

# --- Backend "rules" : mots-clés + négation ----------------------------------
RULE_KEYWORDS = {
    "Cardiomegaly": ["cardiomegaly", "enlarged heart", "heart size enlarged", "cardiac enlargement"],
    "Enlarged Cardiomediastinum": ["mediastinal widening", "widened mediastinum", "enlarged cardiomediastin"],
    "Lung Opacity": ["opacity", "opacities", "airspace disease", "airspace opacity"],
    "Lung Lesion": ["nodule", "mass", "lesion"],
    "Edema": ["edema", "vascular congestion", "pulmonary congestion"],
    "Consolidation": ["consolidation", "consolidative"],
    "Pneumonia": ["pneumonia", "infectious", "infection"],
    "Atelectasis": ["atelectasis", "atelectatic", "volume loss"],
    "Pneumothorax": ["pneumothorax"],
    "Pleural Effusion": ["pleural effusion", "effusion", "pleural fluid"],
    "Pleural Other": ["pleural thickening", "pleural scarring", "fibrothorax"],
    "Fracture": ["fracture"],
    "Support Devices": ["tube", "catheter", "pacemaker", "line", "picc", "port", "device"],
}
NEG_CUES = ["no ", "without ", "free of ", "negative for ", "clear of ",
            "absence of ", "no evidence of ", "resolution of ", "rule out"]


def rules_label(text: str) -> dict:
    """Retourne un vecteur {label: 1/0} via mots-clés + fenêtre de négation."""
    import re
    t = " " + str(text).lower() + " "
    vec = {lab: 0 for lab in CHEXPERT_LABELS}
    positive_any = False
    for lab, kws in RULE_KEYWORDS.items():
        for kw in kws:
            for m in re.finditer(re.escape(kw), t):
                window = t[max(0, m.start() - 25):m.start()]
                if not any(neg in window for neg in NEG_CUES):
                    vec[lab] = 1
                    positive_any = True
    vec["No Finding"] = 0 if positive_any else 1
    return vec


def chexbert_label(texts, checkpoint):
    """Étiquette une liste de textes avec CheXbert. Retourne une liste de dicts."""
    try:
        import torch
        from transformers import BertTokenizer, BertModel  # noqa: F401
    except ImportError:
        raise SystemExit("CheXbert nécessite torch + transformers. "
                         "Installe-les ou utilise --labeler rules.")
    # NB : l'intégration complète de CheXbert dépend de l'implémentation que tu
    # utilises (stanford-crfm/CheXbert). Le plus simple est d'appeler leur
    # script label.py sur un CSV et de relire la sortie. On délègue ici :
    raise SystemExit(
        "Pour CheXbert, utilise le repo officiel stanford-crfm/CheXbert :\n"
        "  1. Écris tes prédictions et références en 2 CSV (colonne 'Report Impression').\n"
        "  2. Lance leur src/label.py sur chacun -> 2 CSV de labels.\n"
        "  3. Relance ce script avec --labeler from-csv --pred-labels ... --ref-labels ...\n"
        "Le backend 'rules' ci-dessous donne une estimation en attendant."
    )


def vectorize(df_labels):
    """DataFrame de dicts -> matrice binaire (lignes × 14 labels), présence positive."""
    import numpy as np
    M = np.zeros((len(df_labels), len(CHEXPERT_LABELS)), dtype=int)
    for i, vec in enumerate(df_labels):
        for j, lab in enumerate(CHEXPERT_LABELS):
            M[i, j] = 1 if vec.get(lab, 0) == 1 else 0
    return M


def prf(y_true, y_pred):
    """Precision/Recall/F1 micro et macro à partir de 2 matrices binaires."""
    from sklearn.metrics import precision_recall_fscore_support
    # micro : agrège sur tous les labels
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    # macro : moyenne des F1 par label
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    # par label
    p_l, r_l, f_l, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
        labels=list(range(len(CHEXPERT_LABELS))))
    return (p_mi, r_mi, f_mi), (p_ma, r_ma, f_ma), (p_l, r_l, f_l, sup)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--labeler", default="rules", choices=["rules", "chexbert", "from-csv"])
    p.add_argument("--chexbert-checkpoint", default=None)
    p.add_argument("--pred-labels", default=None, help="CSV de labels CheXbert des prédictions (from-csv)")
    p.add_argument("--ref-labels", default=None, help="CSV de labels CheXbert des références (from-csv)")
    p.add_argument("--dedup-by-ref", action="store_true",
                   help="Dédupliquer par référence avant de calculer la CE.")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.pred)

    if args.dedup_by_ref:
        import re
        df["_r"] = df["reference"].map(lambda s: re.sub(r"\s+", " ", str(s).strip().lower()).rstrip("."))
        df = df.drop_duplicates(subset="_r", keep="first").reset_index(drop=True)
        print(f"Dédupliqué par référence : {len(df)} cas")

    if args.labeler == "rules":
        pred_vecs = [rules_label(t) for t in df["prediction"]]
        ref_vecs = [rules_label(t) for t in df["reference"]]
        print("Labeler : RULES (estimation — pour un papier, utilise CheXbert)")
    elif args.labeler == "chexbert":
        pred_vecs = chexbert_label(df["prediction"].tolist(), args.chexbert_checkpoint)
        ref_vecs = chexbert_label(df["reference"].tolist(), args.chexbert_checkpoint)
    else:  # from-csv
        raise SystemExit("from-csv : à brancher sur la sortie de CheXbert (voir docstring).")

    y_pred = vectorize(pred_vecs)
    y_true = vectorize(ref_vecs)

    (p_mi, r_mi, f_mi), (p_ma, r_ma, f_ma), (p_l, r_l, f_l, sup) = prf(y_true, y_pred)

    print("\n=== Clinical Efficacy (14 labels CheXpert) ===")
    print(f"{'':<28}{'P':>8}{'R':>8}{'F1':>8}")
    print(f"{'MICRO':<28}{p_mi:>8.4f}{r_mi:>8.4f}{f_mi:>8.4f}")
    print(f"{'MACRO':<28}{p_ma:>8.4f}{r_ma:>8.4f}{f_ma:>8.4f}")
    print("\n--- Par pathologie ---")
    print(f"{'Label':<28}{'P':>8}{'R':>8}{'F1':>8}{'n':>6}")
    for j, lab in enumerate(CHEXPERT_LABELS):
        print(f"{lab:<28}{p_l[j]:>8.4f}{r_l[j]:>8.4f}{f_l[j]:>8.4f}{int(sup[j]):>6}")

    print("\nLes venues RRG rapportent surtout le F1 MICRO et le F1 MACRO.")
    print("Compare-les aux baselines : R2Gen/XPRONET publient ces chiffres sur IU X-Ray.")


if __name__ == "__main__":
    main()