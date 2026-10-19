"""Évalue un checkpoint MCCAG sur le split test OFFICIEL R2Gen.

Charge un checkpoint, génère les rapports sur les études test (beam search
avec anti-répétition), sauvegarde les prédictions, puis calcule BLEU-1..4 /
ROUGE-L / CIDEr via pycocoevalcap (sans Java). METEOR s'ajoute automatiquement
si Java est installé et --meteor est passé.

Les références sont les `report` OFFICIELS R2Gen (colonne du metadata_r2gen.csv),
donc les scores sont comparables au tableau SOTA (R2Gen, XPRONET, KiUT...).

Déduplication en deux temps : d'abord un rapport par étude (uid), puis un
rapport par référence unique — IU X-Ray contient de nombreux rapports "normaux"
identiques partagés entre études différentes, et les compter plusieurs fois
gonfle artificiellement BLEU/CIDEr.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --max-test-samples 16 --n-bootstrap 100
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --meteor   # si Java installé
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --no-dedup-ref  # désactive la dédup par référence
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentation import build_eval_transform  # noqa: E402
from src.data.dataset import IUXRayDataset, collate_fn  # noqa: E402
from src.data.tokenizer import MedicalTokenizer  # noqa: E402
from src.evaluation.metrics_pycoco import compute_all_metrics_with_ci  # noqa: E402
from src.inference.beam_search import beam_search_decode  # noqa: E402
from src.inference.postprocessing import postprocess_generation  # noqa: E402
from src.models.mccag import MCCAG  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évalue MCCAG sur le test R2Gen (pycocoevalcap).")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    p.add_argument("--max-test-samples", type=int, default=None, help="Sous-échantillonne le test (debug).")
    p.add_argument("--n-bootstrap", type=int, default=None, help="Override evaluation.n_bootstrap.")
    p.add_argument("--beam-size", type=int, default=None, help="Override evaluation.beam_size.")
    p.add_argument("--results-dir", type=str, default=None, help="Override paths.results_dir.")
    p.add_argument("--meteor", action="store_true", help="Inclure METEOR (nécessite Java).")
    p.add_argument("--no-dedup-ref", action="store_true",
                   help="Désactive la déduplication par référence (évalue sur toutes les études uniques par uid).")
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_ref(s: str) -> str:
    """Normalisation légère d'une référence pour détecter les doublons
    (espaces multiples, casse, ponctuation finale)."""
    return re.sub(r"\s+", " ", str(s).strip().lower()).rstrip(".")


@torch.no_grad()
def generate_predictions(model, test_loader, tokenizer, eval_cfg, device) -> pd.DataFrame:
    """Beam search sur tout le test loader. Retourne un DataFrame de prédictions."""
    model.eval()
    n_total = len(test_loader.dataset)
    rows = []
    for batch in test_loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = beam_search_decode(
            model,
            batch,
            tokenizer,
            beam_size=eval_cfg["beam_size"],
            max_length=eval_cfg["max_length"],
            min_length=eval_cfg["min_length"],
            length_penalty=eval_cfg["length_penalty"],
            coverage_penalty=eval_cfg["coverage_penalty"],
            temperature=eval_cfg.get("temperature", 1.0),
        )
        for i, out in enumerate(outputs):
            prediction = postprocess_generation(out["ids"], tokenizer)["full_text"]
            # Référence = report OFFICIEL R2Gen (déjà dans le batch via target_text)
            reference = str(batch["target_text"][i])
            rows.append({
                "sample_id": batch["uid"][i],
                "prediction": prediction,
                "reference": reference,
                "beam_score": out["beam_score"],
                "confidence_mean": out["confidence_mean"],
            })
        logger.info(f"Généré {len(rows)}/{n_total} prédictions")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(ROOT / config["dataset_config"])
    model_config = load_yaml(ROOT / config["model_config"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger.info(f"Device : {device}")

    tokenizer = MedicalTokenizer.load(
        ROOT / dataset_config["processed"]["tokenizer_dir"],
        max_length=dataset_config["text"]["max_length"],
    )
    logger.info(f"Tokenizer chargé (vocab_size={tokenizer.vocab_size})")

    image_cfg = dataset_config["image"]
    eval_transform = build_eval_transform(image_cfg["size"], image_cfg["mean"], image_cfg["std"])

    test_dataset = IUXRayDataset(
        metadata_csv=ROOT / dataset_config["processed"]["metadata_csv"],
        split_file="test",
        images_dir=ROOT / dataset_config["raw"]["images_dir"],
        concept_cache_dir=ROOT / dataset_config["processed"]["concept_cache_dir"],
        concept_embeddings_path=ROOT / dataset_config["processed"]["concept_embeddings"],
        tokenizer=tokenizer,
        transform=eval_transform,
        max_concepts=model_config["concepts"]["max_concepts"],
        concept_dim=model_config["concepts"]["hidden_dim"],
    )
    if args.max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(args.max_test_samples, len(test_dataset))))

    eval_cfg = config["evaluation"]
    if args.n_bootstrap is not None:
        eval_cfg["n_bootstrap"] = args.n_bootstrap
    if args.beam_size is not None:
        eval_cfg["beam_size"] = args.beam_size

    # num_workers=0 : évite les deadlocks du DataLoader sous Windows.
    num_workers = eval_cfg.get("num_workers", 0)
    if sys.platform.startswith("win") and num_workers != 0:
        logger.info(f"Windows détecté : num_workers forcé à 0 (était {num_workers}) pour éviter les blocages.")
        num_workers = 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    logger.info(f"Test : {len(test_dataset)} échantillons ({len(test_loader)} batches)")

    model_config["decoder"]["vocab_size"] = tokenizer.vocab_size
    model = MCCAG(model_config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id, pretrained=False)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    logger.info(f"Chargement du checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=str(device), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    # --- Génération ---
    df = generate_predictions(model, test_loader, tokenizer, eval_cfg, device)

    results_dir = Path(args.results_dir) if args.results_dir else ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "test_predictions_r2gen.csv"
    df.to_csv(pred_path, index=False)
    logger.info(f"Prédictions écrites : {pred_path}")

    # --- Déduplication en DEUX temps (protocole comparable au SOTA) ---
    # 1) un rapport par étude (uid) : garde le meilleur beam_score
    if "beam_score" in df.columns:
        df = df.sort_values("beam_score", ascending=False)
    df = df.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
    logger.info(f"Études uniques par uid : {len(df)}")

    # 2) un rapport par RÉFÉRENCE unique : IU X-Ray contient de nombreux
    #    rapports "normaux" identiques partagés entre études différentes.
    #    Les compter plusieurs fois gonfle BLEU/CIDEr.
    if not args.no_dedup_ref:
        df["_ref_norm"] = df["reference"].map(_norm_ref)
        n_before = len(df)
        df = df.drop_duplicates(subset="_ref_norm", keep="first").reset_index(drop=True)
        df = df.drop(columns=["_ref_norm"])
        logger.info(
            f"Références uniques évaluées : {len(df)} "
            f"(retiré {n_before - len(df)} doublons de référence)"
        )
    else:
        logger.info("Déduplication par référence DÉSACTIVÉE (--no-dedup-ref).")

    # --- Métriques pycocoevalcap ---
    metrics = compute_all_metrics_with_ci(
        df["prediction"].tolist(),
        df["reference"].tolist(),
        n_bootstrap=eval_cfg.get("n_bootstrap", 1000),
        confidence=eval_cfg.get("confidence", 0.95),
        seed=eval_cfg.get("seed", 42),
        use_meteor=args.meteor,
    )

    # --- Rapport ---
    label = {
        "bleu1": "BLEU-1", "bleu2": "BLEU-2", "bleu3": "BLEU-3", "bleu4": "BLEU-4",
        "meteor": "METEOR", "rougeL_f1": "ROUGE-L", "cider": "CIDEr",
    }
    order = ["bleu1", "bleu2", "bleu3", "bleu4"]
    if args.meteor:
        order.append("meteor")
    order += ["rougeL_f1", "cider"]

    dedup_note = "dédup uid + référence" if not args.no_dedup_ref else "dédup uid seulement"
    lines = [
        "MCCAG — évaluation test R2Gen (protocole pycocoevalcap)",
        f"Études évaluées : {len(df)}  ({dedup_note})",
        f"Bootstrap : {eval_cfg.get('n_bootstrap', 1000)}  |  IC : {int(eval_cfg.get('confidence', 0.95)*100)}%",
        f"METEOR : {'inclus' if args.meteor else 'omis (Java requis)'}",
        "",
        f"{'Metric':<10}{'Mean':>10}{'95% CI':>26}",
        "-" * 46,
    ]
    for k in order:
        m = metrics[k]
        lines.append(f"{label[k]:<10}{m['mean']:>10.4f}     [{m['ci_low']:.4f}, {m['ci_high']:.4f}]")

    report = "\n".join(lines)
    print("\n" + report)
    out_path = results_dir / "metrics_r2gen.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    logger.info(f"Métriques écrites : {out_path}")


if __name__ == "__main__":
    main()