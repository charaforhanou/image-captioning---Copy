"""Build all processed-data artifacts for MCCAG from the raw IU X-Ray dataset.

Produces:
  - data/processed/metadata.csv          (image, uid, split, findings, impression, target_text)
  - data/splits/{train,val,test}.txt      (one study uid per line)
  - data/processed/concept_cache/<uid>.json   (concepts + KxK edge_weights)
  - data/processed/concept_embeddings.pt      (cui_to_idx + (N,768) embeddings)
  - data/processed/cooccurrence.json           (train-split CUI-pair co-occurrence counts)
  - data/processed/tokenizer/tokenizer.json    (trained byte-level BPE tokenizer)

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --skip-concepts          # metadata/splits/tokenizer only
    python scripts/prepare_data.py --max-reports 50          # smoke test on a subset
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.concept_utils import (  # noqa: E402
    ConceptEmbedder,
    build_edge_weight_matrix,
    compute_cooccurrence,
    extract_concepts_from_doc,
    get_scispacy_pipeline,
)
from src.data.preprocessing import build_metadata, make_splits  # noqa: E402
from src.data.tokenizer import MedicalTokenizer  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402

logger = get_logger(__name__, log_file=ROOT / "logs" / "prepare_data.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MCCAG processed-data artifacts")
    parser.add_argument("--config", default=str(ROOT / "config" / "dataset_config.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-concepts",
        action="store_true",
        help="Skip concept extraction/embeddings/cooccurrence (metadata, splits, tokenizer only).",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=None,
        help="If set, only run concept extraction on the first N unique reports (smoke test).",
    )
    return parser.parse_args()


def build_metadata_and_splits(config: dict) -> tuple:
    raw, processed, splits_cfg = config["raw"], config["processed"], config["splits"]

    metadata = build_metadata(
        images_dir=ROOT / raw["images_dir"],
        reports_dir=ROOT / raw["reports_dir"],
        captions_csv=ROOT / raw["captions_csv"],
    )

    split_map = make_splits(
        metadata["uid"].tolist(),
        train_ratio=splits_cfg["train_ratio"],
        val_ratio=splits_cfg["val_ratio"],
        test_ratio=splits_cfg["test_ratio"],
        seed=splits_cfg["seed"],
    )
    metadata["split"] = metadata["uid"].map(split_map)

    processed_root = ROOT / processed["root"]
    processed_root.mkdir(parents=True, exist_ok=True)
    metadata_csv = ROOT / processed["metadata_csv"]
    metadata.to_csv(metadata_csv, index=False)
    logger.info(f"Wrote {metadata_csv} ({len(metadata)} images, {metadata['uid'].nunique()} studies)")

    splits_dir = ROOT / splits_cfg["dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, file_key in [("train", "train_file"), ("val", "val_file"), ("test", "test_file")]:
        uids = sorted(uid for uid, s in split_map.items() if s == split_name)
        split_path = ROOT / splits_cfg[file_key]
        split_path.write_text("\n".join(uids) + "\n", encoding="utf-8")
        logger.info(f"Wrote {split_path} ({len(uids)} studies)")

    return metadata, split_map


def extract_all_concepts(metadata, split_map, config: dict, device: str, max_reports) -> dict:
    concepts_cfg = config["concepts"]

    unique_reports = metadata.drop_duplicates("uid")[["uid", "findings", "impression"]]
    if max_reports is not None:
        unique_reports = unique_reports.head(max_reports)
    uids = unique_reports["uid"].tolist()
    texts = [
        f"{row.findings} {row.impression}".strip() for row in unique_reports.itertuples()
    ]

    logger.info(f"Loading scispacy pipeline ({concepts_cfg['spacy_model']}) + UMLS linker...")
    nlp = get_scispacy_pipeline(concepts_cfg["spacy_model"], concepts_cfg["linker_threshold"])

    allowed_tuis = concepts_cfg["allowed_tuis"]
    max_concepts = concepts_cfg["max_concepts"]
    threshold = concepts_cfg["linker_threshold"]

    all_concepts = {}
    logger.info(f"Extracting concepts for {len(uids)} unique reports...")
    for uid, doc in tqdm(zip(uids, nlp.pipe(texts, batch_size=32)), total=len(uids), desc="concepts"):
        all_concepts[uid] = extract_concepts_from_doc(
            doc, nlp, allowed_tuis, threshold=threshold, max_concepts=max_concepts
        )

    return all_concepts


def write_cooccurrence_and_caches(all_concepts, split_map, config: dict) -> None:
    processed = config["processed"]
    concepts_cfg = config["concepts"]

    train_concepts = {uid: c for uid, c in all_concepts.items() if split_map.get(uid) == "train"}
    cooccurrence = compute_cooccurrence(train_concepts, threshold=concepts_cfg["cooccurrence_threshold"])
    max_cooc = max(cooccurrence.values(), default=1)
    logger.info(f"Found {len(cooccurrence)} CUI co-occurrence pairs >= {concepts_cfg['cooccurrence_threshold']} (max={max_cooc})")

    cooc_path = ROOT / processed["cooccurrence_file"]
    cooc_json = {f"{a}|{b}": count for (a, b), count in cooccurrence.items()}
    cooc_path.write_text(json.dumps(cooc_json, indent=2), encoding="utf-8")
    logger.info(f"Wrote {cooc_path}")

    cache_dir = ROOT / processed["concept_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    for uid, concepts in tqdm(all_concepts.items(), desc="edge_weights"):
        edge_weights = build_edge_weight_matrix(concepts, cooccurrence, max_cooc)
        cache = {"uid": uid, "concepts": concepts, "edge_weights": edge_weights.tolist()}
        with open(cache_dir / f"{uid}.json", "w", encoding="utf-8") as f:
            json.dump(cache, f)
    logger.info(f"Wrote {len(all_concepts)} concept cache files to {cache_dir}")


def build_concept_embeddings(all_concepts, config: dict, device: str) -> None:
    processed, concepts_cfg = config["processed"], config["concepts"]

    unique_cuis = {}
    for concepts in all_concepts.values():
        for concept in concepts:
            unique_cuis[concept["cui"]] = concept["canonical_name"]

    cuis = sorted(unique_cuis)
    names = [unique_cuis[cui] for cui in cuis]
    logger.info(f"Embedding {len(cuis)} unique concepts with PubMedBERT...")

    embedder = ConceptEmbedder(concepts_cfg["embedding_model"], device=device)
    embeddings = embedder.embed(names) if names else torch.zeros(0, concepts_cfg["embedding_dim"])

    cui_to_idx = {cui: i for i, cui in enumerate(cuis)}
    out_path = ROOT / processed["concept_embeddings"]
    torch.save({"cui_to_idx": cui_to_idx, "embeddings": embeddings}, out_path)
    logger.info(f"Wrote {out_path} ({embeddings.shape})")


def train_tokenizer(metadata, split_map, config: dict) -> None:
    processed, text_cfg = config["processed"], config["text"]

    train_uids = {uid for uid, s in split_map.items() if s == "train"}
    train_texts = (
        metadata[metadata["uid"].isin(train_uids)]["target_text"]
        .drop_duplicates()
        .tolist()
    )
    logger.info(f"Training BPE tokenizer on {len(train_texts)} unique training reports...")

    tokenizer = MedicalTokenizer.train(
        train_texts,
        vocab_size=text_cfg["vocab_size"],
        special_tokens=text_cfg["special_tokens"],
        max_length=text_cfg["max_length"],
    )
    out_dir = ROOT / processed["tokenizer_dir"]
    tokenizer.save(out_dir)
    logger.info(f"Saved tokenizer (vocab_size={tokenizer.vocab_size}) to {out_dir}")


def main() -> None:
    args = parse_args()
    set_seed(42)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = str(get_device(args.device))

    metadata, split_map = build_metadata_and_splits(config)

    if not args.skip_concepts:
        all_concepts = extract_all_concepts(metadata, split_map, config, device, args.max_reports)
        write_cooccurrence_and_caches(all_concepts, split_map, config)
        build_concept_embeddings(all_concepts, config, device)
    else:
        logger.info("Skipping concept extraction (--skip-concepts)")

    train_tokenizer(metadata, split_map, config)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
