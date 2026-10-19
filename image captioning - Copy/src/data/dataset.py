"""IU X-Ray dataset and batch collation for MCCAG training/evaluation.

R2Gen-aligned variant (Chen et al. 2020 official split):
  - reads ``data/processed/metadata_r2gen.csv`` (columns: study_id, uid,
    image_path_list, image, report, split) produced by
    ``scripts/build_r2gen_metadata.py`` ;
  - the generation target is the OFFICIAL R2Gen ``report`` (plain text, no
    FINDINGS/IMPRESSION section tags), so scores are comparable to published
    baselines ;
  - splits are selected via the ``split`` column (no separate .txt files).

The ``split_file`` argument is kept for signature compatibility: pass the
split NAME ("train"/"val"/"test") or a path whose stem is one of those.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.tokenizer import MedicalTokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_SPLITS = {"train", "val", "test"}


def _resolve_split_name(split_file: Union[str, Path]) -> str:
    """Accept 'train' / a path like 'data/splits/train.txt' and return 'train'."""
    s = str(split_file).strip()
    if s in _VALID_SPLITS:
        return s
    stem = Path(s).stem  # 'train.txt' -> 'train'
    if stem in _VALID_SPLITS:
        return stem
    raise ValueError(
        f"Cannot resolve split from {split_file!r}; pass 'train', 'val', or 'test'."
    )


class IUXRayDataset(Dataset):
    """Loads images, target token sequences, and cached concept graphs.

    Args:
        metadata_csv: Path to ``data/processed/metadata_r2gen.csv``.
        split_file: Split name ("train"/"val"/"test") or a path whose stem is
            one of those (kept for backward-compatible signature).
        images_dir: Directory of source PNG images.
        concept_cache_dir: Directory of per-uid concept cache JSON files.
        concept_embeddings_path: Path to ``concept_embeddings.pt`` (dict with
            ``cui_to_idx`` and ``embeddings``).
        tokenizer: A loaded :class:`MedicalTokenizer`.
        transform: Image transform (see :mod:`src.data.augmentation`).
        max_concepts: Per-sample concept slot count (K).
        concept_dim: Concept embedding dimensionality.
    """

    def __init__(
        self,
        metadata_csv: Union[str, Path],
        split_file: Union[str, Path],
        images_dir: Union[str, Path],
        concept_cache_dir: Union[str, Path],
        concept_embeddings_path: Union[str, Path],
        tokenizer: MedicalTokenizer,
        transform: Callable,
        max_concepts: int = 50,
        concept_dim: int = 768,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.concept_cache_dir = Path(concept_cache_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_concepts = max_concepts
        self.concept_dim = concept_dim

        split_name = _resolve_split_name(split_file)
        metadata = pd.read_csv(metadata_csv, dtype={"uid": str})
        if "split" not in metadata.columns:
            raise ValueError(
                f"{metadata_csv} has no 'split' column. Did you point to "
                f"metadata_r2gen.csv built by scripts/build_r2gen_metadata.py?"
            )
        self.metadata = metadata[metadata["split"] == split_name].reset_index(drop=True)
        logger.info(f"Loaded {len(self.metadata)} samples for split '{split_name}'")

        embeddings_data = torch.load(concept_embeddings_path, weights_only=False)
        self.cui_to_idx: Dict[str, int] = embeddings_data["cui_to_idx"]
        self.embeddings: torch.Tensor = embeddings_data["embeddings"]
        
        import pandas as _pd
        self.pathology_labels = {}
        self._pathology_cols = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
            "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices", "No Finding",
        ]
        _tgt_path = Path("data/processed/concept_labels") / f"{split_name}_concept_targets.csv"
        if _tgt_path.is_file():
            _tgt = _pd.read_csv(_tgt_path, dtype={"uid": str})
            for _, _r in _tgt.iterrows():
                self.pathology_labels[str(_r["uid"])] = torch.tensor(
                    [float(_r[c]) for c in self._pathology_cols], dtype=torch.float32
                )

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_concepts(self, uid: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load padded concept embeddings, mask, and edge-weight matrix for ``uid``.

        Returns:
            Tuple ``(embeddings (K, D), mask (K,) bool, edge_weights (K, K))``.
            Missing cache -> all-zero (the model handles an empty concept stream).
        """
        embeddings = torch.zeros(self.max_concepts, self.concept_dim)
        mask = torch.zeros(self.max_concepts, dtype=torch.bool)
        edge_weights = torch.zeros(self.max_concepts, self.max_concepts)

        cache_path = self.concept_cache_dir / f"{uid}.json"
        if not cache_path.is_file():
            return embeddings, mask, edge_weights

        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        concepts = cached.get("concepts", [])[: self.max_concepts]
        k = len(concepts)
        for i, concept in enumerate(concepts):
            if "embedding" in concept:
                embeddings[i] = torch.tensor(concept["embedding"], dtype=torch.float32)
            else:
                idx = self.cui_to_idx.get(concept.get("cui"))
                if idx is not None:
                    embeddings[i] = self.embeddings[idx]
            mask[i] = True

        edge_weights_full = cached.get("edge_weights")
        if edge_weights_full and k > 0:
            edges = torch.tensor(edge_weights_full, dtype=torch.float32)
            edge_weights[:k, :k] = edges[:k, :k]

        return embeddings, mask, edge_weights

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        uid = str(row["uid"])

        # Single (primary) view. image_path_list holds both views if you later
        # want to use them; here we load the primary 'image' column.
        image = Image.open(self.images_dir / row["image"]).convert("RGB")
        image = self.transform(image)

        # R2Gen-aligned target: official plain-text report (no section tags).
        target_text = str(row["report"])
        encoded = self.tokenizer.encode(target_text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # No FINDINGS/IMPRESSION tags in R2Gen text -> all-zero section ids.
        section_ids = self.tokenizer.section_ids(input_ids)

        concept_embeddings, concept_mask, edge_weights = self._load_concepts(uid)

        pathology_label = self.pathology_labels.get(
            uid, torch.zeros(len(self._pathology_cols), dtype=torch.float32)
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "section_ids": section_ids,
            "concept_embeddings": concept_embeddings,
            "concept_mask": concept_mask,
            "edge_weights": edge_weights,
            "pathology_label": pathology_label,
            "uid": uid,
            "image_id": str(row["image"]),
            "target_text": target_text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Stack a list of :class:`IUXRayDataset` items into a batch."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "section_ids": torch.stack([item["section_ids"] for item in batch]),
        "concept_embeddings": torch.stack([item["concept_embeddings"] for item in batch]),
        "concept_mask": torch.stack([item["concept_mask"] for item in batch]),
        "edge_weights": torch.stack([item["edge_weights"] for item in batch]),
        "pathology_label": torch.stack([item["pathology_label"] for item in batch]),
        "uid": [item["uid"] for item in batch],
        "image_id": [item["image_id"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
    }