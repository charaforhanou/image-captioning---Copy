"""Single-image inference: load a checkpoint and generate a radiology report.

Without a ``report_context``, the concept pathway runs with an empty/zero
concept set (the model still has the visual stream, memory, and decoder).
If ``report_context`` text is supplied (e.g. the INDICATION/COMPARISON
section of a real report, or clinician-provided context), concepts are
extracted on-the-fly with the same scispacy + UMLS + PubMedBERT pipeline used
by ``scripts/prepare_data.py`` (see :mod:`src.data.concept_utils`), and the
concept graph's edge weights are built from the cached training-split
co-occurrence statistics.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image

from src.data.tokenizer import MedicalTokenizer
from src.inference.beam_search import beam_search_decode
from src.inference.postprocessing import postprocess_generation
from src.models.mccag import MCCAG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Generates a report for a single chest X-ray image.

    Args:
        model: A :class:`MCCAG` instance with trained weights loaded.
        tokenizer: A loaded :class:`MedicalTokenizer`.
        transform: Eval-time image transform (see
            :func:`src.data.augmentation.build_eval_transform`).
        device: Device to run inference on.
        max_concepts: Concept-slot count (K).
        concept_dim: Concept embedding dimensionality.
        concepts_cfg: The ``concepts`` section of ``config/dataset_config.yaml``
            (used only when ``report_context`` is supplied).
        cooccurrence_path: Path to ``data/processed/cooccurrence.json``
            (used only when ``report_context`` is supplied).
    """

    def __init__(
        self,
        model: MCCAG,
        tokenizer: MedicalTokenizer,
        transform,
        device: torch.device,
        max_concepts: int,
        concept_dim: int,
        concepts_cfg: Optional[dict] = None,
        cooccurrence_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.max_concepts = max_concepts
        self.concept_dim = concept_dim
        self.concepts_cfg = concepts_cfg
        self.cooccurrence_path = Path(cooccurrence_path) if cooccurrence_path else None

        self._nlp = None
        self._embedder = None

    def _empty_concepts(self) -> tuple:
        embeddings = torch.zeros(1, self.max_concepts, self.concept_dim)
        mask = torch.zeros(1, self.max_concepts, dtype=torch.bool)
        edge_weights = torch.zeros(1, self.max_concepts, self.max_concepts)
        return embeddings, mask, edge_weights

    def _concepts_from_text(self, text: str) -> tuple:
        from src.data.concept_utils import ConceptEmbedder, build_edge_weight_matrix, extract_concepts, get_scispacy_pipeline

        if self._nlp is None:
            self._nlp = get_scispacy_pipeline(
                self.concepts_cfg["spacy_model"], linker_threshold=self.concepts_cfg["linker_threshold"]
            )
        if self._embedder is None:
            self._embedder = ConceptEmbedder(self.concepts_cfg["embedding_model"], device=self.device.type)

        concepts = extract_concepts(
            text,
            self._nlp,
            allowed_tuis=self.concepts_cfg["allowed_tuis"],
            threshold=self.concepts_cfg["linker_threshold"],
            max_concepts=self.max_concepts,
        )
        if not concepts:
            return self._empty_concepts()

        cooccurrence: Dict[tuple, int] = {}
        if self.cooccurrence_path and self.cooccurrence_path.is_file():
            raw = json.loads(self.cooccurrence_path.read_text(encoding="utf-8"))
            cooccurrence = {tuple(key.split("|")): count for key, count in raw.items()}
        max_cooc = max(cooccurrence.values(), default=1)

        names = [c["canonical_name"] for c in concepts]
        concept_embeddings = self._embedder.embed(names)
        edge_weights_np = build_edge_weight_matrix(concepts, cooccurrence, max_cooc)

        k = len(concepts)
        embeddings = torch.zeros(1, self.max_concepts, self.concept_dim)
        embeddings[0, :k] = concept_embeddings
        mask = torch.zeros(1, self.max_concepts, dtype=torch.bool)
        mask[0, :k] = True
        edge_weights = torch.zeros(1, self.max_concepts, self.max_concepts)
        edge_weights[0, :k, :k] = torch.from_numpy(edge_weights_np)

        return embeddings, mask, edge_weights

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path], report_context: Optional[str] = None, beam_cfg: Optional[dict] = None) -> Dict:
        """Generate a report for ``image_path``.

        Args:
            image_path: Path to a chest X-ray image (PNG/JPG).
            report_context: Optional free text used to seed the concept
                pathway (see module docstring). If ``None``, the concept
                stream is empty.
            beam_cfg: Keyword args forwarded to
                :func:`src.inference.beam_search.beam_search_decode`
                (``beam_size, max_length, min_length, length_penalty,
                coverage_penalty, temperature``).

        Returns:
            Dict with ``findings``, ``impression``, ``full_text``,
            ``beam_score``, ``confidence_mean``, ``concept_coverage``.
        """
        image = Image.open(image_path).convert("RGB")
        image_t = self.transform(image).unsqueeze(0).to(self.device)

        if report_context:
            concept_embeddings, concept_mask, edge_weights = self._concepts_from_text(report_context)
        else:
            concept_embeddings, concept_mask, edge_weights = self._empty_concepts()

        batch = {
            "image": image_t,
            "concept_embeddings": concept_embeddings.to(self.device),
            "concept_mask": concept_mask.to(self.device),
            "edge_weights": edge_weights.to(self.device),
        }

        outputs = beam_search_decode(self.model, batch, self.tokenizer, **(beam_cfg or {}))
        result = postprocess_generation(outputs[0]["ids"], self.tokenizer)
        result["beam_score"] = outputs[0]["beam_score"]
        result["confidence_mean"] = outputs[0]["confidence_mean"]
        result["concept_coverage"] = outputs[0]["concept_coverage"]
        return result
