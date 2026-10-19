"""Medical concept extraction and grounding.

Practical substitute for RadGraph + UMLS + MetaMap (which require licensed/
credentialed access unavailable in this environment): uses scispacy
``en_core_sci_md`` for NER and scispacy's UMLS ``EntityLinker`` for CUI
grounding, filtered to the semantic types specified in the architecture spec
(T047, T048, T190, T020, T023). Concept embeddings come from PubMedBERT
(mean-pooled, 768-d) rather than a dedicated UMLS embedding table.

Output schema per concept matches the spec:
``{text, cui, canonical_name, semantic_type, confidence, sentence_idx}``.
"""

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Human-readable names for the allowed UMLS semantic types (TUIs). The
# cached "semantic_type" field stores the TUI code itself.
TUI_NAMES: Dict[str, str] = {
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T190": "Anatomical Abnormality",
    "T020": "Acquired Abnormality",
    "T023": "Body Part, Organ, or Organ Component",
}

ANATOMY_TUI = "T023"

# Curated chest-anatomy adjacency table (substring-matched against UMLS
# canonical names, lowercased). Used for the "anatomical proximity" edge type.
ANATOMY_ADJACENCY: Dict[str, List[str]] = {
    "lung": ["pleura", "mediastinum", "diaphragm", "hilum", "bronchus", "rib", "chest wall"],
    "pleura": ["lung", "rib", "diaphragm", "chest wall"],
    "mediastinum": ["lung", "heart", "trachea", "aorta", "spine", "thymus"],
    "heart": ["mediastinum", "diaphragm", "pericardium", "aorta"],
    "diaphragm": ["lung", "heart", "pleura", "costophrenic", "abdomen"],
    "trachea": ["mediastinum", "bronchus", "larynx"],
    "bronchus": ["lung", "trachea", "hilum"],
    "hilum": ["lung", "bronchus", "mediastinum"],
    "rib": ["lung", "pleura", "spine", "chest wall"],
    "spine": ["mediastinum", "rib"],
    "aorta": ["mediastinum", "heart"],
    "costophrenic": ["diaphragm", "pleura", "lung"],
    "chest wall": ["lung", "pleura", "rib"],
    "abdomen": ["diaphragm"],
    "pericardium": ["heart"],
    "thymus": ["mediastinum"],
    "larynx": ["trachea"],
}


def get_scispacy_pipeline(model_name: str = "en_core_sci_md", linker_threshold: float = 0.7):
    """Load the scispacy NER model with a UMLS ``EntityLinker`` pipe.

    Args:
        model_name: scispacy model package name (must be installed).
        linker_threshold: Minimum candidate-linking score the linker itself
            uses internally before returning candidates.

    Returns:
        A loaded spaCy ``Language`` pipeline with ``scispacy_linker`` added.
        Note: on first call this downloads the UMLS knowledge base
        (~1GB) via scispacy's data cache.
    """
    import spacy  # local import: heavy dependency only needed here
    import scispacy.linking  # noqa: F401  registers the "scispacy_linker" factory

    nlp = spacy.load(model_name)
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "resolve_abbreviations": True,
                "linker_name": "umls",
                "threshold": linker_threshold,
            },
        )
    return nlp


def extract_concepts(
    text: str,
    nlp,
    allowed_tuis: Sequence[str],
    threshold: float = 0.7,
    max_concepts: int = 50,
) -> List[Dict]:
    """Extract and UMLS-ground medical concepts from report text.

    Convenience wrapper around :func:`extract_concepts_from_doc` for callers
    that have raw text rather than a pre-built spaCy ``Doc`` (e.g. single-
    image inference). Batch callers should use ``nlp.pipe()`` plus
    :func:`extract_concepts_from_doc` directly to avoid re-running the UMLS
    linker.

    Args:
        text: Cleaned report text (FINDINGS + IMPRESSION).
        nlp: A pipeline returned by :func:`get_scispacy_pipeline`.
        allowed_tuis: UMLS semantic type codes to keep (e.g. T047, T023).
        threshold: Minimum entity-linking confidence score to keep a
            candidate (spec: > 0.7).
        max_concepts: Maximum number of concepts to return per report (K).

    Returns:
        List of up to ``max_concepts`` dicts, sorted by descending
        confidence, each with keys ``text, cui, canonical_name,
        semantic_type, confidence, sentence_idx``.
    """
    if not text.strip():
        return []
    doc = nlp(text)
    return extract_concepts_from_doc(doc, nlp, allowed_tuis, threshold=threshold, max_concepts=max_concepts)


def extract_concepts_from_doc(
    doc,
    nlp,
    allowed_tuis: Sequence[str],
    threshold: float = 0.7,
    max_concepts: int = 50,
) -> List[Dict]:
    """Extract and UMLS-ground medical concepts from an already-processed ``Doc``.

    Args:
        doc: A spaCy ``Doc`` produced by a pipeline with ``scispacy_linker``
            (e.g. via ``nlp.pipe(texts)``).
        nlp: The pipeline that produced ``doc`` (used to fetch the linker's
            knowledge base).
        allowed_tuis: UMLS semantic type codes to keep (e.g. T047, T023).
        threshold: Minimum entity-linking confidence score to keep a
            candidate (spec: > 0.7).
        max_concepts: Maximum number of concepts to return per report (K).

    Returns:
        List of up to ``max_concepts`` dicts, sorted by descending
        confidence, each with keys ``text, cui, canonical_name,
        semantic_type, confidence, sentence_idx``.
    """
    if len(doc) == 0:
        return []

    linker = nlp.get_pipe("scispacy_linker")
    allowed = set(allowed_tuis)

    sentences = list(doc.sents)
    sent_start_to_idx = {sent.start: i for i, sent in enumerate(sentences)}

    candidates: Dict[str, Dict] = {}
    for ent in doc.ents:
        sent_idx = sent_start_to_idx.get(ent.sent.start, 0)
        for cui, score in ent._.kb_ents:
            if score < threshold:
                continue
            kb_entity = linker.kb.cui_to_entity.get(cui)
            if kb_entity is None:
                continue
            matched_tuis = allowed & set(kb_entity.types)
            if not matched_tuis:
                continue
            semantic_type = sorted(matched_tuis)[0]
            existing = candidates.get(cui)
            if existing is None or score > existing["confidence"]:
                candidates[cui] = {
                    "text": ent.text,
                    "cui": cui,
                    "canonical_name": kb_entity.canonical_name,
                    "semantic_type": semantic_type,
                    "confidence": float(score),
                    "sentence_idx": sent_idx,
                }

    concepts = sorted(candidates.values(), key=lambda c: -c["confidence"])
    return concepts[:max_concepts]


class ConceptEmbedder:
    """Encodes concept canonical names to 768-d vectors with PubMedBERT.

    Used in place of a pretrained UMLS embedding table (BioSentVec / BERT-
    UMLS), which is not available offline.

    Args:
        model_name: HF model id, default PubMedBERT abstract+fulltext.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoModel, AutoTokenizer  # local import

        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading concept embedding model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        """Mean-pool PubMedBERT embeddings for a list of concept names.

        Args:
            texts: Concept canonical names (or raw mention text).
            batch_size: Number of texts encoded per forward pass.

        Returns:
            ``(len(texts), 768)`` float tensor on CPU.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=32, return_tensors="pt"
            ).to(self.device)
            output = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (output.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed / counts
            all_embeddings.append(mean_pooled.cpu())
        return torch.cat(all_embeddings, dim=0)


def is_anatomy_adjacent(name_a: str, name_b: str) -> bool:
    """Check whether two anatomy concept names are adjacent per the curated table.

    Args:
        name_a: Canonical name of the first (T023) concept.
        name_b: Canonical name of the second (T023) concept.

    Returns:
        ``True`` if any curated region keyword in ``name_a`` lists a region
        keyword found in ``name_b`` as an adjacent structure.
    """
    a, b = name_a.lower(), name_b.lower()
    for region, neighbors in ANATOMY_ADJACENCY.items():
        if region in a:
            for neighbor in neighbors:
                if neighbor in b:
                    return True
    return False


def compute_cooccurrence(
    concept_lists: Dict[str, List[Dict]],
    threshold: int = 5,
) -> Dict[Tuple[str, str], int]:
    """Count CUI-pair co-occurrences across reports, filtered by frequency.

    Args:
        concept_lists: Mapping from uid to that report's extracted concepts.
        threshold: Minimum co-occurrence count to keep a pair (spec: >= 5).

    Returns:
        Mapping from ``(cui_a, cui_b)`` (``cui_a < cui_b``) to co-occurrence
        count, restricted to pairs meeting ``threshold``.
    """
    counter: Counter = Counter()
    for concepts in concept_lists.values():
        cuis = sorted({c["cui"] for c in concepts})
        for i in range(len(cuis)):
            for j in range(i + 1, len(cuis)):
                counter[(cuis[i], cuis[j])] += 1
    return {pair: count for pair, count in counter.items() if count >= threshold}


def build_edge_weight_matrix(
    concepts: List[Dict],
    cooccurrence: Dict[Tuple[str, str], int],
    max_cooccurrence: int,
) -> np.ndarray:
    """Build the dense ``(K, K)`` concept-graph edge-weight matrix for a report.

    Combines the three edge types from the spec:
      - semantic edges: concepts mentioned in the same sentence, weighted by
        the mean of their linking confidences (RadGraph-relation substitute).
      - anatomical proximity: T023-T023 concept pairs adjacent per
        :data:`ANATOMY_ADJACENCY`, weighted by normalized co-occurrence.
      - co-occurrence edges: any CUI pair with corpus co-occurrence >= the
        configured threshold, weighted by normalized co-occurrence.

    Args:
        concepts: This report's extracted concepts (length K <= 50).
        cooccurrence: Global CUI-pair co-occurrence counts (training split).
        max_cooccurrence: Maximum co-occurrence count, used to normalize
            co-occurrence-based weights into [0, 1].

    Returns:
        Symmetric ``(K, K)`` float32 numpy array with zero diagonal.
    """
    k = len(concepts)
    weights = np.zeros((k, k), dtype=np.float32)
    denom = max(max_cooccurrence, 1)

    for i in range(k):
        for j in range(i + 1, k):
            ci, cj = concepts[i], concepts[j]
            weight = 0.0

            if ci["sentence_idx"] == cj["sentence_idx"]:
                weight = max(weight, (ci["confidence"] + cj["confidence"]) / 2.0)

            pair = tuple(sorted((ci["cui"], cj["cui"])))
            cooc = cooccurrence.get(pair, 0)
            if cooc > 0:
                cooc_weight = min(cooc / denom, 1.0)
                if ci["semantic_type"] == ANATOMY_TUI and cj["semantic_type"] == ANATOMY_TUI:
                    if is_anatomy_adjacent(ci["canonical_name"], cj["canonical_name"]):
                        weight = max(weight, cooc_weight)
                weight = max(weight, cooc_weight)

            if weight > 0:
                weights[i, j] = weight
                weights[j, i] = weight

    return weights
