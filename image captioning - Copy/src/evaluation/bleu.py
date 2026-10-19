"""BLEU-1..4 corpus metrics via NLTK."""

from typing import Dict, List

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Corpus BLEU-1 through BLEU-4 with NLTK's ``method4`` smoothing.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).

    Returns:
        Dict with keys ``bleu1``, ``bleu2``, ``bleu3``, ``bleu4``.
    """
    if not predictions:
        return {f"bleu{n}": 0.0 for n in (1, 2, 3, 4)}

    smoothing = SmoothingFunction().method4
    refs = [[word_tokenize(r)] for r in references]
    hyps = [word_tokenize(p) for p in predictions]
    return {
        f"bleu{n}": corpus_bleu(
            refs, hyps, weights=tuple([1.0 / n] * n + [0.0] * (4 - n)), smoothing_function=smoothing
        )
        for n in (1, 2, 3, 4)
    }
