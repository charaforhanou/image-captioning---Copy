"""METEOR metric via NLTK."""

from typing import List

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


def compute_meteor(predictions: List[str], references: List[str]) -> float:
    """Mean sentence-level METEOR score over a corpus.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).

    Returns:
        Mean METEOR score (``0.0`` for empty hypothesis/reference pairs).
    """
    if not predictions:
        return 0.0

    scores = []
    for pred, ref in zip(predictions, references):
        hyp_tokens = word_tokenize(pred)
        ref_tokens = word_tokenize(ref)
        if not hyp_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        scores.append(meteor_score([ref_tokens], hyp_tokens))
    return sum(scores) / len(scores)
