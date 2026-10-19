"""CIDEr metric via pycocoevalcap.

Note: CIDEr's TF-IDF term weighting is computed over the full corpus of
references, so scores are only meaningful for a reasonably sized test set --
a tiny (e.g. single-document) corpus degenerates towards 0.
"""

from typing import List

from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider


def compute_cider(predictions: List[str], references: List[str]) -> float:
    """Corpus CIDEr score.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).

    Returns:
        Mean CIDEr score over the corpus (``0.0`` if ``predictions`` is empty).
    """
    if not predictions:
        return 0.0

    gts = {i: [" ".join(word_tokenize(references[i]))] for i in range(len(references))}
    res = {i: [" ".join(word_tokenize(predictions[i]))] for i in range(len(predictions))}

    score, _ = Cider().compute_score(gts, res)
    return float(score)
