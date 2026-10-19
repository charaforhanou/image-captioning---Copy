"""ROUGE-L metric via google-research's rouge_score package."""

from typing import Dict, List

from rouge_score import rouge_scorer


def compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Mean ROUGE-L precision/recall/F1 over a corpus.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).

    Returns:
        Dict with keys ``rougeL_precision``, ``rougeL_recall``, ``rougeL_f1``.
    """
    if not predictions:
        return {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    precisions, recalls, f1s = [], [], []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)["rougeL"]
        precisions.append(score.precision)
        recalls.append(score.recall)
        f1s.append(score.fmeasure)

    n = len(f1s)
    return {
        "rougeL_precision": sum(precisions) / n,
        "rougeL_recall": sum(recalls) / n,
        "rougeL_f1": sum(f1s) / n,
    }
