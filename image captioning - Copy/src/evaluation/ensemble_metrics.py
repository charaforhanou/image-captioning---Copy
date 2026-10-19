"""Aggregate BLEU/ROUGE-L/METEOR/CIDEr with bootstrap 95% confidence intervals."""

from typing import Dict, List

import numpy as np

from src.evaluation.bleu import compute_bleu
from src.evaluation.cider import compute_cider
from src.evaluation.meteor import compute_meteor
from src.evaluation.rouge import compute_rouge_l


def compute_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1..4, ROUGE-L (P/R/F1), METEOR, and CIDEr for one corpus.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).

    Returns:
        Flat dict with keys ``bleu1..4``, ``rougeL_precision/recall/f1``,
        ``meteor``, ``cider``.
    """
    metrics: Dict[str, float] = {}
    metrics.update(compute_bleu(predictions, references))
    metrics.update(compute_rouge_l(predictions, references))
    metrics["meteor"] = compute_meteor(predictions, references)
    metrics["cider"] = compute_cider(predictions, references)
    return metrics


def compute_all_metrics_with_ci(
    predictions: List[str],
    references: List[str],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap mean/std/95% CI for each of the 7 captioning metrics.

    Resamples ``(prediction, reference)`` pairs with replacement
    ``n_bootstrap`` times and recomputes every corpus-level metric on each
    resample, giving a distribution from which the standard error and a
    percentile confidence interval are derived. The reported ``mean`` is the
    point estimate computed on the full (un-resampled) corpus.

    Args:
        predictions: Generated report texts.
        references: Ground-truth report texts (one per prediction).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level for the interval (e.g. ``0.95``).
        seed: RNG seed for reproducible resampling.

    Returns:
        Dict keyed by metric name, each mapping to
        ``{"mean", "std", "ci_low", "ci_high"}``.
    """
    point_estimate = compute_all_metrics(predictions, references)

    n = len(predictions)
    if n == 0:
        return {name: {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0} for name in point_estimate}

    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {name: [] for name in point_estimate}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        preds = [predictions[i] for i in idx]
        refs = [references[i] for i in idx]
        for name, value in compute_all_metrics(preds, refs).items():
            samples[name].append(value)

    alpha = (1.0 - confidence) / 2.0
    results: Dict[str, Dict[str, float]] = {}
    for name, values in samples.items():
        arr = np.array(values)
        results[name] = {
            "mean": point_estimate[name],
            "std": float(arr.std(ddof=1)),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        }
    return results
