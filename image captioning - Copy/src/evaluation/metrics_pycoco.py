"""Métriques standard IU X-Ray via pycocoevalcap, SANS Java.

Calcule BLEU-1..4, ROUGE-L et CIDEr avec les implémentations pycocoevalcap
(celles des baselines R2Gen/XPRONET/KiUT), avec une tokenisation Python pure
pour ne PAS dépendre du PTBTokenizer Java.

METEOR omis par défaut (exige Java) ; use_meteor=True l'ajoute si Java présent.

Robustesse : les prédictions/références NaN ou vides sont converties en chaîne
et remplacées par un placeholder, pour éviter que pycocoevalcap ne renvoie 0
partout à cause d'une seule entrée invalide.
"""

import math
import re
from typing import Dict, List

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

_PUNCT = {
    "''", "'", "``", "`", ".", "?", "!", ",", ":", "-", "--", "...", ";",
    "(", ")", "[", "]", "{", "}", '"',
}

# Placeholder pour une génération vide (le modèle n'a rien produit).
# Un token neutre évite les chaînes vides qui cassent pycocoevalcap.
_EMPTY_PLACEHOLDER = "empty"


def _safe_str(x) -> str:
    """Convertit en chaîne ; NaN/None/float-nan -> ''. """
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _ptb_like(text) -> str:
    text = _safe_str(text).replace("\n", " ")
    text = re.sub(r"([.?!,:;()\[\]{}\"])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    toks = [t for t in text.split(" ") if t and t not in _PUNCT]
    out = " ".join(toks)
    return out if out else _EMPTY_PLACEHOLDER


def _tokenize(predictions: List, references: List):
    gts = {i: [_ptb_like(references[i])] for i in range(len(references))}
    res = {i: [_ptb_like(predictions[i])] for i in range(len(predictions))}
    return gts, res


_SCORERS = {}


def _get_scorers(use_meteor: bool):
    if "bleu" not in _SCORERS:
        _SCORERS["bleu"] = Bleu(4)
        _SCORERS["rouge"] = Rouge()
        _SCORERS["cider"] = Cider()
    if use_meteor and "meteor" not in _SCORERS:
        from pycocoevalcap.meteor.meteor import Meteor
        _SCORERS["meteor"] = Meteor()
    return _SCORERS


def compute_all_metrics(predictions: List, references: List, use_meteor: bool = False) -> Dict[str, float]:
    """BLEU-1..4, ROUGE-L, CIDEr (+ METEOR si use_meteor) sur un corpus apparié."""
    import contextlib
    import io

    if not predictions:
        base = {f"bleu{n}": 0.0 for n in (1, 2, 3, 4)}
        base.update({"rougeL_f1": 0.0, "cider": 0.0})
        if use_meteor:
            base["meteor"] = 0.0
        return base

    gts, res = _tokenize(predictions, references)
    s = _get_scorers(use_meteor)
    out: Dict[str, float] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = s["bleu"].compute_score(gts, res)
        rouge = s["rouge"].compute_score(gts, res)[0]
        cider = s["cider"].compute_score(gts, res)[0]
        meteor = s["meteor"].compute_score(gts, res)[0] if use_meteor else None
    out["bleu1"], out["bleu2"], out["bleu3"], out["bleu4"] = (float(b) for b in bleu)
    out["rougeL_f1"] = float(rouge)
    out["cider"] = float(cider)
    if use_meteor:
        out["meteor"] = float(meteor)
    return out


def compute_all_metrics_with_ci(
    predictions: List,
    references: List,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    use_meteor: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap mean/std/IC pour chaque métrique."""
    # Nettoyage en amont : tout en chaîne propre.
    predictions = [_safe_str(p) for p in predictions]
    references = [_safe_str(r) for r in references]

    point = compute_all_metrics(predictions, references, use_meteor=use_meteor)
    n = len(predictions)
    if n == 0:
        return {k: {"mean": 0.0, "std": 0.0, "ci_low": 0.0, "ci_high": 0.0} for k in point}

    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        p = [predictions[i] for i in idx]
        r = [references[i] for i in idx]
        for k, v in compute_all_metrics(p, r, use_meteor=use_meteor).items():
            samples[k].append(v)

    alpha = (1.0 - confidence) / 2.0
    results = {}
    for k, vals in samples.items():
        arr = np.array(vals)
        results[k] = {
            "mean": point[k],
            "std": float(arr.std(ddof=1)),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        }
    return results


def count_empty(predictions: List) -> int:
    """Nombre de prédictions vides/NaN (diagnostic)."""
    return sum(1 for p in predictions if _ptb_like(p) == _EMPTY_PLACEHOLDER)