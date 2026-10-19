"""Fast validation-time metrics and greedy decoding.

These are cheap proxies used inside the training loop to monitor progress
and drive early stopping. The full beam-search generation and the complete
BLEU/ROUGE/METEOR/CIDEr evaluation suite live in ``src/evaluation`` and
``src/inference`` and are run by ``scripts/evaluate.py``.
"""

from typing import List

import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


def token_accuracy(logits: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> float:
    """Next-token teacher-forced accuracy, ignoring ``[PAD]`` targets.

    Args:
        logits: ``(B, T, V)`` model output logits.
        input_ids: ``(B, T)`` target token ids (same tensor used for loss).
        pad_id: Padding token id to exclude from the denominator.

    Returns:
        Fraction of non-padding positions where ``argmax(logits)`` matches
        the shifted target token.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    preds = shift_logits.argmax(dim=-1)
    mask = shift_labels != pad_id
    correct = ((preds == shift_labels) & mask).sum().float()
    return (correct / mask.sum().float().clamp(min=1)).item()


def quick_bleu4(predictions: List[str], references: List[str]) -> float:
    """Corpus BLEU-4 with NLTK's ``method4`` smoothing."""
    if not predictions:
        return 0.0
    smoothing = SmoothingFunction().method4
    refs = [[word_tokenize(r)] for r in references]
    hyps = [word_tokenize(p) for p in predictions]
    return corpus_bleu(refs, hyps, smoothing_function=smoothing)


def quick_meteor(predictions: List[str], references: List[str]) -> float:
    """Mean sentence-level METEOR score."""
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


@torch.no_grad()
def greedy_decode(model, batch: dict, tokenizer, max_length: int = 100) -> List[str]:
    """Greedily decode a batch for quick validation logging (no beam search, no KV cache).

    Args:
        model: A :class:`src.models.mccag.MCCAG` in eval mode.
        batch: Input batch (image, concept_embeddings, concept_mask, edge_weights, ...).
        tokenizer: A :class:`src.data.tokenizer.MedicalTokenizer`.
        max_length: Hard cap on generated length (independent of the
            tokenizer's training-time ``max_length``, kept small for speed).

    Returns:
        List of ``B`` detokenized strings (special tokens stripped).
    """
    device = batch["image"].device
    fused, fused_mask, _ = model.encode(batch)
    b = fused.shape[0]

    input_ids = torch.full((b, 1), tokenizer.report_start_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_length - 1):
        attention_mask = torch.ones_like(input_ids)
        section_ids = tokenizer.section_ids(input_ids)

        decoder_hidden = model.decoder(input_ids, section_ids, attention_mask, fused, fused_mask)

        if model.use_memory:
            memory_context, _ = model.memory(decoder_hidden, fused, fused_mask)
        else:
            memory_context = torch.zeros_like(decoder_hidden)

        logits = model.decoder.compute_logits(decoder_hidden, memory_context)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, tokenizer.pad_id), next_token)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == tokenizer.eos_id)
        if finished.all():
            break

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
