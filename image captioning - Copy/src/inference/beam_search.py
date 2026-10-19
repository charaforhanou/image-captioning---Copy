"""Beam search decoding for MCCAG (Component 7/8 inference).

No KV-cache: at each step the full decoder + memory stack is re-run over the
growing ``input_ids`` sequence (mirrors :func:`src.training.metrics.greedy_decode`).
This keeps the implementation simple and correct, at the cost of O(T^2)
decode time -- acceptable for the modest ``max_length`` values used during
smoke testing and small-batch evaluation.

Implements GNMT-style length normalization, a concept-coverage penalty derived
from :class:`src.models.memory_module.MultimodalMemory`'s per-step coverage
vector, and no-repeat trigram blocking to prevent degenerate repetition loops.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F

from src.data.tokenizer import MedicalTokenizer


def _block_repeat_ngrams(log_probs: torch.Tensor, input_ids: torch.Tensor, no_repeat_ngram_size: int) -> None:
    """In-place: set to -inf the tokens that would create a repeated n-gram.

    For each beam, if the last (n-1) tokens already appeared earlier followed by
    some token c, then emitting c again is forbidden. This is the standard
    no-repeat-ngram trick (Paulus et al. 2017) used by most caption/report
    generators to kill repetition loops.

    Args:
        log_probs: ``(num_beams, vocab)`` next-token log-probabilities (modified).
        input_ids: ``(num_beams, cur_len)`` tokens generated so far.
        no_repeat_ngram_size: n (e.g. 3 blocks repeated trigrams).
    """
    num_beams, cur_len = input_ids.shape
    if cur_len < no_repeat_ngram_size:
        return
    n = no_repeat_ngram_size
    for beam_idx in range(num_beams):
        seq = input_ids[beam_idx].tolist()
        prefix = tuple(seq[-(n - 1):])  # last (n-1) tokens
        banned = set()
        for t in range(len(seq) - n + 1):
            if tuple(seq[t : t + n - 1]) == prefix:
                banned.add(seq[t + n - 1])
        for tok_id in banned:
            log_probs[beam_idx, tok_id] = float("-inf")


@torch.no_grad()
def beam_search_decode(
    model,
    batch: dict,
    tokenizer: MedicalTokenizer,
    beam_size: int = 3,
    max_length: int = 128,
    min_length: int = 3,
    length_penalty: float = 0.6,
    coverage_penalty: float = 0.5,
    temperature: float = 1.0,
    no_repeat_ngram_size: int = 3,
) -> List[Dict[str, object]]:
    """Beam-search-decode a batch of reports.

    Args:
        model: A :class:`src.models.mccag.MCCAG` in eval mode.
        batch: Input batch (image, concept_embeddings, concept_mask, edge_weights, ...).
        tokenizer: A :class:`MedicalTokenizer`.
        beam_size: Number of beams per sample.
        max_length: Maximum generated sequence length (including ``[REPORT_START]``).
        min_length: Minimum length before ``[EOS]`` is allowed.
        length_penalty: GNMT-style length-normalization exponent ``alpha``.
        coverage_penalty: Weight ``beta`` of the concept-coverage penalty.
        temperature: Softmax temperature applied to logits.
        no_repeat_ngram_size: Block repeated n-grams of this size (0/None disables).

    Returns:
        List of ``B`` dicts, each with keys ``ids``, ``beam_score``,
        ``confidence_mean``, ``concept_coverage``.
    """
    fused, fused_mask, concept_mask = model.encode(batch)
    b = fused.shape[0]

    results = []
    for i in range(b):
        results.append(
            _beam_search_single(
                model=model,
                fused=fused[i : i + 1],
                fused_mask=fused_mask[i : i + 1],
                concept_mask=concept_mask[i : i + 1],
                tokenizer=tokenizer,
                beam_size=beam_size,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                coverage_penalty=coverage_penalty,
                temperature=temperature,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        )
    return results


@torch.no_grad()
def _beam_search_single(
    model,
    fused: torch.Tensor,
    fused_mask: torch.Tensor,
    concept_mask: torch.Tensor,
    tokenizer: MedicalTokenizer,
    beam_size: int,
    max_length: int,
    min_length: int,
    length_penalty: float,
    coverage_penalty: float,
    temperature: float,
    no_repeat_ngram_size: int = 3,
) -> Dict[str, object]:
    """Beam search for a single sample (``fused`` has batch size 1)."""
    device = fused.device
    fused = fused.expand(beam_size, -1, -1)
    fused_mask = fused_mask.expand(beam_size, -1)
    concept_mask_f = concept_mask.expand(beam_size, -1).float()

    input_ids = torch.full((beam_size, 1), tokenizer.report_start_id, dtype=torch.long, device=device)
    beam_scores = torch.full((beam_size,), float("-inf"), device=device)
    beam_scores[0] = 0.0
    coverage = torch.zeros(beam_size, model.max_concepts, device=device)

    # (final_score, raw_score, length, ids, concept_coverage) for hypotheses that hit [EOS].
    finished = []

    for _ in range(max_length - 1):
        cur_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        section_ids = tokenizer.section_ids(input_ids)

        decoder_hidden = model.decoder(input_ids, section_ids, attention_mask, fused, fused_mask)
        if model.use_memory:
            memory_context, coverage = model.memory(decoder_hidden, fused, fused_mask)
        else:
            memory_context = torch.zeros_like(decoder_hidden)

        logits = model.decoder.compute_logits(decoder_hidden, memory_context)[:, -1, :].float()
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        # --- No-repeat n-gram blocking: kills degenerate repetition loops ---
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            _block_repeat_ngrams(log_probs, input_ids, no_repeat_ngram_size)

        # Forbid [EOS] before the minimum length is reached.
        if cur_len < min_length:
            log_probs[:, tokenizer.eos_id] = float("-inf")

        vocab_size = log_probs.shape[-1]
        next_scores = (beam_scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = next_scores.topk(min(beam_size * 2, next_scores.numel()))
        src_beams = torch.div(top_idx, vocab_size, rounding_mode="floor")
        next_tokens = top_idx % vocab_size

        kept_src, kept_tok, kept_score = [], [], []
        for score, src, tok in zip(top_scores.tolist(), src_beams.tolist(), next_tokens.tolist()):
            if score == float("-inf"):
                continue
            if tok == tokenizer.eos_id:
                length_norm = ((5 + cur_len) / 6) ** length_penalty
                cov_term = coverage_penalty * (
                    coverage[src].clamp(min=1e-6, max=1.0).log() * concept_mask_f[src]
                ).sum().item()
                concept_cov = _concept_coverage(coverage[src], concept_mask_f[src])
                finished.append((score / length_norm + cov_term, score, cur_len, input_ids[src].tolist(), concept_cov))
                continue
            kept_src.append(src)
            kept_tok.append(tok)
            kept_score.append(score)
            if len(kept_src) == beam_size:
                break

        if not kept_src:
            break

        while len(kept_src) < beam_size:
            kept_src.append(kept_src[-1])
            kept_tok.append(kept_tok[-1])
            kept_score.append(float("-inf"))

        src_idx = torch.tensor(kept_src, device=device)
        next_token_t = torch.tensor(kept_tok, device=device, dtype=torch.long)
        input_ids = torch.cat([input_ids[src_idx], next_token_t.unsqueeze(1)], dim=1)
        beam_scores = torch.tensor(kept_score, device=device)
        coverage = coverage[src_idx]

        if len(finished) >= beam_size:
            break

    if not finished:
        cur_len = input_ids.shape[1]
        length_norm = ((5 + cur_len) / 6) ** length_penalty
        for j in range(beam_size):
            cov_term = coverage_penalty * (
                coverage[j].clamp(min=1e-6, max=1.0).log() * concept_mask_f[j]
            ).sum().item()
            concept_cov = _concept_coverage(coverage[j], concept_mask_f[j])
            finished.append((beam_scores[j].item() / length_norm + cov_term, beam_scores[j].item(), cur_len, input_ids[j].tolist(), concept_cov))

    final_score, raw_score, length, ids, concept_coverage = max(finished, key=lambda x: x[0])
    return {
        "ids": ids,
        "beam_score": final_score,
        "confidence_mean": raw_score / max(length, 1),
        "concept_coverage": concept_coverage,
    }


def _concept_coverage(coverage_row: torch.Tensor, concept_mask_row: torch.Tensor) -> float:
    """Mean memory-attention coverage mass over valid concept slots."""
    denom = concept_mask_row.sum().clamp(min=1.0)
    return (coverage_row * concept_mask_row).sum().item() / denom.item()