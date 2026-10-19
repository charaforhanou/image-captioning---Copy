"""Multimodal memory module (Component 6).

Projects the fused encoder sequence into a fixed-size memory bank (FIFO:
oldest entries evicted first if the sequence exceeds ``memory_size``, though
for the default configuration ``196 + 50 + 32 = 278 <= 512`` so the whole
sequence always fits). Each decoder position cross-attends to the bank's
top-k most relevant slots, with an additive positional-distance decay bias.
Also tracks, per batch element, the cumulative attention mass placed on the
concept-stream slots -- the concept-coverage vector used by the coverage loss
and (at inference) the beam search coverage penalty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalMemory(nn.Module):
    """FIFO memory bank with sparse, distance-decayed cross-attention.

    Args:
        hidden_dim: Model hidden dimension.
        memory_size: Fixed memory bank capacity.
        top_k: Number of bank slots attended to per decoder position.
        decay: Positional-distance decay rate (``exp(-decay * |i - j|)``,
            applied as an additive log-bias ``-decay * |i - j|``).
        num_heads: Number of attention heads.
        num_visual_tokens: Number of visual-stream slots at the start of the
            fused sequence (used to locate the concept-stream slots).
        max_concepts: Number of concept-stream slots (K) immediately
            following the visual slots.
        dropout: Dropout applied to attention weights.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        memory_size: int = 512,
        top_k: int = 32,
        decay: float = 0.01,
        num_heads: int = 12,
        num_visual_tokens: int = 196,
        max_concepts: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.memory_size = memory_size
        self.top_k = top_k
        self.decay = decay
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.concept_start = num_visual_tokens
        self.concept_end = num_visual_tokens + max_concepts

        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_bank(self, fused: torch.Tensor, fused_mask: torch.Tensor) -> tuple:
        """Project the fused sequence into a fixed-size ``memory_size`` bank.

        Args:
            fused: ``(B, N, hidden_dim)`` fused encoder sequence.
            fused_mask: ``(B, N)`` bool, ``True`` for valid positions.

        Returns:
            Tuple ``(bank_k, bank_v, bank_mask)``, each padded/truncated to
            ``memory_size`` along dim 1. Truncation drops the oldest
            (lowest-index) slots first, i.e. FIFO eviction.
        """
        b, n, _ = fused.shape
        bank_k = self.k_proj(fused)
        bank_v = self.v_proj(fused)

        if n < self.memory_size:
            pad = self.memory_size - n
            bank_k = F.pad(bank_k, (0, 0, 0, pad))
            bank_v = F.pad(bank_v, (0, 0, 0, pad))
            bank_mask = F.pad(fused_mask, (0, pad), value=False)
        elif n > self.memory_size:
            bank_k = bank_k[:, -self.memory_size :]
            bank_v = bank_v[:, -self.memory_size :]
            bank_mask = fused_mask[:, -self.memory_size :]
        else:
            bank_mask = fused_mask

        return bank_k, bank_v, bank_mask

    def forward(
        self, decoder_hidden: torch.Tensor, fused: torch.Tensor, fused_mask: torch.Tensor
    ) -> tuple:
        """Cross-attend decoder states to the multimodal memory bank.

        Args:
            decoder_hidden: ``(B, T, hidden_dim)`` decoder hidden states.
            fused: ``(B, N, hidden_dim)`` fused encoder sequence from
                :class:`src.models.triangular_attention.TriangularAttentionFusion`.
            fused_mask: ``(B, N)`` bool, ``True`` for valid positions.

        Returns:
            Tuple ``(memory_context, coverage)``:
              - ``memory_context``: ``(B, T, hidden_dim)``.
              - ``coverage``: ``(B, max_concepts)``, the cumulative
                (head-averaged, time-summed) attention mass on each
                concept-stream slot.
        """
        b, t, d = decoder_hidden.shape
        bank_k, bank_v, bank_mask = self._build_bank(fused, fused_mask)
        n = bank_k.shape[1]

        q = self.q_proj(decoder_hidden).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = bank_k.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = bank_v.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, T, N)

        pos_t = torch.arange(t, device=decoder_hidden.device).view(1, 1, t, 1)
        pos_s = torch.arange(n, device=decoder_hidden.device).view(1, 1, 1, n)
        decay_bias = -self.decay * (pos_t - pos_s).abs().to(attn.dtype)
        attn = attn + decay_bias

        attn = attn.masked_fill(~bank_mask[:, None, None, :], float("-inf"))

        top_k = min(self.top_k, n)
        topk_vals, topk_idx = attn.topk(top_k, dim=-1)
        sparse_attn = torch.full_like(attn, float("-inf"))
        sparse_attn.scatter_(-1, topk_idx, topk_vals)

        sparse_attn = F.softmax(sparse_attn, dim=-1)
        sparse_attn = torch.nan_to_num(sparse_attn)
        sparse_attn = self.dropout(sparse_attn)

        out = torch.matmul(sparse_attn, v).transpose(1, 2).reshape(b, t, d)
        memory_context = self.out_proj(out)

        head_avg_attn = sparse_attn.mean(dim=1)  # (B, T, N)
        coverage = head_avg_attn[:, :, self.concept_start : self.concept_end].sum(dim=1)  # (B, max_concepts)

        return memory_context, coverage
