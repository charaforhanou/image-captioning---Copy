"""Triangular attention fusion (Component 5).

Fuses the visual-token stream (196), concept-token stream (K=50), and a
learnable memory-token stream (32) into a single ``(B, 278, hidden_dim)``
sequence via three fusion layers:

  1. Symmetric cross-attention between the visual and concept streams.
  2. Sparse (top-k) cross-attention from memory tokens into
     ``concat(visual, concept)``.
  3. Joint self-attention over all three streams, followed by a learned
     per-stream sigmoid gate applied as a residual update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Standard multi-head (cross- or self-) attention.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, Lq, hidden_dim)``.
            kv: ``(B, Lk, hidden_dim)`` (pass ``query`` itself for self-attention).
            key_padding_mask: Optional ``(B, Lk)`` bool, ``True`` for valid keys.

        Returns:
            ``(B, Lq, hidden_dim)``.
        """
        b, lq, d = query.shape
        lk = kv.shape[1]

        q = self.q_proj(query).reshape(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv_proj = self.kv_proj(kv).reshape(b, lk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv_proj[0], kv_proj[1]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, Lq, Lk)
        if key_padding_mask is not None:
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, lq, d)
        return self.out_proj(out)


class SparseCrossAttention(nn.Module):
    """Cross-attention restricted to each query's top-k highest-scoring keys.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        top_k: Number of keys retained per query (per head).
        dropout: Dropout applied to attention weights.
    """

    def __init__(self, hidden_dim: int, num_heads: int, top_k: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k = top_k

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            query: ``(B, Lq, hidden_dim)``.
            kv: ``(B, Lk, hidden_dim)``.
            key_padding_mask: Optional ``(B, Lk)`` bool, ``True`` for valid keys.

        Returns:
            ``(B, Lq, hidden_dim)``.
        """
        b, lq, d = query.shape
        lk = kv.shape[1]

        q = self.q_proj(query).reshape(b, lq, self.num_heads, self.head_dim).transpose(1, 2)
        kv_proj = self.kv_proj(kv).reshape(b, lk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv_proj[0], kv_proj[1]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, Lq, Lk)
        if key_padding_mask is not None:
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        top_k = min(self.top_k, lk)
        topk_vals, topk_idx = attn.topk(top_k, dim=-1)
        sparse_attn = torch.full_like(attn, float("-inf"))
        sparse_attn.scatter_(-1, topk_idx, topk_vals)

        sparse_attn = F.softmax(sparse_attn, dim=-1)
        sparse_attn = torch.nan_to_num(sparse_attn)
        sparse_attn = self.dropout(sparse_attn)

        out = torch.matmul(sparse_attn, v).transpose(1, 2).reshape(b, lq, d)
        return self.out_proj(out)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN block used for per-stream refinement.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.attn(h, h, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TriangularAttentionFusion(nn.Module):
    """Fuses visual, concept, and memory token streams.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        memory_tokens: Number of learnable memory tokens (32).
        sparse_topk: Number of keys retained per memory query in fusion layer 2.
        ffn_dim: Feed-forward hidden dimension for stream refinement blocks.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 12,
        memory_tokens: int = 32,
        sparse_topk: int = 50,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.memory_tokens = memory_tokens
        self.memory_token_embed = nn.Parameter(torch.zeros(memory_tokens, hidden_dim))
        nn.init.normal_(self.memory_token_embed, std=0.02)

        self.visual_refine = SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)
        self.concept_refine = SelfAttentionBlock(hidden_dim, num_heads, ffn_dim, dropout)

        # Fusion layer 1: symmetric visual <-> concept cross-attention
        self.visual_to_concept_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.concept_to_visual_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.visual_norm1 = nn.LayerNorm(hidden_dim)
        self.concept_norm1 = nn.LayerNorm(hidden_dim)

        # Fusion layer 2: memory sparse cross-attention over concat(visual, concept)
        self.memory_attn = SparseCrossAttention(hidden_dim, num_heads, sparse_topk, dropout)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # Fusion layer 3: joint self-attention + per-stream sigmoid gate
        self.joint_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.joint_norm = nn.LayerNorm(hidden_dim)
        self.visual_gate = nn.Linear(hidden_dim, hidden_dim)
        self.concept_gate = nn.Linear(hidden_dim, hidden_dim)
        self.memory_gate = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, visual_tokens: torch.Tensor, concept_tokens: torch.Tensor, concept_mask: torch.Tensor
    ) -> tuple:
        """Fuse the three streams.

        Args:
            visual_tokens: ``(B, 196, hidden_dim)`` ViT patch tokens.
            concept_tokens: ``(B, K, hidden_dim)`` refined concept node embeddings
                from :class:`src.models.graph_transformer.GraphTransformerEncoder`.
            concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.

        Returns:
            Tuple ``(fused, fused_mask)``:
              - ``fused``: ``(B, 196 + K + memory_tokens, hidden_dim)``.
              - ``fused_mask``: ``(B, 196 + K + memory_tokens)`` bool, ``True``
                for valid (non-padding) positions.
        """
        b = visual_tokens.shape[0]
        device = visual_tokens.device
        visual_mask = torch.ones(b, visual_tokens.shape[1], dtype=torch.bool, device=device)
        memory_mask = torch.ones(b, self.memory_tokens, dtype=torch.bool, device=device)

        visual = self.visual_refine(visual_tokens, visual_mask)
        concept = self.concept_refine(concept_tokens, concept_mask)

        visual = self.visual_norm1(
            visual + self.dropout(self.visual_to_concept_attn(visual, concept, concept_mask))
        )
        concept = self.concept_norm1(
            concept + self.dropout(self.concept_to_visual_attn(concept, visual, visual_mask))
        )

        memory = self.memory_token_embed.unsqueeze(0).expand(b, -1, -1)
        visual_concept = torch.cat([visual, concept], dim=1)
        visual_concept_mask = torch.cat([visual_mask, concept_mask], dim=1)
        memory = self.memory_norm(
            memory + self.dropout(self.memory_attn(memory, visual_concept, visual_concept_mask))
        )

        joint = torch.cat([visual, concept, memory], dim=1)
        joint_mask = torch.cat([visual_mask, concept_mask, memory_mask], dim=1)
        joint_attn_out = self.joint_norm(self.joint_attn(joint, joint, joint_mask))

        n_visual, n_concept = visual.shape[1], concept.shape[1]
        visual_attn_out, concept_attn_out, memory_attn_out = joint_attn_out.split(
            [n_visual, n_concept, self.memory_tokens], dim=1
        )

        visual_out = visual + torch.sigmoid(self.visual_gate(visual_attn_out)) * visual_attn_out
        concept_out = concept + torch.sigmoid(self.concept_gate(concept_attn_out)) * concept_attn_out
        memory_out = memory + torch.sigmoid(self.memory_gate(memory_attn_out)) * memory_attn_out

        fused = torch.cat([visual_out, concept_out, memory_out], dim=1)
        return fused, joint_mask
