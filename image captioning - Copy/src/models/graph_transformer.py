"""Graph transformer encoder (Component 4).

Refines concept node embeddings with self-attention restricted (via an
additive bias) to graph-neighbor, top-k cosine-similar, and self pairs as
produced by :class:`src.models.graph_builder.ConceptGraphBuilder`, with a
learnable Laplacian-eigenvector structural positional encoding added to the
input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMultiHeadAttention(nn.Module):
    """Multi-head self-attention with an additive per-pair bias matrix.

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

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply graph-masked self-attention.

        Args:
            x: ``(B, K, hidden_dim)`` input node embeddings.
            attention_bias: ``(B, K, K)`` additive bias (0 for connected
                pairs, large negative otherwise).
            key_padding_mask: ``(B, K)`` bool, ``True`` for real (non-padding)
                nodes.

        Returns:
            ``(B, K, hidden_dim)`` attended node embeddings.
        """
        b, k, _ = x.shape
        qkv = self.qkv_proj(x).reshape(b, k, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, K, head_dim)
        q, key, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, key.transpose(-1, -2)) * self.scale  # (B, heads, K, K)
        attn = attn + attention_bias.unsqueeze(1)

        pad_mask = ~key_padding_mask[:, None, None, :]  # (B, 1, 1, K)
        attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)  # rows with no valid keys (all-padding sample)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, heads, K, head_dim)
        out = out.transpose(1, 2).reshape(b, k, -1)
        return self.out_proj(out)


class GraphTransformerLayer(nn.Module):
    """One pre-norm graph-transformer block (graph-attention + FFN).

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = GraphMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_bias: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_bias, key_padding_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GraphTransformerEncoder(nn.Module):
    """Stack of graph-masked transformer layers over concept nodes.

    Args:
        hidden_dim: Model hidden dimension.
        num_layers: Number of stacked layers.
        num_heads: Number of attention heads per layer.
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        concept_tokens: torch.Tensor,
        pos_encoding: torch.Tensor,
        attention_bias: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Refine concept tokens via graph-masked self-attention.

        Args:
            concept_tokens: ``(B, K, hidden_dim)`` projected concept
                embeddings from :class:`src.models.concept_extractor.ConceptExtractor`.
            pos_encoding: ``(B, K, hidden_dim)`` structural positional
                encoding from :class:`src.models.graph_builder.ConceptGraphBuilder`.
            attention_bias: ``(B, K, K)`` additive graph-attention bias.
            concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.

        Returns:
            ``(B, K, hidden_dim)`` refined concept node embeddings.
        """
        x = concept_tokens + pos_encoding
        for layer in self.layers:
            x = layer(x, attention_bias, concept_mask)
        return self.norm(x)
