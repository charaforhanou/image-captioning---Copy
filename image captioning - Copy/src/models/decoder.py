"""Medical report decoder (Component 7).

A 6-layer causal transformer decoder with token + sinusoidal positional +
FINDINGS/IMPRESSION section-type embeddings, cross-attending to the fused
``(B, 278, hidden_dim)`` encoder sequence. The final hidden state is combined
with the per-step :class:`src.models.memory_module.MultimodalMemory` context
via a learned gate before the output projection to vocabulary logits.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.triangular_attention import CrossAttention


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Args:
        hidden_dim: Model hidden dimension.
        max_length: Maximum sequence length to precompute.
    """

    def __init__(self, hidden_dim: int, max_length: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(max_length, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to ``x``.

        Args:
            x: ``(B, T, hidden_dim)``.

        Returns:
            ``(B, T, hidden_dim)``.
        """
        return x + self.pe[: x.shape[1]].unsqueeze(0).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with key-padding masking.

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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, T, hidden_dim)``.
            key_padding_mask: ``(B, T)`` bool, ``True`` for real (non-padding) tokens.

        Returns:
            ``(B, T, hidden_dim)``.
        """
        b, t, d = x.shape
        qkv = self.qkv_proj(x).reshape(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, T, T)

        causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, d)
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    """Pre-norm decoder block: causal self-attention, encoder cross-attention, FFN.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.self_attn_norm(x), self_attn_mask))
        x = x + self.dropout(self.cross_attn(self.cross_attn_norm(x), encoder_out, encoder_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class MedicalReportDecoder(nn.Module):
    """6-layer causal transformer decoder over the fused encoder sequence.

    Args:
        vocab_size: Tokenizer vocabulary size.
        pad_id: Token id used for padding (sets ``embed_tokens`` padding_idx).
        hidden_dim: Model hidden dimension.
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.
        max_length: Maximum decoding length (for positional encoding).
        use_section_embedding: If ``False`` (the ``decoder_type=standard``
            ablation), omit the FINDINGS/IMPRESSION section-type embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        max_length: int = 512,
        use_section_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_section_embedding = use_section_embedding

        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_length)
        self.section_embed = nn.Embedding(2, hidden_dim) if use_section_embedding else None
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.memory_gate = nn.Linear(hidden_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        self.lm_head.weight = self.embed_tokens.weight

        # GPT-style init: nn.Embedding's default N(0,1) makes the (tied) lm_head
        # logits have std ~sqrt(hidden_dim), pushing the initial loss to ~10x
        # ln(vocab_size) instead of starting near it.
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        if self.section_embed is not None:
            nn.init.normal_(self.section_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed_tokens.weight[pad_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        section_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the causal decoder stack.

        Args:
            input_ids: ``(B, T)`` target token ids.
            section_ids: ``(B, T)`` 0/1 FINDINGS/IMPRESSION section ids
                (see :meth:`src.data.tokenizer.MedicalTokenizer.section_ids`).
            attention_mask: ``(B, T)`` 1/0 (or bool), ``1``/``True`` for real tokens.
            encoder_out: ``(B, N, hidden_dim)`` fused encoder sequence.
            encoder_mask: ``(B, N)`` bool, ``True`` for valid encoder positions.

        Returns:
            ``(B, T, hidden_dim)`` final decoder hidden states (pre-LM-head).
        """
        token_emb = self.embed_tokens(input_ids) * math.sqrt(self.hidden_dim)
        x = self.pos_encoding(token_emb)
        if self.use_section_embedding:
            x = x + self.section_embed(section_ids)
        x = self.embed_dropout(x)

        self_attn_mask = attention_mask.bool()
        for layer in self.layers:
            x = layer(x, self_attn_mask, encoder_out, encoder_mask)

        return self.final_norm(x)

    def compute_logits(self, decoder_hidden: torch.Tensor, memory_context: torch.Tensor) -> torch.Tensor:
        """Fuse memory context into the decoder hidden state and project to logits.

        Args:
            decoder_hidden: ``(B, T, hidden_dim)`` output of :meth:`forward`.
            memory_context: ``(B, T, hidden_dim)`` output of
                :class:`src.models.memory_module.MultimodalMemory`.

        Returns:
            ``(B, T, vocab_size)`` unnormalized logits.
        """
        gate = torch.sigmoid(self.memory_gate(memory_context))
        combined = self.memory_norm(decoder_hidden + gate * memory_context)
        return self.lm_head(combined)
