"""Concept token projection (Component 2).

At train/eval time the actual extraction (scispacy NER + UMLS linking +
PubMedBERT embedding) has already been done offline by
``scripts/prepare_data.py`` and cached. This module consumes the resulting
``(B, K, concept_dim)`` cached embeddings + ``(B, K)`` mask supplied by
:class:`src.data.dataset.IUXRayDataset` and projects them into the model's
hidden space, filling padding slots with a learnable pad-concept embedding.
"""

import torch
import torch.nn as nn


class ConceptExtractor(nn.Module):
    """Projects cached PubMedBERT concept embeddings into model space.

    Args:
        input_dim: Dimensionality of the cached concept embeddings.
        hidden_dim: Model hidden dimension.
        dropout: Dropout applied after projection.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_embedding = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.pad_embedding, std=0.02)

    def forward(
        self, concept_embeddings: torch.Tensor, concept_mask: torch.Tensor
    ) -> torch.Tensor:
        """Project concept embeddings, replacing padding with a learned token.

        Args:
            concept_embeddings: ``(B, K, input_dim)`` zero-padded cached
                concept embeddings.
            concept_mask: ``(B, K)`` bool, ``True`` for real (non-padding)
                concept slots.

        Returns:
            ``(B, K, hidden_dim)`` concept token embeddings.
        """
        x = self.proj(concept_embeddings)
        x = self.layer_norm(x)
        x = self.dropout(x)

        pad = self.pad_embedding.view(1, 1, -1).expand_as(x)
        mask = concept_mask.unsqueeze(-1)
        return torch.where(mask, x, pad)
