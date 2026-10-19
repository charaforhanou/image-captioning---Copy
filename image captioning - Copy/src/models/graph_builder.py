"""Concept graph assembly (Component 3).

Combines the cached co-occurrence/anatomy/semantic edge-weight matrix
(produced by ``scripts/prepare_data.py``) with a top-k cosine-similarity
graph computed on the fly from the cached concept embeddings, to produce:

  - an additive attention-bias matrix for :class:`src.models.graph_transformer.GraphTransformerEncoder`
    (0 for connected node pairs, a large negative value otherwise), and
  - a learnable structural positional encoding derived from the graph
    Laplacian's eigenvectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_topk_cosine_adjacency(
    concept_embeddings: torch.Tensor, concept_mask: torch.Tensor, top_k: int = 5
) -> torch.Tensor:
    """Build a symmetric top-k cosine-similarity adjacency among valid concepts.

    Args:
        concept_embeddings: ``(B, K, D)`` cached concept embeddings.
        concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.
        top_k: Number of nearest neighbors connected per node.

    Returns:
        ``(B, K, K)`` bool adjacency, symmetric, with zero diagonal and no
        edges touching padding slots.
    """
    k = concept_embeddings.shape[1]
    x = F.normalize(concept_embeddings, dim=-1, eps=1e-8)
    similarity = torch.matmul(x, x.transpose(-1, -2))  # (B, K, K)

    valid_pair = concept_mask.unsqueeze(1) & concept_mask.unsqueeze(2)
    eye = torch.eye(k, dtype=torch.bool, device=similarity.device).unsqueeze(0)
    similarity = similarity.masked_fill(~valid_pair | eye, float("-inf"))

    n_neighbors = min(top_k, k - 1)
    if n_neighbors <= 0:
        return torch.zeros_like(valid_pair)

    topk_idx = similarity.topk(n_neighbors, dim=-1).indices  # (B, K, n_neighbors)
    adjacency = torch.zeros_like(similarity, dtype=torch.bool)
    adjacency.scatter_(-1, topk_idx, True)
    adjacency = adjacency & valid_pair & ~eye
    return adjacency | adjacency.transpose(-1, -2)


def build_graph_adjacency(
    edge_weights: torch.Tensor,
    concept_embeddings: torch.Tensor,
    concept_mask: torch.Tensor,
    top_k_similar: int = 5,
) -> torch.Tensor:
    """Combine cached edge weights, cosine-similarity edges, and self-loops.

    Args:
        edge_weights: ``(B, K, K)`` cached co-occurrence/anatomy/semantic
            edge weights in ``[0, 1]``.
        concept_embeddings: ``(B, K, D)`` cached concept embeddings.
        concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.
        top_k_similar: Number of cosine-similarity neighbors per node.

    Returns:
        ``(B, K, K)`` bool adjacency including self-loops for valid concepts.
    """
    k = edge_weights.shape[-1]
    eye = torch.eye(k, dtype=torch.bool, device=edge_weights.device).unsqueeze(0)
    cooccurrence_adj = edge_weights > 0
    cosine_adj = compute_topk_cosine_adjacency(concept_embeddings, concept_mask, top_k_similar)
    valid_pair = concept_mask.unsqueeze(1) & concept_mask.unsqueeze(2)
    return (cooccurrence_adj | cosine_adj | eye) & valid_pair


def compute_laplacian_pe(
    adjacency: torch.Tensor, concept_mask: torch.Tensor, pos_enc_dim: int = 6, eps: float = 1e-6
) -> torch.Tensor:
    """Compute Laplacian-eigenvector structural positional encodings.

    Args:
        adjacency: ``(B, K, K)`` symmetric float adjacency with zero
            diagonal (no self-loops).
        concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.
        pos_enc_dim: Number of smallest-eigenvalue eigenvectors to keep.
        eps: Diagonal jitter added before eigendecomposition for stability.

    Returns:
        ``(B, K, pos_enc_dim)`` float positional encodings, zeroed for
        padding slots.
    """
    k = adjacency.shape[-1]
    degree = adjacency.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - adjacency
    eye = torch.eye(k, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
    laplacian = laplacian + eps * eye

    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian.float())
    pos_encoding = eigenvectors[..., :pos_enc_dim]
    pos_encoding = pos_encoding * concept_mask.unsqueeze(-1).to(pos_encoding.dtype)
    return pos_encoding.to(adjacency.dtype)


class ConceptGraphBuilder(nn.Module):
    """Builds the graph attention bias and structural positional encoding.

    Args:
        hidden_dim: Model hidden dimension.
        pos_enc_dim: Number of Laplacian eigenvectors used for the
            structural positional encoding.
        top_k_similar: Number of cosine-similarity neighbors per node.
        neg_bias: Additive attention-bias value for non-neighbor pairs
            (kept moderate, not ``-inf``, to remain finite under fp16 AMP).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        pos_enc_dim: int = 6,
        top_k_similar: int = 5,
        neg_bias: float = -1e4,
    ) -> None:
        super().__init__()
        self.top_k_similar = top_k_similar
        self.pos_enc_dim = pos_enc_dim
        self.neg_bias = neg_bias
        self.pos_proj = nn.Linear(pos_enc_dim, hidden_dim)

    def forward(
        self,
        concept_embeddings: torch.Tensor,
        edge_weights: torch.Tensor,
        concept_mask: torch.Tensor,
    ) -> tuple:
        """Build the attention bias and positional encoding for one batch.

        Args:
            concept_embeddings: ``(B, K, D)`` cached concept embeddings.
            edge_weights: ``(B, K, K)`` cached edge-weight matrix in ``[0, 1]``.
            concept_mask: ``(B, K)`` bool, ``True`` for real concept slots.

        Returns:
            Tuple ``(attention_bias, pos_encoding)``:
              - ``attention_bias``: ``(B, K, K)`` additive bias (0 for
                connected pairs, ``neg_bias`` otherwise).
              - ``pos_encoding``: ``(B, K, hidden_dim)`` projected structural
                positional encoding.
        """
        k = edge_weights.shape[-1]
        eye = torch.eye(k, dtype=torch.bool, device=edge_weights.device).unsqueeze(0)

        adjacency = build_graph_adjacency(edge_weights, concept_embeddings, concept_mask, self.top_k_similar)
        attention_bias = torch.where(
            adjacency,
            torch.zeros_like(edge_weights),
            torch.full_like(edge_weights, self.neg_bias),
        )

        structural_adj = (adjacency & ~eye).to(edge_weights.dtype)
        laplacian_pe = compute_laplacian_pe(structural_adj, concept_mask, self.pos_enc_dim)
        pos_encoding = self.pos_proj(laplacian_pe)

        return attention_bias, pos_encoding
