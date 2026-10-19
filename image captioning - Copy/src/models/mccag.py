"""MCCAG: Medical Concept Cross-Attention Graph (Component 8).

Wires together the seven components into the full report-generation model:

  1. :class:`src.models.vit_encoder.ViTEncoder` (or
     :class:`src.models.vit_encoder.ResNetEncoder`) -- visual stream.
  2. :class:`src.models.concept_extractor.ConceptExtractor` -- concept stream.
  3. :class:`src.models.graph_builder.ConceptGraphBuilder` -- graph attention
     bias + structural positional encoding for the concept stream.
  4. :class:`src.models.graph_transformer.GraphTransformerEncoder` -- refines
     the concept stream via graph-masked self-attention.
  5. :class:`src.models.triangular_attention.TriangularAttentionFusion` (or a
     plain concatenation) -- fuses visual + concept (+ memory) streams.
  6. :class:`src.models.memory_module.MultimodalMemory` -- FIFO memory bank
     cross-attention + concept-coverage tracking.
  7. :class:`src.models.decoder.MedicalReportDecoder` -- causal decoder.

Ablation flags (``model_config["ablation"]``) select the 7 variants from the
architecture spec purely via configuration:

  - ``use_concepts``: if ``False``, the concept mask is forced to all-padding
    before it reaches any module, so the concept stream becomes an inert
    "ghost" stream that contributes nothing to the fused output or the
    coverage vector (ViT-only behavior) without special-casing tensor shapes.
  - ``use_graph``: if ``False``, :class:`ConceptGraphBuilder` and
    :class:`GraphTransformerEncoder` are skipped; concept tokens are the raw
    :class:`ConceptExtractor` output (bag-of-concepts).
  - ``use_memory``: if ``False``, :class:`MultimodalMemory` is skipped; the
    decoder's memory-gated combination receives an all-zero memory context
    and the coverage vector is all-zero.
  - ``fusion_type``: ``"triangular"`` uses :class:`TriangularAttentionFusion`
    (visual + concept + 32 learnable memory tokens); ``"concat"`` simply
    concatenates the visual and concept streams.
  - ``vision_backbone``: ``"vit"`` (196 tokens) or ``"resnet50"`` (49 tokens);
    the visual token count is propagated to :class:`MultimodalMemory` so the
    concept-stream slice of the fused sequence is located correctly.
  - ``decoder_type``: ``"mccag"`` (default) uses the FINDINGS/IMPRESSION
    section-type embedding; ``"standard"`` omits it.
"""

import torch
import torch.nn as nn

from src.models.concept_extractor import ConceptExtractor
from src.models.decoder import MedicalReportDecoder
from src.models.graph_builder import ConceptGraphBuilder
from src.models.graph_transformer import GraphTransformerEncoder
from src.models.memory_module import MultimodalMemory
from src.models.triangular_attention import TriangularAttentionFusion
from src.models.vit_encoder import ResNetEncoder, ViTEncoder


class MCCAG(nn.Module):
    """Full medical report generation model.

    Args:
        model_config: Parsed ``config/model_config.yaml`` dict (must contain
            ``hidden_dim``, ``vision``, ``concepts``, ``graph_transformer``,
            ``fusion``, ``memory``, ``decoder``, and ``ablation`` sections).
        vocab_size: Tokenizer vocabulary size (overrides
            ``model_config["decoder"]["vocab_size"]``).
        pad_id: Tokenizer padding token id.
        pretrained: Whether the vision backbone loads ImageNet-pretrained
            weights (set ``False`` for fast, network-free shape checks).
    """

    def __init__(self, model_config: dict, vocab_size: int, pad_id: int, pretrained: bool = True) -> None:
        super().__init__()
        hidden_dim = model_config["hidden_dim"]
        ablation = model_config["ablation"]
        self.use_concepts = ablation["use_concepts"]
        self.use_graph = ablation["use_graph"]
        self.use_memory = ablation["use_memory"]
        self.fusion_type = ablation["fusion_type"]
        self.vision_backbone = ablation["vision_backbone"]
        self.decoder_type = ablation["decoder_type"]

        vision_cfg = model_config["vision"]
        if self.vision_backbone == "vit":
            self.vision_encoder = ViTEncoder(
                model_name=vision_cfg["backbone"],
                pretrained=pretrained,
                num_frozen_layers=vision_cfg["num_frozen_layers"],
                dropout=vision_cfg["dropout"],
            )
        elif self.vision_backbone == "resnet50":
            self.vision_encoder = ResNetEncoder(hidden_dim=hidden_dim, pretrained=pretrained, dropout=vision_cfg["dropout"])
        else:
            raise ValueError(f"Unknown vision_backbone: {self.vision_backbone}")

        num_visual_tokens = self.vision_encoder.num_tokens if self.vision_backbone == "resnet50" else vision_cfg["num_patches"]
        encoder_dim = self.vision_encoder.hidden_dim
        self.visual_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()

        concepts_cfg = model_config["concepts"]
        self.max_concepts = concepts_cfg["max_concepts"]
        self.concept_extractor = ConceptExtractor(input_dim=concepts_cfg["hidden_dim"], hidden_dim=hidden_dim)

        graph_cfg = model_config["graph_transformer"]
        self.graph_builder = ConceptGraphBuilder(
            hidden_dim=hidden_dim,
            pos_enc_dim=graph_cfg["pos_enc_dim"],
            top_k_similar=graph_cfg["top_k_similar"],
        )
        self.graph_transformer = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=graph_cfg["num_layers"],
            num_heads=graph_cfg["num_heads"],
            ffn_dim=graph_cfg["ffn_dim"],
            dropout=graph_cfg["dropout"],
        )

        fusion_cfg = model_config["fusion"]
        if self.fusion_type == "triangular":
            self.fusion = TriangularAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=fusion_cfg["num_heads"],
                memory_tokens=fusion_cfg["memory_tokens"],
                sparse_topk=fusion_cfg["sparse_topk"],
                dropout=fusion_cfg["dropout"],
            )
        elif self.fusion_type == "concat":
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        if self.use_memory:
            memory_cfg = model_config["memory"]
            self.memory = MultimodalMemory(
                hidden_dim=hidden_dim,
                memory_size=memory_cfg["memory_size"],
                top_k=memory_cfg["top_k"],
                decay=memory_cfg["decay"],
                num_heads=memory_cfg["num_heads"],
                num_visual_tokens=num_visual_tokens,
                max_concepts=self.max_concepts,
            )
        else:
            self.memory = None

        decoder_cfg = model_config["decoder"]
        self.decoder = MedicalReportDecoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            hidden_dim=hidden_dim,
            num_layers=decoder_cfg["num_layers"],
            num_heads=decoder_cfg["num_heads"],
            ffn_dim=decoder_cfg["ffn_dim"],
            dropout=decoder_cfg["dropout"],
            max_length=decoder_cfg["max_length"],
            use_section_embedding=(self.decoder_type == "mccag"),
        )
        self.pathology_head = nn.Linear(hidden_dim, 14)

    def encode(self, batch: dict) -> tuple:
        """Run the full encoder pathway (vision + concept + graph + fusion).

        Args:
            batch: Dict with keys ``image (B,3,H,W)``, ``concept_embeddings
                (B,K,D)``, ``concept_mask (B,K)``, ``edge_weights (B,K,K)``
                as produced by :func:`src.data.dataset.collate_fn`.

        Returns:
            Tuple ``(fused, fused_mask, concept_mask)``:
              - ``fused``: ``(B, N, hidden_dim)`` fused encoder sequence.
              - ``fused_mask``: ``(B, N)`` bool, ``True`` for valid positions.
              - ``concept_mask``: ``(B, max_concepts)`` bool, the (possibly
                ``use_concepts``-overridden) concept mask, for the coverage
                loss.
        """
        images = batch["image"]
        _, visual_tokens = self.vision_encoder(images)
        visual_tokens = self.visual_proj(visual_tokens)
        b, n_visual, _ = visual_tokens.shape
        device = visual_tokens.device

        concept_embeddings = batch["concept_embeddings"]
        concept_mask = batch["concept_mask"]
        if not self.use_concepts:
            concept_mask = torch.zeros_like(concept_mask)

        concept_tokens = self.concept_extractor(concept_embeddings, concept_mask)

        if self.use_graph:
            edge_weights = batch["edge_weights"]
            attention_bias, pos_encoding = self.graph_builder(concept_embeddings, edge_weights, concept_mask)
            concept_tokens = self.graph_transformer(concept_tokens, pos_encoding, attention_bias, concept_mask)

        if self.fusion_type == "triangular":
            fused, fused_mask = self.fusion(visual_tokens, concept_tokens, concept_mask)
        else:
            visual_mask = torch.ones(b, n_visual, dtype=torch.bool, device=device)
            fused = torch.cat([visual_tokens, concept_tokens], dim=1)
            fused_mask = torch.cat([visual_mask, concept_mask], dim=1)

        return fused, fused_mask, concept_mask

    def forward(self, batch: dict) -> dict:
        """Teacher-forced forward pass.

        Args:
            batch: Dict as produced by :func:`src.data.dataset.collate_fn`,
                additionally containing ``input_ids (B,T)``, ``section_ids
                (B,T)``, ``attention_mask (B,T)``.

        Returns:
            Dict with keys:
              - ``logits``: ``(B, T, vocab_size)``.
              - ``coverage``: ``(B, max_concepts)`` concept-attention coverage.
              - ``concept_mask``: ``(B, max_concepts)`` bool (post
                ``use_concepts`` override), for the coverage loss target.
        """
        fused, fused_mask, concept_mask = self.encode(batch)

        decoder_hidden = self.decoder(
            input_ids=batch["input_ids"],
            section_ids=batch["section_ids"],
            attention_mask=batch["attention_mask"],
            encoder_out=fused,
            encoder_mask=fused_mask,
        )

        if self.use_memory:
            memory_context, coverage = self.memory(decoder_hidden, fused, fused_mask)
        else:
            memory_context = torch.zeros_like(decoder_hidden)
            coverage = torch.zeros(
                decoder_hidden.shape[0], self.max_concepts, device=decoder_hidden.device, dtype=decoder_hidden.dtype
            )

        logits = self.decoder.compute_logits(decoder_hidden, memory_context)

        mask_f = fused_mask.unsqueeze(-1).to(fused.dtype)
        pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        pathology_logits = self.pathology_head(pooled)

        return {
            "logits": logits,
            "coverage": coverage,
            "concept_mask": concept_mask,
            "pathology_logits": pathology_logits,
        }