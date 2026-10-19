"""Vision encoder (Component 1): ViT-B/16 with partial fine-tuning."""

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ViTEncoder(nn.Module):
    """ViT-B/16 image encoder with the first N blocks frozen.

    Wraps a ``timm`` Vision Transformer. The patch embedding, position
    embedding, and the first ``num_frozen_layers`` transformer blocks are
    frozen (ImageNet features); the remaining blocks and the final norm are
    fine-tuned on chest X-rays.

    Args:
        model_name: ``timm`` model identifier.
        pretrained: Whether to load ImageNet-pretrained weights.
        num_frozen_layers: Number of leading transformer blocks to freeze.
        dropout: Dropout applied to the output token embeddings.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_frozen_layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        import timm  # local import: heavy dependency only needed here

        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.hidden_dim = self.backbone.embed_dim
        self.dropout = nn.Dropout(dropout)
        self._freeze_layers(num_frozen_layers)

    def _freeze_layers(self, num_frozen_layers: int) -> None:
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        self.backbone.pos_embed.requires_grad = False
        self.backbone.cls_token.requires_grad = False

        for i, block in enumerate(self.backbone.blocks):
            if i < num_frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False

        n_trainable = sum(p.requires_grad for p in self.backbone.parameters())
        n_total = sum(1 for _ in self.backbone.parameters())
        logger.info(f"ViTEncoder: {n_trainable}/{n_total} parameter tensors trainable")

    def forward(self, images: torch.Tensor) -> tuple:
        """Encode a batch of images.

        Args:
            images: ``(B, 3, H, W)`` normalized image tensor (H=W=224).

        Returns:
            Tuple ``(cls_token, patch_tokens)``:
              - ``cls_token``: ``(B, hidden_dim)``
              - ``patch_tokens``: ``(B, num_patches, hidden_dim)`` (196 for 224x224/patch16)
        """
        tokens = self.backbone.forward_features(images)  # (B, 1 + num_patches, hidden_dim)
        tokens = self.dropout(tokens)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return cls_token, patch_tokens


class ResNetEncoder(nn.Module):
    """ResNet-50 image encoder, used for the ``vision_backbone=resnet50`` ablation.

    The final 7x7x2048 conv feature map is flattened into 49 tokens and
    projected to ``hidden_dim``; their mean serves as the CLS-equivalent
    global feature.

    Args:
        hidden_dim: Output token dimensionality.
        pretrained: Whether to load ImageNet-pretrained weights.
        dropout: Dropout applied to the output token embeddings.
    """

    num_tokens = 49

    def __init__(self, hidden_dim: int = 768, pretrained: bool = True, dropout: float = 0.1) -> None:
        super().__init__()
        import torchvision  # local import: heavy dependency only needed here

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # -> (B, 2048, 7, 7)
        self.feature_dim = 2048
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(self.feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> tuple:
        """Encode a batch of images.

        Args:
            images: ``(B, 3, H, W)`` normalized image tensor (H=W=224).

        Returns:
            Tuple ``(cls_token, patch_tokens)``:
              - ``cls_token``: ``(B, hidden_dim)``, mean-pooled feature map.
              - ``patch_tokens``: ``(B, 49, hidden_dim)``.
        """
        features = self.backbone(images)  # (B, 2048, 7, 7)
        b, c, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        tokens = self.dropout(self.proj(tokens))
        cls_token = tokens.mean(dim=1)
        return cls_token, tokens
