"""Training losses for MCCAG.

Combine quatre termes :
  - Cross-entropy lissée sur la séquence cible décalée (génération).
  - Coverage loss : pénalise l'attention mémoire sur les slots concept vides.
  - Diversity loss : entropie négative, décourage les répétitions.
  - Classification auxiliaire : BCE pondérée entre les 14 logits de pathologie
    et les labels CheXpert cibles. Force la représentation fusionnée à encoder
    l'information pathologique (multi-task), pour casser le biais de normalité.
    Supervision uniquement — aucun label en entrée à l'inférence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCCAGLoss(nn.Module):
    def __init__(
        self,
        pad_id: int,
        label_smoothing: float = 0.1,
        coverage_weight: float = 0.15,
        diversity_weight: float = 0.1,
        classification_weight: float = 0.5,
        pathology_pos_weight=None,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.coverage_weight = coverage_weight
        self.diversity_weight = diversity_weight
        self.classification_weight = classification_weight
        if pathology_pos_weight is not None:
            self.register_buffer("pathology_pos_weight", pathology_pos_weight)
        else:
            self.pathology_pos_weight = None

    def forward(self, outputs: dict, batch: dict) -> dict:
        logits = outputs["logits"]
        coverage = outputs["coverage"]
        concept_mask = outputs["concept_mask"]
        input_ids = batch["input_ids"]

        vocab_size = logits.shape[-1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=self.pad_id,
            label_smoothing=self.label_smoothing,
        )

        coverage_loss = (coverage * (~concept_mask).to(coverage.dtype)).sum(dim=-1).mean()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)
        valid = (shift_labels != self.pad_id).to(token_entropy.dtype)
        mean_entropy = (token_entropy * valid).sum() / valid.sum().clamp(min=1)
        diversity_loss = -mean_entropy

        # --- Classification auxiliaire ---
        if "pathology_logits" in outputs and "pathology_label" in batch:
            pos_weight = self.pathology_pos_weight
            if pos_weight is not None:
                pos_weight = pos_weight.to(outputs["pathology_logits"].device)
            classification_loss = F.binary_cross_entropy_with_logits(
                outputs["pathology_logits"],
                batch["pathology_label"],
                pos_weight=pos_weight,
            )
        else:
            classification_loss = torch.zeros((), device=logits.device)

        total = (
            ce_loss
            + self.coverage_weight * coverage_loss
            + self.diversity_weight * diversity_loss
            + self.classification_weight * classification_loss
        )

        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "coverage_loss": coverage_loss.detach(),
            "diversity_loss": diversity_loss.detach(),
            "classification_loss": classification_loss.detach(),
        }
