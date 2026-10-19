"""Shape-check the MCCAG model and every ablation variant with dummy tensors.

Builds a small random batch matching the real collate_fn schema and runs a
forward + backward pass through :class:`src.models.mccag.MCCAG` for the
default configuration plus each ablation variant named in
``config/model_config.yaml``. Uses ``pretrained=False`` vision backbones so
it runs offline in seconds on CPU.

Usage:
    python scripts/check_shapes.py
"""

import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mccag import MCCAG  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402


def make_dummy_batch(batch_size: int, max_concepts: int, seq_len: int, vocab_size: int) -> dict:
    n_real_concepts = max_concepts * 3 // 5
    concept_mask = torch.zeros(batch_size, max_concepts, dtype=torch.bool)
    concept_mask[:, :n_real_concepts] = True

    edge_weights = torch.rand(batch_size, max_concepts, max_concepts)
    edge_weights = (edge_weights + edge_weights.transpose(-1, -2)) / 2
    edge_weights.diagonal(dim1=-2, dim2=-1).zero_()

    return {
        "image": torch.randn(batch_size, 3, 224, 224),
        "input_ids": torch.randint(2, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "section_ids": torch.randint(0, 2, (batch_size, seq_len)),
        "concept_embeddings": torch.randn(batch_size, max_concepts, 768),
        "concept_mask": concept_mask,
        "edge_weights": edge_weights,
    }


def run_variant(name: str, model_config: dict, batch: dict, vocab_size: int) -> None:
    model = MCCAG(model_config, vocab_size=vocab_size, pad_id=0, pretrained=False)
    model.train()

    fused, fused_mask, concept_mask = model.encode(batch)
    out = model(batch)
    logits, coverage = out["logits"], out["coverage"]

    loss = logits.float().pow(2).mean() + coverage.float().pow(2).mean()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    print(
        f"[{name:20s}] fused={tuple(fused.shape)} fused_mask={tuple(fused_mask.shape)} "
        f"logits={tuple(logits.shape)} coverage={tuple(coverage.shape)} "
        f"params={n_params/1e6:.1f}M grad_covered={n_grad}/{n_params}"
    )

    assert fused.shape[0] == batch["image"].shape[0]
    assert fused_mask.shape == fused.shape[:2]
    assert logits.shape == (batch["input_ids"].shape[0], batch["input_ids"].shape[1], vocab_size)
    assert coverage.shape == (batch["input_ids"].shape[0], model_config["concepts"]["max_concepts"])
    assert not torch.isnan(logits).any()
    assert not torch.isnan(coverage).any()


def main() -> None:
    set_seed(42)
    with open(ROOT / "config" / "model_config.yaml") as f:
        base_config = yaml.safe_load(f)

    vocab_size = 100
    batch = make_dummy_batch(
        batch_size=2,
        max_concepts=base_config["concepts"]["max_concepts"],
        seq_len=20,
        vocab_size=vocab_size,
    )

    variants = {
        "MCCAG-Full": {},
        "MCCAG-NoGraph": {"use_graph": False},
        "MCCAG-NoConcept": {"use_concepts": False},
        "MCCAG-NoMemory": {"use_memory": False},
        "MCCAG-SimpleFusion": {"fusion_type": "concat"},
        "CNN-Baseline": {
            "vision_backbone": "resnet50",
            "use_concepts": False,
            "use_memory": False,
            "fusion_type": "concat",
        },
        "Transformer-Baseline": {
            "use_concepts": False,
            "use_memory": False,
            "decoder_type": "standard",
            "fusion_type": "concat",
        },
    }

    for name, overrides in variants.items():
        config = copy.deepcopy(base_config)
        config["ablation"].update(overrides)
        run_variant(name, config, batch, vocab_size)

    print("\nAll shape checks passed.")


if __name__ == "__main__":
    main()
