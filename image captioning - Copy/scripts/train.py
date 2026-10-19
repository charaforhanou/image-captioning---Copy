"""Train MCCAG on the IU X-Ray dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --max-train-samples 100 --max-val-samples 20 --num-epochs 1
    python scripts/train.py --resume checkpoints/latest.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentation import build_eval_transform, build_train_transform  # noqa: E402
from src.data.dataset import IUXRayDataset, collate_fn  # noqa: E402
from src.data.tokenizer import MedicalTokenizer  # noqa: E402
from src.models.mccag import MCCAG  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seeds import seed_worker, set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MCCAG on IU X-Ray.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Subset the train split (smoke testing)."
    )
    parser.add_argument("--max-val-samples", type=int, default=None, help="Subset the val split (smoke testing).")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override training.num_epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override training.grad_accum_steps.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from.")
    parser.add_argument("--run-name", type=str, default=None, help="Subdirectory name for TensorBoard logs.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    config = load_yaml(ROOT / args.config)
    dataset_config = load_yaml(ROOT / config["dataset_config"])
    model_config = load_yaml(ROOT / config["model_config"])

    set_seed(config["seed"])
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    tokenizer = MedicalTokenizer.load(
        ROOT / dataset_config["processed"]["tokenizer_dir"],
        max_length=dataset_config["text"]["max_length"],
    )
    logger.info(f"Loaded tokenizer with vocab_size={tokenizer.vocab_size}")

    image_cfg = dataset_config["image"]
    train_transform = build_train_transform(image_cfg["size"], image_cfg["mean"], image_cfg["std"])
    eval_transform = build_eval_transform(image_cfg["size"], image_cfg["mean"], image_cfg["std"])

    dataset_kwargs = dict(
        metadata_csv=ROOT / dataset_config["processed"]["metadata_csv"],
        images_dir=ROOT / dataset_config["raw"]["images_dir"],
        concept_cache_dir=ROOT / dataset_config["processed"]["concept_cache_dir"],
        concept_embeddings_path=ROOT / dataset_config["processed"]["concept_embeddings"],
        tokenizer=tokenizer,
        max_concepts=model_config["concepts"]["max_concepts"],
        concept_dim=model_config["concepts"]["hidden_dim"],
    )

    train_dataset = IUXRayDataset(
        split_file=ROOT / dataset_config["splits"]["train_file"], transform=train_transform, **dataset_kwargs
    )
    val_dataset = IUXRayDataset(
        split_file=ROOT / dataset_config["splits"]["val_file"], transform=eval_transform, **dataset_kwargs
    )

    if args.max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    training_cfg = config["training"]
    if args.num_epochs is not None:
        training_cfg["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        training_cfg["grad_accum_steps"] = args.grad_accum_steps

    generator = torch.Generator()
    generator.manual_seed(config["seed"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        pin_memory=training_cfg["pin_memory"],
        collate_fn=collate_fn,
    )
    logger.info(f"Train samples: {len(train_dataset)} ({len(train_loader)} batches) | "
                f"Val samples: {len(val_dataset)} ({len(val_loader)} batches)")

    model_config["decoder"]["vocab_size"] = tokenizer.vocab_size
    model = MCCAG(model_config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id, pretrained=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params / 1e6:.1f}M")

    paths = config["paths"]
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / paths["log_dir"] / run_name
    checkpoint_dir = ROOT / paths["checkpoint_dir"]

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        training_cfg=training_cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        device=device,
    )

    if args.resume:
        trainer.resume(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
