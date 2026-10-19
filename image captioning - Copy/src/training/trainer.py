"""AMP training loop with TensorBoard logging, checkpointing, and early stopping."""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.tokenizer import MedicalTokenizer
from src.training.callbacks import EarlyStopping
from src.training.losses import MCCAGLoss
from src.training.metrics import greedy_decode, quick_bleu4, quick_meteor, token_accuracy
from src.training.optimizers import build_optimizer, build_scheduler
from src.utils.checkpointer import Checkpointer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Runs the MCCAG training loop.

    Args:
        model: The :class:`src.models.mccag.MCCAG` instance.
        train_loader: Training :class:`~torch.utils.data.DataLoader`
            (produces batches via :func:`src.data.dataset.collate_fn`).
        val_loader: Validation ``DataLoader``.
        tokenizer: A loaded :class:`MedicalTokenizer`.
        training_cfg: The ``training`` section of ``config/config.yaml``.
        checkpoint_dir: Directory for :class:`Checkpointer` output.
        log_dir: Directory for TensorBoard logs.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: MedicalTokenizer,
        training_cfg: dict,
        checkpoint_dir: Union[str, Path],
        log_dir: Union[str, Path],
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = training_cfg

        self.grad_accum_steps = training_cfg["grad_accum_steps"]
        self.max_grad_norm = training_cfg["max_grad_norm"]
        self.num_epochs = training_cfg["num_epochs"]
        self.log_every = training_cfg["log_every"]
        self.val_log_samples = training_cfg["val_log_samples"]
        self.early_stopping_metric = training_cfg["early_stopping_metric"]
        self.fp16 = training_cfg["fp16"] and device.type == "cuda"

                # pos_weight pour la loss de classification auxiliaire (déséquilibre).
        import pandas as _pd
        _cols = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                 "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
                 "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
                 "Support Devices", "No Finding"]
        _pw = None
        try:
            _tgt = _pd.read_csv("data/processed/concept_labels/train_concept_targets.csv")
            _pos = _tgt[_cols].sum().values.astype("float32")
            _n = len(_tgt)
            _neg = _n - _pos
            import numpy as _np
            _pw_np = _np.clip(_np.where(_pos > 0, _neg / _np.maximum(_pos, 1), 1.0), 1.0, 50.0)
            _pw = torch.tensor(_pw_np, dtype=torch.float32)
        except Exception as _e:
            logger.warning(f"pos_weight non chargé ({_e}); BCE non pondérée.")

        self.criterion = MCCAGLoss(
            pad_id=tokenizer.pad_id,
            label_smoothing=training_cfg["label_smoothing"],
            coverage_weight=training_cfg["coverage_loss_weight"],
            diversity_weight=training_cfg["diversity_loss_weight"],
            classification_weight=training_cfg.get("classification_loss_weight", 0.5),
            pathology_pos_weight=_pw,
        )
        self.optimizer = build_optimizer(
            self.model, lr=training_cfg["learning_rate"], weight_decay=training_cfg["weight_decay"]
        )

        steps_per_epoch = max(1, -(-len(train_loader) // self.grad_accum_steps))  # ceil div
        total_steps = steps_per_epoch * self.num_epochs
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=training_cfg["warmup_steps"],
            total_steps=total_steps,
            base_lr=training_cfg["learning_rate"],
        )

        self.scaler = torch.amp.GradScaler(enabled=self.fp16)
        self.checkpointer = Checkpointer(
            checkpoint_dir=checkpoint_dir, save_every=training_cfg["checkpoint_every"], mode="max"
        )
        self.early_stopping = EarlyStopping(patience=training_cfg["early_stopping_patience"], mode="max")
        self.writer = SummaryWriter(log_dir=str(log_dir))

        self.global_step = 0
        self.start_epoch = 1

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train(self) -> None:
        """Run the full training loop until ``num_epochs`` or early stopping."""
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)

            logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['token_accuracy']:.4f} "
                f"val_bleu4={val_metrics['bleu4']:.4f} val_meteor={val_metrics['meteor']:.4f}"
            )

            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train_epoch/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)

            metric_value = val_metrics[self.early_stopping_metric]
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "global_step": self.global_step,
            }
            self.checkpointer.save(state, epoch=epoch, metric=metric_value)

            if self.early_stopping.step(metric_value):
                logger.info(
                    f"Early stopping after epoch {epoch}: "
                    f"no improvement in {self.early_stopping_metric} for "
                    f"{self.early_stopping.patience} epochs"
                )
                break

        self.writer.close()

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        n_batches = len(self.train_loader)
        totals = {"loss": 0.0, "ce_loss": 0.0, "coverage_loss": 0.0, "diversity_loss": 0.0, "token_accuracy": 0.0}

        self.optimizer.zero_grad()
        for step, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)

            with torch.autocast(device_type=self.device.type, enabled=self.fp16):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)
                loss = loss_dict["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            is_last = (step + 1) == n_batches
            if (step + 1) % self.grad_accum_steps == 0 or is_last:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            else:
                grad_norm = None

            with torch.no_grad():
                acc = token_accuracy(outputs["logits"], batch["input_ids"], self.tokenizer.pad_id)

            totals["loss"] += loss_dict["loss"].item()
            totals["ce_loss"] += loss_dict["ce_loss"].item()
            totals["coverage_loss"] += loss_dict["coverage_loss"].item()
            totals["diversity_loss"] += loss_dict["diversity_loss"].item()
            totals["token_accuracy"] += acc

            if (step + 1) % self.log_every == 0 or is_last:
                lr = self.scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch} step {step + 1}/{n_batches} | "
                    f"loss={loss_dict['loss'].item():.4f} ce={loss_dict['ce_loss'].item():.4f} "
                    f"cov={loss_dict['coverage_loss'].item():.4f} div={loss_dict['diversity_loss'].item():.4f} "
                    f"cls={loss_dict['classification_loss'].item():.4f} "
                    f"acc={acc:.4f} lr={lr:.2e}"
                )
                self.writer.add_scalar("train/loss", loss_dict["loss"].item(), self.global_step)
                self.writer.add_scalar("train/ce_loss", loss_dict["ce_loss"].item(), self.global_step)
                self.writer.add_scalar("train/coverage_loss", loss_dict["coverage_loss"].item(), self.global_step)
                self.writer.add_scalar("train/diversity_loss", loss_dict["diversity_loss"].item(), self.global_step)
                self.writer.add_scalar("train/token_accuracy", acc, self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)
                if grad_norm is not None:
                    self.writer.add_scalar("train/grad_norm", float(grad_norm), self.global_step)

        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        n_batches = len(self.val_loader)
        totals = {"loss": 0.0, "ce_loss": 0.0, "coverage_loss": 0.0, "diversity_loss": 0.0, "token_accuracy": 0.0}

        predictions, references = [], []
        for step, batch in enumerate(self.val_loader):
            batch = self._to_device(batch)

            with torch.autocast(device_type=self.device.type, enabled=self.fp16):
                outputs = self.model(batch)
                loss_dict = self.criterion(outputs, batch)

            acc = token_accuracy(outputs["logits"], batch["input_ids"], self.tokenizer.pad_id)
            totals["loss"] += loss_dict["loss"].item()
            totals["ce_loss"] += loss_dict["ce_loss"].item()
            totals["coverage_loss"] += loss_dict["coverage_loss"].item()
            totals["diversity_loss"] += loss_dict["diversity_loss"].item()
            totals["token_accuracy"] += acc

            if step == 0:
                n_samples = min(self.val_log_samples, batch["image"].shape[0])
                sample_batch = {
                    k: (v[:n_samples] if torch.is_tensor(v) else v[:n_samples]) for k, v in batch.items()
                }
                preds = greedy_decode(self.model, sample_batch, self.tokenizer, max_length=100)
                refs = [
                    self.tokenizer.decode(self.tokenizer.encode(t)["input_ids"], skip_special_tokens=True)
                    for t in sample_batch["target_text"]
                ]
                predictions.extend(preds)
                references.extend(refs)

        metrics = {k: v / n_batches for k, v in totals.items()}
        metrics["bleu4"] = quick_bleu4(predictions, references)
        metrics["meteor"] = quick_meteor(predictions, references)

        if epoch == 1 or epoch % 5 == 0:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                logger.info(f"  [val sample {i}] PRED: {pred}")
                logger.info(f"  [val sample {i}] REF:  {ref}")

        return metrics

    def resume(self, checkpoint_path: Union[str, Path]) -> None:
        """Restore model/optimizer/scheduler/scaler state and resume from the next epoch.

        Args:
            checkpoint_path: Path to a checkpoint saved by :class:`Checkpointer`.
        """
        ckpt = self.checkpointer.load(str(checkpoint_path), map_location=str(self.device))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {checkpoint_path} at epoch {self.start_epoch}")
