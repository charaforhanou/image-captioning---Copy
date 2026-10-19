"""Beam-search generation and the full BLEU/ROUGE/METEOR/CIDEr evaluation suite."""

from pathlib import Path
from typing import Dict, Union

import pandas as pd
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from src.data.tokenizer import MedicalTokenizer
from src.evaluation.ensemble_metrics import compute_all_metrics_with_ci
from src.inference.beam_search import beam_search_decode
from src.inference.postprocessing import postprocess_generation
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Generates beam-search predictions over a test loader and scores them.

    Args:
        model: A :class:`src.models.mccag.MCCAG` instance.
        test_loader: Test :class:`~torch.utils.data.DataLoader`.
        tokenizer: A loaded :class:`MedicalTokenizer`.
        eval_cfg: The ``evaluation`` section of ``config/config.yaml``
            (``beam_size, max_length, min_length, length_penalty,
            coverage_penalty, temperature, n_bootstrap, confidence, seed``).
        device: Device to run generation on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        tokenizer: MedicalTokenizer,
        eval_cfg: dict,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.cfg = eval_cfg
        self.device = device

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

    @torch.no_grad()
    def generate(self) -> pd.DataFrame:
        """Run beam search over the full test loader.

        Returns:
            DataFrame with one row per sample: ``sample_id, prediction,
            reference, beam_score, confidence_mean, concept_coverage,
            num_concepts``.
        """
        self.model.eval()
        n_total = len(self.test_loader.dataset)
        rows = []
        for batch in self.test_loader:
            batch = self._to_device(batch)
            outputs = beam_search_decode(
                self.model,
                batch,
                self.tokenizer,
                beam_size=self.cfg["beam_size"],
                max_length=self.cfg["max_length"],
                min_length=self.cfg["min_length"],
                length_penalty=self.cfg["length_penalty"],
                coverage_penalty=self.cfg["coverage_penalty"],
                temperature=self.cfg.get("temperature", 1.0),
            )
            for i, out in enumerate(outputs):
                prediction = postprocess_generation(out["ids"], self.tokenizer)["full_text"]
                ref_ids = self.tokenizer.encode(batch["target_text"][i])["input_ids"]
                reference = postprocess_generation(ref_ids, self.tokenizer)["full_text"]
                rows.append(
                    {
                        "sample_id": batch["uid"][i],
                        "prediction": prediction,
                        "reference": reference,
                        "beam_score": out["beam_score"],
                        "confidence_mean": out["confidence_mean"],
                        "concept_coverage": out["concept_coverage"],
                        "num_concepts": int(batch["concept_mask"][i].sum().item()),
                    }
                )
            logger.info(f"Generated {len(rows)}/{n_total} predictions")
        return pd.DataFrame(rows)

    def evaluate(self, results_dir: Union[str, Path]) -> Dict[str, Dict[str, float]]:
        """Generate predictions, write the CSV + report, and return the metrics.

        Args:
            results_dir: Output directory for ``test_predictions.csv`` and
                ``evaluation_report.md``.

        Returns:
            The bootstrap metric dict from :func:`compute_all_metrics_with_ci`.
        """
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        df = self.generate()
        csv_columns = ["sample_id", "prediction", "reference", "beam_score", "confidence_mean"]
        predictions_path = results_dir / "test_predictions.csv"
        df[csv_columns].to_csv(predictions_path, index=False)
        logger.info(f"Wrote {len(df)} predictions to {predictions_path}")

        metrics = compute_all_metrics_with_ci(
            df["prediction"].tolist(),
            df["reference"].tolist(),
            n_bootstrap=self.cfg.get("n_bootstrap", 1000),
            confidence=self.cfg.get("confidence", 0.95),
            seed=self.cfg.get("seed", 42),
        )

        report_path = results_dir / "evaluation_report.md"
        self._write_report(df, metrics, report_path)

        metrics_path = results_dir / "metrics.txt"
        self._write_metrics_txt(df, metrics, metrics_path)
        return metrics

    def _write_metrics_txt(self, df: pd.DataFrame, metrics: Dict[str, Dict[str, float]], metrics_path: Path) -> None:
        label_map = {
            "bleu1": "BLEU-1",
            "bleu2": "BLEU-2",
            "bleu3": "BLEU-3",
            "bleu4": "BLEU-4",
            "rougeL_precision": "ROUGE-L (P)",
            "rougeL_recall": "ROUGE-L (R)",
            "rougeL_f1": "ROUGE-L (F1)",
            "meteor": "METEOR",
            "cider": "CIDEr",
        }
        order = list(label_map)

        lines = [f"Test set size: {len(df)} samples", ""]
        for key in order:
            m = metrics[key]
            lines.append(
                f"{label_map[key]}: mean={m['mean']:.4f} std={m['std']:.4f} "
                f"95% CI=[{m['ci_low']:.4f}, {m['ci_high']:.4f}]"
            )
        metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Wrote metrics summary to {metrics_path}")

    def _write_report(self, df: pd.DataFrame, metrics: Dict[str, Dict[str, float]], report_path: Path) -> None:
        lengths_pred = df["prediction"].str.split().apply(len)
        lengths_ref = df["reference"].str.split().apply(len)

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        df = df.copy()
        df["rougeL_f1"] = [
            scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(df["prediction"], df["reference"])
        ]
        worst = df.nsmallest(min(10, len(df)), "rougeL_f1")

        label_map = {
            "bleu1": "BLEU-1",
            "bleu2": "BLEU-2",
            "bleu3": "BLEU-3",
            "bleu4": "BLEU-4",
            "rougeL_precision": "ROUGE-L (P)",
            "rougeL_recall": "ROUGE-L (R)",
            "rougeL_f1": "ROUGE-L (F1)",
            "meteor": "METEOR",
            "cider": "CIDEr",
        }
        order = [
            "bleu1",
            "bleu2",
            "bleu3",
            "bleu4",
            "rougeL_precision",
            "rougeL_recall",
            "rougeL_f1",
            "meteor",
            "cider",
        ]

        lines = [
            "# MCCAG Evaluation Report",
            "",
            f"Test set size: {len(df)} samples",
            "",
            "## Captioning Metrics",
            "",
            "| Metric | Mean | Std | 95% CI |",
            "|---|---|---|---|",
        ]
        for key in order:
            m = metrics[key]
            lines.append(f"| {label_map[key]} | {m['mean']:.4f} | {m['std']:.4f} | [{m['ci_low']:.4f}, {m['ci_high']:.4f}] |")

        lines += [
            "",
            "## Length Statistics (words)",
            "",
            "| | Mean | Std | Min | Max |",
            "|---|---|---|---|---|",
            f"| Prediction | {lengths_pred.mean():.1f} | {lengths_pred.std():.1f} | {lengths_pred.min()} | {lengths_pred.max()} |",
            f"| Reference | {lengths_ref.mean():.1f} | {lengths_ref.std():.1f} | {lengths_ref.min()} | {lengths_ref.max()} |",
            "",
            "## Concept Coverage",
            "",
            f"- Mean concepts available per sample: {df['num_concepts'].mean():.2f}",
            f"- Mean memory-attention coverage of valid concept slots: {df['concept_coverage'].mean():.4f}",
            "",
            "## Worst 10 Examples (by ROUGE-L F1)",
            "",
        ]
        for _, row in worst.iterrows():
            lines += [
                f"### {row['sample_id']} (ROUGE-L F1 = {row['rougeL_f1']:.4f})",
                "",
                f"- **Prediction**: {row['prediction']}",
                f"- **Reference**: {row['reference']}",
                "",
            ]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote evaluation report to {report_path}")
