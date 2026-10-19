"""Generate a radiology report for a single chest X-ray image.

Usage:
    python scripts/predict.py --image dataset/NLMCXR_png/CXR1_1_IM-0001-3001.png
    python scripts/predict.py --image path/to/scan.png --report-context "history of cough" --beam-size 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentation import build_eval_transform  # noqa: E402
from src.data.tokenizer import MedicalTokenizer  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.models.mccag import MCCAG  # noqa: E402
from src.utils.device import get_device  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seeds import set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a report for a single chest X-ray image.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to a checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Path to a chest X-ray image.")
    parser.add_argument(
        "--report-context",
        type=str,
        default=None,
        help="Optional free text used to seed the concept pathway (e.g. clinical history).",
    )
    parser.add_argument("--beam-size", type=int, default=None, help="Override evaluation.beam_size.")
    parser.add_argument("--max-length", type=int, default=None, help="Override evaluation.max_length.")
    parser.add_argument("--min-length", type=int, default=None, help="Override evaluation.min_length.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write the result as JSON.")
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

    image_cfg = dataset_config["image"]
    transform = build_eval_transform(image_cfg["size"], image_cfg["mean"], image_cfg["std"])

    model_config["decoder"]["vocab_size"] = tokenizer.vocab_size
    model = MCCAG(model_config, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id, pretrained=False)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    predictor = Predictor(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device=device,
        max_concepts=model_config["concepts"]["max_concepts"],
        concept_dim=model_config["concepts"]["hidden_dim"],
        concepts_cfg=dataset_config["concepts"],
        cooccurrence_path=ROOT / dataset_config["processed"]["cooccurrence_file"],
    )

    eval_cfg = config["evaluation"]
    beam_cfg = {
        "beam_size": args.beam_size or eval_cfg["beam_size"],
        "max_length": args.max_length or eval_cfg["max_length"],
        "min_length": args.min_length or eval_cfg["min_length"],
        "length_penalty": eval_cfg["length_penalty"],
        "coverage_penalty": eval_cfg["coverage_penalty"],
        "temperature": eval_cfg.get("temperature", 1.0),
    }

    result = predictor.predict(args.image, report_context=args.report_context, beam_cfg=beam_cfg)

    logger.info(f"FINDINGS: {result['findings']}")
    logger.info(f"IMPRESSION: {result['impression']}")
    logger.info(
        f"beam_score={result['beam_score']:.4f} confidence_mean={result['confidence_mean']:.4f} "
        f"concept_coverage={result['concept_coverage']:.4f}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Wrote result to {output_path}")


if __name__ == "__main__":
    main()
