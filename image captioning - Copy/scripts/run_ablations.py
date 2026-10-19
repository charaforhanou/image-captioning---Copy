import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- ablations
# Each entry: name -> dict of overrides applied to the MODEL config.
# Adapt keys to match config/model_config.yaml exactly.
ABLATIONS = {
    "full": {},  # baseline: complete MCCAG

    # --- Concept graph branch ---
    "no_concept_graph": {
        "concept_graph.enabled": False,
    },
    "concepts_no_graph": {  # concepts as flat set, no GraphTransformerEncoder
        "concept_graph.use_graph_encoder": False,
    },
    "graph_1layer": {
        "concept_graph.num_layers": 1,
    },

    # --- Triangular cross-modal attention ---
    "no_triangular_fusion": {  # fall back to plain text->image cross-attn
        "fusion.type": "vanilla_cross_attention",
    },
    "no_text_to_concept": {
        "fusion.branches": ["text_to_image", "concept_to_image"],
    },
    "no_concept_to_image": {
        "fusion.branches": ["text_to_image", "text_to_concept"],
    },

    # --- Multimodal memory ---
    "no_memory": {
        "memory.enabled": False,
    },
    "no_coverage_tracking": {
        "memory.coverage_tracking": False,
    },

    # --- Decoding ---
    "greedy_decoding": {
        "__train__": {"evaluation.beam_size": 1},
    },
    "no_coverage_penalty": {
        "__train__": {"evaluation.coverage_penalty": 0.0},
    },
}

# Loss-term ablations live in the TRAIN config
LOSS_ABLATIONS = {
    "no_coverage_loss": {"training.coverage_loss_weight": 0.0},
    "no_diversity_loss": {"training.diversity_loss_weight": 0.0},
    "no_label_smoothing": {"training.label_smoothing": 0.0},
}


# ---------------------------------------------------------------- utils
def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def set_nested(cfg: dict, dotted_key: str, value):
    """set_nested(cfg, 'fusion.branches', [...]) -> cfg['fusion']['branches'] = [...]"""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if k == "__train__":
            continue
        set_nested(cfg, k, v)
    return cfg


def run(cmd: list) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(map(str, cmd))}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="config/config.yaml")
    ap.add_argument("--out", default="results/ablations")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Run only these ablation names")
    ap.add_argument("--skip-train", action="store_true",
                    help="Evaluate existing checkpoints only")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override num_epochs (useful for short ablation runs)")
    args = ap.parse_args()

    base_cfg = load_yaml(ROOT / args.base_config)
    base_model_cfg = load_yaml(ROOT / base_cfg["model_config"])

    out_dir = ROOT / args.out
    cfg_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_ablations = {**ABLATIONS, **{k: {"__train__": v} for k, v in LOSS_ABLATIONS.items()}}
    names = args.only or list(all_ablations.keys())

    summary = {}

    for name in names:
        overrides = all_ablations[name]
        print(f"\n{'=' * 70}\n  ABLATION: {name}\n{'=' * 70}")

        # --- build variant configs
        model_cfg = apply_overrides(base_model_cfg, overrides)
        train_cfg = copy.deepcopy(base_cfg)
        for k, v in overrides.get("__train__", {}).items():
            set_nested(train_cfg, k, v)
        if args.epochs is not None:
            set_nested(train_cfg, "training.num_epochs", args.epochs)

        model_cfg_path = cfg_dir / f"{name}_model.yaml"
        train_cfg_path = cfg_dir / f"{name}_config.yaml"

        ckpt_dir = out_dir / name / "checkpoints"
        res_dir = out_dir / name / "results"
        set_nested(train_cfg, "model_config", str(model_cfg_path.relative_to(ROOT)).replace("\\", "/"))
        set_nested(train_cfg, "paths.checkpoint_dir", str(ckpt_dir.relative_to(ROOT)).replace("\\", "/"))
        set_nested(train_cfg, "paths.results_dir", str(res_dir.relative_to(ROOT)).replace("\\", "/"))

        dump_yaml(model_cfg, model_cfg_path)
        dump_yaml(train_cfg, train_cfg_path)

        # --- train
        if not args.skip_train:
            run([sys.executable, "scripts/train.py", "--config", str(train_cfg_path.relative_to(ROOT))])

        # --- evaluate
        ckpt = ckpt_dir / "best_model.pth"   # adapt to your checkpoint naming
        run([
            sys.executable, "scripts/evaluate.py",
            "--config", str(train_cfg_path.relative_to(ROOT)),
            "--checkpoint", str(ckpt.relative_to(ROOT)),
            "--split", "test",
        ])

        # --- collect metrics
        metrics_path = res_dir / "metrics.json"   # adapt to evaluate.py output
        if metrics_path.exists():
            summary[name] = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            summary[name] = {"error": "metrics.json not found"}
            print(f"[warn] no metrics at {metrics_path}")

    # ---------------------------------------------------------------- report
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_table(summary)


def print_table(summary: dict):
    metrics = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L"]
    header = f"{'Ablation':<24}" + "".join(f"{m:>10}" for m in metrics)
    print("\n" + header)
    print("-" * len(header))

    base = summary.get("full", {})
    for name, m in summary.items():
        row = f"{name:<24}"
        for k in metrics:
            v = m.get(k)
            row += f"{v:>10.3f}" if isinstance(v, (int, float)) else f"{'--':>10}"
        print(row)

    # deltas vs full model
    if base:
        print("\nDelta vs full model:")
        print("-" * len(header))
        for name, m in summary.items():
            if name == "full":
                continue
            row = f"{name:<24}"
            for k in metrics:
                v, b = m.get(k), base.get(k)
                if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                    row += f"{v - b:>+10.3f}"
                else:
                    row += f"{'--':>10}"
            print(row)


def latex_table(summary: dict, path: Path):
    """Optional: emit a SCITEPRESS-ready LaTeX table."""
    metrics = ["BLEU_1", "BLEU_4", "METEOR", "ROUGE_L"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on IU X-Ray (test split, 590 studies).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\hline",
        "Model & " + " & ".join(m.replace("_", "-") for m in metrics) + r" \\",
        r"\hline",
    ]
    for name, m in summary.items():
        vals = " & ".join(
            f"{m[k]:.3f}" if isinstance(m.get(k), (int, float)) else "--" for k in metrics
        )
        lines.append(f"{name.replace('_', ' ')} & {vals} " + r"\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()