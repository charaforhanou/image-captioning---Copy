import yaml, copy
from pathlib import Path

base = yaml.safe_load(open("config/config_honest.yaml"))
outdir = Path("config/ablations")
outdir.mkdir(parents=True, exist_ok=True)

variants = ["full", "noconcept", "nograph", "nomemory", "simplefusion"]

for name in variants:
    cfg = copy.deepcopy(base)
    cfg["model_config"] = f"config/ablations/model_{name}.yaml"
    cfg["paths"]["checkpoint_dir"] = f"checkpoints_abl_{name}"
    # Option A : ablation SANS loss auxiliaire -> poids classification = 0
    cfg["training"]["classification_loss_weight"] = 0.0
    with open(outdir / f"config_{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Ecrit: config/ablations/config_{name}.yaml  (classification_loss_weight=0)")
