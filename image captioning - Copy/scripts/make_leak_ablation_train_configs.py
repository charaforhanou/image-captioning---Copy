import yaml, copy
from pathlib import Path

base = yaml.safe_load(open("results/ablations/configs/full_config.yaml"))
outdir = Path("config/ablations_leak")
outdir.mkdir(parents=True, exist_ok=True)

variants = ["full", "noconcept", "nograph", "nomemory", "simplefusion"]

for name in variants:
    cfg = copy.deepcopy(base)
    cfg["model_config"] = f"config/ablations_leak/model_{name}.yaml"
    cfg["paths"]["checkpoint_dir"] = f"checkpoints_leak_{name}"
    cfg["paths"]["results_dir"] = f"results/ablations_leak/{name}"
    cfg["training"]["num_workers"] = 0
    cfg["evaluation"]["num_workers"] = 0
    # pas de loss auxiliaire dans cette version (poids 0 par securite si le trainer la calcule)
    cfg["training"]["classification_loss_weight"] = 0.0
    with open(outdir / f"config_{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Ecrit: config/ablations_leak/config_{name}.yaml  (cache texte, no aux, ckpt=checkpoints_leak_{name})")
