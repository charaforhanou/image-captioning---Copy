import yaml, copy
from pathlib import Path

base = yaml.safe_load(open("results/ablations/configs/full_model.yaml"))
outdir = Path("config/ablations_leak")
outdir.mkdir(parents=True, exist_ok=True)

if "ablation" not in base:
    raise SystemExit("full_model.yaml n a pas de section 'ablation' — arrete, on doit l ajouter avant.")

variants = {
    "full":         {},
    "noconcept":    {"use_concepts": False},
    "nograph":      {"use_graph": False},
    "nomemory":     {"use_memory": False},
    "simplefusion": {"fusion_type": "concat"},
}

for name, changes in variants.items():
    cfg = copy.deepcopy(base)
    cfg["ablation"].update(changes)
    with open(outdir / f"model_{name}.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Ecrit: config/ablations_leak/model_{name}.yaml  ->  {cfg['ablation']}")
