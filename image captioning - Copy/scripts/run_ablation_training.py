import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["full", "noconcept", "nograph", "nomemory", "simplefusion"]

for name in VARIANTS:
    cfg = f"config/ablations/config_{name}.yaml"
    print(f"\n{'='*60}\n=== ENTRAINEMENT : {name} ===\n{'='*60}")
    cmd = [sys.executable, "scripts/train.py", "--config", cfg, "--run-name", f"abl_{name}"]
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[ERREUR] {name} : code {result.returncode} — on continue avec les suivants.")
    else:
        print(f"[OK] {name} terminé.")

print("\nTous les entrainements lances. Checkpoints dans checkpoints_abl_<variante>/best.pt")
