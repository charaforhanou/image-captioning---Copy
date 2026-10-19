"""Évalue toutes les ablations et agrège les métriques dans un CSV comparatif.

Pour chaque sous-dossier de results/ablations/ (hors 'configs') contenant un
checkpoint, lance scripts/evaluate.py avec la config associée, puis parse les
metrics_r2gen.txt produits pour construire un tableau comparatif.

Convention attendue :
    results/ablations/<nom>/checkpoints/best.pt   (checkpoint)
    results/ablations/configs/<nom>_config.yaml   (config d'éval)

Usage :
    python scripts/evaluate_ablations.py
    python scripts/evaluate_ablations.py --meteor
    python scripts/evaluate_ablations.py --only full sans_concepts
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABL_DIR = ROOT / "results" / "ablations"
CFG_DIR = ABL_DIR / "configs"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate.py"

# ligne type :  BLEU-1      0.4821     [0.4610, 0.5033]
METRIC_RE = re.compile(
    r"^(BLEU-1|BLEU-2|BLEU-3|BLEU-4|METEOR|ROUGE-L|CIDEr)\s+"
    r"([-\d.]+)\s+\[([-\d.]+),\s*([-\d.]+)\]"
)


def parse_args():
    p = argparse.ArgumentParser(description="Évalue et compare toutes les ablations.")
    p.add_argument("--meteor", action="store_true", help="Inclure METEOR (Java requis).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Limiter à ces ablations (noms de dossiers).")
    p.add_argument("--config-suffix", default="_config.yaml",
                   help="Suffixe des fichiers de config dans configs/.")
    p.add_argument("--skip-eval", action="store_true",
                   help="Ne relance pas l'éval, agrège seulement les metrics existants.")
    return p.parse_args()


def find_ablations(only=None):
    """Retourne [(nom, checkpoint_path, config_path), ...] pour les ablations valides."""
    found = []
    for d in sorted(ABL_DIR.iterdir()):
        if not d.is_dir() or d.name == "configs":
            continue
        if only and d.name not in only:
            continue
        ckpt = d / "checkpoints" / "best.pt"
        cfg = CFG_DIR / f"{d.name}_config.yaml"
        found.append((d.name, ckpt, cfg))
    return found


def parse_metrics_file(path: Path) -> dict:
    """Parse un metrics_r2gen.txt -> {'BLEU-1': (mean, lo, hi), ...}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = METRIC_RE.match(line.strip())
        if m:
            name, mean, lo, hi = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
            out[name] = (mean, lo, hi)
    return out


def main():
    args = parse_args()
    ablations = find_ablations(args.only)

    if not ablations:
        print(f"Aucune ablation trouvée dans {ABL_DIR}")
        return

    print(f"{len(ablations)} ablation(s) détectée(s) : {', '.join(a[0] for a in ablations)}\n")

    rows = []
    for name, ckpt, cfg in ablations:
        eval_dir = ABL_DIR / name / "eval"

        if not args.skip_eval:
            if not ckpt.exists():
                print(f"[SKIP] {name} : checkpoint absent -> {ckpt}")
                continue
            if not cfg.exists():
                print(f"[SKIP] {name} : config absente -> {cfg}")
                continue

            cmd = [
                sys.executable, str(EVAL_SCRIPT),
                "--config", str(cfg.relative_to(ROOT)),
                "--checkpoint", str(ckpt.relative_to(ROOT)),
                "--results-dir", str(eval_dir),
            ]
            if args.meteor:
                cmd.append("--meteor")

            print(f"=== Évaluation : {name} ===")
            print("  " + " ".join(cmd))
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"[ERREUR] {name} : evaluate.py a renvoyé le code {result.returncode}\n")
                continue

        metrics_file = eval_dir / "metrics_r2gen.txt"
        if not metrics_file.exists():
            print(f"[WARN] {name} : {metrics_file} introuvable, ablation ignorée dans le tableau.\n")
            continue

        parsed = parse_metrics_file(metrics_file)
        row = {"ablation": name}
        for metric, (mean, lo, hi) in parsed.items():
            row[metric] = round(mean, 4)
            row[f"{metric}_CI"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(row)
        print()

    if not rows:
        print("Aucune métrique agrégée.")
        return

    df = pd.DataFrame(rows)
    metric_order = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]
    cols = ["ablation"]
    for m in metric_order:
        if m in df.columns:
            cols += [m, f"{m}_CI"]
    df = df[[c for c in cols if c in df.columns]]

    out_csv = ABL_DIR / "ablations_comparison.csv"
    df.to_csv(out_csv, index=False)

    # affichage compact (moyennes uniquement)
    mean_cols = ["ablation"] + [m for m in metric_order if m in df.columns]
    print("\n=== Comparatif des ablations (moyennes) ===")
    print(df[mean_cols].to_string(index=False))
    print(f"\nCSV complet (avec IC) écrit : {out_csv}")


if __name__ == "__main__":
    main()