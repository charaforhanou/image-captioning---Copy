import subprocess, sys
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["full", "noconcept", "nograph", "nomemory", "simplefusion"]

METRIC_RE = re.compile(r"^(BLEU-1|BLEU-2|BLEU-3|BLEU-4|ROUGE-L|CIDEr)\s+([-\d.]+)")

rows = []
for name in VARIANTS:
    ckpt = ROOT / f"checkpoints_leak_{name}" / "best.pt"
    cfg = f"config/ablations_leak/config_{name}.yaml"
    evaldir = ROOT / f"results/ablations_leak/{name}/eval"
    if not ckpt.exists():
        print(f"[SKIP] {name}: checkpoint absent ({ckpt})")
        continue
    print(f"\n=== EVALUATION : {name} ===")
    subprocess.run([sys.executable, "scripts/evaluate.py", "--config", cfg,
                    "--checkpoint", str(ckpt), "--results-dir", str(evaldir)], cwd=str(ROOT))

    metrics_file = evaldir / "metrics_r2gen.txt"
    row = {"variante": name}
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            m = METRIC_RE.match(line.strip())
            if m:
                row[m.group(1)] = float(m.group(2))
    rows.append(row)

if rows:
    df = pd.DataFrame(rows)
    order = ["variante", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "CIDEr"]
    df = df[[c for c in order if c in df.columns]]
    out = ROOT / "results/ablations_leak/ablation_nlg_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== TABLEAU COMPARATIF NLG (version avec fuite) ===")
    print(df.to_string(index=False))
    print(f"\nCSV: {out}")
