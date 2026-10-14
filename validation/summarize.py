"""Markdown summary table over validation/results/eval_*.json."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
rows = []
for f in sorted(d.glob("eval_*.json")):
    r = json.load(open(f))
    s = r["summary"]
    rows.append((f.stem.replace("eval_", ""), s["mean_other_pct"], s["mean_random_pct_median"],
                 s["wall_to_wall_pct"] or 0.0, s["mean_hits"]["1000"], s["mean_hits"]["10000"], s["mean_AP"]))
print("| vectors | other sites mean pct | random block median | wall-to-wall pct | hits@1k | hits@10k | mean AP |")
print("|---|---|---|---|---|---|---|")
for name, o, rnd, w, h1, h10, ap in rows:
    print(f"| {name} | {o:.1f} | {rnd:.1f} | {w:.1f} | {h1:.2f} | {h10:.2f} | {ap:.4f} |")
