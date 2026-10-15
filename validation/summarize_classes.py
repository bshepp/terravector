"""Cross-configuration table over validation/results/classes_*.json.

Prints, per class, the top-1000 enrichment (P@1000 / base rate) with its 95%
bootstrap interval for every configuration, then the macro averages.

    python validation/summarize_classes.py [results_dir] [--k 100|1000]
"""
import json
import sys
from pathlib import Path

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
k = "1000"
if "--k" in sys.argv:
    k = sys.argv[sys.argv.index("--k") + 1]
d = Path(args[0]) if args else Path(__file__).resolve().parent / "results"
files = sorted(d.glob("classes_*.json"))
reports = {f.stem.replace("classes_", ""): json.load(open(f)) for f in files}
names = list(reports)
classes = []
for r in reports.values():
    for c in r["classes"]:
        if c not in classes:
            classes.append(c)

print(f"Top-{k} enrichment over the class base rate (1x = random); point [95% CI over queries]\n")
print("| class | cells | base % | " + " | ".join(names) + " |")
print("|---|---|---|" + "---|" * len(names))
for c in classes:
    cells = base = None
    row = []
    for n in names:
        s = reports[n]["classes"].get(c, {}).get("summary")
        if s is None:
            row.append("-"); continue
        cells, base = s["n_cells"], s["base_rate"]
        p, lo, hi = s[f"enr@{k}"]
        row.append(f"{p:.1f}x [{lo:.1f}, {hi:.1f}]")
    print(f"| {c} | {cells} | {100 * base:.2f} | " + " | ".join(row) + " |")
print("\n| macro | " + " | ".join(names) + " |")
print("|---|" + "---|" * len(names))
for key, label in ((f"enr@{k}", f"mean enrichment@{k}"), (f"geo_mean_enr@{k}", f"geometric mean enrichment@{k}"),
                   ("median_pct", "median percentile of same-class cells")):
    vals = []
    for n in names:
        s = reports[n]["summary"]
        if key in s:
            vals.append(f"{s[key]:.1f}" + ("x" if "enr" in key else ""))
        elif key.startswith("geo_mean"):
            v = np.exp(np.mean([np.log(max(c["summary"][f"enr@{k}"][0], 1e-3)) for c in reports[n]["classes"].values()]))
            vals.append(f"{v:.1f}x")
        else:
            vals.append("-")
    print(f"| {label} | " + " | ".join(vals) + " |")
