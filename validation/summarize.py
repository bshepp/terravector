"""Markdown summary table over validation/results/eval_*.json.

Enrichment = (fraction of other ground-truth sites whose best 3x3 cell is in the
top 0.1% / 0.5% of the county) / (same fraction for 500 random 3x3 control
blocks). "matched" uses landform-matched controls (evaluate_retrieval.py
--landform) and is "-" for reports run without them.
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"
rows = []
for f in sorted(d.glob("eval_*.json")):
    r = json.load(open(f))
    s = r["summary"]
    enr = s.get("enrichment", {})
    menr = s.get("matched_enrichment", {})
    ofa = s.get("others_frac_above", {})
    rows.append((f.stem.replace("eval_", ""), s["mean_other_pct"], s["mean_random_pct_median"], s.get("mean_matched_random_pct_median"),
                 s["wall_to_wall_pct"] or 0.0, ofa.get("99.9"), enr.get("99.9"), menr.get("99.9"), ofa.get("99.5"), enr.get("99.5"), menr.get("99.5"), s["mean_AP"]))
fmt = lambda v, f, suffix="": ("-" if v is None else format(v, f) + suffix)  # noqa: E731
print("| vectors | other sites mean pct | random block median | matched block median | wall-to-wall pct | frac. of other sites in top 0.1% | enrichment vs random | vs matched | in top 0.5% | enrichment | vs matched | mean AP |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, o, rnd, mrnd, w, f1, e1, m1, f5, e5, m5, ap in rows:
    print(f"| {name} | {o:.1f} | {rnd:.1f} | {fmt(mrnd, '.1f')} | {w:.1f} | {fmt(f1, '.2f')} | {fmt(e1, '.1f', 'x')} | {fmt(m1, '.1f', 'x')} | {fmt(f5, '.2f')} | {fmt(e5, '.1f', 'x')} | {fmt(m5, '.1f', 'x')} | {ap:.4f} |")
