"""Query-level bootstrap confidence intervals for an evaluate_retrieval.py report.

The dominant uncertainty in the protocol is the small number of ground-truth
queries (nine), not the 500 control blocks, so queries are resampled with
replacement and each summary statistic is recomputed per replicate. Also
prints the per-class breakdown (query class x other-site class) of the
fraction of other sites at or above the 99.5th percentile.

    python validation/bootstrap_ci.py validation/results/eval_full_p128_pool3.json [more.json ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

B = 5000
THRESH = ("99.9", "99.5")


def stats(qs):
    """Summary statistics over a list of per-query dicts."""
    out = {}
    for t in THRESH:
        o = np.mean([q["others_frac_above"][t] for q in qs])
        r = np.mean([q["random_frac_above"][t] for q in qs])
        out[f"frac_above_{t}"] = o
        out[f"enrich_{t}"] = o / max(r, 1e-6)
        if "matched_random_frac_above" in qs[0]:
            m = np.mean([q["matched_random_frac_above"][t] for q in qs])
            out[f"matched_enrich_{t}"] = o / max(m, 1e-6)
    ww = [p for q in qs if q["cls"] == "wall" for n, p in q["others"].items() if q.get("others_cls", {}).get(n) == "wall"]
    out["wall_to_wall_pct"] = np.mean(ww) if ww else np.nan
    out["mean_other_pct"] = np.mean([p for q in qs for p in q["others"].values()])
    return out


def ci(report, rng):
    qs = list(report["queries"].values())
    point = stats(qs)
    boots = {k: [] for k in point}
    for _ in range(B):
        samp = [qs[i] for i in rng.integers(0, len(qs), len(qs))]
        for k, v in stats(samp).items():
            boots[k].append(v)
    return {k: (point[k], np.nanpercentile(boots[k], 2.5), np.nanpercentile(boots[k], 97.5)) for k in point}


def per_class(report):
    cells = {}
    for q in report["queries"].values():
        for n, p in q["others"].items():
            oc = q.get("others_cls", {}).get(n)
            if oc is None:
                continue
            cells.setdefault((q["cls"], oc), []).append(p)
    return {k: (np.mean([p >= 99.5 for p in v]), len(v)) for k, v in sorted(cells.items())}


def main():
    rng = np.random.default_rng(0)
    files = [Path(f) for f in sys.argv[1:]]
    rows = {f.stem.replace("eval_", ""): ci(json.load(open(f)), rng) for f in files}
    keys = ["frac_above_99.9", "enrich_99.9", "matched_enrich_99.9", "enrich_99.5", "matched_enrich_99.5", "wall_to_wall_pct"]
    labels = {"frac_above_99.9": "other sites in top 0.1%", "enrich_99.9": "enrichment vs uniform (top 0.1%)",
              "matched_enrich_99.9": "enrichment vs landform-matched (top 0.1%)", "enrich_99.5": "enrichment vs uniform (top 0.5%)",
              "matched_enrich_99.5": "enrichment vs landform-matched (top 0.5%)", "wall_to_wall_pct": "wall-to-wall pct"}
    print(f"Point estimate [95% bootstrap CI over queries, B={B}]\n")
    print("| vectors | " + " | ".join(labels[k] for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name, r in rows.items():
        cells = []
        for k in keys:
            if k not in r:
                cells.append("-"); continue
            p, lo, hi = r[k]
            fmt = "{:.2f}" if k.startswith("frac") else ("{:.1f}" if "pct" in k else "{:.0f}x")
            cells.append(f"{fmt.format(p)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nFraction of other sites at or above the 99.5th percentile, by query class -> other-site class (n pairs):\n")
    classes = ["wall", "mound", "pits"]
    print("| vectors | " + " | ".join(f"{a}->{b}" for a in classes for b in classes) + " |")
    print("|---|" + "---|" * 9)
    for f in files:
        pc = per_class(json.load(open(f)))
        print(f"| {f.stem.replace('eval_', '')} | " + " | ".join(
            (f"{pc[(a, b)][0]:.2f} ({pc[(a, b)][1]})" if (a, b) in pc else "-") for a in classes for b in classes) + " |")


if __name__ == "__main__":
    main()
