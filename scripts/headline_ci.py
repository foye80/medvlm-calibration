#!/usr/bin/env python
"""Cell-level paired bootstrap 95% CIs for the three headline deltas in Table 1.
Pairing unit = (model, dataset) cell / (model, train-dataset) adapter, matching
the paper's 'point estimates; 95% bootstrap CI over 1000 resamples' claim.

  d_acc   = mean(FT in-dist acc) - mean(zero-shot acc)          paired by (model,dataset)
  d_ece   = mean(FT in-dist ECE) - mean(zero-shot ECE)          paired by (model,dataset)
  d_ece_x = mean(cross-dataset ECE) - mean(FT in-dist ECE)      paired per adapter (model,train)
"""
import csv, numpy as np

rows = list(csv.DictReader(open("results/master_metrics.csv")))
def f(x, k): return float(x[k])
models = sorted({r["model"] for r in rows})
dsets = sorted({r["eval_dataset"] for r in rows})
get = {(r["model"], r["condition"], r["eval_dataset"]): r for r in rows}

# paired by (model, dataset): zero-shot vs FT in-distribution
zs_acc, ft_acc, zs_ece, ft_ece = [], [], [], []
for m in models:
    for d in dsets:
        zs, ft = get[(m, "zero_shot", d)], get[(m, d, d)]
        zs_acc.append(f(zs, "accuracy")); ft_acc.append(f(ft, "accuracy"))
        zs_ece.append(f(zs, "ece"));      ft_ece.append(f(ft, "ece"))
zs_acc, ft_acc, zs_ece, ft_ece = map(np.array, (zs_acc, ft_acc, zs_ece, ft_ece))

# paired per adapter (model, train-dataset): in-dist vs cross ECE
indist_ece, cross_ece = [], []
for m in models:
    for tr in dsets:
        indist_ece.append(f(get[(m, tr, tr)], "ece"))
        cross_ece.append(np.mean([f(get[(m, tr, ev)], "ece") for ev in dsets if ev != tr]))
indist_ece, cross_ece = np.array(indist_ece), np.array(cross_ece)

def boot(diff, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(diff)
    stats = [diff[rng.integers(0, N, N)].mean() for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])

for name, a, b in [("d_acc (FT-id - zero-shot)", ft_acc, zs_acc),
                   ("d_ece (FT-id - zero-shot)", ft_ece, zs_ece),
                   ("d_ece_x (cross - FT-id)",   cross_ece, indist_ece)]:
    diff = a - b
    lo, hi = boot(diff)
    print(f"{name:30s} mean={diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={len(diff)})")
