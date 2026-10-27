"""Round-robin re-scoring with draws counted as half a point, and a
sparse-vs-shaped label-permutation test on the common opponent set.

Input:  experiments/replicate_eval.json (round_robin, 253 pairs, 200 games each)
Output: experiments/roundrobin_points.json
"""
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "experiments" / "replicate_eval.json").read_text(encoding="utf-8"))
rr = data["round_robin"]

runs = sorted({p["a"] for p in rr} | {p["b"] for p in rr})
idx = {r: i for i, r in enumerate(runs)}
n = len(runs)

# pts[i][j]: points of i against j (win 1, draw 0.5) over games; wins[i][j]: wins only
pts = [[None] * n for _ in range(n)]
wins = [[None] * n for _ in range(n)]
games = [[0] * n for _ in range(n)]
for p in rr:
    a, b, g = idx[p["a"]], idx[p["b"]], p["games"]
    assert p["a_wins"] + p["b_wins"] + p["draws"] == g
    pts[a][b] = (p["a_wins"] + 0.5 * p["draws"]) / g
    pts[b][a] = (p["b_wins"] + 0.5 * p["draws"]) / g
    wins[a][b] = p["a_wins"] / g
    wins[b][a] = p["b_wins"] / g
    games[a][b] = games[b][a] = g


def mean_vs_others(m, i):
    vals = [m[i][j] for j in range(n) if j != i]
    return sum(vals) / len(vals)


score = {r: mean_vs_others(pts, idx[r]) for r in runs}
winrate = {r: mean_vs_others(wins, idx[r]) for r in runs}
rank_pts = sorted(runs, key=lambda r: -score[r])
rank_win = sorted(runs, key=lambda r: -winrate[r])

is_sparse = [r.startswith("sparse_") for r in runs]


def stats(labels):
    sp = [i for i in range(n) if labels[i]]
    sh = [i for i in range(n) if not labels[i]]
    mean_sp = sum(score[runs[i]] for i in sp) / len(sp)
    mean_sh = sum(score[runs[i]] for i in sh) / len(sh)
    cross = [pts[i][j] for i in sp for j in sh]
    return mean_sp - mean_sh, sum(cross) / len(cross)


obs_diff, obs_cross = stats(is_sparse)

rng = random.Random(0)
N_PERM = 10_000
ge_diff = ge_cross = 0
labels = list(is_sparse)
for _ in range(N_PERM):
    rng.shuffle(labels)
    d, c = stats(labels)
    if abs(d) >= abs(obs_diff) - 1e-12:
        ge_diff += 1
    if abs(c - 0.5) >= abs(obs_cross - 0.5) - 1e-12:
        ge_cross += 1
p_diff = (ge_diff + 1) / (N_PERM + 1)
p_cross = (ge_cross + 1) / (N_PERM + 1)

# Spearman between new score and old win-rate rank
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    r = [0.0] * len(xs)
    for pos, k in enumerate(order):
        r[k] = pos + 1
    return r

def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den

rho_old_new = spearman([score[r] for r in runs], [winrate[r] for r in runs])

sp_scores = [score[r] for r in runs if r.startswith("sparse_")]
sh_scores = [score[r] for r in runs if not r.startswith("sparse_")]

out = {
    "scoring": "win = 1, draw = 0.5, loss = 0; mean over 22 opponents, 200 games per pair",
    "ranking_points": [
        {"rank": k + 1, "run": r, "score": round(score[r], 4),
         "win_rate_excl_draws": round(winrate[r], 4),
         "old_rank": rank_win.index(r) + 1}
        for k, r in enumerate(rank_pts)
    ],
    "spearman_points_vs_winrate": round(rho_old_new, 4),
    "sparse_vs_shaped": {
        "n_sparse": len(sp_scores), "n_shaped": len(sh_scores),
        "mean_score_sparse": round(sum(sp_scores) / len(sp_scores), 4),
        "mean_score_shaped": round(sum(sh_scores) / len(sh_scores), 4),
        "diff": round(obs_diff, 4),
        "p_perm_diff": round(p_diff, 4),
        "cross_group_sparse_points": round(obs_cross, 4),
        "cross_group_pairs": len(sp_scores) * len(sh_scores),
        "p_perm_cross": round(p_cross, 4),
        "n_perm": N_PERM,
    },
}
(ROOT / "experiments" / "roundrobin_points.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"{'rank':>4} {'run':<12} {'score':>6} {'winrate':>8} {'old':>4}")
for e in out["ranking_points"]:
    print(f"{e['rank']:>4} {e['run']:<12} {e['score']:>6.3f} {e['win_rate_excl_draws']:>8.3f} {e['old_rank']:>4}")
print("spearman(points, winrate) =", out["spearman_points_vs_winrate"])
print(json.dumps(out["sparse_vs_shaped"], indent=2))
