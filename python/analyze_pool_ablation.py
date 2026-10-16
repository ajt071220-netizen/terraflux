"""analyze_pool_ablation.py -- opponent-pool ablation (report section 4.3).

Arm A: opponent sampled from the last 5 snapshots (--pool-window 5)
Arm B: opponent sampled from all snapshots (--pool-window 0)
Both: 25x25 duel gap=3 black-first, 500k steps, step penalty 0.002,
heuristic ratio 0.65, snapshot every 50k steps, seeds 1..10.

Late window = logged rows with steps >= 300k (last 40% of training).
Paired by seed; two-sided exact sign-flip test over 10 seeds.
Run from python/ with: python analyze_pool_ablation.py
"""
import itertools
import json
import statistics

ARMS = {'A': 'abl2_A', 'B': 'abl2_B'}
LATE_STEPS = 300_000
SEEDS = range(1, 11)


def rows(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def late_win_rates(arm, seed):
    rs = rows(f'training_log_{arm}_s{seed}.jsonl')
    return [r['win_rate'] for r in rs if r['win_rate'] is not None and r['steps'] >= LATE_STEPS]


def per_seed(arm):
    means, sds = [], []
    for s in SEEDS:
        w = late_win_rates(arm, s)
        means.append(statistics.mean(w))
        sds.append(statistics.stdev(w))
    return means, sds


def signflip_p(d):
    obs = abs(statistics.mean(d))
    n = len(d)
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        if abs(sum(s * x for s, x in zip(signs, d)) / n) >= obs - 1e-12:
            hits += 1
    return hits / total


def main():
    ma, sa = per_seed(ARMS['A'])
    mb, sb = per_seed(ARMS['B'])
    dm = [b - a for a, b in zip(ma, mb)]
    ds = [b - a for a, b in zip(sa, sb)]
    print(f"late win rate  A={statistics.mean(ma):.3f}  B={statistics.mean(mb):.3f}  "
          f"diff(B-A)={statistics.mean(dm):+.4f}  p={signflip_p(dm):.3f}")
    print(f"late win SD    A={statistics.mean(sa):.3f}  B={statistics.mean(sb):.3f}  "
          f"diff(B-A)={statistics.mean(ds):+.4f}  p={signflip_p(ds):.3f}")


if __name__ == '__main__':
    main()
