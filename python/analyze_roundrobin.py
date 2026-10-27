"""analyze_roundrobin.py —— 循环赛汇总（报告 §5.3 / §6.3）

读取 experiments/replicate_eval.json 中的循环赛与自对弈结果，输出：
  - 每个动态模型的平均循环赛胜率（对所有其它动态模型）与 Copeland 式胜场
  - 自对弈白胜率与平均循环赛胜率的 Spearman 相关
  - 三元组成环（非传递）的数量
  - 同奖励对 vs 跨奖励对的 |胜率−50| 平均值（同奖励 = shaped 配置内两两；跨奖励 = shaped 配置 vs sparse）
输出 JSON：experiments/roundrobin_summary.json
"""
import itertools
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ev = json.load(open('experiments/replicate_eval.json', encoding='utf-8'))
sp = {r['model']: r for r in ev['selfplay']}
rr = ev['round_robin']
names = sorted({x['a'] for x in rr} | {x['b'] for x in rr})
# W：得分率（胜 1、平 0.5、负 0）；WIN：只计胜局的胜率（v2 初版口径，保留对照）
W, WIN = {}, {}
for x in rr:
    W[(x['a'], x['b'])] = x['a_win'] + 0.5 * x['draw']
    W[(x['b'], x['a'])] = x['b_win'] + 0.5 * x['draw']
    WIN[(x['a'], x['b'])] = x['a_win']
    WIN[(x['b'], x['a'])] = x['b_win']

avg = {n: sum(W[(n, m)] for m in names if m != n) / (len(names) - 1) for n in names}
avg_win = {n: sum(WIN[(n, m)] for m in names if m != n) / (len(names) - 1) for n in names}
cope = {n: sum(1 for m in names if m != n and W[(n, m)] > W[(m, n)]) for n in names}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for k, i in enumerate(order):
        r[i] = k + 1
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


xs = [avg[n] for n in names]
ys = [sp[n]['white_rate'] for n in names]
rho = pearson(ranks(xs), ranks(ys))

beats = lambda x, y: W[(x, y)] > W[(y, x)]
triples = list(itertools.combinations(names, 3))
cyc = sum(1 for a, b, c in triples
          if (beats(a, b) and beats(b, c) and beats(c, a)) or (beats(b, a) and beats(c, b) and beats(a, c)))

sh = [n for n in names if sp[n]['static'] is False and (n.startswith('shaped') or n in ('main', 'abl_shaped', 'abl_sparse'))]
spr = [n for n in names if n.startswith('sparse')]
same = list(itertools.combinations(sh, 2))
cross = [(a, b) for a in sh for b in spr]
m_same = sum(abs(W[(a, b)] * 100 - 50) for a, b in same) / len(same) if same else float('nan')
m_cross = sum(abs(W[(a, b)] * 100 - 50) for a, b in cross) / len(cross) if cross else float('nan')

print(f'动态模型 {len(names)} 个，循环赛 {len(rr)} 对')
rho_win = pearson(ranks([avg_win[n] for n in names]), ranks(ys))
rank_win = sorted(names, key=lambda n: -avg_win[n])

print(f"{'model':12s} {'score':>7s} {'winonly':>8s} {'oldrk':>6s} {'copeland':>9s} {'duelWhite':>10s}")
for n in sorted(names, key=lambda n: -avg[n]):
    print(f"{n:12s} {avg[n]*100:6.1f}% {avg_win[n]*100:7.1f}% {rank_win.index(n)+1:6d} "
          f"{cope[n]:4d}/{len(names)-1:<4d} {sp[n]['white_rate']*100:9.1f}%")
print(f'Spearman(avg RR score, duel white) = {rho:.2f}；只计胜局口径 = {rho_win:.2f}  (n={len(names)})')
print(f'成环三元组 {cyc} / {len(triples)}')
print(f'同奖励对 n={len(same)}：mean|win-50| = {m_same:.1f}；跨奖励对 n={len(cross)}：mean|win-50| = {m_cross:.1f}')

out = dict(scoring='win 1, draw 0.5, loss 0', n_models=len(names), n_pairs=len(rr),
           spearman_rho=rho, spearman_rho_win_only=rho_win, cyclic_triples=cyc, total_triples=len(triples),
           avg_rr_win_only={n: avg_win[n] for n in names},
           identical_reward_pairs=len(same), identical_reward_mean_abs_dev=m_same,
           cross_reward_pairs=len(cross), cross_reward_mean_abs_dev=m_cross,
           avg_rr={n: avg[n] for n in names}, copeland={n: cope[n] for n in names},
           duel_white={n: sp[n]['white_rate'] for n in names})
os.makedirs('experiments', exist_ok=True)
with open('experiments/roundrobin_summary.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print('written experiments/roundrobin_summary.json')
