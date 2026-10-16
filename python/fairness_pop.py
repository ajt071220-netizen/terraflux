"""fairness_pop.py —— 公平性的群体评估（事前登记准则）

布局：duel gap=3，25×25，黑先（与主实验一致）。游戏规则不变。
群体（20 个训练模型 + 启发式）：
  shaped_s2..s11（10 个）、sparse_s1..s10（10 个）、启发式 heuristic。
配对（所有无序对）：
  - 模型 vs 模型：C(20,2) = 190 对
  - 模型 vs 启发式：20 对
  - 自对弈（同一模型双方）：20 对，仅作为旧测量方法的对照
每对每种颜色各 GAMES_PER_COLOR 局（A 执白、B 执白各一半），统计全局白胜率。

事前登记的判定准则（运行前写死）：
  主指标：所有非自对弈配对的全局白胜率均值（每对等权）。
  不确定性：按配对做 cluster bootstrap（10000 次，seed 20260815）得到 95% CI。
  判定：CI 包含 0.5 → 该布局在此群体上判定为"公平"；否则按均值方向判定偏向白方或黑方。
  同时报告：自对弈白胜率（旧方法）与群体白胜率的对比。
用法（仓库根目录）：
  ..\.venv-py312\Scripts\python.exe python\fairness_pop.py [每色局数，默认 10]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import itertools
import json
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BOOT_SEED = 20260815
MAX_STEPS = 800
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

AGENTS = {}
for s in range(2, 12):
    AGENTS[f'shaped_s{s}'] = os.path.join(ROOT, 'models', f'ppo25_shaped_s{s}.zip')
for s in range(1, 11):
    AGENTS[f'sparse_s{s}'] = os.path.join(ROOT, 'models', f'ppo25_sparse_s{s}.zip')
AGENTS['heuristic'] = None

_cache = {}


def get_agent(name):
    if name not in _cache:
        from model_vs_model import load_model
        _cache[name] = load_model(AGENTS[name])
    return _cache[name]


def choose(name, state, color, rng):
    import heuristic
    if name == 'heuristic':
        return heuristic.choose_move(state, color, rng=rng)
    from model_vs_model import model_move
    return model_move(get_agent(name), state, color, rng)


def play_game(a, b, a_color, rng, gap):
    import engine as E
    state = E.create_game({'size': 25, 'startLayout': 'duel', 'gap': gap, 'firstPlayer': 'black'}, rng)
    players = {'white': a if a_color == 'white' else b,
               'black': b if a_color == 'white' else a}
    steps = 0
    while state['status'] == 'playing' and steps < MAX_STEPS:
        color = state['turn']
        mv = choose(players[color], state, color, rng)
        E.apply_move(state, color, mv)
        steps += 1
    return state['winner'] if state['winner'] in ('white', 'black') else 'draw'


def kind(a, b):
    if a == b:
        return 'self'
    if 'heuristic' in (a, b):
        return 'vs_heuristic'
    return 'model_vs_model'


def run_pair(task):
    a, b, seed, gpc, gap = task
    rng = random.Random(seed)
    white_wins = black_wins = draws = 0
    for a_color in ('white', 'black'):
        for _ in range(gpc):
            w = play_game(a, b, a_color, rng, gap)
            if w == 'white':
                white_wins += 1
            elif w == 'black':
                black_wins += 1
            else:
                draws += 1
    n = white_wins + black_wins + draws
    return {'a': a, 'b': b, 'kind': kind(a, b), 'games': n, 'gap': gap,
            'white_wins': white_wins, 'black_wins': black_wins, 'draws': draws,
            'white_wr': white_wins / n}


def cluster_boot(values, n_boot=10000, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarize(rows):
    vals = [r['white_wr'] for r in rows]
    mean = sum(vals) / len(vals)
    lo, hi = cluster_boot(vals)
    tw = sum(r['white_wins'] for r in rows)
    tb = sum(r['black_wins'] for r in rows)
    td = sum(r['draws'] for r in rows)
    return {'pairs': len(rows), 'games': sum(r['games'] for r in rows),
            'mean_pair_white_wr': mean, 'ci95': [lo, hi],
            'pooled_white_wr': tw / (tw + tb + td),
            'white_wins': tw, 'black_wins': tb, 'draws': td}


def main(gpc, gap):
    names = list(AGENTS.keys())
    tasks = []
    seed = 20260815
    for a, b in itertools.combinations(names, 2):
        tasks.append((a, b, seed, gpc, gap)); seed += 1
    for name in names:
        if name != 'heuristic':
            tasks.append((name, name, seed, gpc, gap)); seed += 1
    print(f'pairings: {len(tasks)}  games/pairing: {2 * gpc}', flush=True)
    rows = []
    with Pool(8) as pool:
        for i, r in enumerate(pool.imap_unordered(run_pair, tasks), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f'  done {i}/{len(tasks)}', flush=True)

    mm = [r for r in rows if r['kind'] == 'model_vs_model']
    vh = [r for r in rows if r['kind'] == 'vs_heuristic']
    sp = [r for r in rows if r['kind'] == 'self']
    result = {
        'criterion': 'mean over non-self pairings of white win rate; cluster bootstrap 95% CI (10000, seed 20260815); fair if CI contains 0.5',
        'games_per_color': gpc,
        'gap': gap,
        'population_all_non_self': summarize(mm + vh),
        'model_vs_model': summarize(mm),
        'vs_heuristic': summarize(vh),
        'self_play_old_method': summarize(sp),
        'pairs': sorted(rows, key=lambda r: (r['kind'], r['a'], r['b'])),
    }
    p = result['population_all_non_self']
    lo, hi = p['ci95']
    if lo <= 0.5 <= hi:
        verdict = 'fair (CI contains 0.5)'
    else:
        verdict = 'white-favoured' if p['mean_pair_white_wr'] > 0.5 else 'black-favoured'
    print(f"\n群体（非自对弈）平均白胜率 = {p['mean_pair_white_wr']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  -> {verdict}")
    print(f"  模型 vs 模型: {result['model_vs_model']['mean_pair_white_wr']:.3f} {result['model_vs_model']['ci95']}")
    print(f"  模型 vs 启发式: {result['vs_heuristic']['mean_pair_white_wr']:.3f} {result['vs_heuristic']['ci95']}")
    s = result['self_play_old_method']
    print(f"  自对弈（旧方法）: {s['mean_pair_white_wr']:.3f} {s['ci95']}")
    out = os.path.join(ROOT, 'experiments', f'fairness_pop_gap{gap}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f'已写入 {out}')


if __name__ == '__main__':
    gpc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    gaps = [int(g) for g in sys.argv[2].split(',')] if len(sys.argv) > 2 else [3]
    for g in gaps:
        main(gpc, g)