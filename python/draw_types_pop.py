"""draw_types_pop.py —— 群体互弈中各类终局（含同瞬平局 border-capture）的频率

与 fairness_pop.py 相同的群体、配对、种子与对局设置（duel，25×25，黑先），
只额外记录每局的终局原因（engine 的 winReason）。游戏规则不变。
终局原因：escape（白逃脱）、capture（黑捕获）、border-capture（同瞬平局）、
repetition、max-moves、stalemate、mutual-pass；超过 MAX_STEPS 未结束记为 unfinished。
模型动作是采样的（deterministic=False，用 torch 的随机数，不受 rng 种子控制），
所以对局无法逐局重放；与 experiments/fairness_pop_gap{gap}.json 的逐对比较只用于提示差异，
两次运行应视为同一设计下的两个独立样本。
用法（仓库根目录）：
  .venv-py312\\Scripts\\python.exe python\\draw_types_pop.py [每色局数，默认 10] [gap，默认 3]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import collections
import itertools
import json
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fairness_pop as FP


def play_game(a, b, a_color, rng, gap):
    import engine as E
    state = E.create_game({'size': 25, 'startLayout': 'duel', 'gap': gap, 'firstPlayer': 'black'}, rng)
    players = {'white': a if a_color == 'white' else b,
               'black': b if a_color == 'white' else a}
    steps = 0
    while state['status'] == 'playing' and steps < FP.MAX_STEPS:
        color = state['turn']
        mv = FP.choose(players[color], state, color, rng)
        E.apply_move(state, color, mv)
        steps += 1
    if state['status'] == 'playing':
        return 'draw', 'unfinished'
    return state['winner'], state['winReason']


def run_pair(task):
    a, b, seed, gpc, gap = task
    rng = random.Random(seed)
    reasons = collections.Counter()
    white_wins = black_wins = 0
    for a_color in ('white', 'black'):
        for _ in range(gpc):
            winner, reason = play_game(a, b, a_color, rng, gap)
            reasons[reason] += 1
            if winner == 'white':
                white_wins += 1
            elif winner == 'black':
                black_wins += 1
    return {'a': a, 'b': b, 'kind': FP.kind(a, b), 'games': 2 * gpc,
            'white_wins': white_wins, 'black_wins': black_wins, 'reasons': dict(reasons)}


def main(gpc, gap):
    names = list(FP.AGENTS.keys())
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

    ref_path = os.path.join(FP.ROOT, 'experiments', f'fairness_pop_gap{gap}.json')
    mismatches = None
    if os.path.exists(ref_path):
        with open(ref_path, encoding='utf-8') as f:
            ref = {(p['a'], p['b']): p for p in json.load(f)['pairs']}
        mismatches = [(r['a'], r['b']) for r in rows
                      if (r['a'], r['b']) in ref and
                      (ref[(r['a'], r['b'])]['white_wins'], ref[(r['a'], r['b'])]['black_wins'])
                      != (r['white_wins'], r['black_wins'])]

    by_kind = {}
    for k in ('model_vs_model', 'vs_heuristic', 'self'):
        sel = [r for r in rows if r['kind'] == k]
        c = collections.Counter()
        for r in sel:
            c.update(r['reasons'])
        by_kind[k] = {'pairs': len(sel), 'games': sum(r['games'] for r in sel), 'reasons': dict(c)}
    result = {'gap': gap, 'games_per_color': gpc, 'by_kind': by_kind,
              'matches_fairness_pop': None if mismatches is None else len(mismatches) == 0,
              'mismatched_pairs': mismatches,
              'pairs': sorted(rows, key=lambda r: (r['kind'], r['a'], r['b']))}
    for k, v in by_kind.items():
        print(k, v)
    print('matches fairness_pop:', result['matches_fairness_pop'],
          '' if not mismatches else f'({len(mismatches)} mismatched pairs)')
    out = os.path.join(FP.ROOT, 'experiments', f'draw_types_pop_gap{gap}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f'已写入 {out}')


if __name__ == '__main__':
    gpc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    gap = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(gpc, gap)
