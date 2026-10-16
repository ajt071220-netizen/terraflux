"""scan_gap.py —— 布局选择的可追溯扫描（事前登记准则）

duel 布局（黑镇中央、白正北隔 gap 格对峙），25×25，启发式 vs 启发式互弈。
对每个 gap 取值、每种先手各 GAMES 局，把全部结果落盘到 experiments/gap_scan.json，
不做事后挑选。

事前登记的选择准则（运行前写死，不根据结果修改）：
  主准则：在"黑先"条件下，使白胜率与 50% 的差距 |白胜率 - 0.5| 最小的 gap。
  并列时取较小的 gap。
  同时报告 Wilson 95% 置信区间和白先条件下的结果，作为敏感性信息。
游戏规则不变；只改变布局参数 gap。
"""
import json
import math
import random
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'python')

import engine as E
import heuristic

SIZE = 25
GAPS = [1, 2, 3, 4, 5, 6, 7, 8]
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 20260815
MAX_STEPS = 1200


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def play_one(rng, gap, first):
    state = E.create_game({'size': SIZE, 'startLayout': 'duel', 'gap': gap,
                           'firstPlayer': first}, rng)
    steps = 0
    while state['status'] == 'playing' and steps < MAX_STEPS:
        color = state['turn']
        mv = heuristic.choose_move(state, color, rng=rng)
        E.apply_move(state, color, mv)
        steps += 1
    return state


def scan_gap(gap):
    out = {}
    for first in ('black', 'white'):
        rng = random.Random(SEED * 1000 + gap * 10 + (0 if first == 'black' else 1))
        wins = {'white': 0, 'black': 0, 'draw': 0}
        turns = 0
        for _ in range(GAMES):
            st = play_one(rng, gap, first)
            wins[st['winner'] if st['winner'] in wins else 'draw'] += 1
            turns += st['moveCount']['white'] + st['moveCount']['black']
        lo, hi = wilson(wins['white'], GAMES)
        out[first] = {
            'games': GAMES,
            'white_wins': wins['white'], 'black_wins': wins['black'], 'draws': wins['draw'],
            'white_wr': wins['white'] / GAMES,
            'white_wr_ci95': [lo, hi],
            'avg_len': turns / GAMES,
        }
    return out


def main():
    results = {}
    for gap in GAPS:
        r = scan_gap(gap)
        results[gap] = r
        b, w = r['black'], r['white']
        print(f"gap={gap} | 黑先: 白胜率 {b['white_wr']:.1%} [{b['white_wr_ci95'][0]:.1%},{b['white_wr_ci95'][1]:.1%}]"
              f" 平 {b['draws']}/{GAMES} | 白先: 白胜率 {w['white_wr']:.1%} [{w['white_wr_ci95'][0]:.1%},{w['white_wr_ci95'][1]:.1%}]", flush=True)

    best = min(GAPS, key=lambda g: (abs(results[g]['black']['white_wr'] - 0.5), g))
    print(f'\n事前准则选出的 gap（黑先，|白胜率-0.5| 最小）: {best}')
    print(f'gap=3 是否为最优: {best == 3}')
    json.dump({'criterion': 'black-first, minimise |white_wr-0.5|, ties -> smaller gap',
               'selected_gap': best, 'results': results},
              open('experiments/gap_scan.json', 'w'), indent=1, ensure_ascii=False)
    print('已写入 experiments/gap_scan.json')


if __name__ == '__main__':
    main()