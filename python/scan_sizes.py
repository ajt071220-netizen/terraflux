"""scan_sizes.py —— 实验③：棋盘尺寸公平性相变扫描

duel 布局（黑镇中央、白正北隔 3 格对峙）下，不同棋盘尺寸 × 先手方的公平性扫描。
启发式 vs 启发式互弈，每个条件 300 局，输出：
1. 结果表（终端 + experiments/size_scan.json）
2. 白胜率-尺寸曲线 → experiments/fig_size_fairness.png
"""
import json
import random
import sys

sys.path.insert(0, '.')
sys.path.insert(0, 'python')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import engine as E
import heuristic

SIZES = [15, 17, 19, 21, 23, 25, 27]
GAMES = 300
SEED = 42


def play_one(rng, size, first):
    state = E.create_game({'size': size, 'startLayout': 'duel', 'gap': 3, 'firstPlayer': first}, rng)
    steps = 0
    while state['status'] == 'playing' and steps < 1200:
        color = state['turn']
        mv = heuristic.choose_move(state, color, rng=rng)
        E.apply_move(state, color, mv)
        steps += 1
    return state


def scan_size(size):
    out = {}
    for first in ('black', 'white'):
        rng = random.Random(SEED * 1000 + size * 10 + (0 if first == 'black' else 1))
        wins = {'white': 0, 'black': 0, 'draw': 0}
        turns = 0
        for _ in range(GAMES):
            st = play_one(rng, size, first)
            wins[st['winner']] += 1
            turns += st['moveCount']['white'] + st['moveCount']['black']
        out[first] = {
            'white_wr': wins['white'] / GAMES,
            'black_wr': wins['black'] / GAMES,
            'draw': wins['draw'] / GAMES,
            'avg_len': turns / GAMES,
        }
    return out


def main():
    results = {}
    for size in SIZES:
        r = scan_size(size)
        results[size] = r
        b, w = r['black'], r['white']
        print(f'{size}×{size} | 黑先: 白胜率 {b["white_wr"]:.1%}（平 {b["draw"]:.1%}，{b["avg_len"]:.0f} 步）'
              f' | 白先: 白胜率 {w["white_wr"]:.1%}（平 {w["draw"]:.1%}，{w["avg_len"]:.0f} 步）')

    json.dump(results, open('experiments/size_scan.json', 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for first, marker, color in [('black', 'o', '#e07020'), ('white', 's', '#5a6a80')]:
        ys = [results[s][first]['white_wr'] * 100 for s in SIZES]
        ax.plot(SIZES, ys, marker=marker, color=color, lw=1.8,
                label='Black first' if first == 'black' else 'White first')
    ax.axhline(50, color='#888', ls='--', lw=1, alpha=0.6)
    ax.set_xlabel('Board size')
    ax.set_ylabel('White win rate (%)')
    ax.set_title(f'Fairness vs board size (duel gap=3, heuristic self-play, {GAMES} games/point)')
    ax.set_xticks(SIZES)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig('experiments/fig_size_fairness.png', dpi=140)
    print('[图] experiments/fig_size_fairness.png')


if __name__ == '__main__':
    main()
