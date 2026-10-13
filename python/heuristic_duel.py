"""heuristic_duel.py —— 启发式 vs 启发式的 duel 白胜率（报告 §6.1 的基线，v2 重跑）

与训练对手一致的启发式（默认 noise=0.4），duel 布局（gap=3，黑先，25×25），N 局。
用法（仓库根目录）：  .\\.venv-py312\\Scripts\\python.exe python\\heuristic_duel.py --games 300
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E          # noqa: E402
import heuristic            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20260813)
    ap.add_argument('--out', type=str, default='experiments/heuristic_duel.json')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = {'size': 25, 'gap': 3, 'firstPlayer': 'black'}
    res, reasons, lens = Counter(), Counter(), []
    for _ in range(args.games):
        st = E.create_game(cfg, rng)
        steps = 0
        while st['status'] == 'playing' and steps < 800:
            color = st['turn']
            mv = heuristic.choose_move(st, color, rng=rng)
            E.apply_move(st, color, mv)
            steps += 1
        w = st['winner'] or 'unfinished'
        res[w] += 1
        if w == 'draw':
            reasons[st.get('winReason') or '?'] += 1
        lens.append(steps)

    n = args.games
    p = res['white'] / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    out = dict(games=n, white=res['white'], black=res['black'], draw=res['draw'],
               white_rate=p, white_ci=[max(0.0, c - h), min(1.0, c + h)],
               draw_reasons=dict(reasons), mean_len=sum(lens) / n, seed=args.seed)
    print(f"启发式 vs 启发式（duel，黑先，{n} 局）：白胜 {p:.1%} [{out['white_ci'][0]:.1%},{out['white_ci'][1]:.1%}]  "
          f"黑胜 {res['black']/n:.1%}  平 {res['draw']/n:.1%} {dict(reasons)}  平均 {out['mean_len']:.1f} 步")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('written', args.out)


if __name__ == '__main__':
    main()
