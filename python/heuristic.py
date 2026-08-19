"""heuristic.py —— 启发式 AI（Python 版，与 src/ai.js 同思路）

白方：冲向胜利边 + 拉开距离；黑方：逼近 + 争夺高位。
1 层前瞻 + 少量噪声，用作 PPO 训练的保底对手。
"""
from __future__ import annotations
import math
import random

import engine as E


def _goal_dist(state, pos):
    s = state['size']
    g = state['config']['goalEdges']
    best = s
    if 'N' in g:
        best = min(best, pos['r'] - 1)
    if 'S' in g:
        best = min(best, s - pos['r'])
    if 'W' in g:
        best = min(best, pos['c'] - 1)
    if 'E' in g:
        best = min(best, s - pos['c'])
    return best


def evaluate(state, me):
    """从 me 视角打分，越大越好。"""
    if state['status'] == 'over':
        if state['winner'] == me:
            return 1000.0
        if state['winner'] == 'draw':
            return 0.0
        return -1000.0

    opp = 'white' if me == 'black' else 'black'
    m, o = state[me], state[opp]
    dist = max(abs(m['r'] - o['r']), abs(m['c'] - o['c']))
    my_h = E.pillar_height(state, m['r'], m['c'])
    opp_h = E.pillar_height(state, o['r'], o['c'])
    score = 0.0

    if me == 'white':
        gd = _goal_dist(state, m)
        score += (state['size'] - gd) * 4.0          # 离胜利边越近越好
        score += min(dist, 6) * 2.0                  # 与黑保持距离
        score += my_h * 6.0 - opp_h * 4.0            # 我要站高、压低黑
        if gd == 0:
            score += 500.0
        if dist <= 2 and my_h <= opp_h:
            score -= 80.0                            # 相邻且没有高度优势，危险
    else:
        score += max(0, 8 - dist) * 6.0              # 逼近白
        score += my_h * 6.0 - opp_h * 4.0
        score -= _goal_dist(state, o) * 2.0          # 别让白接近边
        if dist == 1 and my_h > opp_h:
            score += 500.0                           # 下一步可擒
        if dist == 1:
            score += 60.0
    return score


def choose_move(state, color, noise=0.4, rng=None):
    rng = rng or random
    moves = E.legal_moves(state, color)
    if not moves:
        return None
    best, best_score = None, -math.inf
    for mv in moves:
        s2 = E.clone_state(state)
        E.apply_move(s2, color, mv)
        sc = evaluate(s2, color) + rng.uniform(0, noise * 10)
        if sc > best_score:
            best, best_score = mv, sc
    return best
