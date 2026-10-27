"""parity_gen.py —— 生成 JS/Python 引擎一致性测试用的随机对局轨迹

用 python/engine.py 跑大量随机对局（覆盖所有规则开关），逐步记录完整状态，
写到 experiments/parity_traces.json；再由 tests/parity_check.mjs 用 src/engine.js
重放同一串动作并逐步比对。

用法：py python/parity_gen.py [局数=3000] [种子=20261027]
"""
from __future__ import annotations
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import engine as E

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261027
OUT = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'parity_traces.json')

# JS 版没有步数上限，这里设得足够大，保证对局只因规则本身结束
MAX_PLIES = 600


def random_config(rng):
    size = rng.choice([7, 9, 11, 21, 25])
    m = E.mid(size)
    return dict(
        size=size,
        goalEdges=rng.choice([['S'], ['N'], ['E', 'W'], ['N', 'S', 'E', 'W'], ['S', 'E']]),
        captureRule=rng.choice(['gt', 'gte']),
        firstPlayer=rng.choice(['black', 'white']),
        phaseMode=rng.choice(['off', 'sync', 'all8']),
        startLayout=rng.choice(['duel', 'blockade', 'classic']),
        gap=rng.randint(1, m - 1),
        passAllowed=rng.random() < 0.3,
        stalemateLoses=rng.random() < 0.7,
        repetitionDraw=rng.random() < 0.85,
        maxMoves=10 ** 9,
    )


def dump_state(s):
    return dict(
        pillars=''.join(str(x) for x in s['pillars']),
        white=[s['white']['r'], s['white']['c']],
        black=[s['black']['r'], s['black']['c']],
        turn=s['turn'],
        moveCount=[s['moveCount']['white'], s['moveCount']['black']],
        passes=s['passes'],
        status=s['status'],
        winner=s['winner'],
        winReason=s['winReason'],
    )


def dump_events(ev):
    if ev is None:
        return None
    return dict(
        moved=list(ev['moved']) if ev['moved'] else None,
        passed=ev['passed'],
        phase=ev['phase'],
        changes=[list(x) for x in ev['changes']],
        result=list(ev['result']) if ev['result'] else None,
    )


def play(rng, cfg):
    s = E.create_game(cfg, rng=rng)
    start = dict(white=[s['white']['r'], s['white']['c']], black=[s['black']['r'], s['black']['c']])
    steps = []
    for _ in range(MAX_PLIES):
        if s['status'] != 'playing':
            break
        color = s['turn']
        legal = E.legal_moves(s, color)
        # 偶尔故意尝试非法动作，检查两边是否同样拒绝
        roll = rng.random()
        if roll < 0.05:
            r, c = s[color]['r'] + rng.randint(-2, 2), s[color]['c'] + rng.randint(-2, 2)
            move = (r, c)
        elif roll < 0.10:
            move = None
        elif legal:
            # 偏向少数几个方向，制造来回走动以触发三次重复
            move = legal[0] if rng.random() < 0.15 else rng.choice(legal)
        else:
            move = None
        wrong_side = rng.random() < 0.02
        actor = ('white' if color == 'black' else 'black') if wrong_side else color
        ev = E.apply_move(s, actor, move)
        steps.append(dict(
            actor=actor,
            move=list(move) if move else None,
            legal=[list(m) for m in legal],
            events=dump_events(ev),
            state=dump_state(s),
        ))
    return dict(config=cfg, start=start, steps=steps)


def main():
    rng = random.Random(SEED)
    games = []
    for _ in range(N_GAMES):
        cfg = random_config(rng)
        g = play(rng, cfg)
        games.append(g)
    reasons = {}
    for g in games:
        st = g['steps'][-1]['state'] if g['steps'] else None
        key = st['winReason'] if st and st['status'] == 'over' else 'unfinished'
        reasons[key] = reasons.get(key, 0) + 1
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(dict(seed=SEED, games=games), f, separators=(',', ':'))
    n_steps = sum(len(g['steps']) for g in games)
    print(f'games={len(games)} steps={n_steps} seed={SEED}')
    print('end reasons:', dict(sorted(reasons.items())))
    print('wrote', os.path.normpath(OUT))


if __name__ == '__main__':
    main()
