"""model_vs_model.py —— 两模型对打评估（循环赛强度排名用）

每对组合交换颜色各 N 局（A 执白 vs B 执黑 + B 执白 vs A 执黑），报 A 的总胜率。
采样模式 + duel 默认布局。

用法：
  python model_vs_model.py --a experiments\ppo25_duel_shaped.zip --b experiments\ppo25_duel_sparse.zip --games 100
"""
from __future__ import annotations
import argparse
import random

import numpy as np

import engine as E
from gym_env import action_mask
from train_ppo import env_obs_for, action_to_move


def load_model(path):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path)


def model_move(model, state, color, rng):
    legal = E.legal_moves(state, color)
    if not legal:
        return None
    obs = env_obs_for(state, color)
    act, _ = model.predict(obs, action_masks=np.array([action_mask(state, color)]), deterministic=False)
    move = action_to_move(state, color, int(act.item() if hasattr(act, 'item') else act))
    if move is None:
        return None if state['config']['passAllowed'] else rng.choice(legal)
    return move if move in legal else rng.choice(legal)


def play_one(ma, mb, a_color, rng):
    state = E.create_game({'size': 25}, rng)
    players = {'white': ma if a_color == 'white' else mb,
               'black': mb if a_color == 'white' else ma}
    steps = 0
    while state['status'] == 'playing' and steps < 800:
        color = state['turn']
        mv = model_move(players[color], state, color, rng)
        E.apply_move(state, color, mv)
        steps += 1
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--games', type=int, default=100, help='每种颜色各打多少局')
    args = ap.parse_args()

    ma, mb = load_model(args.a), load_model(args.b)
    rng = random.Random()

    total = {'a': 0, 'b': 0, 'draw': 0}
    for a_color in ('white', 'black'):
        for _ in range(args.games):
            st = play_one(ma, mb, a_color, rng)
            if st['winner'] == 'draw':
                total['draw'] += 1
            elif st['winner'] == a_color:
                total['a'] += 1
            else:
                total['b'] += 1
    n = args.games * 2
    print(f'A({args.a}) vs B({args.b}) 各 {args.games}×2 局:')
    print(f'  A 胜 {total["a"]/n:.1%}  B 胜 {total["b"]/n:.1%}  平 {total["draw"]/n:.1%}')


if __name__ == '__main__':
    main()
