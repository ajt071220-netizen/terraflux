r"""eval_vs.py —— 用训练好的 PPO 模型对启发式 AI 跑评估局，量化训练效果

用法：
  .\.venv-py312\Scripts\python.exe python\eval_vs.py --model ppo_terrain.zip --games 100 --size 21
"""
from __future__ import annotations
import argparse
import random

import numpy as np

import engine as E
import heuristic
from gym_env import action_mask
from train_ppo import env_obs_for, action_to_move


def load_model(path):
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path), True
    except ImportError:
        from stable_baselines3 import PPO
        return PPO.load(path), False


def ppo_move(model, use_mask, state, color):
    obs = env_obs_for(state, color)
    if use_mask:
        act, _ = model.predict(obs, action_masks=np.array([action_mask(state, color)]), deterministic=True)
    else:
        act, _ = model.predict(obs, deterministic=True)
    action = int(act.item() if hasattr(act, 'item') else act)
    move = action_to_move(state, color, action)
    legal = E.legal_moves(state, color)
    if move is None:
        return None if state['config']['passAllowed'] else (legal[0] if legal else None)
    return move if move in legal else (legal[0] if legal else None)


def play_one(model, use_mask, ppo_color, size, rng):
    state = E.create_game({'size': size}, rng)
    steps = 0
    while state['status'] == 'playing' and steps < 800:
        color = state['turn']
        if color == ppo_color:
            mv = ppo_move(model, use_mask, state, color)
        else:
            mv = heuristic.choose_move(state, color, rng=rng)
        E.apply_move(state, color, mv)
        steps += 1
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='ppo_terrain.zip')
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--size', type=int, default=21)
    args = ap.parse_args()

    model, use_mask = load_model(args.model)
    rng = random.Random(7)

    for ppo_color in ('white', 'black'):
        wins = {'white': 0, 'black': 0, 'draw': 0}
        for _ in range(args.games):
            st = play_one(model, use_mask, ppo_color, args.size, rng)
            wins[st['winner']] += 1
        n = args.games
        print(f'PPO 执{"白" if ppo_color == "white" else "黑"} {n} 局 vs 启发式：'
              f'白胜 {wins["white"]/n:.0%}  黑胜 {wins["black"]/n:.0%}  平 {wins["draw"]/n:.0%}')

    print('参考基线（启发式 vs 启发式）：白 45% / 黑 54% / 平 1.5%（200 局自检）')


if __name__ == '__main__':
    main()
