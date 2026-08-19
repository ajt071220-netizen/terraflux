"""eval_selfplay.py —— PPO 自对弈：测量"高手 vs 高手"下的黑白胜率（公平性基线）

用法：
  .\.venv-py312\Scripts\python.exe python\eval_selfplay.py --model models\ppo_25.zip --games 100 --size 25

注意：双方用同一模型、采样模式（deterministic=False）+ 随机布子，
否则同一模型的确定性输出会让每局完全相同，失去统计意义。
"""
from __future__ import annotations
import argparse
import random

import numpy as np

import engine as E
from gym_env import action_mask
from train_ppo import env_obs_for, action_to_move


def load_model(path):
    try:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path), True
    except ImportError:
        from stable_baselines3 import PPO
        return PPO.load(path), False


def model_move(model, use_mask, state, color, rng, deterministic):
    legal = E.legal_moves(state, color)
    if not legal:
        return None
    obs = env_obs_for(state, color)
    if use_mask:
        act, _ = model.predict(obs, action_masks=np.array([action_mask(state, color)]), deterministic=deterministic)
    else:
        act, _ = model.predict(obs, deterministic=deterministic)
    move = action_to_move(state, color, int(act.item() if hasattr(act, 'item') else act))
    if move is None:
        return None if state['config']['passAllowed'] else rng.choice(legal)
    return move if move in legal else rng.choice(legal)


def play_one(model, use_mask, config, rng, deterministic, setup=None):
    state = E.create_game(config, rng)
    if setup:  # 固定布子（公平性扫描用）：覆盖随机布子并重置重复局面记录
        state['white'] = {'r': setup[0][0], 'c': setup[0][1]}
        state['black'] = {'r': setup[1][0], 'c': setup[1][1]}
        state['repetition'] = {}
        E.record_repetition(state)
    steps = 0
    while state['status'] == 'playing' and steps < 800:
        mv = model_move(model, use_mask, state, state['turn'], rng, deterministic)
        E.apply_move(state, state['turn'], mv)
        steps += 1
    return state, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--size', type=int, default=25)
    ap.add_argument('--deterministic', action='store_true',
                    help='用确定性输出（每局仍不同，因为布子随机）')
    # 规则旋钮覆盖（用于公平性扫描；模型是在默认规则下训的，换规则后是方向性指示）
    ap.add_argument('--capture-rule', type=str, default=None, choices=['gt', 'gte'])
    ap.add_argument('--gap', type=int, default=None)
    ap.add_argument('--first-player', type=str, default=None, choices=['white', 'black'])
    ap.add_argument('--goal-edges', type=str, default=None, help='如 S 或 NS')
    ap.add_argument('--phase-mode', type=str, default=None, choices=['off', 'sync', 'all8'])
    ap.add_argument('--setup', type=str, default=None,
                    help='固定布子 "白r,白c:黑r,黑c"，如 "14,13:13,13"（覆盖随机布子）')
    args = ap.parse_args()

    model, use_mask = load_model(args.model)
    rng = random.Random()

    config = {'size': args.size}
    overrides = []
    if args.capture_rule: config['captureRule'] = args.capture_rule; overrides.append(f'捕捉={args.capture_rule}')
    if args.gap is not None: config['gap'] = args.gap; overrides.append(f'间距={args.gap}')
    if args.first_player: config['firstPlayer'] = args.first_player; overrides.append(f'先手={args.first_player}')
    if args.goal_edges: config['goalEdges'] = list(args.goal_edges); overrides.append(f'胜利边={args.goal_edges}')
    if args.phase_mode: config['phaseMode'] = args.phase_mode; overrides.append(f'相位={args.phase_mode}')

    setup = None
    if args.setup:
        w, b = args.setup.split(':')
        setup = (tuple(int(x) for x in w.split(',')), tuple(int(x) for x in b.split(',')))
        overrides.append(f'布子=白{setup[0]}黑{setup[1]}')

    wins = {'white': 0, 'black': 0, 'draw': 0, 'unfinished': 0}
    lens = []
    for _ in range(args.games):
        st, steps = play_one(model, use_mask, config, rng, args.deterministic, setup)
        wins[st['winner'] if st['winner'] else 'unfinished'] += 1
        lens.append(steps)

    n = args.games
    print(f'PPO 自对弈 {n} 局（{args.size}×{args.size}，{"确定性" if args.deterministic else "采样"}模式'
          + (f'，覆盖：{"，".join(overrides)}' if overrides else '，默认规则') + '）：')
    print(f'  白胜 {wins["white"]/n:.0%}  黑胜 {wins["black"]/n:.0%}  平 {wins["draw"]/n:.0%}' +
          (f'  未分胜负 {wins["unfinished"]/n:.0%}' if wins['unfinished'] else ''))
    print(f'  平均回合 {sum(lens)/len(lens):.0f} 步')
    se = (wins['white']/n * (1 - wins['white']/n) / n) ** 0.5
    print(f'  白胜率标准误 ±{se:.1%}（95% 置信区间约 ±{1.96*se:.1%}）')


if __name__ == '__main__':
    main()
