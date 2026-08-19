"""train_ppo.py —— PPO 自我对弈训练

用法：
  pip install -r requirements.txt
  python train_ppo.py --steps 2000000 --size 21

产物：
  ppo_terrain.zip      —— 最终模型（serve_ppo.py 加载它）
  checkpoints/         —— 周期快照（可用作 self-play 对手）

self-play 方案：对手 = 以概率 p_snapshot 用历史快照、否则用启发式 AI，
每 --snapshot-every 步把当前模型存入快照池。这样模型始终面对
"和自己水平相当 + 风格多样"的对手，避免过拟合单一对手。
"""
from __future__ import annotations
import argparse
import os
import random

import numpy as np

import engine as E
import heuristic
from gym_env import TerrainChessEnv, action_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2_000_000)
    ap.add_argument('--size', type=int, default=21)
    ap.add_argument('--snapshot-every', type=int, default=100_000)
    ap.add_argument('--heuristic-ratio', type=float, default=0.5,
                    help='对手用启发式的比例（其余用历史快照）')
    ap.add_argument('--out', type=str, default='ppo_terrain.zip')
    ap.add_argument('--resume', type=str, default=None,
                    help='从已有模型继续训练（填 .zip 路径）')
    ap.add_argument('--tb', type=str, default='runs',
                    help='TensorBoard 日志目录（设为空字符串关闭）')
    ap.add_argument('--step-penalty', type=float, default=0.002,
                    help='每步奖励整形惩罚（消融实验设 0）')
    ap.add_argument('--exp', type=str, default='',
                    help='实验标签：隔离日志/快照/TensorBoard 产物（消融对照组用）')
    args = ap.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        USE_MASK = True
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO
        ActionMasker = None
        USE_MASK = False
        print('[提示] 未安装 sb3-contrib，退化为普通 PPO（非法动作靠奖励惩罚学习）')

    CKPT_DIR = os.path.join('checkpoints', args.exp) if args.exp else 'checkpoints'
    os.makedirs(CKPT_DIR, exist_ok=True)
    snapshots = []  # 历史模型路径
    if args.resume:
        import glob
        snapshots.extend(sorted(glob.glob(os.path.join(CKPT_DIR, 'snap_*.zip'))))
        print(f'[续训] 预载对手池 {len(snapshots)} 个历史快照')
    SelfPlayEnv.model_cls = MaskablePPO

    config = dict(size=args.size, stepPenalty=args.step_penalty)

    # 对手策略：优先从快照池采样（只用最近 5 个，早期弱快照价值低且省内存），否则启发式
    def make_opponent(get_model):
        def opponent(state, color, rng):
            if snapshots and rng.random() > args.heuristic_ratio:
                try:
                    model = get_model(rng.choice(snapshots[-5:]))
                    obs = env_obs_for(state, color)
                    if USE_MASK:
                        act, _ = model.predict(obs, action_masks=np.array([action_mask(state, color)]), deterministic=False)
                    else:
                        act, _ = model.predict(obs, deterministic=False)
                    return action_to_move(state, color, int(act.item() if hasattr(act, 'item') else act))
                except Exception:
                    return heuristic.choose_move(state, color, rng=rng)
            return heuristic.choose_move(state, color, rng=rng)
        return opponent

    # 说明：环境里的"学习方"每步走完后由对手回应；学习方颜色每局随机
    env = SelfPlayEnv(config, make_opponent)
    if USE_MASK:
        env = ActionMasker(env, lambda e: e.unwrapped.action_mask())

    # 自定义特征提取器：棋盘空间信息压平 + 元数据拼接 → MLP
    # （默认 CNN 为 Atari 大图设计，小棋盘上直接打平接 MLP 更稳、更快）
    import torch as th
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class BoardMLP(BaseFeaturesExtractor):
        def __init__(self, obs_space, features_dim=256):
            super().__init__(obs_space, features_dim)
            s = obs_space['board'].shape[0]
            in_dim = s * s * 4 + s * s * 2 + 8
            self.net = th.nn.Sequential(
                th.nn.Linear(in_dim, 512), th.nn.ReLU(),
                th.nn.Linear(512, 512), th.nn.ReLU(),
                th.nn.Linear(512, features_dim), th.nn.ReLU(),
            )

        def forward(self, obs):
            x = th.cat([
                obs['board'].reshape(obs['board'].shape[0], -1),
                obs['pieces'].reshape(obs['pieces'].shape[0], -1),
                obs['meta'],
            ], dim=1)
            return self.net(x)

    policy_kwargs = dict(
        features_extractor_class=BoardMLP,
        net_arch=[256, 256],
    )
    tb_kwargs = dict(tensorboard_log=args.tb) if args.tb else {}
    if args.resume:
        model = MaskablePPO.load(args.resume, env=env, verbose=1, **tb_kwargs)
        print(f'[续训] 从 {args.resume} 继续')
    else:
        model = MaskablePPO('MultiInputPolicy', env, verbose=1,
                            learning_rate=3e-4, n_steps=2048, batch_size=256,
                            policy_kwargs=policy_kwargs, **tb_kwargs)

    # 周期快照回调（_on_rollout_end 在 n_steps 边界触发，不依赖逐步取模的巧合）
    from stable_baselines3.common.callbacks import BaseCallback
    import json
    import time
    from collections import deque

    LOG_FILE = f'training_log{"_" + args.exp if args.exp else ""}.jsonl'

    class SnapshotCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self._next = None  # 惰性初始化，兼容续训时 num_timesteps 不从 0 开始

        def _on_rollout_end(self) -> None:
            if self._next is None:
                self._next = (self.num_timesteps // args.snapshot_every + 1) * args.snapshot_every
            if self.num_timesteps >= self._next:
                self._next += args.snapshot_every
                path = os.path.join(CKPT_DIR, f'snap_{self.num_timesteps}.zip')
                self.model.save(path)
                snapshots.append(path)
                print(f'[快照] {path}（池大小 {len(snapshots)}）')

        def _on_step(self) -> bool:
            return True

    class MetricsCallback(BaseCallback):
        """每次 rollout 结束把训练指标追加到 training_log.jsonl（供网页监控面板读取）。"""

        def __init__(self):
            super().__init__()
            self._results = deque(maxlen=100)  # 近100局: 1=学习方胜 0=平 -1=负
            self._lens = deque(maxlen=100)
            self._ep_rew = deque(maxlen=100)
            self._t0 = time.time()
            self._base_steps = None  # 续训时 num_timesteps 不从 0 开始，fps 需减去基数
            # 全新训练才清空日志；续训保留历史，曲线连续
            if not args.resume:
                open(LOG_FILE, 'w').close()

        def _on_step(self) -> bool:
            for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
                if done and 'winner' in info:
                    w, learner = info['winner'], info.get('learner')
                    self._results.append(1 if w == learner else (0 if w == 'draw' else -1))
            return True

        def _on_rollout_end(self) -> None:
            lv = self.model.logger.name_to_value
            ep_info = list(getattr(self.model, 'ep_info_buffer', []))
            if ep_info:
                self._ep_rew.extend(e['r'] for e in ep_info[-20:])
                self._lens.extend(e['l'] for e in ep_info[-20:])
            n = max(len(self._results), 1)
            win_rate = sum(1 for r in self._results if r == 1) / n
            draw_rate = sum(1 for r in self._results if r == 0) / n
            # 自定义指标写入 TensorBoard（与 SB3 内置指标同面板）
            self.logger.record('terrain/win_rate_100', win_rate)
            self.logger.record('terrain/draw_rate_100', draw_rate)
            self.logger.record('terrain/loss_rate_100', sum(1 for r in self._results if r == -1) / n)
            if self._lens:
                self.logger.record('terrain/ep_len_100', sum(self._lens) / len(self._lens))
            self.logger.record('terrain/snapshot_pool', len(snapshots))
            if self._base_steps is None:
                self._base_steps = self.num_timesteps
            elapsed = time.time() - self._t0
            f32 = lambda v: None if v is None else round(float(v), 5)  # logger 值是 numpy float32
            row = {
                'ts': round(time.time(), 1),
                'steps': self.num_timesteps,
                'total': self.model._total_timesteps,
                'fps': round((self.num_timesteps - self._base_steps) / max(elapsed, 1e-6)),
                'ep_rew': round(sum(self._ep_rew) / max(len(self._ep_rew), 1), 4) if self._ep_rew else None,
                'ep_len': round(sum(self._lens) / max(len(self._lens), 1), 1) if self._lens else None,
                'win_rate': round(win_rate, 4),
                'draw_rate': round(draw_rate, 4),
                'loss_rate': round(sum(1 for r in self._results if r == -1) / n, 4),
                'entropy': f32(lv.get('train/entropy_loss')),
                'value_loss': f32(lv.get('train/value_loss')),
                'expl_var': f32(lv.get('train/explained_variance')),
                'approx_kl': f32(lv.get('train/approx_kl')),
                'snapshots': len(snapshots),
            }
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')

    from stable_baselines3.common.callbacks import CallbackList
    learn_kwargs = dict(tb_log_name=f'ppo_terrain_{args.size}{"_" + args.exp if args.exp else ""}') if args.tb else {}
    model.learn(total_timesteps=args.steps,
                callback=CallbackList([SnapshotCallback(), MetricsCallback()]),
                reset_num_timesteps=not args.resume,  # 续训保持计数连续（steps 目标为绝对总量）
                **learn_kwargs)
    model.save(args.out)
    print(f'训练完成 → {args.out}')


# ---- 环境封装：随机分配学习方颜色 ----
from gym_env import DIRS


def action_to_move(state, color, action):
    me = state[color]
    if action == 8:
        return None
    dr, dc = DIRS[action]
    return (me['r'] + dr, me['c'] + dc)


def env_obs_for(state, color):
    """从任意局面构造观测（与 gym_env._obs 同布局，但显式指定视角颜色）。"""
    s = state['size']
    board = np.zeros((s, s, 4), dtype=np.float32)
    for i, v in enumerate(state['pillars']):
        board[i // s, i % s, v] = 1.0
    me = state[color]
    opp = state['white' if color == 'black' else 'black']
    pieces = np.zeros((s, s, 2), dtype=np.float32)
    pieces[me['r'] - 1, me['c'] - 1, 0] = 1.0
    pieces[opp['r'] - 1, opp['c'] - 1, 1] = 1.0
    dist = max(abs(me['r'] - opp['r']), abs(me['c'] - opp['c']))
    phase = E.phase_for(state, color)
    g = state['config']['goalEdges']
    gd = s
    if 'N' in g:
        gd = min(gd, me['r'] - 1)
    if 'S' in g:
        gd = min(gd, s - me['r'])
    if 'W' in g:
        gd = min(gd, me['c'] - 1)
    if 'E' in g:
        gd = min(gd, s - me['c'])
    meta = np.array([
        float(E.pillar_height(state, me['r'], me['c'])),
        float(E.pillar_height(state, opp['r'], opp['c'])),
        dist / s, gd / s,
        1.0 if phase == 'cross' else 0.0,
        1.0 if phase == 'x' else 0.0,
        1.0 if phase == 'all8' else 0.0,
        1.0 if color == 'white' else 0.0,
    ], dtype=np.float32)
    return {'board': board, 'pieces': pieces, 'meta': meta}


class SelfPlayEnv(TerrainChessEnv):
    """每局随机决定学习方执白还是执黑；另一方由 opponent_fn 驱动。"""

    model_cls = None  # 由 main() 注入（MaskablePPO 或 PPO），供加载快照用

    def __init__(self, config, make_opponent_fn):
        self._make_opponent = make_opponent_fn
        self._model_cache = {}

        def get_model(path):
            if path not in self._model_cache:
                self._model_cache[path] = SelfPlayEnv.model_cls.load(path)
                if len(self._model_cache) > 12:  # 对手池窗口为 5，缓存 12 保证命中、控制内存
                    self._model_cache.pop(next(iter(self._model_cache)))
            return self._model_cache[path]

        self._opponent = make_opponent_fn(get_model)
        super().__init__(config, opponent_fn=None)
        self.learner_color = 'white'

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.learner_color = self.rng.choice(['white', 'black'])
        # 对手先走（若对方先手）
        while self.state['status'] == 'playing' and self.state['turn'] != self.learner_color:
            self._opp_step()
        return self._obs(), info

    def action_mask(self):
        """ActionMasker 通过它取学习方视角的合法动作掩码。"""
        return np.array(action_mask(self.state, self.learner_color), dtype=np.int8)

    def _obs(self):
        return env_obs_for(self.state, self.learner_color) if self.state['status'] == 'playing' else super()._obs()

    def _opp_step(self):
        color = self.state['turn']
        mv = self._opponent(self.state, color, self.rng)
        legal = E.legal_moves(self.state, color)
        if mv is None and not self.state['config']['passAllowed']:
            mv = self.rng.choice(legal) if legal else None
        elif mv is not None and mv not in legal:
            mv = self.rng.choice(legal) if legal else None
        E.apply_move(self.state, color, mv)

    def step(self, action):
        st = self.state
        move = action_to_move(st, self.learner_color, int(action))
        legal = E.legal_moves(st, self.learner_color)
        invalid = False
        if move is None:
            if not st['config']['passAllowed']:
                invalid = True
        elif move not in legal:
            invalid = True
        if invalid:
            move = self.rng.choice(legal) if legal else None

        E.apply_move(st, self.learner_color, move)
        # 对手回应直到又轮到学习方或终局
        while st['status'] == 'playing' and st['turn'] != self.learner_color:
            self._opp_step()

        done = st['status'] == 'over'
        reward = 0.0
        if done:
            if st['winner'] == self.learner_color:
                reward = 1.0
            elif st['winner'] != 'draw':
                reward = -1.0
        reward -= 0.002
        if invalid:
            reward -= 0.05
        info = {'winner': st['winner'], 'reason': st.get('winReason'), 'learner': self.learner_color}
        return self._obs(), reward, done, False, info


if __name__ == '__main__':
    main()
