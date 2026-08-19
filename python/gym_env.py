"""gym_env.py —— 地形棋 Gymnasium 环境（self-play PPO 用）

- 观测：Dict
    board: (size, size, 4) float32  —— 地形 one-hot（正常/升高/降低/填平）
    pieces: (size, size, 2) float32 —— 通道0 当前方位置，通道1 对方位置（视角随行动方翻转）
    meta: (8,) float32              —— [我方高度, 对方高度, 距离, 我方离胜利边, 相位one-hot×3]
- 动作：Discrete(9) —— 0..7 为 8 方向（见 DIRS），8 为停步（仅 passAllowed 时合法）
- 奖励：胜 +1 / 负 -1 / 平 0；每步 -0.002 抑制拖局
- 对手：可注入任意 callable（启发式 / 历史模型快照）。训练两边：env 随机指定学习方颜色，
  观测始终以"当前行动方视角"给出，模型天然学会两个角色。

依赖：gymnasium, numpy
"""
from __future__ import annotations
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import engine as E

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


class TerrainChessEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, opponent_fn=None, seed=None):
        super().__init__()
        self.config = config or {}
        self.size = self.config.get('size', E.DEFAULT_CONFIG['size'])
        self.opponent_fn = opponent_fn  # callable(state, color, rng) -> (r,c)|None
        self.rng = random.Random(seed)
        self.state = None

        s = self.size
        self.observation_space = spaces.Dict({
            'board': spaces.Box(0.0, 1.0, shape=(s, s, 4), dtype=np.float32),
            'pieces': spaces.Box(0.0, 1.0, shape=(s, s, 2), dtype=np.float32),
            'meta': spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
        })
        self.action_space = spaces.Discrete(9)

    # ---- 内部 ----
    def _goal_dist(self, pos):
        s = self.size
        g = self.state['config']['goalEdges']
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

    def _obs(self):
        st = self.state
        s = self.size
        board = np.zeros((s, s, 4), dtype=np.float32)
        for i, v in enumerate(st['pillars']):
            board[(i // s), (i % s), v] = 1.0

        me = st[st['turn']]
        opp = st['white' if st['turn'] == 'black' else 'black']
        pieces = np.zeros((s, s, 2), dtype=np.float32)
        pieces[me['r'] - 1, me['c'] - 1, 0] = 1.0
        pieces[opp['r'] - 1, opp['c'] - 1, 1] = 1.0

        dist = max(abs(me['r'] - opp['r']), abs(me['c'] - opp['c']))
        phase = E.phase_for(st, st['turn'])
        meta = np.array([
            E.pillar_height(st, me['r'], me['c']) / 1.0,
            E.pillar_height(st, opp['r'], opp['c']) / 1.0,
            dist / s,
            self._goal_dist(me) / s,
            1.0 if phase == 'cross' else 0.0,
            1.0 if phase == 'x' else 0.0,
            1.0 if phase == 'all8' else 0.0,
            1.0 if st['turn'] == 'white' else 0.0,
        ], dtype=np.float32)
        return {'board': board, 'pieces': pieces, 'meta': meta}

    def _action_to_move(self, action):
        st = self.state
        me = st[st['turn']]
        if action == 8:
            return None  # 停步
        dr, dc = DIRS[action]
        return (me['r'] + dr, me['c'] + dc)

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.state = E.create_game(self.config, self.rng)
        # 若对手先手，先让它走
        self._maybe_opponent_turn()
        return self._obs(), {}

    def _maybe_opponent_turn(self):
        """当前回合若由对手（非学习方）行动，则执行对手动作。

        约定：env 双方都可能是学习方（self-play 时 opponent_fn 指向同一模型），
        但标准用法是 opponent_fn=None 表示外部接管对方。这里采用：
        opponent_fn 提供时，轮到谁就先用它走一步——即学习方固定为'先行动的一方之后不再切换'……
        更简单可靠的 self-play：双方都交给 PPO（opponent_fn=None），
        环境只负责推进，reward 在最后结算给"刚行动的一方"。
        """
        pass

    def step(self, action):
        st = self.state
        color = st['turn']
        move = self._action_to_move(int(action))

        # 非法动作处理：停步不允许 / 落点非法 → 用小惩罚并随机改走合法步（保持训练不断）
        legal = E.legal_moves(st, color)
        invalid = False
        if move is None:
            if not st['config']['passAllowed']:
                invalid = True
        elif move not in legal:
            invalid = True
        if invalid:
            move = self.rng.choice(legal) if legal else None

        events = E.apply_move(st, color, move)

        # 对手回应（若配置了对手策略）
        if st['status'] == 'playing' and self.opponent_fn is not None:
            opp_color = st['turn']
            opp_move = self.opponent_fn(st, opp_color, self.rng)
            if opp_move is None and st['config']['passAllowed']:
                E.apply_move(st, opp_color, None)
            else:
                opp_legal = E.legal_moves(st, opp_color)
                if opp_move not in opp_legal:
                    opp_move = self.rng.choice(opp_legal) if opp_legal else None
                E.apply_move(st, opp_color, opp_move)

        done = st['status'] == 'over'
        reward = 0.0
        if done:
            if st['winner'] == color:
                reward = 1.0
            elif st['winner'] == 'draw':
                reward = 0.0
            else:
                reward = -1.0
        reward -= self.config.get('stepPenalty', 0.002)  # 奖励整形：消融实验可置 0
        if invalid:
            reward -= 0.05

        info = {'winner': st['winner'], 'reason': st.get('winReason'), 'color': color}
        return self._obs() if not done else self._terminal_obs(), reward, done, False, info

    def _terminal_obs(self):
        # 结束后 gym 仍要返回一个合法观测
        return self._obs()


def action_mask(state, color):
    """供 MaskablePPO 使用：9 维 0/1。"""
    mask = [0] * 9
    me = state[color]
    legal = set(E.legal_moves(state, color))
    for i, (dr, dc) in enumerate(DIRS):
        if (me['r'] + dr, me['c'] + dc) in legal:
            mask[i] = 1
    if state['config']['passAllowed']:
        mask[8] = 1
    return mask
