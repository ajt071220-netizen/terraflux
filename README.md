# Terrain Chess · 地形棋

**An original board game with a living board — and a reinforcement-learning study of what "fair rules" even mean.**

Every move rewrites the terrain under the pieces: pillars rise, sink, and fill in.
The pursuer (black) can only capture from a height advantage, so the whole game is a fight
over the *shape of the board itself*. This repo contains a playable 3D simulator (Three.js),
three kinds of agents (heuristic / PPO self-play / LLM), and a set of experiments on
rule fairness, reward shaping, and self-play training dynamics.

## Research highlights

**1. Fairness is strength-conditional.** The "balanced" opening (duel, gap 3, black first)
shows a perfect monotone ladder — 0% → 20% → 34% → 46% white win rate as the policy gets
stronger, converging toward 50% only for the strongest agent. A rule set cannot be called
fair or unfair without specifying *who is playing*. Heuristic-based playtesting would have
declared this layout hopeless for white; converged PPO self-play shows it is nearly balanced.

![Strength-conditional fairness](experiments/fig_strength_fairness.png)

**2. Reward shaping: accelerator, not ceiling-raiser — and log win-rates lie.**
An ablation pair (50k-step rolling window, 500k steps each, identical except for a
−0.002/step penalty) shows shaping speeds up early learning by ~24% (59k vs 73k steps to
55% win rate). Training logs suggest both converge to the same strength (0.716 vs 0.709) —
but a direct round-robin match exposes that as an artifact of self-play relative win rates:
the shaped agent crushes the sparse one **66.0% : 14.5%**.

![Reward shaping ablation](experiments/fig_ablation.png)

**3. Self-play dynamics: oscillation is the signature, not a bug.**
Mid-training win rate oscillates persistently because the opponent pool is made of the
learner's own snapshots — both sides improve together. Explained variance climbs 0.2 → 0.8
but never saturates: the environment is permanently non-stationary.

![Training dynamics](experiments/fig_training_dynamics.png)

**4. Board size does not move fairness** (for weak policies, 15×15–27×27 is a flat line),
while policy strength moves it decisively — size is the wrong knob, strength is the right lens.

![Size scan](experiments/fig_size_fairness.png)

Full experiment write-up with all numbers, confidence intervals, and limitations:
[experiments/REPORT_DATA.md](experiments/REPORT_DATA.md)

## Run it

```bash
node server.js        # then open http://localhost:5173
```

Zero npm runtime dependencies — Three.js is vendored in `lib/`.
White and Black can each be set to **Human / Heuristic / LLM / PPO** in the settings panel
(gear icon, top right). UI is bilingual (中文 / English toggle in the top bar).

- **PPO agent**: train with `python/train_ppo.py` (MaskablePPO self-play, Gymnasium),
  serve with `python/serve_ppo.py` — see [python/README.md](python/README.md).
- **LLM agent**: copy `config.example.json` → `config.json`, fill in any OpenAI-compatible
  endpoint (developed against Qwen). The server builds a prompt with per-move outcome
  previews; the engine vets unsafe LLM moves (hybrid propose-and-veto architecture).

## How the game works

- 25×25 board of pillars, each with a dimple "seat". White escapes, black pursues.
- Moving to any of 8 neighbors triggers a terrain mutation around the landing cell,
  alternating between two phases: **cross** (N rises, S sinks, E/W fill) and
  **X** (NE/NW rise, SE/SW sink). Filled cells become impassable.
- Capture: adjacent + black **higher** than white. White wins by reaching the goal edge
  alive; simultaneous capture-and-escape is a draw.

## Repository layout

```
src/engine.js     pure rule engine (no rendering deps) — single source of truth
src/scene.js      Three.js rendering (instanced pillars, raycast picking)
src/ai.js         heuristic agent (2-ply greedy + noise)
server.js         static server + /api/agent (LLM proxy, PPO bridge) + replay storage
python/           engine port, Gymnasium env, MaskablePPO self-play training,
                  model serving, and the experiment scripts behind every figure above
experiments/      REPORT_DATA.md + figures + raw logs (the evidence chain)
```

---

<details>
<summary>中文版（精简）</summary>

地形棋是一个原创动态地形棋类游戏：每一步落子都会改写周围地形（隆起/下陷/填平），
黑白两球在高度差里分出胜负。本仓库包含：

- **3D 可玩模拟器**（Three.js，零 npm 依赖）：`node server.js` 后打开 http://localhost:5173
- **三类智能体**：启发式 / PPO 自我对弈（MaskablePPO）/ 大语言模型（OpenAI 兼容接口）
- **强化学习实验**：奖励整形消融、自我对弈训练动态解剖、规则公平性的强度条件化研究——
  全部数据与图表见 [experiments/REPORT_DATA.md](experiments/REPORT_DATA.md)

核心发现：规则公平性是策略强度条件化的——同一开局布局，白胜率随策略强度从 0% 单调
收敛至 46%；奖励整形加速学习约 24%，且自我对弈的日志胜率会掩盖真实强度差（循环赛
66:14.5）；棋盘尺寸对公平性几乎无影响。

</details>
