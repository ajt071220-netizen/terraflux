# Terraflux · 涌陆

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16898141.svg)](https://doi.org/10.5281/zenodo.16898141)

**A pursuit-evasion board game on a self-modifying board — and a reinforcement-learning study of what "fair rules" even mean.**
*(formerly "Terrain Chess")*

Every move rewrites the terrain under the pieces: pillars rise, sink, and fill in.
The pursuer (black) can only capture from a height advantage, so the whole game is a fight
over the *shape of the board itself*. This repo contains a playable 3D simulator (Three.js),
three kinds of agents (heuristic / PPO self-play / LLM), and a set of experiments on
rule fairness, reward shaping, and self-play training dynamics.

## Research status (v2 — corrected)

> The first version of the study had a **reward bug**: the training environment hard-coded the step penalty, so the "sparse" arm of the reward-shaping ablation was trained with the same penalty as the "shaped" arm. Conclusions drawn from that comparison are withdrawn. Replicate runs (seeds recorded) show that outcomes vary more across training runs than across the configurations compared. See **[report/REPORT.md](report/REPORT.md)** (v2) for the full revision note; the superseded v1 text is kept in `report/REPORT_v1_superseded.md`.

What the replicates currently support:

1. **Training does not settle.** Rolling win rates keep oscillating in every dynamic-terrain run, and the value function's explained variance plateaus well below 1. A twenty-run opponent-pool ablation leaves the oscillation unchanged (p = 0.50). Without terrain rewriting the duel layout is decided by colour, so a static-board control cannot isolate the cause.
2. **Sparse rewards are viable.** All ten sparse runs learned the game; they do not differ significantly from thirteen penalised runs in learning speed (p ≈ 0.43) or late-stage win rate (p ≈ 0.34).
3. **Balance depends on who plays.** In self-play, the White win rate across runs with identical settings ranges from 9% to 84%. In population cross-play the duel layout at its design gap of 3 favours White (71% of games), while heuristic self-play gives White 0%, and the gap closest to 50% depends on the evaluator. Gap 3 is kept as a design setting.
4. **The same-instant draw is rare but real.** It ends 0.8% of dynamic-terrain self-play games and never occurs on a static board.

![White win rate by training run](experiments/fig_selfplay.png)

Scripts: `python/run_robustness.ps1` (training batch), `python/analyze_robustness.py`, `python/replicate_eval.py`, `python/fig_replicates.py`; opponent-pool ablation `python/run_ablation2.ps1` and `python/analyze_pool_ablation.py`; layout balance `python/fairness_pop.py` and `python/scan_gap.py`; end-of-game reasons `python/draw_types_pop.py`. Data: `experiments/robustness_training.json`, `experiments/replicate_eval.json`, `experiments/fairness_pop_gap{1..8}.json`, `experiments/gap_scan.json`, `experiments/draw_types_pop_gap3.json`. All numbers are collected in `experiments/REPORT_DATA.md`.

## Run it

```bash
node server.js        # then open http://localhost:5173
```

Zero npm runtime dependencies — Three.js is vendored in `lib/`.
White and Black can each be set to **Human / Heuristic / LLM / PPO** in the settings panel
(gear icon, top right). UI is bilingual (中文 / English toggle in the top bar).

New visitors get an interactive tutorial: three illustrated rule cards, then an optional
**narrated demo game** — two heuristic agents play a full match on the 3D board while a
caption bar explains the goal, terrain phases, and height-based capture as they happen
(works fully offline on the static Pages build).

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

涌陆（Terraflux，原名"地形棋"）是一个原创动态地形追逃棋：每一步落子都会改写周围
地形（隆起/下陷/填平），黑白两球在高度差里分出胜负。本仓库包含：

- **3D 可玩模拟器**（Three.js，零 npm 依赖）：`node server.js` 后打开 http://localhost:5173
- **三类智能体**：启发式 / PPO 自我对弈（MaskablePPO）/ 大语言模型（OpenAI 兼容接口）
- **强化学习实验（v2，已修正）**：完整报告见 [report/REPORT.md](report/REPORT.md)；
  v1 的结论已作废，更正说明见 [report/ERRATUM_v1.md](report/ERRATUM_v1.md)

v2 的结论（多种子复现）：动态地形下胜率持续震荡、不会稳定，对手池消融也不能消除这种震荡；
稀疏奖励可行，10 个稀疏奖励运行都学会了这个游戏，步罚对学习速度的影响与运行间随机差异无法区分；
平衡取决于由谁来下：同一设置下自对弈白胜率约 9%–84%，群体互弈中 gap 3 的白胜率为 71%，
启发式自对弈为 0%，最接近 50% 的 gap 也因评估者而异（gap 3 作为游戏设定保留）；
同瞬平局罕见但确实存在，约占动态地形自对弈对局的 0.8%，静态棋盘上从未出现。
v1 中"公平性随策略强度单调上升"与"稀疏奖励慢约 24%"均因代码 bug 与单次运行而不成立。
全部数字汇总在 `experiments/REPORT_DATA.md`。

</details>
