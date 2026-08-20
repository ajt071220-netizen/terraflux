# Terraflux (涌陆): Training Self-Play RL on a Board Game That Rewrites Itself

**Reward shaping, self-play dynamics, and what "fair rules" mean in an original pursuit-evasion game**

*Your Name · August 2026*
*Code: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) · Playable demo: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)*

<!-- ============================================================
写作总纲（写完请删除所有此类注释块）

报告主线（定稿）：我设计了一个"棋盘会自我改写"的原创棋（§2 简述，
含三个设计突破作为亮点）；把它变成 RL 实验台（§3）；三个实验（§4，
全文主菜）：奖励整形消融、自我对弈训练动态、公平性的强度条件化。

目标篇幅：5 页 PDF（约 2000 词正文 + 4 图 4 表）。
语气：第一人称、主动语态。
数字纪律：所有数字已预填自 experiments/REPORT_DATA.md，【不要手改】。
机制/配置细节全部查 report/写作底稿.md。
============================================================ -->

## Abstract

<!-- 【你来写】≤150 词，三段式：
第 1 段（2 句）：我设计了 Terraflux——一个每步落子都改写地形的追逃棋
  （流体地形/地形高度/非对称追逃，各半句）。
第 2 段（2 句）：我把它变成 RL 实验台（JS+Python 双实现、PPO 自我对弈、
  三类智能体、可玩 3D 模拟器）。
第 3 段（2–3 句）：三个发现各一句——①步罚加速学习约 24%，且自我对弈的
  日志胜率会掩盖真实强度差（循环赛 66:14.5）；②中期胜率宽幅震荡是
  共进化签名而非训练崩溃；③"公平"不是布局的固有属性——同一开局白胜率
  随策略强度 0%→46% 单调爬升。
-->

> [Abstract to be written]

## 1. Introduction

<!-- 【你来写】约 250–300 词：
1) 钩子（1–2 句）：棋类几百年默认"棋盘是静止的舞台"——如果舞台本身
   也是演员呢？我设计了 Terraflux（涌陆）。
2) 游戏速览（3–4 句）：25×25 柱阵；白逃向南边线、黑追捕（相邻且黑严格
   更高才算抓到）；每步落子改写落点周围地形；相位按各白手数在十字/斜向
   间轮换。
3) 研究动机（2–3 句）：这个动态棋盘带来一串 RL 问题——稀疏奖励下怎么
   训练？自我对弈的训练曲线长什么样？以及最要命的：这游戏公平吗，
   谁来裁决？
4) 预告（1 句）：本文结构——游戏设计（§2）、实验台（§3）、三个实验
   （§4）、讨论（§5）。
-->

> [Section 1 to be written]

![Figure 1. The playable 3D simulator — visitors can watch a narrated demo game in the browser](../experiments/fig_size_fairness.png)
<!-- 占位图：换一张网站截图（3D 棋盘 + 演示解说条），保存为
     report/fig_screenshot.png 并替换上面的路径。 -->

## 2. The Game — Design Highlights

<!-- 【你来写】补充章节，约 250–300 词。三个设计突破各用 2–3 句
（不展开，点到为止——主菜在 §4）：
1) 流体地形：格子三态循环（隆起/下陷/填平），填平可被后续相位复活——
   地形是流体不是消耗品；"好位置"不再永恒。
2) 高度属于土地：高度是地形属性而非棋子属性；抓捕＝相邻＋严格更高，
   吃子从"位移"变成"居高临下"。
3) 非对称追逃＋相位轮换＋同瞬平局：胜负同帧发生的设计几乎没有先例。
最后用一小段（3–4 句）讲设计哲学——价值镜像检验：为什么地形词汇表
只有高度和通行性（传送偏白、冻结偏黑，单边武器全被淘汰）；
候选机制清单（退潮格/自主演化/潮汐线等）留作未来工作（§5 呼应）。
素材查底稿 §1.1 与 §5.5。Table 1 已预填。
-->

> [Section 2 to be written]

**Table 1. Terraflux in the design space of classic abstract games**

| Dimension | Go / Chess | Zertz | Dvonn | **Terraflux** |
|---|---|---|---|---|
| Board | Static forever | Shrinks (tiles removed) | Static | **Rewritten by every move** |
| Height | — | — | Stacked pieces | **Terrain lifts the piece** |
| Objective | Symmetric | Symmetric | Symmetric | **Asymmetric (flee vs. hunt)** |
| Capture | Move onto piece | Jump over | Surround | **Adjacent + strictly higher** |

## 3. Methods — From a Game to an RL Testbed

<!-- 【你来写】约 300 词，三段：
1) 双实现互验：JS 引擎驱动 Three.js 3D 前端，Python 逐行移植驱动训练；
   另接三类智能体（启发式 / LLM 混合架构 / PPO）。
2) PPO 配置：MaskablePPO + 非法动作掩码；观测 = 25×25×4 地形 one-hot
   + 双方位置 + 8 维元信息；动作 = 8 方向+停步；奖励 = 终局 ±1（稀疏），
   实验组加每步 −0.002 步罚；对手池 = 65% 启发式 + 35% 最近 5 个历史
   快照；主训练 50 万步。
3) 评估协议：自对弈胜率 + 模型间循环赛直接对局，每条件 200–300 局、
   95% 置信区间；方法论纪律——跨模型强度只用循环赛，不信训练日志
   胜率（§4.1 的伏笔）。
-->

> [Section 3 to be written]

## 4. Results

### 4.1 Reward shaping speeds learning — and self-play logs lie

两组 MaskablePPO 从零训练，唯一变量为步罚（shaped: −0.002/step；sparse: 0）。
25×25 duel 布局（gap 3、黑先），各 50 万步。

**Table 2. Reward shaping ablation**

| Metric | Shaped | Sparse |
|---|---|---|
| Steps to rolling win rate 55% | **59k** | 73k (+24% slower) |
| Steps to rolling win rate 65% | **65k** | 77k (+18% slower) |
| Late-stage win rate vs own snapshot pool | 0.716 | 0.709 (looks tied) |
| **Head-to-head round robin** | **66.0%** | 14.5% (19.5% draws) |
| Mean game length (late stage) | 28.8 | 29.2 |

![Figure 2. Reward shaping ablation: win rate / episode length / policy entropy](../experiments/fig_ablation.png)

<!-- 【你来写】约 250–300 词，按"被骗 → 醒悟 → 验证"的弧线：
1) 末段日志胜率 0.716 vs 0.709——看似打平；
2) 但日志胜率是"各自对自身快照池"的相对记账，池子强弱不同就不可比；
3) 循环赛直接对局 66 : 14.5——sparse 组学到的是"对弱对手苟活"，遇强即崩；
4) 收束：在自我对弈里，日志胜率是记账单位，不是强度单位。
-->

> [4.1 narrative to be written]

### 4.2 Self-play oscillates — it does not converge, and that is not a bug

主训练 50 万步 rollout 级日志（245 采样点）：

**Table 3. Training phases**

| Phase | Steps | Rolling win rate | Policy entropy | Episode length |
|---|---|---|---|---|
| Early | 2k–165k | 0.419 | −1.119 | 23.1 |
| Mid | 167k–331k | 0.672 | −0.850 | 25.8 |
| Late | 333k–501k | 0.712 | −0.757 | 20.0 |

![Figure 3. Training dynamics: win rate, entropy, episode length, explained variance](../experiments/fig_training_dynamics.png)

<!-- 【你来写】约 200–250 词。要点：
1) 胜率从随机水平爬到 0.87 区间后，中期（150k–400k）持续宽幅震荡；
2) 解释：学习方变强的同时，对手池（自身快照）同步变强——胜率拉锯是
   自我对弈共进化的固有形态，不是训练出错；
3) 诊断证据：全程无 KL 尖峰（>0.15）；仅 94k/104k 两次熵跳变，正好卡在
   快照注入时刻；值函数解释方差 0.2→0.8 未饱和——对手一直在变，
   "这局能不能赢"没有稳定答案。
-->

> [4.2 narrative to be written]

### 4.3 Fairness is strength-conditional

duel 布局（黑镇中央、白正北隔三格、黑先）下，四种强度策略的白胜率：

**Table 4. White win rate rises with policy strength**

| Policy | Round-robin rank | White win rate (duel self-play) | Mean length |
|---|---|---|---|
| Heuristic vs itself | 4 (weakest) | **0%** | 10 |
| PPO sparse | 3 | **20%** (95% CI ±5.6%) | 59 |
| PPO shaped | 2 | **34%** (±6.5%) | 75 |
| PPO main (500k steps) | 1 (strongest) | **46%** (±6.9%) | 54 |

![Figure 4. Left: white win rate rises monotonically with policy strength toward 50%. Right: round-robin strength matrix](../experiments/fig_strength_fairness.png)

<!-- 【你来写】约 300–350 词，四段结构：
1) 现象：同一开局，白胜率随裁判强度 0% → 20% → 34% → 46% 严格单调，
   向 50% 的完美平衡爬升；
2) 对照：七种棋盘尺寸（15×15–27×27）扫描全是平线——几何不是
   公平性变量，强度才是；
3) 机制：弱白方的评估里间距项被趋边冲动压倒，事实上不会逃，黑直线
   追击平均 10 步抓死；强策略自学了保持间距与借地形卡位——
   逃跑是门要学的技术，学会多少，开局就公平多少；
4) 论点：规则公平性不能脱离玩家强度定义；"平衡开局"是强度依赖的谱，
   不是布尔属性；只有收敛的 RL 策略有资格当公平性的裁判。
-->

> [4.3 narrative to be written]

## 5. Discussion — Limitations and Future Work

<!-- 局限性四条，每条 1–2 句英文改写（素材在底稿 §4.2）：
单随机种子；弱策略仅启发式一类代表；强策略尺寸扫描受算力所限；
样本 200–300 局、95% CI 约 ±6%。
未来工作挑 2–3 条：多种子复核；强策略尺寸扫描；LLM 智能体作为第三类
强度样本；机制向流水线——候选机制（退潮格/地形自主演化/潮汐线等，
见底稿 §5.5 八项清单）各称通过价值镜像检验，但每一项都须经 RL 裁判
实测才敢入列："新机制想法 → 实现 → RL 测量 → 数据裁决"。
-->

> [Section 5 to be written]

## 6. References and Links

- **Code & full experiment data**: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux)
- **Playable 3D demo (bilingual, with narrated tutorial game)**: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Huang, S. & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. arXiv:2006.14171.
