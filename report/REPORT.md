# Terraflux (涌陆): A Board Game Where the Board Plays Back

**Design of an original pursuit-evasion game on a self-modifying board — and what self-play reinforcement learning reveals about it**

*Your Name · August 2026*
*Code: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) · Playable demo: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)*

<!-- ============================================================
写作总纲（写完请删除所有此类注释块）

报告主线（新）：我设计了一个"棋盘会自我改写"的棋，三个维度打破了
棋类几百年的默认设定；然后我把它变成 RL 实验台，定量研究这个
前所未见的对象，发现了三个反直觉的现象。

目标篇幅：5–6 页 PDF（约 2200 词正文 + 4 图 4 表）。
语气：第一人称、主动语态。
数字纪律：所有数字已预填自 experiments/REPORT_DATA.md，【不要手改】。
所有机制细节（规则/超参/智能体配置）查 report/写作底稿.md。
============================================================ -->

## Abstract

<!-- 【你来写】150 词以内，三段式：
第 1 段（2 句）：传统棋类的棋盘是静止的舞台；我设计了一个每步落子都会
  改写地形的追逃棋——Terraflux。
第 2 段（2 句）：设计上的三个突破点各半句（流体地形 / 高度属于土地 /
  非对称追逃与轮换相位）；我用 PPO 自我对弈把它变成实验台。
第 3 段（2 句）：三个发现各一句——奖励整形加速学习但日志胜率会骗人；
  自我对弈的震荡是共进化而非崩溃；"公平"不是布局的固有属性，
  而是随玩家强度爬升的光谱（0%→46%）。
-->

> [Abstract to be written]

## 1. Introduction — Boards That Don't Sit Still

<!-- 【你来写】约 250–300 词。结构：
1) 钩子开场（1–2 句）：围棋下在 19×19 的永恒木盘上，象棋的九宫格
   千年不动——棋类设计几百年来的默认设定是"棋盘是舞台，棋子是演员"。
   如果舞台本身也是演员呢？
2) 谱系定位（3–4 句）：棋类史上棋盘并非没有变化——Zertz 的板块会拆除
   收缩、Dvonn 把高度做进了棋子堆叠——但"落子动作本身持续改写地形"
   没有先例（参考 §2 的 Table 1）。
3) 问题引出（2 句）：当棋盘都变成动态的，一个传统问题反而变得无解——
   "这游戏公平吗？"人工推演在流动的地形上失效，没有人类高手、没有定式。
   唯一的出路：训练一个足够强的 AI 来当裁判。
-->

> [Section 1 to be written]

![Figure 1. The playable 3D simulator — every visitor can watch a narrated demo game in the browser](../experiments/fig_size_fairness.png)
<!-- 上图是占位：换一张网站截图（棋盘+演示解说条），命名
     report/fig_screenshot.png 并替换上面的路径。 -->

## 2. Design — Three Breaks from Tradition

<!-- 【你来写】约 400–450 词。这是全文最独特的一节，三个小节各约 130–150 词。
每节写法固定：这机制是什么（2 句）→ 传统棋类为什么从不这么做（1 句）→
它改变了什么（1–2 句）。规则细节全部可查 report/写作底稿.md §1.1。
-->

### 2.1 The Fluid Board

<!-- 要点：25×25 柱阵；每格三态循环（隆起+1 / 下陷−1 / 填平不可进入）；
每次落子改写落点周围（相位决定改哪些方向）；关键强调——填平不是永久的，
后续相位可以让它复活，地形是流体而非消耗品。传统棋类里"好位置"是永恒的
（金角银边）；这里只有暂时的好位置。 -->

> [2.1 to be written]

### 2.2 Height Belongs to the Land

<!-- 要点：高度不是棋子的属性（Dvonn 的堆叠是棋子叠棋子），而是地形的
属性——棋子不变，地把它顶起来。抓捕条件是"相邻 + 黑严格高于白"：
吃子不再是"移动到你所在的格子"，而是"居高临下"。这制造了高度军备竞赛：
你站高一步，我就得站得更高。 -->

> [2.2 to be written]

### 2.3 Asymmetric Goals, Alternating Physics, Simultaneous Fates

<!-- 要点：白逃向南边线、黑追缉——双方目标根本不同（传统棋类目标对称）；
地形改写规则按各白手数在十字/斜向两个相位间轮换，等于物理定律每手换
一次；最罕见的是同瞬平局——白踏上胜利边的一步若恰好同时满足抓捕条件，
逃脱与死亡同一瞬间发生，判平。棋类里几乎没有第二种"胜负同瞬"的设计。 -->

> [2.3 to be written]

**Table 1. Terraflux in the design space of classic abstract games**

| Dimension | Go / Chess | Zertz | Dvonn | **Terraflux** |
|---|---|---|---|---|
| Board | Static forever | Shrinks (tiles removed) | Static | **Rewritten by every move** |
| Height | — | — | Stacked pieces | **Terrain lifts the piece** |
| Objective | Symmetric | Symmetric | Symmetric | **Asymmetric (flee vs. hunt)** |
| Capture | Move onto piece | Jump over | Surround | **Adjacent + strictly higher** |

## 3. Methods — Turning the Game into an RL Laboratory

<!-- 【你来写】约 250–300 词，三段：
1) 双实现互验：JS 引擎（驱动 Three.js 3D 前端）+ Python 逐行移植
   （驱动训练与批量评测），保证"网页上玩的规则"="AI 训练的规则"。
2) PPO 配置：MaskablePPO + 非法动作掩码；观测 = 25×25×4 地形 one-hot
   + 双方位置 + 8 维元信息；动作 = 8 方向+停步；奖励 = 终局 ±1（稀疏），
   实验组加每步 −0.002 步罚；自我对弈对手池 = 65% 启发式 + 35% 最近 5 个
   历史快照；主训练 50 万步。
3) 评估协议：自对弈胜率 + 模型间循环赛直接对局（200–300 局、95% CI）；
   方法论纪律：跨模型强度只用循环赛，不信训练日志胜率（§4.1 的伏笔）。
-->

> [Section 3 to be written]

## 4. Results — Three Findings

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
1) 末段日志胜率 0.716 vs 0.709——我当时以为两组一样强；
2) 但日志胜率是"各自对自己的快照池"的相对记账，池子强弱不同就不可比；
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
3) 诊断证据：全程无 KL 尖峰（>0.15）；仅 94k/104k 两次熵跳变，且正好卡在
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

<!-- 【你来写】约 300 词。四段结构：
1) 现象：白胜率随策略强度 0% → 20% → 34% → 46% 严格单调，向 50% 收敛；
2) 对照：尺寸扫描（15×15–27×27）全是平线——尺寸不是公平性变量，强度才是；
3) 机制：弱白的评估里间距项被趋边冲动压倒，事实上不会逃，黑直线追击
   10 步抓死；PPO 自学了保持间距与借地形卡位——逃跑是门要学的技术；
4) 核心论点：规则公平性不能脱离玩家强度定义；"平衡开局"是强度依赖的谱，
   不是布尔属性；只有收敛的 RL 策略有资格当公平性的裁判。
-->

> [4.3 narrative to be written]

## 5. Discussion — Limitations and Future Work

<!-- 四条局限性，每条 1–2 句英文改写（素材在写作底稿 §4.2）：
单种子消融；弱策略仅启发式一类代表；强策略尺寸扫描受算力所限；
样本 200–300 局、95% CI 约 ±6%。
未来工作挑 2 条：多种子复核；强策略尺寸×公平性扫描；
LLM 智能体作为第三类强度样本；以及机制向的新问题——当环境被双方动作
持续改写时，策略学习会发生什么（呼应 §2 的独特性）。
-->

> [Section 5 to be written]

## 6. References and Links

- **Code & full experiment data**: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux)
- **Playable 3D demo (bilingual, with narrated tutorial game)**: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Huang, S. & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. arXiv:2006.14171.
