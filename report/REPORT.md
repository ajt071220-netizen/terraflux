# Terraflux (涌陆): A Board Game Where the Board Plays Back

**Three breaks from board-game tradition — and what self-play reinforcement learning revealed about each**

*Your Name · August 2026*
*Code: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) · Playable demo: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)*

<!-- ============================================================
写作总纲（写完请删除所有此类注释块）——定稿，不再改版

主线：报告只研究 Terraflux 的三个独特设计突破（§4/§5/§6 各一个）。
每个突破一章，内部固定两段式：
  ① 设计是什么（~150 词，机制描述压缩到此为止——这是"补充"性质）
  ② RL 裁决（~250–300 词，主体——用实验数据回答这个设计引出的问题）
三个实验数据全部编进对应章节，不设独立的"实验章节"：
  突破Ⅰ 流体地形 ← 训练动态数据（非平稳指纹）
  突破Ⅱ 高度属于土地 ← 奖励整形消融 + 日志胜率撒谎
  突破Ⅲ 非对称追逃 ← 公平性强度谱
教科书式的独立 RL 章节（"消融""训练曲线"）不存在——实验为设计问题服务。

目标篇幅：5–6 页 PDF（约 2200 词正文 + 4 图 4 表）。
语气：第一人称、主动语态。
数字纪律：所有数字预填自 experiments/REPORT_DATA.md，【不要手改】。
机制/配置细节查 report/写作底稿.md。
写作顺序：§6 → §5 → §4 → §2 → §3 → §1 → §7 → Abstract。
============================================================ -->

## Abstract

<!-- 【你来写】≤150 词，三段式：
第 1 段（2 句）：我设计了 Terraflux——一个棋盘会自我改写的追逃棋，
  它对棋类传统做了三个突破：流体地形 / 高度属于土地 / 非对称追逃
  （相位轮换与同瞬平局）。
第 2 段（1–2 句）：我用自我对弈强化学习（PPO）当裁判，逐一检验这三个
  设计带来的新问题。
第 3 段（3 句，一突破一句）：①流体地形在训练曲线上留下非平稳指纹——
  胜率永不收敛、值函数永不饱和；②高度压制的长因果链下稀疏奖励可行，
  但自我对弈日志胜率会撒谎，强度只能循环赛裁决（66 : 14.5）；
  ③"公平"不是布局的固有属性——同一开局白胜率随策略强度 0%→46% 爬升。
-->

> [Abstract to be written]

## 1. Introduction — Boards That Don't Sit Still

<!-- 【你来写】约 250 词：
1) 钩子（1–2 句）：棋类几百年默认"棋盘是静止的舞台"——如果舞台本身
   也是演员呢？
2) 我设计了 Terraflux（涌陆），并刻意押上三个打破传统的赌注（各一句
   预告，不展开）：每步落子改写地形；高度属于土地而非棋子；白逃黑捕的
   非对称目标，外加相位轮换与同帧平局。
3) 每个赌注都带来一个传统棋类从未遇到过的问题（各半句）：没有稳态的
   棋盘上学到的策略长什么样？高度决定生死时稀疏奖励够用吗？
   非对称游戏该由谁裁决公平？
4) 方法预告（1–2 句）：我训练自我对弈 PPO 当裁判——本文是三个设计
   突破与它们各自的 RL 裁决。
-->

> [Section 1 to be written]

## 2. The Game in One Page

<!-- 【你来写】约 200 词 + Figure 1。只讲"怎么玩"，不评价设计：
25×25 柱阵；白从北边线附近出发逃向南边线，黑追捕；八向走一格；
回合制（白手数 1/5/9…白先，可选黑先）；每步落子改写落点周围地形
（相位按各白手数在十字/斜向间轮换）；格子三态（隆起/下陷/填平）；
抓捕＝相邻且黑严格更高（默认规则）；平局＝同一落子后黑白同时达边
（同瞬平局）；憋死＝无合法步者判负。一句话带过三类智能体与
双语 3D 模拟器（Figure 1）。素材查底稿 §1.1。
Table 1（下方预填）用来一眼定位 Terraflux 在设计空间中的位置。
-->

> [Section 2 to be written]

![Figure 1. The playable 3D simulator — a narrated demo game runs in the browser](../experiments/fig_size_fairness.png)
<!-- 占位图：换成网站截图（3D 棋盘 + 解说条），存为 report/fig_screenshot.png 后替换路径 -->

**Table 1. Terraflux in the design space of classic abstract games**

| Dimension | Go / Chess | Zertz | Dvonn | **Terraflux** |
|---|---|---|---|---|
| Board | Static forever | Shrinks (tiles removed) | Static | **Rewritten by every move** |
| Height | — | — | Stacked pieces | **Terrain lifts the piece** |
| Objective | Symmetric | Symmetric | Symmetric | **Asymmetric (flee vs. hunt)** |
| Capture | Move onto piece | Jump over | Surround | **Adjacent + strictly higher** |

## 3. Manufacturing the Referee

<!-- 【你来写】约 250 词，简短三段：
1) 为什么需要 AI 裁判：没有人类玩家社群，只能"制造裁判"；JS 引擎驱动
   Three.js 3D 前端，Python 逐行移植驱动训练，双实现互验。
2) PPO 配置：MaskablePPO + 非法动作掩码；观测 = 25×25×4 地形 one-hot
   + 双方位置 + 8 维元信息；动作 = 8 方向+停步；奖励 = 终局 ±1（稀疏），
   对照组加每步 −0.002；对手池 = 65% 启发式 + 35% 最近 5 个历史快照；
   主训练 50 万步。
3) 评估协议与纪律：自对弈胜率 + 模型间循环赛，每条件 200–300 局、
   95% 置信区间；跨模型强度只信循环赛——§5 会证明这条纪律的必要性。
-->

> [Section 3 to be written]

## 4. Break I — The Board Plays Back

<!-- 【设计段·你来写】约 150 词：
传统棋的地形消耗且单向（Zertz 抽板只会变少）；Terraflux 的格子在
隆起/下陷/填平间循环，填平可被后续相位复活——地形是流体不是消耗品。
后果：没有永恒的好位置；安全区会塌陷，死路会重生。
引出问题：棋盘没有稳态，学出来的策略长什么样？训练会收敛吗？
-->

> [4-design to be written]

**The RL verdict.** 主训练 50 万步 rollout 级日志（245 采样点）：

**Table 2. Training phases on a fluid board**

| Phase | Steps | Rolling win rate | Policy entropy | Episode length |
|---|---|---|---|---|
| Early | 2k–165k | 0.419 | −1.119 | 23.1 |
| Mid | 167k–331k | 0.672 | −0.850 | 25.8 |
| Late | 333k–501k | 0.712 | −0.757 | 20.0 |

![Figure 2. Training dynamics: win rate, entropy, episode length, explained variance](../experiments/fig_training_dynamics.png)

<!-- 【裁决段·你来写】约 250–300 词：
1) 现象：胜率爬到 0.87 区间后，中期（150k–400k）持续宽幅震荡，
   不收敛；
2) 解释：双重非平稳——地形每步被改写 + 对手池（自身快照）同步变强；
   震荡不是训练出错，而是"流体棋盘 + 共进化"的固有形态；
3) 诊断：全程无 KL 尖峰（>0.15）；仅 94k/104k 两次熵跳变，正好卡在
   快照注入时刻；值函数解释方差 0.2→0.8 后未饱和——对手与棋盘一直
   在变，"这局能不能赢"没有稳定答案；
4) 收束：这就是"舞台也是演员"在训练曲线上的指纹。流体地形不是
   视觉噱头——它从根上改变了学习问题的形态。
-->

> [4-verdict to be written]

## 5. Break II — The Terrain Owns the Height

<!-- 【设计段·你来写】约 150 词：
传统棋的高度属于棋子（Dvonn 堆叠、军棋等级）；Terraflux 的高度属于
土地——棋子站上去才被抬高。抓捕从"位移"（走上去吃子）变成
"居高临下"（相邻＋严格更高）；地势差可白赚（黑 1 白 −1 差 2）。
引出问题："建立高度优势 → 完成抓捕"是一条长因果链，中间没有任何
奖励信号——±1 终局稀疏奖励，够学会居高临下吗？
-->

> [5-design to be written]

**The RL verdict.** 两组 MaskablePPO 从零训练，唯一变量为步罚
（shaped: −0.002/step；sparse: 0）。25×25 duel 布局（gap 3、黑先），各 50 万步。

**Table 3. Sparse reward suffices — but self-play logs lie**

| Metric | Shaped | Sparse |
|---|---|---|
| Steps to rolling win rate 55% | **59k** | 73k (+24% slower) |
| Steps to rolling win rate 65% | **65k** | 77k (+18% slower) |
| Late-stage win rate vs own snapshot pool | 0.716 | 0.709 (looks tied) |
| **Head-to-head round robin** | **66.0%** | 14.5% (19.5% draws) |
| Mean game length (late stage) | 28.8 | 29.2 |

![Figure 3. Reward shaping ablation: win rate / episode length / policy entropy](../experiments/fig_ablation.png)

<!-- 【裁决段·你来写】约 250–300 词，按"被骗 → 醒悟 → 验证"：
1) 稀疏奖励确实可行：纯 ±1 也能学会高度压制，只是慢约 24%；
2) 但日志胜率撒谎：末段 0.716 vs 0.709 看似打平——日志胜率是
   "各自对自身快照池"的相对记账，池子强弱不同就不可比；
3) 循环赛戳穿：66 : 14.5——sparse 组学到的是"对弱对手苟活"，
   遇强即崩；
4) 收束：在自我对弈里，日志胜率是记账单位，不是强度单位；任何想拿
   RL 当裁判的人，先学会怀疑裁判的记分牌。
-->

> [5-verdict to be written]

## 6. Break III — Asymmetric Goals, Rotating Phases, and the Same-Instant Draw

<!-- 【设计段·你来写】约 150 词：
传统抽象棋对称目标；Terraflux 白逃黑捕，胜负条件不同；相位按各自
手数独立轮换（同一回合双方相位可以不同）；平局定义为"同一落子后
黑白同时达边"——胜负同帧发生的设计几乎没有先例（回合制里如何
"同时"，写清那个 X 相位对角触发链的具体例子，查底稿 §5.2）。
引出问题：非对称游戏没有对称性托底——它公平吗？谁来裁决？
-->

> [6-design to be written]

**The RL verdict.** duel 布局（黑镇中央、白正北隔三格、黑先）下，
四种强度策略的白胜率：

**Table 4. White win rate rises with policy strength**

| Policy | Round-robin rank | White win rate (duel self-play) | Mean length |
|---|---|---|---|
| Heuristic vs itself | 4 (weakest) | **0%** | 10 |
| PPO sparse | 3 | **20%** (95% CI ±5.6%) | 59 |
| PPO shaped | 2 | **34%** (±6.5%) | 75 |
| PPO main (500k steps) | 1 (strongest) | **46%** (±6.9%) | 54 |

![Figure 4. Left: white win rate rises monotonically with policy strength toward 50%. Right: round-robin strength matrix](../experiments/fig_strength_fairness.png)

<!-- 【裁决段·你来写】约 300 词，四段：
1) 现象：同一开局，白胜率随裁判强度 0% → 20% → 34% → 46% 严格单调，
   向 50% 的完美平衡爬升；
2) 对照：七种棋盘尺寸（15×15–27×27）扫描全是平线——几何不是
   公平性变量，强度才是；
3) 机制：弱白方的评估里间距项被趋边冲动压倒，事实上不会逃，黑直线
   追击平均 10 步抓死；强策略自学了保持间距与借地形卡位——
   逃跑是门要学的技术，学会多少，开局就公平多少；
4) 收束：规则公平性不能脱离玩家强度定义——"平衡开局"是强度依赖的
   谱，不是布尔属性。对非对称游戏，只有收敛的 RL 策略有资格当裁判；
   这也正是本设计的安身之处：相位轮换与同瞬平局的平衡效果，
   从此可以被测量而不是被争论。
-->

> [6-verdict to be written]

## 7. Discussion — The Mirror-Value Test, Limitations, and Future Mechanisms

<!-- 【你来写】约 250–300 词，三块：
1) 设计哲学——价值镜像检验（全文设计观的收束，约 100 词，素材查
   底稿 §5.5）：非对称追逃里，地形词汇表每个词都必须通过检验
   "对逃方与捕方价值是否大致对称"——传送偏白、冻结偏黑，
   单边武器全被淘汰；高度与通行性是唯二幸存者。
2) 局限性四条，每条 1–2 句（底稿 §4.2）：单随机种子；弱策略仅
   启发式一类代表；强策略尺寸扫描受算力所限；样本 200–300 局、
   95% CI 约 ±6%。
3) 未来（2–3 句）：机制流水线——候选机制（退潮格/地形自主演化/
   潮汐线等，八项清单见底稿 §5.5）各称通过镜像检验，但每一项都须经
   RL 裁判实测才敢入列："想法 → 实现 → RL 测量 → 数据裁决"。
-->

> [Section 7 to be written]

## 8. References and Links

- **Code & full experiment data**: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux)
- **Playable 3D demo (bilingual, with narrated tutorial game)**: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Huang, S. & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. arXiv:2006.14171.
