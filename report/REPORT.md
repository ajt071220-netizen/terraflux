# Terraflux (涌陆): A Board Game Where the Board Plays Back

**Design of an original pursuit-evasion game on a self-modifying board — and an experiment on what "fairness" means inside it**

*Your Name · August 2026*
*Code: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) · Playable demo: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)*

<!-- ============================================================
写作总纲（写完请删除所有此类注释块）

报告主线：我设计了一个"棋盘会自我改写"的棋，三个维度打破棋类
几百年的默认设定（§2，全文主菜）；这个独特对象逼出了一个传统
棋类不存在的问题——"公平"在这里无法被人工定义；于是训练 RL 裁判
来测量（§3），实验发现公平性是随玩家强度爬升的光谱（§4，核心发现）。

目标篇幅：4–5 页 PDF（约 1500–1700 词正文 + 2–3 图 2 表）。
语气：第一人称、主动语态。
数字纪律：所有数字已预填自 experiments/REPORT_DATA.md，【不要手改】。
所有机制细节（规则/训练配置）查 report/写作底稿.md。
============================================================ -->

## Abstract

<!-- 【你来写】≤150 词，三段式：
第 1 段（2 句）：棋类几百年来默认"棋盘是静止的舞台"；我设计的
  Terraflux 让每步落子都改写地形——三个机制突破各半句带过。
第 2 段（2 句）：动态地形 + 非对称追逃让"这游戏公平吗"无法人工回答；
  我用 PPO 自我对弈训练出不同强度的 AI 当裁判。
第 3 段（2 句）：核心发现——同一开局，白胜率随策略强度 0%→46%
  单调爬升；"平衡"不是布局的固有属性，而是强度依赖的光谱。
-->

> [Abstract to be written]

## 1. Introduction — Boards That Don't Sit Still

<!-- 【你来写】约 200–250 词：
1) 钩子（1–2 句）：围棋下在永恒的 19×19 木盘上，象棋的九宫格千年
   不动——几百年来棋类的默认设定是"棋盘是舞台，棋子是演员"。
   如果舞台本身也是演员呢？
2) 谱系（2–3 句）：棋盘并非从未变化——Zertz 的板块会拆除收缩、
   Dvonn 把高度做进棋子堆叠——但"落子动作本身持续改写地形"
   没有先例（见 Table 1）。
3) 引出（2 句）：本文先讲 Terraflux 的三个设计突破（§2），再讲它们
   逼出的新问题，以及我用强化学习给出的定量答案（§3–4）。
-->

> [Section 1 to be written]

![Figure 1. The playable 3D simulator](../experiments/fig_size_fairness.png)
<!-- 占位图：换一张网站截图（3D 棋盘 + 演示解说条），保存为
     report/fig_screenshot.png 并替换上面的路径。 -->

## 2. Design — Three Breaks from Tradition

<!-- 【你来写】全文主菜，约 450 词。三个小节各约 150 词，固定写法：
这机制是什么（2 句）→ 传统棋类为什么从不这样做（1 句）→
它改变了什么（1–2 句）。规则细节全部查 report/写作底稿.md §1.1。 -->

### 2.1 The Fluid Board

<!-- 要点：25×25 柱阵；每格三态循环（隆起 +1 / 下陷 −1 / 填平不可进入）；
每次落子按当前相位改写落点周围（十字相位：北隆、南陷、东西填平；
斜向相位：东北西北隆、东南西南陷）。关键强调：填平不是永久的——
后续相位可使其复活，地形是流体而非消耗品。围棋里"金角银边"是永恒
真理；这里只有暂时的好位置。 -->

> [2.1 to be written]

### 2.2 Height Belongs to the Land

<!-- 要点：高度不是棋子的属性（Dvonn 的堆叠是棋子叠棋子），而是地形的
属性——棋子不变，地把它顶起来。抓捕条件是"相邻 + 黑严格高于白"：
吃子不再是"移动到你所在的格子"，而是"居高临下"。这制造了高度
军备竞赛：你站高一步，我就得站得更高。 -->

> [2.2 to be written]

### 2.3 Asymmetric Goals, Alternating Physics, Simultaneous Fates

<!-- 要点：白逃向南边线、黑追缉——双方目标根本不同（传统棋类目标对称）；
地形改写规则按各白手数在十字/斜向相位间轮换，等于物理定律每手换
一次；最罕见的是同瞬平局——白踏上胜利边的一步若恰好同时满足抓捕
条件，逃脱与死亡在同一瞬间发生，判平。棋类里几乎没有第二种
"胜负同瞬"的设计。 -->

> [2.3 to be written]

**Table 1. Terraflux in the design space of classic abstract games**

| Dimension | Go / Chess | Zertz | Dvonn | **Terraflux** |
|---|---|---|---|---|
| Board | Static forever | Shrinks (tiles removed) | Static | **Rewritten by every move** |
| Height | — | — | Stacked pieces | **Terrain lifts the piece** |
| Objective | Symmetric | Symmetric | Symmetric | **Asymmetric (flee vs. hunt)** |
| Capture | Move onto piece | Jump over | Surround | **Adjacent + strictly higher** |

## 3. Methods — Manufacturing Referees

<!-- 【你来写】约 200–250 词。这一节回答一个问题：这三个机制凑在一起，
"这游戏公平吗"为什么没法人工回答，以及我怎么造出"裁判"。
1) 为什么难（3 句）：非对称追逃意味着双方的目标函数根本不同，
   没有"对杀定式"可循；地形每步流动，推演两三步后局面面目全非；
   新游戏没有人类高手——"最佳棋"无人知晓。
2) 造裁判（4–5 句）：MaskablePPO 自我对弈（观测 = 地形张量+位置+
   8 维元信息；动作 = 8 方向+停步；奖励 = 终局 ±1；对手池 = 65% 启发式
   + 35% 自身历史快照；50 万步）。通过控制训练量与奖励设计，
   得到四个强度分明的裁判，循环赛（两两互打 200 局）确定强弱排名。
3) 评估协议（1–2 句）：同一开局让不同强度的裁判自对弈 200–300 局，
   报 95% 置信区间。
-->

> [Section 3 to be written]

## 4. The Experiment — Fairness Is Strength-Conditional

duel 布局（黑镇中央、白正北隔三格、黑先）下，四种强度策略的白胜率：

**Table 2. White win rate rises with policy strength**

| Policy | Round-robin rank | White win rate (duel self-play) | Mean length |
|---|---|---|---|
| Heuristic vs itself | 4 (weakest) | **0%** | 10 |
| PPO sparse | 3 | **20%** (95% CI ±5.6%) | 59 |
| PPO shaped | 2 | **34%** (±6.5%) | 75 |
| PPO main (500k steps) | 1 (strongest) | **46%** (±6.9%) | 54 |

![Figure 2. Left: white win rate rises monotonically with policy strength toward 50%. Right: round-robin strength matrix](../experiments/fig_strength_fairness.png)

<!-- 【你来写】约 300–350 词，四段结构：
1) 现象：同一开局，白胜率随裁判强度 0% → 20% → 34% → 46% 严格单调，
   向 50% 的完美平衡爬升；
2) 对照：七种棋盘尺寸（15×15–27×27）扫描全是平线——几何不是
   公平性变量，强度才是；
3) 机制：弱白方的评估里间距项被趋边冲动压倒，事实上不会逃，黑直线
   追击平均 10 步抓死；强策略在残酷训练中自学了保持间距与借地形
   卡位——逃跑是门要学的技术，学会多少，开局就公平多少；
4) 论点：规则公平性不能脱离玩家强度来定义；"平衡开局"是强度依赖的
   谱而非布尔属性；只有收敛的 RL 策略有资格当公平性的裁判——
   这正是 §2 三个机制凑出来的全新问题：在静态对称棋里，
   "公平"从不需要这样被追问。
-->

> [Section 4 narrative to be written]

## 5. Discussion — Limitations and Future Work

<!-- 局限性四条，每条 1–2 句英文改写（素材在写作底稿 §4.2）：
单随机种子；弱策略仅启发式一类代表；强策略尺寸扫描受算力所限；
样本 200–300 局、95% CI 约 ±6%。
未来工作挑 2 条：多种子复核；强策略下的尺寸扫描；LLM 智能体作为
第三类强度样本；以及机制向的开放问题——当环境本身被双方动作持续
改写时，策略学习会发生什么（呼应 §2，告诉读者你看到的问题比解决的
多）。
（训练方法学的两项附加分析——奖励整形消融、自我对弈训练动态——
数据完整保留在仓库 experiments/REPORT_DATA.md，可在本节点一句带过。）
-->

> [Section 5 to be written]

## 6. References and Links

- **Code & full experiment data**: [github.com/ajt071220-netizen/terraflux](https://github.com/ajt071220-netizen/terraflux) — including two supplementary training-methodology analyses ([experiments/REPORT_DATA.md](https://github.com/ajt071220-netizen/terraflux/blob/main/experiments/REPORT_DATA.md))
- **Playable 3D demo (bilingual, with narrated tutorial game)**: [ajt071220-netizen.github.io/terraflux](https://ajt071220-netizen.github.io/terraflux/)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268).
- Huang, S. & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. arXiv:2006.14171.
