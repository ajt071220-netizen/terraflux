# Erratum to the August 2026 report (v1) — Terraflux

**English**

The v1 report (archived as v1.0.0, DOI 10.5281/zenodo.16898141) contains results that do not hold. The corrected report is `REPORT.md` (v2).

1. **Reward bug.** The self-play training environment hard-coded the per-move step penalty as −0.002, so the `--step-penalty` setting had no effect during training. The "sparse" arm of the reward-shaping ablation was therefore trained with the same reward as the "shaped" arm. The claims that sparse training is ~24% slower, that logged win rates "lie", and the 66%:14.5% head-to-head as evidence about reward shaping are **withdrawn**.
2. **Run-to-run variance.** Training runs with identical settings differ widely in the White win rate on the duel layout (about 9% to 76% in our replicates). The monotone "0% → 20% → 34% → 46%" fairness ladder with policy strength, and the balance of the duel layout at 49:50, are **not supported**.
3. **Static-board control.** Freezing the terrain changes the game's balance (White wins almost every game in some seeds; repetition draws dominate in others), so it does not isolate the cause of training oscillation. The claim that terrain causes non-convergence is **not established**.

What remains: training curves are noisy; the heuristic opponent loses 100% of duel games as White (re-run: 0.0%, 95% CI 0.0–1.3%); the board-size scan (heuristic only) shows no size effect for that weak policy.

The v1 code and logs are retained in the repository's history. The corrected code is in `python/train_ppo.py` (the penalty is read from the configuration).

**中文**

v1 报告（2026 年 8 月，Zenodo v1.0.0，DOI 10.5281/zenodo.16898141）中有以下结论不成立，更正版见 `REPORT.md`（v2）：

1. **奖励 bug。** 训练环境硬编码了每步 −0.002 的步罚，`--step-penalty` 在训练中不生效。消融中的 sparse 组实际与 shaped 组奖励相同。因此"奖励整形使学习慢约 24%"、"日志胜率会骗人"、以及"循环赛 66%:14.5%"作为奖励整形证据的说法**全部撤回**。
2. **运行间随机差异。** 设置相同的训练运行，其 duel 布局上的白胜率差异很大（复现中约 9% 至 76%）。因此"公平性随策略强度单调上升（0%→20%→34%→46%）"以及"布局平衡于 49:50"**不成立**。
3. **静止棋盘对照。** 冻结地形改变了游戏平衡（有的种子白方几乎全胜，有的以重复和棋为主），不能隔离训练震荡的原因。"地形导致训练不收敛"的说法**尚未证实**。

仍然成立的：训练曲线有噪声；启发式对手作白方在 duel 中 0% 获胜（重跑结果 0.0%，95% CI 0.0–1.3%）；仅针对启发式的板宽扫描未见尺寸效应（该弱策略下）。

代码与日志保留在仓库历史中；修正后的训练代码读取配置中的步罚（`python/train_ppo.py`）。
