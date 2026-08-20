"""fig_strength_fairness.py —— 实验③汇总图：策略强度 × duel 布局公平性

左图：四种策略 duel 自对弈白胜率（按循环赛强度排序，带 95% CI）
右图：四策略对局矩阵（行胜率）
数据源：eval_selfplay / scan_sizes / model_vs_model 实测结果（硬编码自本轮实验输出）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 实测数据（200 局自对弈 / 循环赛各 200 局）
NAMES = ['Heuristic', 'PPO sparse', 'PPO shaped', 'PPO main']
WHITE_WR = [0.0, 20.0, 34.0, 46.0]          # duel 自对弈白胜率 %
N_GAMES = [300, 200, 200, 200]

# 对局矩阵：行策略对列策略的胜率 %（平局按 0.5 折算）；对角线=自对弈白胜率无意义，置 NaN
# 行: 0=启发式 1=sparse 2=shaped 3=blockade(ppo_25)
M = np.array([
    [np.nan, 18.0, 3.0, 21.0],     # 启发式 vs 三 PPO（eval_vs 折算）
    [82.0, np.nan, 14.5, 37.5],    # sparse
    [97.0, 66.0, np.nan, 31.5],    # shaped
    [79.0, 43.0, 53.5, np.nan],    # blockade
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# 左：强度排序后的 duel 白胜率
x = np.arange(4)
colors = ['#8a8578', '#5a6a80', '#e07020', '#a04030']
ax1.bar(x, WHITE_WR, color=colors, width=0.58)
for i, (wr, n) in enumerate(zip(WHITE_WR, N_GAMES)):
    se = 100 * (wr / 100 * (1 - wr / 100) / n) ** 0.5 * 1.96
    ax1.errorbar(i, wr, yerr=se, fmt='none', ecolor='#333', capsize=4, lw=1.2)
    ax1.text(i, wr + se + 1.2, f'{wr:.0f}%', ha='center', fontsize=10)
ax1.axhline(50, color='#888', ls='--', lw=1, alpha=0.6)
ax1.text(-0.45, 52.5, 'perfectly balanced (50%)', fontsize=8, color='#666', ha='left')
ax1.set_xticks(x)
ax1.set_xticklabels(NAMES, fontsize=9)
ax1.set_ylabel('White win rate in duel self-play (%)')
ax1.set_title('Fairness is strength-conditional\n(duel gap=3, black first; ordered by round-robin strength →)', fontsize=10)
ax1.set_ylim(0, 68)
ax1.grid(axis='y', alpha=0.25)

# 右：对局矩阵热图
im = ax2.imshow(M, cmap='RdYlGn', vmin=0, vmax=100)
ax2.set_xticks(range(4))
ax2.set_yticks(range(4))
ax2.set_xticklabels(NAMES, fontsize=8, rotation=18)
ax2.set_yticklabels(NAMES, fontsize=8)
ax2.set_xlabel('Opponent')
ax2.set_title('Round-robin matrix: row win rate vs column (%)', fontsize=10)
for i in range(4):
    for j in range(4):
        if not np.isnan(M[i, j]):
            ax2.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=9,
                     color='black' if 25 < M[i, j] < 75 else 'white')
fig.colorbar(im, ax=ax2, shrink=0.85)
fig.tight_layout()
fig.savefig('experiments/fig_strength_fairness.png', dpi=140)
print('[图] experiments/fig_strength_fairness.png')
