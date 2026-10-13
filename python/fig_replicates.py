"""fig_replicates.py —— 实验④ 数据驱动的汇总图

输入：experiments/replicate_eval.json（自对弈 + 循环赛）、experiments/robustness_training.json（训练统计）
输出：experiments/fig_replicates.png
  上：各训练运行的 duel 自对弈白胜率（Wilson 95% CI），按配置分组；50% 虚线
  中：动态棋盘模型两两循环赛（行模型对列模型的胜率，平局不计入胜率）
  下：训练后期（150k 步之后）的解释方差均值 与 胜率标准差，按配置着色（不逐点标注，避免重叠）
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ev = json.load(open('experiments/replicate_eval.json', encoding='utf-8'))
tr = json.load(open('experiments/robustness_training.json', encoding='utf-8'))

GROUP_COLOR = {'shaped-config': '#e07020', 'sparse': '#5a6a80', 'static': '#8a8578'}
GROUP_ORDER = ['shaped-config', 'sparse', 'static']

# ---- 数据顺序：按配置分组，组内按名称 ----
sp = ev['selfplay']
sp_sorted = sorted(sp, key=lambda r: (GROUP_ORDER.index(tr['runs'][r['model']]['group']), r['model']))

fig = plt.figure(figsize=(7.2, 11.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.45, 0.9], hspace=0.5)

# ---- 上：自对弈白胜率 ----
ax = fig.add_subplot(gs[0])
for i, r in enumerate(sp_sorted):
    g = tr['runs'][r['model']]['group']
    lo, hi = r['white_ci']
    ax.errorbar(i, r['white_rate'] * 100,
                yerr=[[max(0.0, (r['white_rate'] - lo) * 100)], [max(0.0, (hi - r['white_rate']) * 100)]],
                fmt='o', color=GROUP_COLOR[g], ecolor=GROUP_COLOR[g], capsize=3, ms=6)
ax.axhline(50, color='#333', ls='--', lw=0.9)
ax.set_xticks(range(len(sp_sorted)))
ax.set_xticklabels([r['model'] for r in sp_sorted], rotation=90, ha='center', fontsize=6)
ax.set_xlim(-0.7, len(sp_sorted) - 0.3)
ax.set_ylabel('White win rate (%)', fontsize=7)
ax.set_ylim(-3, 100)
ax.set_title('(a) Duel self-play, 300 games per run', fontsize=8.5)
ax.grid(alpha=0.25, axis='y')
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLOR[g], ms=7, label=g)
                   for g in GROUP_ORDER], fontsize=6.5, loc='upper right', ncol=3, frameon=False)

# ---- 中：循环赛矩阵 ----
ax2 = fig.add_subplot(gs[1])
_all = sorted({x['a'] for x in ev['round_robin']} | {x['b'] for x in ev['round_robin']})
_avg = {n: (sum(x['a_win'] for x in ev['round_robin'] if x['a'] == n) + sum(x['b_win'] for x in ev['round_robin'] if x['b'] == n))
        / max(1, sum(1 for x in ev['round_robin'] if n in (x['a'], x['b']))) for n in _all}
names = sorted(_all, key=lambda n: -_avg[n])
M = np.full((len(names), len(names)), np.nan)
for x in ev['round_robin']:
    i, j = names.index(x['a']), names.index(x['b'])
    M[i, j] = x['a_win'] * 100
    M[j, i] = x['b_win'] * 100
im = ax2.imshow(M, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
for i in range(len(names)):
    for j in range(len(names)):
        if i == j or np.isnan(M[i, j]):
            continue
        ax2.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=4.2,
                 color='white' if abs(M[i, j] - 50) > 28 else 'black')
ax2.set_xticks(range(len(names)))
ax2.set_yticks(range(len(names)))
ax2.set_xticklabels(names, rotation=90, ha='center', fontsize=6)
ax2.set_yticklabels(names, fontsize=6)
ax2.set_xlabel('opponent (column)')
ax2.set_ylabel('row model')
ax2.set_title('(b) Round robin, row model win % (100 games per colour per pair)', fontsize=8.5)
cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.01)
cb.set_label('row win %', fontsize=6); cb.ax.tick_params(labelsize=6)

# ---- 下：训练统计（无逐点文字，按配置着色） ----
ax3 = fig.add_subplot(gs[2])
for n, r in tr['runs'].items():
    ax3.scatter(r['expl_var_mean_after150k'], r['win_sd_after150k'], s=60,
                color=GROUP_COLOR[r['group']], edgecolor='k', lw=0.4, zorder=3)
ax3.set_xlabel('explained variance, mean after 150k steps', fontsize=7)
ax3.set_ylabel('SD of rolling win rate\nafter 150k steps')
ax3.set_title('(c) Training statistics per run', fontsize=8.5)
ax3.grid(alpha=0.25)
ax3.set_xlim(0.45, 1.05)
ax3.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLOR[g], ms=7, label=g)
                    for g in GROUP_ORDER], fontsize=6.5, loc='upper left', frameon=False)

fig.suptitle('Terraflux replicate study: outcomes vary by training run more than by configuration', fontsize=9, y=0.995)
out = 'experiments/fig_replicates.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('written', out)
