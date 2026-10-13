"""fig_replicates.py —— 实验④ 数据驱动的汇总图（替代手写数字的旧 fig_strength_fairness.py）

输入：experiments/replicate_eval.json（自对弈 + 循环赛）、experiments/robustness_training.json（训练统计）
输出：experiments/fig_replicates.png
  左：各训练运行的 duel 自对弈白胜率（含 Wilson 95% CI），按配置分组；50% 虚线
  中：动态棋盘模型两两循环赛（行模型对列模型的胜率，平局单独不计入胜率）
  右：训练后期（150k 步之后）的解释方差均值与胜率标准差，按运行
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ev = json.load(open('experiments/replicate_eval.json', encoding='utf-8'))
tr = json.load(open('experiments/robustness_training.json', encoding='utf-8'))

GROUP_COLOR = {'shaped-config': '#e07020', 'sparse': '#5a6a80', 'static': '#8a8578'}
GROUP_ORDER = ['shaped-config', 'sparse', 'static']

sp = ev['selfplay']
sp_sorted = sorted(sp, key=lambda r: (GROUP_ORDER.index(tr['runs'][r['model']]['group']), r['model']))

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.55, wspace=0.28)

# ---- 左：自对弈白胜率 ----
ax = fig.add_subplot(gs[0, 0])
for i, r in enumerate(sp_sorted):
    g = tr['runs'][r['model']]['group']
    lo, hi = r['white_ci']
    ax.errorbar(i, r['white_rate'] * 100, yerr=[[max(0.0, (r['white_rate'] - lo) * 100)], [max(0.0, (hi - r['white_rate']) * 100)]],
                fmt='o', color=GROUP_COLOR[g], ecolor=GROUP_COLOR[g], capsize=3, ms=6)
ax.axhline(50, color='#333', ls='--', lw=0.9)
ax.set_xticks(range(len(sp_sorted)))
ax.set_xticklabels([r['model'] for r in sp_sorted], rotation=60, ha='right', fontsize=8)
ax.set_ylabel('White win rate, duel self-play (%)')
ax.set_ylim(-3, 105)
ax.set_title('(a) Same layout, same reward, different training runs', fontsize=10)
ax.grid(alpha=0.25, axis='y')
for g in GROUP_ORDER:
    ax.plot([], [], 'o', color=GROUP_COLOR[g], label=g)
ax.legend(fontsize=8, loc='center right')

# ---- 中：循环赛矩阵 ----
ax2 = fig.add_subplot(gs[1, :])
_all = sorted({x['a'] for x in ev['round_robin']} | {x['b'] for x in ev['round_robin']})
_avg = {n: (sum(x['a_win'] for x in ev['round_robin'] if x['a'] == n) + sum(x['b_win'] for x in ev['round_robin'] if x['b'] == n)) / max(1, sum(1 for x in ev['round_robin'] if n in (x['a'], x['b']))) for n in _all}
names = sorted(_all, key=lambda n: -_avg[n])
M = np.full((len(names), len(names)), np.nan)
D = np.full((len(names), len(names)), np.nan)
for x in ev['round_robin']:
    i, j = names.index(x['a']), names.index(x['b'])
    M[i, j] = x['a_win'] * 100
    M[j, i] = x['b_win'] * 100
    D[i, j] = D[j, i] = x['draw'] * 100
im = ax2.imshow(M, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
for i in range(len(names)):
    for j in range(len(names)):
        if i == j:
            continue
        ax2.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=8,
                 color='white' if abs(M[i, j] - 50) > 28 else 'black')
ax2.set_xticks(range(len(names)))
ax2.set_yticks(range(len(names)))
ax2.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
ax2.set_yticklabels(names, fontsize=8)
ax2.set_xlabel('opponent (column)')
ax2.set_ylabel('row model')
ax2.set_title('(b) Round robin, row model win % (draws excluded, 100 games/colour)', fontsize=10)
cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
cb.set_label('row win %', fontsize=8)

# ---- 右：训练统计 ----
ax3 = fig.add_subplot(gs[0, 1])
runs = [n for n in tr['runs']]
for i, n in enumerate(runs):
    r = tr['runs'][n]
    ax3.scatter(r['expl_var_mean_after150k'], r['win_sd_after150k'], s=55,
                color=GROUP_COLOR[r['group']], edgecolor='k', lw=0.4, zorder=3)
    ax3.annotate(n, (r['expl_var_mean_after150k'], r['win_sd_after150k']), fontsize=7,
                 xytext=(4, 3), textcoords='offset points')
ax3.set_xlabel('explained variance, mean after 150k steps')
ax3.set_ylabel('SD of rolling win rate after 150k')
ax3.set_title('(c) Training stats per run', fontsize=10)
ax3.grid(alpha=0.25)
ax3.set_xlim(0.4, 1.05)

fig.suptitle('Terraflux replicate study: outcomes vary by training run more than by configuration', fontsize=12)
out = 'experiments/fig_replicates.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print('written', out)
