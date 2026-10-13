"""fig_replicates.py —— 实验④ 数据驱动的三张独立图（按 A4 版心宽度设计，字号以印刷为准）

输入：experiments/replicate_eval.json、experiments/robustness_training.json
输出：
  experiments/fig_curves.png     图3：训练曲线对比（滚动胜率与解释方差随步数变化；细线=单个运行，粗线=组均值）
  experiments/fig_selfplay.png   图4：各训练运行的 duel 自对弈白胜率（水平点图，含 Wilson 95% CI）
  experiments/fig_roundrobin.png 图5：循环赛——上方为各运行平均胜率，下方为 23×23 对战矩阵（无单元格数字，数值见 JSON）
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
                     'font.family': 'DejaVu Sans'})

ev = json.load(open('experiments/replicate_eval.json', encoding='utf-8'))
tr = json.load(open('experiments/robustness_training.json', encoding='utf-8'))

GROUP_COLOR = {'shaped-config': '#d9701c', 'sparse': '#3f5f8a', 'static': '#7a756a'}
GROUP_LABEL = {'shaped-config': 'shaped (step penalty −0.002)', 'sparse': 'sparse (no step penalty)',
               'static': 'static board (terrain frozen)'}
GROUP_ORDER = ['shaped-config', 'sparse', 'static']


def seed_of(name):
    m = re.search(r'_s(\d+)$', name)
    return int(m.group(1)) if m else -1


def run_key(name):
    g = GROUP_ORDER.index(tr['runs'][name]['group'])
    first = {'main': 0, 'abl_shaped': 1, 'abl_sparse': 2}.get(name, 3)
    return (g, first, seed_of(name))


def legend_handles():
    return [Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLOR[g], markersize=7,
                   label=GROUP_LABEL[g]) for g in GROUP_ORDER]


# =============== 图3：自对弈白胜率（水平点图）===============
sp = sorted(ev['selfplay'], key=lambda r: run_key(r['model']))
ypos, y = {}, 0
prev_g = None
for r in sp:
    g = tr['runs'][r['model']]['group']
    if prev_g is not None and g != prev_g:
        y += 0.9                    # 组间空隙
    ypos[r['model']] = y
    y += 1
    prev_g = g

fig, ax = plt.subplots(figsize=(6.6, 8.2))
for r in sp:
    n = r['model']
    g = tr['runs'][n]['group']
    lo, hi = r['white_ci']
    yy = ypos[n]
    ax.errorbar(r['white_rate'] * 100, yy,
                xerr=[[max(0.0, (r['white_rate'] - lo) * 100)], [max(0.0, (hi - r['white_rate']) * 100)]],
                fmt='o', color=GROUP_COLOR[g], ecolor=GROUP_COLOR[g], capsize=2.5, ms=5, lw=1.2)
    if r['draw_rate'] >= 0.5:
        ax.annotate(f"{r['draw_rate']*100:.0f}% draws", (r['white_rate'] * 100, yy),
                    xytext=(16 if r['white_rate'] < 0.5 else -16, 0), textcoords='offset points',
                    fontsize=7, color='#444', va='center', ha='left' if r['white_rate'] < 0.5 else 'right')
ax.axvline(50, color='#333', ls='--', lw=0.9)
ax.set_yticks([ypos[r['model']] for r in sp])
ax.set_yticklabels([r['model'] for r in sp])
ax.invert_yaxis()
ax.set_xlim(-3, 103)
ax.set_xlabel('White win rate in duel self-play (%), 300 games per run; bars = 95% Wilson interval')
ax.grid(axis='x', alpha=0.25)
ax.set_title('White win rate by training run', loc='left')
ax.legend(handles=legend_handles(), loc='upper center', bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig('experiments/fig_selfplay.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('written experiments/fig_selfplay.png')

# =============== 图4：循环赛 ===============
rr = ev['round_robin']
allm = sorted({x['a'] for x in rr} | {x['b'] for x in rr})
avg = {n: (sum(x['a_win'] for x in rr if x['a'] == n) + sum(x['b_win'] for x in rr if x['b'] == n))
       / max(1, sum(1 for x in rr if n in (x['a'], x['b']))) for n in allm}
names = sorted(allm, key=lambda n: -avg[n])
M = np.full((len(names), len(names)), np.nan)
for x in rr:
    i, j = names.index(x['a']), names.index(x['b'])
    M[i, j] = x['a_win'] * 100
    M[j, i] = x['b_win'] * 100

fig = plt.figure(figsize=(6.6, 8.6))
gs = fig.add_gridspec(2, 2, width_ratios=[40, 1.1], height_ratios=[1.0, 4.6], hspace=0.06, wspace=0.03)
axb = fig.add_subplot(gs[0, 0])
axh = fig.add_subplot(gs[1, 0], sharex=axb)
cax = fig.add_subplot(gs[1, 1])
fig.add_subplot(gs[0, 1]).axis('off')

cols = [GROUP_COLOR[tr['runs'][n]['group']] if n in tr['runs'] else '#999' for n in names]
axb.bar(range(len(names)), [avg[n] * 100 for n in names], color=cols, width=0.8)
axb.axhline(50, color='#333', ls='--', lw=0.8)
axb.set_ylim(0, 70)
axb.set_ylabel('mean win %\nvs all others')
axb.tick_params(axis='x', labelbottom=False)
axb.set_title('Round robin among the 23 dynamic-terrain runs (100 games per colour per pair)', loc='left')
axb.legend(handles=[h for h in legend_handles() if h.get_label().startswith(('shaped', 'sparse'))],
           loc='upper right', fontsize=7, frameon=False, ncol=2)

im = axh.imshow(M, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
axh.set_xticks(range(len(names)))
axh.set_yticks(range(len(names)))
axh.set_xticklabels(names, rotation=90, ha='center', fontsize=7)
axh.set_yticklabels(names, fontsize=7)
for lbl, c in zip(axh.get_xticklabels(), cols):
    lbl.set_color(c)
for lbl, c in zip(axh.get_yticklabels(), cols):
    lbl.set_color(c)
axh.set_xticks(np.arange(-.5, len(names), 1), minor=True)
axh.set_yticks(np.arange(-.5, len(names), 1), minor=True)
axh.grid(which='minor', color='white', lw=0.6)
axh.tick_params(which='minor', length=0)
axh.set_xlabel('opponent (column)')
axh.set_ylabel('row run: its win rate against the column run')
cb = fig.colorbar(im, cax=cax)
cb.set_label('row run win % (draws not counted)', fontsize=7)
cb.ax.tick_params(labelsize=7)
fig.savefig('experiments/fig_roundrobin.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('written experiments/fig_roundrobin.png')

# =============== 图3：训练曲线对比 ===============
LOG_FILES = {'main': 'training_log.jsonl', 'abl_shaped': 'training_log_abl_shaped.jsonl',
             'abl_sparse': 'training_log_abl_sparse.jsonl'}


def roll9(xs):
    out = []
    for i in range(len(xs)):
        seg = [x for x in xs[max(0, i - 8):i + 1] if x is not None]
        out.append(sum(seg) / len(seg) if seg else np.nan)
    return np.array(out, dtype=float)


curves = {}
steps_ref = None
for name, info in tr['runs'].items():
    path = LOG_FILES.get(name, f'training_log_{name}.jsonl')
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    st = np.array([r['steps'] for r in rows]) / 1000.0
    if steps_ref is None:
        steps_ref = st
    assert len(st) == len(steps_ref) and np.allclose(st, steps_ref), f'step grid differs for {name}'
    curves[name] = (roll9([r['win_rate'] for r in rows]), roll9([r['expl_var'] for r in rows]), info['group'])

fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True)
panels = [(0, 'rolling win rate against the run\'s own opponent pool', (0.0, 1.0)),
          (1, 'value-function explained variance', (-0.1, 1.05))]
for k, ylabel, ylim in panels:
    ax = axes[k]
    for g in GROUP_ORDER:
        members = [c for c in curves.values() if c[2] == g]
        for c in members:
            ax.plot(steps_ref, c[k], color=GROUP_COLOR[g], lw=0.6, alpha=0.22)
        mean = np.nanmean(np.vstack([c[k] for c in members]), axis=0)
        ax.plot(steps_ref, mean, color=GROUP_COLOR[g], lw=2.4, label=f"{GROUP_LABEL[g]}, n = {len(members)}")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, steps_ref[-1])
axes[0].set_title('(a) Rolling win rate (thin: single runs; thick: group mean)', loc='left')
axes[1].set_title('(b) Explained variance (thin: single runs; thick: group mean)', loc='left')
axes[1].set_xlabel('training steps (thousands)')
axes[0].legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=7.5)
fig.tight_layout()
fig.savefig('experiments/fig_curves.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('written experiments/fig_curves.png')
