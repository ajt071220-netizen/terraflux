"""analyze_collapse.py —— 实验②：PPO 自我对弈训练动态解剖

读 training_log.jsonl（rollout 粒度指标），输出：
1. 四联学习曲线图（胜率/局长/熵/解释方差）→ experiments/fig_training_dynamics.png
2. 不稳定性事件清单：胜率骤降 >0.25、KL 尖峰 >0.15、熵跳变
3. 终端打印摘要统计（供报告引用）
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = sys.argv[1] if len(sys.argv) > 1 else 'training_log.jsonl'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'experiments/fig_training_dynamics.png'

rows = [json.loads(l) for l in open(LOG, encoding='utf-8')]
steps = [r['steps'] / 1000 for r in rows]

def col(k):
    return [r.get(k) for r in rows]

def rolling(xs, w=7):
    out = []
    for i in range(len(xs)):
        seg = [x for x in xs[max(0, i - w + 1):i + 1] if x is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
fig.suptitle(f'PPO Self-Play Training Dynamics ({rows[-1]["steps"]//1000}k steps, 25×25 duel)', fontsize=13)

panels = [
    ('win_rate', 'Win rate (rolling-100 games)', (0, 1)),
    ('ep_len', 'Episode length (moves)', None),
    ('entropy', 'Policy entropy (nats)', None),
    ('expl_var', 'Value fn explained variance', (-0.05, 1.05)),
]
for ax, (key, title, ylim) in zip(axes.flat, panels):
    raw, sm = col(key), rolling(col(key))
    ax.plot(steps, raw, color='#c9a86a', alpha=0.28, lw=0.7)
    ax.plot(steps, sm, color='#e07020', lw=1.8, label='rolling-7')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('steps (k)')
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=140)
print(f'[图] {OUT}')

# ---- 不稳定性事件检测 ----
wr = col('win_rate')
kl = col('approx_kl')
ent = col('entropy')

events = []
for i in range(1, len(rows)):
    if wr[i] is not None and wr[i - 1] is not None and wr[i] - wr[i - 1] <= -0.25:
        events.append((rows[i]['steps'], f'win_rate 骤降 {wr[i-1]:.2f}→{wr[i]:.2f}'))
    if kl[i] is not None and kl[i] > 0.15:
        events.append((rows[i]['steps'], f'KL 尖峰 {kl[i]:.3f}'))
    if ent[i] is not None and ent[i - 1] is not None and abs(ent[i] - ent[i - 1]) > 0.25:
        events.append((rows[i]['steps'], f'entropy 跳变 {ent[i-1]:.2f}→{ent[i]:.2f}'))

print(f'\n[不稳定性事件] 共 {len(events)} 次')
for s, msg in events[:20]:
    print(f'  @{s//1000}k  {msg}')
if len(events) > 20:
    print(f'  … 其余 {len(events)-20} 次省略')

# ---- 阶段摘要 ----
third = len(rows) // 3
for name, seg in [('早期', rows[:third]), ('中期', rows[third:2 * third]), ('后期', rows[2 * third:])]:
    w = [r['win_rate'] for r in seg if r.get('win_rate') is not None]
    e = [r['entropy'] for r in seg if r.get('entropy') is not None]
    l = [r['ep_len'] for r in seg if r.get('ep_len') is not None]
    print(f'[{name}] steps {seg[0]["steps"]//1000}k–{seg[-1]["steps"]//1000}k | '
          f'win {sum(w)/len(w):.3f} | entropy {sum(e)/len(e):.3f} | ep_len {sum(l)/len(l):.1f}')
