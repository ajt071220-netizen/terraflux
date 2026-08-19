"""analyze_ablation.py —— 实验①：奖励整形消融对比

读 training_log_abl_shaped.jsonl（有步罚 -0.002）与 training_log_abl_sparse.jsonl（无步罚），
输出三联对比图 → experiments/fig_ablation.png：
1. win_rate：稀疏组学习是否更慢/更不稳（核心问题：整形是否加速学习）
2. ep_len：步罚的直接目标——局长是否被压缩（稀疏组预期拖长）
3. entropy：探索衰减节奏差异

注意：ep_rew 不可直接对比（步罚本身改变回报数值，属平凡差异），故不画。
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    return [r['steps'] / 1000 for r in rows], rows


def rolling(xs, w=9):
    out = []
    for i in range(len(xs)):
        seg = [x for x in xs[max(0, i - w + 1):i + 1] if x is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


s_steps, s_rows = load('training_log_abl_shaped.jsonl')
p_steps, p_rows = load('training_log_abl_sparse.jsonl')

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.suptitle('Reward shaping ablation: step penalty −0.002 vs none (50k-step rolling window)', fontsize=12)

for ax, key, title in [
    (axes[0], 'win_rate', 'Win rate vs opponent pool'),
    (axes[1], 'ep_len', 'Episode length (moves)'),
    (axes[2], 'entropy', 'Policy entropy (nats)'),
]:
    ax.plot(s_steps, rolling([r.get(key) for r in s_rows]), color='#e07020', lw=1.8, label='shaped (−0.002/step)')
    ax.plot(p_steps, rolling([r.get(key) for r in p_rows]), color='#5a6a80', lw=1.8, label='sparse (no penalty)')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('steps (k)')
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig('experiments/fig_ablation.png', dpi=140)
print('[图] experiments/fig_ablation.png')

# 数字摘要：达到 55%/65% 胜率各需多少步（学习效率指标）
def steps_to(rows, thr):
    sm = rolling([r['win_rate'] for r in rows])
    for r, v in zip(rows, sm):
        if v is not None and v >= thr:
            return r['steps'] // 1000
    return None

for thr in (0.55, 0.65):
    a, b = steps_to(s_rows, thr), steps_to(p_rows, thr)
    print(f'达到滚动胜率 {thr:.0%}: shaped={a}k  sparse={b}k')

for name, rows in [('shaped', s_rows), ('sparse', p_rows)]:
    tail = rows[-40:]
    wl = [r['win_rate'] for r in tail if r.get('win_rate') is not None]
    el = [r['ep_len'] for r in tail if r.get('ep_len') is not None]
    print(f'[{name}] 末段 win {sum(wl)/len(wl):.3f}  ep_len {sum(el)/len(el):.1f}')
