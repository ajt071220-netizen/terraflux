"""analyze_robustness.py —— 实验④：训练日志的复现统计（多种子 + 静止棋盘对照）

对每个训练运行计算：
  - 达到滚动胜率 55% / 65% 的步数（两种平滑口径：9 点滚动均值 = analyze_ablation.py；
    以及原始日志值，即 100 局滚动窗口）
  - 末段胜率（最后 40 个采样点）与 ≥333k 步的均值
  - 150k 步之后胜率的标准差（非平稳/震荡的度量）
  - 解释方差：150k 步之后的均值与最大值
  - 熵跳变次数（|Δ熵|>0.25）与 KL 尖峰次数（>0.15）
  - 平局率均值

分组比较使用置换检验（双侧，10,000 次）。输出：experiments/robustness_training.json 与终端表格。
注意：日志中的 entropy 字段为 SB3 的 entropy_loss（= −熵），本脚本取负还原为正的策略熵。
"""
import json
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

RUNS = {
    # 名称: (日志文件, 分组, 备注)
    'main':       ('training_log.jsonl',               'shaped-config', '原主训练（未记录种子）'),
    'abl_shaped': ('training_log_abl_shaped.jsonl',    'shaped-config', '原消融 shaped（未记录种子）'),
    'abl_sparse': ('training_log_abl_sparse.jsonl',    'shaped-config', '原消融 sparse：实际奖励与 shaped 相同（bug），作同配置重复'),
    'shaped_s2':  ('training_log_shaped_s2.jsonl',     'shaped-config', 'shaped seed 2'),
    'shaped_s3':  ('training_log_shaped_s3.jsonl',     'shaped-config', 'shaped seed 3'),
    'sparse_s1':  ('training_log_sparse_s1.jsonl',     'sparse',        'sparse seed 1（修复后）'),
    'sparse_s2':  ('training_log_sparse_s2.jsonl',     'sparse',        'sparse seed 2（修复后）'),
    'sparse_s3':  ('training_log_sparse_s3.jsonl',     'sparse',        'sparse seed 3（修复后）'),
    'static_s1':  ('training_log_static_s1.jsonl',     'static',        '静止棋盘 seed 1'),
    'static_s2':  ('training_log_static_s2.jsonl',     'static',        '静止棋盘 seed 2'),
    'static_s3':  ('training_log_static_s3.jsonl',     'static',        '静止棋盘 seed 3'),
    'shaped_s4':  ('training_log_shaped_s4.jsonl',     'shaped-config', 'shaped seed 4（批次二）'),
    'shaped_s5':  ('training_log_shaped_s5.jsonl',     'shaped-config', 'shaped seed 5（批次二）'),
    'sparse_s4':  ('training_log_sparse_s4.jsonl',     'sparse',        'sparse seed 4（批次二）'),
    'sparse_s5':  ('training_log_sparse_s5.jsonl',     'sparse',        'sparse seed 5（批次二）'),
    'sparse_s6':  ('training_log_sparse_s6.jsonl',     'sparse',        'sparse seed 6（批次二）'),
    'static_s4':  ('training_log_static_s4.jsonl',     'static',        '静止棋盘 seed 4（批次二）'),
    'static_s5':  ('training_log_static_s5.jsonl',     'static',        '静止棋盘 seed 5（批次二）'),
    'static_s6':  ('training_log_static_s6.jsonl',     'static',        '静止棋盘 seed 6（批次二）'),
}


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def rolling9(xs):
    out = []
    for i in range(len(xs)):
        seg = [x for x in xs[max(0, i - 8):i + 1] if x is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


def steps_to(rows, thr, smooth):
    vals = [r['win_rate'] for r in rows]
    if smooth:
        vals = rolling9(vals)
    for r, v in zip(rows, vals):
        if v is not None and v >= thr:
            return r['steps']
    return None


def stats_for(rows):
    late40 = [r['win_rate'] for r in rows[-40:]]
    late_ge333 = [r['win_rate'] for r in rows if r['steps'] >= 333000]
    tail = [r for r in rows if r['steps'] >= 150000]
    wr_tail = [r['win_rate'] for r in tail]
    ev = [r['expl_var'] for r in tail if r.get('expl_var') is not None]
    ent = [(-r['entropy']) for r in rows if r.get('entropy') is not None]
    ent_jumps = sum(1 for a, b in zip(ent, ent[1:]) if abs(b - a) > 0.25)
    kl = [r['approx_kl'] for r in rows if r.get('approx_kl') is not None]
    kl_spikes = sum(1 for k in kl if k > 0.15)
    return dict(
        n_rows=len(rows), final_steps=rows[-1]['steps'],
        steps_to_55_smooth9=steps_to(rows, 0.55, True),
        steps_to_65_smooth9=steps_to(rows, 0.65, True),
        steps_to_55_raw=steps_to(rows, 0.55, False),
        steps_to_65_raw=steps_to(rows, 0.65, False),
        late_win_last40=statistics.mean(late40),
        late_win_ge333k=statistics.mean(late_ge333),
        win_sd_after150k=statistics.pstdev(wr_tail),
        win_max=max(r['win_rate'] for r in rows),
        expl_var_mean_after150k=statistics.mean(ev),
        expl_var_max_after150k=max(ev),
        entropy_jumps_gt025=ent_jumps,
        kl_spikes_gt015=kl_spikes,
        draw_rate_mean=statistics.mean(r['draw_rate'] for r in rows),
    )


def perm_test(a, b, n=10000, seed=0):
    """双侧置换检验：两组均值差的绝对值。"""
    rng = random.Random(seed)
    obs = abs(statistics.mean(a) - statistics.mean(b))
    pooled = list(a) + list(b)
    k = len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        d = abs(statistics.mean(pooled[:k]) - statistics.mean(pooled[k:]))
        if d >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def fmt(x, pct=False, nd=3):
    if x is None:
        return '  —  '
    return f'{x*100:6.1f}%' if pct else f'{x:{nd+4}.{nd}f}'


def main():
    results = {}
    for name, (f, grp, note) in RUNS.items():
        if not os.path.exists(f):
            print(f'[缺少日志] {name}: {f}')
            continue
        rows = load(f)
        results[name] = dict(group=grp, note=note, **stats_for(rows))

    print('=' * 120)
    print(f"{'run':11s} {'group':14s} {'steps':>7s} {'→55%(s9)':>9s} {'→65%(s9)':>9s} {'→55%(raw)':>10s} "
          f"{'late40':>7s} {'late≥333k':>9s} {'winSD≥150k':>10s} {'EVmean':>7s} {'EVmax':>6s} {'Hjump':>5s} {'KL>.15':>6s} {'draw':>6s}")
    print('-' * 120)
    for name, r in results.items():
        print(f"{name:11s} {r['group']:14s} {r['final_steps']//1000:6d}k {str(r['steps_to_55_smooth9'] and r['steps_to_55_smooth9']//1000)+'k':>9s} "
              f"{str(r['steps_to_65_smooth9'] and r['steps_to_65_smooth9']//1000)+'k':>9s} "
              f"{str(r['steps_to_55_raw'] and r['steps_to_55_raw']//1000)+'k':>10s} "
              f"{r['late_win_last40']:7.3f} {r['late_win_ge333k']:9.3f} {r['win_sd_after150k']:10.3f} "
              f"{r['expl_var_mean_after150k']:7.3f} {r['expl_var_max_after150k']:6.3f} {r['entropy_jumps_gt025']:5d} "
              f"{r['kl_spikes_gt015']:6d} {r['draw_rate_mean']:6.3f}")

    # 分组比较（置换检验）
    def group(g):
        return [n for n, r in results.items() if r['group'] == g]

    print('\n分组比较（均值差 = A − B；双侧置换检验，10,000 次）')
    comparisons = [
        ('sparse', 'shaped-config', 'steps_to_55_smooth9'),
        ('sparse', 'shaped-config', 'steps_to_55_raw'),
        ('sparse', 'shaped-config', 'late_win_ge333k'),
        ('static', 'shaped-config', 'win_sd_after150k'),
        ('static', 'shaped-config', 'expl_var_mean_after150k'),
        ('static', 'shaped-config', 'entropy_jumps_gt025'),
        ('static', 'shaped-config', 'draw_rate_mean'),
    ]
    comp_out = []
    for a, b, key in comparisons:
        va = [results[n][key] for n in group(a) if results[n][key] is not None]
        vb = [results[n][key] for n in group(b) if results[n][key] is not None]
        if len(va) < 2 or len(vb) < 2:
            print(f'  {a} − {b} [{key}]: 样本不足')
            continue
        p = perm_test(va, vb)
        diff = statistics.mean(va) - statistics.mean(vb)
        comp_out.append(dict(a=a, b=b, metric=key, n_a=len(va), n_b=len(vb), mean_a=statistics.mean(va),
                             mean_b=statistics.mean(vb), diff=diff, perm_p=p))
        print(f'  {a:7s}(n={len(va)}) {statistics.mean(va):9.3f}  vs  {b:14s}(n={len(vb)}) {statistics.mean(vb):9.3f}  '
              f'diff {diff:+.3f}  perm p = {p:.3f}   [{key}]')

    out = dict(runs=results, comparisons=comp_out,
               notes=dict(
                   entropy='日志 entropy 为 SB3 entropy_loss（=−熵），本脚本已取负',
                   reward_bug='原 SelfPlayEnv.step 硬编码 −0.002，使原 sparse 消融与 shaped 奖励相同；修复后重跑 sparse_s1–s3',
               ))
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/robustness_training.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nwritten experiments/robustness_training.json')


if __name__ == '__main__':
    main()
