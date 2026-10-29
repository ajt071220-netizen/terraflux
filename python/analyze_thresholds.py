"""analyze_thresholds.py —— 训练统计阈值的敏感性分析

报告 §4.1/§4.2/§5.3 用了三个阈值：熵跳变 |Δ熵|>0.25、KL 尖峰 >0.15、滚动胜率 55%。
本脚本在每个阈值附近取若干档，重算每个运行的计数/步数，并重做报告中用到的分组置换检验
（static vs shaped-config：熵跳变；sparse vs shaped-config：达到门槛的步数），
检查结论是否依赖于阈值的选取。输出：experiments/threshold_sensitivity.json
"""
import json
import statistics

from analyze_robustness import RUNS, load, perm_test, steps_to

ENT_THR = [0.15, 0.20, 0.25, 0.30, 0.40]
KL_THR = [0.05, 0.10, 0.15, 0.20, 0.30]
WIN_THR = [0.50, 0.55, 0.60, 0.65]


def ent_jumps(rows, thr):
    ent = [-r['entropy'] for r in rows if r.get('entropy') is not None]
    return sum(1 for a, b in zip(ent, ent[1:]) if abs(b - a) > thr)


def kl_spikes(rows, thr):
    return sum(1 for r in rows if r.get('approx_kl') is not None and r['approx_kl'] > thr)


def main():
    logs = {n: (load(f), g) for n, (f, g, _) in RUNS.items()}
    groups = {g: [n for n, (_, gg) in logs.items() if gg == g] for g in ('shaped-config', 'sparse', 'static')}
    dynamic = groups['shaped-config'] + groups['sparse']
    out = dict(entropy=[], kl=[], win=[])

    print('熵跳变阈值   动态运行范围   static 均值  shaped 均值   perm p')
    for t in ENT_THR:
        c = {n: ent_jumps(logs[n][0], t) for n in logs}
        a = [c[n] for n in groups['static']]
        b = [c[n] for n in groups['shaped-config']]
        dyn = [c[n] for n in dynamic]
        p = perm_test(a, b)
        out['entropy'].append(dict(thr=t, dynamic_min=min(dyn), dynamic_max=max(dyn),
                                   static_mean=statistics.mean(a), shaped_mean=statistics.mean(b), perm_p=p))
        print(f'  >{t:.2f}       {min(dyn)}–{max(dyn):<10d} {statistics.mean(a):9.2f}  {statistics.mean(b):10.2f}   {p:.3f}')

    print('\nKL 尖峰阈值   动态运行范围   static 均值  shaped 均值   perm p')
    for t in KL_THR:
        c = {n: kl_spikes(logs[n][0], t) for n in logs}
        a = [c[n] for n in groups['static']]
        b = [c[n] for n in groups['shaped-config']]
        dyn = [c[n] for n in dynamic]
        p = perm_test(a, b)
        out['kl'].append(dict(thr=t, dynamic_min=min(dyn), dynamic_max=max(dyn),
                              static_mean=statistics.mean(a), shaped_mean=statistics.mean(b), perm_p=p))
        print(f'  >{t:.2f}       {min(dyn)}–{max(dyn):<10d} {statistics.mean(a):9.2f}  {statistics.mean(b):10.2f}   {p:.3f}')

    print('\n胜率门槛（9 点平滑）  sparse 到达数/均值   shaped 到达数/均值   perm p')
    for t in WIN_THR:
        s = {n: steps_to(logs[n][0], t, True) for n in dynamic}
        a = [s[n] for n in groups['sparse'] if s[n] is not None]
        b = [s[n] for n in groups['shaped-config'] if s[n] is not None]
        p = perm_test(a, b) if len(a) >= 2 and len(b) >= 2 else None
        out['win'].append(dict(thr=t, sparse_reached=len(a), sparse_n=len(groups['sparse']),
                               shaped_reached=len(b), shaped_n=len(groups['shaped-config']),
                               sparse_mean_k=statistics.mean(a) / 1000, shaped_mean_k=statistics.mean(b) / 1000,
                               perm_p=p))
        print(f'  {t:.2f}               {len(a)}/{len(groups["sparse"])} {statistics.mean(a)/1000:6.0f}k'
              f'          {len(b)}/{len(groups["shaped-config"])} {statistics.mean(b)/1000:6.0f}k        {p:.3f}')

    with open('experiments/threshold_sensitivity.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\nwritten experiments/threshold_sensitivity.json')


if __name__ == '__main__':
    main()
