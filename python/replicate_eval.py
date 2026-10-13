"""replicate_eval.py —— 实验④评估：duel 自对弈白胜率、平局原因、循环赛

1) 自对弈：每个模型在 duel 布局（gap=3，黑先，25×25）上自对弈 N 局（采样模式），统计白胜/黑胜/平局及原因。
   静止棋盘模型（static）按静止规则评估。
2) 循环赛：动态棋盘模型两两对打，每种颜色各 N 局（采样模式，与 model_vs_model.py 一致）。

增量与分片（v2 批次二）：
  --new-only         跳过 --out 中已有的模型与对局，只计算新增部分
  --shard i --nshards n   只计算第 i 份（任务按序号取模分配），结果写入 experiments/replicate_shards/shard_i.json
  --merge            把 experiments/replicate_shards/*.json 合并进 --out（已存在的条目不重复）

用法（仓库根目录）：
  .\\.venv-py312\\Scripts\\python.exe python\\replicate_eval.py --games 300 --rr-games 100 --new-only --shard 0 --nshards 4
  .\\.venv-py312\\Scripts\\python.exe python\\replicate_eval.py --merge
"""
from __future__ import annotations
import argparse
import glob
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_selfplay as ES              # noqa: E402
import model_vs_model as MVM            # noqa: E402

# 名称 -> (模型路径, 是否静止棋盘, 说明)
MODELS = {
    'main':       ('models/ppo_25.zip',                  False, '原主模型（shaped 配置，未记录种子）'),
    'abl_shaped': ('experiments/ppo25_duel_shaped.zip', False, '原消融 shaped（未记录种子）'),
    'abl_sparse': ('experiments/ppo25_duel_sparse.zip', False, '原消融 sparse：实际与 shaped 奖励相同（硬编码 bug），仅作同配置重复'),
    'shaped_s2': ('models/ppo25_shaped_s2.zip',         False, 'shaped，seed 2'),
    'shaped_s3': ('models/ppo25_shaped_s3.zip',         False, 'shaped，seed 3'),
    'shaped_s4': ('models/ppo25_shaped_s4.zip',         False, 'shaped，seed 4'),
    'shaped_s5': ('models/ppo25_shaped_s5.zip',         False, 'shaped，seed 5'),
    'sparse_s1': ('models/ppo25_sparse_s1.zip',         False, 'sparse（无步罚），seed 1'),
    'sparse_s2': ('models/ppo25_sparse_s2.zip',         False, 'sparse（无步罚），seed 2'),
    'sparse_s3': ('models/ppo25_sparse_s3.zip',         False, 'sparse（无步罚），seed 3'),
    'sparse_s4': ('models/ppo25_sparse_s4.zip',         False, 'sparse（无步罚），seed 4'),
    'sparse_s5': ('models/ppo25_sparse_s5.zip',         False, 'sparse（无步罚），seed 5'),
    'sparse_s6': ('models/ppo25_sparse_s6.zip',         False, 'sparse（无步罚），seed 6'),
    'static_s1': ('models/ppo25_static_s1.zip',         True,  '静止棋盘，seed 1'),
    'static_s2': ('models/ppo25_static_s2.zip',         True,  '静止棋盘，seed 2'),
    'static_s3': ('models/ppo25_static_s3.zip',         True,  '静止棋盘，seed 3'),
    'static_s4': ('models/ppo25_static_s4.zip',         True,  '静止棋盘，seed 4'),
    'static_s5': ('models/ppo25_static_s5.zip',         True,  '静止棋盘，seed 5'),
    'static_s6': ('models/ppo25_static_s6.zip',         True,  '静止棋盘，seed 6'),
    # 第三批（步罚组补到每组 10 个种子）
    'shaped_s6':  ('models/ppo25_shaped_s6.zip',        False, 'shaped，seed 6'),
    'shaped_s7':  ('models/ppo25_shaped_s7.zip',        False, 'shaped，seed 7'),
    'shaped_s8':  ('models/ppo25_shaped_s8.zip',        False, 'shaped，seed 8'),
    'shaped_s9':  ('models/ppo25_shaped_s9.zip',        False, 'shaped，seed 9'),
    'shaped_s10': ('models/ppo25_shaped_s10.zip',       False, 'shaped，seed 10'),
    'shaped_s11': ('models/ppo25_shaped_s11.zip',       False, 'shaped，seed 11'),
    'sparse_s7':  ('models/ppo25_sparse_s7.zip',        False, 'sparse（无步罚），seed 7'),
    'sparse_s8':  ('models/ppo25_sparse_s8.zip',        False, 'sparse（无步罚），seed 8'),
    'sparse_s9':  ('models/ppo25_sparse_s9.zip',        False, 'sparse（无步罚），seed 9'),
    'sparse_s10': ('models/ppo25_sparse_s10.zip',       False, 'sparse（无步罚），seed 10'),
}
RR_CANDIDATES = [n for n, v in MODELS.items() if not v[1]]   # 只有动态棋盘参与循环赛


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def self_play(name, path, static, games, seed):
    model, use_mask = ES.load_model(path)
    config = {'size': 25, 'gap': 3, 'firstPlayer': 'black'}
    if static:
        config['staticTerrain'] = True
    rng = random.Random(seed)
    res = Counter()
    reasons = Counter()
    lens = []
    t0 = time.time()
    for _ in range(games):
        st, steps = ES.play_one(model, use_mask, config, rng, deterministic=False)
        w = st['winner'] or 'unfinished'
        res[w] += 1
        if w == 'draw':
            reasons[st.get('winReason') or '?'] += 1
        lens.append(steps)
    n = games
    lo, hi = wilson(res['white'], n)
    out = dict(model=name, path=path, static=static, games=n, seed=seed,
               white=res['white'], black=res['black'], draw=res['draw'], unfinished=res['unfinished'],
               white_rate=res['white'] / n, white_ci=[lo, hi],
               black_rate=res['black'] / n, draw_rate=res['draw'] / n,
               draw_reasons=dict(reasons), mean_len=sum(lens) / n, seconds=round(time.time() - t0, 1))
    print(f"[自对弈] {name:10s} 白 {out['white_rate']:.1%} [{lo:.1%},{hi:.1%}]  黑 {out['black_rate']:.1%}  "
          f"平 {out['draw_rate']:.1%} {dict(reasons)}  平均 {out['mean_len']:.0f} 步  ({out['seconds']}s)", flush=True)
    return out


def round_robin_pair(a, b, games, seed, cache):
    for n in (a, b):
        if n not in cache:
            cache[n] = MVM.load_model(MODELS[n][0])
    rng = random.Random(seed)
    total = Counter()
    t0 = time.time()
    for a_color in ('white', 'black'):
        for _ in range(games):
            st = MVM.play_one(cache[a], cache[b], a_color, rng)
            if st['winner'] == 'draw':
                total['draw'] += 1
            elif st['winner'] == a_color:
                total['a'] += 1
            else:
                total['b'] += 1
    n = games * 2
    row = dict(a=a, b=b, games=n, seed=seed, a_win=total['a'] / n, b_win=total['b'] / n,
               draw=total['draw'] / n, a_wins=total['a'], b_wins=total['b'], draws=total['draw'])
    print(f"[循环赛] {a:10s} vs {b:10s}  {row['a_win']:.1%} : {row['b_win']:.1%}  平 {row['draw']:.1%}  ({round(time.time()-t0)}s)", flush=True)
    return row


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return dict(selfplay=[], round_robin=[], missing=[], models={})


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def merge(out):
    base = load_json(out)
    have_sp = {r['model'] for r in base['selfplay']}
    have_rr = {(r['a'], r['b']) for r in base['round_robin']} | {(r['b'], r['a']) for r in base['round_robin']}
    added = 0
    for f in sorted(glob.glob(os.path.join('experiments', 'replicate_shards', 'shard_*.json'))):
        sh = load_json(f)
        for r in sh['selfplay']:
            if r['model'] not in have_sp:
                base['selfplay'].append(r); have_sp.add(r['model']); added += 1
        for r in sh['round_robin']:
            if (r['a'], r['b']) not in have_rr:
                base['round_robin'].append(r); have_rr.add((r['a'], r['b'])); added += 1
    base['models'] = {k: dict(path=v[0], static=v[1], note=v[2]) for k, v in MODELS.items()}
    save_json(out, base)
    print(f'merged {added} new entries into {out}: selfplay={len(base["selfplay"])} rr_pairs={len(base["round_robin"])}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=300, help='每个模型自对弈局数')
    ap.add_argument('--rr-games', type=int, default=100, help='循环赛每种颜色局数')
    ap.add_argument('--only', type=str, default='', help='逗号分隔，只评估这些模型（默认全部可用模型）')
    ap.add_argument('--skip-rr', action='store_true')
    ap.add_argument('--new-only', action='store_true', help='跳过 --out 中已有的模型与对局')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--merge', action='store_true', help='合并 experiments/replicate_shards/*.json 到 --out')
    ap.add_argument('--out', type=str, default='experiments/replicate_eval.json')
    args = ap.parse_args()

    if args.merge:
        merge(args.out)
        return

    base = load_json(args.out)
    done_sp = {r['model'] for r in base['selfplay']} if args.new_only else set()
    done_pairs = set()
    if args.new_only:
        for r in base['round_robin']:
            done_pairs.add((r['a'], r['b'])); done_pairs.add((r['b'], r['a']))

    names = [n for n in MODELS if (not args.only or n in args.only.split(','))]
    present = [n for n in names if os.path.exists(MODELS[n][0])]
    missing = [n for n in names if n not in present]
    if missing:
        print('缺少模型（跳过）：', missing)

    sp_todo = [n for n in present if n not in done_sp]
    sp_todo = sp_todo[args.shard::args.nshards]
    selfplay = []
    for n in sp_todo:
        seed = 2000 + list(MODELS).index(n)
        selfplay.append(self_play(n, MODELS[n][0], MODELS[n][1], args.games, seed=seed))

    rr = []
    if not args.skip_rr:
        rr_names = [n for n in RR_CANDIDATES if n in present]
        all_pairs = list(itertools.combinations(rr_names, 2))
        todo = [p for p in all_pairs if p not in done_pairs and (p[1], p[0]) not in done_pairs]
        todo = todo[args.shard::args.nshards]
        cache = {}
        for idx, (a, b) in enumerate(todo):
            seed = 7 + (all_pairs.index((a, b)) if (a, b) in all_pairs else 0)
            rr.append(round_robin_pair(a, b, args.rr_games, seed, cache))

    if args.nshards > 1 or args.new_only:
        out = os.path.join('experiments', 'replicate_shards', f'shard_{args.shard}.json')
        save_json(out, dict(selfplay=selfplay, round_robin=rr))
        print('shard written', out)
    else:
        base['selfplay'] += selfplay
        base['round_robin'] += rr
        base['missing'] = missing
        base['models'] = {k: dict(path=v[0], static=v[1], note=v[2]) for k, v in MODELS.items()}
        save_json(args.out, base)
        print('written', args.out)


if __name__ == '__main__':
    main()
