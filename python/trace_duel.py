"""trace_duel.py —— 解剖 duel 布局下启发式白方为何速败：打印一局前 12 步轨迹"""
import random
import sys

sys.path.insert(0, 'python')

import engine as E
import heuristic

rng = random.Random(42)
state = E.create_game({'size': 25, 'startLayout': 'duel', 'gap': 3, 'firstPlayer': 'black'}, rng)

print(f'开局: 黑 {state["black"]} 白 {state["white"]}  (黑先)')
print(f'{"步":>2} {"方":^2} {"从":>10} {"到":>10}  白高/黑高  事件')
for step in range(12):
    if state['status'] != 'playing':
        break
    color = state['turn']
    me = state[color]
    wh = E.pillar_height(state, state['white']['r'], state['white']['c'])
    bh = E.pillar_height(state, state['black']['r'], state['black']['c'])
    mv = heuristic.choose_move(state, color, rng=rng)
    ev = E.apply_move(state, color, mv)
    evtxt = ' '.join(str(e if isinstance(e, str) else e.get('type', '?')) for e in (ev or [])) if ev else ''
    print(f'{step + 1:>2} {color:^6} ({me["r"]},{me["c"]}) → {str(mv):>10}   {wh}/{bh}   {evtxt}')

print(f'\n终局: {state["status"]} winner={state.get("winner")} reason={state.get("winReason")}')
print(f'黑白各走: {state["moveCount"]}')
