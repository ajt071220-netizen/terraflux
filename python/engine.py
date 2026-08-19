"""engine.py —— 涌陆（Terraflux）规则引擎（Python 版，与 src/engine.js 逐条对稿）

用于 PPO 训练。所有函数为纯逻辑，状态用 dict 表示，坐标 1-indexed 与 JS 版一致。
地形状态：0 正常 / 1 升高(+1) / 2 降低(-1) / 3 填平(不可进入)
"""
from __future__ import annotations
import random

NORMAL, RAISED, LOWERED, FILLED = 0, 1, 2, 3
HEIGHT = (0, 1, -1, 0)

DEFAULT_CONFIG = dict(
    size=25,
    goalEdges=('S',),
    captureRule='gt',      # 'gte' 同高即捕 | 'gt' 严格高于
    firstPlayer='black',  # duel gap3 下黑先实测 49:50
    phaseMode='sync',      # 'off' | 'sync' | 'all8'
    startLayout='duel',    # 'duel' 黑镇中央+白正北贴身（实测最平衡）| 'blockade' | 'classic'
    gap=3,                # duel 实测最平衡（49:50）
    passAllowed=False,
    stalemateLoses=True,
    repetitionDraw=True,
    maxMoves=400,          # 训练保险：超过判平（网页端无上限）
)

DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def mid(size):
    return (size + 1) // 2


def in_bounds(size, r, c):
    return 1 <= r <= size and 1 <= c <= size


def idx(state, r, c):
    return (r - 1) * state['size'] + (c - 1)


def pillar_height(state, r, c):
    return HEIGHT[state['pillars'][idx(state, r, c)]]


def start_options(config):
    m, g = mid(config['size']), config['gap']
    if config['startLayout'] == 'duel':
        # 对垒式：黑镇中央，白正北隔 gap 格对峙——无选位
        return {
            'fixed': ('black', m, m),
            'chooser': None,
            'options': None,
            'white_start': (m - g, m),
        }
    if config['startLayout'] == 'blockade':
        return {
            'fixed': ('black', m, m),
            'chooser': 'white',
            'options': [(m - g, m), (m - g, m - g), (m - g, m + g)],
        }
    return {
        'fixed': ('white', m, m),
        'chooser': 'black',
        'options': [
            (m - g, m), (m + g, m), (m, m - g), (m, m + g),
            (m - g, m - g), (m - g, m + g), (m + g, m - g), (m + g, m + g),
        ],
    }


def create_game(config=None, rng=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    setup = start_options(cfg)
    state = {
        'config': cfg,
        'size': cfg['size'],
        'pillars': bytearray(cfg['size'] * cfg['size']),
        'white': None,
        'black': None,
        'turn': cfg['firstPlayer'],
        'moveCount': {'white': 0, 'black': 0},
        'passes': 0,
        'status': 'setup',
        'winner': None,
        'winReason': None,
        'repetition': {},
    }
    color, r, c = setup['fixed']
    state[color] = {'r': r, 'c': c}
    if setup['chooser'] is None:
        # duel：双方位置全固定
        wr, wc = setup['white_start']
        state['white'] = {'r': wr, 'c': wc}
    else:
        # 训练时由环境随机选定起点（不算动作）
        rng = rng or random
        r, c = setup['options'][rng.randrange(len(setup['options']))]
        state[setup['chooser']] = {'r': r, 'c': c}
    state['status'] = 'playing'
    record_repetition(state)
    return state


def phase_for(state, color):
    mode = state['config']['phaseMode']
    if mode == 'off':
        return 'cross'
    if mode == 'all8':
        return 'all8'
    n = state['moveCount'][color] + 1
    return 'cross' if n % 2 == 1 else 'x'


def affected_cells(state, r, c, phase):
    cells = []
    size = state['size']

    def push(rr, cc, to):
        if in_bounds(size, rr, cc):
            cells.append((rr, cc, to))

    if phase in ('cross', 'all8'):
        push(r - 1, c, RAISED)
        push(r + 1, c, LOWERED)
        push(r, c - 1, FILLED)
        push(r, c + 1, FILLED)
    if phase in ('x', 'all8'):
        push(r - 1, c - 1, RAISED)
        push(r - 1, c + 1, RAISED)
        push(r + 1, c - 1, LOWERED)
        push(r + 1, c + 1, LOWERED)
    return cells


def legal_moves(state, color):
    if state['status'] != 'playing':
        return []
    me = state[color]
    opp = state['white' if color == 'black' else 'black']
    moves = []
    for dr, dc in DIR8:
        r, c = me['r'] + dr, me['c'] + dc
        if not in_bounds(state['size'], r, c):
            continue
        if opp and opp['r'] == r and opp['c'] == c:
            continue
        if state['pillars'][idx(state, r, c)] == FILLED:
            continue
        moves.append((r, c))
    return moves


def is_on_goal(state, pos):
    g = state['config']['goalEdges']
    s = state['size']
    return (('N' in g and pos['r'] == 1) or ('S' in g and pos['r'] == s)
            or ('W' in g and pos['c'] == 1) or ('E' in g and pos['c'] == s))


def _judge(state, just_moved):
    w, b = state['white'], state['black']
    dist = max(abs(w['r'] - b['r']), abs(w['c'] - b['c']))
    wh = pillar_height(state, w['r'], w['c'])
    bh = pillar_height(state, b['r'], b['c'])
    capture_ok = bh >= wh if state['config']['captureRule'] == 'gte' else bh > wh
    if dist == 1 and capture_ok:
        if just_moved == 'white' and is_on_goal(state, w):
            return ('draw', 'border-capture')
        return ('black', 'capture')
    if just_moved == 'white' and is_on_goal(state, w):
        return ('white', 'escape')
    return None


def _snapshot(state):
    return (state['turn'], state['white']['r'], state['white']['c'],
            state['black']['r'], state['black']['c'], bytes(state['pillars']))


def record_repetition(state):
    if not state['config']['repetitionDraw']:
        return False
    key = _snapshot(state)
    n = state['repetition'].get(key, 0) + 1
    state['repetition'][key] = n
    return n >= 3


def apply_move(state, color, move):
    """move=(r,c) 或 None（停步）。原地修改 state，返回事件 dict，非法返回 None。"""
    if state['status'] != 'playing' or state['turn'] != color:
        return None
    events = {'color': color, 'moved': None, 'passed': False, 'phase': None, 'changes': [], 'result': None}

    if move is None:
        if not state['config']['passAllowed']:
            return None
        events['passed'] = True
        state['passes'] += 1
    else:
        legal = any(m == move for m in legal_moves(state, color))
        if not legal:
            return None
        phase = phase_for(state, color)
        state[color]['r'], state[color]['c'] = move
        events['moved'] = move
        events['phase'] = phase
        state['passes'] = 0
        for (r, c, to) in affected_cells(state, move[0], move[1], phase):
            i = idx(state, r, c)
            frm = state['pillars'][i]
            if frm != to:
                state['pillars'][i] = to
                events['changes'].append((r, c, frm, to))
    state['moveCount'][color] += 1

    result = _judge(state, color)
    if result:
        state['status'] = 'over'
        state['winner'], state['winReason'] = result
        events['result'] = result
        return events

    if state['config']['passAllowed'] and state['passes'] >= 2:
        state['status'] = 'over'
        state['winner'], state['winReason'] = 'draw', 'mutual-pass'
        events['result'] = ('draw', 'mutual-pass')
        return events

    state['turn'] = 'black' if color == 'white' else 'white'

    if record_repetition(state):
        state['status'] = 'over'
        state['winner'], state['winReason'] = 'draw', 'repetition'
        events['result'] = ('draw', 'repetition')
        return events

    if not state['config']['passAllowed'] and not legal_moves(state, state['turn']):
        state['status'] = 'over'
        if state['config']['stalemateLoses']:
            state['winner'], state['winReason'] = color, 'stalemate'
        else:
            state['winner'], state['winReason'] = 'draw', 'stalemate'
        events['result'] = (state['winner'], 'stalemate')
        return events

    total = state['moveCount']['white'] + state['moveCount']['black']
    if total >= state['config']['maxMoves']:
        state['status'] = 'over'
        state['winner'], state['winReason'] = 'draw', 'max-moves'
        events['result'] = ('draw', 'max-moves')

    return events


def clone_state(state):
    return {
        **state,
        'pillars': bytearray(state['pillars']),
        'white': dict(state['white']),
        'black': dict(state['black']),
        'moveCount': dict(state['moveCount']),
        'repetition': {},
    }
