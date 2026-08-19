"""selfcheck.py —— Python 引擎自检：与启发式 AI 跑若干局，确认无异常、胜负分布合理"""
import random
import engine as E
import heuristic


def play_one(rng, size=21):
    state = E.create_game({'size': size}, rng)
    steps = 0
    while state['status'] == 'playing' and steps < 800:
        color = state['turn']
        mv = heuristic.choose_move(state, color, rng=rng)
        ev = E.apply_move(state, color, mv)
        assert ev is not None, f'非法步: {color} {mv}'
        steps += 1
    return state


def main():
    rng = random.Random(2026)
    wins = {'white': 0, 'black': 0, 'draw': 0}
    reasons = {}
    turns = 0
    N = 200
    for _ in range(N):
        st = play_one(rng)
        wins[st['winner']] += 1
        reasons[st['winReason']] = reasons.get(st['winReason'], 0) + 1
        turns += st['moveCount']['white'] + st['moveCount']['black']
    print(f'{N} 局（启发式 vs 启发式，21×21，堵截式）:')
    print(f'  白胜 {wins["white"]} ({wins["white"]/N:.0%})  黑胜 {wins["black"]} ({wins["black"]/N:.0%})  平 {wins["draw"]}')
    print(f'  平均步数 {turns/N:.0f}，终局原因: {reasons}')


if __name__ == '__main__':
    main()
