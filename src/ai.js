// ai.js —— 启发式 AI：用于自动对局胜率测试
// 策略：一层贪心 + 对手最优响应（两层极小化），附带温度噪声保证对局多样性。

import { legalMoves, applyMove, cloneState, pillarHeight } from './engine.js';

function distanceToGoal(state, pos) {
  const s = state.size;
  const g = state.config.goalEdges;
  let d = Infinity;
  if (g.includes('S')) d = Math.min(d, s - pos.r);
  if (g.includes('N')) d = Math.min(d, pos.r - 1);
  if (g.includes('E')) d = Math.min(d, s - pos.c);
  if (g.includes('W')) d = Math.min(d, pos.c - 1);
  return d;
}

// 从 color 视角评估局面：越大越好
export function evaluate(state, color) {
  if (state.status === 'over') {
    if (state.winner === color) return 100000;
    if (state.winner === 'draw') return 0;
    return -100000;
  }
  const w = state.white;
  const b = state.black;
  const dist = Math.max(Math.abs(w.r - b.r), Math.abs(w.c - b.c));
  const wh = pillarHeight(state, w.r, w.c);
  const bh = pillarHeight(state, b.r, b.c);
  const goalDist = distanceToGoal(state, w);
  // 白：贴近胜利线、拉远距离、占高度优势
  const whiteScore = -goalDist * 12 + dist * 6 + (wh - bh) * 8;
  // 黑：阻止白贴近胜利线、压缩距离、占高度优势
  const blackScore = goalDist * 12 - dist * 6 + (bh - wh) * 8;
  return color === 'white' ? whiteScore : blackScore;
}

export function chooseMove(state, color, noise = 0, movesFilter = null) {
  const moves = movesFilter || legalMoves(state, color);
  if (moves.length === 0) return undefined;
  const opp = color === 'white' ? 'black' : 'white';
  let best = null;
  let bestScore = -Infinity;
  for (const m of moves) {
    const s2 = cloneState(state);
    applyMove(s2, color, m);
    let score;
    if (s2.status === 'over') {
      score = evaluate(s2, color);
    } else {
      // 假设对手走出其最优应招，我方取最差情况
      const oppMoves = legalMoves(s2, opp);
      let worst = Infinity;
      for (const om of oppMoves) {
        const s3 = cloneState(s2);
        applyMove(s3, opp, om);
        worst = Math.min(worst, evaluate(s3, color));
      }
      score = oppMoves.length === 0 ? evaluate(s2, color) : worst;
    }
    if (noise > 0) score += (Math.random() - 0.5) * noise * 20;
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best;
}
