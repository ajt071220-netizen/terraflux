// batch.js —— 自动对局胜率统计
// 分批异步执行，避免阻塞 UI。SAFETY 是模拟器内部的步数保险丝
// （规则本身不设回合上限；超出保险丝的对局记为和局，防止病态对局拖死统计）。

import { createGame, chooseStart, applyMove } from './engine.js';
import { chooseMove } from './ai.js';

const SAFETY = 4000;
const BATCH_SIZE = 20;

function playOne(config) {
  const state = createGame(config);
  const opt = state.setupOptions[Math.floor(Math.random() * state.setupOptions.length)];
  chooseStart(state, opt.r, opt.c);
  let safety = SAFETY;
  while (state.status === 'playing' && safety-- > 0) {
    const color = state.turn;
    const m = chooseMove(state, color, 1.0);
    if (m === undefined) {
      if (state.config.passAllowed) applyMove(state, color, null);
      else break; // 引擎已处理无棋可走判负，这里是兜底
    } else {
      applyMove(state, color, m);
    }
  }
  return {
    winner: state.status === 'over' ? state.winner : 'draw',
    turns: state.moveCount.white + state.moveCount.black,
  };
}

export function runBatch(config, totalGames, onProgress) {
  return new Promise((resolve) => {
    const stats = { white: 0, black: 0, draw: 0, turns: 0, done: 0, total: totalGames };
    function step() {
      for (let i = 0; i < BATCH_SIZE && stats.done < totalGames; i++) {
        const r = playOne(config);
        if (r.winner === 'white') stats.white++;
        else if (r.winner === 'black') stats.black++;
        else stats.draw++;
        stats.turns += r.turns;
        stats.done++;
      }
      onProgress(stats);
      if (stats.done < totalGames) setTimeout(step, 0);
      else resolve(stats);
    }
    step();
  });
}
