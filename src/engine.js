// engine.js —— 涌陆（Terraflux）规则引擎（纯逻辑，无渲染依赖）
//
// 地形状态机（覆盖式，已与设计者对稿）：
//   落点的 4 个正方向邻居（十字相位）：北→升高 南→降低 西/东→填平
//   落点的 4 个斜向邻居（X 相位）：  西北/东北→升高 西南/东南→降低（不填平）
//   任何旧状态都会被新角色覆盖；被填平的柱子被升/降角色影响时会恢复成凹槽。
//
// 判定顺序：落定 → 地形变化（对方球站在受影响柱上会随之升降/被填平）→ 胜负判定。

export const T = { NORMAL: 0, RAISED: 1, LOWERED: 2, FILLED: 3 };

// 各状态对应的高度值：升高 +1，降低 -1，正常与填平都是 0
export const HEIGHT = [0, 1, -1, 0];

export const DEFAULT_CONFIG = {
  size: 25,               // 棋盘边长（21 或 25）
  goalEdges: ['S'],       // 胜利边：'N' 'S' 'E' 'W' 任意子集
  captureRule: 'gt',      // 'gte' 同高即捕（≥） | 'gt' 须严格高于（>）
  firstPlayer: 'black',   // 先手方（duel gap3 下黑先实测 49:50）
  phaseMode: 'sync',      // 'off' 恒定十字 | 'sync' 双方同步 十字/X 交替 | 'all8' 每步八向
  startLayout: 'duel',    // 'duel' 黑镇中央+白正北对峙（实测最平衡）| 'blockade' 黑镇中央白三选一 | 'classic' 白中央黑八选一
  gap: 3,                 // 开局间距（切比雪夫距离，3 = 中间隔两格；duel 实测 49:50）
  passAllowed: false,     // 是否允许停步
  stalemateLoses: true,   // 无合法步的一方判负
  repetitionDraw: true,   // 同一局面出现三次判和
};

export function mid(size) {
  return (size + 1) >> 1; // 1-indexed 中心点
}

export function inBounds(size, r, c) {
  return r >= 1 && r <= size && c >= 1 && c <= size;
}

function idx(state, r, c) {
  return (r - 1) * state.size + (c - 1);
}

export function pillarState(state, r, c) {
  return state.pillars[idx(state, r, c)];
}

export function pillarHeight(state, r, c) {
  return HEIGHT[state.pillars[idx(state, r, c)]];
}

// 开局布置：返回固定方位置与待选方候选点
export function startOptions(config) {
  const m = mid(config.size);
  const g = config.gap;
  if (config.startLayout === 'duel') {
    // 对垒式：黑镇中央，白正北隔 gap 格对峙——无选位，直接落子
    return {
      fixed: { color: 'black', r: m, c: m },
      chooser: null,
      options: null,
      whiteStart: { r: m - g, c: m },
    };
  }
  if (config.startLayout === 'blockade') {
    // 黑镇中央，白在正后方/两侧斜后方三选一
    return {
      fixed: { color: 'black', r: m, c: m },
      chooser: 'white',
      options: [
        { r: m - g, c: m },
        { r: m - g, c: m - g },
        { r: m - g, c: m + g },
      ],
    };
  }
  // classic：白在中央，黑在距其 gap 的 8 个方向选一
  return {
    fixed: { color: 'white', r: m, c: m },
    chooser: 'black',
    options: [
      { r: m - g, c: m }, { r: m + g, c: m }, { r: m, c: m - g }, { r: m, c: m + g },
      { r: m - g, c: m - g }, { r: m - g, c: m + g }, { r: m + g, c: m - g }, { r: m + g, c: m + g },
    ],
  };
}

export function createGame(config = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const setup = startOptions(cfg);
  const state = {
    config: cfg,
    size: cfg.size,
    pillars: new Uint8Array(cfg.size * cfg.size), // 全 0 = 全部正常
    white: null,
    black: null,
    turn: cfg.firstPlayer,
    moveCount: { white: 0, black: 0 },
    passes: 0,
    status: 'setup', // 'setup' | 'playing' | 'over'
    winner: null,    // 'white' | 'black' | 'draw'
    winReason: null,
    setupOptions: setup.options,
    setupChooser: setup.chooser,
    repetition: new Map(),
  };
  if (setup.fixed.color === 'black') state.black = { r: setup.fixed.r, c: setup.fixed.c };
  else state.white = { r: setup.fixed.r, c: setup.fixed.c };
  if (setup.chooser === null) {
    // duel：双方位置全固定，跳过选位阶段
    state.white = { r: setup.whiteStart.r, c: setup.whiteStart.c };
    state.status = 'playing';
    state.setupOptions = null;
    recordRepetition(state);
  }
  return state;
}

// 待选方选择起始点
export function chooseStart(state, r, c) {
  if (state.status !== 'setup') return false;
  const ok = state.setupOptions.some((o) => o.r === r && o.c === c);
  if (!ok) return false;
  if (state.setupChooser === 'white') state.white = { r, c };
  else state.black = { r, c };
  state.status = 'playing';
  state.setupOptions = null;
  recordRepetition(state);
  return true;
}

// 当前行动方这一步触发的相位
export function phaseFor(state, color) {
  const mode = state.config.phaseMode;
  if (mode === 'off') return 'cross';
  if (mode === 'all8') return 'all8';
  // sync：双方各自计步、同相位——第 1 步十字，第 2 步 X，循环
  const n = state.moveCount[color] + 1;
  return n % 2 === 1 ? 'cross' : 'x';
}

// 落点 (r,c) 在给定相位下将影响的地形格子
export function affectedCells(state, r, c, phase) {
  const cells = [];
  const push = (rr, cc, to) => {
    if (inBounds(state.size, rr, cc)) cells.push({ r: rr, c: cc, to });
  };
  if (phase === 'cross' || phase === 'all8') {
    push(r - 1, c, T.RAISED);
    push(r + 1, c, T.LOWERED);
    push(r, c - 1, T.FILLED);
    push(r, c + 1, T.FILLED);
  }
  if (phase === 'x' || phase === 'all8') {
    push(r - 1, c - 1, T.RAISED);
    push(r - 1, c + 1, T.RAISED);
    push(r + 1, c - 1, T.LOWERED);
    push(r + 1, c + 1, T.LOWERED);
  }
  return cells;
}

export function legalMoves(state, color) {
  if (state.status !== 'playing') return [];
  const me = state[color];
  const opp = state[color === 'white' ? 'black' : 'white'];
  const moves = [];
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      const r = me.r + dr;
      const c = me.c + dc;
      if (!inBounds(state.size, r, c)) continue;
      if (opp && opp.r === r && opp.c === c) continue;
      if (state.pillars[idx(state, r, c)] === T.FILLED) continue;
      moves.push({ r, c });
    }
  }
  return moves;
}

export function isOnGoal(state, pos) {
  const g = state.config.goalEdges;
  const s = state.size;
  return (
    (g.includes('N') && pos.r === 1) ||
    (g.includes('S') && pos.r === s) ||
    (g.includes('W') && pos.c === 1) ||
    (g.includes('E') && pos.c === s)
  );
}

// 落定 → 地形 → 判定
function judge(state, justMoved) {
  const w = state.white;
  const b = state.black;
  const dist = Math.max(Math.abs(w.r - b.r), Math.abs(w.c - b.c));
  const wh = pillarHeight(state, w.r, w.c);
  const bh = pillarHeight(state, b.r, b.c);
  const captureOk = state.config.captureRule === 'gte' ? bh >= wh : bh > wh;
  if (dist === 1 && captureOk) {
    // 白这一步刚踏上胜利线时被追上 → 平局
    if (justMoved === 'white' && isOnGoal(state, w)) return { winner: 'draw', reason: 'border-capture' };
    return { winner: 'black', reason: 'capture' };
  }
  if (justMoved === 'white' && isOnGoal(state, w)) return { winner: 'white', reason: 'escape' };
  return null;
}

function snapshot(state) {
  let s = state.turn + '|' + state.white.r + ',' + state.white.c + '|' + state.black.r + ',' + state.black.c + '|';
  for (let i = 0; i < state.pillars.length; i++) s += state.pillars[i];
  return s;
}

function recordRepetition(state) {
  if (!state.config.repetitionDraw) return false;
  const key = snapshot(state);
  const n = (state.repetition.get(key) || 0) + 1;
  state.repetition.set(key, n);
  return n >= 3;
}

// 执行一步。move 为 {r,c}，或 null 表示停步（需 passAllowed）。
// 原地修改 state，返回事件对象（供动画/UI），非法步返回 null。
export function applyMove(state, color, move) {
  if (state.status !== 'playing' || state.turn !== color) return null;
  const events = { color, moved: null, passed: false, phase: null, changes: [], result: null };

  if (move === null) {
    if (!state.config.passAllowed) return null;
    events.passed = true;
    state.passes += 1;
  } else {
    const legal = legalMoves(state, color).some((m) => m.r === move.r && m.c === move.c);
    if (!legal) return null;
    const phase = phaseFor(state, color); // 在步数增加前取相位
    state[color].r = move.r;
    state[color].c = move.c;
    events.moved = { r: move.r, c: move.c };
    events.phase = phase;
    state.passes = 0;
    for (const cell of affectedCells(state, move.r, move.c, phase)) {
      const i = idx(state, cell.r, cell.c);
      const from = state.pillars[i];
      if (from !== cell.to) {
        state.pillars[i] = cell.to;
        events.changes.push({ r: cell.r, c: cell.c, from, to: cell.to });
      }
    }
  }
  state.moveCount[color] += 1;

  // 胜负判定（先判定，再换手）
  const result = judge(state, color);
  if (result) {
    state.status = 'over';
    state.winner = result.winner;
    state.winReason = result.reason;
    events.result = result;
    return events;
  }

  // 双方连续停步 → 和棋
  if (state.config.passAllowed && state.passes >= 2) {
    state.status = 'over';
    state.winner = 'draw';
    state.winReason = 'mutual-pass';
    events.result = { winner: 'draw', reason: 'mutual-pass' };
    return events;
  }

  state.turn = color === 'white' ? 'black' : 'white';

  // 重复局面三次判和
  if (recordRepetition(state)) {
    state.status = 'over';
    state.winner = 'draw';
    state.winReason = 'repetition';
    events.result = { winner: 'draw', reason: 'repetition' };
    return events;
  }

  // 下一步方无合法步（不允许停步时）→ 判负
  if (!state.config.passAllowed && legalMoves(state, state.turn).length === 0) {
    state.status = 'over';
    if (state.config.stalemateLoses) {
      state.winner = color; // 对方无棋可走，刚走完的一方胜
      state.winReason = 'stalemate';
    } else {
      state.winner = 'draw';
      state.winReason = 'stalemate';
    }
    events.result = { winner: state.winner, reason: 'stalemate' };
    return events;
  }

  return events;
}

// AI 搜索用的轻量克隆（不携带重复局面记录）
export function cloneState(state) {
  return {
    ...state,
    pillars: state.pillars.slice(),
    white: { ...state.white },
    black: { ...state.black },
    moveCount: { ...state.moveCount },
    repetition: new Map(),
  };
}

export const REASON_TEXT = {
  capture: '追捕成功',
  escape: '成功抵达边界',
  'border-capture': '抵达边界的同时被追上',
  stalemate: '无棋可走',
  repetition: '局面三次重复',
  'mutual-pass': '双方连续停步',
};
