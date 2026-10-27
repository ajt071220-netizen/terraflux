// parity_check.mjs —— 用 src/engine.js 重放 python/parity_gen.py 生成的轨迹，逐步比对两套引擎
//
// 用法：node tests/parity_check.mjs [experiments/parity_traces.json]
import { readFileSync } from 'node:fs';
import { createGame, chooseStart, legalMoves, applyMove } from '../src/engine.js';

const path = process.argv[2] || new URL('../experiments/parity_traces.json', import.meta.url);
const { seed, games } = JSON.parse(readFileSync(path, 'utf-8'));

function dumpState(s) {
  return {
    pillars: Array.from(s.pillars).join(''),
    white: [s.white.r, s.white.c],
    black: [s.black.r, s.black.c],
    turn: s.turn,
    moveCount: [s.moveCount.white, s.moveCount.black],
    passes: s.passes,
    status: s.status,
    winner: s.winner,
    winReason: s.winReason,
  };
}

function dumpEvents(ev) {
  if (ev === null) return null;
  return {
    moved: ev.moved ? [ev.moved.r, ev.moved.c] : null,
    passed: ev.passed,
    phase: ev.phase,
    changes: ev.changes.map((x) => [x.r, x.c, x.from, x.to]),
    result: ev.result ? [ev.result.winner, ev.result.reason] : null,
  };
}

const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

let nSteps = 0;
let nIllegal = 0;
const failures = [];
const reasons = {};

for (let gi = 0; gi < games.length; gi++) {
  const g = games[gi];
  const { maxMoves, ...cfg } = g.config;
  const s = createGame(cfg);
  if (s.status === 'setup') {
    const [r, c] = g.start[s.setupChooser];
    if (!chooseStart(s, r, c)) failures.push({ game: gi, step: -1, what: 'chooseStart rejected' });
  }
  const start = { white: [s.white.r, s.white.c], black: [s.black.r, s.black.c] };
  if (!eq(start, g.start)) failures.push({ game: gi, step: -1, what: 'start', js: start, py: g.start });

  for (let si = 0; si < g.steps.length; si++) {
    const st = g.steps[si];
    const legal = s.status === 'playing' ? legalMoves(s, s.turn).map((m) => [m.r, m.c]) : [];
    if (!eq(legal, st.legal)) failures.push({ game: gi, step: si, what: 'legal', js: legal, py: st.legal });
    const move = st.move ? { r: st.move[0], c: st.move[1] } : null;
    const ev = dumpEvents(applyMove(s, st.actor, move));
    if (ev === null) nIllegal++;
    if (!eq(ev, st.events)) failures.push({ game: gi, step: si, what: 'events', js: ev, py: st.events });
    const state = dumpState(s);
    if (!eq(state, st.state)) failures.push({ game: gi, step: si, what: 'state', js: state, py: st.state });
    nSteps++;
    if (failures.length > 20) break;
  }
  const last = dumpState(s);
  const key = last.status === 'over' ? last.winReason : 'unfinished';
  reasons[key] = (reasons[key] || 0) + 1;
  if (failures.length > 20) break;
}

console.log(`seed=${seed} games=${games.length} steps=${nSteps} rejected-moves=${nIllegal}`);
console.log('end reasons (JS):', Object.fromEntries(Object.entries(reasons).sort()));
if (failures.length) {
  console.log(`MISMATCH: ${failures.length}${failures.length > 20 ? '+' : ''} discrepancies; first:`);
  for (const f of failures.slice(0, 5)) console.log(JSON.stringify(f));
  process.exit(1);
}
console.log('PASS: every step identical (legal moves, events, full state)');
