// main.js —— 入口：组装引擎、3D 场景、UI 与智能体调度

import * as E from './engine.js';
import { BoardScene } from './scene.js';
import * as UI from './ui.js';
import { runBatch } from './batch.js';
import { requestMove, chooseSetup, saveReplay, formatMeta, detectBackend } from './agent.js';
import { initTrainingMonitor, setTrainVisible } from './training.js';
import { t, toggleLang, applyStatic, onLangChange } from './i18n.js';

const scene = new BoardScene(document.getElementById('viewport'));

let state = null;
let currentConfig = null;
let players = { white: 'human', black: 'human' };
let lastHover = null;
let aiBusy = false;         // 防重入：AI 请求进行中
let replayMoves = [];       // 棋谱

const kindLabel = (id) => t('agent.' + id);
const sideName = (color) => (color === 'white' ? t('hint.white') : t('hint.black'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function startGame(config, pl) {
  currentConfig = config;
  players = pl || { white: 'human', black: 'human' };
  replayMoves = [];
  aiBusy = false;
  state = E.createGame(config);
  scene.buildBoard(config.size);
  scene.syncState(state);
  UI.hideBanner();
  UI.aiToast('');
  refresh();
  maybeAI(); // setup 阶段也可能是 AI
}

function refresh() {
  UI.updateHUD(state);
  if (state.status === 'setup') {
    const chooser = state.setupChooser;
    if (players[chooser] === 'human') {
      scene.showHints(state.setupOptions, 0xffd479);
      UI.setHint(chooser === 'white' ? t('hint.setupWhite') : t('hint.setupBlack'));
    } else {
      scene.showHints(state.setupOptions, 0xffd479);
      UI.setHint(`${kindLabel(players[chooser])} ${t('hint.choosing')}`);
    }
    UI.setPassVisible(false);
    return;
  }
  if (state.status === 'playing') {
    const isHuman = players[state.turn] === 'human';
    scene.showHints(isHuman ? E.legalMoves(state, state.turn) : [], state.turn === 'white' ? 0xc9a86a : 0x8a744a);
    const phase = E.phaseFor(state, state.turn);
    const phaseText = phase === 'cross' ? t('hint.phaseCross')
      : phase === 'x' ? t('hint.phaseX')
      : t('hint.phaseAll8');
    const who = sideName(state.turn);
    UI.setHint(isHuman
      ? `${who} ${t('hint.move')} — ${phaseText}`
      : `${who} · ${kindLabel(players[state.turn])} ${t('hint.thinking')} — ${phaseText}`);
    UI.setPassVisible(state.config.passAllowed && isHuman);
    return;
  }
  // over
  scene.showHints([]);
  UI.setPassVisible(false);
  UI.setHint(t('hint.over'));
  UI.showBanner(state);
  saveReplay({
    config: currentConfig,
    players,
    moves: replayMoves,
    result: { winner: state.winner, reason: state.winReason },
    endedAt: new Date().toISOString(),
  });
}

// ---------- AI 调度 ----------
async function maybeAI() {
  if (!state || aiBusy) return;

  // setup 阶段：AI 自动选起点
  if (state.status === 'setup') {
    const chooser = state.setupChooser;
    if (players[chooser] !== 'human') {
      aiBusy = true;
      await sleep(500);
      if (!state || state.status !== 'setup') { aiBusy = false; return; }
      const opt = chooseSetup(state);
      E.chooseStart(state, opt.r, opt.c);
      replayMoves.push({ setup: chooser, r: opt.r, c: opt.c });
      scene.syncState(state);
      aiBusy = false;
      refresh();
      maybeAI();
    }
    return;
  }

  if (state.status !== 'playing') return;
  const kind = players[state.turn];
  if (kind === 'human') return;

  aiBusy = true;
  const color = state.turn;
  const whoTag = `<span class="who">${sideName(color)} · ${kindLabel(kind)}</span>`;
  UI.aiToast(`${whoTag}<span class="thinking">${t('toast.thinking')}</span>`, { sticky: true });

  await sleep(kind === 'heuristic' ? 350 : 150); // 让上一手动画落地
  if (!state || state.status !== 'playing' || state.turn !== color) { aiBusy = false; return; }

  const result = await requestMove(kind, state);
  if (!state || state.status !== 'playing' || state.turn !== color) { aiBusy = false; return; }

  let move;
  if (result.error) {
    UI.aiToast(`${whoTag}<span class="err">${result.error} — ${t('toast.fallback')}</span>`);
    const legal = E.legalMoves(state, color);
    move = legal.length ? legal[Math.floor(Math.random() * legal.length)] : null;
  } else {
    move = result.move;
    const note = formatMeta(result.meta);
    UI.aiToast(`${whoTag}${note ? ' · ' + note : t('toast.moved')}`);
  }

  E.applyMove(state, color, move);
  replayMoves.push({ color, move: move ? { r: move.r, c: move.c } : null, agent: kind });
  scene.syncState(state);
  aiBusy = false;
  refresh();

  // 观战/人机链式推进：下一手仍是 AI 则继续
  await sleep(600);
  maybeAI();
}

// ---------- 交互 ----------
scene.onCellClick((r, c) => {
  if (!state || aiBusy) return;
  if (state.status === 'setup') {
    if (players[state.setupChooser] !== 'human') return;
    if (E.chooseStart(state, r, c)) {
      replayMoves.push({ setup: state.setupChooser, r, c });
      scene.syncState(state);
      refresh();
      maybeAI();
    }
    return;
  }
  if (state.status === 'playing') {
    if (players[state.turn] !== 'human') return;
    const color = state.turn;
    const events = E.applyMove(state, color, { r, c });
    if (events) {
      lastHover = null;
      replayMoves.push({ color, move: { r, c }, agent: 'human' });
      scene.syncState(state);
      refresh();
      maybeAI();
    }
  }
});

scene.onCellHover((r, c) => {
  if (!state || state.status !== 'playing' || aiBusy) return;
  if (players[state.turn] !== 'human') return;
  const key = r === null ? null : r + ',' + c;
  if (key === lastHover) return;
  lastHover = key;
  scene.restoreColors(state);
  if (r === null) return;
  const legal = E.legalMoves(state, state.turn).some((m) => m.r === r && m.c === c);
  if (!legal) return;
  const phase = E.phaseFor(state, state.turn);
  scene.previewAffects(E.affectedCells(state, r, c, phase));
});

UI.initUI({
  onNewGame: (config) => startGame(config, UI.readPlayers()),
  onToggleLang: () => toggleLang(),
  onPass: () => {
    if (state && state.status === 'playing' && state.config.passAllowed
        && players[state.turn] === 'human' && !aiBusy) {
      const color = state.turn;
      E.applyMove(state, color, null);
      replayMoves.push({ color, move: null, agent: 'human' });
      scene.syncState(state);
      refresh();
      maybeAI();
    }
  },
  onBatch: (config, games) => {
    UI.setBatchRunning(true);
    setTimeout(() => {
      runBatch(config, games, UI.updateBatchProgress).then((stats) => {
        UI.setBatchRunning(false);
        UI.showBatchResult(stats);
      });
    }, 30);
  },
});

applyStatic();
document.getElementById('lang-toggle').textContent = t('lang.switch');
onLangChange(() => {
  document.getElementById('lang-toggle').textContent = t('lang.switch');
  if (state) refresh();
});

startGame(UI.readConfig(), UI.readPlayers());

// 探测后端：纯静态托管（GitHub Pages）降级为人类/启发式，不启动训练监控
detectBackend().then((ok) => {
  if (ok) {
    initTrainingMonitor();
  } else {
    UI.setStaticMode(true);
    setTrainVisible(false);
  }
});
