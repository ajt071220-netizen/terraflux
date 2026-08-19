// ui.js —— 面板、HUD 与结算横幅（动态文案经 i18n）

import { phaseFor, pillarHeight } from './engine.js';
import { t } from './i18n.js';

const $ = (id) => document.getElementById(id);

export function readConfig() {
  const goalEdges = [...document.querySelectorAll('#cfg-goal input:checked')].map((i) => i.value);
  return {
    size: parseInt($('cfg-size').value, 10),
    startLayout: $('cfg-layout').value,
    gap: parseInt($('cfg-gap').value, 10),
    firstPlayer: $('cfg-first').value,
    goalEdges: goalEdges.length > 0 ? goalEdges : ['S'],
    captureRule: $('cfg-capture').value,
    phaseMode: $('cfg-phase').value,
    passAllowed: $('cfg-pass').checked,
    repetitionDraw: $('cfg-repetition').checked,
    stalemateLoses: true,
  };
}

// 双方玩家类型：'human' | 'heuristic' | 'llm' | 'ppo'
export function readPlayers() {
  return {
    white: $('cfg-player-white').value,
    black: $('cfg-player-black').value,
  };
}

// AI 思考/出错提示
export function aiToast(text, opts = {}) {
  const el = $('ai-toast');
  if (!text) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');
  el.innerHTML = text;
  if (opts.sticky) clearTimeout(el._timer);
  else {
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.add('hidden'), 6000);
  }
}

export function setPanelOpen(open) {
  document.body.classList.toggle('panel-open', open);
}

// 纯静态托管（GitHub Pages）降级：禁用依赖后端的 LLM/PPO 选项
export function setStaticMode(on) {
  for (const id of ['cfg-player-white', 'cfg-player-black']) {
    const sel = $(id);
    for (const opt of sel.options) {
      if (opt.value === 'llm' || opt.value === 'ppo') opt.disabled = on;
    }
    if (on && (sel.value === 'llm' || sel.value === 'ppo')) sel.value = 'heuristic';
  }
  const note = $('offline-note');
  if (note) note.classList.toggle('hidden', !on);
}

export function initUI(handlers) {
  $('btn-new').addEventListener('click', () => {
    handlers.onNewGame(readConfig());
    setPanelOpen(false); // 开局后收起抽屉，棋盘是主角
  });
  $('btn-again').addEventListener('click', () => handlers.onNewGame(readConfig()));
  $('btn-pass').addEventListener('click', () => handlers.onPass());
  $('btn-batch').addEventListener('click', () => {
    const games = parseInt($('batch-games').value, 10);
    handlers.onBatch(readConfig(), games);
  });
  $('lang-toggle').addEventListener('click', () => handlers.onToggleLang());
  $('panel-toggle').addEventListener('click', () => setPanelOpen(true));
  $('panel-close').addEventListener('click', () => setPanelOpen(false));
  $('panel-scrim').addEventListener('click', () => setPanelOpen(false));
}

export function updateHUD(state) {
  const turnEl = $('hud-turn');
  const round = Math.floor((state.moveCount.white + state.moveCount.black) / 2) + 1;
  $('hud-round').textContent = round;

  if (state.status === 'setup') {
    turnEl.className = 'hud-item ' + state.setupChooser;
    turnEl.querySelector('span').textContent =
      state.setupChooser === 'white' ? t('hud.setupWhite') : t('hud.setupBlack');
    $('hud-phase').textContent = '—';
    $('hud-heights').textContent = '—';
    return;
  }

  const isWhite = state.turn === 'white';
  turnEl.className = 'hud-item ' + state.turn;
  turnEl.querySelector('span').textContent =
    state.status === 'over'
      ? t('hud.gameOver')
      : isWhite
        ? t('hud.turnWhite')
        : t('hud.turnBlack');

  if (state.status === 'over') {
    $('hud-phase').textContent = '—';
  } else {
    const phase = phaseFor(state, state.turn);
    $('hud-phase').textContent =
      phase === 'cross' ? t('phase.cross') : phase === 'x' ? t('phase.x') : t('phase.all8');
  }

  if (state.white && state.black) {
    const wh = pillarHeight(state, state.white.r, state.white.c);
    const bh = pillarHeight(state, state.black.r, state.black.c);
    const fmt = (h) => (h > 0 ? '+' + h : String(h));
    $('hud-heights').innerHTML =
      `<em>${t('hud.height')}</em> <b>W ${fmt(wh)} / B ${fmt(bh)}</b>`;
  }
}

export function setHint(text) {
  $('hint').textContent = text;
}

export function showBanner(state) {
  const title = $('banner-title');
  if (state.winner === 'white') title.textContent = t('banner.whiteWin');
  else if (state.winner === 'black') title.textContent = t('banner.blackWin');
  else title.textContent = t('banner.draw');
  $('banner-reason').textContent = t('reason.' + state.winReason);
  $('banner').classList.remove('hidden');
}

export function hideBanner() {
  $('banner').classList.add('hidden');
}

export function setPassVisible(visible) {
  $('btn-pass').classList.toggle('hidden', !visible);
}

export function setBatchRunning(running) {
  $('btn-batch').disabled = running;
  $('btn-batch').textContent = running ? t('batch.running') : t('batch.run');
  $('batch-progress').classList.toggle('hidden', !running);
  if (!running) $('batch-progress').querySelector('i').style.width = '0%';
}

export function updateBatchProgress(stats) {
  $('batch-progress').querySelector('i').style.width = ((stats.done / stats.total) * 100).toFixed(1) + '%';
}

export function showBatchResult(stats) {
  const pct = (x) => ((x / stats.done) * 100).toFixed(1);
  const avgTurns = (stats.turns / stats.done).toFixed(1);
  $('batch-result').innerHTML = `
    <div class="bar">
      <i class="w" style="width:${pct(stats.white)}%"></i>
      <i class="b" style="width:${pct(stats.black)}%"></i>
      <i class="d" style="width:${pct(stats.draw)}%"></i>
    </div>
    <div class="row"><span>${t('batch.white')} ${pct(stats.white)}%</span><span>${stats.white}</span></div>
    <div class="row"><span>${t('batch.black')} ${pct(stats.black)}%</span><span>${stats.black}</span></div>
    <div class="row"><span>${t('batch.draw')} ${pct(stats.draw)}%</span><span>${stats.draw}</span></div>
    <div class="row"><span>${t('batch.avgTurns')}</span><span>${avgTurns}</span></div>
  `;
}
