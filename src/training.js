// training.js —— 训练监控面板：轮询 /api/training，Canvas 手绘曲线（零依赖）

import { t, getLang, onLangChange } from './i18n.js';

const $ = (id) => document.getElementById(id);

function drawLine(canvas, rows, pick, color, opts = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth * dpr;
  const H = canvas.clientHeight * dpr;
  if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
  ctx.clearRect(0, 0, W, H);

  const pts = rows.map(pick).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
  if (pts.length < 2) {
    ctx.fillStyle = 'rgba(162,152,133,0.5)';
    ctx.font = `${11 * dpr}px sans-serif`;
    ctx.fillText(t('train.nodata'), 10 * dpr, H / 2);
    return;
  }
  let min = opts.min !== undefined ? opts.min : Math.min(...pts);
  let max = opts.max !== undefined ? opts.max : Math.max(...pts);
  if (max - min < 1e-6) { max += 0.5; min -= 0.5; }
  const pad = 6 * dpr;
  const x = (i) => pad + (i / (pts.length - 1)) * (W - 2 * pad);
  const y = (v) => H - pad - ((v - min) / (max - min)) * (H - 2 * pad);

  // 网格中线（0 或中值）
  if (opts.zeroLine && min < 0 && max > 0) {
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(W, y(0));
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6 * dpr;
  ctx.lineJoin = 'round';
  ctx.beginPath();
  pts.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.stroke();

  // 末端数值
  ctx.fillStyle = color;
  ctx.font = `${10 * dpr}px sans-serif`;
  const last = pts[pts.length - 1];
  ctx.fillText(String(typeof last === 'number' ? last.toFixed(opts.decimals ?? 2) : last), W - 44 * dpr, y(last) - 4 * dpr);
}

function drawTwoLines(canvas, rows, pickA, pickB, colorA, colorB) {
  drawLine(canvas, rows, pickA, colorA, { min: 0, max: 1, decimals: 2 });
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width;
  const H = canvas.height;
  const pts = rows.map(pickB).filter((v) => v != null);
  if (pts.length < 2) return;
  const pad = 6 * dpr;
  const x = (i) => pad + (i / (pts.length - 1)) * (W - 2 * pad);
  const y = (v) => H - pad - v * (H - 2 * pad);
  ctx.strokeStyle = colorB;
  ctx.lineWidth = 1.2 * dpr;
  ctx.beginPath();
  pts.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
  ctx.stroke();
}

function fmtEta(sec) {
  if (!isFinite(sec) || sec < 0) return '—';
  const m = Math.floor(sec / 60);
  if (getLang() === 'en') {
    return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : m >= 1 ? `${m}m` : `${Math.round(sec)}s`;
  }
  return m >= 60 ? `${Math.floor(m / 60)} 时 ${m % 60} 分` : m >= 1 ? `${m} 分` : `${Math.round(sec)} 秒`;
}

function fmtSteps(steps, total, pct) {
  if (getLang() === 'en') {
    const k = (n) => (n >= 1000 ? (n / 1000).toFixed(0) + 'k' : n);
    return `${(steps / 1000).toFixed(1)}k / ${k(total)} (${pct}%)`;
  }
  return `${(steps / 10000).toFixed(1)} 万 / ${(total / 10000).toFixed(0)} 万（${pct}%）`;
}

let timer = null;

export function setTrainVisible(visible) {
  const sec = $('train-section');
  if (sec) sec.classList.toggle('hidden', !visible);
  if (!visible && timer) {
    clearInterval(timer);
    timer = null;
  }
}

export async function pollTraining() {
  let data;
  try {
    const resp = await fetch('/api/training');
    if (!resp.ok) throw new Error('no backend');
    data = await resp.json();
  } catch {
    // 纯静态托管：整个监控区隐藏并停止轮询
    if (document.getElementById('train-section')) {
      setTrainVisible(false);
    } else {
      $('ts-status').textContent = t('train.unreachable');
    }
    return;
  }
  const rows = data.rows || [];
  const latest = rows[rows.length - 1];
  const badge = $('train-badge');

  if (!latest) {
    $('ts-status').textContent = t('train.waiting');
    badge.classList.add('hidden');
    return;
  }

  badge.classList.toggle('hidden', !data.active);
  $('ts-status').textContent = data.active ? t('train.running') : t('train.stopped');

  const pct = latest.total ? ((latest.steps / latest.total) * 100).toFixed(1) : '?';
  $('ts-steps').textContent = fmtSteps(latest.steps, latest.total, pct);
  $('ts-fps').textContent = `${latest.fps} ${getLang() === 'en' ? 'steps/s' : '步/秒'}`;
  const eta = latest.total && latest.fps ? (latest.total - latest.steps) / latest.fps : null;
  $('ts-eta').textContent = data.active && eta ? fmtEta(eta) : '—';
  const W = getLang() === 'en' ? 'W' : '胜';
  const D = getLang() === 'en' ? 'D' : '平';
  const L = getLang() === 'en' ? 'L' : '负';
  $('ts-win').textContent = latest.win_rate !== null
    ? `${W} ${(latest.win_rate * 100).toFixed(0)}% / ${D} ${(latest.draw_rate * 100).toFixed(0)}% / ${L} ${(latest.loss_rate * 100).toFixed(0)}%`
    : '—';

  drawLine($('chart-rew'), rows, (r) => r.ep_rew, '#c9a86a', { zeroLine: true, decimals: 3 });
  drawTwoLines($('chart-win'), rows, (r) => r.win_rate, (r) => r.draw_rate, '#8fae7c', '#9b9184');
  drawLine($('chart-loss'), rows, (r) => r.value_loss, '#a3806e', { decimals: 4 });
}

export function initTrainingMonitor(intervalMs = 3000) {
  pollTraining();
  timer = setInterval(pollTraining, intervalMs);
  onLangChange(() => pollTraining());
}
