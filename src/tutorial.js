// tutorial.js —— 首访教学卡 + 演示对局解说引擎
// 演示对局由 main.js 驱动（双启发式自动走子），本模块负责：
// 首访弹层、解说条渲染、按局面事件挑选教学文案（带冷却，不刷屏）

import { t, onLangChange } from './i18n.js';

const $ = (id) => document.getElementById(id);
const SEEN_KEY = 'tc-seen';

let demoActive = false;
let handlers = { onExitDemo: null };

// 解说状态（每局演示重置）
let taught = null;
function resetNarration() {
  taught = { cross: 0, x: 0, closeShown: false, escapeShown: false, lastLead: null };
}

export function isDemo() {
  return demoActive;
}

// ---------- 欢迎卡 ----------
export function bindWelcome(onWatchDemo) {
  $('btn-watch-demo').addEventListener('click', () => {
    hideWelcome();
    localStorage.setItem(SEEN_KEY, '1');
    onWatchDemo();
  });
  $('btn-play-now').addEventListener('click', () => {
    hideWelcome();
    localStorage.setItem(SEEN_KEY, '1');
  });
  $('demo-exit').addEventListener('click', () => {
    if (handlers.onExitDemo) handlers.onExitDemo();
  });
  onLangChange(() => {
    // 切换语言时若解说条开着，刷新为通用提示（事件解说随下一手更新）
    if (demoActive) $('demo-caption-text').textContent = t('demo.intro');
  });
}

export function maybeWelcome() {
  if (localStorage.getItem(SEEN_KEY)) return false;
  $('welcome-overlay').classList.remove('hidden');
  return true;
}

function hideWelcome() {
  $('welcome-overlay').classList.add('hidden');
}

// ---------- 演示模式 ----------
export function setDemo(on, onExit) {
  demoActive = on;
  handlers.onExitDemo = on ? onExit : null;
  if (on) {
    resetNarration();
    hideWelcome();
    $('demo-caption').classList.remove('hidden');
    $('demo-caption-text').textContent = t('demo.intro');
    $('demo-exit').textContent = t('demo.exit');
    $('hint').style.visibility = 'hidden'; // 操作提示让位给解说条
  } else {
    $('demo-caption').classList.add('hidden');
    $('hint').style.visibility = '';
  }
}

// ---------- 解说引擎 ----------
// 每手落定后由 main.js 调用。events 为 applyMove 返回的事件数组。
export function narrate(state, events, moverColor) {
  if (!demoActive) return;
  const set = (key) => { $('demo-caption-text').textContent = t(key); };

  // 终局：最高优先级
  if (state.status === 'over') {
    if (state.winner === 'white') set('demo.endWhite');
    else if (state.winner === 'black') set('demo.endBlack');
    else set('demo.endDraw');
    return;
  }

  const wh = state.white;
  const bk = state.black;
  const dist = Math.max(Math.abs(wh.r - bk.r), Math.abs(wh.c - bk.c));
  const goalDist = state.size - wh.r; // 默认胜利边为南（最后一行）

  // 高度翻转：黑从"不够高"变"够高"（或反之）——教学价值高
  // pillarHeight 由 main.js 通过事件附带，避免循环依赖
  if (events.heightLead) {
    if (events.heightLead === 'black' && taught.lastLead !== 'black') {
      taught.lastLead = 'black';
      set('demo.heightBlack');
      return;
    }
    if (events.heightLead === 'white' && taught.lastLead !== 'white') {
      taught.lastLead = 'white';
      set('demo.heightWhite');
      return;
    }
  }

  // 距离逼近（一次性）
  if (!taught.closeShown && dist <= 2) {
    taught.closeShown = true;
    set('demo.close');
    return;
  }

  // 白方接近胜利边（一次性）
  if (!taught.escapeShown && goalDist <= 3) {
    taught.escapeShown = true;
    set('demo.escape');
    return;
  }

  // 相位教学：前两手各讲一次
  if (events.phase === 'cross' && taught.cross < 2) {
    taught.cross += 1;
    set('demo.cross');
    return;
  }
  if (events.phase === 'x' && taught.x < 2) {
    taught.x += 1;
    set('demo.xphase');
    return;
  }
  // 其余手保持上一条解说（教学信息密度已够，不刷屏）
}
