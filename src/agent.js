// agent.js —— 前端 AI 调度：内置启发式本地算，LLM/PPO 走服务器 /api/agent
import * as E from './engine.js';
import { chooseMove } from './ai.js';

export const AGENT_KINDS = [
  { id: 'human' },
  { id: 'heuristic' },
  { id: 'llm' },
  { id: 'ppo' },
];

// 后端探测：GitHub Pages 等纯静态托管上没有 /api/*，降级为人类/启发式
let backendOk = null;

export async function detectBackend() {
  if (backendOk !== null) return backendOk;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 3000);
    const resp = await fetch('/api/health', { signal: ctrl.signal });
    clearTimeout(timer);
    backendOk = resp.ok;
  } catch {
    backendOk = false;
  }
  return backendOk;
}

export function isBackendOk() {
  return backendOk !== false;
}

export function serializeState(state) {
  return {
    config: state.config,
    size: state.size,
    pillars: Array.from(state.pillars),
    white: state.white,
    black: state.black,
    turn: state.turn,
    moveCount: state.moveCount,
    passes: state.passes,
    status: state.status,
    winner: state.winner,
    winReason: state.winReason,
  };
}

// 返回 { move, meta } 或 { move: null, error }
export async function requestMove(kind, state) {
  if (kind === 'human') return { move: null, error: 'human 不需要决策' };
  if (kind === 'heuristic') {
    return { move: chooseMove(state, state.turn, 0.4) };
  }
  try {
    const resp = await fetch('/api/agent', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent: kind, state: serializeState(state) }),
    });
    return await resp.json();
  } catch (e) {
    return { move: null, error: '服务器不可达（LLM/PPO 需要通过 node server.js 启动）' };
  }
}

// setup 阶段：AI 随机选一个候选起点
export function chooseSetup(state) {
  const opts = state.setupOptions || [];
  if (!opts.length) return null;
  return opts[Math.floor(Math.random() * opts.length)];
}

export async function saveReplay(record) {
  try {
    await fetch('/api/replay', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record),
    });
  } catch { /* 无服务器时静默失败 */ }
}

// 从局面提取可解释的 LLM 决策备注（如果服务器返回了 reasoning）
export function formatMeta(meta) {
  if (!meta) return '';
  if (meta.reasoning) return meta.reasoning;
  return '';
}
