// server.js —— 涌陆（Terraflux）服务器：静态文件 + 智能体决策 API + 棋谱保存
// 零 npm 依赖（只用 Node 内置模块）。用法：node server.js
//
// API:
//   POST /api/agent   { agent: 'heuristic'|'llm'|'ppo', state: <序列化局面> }
//                     → { move: {r,c} | null, meta? }
//   POST /api/replay  { config, players, moves, result } → 保存到 replays/
//
// 配置文件 config.json（可选，见 config.example.json）：
//   llm: OpenAI 兼容接口的 baseURL / apiKey / model
//   ppo: 本地 PPO 模型服务地址（默认 http://127.0.0.1:8765）

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as E from './src/engine.js';
import { chooseMove } from './src/ai.js';

const root = path.dirname(fileURLToPath(import.meta.url));

// ---------- 配置 ----------
function loadConfig() {
  try {
    return JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf-8'));
  } catch {
    return {};
  }
}
const appConfig = loadConfig();
const LLM_CFG = appConfig.llm || null;
const PPO_URL = (appConfig.ppo && appConfig.ppo.url) || 'http://127.0.0.1:8765';

// PPO 模型支持的棋盘尺寸列表（启动时从 PPO 服务探测，用于给出友好报错）
let PPO_SIZES = [];
fetch(PPO_URL.replace(/\/$/, '') + '/health')
  .then((r) => r.json())
  .then((d) => { PPO_SIZES = d.sizes || (d.size ? [d.size] : []); })
  .catch(() => { /* 服务未启动时保持空 */ });

// ---------- 局面序列化/还原 ----------
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

export function deserializeState(raw) {
  return {
    ...raw,
    pillars: Uint8Array.from(raw.pillars),
    white: raw.white && { ...raw.white },
    black: raw.black && { ...raw.black },
    moveCount: { ...raw.moveCount },
    repetition: new Map(),
    setupOptions: null,
    setupChooser: null,
  };
}

// ---------- LLM 局面描述（给大语言模型看的局面摘要） ----------
const DIR8 = [
  { dr: -1, dc: 0, name: '北' }, { dr: 1, dc: 0, name: '南' },
  { dr: 0, dc: -1, name: '西' }, { dr: 0, dc: 1, name: '东' },
  { dr: -1, dc: -1, name: '西北' }, { dr: -1, dc: 1, name: '东北' },
  { dr: 1, dc: -1, name: '西南' }, { dr: 1, dc: 1, name: '东南' },
];
const TERRAIN_NAME = ['正常', '升高(+1)', '降低(-1)', '填平(不可进入)'];

// 两层前瞻安全判定：我走完 m 后，对方是否存在下一手直接获胜
// 返回 { level: 'win'|'lose'|'draw'|'danger'|'safe', killMove? }
function moveSafety(state, color, m) {
  const opp = color === 'white' ? 'black' : 'white';
  const s2 = E.cloneState(state);
  E.applyMove(s2, color, m);
  if (s2.status === 'over') {
    if (s2.winner === color) return { level: 'win' };
    if (s2.winner === 'draw') return { level: 'draw' };
    return { level: 'lose' };
  }
  for (const om of E.legalMoves(s2, opp)) {
    const s3 = E.cloneState(s2);
    E.applyMove(s3, opp, om);
    if (s3.status === 'over' && s3.winner === opp) return { level: 'danger', killMove: om };
  }
  return { level: 'safe' };
}

const SAFETY_TAG = {
  win: '【★直接获胜】',
  lose: '【✗自杀：走这步直接输】',
  draw: '【=此步和局】',
  danger: null, // 动态生成
  safe: '',
};

// 到胜利边的最少步数（切比雪夫意义下的边距）
function goalDist(state, pos) {
  const s = state.size;
  const ds = [];
  for (const e of state.config.goalEdges) {
    if (e === 'N') ds.push(pos.r - 1);
    if (e === 'S') ds.push(s - pos.r);
    if (e === 'W') ds.push(pos.c - 1);
    if (e === 'E') ds.push(s - pos.c);
  }
  return ds.length ? Math.min(...ds) : 0;
}

const EDGE_NAME = { N: '北（减小行号）', S: '南（增大行号）', W: '西（减小列号）', E: '东（增大列号）' };

function describeMoveForLLM(state, color, m, safety) {
  const s2 = E.cloneState(state);
  const events = E.applyMove(s2, color, m);
  const opp = color === 'white' ? 'black' : 'white';
  const meBefore = state[color];
  const meAfter = s2[color];
  const oppAfter = s2[opp];
  const parts = [];
  if (safety) {
    const tag = safety.level === 'danger'
      ? `【⚠危险：对方下一手走(${safety.killMove.r},${safety.killMove.c})即可获胜】`
      : SAFETY_TAG[safety.level];
    if (tag) parts.push(tag);
  }
  const phase = events.phase === 'x' ? 'X相位' : events.phase === 'all8' ? '八向' : '十字相位';
  const changes = events.changes.map((ch) => `(${ch.r},${ch.c})→${TERRAIN_NAME[ch.to]}`).join('，');
  parts.push(`落在(${m.r},${m.c})，触发${phase}${changes ? '：' + changes : '（无地形变化）'}`);
  const distBefore = Math.max(Math.abs(meBefore.r - state[opp].r), Math.abs(meBefore.c - state[opp].c));
  const dist = Math.max(Math.abs(meAfter.r - oppAfter.r), Math.abs(meAfter.c - oppAfter.c));
  parts.push(`；与对方距离 ${distBefore}→${dist}`);
  if (color === 'white') parts.push(`，距胜利边 ${goalDist(state, meBefore)}→${goalDist(state, meAfter)} 步`);
  const myH = E.pillarHeight(s2, meAfter.r, meAfter.c);
  const oppH = E.pillarHeight(s2, oppAfter.r, oppAfter.c);
  if (dist === 1) parts.push(`；落地后与对方相邻（我方高度${myH}，对方高度${oppH}）`);
  if (s2.status === 'over') {
    parts.push(s2.winner === color ? '；此步直接获胜！' : s2.winner === 'draw' ? '；此步导致平局' : '；警告：此步导致我方落败！');
  } else if (color === 'white' && E.isOnGoal(s2, meAfter)) {
    parts.push('；此步抵达胜利边！');
  }
  return parts.join('');
}

export function buildLLMPrompt(state, color) {
  const role = color === 'white' ? '逃离者（白球）' : '追捕者（黑球）';
  const opp = color === 'white' ? 'black' : 'white';
  const me = state[color];
  const op = state[opp];
  const phase = E.phaseFor(state, color);
  const captureRule = state.config.captureRule === 'gte' ? '黑高度≥白高度（同高即捕）' : '黑高度必须严格高于白（>）';
  const moves = E.legalMoves(state, color);
  const safetyMap = new Map(moves.map((m) => [m.r + ',' + m.c, moveSafety(state, color, m)]));
  const edgesCN = state.config.goalEdges.map((e) => `${e}=${EDGE_NAME[e]}`).join('、');
  const lines = [
    `你正在下一盘"涌陆"（Terraflux），你执${role}。`,
    ``,
    `【坐标系】棋盘 ${state.size}×${state.size}。位置写作 (行,列)：行号从北边 1 递增到南边 ${state.size}，列号从西边 1 递增到东边 ${state.size}。`,
    `方向与坐标的关系：北=行-1，南=行+1，西=列-1，东=列+1，东北=(行-1,列+1)，以此类推。胜利边方位：${edgesCN}。`,
    ``,
    `【规则速览】每格是一根顶面带凹槽的柱子（状态：正常/升高+1/降低-1/填平不可入）。`,
    `每步移动到相邻8格之一；落子触发地形变化——十字相位：北邻升高、南邻降低、东西邻填平；X相位：东北/西北邻升高、东南/西南邻降低。地形变化会带着站在上面的球一起动。`,
    `黑球获胜条件：与白球8方向相邻，且${captureRule}。白球获胜条件：抵达胜利边（${state.config.goalEdges.join('/')}）且未被擒；抵达边界同时被擒为平局。`,
    ``,
    `【当前局面】第 ${Math.floor((state.moveCount.white + state.moveCount.black) / 2) + 1} 轮，你这一步的相位：${phase === 'x' ? 'X相位' : phase === 'all8' ? '八向' : '十字相位'}`,
    `白球位置 (${state.white.r},${state.white.c})，脚下高度 ${E.pillarHeight(state, state.white.r, state.white.c)}，距胜利边 ${goalDist(state, state.white)} 步`,
    `黑球位置 (${state.black.r},${state.black.c})，脚下高度 ${E.pillarHeight(state, state.black.r, state.black.c)}`,
    `你在 (${me.r},${me.c})，对方在 (${op.r},${op.c})，切比雪夫距离 ${Math.max(Math.abs(me.r - op.r), Math.abs(me.c - op.c))}`,
    ``,
    `【合法落点及后果预览】（坐标为 行,列；已帮你算好每步的距离变化，并标注了安全等级——标 ✗ 和 ⚠ 的不要选）`,
    ...moves.map((m) => `- ${describeMoveForLLM(state, color, m, safetyMap.get(m.r + ',' + m.c))}`),
    ``,
    `思考要点：${color === 'white' ? '优先缩短与胜利边的距离，同时避免走入被对方贴身的格子；注意落子触发的地形变化可能抬高/降低你或对方脚下。' : '优先缩短与对方的切比雪夫距离；贴身前确保自己脚下高度严格高于对方，否则无法捕捉；可以利用地形变化抬高自己或压低对方。'}`,
    `请只输出 JSON：{"r": 行号, "c": 列号}，并在 reasoning 字段用一两句话说明理由：{"r":…,"c":…,"reasoning":"…"}`,
  ];
  return lines.join('\n');
}

// ---------- 智能体决策 ----------
async function llmMove(state) {
  if (!LLM_CFG || !LLM_CFG.apiKey) {
    return { move: null, error: '未配置 LLM：请复制 config.example.json 为 config.json 并填入 apiKey' };
  }
  const color = state.turn;
  const prompt = buildLLMPrompt(state, color);
  // 混合架构的安全网：预先算好每个候选的两层前瞻安全等级
  const moves = E.legalMoves(state, color);
  const safetyMap = new Map(moves.map((m) => [m.r + ',' + m.c, moveSafety(state, color, m)]));
  const body = {
    model: LLM_CFG.model || 'gpt-4o',
    messages: [
      { role: 'system', content: '你是一名涌陆（Terraflux）棋手。只输出 JSON，不要输出任何其他内容。' },
      { role: 'user', content: prompt },
    ],
    temperature: 0.4,
    // extraBody 透传平台自定义参数（如通义 enable_thinking、Kimi K3 reasoning_effort）
    ...(LLM_CFG.extraBody || {}),
  };
  const resp = await fetch(LLM_CFG.baseURL.replace(/\/$/, '') + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + LLM_CFG.apiKey },
    body: JSON.stringify(body),
  });
  if (!resp.ok) return { move: null, error: 'LLM 接口错误：HTTP ' + resp.status };
  const data = await resp.json();
  const text = data.choices?.[0]?.message?.content || '';
  const jsonMatch = text.match(/\{[^{}]*"r"[^{}]*\}/);
  if (!jsonMatch) return { move: null, error: 'LLM 未返回合法 JSON', raw: text };
  let parsed;
  try { parsed = JSON.parse(jsonMatch[0]); } catch { return { move: null, error: 'LLM JSON 解析失败', raw: text }; }
  const move = { r: Number(parsed.r), c: Number(parsed.c) };
  const legal = E.legalMoves(state, color).some((m) => m.r === move.r && m.c === move.c);
  if (!legal) return { move: null, error: 'LLM 选择了非法落点', raw: text };

  // 引擎否决制：LLM 漏看必杀 / 走了送死步或给对方留必赢着时，引擎接管
  const picked = safetyMap.get(move.r + ',' + move.c);
  const winEntry = [...safetyMap.entries()].find(([, v]) => v.level === 'win');
  if (winEntry && picked.level !== 'win') {
    const [wk] = winEntry;
    const [wr, wc] = wk.split(',').map(Number);
    return {
      move: { r: wr, c: wc },
      meta: { reasoning: `⚡ LLM 原选 (${move.r},${move.c})，但引擎发现 (${wr},${wc}) 可直接获胜，已改走必杀。LLM 原理由：${parsed.reasoning || '无'}`, vetoed: true },
    };
  }
  if (picked.level === 'lose' || picked.level === 'danger') {
    const safe = moves.filter((m) => {
      const lv = safetyMap.get(m.r + ',' + m.c).level;
      return lv === 'safe' || lv === 'draw';
    });
    if (safe.length > 0) {
      const fallback = chooseMove(state, color, 0, safe);
      const why = picked.level === 'lose' ? '走这步直接输' : `对方下一手走(${picked.killMove.r},${picked.killMove.c})即可获胜`;
      return {
        move: { r: fallback.r, c: fallback.c },
        meta: { reasoning: `🛡️ LLM 原选 (${move.r},${move.c})，引擎判定危险（${why}），已否决改走 (${fallback.r},${fallback.c})。LLM 原理由：${parsed.reasoning || '无'}`, vetoed: true },
      };
    }
    // 所有走法都危险：尊重 LLM 的选择（横竖都是输，让它选个体面的）
  }
  return { move, meta: { reasoning: parsed.reasoning || '' } };
}

async function ppoMove(state) {
  if (PPO_SIZES.length > 0 && !PPO_SIZES.includes(state.size)) {
    return { move: null, error: `PPO 暂不支持 ${state.size}×${state.size} 棋盘，当前可用尺寸：${PPO_SIZES.join('、')}。（25×25 模型正在训练中，完成后自动可用）` };
  }
  try {
    const resp = await fetch(PPO_URL.replace(/\/$/, '') + '/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: serializeState(state) }),
    });
    if (!resp.ok) {
      let detail = '';
      try { detail = (await resp.json()).error || ''; } catch { /* 忽略解析失败 */ }
      return { move: null, error: 'PPO 服务错误：HTTP ' + resp.status + (detail ? '（' + detail + '）' : '') };
    }
    const data = await resp.json();
    if (data.move === null) return { move: null };
    const move = { r: Number(data.move.r), c: Number(data.move.c) };
    const legal = E.legalMoves(state, state.turn).some((m) => m.r === move.r && m.c === move.c);
    if (!legal) return { move: null, error: 'PPO 返回非法落点' };
    return { move, meta: data.meta };
  } catch (e) {
    return { move: null, error: '连不上 PPO 服务（' + PPO_URL + '）：请先运行 python/serve_ppo.py' };
  }
}

async function agentMove(agent, state) {
  if (agent === 'heuristic') return { move: chooseMove(state, state.turn, 0.4) };
  if (agent === 'llm') return llmMove(state);
  if (agent === 'ppo') return ppoMove(state);
  return { move: null, error: '未知智能体类型: ' + agent };
}

// ---------- HTTP ----------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
};

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => { data += c; if (data.length > 5e6) req.destroy(); });
    req.on('end', () => { try { resolve(JSON.parse(data || '{}')); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

function sendJSON(res, obj, code = 200) {
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Access-Control-Allow-Origin': '*' });
  res.end(JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];

  if (req.method === 'POST' && url === '/api/agent') {
    try {
      const body = await readBody(req);
      const state = deserializeState(body.state);
      const result = await agentMove(body.agent, state);
      sendJSON(res, result);
    } catch (e) {
      sendJSON(res, { move: null, error: String(e && e.message || e) }, 400);
    }
    return;
  }

  if (req.method === 'POST' && url === '/api/replay') {
    try {
      const body = await readBody(req);
      const dir = path.join(root, 'replays');
      if (!fs.existsSync(dir)) fs.mkdirSync(dir);
      const name = new Date().toISOString().replace(/[:.]/g, '-') + '.json';
      fs.writeFileSync(path.join(dir, name), JSON.stringify(body, null, 2));
      sendJSON(res, { ok: true, file: name });
    } catch (e) {
      sendJSON(res, { ok: false, error: String(e && e.message || e) }, 400);
    }
    return;
  }

  if (req.method === 'GET' && url === '/api/health') {
    sendJSON(res, { ok: true, llm: !!(LLM_CFG && LLM_CFG.apiKey), ppoUrl: PPO_URL });
    return;
  }

  // 训练监控：读取 python 训练进程写的 training_log.jsonl
  if (req.method === 'GET' && url === '/api/training') {
    try {
      const file = path.join(root, 'training_log.jsonl');
      if (!fs.existsSync(file)) return sendJSON(res, { active: false, rows: [] });
      const stat = fs.statSync(file);
      const lines = fs.readFileSync(file, 'utf-8').trim().split('\n').filter(Boolean);
      const tail = lines.slice(-800).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
      sendJSON(res, {
        active: Date.now() - stat.mtimeMs < 20000, // 20 秒内有写入视为训练中
        updatedAt: stat.mtimeMs,
        rows: tail,
      });
    } catch (e) {
      sendJSON(res, { active: false, rows: [], error: String(e && e.message || e) });
    }
    return;
  }

  if (req.method === 'OPTIONS') {
    res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Allow-Methods': 'POST,GET' });
    return res.end();
  }

  // 静态文件
  if (req.method !== 'GET') { res.writeHead(405); return res.end(); }
  let p = decodeURIComponent(url);
  if (p === '/') p = '/index.html';
  const file = path.join(root, p);
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not Found: ' + p); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream' });
    res.end(data);
  });
});

const PORT = process.env.PORT || 5173;
server.listen(PORT, () => {
  console.log('');
  console.log('  涌陆（Terraflux）已启动：  http://localhost:' + PORT);
  console.log('  LLM 智能体：  ' + (LLM_CFG && LLM_CFG.apiKey ? '已配置（' + (LLM_CFG.model || 'gpt-4o') + '）' : '未配置（复制 config.example.json 为 config.json 并填入 apiKey）'));
  console.log('  PPO 服务地址：' + PPO_URL + '（训练后运行 python/serve_ppo.py）');
  console.log('');
});
