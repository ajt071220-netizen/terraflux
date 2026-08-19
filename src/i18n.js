// i18n.js —— 中英双语：字典 + 切换 + 静态标注应用
// 用法：HTML 里 <span data-i18n="setup.title"></span>；JS 动态文案用 t('key')

const DICTS = {
  zh: {
    'app.title': '涌陆',
    'app.subtitle': 'TERRAFLUX',

    'hud.round': '轮次',
    'hud.phase': '相位',
    'hud.height': '高度',
    'hud.setupWhite': '布置：白方选起点',
    'hud.setupBlack': '布置：黑方选起点',
    'hud.gameOver': '对局结束',
    'hud.turnWhite': '白方 · 逃离者',
    'hud.turnBlack': '黑方 · 追捕者',
    'phase.cross': '十字',
    'phase.x': 'X 相位',
    'phase.all8': '八向',

    'setup.title': '对局设置',
    'setup.white': '白 · 逃离者',
    'setup.black': '黑 · 追捕者',
    'agent.human': '人类',
    'agent.heuristic': '内置 AI',
    'agent.llm': 'LLM',
    'agent.ppo': 'PPO',
    'setup.size': '棋盘尺寸',
    'setup.layout': '开局布局',
    'setup.layout.duel': '对垒式 · 黑镇中央，白正北对峙（实测最平衡）',
    'setup.layout.blockade': '堵截式 · 黑镇中央，白后方选择',
    'setup.layout.classic': '经典式 · 白居中央，黑八向选择',
    'setup.gap': '开局间距',
    'setup.gap2': '隔一格（距离 2）',
    'setup.gap3': '隔两格（距离 3）',
    'setup.first': '先手',
    'setup.firstWhite': '逃离者（白）先行',
    'setup.firstBlack': '追捕者（黑）先行',
    'setup.offline': '在线版仅含内置 AI —— LLM / PPO 对局请按 README 本地运行',

    'rules.title': '规则旋钮',
    'rules.goal': '逃离胜利边（可多选）',
    'rules.edgeN': '北（第 1 行）',
    'rules.edgeS': '南（最后一行）',
    'rules.edgeW': '西（第 1 列）',
    'rules.edgeE': '东（最后一列）',
    'rules.capture': '抓捕条件',
    'rules.captureGt': '严格高于（黑高 > 白高）',
    'rules.captureGte': '同高即捕（黑高 ≥ 白高）',
    'rules.phase': '相位循环',
    'rules.phaseSync': '十字 / X 双方同步交替',
    'rules.phaseOff': '恒定十字（原版）',
    'rules.phaseAll8': '每步八向齐变',
    'rules.pass': '允许停步（放弃移动）',
    'rules.repetition': '局面三次重复判和',

    'action.new': '开始新对局',
    'action.pass': '停步',
    'action.again': '再来一局',

    'train.title': '训练监控',
    'train.live': '训练中',
    'train.status': '状态',
    'train.progress': '进度',
    'train.speed': '速度',
    'train.eta': '预计剩余',
    'train.winrate': '近100局胜率',
    'train.idle': '未启动',
    'train.unreachable': '服务器不可达',
    'train.waiting': '等待训练启动',
    'train.running': '训练中',
    'train.stopped': '已结束/暂停',
    'train.nodata': '等待数据…',
    'train.legendRew': '回报曲线',
    'train.legendWin': '胜率(绿)/平局(灰)',
    'train.legendLoss': '价值损失',
    'train.tb': '打开专业训练面板（TensorBoard）↗',
    'train.tbNote': '完整指标：奖励/损失/熵/KL/自定义胜率（terrain/*）',

    'batch.title': '自动对局 · 胜率测试',
    'batch.games': '对局数',
    'batch.run': '用当前配置跑胜率',
    'batch.running': '模拟中……',
    'batch.white': '白（逃离）',
    'batch.black': '黑（追捕）',
    'batch.draw': '平局',
    'batch.avgTurns': '平均回合数',
    'batch.gamesUnit': '局',

    'banner.whiteWin': '逃离者胜利',
    'banner.blackWin': '追捕者胜利',
    'banner.draw': '平局',
    'reason.capture': '追捕成功',
    'reason.escape': '成功抵达边界',
    'reason.border-capture': '抵达边界的同时被追上',
    'reason.stalemate': '无棋可走',
    'reason.repetition': '局面三次重复',
    'reason.mutual-pass': '双方连续停步',

    'hint.setupWhite': '白方（逃离者）：点击金色圆圈，选择起始位置',
    'hint.setupBlack': '黑方（追捕者）：点击金色圆圈，选择起始位置',
    'hint.choosing': '正在选择起点……',
    'hint.move': '行动：点击发光圆圈移动',
    'hint.thinking': '思考中',
    'hint.phaseCross': '十字相位（北升·南降·东西填平）',
    'hint.phaseX': 'X 相位（东北/西北升·东南/西南降）',
    'hint.phaseAll8': '八向齐变',
    'hint.over': '对局结束',
    'hint.white': '白方',
    'hint.black': '黑方',

    'toast.thinking': '思考中……',
    'toast.moved': '已落子',
    'toast.fallback': '本手随机落子',

    'lang.switch': 'EN',

    // ---- 教学引导 ----
    'tut.tagline': '在一方会自我改写的棋盘上追捕与逃脱。',
    'tut.goal.title': '目标',
    'tut.goal.body': '白球逃到金色南边线即胜；黑球要在相邻格、且站得更高才能抓捕。',
    'tut.terrain.title': '活体地形',
    'tut.terrain.body': '每步落子改写落点周围：北邻隆起、南邻下陷、东西填平——与斜向相位交替。',
    'tut.capture.title': '高度法则',
    'tut.capture.body': '棋子随柱子升降。抓捕的唯一条件：黑严格高于白。同高不算。',
    'tut.watch': '▶ 观看演示对局',
    'tut.play': '直接开玩',
    'tut.again': '再看一遍演示',

    'demo.intro': '白球要逃到金线，黑球要追捕——但抓捕需要站在更高处。注意看：地形随每一步流动。',
    'demo.cross': '十字相位：落点北邻隆起、南邻下陷、东西两邻被填平（不可通行）。',
    'demo.xphase': '斜向相位：东北/西北隆起、东南/西北下陷——本相位不填平。',
    'demo.heightBlack': '黑方登顶成功！黑高 > 白高，抓捕条件已满足。',
    'demo.heightWhite': '白方抢到了高地——黑方此刻无法抓捕。',
    'demo.close': '黑方逼近到两格之内，白方危险了！',
    'demo.escape': '白方离金色胜利线只剩几步……',
    'demo.endWhite': '白球踏上金线——逃脱成功！',
    'demo.endBlack': '抓捕！黑方在相邻格站得更高。',
    'demo.endDraw': '逃脱与抓捕同一瞬间发生——平局。',
    'demo.exit': '退出演示',
    'demo.done': '演示结束。轮到你上手了——',
    'demo.playNow': '开始我的对局',
  },

  en: {
    'app.title': 'Terraflux',
    'app.subtitle': '涌陆',

    'hud.round': 'Round',
    'hud.phase': 'Phase',
    'hud.height': 'Height',
    'hud.setupWhite': 'Setup: White picks a start',
    'hud.setupBlack': 'Setup: Black picks a start',
    'hud.gameOver': 'Game Over',
    'hud.turnWhite': 'White · Runner',
    'hud.turnBlack': 'Black · Chaser',
    'phase.cross': 'Cross',
    'phase.x': 'X',
    'phase.all8': 'All-8',

    'setup.title': 'Match Setup',
    'setup.white': 'White · Runner',
    'setup.black': 'Black · Chaser',
    'agent.human': 'Human',
    'agent.heuristic': 'Built-in AI',
    'agent.llm': 'LLM',
    'agent.ppo': 'PPO',
    'setup.size': 'Board Size',
    'setup.layout': 'Opening Layout',
    'setup.layout.duel': 'Duel — Black center, White faces it from north (fairest)',
    'setup.layout.blockade': 'Blockade — Black holds center, White behind',
    'setup.layout.classic': 'Classic — White centered, Black surrounds',
    'setup.gap': 'Opening Gap',
    'setup.gap2': '1 cell apart (dist 2)',
    'setup.gap3': '2 cells apart (dist 3)',
    'setup.first': 'First Move',
    'setup.firstWhite': 'Runner (White) first',
    'setup.firstBlack': 'Chaser (Black) first',
    'setup.offline': 'Online build: heuristic AI only — run locally (see README) for LLM / PPO play',

    'rules.title': 'Rule Knobs',
    'rules.goal': 'Escape Edges (multi-select)',
    'rules.edgeN': 'North (row 1)',
    'rules.edgeS': 'South (last row)',
    'rules.edgeW': 'West (column 1)',
    'rules.edgeE': 'East (last column)',
    'rules.capture': 'Capture Rule',
    'rules.captureGt': 'Strictly higher (Black H > White H)',
    'rules.captureGte': 'Equal or higher (Black H ≥ White H)',
    'rules.phase': 'Phase Cycle',
    'rules.phaseSync': 'Cross / X alternating in sync',
    'rules.phaseOff': 'Always Cross (original)',
    'rules.phaseAll8': 'All-8 every move',
    'rules.pass': 'Allow pass (skip a move)',
    'rules.repetition': 'Threefold repetition is a draw',

    'action.new': 'New Game',
    'action.pass': 'Pass',
    'action.again': 'Play Again',

    'train.title': 'Training Monitor',
    'train.live': 'LIVE',
    'train.status': 'Status',
    'train.progress': 'Progress',
    'train.speed': 'Speed',
    'train.eta': 'ETA',
    'train.winrate': 'Win rate (100 ep)',
    'train.idle': 'Not started',
    'train.unreachable': 'Server unreachable',
    'train.waiting': 'Waiting for training',
    'train.running': 'Training',
    'train.stopped': 'Ended / Paused',
    'train.nodata': 'Waiting for data…',
    'train.legendRew': 'Episode reward',
    'train.legendWin': 'Win(green)/Draw(gray)',
    'train.legendLoss': 'Value loss',
    'train.tb': 'Open TensorBoard ↗',
    'train.tbNote': 'Full metrics: reward/loss/entropy/KL/win rates (terrain/*)',

    'batch.title': 'Self-Play · Win-Rate Test',
    'batch.games': 'Games',
    'batch.run': 'Run with current config',
    'batch.running': 'Simulating…',
    'batch.white': 'White (Runner)',
    'batch.black': 'Black (Chaser)',
    'batch.draw': 'Draw',
    'batch.avgTurns': 'Avg. turns',
    'batch.gamesUnit': '',

    'banner.whiteWin': 'Runner Wins',
    'banner.blackWin': 'Chaser Wins',
    'banner.draw': 'Draw',
    'reason.capture': 'Captured on higher ground',
    'reason.escape': 'Reached the escape edge',
    'reason.border-capture': 'Caught right at the edge',
    'reason.stalemate': 'No legal moves',
    'reason.repetition': 'Threefold repetition',
    'reason.mutual-pass': 'Both players passed',

    'hint.setupWhite': 'White (Runner): click a gold ring to choose your start',
    'hint.setupBlack': 'Black (Chaser): click a gold ring to choose your start',
    'hint.choosing': 'is choosing a start…',
    'hint.move': 'to move: click a glowing ring',
    'hint.thinking': 'is thinking',
    'hint.phaseCross': 'Cross phase (N raises, S lowers, E/W fill)',
    'hint.phaseX': 'X phase (NE/NW raise, SE/SW lower)',
    'hint.phaseAll8': 'All-8 phase',
    'hint.over': 'Game over',
    'hint.white': 'White',
    'hint.black': 'Black',

    'toast.thinking': 'thinking…',
    'toast.moved': 'moved',
    'toast.fallback': 'random move instead',

    'lang.switch': '中文',

    // ---- Tutorial ----
    'tut.tagline': 'A hunt on a board that rewrites itself.',
    'tut.goal.title': 'The Goal',
    'tut.goal.body': 'White wins by escaping to the golden south edge. Black captures — only from adjacent, higher ground.',
    'tut.terrain.title': 'Living Terrain',
    'tut.terrain.body': 'Every move reshapes the landing cell\u2019s neighbors: north rises, south sinks, east & west fill — alternating with a diagonal phase.',
    'tut.capture.title': 'Height Rules',
    'tut.capture.body': 'Pieces ride the pillars up and down. Capture requires black strictly higher than white. Equal height is not enough.',
    'tut.watch': '▶ Watch a Demo Game',
    'tut.play': 'Play Now',
    'tut.again': 'Watch Demo Again',

    'demo.intro': 'White runs for the golden edge; black hunts — but capture needs higher ground. Watch the terrain flow with every move.',
    'demo.cross': 'Cross phase: north neighbor rises, south sinks, east & west fill in (impassable).',
    'demo.xphase': 'Diagonal phase: NE/NW rise, SE/SW sink — nothing fills this phase.',
    'demo.heightBlack': 'Black has the high ground — capture is now possible!',
    'demo.heightWhite': 'White took the high ground — black cannot capture right now.',
    'demo.close': 'Black closes to within two cells — danger for white!',
    'demo.escape': 'White is only a few steps from the golden edge…',
    'demo.endWhite': 'White reached the golden edge — escape succeeds!',
    'demo.endBlack': 'Capture! Black stood higher on adjacent ground.',
    'demo.endDraw': 'Escape and capture in the same instant — a draw.',
    'demo.exit': 'Exit demo',
    'demo.done': 'Demo over. Your turn —',
    'demo.playNow': 'Start My Game',
  },
};

let lang = localStorage.getItem('tc-lang') || 'en';
const listeners = [];

export function getLang() { return lang; }

export function t(key) {
  return (DICTS[lang] && DICTS[lang][key]) ?? DICTS.zh[key] ?? key;
}

// 把 data-i18n 标注应用到静态 DOM
export function applyStatic() {
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.title = lang === 'zh' ? '涌陆 · Terraflux' : 'Terraflux · 涌陆';
}

export function setLang(next) {
  lang = next;
  localStorage.setItem('tc-lang', next);
  applyStatic();
  listeners.forEach((cb) => cb());
}

export function toggleLang() {
  setLang(lang === 'zh' ? 'en' : 'zh');
}

// 动态内容（HUD/hint/banner/训练面板）在语言切换后重绘
export function onLangChange(cb) {
  listeners.push(cb);
}
