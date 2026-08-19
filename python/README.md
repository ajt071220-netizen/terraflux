# PPO 训练管线（Python）

> 规则引擎与 JS 版逐条对稿（`engine.py` ↔ `src/engine.js`）。
> 本机实测环境：**Python 3.12.9**（装在 `.venv-py312/` 虚拟环境，与系统 3.15 互不干扰）。

## 环境（已配置好）

本机系统 Python 是 3.15 测试版，PyTorch 尚无对应预编译包，
因此训练用的是项目根目录下的 `.venv-py312` 虚拟环境（Python 3.12.9）。
所有命令都用这个解释器：

```powershell
.\.venv-py312\Scripts\python.exe python\train_ppo.py --steps 2000000 --size 21
```

如需重装依赖（镜像源：阿里云）：

```powershell
.\.venv-py312\Scripts\python.exe -m pip install -r python\requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 训练

```powershell
# 在 项目根目录 下执行：
.\.venv-py312\Scripts\python.exe python\train_ppo.py --steps 2000000 --size 21
```

- `--size 21` 先用 21×21 训练（收敛快），之后可 `--resume` 微调迁移
- 对手 = 启发式 AI（50%）+ 历史快照池（50%）的自我对弈
- 产物：`ppo_terrain.zip` + `checkpoints/` 周期快照
- 中断续训：`--resume checkpoints\snap_500000.zip`
- 2M 步在普通 CPU 上大约 4–8 小时；先看 200k 步的效果即可试玩

## 部署到 3D 对战端

```powershell
.\.venv-py312\Scripts\python.exe python\serve_ppo.py --model ppo_terrain.zip --port 8765
```

然后在 3D 页面把某一方的玩家类型选成 **PPO** 即可。
（网页端 server.js 会把局面转发到 `http://127.0.0.1:8765/move`。）

## 自检（纯标准库，随时可跑）

```powershell
py python\selfcheck.py
```

## 文件

| 文件 | 作用 |
|---|---|
| `engine.py` | 规则引擎（与 JS 版一致） |
| `heuristic.py` | 启发式 AI（训练保底对手） |
| `gym_env.py` | Gymnasium 环境 + 动作掩码 |
| `train_ppo.py` | self-play 训练入口（含 BoardMLP 特征提取器） |
| `serve_ppo.py` | 训练产物的 HTTP 服务 |
| `selfcheck.py` | 引擎自检 + 基线胜率 |
