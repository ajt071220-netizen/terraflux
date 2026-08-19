"""serve_ppo.py —— PPO 模型 HTTP 服务，供 3D 对战端调用

用法：
  python serve_ppo.py --port 8765
  启动时自动加载 models/ 下所有 ppo_<尺寸>.zip（如 ppo_21.zip、ppo_25.zip），
  按对局的棋盘尺寸自动路由到对应模型。

协议（与 server.js 的 /api/agent 转发对接）：
  POST /move  { "state": <JS 端序列化局面> }
  → { "move": {"r": int, "c": int} | null }
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

import engine as E
from gym_env import DIRS
from train_ppo import env_obs_for


def deserialize_state(raw):
    return {
        'config': raw['config'],
        'size': raw['size'],
        'pillars': bytearray(raw['pillars']),
        'white': raw['white'] and dict(raw['white']),
        'black': raw['black'] and dict(raw['black']),
        'turn': raw['turn'],
        'moveCount': dict(raw['moveCount']),
        'passes': raw.get('passes', 0),
        'status': raw['status'],
        'winner': raw.get('winner'),
        'winReason': raw.get('winReason'),
        'repetition': {},
    }


def build_handler(models, use_mask):
    """models: {尺寸: 模型}"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != '/move':
                return self._send({'error': 'not found'}, 404)
            try:
                length = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(length) or b'{}')
                state = deserialize_state(body['state'])
                color = state['turn']

                model = models.get(state['size'])
                if model is None:
                    avail = '、'.join(str(s) for s in sorted(models))
                    return self._send({'move': None, 'error': f'暂无 {state["size"]}×{state["size"]} 的 PPO 模型（可用尺寸：{avail}）'}, 400)

                legal = E.legal_moves(state, color)
                if not legal:
                    return self._send({'move': None})

                obs = env_obs_for(state, color)
                if use_mask:
                    from gym_env import action_mask
                    act, _ = model.predict(obs, action_masks=np.array([action_mask(state, color)]), deterministic=True)
                else:
                    act, _ = model.predict(obs, deterministic=True)
                action = int(act.item() if hasattr(act, 'item') else act)

                if action == 8:
                    move = None if state['config']['passAllowed'] else legal[0]
                else:
                    dr, dc = DIRS[action]
                    me = state[color]
                    cand = (me['r'] + dr, me['c'] + dc)
                    move = cand if cand in [tuple(m) for m in legal] else legal[0]
                self._send({'move': None if move is None else {'r': move[0], 'c': move[1]}})
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send({'move': None, 'error': str(e)}, 400)

        def do_GET(self):
            if self.path == '/health':
                self._send({'ok': True, 'sizes': sorted(models)})
            else:
                self._send({'error': 'not found'}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', type=str, default='models')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()

    try:
        from sb3_contrib import MaskablePPO as cls
        use_mask = True
    except ImportError:
        from stable_baselines3 import PPO as cls
        use_mask = False

    models = {}
    for f in sorted(glob.glob(os.path.join(args.models_dir, 'ppo_*.zip'))):
        m = re.search(r'ppo_(\d+)\.zip$', os.path.basename(f))
        if not m:
            continue
        size = int(m.group(1))
        models[size] = cls.load(f)
        print(f'  已加载 {f}（{size}×{size}）')
    if not models:
        raise SystemExit(f'{args.models_dir}/ 下没有找到 ppo_<尺寸>.zip 模型文件')

    print(f'PPO 服务已启动: http://127.0.0.1:{args.port}  （支持尺寸：{sorted(models)}）')
    HTTPServer(('127.0.0.1', args.port), build_handler(models, use_mask)).serve_forever()


if __name__ == '__main__':
    main()
