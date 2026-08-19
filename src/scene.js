// scene.js —— Three.js 3D 棋盘场景
// 625 根柱子用 InstancedMesh 一次绘制；凹槽是 BackSide 半球（看到的内壁即碗状凹陷）；
// 高度/填平变化通过每帧向目标值平滑插值形成动画。

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { T } from './engine.js';

const PILLAR_W = 0.88;   // 柱截面边长
const PILLAR_H = 0.7;    // 柱体基准高度
const RISE = 0.34;       // 升高/降低的幅度
const DIMPLE_R = 0.3;    // 凹槽半径
const BALL_R = 0.24;     // 球半径

const PILLAR_COLORS = {
  [T.NORMAL]: new THREE.Color(0x9a9284),
  [T.RAISED]: new THREE.Color(0xd9b26a),
  [T.LOWERED]: new THREE.Color(0x5a6a80),
  [T.FILLED]: new THREE.Color(0x453e33),
};
const PREVIEW_COLORS = {
  [T.NORMAL]: new THREE.Color(0xaba393),
  [T.RAISED]: new THREE.Color(0xffe08a),
  [T.LOWERED]: new THREE.Color(0x8aa0c0),
  [T.FILLED]: new THREE.Color(0x5a5244),
};

export class BoardScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x16130f);
    this.scene.fog = new THREE.Fog(0x16130f, 45, 100);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 250);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 6;
    this.controls.maxDistance = 70;

    this.scene.add(new THREE.AmbientLight(0x9a8f78, 0.55));
    const sun = new THREE.DirectionalLight(0xfff1da, 1.7);
    sun.position.set(18, 32, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -20;
    sun.shadow.camera.right = 20;
    sun.shadow.camera.top = 20;
    sun.shadow.camera.bottom = -20;
    sun.shadow.camera.far = 80;
    sun.shadow.bias = -0.0004;
    this.scene.add(sun);
    const rim = new THREE.DirectionalLight(0x6a5a40, 0.5);
    rim.position.set(-14, 18, -16);
    this.scene.add(rim);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this._clickHandlers = [];
    this._hoverHandlers = [];
    this._downPos = null;
    const el = this.renderer.domElement;
    el.addEventListener('pointerdown', (e) => { this._downPos = { x: e.clientX, y: e.clientY }; });
    el.addEventListener('pointerup', (e) => {
      if (!this._downPos) return;
      const dx = e.clientX - this._downPos.x;
      const dy = e.clientY - this._downPos.y;
      this._downPos = null;
      if (dx * dx + dy * dy < 36) this._pick(e, this._clickHandlers); // 位移小才算点击，避免与旋转冲突
    });
    el.addEventListener('pointermove', (e) => this._pick(e, this._hoverHandlers));

    this.clock = new THREE.Clock();
    this._dummy = new THREE.Object3D();
    this._board = null;
    this._resize = () => {
      const w = this.container.clientWidth || window.innerWidth;
      const h = this.container.clientHeight || window.innerHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    };
    window.addEventListener('resize', this._resize);
    this._resize();
    this.renderer.setAnimationLoop(() => this._tick());
  }

  onCellClick(fn) { this._clickHandlers.push(fn); }
  onCellHover(fn) { this._hoverHandlers.push(fn); }

  _pick(e, handlers) {
    if (!this._board || handlers.length === 0) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this._board.pillars);
    if (hits.length > 0 && hits[0].instanceId !== undefined) {
      const id = hits[0].instanceId;
      const r = Math.floor(id / this.size) + 1;
      const c = (id % this.size) + 1;
      handlers.forEach((f) => f(r, c));
    } else if (handlers === this._hoverHandlers) {
      handlers.forEach((f) => f(null));
    }
  }

  cellToWorld(r, c) {
    return { x: (c - this.mid) * 1, z: (r - this.mid) * 1 };
  }

  buildBoard(size) {
    if (this._board) {
      this.scene.remove(this._board.group);
      this._board.group.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
    }
    this.size = size;
    this.mid = (size + 1) / 2;
    const n = size * size;
    this.offsets = new Float32Array(n);       // 柱子当前的纵向偏移（动画值）
    this.offsetTargets = new Float32Array(n); // 目标偏移
    this.dimpleScales = new Float32Array(n).fill(1);
    this.dimpleTargets = new Float32Array(n).fill(1);
    this.pillarStates = new Uint8Array(n);

    const group = new THREE.Group();

    // 柱体
    const geo = new THREE.BoxGeometry(PILLAR_W, PILLAR_H, PILLAR_W);
    geo.translate(0, PILLAR_H / 2, 0); // 原点移到柱底，方便纵向偏移
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.12 });
    const pillars = new THREE.InstancedMesh(geo, mat, n);
    pillars.castShadow = true;
    pillars.receiveShadow = true;
    pillars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    group.add(pillars);

    // 凹槽：BackSide 半球，看到的内壁就是碗状凹陷
    const dgeo = new THREE.SphereGeometry(DIMPLE_R, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2);
    const dmat = new THREE.MeshStandardMaterial({ color: 0x171310, roughness: 0.95, side: THREE.BackSide });
    const dimples = new THREE.InstancedMesh(dgeo, dmat, n);
    dimples.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    group.add(dimples);

    // 底座
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(size + 1.4, 0.3, size + 1.4),
      new THREE.MeshStandardMaterial({ color: 0x1e1a14, roughness: 0.85 })
    );
    base.position.y = -0.19;
    base.receiveShadow = true;
    group.add(base);

    // 胜利边指示条（默认全隐藏，按配置点亮）
    const edgeGeoH = new THREE.PlaneGeometry(size + 0.6, 0.22);
    edgeGeoH.rotateX(-Math.PI / 2);
    const edgeGeoV = edgeGeoH.clone();
    edgeGeoV.rotateY(Math.PI / 2);
    const edgeMat = new THREE.MeshBasicMaterial({ color: 0xc9a86a, transparent: true, opacity: 0.6 });
    const half = (size + 1) / 2 - 0.5; // 第 1/最后一格的中心偏移
    const off = half + 0.62;
    const mk = (geo, x, z) => {
      const m = new THREE.Mesh(geo, edgeMat);
      m.position.set(x, 0.02, z);
      m.visible = false;
      group.add(m);
      return m;
    };
    this._edgeBars = {
      N: mk(edgeGeoH, 0, -off),
      S: mk(edgeGeoH, 0, off),
      W: mk(edgeGeoV, -off, 0),
      E: mk(edgeGeoV, off, 0),
    };

    this.scene.add(group);
    this._board = { group, pillars, dimples };

    // 高亮圈池与球只创建一次（重开游戏时复用，避免残留）
    if (!this._rings) {
      const rgeo = new THREE.RingGeometry(0.3, 0.43, 32);
      rgeo.rotateX(-Math.PI / 2);
      this._rings = [];
      for (let i = 0; i < 12; i++) {
        const rm = new THREE.Mesh(
          rgeo,
          new THREE.MeshBasicMaterial({ color: 0xd9ba7e, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false })
        );
        rm.visible = false;
        rm.renderOrder = 5;
        this.scene.add(rm);
        this._rings.push(rm);
      }
    }

    if (!this.whiteBall) {
      const bgeo = new THREE.SphereGeometry(BALL_R, 32, 24);
      this.whiteBall = new THREE.Mesh(
        bgeo,
        new THREE.MeshStandardMaterial({ color: 0xfff6e6, emissive: 0xffdf9e, emissiveIntensity: 0.5, roughness: 0.22 })
      );
      this.blackBall = new THREE.Mesh(
        bgeo,
        new THREE.MeshStandardMaterial({ color: 0x1c1610, emissive: 0x8a6a30, emissiveIntensity: 0.4, roughness: 0.3 })
      );
      this.whiteBall.castShadow = true;
      this.blackBall.castShadow = true;
      this.scene.add(this.whiteBall, this.blackBall);
      this._ballTargets = { white: new THREE.Vector3(), black: new THREE.Vector3() };
    }
    this.whiteBall.visible = false;
    this.blackBall.visible = false;

    this._writeAllMatrices();
    this.camera.position.set(0, size * 0.95, size * 1.1);
    this.controls.target.set(0, 0, 0);
  }

  _idx(r, c) { return (r - 1) * this.size + (c - 1); }

  _writeAllMatrices() {
    const d = this._dummy;
    for (let r = 1; r <= this.size; r++) {
      for (let c = 1; c <= this.size; c++) {
        const i = this._idx(r, c);
        const { x, z } = this.cellToWorld(r, c);
        d.position.set(x, this.offsets[i], z);
        d.scale.setScalar(1);
        d.updateMatrix();
        this._board.pillars.setMatrixAt(i, d.matrix);
        d.position.set(x, this.offsets[i] + PILLAR_H + 0.001, z);
        d.scale.setScalar(Math.max(this.dimpleScales[i], 0.0001));
        d.updateMatrix();
        this._board.dimples.setMatrixAt(i, d.matrix);
      }
    }
    d.scale.setScalar(1);
    this._board.pillars.instanceMatrix.needsUpdate = true;
    this._board.dimples.instanceMatrix.needsUpdate = true;
  }

  // 全量同步引擎状态 → 场景目标值（动画由每帧插值完成）
  syncState(state) {
    for (let r = 1; r <= this.size; r++) {
      for (let c = 1; c <= this.size; c++) {
        const i = this._idx(r, c);
        const t = state.pillars[i];
        this.pillarStates[i] = t;
        this.offsetTargets[i] = t === T.RAISED ? RISE : t === T.LOWERED ? -RISE : 0;
        this.dimpleTargets[i] = t === T.FILLED ? 0 : 1;
        this._board.pillars.setColorAt(i, PILLAR_COLORS[t]);
      }
    }
    this._board.pillars.instanceColor.needsUpdate = true;

    if (state.white) {
      this.whiteBall.visible = true;
      this._ballTargets.white.copy(this._ballPos(state, state.white));
    }
    if (state.black) {
      this.blackBall.visible = true;
      this._ballTargets.black.copy(this._ballPos(state, state.black));
    }
    for (const [edge, bar] of Object.entries(this._edgeBars)) {
      bar.visible = state.config.goalEdges.includes(edge);
    }
  }

  _ballPos(state, pos) {
    const { x, z } = this.cellToWorld(pos.r, pos.c);
    const i = this._idx(pos.r, pos.c);
    return new THREE.Vector3(x, this.offsetTargets[i] + PILLAR_H + 0.04, z);
  }

  // 高亮一组格子（合法落点或候选起点）
  showHints(cells, color = 0xc9a86a) {
    this._rings.forEach((ring, i) => {
      if (i < cells.length) {
        const { x, z } = this.cellToWorld(cells[i].r, cells[i].c);
        const idx = this._idx(cells[i].r, cells[i].c);
        ring.position.set(x, this.offsetTargets[idx] + PILLAR_H + 0.03, z);
        ring.material.color.setHex(color);
        ring.visible = true;
      } else {
        ring.visible = false;
      }
    });
  }

  // 悬停预览：把将被影响的柱子染成预览色
  previewAffects(cells) {
    for (const cell of cells) {
      const i = this._idx(cell.r, cell.c);
      this._board.pillars.setColorAt(i, PREVIEW_COLORS[cell.to]);
    }
    this._board.pillars.instanceColor.needsUpdate = true;
  }

  // 恢复真实颜色（清除预览）
  restoreColors(state) {
    for (let i = 0; i < this.pillarStates.length; i++) {
      this._board.pillars.setColorAt(i, PILLAR_COLORS[state.pillars[i]]);
    }
    this._board.pillars.instanceColor.needsUpdate = true;
  }

  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const k = 1 - Math.exp(-dt * 9);
    const t = this.clock.elapsedTime;

    if (this._board) {
      let dirty = false;
      for (let i = 0; i < this.offsets.length; i++) {
        const doff = this.offsetTargets[i] - this.offsets[i];
        if (Math.abs(doff) > 0.0004) {
          this.offsets[i] += doff * k;
          dirty = true;
        }
        const dscale = this.dimpleTargets[i] - this.dimpleScales[i];
        if (Math.abs(dscale) > 0.004) {
          this.dimpleScales[i] += dscale * k;
          dirty = true;
        }
      }
      if (dirty) this._writeAllMatrices();
      // 高亮圈呼吸
      for (const ring of this._rings) {
        if (ring.visible) ring.material.opacity = 0.55 + 0.3 * Math.sin(t * 4);
      }
    }

    for (const [key, ball] of [['white', this.whiteBall], ['black', this.blackBall]]) {
      if (ball.visible) ball.position.lerp(this._ballTargets[key], 1 - Math.exp(-dt * 8));
    }

    // 胜利边微光脉冲
    for (const bar of Object.values(this._edgeBars || {})) {
      if (bar.visible) bar.material.opacity = 0.4 + 0.2 * Math.sin(t * 2.4);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
