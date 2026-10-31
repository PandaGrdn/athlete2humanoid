import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { blend, newQpos, sample, type Frames, type Loaded, type Moment, type PushRow, type Robot } from "./data";

// The showcase's 3D view and its player. The robot is posed every frame from recorded qpos (forward
// kinematics here, from the MJCF's kinematic tree): the policy's unpushed run, or, after a click, the
// recorded run of the same policy pushed at that moment in that direction. Nothing is simulated here.
// Stage (renderer, camera, lights, floor), G1 and PolicyActor (a policy and its recorded pushes) are shared
// with the build view (buildview/scene.ts).

export type PushState = {
  status: "queued" | "live" | "rejoined" | "fell";
  mps: number;
  angle: number; // the recording's push direction (rad, world frame: MuJoCo x / y)
  step: number; // the push moment (50 Hz frame)
  rejoin: number | null;
  fall: number | null;
  since: number; // performance.now() of the last status change
};
export type PlayerState = { t: number; playing: boolean; speed: number; push: PushState | null; ghost: boolean; ready: boolean; note: string | null };

export const UP_FIX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2); // MuJoCo z-up -> three y-up
const FADE_FRAMES = 15; // a rejoin eases the last few cm back onto the unpushed run over 0.3 s
const FALL_HOLD_MS = 900;
const TARGET_Y = 0.66; // the camera orbits a point at the robot's hips
const MIN_LEAD = 6; // frames between a click and its push: the crossfade into the recording runs there

type Pose = { pos: THREE.Vector3[]; quat: THREE.Quaternion[] };

export class G1 {
  readonly group = new THREE.Group();
  readonly bodies: THREE.Group[] = [];
  readonly meshes: THREE.Mesh[] = [];
  private pose: Pose;
  private tmpQ = new THREE.Quaternion();
  private tmpV = new THREE.Vector3();
  private axis = new THREE.Vector3();

  constructor(private robot: Robot, geoms: THREE.BufferGeometry[], mats: Record<string, THREE.Material>, shadow: boolean) {
    for (let i = 0; i < robot.bodies.length; i++) {
      const g = new THREE.Group();
      g.matrixAutoUpdate = true;
      this.bodies.push(g);
      this.group.add(g);
    }
    robot.parts.forEach((p, i) => {
      const m = new THREE.Mesh(geoms[i], mats[p.tone]);
      m.castShadow = shadow;
      this.bodies[p.body].add(m);
      this.meshes.push(m);
    });
    this.pose = { pos: robot.bodies.map(() => new THREE.Vector3()), quat: robot.bodies.map(() => new THREE.Quaternion()) };
  }

  /** Forward kinematics from qpos (MuJoCo frame), into the body groups. */
  set(q: Float32Array) {
    const { pos, quat } = this.pose;
    this.robot.bodies.forEach((b, i) => {
      const p = pos[i];
      const r = quat[i];
      if (b.free) {
        p.set(q[0], q[1], q[2]);
        r.set(q[4], q[5], q[6], q[3]).normalize();
      } else {
        const pp = pos[b.parent - 1] ?? new THREE.Vector3();
        const pq = quat[b.parent - 1] ?? new THREE.Quaternion();
        p.set(b.pos[0], b.pos[1], b.pos[2]).applyQuaternion(pq).add(pp);
        r.copy(pq).multiply(this.tmpQ.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]));
        if (b.joint >= 0) {
          const jr = this.tmpQ.setFromAxisAngle(this.axis.set(b.axis[0], b.axis[1], b.axis[2]), q[7 + b.joint]);
          if (b.jpos[0] || b.jpos[1] || b.jpos[2]) {
            const jp = this.tmpV.set(b.jpos[0], b.jpos[1], b.jpos[2]);
            const moved = jp.clone().applyQuaternion(jr);
            p.add(jp.sub(moved).applyQuaternion(r));
          }
          r.multiply(jr);
        }
      }
      this.bodies[i].position.copy(p);
      this.bodies[i].quaternion.copy(r);
    });
  }

  root() { return this.pose.pos[0]; }
  /** A body's position (MuJoCo frame, the robot's own group) after the last set(), by name. */
  body(name: string) {
    const i = this.robot.bodies.findIndex((b) => b.name === name);
    return i >= 0 ? this.pose.pos[i] : null;
  }
}

export function geometries(robot: Robot) {
  return robot.parts.map((p) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(robot.pos.slice(p.v[0] * 3, (p.v[0] + p.v[1]) * 3), 3));
    g.setIndex(new THREE.BufferAttribute(robot.idx.slice(p.i[0], p.i[0] + p.i[1]), 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    return g;
  });
}

export const solidMaterials = () => ({
  light: new THREE.MeshStandardMaterial({ color: 0xd8d8d8, roughness: 0.42, metalness: 0.15 }),
  dark: new THREE.MeshStandardMaterial({ color: 0x1c1c1c, roughness: 0.55, metalness: 0.25 }),
});

/** The halo of the aether design, engraved on the floor: rings and radiating hairlines around the robot. */
function floorHalo() {
  const g = new THREE.Group();
  let s = 7;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const ray: number[] = [];
  for (let i = 0; i < 360; i++) {
    const a = (i / 360) * Math.PI * 2 + rnd() * 0.01;
    const r0 = 0.78 + rnd() * 0.04;
    const r1 = r0 + 0.08 + rnd() ** 2.2 * 0.6;
    ray.push(Math.cos(a) * r0, Math.sin(a) * r0, 0, Math.cos(a) * r1, Math.sin(a) * r1, 0);
  }
  const rg = new THREE.BufferGeometry();
  rg.setAttribute("position", new THREE.Float32BufferAttribute(ray, 3));
  g.add(new THREE.LineSegments(rg, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.22, depthWrite: false })));
  for (const [r, o, dash] of [[0.76, 0.5, false], [1.55, 0.14, false], [1.68, 0.16, true]] as const) {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 256; i++) pts.push(new THREE.Vector3(Math.cos((i / 256) * Math.PI * 2) * r, Math.sin((i / 256) * Math.PI * 2) * r, 0));
    const cg = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = dash ? new THREE.LineDashedMaterial({ color: 0xffffff, transparent: true, opacity: o, dashSize: 0.02, gapSize: 0.05, depthWrite: false })
      : new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: o, depthWrite: false });
    const line = new THREE.Line(cg, mat);
    if (dash) line.computeLineDistances();
    g.add(line);
  }
  // a soft pool of light under the robot
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const x = c.getContext("2d")!;
  const grad = x.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, "rgba(255,255,255,0.20)");
  grad.addColorStop(0.45, "rgba(255,255,255,0.05)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = grad;
  x.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const glow = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
  glow.position.z = -0.001;
  g.add(glow);
  g.position.z = 0.001;
  return g;
}

/** The black exhibit: renderer, camera with orbit controls, lights, a shadowed floor with its grid, and the
 *  render loop (paused while off screen). Subclasses add actors to `world` (MuJoCo frame) and move them in update(). */
export class Stage {
  protected renderer: THREE.WebGLRenderer;
  protected scene = new THREE.Scene();
  protected camera = new THREE.PerspectiveCamera(32, 1, 0.05, 60);
  protected controls: OrbitControls;
  protected world = new THREE.Group();
  protected key: THREE.DirectionalLight;
  protected halo: THREE.Group;
  private last = 0;
  private raf = 0;
  private visible = true;
  private ro: ResizeObserver;
  private io: IntersectionObserver;
  private wheelRoot: HTMLElement;

  constructor(protected canvas: HTMLCanvasElement, opts: { maxDistance?: number; shadowBox?: number } = {}) {
    const r = (this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" }));
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.fog = new THREE.Fog(0x000000, 5, 13);

    this.world.quaternion.copy(UP_FIX);
    this.scene.add(this.world);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x101010, 0.9));
    const key = (this.key = new THREE.DirectionalLight(0xffffff, 2.4));
    key.position.set(2.2, 4, 2.6);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    const b = opts.shadowBox ?? 2.5;
    Object.assign(key.shadow.camera, { left: -b, right: b, top: b, bottom: -b, near: 0.5, far: 12 });
    key.shadow.bias = -0.0004;
    key.shadow.radius = 4;
    this.scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0xffffff, 2.2);
    rim.position.set(-2.5, 2.2, -3.2);
    this.scene.add(rim);

    const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.ShadowMaterial({ opacity: 0.55 }));
    floor.receiveShadow = true;
    this.world.add(floor);
    const grid = new THREE.GridHelper(24, 96, 0x2a2a2a, 0x141414);
    grid.quaternion.copy(UP_FIX.clone().invert());
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.7;
    this.world.add(grid);
    this.halo = floorHalo();
    this.world.add(this.halo);

    const c = (this.controls = new OrbitControls(this.camera, canvas));
    c.enableDamping = true;
    c.dampingFactor = 0.08;
    c.enablePan = false;
    c.minDistance = 1.3;
    c.maxDistance = opts.maxDistance ?? 7;
    c.minPolarAngle = 0.15;
    c.maxPolarAngle = Math.PI / 2 - 0.04;
    c.rotateSpeed = 0.7;
    c.zoomSpeed = 0.8;

    // A plain wheel / two-finger swipe scrolls the page; only a pinch (a trackpad pinch is a ctrl+wheel) or
    // ctrl/cmd+wheel zooms. Caught on the way down, before OrbitControls sees it.
    this.wheelRoot = canvas.parentElement ?? canvas;
    this.wheelRoot.addEventListener("wheel", this.onWheel, { capture: true });
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas.parentElement ?? canvas);
    this.io = new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; });
    this.io.observe(canvas);
  }

  /** Point the camera at `target` (three's frame) from `target + offset`. */
  protected frame(target: THREE.Vector3, offset: THREE.Vector3) {
    this.controls.target.copy(target);
    this.camera.position.copy(target).add(offset);
    this.controls.update();
  }

  /** Call at the end of a subclass's constructor: sizes the canvas and starts the loop. */
  protected start() {
    this.resize();
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }

  /** Once per animation frame while on screen: dt in seconds (capped), now from performance.now(). */
  protected update(_dt: number, _now: number) {}

  private onWheel = (e: WheelEvent) => { if (!e.ctrlKey && !e.metaKey) e.stopPropagation(); };

  private tick = (now: number) => {
    this.raf = requestAnimationFrame(this.tick);
    const dt = Math.max(0, Math.min(0.1, (now - this.last) / 1000)); // a frame's timestamp can predate the constructor
    this.last = now;
    if (!this.visible || document.hidden) return;
    this.update(dt, now);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  protected resize() {
    const el = this.canvas.parentElement ?? this.canvas;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.fov = w < 560 ? 40 : 32;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.io.disconnect();
    this.wheelRoot.removeEventListener("wheel", this.onWheel, { capture: true });
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => {
        (x as THREE.MeshBasicMaterial).map?.dispose();
        x.dispose();
      });
    });
    this.renderer.dispose();
  }
}

/** A trained policy's recorded run and its recorded pushes, on its own 50 Hz clock: the robot (and the
 *  retarget as a ghost) in `group` (MuJoCo frame), the push arrow with it, and the click ripple (three's frame,
 *  for the host's scene). The host feeds it time (update) and clicks (push). */
export class PolicyActor {
  state: PlayerState = { t: 0, playing: true, speed: 1, push: null, ghost: false, ready: false, note: null };
  readonly group = new THREE.Group();
  readonly robot: G1;
  readonly ghost: G1;
  readonly ripple: THREE.Mesh;
  private listeners = new Set<() => void>();
  private arrow: THREE.Group;
  private f = 0; // current frame (fractional, 50 Hz)
  private q = newQpos();
  private q2 = newQpos();
  // a recording starts a few frames before its push (pre-roll, its own env's run): blended into from b0 on
  private pushed: { m: Moment; r: PushRow; frames: Frames | null; loopAhead: boolean; b0: number | null } | null = null;
  private fade: { delta: Float32Array; from: number } | null = null;
  private fellAt = 0;
  private fx = { arrowAt: -1e9, rippleAt: -1e9 };
  private lastPrefetch = 0;

  constructor(readonly data: Loaded, private mps: number, geoms = geometries(data.robot)) {
    const ghostMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.1, depthWrite: false });
    this.robot = new G1(data.robot, geoms, solidMaterials(), true);
    this.ghost = new G1(data.robot, geoms, { light: ghostMat, dark: ghostMat }, false);
    this.ghost.group.visible = false;
    this.group.add(this.robot.group, this.ghost.group);

    // the push arrow (along +x in its own frame, laid on the floor plane at pelvis height) and the click ripple
    this.arrow = new THREE.Group();
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.42, 8), arrowMat);
    shaft.rotation.z = -Math.PI / 2;
    shaft.position.x = -0.05;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.1, 16), arrowMat);
    head.rotation.z = -Math.PI / 2;
    head.position.x = 0.2;
    this.arrow.add(shaft, head);
    this.group.add(this.arrow);
    this.ripple = new THREE.Mesh(new THREE.RingGeometry(0.03, 0.036, 48), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));

    sample(data.nominal, 0, this.q);
    this.robot.set(this.q);
    this.state = { ...this.state, ready: true };
  }

  // --- the UI's side ------------------------------------------------------------------------------
  subscribe = (fn: () => void) => { this.listeners.add(fn); return () => { this.listeners.delete(fn); }; };
  getState = () => this.state;
  private emit(patch: Partial<PlayerState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn());
  }

  setPlaying(playing: boolean) { this.emit({ playing }); }
  setSpeed(speed: number) { this.emit({ speed }); }
  setStrength(mps: number) { this.mps = mps; }
  clearNote() { if (this.state.note) this.emit({ note: null }); }
  note(note: string) { this.emit({ note }); }
  setGhost(ghost: boolean) { this.ghost.group.visible = ghost; this.emit({ ghost }); }
  /** Seek (seconds). Any push in progress ends: the recordings start from the unpushed run. */
  seek(t: number) {
    this.f = Math.max(0, Math.min(this.data.show.frames - 1, t * this.data.show.fps));
    this.pushed = null;
    this.fade = null;
    this.emit({ t: this.f / this.data.show.fps, push: null, note: null });
  }

  /** Warm the cache for the next push moments (a click may follow); at most every 150 ms. */
  prefetch(force = false) {
    const now = performance.now();
    if (!force && now - this.lastPrefetch < 150) return;
    this.lastPrefetch = now;
    this.data.pushes.prefetch(Math.floor(this.f) + MIN_LEAD);
  }

  /** Push toward `angle` (rad, MuJoCo world frame); `at` (three's frame) is where the click landed, for the ripple. */
  push(angle: number, at: THREE.Vector3 | null) {
    const s = this.state.push;
    if (s && (s.status === "live" || (s.status === "fell"))) {
      this.emit({ note: "Each recording starts from the unpushed run: wait until it has recovered." });
      return;
    }
    const store = this.data.pushes;
    const next = store.next(Math.floor(this.f) + MIN_LEAD);
    if (!next) return;
    const r = store.pick(next.moment, angle, this.mps);
    const loopAhead = next.moment.step <= this.f;
    this.pushed = { m: next.moment, r, frames: null, loopAhead, b0: null };
    store.frames(next.moment, r).then((fr) => { if (this.pushed?.r === r) this.pushed.frames = fr; })
      .catch(() => { this.pushed = null; this.emit({ push: null, note: "Couldn't load that recording." }); });
    if (at) {
      this.ripple.position.copy(at);
      this.fx.rippleAt = performance.now();
    }
    this.fx.arrowAt = performance.now() + (next.wait / this.data.show.fps) * 1000 / Math.max(0.1, this.state.speed);
    this.arrow.rotation.set(0, 0, r.angle);
    this.emit({ note: null, push: { status: "queued", mps: r.mps, angle: r.angle, step: next.moment.step, rejoin: r.rejoin, fall: r.fall, since: performance.now() } });
    if (!this.state.playing) this.emit({ playing: true });
  }

  /** Advance the clock by dt seconds and pose the robot (and the ghost, if shown). */
  update(dt: number, now: number, camera: THREE.Camera) {
    const { fps, frames } = this.data.show;
    const p = this.pushed;
    const s = this.state.push;

    if (s?.status === "fell" && now - this.fellAt > FALL_HOLD_MS) { // after a fall: the clip again from the start
      this.pushed = null;
      this.f = 0;
      this.emit({ push: null });
    } else if (this.state.playing && s?.status !== "fell") {
      let nf = this.f + dt * fps * this.state.speed;
      if (p && s?.status === "queued" && !p.loopAhead && nf >= p.m.step && !p.frames) nf = p.m.step - 1e-3; // hold until it loads
      if (nf >= frames - 1) {
        nf = 0;
        if (p?.loopAhead) p.loopAhead = false;
        else if (p && s?.status === "live") { this.pushed = null; this.emit({ push: null }); } // ran to the clip's end
      }
      this.f = nf;
    }

    // which run the robot is on
    const pp = this.pushed;
    const st = this.state.push;
    let posed = false;
    if (pp && pp.frames && st?.status === "queued" && !pp.loopAhead && this.f >= pp.r.start) {
      if (this.f >= pp.m.step) this.emit({ push: { ...st, status: "live", since: now } });
      else { // crossfade from the unpushed run into this env's own run, before the push lands
        pp.b0 ??= this.f;
        const w = Math.min(1, (this.f - pp.b0) / Math.max(1, pp.m.step - pp.b0));
        sample(this.data.nominal, this.f, this.q);
        sample(pp.frames, this.f - pp.r.start, this.q2);
        blend(this.q, this.q2, w * w * (3 - 2 * w));
        posed = true;
      }
    }
    const live = this.pushed && this.state.push?.status === "live" ? this.pushed : null;
    if (live && live.frames) {
      const i = this.f - live.r.start;
      if (i >= live.frames.n - 1 && live.r.fall == null && live.r.rejoin == null) {
        sample(live.frames, live.frames.n - 1, this.q); // recorded to the clip's end: the loop restarts it
      } else if (i >= live.frames.n - 1) {
        sample(live.frames, live.frames.n - 1, this.q);
        if (live.r.fall != null) {
          this.fellAt = now;
          this.emit({ push: { ...this.state.push!, status: "fell", since: now } });
        } else {
          // back on the unpushed run: ease the remaining offset (< a few cm) out over FADE_FRAMES
          sample(this.data.nominal, this.f, this.q2);
          const delta = new Float32Array(this.q.length);
          for (let k = 0; k < delta.length; k++) delta[k] = this.q[k] - this.q2[k];
          this.fade = { delta, from: this.f };
          this.pushed = null;
          this.emit({ push: { ...this.state.push!, status: "rejoined", since: now } });
        }
      } else {
        sample(live.frames, Math.max(0, i), this.q);
      }
      posed = true;
    }
    if (!posed && this.state.push?.status !== "fell") {
      sample(this.data.nominal, this.f, this.q);
      if (this.fade) {
        const u = (this.f - this.fade.from) / FADE_FRAMES;
        if (u >= 1 || u < 0) this.fade = null;
        else {
          for (let k = 0; k < this.q.length; k++) this.q2[k] = this.q[k] + this.fade.delta[k];
          blend(this.q, this.q2, 1 - u);
        }
      }
    }
    if (this.state.push?.status === "rejoined" && now - this.state.push.since > 2600) this.emit({ push: null });

    this.robot.set(this.q);
    if (this.state.ghost) {
      sample(this.data.reference, this.f, this.q2);
      this.ghost.set(this.q2);
    }
    this.effects(now, camera);
    if (Math.abs(this.f / fps - this.state.t) > 1 / 30 || this.f / fps < this.state.t) this.emit({ t: this.f / fps });
  }

  private effects(now: number, camera: THREE.Camera) {
    const ra = (now - this.fx.rippleAt) / 700;
    const rm = this.ripple.material as THREE.MeshBasicMaterial;
    if (ra >= 0 && ra < 1) {
      this.ripple.quaternion.copy(camera.quaternion);
      this.ripple.scale.setScalar(1 + ra * 3.2);
      rm.opacity = 0.9 * (1 - ra);
    } else rm.opacity = 0;
    const aa = (now - this.fx.arrowAt) / 900;
    const am = (this.arrow.children[0] as THREE.Mesh).material as THREE.MeshBasicMaterial;
    const root = this.robot.root();
    const ang = this.arrow.rotation.z;
    if (aa >= -0.6 && aa < 1) {
      const lead = aa < 0 ? 0.25 : 0.25 - aa * 0.12; // the arrow arrives from behind the robot, then fades
      this.arrow.position.set(root.x - Math.cos(ang) * (lead + 0.18), root.y - Math.sin(ang) * (lead + 0.18), root.z);
      am.opacity = aa < 0 ? 0.5 * (1 + aa / 0.6) : 0.95 * (1 - aa);
    } else am.opacity = 0;
  }
}

/** A click (not a drag) on a canvas: calls `onClick` with the ray through that point. */
export function clickRays(canvas: HTMLCanvasElement, camera: THREE.Camera, onDown: () => void, onClick: (ray: THREE.Raycaster) => void) {
  let down: { x: number; y: number; t: number } | null = null;
  const pd = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; onDown(); };
  const pu = (e: PointerEvent) => {
    const d = down;
    down = null;
    if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 7 || performance.now() - d.t > 450) return; // a drag
    const rect = canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    onClick(ray);
  };
  canvas.addEventListener("pointerdown", pd);
  canvas.addEventListener("pointerup", pu);
  return () => {
    canvas.removeEventListener("pointerdown", pd);
    canvas.removeEventListener("pointerup", pu);
  };
}

/** A three.js direction -> its heading in MuJoCo's world frame (rad). */
export const heading = (d: THREE.Vector3) => Math.atan2(-d.z, d.x);

export class ShowcasePlayer extends Stage {
  private actor: PolicyActor;
  private unclick: () => void;

  constructor(canvas: HTMLCanvasElement, data: Loaded, mps: number) {
    super(canvas);
    const a = (this.actor = new PolicyActor(data, mps));
    this.world.add(a.group);
    this.scene.add(a.ripple);

    const root = a.robot.root();
    const target = new THREE.Vector3(root.x, TARGET_Y, -root.y);
    this.frame(target, new THREE.Vector3(2.5, 1.35 - TARGET_Y, 3.1));

    this.unclick = clickRays(canvas, this.camera, () => a.prefetch(true), (ray) => {
      const hit = ray.intersectObjects(a.robot.meshes, false)[0];
      if (!hit) return a.note("Click the robot itself to push it.");
      a.push(heading(ray.ray.direction), hit.point);
    });
    canvas.addEventListener("pointermove", this.onMove);
    this.start();
  }

  // --- the UI's side: the actor's ----------------------------------------------------------------
  get state() { return this.actor.state; }
  subscribe = (fn: () => void) => this.actor.subscribe(fn);
  getState = () => this.actor.state;
  setPlaying(playing: boolean) { this.actor.setPlaying(playing); }
  setSpeed(speed: number) { this.actor.setSpeed(speed); }
  setStrength(mps: number) { this.actor.setStrength(mps); }
  clearNote() { this.actor.clearNote(); }
  setGhost(ghost: boolean) { this.actor.setGhost(ghost); }
  seek(t: number) { this.actor.seek(t); }

  /** Push from the camera's side (the keyboard / button route to the same recordings). */
  pushFromView() {
    const d = new THREE.Vector3();
    this.camera.getWorldDirection(d);
    this.actor.push(heading(d), null);
  }

  dispose() {
    this.unclick();
    this.canvas.removeEventListener("pointermove", this.onMove);
    super.dispose();
  }

  private onMove = () => this.actor.prefetch();

  protected update(dt: number, now: number) {
    this.actor.update(dt, now, this.camera);
    this.follow();
  }

  private follow() {
    const root = this.actor.robot.root();
    const want = new THREE.Vector3(root.x, TARGET_Y, -root.y);
    const delta = want.sub(this.controls.target).multiplyScalar(0.06);
    this.controls.target.add(delta);
    this.camera.position.add(delta);
    this.key.target.position.set(root.x, 0, -root.y);
    this.key.position.set(root.x + 2.2, 4, -root.y + 2.6);
  }
}
