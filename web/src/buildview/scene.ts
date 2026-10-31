import * as THREE from "three";
import { newQpos, sample, type Frames, type Loaded, type Robot } from "../showcase/data";
import { clickRays, G1, geometries, heading, PolicyActor, solidMaterials, Stage } from "../showcase/scene";
import type { Skeleton, StageKey, Variants } from "./data";

// The build view's 3D scene: the pipeline's stages as actors on one floor, in a row from left to right: the
// 3D skeleton the pose model lifted from the video, the G1 retargeted onto it, and the trained policy; behind
// them, the variants as ghosts. The skeleton and the retarget follow the video's clock; the variants loop on
// their own tempos; the policy runs on its own clock and takes pushes (showcase/scene.ts PolicyActor).

// slots along MuJoCo's y (the camera looks from +x, so +y is screen right)
const SLOT = { skeleton: -1.2, retarget: 0, policy: 1.2 };
const GHOST_X = -1.7;
const GHOST_Y = [-1.65, -0.55, 0.55, 1.65];
const C = { left: 0x3987e5, right: 0xd95926, mid: 0xe8edf2, link: 0x0297de };
const side = (n: string) => (/^left/.test(n) ? C.left : /^right/.test(n) ? C.right : C.mid);
const ease = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
// skeleton joint -> the G1 body the retarget puts on it (the IK's end effectors, roughly)
const LINKS: [string, string][] = [["left wrist", "left_wrist_yaw_link"], ["right wrist", "right_wrist_yaw_link"],
  ["left elbow", "left_elbow_link"], ["right elbow", "right_elbow_link"], ["left knee", "left_knee_link"],
  ["right knee", "right_knee_link"], ["left ankle", "left_ankle_roll_link"], ["right ankle", "right_ankle_roll_link"]];

// camera framings (three's frame: x = MuJoCo x, y = MuJoCo z, z = -MuJoCo y): [target, offset]
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const VIEWS: Record<StageKey | "all", [THREE.Vector3, THREE.Vector3]> = {
  all: [V(-0.35, 0.72, 0), V(4.6, 1.05, 1.15)],
  video: [V(-0.35, 0.72, 0), V(4.6, 1.05, 1.15)],
  skeleton: [V(0, 0.8, -SLOT.skeleton), V(2.6, 0.45, 1.0)],
  retarget: [V(0, 0.75, -SLOT.skeleton / 2), V(3.2, 0.55, 0.9)],
  moments: [V(0, 0.7, 0), V(2.5, 0.5, 0.7)],
  variants: [V(-1.1, 0.7, 0), V(4.4, 1.9, 1.4)],
  policy: [V(0, 0.7, -SLOT.policy), V(2.4, 0.5, -1.0)],
};

/** A text label lying on the floor, drawn into a canvas texture. */
function label(text: string, opts: { size?: number; color?: string } = {}) {
  const px = 64;
  const c = document.createElement("canvas");
  const x = c.getContext("2d")!;
  const font = `600 ${px * 0.42}px ui-monospace, "SF Mono", Menlo, monospace`;
  x.font = font;
  c.width = Math.ceil(x.measureText(text).width + px * 0.6);
  c.height = px;
  x.font = font;
  x.fillStyle = opts.color ?? "rgba(255,255,255,0.9)";
  x.textBaseline = "middle";
  x.fillText(text, px * 0.3, px / 2);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const h = opts.size ?? 0.1;
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const m = new THREE.Mesh(new THREE.PlaneGeometry((h * c.width) / c.height, h), mat);
  m.rotation.z = Math.PI / 2; // lying on the floor (MuJoCo frame), reading left to right from the camera's side (+x)
  return m;
}

class StickFigure {
  readonly group = new THREE.Group();
  private joints: THREE.Mesh[] = [];
  private bones: THREE.Mesh[] = [];
  private neck: { nose: number; l: number; r: number; m: THREE.Mesh } | null = null; // nose to mid-shoulders
  private mid = new THREE.Vector3();
  private up = new THREE.Vector3(0, 1, 0);
  private a = new THREE.Vector3();
  private b = new THREE.Vector3();
  readonly pts: THREE.Vector3[];

  constructor(readonly sk: Skeleton) {
    const jg = new THREE.SphereGeometry(0.024, 14, 10);
    const bg = new THREE.CylinderGeometry(0.011, 0.011, 1, 8);
    const mats = new Map<number, THREE.MeshBasicMaterial>();
    const mat = (c: number) => mats.get(c) ?? mats.set(c, new THREE.MeshBasicMaterial({ color: c, transparent: true })).get(c)!;
    sk.names.forEach((n) => { const m = new THREE.Mesh(jg, mat(side(n))); this.joints.push(m); this.group.add(m); });
    sk.edges.forEach(([i, k]) => {
      const ca = side(sk.names[i]);
      const m = new THREE.Mesh(bg, mat(ca === side(sk.names[k]) ? ca : C.mid));
      this.bones.push(m);
      this.group.add(m);
    });
    this.pts = sk.names.map(() => new THREE.Vector3());
    const [nose, l, r] = ["nose", "left shoulder", "right shoulder"].map((n) => sk.names.indexOf(n));
    if (nose >= 0 && l >= 0 && r >= 0) {
      this.neck = { nose, l, r, m: new THREE.Mesh(bg, mat(C.mid)) };
      this.group.add(this.neck.m);
      this.joints[nose].scale.setScalar(2.2); // a head
    }
  }

  private bone(m: THREE.Mesh, pa: THREE.Vector3, pb: THREE.Vector3, grow: number) {
    m.visible = !Number.isNaN(pa.x) && !Number.isNaN(pb.x) && grow > 0.05;
    if (!m.visible) return;
    this.a.copy(pb).sub(pa);
    const len = this.a.length() * ease(grow);
    this.b.copy(this.a).normalize();
    m.quaternion.setFromUnitVectors(this.up, this.b);
    m.position.copy(pa).addScaledVector(this.b, len / 2);
    m.scale.set(1, Math.max(1e-4, len), 1);
  }

  /** Pose at fractional frame f; grow (0..1) draws the bones out from each joint. */
  set(f: number, offset: THREE.Vector3, grow: number) {
    const { n, j, data } = this.sk;
    const x = Math.max(0, Math.min(n - 1, f));
    const i0 = Math.floor(x);
    const i1 = Math.min(n - 1, i0 + 1);
    const w = x - i0;
    for (let k = 0; k < j; k++) {
      const a = (i0 * j + k) * 3;
      const b = (i1 * j + k) * 3;
      const p = this.pts[k];
      if (Number.isNaN(data[a]) || Number.isNaN(data[b])) p.set(NaN, NaN, NaN);
      else p.set(data[a] * (1 - w) + data[b] * w, data[a + 1] * (1 - w) + data[b + 1] * w, data[a + 2] * (1 - w) + data[b + 2] * w).add(offset);
      const m = this.joints[k];
      m.visible = !Number.isNaN(p.x);
      m.position.copy(p);
      m.scale.setScalar(Math.max(0.001, ease(grow * 1.6)) * (this.neck?.nose === k ? 2.2 : 1));
    }
    this.sk.edges.forEach(([i, k], e) => this.bone(this.bones[e], this.pts[i], this.pts[k], grow));
    if (this.neck) {
      const { nose, l, r, m } = this.neck;
      this.mid.copy(this.pts[l]).add(this.pts[r]).multiplyScalar(0.5);
      this.bone(m, this.mid, this.pts[nose], grow);
    }
  }
}

type Ghost = { g1: G1; frames: Frames; fps: number; slot: THREE.Vector3; root0: THREE.Vector2; label: THREE.Mesh; mat: THREE.MeshBasicMaterial };

export class BuildScene extends Stage {
  private geoms: THREE.BufferGeometry[];
  private fig: StickFigure | null = null;
  private figAt = 0; // performance.now() it arrived
  private figOffset = new THREE.Vector3();
  private ret: G1 | null = null;
  private retFrames: Frames | null = null;
  private retAt = 0;
  private retRoot0 = new THREE.Vector2();
  private links: THREE.LineSegments;
  private ghosts: Ghost[] = [];
  private ghostsAt = 0;
  private actor: PolicyActor | null = null;
  private actorAt = 0;
  private unclick: () => void;
  private labels: Partial<Record<"skeleton" | "retarget" | "policy" | "variants", THREE.Mesh>> = {};
  private pulse: THREE.Mesh;
  private pulseAt = -1e9;
  private moments: number[] = [];
  private lastT = 0;
  private q = newQpos();
  private elapsed = 0;
  private fps = 30;
  private tween: { from: [THREE.Vector3, THREE.Vector3]; to: [THREE.Vector3, THREE.Vector3]; at: number } | null = null;
  private onPolicyClick: ((hit: boolean) => void) | null = null;

  constructor(canvas: HTMLCanvasElement, private robot: Robot, private clock: () => number) {
    super(canvas, { maxDistance: 10, shadowBox: 3.4 });
    this.geoms = geometries(robot);
    this.halo.position.y = SLOT.retarget;
    this.links = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: C.link, transparent: true, opacity: 0, depthWrite: false }));
    this.links.geometry.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(LINKS.length * 6), 3));
    this.links.frustumCulled = false;
    this.world.add(this.links);
    this.pulse = new THREE.Mesh(new THREE.RingGeometry(0.62, 0.66, 96), new THREE.MeshBasicMaterial({ color: C.link, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
    this.pulse.position.set(0, SLOT.retarget, 0.004);
    this.world.add(this.pulse);
    const [t, o] = VIEWS.all;
    this.frame(t, o);
    this.controls.addEventListener("start", () => { this.tween = null; });
    this.unclick = clickRays(canvas, this.camera, () => this.actor?.prefetch(true), (ray) => {
      if (!this.actor) return;
      const hit = ray.intersectObjects(this.actor.robot.meshes, false)[0];
      this.onPolicyClick?.(!!hit);
      if (hit) this.actor.push(heading(ray.ray.direction), hit.point);
    });
    this.start();
  }

  /** Called on every click on the canvas once a policy is in: whether it hit the robot. */
  onClickPolicy(fn: ((hit: boolean) => void) | null) { this.onPolicyClick = fn; }

  private floorLabel(key: keyof BuildScene["labels"], text: string, x: number, y: number) {
    this.labels[key]?.removeFromParent();
    const m = label(text);
    m.position.set(x, y, 0.003);
    this.world.add(m);
    this.labels[key] = m;
  }

  setFps(fps: number) { this.fps = fps || 30; }

  setSkeleton(sk: Skeleton | null, model: string | null) {
    this.fig?.group.removeFromParent();
    this.fig = null;
    if (!sk || !sk.n) return;
    this.fig = new StickFigure(sk);
    this.figAt = performance.now();
    // feet on the floor (the lowest foot point, robustly), the first frame's hips over the slot
    const lows: number[] = [];
    const feet = sk.names.map((n, i) => (/ankle|heel|toe/.test(n) ? i : -1)).filter((i) => i >= 0);
    const hips = [sk.names.indexOf("left hip"), sk.names.indexOf("right hip")];
    for (let i = 0; i < sk.n; i++) {
      let lo = Infinity;
      for (const k of feet) { const z = sk.data[(i * sk.j + k) * 3 + 2]; if (z < lo) lo = z; }
      if (Number.isFinite(lo)) lows.push(lo);
    }
    lows.sort((a, b) => a - b);
    const ground = lows.length ? lows[Math.floor(lows.length * 0.05)] : 0;
    let hx = 0, hy = 0;
    for (let i = 0; i < sk.n; i++) { // the first frame with both hips
      const a = (i * sk.j + hips[0]) * 3;
      const b = (i * sk.j + hips[1]) * 3;
      if (!Number.isNaN(sk.data[a]) && !Number.isNaN(sk.data[b])) { hx = (sk.data[a] + sk.data[b]) / 2; hy = (sk.data[a + 1] + sk.data[b + 1]) / 2; break; }
    }
    this.figOffset.set(-hx, SLOT.skeleton - hy, -ground);
    this.world.add(this.fig.group);
    this.floorLabel("skeleton", `${(model ?? "pose").toUpperCase()} · 3D`, 0.55, SLOT.skeleton);
  }

  setRetarget(frames: Frames | null) {
    this.ret?.group.removeFromParent();
    this.ret = null;
    this.retFrames = frames;
    if (!frames || !frames.n) return;
    this.ret = new G1(this.robot, this.geoms, solidMaterials(), true);
    sample(frames, 0, this.q);
    this.retRoot0.set(this.q[0], this.q[1]);
    this.ret.set(this.q);
    this.retAt = performance.now();
    this.world.add(this.ret.group);
    this.floorLabel("retarget", "RETARGET · UNITREE G1", 0.55, SLOT.retarget);
  }

  setVariants(v: Variants | null) {
    for (const g of this.ghosts) { g.g1.group.removeFromParent(); g.label.removeFromParent(); }
    this.ghosts = [];
    this.labels.variants?.removeFromParent();
    if (!v) return;
    this.ghostsAt = performance.now();
    v.ghosts.slice(0, GHOST_Y.length).forEach((gh, i) => {
      // additive: overlapping parts glow brighter instead of sorting badly
      const mat = new THREE.MeshBasicMaterial({ color: 0x5fb8ec, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
      const g1 = new G1(this.robot, this.geoms, { light: mat, dark: mat }, false);
      sample(gh.frames, 0, this.q);
      const root0 = new THREE.Vector2(this.q[0], this.q[1]);
      const lab = label(gh.label.toUpperCase(), { size: 0.085, color: "rgba(150,210,245,0.95)" });
      this.world.add(g1.group, lab);
      this.ghosts.push({ g1, frames: gh.frames, fps: gh.fps, slot: new THREE.Vector3(GHOST_X, GHOST_Y[i], 0), root0, label: lab, mat });
    });
    if (v.total) {
      const m = label(`${v.kept} OF ${v.total} VARIANTS KEPT`, { size: 0.09 });
      m.position.set(GHOST_X + 0.62, 0, 0.003);
      this.world.add(m);
      this.labels.variants = m;
    }
  }

  setMoments(times: number[]) { this.moments = [...times].sort((a, b) => a - b); }

  setPolicy(data: Loaded | null, mps: number, text: string) {
    if (this.actor) { this.actor.group.removeFromParent(); this.actor.ripple.removeFromParent(); }
    this.actor = null;
    this.labels.policy?.removeFromParent();
    if (!data) return null;
    const a = (this.actor = new PolicyActor(data, mps, this.geoms));
    const r = a.robot.root();
    a.group.position.set(-r.x, SLOT.policy - r.y, 0);
    this.world.add(a.group);
    this.scene.add(a.ripple);
    this.actorAt = performance.now();
    this.floorLabel("policy", text.toUpperCase(), 0.55, SLOT.policy);
    return a;
  }

  /** Push the policy from the camera's side. */
  pushFromView() {
    if (!this.actor) return;
    const d = new THREE.Vector3();
    this.camera.getWorldDirection(d);
    this.actor.push(heading(d), null);
  }

  /** Swing the camera to a stage's actors (or all of them). */
  focus(key: StageKey | "all") {
    const [t, o] = VIEWS[key];
    const w = this.canvas.clientWidth;
    const k = w && w < 640 ? 1.45 : 1; // a narrow screen: step back
    this.tween = { from: [this.controls.target.clone(), this.camera.position.clone()], to: [t.clone(), t.clone().addScaledVector(o, k)], at: performance.now() };
  }

  dispose() {
    this.unclick();
    super.dispose();
  }

  protected update(dt: number, now: number) {
    const t = this.clock();
    const f = t * this.fps;
    this.elapsed += dt;

    if (this.fig) this.fig.set(f, this.figOffset, (now - this.figAt) / 900);

    if (this.ret && this.retFrames) {
      sample(this.retFrames, f, this.q);
      this.ret.set(this.q);
      // the snap: it arrives on the skeleton, then steps aside to its own place
      const u = ease((now - this.retAt - 500) / 1300);
      const fromY = this.fig ? SLOT.skeleton : SLOT.retarget;
      this.ret.group.position.set(-this.retRoot0.x, fromY + (SLOT.retarget - fromY) * u - this.retRoot0.y, 0);
      this.links.visible = !!this.fig;
      if (this.fig) this.updateLinks(Math.min(1, (now - this.retAt) / 400) * (0.75 - 0.45 * u));
    } else this.links.visible = false;

    if (this.ghosts.length) {
      const u0 = (now - this.ghostsAt) / 1000;
      this.ghosts.forEach((g, i) => {
        const u = ease(u0 - i * 0.18);
        const dur = g.frames.n / g.fps;
        sample(g.frames, ((this.elapsed % dur) / dur) * (g.frames.n - 1), this.q);
        g.g1.set(this.q);
        const x = g.slot.x * u;
        const y = SLOT.retarget + (g.slot.y - SLOT.retarget) * u;
        g.g1.group.position.set(x - g.root0.x, y - g.root0.y, 0);
        g.mat.opacity = 0.22 * u;
        g.label.position.set(x + 0.42, y, 0.003);
        (g.label.material as THREE.MeshBasicMaterial).opacity = u;
      });
    }

    if (this.actor) {
      this.actor.update(dt, now, this.camera);
      const u = ease((now - this.actorAt) / 700);
      this.actor.group.scale.setScalar(0.6 + 0.4 * u);
    }

    // a key moment passing: a ring pulses under the retarget
    if (this.ret && this.moments.some((m) => m > this.lastT && m <= t)) this.pulseAt = now;
    this.lastT = t;
    const pa = (now - this.pulseAt) / 650;
    const pm = this.pulse.material as THREE.MeshBasicMaterial;
    if (pa >= 0 && pa < 1) { this.pulse.scale.setScalar(0.7 + pa * 0.6); pm.opacity = 0.85 * (1 - pa); } else pm.opacity = 0;

    if (this.tween) {
      const u = ease((now - this.tween.at) / 1100);
      this.controls.target.lerpVectors(this.tween.from[0], this.tween.to[0], u);
      this.camera.position.lerpVectors(this.tween.from[1], this.tween.to[1], u);
      if (u >= 1) this.tween = null;
    }
  }

  private updateLinks(opacity: number) {
    const fig = this.fig!;
    const ret = this.ret!;
    const pos = this.links.geometry.getAttribute("position") as THREE.BufferAttribute;
    LINKS.forEach(([j, b], i) => {
      const p = fig.pts[fig.sk.names.indexOf(j)];
      const q = ret.body(b);
      if (!p || !q || Number.isNaN(p.x)) { pos.setXYZ(i * 2, 0, 0, 0); pos.setXYZ(i * 2 + 1, 0, 0, 0); return; }
      const g = ret.group.position;
      pos.setXYZ(i * 2, p.x, p.y, p.z);
      pos.setXYZ(i * 2 + 1, q.x + g.x, q.y + g.y, q.z + g.z);
    });
    pos.needsUpdate = true;
    (this.links.material as THREE.LineBasicMaterial).opacity = opacity;
  }
}
