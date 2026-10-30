"""Objects in the G1's scene: a ball for the front kick (docs/OBJECTS_PLAN.md, stage 1). Runs in
mjlab's virtualenv, like mjlab_tasks.py, whose tasks it registers too:

    $MJLAB_PY mjlab_objects.py A2H-KickBall-Flat-Unitree-G1 --env.commands.motion.motion-file motion.npz ...

A2H-KickBall-Flat-Unitree-G1 is mjlab's tracking task (the one kp_front_kick_tracking was trained on)
plus a soccer ball:
- Where: the clip's front kick is a high kick (the foot rises past 0.8 m at ~5 m/s), so a ball on
  the floor is never in its path. The ball is held still at the point where the reference toe is
  fastest on the way up (kick_strikes), like a target a coach holds out, and let go when something
  moves it (or 0.15 s after the reference strike, a miss). Each of the clip's kicks gets a fresh ball.
- Frame: placed relative to the robot, not the clip, the way mjlab's tracking rewards compare bodies
  (MotionCommand.body_pos_relative_w): a robot that started 5 cm off or turned 0.2 rad still finds
  the ball where its own kick goes. Training jitters it by `jitter` m.
- The policy sees the ball (its position and velocity in the torso's yaw frame, and whether one is
  waiting to be kicked), appended to the tracking observation, so a tracking checkpoint warm-starts
  it acting exactly as before (policy_weights.py: new inputs get zero weight).
- Reward: the tracking terms, unchanged, plus a bonus when the kicking leg makes the hit and, for
  BALL_WINDOW_S after it, the ball's horizontal speed along the reference toe's direction at the strike.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mjlab_tasks  # noqa: E402  (registers our other tasks; its helpers)

import mujoco  # noqa: E402
from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.tasks.registry import register_mjlab_task  # noqa: E402
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat  # noqa: E402

BALL_R, BALL_M = 0.11, 0.43  # a size-5 soccer ball
KICK_BODY = "right_ankle_roll_link"
TOE = (0.13, 0.0, -0.025)  # the front of the G1's foot capsules, in the ankle roll link's frame
LEG_BODIES = ("right_ankle_roll_link", "right_knee_link")  # a hit by these counts as a kick
LEG_REACH = 0.1  # ...when the toe or one of their origins is this close to the ball's surface
PARK = (-2.0, 2.0, BALL_R)  # where the ball waits between kicks, in the env's frame, out of reach
HOLD_AFTER_S = 0.15  # an untouched ball is let go this long after the reference strike
SPAWN_GAP_S = 1.0  # a kick's ball appears this long after the previous strike (at t = 0 for the first)
BALL_WINDOW_S = 1.0  # the speed reward counts for this long after a hit
V_MAX = 6.0  # m/s: ball speed past this earns nothing more
HIT_SPEED = 0.6  # m/s: a held ball moving faster than this was hit (gravity alone gives 0.2 per step)


def ball_spec():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint(name="ball_joint")
    body.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(BALL_R, 0, 0), mass=BALL_M,
                  friction=(0.6, 0.005, 0.0001), condim=3, rgba=(0.95, 0.95, 0.95, 1.0))
    return spec


def toe_track(pos, quat):
    """(T, 3) ankle positions, (T, 4) wxyz quats -> (T, 3) toe positions."""
    toe = torch.tensor(TOE, dtype=pos.dtype, device=pos.device).expand_as(pos)
    return pos + quat_apply(quat, toe)


def kick_strikes(toe, fps, lift=0.15, min_len_s=0.2):
    """The strike frame of each kick: while the toe is up (above `lift` over its lowest point) for at
    least min_len_s, the frame where it moves fastest while rising. -> [(frame, toe pos, unit velocity)]."""
    vel = torch.gradient(toe, spacing=1 / fps, dim=0)[0]
    up = toe[:, 2] > toe[:, 2].min() + lift
    out, t, T = [], 0, len(toe)
    while t < T:
        if not up[t]:
            t += 1
            continue
        e = t
        while e < T and up[e]:
            e += 1
        if e - t >= min_len_s * fps:
            speed = vel[t:e].norm(dim=-1) * (vel[t:e, 2] > 0)
            s = t + int(speed.argmax())
            out.append((s, toe[s], vel[s] / vel[s].norm().clamp(min=1e-6)))
        t = e
    return out


class BallKick:
    """Step event: puts each kick's ball in front of the robot, holds it there, lets it go when hit,
    and keeps score (env.a2h_ball) for the observation, the rewards and the evaluation."""

    def __init__(self, cfg, env):
        # The event manager is built before the command manager: set up on first use.
        self.env, self.jitter, self.ready = env, float(cfg.params.get("jitter", 0.0)), False
        env.a2h_ball = self

    def _setup(self):
        env, self.ready = self.env, True
        self.cmd = env.command_manager.get_term("motion")
        self.ball, robot = env.scene["ball"], self.cmd.robot
        m, dt = self.cmd.motion, env.step_dt
        k = robot.body_names.index(KICK_BODY)
        toe = toe_track(m._body_pos_w[:, k], m._body_quat_w[:, k])
        strikes = kick_strikes(toe, 1 / dt)
        if not strikes:
            raise RuntimeError("no kick found in the motion (the right toe never rises 15 cm)")
        self.strike = torch.tensor([s for s, _, _ in strikes], device=env.device)
        # The ball's centre just past the toe, along its path: the toe reaches it at the strike.
        self.center = torch.stack([p + 0.8 * BALL_R * u for _, p, u in strikes])
        d = torch.stack([u for _, _, u in strikes])
        d[:, 2] = 0
        self.dir = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        prev = torch.cat([self.strike.new_full((1,), -10**6), self.strike[:-1]])
        self.spawn = torch.clamp(prev + round(SPAWN_GAP_S / dt), min=0)
        self.hold_end = self.strike + round(HOLD_AFTER_S / dt)
        self.window = round(BALL_WINDOW_S / dt)
        self.leg_ids = [robot.body_names.index(n) for n in LEG_BODIES]
        self.kick_body = k
        n = env.num_envs
        z = lambda dtype=torch.bool: torch.zeros(n, dtype=dtype, device=env.device)  # noqa: E731
        self.kick = z(torch.long) - 1  # which kick the ball is for (-1: parked)
        self.held, self.hit, self.leg_hit = z(), z(), z()
        self.hit_t, self.last_t = z(torch.long), z(torch.long)
        self.target, self.d = torch.zeros(n, 3, device=env.device), torch.zeros(n, 3, device=env.device)
        self.hits = self.leg_hits = self.kicks = 0  # running totals, for the evaluation

    def _park(self, ids):
        if len(ids) == 0:
            return
        self.kick[ids], self.held[ids], self.hit[ids], self.leg_hit[ids] = -1, False, False, False
        state = torch.zeros(len(ids), 13, device=self.env.device)
        state[:, :3] = self.env.scene.env_origins[ids] + torch.tensor(PARK, device=self.env.device)
        state[:, 3] = 1
        self.ball.write_root_state_to_sim(state, env_ids=ids)

    def reset(self, env_ids=None):
        if not self.ready:
            self._setup()
        ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
        self._park(ids)
        self.last_t[ids] = 0

    def _spawn(self, ids, k):
        """Kick k's ball for envs ids, placed as body_pos_relative_w places the reference's bodies."""
        cmd, origin = self.cmd, self.env.scene.env_origins[ids]
        ref_anchor = cmd.anchor_pos_w[ids]
        delta_q = yaw_quat(quat_mul(cmd.robot_anchor_quat_w[ids], quat_inv(cmd.anchor_quat_w[ids])))
        base = cmd.robot_anchor_pos_w[ids].clone()
        base[:, 2] = ref_anchor[:, 2]
        tgt = base + quat_apply(delta_q, self.center[k] + origin - ref_anchor)
        if self.jitter:
            tgt += (torch.rand_like(tgt) * 2 - 1) * self.jitter
        self.target[ids], self.d[ids] = tgt, quat_apply(delta_q, self.dir[k].expand(len(ids), 3))
        self.kick[ids], self.held[ids], self.hit[ids], self.leg_hit[ids] = k, True, False, False
        self.kicks += len(ids)

    def __call__(self, env, env_ids, **kwargs):
        if not self.ready:
            self._setup()
        t = self.cmd.time_steps
        # Read the ball before anything below moves it: after a write, its derived state (velocity,
        # position) is stale until the env's next forward(). A ball spawned while the last one
        # still flew read as already hit, so the clip's second ball never appeared.
        # Hit: a held ball that moved since the last step. By the kicking leg if that is nearest.
        v = self.ball.data.root_link_lin_vel_w
        moved = self.held & (v.norm(dim=-1) > HIT_SPEED)
        if moved.any():
            p = self.ball.data.root_link_pos_w
            robot = self.cmd.robot.data
            toe = toe_track(robot.body_link_pos_w[:, self.kick_body], robot.body_link_quat_w[:, self.kick_body])
            pts = torch.cat([toe[:, None], robot.body_link_pos_w[:, self.leg_ids]], 1)
            leg = (pts - p[:, None]).norm(dim=-1).min(-1).values < BALL_R + LEG_REACH
            self.hit |= moved
            self.leg_hit |= moved & leg
            self.hit_t = torch.where(moved, t, self.hit_t)
            self.hits += int(moved.sum())
            self.leg_hits += int((moved & leg).sum())
        kick = self.kick.clamp(min=0)
        self.held &= ~moved & (t <= self.hold_end[kick])
        self._park(torch.nonzero(t < self.last_t).flatten())  # the motion wrapped: a new start
        self.last_t = t.clone()
        for k in range(len(self.strike)):
            ids = torch.nonzero((t >= self.spawn[k]) & (t <= self.strike[k]) & (self.kick < k)).flatten()
            if len(ids):
                self._spawn(ids, k)
        ids = torch.nonzero(self.held).flatten()
        if len(ids):
            state = torch.zeros(len(ids), 13, device=env.device)
            state[:, :3], state[:, 3] = self.target[ids], 1
            self.ball.write_root_state_to_sim(state, env_ids=ids)

    def flying(self):
        """Envs whose ball was kicked by the leg less than BALL_WINDOW_S ago."""
        return self.leg_hit & (self.cmd.time_steps - self.hit_t <= self.window)


def ball_obs(env):
    """(N, 7): the ball's position (clipped to 2 m) and velocity (/5) in the torso's yaw frame, and
    1 while a ball waits to be kicked. Zeros before the ball event exists (the observation is sized)."""
    b = getattr(env, "a2h_ball", None)
    if b is None or not b.ready:
        return torch.zeros(env.num_envs, 7, device=env.device)
    q = yaw_quat(b.cmd.robot_anchor_quat_w)
    pos = quat_apply_inverse(q, b.ball.data.root_link_pos_w - b.cmd.robot_anchor_pos_w).clamp(-2, 2)
    vel = quat_apply_inverse(q, b.ball.data.root_link_lin_vel_w) / 5
    return torch.cat([pos, vel.clamp(-2, 2), b.held.float()[:, None]], -1)


def ball_speed(env):
    """The kicked ball's horizontal speed along the strike direction, 0..1 (V_MAX)."""
    b = env.a2h_ball
    v = (b.ball.data.root_link_lin_vel_w[:, :2] * b.d[:, :2]).sum(-1)
    return (v.clamp(0, V_MAX) / V_MAX) * b.flying()


def ball_reach(env, std=0.1, before_s=0.1):
    """Guides the toe to a waiting ball: exp(-|toe - ball|^2 / std^2), from before_s before the
    reference strike until the ball is let go. The tracker alone kicks late and low (kp_front_kick:
    zero-shot it brushes about half the balls, at 0.4 m/s)."""
    b = env.a2h_ball
    k = b.kick.clamp(min=0)
    on = b.held & (b.cmd.time_steps >= b.strike[k] - round(before_s / env.step_dt))
    robot = b.cmd.robot
    toe = toe_track(robot.data.body_link_pos_w[:, b.kick_body], robot.data.body_link_quat_w[:, b.kick_body])
    d2 = (toe - b.ball.data.root_link_pos_w).square().sum(-1)
    return torch.exp(-d2 / std**2) * on


def ball_leg_hit(env):
    """1 on the step the kicking leg hits the ball."""
    b = env.a2h_ball
    return (b.leg_hit & (b.hit_t == b.cmd.time_steps)).float()


def kick_ball_cfg(play=False, jitter=0.03, speed_weight=3.0, hit_weight=20.0, reach_weight=2.0):
    cfg = unitree_g1_flat_tracking_env_cfg(play=play)
    # A camera that sees the ball fly: beside and in front of the robot, close.
    v = cfg.viewer
    v.origin_type, v.entity_name, v.body_name = type(v).OriginType.ASSET_BODY, "robot", "torso_link"
    v.distance, v.elevation, v.azimuth, v.width, v.height = 2.6, -8.0, 135.0, 640, 480
    if play:
        cfg.commands["motion"].debug_vis = False  # no reference ghost: the robot and the ball only
    cfg.scene.entities["ball"] = EntityCfg(spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=PARK))
    cfg.events["ball"] = EventTermCfg(func=BallKick, mode="step", params={"jitter": 0.0 if play else jitter})
    for group in ("actor", "critic"):
        cfg.observations[group].terms["ball"] = ObservationTermCfg(func=ball_obs)
    cfg.rewards["ball_speed"] = RewardTermCfg(func=ball_speed, weight=speed_weight)
    cfg.rewards["ball_hit"] = RewardTermCfg(func=ball_leg_hit, weight=hit_weight)
    cfg.rewards["ball_reach"] = RewardTermCfg(func=ball_reach, weight=reach_weight)
    return cfg


register_mjlab_task(task_id="A2H-KickBall-Flat-Unitree-G1", env_cfg=kick_ball_cfg(),
                    play_env_cfg=kick_ball_cfg(play=True), rl_cfg=mjlab_tasks.unitree_g1_tracking_ppo_runner_cfg(),
                    runner_cls=mjlab_tasks.MotionTrackingOnPolicyRunner)


if __name__ == "__main__":
    import policy_weights
    from mjlab.scripts.train import main
    policy_weights.patch_mjlab_train()  # A2H_INIT_CHECKPOINT: build off the kick tracker's weights
    main()
