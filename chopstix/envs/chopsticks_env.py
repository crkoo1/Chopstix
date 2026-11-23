"""Shadow Hand + chopsticks in MuJoCo: Makis' ShadowChopsticksEnv (chops.envs.chopsticks_env) ported from SuperDex.

Same task, phases, rewards, success tests, and the SAME observation (proprio 81 + task 40 = 121) and action (28)
layouts, joint order included (SuperDex's: FF, LF, MF, RF, TH), so a chops actor (chops-actor-v1) can be loaded here.

Stages (cfg.stage):
  * "hold": start in a tripod grip under gravity and random pushes (0.03 N) and keep both sticks where the grip put
    them (palm frame). Wrist locked; actions are residuals of up to hold_residual rad on the grip's joint targets, so
    a zero action is the scripted grip. With hold_grip_closed set, a scheduled command alpha (task obs 33:36) moves
    the reference between hold_grip (0) and hold_grip_closed (1).
  * "pick": start holding the sticks; the palm follows a scripted path (descend onto a cube, pause, lift; with
    pick_place also carry, lower, release, retreat) and the policy moves only the fingers (residuals as in "hold").

Physics knobs are chopstix.sim.SimCfg fields and can be passed in the same override dict (e.g. {"stage": "hold",
"timestep": 0.005}). chops' ball_* names map onto the object (ball_radius -> object_size, ball_cube -> object="cube").
"""

from dataclasses import dataclass, field, fields
from typing import Any

import gymnasium as gym
import numpy as np

from chopstix.geometry import CHOPSTICK_LENGTH, NUM_HAND_JOINTS, SR_JOINT_NAMES, mat_to_rotvec, rot6d, rotvec_to_mat, tip_offset
from chopstix.sim import STICKS, Sim, SimCfg, load_grip

PROPRIO_DIM = 3 * NUM_HAND_JOINTS + 3 + 6 + 6
TASK_DIM = 2 * (3 + 6 + 3 + 3) + (3 + 3) + 3 + 1
ACTION_DIM = 6 + NUM_HAND_JOINTS
# SuperDex's hand DoF order (by link name: FF, LF, MF, RF, TH), read from chops' env (_sim_from_sr).
# hand_in_obs_order = hand_in_sr_order[OBS_FROM_SR]
OBS_FROM_SR = np.array([0, 1, 2, 3, 12, 13, 14, 15, 16, 4, 5, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21])
# (finger, stick index) pairs of the tripod grip (chops.envs.chopsticks_env.HOLD_CONTACTS).
HOLD_CONTACTS = [("th", 0), ("ff", 0), ("mf", 0), ("th", 1), ("rf", 1)]
# Physics defaults for training (see the grip lab in README_MUJOCO.md): 5 ms steps (10 per 20 Hz control step), a
# no-slip pass (without it soft-contact creep walks the sticks out of the grip) and a 0.15 s lag on the finger targets
# (without it exploration noise of ~0.01 rad per step shakes the lower stick loose within 10 s).
ENV_SIM_DEFAULTS = {"timestep": 0.005, "noslip_iterations": 10, "actuator_timeconst": 0.15}
ALIASES = {"ball_radius": "object_size", "ball_mass": "object_mass", "ball_friction": "object_friction"}


@dataclass
class ChopsticksEnvCfg:
    stage: str = "hold"
    control_frequency: int = 20
    steps_per_episode: int = 200
    sim: dict = field(default_factory=dict)  # SimCfg overrides (on top of ENV_SIM_DEFAULTS)

    w_action: float = 0.01
    # Hold stage.
    hold_grip: str = "closed_v5_mj4"  # name in chopstix/grips/ or a path (grip lab: README_MUJOCO.md)
    hold_palm_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.3])
    hold_residual: float = 0.1
    hold_push: float = 0.03
    hold_push_every: int = 2
    hold_joint_noise: float = 0.01
    hold_stick_noise: float = 0.0
    hold_max_drift: float = 0.03
    hold_max_turn: float = 30.0
    # How stick rotation is measured for hold_max_turn, hold_success_turn and the hold_rot reward: "tilt" = angle
    # between the stick's axis and the grip's (spin about its own axis is free: the stick is round); "full" = chops'
    # full rotation angle, spin included (a few degrees of harmless rolling then end episodes and fail successes).
    hold_turn: str = "tilt"
    # Palm motion in the hold stage (0 = palm locked, chops' behaviour). Each episode draws an amplitude uniformly in
    # [0, max] and a random direction/axis: wobble = rocking about the axis at 2 Hz (+-deg), shake = translation along the
    # direction at 3 Hz (peak m/s^2). Grip lab: the grip alone does not survive +-5 deg of wobble; the policy must.
    hold_wobble: float = 0.0
    hold_shake: float = 0.0
    hold_success_drift: float = 0.01
    hold_success_turn: float = 10.0
    w_hold_pos: float = 1.0
    w_hold_rot: float = 1.0
    w_hold_pose: float = 0.3
    w_hold_success: float = 1.0
    w_hold_pair: float = 0.0
    hold_pair_drift_scale: float = 0.005
    hold_pair_turn_scale: float = 5.0
    w_hold_slip: float = 0.0
    hold_slip_lin_scale: float = 0.01
    hold_slip_ang_scale: float = 0.5
    w_hold_squeeze: float = 0.0
    w_hold_no_contact: float = 0.0
    hold_force_min: float = 0.03
    hold_force_max: float = 1.0
    hold_force_zero: float = 0.005
    hold_grip_closed: str = ""
    hold_alpha_rate: float = 0.5
    hold_alpha_dwell_min: float = 0.5
    hold_alpha_dwell_max: float = 2.0
    w_hold_gap: float = 2.0
    hold_gap_scale: float = 0.005
    hold_gap_success: float = 0.008
    # Pick stage.
    pick_settle: float = 0.5
    pick_descend: float = 1.5
    pick_close: float = 1.5
    pick_lift: float = 1.5
    pick_approach: float = 0.05
    pick_lift_height: float = 0.06
    pick_back: float = 0.022
    pick_push: float = 0.0  # N: random pushes on each stick during "pick" too (as hold_push, every hold_push_every steps); 0 = off
    pick_stick_pitch: float = 0.0  # deg below horizontal for the sticks' mean axis in the pick pose; 0 = the grip's own
    pick_hand_flip: bool = False  # turn the hand 180 deg about the stick axis: back of the hand up, as a person holds chopsticks
    pick_hand_tilt: float = 0.0  # deg: then pitch the hand so fingers and sticks point that much further down
    pick_level_open: bool = False  # open/close pair: turn the hand so the tips open and close sideways (level), not up/down
    pick_hand_roll: float = 0.0  # deg: then roll the hand about the sticks so the back of the hand faces further up
    pick_object_noise: float = 0.005
    pick_object_yaw_noise: float = 0.1
    pick_success_height: float = 0.04
    pick_max_drift: float = 0.03
    pick_max_away: float = 0.12
    pick_touch_force: float = 0.01
    w_pick_hold: float = 1.0
    w_pick_near: float = 0.5
    w_pick_touch: float = 1.0
    w_pick_lift: float = 5.0
    w_pick_success: float = 10.0
    pick_place: bool = False
    pick_carry: float = 2.0
    pick_lower: float = 1.0
    pick_release: float = 1.5
    pick_retreat: float = 1.0
    pick_place_dist: float = 0.10
    pick_place_drop: float = 0.005
    pick_place_tolerance: float = 0.02
    pick_open_gap: float = 0.008
    w_pick_open: float = 1.0
    w_pick_released: float = 2.0
    pick_grip_ramp: float = 1.0
    w_pick_squeeze: float = 0.0
    pick_squeeze_min: float = 0.05
    pick_squeeze_max: float = 1.0
    w_pick_stick_lost: float = 0.0
    pick_lost_time: float = 0.15
    w_pick_drop: float = 0.0
    pick_drop_lift: float = 0.015
    pick_drop_time: float = 0.15

    @classmethod
    def from_overrides(cls, overrides: dict | None = None) -> "ChopsticksEnvCfg":
        """Split one flat override dict into env fields and SimCfg fields (chops' ball_* names accepted)."""
        env_names, sim_names = {f.name for f in fields(cls)}, {f.name for f in fields(SimCfg)}
        env_kw, sim_kw = {}, {}
        for k, v in (overrides or {}).items():
            if k == "ball_cube":
                sim_kw["object"] = "cube" if v else "ball"
            elif k in ALIASES:
                sim_kw[ALIASES[k]] = v
            elif k == "sim":
                sim_kw.update(v)
            elif k in env_names:
                env_kw[k] = v
            elif k in sim_names:
                sim_kw[k] = v
            else:
                raise KeyError(f"unknown env/sim setting {k!r}")
        cfg = cls(**env_kw)
        cfg.sim = {**ENV_SIM_DEFAULTS, **sim_kw}
        return cfg


def _angle_deg(cos):
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class ChopsticksEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    PICK_PHASES = ("settle", "descend", "close", "lift", "carry", "lower", "release", "retreat")

    def __init__(self, cfg: ChopsticksEnvCfg | dict[str, Any] | None = None, render_mode: str | None = None):
        if not isinstance(cfg, ChopsticksEnvCfg):
            cfg = ChopsticksEnvCfg.from_overrides(cfg)
        if cfg.stage not in ("hold", "pick"):
            raise ValueError(f"stage {cfg.stage!r}: only 'hold' and 'pick' are ported so far")
        self.cfg = cfg
        self.render_mode = render_mode
        self.sim = Sim(SimCfg.from_overrides(cfg.sim))
        self.dt = 1.0 / cfg.control_frequency
        self.substeps = max(1, round(self.dt / self.sim.model.opt.timestep))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (PROPRIO_DIM + TASK_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self._grips = [self._read_grip(cfg.hold_grip), self._read_grip(cfg.hold_grip_closed) if cfg.hold_grip_closed else None]
        if self._actuated and not np.allclose(self._grips[0]["palm_rot"], self._grips[1]["palm_rot"], atol=1e-4):
            raise ValueError("hold_grip and hold_grip_closed have different palm orientations")
        self._hold = self._grips[0]
        self._alpha = self._alpha_goal = self._alpha_vel = 0.0
        self._alpha_speed = self._dwell = 0.0
        self._push = np.zeros((2, 3))
        self._wobble = self._shake = None
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._targets = self._hold["targets"].copy()
        if cfg.stage == "pick":
            if not self.sim.has_object:
                raise ValueError("the pick stage needs an object (sim object='cube' or 'ball')")
            self._init_pick()
        self._renderer = None

    # ------------------------------------------------------------------ grips
    def _read_grip(self, name: str) -> dict:
        g = load_grip(name)
        sticks = [g["sticks"][n] for n in STICKS]  # upper (stick_a), lower (stick_b)
        tips = [p + r @ tip_offset() for p, r in sticks]
        return {"palm_rot": g["palm_rot"], "joints": g["joints"], "targets": g["targets"], "sticks": sticks,
                "gap": float(np.linalg.norm(tips[0] - tips[1]))}  # fmt: skip

    @property
    def _actuated(self) -> bool:
        return self._grips[1] is not None

    def _ref(self) -> dict:
        if not self._actuated:
            return self._hold
        a, (g0, g1) = self._alpha, self._grips
        lerp = lambda k: g0[k] + a * (g1[k] - g0[k])  # noqa: E731
        sticks = [(p0 + a * (p1 - p0), r0 @ rotvec_to_mat(a * mat_to_rotvec(r0.T @ r1))) for (p0, r0), (p1, r1) in zip(g0["sticks"], g1["sticks"])]
        return {"palm_rot": g0["palm_rot"], "joints": lerp("joints"), "targets": lerp("targets"), "sticks": sticks, "gap": lerp("gap")}

    def _step_alpha(self):
        cfg, rng = self.cfg, self.np_random
        if self._alpha == self._alpha_goal:
            self._dwell -= self.dt
            if self._dwell <= 0:
                self._alpha_goal = float(rng.choice([0.0, 1.0, rng.uniform()], p=[0.35, 0.35, 0.3]))
                self._dwell = rng.uniform(cfg.hold_alpha_dwell_min, cfg.hold_alpha_dwell_max)
        step = self._alpha_speed * self.dt
        prev = self._alpha
        self._alpha = float(np.clip(self._alpha_goal, prev - step, prev + step))
        self._alpha_vel = (self._alpha - prev) / self.dt

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM)
        self._push = np.zeros((2, 3))
        if self.cfg.stage == "hold":
            self._reset_hold()
        else:
            self._reset_pick()
        self.sim.forward()
        self._tips_rest_height = self._stick_tips()[:, 2].copy()
        obs, info = self._observe()
        return obs, info

    def _place(self, palm_pos, palm_rot, grip: dict, stick_noise: float = 0.0):
        cfg, rng, sim = self.cfg, self.np_random, self.sim
        sim.reset()
        sim.set_palm(palm_pos, palm_rot)
        sim.set_joints(grip["joints"] + rng.uniform(-cfg.hold_joint_noise, cfg.hold_joint_noise, NUM_HAND_JOINTS))
        self._targets = np.clip(grip["targets"], *sim.joint_limits.T)
        sim.set_targets(self._targets)
        for n, (pos, rot) in zip(STICKS, grip["sticks"]):
            sim.place_stick(n, pos + (rng.uniform(-stick_noise, stick_noise, 3) if stick_noise else 0.0), rot)

    def _reset_hold(self):
        cfg, rng, h = self.cfg, self.np_random, self._hold
        if self._actuated:
            end = int(rng.integers(2))
            h = self._grips[end]
            self._alpha = self._alpha_goal = float(end)
            self._alpha_vel = 0.0
            self._alpha_speed = cfg.hold_alpha_rate * rng.uniform(0.3, 1.0)
            self._dwell = rng.uniform(cfg.hold_alpha_dwell_min, cfg.hold_alpha_dwell_max)
        self._place(np.array(cfg.hold_palm_pos), h["palm_rot"], h, cfg.hold_stick_noise)
        self._palm_base = (np.array(cfg.hold_palm_pos, float), h["palm_rot"].copy())
        unit = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=3))  # noqa: E731
        self._wobble = (unit(), np.radians(rng.uniform(0, cfg.hold_wobble))) if cfg.hold_wobble > 0 else None
        self._shake = (unit(), rng.uniform(0, cfg.hold_shake) / (2 * np.pi * 3.0) ** 2) if cfg.hold_shake > 0 else None
        if self.sim.has_object:
            self.sim.place_object([0.3, 0.0, self.sim.cfg.object_size])  # parked, out of the way

    # ------------------------------------------------------------------ pick stage
    def _init_pick(self):
        cfg = self.cfg
        half = CHOPSTICK_LENGTH / 2

        def pinch_points(grip):
            (pu, ru), (pl, rl) = grip["sticks"]
            a = pu + ru[:, 0] * (half - cfg.pick_back)
            b = pl + rl[:, 0] * float((a - pl) @ rl[:, 0])
            return a, b, ru[:, 0] + rl[:, 0]

        h = self._grips[1] if self._actuated else self._hold
        a, b, axis = pinch_points(h)
        sep = a - b
        if cfg.pick_level_open and self._actuated:  # level the direction the tips open in (open minus closed gap), not the
            ao, bo, _ = pinch_points(self._hold)  # closed gap (nearly 0 when the closed tips touch: its direction is noise)
            sep = (ao - bo) - sep
        base = h["palm_rot"]
        w_axis = base @ axis
        w_axis /= np.linalg.norm(w_axis)
        if cfg.pick_stick_pitch:  # tilt the hand (same heading) so the sticks point pick_stick_pitch deg below horizontal
            th = np.radians(cfg.pick_stick_pitch)
            head = np.array([w_axis[0], w_axis[1], 0.0]) / np.linalg.norm(w_axis[:2])
            target = np.cos(th) * head + [0.0, 0.0, -np.sin(th)]
            c = np.cross(w_axis, target)
            ang = np.arctan2(np.linalg.norm(c), float(w_axis @ target))
            base = rotvec_to_mat(c / max(np.linalg.norm(c), 1e-12) * ang) @ base
            w_axis = target
        w_sep = base @ sep

        def spun(t):
            return rotvec_to_mat(w_axis * t) @ w_sep

        best = min(np.linspace(-np.pi, np.pi, 3601), key=lambda t: abs(spun(t)[2]) - 1e-3 * (spun(t) @ base @ sep))
        if cfg.pick_hand_flip:  # of the two level-sticks solutions, the one with the palm (its -y) facing the table
            best = min((best, best + np.pi), key=lambda t: -(rotvec_to_mat(w_axis * t) @ base)[2, 1])
        self._pick_palm_rot = rotvec_to_mat(w_axis * best) @ base
        if cfg.pick_hand_roll:  # then roll about the stick axis, + towards the back of the hand facing up
            r = rotvec_to_mat(w_axis * np.radians(cfg.pick_hand_roll))
            if (rotvec_to_mat(w_axis * -np.radians(cfg.pick_hand_roll)) @ self._pick_palm_rot)[2, 1] > (r @ self._pick_palm_rot)[2, 1]:
                r = r.T
            self._pick_palm_rot = r @ self._pick_palm_rot
        if cfg.pick_hand_tilt:  # pitch the whole hand (fingers and sticks) down about the horizontal across the sticks
            head = np.array([w_axis[0], w_axis[1], 0.0]) / np.linalg.norm(w_axis[:2])
            self._pick_palm_rot = rotvec_to_mat(np.cross([0.0, 0.0, 1.0], head) * np.radians(cfg.pick_hand_tilt)) @ self._pick_palm_rot
        if self._actuated:
            a, b, _ = pinch_points(self._hold)
        self._pick_spot = (a + b) / 2
        s = self._pick_palm_rot @ sep
        s[2] = 0.0
        x = s / np.linalg.norm(s)
        self._pick_object_rot = np.stack([x, np.cross([0, 0, 1.0], x), [0, 0, 1.0]], axis=1)
        durations = [cfg.pick_settle, cfg.pick_descend, cfg.pick_close, cfg.pick_lift]
        if cfg.pick_place:
            durations += [cfg.pick_carry, cfg.pick_lower, cfg.pick_release, cfg.pick_retreat]
        self._pick_steps = [max(1, round(t * cfg.control_frequency)) for t in durations]
        self._pick_open, self._pick_lifted = False, 0.0
        self._reset_pick_counters()

    def _reset_pick_counters(self):
        self._stick_unforced = np.zeros(2, dtype=int)
        self._pick_held = False
        self._unpinched = 0

    def _pick_alpha(self, step: int) -> float:
        starts = np.concatenate([[0], np.cumsum(self._pick_steps)])
        ramp = max(1, round(self.cfg.pick_grip_ramp * self.cfg.control_frequency))
        alpha = float(np.clip((step + 1 - starts[2]) / ramp, 0.0, 1.0))
        if self.cfg.pick_place:
            alpha = min(alpha, float(np.clip(1.0 - (step + 1 - starts[6]) / ramp, 0.0, 1.0)))
        return alpha

    def _reset_pick(self):
        cfg, rng, h = self.cfg, self.np_random, self._hold
        size = self.sim.cfg.object_size
        palm_rot = self._pick_palm_rot
        obj = np.array([0.0, 0.0, size]) + rng.uniform(-cfg.pick_object_noise, cfg.pick_object_noise, 3) * [1, 1, 0]
        obj_rot = rotvec_to_mat([0.0, 0.0, rng.uniform(-cfg.pick_object_yaw_noise, cfg.pick_object_yaw_noise)]) @ self._pick_object_rot
        grasp = np.array([0.0, 0.0, size]) - palm_rot @ self._pick_spot
        start, lifted = grasp + [0, 0, cfg.pick_approach], grasp + [0, 0, cfg.pick_lift_height]
        knots = [start, start, grasp, grasp, lifted]
        self._pick_b = None
        if cfg.pick_place:
            ang = rng.uniform(-np.pi, np.pi)
            shift = cfg.pick_place_dist * np.array([np.cos(ang), np.sin(ang), 0.0])
            self._pick_b = obj + shift
            above_b, at_b = lifted + shift, grasp + shift + [0, 0, cfg.pick_place_drop]
            knots += [above_b, at_b, at_b, above_b]
        self._pick_knots = knots
        self._pick_open, self._pick_lifted = False, 0.0
        self._reset_pick_counters()
        self._alpha = self._alpha_goal = self._alpha_vel = 0.0
        self._place(start, palm_rot, h)
        self.sim.place_object(obj, obj_rot)

    def _pick_palm_target(self, step: int) -> np.ndarray:
        k, t0 = self._pick_knots, 0
        for i, n in enumerate(self._pick_steps):
            if step < t0 + n:
                return k[i] + (k[i + 1] - k[i]) * (step - t0 + 1) / n
            t0 += n
        return k[-1]

    def _pick_phase(self, step: int) -> str:
        i = int(np.searchsorted(np.cumsum(self._pick_steps), step, side="right"))
        if i >= len(self._pick_steps):
            return "retreat" if self.cfg.pick_place else "carry"
        return self.PICK_PHASES[i]

    # ------------------------------------------------------------------ step
    def step(self, action):
        cfg, sim = self.cfg, self.sim
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0).copy()
        a[:6] = 0.0  # the wrist is locked (hold) or scripted (pick): only the fingers are the policy's
        step = self._step_count
        push = cfg.hold_push if cfg.stage == "hold" else cfg.pick_push
        if push and step % max(1, cfg.hold_push_every) == 0:
            self._push = self.np_random.uniform(-push, push, size=(2, 3))
        if cfg.stage == "pick":
            self._pick_open = cfg.pick_place and self._pick_phase(step) in ("release", "retreat")
            sim.set_palm_target(self._pick_palm_target(step), self._pick_palm_rot)
            if self._actuated:
                prev = self._alpha
                self._alpha = self._alpha_goal = self._pick_alpha(step)
                self._alpha_vel = (self._alpha - prev) * cfg.control_frequency
        elif self._actuated:
            self._step_alpha()
        residual = np.zeros(NUM_HAND_JOINTS)
        residual[OBS_FROM_SR] = a[6:]  # obs/action order -> SR order
        self._targets = np.clip(self._ref()["targets"] + residual * cfg.hold_residual, *sim.joint_limits.T)
        sim.set_targets(self._targets)
        self._prev_action = a
        forces = {n: f for n, f in zip(STICKS, self._push)} if (cfg.stage == "hold" or cfg.pick_push) else None
        if cfg.stage == "hold" and (self._wobble or self._shake):
            pos0, rot0 = self._palm_base
            h = sim.model.opt.timestep
            for j in range(self.substeps):  # move the palm target every physics step
                t = step * self.dt + (j + 1) * h
                pos, rot = pos0, rot0
                if self._shake:
                    pos = pos0 + self._shake[0] * self._shake[1] * np.sin(2 * np.pi * 3.0 * t)
                if self._wobble:
                    rot = rotvec_to_mat(self._wobble[0] * self._wobble[1] * np.sin(2 * np.pi * 2.0 * t)) @ rot0
                sim.set_palm_target(pos, rot)
                sim.step(1, forces)
        else:
            sim.step(self.substeps, forces)
        self._step_count += 1

        obs, info = self._observe()
        terms = self._reward_terms(info)
        reward = float(sum(terms.values()))
        info.update({f"reward_{k}": float(v) for k, v in terms.items()})
        terminated = self._lost(info) or not np.all(np.isfinite(obs))
        truncated = self._step_count >= cfg.steps_per_episode
        return obs, reward, bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ observation & info
    def _stick_tips(self) -> np.ndarray:
        return np.array([p + r @ tip_offset() for p, r in (self.sim.stick_pose(n) for n in STICKS)])

    def _observe(self):
        sim, cfg = self.sim, self.cfg
        palm_pos, palm_rot = sim.palm_pose()
        palm_v, palm_w = sim.body_vel(sim.palm_body)
        to_palm = palm_rot.T
        q, qd = sim.joints(), sim.joint_vel()
        proprio = np.concatenate([q[OBS_FROM_SR], 0.1 * qd[OBS_FROM_SR], self._targets[OBS_FROM_SR], palm_pos,
                                  rot6d(palm_rot), 0.1 * palm_v, 0.1 * palm_w])  # fmt: skip
        task, heights = [], []
        poses = [sim.stick_pose(n) for n in STICKS]
        for n, (pos, rot) in zip(STICKS, poses):
            heights.append(pos[2])
            v, _ = sim.body_vel(sim.stick_body[n])
            task += [to_palm @ (pos - palm_pos), rot6d(to_palm @ rot), to_palm @ v, to_palm @ (pos + rot @ tip_offset() - palm_pos)]
        obj_pos = sim.object_pose()[0] if sim.has_object else np.array([0.3, 0.0, 0.0])
        tips = self._stick_tips()
        tip_mid = tips.mean(0)
        if cfg.stage == "hold" and self._actuated:
            obj_vel = np.array([self._alpha, self._alpha_vel / max(cfg.hold_alpha_rate, 1e-6), self._alpha_goal])
        else:
            obj_vel = to_palm @ sim.body_vel(sim.object_body)[0] if sim.has_object else np.zeros(3)
        if cfg.stage == "pick" and cfg.pick_place:
            obj_vel[0] = float(self._pick_open)
        task += [to_palm @ (obj_pos - palm_pos), obj_vel, to_palm @ (obj_pos - tip_mid), [np.linalg.norm(tips[0] - tips[1])]]
        obs = np.concatenate([proprio, np.concatenate(task)]).astype(np.float32)

        info = {
            "stick_lift": float(min(heights) - 0.0035),
            "tip_lift": float(np.min(tips[:, 2] - self._tips_rest_height)) if hasattr(self, "_tips_rest_height") else 0.0,
            "tip_gap": float(np.linalg.norm(tips[0] - tips[1])),
            "palm_height": float(palm_pos[2]),
        }
        forces = sim.force_vectors()
        info.update(self._hold_stability(palm_pos, palm_rot, palm_v, palm_w, poses, forces))
        if cfg.stage == "hold":
            ref = self._ref()
            for (pos, rot), (ref_pos, ref_rot), name in zip(poses, ref["sticks"], ("upper", "lower")):
                info[f"{name}_drift"] = float(np.linalg.norm(to_palm @ (pos - palm_pos) - ref_pos))
                rel = to_palm @ rot
                info[f"{name}_spin"] = _angle_deg((np.trace(ref_rot.T @ rel) - 1) / 2)  # full rotation, spin included
                info[f"{name}_turn"] = info[f"{name}_spin"] if cfg.hold_turn == "full" else _angle_deg(ref_rot[:, 0] @ rel[:, 0])
            info["pose_error"] = float(np.sqrt(np.mean(np.square(q - ref["joints"]))))
            if self._actuated:
                info["alpha"], info["gap_ref"] = self._alpha, ref["gap"]
                info["gap_error"] = abs(info["tip_gap"] - ref["gap"])
                info["is_success"] = bool(info["gap_error"] < cfg.hold_gap_success and max(info["upper_drift"], info["lower_drift"]) < cfg.hold_max_drift / 2)
            else:
                info["is_success"] = bool(
                    max(info["upper_drift"], info["lower_drift"]) < cfg.hold_success_drift
                    and max(info["upper_turn"], info["lower_turn"]) < cfg.hold_success_turn
                )
        else:
            self._pick_info(palm_pos, palm_rot, poses, obj_pos, forces, info)
        return obs, info

    def _hold_stability(self, palm_pos, palm_rot, palm_v, palm_w, poses, forces) -> dict:
        sim = self.sim
        (pu, ru), (pl, rl) = poses
        (ref_pu, ref_ru), (ref_pl, ref_rl) = self._ref()["sticks"]
        out = {"pair_drift": float(np.linalg.norm(palm_rot.T @ (pl - pu) - (ref_pl - ref_pu)))}
        out["pair_turn"] = float(abs(_angle_deg(ru[:, 0] @ rl[:, 0]) - _angle_deg(ref_ru[:, 0] @ ref_rl[:, 0])))
        lin, ang = [], []
        for n, pos in zip(STICKS, (pu, pl)):
            v, w = sim.body_vel(sim.stick_body[n])
            lin.append(np.linalg.norm(v - palm_v - np.cross(palm_w, pos - palm_pos)))
            ang.append(np.linalg.norm(w - palm_w))
        out["slip_lin"], out["slip_ang"] = float(np.mean(lin)), float(np.mean(ang))
        for finger, i in HOLD_CONTACTS:
            out[f"force_{finger}_{('upper', 'lower')[i]}"] = float(np.linalg.norm(forces.get((finger, STICKS[i]), 0.0)))
        return out

    def _pick_info(self, palm_pos, palm_rot, poses, obj, forces, info):
        cfg, to_palm = self.cfg, palm_rot.T
        (pu, ru), (pl, rl) = poses
        (ref_pu, ref_ru), (ref_pl, _) = self._ref()["sticks"]
        web = np.array([-CHOPSTICK_LENGTH / 2 + 0.07, 0, 0])
        info["upper_drift"] = float(np.linalg.norm(to_palm @ (pu + ru @ web - palm_pos) - (ref_pu + ref_ru @ web)))
        info["lower_drift"] = float(np.linalg.norm(to_palm @ (pl - palm_pos) - ref_pl))
        half = CHOPSTICK_LENGTH / 2
        a = pu + ru[:, 0] * (half - cfg.pick_back)
        b = pl + rl[:, 0] * float((a - pl) @ rl[:, 0])
        info["object_to_spot"] = float(np.linalg.norm(obj - (a + b) / 2))
        info["object_lift"] = float(obj[2] - self.sim.cfg.object_size)
        pinch_axis = (a - b) / max(float(np.linalg.norm(a - b)), 1e-9)
        for name in STICKS:
            f = np.asarray(forces.get((name, "object"), np.zeros(3)))
            info[f"touch_{name}"] = float(np.linalg.norm(f))
            info[f"pinch_{name}"] = float(abs(f @ pinch_axis))
        info["pinch"] = min(info["pinch_upper"], info["pinch_lower"])
        info["both_touch"] = float(min(info["touch_upper"], info["touch_lower"]) > cfg.pick_touch_force)
        lost_steps = max(1, round(cfg.pick_lost_time * cfg.control_frequency))
        for i, name in enumerate(STICKS):
            total = sum(info[f"force_{f}_{name}"] for f, j in HOLD_CONTACTS if j == i)
            info[f"stick_force_{name}"] = float(total)
            self._stick_unforced[i] = self._stick_unforced[i] + 1 if total < cfg.hold_force_zero else 0
        info["sticks_lost"] = float(np.sum(self._stick_unforced >= lost_steps))
        phase = self._pick_phase(self._step_count)
        info["phase"] = self.PICK_PHASES.index(phase)
        held = max(info["upper_drift"], info["lower_drift"]) < cfg.pick_max_drift
        if info["both_touch"]:
            self._pick_lifted = max(self._pick_lifted, info["object_lift"])
        info["max_lift"] = self._pick_lifted
        info["open_cmd"] = float(self._pick_open)
        closed_gap = self._grips[1]["gap"] if self._actuated else self._hold["gap"]
        info["gap_open"] = info["tip_gap"] - closed_gap
        if self._actuated:
            info["alpha"] = self._alpha
        info["dropped"] = 0.0
        if info["both_touch"] and not self._pick_open:
            self._pick_held, self._unpinched = info["object_lift"] > cfg.pick_drop_lift, 0
        elif self._pick_held:
            if self._pick_open:
                self._pick_held = False
            elif not info["both_touch"]:
                self._unpinched += 1
                if self._unpinched >= max(1, round(cfg.pick_drop_time * cfg.control_frequency)):
                    info["dropped"], self._pick_held = 1.0, False
            else:
                self._unpinched = 0
        if not cfg.pick_place:
            info["is_success"] = bool(info["phase"] >= 3 and info["object_lift"] > cfg.pick_success_height and info["both_touch"] and held)
            return
        info["object_to_b"] = float(np.linalg.norm((obj - self._pick_b)[:2]))
        touching = max(info["touch_upper"], info["touch_lower"]) > cfg.pick_touch_force
        info["released"] = float(not touching and info["object_lift"] < 0.01)
        info["is_success"] = bool(
            self._pick_lifted > cfg.pick_success_height and info["released"]
            and info["object_to_b"] < cfg.pick_place_tolerance and held and phase == "retreat"
        )  # fmt: skip

    # ------------------------------------------------------------------ reward & termination
    def _hold_force_terms(self, info) -> tuple[float, float]:
        cfg = self.cfg
        forces = np.array([info[f"force_{f}_{('upper', 'lower')[i]}"] for f, i in HOLD_CONTACTS])
        below = np.clip(forces / cfg.hold_force_min, 0.0, 1.0)
        above = np.exp(-np.maximum(forces - cfg.hold_force_max, 0.0) / cfg.hold_force_max)
        return float(np.mean(below * above)), float(np.mean(forces < cfg.hold_force_zero))

    def _reward_terms(self, info) -> dict[str, float]:
        cfg = self.cfg
        act = -cfg.w_action * float(np.sum(np.square(self._prev_action[6:])))
        squeeze, lost = self._hold_force_terms(info)
        if cfg.stage == "hold":
            slip = info["slip_lin"] / cfg.hold_slip_lin_scale + info["slip_ang"] / cfg.hold_slip_ang_scale
            pair = np.exp(-info["pair_drift"] / cfg.hold_pair_drift_scale) + np.exp(-info["pair_turn"] / cfg.hold_pair_turn_scale)
            return {
                "hold_pos": cfg.w_hold_pos * np.mean([np.exp(-info[f"{n}_drift"] / 0.005) for n in STICKS]),
                "hold_rot": cfg.w_hold_rot * np.mean([np.exp(-info[f"{n}_turn"] / 5.0) for n in STICKS]),
                "hold_pose": cfg.w_hold_pose * np.exp(-info["pose_error"] / 0.1),
                "action": act,
                "success": cfg.w_hold_success * float(info["is_success"]),
                "hold_pair": cfg.w_hold_pair * 0.5 * pair,
                "hold_slip": -cfg.w_hold_slip * (1.0 - np.exp(-slip)),
                "hold_squeeze": cfg.w_hold_squeeze * squeeze,
                "hold_no_contact": -cfg.w_hold_no_contact * lost,
                "hold_gap": cfg.w_hold_gap * np.exp(-info["gap_error"] / cfg.hold_gap_scale) if self._actuated else 0.0,
            }  # fmt: skip
        touching = float(info["touch_upper"] > cfg.pick_touch_force) + float(info["touch_lower"] > cfg.pick_touch_force)
        not_open = 1.0 - info["open_cmd"]
        return {
            "pick_hold": cfg.w_pick_hold * np.mean([np.exp(-info[f"{n}_drift"] / 0.01) for n in STICKS]),
            "hold_squeeze": cfg.w_hold_squeeze * squeeze,
            "hold_no_contact": -cfg.w_hold_no_contact * lost,
            "pick_near": cfg.w_pick_near * np.exp(-info["object_to_spot"] / 0.02) * not_open,
            "pick_touch": cfg.w_pick_touch * 0.5 * touching * not_open,
            "pick_lift": cfg.w_pick_lift * info["both_touch"] * (
                np.clip(info["object_lift"] / cfg.pick_lift_height, 0.0, 1.0) if info["phase"] <= 4 else 1.0
            ) * not_open,
            "pick_open": cfg.w_pick_open * info["open_cmd"] * float(np.clip(info["gap_open"] / cfg.pick_open_gap, 0.0, 1.0)),
            "pick_released": cfg.w_pick_released * info["open_cmd"] * info.get("released", 0.0)
            * float(np.exp(-info.get("object_to_b", 0.0) / cfg.pick_place_tolerance)),
            "success": cfg.w_pick_success * float(info["is_success"]),
            "action": act,
            "pick_squeeze": cfg.w_pick_squeeze * not_open * float(
                np.clip(info["pinch"] / cfg.pick_squeeze_min, 0.0, 1.0)
                * np.exp(-max(info["pinch"] - cfg.pick_squeeze_max, 0.0) / cfg.pick_squeeze_max)
            ),
            "pick_stick_lost": -cfg.w_pick_stick_lost * info["sticks_lost"],
            "pick_drop": -cfg.w_pick_drop * info["dropped"],
        }  # fmt: skip

    def _lost(self, info) -> bool:
        cfg = self.cfg
        if cfg.stage == "hold":
            return max(info["upper_drift"], info["lower_drift"]) > cfg.hold_max_drift or max(info["upper_turn"], info["lower_turn"]) > cfg.hold_max_turn
        lost = max(info["upper_drift"], info["lower_drift"]) > cfg.pick_max_drift
        return bool(lost or (info["object_to_spot"] > cfg.pick_max_away and not info["open_cmd"]))

    # ------------------------------------------------------------------ rendering
    def render(self):
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.sim.model, 480, 640)
            self._cam = mujoco.MjvCamera()
        centre = np.mean([self.sim.stick_pose(n)[0] for n in STICKS] + [self.sim.palm_pose()[0]], axis=0)
        self._cam.lookat[:] = centre
        self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.4, 150, -25
        self._renderer.update_scene(self.sim.data, self._cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(overrides: dict | None = None) -> ChopsticksEnv:
    return ChopsticksEnv(ChopsticksEnvCfg.from_overrides(overrides))


