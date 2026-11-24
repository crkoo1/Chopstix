"""MuJoCo scene for the chopsticks task, built to match Makis' SuperDex setup (chops.envs.chopsticks_env).

    Shadow Hand E from sr_description's MuJoCo model (the same convex collision meshes chops baked for SuperDex),
    with the forearm/wrist joints removed: the hand root is a free joint and the palm (rh_palm) is pulled onto a
    mocap target by a weld, the MuJoCo counterpart of SuperDex's palm pose controller.
    22 independent position actuators, one per sr_description joint (the J1/J2 tendon coupling is dropped so the
    action layout matches chops: 22 joint targets).
    Two free tapered chopsticks ("upper" = chops' stick_a, "lower" = stick_b), an optional cube/ball, a table at z=0.

Every physics knob lives in SimCfg so calibration (chopstix.tools.hold_test --set key=value) can sweep it.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import mujoco
import numpy as np

from chopstix.geometry import (
    CHOPSTICK_BASE_RADIUS,
    CHOPSTICK_LENGTH,
    CHOPSTICK_MASS,
    CHOPSTICK_TIP_RADIUS,
    SR_JOINT_NAMES,
    chopstick_vertices,
    mat_to_quat,

)

ROOT = Path(__file__).resolve().parents[1]
SR_MODELS = ROOT / "sr_common" / "sr_description" / "mujoco_models"
HAND_XML = SR_MODELS / "sr_hand_e_environment.xml"
STICKS = ("upper", "lower")

FINGER_BODIES = {
    "th": ["rh_thbase", "rh_thproximal", "rh_thhub", "rh_thmiddle", "rh_thdistal"],
    "ff": ["rh_ffknuckle", "rh_ffproximal", "rh_ffmiddle", "rh_ffdistal"],
    "mf": ["rh_mfknuckle", "rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"],
    "rf": ["rh_rfknuckle", "rh_rfproximal", "rh_rfmiddle", "rh_rfdistal"],
    "lf": ["rh_lfmetacarpal", "rh_lfknuckle", "rh_lfproximal", "rh_lfmiddle", "rh_lfdistal"],
    "palm": ["rh_palm"],
}


@dataclass
class SimCfg:
    # Solver.
    timestep: float = 0.002  # s
    integrator: str = "implicitfast"
    cone: str = "elliptic"
    impratio: float = 10.0
    noslip_iterations: int = 0
    multiccd: bool = True  # several contact points per convex pair (a stick on a finger pad is a line contact)
    iterations: int = 100
    # Hand actuation (chops defaults: joint_stiffness 5, joint_damping 0.1).
    joint_kp: float = 5.0  # N m / rad
    joint_kv: float = 0.1  # N m s / rad
    # First-order lag on each finger target (s; 0 = none). The real hand's tendon-driven joints can't follow a target
    # that jumps every control step; a lag also keeps a policy's exploration noise from shaking the sticks loose.
    actuator_timeconst: float = 0.0
    joint_damping: float = 0.05  # sr_description's passive joint damping
    joint_armature: float = 0.005
    joint_frictionloss: float = 0.0  # sr_description has 0.1; SuperDex's URDF hand had none
    palm_weld_solref: list[float] = field(default_factory=lambda: [0.01, 1.0])
    # Contacts. MuJoCo mixes two geoms' friction with max() (SuperDex seems to use the lower one).
    friction: float = 0.8  # hand and table
    stick_friction: float = 0.0  # 0 = friction
    object_friction: float = 0.0  # 0 = friction
    torsional_friction: float = 0.005  # m (condim >= 4)
    rolling_friction: float = 0.0001  # m (condim 6)
    condim: int = 4
    solref: list[float] = field(default_factory=lambda: [0.02, 1.0])
    solimp: list[float] = field(default_factory=lambda: [0.9, 0.95, 0.001, 0.5, 2.0])
    margin: float = 0.0
    # Stiffer contacts for the cube (stick-cube via solmix, table-cube via the table's own params): with the shared
    # solref/solimp a 5 g cube squeezed by the hand let a stick sink 11 mm into it and itself sink 10 mm into the table;
    # these bring both under 2 mm. Hand-stick contacts (the grip) keep solref/solimp above.
    object_solref: list[float] = field(default_factory=lambda: [0.01, 1.0])
    object_solimp: list[float] = field(default_factory=lambda: [0.99, 0.999, 0.001, 0.5, 2.0])
    # Props.
    stick_shape: str = "mesh"  # "mesh" (exact tapered stick) or "capsule" (mean radius)
    stick_mass: float = CHOPSTICK_MASS
    object: str = "cube"  # "cube", "ball" or "none"
    object_size: float = 0.015  # m, half side (cube) or radius (ball)
    object_mass: float = 0.005

    @classmethod
    def from_overrides(cls, overrides: dict | None = None) -> "SimCfg":
        cfg = cls()
        for k, v in (overrides or {}).items():
            if k not in {f.name for f in fields(cls)}:
                raise KeyError(f"unknown SimCfg field {k!r}")
            setattr(cfg, k, v)
        return cfg


def _symlink_assets(root: Path) -> dict[str, bytes]:
    """sr_common keeps duplicate_geoms/*.xml as git symlinks; on Windows they check out as text files holding the
    target path. Serve the real contents to MuJoCo instead (same fix as chops.dexgraspnet.shadow_hand)."""
    assets = {}
    for p in (root / "meshes" / "arm_and_hand_meshes" / "duplicate_geoms").glob("*.xml"):
        text = p.read_bytes()
        if not text.lstrip().startswith(b"<"):
            assets[p.relative_to(root).as_posix()] = (p.parent / text.decode().strip()).read_bytes()
    return assets


def build_model(cfg: SimCfg) -> mujoco.MjModel:
    if not HAND_XML.exists():
        raise FileNotFoundError(f"{HAND_XML} missing: run `git submodule update --init`")
    spec = mujoco.MjSpec.from_file(str(HAND_XML), include=_symlink_assets(SR_MODELS))

    opt = spec.option
    opt.timestep = cfg.timestep
    opt.integrator = {"euler": mujoco.mjtIntegrator.mjINT_EULER, "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                      "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT, "rk4": mujoco.mjtIntegrator.mjINT_RK4}[cfg.integrator]  # fmt: skip
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if cfg.cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.impratio = cfg.impratio
    opt.noslip_iterations = cfg.noslip_iterations
    opt.iterations = cfg.iterations
    if not cfg.multiccd:  # on by default in MuJoCo >= 3.3
        opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD

    # Hand: drop sr_description's motors, tendons and wrist joints; free root; light forearm/wrist without geoms.
    for a in list(spec.actuators):
        spec.delete(a)
    for t in list(spec.tendons):
        spec.delete(t)
    for j in ("rh_WRJ1", "rh_WRJ2"):
        spec.delete(spec.joint(j))
    for name in ("rh_forearm", "rh_wrist"):
        body = spec.body(name)
        for g in list(body.geoms):
            spec.delete(g)
        body.mass, body.inertia = 1e-3, [1e-7, 1e-7, 1e-7]
    spec.body("rh_forearm").add_freejoint(name="hand_root")

    friction = [cfg.friction, cfg.torsional_friction, cfg.rolling_friction]
    for g in spec.geoms:
        if g.contype or g.conaffinity:  # collision geoms (hand meshes, floor)
            g.friction, g.condim, g.solref, g.solimp, g.margin = friction, cfg.condim, cfg.solref, cfg.solimp, cfg.margin
            # Priority: in a hand-stick, hand-object or table-object contact the hand's/table's friction wins (chops
            # gives sticks and cube friction 2.0 so they grip each other, while hand and table stay at `friction`;
            # without priority MuJoCo would use the larger of the two everywhere).
            g.priority = 1
    for j in SR_JOINT_NAMES:
        joint = spec.joint(j)
        joint.damping = [cfg.joint_damping, 0.0, 0.0]  # (linear, polynomial terms) in MuJoCo >= 3.4
        joint.armature, joint.frictionloss = cfg.joint_armature, cfg.joint_frictionloss
        act = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if cfg.actuator_timeconst > 0:
            act.set_to_position(kp=cfg.joint_kp, kv=cfg.joint_kv, timeconst=cfg.actuator_timeconst)
        else:
            act.set_to_position(kp=cfg.joint_kp, kv=cfg.joint_kv)
        act.ctrlrange = list(joint.range)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # Palm target: a mocap body the palm is welded to (anchor, relative pose = identity, torque scale).
    spec.worldbody.add_body(name="palm_target", mocap=True)
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="palm_target", name2="rh_palm",
        data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=cfg.palm_weld_solref,
    )  # fmt: skip

    # Chopsticks.
    stick_fr = [cfg.stick_friction or cfg.friction, cfg.torsional_friction, cfg.rolling_friction]
    if cfg.stick_shape == "mesh":
        # scale: sr_description's default class scales meshes by 0.001 (its STLs are in mm); ours are in metres.
        spec.add_mesh(name="chopstick", uservert=chopstick_vertices().ravel().tolist(), scale=[1.0, 1.0, 1.0])
    for i, name in enumerate(STICKS):
        body = spec.worldbody.add_body(name=f"stick_{name}", pos=[0.0, 0.5 + 0.1 * i, 0.5])
        body.add_freejoint(name=f"stick_{name}")
        kw = dict(name=f"stick_{name}", mass=cfg.stick_mass, friction=stick_fr, condim=cfg.condim, solref=cfg.solref,
                  solimp=cfg.solimp, margin=cfg.margin, rgba=[0.72, 0.52, 0.30, 1])  # fmt: skip
        if cfg.stick_shape == "mesh":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="chopstick", **kw)
        else:
            r = (CHOPSTICK_BASE_RADIUS + CHOPSTICK_TIP_RADIUS) / 2
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, CHOPSTICK_LENGTH / 2 - r, 0],
                          quat=[0.70710678, 0, 0.70710678, 0], **kw)  # fmt: skip
        # Clamp (off by default): a mocap body the stick can be welded to while a grip is formed.
        spec.worldbody.add_body(name=f"clamp_{name}", mocap=True)
        spec.add_equality(
            name=f"clamp_{name}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"clamp_{name}", name2=f"stick_{name}", data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            solref=[0.005, 1.0], active=False,
        )  # fmt: skip

    if cfg.object != "none":
        body = spec.worldbody.add_body(name="object", pos=[0.3, 0.0, cfg.object_size])
        body.add_freejoint(name="object")
        body.add_geom(
            name="object", mass=cfg.object_mass, condim=cfg.condim, solref=cfg.solref, solimp=cfg.solimp,
            friction=[cfg.object_friction or cfg.friction, cfg.torsional_friction, cfg.rolling_friction],
            type=mujoco.mjtGeom.mjGEOM_BOX if cfg.object == "cube" else mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[cfg.object_size] * 3, rgba=[0.9, 0.15, 0.1, 1],
        )  # fmt: skip
        obj = spec.geom("object")
        obj.solref, obj.solimp, obj.solmix = cfg.object_solref, cfg.object_solimp, 1000.0  # its params win against a stick
        for g in spec.worldbody.geoms:  # the table (priority 1, so its params decide a table-cube contact)
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                g.solref, g.solimp = cfg.object_solref, cfg.object_solimp
    return spec.compile()


class Sim:
    """A built scene plus the helpers the grip tools and the env share. Joint arrays are in SR_JOINT_NAMES order;
    stick poses are (position of the stick's centre, rotation), in the world or the palm frame."""

    def __init__(self, cfg: SimCfg | None = None):
        self.cfg = cfg or SimCfg()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.gravity = m.opt.gravity.copy()
        self.substeps = 1
        self._root_q = m.jnt_qposadr[m.joint("hand_root").id]
        self._root_v = m.jnt_dofadr[m.joint("hand_root").id]
        self._jq = np.array([m.jnt_qposadr[m.joint(j).id] for j in SR_JOINT_NAMES])
        self._jv = np.array([m.jnt_dofadr[m.joint(j).id] for j in SR_JOINT_NAMES])
        self._act = np.array([m.actuator(j).id for j in SR_JOINT_NAMES])
        self._act_state = np.array([m.actuator_actadr[a] for a in self._act])  # -1 without a lag
        self.joint_limits = np.array([m.jnt_range[m.joint(j).id] for j in SR_JOINT_NAMES])
        self._mocap = m.body_mocapid[m.body("palm_target").id]
        self.palm_body = m.body("rh_palm").id
        self.stick_body = {n: m.body(f"stick_{n}").id for n in STICKS}
        self._stick_q = {n: m.jnt_qposadr[m.joint(f"stick_{n}").id] for n in STICKS}
        self._stick_v = {n: m.jnt_dofadr[m.joint(f"stick_{n}").id] for n in STICKS}
        self._clamp_mocap = {n: m.body_mocapid[m.body(f"clamp_{n}").id] for n in STICKS}
        self._clamp_eq = {n: m.equality(f"clamp_{n}").id for n in STICKS}
        self.has_object = self.cfg.object != "none"
        if self.has_object:
            self.object_body = m.body("object").id
            self._obj_q = m.jnt_qposadr[m.joint("object").id]
            self._obj_v = m.jnt_dofadr[m.joint("object").id]
        # Body -> finger name, for contact force bookkeeping.
        self._finger_of_body = {}
        for f, names in FINGER_BODIES.items():
            for n in names:
                self._finger_of_body[m.body(n).id] = f
        self._stick_of_body = {b: n for n, b in self.stick_body.items()}
        # rh_palm in the hand root (forearm) frame, to place the free root from a palm pose.
        mujoco.mj_kinematics(m, self.data)
        f = m.body("rh_forearm").id
        rf, rp = self.data.xmat[f].reshape(3, 3), self.data.xmat[self.palm_body].reshape(3, 3)
        self._palm_in_root = (rf.T @ (self.data.xpos[self.palm_body] - self.data.xpos[f]), rf.T @ rp)
        self.palm_cmd = (np.zeros(3), np.eye(3))  # commanded palm pose (mocap)

    # ------------------------------------------------------------------ setting state
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model.opt.gravity[:] = self.gravity

    def set_gravity(self, on: bool):
        self.model.opt.gravity[:] = self.gravity if on else 0.0

    def set_palm(self, pos, rot):
        """Teleport the hand so rh_palm is at (pos, rot), and put the palm target there."""
        p_in_r, r_in_r = self._palm_in_root
        root_rot = rot @ r_in_r.T
        root_pos = np.asarray(pos) - root_rot @ p_in_r
        self.data.qpos[self._root_q : self._root_q + 3] = root_pos
        self.data.qpos[self._root_q + 3 : self._root_q + 7] = mat_to_quat(root_rot)
        self.data.qvel[self._root_v : self._root_v + 6] = 0.0
        self.set_palm_target(pos, rot)

    def set_palm_target(self, pos, rot):
        self.palm_cmd = (np.array(pos, dtype=float), np.array(rot, dtype=float))
        self.data.mocap_pos[self._mocap] = pos
        self.data.mocap_quat[self._mocap] = mat_to_quat(rot)

    def set_joints(self, q):
        q = np.clip(q, *self.joint_limits.T)
        self.data.qpos[self._jq] = q
        self.data.qvel[self._jv] = 0.0
        self.set_targets(q)
        if (self._act_state >= 0).all():  # with a lag, its filtered target starts at the joints too
            self.data.act[self._act_state] = q

    def set_targets(self, q):
        self.data.ctrl[self._act] = np.clip(q, *self.joint_limits.T)

    def place_stick(self, name, pos, rot, frame="palm"):
        if frame == "palm":
            pp, pr = self.palm_cmd
            pos, rot = pp + pr @ pos, pr @ rot
        a = self._stick_q[name]
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = mat_to_quat(rot)
        self.data.qvel[self._stick_v[name] : self._stick_v[name] + 6] = 0.0

    def clamp(self, name, on: bool = True):
        """Weld a stick where it is now (on) or let it go (off)."""
        if on:
            pos, rot = self.data.qpos[self._stick_q[name] : self._stick_q[name] + 3], self.data.qpos[self._stick_q[name] + 3 : self._stick_q[name] + 7]
            self.data.mocap_pos[self._clamp_mocap[name]] = pos
            self.data.mocap_quat[self._clamp_mocap[name]] = rot
        self.data.eq_active[self._clamp_eq[name]] = on

    def set_stick_damping(self, linear: float = 0.0, angular: float = 0.0):
        """Viscous damping on the sticks' free joints (N s/m, N m s/rad); used while forming grips, 0 otherwise."""
        for n in STICKS:
            v = self._stick_v[n]
            self.model.dof_damping[v : v + 3] = linear
            self.model.dof_damping[v + 3 : v + 6] = angular

    def place_object(self, pos, rot=np.eye(3)):
        self.data.qpos[self._obj_q : self._obj_q + 3] = pos
        self.data.qpos[self._obj_q + 3 : self._obj_q + 7] = mat_to_quat(rot)
        self.data.qvel[self._obj_v : self._obj_v + 6] = 0.0

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1, forces: dict | None = None):
        """n physics steps; `forces` = {stick name or "object": world force (N) at its centre of mass}, held for all n."""
        self.data.xfrc_applied[:] = 0.0
        for name, f in (forces or {}).items():
            self.data.xfrc_applied[self.object_body if name == "object" else self.stick_body[name], :3] = f
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ------------------------------------------------------------------ reading state
    def joints(self) -> np.ndarray:
        return self.data.qpos[self._jq].copy()

    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self._jv].copy()

    def targets(self) -> np.ndarray:
        return self.data.ctrl[self._act].copy()

    def body_pose(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[body].copy(), self.data.xmat[body].reshape(3, 3).copy()

    def body_vel(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        """(linear, angular) velocity of a body's frame origin, in the world frame."""
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, v, 0)
        return v[3:], v[:3]

    def palm_pose(self):
        return self.body_pose(self.palm_body)

    def stick_pose(self, name, frame="world"):
        pos, rot = self.body_pose(self.stick_body[name])
        if frame == "palm":
            pp, pr = self.palm_pose()
            return pr.T @ (pos - pp), pr.T @ rot
        return pos, rot

    def object_pose(self):
        return self.body_pose(self.object_body)

    def _group(self, body: int) -> str:
        if body in self._finger_of_body:
            return self._finger_of_body[body]
        if body in self._stick_of_body:
            return self._stick_of_body[body]
        if self.has_object and body == self.object_body:
            return "object"
        return "table" if body == 0 else "other"

    def force_vectors(self) -> dict:
        """World-frame contact force on group A exerted by group B, as {(A, B): vector}, for every touching pair (both
        orders). Groups: finger names, 'palm', 'upper', 'lower', 'object', 'table'."""
        m, d = self.model, self.data
        out: dict = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = self._group(m.geom_bodyid[c.geom1]), self._group(m.geom_bodyid[c.geom2])
            if g1 == g2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            f = c.frame.reshape(3, 3).T @ f6[:3]  # force on geom2 from geom1 (the normal points geom1 -> geom2)
            out[(g2, g1)] = out.get((g2, g1), 0.0) + f
            out[(g1, g2)] = out.get((g1, g2), 0.0) - f
        return out

    def contact_forces(self) -> dict:
        """Total contact force magnitude (N) per (body group, body group) pair: groups are finger names, 'upper',
        'lower', 'object', 'table'. Keys are sorted tuples."""
        m, d = self.model, self.data
        out: dict = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            groups = []
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                if b in self._finger_of_body:
                    groups.append(self._finger_of_body[b])
                elif b in self._stick_of_body:
                    groups.append(self._stick_of_body[b])
                elif self.has_object and b == self.object_body:
                    groups.append("object")
                elif b == 0:
                    groups.append("table")
                else:
                    groups.append("other")
            mujoco.mj_contactForce(m, d, i, f6)
            key = tuple(sorted(groups))
            out[key] = out.get(key, 0.0) + float(np.linalg.norm(f6[:3]))
        return out

    def finger_forces(self, forces: dict | None = None) -> dict[str, dict[str, float]]:
        forces = self.contact_forces() if forces is None else forces
        return {f: {s: forces.get(tuple(sorted((f, s))), 0.0) for s in STICKS} for f in FINGER_BODIES}

    def penetration(self) -> dict:
        """Deepest penetration (m, positive = overlap) per body-group pair, from the current contacts."""
        m, d = self.model, self.data
        out: dict = {}
        for i in range(d.ncon):
            c = d.contact[i]
            names = []
            for g in (c.geom1, c.geom2):
                b = m.geom_bodyid[g]
                names.append(self._finger_of_body.get(b) or self._stick_of_body.get(b) or m.body(b).name)
            key = tuple(sorted(names))
            out[key] = max(out.get(key, -np.inf), -float(c.dist))
        return out


def load_grip(name_or_path: str) -> dict:
    """A chops grip file: palm rotation, joints and joint targets (SR order), stick poses in the palm frame."""
    import json

    from chopstix.geometry import rotvec_to_mat

    p = Path(name_or_path)
    if not p.suffix:
        p = Path(__file__).resolve().parent / "grips" / f"{name_or_path}.json"
    g = json.loads(p.read_text())
    return {
        "name": p.stem,
        "palm_rot": rotvec_to_mat(g["palm_rotvec"]),
        "joints": np.array([g["joints"][n] for n in SR_JOINT_NAMES]),
        "targets": np.array([g["joint_targets"][n] for n in SR_JOINT_NAMES]),
        "sticks": {n: (np.array(g["sticks"][n]["pos"]), rotvec_to_mat(g["sticks"][n]["rotvec"])) for n in STICKS},
        "raw": g,
    }


