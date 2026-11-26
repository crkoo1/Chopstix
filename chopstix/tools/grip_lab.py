"""Grip lab: how a grip copes with gravity, then with growing disturbances. Writes runs/griplab/<grip>.json.

    .venv/Scripts/python.exe -m chopstix.tools.grip_lab --grip closed_v5_mj
    .venv/Scripts/python.exe -m chopstix.tools.grip_lab --grip closed_v5_mj --quick      # fewer seeds, for search

Tests (palm held still by its weld, finger targets fixed at the grip's unless jittered; physics = the training env's):
  gravity    no pushes, no noise, 10 s, at several palm orientations: the grip's own, the pick task's (the hand turned
             about the sticks so they sit side by side), and tilts of the grip's orientation. Per orientation: how far
             each stick moves over time, the finger forces at the end, and friction use (tangential / (mu x normal)
             force at each finger-stick contact; 1 = slipping).
  pushes     random force on each stick, uniform up to F per axis, new every 0.1 s, at F = 0 ... 0.2 N
  jitter     Gaussian noise on every finger target at each 20 Hz control step (what a learning policy's exploration
             does), sigma = 0 ... 0.03 rad
  start      uniform noise on the starting joint angles, 0 ... 0.06 rad
A run HOLDS if both sticks end within 1 cm of where they settled and their axes tilt less than 10 deg (palm frame),
and >= 3 of the thumb / index / middle / ring fingers still touch a stick. chops' stage-1 test limits the full
rotation instead; a chopstick is round, so spin about its own axis changes nothing about the grip and is only
reported ("spin_deg"), not judged.
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from chopstix.geometry import rot_angle_deg, rotvec_to_mat


def tilt_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    """Angle between two stick axes (local x), ignoring spin about the axis."""
    return float(np.degrees(np.arccos(np.clip(r0[:, 0] @ r1[:, 0], -1.0, 1.0))))
from chopstix.sim import STICKS, Sim, SimCfg, load_grip

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "griplab"
PALM_POS = np.array([0.0, 0.0, 0.3])
GRIP_FINGERS = ("th", "ff", "mf", "rf")
from chopstix.envs.chopsticks_env import ENV_SIM_DEFAULTS as TRAIN_SIM  # noqa: E402  (the training physics)
CTRL_DT = 0.05
# Light, training-like noise under the tests that change something else (motion, mass, friction, step size), so the
# runs differ from seed to seed: 0.03 N pushes (as in the hold stage) and 0.01 rad finger jitter.
BASE_NOISE = {"push": 0.03, "jitter": 0.01}
# name: (job field or "sim:<SimCfg field>", levels from easy to hard, palm pose, base noise on?)
TESTS = {
    "pushes": ("push", [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5], "grip", False),
    "pushes_pick": ("push", [0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5], "pick", False),
    "jitter": ("jitter", [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08], "grip", False),
    "start": ("start", [0.0, 0.01, 0.02, 0.04, 0.06, 0.1], "grip", False),
    "path": ("path", [1.0, 2.0, 3.0, 4.0], "pick", True),
    "shake": ("shake", [0.0, 2.0, 5.0, 10.0, 20.0, 30.0], "pick", True),
    "wobble": ("wobble", [0.0, 5.0, 10.0, 20.0, 30.0], "pick", True),
    "mass": ("sim:stick_mass", [0.01, 0.015, 0.02, 0.03], "pick", True),
    "friction": ("sim:friction", [1.0, 0.8, 0.6, 0.5, 0.4, 0.3], "pick", True),
    "timestep": ("sim:timestep", [0.002, 0.005, 0.01], "pick", True),
}
SHAKE_HZ, WOBBLE_HZ = 3.0, 2.0
# Palm path of the pick task (palm moves; no table, no cube): (seconds at 1x, displacement); "carry" is horizontal in a
# random direction. Played forward then backward to fill the run; faster speeds mean larger accelerations.
PATH = [(1.5, (0, 0, -0.05)), (1.5, None), (1.5, (0, 0, 0.06)), (2.0, "carry"), (1.0, (0, 0, -0.055)), (1.5, None), (1.0, (0, 0, 0.045))]
_SIM: dict = {}


def palm_motion(job: dict, rng, base_rot: np.ndarray):
    """A function t -> (palm position, palm rotation) for the job's motion, or None if the palm stays still."""
    if job.get("path"):
        ang = rng.uniform(-np.pi, np.pi)
        carry = 0.10 * np.array([np.cos(ang), np.sin(ang), 0.0])
        knots, times, t = [PALM_POS.copy()], [0.0], 0.0
        for dur, d in PATH:
            delta = carry if d == "carry" else np.zeros(3) if d is None else np.array(d, float)
            knots.append(knots[-1] + delta)
            t += dur
            times.append(t)
        knots, times, total = np.array(knots), np.array(times), t

        def f(tt):
            tau = (tt * job["path"]) % (2 * total)
            tau = tau if tau <= total else 2 * total - tau
            return np.array([np.interp(tau, times, knots[:, k]) for k in range(3)]), base_rot

        return f
    if job.get("shake"):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        w = 2 * np.pi * SHAKE_HZ
        amp = job["shake"] / w**2  # peak acceleration = amp * w^2
        return lambda tt: (PALM_POS + u * amp * np.sin(w * tt), base_rot)
    if job.get("wobble"):
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        w, th = 2 * np.pi * WOBBLE_HZ, np.radians(job["wobble"])
        return lambda tt: (PALM_POS, rotvec_to_mat(a * th * np.sin(w * tt)) @ base_rot)
    return None


def orientations(grip_name: str) -> dict[str, np.ndarray]:
    """Palm orientations to test gravity at (world-frame rotations of rh_palm)."""
    from chopstix.envs.chopsticks_env import make_env

    g = load_grip(grip_name)
    base = g["palm_rot"]
    pick = make_env({"stage": "pick", "hold_grip": grip_name})._pick_palm_rot
    out = {"grip pose": base, "pick task pose": pick}
    for name, axis, deg in (("tilted 45° forward", [0, 1, 0], 45), ("tilted 45° back", [0, 1, 0], -45),
                            ("rolled 45° left", [1, 0, 0], 45), ("rolled 45° right", [1, 0, 0], -45),
                            ("rolled 90° left", [1, 0, 0], 90), ("rolled 90° right", [1, 0, 0], -90),
                            ("upside down", [1, 0, 0], 180)):  # fmt: skip
        out[name] = rotvec_to_mat(np.radians(deg) * np.array(axis, float)) @ base
    return out


def _sim(sim_over: dict) -> Sim:
    key = json.dumps(sim_over, sort_keys=True)
    if key not in _SIM:
        _SIM[key] = Sim(SimCfg.from_overrides(sim_over))
    return _SIM[key]


def friction_use(sim: Sim) -> dict[str, float]:
    """Per grip finger, the highest tangential / (mu x normal) over its contacts with either stick."""
    m, d = sim.model, sim.data
    f6 = np.zeros(6)
    out = {f: 0.0 for f in GRIP_FINGERS}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        finger = sim._finger_of_body.get(b1) or sim._finger_of_body.get(b2)
        stick = sim._stick_of_body.get(b1) or sim._stick_of_body.get(b2)
        if finger not in out or not stick:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] > 1e-4:
            out[finger] = max(out[finger], float(np.hypot(f6[1], f6[2]) / (c.friction[0] * f6[0])))
    return out


def run(job: dict) -> dict:
    """One episode. job: grip, palm_rot (list), push, jitter, start, seed, duration, series (bool), record (bodies)."""
    from chopstix.tools.hold_test import load_into

    sim = _sim(job.get("sim", TRAIN_SIM))
    grip = load_grip(job["grip"])
    grip = {**grip, "palm_rot": np.array(job["palm_rot"]) if job.get("palm_rot") is not None else grip["palm_rot"]}
    rng = np.random.default_rng(job["seed"])
    dt = sim.model.opt.timestep
    n_ctrl = max(1, round(CTRL_DT / dt))
    sim.reset()
    sim.set_palm(PALM_POS, grip["palm_rot"])
    sim.set_joints(np.clip(grip["joints"] + rng.uniform(-job["start"], job["start"], 22), *sim.joint_limits.T) if job["start"] else grip["joints"])
    sim.set_targets(grip["targets"])
    for n in STICKS:
        sim.place_stick(n, *grip["sticks"][n])
    sim.set_gravity(False)
    sim.step(round(0.2 / dt))
    start = {n: sim.stick_pose(n, "palm") for n in STICKS}
    sim.set_gravity(True)
    bodies = [sim.model.body(b).id for b in job.get("record", [])]
    extra = [sim.stick_body["upper"], sim.stick_body["lower"]] + ([sim.object_body] if sim.has_object else [])

    def pose(b):
        p, q = sim.data.xpos[b], sim.data.xquat[b]
        return [round(float(x), 5) for x in (*p, q[1], q[2], q[3], q[0])]

    frames = [[pose(b) for b in bodies + extra]] if bodies else []
    series, first_fail = [], None
    steps = round(job["duration"] / CTRL_DT)
    forces = {n: np.zeros(3) for n in STICKS}
    motion = palm_motion(job, rng, grip["palm_rot"])
    for k in range(steps):
        if job["push"] and k % 2 == 0:  # new push every 0.1 s
            forces = {n: rng.uniform(-job["push"], job["push"], 3) for n in STICKS}
        if job["jitter"]:
            sim.set_targets(grip["targets"] + rng.normal(0.0, job["jitter"], 22))
        if motion is None:
            sim.step(n_ctrl, forces if job["push"] else None)
        else:  # move the palm target every physics step
            for j in range(n_ctrl):
                sim.set_palm_target(*motion((k * n_ctrl + j + 1) * dt))
                sim.step(1, forces if job["push"] else None)
        drift = {n: float(np.linalg.norm(sim.stick_pose(n, "palm")[0] - start[n][0])) for n in STICKS}
        rots = {n: sim.stick_pose(n, "palm")[1] for n in STICKS}
        turn = {n: tilt_deg(start[n][1], rots[n]) for n in STICKS}
        spin = {n: rot_angle_deg(start[n][1].T @ rots[n]) for n in STICKS}
        if first_fail is None and (max(drift.values()) > 0.01 or max(turn.values()) > 10):
            first_fail = round((k + 1) * CTRL_DT, 2)
        if job.get("series"):
            series.append([round(1000 * drift["upper"], 2), round(1000 * drift["lower"], 2), round(turn["upper"], 2), round(turn["lower"], 2)])
        if bodies:
            frames.append([pose(b) for b in bodies + extra])
    ff = sim.finger_forces()
    touching = [f for f in GRIP_FINGERS if max(ff[f].values()) > 0.01]
    holds = max(drift.values()) < 0.01 and max(turn.values()) < 10 and len(touching) >= 3
    out = {"seed": job["seed"], "holds": bool(holds), "first_fail": first_fail, "drift_mm": {n: round(1000 * v, 2) for n, v in drift.items()},
           "turn_deg": {n: round(v, 2) for n, v in turn.items()}, "spin_deg": {n: round(v, 2) for n, v in spin.items()},
           "touching": touching}  # fmt: skip
    if job.get("series"):
        out["series"] = series
        out["forces"] = {f: {s: round(v, 3) for s, v in ff[f].items()} for f in ("th", "ff", "mf", "rf", "lf", "palm")}
        out["friction_use"] = {f: round(v, 3) for f, v in friction_use(sim).items()}
    if bodies:
        out["frames"] = frames
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def test_job(test: str, level: float, grip: str, seed_index: int, duration: float, sim_over: dict, orient: dict) -> dict:
    field, _, pose, base = TESTS[test]
    job = {"grip": grip, "palm_rot": orient["pick task pose"].tolist() if pose == "pick" else None, "push": 0.0, "jitter": 0.0,
           "start": 0.0, "seed": 1000 * (1 + seed_index), "duration": duration, "sim": dict(sim_over)}  # fmt: skip
    if base:
        job.update(BASE_NOISE)
    if field.startswith("sim:"):
        job["sim"][field[4:]] = level
    else:
        job[field] = level
    return job


def lab(grip: str, seeds: int = 20, duration: float = 10.0, workers: int = 8, record_bodies=None, sim_over=None, log=print,
        tests: list[str] | None = None) -> dict:
    """Run the battery (all TESTS, or `tests`) on one grip under sim_over physics (default: the training env's)."""
    sim_over = sim_over or TRAIN_SIM
    orient = orientations(grip)
    jobs, tags = [], []
    for name, rot in orient.items():
        rec = record_bodies if (record_bodies and name in ("grip pose", "pick task pose")) else None
        jobs.append({"grip": grip, "palm_rot": rot.tolist(), "push": 0.0, "jitter": 0.0, "start": 0.0, "seed": 0,
                     "duration": duration, "series": True, "record": rec or [], "sim": sim_over})  # fmt: skip
        tags.append(("gravity", name))
    if record_bodies:  # 3-D clips of the palm-motion tests, at a level the grip should survive
        for test, lv, title in (("path", 2.0, "pick path at 2x speed"), ("shake", 10.0, "palm shaking at 10 m/s²"),
                                ("wobble", 20.0, "palm wobbling ±20°")):  # fmt: skip
            jobs.append({**test_job(test, lv, grip, 0, duration, sim_over, orient), "series": True, "record": record_bodies})
            tags.append(("clip", title))
    for test in tests or TESTS:
        for lv in TESTS[test][1]:
            for s in range(seeds):
                jobs.append(test_job(test, lv, grip, s, duration, sim_over, orient))
                tags.append((test, lv))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(run, jobs, chunksize=2))
    gravity, ladders, clips = [], {}, []
    for (test, key), r in zip(tags, res):
        if test == "gravity":
            g = {"pose": key, **{k: r[k] for k in ("holds", "first_fail", "drift_mm", "turn_deg", "touching", "series", "forces", "friction_use")}}
            gravity.append(g)
            if "frames" in r:
                clips.append({"pose": key, "title": f"gravity only · {key}", "frames": r["frames"], "series": r["series"]})
        elif test == "clip":
            clips.append({"pose": key, "title": key, "holds": r["holds"], "frames": r["frames"], "series": r["series"]})
        else:
            ladders.setdefault(test, {}).setdefault(key, []).append(r)
    out = {"grip": grip, "seeds": seeds, "duration": duration, "sim": sim_over, "gravity": gravity, "ladders": {}}
    for test, by in ladders.items():
        rows = []
        for lv in TESTS[test][1]:  # easy -> hard, as listed
            rs = by[lv]
            k = sum(r["holds"] for r in rs)
            rows.append({"level": lv, "k": k, "n": len(rs), "p": round(k / len(rs), 4), "ci": wilson(k, len(rs)),
                         "median_fail_s": float(np.median([r["first_fail"] for r in rs if r["first_fail"] is not None])) if any(r["first_fail"] is not None for r in rs) else None,
                         "worst_mm": round(max(max(r["drift_mm"].values()) for r in rs), 1)})  # fmt: skip
        out["ladders"][test] = rows
    out["score"] = score(out)
    out["seconds"] = round(time.time() - t0)
    log(summary(out))
    return out | ({"clips": clips} if clips else {})


def score(out: dict) -> dict:
    """Headline robustness numbers: the largest level each ladder still holds at >= 90%, gravity worst case."""
    def max_ok(test):
        ok = [r["level"] for r in out["ladders"][test] if r["p"] >= 0.9]
        # the ladder is only as good as its first failing rung
        lv = 0.0
        for r in out["ladders"][test]:
            if r["p"] < 0.9:
                break
            lv = r["level"]
        return lv if ok else None
    grav = {g["pose"]: g for g in out["gravity"]}
    s = {
        "gravity_poses_held": sum(g["holds"] for g in out["gravity"]), "gravity_poses": len(out["gravity"]),
        "grip_pose_held": grav["grip pose"]["holds"], "pick_pose_held": grav["pick task pose"]["holds"],
        "max_friction_use": round(max(grav["grip pose"]["friction_use"].values()), 3),
        "max_finger_force_N": round(max(max(v.values()) for f, v in grav["grip pose"]["forces"].items() if f != "palm"), 2),
    }  # fmt: skip
    names = {"pushes": "max_push_N", "pushes_pick": "max_push_pick_N", "jitter": "max_jitter_rad", "start": "max_start_rad"}
    for test in out["ladders"]:
        s[names.get(test, f"held_{test}")] = max_ok(test)
    return s


def summary(out: dict) -> str:
    lines = [f"grip {out['grip']}  ({out['seeds']} seeds x {out['duration']:.0f} s per rung)"]
    lines.append("  gravity (no disturbance), stick move after 10 s [upper/lower mm, axis tilt deg]:")
    for g in out["gravity"]:
        lines.append(f"    {g['pose']:20s} {'HOLDS' if g['holds'] else 'drops'}  {g['drift_mm']['upper']:6.1f}/{g['drift_mm']['lower']:6.1f} mm"
                     f"  {g['turn_deg']['upper']:5.1f}/{g['turn_deg']['lower']:5.1f} deg  friction use {max(g['friction_use'].values()):.2f}"
                     f"  first fail {g['first_fail']}")  # fmt: skip
    for test, rows in out["ladders"].items():
        lines.append(f"  {test:12s} " + "  ".join(f"{r['level']:g}:{r['p']:.0%}" for r in rows))
    lines.append(f"  score {out['score']}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grip", default="closed_v5_mj4")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="8 seeds x 5 s")
    p.add_argument("--record", action="store_true", help="record 3-D clips of the gravity runs (grip and pick poses)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="physics overrides on top of the training env's")
    p.add_argument("--tag", default="", help="suffix for the output file (e.g. when testing other physics)")
    p.add_argument("--tests", nargs="*", default=None, choices=list(TESTS), help="only these ladders (default: all)")
    args = p.parse_args(argv)
    if args.quick:
        args.seeds, args.duration = 8, 5.0
    bodies = None
    if args.record:
        from chopstix.tools.report import scene_meshes

        bodies = scene_meshes()[1]
    from chopstix.tools.common import parse_overrides

    sim_over = {**TRAIN_SIM, **parse_overrides(args.set)}
    out = lab(args.grip, args.seeds, args.duration, args.workers, bodies, sim_over, tests=args.tests)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.grip}{'_' + args.tag if args.tag else ''}{'_quick' if args.quick else ''}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
