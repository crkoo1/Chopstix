"""Design a human-style chopstick grip (user, 2026-11-23), for the back-of-the-hand-up pick pose.

The working grips hold both sticks across the fingers, pointing out of the palm, with the upper stick squeezed between
the index and middle fingers and no thumb on it. This designs the grip a person uses instead:

    upper stick: thumb pad, index fingertip, middle fingertip (it pivots on the thumb to open and close)
    lower stick: wedged between the thumb's base and the base of the index finger, resting on the ring fingertip
    both sticks run along the fingers and leave the hand past the fingertips, so the hand stays high above the cube

Stage 1 (`design`, kinematic, fast): the finger joints (thumb, index, middle, ring) and both stick poses in the palm
frame are searched together (cross-entropy method) so that exactly those finger parts touch those sticks (MuJoCo
mj_geomDistance on the collision meshes, nothing else overlapping a stick), the sticks point along the fingers within
--max-dir-dev deg of a human hold, stick out at least --protrude past the hand, and their tips meet.

Stage 2 (`physics`): each design is formed in MuJoCo (form_grip: relax with gravity off, then each finger presses
into its stick by a pinch amount, then the sticks are let go) and held in the pick pose (back of the hand up, sticks
--pitch deg down) under gravity with finger jitter and stick pushes. Random pinch amounts per design; the best
(sticks stay put, every designed contact still pressing) are saved as grips hg_<design>_<k>.

    .venv/Scripts/python.exe -m chopstix.tools.human_grip design --out runs/human_grip/design --restarts 8
    .venv/Scripts/python.exe -m chopstix.tools.human_grip physics --designs 4,12,3 --tries 24
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

from chopstix.geometry import CHOPSTICK_LENGTH, SR_JOINT_NAMES, mat_to_rotvec
from chopstix.sim import STICKS, Sim, SimCfg, load_grip

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SIM = {"timestep": 0.005, "noslip_iterations": 10, "actuator_timeconst": 0.15}
BASE_GRIP = "closed_v5_mj3_sq35_makis240"  # start posture for the fingers (its sticks are discarded)
FREE = [j for j in SR_JOINT_NAMES if j[3:5] in ("TH", "FF", "MF", "RF")]  # little finger stays curled out of the way
HALF = CHOPSTICK_LENGTH / 2
# Human hold in the palm frame (z: towards the fingertips, -y: the palm side, +x: the thumb side).
HUMAN_DIR = np.array([-0.25, -0.40, 0.88]) / np.linalg.norm([-0.25, -0.40, 0.88])

TOUCH = {  # stick -> groups of bodies; each group must touch that stick (closest body of the group counts)
    "upper": [["rh_thdistal"], ["rh_ffdistal", "rh_ffmiddle"], ["rh_mfdistal", "rh_mfmiddle"]],
    "lower": [["rh_thproximal", "rh_thmiddle"], ["rh_ffproximal", "rh_ffknuckle", "rh_palm"], ["rh_rfdistal", "rh_rfmiddle"]],
}

_W: dict = {}
PALM0 = np.array([0.0, 0.0, 0.3])


def _world():
    if "sim" not in _W:
        sim = Sim(SimCfg.from_overrides(TRAIN_SIM))
        sim.set_palm(PALM0, np.eye(3))  # off the floor; sticks are placed in the palm frame
        m = sim.model
        coll = lambda b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]  # noqa: E731
        hand = {m.body(n).id: m.body(n).name for n in range(m.nbody) if m.body(n).name.startswith("rh_")}
        _W.update(sim=sim, geoms={name: coll(b) for b, name in hand.items()},
                  stick_geom={n: coll(sim.stick_body[n])[0] for n in STICKS}, base=load_grip(BASE_GRIP))  # fmt: skip
        _W["hand_ids"] = set(hand)
        _W["free"] = np.array([SR_JOINT_NAMES.index(j) for j in FREE])
    return _W


def _dir(th, ph):
    """Unit vector at angle th from HUMAN_DIR, rotated ph about it."""
    a = np.cross(HUMAN_DIR, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(HUMAN_DIR, a)
    return np.cos(th) * HUMAN_DIR + np.sin(th) * (np.cos(ph) * a + np.sin(ph) * b)


def _stick_rot(d):
    x = d / np.linalg.norm(d)
    y = np.cross([0.0, 1.0, 0.0] if abs(x[1]) < 0.9 else [1.0, 0.0, 0.0], x)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def unpack(x):
    """x: free joints (len FREE), then per stick: centre (3, m) and direction (th, ph)."""
    w = _world()
    q = w["base"]["joints"].copy()
    q[w["free"]] = x[: len(FREE)]
    k = len(FREE)
    sticks = {}
    for n in STICKS:
        c, th, ph = x[k : k + 3], x[k + 3], x[k + 4]
        sticks[n] = (np.array(c), _stick_rot(_dir(th, ph)))
        k += 5
    return q, sticks


def x0():
    w = _world()
    q = w["base"]["joints"][w["free"]]
    # upper: through the point between the thumb pad and the index tip; lower: 15 mm further to the palm side
    up = np.array([25.0, -45.0, 120.0]) / 1000 + HUMAN_DIR * 0.05
    lo = np.array([20.0, -60.0, 105.0]) / 1000 + HUMAN_DIR * 0.05
    return np.concatenate([q, up, [0.05, 0.0], lo, [0.05, np.pi]])


def scale():
    return np.concatenate([np.full(len(FREE), 0.25), [0.012] * 3, [0.12, 1.0], [0.012] * 3, [0.12, 1.0]])


def _place(q, sticks):
    sim = _world()["sim"]
    sim.data.qpos[sim._jq] = np.clip(q, *sim.joint_limits.T)
    for n in STICKS:
        sim.place_stick(n, *sticks[n])
    mujoco.mj_fwdPosition(sim.model, sim.data)  # kinematics + contacts (for finger-finger overlaps)


def _dist(body, stick, cap=0.03):
    sim, w = _world()["sim"], _world()
    ft = np.zeros(6)
    return min((mujoco.mj_geomDistance(sim.model, sim.data, g, w["stick_geom"][stick], cap, ft) for g in w["geoms"][body]),
               default=cap)  # fmt: skip


def evaluate(x, detail=False):
    """Cost (lower is better) of a design; with detail, the parts."""
    w = _world()
    sim = w["sim"]
    q, sticks = unpack(x)
    lo, hi = sim.joint_limits[w["free"]].T
    cost = 1e3 * float(np.sum(np.clip(lo - x[: len(FREE)], 0, None) + np.clip(x[: len(FREE)] - hi, 0, None)))
    _place(q, sticks)
    parts = {}
    used = {n: set() for n in STICKS}
    for n, groups in TOUCH.items():
        for grp in groups:
            d = min(_dist(b, n) for b in grp)
            used[n].update(grp)
            mm = 1000 * d
            c = (3 * (mm + 0.2)) ** 2 if mm > -0.2 else (3 * (-mm - 0.9)) ** 2 if mm < -0.9 else 0.0  # pressing in 0.2 .. 0.9 mm (form_grip needs real contacts)
            parts[f"{n}:{grp[0][3:]}"] = round(mm, 2)
            cost += c
    over = {}
    for n in STICKS:
        for b in w["geoms"]:
            if b in used[n] or not w["geoms"][b]:
                continue
            mm = 1000 * _dist(b, n, 0.002)
            if mm < 0.5:  # nothing else may touch or cut into a stick
                over[f"{n}:{b[3:]}"] = round(mm, 2)
                cost += 4 * (0.5 - mm) ** 2 + 2
    m, d = sim.model, sim.data
    hand_ids = w["hand_ids"]
    self_mm = 0.0
    for i in range(d.ncon):  # fingers must not overlap each other (the physics would push them apart)
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] in hand_ids and m.geom_bodyid[c.geom2] in hand_ids and c.dist < -0.0002:
            self_mm = max(self_mm, -1000 * c.dist)
    if self_mm:
        cost += (3 * (self_mm - 0.2)) ** 2 + 2
    parts["finger_overlap"] = round(self_mm, 2)
    # sticks: along the fingers, tips meeting ahead of the hand, sticking out far enough, not through each other
    pts = {}
    for n in STICKS:
        c, r = sticks[n]
        pts[n] = (c + r[:, 0] * HALF, c - r[:, 0] * HALF, r[:, 0])
    gap = np.linalg.norm(pts["upper"][0] - pts["lower"][0])
    cost += 0.5 * (1000 * max(0.0, gap - 0.008)) ** 2 / 10
    far = max(float((sim.data.xpos[sim.model.body(b).id] - PALM0) @ pts[n][2]) for b in ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_thdistal") for n in STICKS)
    out = {n: float(pts[n][0] @ pts[n][2]) - far - 0.015 for n in STICKS}  # tip past the furthest fingertip (minus its pad)
    for n in STICKS:
        cost += (1000 * max(0.0, ARGS_PROTRUDE[0] - out[n])) ** 2 / 20
    ss = 1000 * mujoco.mj_geomDistance(sim.model, sim.data, w["stick_geom"]["upper"], w["stick_geom"]["lower"], 0.01, np.zeros(6))
    cost += (3 * max(0.0, -ss)) ** 2  # sticks may touch at the tips, not cross through each other
    dev = {n: float(np.degrees(np.arccos(np.clip(pts[n][2] @ HUMAN_DIR, -1, 1)))) for n in STICKS}
    for n in STICKS:
        cost += max(0.0, dev[n] - ARGS_MAX_DEV[0]) ** 2
    if not detail:
        return cost
    return {"cost": round(cost, 3), "touch_mm": parts, "unwanted_mm": over, "tip_gap_mm": round(1000 * gap, 1),
            "protrude_mm": {n: round(1000 * v, 1) for n, v in out.items()}, "dir_dev_deg": {n: round(v, 1) for n, v in dev.items()},
            "stick_stick_mm": round(ss, 2)}  # fmt: skip


def _set_env(env, rot=None):
    global PICK_ENV, FIXED_ROT
    PICK_ENV, FIXED_ROT = env, rot


ARGS_PROTRUDE = [0.13]
ARGS_MAX_DEV = [25.0]


def cem(seed, iters=250, pop=64, elite=8, start=None, protrude=0.13, max_dev=25.0):
    ARGS_PROTRUDE[0], ARGS_MAX_DEV[0] = protrude, max_dev
    rng = np.random.default_rng(seed)
    mu = x0() if start is None else np.array(start)
    sd = scale() * (1.0 if start is None else 0.3)
    if start is None:
        mu = mu + rng.normal(0, 0.5, mu.shape) * scale()
    best = (evaluate(mu), mu)
    for it in range(iters):
        xs = mu + rng.normal(size=(pop, len(mu))) * sd
        cs = np.array([evaluate(x) for x in xs])
        idx = np.argsort(cs)[:elite]
        if cs[idx[0]] < best[0]:
            best = (float(cs[idx[0]]), xs[idx[0]].copy())
        mu = 0.7 * xs[idx].mean(0) + 0.3 * best[1]
        sd = np.maximum(0.85 * sd + 0.15 * xs[idx].std(0), scale() * 0.01)
    return {"seed": seed, "cost": best[0], "x": best[1].tolist(), "detail": evaluate(best[1], detail=True)}


def to_grip(x, name: str, palm_rot=None) -> dict:
    q, sticks = unpack(np.array(x))
    return {"palm_rotvec": mat_to_rotvec(np.eye(3) if palm_rot is None else palm_rot).tolist(),
            "joint_targets": dict(zip(SR_JOINT_NAMES, map(float, q))), "joints": dict(zip(SR_JOINT_NAMES, map(float, q))),
            "sticks": {n: {"pos": sticks[n][0].tolist(), "rotvec": mat_to_rotvec(sticks[n][1]).tolist()} for n in STICKS},
            "made_by": f"chopstix.tools.human_grip design ({name})", "sim": TRAIN_SIM, "env": {}}  # fmt: skip


PICK_ENV = None  # s3_pick25's env + the new hand pose; set in main
PINCH_KEYS = ["th-upper", "ff-upper", "mf-upper", "th-lower", "ff-lower", "rf-lower"]
GRIPS = ROOT / "chopstix" / "grips"


FIXED_ROT = None  # --rot-pair: one pick pose for every candidate (that pair's pose in the env), else per grip


def pick_rot(grip_name: str) -> np.ndarray:
    from chopstix.envs.chopsticks_env import make_env

    if FIXED_ROT is not None:
        return np.array(FIXED_ROT)
    return make_env({**PICK_ENV, "hold_grip": grip_name})._pick_palm_rot


def pair_rot(open_grip: str, closed_grip: str, level_open: bool) -> np.ndarray:
    from chopstix.envs.chopsticks_env import make_env

    return make_env({**PICK_ENV, "hold_grip": open_grip, "hold_grip_closed": closed_grip, "pick_level_open": level_open})._pick_palm_rot


def hold_check(grip: dict, rot, seed: int, secs: float = 3.0, jitter: float = 0.01, push: float = 0.03) -> dict:
    """Gravity on, palm still at rot; finger jitter every 0.05 s, stick pushes every 0.1 s. Stick drift in the palm frame."""
    from chopstix.tools.grip_lab import PALM_POS, tilt_deg

    sim = Sim(SimCfg.from_overrides(TRAIN_SIM))
    rng = np.random.default_rng(seed)
    sim.reset()
    sim.set_palm(PALM_POS, rot)
    sim.set_joints(grip["joints"])
    sim.set_targets(grip["targets"])
    for n in STICKS:
        sim.place_stick(n, *grip["sticks"][n])
    sim.forward()
    dt = sim.model.opt.timestep
    start = {n: sim.stick_pose(n, "palm") for n in STICKS}
    worst = {n: [0.0, 0.0] for n in STICKS}
    forces = None
    for k in range(round(secs / 0.05)):
        sim.set_targets(np.clip(grip["targets"] + rng.normal(0, jitter, 22) * (k > 4), *sim.joint_limits.T))
        f = {n: rng.uniform(-push, push, 3) for n in STICKS} if k % 2 == 0 and k > 4 else (f if k > 4 else None)
        sim.step(round(0.05 / dt), forces=f)
        for n in STICKS:
            p, r = sim.stick_pose(n, "palm")
            worst[n] = [max(worst[n][0], float(np.linalg.norm(p - start[n][0]))), max(worst[n][1], tilt_deg(start[n][1], r))]
            if worst[n][0] > 0.05:
                return {"ok": False, "lost": n, "drift_mm": {m: round(1000 * w[0], 1) for m, w in worst.items()}}
    forces = sim.finger_forces()
    return {"ok": True, "drift_mm": {n: round(1000 * w[0], 1) for n, w in worst.items()},
            "tilt_deg": {n: round(w[1], 1) for n, w in worst.items()},
            "forces": {f"{a}-{s}": round(v, 3) for a, d in forces.items() for s, v in d.items() if v > 0.005}}  # fmt: skip


def physics_job(job: dict) -> dict:
    from types import SimpleNamespace

    from chopstix.tools.form_grip import form

    q, sticks = unpack(np.array(job["x"]))
    src = {"palm_rot": np.eye(3), "joints": q, "sticks": sticks}
    a = SimpleNamespace(pinch=[f"{k}={v}" for k, v in job["pinch"].items()], relax=1.0, clamp_solref=[0.05, 1.0],
                        stick_damping=[1.0, 0.01])  # fmt: skip
    sim = Sim(SimCfg.from_overrides(TRAIN_SIM))
    g = form(sim, src, a, log=lambda *x: None)
    g.update(made_by=f"chopstix.tools.human_grip physics (design {job['design']}, pinch {job['pinch']})", sim=TRAIN_SIM)
    name = f"hg_tmp_{job['design']}_{job['k']}" if not job.get("seed") else f"hg_tmp_s{job['seed']}_{job['design']}_{job['k']}"
    (GRIPS / f"{name}.json").write_text(json.dumps(g, indent=1, default=str))
    grip = load_grip(name)
    rot = pick_rot(name)
    checks = [hold_check(grip, rot, s) for s in range(job.get("seeds", 3))]
    ok = sum(c["ok"] for c in checks)
    drift = max((max(c["drift_mm"].values()) for c in checks), default=99)
    fz = [c.get("forces", {}) for c in checks if c["ok"]]
    need = ["th-upper", "ff-upper", "rf-lower"]
    have = min((sum(1 for k in need if f.get(k, 0) > 0.02) for f in fz), default=0)
    score = ok * 100 + have * 20 - drift
    return {**job, "name": name, "ok": ok, "drift_mm": drift, "have": have, "score": score, "checks": checks}


OPEN_JOINTS = ["rh_FFJ3", "rh_FFJ2", "rh_FFJ1", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_FFJ4", "rh_MFJ4", "rh_THJ1", "rh_THJ2"]


def cycle_job(job: dict) -> dict:
    from chopstix.tools import cycle_test

    r = cycle_test.run(job)
    opens = [x["gap_mm"] for x in r["checks"] if x["kind"] == "open"]
    return {"key": job["key"], "seed": job["seed"], "ok": r["ok"], "dropped": r["dropped"], "open_mm": min(opens) if opens else 0.0,
            "checks": r["checks"]}  # fmt: skip


def open_state(closed: str, open_targets, rot) -> dict:
    """Simulate closed -> open in the pick pose (no noise); the open grip file is that state (palm frame)."""
    from chopstix.tools.grip_lab import PALM_POS

    sim = Sim(SimCfg.from_overrides(TRAIN_SIM))
    c = load_grip(closed)
    dt = sim.model.opt.timestep
    sim.reset()
    sim.set_palm(PALM_POS, rot)
    sim.set_joints(c["joints"])
    sim.set_targets(c["targets"])
    for n in STICKS:
        sim.place_stick(n, *c["sticks"][n])
    sim.set_gravity(False)
    sim.step(round(0.2 / dt))
    sim.set_gravity(True)
    tg = np.array(open_targets)
    for k in range(20):
        sim.set_targets(c["targets"] + (k + 1) / 20 * (tg - c["targets"]))
        sim.step(round(0.05 / dt))
    sim.step(round(1.5 / dt))
    raw = json.loads((GRIPS / f"{closed}.json").read_text())
    q = sim.joints()
    poses = {n: sim.stick_pose(n, "palm") for n in STICKS}
    return {**raw, "joints": dict(zip(SR_JOINT_NAMES, map(float, q))), "joint_targets": dict(zip(SR_JOINT_NAMES, map(float, tg))),
            "sticks": {n: {"pos": p.tolist(), "rotvec": mat_to_rotvec(r).tolist()} for n, (p, r) in poses.items()},
            "pairs_with": closed, "made_by": f"chopstix.tools.human_grip opening (from {closed})"}  # fmt: skip


def opening(args):
    import shutil

    res = [r for r in json.loads(args.physics.read_text()) if r["ok"] == 3 and r["have"] == 3]
    res = sorted(res, key=lambda r: r["drift_mm"])[: args.closed]
    rng = np.random.default_rng(args.seed)
    idx = [SR_JOINT_NAMES.index(j) for j in OPEN_JOINTS]
    cands, rots = {}, {}
    screen_seeds = [0]
    if args.refine:  # perturb the best opening motions of an earlier search (pass >= 50% there)
        prev = [f for f in json.loads(args.refine.read_text())["finals"] if f["pass"] >= 0.5]
        screen_seeds = [100, 101, 102]  # not the gate's seeds
        for i, f in enumerate(prev):
            cl = f["closed"]
            rots.setdefault(cl, pick_rot(cl).tolist())
            ct, ot = load_grip(cl)["targets"], np.array(f["open_targets"])
            cands[f"{f['key']}|r{i}|0"] = (cl, ot.tolist())
            for k in range(args.dirs):
                noise = np.zeros(22)
                noise[idx] = rng.normal(size=len(idx)) * 0.06
                cands[f"{f['key']}|r{i}|{k + 1}"] = (cl, (ct + (ot - ct) * rng.uniform(0.8, 1.2) + noise).tolist())
    for r in [] if args.refine else res:
        rots[r["name"]] = pick_rot(r["name"]).tolist()
        c = load_grip(r["name"])
        for k in range(args.dirs):
            d = np.zeros(22)
            d[idx] = rng.normal(size=len(idx)) * np.array([1.0] * 8 + [0.5, 0.5])
            d /= np.linalg.norm(d)
            for size in (0.3, 0.5, 0.7):
                cands[f"{r['name']}|{k}|{size}"] = (r["name"], (c["targets"] + size * d).tolist())
    base = {"min_open": args.min_open, "back_tol": 0.02, "jitter": 0.01, "push": 0.03, "cycles": 3}
    jobs = [{**base, "key": key, "closed": cl, "open_targets": ot, "palm_rot": rots[cl], "seed": s}
            for key, (cl, ot) in cands.items() for s in screen_seeds]  # fmt: skip
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        first = list(ex.map(cycle_job, jobs, chunksize=4))
    by = {}
    for f in first:
        by.setdefault(f["key"], []).append(f)
    scored = [{"key": k, "ok": sum(x["ok"] for x in v), "open_mm": min(x["open_mm"] for x in v)} for k, v in by.items()]
    good = sorted([f for f in scored if f["ok"]], key=lambda f: (-f["ok"], -f["open_mm"]))
    print(f"screen: {len(good)} of {len(scored)} opening motions passed (runs each: {len(screen_seeds)}; {time.time() - t0:.0f} s)", flush=True)
    first = scored
    finals = []
    for f in good[: args.finalists]:
        cl, ot = cands[f["key"]]
        js = [{**base, "key": f["key"], "closed": cl, "open_targets": ot, "palm_rot": rots[cl], "seed": s} for s in range(args.gate_runs)]
        with ProcessPoolExecutor(args.workers) as ex:
            rr = list(ex.map(cycle_job, js))
        rate = sum(x["ok"] for x in rr) / len(rr)
        finals.append({"key": f["key"], "closed": cl, "open_targets": ot, "pass": rate, "open_mm": float(np.median([x["open_mm"] for x in rr]))})
        print(f"  {f['key']}: pass {rate:.0%} of {len(rr)}, tips open {finals[-1]['open_mm']:.0f} mm", flush=True)
    finals.sort(key=lambda x: (-x["pass"], -x["open_mm"]))
    out = {"finals": finals, "screen_passed": len(good), "screened": len(first)}
    if finals and finals[0]["pass"] >= args.gate:
        b = finals[0]
        n = 1 + max([int(x.stem.split("_")[1]) for x in GRIPS.glob("hg_*.json") if x.stem.split("_")[1].isdigit()] + [0])
        closed = f"hg_{n}"
        shutil.copy(GRIPS / f"{b['closed']}.json", GRIPS / f"{closed}.json")
        o = open_state(closed, b["open_targets"], np.array(rots[b["closed"]]))
        o["cycle_pass"] = b["pass"]
        opened = f"hg_{n}_open{round(b['open_mm'])}mm"
        (GRIPS / f"{opened}.json").write_text(json.dumps(o, indent=1, default=str))
        out["result"] = {"closed": closed, "open": opened, "pass": b["pass"], "open_mm": b["open_mm"], "from": b["key"]}
        print(f"PASS: {opened} -> {closed}: {b['pass']:.0%}", flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("design")
    d.add_argument("--out", type=Path, default=ROOT / "runs/human_grip/design")
    d.add_argument("--restarts", type=int, default=8)
    d.add_argument("--iters", type=int, default=250)
    d.add_argument("--workers", type=int, default=6)
    d.add_argument("--protrude", type=float, default=0.13, help="m: each stick's tip past the furthest fingertip")
    d.add_argument("--max-dir-dev", type=float, default=25.0, help="deg: stick direction vs a human hold")
    ph = sub.add_parser("physics")
    ph.add_argument("--design-file", type=Path, default=ROOT / "runs/human_grip/design/designs.json")
    ph.add_argument("--designs", default="", help="comma-separated design seeds (default: all)")
    ph.add_argument("--tries", type=int, default=24, help="random pinch settings per design")
    ph.add_argument("--workers", type=int, default=8)
    ph.add_argument("--pitch", type=float, default=20.0)
    ph.add_argument("--out", type=Path, default=ROOT / "runs/human_grip/physics.json")
    ph.add_argument("--seed", type=int, default=0, help="pinch sampling seed (also in the grip names when not 0)")
    op = sub.add_parser("opening")
    op.add_argument("--physics", type=Path, default=ROOT / "runs/human_grip/physics.json")
    op.add_argument("--closed", type=int, default=12, help="best closed grips to try")
    op.add_argument("--dirs", type=int, default=40, help="random opening directions per closed grip (x3 sizes)")
    op.add_argument("--min-open", type=float, default=0.045, help="m: tip gap when open (25 mm cube + margin)")
    op.add_argument("--finalists", type=int, default=8)
    op.add_argument("--gate-runs", type=int, default=20)
    op.add_argument("--gate", type=float, default=0.9)
    op.add_argument("--pitch", type=float, default=20.0)
    op.add_argument("--seed", type=int, default=0)
    op.add_argument("--refine", type=Path, help="an earlier opening.json: perturb its best motions instead of random ones")
    for sp in (ph, op):
        sp.add_argument("--rot-pair", default="", help="OPEN,CLOSED: test every candidate in this pair's pick pose")
        sp.add_argument("--level-open", action="store_true", help="with --rot-pair: the pose where the tips open sideways")
    op.add_argument("--workers", type=int, default=12)
    op.add_argument("--out", type=Path, default=ROOT / "runs/human_grip/opening.json")
    args = p.parse_args(argv)
    if args.cmd in ("physics", "opening"):
        global PICK_ENV, FIXED_ROT
        spec = json.loads((ROOT / "site/lineup.json").read_text(encoding="utf-8-sig"))["setups"]["s3_pick25"]["env"]
        PICK_ENV = {**{k: v for k, v in spec.items() if k != "hold_grip_closed"}, "pick_hand_flip": True, "pick_stick_pitch": args.pitch}
        if args.rot_pair:
            o, c = args.rot_pair.split(",")
            FIXED_ROT = pair_rot(o, c, args.level_open).tolist()
    if args.cmd == "opening":
        return opening(args)
    if args.cmd == "physics":
        designs = {r["seed"]: r for r in json.loads(args.design_file.read_text())}
        use = [int(s) for s in args.designs.split(",")] if args.designs else sorted(designs)
        rng = np.random.default_rng(args.seed)
        jobs = []
        for dsn in use:
            for k in range(args.tries):
                pinch = {key: float(np.round(np.exp(rng.uniform(np.log(0.005), np.log(0.15))), 4)) for key in PINCH_KEYS}
                jobs.append({"design": dsn, "k": k, "x": designs[dsn]["x"], "pinch": pinch, "seed": args.seed})
        t0 = time.time()
        with ProcessPoolExecutor(args.workers, initializer=_set_env, initargs=(PICK_ENV, FIXED_ROT)) as ex:
            res = sorted(ex.map(physics_job, jobs), key=lambda r: -r["score"])
        for r in res[:15]:
            print(f"design {r['design']} #{r['k']}: held {r['ok']}/3, drift {r['drift_mm']} mm, contacts {r['have']}/3, pinch {r['pinch']}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([{k: v for k, v in r.items() if k != "x"} for r in res], indent=1))
        print(f"{time.time() - t0:.0f} s; held 3/3: {sum(r['ok'] == 3 for r in res)} of {len(res)}")
    if args.cmd == "design":
        args.out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(cem, s, args.iters, 64, 8, None, args.protrude, args.max_dir_dev) for s in range(args.restarts)]
            res = sorted((f.result() for f in futs), key=lambda r: r["cost"])
        for r in res:
            print(f"seed {r['seed']}: cost {r['cost']:.2f}  {json.dumps(r['detail'])}", flush=True)
        (args.out / "designs.json").write_text(json.dumps(res, indent=1))
        print(f"{time.time() - t0:.0f} s; wrote {args.out / 'designs.json'}")


if __name__ == "__main__":
    main()
