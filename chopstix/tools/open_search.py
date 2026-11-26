"""Find an OPEN grip that pairs with a closed grip (same palm), for the open/close stage.

The lower stick stays pinched by the thumb's base (middle phalanx), the ring fingertip and the palm; opening moves only
the joints in --joints (default: index and middle fingers, plus the thumb's distal joints), so the upper stick pivots
against the thumb as in a human tripod grip. For each opening direction (Makis' open_v4 - closed_v4 finger motion
restricted to those joints, plus random extension directions) and each size, under gravity with the training physics:

    1. start in the closed grip, settle, then move the finger targets to closed + size x direction over 1 s, hold 2 s
    2. measure: tip gap, lower stick drift/tilt (must stay put), upper stick still touched by the thumb and the index
       or middle finger, upper stick not lost (< 3 cm from where the opening put it after 0.5 s more)
    3. close back over 1 s, hold 1 s: both sticks still held (lower stick within 1 cm / 10 deg of where it started, the
       upper touched by the thumb and the index or middle finger) and the tips closed again to within --back-tol of the
       closed grip's gap (closing does not return exactly: the upper stick re-seats a little differently)

Candidates that pass are saved as grips (chopstix/grips/<closed>_open<gap>mm.json, the open state's joints, targets
and stick poses) for the grip lab. Results: runs/griplab/open_search_<closed>.json.

    .venv/Scripts/python.exe -m chopstix.tools.open_search --closed closed_v5_mj4 --random 40
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chopstix.geometry import SR_JOINT_NAMES, mat_to_rotvec, tip_offset
from chopstix.sim import STICKS, load_grip
from chopstix.tools.grip_lab import PALM_POS, TRAIN_SIM, _sim, tilt_deg

ROOT = Path(__file__).resolve().parents[2]
GRIPS = ROOT / "chopstix" / "grips"
DEFAULT_JOINTS = ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1", "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1", "rh_THJ2", "rh_THJ1"]
SIZES = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]


def tip_gap(sim) -> float:
    t = [sim.stick_pose(n)[0] + sim.stick_pose(n)[1] @ tip_offset() for n in STICKS]
    return float(np.linalg.norm(t[0] - t[1]))


def trial(job: dict) -> dict:
    sim = _sim(job.get("sim", TRAIN_SIM))
    c = load_grip(job["closed"])
    d = np.array(job["dir"])
    dt = sim.model.opt.timestep
    n_ctrl = max(1, round(0.05 / dt))
    rng = np.random.default_rng(job.get("seed", 0))
    sim.reset()
    sim.set_palm(PALM_POS, c["palm_rot"])
    sim.set_joints(c["joints"])
    sim.set_targets(c["targets"])
    for n in STICKS:
        sim.place_stick(n, *c["sticks"][n])
    sim.set_gravity(False)
    sim.step(round(0.2 / dt))
    start = {n: sim.stick_pose(n, "palm") for n in STICKS}
    gap0 = tip_gap(sim)
    sim.set_gravity(True)
    jit = job.get("jitter", 0.0)

    def go(target, seconds):
        for k in range(round(seconds / 0.05)):
            frac = min(1.0, (k + 1) / max(1, round(1.0 / 0.05)))
            tgt = cur[0] + frac * (target - cur[0])
            sim.set_targets(np.clip(tgt + (rng.normal(0, jit, 22) if jit else 0.0), *sim.joint_limits.T))
            sim.step(n_ctrl)
        cur[0] = target

    cur = [c["targets"].copy()]
    open_t = np.clip(c["targets"] + job["size"] * d, *sim.joint_limits.T)
    go(open_t, 3.0)
    open_state = {"joints": sim.joints(), "sticks": {n: sim.stick_pose(n, "palm") for n in STICKS}}
    gap_open = tip_gap(sim)
    ff = sim.finger_forces()
    upper_by = [f for f in ("th", "ff", "mf") if ff[f]["upper"] > 0.01]
    lower_by = [f for f in ("th", "rf", "palm") if ff[f]["lower"] > 0.01]
    lower_drift = float(np.linalg.norm(open_state["sticks"]["lower"][0] - start["lower"][0]))
    lower_tilt = tilt_deg(start["lower"][1], open_state["sticks"]["lower"][1])
    sim.step(round(0.5 / dt))
    upper_settle = float(np.linalg.norm(sim.stick_pose("upper", "palm")[0] - open_state["sticks"]["upper"][0]))
    held_open = ("th" in upper_by and len(upper_by) >= 2 and len(lower_by) >= 2 and lower_drift < 0.01 and lower_tilt < 10
                 and upper_settle < 0.01)  # fmt: skip
    go(c["targets"], 2.0)
    back = {n: sim.stick_pose(n, "palm") for n in STICKS}
    gap_back = tip_gap(sim)
    ffb = sim.finger_forces()
    upper_back = [f for f in ("th", "ff", "mf") if ffb[f]["upper"] > 0.01]
    back_ok = (gap_back - gap0 < job.get("back_tol", 0.02) and "th" in upper_back and len(upper_back) >= 2
               and np.linalg.norm(back["lower"][0] - start["lower"][0]) < 0.01 and tilt_deg(start["lower"][1], back["lower"][1]) < 10)  # fmt: skip
    out = {**{k: job[k] for k in ("name", "size", "closed")}, "gap_closed_mm": round(1000 * gap0, 1),
           "gap_open_mm": round(1000 * gap_open, 1), "gap_back_mm": round(1000 * gap_back, 1), "upper_by": upper_by,
           "lower_by": lower_by, "lower_drift_mm": round(1000 * lower_drift, 1), "lower_tilt_deg": round(lower_tilt, 1),
           "upper_settle_mm": round(1000 * upper_settle, 1), "held_open": bool(held_open), "closes_back": bool(back_ok),
           "ok": bool(held_open and back_ok)}  # fmt: skip
    if job.get("keep"):
        out["open_state"] = {"joints": open_state["joints"].tolist(), "targets": open_t.tolist(),
                             "sticks": {n: {"pos": p.tolist(), "rotvec": mat_to_rotvec(r).tolist()} for n, (p, r) in open_state["sticks"].items()}}  # fmt: skip
    return out


def directions(closed: str, joints: list[str], n_random: int, seed: int = 0) -> dict[str, np.ndarray]:
    idx = [SR_JOINT_NAMES.index(j) for j in joints]
    mask = np.zeros(22)
    mask[idx] = 1.0
    o4, c4 = load_grip("open_v4"), load_grip("closed_v4")
    out = {}
    makis = (o4["targets"] - c4["targets"]) * mask
    out["makis"] = makis / np.linalg.norm(makis)
    # plain extension of index + middle (flexion joints back), thumb tip extended a little
    ext = np.zeros(22)
    for j in joints:
        if j.endswith(("J3", "J2", "J1")) and not j.startswith("rh_TH"):
            ext[SR_JOINT_NAMES.index(j)] = -1.0
    out["extend"] = ext / np.linalg.norm(ext)
    rng = np.random.default_rng(seed)
    for k in range(n_random):  # mostly extension, random mix
        v = np.zeros(22)
        v[idx] = rng.normal(size=len(idx))
        v = v - 0.8 * np.abs(ext) * np.abs(v)  # bias the flexion joints towards extension
        out[f"rand{k:02d}"] = v / np.linalg.norm(v)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--closed", default="closed_v5_mj4")
    p.add_argument("--joints", nargs="*", default=DEFAULT_JOINTS)
    p.add_argument("--random", type=int, default=40, help="random opening directions besides makis/extend")
    p.add_argument("--save", type=int, default=6, help="save this many passing candidates (widest gaps, distinct dirs)")
    p.add_argument("--sizes", type=float, nargs="*", default=SIZES, help="opening sizes (rad, along each unit direction)")
    p.add_argument("--back-tol", type=float, default=0.02, help="m: closing again must bring the tips this close to the closed gap")
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args(argv)
    dirs = directions(args.closed, args.joints, args.random)
    jobs = [{"name": n, "dir": d.tolist(), "size": s, "closed": args.closed, "back_tol": args.back_tol} for n, d in dirs.items() for s in args.sizes]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(trial, jobs, chunksize=2))
    ok = [r for r in res if r["ok"]]
    print(f"{len(res)} trials in {time.time() - t0:.0f} s, {len(ok)} pass (held open and closed back)")
    # the widest-opening passing size per direction, then robustness check with finger jitter on the best ones
    best = {}
    for r in ok:
        if r["name"] not in best or r["gap_open_mm"] > best[r["name"]]["gap_open_mm"]:
            best[r["name"]] = r
    ranked = sorted(best.values(), key=lambda r: -r["gap_open_mm"])
    for r in ranked[:12]:
        print(f"  {r['name']:8s} size {r['size']:.2f}: gap {r['gap_closed_mm']} -> {r['gap_open_mm']} mm (back {r['gap_back_mm']})"
              f"  upper by {r['upper_by']}  lower by {r['lower_by']}  lower drift {r['lower_drift_mm']} mm")  # fmt: skip
    # jitter re-test (8 seeds x 0.01 rad) of the top 3 x size per direction, keep the ones that pass all seeds
    cands = ranked[: max(args.save * 3, 12)]
    jjobs = [{**next(j for j in jobs if j["name"] == r["name"] and j["size"] == r["size"]), "jitter": 0.01, "seed": s} for r in cands for s in range(8)]
    with ProcessPoolExecutor(args.workers) as ex:
        jres = list(ex.map(trial, jjobs, chunksize=2))
    robust = []
    for r in cands:
        mine = [x for x in jres if x["name"] == r["name"] and x["size"] == r["size"]]
        r["jitter_pass"] = sum(x["ok"] for x in mine) / len(mine)
        if r["jitter_pass"] >= 0.875:
            robust.append(r)
    saved = []
    for r in robust[: args.save]:
        job = {**next(j for j in jobs if j["name"] == r["name"] and j["size"] == r["size"]), "keep": True}
        st = trial(job)["open_state"]
        c = load_grip(args.closed)
        name = f"{args.closed}_open{round(r['gap_open_mm'])}mm_{r['name']}"
        grip = {"palm_rotvec": mat_to_rotvec(c["palm_rot"]).tolist(), "joint_targets": dict(zip(SR_JOINT_NAMES, st["targets"])),
                "joints": dict(zip(SR_JOINT_NAMES, st["joints"])), "sticks": st["sticks"], "env": {}, "sim": TRAIN_SIM,
                "made_by": f"chopstix.tools.open_search from {args.closed}: direction {r['name']}, size {r['size']}",
                "pairs_with": args.closed, "open_search": {k: v for k, v in r.items() if k != "open_state"}}  # fmt: skip
        (GRIPS / f"{name}.json").write_text(json.dumps(grip, indent=1))
        saved.append(name)
        print(f"  saved {name}  (jitter pass {r['jitter_pass']:.0%})")
    out = ROOT / "runs" / "griplab" / f"open_search_{args.closed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"closed": args.closed, "joints": args.joints, "trials": res, "ranked": ranked, "robust": robust, "saved": saved}, indent=1))
    print(f"wrote {out}")
    return saved


if __name__ == "__main__":
    main()
