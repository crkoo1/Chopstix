"""Open <-> closed cycle test of a grip pair: the grip gate before training.

Starts in the closed grip, under gravity, then cycles closed -> open -> closed (each move over 1 s, then 1 s still)
--cycles times, with random 0.03 N pushes on each stick every 0.1 s (as in the hold stage) and --jitter rad of finger
target noise every 0.05 s. A run passes when, at the end of every still phase:

    - the lower stick is within 1 cm / 10 deg (axis tilt) of where it started, in the palm frame
    - the upper stick is touched by the thumb and the index or middle finger
    - open: the tips are at least --min-open apart; closed again: within --back-tol of the closed gap
and no stick ever moves more than 5 cm from where it started (dropped).

    .venv/Scripts/python.exe -m chopstix.tools.cycle_test --open closed_v5_mj4_open57mm_makis
Results: runs/griplab/cycle_<open>.json ("pass" = share of runs that passed).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chopstix.sim import STICKS, load_grip
from chopstix.tools.grip_lab import PALM_POS, TRAIN_SIM, _sim, tilt_deg
from chopstix.tools.open_search import tip_gap

ROOT = Path(__file__).resolve().parents[2]
GRIPS = ROOT / "chopstix" / "grips"


def pair_of(open_grip: str) -> str:
    return json.loads((GRIPS / f"{open_grip}.json").read_text())["pairs_with"]


def run(job: dict) -> dict:
    sim = _sim(job.get("sim", TRAIN_SIM))
    c = load_grip(job["closed"])
    if "closed_targets" in job:  # a squeezed closed grip being tried (start and closing both use these targets)
        c["targets"] = np.array(job["closed_targets"])
    o = {"targets": np.array(job["open_targets"])} if "open_targets" in job else load_grip(job["open"])
    dt = sim.model.opt.timestep
    n_ctrl = max(1, round(0.05 / dt))
    rng = np.random.default_rng(1000 + job["seed"])
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
    push, jit = job.get("push", 0.03), job.get("jitter", 0.01)
    state = {"cur": c["targets"].copy(), "dropped": False, "k": 0, "forces": {}}

    def move(target, seconds):
        steps = round(seconds / 0.05)
        for k in range(steps):
            frac = min(1.0, (k + 1) / round(job.get("ramp", 1.0) / 0.05))
            tgt = state["cur"] + frac * (target - state["cur"])
            sim.set_targets(np.clip(tgt + rng.normal(0, jit, 22), *sim.joint_limits.T))
            if state["k"] % 2 == 0:
                state["forces"] = {n: rng.uniform(-push, push, 3) for n in STICKS}
            state["k"] += 1
            sim.step(n_ctrl, forces=state["forces"])
            for n in STICKS:
                if np.linalg.norm(sim.stick_pose(n, "palm")[0] - start[n][0]) > 0.05:
                    state["dropped"] = True
        state["cur"] = target

    def check(kind):
        ff = sim.finger_forces()
        upper_by = [f for f in ("th", "ff", "mf") if ff[f]["upper"] > 0.01]
        lo = sim.stick_pose("lower", "palm")
        gap = tip_gap(sim)
        ok = ("th" in upper_by and len(upper_by) >= 2 and np.linalg.norm(lo[0] - start["lower"][0]) < 0.01
              and tilt_deg(start["lower"][1], lo[1]) < 10)  # fmt: skip
        ok = ok and (gap >= job["min_open"] if kind == "open" else gap - gap0 < job["back_tol"])
        return {"kind": kind, "gap_mm": round(1000 * gap, 1), "upper_by": upper_by, "ok": bool(ok)}

    checks = []
    for _ in range(job["cycles"]):
        move(o["targets"], job.get("ramp", 1.0) + 1.0)
        checks.append(check("open"))
        move(c["targets"], job.get("ramp", 1.0) + 1.0)
        checks.append(check("closed"))
    ok = all(x["ok"] for x in checks) and not state["dropped"]
    return {"seed": job["seed"], "ok": bool(ok), "dropped": state["dropped"], "gap_closed_mm": round(1000 * gap0, 1), "checks": checks}


def cycle_test(open_grip: str, runs: int = 20, cycles: int = 3, min_open: float = 0.065, back_tol: float = 0.02,
               jitter: float = 0.01, push: float = 0.03, workers: int = 14) -> dict:  # fmt: skip
    closed = pair_of(open_grip)
    jobs = [{"open": open_grip, "closed": closed, "seed": s, "cycles": cycles, "min_open": min_open, "back_tol": back_tol,
             "jitter": jitter, "push": push} for s in range(runs)]  # fmt: skip
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(run, jobs))
    opens = [x["gap_mm"] for r in res for x in r["checks"] if x["kind"] == "open"]
    backs = [x["gap_mm"] for r in res for x in r["checks"] if x["kind"] == "closed"]
    out = {"open": open_grip, "closed": closed, "runs": runs, "cycles": cycles, "min_open_mm": 1000 * min_open,
           "back_tol_mm": 1000 * back_tol, "jitter": jitter, "push": push, "pass": sum(r["ok"] for r in res) / runs,
           "dropped": sum(r["dropped"] for r in res), "open_gap_mm_median": float(np.median(opens)),
           "closed_gap_mm_median": float(np.median(backs)), "results": res}  # fmt: skip
    f = ROOT / "runs" / "griplab" / f"cycle_{open_grip}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--open", nargs="+", required=True, help="open grip name(s); each pairs with its closed grip")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--min-open", type=float, default=0.065, help="m: tip gap the open grip must reach (60 mm cube)")
    p.add_argument("--back-tol", type=float, default=0.02)
    p.add_argument("--jitter", type=float, default=0.01)
    p.add_argument("--push", type=float, default=0.03)
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args(argv)
    for g in args.open:
        r = cycle_test(g, args.runs, args.cycles, args.min_open, args.back_tol, args.jitter, args.push, args.workers)
        print(f"{g}: pass {r['pass']:.0%} ({r['runs']} runs x {r['cycles']} cycles), dropped {r['dropped']}, "
              f"tips open {r['open_gap_mm_median']} mm, closed again {r['closed_gap_mm_median']} mm (closed {r['results'][0]['gap_closed_mm']})", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
