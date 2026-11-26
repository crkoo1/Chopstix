"""Step 2 check: does a saved tripod grip hold both chopsticks in MuJoCo? (port of `chops.stage1.grip test`)

Per seed: gravity off, hand + sticks placed as the grip file says (palm 30 cm above the table), settle 0.2 s, then
gravity on and random pushes on each stick (uniform, up to --push N per axis, new every --push-every s) for
--duration s with the finger targets fixed at the grip's. The grip HOLDS if both sticks stay within --max-drift m and
--max-turn deg of where they settled (palm frame) and at least 3 of the thumb/index/middle/ring fingers still touch a
stick. Same criteria as Makis' SuperDex stage-1 test.

    .venv/Scripts/python.exe -m chopstix.tools.hold_test --grip closed_v5
    .venv/Scripts/python.exe -m chopstix.tools.hold_test --grip closed_v5 --set impratio=100 condim=6 --seeds 10
    .venv/Scripts/python.exe -m chopstix.tools.hold_test --grip closed_v5 --view        # watch seed 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chopstix.geometry import rot_angle_deg
from chopstix.sim import STICKS, Sim, SimCfg, load_grip
from chopstix.tools.common import parse_overrides

PALM_POS = np.array([0.0, 0.0, 0.3])
GRIP_FINGERS = ("th", "ff", "mf", "rf")


def load_into(sim: Sim, grip: dict, joint_noise: float = 0.0, rng=None):
    sim.reset()
    sim.set_palm(PALM_POS, grip["palm_rot"])
    q = grip["joints"] + (rng.uniform(-joint_noise, joint_noise, len(grip["joints"])) if joint_noise else 0.0)
    sim.set_joints(q)
    sim.set_targets(grip["targets"])
    for n in STICKS:
        sim.place_stick(n, *grip["sticks"][n])
    sim.forward()


def initial_report(sim: Sim, grip: dict) -> dict:
    """Overlap between sticks and hand right after placing (before any physics), and where things go in 0.2 s."""
    load_into(sim, grip)
    pen = {f"{a}-{b}": round(v * 1000, 2) for (a, b), v in sorted(sim.penetration().items()) if "upper" in (a, b) or "lower" in (a, b)}
    sim.set_gravity(False)
    sim.step(round(0.2 / sim.model.opt.timestep))
    moved = {}
    for n in STICKS:
        pos, rot = sim.stick_pose(n, "palm")
        p0, r0 = grip["sticks"][n]
        moved[n] = {"mm": round(float(np.linalg.norm(pos - p0)) * 1000, 2), "deg": round(rot_angle_deg(r0.T @ rot), 2)}
    forces = {f: {s: round(v, 3) for s, v in by.items()} for f, by in sim.finger_forces().items()}
    joint_err = np.abs(sim.joints() - grip["joints"])
    return {"overlap_mm_at_load": pen, "settle_0.2s_no_gravity": moved, "finger_forces_N": forces,
            "joint_drift_rad_max": round(float(joint_err.max()), 4)}  # fmt: skip


def run_seed(sim: Sim, grip: dict, seed: int, args, viewer=None) -> dict:
    rng = np.random.default_rng(seed)
    dt = sim.model.opt.timestep
    load_into(sim, grip, args.joint_noise, rng)
    sim.set_gravity(False)
    sim.step(round(0.2 / dt))
    start = {n: sim.stick_pose(n, "palm") for n in STICKS}
    sim.set_gravity(True)
    every = max(1, round(args.push_every / dt))
    steps = round(args.duration / dt)
    first_fail = None
    for i in range(0, steps, every):
        forces = {n: rng.uniform(-args.push, args.push, 3) for n in STICKS}
        sim.step(min(every, steps - i), forces)
        if viewer is not None:
            viewer.sync()
            time.sleep(every * dt)
        if first_fail is None:
            for n in STICKS:
                pos, rot = sim.stick_pose(n, "palm")
                if np.linalg.norm(pos - start[n][0]) > args.max_drift or rot_angle_deg(start[n][1].T @ rot) > args.max_turn:
                    first_fail = round((i + every) * dt, 2)
                    break
    drift, turn = {}, {}
    for n in STICKS:
        pos, rot = sim.stick_pose(n, "palm")
        drift[n] = float(np.linalg.norm(pos - start[n][0]))
        turn[n] = rot_angle_deg(start[n][1].T @ rot)
    ff = sim.finger_forces()
    touching = [f for f in GRIP_FINGERS if max(ff[f].values()) > 0.01]
    holds = all(d < args.max_drift for d in drift.values()) and all(t < args.max_turn for t in turn.values())
    holds = holds and len(touching) >= 3
    return {"seed": seed, "holds": bool(holds), "drift_m": drift, "turn_deg": turn, "touching": touching,
            "first_fail_s": first_fail, "forces": {f: ff[f] for f in GRIP_FINGERS}}  # fmt: skip


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grip", default="closed_v5", help="name in chopstix/grips/ or a path")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--push", type=float, default=0.03, help="N, max per axis, on each stick")
    p.add_argument("--push-every", type=float, default=0.1)
    p.add_argument("--max-drift", type=float, default=0.01)
    p.add_argument("--max-turn", type=float, default=10.0)
    p.add_argument("--joint-noise", type=float, default=0.0, help="rad, uniform, on the starting joints")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="SimCfg overrides")
    p.add_argument("--out", type=Path, help="write the results as JSON")
    p.add_argument("--view", action="store_true", help="watch seed 0 in the MuJoCo viewer")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    overrides = parse_overrides(args.set)
    sim = Sim(SimCfg.from_overrides(overrides))
    grip = load_grip(args.grip)
    report = initial_report(sim, grip)
    if not args.quiet:
        print(f"grip {grip['name']}  sim {overrides or 'defaults'}")
        print(json.dumps(report, indent=1))

    viewer = None
    if args.view:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    t0 = time.perf_counter()
    runs = []
    for seed in range(args.seeds):
        r = run_seed(sim, grip, seed, args, viewer if seed == 0 else None)
        runs.append(r)
        if not args.quiet:
            print(
                f"seed {seed:3d}  {'HOLDS' if r['holds'] else 'drops'}  drift upper {r['drift_m']['upper'] * 100:5.2f} cm"
                f"  lower {r['drift_m']['lower'] * 100:5.2f} cm  turn {r['turn_deg']['upper']:5.1f}/{r['turn_deg']['lower']:5.1f} deg"
                f"  touching {','.join(r['touching']) or '-'}  first fail {r['first_fail_s']}"
            )  # fmt: skip
    wall = time.perf_counter() - t0
    rate = float(np.mean([r["holds"] for r in runs]))
    sim_s = args.seeds * (args.duration + 0.2)
    print(f"{grip['name']} {overrides or ''}: holds in {rate:.0%} of {args.seeds} runs ({args.duration:.0f} s, pushes "
          f"{args.push} N)  [{sim_s / wall:.0f}x real time]")  # fmt: skip
    out = {"grip": grip["name"], "sim": overrides, "settings": {k: v for k, v in vars(args).items() if k not in ("out", "set")},
           "hold_rate": rate, "initial": report, "runs": runs, "realtime_factor": sim_s / wall}  # fmt: skip
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
