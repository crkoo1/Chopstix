"""Tighter versions of the human-style grip hg_1 (user, 2026-11-23).

Why: stage 1 (hold under noise, sideways pick pose) sat at 80% for an hour; the failing episodes are the strongest
wobbles (4-5 deg), where both sticks slide 10-20 mm together through the hand, and the trained policy failed the same
episodes as the bare grip. hg_1's finger targets sit within 0.007 rad of where the fingers rest, so it barely squeezes.

For each scale s, hg_1 is formed again from its design (human_grip design 14) with every finger's pinch x s
(the pinch is how far each finger is pressed into its stick while forming), then given hg_1's opening motion
(open targets = new closed targets + (hg_1_open74mm - hg_1) targets, settled in the pick pose). Each pair is checked
untrained in the sideways pick pose: stage 1 (hold under noise) and stage 2 (open/close on command) on 56 episodes,
plus grip_level's verify (40 open/close cycle runs, 32-episode pick). Scale 1 is the control (should match hg_1).

    .venv/Scripts/python.exe -m chopstix.tools.tighten_hg [--scales 1 1.5 2 3]
Result: runs/human_grip/tighten.json
"""

import argparse
import ast
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chopstix.tools.grip_level import env_for, verify
from chopstix.tools import human_grip
from chopstix.tools.human_grip import GRIPS, open_state, physics_job
from chopstix.tools.overnight_next import ROOT, log

OUT = ROOT / "runs" / "human_grip" / "tighten.json"
BASE_OPEN, BASE_CLOSED, DESIGN = "hg_1_open74mm", "hg_1", 14


def stage_envs(open_grip: str, closed_grip: str) -> dict:
    from chopstix.envs.chopsticks_env import make_env

    rot = make_env(env_for(open_grip, closed_grip))._pick_palm_rot.flatten().tolist()
    return {"s1": {"stage": "hold", "hold_grip": closed_grip, "hold_joint_noise": 0.02, "hold_push": 0.1, "hold_wobble": 5,
                   "hold_shake": 10, "hold_palm_rot": rot},
            "s2": {"stage": "hold", "hold_grip": open_grip, "hold_grip_closed": closed_grip, "hold_joint_noise": 0.02,
                   "hold_push": 0.02, "hold_palm_rot": rot}}  # fmt: skip


def episode(job) -> bool:
    from chopstix.envs.chopsticks_env import make_env

    env, seed = job
    e = make_env(env)
    e.reset(seed=seed)
    z = np.zeros(e.action_space.shape)
    while True:
        _, _, term, trunc, info = e.step(z)
        if term or trunc:
            return bool(info["is_success"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--workers", type=int, default=14)
    args = p.parse_args()
    from chopstix.sim import load_grip

    design = {r["seed"]: r for r in json.loads((ROOT / "runs/human_grip/design/designs.json").read_text())}[DESIGN]
    made = json.loads((GRIPS / f"{BASE_CLOSED}.json").read_text())["made_by"]
    pinch0 = ast.literal_eval(made.split("pinch ", 1)[1].rstrip(")"))  # the dict literal hg_1 was formed with
    d_open = load_grip(BASE_OPEN)["targets"] - load_grip(BASE_CLOSED)["targets"]
    from chopstix.envs.chopsticks_env import make_env

    rot = make_env(env_for(BASE_OPEN, BASE_CLOSED))._pick_palm_rot
    human_grip.FIXED_ROT = rot.tolist()  # the forming hold check then runs in the sideways pick pose
    results = []
    pairs = [(BASE_OPEN, BASE_CLOSED, 0.0)]
    for s in args.scales:
        tag = f"hg_1t{round(100 * s)}"
        pinch = {k: round(min(0.3, v * s), 4) for k, v in pinch0.items()}
        r = physics_job({"design": DESIGN, "k": f"t{round(100 * s)}", "x": design["x"], "pinch": pinch, "seed": 0})
        shutil.copy(GRIPS / f"{r['name']}.json", GRIPS / f"{tag}.json")
        ct = load_grip(tag)["targets"]
        o = open_state(tag, ct + d_open, rot)
        (GRIPS / f"{tag}_open.json").write_text(json.dumps(o, indent=1))
        log(f"tighten_hg: {tag} (pinch x{s}: {pinch}): form hold check {r['ok']}/3, drift {r['drift_mm']} mm")
        pairs.append((f"{tag}_open", tag, s))
    with ProcessPoolExecutor(args.workers) as ex:
        for op, cl, s in pairs:
            envs = stage_envs(op, cl)
            s1 = sum(ex.map(episode, [(envs["s1"], k) for k in range(56)]))
            s2 = sum(ex.map(episode, [(envs["s2"], k) for k in range(56)]))
            v = verify(op, cl, workers=args.workers)
            res = {"open": op, "closed": cl, "scale": s, "stage1": s1, "stage2": s2, "cycle_pass": v["cycle_pass"],
                   "open_mm": v["open_mm"], "pick": v["pick"], "lifted": v["lifted"]}  # fmt: skip
            results.append(res)
            log(f"tighten_hg: {cl} (x{s or 'hg_1'}): stage 1 {s1}/56, stage 2 {s2}/56, open/close {v['cycle_pass']:.0%} of 40 "
                f"(tips open {v['open_mm']:.0f} mm), untrained pick {v['pick']}")  # fmt: skip
            OUT.write_text(json.dumps(results, indent=1))
    log("tighten_hg: done")


if __name__ == "__main__":
    main()
