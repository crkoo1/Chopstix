"""Stage 4b (user, 2026-11-23): get the human-style grip working in the pose where its tips close SIDEWAYS
(`pick_level_open`), before any training. No training here: the user checks the grip first.

Why: with the back of the hand up, hg_1's tips open and close up/down (the upper tip ends 8 cm above the cube), so
s4_human never pinched (0/32, stopped at 0%). Turning the hand so the opening is level picks 4/16 untrained, but hg_1
only passes 17/20 open/close runs in that pose.

Rounds (all candidates tested in the sideways-closing pose of the current best pair):
    odd rounds: refine the best opening motions so far (small changes to the open targets)
    even rounds: re-form the closed grips with gravity in this pose (new pinch amounts), then a random opening search
Every pair a round saves (>= 90% of 20 open/close runs) is verified in ITS OWN env pose: 40 open/close runs and a
32-episode untrained pick. Accepted: >= 90% open/close and the untrained grip lifting the cube in >= 40% of picks.
Keeps improving for at least --min-rounds rounds, then stops at the first accepted pair (or after --rounds).

    .venv/Scripts/python.exe -m chopstix.tools.grip_level        # usually via the ChopstixGrip task
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chopstix.tools.overnight_next import PY, ROOT, keep_awake, log, summary

OUT = ROOT / "runs" / "human_grip" / "level"
PITCH = 20


def env_for(open_grip: str, closed_grip: str) -> dict:
    spec = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["s3_pick25"]["env"]
    return {**spec, "hold_grip": open_grip, "hold_grip_closed": closed_grip, "pick_hand_flip": True, "pick_stick_pitch": PITCH,
            "pick_level_open": True}  # fmt: skip


def verify(open_grip: str, closed_grip: str, workers: int = 10) -> dict:
    from chopstix.envs.chopsticks_env import make_env
    from chopstix.sim import load_grip
    from chopstix.tools.human_grip import cycle_job
    from chopstix.tools.report import rollout, summarize_pick

    env = env_for(open_grip, closed_grip)
    e = make_env(env)
    rot = e._pick_palm_rot.tolist()
    ot = load_grip(open_grip)["targets"].tolist()
    jobs = [{"key": "v", "closed": closed_grip, "open_targets": ot, "palm_rot": rot, "seed": 200 + s, "min_open": 0.045,
             "back_tol": 0.02, "jitter": 0.01, "push": 0.03, "cycles": 3} for s in range(40)]  # fmt: skip
    with ProcessPoolExecutor(workers) as ex:
        cyc = list(ex.map(cycle_job, jobs))
        eps = list(ex.map(rollout, [{"env": env, "actor": None, "seed": s, "record": False} for s in range(32)]))
    summarize_pick(eps, e.cfg.steps_per_episode, e.cfg)
    out = {}
    for x in eps:
        out[x["outcome"]] = out.get(x["outcome"], 0) + 1
    lifted = sum(out.get(k, 0) for k in ("success", "dropped", "not_placed"))
    r = {"open": open_grip, "closed": closed_grip, "cycle_pass": sum(c["ok"] for c in cyc) / len(cyc),
         "open_mm": float(np.median([c["open_mm"] for c in cyc])), "pick": out, "lifted": lifted / 32}  # fmt: skip
    r["accepted"] = r["cycle_pass"] >= 0.9 and r["lifted"] >= 0.4
    r["score"] = r["cycle_pass"] + r["lifted"] + 0.5 * out.get("success", 0) / 32
    return r


def hg(args: list[str]) -> None:
    log("grip_level: human_grip " + " ".join(args))
    with (OUT / "rounds_log.txt").open("a", encoding="utf-8") as f:
        subprocess.run([PY, "-m", "chopstix.tools.human_grip", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def top(path) -> float:
    try:
        return max((f["pass"] for f in json.loads(path.read_text())["finals"]), default=0.0)
    except (OSError, ValueError, KeyError):
        return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--min-rounds", type=int, default=4)
    p.add_argument("--start", default="hg_1_open74mm,hg_1", help="the pair whose sideways pose the search starts in")
    p.add_argument("--start-refine", default=str(ROOT / "runs/human_grip/opening_r1.json"))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keep_awake()
    t0 = time.time()
    pose = args.start
    base = verify(*args.start.split(","))
    log(f"grip_level: start pair {args.start}: {base}")
    verified = [base]
    best_file = (0.0, args.start_refine)
    phys = ROOT / "runs/human_grip/physics.json"
    for rnd in range(1, args.rounds + 1):
        out = OUT / f"opening_r{rnd}.json"
        common = ["--rot-pair", pose, "--level-open", "--pitch", str(PITCH), "--workers", "12", "--seed", str(100 + rnd), "--out", str(out)]
        if rnd % 2 == 1:
            hg(["opening", "--refine", best_file[1], "--dirs", "80", *common])
        else:
            phys = OUT / f"physics_r{rnd}.json"
            hg(["physics", "--tries", "24", "--workers", "10", "--rot-pair", pose, "--level-open", "--pitch", str(PITCH),
                "--seed", str(100 + rnd), "--out", str(phys)])  # fmt: skip
            hg(["opening", "--physics", str(phys), "--closed", "16", "--dirs", "60", *common])
        if top(out) >= best_file[0]:
            best_file = (top(out), str(out))
        try:
            res = json.loads(out.read_text()).get("result")
        except (OSError, ValueError):
            res = None
        if res:
            v = verify(res["open"], res["closed"])
            verified.append(v)
            log(f"grip_level: round {rnd}: {res['open']} -> {res['closed']}: open/close {v['cycle_pass']:.0%} of 40 in its own pose, "
                f"untrained pick {v['pick']} (lifted {v['lifted']:.0%})")  # fmt: skip
            if v["accepted"] and v["score"] >= max(x["score"] for x in verified):
                pose = f"{res['open']},{res['closed']}"  # search on in the best accepted pair's pose
        else:
            log(f"grip_level: round {rnd}: no pair passed 90% of 20 (best {top(out):.0%})")
        best = max(verified, key=lambda x: (x["accepted"], x["score"]))
        (OUT / "best.json").write_text(json.dumps({"best": best, "verified": verified}, indent=1))
        if best["accepted"] and rnd >= args.min_rounds:
            break
    best = max(verified, key=lambda x: (x["accepted"], x["score"]))
    summary(["## Stage 4b: human-style grip, sticks closing sideways (no training yet: user checks the grip first)", "",
             f"- Best pair: {best['open']} -> {best['closed']}: open/close {best['cycle_pass']:.0%} of 40 in its pick pose, "
             f"tips open {best['open_mm']:.0f} mm; untrained pick {best['pick']} (lifted {best['lifted']:.0%}); accepted: {best['accepted']}",
             f"- {len(verified) - 1} pairs verified; {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log(f"grip_level: done; best {best}")


if __name__ == "__main__":
    main()
