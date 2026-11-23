"""Stage 4 (user, 2026-11-23): the human-style hold, then training (overnight rule: keep altering the grip until a pair
passes 90% on the open/close test in the pick pose, then train; never leave the PC idle).

Hand turned over (back of the hand up, `pick_hand_flip`), sticks 20 deg down, a 25 mm cube. The grip comes from
`chopstix.tools.human_grip` (thumb pad + index + middle on the upper stick; thumb base + index base + ring fingertip
on the lower; sticks along the fingers). Rounds: an opening search on the best closed grips; if nothing passes, a
wider search with a new seed; every 3rd failed round also re-forms the closed grips with new pinch amounts. Then a
32-episode scripted check, then PPO from scratch (the old policies are for a different hand pose) with the steadier
stopping rule (56-episode evals, 60-eval plateau).

    .venv/Scripts/python.exe -m chopstix.tools.train_human --wait-pid <pid>     # usually via the ChopstixHuman task
"""

import argparse
import json
import subprocess
import time

from chopstix.tools.overnight_next import PY, ROOT, alive, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO
from chopstix.tools.train_next import as_args

OUT = ROOT / "runs" / "human_grip"
PITCH = 20


def run(args: list[str]) -> None:
    log("human_grip: " + " ".join(args))
    with (OUT / "rounds_log.txt").open("a", encoding="utf-8") as f:
        subprocess.run([PY, "-m", "chopstix.tools.human_grip", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def result(path) -> dict | None:
    try:
        return json.loads(path.read_text()).get("result")
    except (OSError, ValueError):
        return None


def top_pass(path) -> float:
    try:
        return max((f["pass"] for f in json.loads(path.read_text())["finals"]), default=0.0)
    except (OSError, ValueError, KeyError):
        return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=0, help="the opening search already running")
    p.add_argument("--rounds", type=int, default=12)
    args = p.parse_args()
    keep_awake()
    t0 = time.time()
    if args.wait_pid:
        log(f"train_human: waiting for pid {args.wait_pid} (opening search)")
        while alive(args.wait_pid):
            time.sleep(60)
    res = result(OUT / "opening.json")
    best = (top_pass(OUT / "opening.json"), OUT / "opening.json")
    rnd = 0
    while res is None and rnd < args.rounds:
        rnd += 1
        out = OUT / f"opening_r{rnd}.json"
        if rnd % 2 == 1 and best[0] >= 0.5:  # refine around the best pairs so far
            run(["opening", "--refine", str(best[1]), "--dirs", "80", "--seed", str(rnd), "--workers", "12", "--pitch", str(PITCH), "--out", str(out)])
        else:
            phys = OUT / "physics.json"
            if rnd % 4 == 0:  # new closed grips: other pinch amounts (seeded by the round)
                phys = OUT / f"physics_r{rnd}.json"
                run(["physics", "--tries", "24", "--workers", "10", "--pitch", str(PITCH), "--seed", str(rnd), "--out", str(phys)])
            run(["opening", "--physics", str(phys), "--closed", "16", "--dirs", "60", "--seed", str(rnd), "--workers", "12",
                 "--pitch", str(PITCH), "--out", str(out)])  # fmt: skip
        res = result(out)
        if top_pass(out) > best[0]:
            best = (top_pass(out), out)
        log(f"train_human: round {rnd}: best pass so far {best[0]:.0%} ({best[1].name})")
    if res is None:
        summary(["## Stage 4: human-style grip", "", f"- No open/close pair reached 90% after {rnd} extra rounds; see runs/human_grip"])
        log("train_human: no grip passed; stopping")
        return
    log(f"train_human: grip pair {res['open']} -> {res['closed']} passed {res['pass']:.0%}")
    spec = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["s3_pick25"]["env"]
    env = {**spec, "hold_grip": res["open"], "hold_grip_closed": res["closed"], "pick_hand_flip": True, "pick_stick_pitch": PITCH}
    env.pop("pick_hand_tilt", None)
    label = "Human-style hold: pick and place, 25 mm cube"
    desc = ("Back of the hand up; thumb, index and middle on the upper stick, thumb base, index base and ring finger on the "
            "lower; sticks 20 deg down. Palm comes down 5 cm with the grip open, closes on a 25 mm cube, lifts it 6 cm, "
            "carries it 10 cm to B, lowers it and opens. 10.5 s.")  # fmt: skip
    done = train("s4_human", as_args(env), "Stage 4: human-style hold, 25 mm cube", f"From scratch; grip {res['closed']} (open/close test {res['pass']:.0%}).",
                 label, desc, pretrained=None, extra=PPO + ["--eval-rounds", "4", "--eval-plateau-patience", "60"], total=60e6)  # fmt: skip
    summary(["## Stage 4: human-style hold (back of the hand up, 20 deg), 25 mm cube", "",
             f"- Grip: {res['open']} -> {res['closed']}, open/close test in the pick pose {res['pass']:.0%}, tips open {res['open_mm']:.0f} mm",
             f"- `runs/train/s4_human`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_human: done")


if __name__ == "__main__":
    main()
