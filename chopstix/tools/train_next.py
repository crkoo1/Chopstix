"""After s3_pick (user, 2026-11-22): flatter sticks now, a smaller cube once its grip passes the gate.

    1. wait for --wait-pid (the running train_pick job)
    2. s3_pick15: pick and place with the sticks 15 deg below horizontal (was ~31 deg; 20 deg puts the fingers
       through the table), 30 mm cube, from s3_pick's best policy (which already places 20/32 at 15 deg untrained)
    3. wait for chopstix.tools.tight_grip (runs/tight_grip/RESULT.json), then s3_pick20: 15 deg sticks, 20 mm cube,
       its tighter open/closed pair, from s3_pick15's best
Same PPO settings and stopping rules as train_chain / train_pick. Log: runs/overnight/next_log.txt.

    .venv/Scripts/python.exe -m chopstix.tools.train_next --wait-pid <pid>     # usually via the ChopstixNext task
"""

import argparse
import json
import time

from chopstix.tools.overnight_next import ROOT, alive, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO

DESC = ("Palm comes down 5 cm with the grip open, sticks 15 deg below horizontal, closes on a {cube} mm cube, lifts it 6 cm, "
        "carries it 10 cm to B, lowers it and opens to let go. 10.5 s.")  # fmt: skip


def as_args(env: dict) -> list[str]:
    return [f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in env.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=0)
    p.add_argument("--wait-pid-20", type=int, default=0, help="also wait for this pid before the 20 mm run (one training at a time)")
    args = p.parse_args()
    keep_awake()
    if args.wait_pid:
        log(f"train_next: waiting for pid {args.wait_pid} (s3_pick)")
        while alive(args.wait_pid):
            time.sleep(60)
    t0 = time.time()
    base = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["pick_oc"]["env"]
    env15 = {**base, "pick_stick_pitch": 15}
    done15 = ROOT / "runs/train/s3_pick15/DONE"
    d15 = done15.read_text().strip() if done15.exists() else train("s3_pick15", as_args(env15), "Stage 3b: pick and place, flatter sticks", "From s3_pick's best; sticks 15 deg below horizontal.",
                "Pick and place, 15° sticks", DESC.format(cube=30), pretrained=str(ROOT / "runs/train/s3_pick/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    lines = [f"- Sticks 15 deg, 30 mm cube (`runs/train/s3_pick15`): {d15 or 'reached total steps'}"]
    res = ROOT / "runs" / "tight_grip" / "RESULT.json"
    if not res.exists():
        log("train_next: waiting for the tight grip search")
    while not res.exists():
        time.sleep(120)
    r = json.loads(res.read_text())
    if args.wait_pid_20 and r.get("open"):
        log(f"train_next: grip found; waiting for pid {args.wait_pid_20} before s3_pick20 (one training at a time)")
        while alive(args.wait_pid_20):
            time.sleep(60)
    if r.get("open"):
        env20 = {**env15, "ball_radius": 0.010, "hold_grip": r["open"], "hold_grip_closed": r["closed"]}
        d20 = train("s3_pick20", as_args(env20), "Stage 3c: pick and place, 20 mm cube", f"From s3_pick15's best; grips {r['open']} -> {r['closed']}.",
                    "Pick and place, 15° sticks, 20 mm cube", DESC.format(cube=20), pretrained=str(ROOT / "runs/train/s3_pick15/actor_best.pt"),
                    extra=PPO, total=30e6)  # fmt: skip
        lines.append(f"- Sticks 15 deg, 20 mm cube (`runs/train/s3_pick20`, grips {r['open']} -> {r['closed']}): {d20 or 'reached total steps'}")
    else:
        lines.append("- 20 mm cube: no tighter grip passed the gate, so no run")
    summary(["## Stage 3b/3c: flatter sticks, then a 20 mm cube", "", *lines,
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_next: done")


if __name__ == "__main__":
    main()
