"""Stage 3e (user, 2026-11-22 21:50): pick and place a 25 mm cube with the working grip pair, sticks at 15 deg.

Replaces the 20 mm plan: no tighter grip passed in the pick pose (tight_grip v4: 22 rounds, best closed back to ~22 mm,
need <= 14 mm), while the current pair (closed_v5_mj3_open68mm_g_makis240 -> closed_v5_mj3_sq35_makis240) already
places a 25 mm cube 3/32 untrained (runs/pick_test2). From s3_pick15's best; waits for --wait-pid (s3_pick_hard) so
only one training runs at a time. Same PPO settings and stopping rules as the other stage 3 runs.

    .venv/Scripts/python.exe -m chopstix.tools.train_pick25 --wait-pid <pid>     # usually via the ChopstixPick25 task
"""

import argparse
import json
import time

from chopstix.tools.overnight_next import ROOT, alive, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO
from chopstix.tools.train_next import DESC, as_args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=0)
    args = p.parse_args()
    keep_awake()
    if args.wait_pid:
        log(f"train_pick25: waiting for pid {args.wait_pid} (s3_pick_hard)")
        while alive(args.wait_pid):
            time.sleep(60)
    t0 = time.time()
    spec = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))
    env = {**spec["setups"]["s3_pick15"]["env"], "ball_radius": 0.0125}
    done = train("s3_pick25", as_args(env), "Stage 3e: pick and place, 25 mm cube", "From s3_pick15's best; 25 mm cube, sticks 15 deg.",
                 "Pick and place, 15° sticks, 25 mm cube", DESC.format(cube=25), pretrained=str(ROOT / "runs/train/s3_pick15/actor_best.pt"),
                 extra=PPO, total=30e6)  # fmt: skip
    summary(["## Stage 3e: 25 mm cube (instead of 20 mm)", "",
             "- The tighter-grip search for a 20 mm cube found nothing that holds in the pick pose, so the working pair stays",
             "- Before training: s3_pick 3/32 on a 25 mm cube at 15 deg",
             f"- `runs/train/s3_pick25`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_pick25: done")


if __name__ == "__main__":
    main()
