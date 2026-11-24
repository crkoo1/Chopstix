"""Stage 3h (user, 2026-11-23): improve the 25 mm pick and place in the previous hand pose (backup while the human-style
grip is searched) and keep the sticks steady at the drop.

s3_pick25b places 26/32, but at the release the upper stick can swing up to 65 mm from where the open grip holds it,
and in 2 of 16 episodes the sticks end 23 / 60 mm out of place. New: success also needs both stick tips within 15 mm
of the grip's pose at the end (`pick_end_tip_err`; s3_pick25b: 25/32 under it), and during release and retreat a
reward for the tips staying where the opening grip holds them (`w_pick_settle`). From s3_pick25b's best, the same PPO
settings and the steadier stopping rule; 8 envs (not 14) so it fits in memory beside the grip search.

    .venv/Scripts/python.exe -m chopstix.tools.train_pick25s            # usually via the ChopstixSteady task
"""

import json
import time

from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO
from chopstix.tools.train_next import as_args


def main():
    keep_awake()
    t0 = time.time()
    s = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["s3_pick25"]
    env = {**s["env"], "pick_end_tip_err": 0.015, "w_pick_settle": 2.0}
    label = "Pick and place, 25 mm cube, sticks steady at the drop"
    desc = (s["desc"] + " Success also needs both stick tips back within 15 mm of where the open grip holds them at the end.")
    done = train("s3_pick25s", as_args(env), "Stage 3h: 25 mm cube, steady sticks at the drop",
                 "From s3_pick25b's best; tips must end within 15 mm of the grip's pose; reward for steady sticks while letting go.",
                 label, desc, pretrained=str(ROOT / "runs/train/s3_pick25b/actor_best.pt"),
                 extra=PPO + ["--eval-rounds", "4", "--eval-plateau-patience", "60", "--num-envs", "8"], total=30e6)  # fmt: skip
    summary(["## Stage 3h: 25 mm cube, steady sticks at the drop (previous hand pose)", "",
             "- s3_pick25b: 26/32; 25/32 when the stick tips must end within 15 mm of the grip's pose",
             f"- `runs/train/s3_pick25s`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_pick25s: done")


if __name__ == "__main__":
    main()
