"""Stage 3g (2026-11-23 01:40, overnight keep-training rule): continue s3_pick25 (25 mm cube, 15 deg, 0.1 rad) with
steadier stopping. Evals use 4 rounds (56 episodes, was 28) and the plateau rule waits 60 evals (was 30): s3_pick25's
28-episode evals swung 36-71%, so the plateau rule may have stopped it on noise. The 0.2 rad variant (s3_pick25r) did
worse (best 57%: fewer missed pinches, more drops). Same env and PPO settings; same page tab as s3_pick25.

    .venv/Scripts/python.exe -m chopstix.tools.train_pick25b            # usually via the ChopstixPick25b task
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
    done = train("s3_pick25b", as_args(s["env"]), "Stage 3g: 25 mm cube, continued", "From s3_pick25's best; 56-episode evals, 60-eval plateau.",
                 s["label"], s["desc"], pretrained=str(ROOT / "runs/train/s3_pick25/actor_best.pt"),
                 extra=PPO + ["--eval-rounds", "4", "--eval-plateau-patience", "60"], total=30e6)  # fmt: skip
    summary(["## Stage 3g: 25 mm cube continued, steadier stopping (overnight keep-training rule)", "",
             "- s3_pick25: 20/32, best eval 71%; s3_pick25r (0.2 rad fingers): 14/32, best eval 57%",
             f"- `runs/train/s3_pick25b`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_pick25b: done")


if __name__ == "__main__":
    main()
