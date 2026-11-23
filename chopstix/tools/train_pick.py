"""Stage 3 (2026-11-22): pick and place with the open/close grip, continued from the best stage 2 policy.

The palm follows the scripted path (down 5 cm with the grip open, close on a 30 mm cube, lift 6 cm, carry 10 cm to B,
lower, open, back off); the policy moves only the fingers, as in stage 2, whose open/close skill is exactly what this
needs. Before any pick training (runs/pick_test.json): grip alone 4/32, s2_openclose_v2 14/32.

Same PPO settings as train_chain (lr 1e-4, desired_kl 0.02, entropy 0.005, batch 1792, 3 epochs, critic alone for the
first 20 updates). Finger corrections up to 0.1 rad, as in stage 2. Stops at 90% on 3 evals in a row, at a plateau, or
at 30M steps. Log: runs/overnight/next_log.txt.

    .venv/Scripts/python.exe -m chopstix.tools.train_pick                  # usually via the ChopstixPick scheduled task
"""

import json
import time

from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO


def main():
    keep_awake()
    t0 = time.time()
    setup = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["pick_oc"]
    env = [f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in setup["env"].items()]
    done = train("s3_pick", env, "Stage 3: pick and place", "From s2_openclose_v2's best; open/close grip on the pick-and-place path.",
                 setup["label"], setup["desc"], pretrained=str(ROOT / "runs/train/s2_openclose_v2/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    summary(["## Stage 3: pick and place with the open/close grip", "",
             "- Before training: grip alone 4/32, stage 2 policy (`s2_openclose_v2`) 14/32 on the fixed test",
             f"- `runs/train/s3_pick`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_pick: done")


if __name__ == "__main__":
    main()
