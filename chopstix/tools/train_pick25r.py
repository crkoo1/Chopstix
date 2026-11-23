"""Stage 3f (2026-11-22 23:50, overnight rule: keep the PC training): the 25 mm cube with twice the finger range.

s3_pick25 plateaued at 71% best eval (20/32 on the fixed test); 9 of its 12 failures never pinch the cube (the grip
closes to ~27 mm for a 25 mm cube). Finger corrections up to 0.2 rad (was 0.1) let the policy squeeze harder; the
unchanged s3_pick25 policy already does 17/32 with 0.2. From s3_pick25's best; same PPO settings and stopping rules.

    .venv/Scripts/python.exe -m chopstix.tools.train_pick25r            # usually via the ChopstixPick25r task
"""

import json
import time

from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO
from chopstix.tools.train_next import as_args

SETUP = "s3_pick25r"


def main():
    keep_awake()
    t0 = time.time()
    p = ROOT / "site" / "lineup.json"
    spec = json.loads(p.read_text(encoding="utf-8-sig"))
    env = {**spec["setups"]["s3_pick25"]["env"], "hold_residual": 0.2}
    label = "Pick and place, 25 mm cube, 0.2 rad finger range"
    desc = "As the 25 mm pick and place (15° sticks), but the policy may move each finger joint up to 0.2 rad from the grip (was 0.1). 10.5 s."
    spec["setups"][SETUP] = {"label": label, "desc": desc, "env": env, "episodes": 32, "clip_seeds": [0]}
    for e in spec["entries"]:
        if e["id"] in ("scripted", "s3_pick25") and SETUP not in e["setups"]:
            e["setups"].append(SETUP)
    p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    done = train("s3_pick25r", as_args(env), "Stage 3f: 25 mm cube, wider finger range", "From s3_pick25's best; finger corrections up to 0.2 rad.",
                 label, desc, pretrained=str(ROOT / "runs/train/s3_pick25/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    summary(["## Stage 3f: 25 mm cube, finger range 0.2 rad (started overnight by the keep-training rule)", "",
             "- s3_pick25 (0.1 rad) plateaued: best eval 71%, 20/32 fixed test; unchanged policy at 0.2 rad: 17/32",
             f"- `runs/train/s3_pick25r`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_pick25r: done")


if __name__ == "__main__":
    main()
