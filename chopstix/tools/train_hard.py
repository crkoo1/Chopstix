"""Harder pick (user, 2026-11-22 18:40): the 15 deg pick-and-place with more variation, from s3_pick15's best policy.

Cube start position +-1.5 cm (was +-0.5 cm), yaw +-0.3 rad / 17 deg (was +-0.1 rad), and random 0.02 N pushes on each
stick every 0.1 s during the whole pick (new env option pick_push). Before training, s3_pick15 places 21/32 here
(29/32 on its own task): mostly the cube gets knocked when it starts off centre; the pushes barely matter.
Same PPO settings and stopping rules as train_chain / train_pick. Runs beside chopstix.tools.tight_grip (6 workers).

    .venv/Scripts/python.exe -m chopstix.tools.train_hard              # usually via the ChopstixHard scheduled task
"""

import json
import time

from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_chain import PPO
from chopstix.tools.train_next import as_args

SETUP = "s3_pick_hard"


def main():
    keep_awake()
    t0 = time.time()
    p = ROOT / "site" / "lineup.json"
    spec = json.loads(p.read_text(encoding="utf-8-sig"))
    env = {**spec["setups"]["s3_pick15"]["env"], "pick_object_noise": 0.015, "pick_object_yaw_noise": 0.3, "pick_push": 0.02}
    label = "Harder pick: 15° sticks, cube ±1.5 cm / ±17°, 0.02 N pushes"
    desc = ("As the 15° pick and place (30 mm cube, 10 cm carry), but the cube starts anywhere within ±1.5 cm and ±17° "
            "and random 0.02 N pushes hit each stick every 0.1 s. 10.5 s.")  # fmt: skip
    # the test exists before the run, with the grip alone and s3_pick15 in it, so the new policy sits beside both
    spec["setups"][SETUP] = {"label": label, "desc": desc, "env": env, "episodes": 32, "clip_seeds": [0]}
    for e in spec["entries"]:
        if e["id"] in ("scripted", "s3_pick15") and SETUP not in e["setups"]:
            e["setups"].append(SETUP)
    p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    done = train("s3_pick_hard", as_args(env), "Stage 3d: harder pick", "From s3_pick15's best; cube ±1.5 cm / ±17°, 0.02 N pushes.",
                 label, desc, pretrained=str(ROOT / "runs/train/s3_pick15/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    summary(["## Stage 3d: harder pick (cube ±1.5 cm / ±17°, 0.02 N pushes)", "",
             "- Before training: s3_pick15 21/32 on this test (29/32 on its own)",
             f"- `runs/train/s3_pick_hard`: {done or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_hard: done")


if __name__ == "__main__":
    main()
