"""Stage 4d (user, 2026-11-24): robustness training of the pick stage only, after the 98% second pass (s4v) finishes.

The user wants the policy taken to near-failure states so it learns to recover, and no entropy bonus ("it reduces the
quality"). Pose 4 (grip hg_1s125, sideways closing, stiffer contacts, finger range +-0.2 rad), pick and place of the
25 mm cube, starting from the best pick policy so far (s4u_pick or an s4v_pick run, by best check). On top of the task:
    - pushes on the sticks during the whole pick (0.05 N per axis, new every 2 steps)
    - shoves on the cube at random moments of the lift and carry (0.1 N, 0.2 s each, 5% chance per step)
    - wider starts: cube +-2 cm and +-0.4 rad, finger joints +-0.04 rad
    - physics randomization drawn per env: stick and cube friction 0.6-1.4x, cube mass 0.5-2x, cube size +-12%
    - restart from failures: 20% of training episodes start from a state saved 0.25 s before the cube began to slip
Calibration (s4u_pick's best, 32 episodes): 30/32 without these, 17-19/32 with all of them on.

PPO: as the second pass (lr 2e-5 adapting 5e-6..1e-4, 512-step batches, 2 epochs, 112-episode checks every 12 updates,
stop at 95% on 3 checks or 100 checks without a new best) but entropy bonus 0. Gate 90%: below it, another run from
this stage's best with a new seed, at most 4 runs. Checks use the same pushes, starts and per-env physics, never restarts.

    .venv/Scripts/python.exe -m chopstix.tools.train_robust --wait-done      # usually via the ChopstixRobust task
"""

import argparse
import json
import re
import time

from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_next import as_args

TRAIN = ROOT / "runs" / "train"
GATE, MAX_RUNS = 0.90, 4
ROBUST = {
    "pick_push": 0.05, "pick_object_push": 0.1, "pick_object_noise": 0.02, "pick_object_yaw_noise": 0.4, "hold_joint_noise": 0.04,
    "domain_rand": {"stick_friction": [0.6, 1.4], "object_friction": [0.6, 1.4], "object_mass": [0.5, 2.0], "object_size": [0.88, 1.12]},
    "pick_restart_prob": 0.2,
}  # fmt: skip
PPO_R = ["--ppo", "lr=2e-5", "lr_min=5e-6", "lr_max=1e-4", "desired_kl=0.02", "entropy_coef=0.0", "entropy_skip_dims=6",
         "rollout_steps=512", "epochs=2", "actor_freeze_iters=20"]  # fmt: skip
STOP = ["--eval-every", "12", "--eval-rounds", "8", "--eval-plateau-patience", "100", "--eval-target-success", "0.95", "--num-envs", "14"]


def best_eval(name: str) -> float:
    try:
        found = re.findall(r"eval it\s+\d+: success ([0-9.]+)", (TRAIN / name / "train_log.txt").read_text(encoding="utf-8"))
        return max(map(float, found)) if found else 0.0
    except OSError:
        return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-done", action="store_true", help="wait for the running chain (second pass) to log 'train_level: done'")
    p.add_argument("--prefix", default="s4r")
    args = p.parse_args()
    keep_awake()
    if args.wait_done:
        logf = ROOT / "runs" / "overnight" / "next_log.txt"
        seen = logf.stat().st_size
        log("train_robust: waiting for the second pass (s4v) to finish")
        while "train_level: done" not in logf.read_text(encoding="utf-8", errors="replace")[seen:]:
            time.sleep(120)
    t0 = time.time()
    picks = [d.name for d in TRAIN.iterdir() if d.is_dir() and re.fullmatch(r"s4[uv]_pick(_\d+)?", d.name) and (d / "actor_best.pt").exists()]
    src = max(picks, key=lambda n: (best_eval(n), n))
    env = {**json.loads((TRAIN / src / "config.json").read_text())["env"], **ROBUST}
    log(f"train_robust: from {src} (best check {best_eval(src):.1%}); robustness {ROBUST}")
    lines, best, run = [], (0.0, src), 0
    while run < MAX_RUNS:
        run += 1
        name = f"{args.prefix}_pick" + (f"_{run}" if run > 1 else "")
        done = train(name, as_args(env), f"Stage 4d (run {run}): robust pick, pushes + wide starts + random physics + restarts",
                     f"From {best[1]}'s best; no entropy bonus; gate {GATE:.0%}.", "hg_1s125 sideways: robust pick and place, 25 mm cube",
                     "Pick and place with stick pushes, cube shoves during lift and carry, cube placed up to 2 cm off and turned up "
                     "to 0.4 rad, finger joint noise, and friction, cube mass and cube size varied per env.",
                     pretrained=str(TRAIN / best[1] / "actor_best.pt"), extra=PPO_R + STOP + ["--seed", str(run - 1)], total=60e6)  # fmt: skip
        b = best_eval(name)
        lines.append(f"- `runs/train/{name}` (from {best[1]}): best check {b:.1%}; {done or 'reached total steps'}")
        log(f"train_robust: {name}: best check {b:.1%}")
        if run == 1 or b > best[0]:
            best = (b, name)
        if best[0] >= GATE:
            break
    summary(["## Stage 4d: robust pick (pushes, wide starts, random physics, restarts; no entropy bonus)", "", *lines,
             f"- Best: {best[1]} at {best[0]:.1%} (gate {GATE:.0%}); {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_robust: done")


if __name__ == "__main__":
    main()
