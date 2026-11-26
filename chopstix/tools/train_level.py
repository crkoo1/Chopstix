"""Stage 4c (user, 2026-11-23): train the human-style grip hg_1 in the sideways-closing pose, stage by stage, as with the
previous hand pose: 1) hold under disturbances, 2) open/close on command, 3) pick and place. Each stage starts from the
previous stage's best policy, and only hands over once its best eval reaches 90% (user: "make sure that each step reaches
the 90% success rate").

Gate (per stage):
    - a run stops at 95% on 3 evals in a row, or after 60 evals without a new best;
    - best eval >= 90% -> next stage, from this stage's best policy;
    - below 90% -> another run continuing from this stage's best so far (new seed);
    - a second miss in stage 2 or 3 -> go back a stage once (continue the previous stage from its best, new seed), then
      redo this stage from the refreshed policy;
    - at most 4 runs per stage; if a stage still misses 90%, the chain stops there (no next stage).

Pose: back of the hand up (`pick_hand_flip`), sticks 20 deg down, hand turned so the tips open and close sideways
(`pick_level_open`). Stages 1 and 2 hold the palm in that same pose (`hold_palm_rot`, 30 cm up), so what they learn
carries over to the pick. Untrained hg_1 in this pose: stage 1 24/28, stage 2 22/28, pick 17/32 placed.

PPO (user): batch 256 steps x 14 envs = 3584 per update (was 1792), 2 epochs (was 3), and a low learning rate for
accuracy: starts at 5e-5, and the adaptive rule may move it only between 1e-5 and --lr-max (the first try at 2e-4 had
it climb to the 1e-2 cap within 15 updates). 56-episode evals every 12 updates.

Second try (user, 16:50): the first stage 1 run (s4l_hold, hg_1, +-0.1 rad, cap 1e-4) sat at 80% for an hour, failing
the same strong-wobble episodes as the bare grip (the pair slid 10-20 mm through the hand). Now: a tighter grip
(chopstix.tools.tighten_hg), the policy's finger range doubled to +-0.2 rad in every stage (`hold_residual`, kept the
same across stages so the policies carry over), and the learning-rate cap raised to 3e-4.

    .venv/Scripts/python.exe -m chopstix.tools.train_level --open hg_1t200_open --closed hg_1t200 --prefix s4t
"""

import argparse
import re
import time

from chopstix.envs.chopsticks_env import make_env
from chopstix.tools.grip_level import env_for
from chopstix.tools.overnight_next import ROOT, keep_awake, log, summary, train
from chopstix.tools.train_next import as_args

GATE, MAX_RUNS = 0.90, 4
STOP = ["--eval-every", "12", "--eval-rounds", "4", "--eval-plateau-patience", "60", "--eval-target-success", "0.95", "--num-envs", "14"]
POSE = ("Back of the hand up, hand turned so the tips close sideways, sticks 20 deg down; thumb pad, index and middle "
        "on the upper stick, thumb base, index base and ring finger on the lower.")  # fmt: skip


def best_eval(name: str) -> float:
    try:
        found = re.findall(r"best ([0-9.]+)", (ROOT / "runs" / "train" / name / "train_log.txt").read_text(encoding="utf-8"))
        return float(found[-1]) if found else 0.0
    except OSError:
        return 0.0


def main():
    global GATE, STOP, PPO_L
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--open", default="hg_1_open74mm")
    p.add_argument("--closed", default="hg_1")
    p.add_argument("--prefix", default="s4l", help="run names: <prefix>_hold, <prefix>_openclose, <prefix>_pick (+ _<n>)")
    p.add_argument("--residual", type=float, default=0.2, help="policy finger range (rad), the same in every stage")
    p.add_argument("--lr-max", type=float, default=3e-4)
    p.add_argument("--solref", type=float, default=0.0, help="hand-stick contact stiffness (MuJoCo solref time constant) in every stage; 0 = default 0.02")
    # second pass (user, 2026-11-23 21:45): smaller learning rate, more episodes, 98% gate, from the first pass's bests
    p.add_argument("--gate", type=float, default=GATE, help="best eval a stage needs before the next (also its early-stop level)")
    p.add_argument("--lr", type=float, default=5e-5, help="starting learning rate")
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--rollout", type=int, default=256, help="steps per env per update (x 14 envs = the batch)")
    p.add_argument("--eval-rounds", type=int, default=4, help="eval episodes = rounds x 14")
    p.add_argument("--patience", type=int, default=60, help="stop a run after this many evals without a new best")
    p.add_argument("--total", type=float, default=0.0, help="max env steps per run (0 = 30M, 40M for the pick)")
    p.add_argument("--init-from", default="", help="first-pass prefix: each stage's first run continues from that pass's best run of the same stage")
    p.add_argument("--wait-done", action="store_true", help="first wait for the chain already running to log 'train_level: done'")
    args = p.parse_args()
    OPEN, CLOSED = args.open, args.closed
    GATE = args.gate
    STOP = ["--eval-every", "12", "--eval-rounds", str(args.eval_rounds), "--eval-plateau-patience", str(args.patience),
            "--eval-target-success", str(max(GATE, 0.95)), "--num-envs", "14"]  # fmt: skip
    PPO_L = ["--ppo", f"lr={args.lr}", f"lr_min={args.lr_min}", f"lr_max={args.lr_max}", "desired_kl=0.02", "entropy_coef=0.005",
             "entropy_skip_dims=6", f"rollout_steps={args.rollout}", "epochs=2"]  # fmt: skip
    keep_awake()
    if args.wait_done:
        logf = ROOT / "runs" / "overnight" / "next_log.txt"
        seen = logf.stat().st_size
        log(f"train_level: waiting for the running chain to finish before {args.prefix}_*")
        while "train_level: done" not in logf.read_text(encoding="utf-8", errors="replace")[seen:]:
            time.sleep(120)
    init = {}
    if args.init_from:  # the last "stage best X% in RUN" line per stage of that pass
        text = (ROOT / "runs" / "overnight" / "next_log.txt").read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(rf"train_level: {re.escape(args.init_from)}_(hold|openclose|pick)\S*: best \S+ \(stage best \S+ in (\S+)\)", text):
            init[m.group(1)] = m.group(2)
        log(f"train_level: {args.prefix}_* starts each stage from {init}")
    t0 = time.time()
    sim = {"sim": {"solref": [args.solref, 1.0]}} if args.solref else {}
    pick = {**env_for(OPEN, CLOSED), "hold_residual": args.residual, **sim}
    rot = make_env(pick)._pick_palm_rot.flatten().tolist()
    log(f"train_level: {OPEN} -> {CLOSED}, finger range +-{args.residual} rad, lr cap {args.lr_max}, sim {sim or "default"}, runs {args.prefix}_*")
    stages = [
        ("hold", {"stage": "hold", "hold_grip": CLOSED, "hold_joint_noise": 0.02, "hold_push": 0.1, "hold_wobble": 5,
                  "hold_shake": 10, "hold_palm_rot": rot, "hold_residual": args.residual, **sim},
         "hold under noise", "Palm 30 cm up; 0.1 N pushes, wobble 5 deg, shake 10 m/s2. 10 s."),
        ("openclose", {"stage": "hold", "hold_grip": OPEN, "hold_grip_closed": CLOSED, "hold_joint_noise": 0.02, "hold_push": 0.02,
                       "hold_palm_rot": rot, "hold_residual": args.residual, **sim},
         "open/close on command", "Palm 30 cm up; the grip opens and closes on command; 0.02 N pushes."),
        ("pick", pick, "pick and place, 25 mm cube",
         "Palm comes down with the grip open, closes on a 25 mm cube, lifts it, carries it to B, lowers it and opens."),
    ]  # fmt: skip
    runs = {k: 0 for k, *_ in stages}
    best = {}  # stage -> (best eval, run name)
    from_prev = {k: True for k, *_ in stages}  # next run of this stage starts from the previous stage's best
    went_back = set()
    lines = []
    i = 0
    while i < len(stages):
        key, env, what, desc = stages[i]
        runs[key] += 1
        if runs[key] > MAX_RUNS:
            log(f"train_level: stage {i + 1} ({what}) missed {GATE:.0%} after {MAX_RUNS} runs; stopping the chain")
            lines.append(f"- Stage {i + 1} ({what}) missed {GATE:.0%} after {MAX_RUNS} runs (best {best[key][0]:.0%}); chain stopped")
            break
        name = f"{args.prefix}_{key}" + (f"_{runs[key]}" if runs[key] > 1 else "")
        if runs[key] == 1 and key in init:
            src = init[key]  # second pass: carry on from the first pass's best for this stage
        else:
            src = (best[stages[i - 1][0]][1] if i else None) if from_prev[key] else best[key][1]
        pre = str(ROOT / "runs" / "train" / src / "actor_best.pt") if src else None
        extra = PPO_L + (["actor_freeze_iters=20"] if pre else []) + STOP + ["--seed", str(runs[key] - 1)]
        start = f"from {src}'s best" if src else "from scratch"
        done = train(name, as_args(env), f"Stage 4c-{i + 1} (run {runs[key]}): {CLOSED} sideways, {what}", f"{start}; gate {GATE:.0%}.",
                     f"{CLOSED} sideways: {what}", POSE + " " + desc, pretrained=pre, extra=extra,
                     total=args.total or (40e6 if key == "pick" else 30e6))  # fmt: skip
        b = best_eval(name)
        if key not in best or b > best[key][0] or b >= GATE:  # a passing rerun (after going back) is the fresher policy
            best[key] = (b, name)
        from_prev[key] = False
        lines.append(f"- Stage {i + 1} ({what}), `runs/train/{name}` ({start}): best {b:.1%}; {done or 'reached total steps'}")
        log(f"train_level: {name}: best {b:.1%} (stage best {best[key][0]:.1%} in {best[key][1]})")
        if best[key][0] >= GATE:
            i += 1
            if i < len(stages):
                from_prev[stages[i][0]] = True
        elif runs[key] >= 2 and i > 0 and key not in went_back:
            went_back.add(key)
            from_prev[key] = True  # redo this stage from the refreshed previous stage
            i -= 1
            log(f"train_level: going back to stage {i + 1} once before retrying {key}")
    summary([f"## Stage 4c ({args.prefix}): grip {CLOSED}, sideways closing: hold, open/close, pick ({GATE:.0%} gate per stage)", "", *lines,
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_level: done")


if __name__ == "__main__":
    main()
