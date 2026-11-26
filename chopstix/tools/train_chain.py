"""Follow-up training with the user's PPO settings (2026-11-22 10:50), then the "go back a stage" rule.

    1. wait for --wait-pid (if any)
    2. s2_openclose_v2: stage 2 (open/close on command) continued from s2_openclose's best policy, new PPO settings
    3. s1_hold95: stage 1 (hold under noise) continued from s1_hold's best, stops at 95% on 3 evals in a row
       (4 eval rounds = 56 episodes per eval, so a lucky eval can't stop it) or at a plateau
    4. s2_openclose_b: stage 2 again from s1_hold95's best policy, stops at 90% or at a plateau

PPO settings (user, for a first look at results before refining): starting lr 1e-4, desired_kl 0.02 (the adaptive learning rate was
pinned at its 1e-5 floor), entropy bonus 0.005 on the finger actions (the 6 wrist actions are fixed in these stages),
a bigger batch (128 steps x 14 envs = 1792 per update, was 896) with fewer epochs (3, was 5); the critic trains alone
for the first 20 updates since only the actor is carried over. Evals every 12 updates (about the same number of env
steps as before). Same physics, grips and noise as overnight_next. Log: runs/overnight/next_log.txt.

    .venv/Scripts/python.exe -m chopstix.tools.train_chain                 # usually via the ChopstixChain scheduled task
"""

import argparse
import time

from chopstix.tools.overnight_next import ROOT, alive, keep_awake, log, summary, train

OPEN, CLOSED, S1_CLOSED = "closed_v5_mj3_open68mm_g_makis240", "closed_v5_mj3_sq35_makis240", "closed_v5_mj3"
S1_ENV = ["stage=hold", f"hold_grip={S1_CLOSED}", "hold_joint_noise=0.02", "hold_push=0.1", "hold_wobble=5", "hold_shake=10"]
S2_ENV = ["stage=hold", f"hold_grip={OPEN}", f"hold_grip_closed={CLOSED}", "hold_joint_noise=0.02", "hold_push=0.02"]
PPO = ["--ppo", "lr=1e-4", "desired_kl=0.02", "entropy_coef=0.005", "entropy_skip_dims=6", "rollout_steps=128", "epochs=3",
       "actor_freeze_iters=20", "--eval-every", "12"]  # fmt: skip
S1_DESC = "Palm 30 cm up, gravity, 0.1 N pushes every 0.1 s, 0.02 rad start noise, palm wobble up to +-5 deg (2 Hz) and shake up to 10 m/s2 (3 Hz). 10 s."
S2_DESC = "The command moves the grip between open and closed (random goals, speeds and dwells); tips must follow, sticks held. 0.02 N pushes."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=0)
    args = p.parse_args()
    keep_awake()
    if args.wait_pid:
        log(f"train_chain: waiting for pid {args.wait_pid}")
        while alive(args.wait_pid):
            time.sleep(60)
    t0 = time.time()
    log("train_chain: new PPO settings " + " ".join(PPO))
    d0 = train("s2_openclose_v2", S2_ENV, "Stage 2 (new PPO settings): open/close on command",
               "Continued from s2_openclose's best with desired_kl 0.02, entropy 0.005, batch 1792, 3 epochs.",
               "Open/close on command", S2_DESC, pretrained=str(ROOT / "runs/train/s2_openclose/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    d1 = train("s1_hold95", S1_ENV, "Stage 1 at 95%: hold under noise", "Continued from s1_hold's best; stops at 95%.",
               "Hold under noise", S1_DESC, pretrained=str(ROOT / "runs/train/s1_hold/actor_best.pt"),
               extra=PPO + ["--eval-target-success", "0.95", "--eval-rounds", "4"], total=30e6)  # fmt: skip
    d2 = train("s2_openclose_b", S2_ENV, "Stage 2 (again): open/close on command", f"From s1_hold95's policy; open {OPEN}, closed {CLOSED}.",
               "Open/close on command", S2_DESC, pretrained=str(ROOT / "runs/train/s1_hold95/actor_best.pt"), extra=PPO, total=30e6)  # fmt: skip
    summary(["## Follow-up training (new PPO settings; stage 1 at 95%, then stage 2 again)", "",
             f"- Stage 2 continued (`runs/train/s2_openclose_v2`): {d0 or 'reached total steps'}",
             f"- Stage 1 at 95% (`runs/train/s1_hold95`): {d1 or 'reached total steps'}",
             f"- Stage 2 again (`runs/train/s2_openclose_b`): {d2 or 'reached total steps'}",
             f"- {(time.time() - t0) / 3600:.1f} h; ended {time.strftime('%Y-%m-%d %H:%M')}"])  # fmt: skip
    log("train_chain: done")


if __name__ == "__main__":
    main()
