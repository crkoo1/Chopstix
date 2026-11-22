# Chopstix on MuJoCo: plan and rules

Decisions made with Christopher on 2026-11-21. Makis' chops (SuperDex) is a reference only; nothing here depends on
its runs.

## Goal

Reliability first: a stage is only built on a stage that is solid.

## Stages (in order, each starting from the previous stage's policy, trained from scratch at stage 1)

0. **Grip** (no policy). Done when the grip lab passes. Current grip and its results: README_MUJOCO.md, "Grip lab".
   An open grip is formed and grip-lab-tested the same way before stage 2.
1. **Hold** under noise: keep both sticks in the grip under gravity, pushes, start noise and the policy's own
   exploration noise.
2. **Open / close on command**: move between the open and the closed grip while keeping both sticks.
3. **Pick and lift**, with the object made harder step by step:
   1. cube twice the size (60 mm) and half the weight (2.5 g)
   2. the same cube at full weight (5 g)
   3. the real cube: 30 mm, 5 g (as chops V4 stage C)
4. **Pick, carry and place** at B.

## When a stage is done, and when to go back

- **Done:** >= 90% success on the results page's fixed test episodes, steady over the last few checks.
- **Going back (cascade rule):** if a stage can't reach 90% after real attempts at fixing it, the problem is assumed to
  come from the stage before. Go back to that stage and raise it to **95%**, then retry. If that stage can't reach
  95% either, go back one more stage, and so on.

## Overnight runs

- Training runs at below-normal priority on all cores, any time of day. The runner keeps Windows awake while it
  trains (the PC otherwise sleeps after 5 h without input); the screen/TV may turn off. Don't log out or shut down.
- A run stops at a **plateau** (no eval improvement for a set number of evals) or at its **target** success.
- Claude may change on its own: run settings (learning rate, run length, batch sizes, seeds) and reward weights.
  Changes to env code or physics go to Christopher first.
- Claude never commits or pushes. It leaves the changes in the working tree and writes the git commands for
  Christopher to run.

## Reporting

- Results page (private): updated **every hour** during a run, and after each run.
- **Morning summary** as a separate file: `logs/morning/YYYY-MM-DD.md` (what ran, what changed, what failed, what
  next).
