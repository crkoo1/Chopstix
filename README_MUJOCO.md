# Chopstix on MuJoCo (branch `mujoco`)

Makis' [chops](https://github.com/makislam/chops) framework (tripod grip, hold/pick stages, PPO), ported from SuperDex to
MuJoCo so it trains on an ordinary PC. Everything new lives in `chopstix/`; the older `src/` skills are untouched.

```bash
git submodule update --init                     # sr_common (Shadow Hand E model, BSD-3)
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe mujoco==3.15.0 numpy gymnasium==1.4.0 tensorboard pillow trimesh scipy \
    torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## What is where

| Path | What |
|---|---|
| `chopstix/sim.py` | Scene: sr_description Shadow Hand E (the same convex colliders chops baked for SuperDex), free root with the palm welded to a mocap target, 22 position actuators, two free tapered chopsticks (23 cm, 3.5 → 2 mm), optional cube. Every physics knob is a `SimCfg` field. |
| `chopstix/envs/chopsticks_env.py` | `hold` and `pick` stages of chops' env with the same phases, rewards and success tests, and the **same observation (81 + 40) and action (28) layout and joint order**, so chops actors load here. |
| `chopstix/rl/`, `chopstix/train.py` | chops' actor-critic and PPO (copied), training loop on gymnasium `AsyncVectorEnv`. |
| `chopstix/grips/` | chops' grip files, plus `closed_v5_mj` (closed_v5 re-formed in MuJoCo). |
| `chopstix/tools/` | `hold_test` (stage-1 grip test), `form_grip` (re-form a grip in MuJoCo), `sweep`/`form_sweep` (parallel), `render`, `rollout` (GIF). |

## Step 2: hand posture and grip

A SuperDex grip loaded as-is fails in MuJoCo (0% hold): SuperDex's contacts let the sticks rest 3-6 mm inside the
fingers, so MuJoCo pushes back with ~6 N and the upper stick creeps out in ~4 s. `form_grip` keeps the grip's hand
posture and stick placement and lets MuJoCo settle it (relax with soft clamps, then pinch each finger-stick contact
along its contact Jacobian). The key was the lower stick: the thumb's middle phalanx must press it into the palm, with
the ring fingertip near its middle.

`closed_v5_mj` (joints within 0.037 rad rms of closed_v5, sticks within 4 mm):

| Physics | Hold rate, 20 runs x 10 s, 0.02 rad joint noise, 0.03 N pushes |
|---|---|
| timestep 5 ms, `noslip_iterations=10` (env default) | **100%** |
| timestep 4 ms / 2 ms, noslip 10 | 100% / 100% |
| friction 0.7 / 1.0 | 100% / 100% |
| noslip 5 / 2 / 0 | 95% / 80% / 25% |
| joint gains -20% | 95% |
| sticks 15 g instead of 10 g | 90% |

```bash
.venv/Scripts/python.exe -m chopstix.tools.hold_test --grip closed_v5_mj --seeds 20 --duration 10 --joint-noise 0.02 \
    --set noslip_iterations=10 timestep=0.005
.venv/Scripts/python.exe -m chopstix.tools.form_grip --source closed_v5 --save-as closed_v5_mj \
    --pinch th-upper=0.01 th-lower=0.15 ff-upper=0.01 mf-upper=0.01 rf-lower=0.1 --set noslip_iterations=5
```

## Step 1: environment and training

- `hold`, zero action (the scripted grip): 8/8 full 10 s episodes successful, sticks within 2 mm.
- `pick` (V4 stage C settings, 30 mm cube, place at B), zero action: the palm follows the scripted path; the bare
  grip lifted the cube 60 mm in one of six seeds, the rest miss or drop it (that is what the policy must learn).
- Speed: ~135 env steps/s per process (SuperDex: 6.6). 5.4 ms of each step is physics, ~3 ms Python.

```bash
.venv/Scripts/python.exe -m chopstix.train --run-dir runs/hold_s0 --num-envs 14 --device cpu \
    --env stage=hold --policy init_log_std=-2.5 --eval-every 25
.venv/Scripts/python.exe -m chopstix.tools.rollout --env stage=hold --out runs/gifs/hold.gif
```

Notes: MuJoCo mixes contact friction with max(); hand and table geoms get priority so they keep `friction` while
sticks and cube use their own (chops: 2.0) between themselves. chops' SuperDex-only settings (`joint_stiffness`,
`solver_newton_iters`, ...) don't apply here.

## Pictures

- `docs/mujoco/closed_v5_mj.png`: the MuJoCo grip at load (top) and after 5 s of gravity and pushes (bottom)
- `docs/mujoco/closed_v5_superdex_as_is.png`: chops' closed_v5 loaded as-is (the upper stick turns)
- `docs/mujoco/hold_scripted.gif`, `docs/mujoco/pick_scripted_seed0.gif`: zero-action episodes

## Results page

Live page: https://claude.ai/artifact/V9EtFBWUnWTDE3M9tjaHtU (private until shared from its Share menu).
`site/index.html` reads `site/data/`, which `chopstix.tools.report` writes:

```bash
# after each training run or check: rerun the fixed test episodes and add a dated snapshot to the history
.venv/Scripts/python.exe -m chopstix.tools.report --note "what changed"
```

Then republish `site/` to the same link. `site/lineup.json` lists the test setups and the policies on the bench; add a
trained policy as an entry with its `actor` (and `run`, for its training curve). Every snapshot reruns the same seeds
with the policy's mean action, so snapshots are comparable.

## Grip lab (training grip: `closed_v5_mj4`)

`chopstix.tools.grip_lab` tests a grip alone (no policy), 20 runs x 10 s per level: gravity at 9 hand orientations,
then disturbances stepped up: pushes on the sticks (grip and pick pose), finger-target jitter, start-joint noise, the
palm along the pick path (up to 4x speed), palm shaking (3 Hz), palm wobble (2 Hz), heavier sticks, slippery fingers and
the physics step size. A run holds when both sticks stay within 1 cm and their axes tilt < 10 deg (spin about a round
stick's own axis is ignored), with 3 fingertips still on them.

| Hardest level held at >= 90% | mj2 (~5 N) | mj3 (~8 N) | **mj4 (~10.5 N, default)** |
|---|---|---|---|
| Gravity, 9 orientations | 9/9 | 9/9 | 9/9 |
| Pushes, grip / pick pose | 0.5 / 0.3 N | 0.5 / 0.5 N | 0.5 / 0.5 N |
| Finger jitter | 0.05 rad | 0.08 rad | 0.05 rad (70% at 0.08) |
| Start-joint noise | 0.1 rad | 0.1 rad | 0.1 rad |
| Palm along the pick path | 4x | 4x | 4x |
| Palm shaking | 10 m/s2 | 20 m/s2 | 30 m/s2 |
| Palm wobble | none (25% at 5 deg) | none (20%) | none (45%) |
| Stick mass | 30 g | 30 g | 30 g |
| Hand friction down to | 0.5 | 0.3 | 0.3 |
| Physics step | 2-10 ms | 2-10 ms | 2-10 ms |

What made the difference: a firm pinch on the lower stick (the thumb's middle phalanx presses it into the palm, the
ring fingertip near its middle) and a 0.15 s first-order lag on the finger targets (`actuator_timeconst`; without it,
~0.01 rad of exploration noise per control step walks the lower stick out within 10 s). Open issue: rocking the palm
(wobble) slowly walks both sticks out; more force, friction, rolling friction, contact stiffness or finger stiffness
don't fix it. The current tasks don't rotate the palm; the plan is to have the policy learn to counter it.

```bash
.venv/Scripts/python.exe -m chopstix.tools.grip_lab --grip closed_v5_mj4 --record
.venv/Scripts/python.exe -m chopstix.tools.form_grip --source closed_v5 --save-as closed_v5_mj4     --pinch th-upper=0.02 ff-upper=0.02 mf-upper=0.02 th-lower=0.6 rf-lower=0.5     --set timestep=0.005 noslip_iterations=10 actuator_timeconst=0.15
```
