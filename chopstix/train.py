"""Train a Shadow Hand chopsticks policy with PPO in MuJoCo (chops' chops/train.py, with SuperDex's HybridVectorEnv
replaced by gymnasium's AsyncVectorEnv: one subprocess per env, NEXT_STEP autoreset as chops' PPO expects).

    python -m chopstix.train --run-dir runs/hold_s0 --num-envs 14 --env stage=hold
    python -m chopstix.train --run-dir runs/debug --num-envs 2 --total-steps 2000

Re-running with the same --run-dir resumes from its latest checkpoint (this is how Slurm requeues continue).
On SIGTERM/SIGINT the run saves a checkpoint after the current iteration and exits with code 3.
Overrides: --env key=value (ChopsticksEnvCfg or chopstix.sim.SimCfg fields), --ppo key=value (PPOCfg),
--policy key=value (PolicyCfg); values are parsed as JSON when possible.

With --eval-every N, every N iterations the deterministic policy plays one episode in every env (on fixed seeds, so
evaluations are comparable). The best one is kept as checkpoints/best.pt and actor_best.pt (chops-actor-v1). The
run stops early once the eval success rate reaches --eval-target-success on --eval-patience consecutive evals, or
when it hasn't improved by --eval-min-delta for --eval-plateau-patience evals (0 disables either rule).
"""

import argparse
import dataclasses
import functools
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from chopstix.rl.policy import ActorCritic, PolicyCfg, save_pretrained
from chopstix.rl.ppo import PPO, PPOCfg, episode_tracker

EXIT_INTERRUPTED = 3
EVAL_SEED = 987_654  # same episodes at every evaluation


def make_env(env_overrides: dict):
    from chopstix.envs.chopsticks_env import make_env as _make

    return _make(env_overrides)


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def apply(cfg, overrides: dict):
    names = {f.name for f in dataclasses.fields(cfg)}
    unknown = set(overrides) - names
    if unknown:
        raise SystemExit(f"unknown {type(cfg).__name__} keys: {sorted(unknown)}")
    return dataclasses.replace(cfg, **overrides)


class Interrupt:
    """Sets a flag on SIGTERM / SIGUSR1 so the loop can checkpoint cleanly instead of dying mid-write."""

    def __init__(self):
        self.requested = False
        for sig in ("SIGTERM", "SIGUSR1", "SIGINT"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), self._handle)

    def _handle(self, signum, frame):
        print(f"signal {signum}: checkpointing after this iteration", flush=True)
        self.requested = True


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a preempted job never leaves a half-written latest.pt


@torch.no_grad()
def evaluate(envs, model: ActorCritic, num_envs: int, device, max_steps: int, rounds: int = 1) -> dict[str, float]:
    """`rounds` deterministic episodes per env from fixed seeds (round 0 uses the original EVAL_SEED, so a 1-round
    eval matches older runs); the envs must be reset again before training resumes."""
    successes, lengths, finishes = [], [], []
    for r in range(rounds):
        obs, _ = envs.reset(seed=EVAL_SEED + 1000 * r)
        finished = np.zeros(num_envs, dtype=bool)
        success, length = np.zeros(num_envs), np.zeros(num_envs)
        for _ in range(max_steps):
            action, _, _ = model.act(torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True)
            obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())
            length += ~finished
            ended = (terminated | truncated) & ~finished
            success[ended] = np.asarray(info.get("is_success", np.zeros(num_envs)), dtype=float)[ended]
            finished |= ended
            if finished.all():
                break
        successes.append(success), lengths.append(length), finishes.append(finished)
    return {
        "eval/success": float(np.mean(successes)), "eval/length": float(np.mean(lengths)),
        "eval/finished": float(np.mean(finishes)), "eval/episodes": float(rounds * num_envs),
    }  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=os.cpu_count())
    p.add_argument("--envs-per-worker", type=int, default=1, help="(kept for chops command lines; must be 1)")
    p.add_argument("--total-steps", type=int, default=200_000_000, help="env steps (all envs)")
    p.add_argument("--pretrained", type=str, default=None, help="BC actor checkpoint (chops-actor-v1)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint-every", type=int, default=50, help="iterations between kept checkpoints")
    p.add_argument("--checkpoint-at-steps", type=float, nargs="*", default=[], metavar="STEPS",
                   help="also keep a checkpoint once training passes each of these env-step counts, e.g. 5e6 1e7 1.5e7 "
                   "-> checkpoints/steps_5M.pt and actor_steps_5M.pt")
    p.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--ppo", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--policy", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--eval-every", type=int, default=0, help="iterations between deterministic evals (0 = off)")
    p.add_argument("--eval-target-success", type=float, default=0.0, help="stop once eval success reaches this ...")
    p.add_argument("--eval-patience", type=int, default=3, help="... on this many consecutive evals")
    p.add_argument("--eval-plateau-patience", type=int, default=0, help="stop after this many evals without a new best")
    p.add_argument("--eval-min-delta", type=float, default=0.02, help="success gain that counts as a new best")
    p.add_argument("--eval-rounds", type=int, default=1, help="eval episodes per env (more = less noisy evals)")
    args = p.parse_args()

    run_dir: Path = args.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    resuming = latest.exists()

    config_path = run_dir / "config.json"
    if resuming:
        # The run's original config wins over the command line, so a requeued job can't silently change it.
        config = json.loads(config_path.read_text())
        print(f"resuming {run_dir} from {latest}", flush=True)
    else:
        config = {
            "num_envs": args.num_envs,
            "envs_per_worker": args.envs_per_worker,
            "total_steps": args.total_steps,
            "pretrained": args.pretrained,
            "seed": args.seed,
            "env": parse_overrides(args.env),
            "ppo": dataclasses.asdict(apply(PPOCfg(), parse_overrides(args.ppo))),
            "policy": dataclasses.asdict(apply(PolicyCfg(), parse_overrides(args.policy))),
            "eval": {
                "every": args.eval_every, "target_success": args.eval_target_success, "patience": args.eval_patience,
                "plateau_patience": args.eval_plateau_patience, "min_delta": args.eval_min_delta,
                "rounds": args.eval_rounds,
            },
        }  # fmt: skip
        config_path.write_text(json.dumps(config, indent=2))
    ppo_cfg, policy_cfg = PPOCfg(**config["ppo"]), PolicyCfg(**config["policy"])
    eval_cfg = config.get("eval", {"every": 0})  # runs started before evals existed have none

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(1)  # cores belong to the simulator workers

    from gymnasium.vector import AsyncVectorEnv, AutoresetMode

    num_envs = config["num_envs"]
    if config.get("envs_per_worker", 1) != 1:
        raise SystemExit("--envs-per-worker > 1 is not supported with AsyncVectorEnv")
    from chopstix.envs.chopsticks_env import ChopsticksEnvCfg

    ChopsticksEnvCfg.from_overrides(config["env"])  # fail fast on a bad --env key, before spawning workers
    env_fn = functools.partial(make_env, config["env"])
    envs = AsyncVectorEnv([env_fn] * num_envs, autoreset_mode=AutoresetMode.NEXT_STEP, context="spawn")
    obs_dim, action_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]

    from chopstix.envs.chopsticks_env import ACTION_DIM, PROPRIO_DIM, TASK_DIM

    assert (obs_dim, action_dim) == (PROPRIO_DIM + TASK_DIM, ACTION_DIM), (obs_dim, action_dim)
    device = torch.device(args.device)
    model = ActorCritic(PROPRIO_DIM, TASK_DIM, ACTION_DIM, policy_cfg).to(device)
    ppo = PPO(model, ppo_cfg, num_envs, device)

    global_step = 0
    # best: best eval success so far; at_target / since_best: consecutive evals at the target / without a new best.
    early = {"best": -1.0, "at_target": 0, "since_best": 0, "stopped": None}
    if resuming:
        state = torch.load(latest, map_location=device, weights_only=False)
        ppo.load_state_dict(state["ppo"])
        global_step = state["global_step"]
        early = state.get("early_stop", early)
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
    elif config["pretrained"]:
        env_over = config["env"]
        # Hold-stage actuation puts its command in the parked ball's velocity slots (task 33:36), which the
        # checkpoint only ever saw at 0: start those inputs fresh.
        fresh = [33, 34, 35] if env_over.get("stage") == "hold" and env_over.get("hold_grip_closed") else []
        fresh_proprio = []
        if env_over.get("stage") == "pick":
            # A hold checkpoint saw the palm locked and the ball parked: those inputs never varied (variance ~0), and
            # in the pick stage they move. Start them fresh: palm pos/rot6d/vel (proprio 66:81), object (task 30:39;
            # with pick_place, task 33 is the OPEN command).
            fresh, fresh_proprio = list(range(30, 39)), list(range(66, 81))
            # Unless the checkpoint already saw them move: a pick checkpoint (V3 -> V4) keeps what it learned.
            saved = torch.load(config["pretrained"], map_location="cpu", weights_only=True)["actor"]
            if float(saved["proprio_norm.var"][66:81].min()) > 1e-6:
                fresh, fresh_proprio = [], []
        report = model.load_pretrained(config["pretrained"], fresh_task_inputs=fresh, fresh_proprio_inputs=fresh_proprio)
        if fresh or fresh_proprio:
            print(f"pretrained: task inputs {fresh} and proprio inputs {fresh_proprio} start fresh", flush=True)
        print(f"pretrained: loaded {len(report['loaded'])} tensors, skipped {report['skipped']}", flush=True)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(run_dir / "tb")
    tracker = None  # chops' optional W&B tracking is not ported
    interrupt = Interrupt()
    track = episode_tracker(num_envs)
    # Distinct seed per (run, resume) so a requeued job doesn't replay the same episodes.
    obs, _ = envs.reset(seed=config["seed"] * 100_003 + ppo.iteration)
    prev_done = np.zeros(num_envs, dtype=bool)
    recent: list[dict] = []
    steps_per_iter = ppo_cfg.rollout_steps * num_envs
    print(f"{num_envs} envs (one process each), {steps_per_iter} steps/iter, device {device}", flush=True)

    episode_steps = ChopsticksEnvCfg.from_overrides(config["env"]).steps_per_episode
    while global_step < config["total_steps"] and not interrupt.requested and not early["stopped"]:
        t0 = time.time()
        policy_time = env_step_time = 0.0
        model.eval()
        for _ in range(ppo_cfg.rollout_steps):
            policy_t0 = time.perf_counter()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            valid = ~prev_done  # the step after a done is the autoreset step: not a real transition
            model.update_normalizers(obs_t[torch.as_tensor(valid, device=device)])
            action, log_prob, value = model.act(obs_t)
            action_np = action.cpu().numpy()  # includes the wait for GPU policy inference
            policy_time += time.perf_counter() - policy_t0
            env_t0 = time.perf_counter()
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            env_step_time += time.perf_counter() - env_t0
            done = terminated | truncated

            reward = reward.astype(np.float64)
            if truncated.any():
                # Time-limit truncation: bootstrap from the final observation (NEXT_STEP mode returns it here).
                with torch.no_grad():
                    final_v = model.value(torch.as_tensor(next_obs, dtype=torch.float32, device=device)).cpu().numpy()
                reward = reward + ppo_cfg.gamma * final_v * (truncated & ~terminated)

            ppo.buffer.add(
                obs=obs_t, actions=action, log_probs=log_prob, values=value,
                rewards=torch.as_tensor(reward * valid, dtype=torch.float32, device=device),
                dones=torch.as_tensor(done & valid, dtype=torch.float32, device=device),
                valid=torch.as_tensor(valid, dtype=torch.float32, device=device),
            )  # fmt: skip
            recent += track(info_reward(info, reward), done & valid, valid, info)
            obs, prev_done = next_obs, done & valid
            global_step += num_envs
        rollout_time = time.time() - t0
        update_t0 = time.perf_counter()

        with torch.no_grad():
            last_value = model.value(torch.as_tensor(obs, dtype=torch.float32, device=device))
        ppo.buffer.compute_gae(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        model.train()
        stats = ppo.update()
        update_time = time.perf_counter() - update_t0
        it = ppo.iteration

        fps = steps_per_iter / (time.time() - t0)
        metrics = {f"train/{k}": v for k, v in stats.items()}
        metrics.update({
            "perf/fps": fps,
            "perf/rollout_fraction": rollout_time / (time.time() - t0),
            "perf/policy_inference_seconds": policy_time,
            "perf/env_step_seconds": env_step_time,
            "perf/ppo_update_seconds": update_time,
        })
        for k, v in metrics.items():
            writer.add_scalar(k, v, global_step)
        if recent:
            keys = sorted({k for ep in recent for k in ep})
            for k in keys:
                value = float(np.mean([ep[k] for ep in recent if k in ep]))
                writer.add_scalar(f"episode/{k}", value, global_step)
                metrics[f"episode/{k}"] = value
            print(
                f"it {it:6d}  steps {global_step:11d}  fps {fps:6.0f}  return {np.mean([e['return'] for e in recent]):8.2f}"
                f"  success {np.mean([e.get('is_success', 0) for e in recent]):.3f}  len {np.mean([e['length'] for e in recent]):5.1f}"
                f"  kl {stats['kl']:.4f}  lr {stats['lr']:.1e}",
                flush=True,
            )
            recent = []

        new_best = False
        if eval_cfg["every"] > 0 and it % eval_cfg["every"] == 0:
            model.eval()
            result = evaluate(envs, model, num_envs, device, episode_steps + 1, eval_cfg.get("rounds", 1))
            # The evals ended every env's episode mid-flight: start training episodes afresh.
            obs, _ = envs.reset(seed=config["seed"] * 100_003 + it)
            prev_done = np.zeros(num_envs, dtype=bool)
            track = episode_tracker(num_envs)
            succ = result["eval/success"]
            new_best = succ >= early["best"] + eval_cfg["min_delta"] or early["best"] < 0
            if new_best:
                early["best"], early["since_best"] = succ, 0
            else:
                early["since_best"] += 1
            target = eval_cfg["target_success"]
            early["at_target"] = early["at_target"] + 1 if target > 0 and succ >= target else 0
            if target > 0 and early["at_target"] >= eval_cfg["patience"]:
                early["stopped"] = f"eval success >= {target} on {early['at_target']} consecutive evals"
            elif eval_cfg["plateau_patience"] > 0 and early["since_best"] >= eval_cfg["plateau_patience"]:
                early["stopped"] = f"no eval improvement in {early['since_best']} evals (best {early['best']:.3f})"
            result["eval/best_success"] = early["best"]
            for k, v in result.items():
                writer.add_scalar(k, v, global_step)
            metrics.update(result)
            print(
                f"eval it {it}: success {succ:.3f}  len {result['eval/length']:.1f}  best {early['best']:.3f}"
                + (f"  -> stopping: {early['stopped']}" if early["stopped"] else ""),
                flush=True,
            )

        state = {
            "ppo": ppo.state_dict(), "global_step": global_step, "config": config,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "early_stop": early,
        }  # fmt: skip
        save_checkpoint(latest, state)
        if it % args.checkpoint_every == 0:
            save_checkpoint(ckpt_dir / f"it_{it:06d}.pt", state)
        for mark in args.checkpoint_at_steps:  # first iteration at or past each mark (once: the file is the record)
            name = f"steps_{mark / 1e6:g}M"
            if global_step >= mark and not (ckpt_dir / f"{name}.pt").exists():
                save_checkpoint(ckpt_dir / f"{name}.pt", state)
                save_pretrained(model.actor, model.dims, str(run_dir / f"actor_{name}.pt"))
                print(f"kept checkpoint {name} at {global_step} steps", flush=True)
        if new_best:
            save_checkpoint(ckpt_dir / "best.pt", state)
            save_pretrained(model.actor, model.dims, str(run_dir / "actor_best.pt"))
        if tracker:
            tracker.log(metrics, global_step)

    writer.close()
    envs.close()
    if tracker:
        tracker.finish(interrupt.requested)
    if interrupt.requested and global_step < config["total_steps"] and not early["stopped"]:
        print("interrupted; checkpoint saved, exiting for requeue", flush=True)
        return EXIT_INTERRUPTED
    (run_dir / "DONE").write_text(f"early stop: {early['stopped']}\n" if early["stopped"] else "")
    return 0


def info_reward(info: dict, reward: np.ndarray) -> np.ndarray:
    """Episode returns are tracked on the raw env reward (sum of reward_* terms), without truncation bootstrapping."""
    terms = [v for k, v in info.items() if k.startswith("reward_") and isinstance(v, np.ndarray)]
    return np.sum(terms, axis=0) if terms else reward


if __name__ == "__main__":
    sys.exit(main())
