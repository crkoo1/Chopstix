# Copied from makislam/chops (chops/rl/ppo.py); only the import path differs.
"""PPO for CPU-simulated vector envs (MuJoCo here, SuperDex in chops) (Gymnasium NEXT_STEP autoreset).

With NEXT_STEP autoreset, the step that ends an episode returns that episode's final observation, and the following
step() call resets the env, ignores its action and returns the first observation of the next episode. That reset step
is not a real transition, so it is masked out of the loss (`valid`), and truncated episodes bootstrap from the final
observation's value.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from chopstix.rl.policy import ActorCritic


@dataclass
class PPOCfg:
    rollout_steps: int = 64  # per env, per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 1.0
    entropy_coef: float = 0.0
    # Leading action dims left out of the entropy bonus. Set 6 in the hold stage: the env zeroes the wrist actions, so
    # nothing in the loss would stop the bonus inflating their std without bound.
    entropy_skip_dims: int = 0
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    desired_kl: float = 0.01  # adaptive learning rate; <= 0 disables
    actor_freeze_iters: int = 0  # train only the critic for this many iterations (useful after pretraining)


class RolloutBuffer:
    def __init__(self, steps: int, num_envs: int, obs_dim: int, action_dim: int, device):
        def z(*shape, dtype=torch.float32):
            return torch.zeros(steps, num_envs, *shape, dtype=dtype, device=device)

        self.obs, self.actions = z(obs_dim), z(action_dim)
        self.log_probs, self.values, self.rewards = z(), z(), z()
        self.dones, self.valid = z(), z()
        self.advantages, self.returns = z(), z()
        self.step = 0

    def add(self, **tensors):
        for k, v in tensors.items():
            getattr(self, k)[self.step] = v
        self.step += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        gae = torch.zeros_like(last_value)
        for t in reversed(range(self.step)):
            next_value = last_value if t == self.step - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
        self.advantages *= self.valid
        self.returns = self.advantages + self.values
        self.step = 0


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOCfg, num_envs: int, device):
        self.model, self.cfg, self.device = model, cfg, device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        self.buffer = RolloutBuffer(
            cfg.rollout_steps, num_envs, sum(model.dims[k] for k in ("proprio", "task")), model.dims["action"], device
        )
        self.iteration = 0

    def update(self) -> dict[str, float]:
        cfg, buf, model = self.cfg, self.buffer, self.model
        n = buf.obs.shape[0] * buf.obs.shape[1]
        flat = lambda x: x.reshape(n, *x.shape[2:])  # noqa: E731
        obs, actions, old_log_probs = flat(buf.obs), flat(buf.actions), flat(buf.log_probs)
        advantages, returns, old_values, valid = flat(buf.advantages), flat(buf.returns), flat(buf.values), flat(buf.valid)
        idx_valid = valid > 0
        adv_mean, adv_std = advantages[idx_valid].mean(), advantages[idx_valid].std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        train_actor = self.iteration >= cfg.actor_freeze_iters
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        updates = 0
        for _ in range(cfg.epochs):
            for mb in torch.randperm(n, device=self.device).chunk(cfg.minibatches):
                w = valid[mb]
                denom = w.sum().clamp(min=1.0)
                dist = model.distribution(obs[mb])
                log_probs = dist.log_prob(actions[mb]).sum(-1)
                ratio = (log_probs - old_log_probs[mb]).exp()
                surrogate = torch.min(
                    ratio * advantages[mb], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages[mb]
                )
                policy_loss = -(surrogate * w).sum() / denom
                entropy = (dist.entropy()[:, cfg.entropy_skip_dims :].sum(-1) * w).sum() / denom

                values = model.value(obs[mb])
                clipped = old_values[mb] + (values - old_values[mb]).clamp(-cfg.clip, cfg.clip)
                value_loss = (torch.max((values - returns[mb]).square(), (clipped - returns[mb]).square()) * w).sum() / denom

                loss = cfg.value_coef * value_loss
                if train_actor:
                    loss = loss + policy_loss - cfg.entropy_coef * entropy

                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[mb]
                    kl = (((ratio - 1) - log_ratio) * w).sum() / denom
                    clip_frac = (((ratio - 1).abs() > cfg.clip).float() * w).sum() / denom
                if cfg.desired_kl > 0 and train_actor:
                    if kl > 2 * cfg.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < cfg.desired_kl / 2:
                        self.lr = min(1e-2, self.lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.lr

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", entropy), ("kl", kl), ("clip_frac", clip_frac)):
                    stats[k] += v.item()
                updates += 1

        self.iteration += 1
        stats = {k: v / updates for k, v in stats.items()}
        stats["lr"] = self.lr
        stats["action_std"] = float(model.actor.log_std.detach()[cfg.entropy_skip_dims :].exp().mean())
        stats["train_actor"] = float(train_actor)
        return stats

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "lr": self.lr, "iteration": self.iteration}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr, self.iteration = state["lr"], state["iteration"]


def episode_tracker(num_envs: int):
    """Accumulates per-env returns; call with each step's (reward, done, valid, info) to get finished episodes."""
    returns = np.zeros(num_envs)
    lengths = np.zeros(num_envs, dtype=int)

    def update(reward, done, valid, info):
        nonlocal returns, lengths
        returns += reward * valid
        lengths += valid.astype(int)
        finished = []
        for i in np.flatnonzero(done):
            ep = {"return": float(returns[i]), "length": int(lengths[i])}
            for key, value in info.items():
                if key.startswith("_") or not isinstance(value, np.ndarray) or value.shape[:1] != (num_envs,):
                    continue
                if info.get(f"_{key}", np.ones(num_envs, bool))[i] and np.issubdtype(value.dtype, np.number) | (value.dtype == bool):
                    ep[key] = float(value[i])
            finished.append(ep)
            returns[i], lengths[i] = 0.0, 0
        return finished

    return update
