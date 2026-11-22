# Copied unchanged from makislam/chops (chops/rl/policy.py); only the import path differs.
"""Actor-critic shared by DexGraspNet behaviour-cloning pretraining and SuperDex PPO fine-tuning.

Observations come in two groups (see chops.envs.chopsticks_env):
    proprio: hand-only state, same layout in every task  -> proprio encoder (transfers)
    task:    object state, task-specific                 -> task encoder (re-initialised if its size changes)
Both are normalised by running mean/std, concatenated after encoding and fed to a shared trunk that outputs the
mean of a Gaussian over actions (palm pose delta + 22 joint target deltas).

Pretrained checkpoint format (what the BC pretraining must save, see `save_pretrained`):
    {"format": "chops-actor-v1", "dims": {"proprio": P, "task": T, "action": A}, "actor": Actor.state_dict()}
Fine-tuning loads every tensor whose name and shape match and reports what it skipped.
"""

from dataclasses import dataclass, field

import torch
from torch import nn

PRETRAINED_FORMAT = "chops-actor-v1"


class RunningNorm(nn.Module):
    """Running mean/variance normaliser (parallel-variance update), stored as buffers so it is checkpointed."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        if self.frozen or x.shape[0] == 0:
            return
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(sizes: list[int], activation=nn.ELU, last_activation: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


@dataclass
class PolicyCfg:
    proprio_hidden: list[int] = field(default_factory=lambda: [256])
    task_hidden: list[int] = field(default_factory=lambda: [256])
    trunk_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden: list[int] = field(default_factory=lambda: [512, 256, 128])
    init_log_std: float = -1.0


class Actor(nn.Module):
    def __init__(self, proprio_dim: int, task_dim: int, action_dim: int, cfg: PolicyCfg):
        super().__init__()
        self.proprio_norm = RunningNorm(proprio_dim)
        self.task_norm = RunningNorm(task_dim)
        self.proprio_enc = mlp([proprio_dim, *cfg.proprio_hidden])
        self.task_enc = mlp([task_dim, *cfg.task_hidden])
        self.trunk = mlp([cfg.proprio_hidden[-1] + cfg.task_hidden[-1], *cfg.trunk_hidden, action_dim], last_activation=False)
        self.log_std = nn.Parameter(torch.full((action_dim,), cfg.init_log_std))
        # Small last layer: the initial policy stays near "hold still".
        nn.init.uniform_(self.trunk[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, proprio: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.proprio_enc(self.proprio_norm(proprio)), self.task_enc(self.task_norm(task))], -1)
        return self.trunk(h)


class ActorCritic(nn.Module):
    def __init__(self, proprio_dim: int, task_dim: int, action_dim: int, cfg: PolicyCfg | None = None):
        super().__init__()
        cfg = cfg or PolicyCfg()
        self.dims = {"proprio": proprio_dim, "task": task_dim, "action": action_dim}
        self.actor = Actor(proprio_dim, task_dim, action_dim, cfg)
        self.critic_norm = RunningNorm(proprio_dim + task_dim)
        self.critic = mlp([proprio_dim + task_dim, *cfg.critic_hidden, 1], last_activation=False)

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return obs[..., : self.dims["proprio"]], obs[..., self.dims["proprio"] :]

    def update_normalizers(self, obs: torch.Tensor) -> None:
        proprio, task = self.split(obs)
        self.actor.proprio_norm.update(proprio)
        self.actor.task_norm.update(task)
        self.critic_norm.update(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(*self.split(obs))
        return torch.distributions.Normal(mean, self.actor.log_std.exp().expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.critic_norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def load_pretrained(self, path: str, fresh_task_inputs: list[int] = (), fresh_proprio_inputs: list[int] = ()) -> dict[str, list[str]]:
        """Initialise the actor from a BC checkpoint. Returns {"loaded": [...], "skipped": [...]}.

        fresh_task_inputs: task-observation indices that carry a new signal the checkpoint never saw vary (e.g. the
        hold stage's open/close command in the parked ball's velocity slots). Their normaliser is reset to identity
        and their first-layer weights zeroed, so the loaded actor starts out exactly as it was trained.
        fresh_proprio_inputs: the same for proprio indices (e.g. the palm pose and velocity, constant in the hold stage
        but moving in the pick stage; their checkpoint variance is ~0, so they would otherwise saturate the input)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != PRETRAINED_FORMAT:
            raise ValueError(f"{path}: expected format {PRETRAINED_FORMAT!r}, got {ckpt.get('format')!r}")
        if ckpt["dims"]["action"] != self.dims["action"]:
            raise ValueError(f"action dim mismatch: pretrained {ckpt['dims']['action']}, env {self.dims['action']}")
        own = self.actor.state_dict()
        loaded, skipped = {}, []
        for name, tensor in ckpt["actor"].items():
            if name in own and own[name].shape == tensor.shape:
                loaded[name] = tensor
            else:
                skipped.append(name)
        self.actor.load_state_dict(loaded, strict=False)
        with torch.no_grad():
            for i in fresh_task_inputs:
                self.actor.task_norm.mean[i], self.actor.task_norm.var[i] = 0.0, 1.0
                self.actor.task_enc[0].weight[:, i] = 0.0
            for i in fresh_proprio_inputs:
                self.actor.proprio_norm.mean[i], self.actor.proprio_norm.var[i] = 0.0, 1.0
                self.actor.proprio_enc[0].weight[:, i] = 0.0
        # Proprio statistics come from DexGraspNet hand states; keep them but let fine-tuning keep updating them.
        return {"loaded": sorted(loaded), "skipped": sorted(skipped)}


def save_pretrained(actor: Actor, dims: dict[str, int], path: str) -> None:
    """Save an actor (e.g. trained by BC on DexGraspNet) in the format `ActorCritic.load_pretrained` expects."""
    torch.save({"format": PRETRAINED_FORMAT, "dims": dict(dims), "actor": actor.state_dict()}, path)
