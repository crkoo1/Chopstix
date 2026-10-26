import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from chopstick_skill1_env import ChopstickSkill1Env


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

print("\nCreating Skill 1 environment...")

env = ChopstickSkill1Env()


# ==================================================
# VERIFY ENVIRONMENT
# ==================================================

print("Checking Skill 1 environment...")

check_env(
    env,
    warn=True
)

print("Environment check passed!")


# ==================================================
# OUTPUT DIRECTORIES
# ==================================================

os.makedirs(
    "trained_models",
    exist_ok=True
)

os.makedirs(
    "logs/chopstick_skill1",
    exist_ok=True
)


# ==================================================
# PPO MODEL
# ==================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=3e-4,

    n_steps=512,

    batch_size=64,

    gamma=0.98,

    verbose=1,

    tensorboard_log="logs/chopstick_skill1",

    device="cpu"
)


# ==================================================
# TRAINING
# ==================================================

print("\n====================================")
print("TRAINING CHOPSTICK SKILL 1")
print("====================================")


model.learn(
    total_timesteps=30_000
)


# ==================================================
# SAVE MODEL
# ==================================================

model.save(
    "trained_models/ppo_chopstick_skill1"
)


print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print(
    "Saved:"
)

print(
    "trained_models/ppo_chopstick_skill1.zip"
)


env.close()