import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from chopstick_grasp_env import ChopstickGraspEnv


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

print("\nCreating grasp environment...")

env = ChopstickGraspEnv()


print("Checking environment...")

check_env(
    env,
    warn=True
)

print("Environment check passed!")


# ==================================================
# OUTPUT
# ==================================================

os.makedirs(
    "trained_models",
    exist_ok=True
)

os.makedirs(
    "logs/grasp_stage1",
    exist_ok=True
)


# ==================================================
# PPO
# ==================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=3e-4,

    n_steps=512,

    batch_size=64,

    gamma=0.98,

    verbose=1,

    tensorboard_log="logs/grasp_stage1",

    device="cpu"
)


# ==================================================
# TRAIN
# ==================================================

print("\n====================================")
print("STAGE 1: PPO CHOPSTICK GRASP")
print("====================================")


model.learn(
    total_timesteps=20_000
)


# ==================================================
# SAVE
# ==================================================

model.save(
    "trained_models/ppo_grasp_stage1"
)


print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print(
    "Saved:",
    "trained_models/ppo_grasp_stage1.zip"
)