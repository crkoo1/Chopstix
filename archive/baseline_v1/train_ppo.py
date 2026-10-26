from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from chopstick_rl_env import ChopstickEnv


print("\nCreating environment...")

env = ChopstickEnv()


# Make sure our Gym environment is valid
print("Checking environment...")

check_env(env)

print("Environment check passed!")


# ------------------------------------------
# Create PPO
# ------------------------------------------

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.98,
    verbose=1,
    device="cpu"
)


print("\n====================================")
print("FIRST PPO TRAINING RUN")
print("====================================")


# Only 10,000 for our first sanity check
model.learn(
    total_timesteps=10_000
)


# Save it
model.save(
    "ppo_first_chopstick_model"
)


print("\n====================================")
print("FIRST TRAINING COMPLETE")
print("====================================")

print("Saved: ppo_first_chopstick_model.zip")