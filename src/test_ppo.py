from stable_baselines3 import PPO
import numpy as np

from chopstick_rl_env import ChopstickEnv


env = ChopstickEnv()

model = PPO.load(
    "ppo_first_chopstick_model"
)

NUM_EPISODES = 20

successes = 0
episode_lengths = []
final_angles = []

print("\n====================================")
print("TESTING TRAINED PPO POLICY")
print("====================================")


for episode in range(NUM_EPISODES):

    obs, info = env.reset()

    terminated = False
    truncated = False

    steps = 0

    while not terminated and not truncated:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        steps += 1

    success = info["success"]

    if success:
        successes += 1

    episode_lengths.append(steps)
    final_angles.append(
        info["hinge_angle_deg"]
    )

    print(
        f"Episode {episode + 1:02d} | "
        f"Success: {success} | "
        f"Steps: {steps} | "
        f"Final angle: {info['hinge_angle_deg']:.2f}° | "
        f"Error: {info['error_deg']:.2f}°"
    )


success_rate = (
    successes / NUM_EPISODES * 100
)

print("\n====================================")
print("RESULTS")
print("====================================")

print(
    f"Success rate: "
    f"{successes}/{NUM_EPISODES} "
    f"({success_rate:.1f}%)"
)

print(
    f"Average episode length: "
    f"{np.mean(episode_lengths):.1f} steps"
)

print(
    f"Average final hinge angle: "
    f"{np.mean(final_angles):.2f}°"
)

print("Target hinge angle: 2.00°")