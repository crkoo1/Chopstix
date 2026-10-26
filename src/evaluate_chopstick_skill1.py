import numpy as np

from stable_baselines3 import PPO

from chopstick_skill1_env import ChopstickSkill1Env


NUM_EPISODES = 100

MODEL_PATH = "trained_models/ppo_chopstick_skill1"


env = ChopstickSkill1Env()

model = PPO.load(
    MODEL_PATH
)


successes = 0

final_errors = []
episode_steps = []

successful_errors = []
successful_steps = []


print("\n====================================")
print("EVALUATING CHOPSTICK SKILL 1")
print("====================================")


for episode in range(NUM_EPISODES):

    obs, info = env.reset(
        seed=1000 + episode
    )

    start_deg = info["start_deg"]
    target_deg = info["target_deg"]

    final_info = None


    for step in range(env.max_steps):

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            final_info
        ) = env.step(action)


        if terminated or truncated:
            break


    steps_taken = step + 1

    final_error = abs(
        final_info["error_deg"]
    )

    success = final_info["success"]


    final_errors.append(
        final_error
    )

    episode_steps.append(
        steps_taken
    )


    if success:

        successes += 1

        successful_errors.append(
            final_error
        )

        successful_steps.append(
            steps_taken
        )


    # Only show first 10 episodes
    if episode < 10:

        print(
            f"\nEpisode {episode + 1}"
        )

        print(
            f"Start:  {start_deg:.3f} deg"
        )

        print(
            f"Target: {target_deg:.3f} deg"
        )

        print(
            "Final:",
            f"{final_info['current_deg']:.3f} deg"
        )

        print(
            "Error:",
            f"{final_error:.3f} deg"
        )

        print(
            "Steps:",
            steps_taken
        )

        print(
            "Success:",
            success
        )


success_rate = (
    successes
    / NUM_EPISODES
    * 100.0
)


print("\n====================================")
print("FINAL EVALUATION")
print("====================================")

print(
    f"Successes: "
    f"{successes}/{NUM_EPISODES}"
)

print(
    f"Success rate: "
    f"{success_rate:.1f}%"
)


print("\nERROR")

print(
    "Average final error:",
    f"{np.mean(final_errors):.3f} deg"
)

print(
    "Median final error:",
    f"{np.median(final_errors):.3f} deg"
)

print(
    "Worst final error:",
    f"{np.max(final_errors):.3f} deg"
)


print("\nEPISODE LENGTH")

print(
    "Average steps:",
    f"{np.mean(episode_steps):.1f}"
)


if successes > 0:

    print(
        "Average successful steps:",
        f"{np.mean(successful_steps):.1f}"
    )

    print(
        "Average successful error:",
        f"{np.mean(successful_errors):.3f} deg"
    )


print("\n====================================")
print("VERDICT")
print("====================================")


if success_rate >= 90:

    print(
        "PASS: Skill 1 learned reliably."
    )

    print(
        "Ready for Skill 2: cube grasping."
    )


elif success_rate >= 70:

    print(
        "PARTIAL PASS: PPO learned the task "
        "but needs more improvement."
    )


else:

    print(
        "FAIL: PPO is not reliable enough yet."
    )


env.close()