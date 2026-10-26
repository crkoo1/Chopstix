import numpy as np

from chopstick_skill1_env import ChopstickSkill1Env


env = ChopstickSkill1Env()


TOTAL_EPISODES = 10

successes = 0


print("\n====================================")
print("SKILL 1 SANITY TEST")
print("====================================")


for episode in range(
    TOTAL_EPISODES
):

    obs, info = env.reset(
        seed=100 + episode
    )


    start = info[
        "start_deg"
    ]

    target = info[
        "target_deg"
    ]


    final_info = None


    for step in range(
        env.max_steps
    ):

        current = env._hinge_deg()

        error = (
            target
            - current
        )


        # Simple proportional-style policy
        action_value = np.clip(
            error / 2.0,
            -1.0,
            1.0
        )


        action = np.array(
            [action_value],
            dtype=np.float32
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            final_info
        ) = env.step(
            action
        )


        if terminated or truncated:
            break


    if final_info[
        "success"
    ]:

        successes += 1


    print(
        f"\nEpisode {episode + 1}"
    )

    print(
        f"Start:  {start:.2f} deg"
    )

    print(
        f"Target: {target:.2f} deg"
    )

    print(
        "Final:",
        f"{final_info['current_deg']:.2f} deg"
    )

    print(
        "Error:",
        f"{abs(final_info['error_deg']):.3f} deg"
    )

    print(
        "Steps:",
        step + 1
    )

    print(
        "Success:",
        final_info["success"]
    )


print("\n====================================")
print("SUMMARY")
print("====================================")

print(
    f"Successes: "
    f"{successes}/{TOTAL_EPISODES}"
)