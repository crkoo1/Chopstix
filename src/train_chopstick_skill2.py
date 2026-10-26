import os

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import (
    check_env
)

from chopstick_skill2_env import (
    ChopstickSkill2Env
)


# ==========================================================
# SETTINGS
# ==========================================================

TOTAL_TIMESTEPS = 50_000

NUM_EVAL_EPISODES = 100


MODEL_PATH = (
    "trained_models/"
    "ppo_chopstick_skill2"
)


# ==========================================================
# CREATE ENVIRONMENT
# ==========================================================

print(
    "\nCreating Skill 2 environment..."
)


env = ChopstickSkill2Env()


# ==========================================================
# CHECK ENVIRONMENT
# ==========================================================

print(
    "Checking environment..."
)


check_env(
    env,
    warn=True
)


print(
    "Environment check passed!"
)


# ==========================================================
# OUTPUT FOLDERS
# ==========================================================

os.makedirs(
    "trained_models",
    exist_ok=True
)


os.makedirs(
    "logs/chopstick_skill2",
    exist_ok=True
)


# ==========================================================
# PPO MODEL
# ==========================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=3e-4,

    n_steps=512,

    batch_size=64,

    gamma=0.98,

    # Encourage exploration.
    ent_coef=0.01,

    verbose=1,

    tensorboard_log=(
        "logs/chopstick_skill2"
    ),

    device="cpu"
)


# ==========================================================
# TRAIN
# ==========================================================

print(
    "\n===================================="
)

print(
    "TRAINING CHOPSTICK SKILL 2"
)

print(
    "===================================="
)


model.learn(
    total_timesteps=TOTAL_TIMESTEPS
)


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    MODEL_PATH
)


print(
    "\n===================================="
)

print(
    "TRAINING COMPLETE"
)

print(
    "===================================="
)


print(
    "Saved:",
    MODEL_PATH + ".zip"
)


env.close()


# ==========================================================
# EVALUATION
# ==========================================================

print(
    "\n===================================="
)

print(
    "EVALUATING CHOPSTICK SKILL 2"
)

print(
    "===================================="
)


eval_env = (
    ChopstickSkill2Env()
)


trained_model = PPO.load(
    MODEL_PATH
)


successes = 0


episode_lengths = []

final_angles = []

final_commands = []

final_grasp_errors = []

max_hold_steps = []


# ==========================================================
# RUN EVALUATION
# ==========================================================

for episode in range(
    NUM_EVAL_EPISODES
):

    obs, info = (
        eval_env.reset(
            seed=2000 + episode
        )
    )


    episode_max_hold = 0

    final_info = None


    for step in range(
        eval_env.max_steps
    ):

        action, _ = (
            trained_model.predict(
                obs,
                deterministic=True
            )
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            final_info
        ) = eval_env.step(
            action
        )


        episode_max_hold = max(
            episode_max_hold,
            final_info[
                "hold_steps"
            ]
        )


        if (
            terminated
            or truncated
        ):

            break


    steps_taken = (
        step + 1
    )


    episode_lengths.append(
        steps_taken
    )


    final_angles.append(
        final_info[
            "hinge_deg"
        ]
    )


    final_commands.append(
        final_info[
            "command_deg"
        ]
    )


    final_grasp_errors.append(
        final_info[
            "grasp_error_deg"
        ]
    )


    max_hold_steps.append(
        episode_max_hold
    )


    if final_info[
        "success"
    ]:

        successes += 1


    # ======================================================
    # PRINT FIRST 10 EPISODES
    # ======================================================

    if episode < 10:

        print(
            f"\nEpisode {episode + 1}"
        )


        print(
            "Final hinge:",
            f"{final_info['hinge_deg']:.3f} deg"
        )


        print(
            "Command:",
            f"{final_info['command_deg']:.3f} deg"
        )


        print(
            "Grasp-center error:",
            f"{final_info['grasp_error_deg']:.3f} deg"
        )


        print(
            "Lower contact:",
            final_info[
                "lower_contact"
            ]
        )


        print(
            "Upper contact:",
            final_info[
                "upper_contact"
            ]
        )


        print(
            "Both contact:",
            final_info[
                "both_contact"
            ]
        )


        print(
            "Max hold:",
            episode_max_hold
        )


        print(
            "Steps:",
            steps_taken
        )


        print(
            "Success:",
            final_info[
                "success"
            ]
        )


# ==========================================================
# FINAL STATISTICS
# ==========================================================

success_rate = (
    successes
    / NUM_EVAL_EPISODES
    * 100.0
)


print(
    "\n===================================="
)

print(
    "FINAL SKILL 2 EVALUATION"
)

print(
    "===================================="
)


print(
    f"Successes: "
    f"{successes}/{NUM_EVAL_EPISODES}"
)


print(
    f"Success rate: "
    f"{success_rate:.1f}%"
)


print(
    "Average episode length:",
    f"{np.mean(episode_lengths):.1f}"
)


print(
    "Average final hinge angle:",
    f"{np.mean(final_angles):.3f} deg"
)


print(
    "Average final command:",
    f"{np.mean(final_commands):.3f} deg"
)


print(
    "Average grasp-center error:",
    f"{np.mean(final_grasp_errors):.3f} deg"
)


print(
    "Average maximum hold:",
    f"{np.mean(max_hold_steps):.1f} steps"
)


# ==========================================================
# VERDICT
# ==========================================================

print(
    "\n===================================="
)

print(
    "VERDICT"
)

print(
    "===================================="
)


if success_rate >= 90.0:

    print(
        "PASS: Skill 2 learned reliable "
        "two-sided cube grasping."
    )

    print(
        "Ready for Skill 3: "
        "dynamic cube and lifting."
    )


elif success_rate >= 70.0:

    print(
        "PARTIAL PASS: Skill 2 was learned "
        "but needs improved reliability."
    )


else:

    print(
        "FAIL: Skill 2 is still not "
        "reliable enough."
    )


eval_env.close()