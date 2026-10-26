import os

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from chopstick_skill3_env import ChopstickSkill3Env


TOTAL_TIMESTEPS = 40_000

NUM_FINAL_EVAL_EPISODES = 100

ZERO_ACTION_BASELINE = 31.0


MODEL_PATH = (
    "trained_models/"
    "ppo_chopstick_skill3"
)

BEST_MODEL_PATH = (
    "trained_models/"
    "ppo_chopstick_skill3_best"
)


class Skill3EvalCallback(BaseCallback):

    def __init__(
        self,
        eval_freq=5000,
        n_eval_episodes=50,
        verbose=1,
    ):

        super().__init__(
            verbose=verbose
        )

        self.eval_freq = eval_freq

        self.n_eval_episodes = (
            n_eval_episodes
        )

        self.eval_env = (
            ChopstickSkill3Env()
        )

        self.best_success_rate = -1.0

        self.best_mean_length = -1.0


    def _on_step(self):

        if (
            self.num_timesteps
            % self.eval_freq
            != 0
        ):

            return True

        successes = 0
        drops = 0

        lengths = []

        final_errors = []

        actions_used = []


        for episode in range(
            self.n_eval_episodes
        ):

            obs, reset_info = (
                self.eval_env.reset(
                    seed=7000 + episode
                )
            )

            final_info = None

            episode_actions = []


            for step in range(
                self.eval_env.max_steps
            ):

                action, _ = (
                    self.model.predict(
                        obs,
                        deterministic=True
                    )
                )

                episode_actions.append(
                    float(action[0])
                )

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    final_info,
                ) = self.eval_env.step(
                    action
                )

                if (
                    terminated
                    or truncated
                ):

                    break


            lengths.append(
                step + 1
            )

            final_errors.append(
                final_info[
                    "command_error_deg"
                ]
            )

            actions_used.extend(
                episode_actions
            )

            if final_info[
                "success"
            ]:

                successes += 1

            if final_info[
                "dropped"
            ]:

                drops += 1


        success_rate = (
            successes
            / self.n_eval_episodes
            * 100.0
        )

        mean_length = float(
            np.mean(lengths)
        )

        mean_final_error = float(
            np.mean(final_errors)
        )

        mean_abs_action = float(
            np.mean(
                np.abs(actions_used)
            )
        )


        print()
        print(
            "===================================="
        )
        print(
            "DETERMINISTIC PPO CHECK - SKILL 3 V4"
        )
        print(
            "===================================="
        )

        print(
            "Timesteps:",
            self.num_timesteps
        )

        print(
            f"Success: "
            f"{successes}/{self.n_eval_episodes}"
        )

        print(
            f"Success rate: "
            f"{success_rate:.1f}%"
        )

        print(
            "Zero-action baseline:",
            f"{ZERO_ACTION_BASELINE:.1f}%"
        )

        print(
            "Average episode length:",
            f"{mean_length:.1f}"
        )

        print(
            "Average final command error:",
            f"{mean_final_error:.4f} deg"
        )

        print(
            "Average absolute action:",
            f"{mean_abs_action:.3f}"
        )

        print(
            "Drops:",
            drops
        )


        better_model = False

        if (
            success_rate
            > self.best_success_rate
        ):

            better_model = True

        elif (
            success_rate
            == self.best_success_rate
            and
            mean_length
            > self.best_mean_length
        ):

            better_model = True


        if better_model:

            self.best_success_rate = (
                success_rate
            )

            self.best_mean_length = (
                mean_length
            )

            self.model.save(
                BEST_MODEL_PATH
            )

            print(
                "New best Skill 3 V4 model saved."
            )


        print(
            "===================================="
        )
        print()


        return True


    def _on_training_end(self):

        self.eval_env.close()


print()
print(
    "Creating Skill 3 V4 environment..."
)


env = ChopstickSkill3Env()


print(
    "Checking environment..."
)


check_env(
    env,
    warn=True,
)


print(
    "Environment check passed!"
)


os.makedirs(
    "trained_models",
    exist_ok=True,
)

os.makedirs(
    "logs/chopstick_skill3_v4",
    exist_ok=True,
)


policy_kwargs = {

    "log_std_init":
        -1.5,

    "net_arch":
        dict(
            pi=[
                64,
                64,
            ],
            vf=[
                64,
                64,
            ],
        ),
}


model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=3e-4,

    n_steps=1024,

    batch_size=64,

    gamma=0.99,

    gae_lambda=0.95,

    ent_coef=0.0,

    policy_kwargs=policy_kwargs,

    verbose=1,

    tensorboard_log=(
        "logs/chopstick_skill3_v4"
    ),

    device="cpu",

    seed=42,
)


evaluation_callback = (
    Skill3EvalCallback(
        eval_freq=5000,
        n_eval_episodes=50,
    )
)


print()
print(
    "===================================="
)
print(
    "TRAINING CHOPSTICK SKILL 3 V4"
)
print(
    "===================================="
)

print(
    "Task:"
)

print(
    "Recover the disturbed chopstick command "
    "and maintain a dynamic cube grasp for 1 second."
)

print()

print(
    "Zero-action baseline:",
    f"{ZERO_ACTION_BASELINE:.1f}%"
)

print(
    "Classical controller benchmark: 100.0%"
)

print()

print(
    "Stable grasp command:"
)

print(
    f"{env.grasp_command_deg if env.grasp_command_deg is not None else 'found on reset'}"
)

print(
    "Maximum action authority: "
    "0.25 deg/step"
)

print(
    "Setpoint-recovery reward: ON"
)

print(
    "Horizontal velocity disturbance: OFF"
)

print(
    "Low exploration: ON"
)

print(
    "Deterministic evaluation: ON"
)

print(
    "Cached grasp initialization: ON"
)


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=evaluation_callback,
)


model.save(
    MODEL_PATH
)


print()
print(
    "===================================="
)
print(
    "TRAINING COMPLETE"
)
print(
    "===================================="
)

print(
    "Final model:",
    MODEL_PATH + ".zip",
)


if os.path.exists(
    BEST_MODEL_PATH + ".zip"
):

    evaluation_model_path = (
        BEST_MODEL_PATH
    )

    print(
        "Evaluating best checkpoint:",
        BEST_MODEL_PATH + ".zip",
    )

else:

    evaluation_model_path = (
        MODEL_PATH
    )

    print(
        "Evaluating final model."
    )


env.close()


print()
print(
    "===================================="
)
print(
    "FINAL SKILL 3 V4 EVALUATION"
)
print(
    "===================================="
)


eval_env = (
    ChopstickSkill3Env()
)


trained_model = PPO.load(
    evaluation_model_path
)


successes = 0
drops = 0

episode_lengths = []

maximum_displacements = []
maximum_vertical_drops = []

final_commands = []
final_hinges = []
final_errors = []

first_actions = []


for episode in range(
    NUM_FINAL_EVAL_EPISODES
):

    obs, reset_info = (
        eval_env.reset(
            seed=10000 + episode
        )
    )

    episode_max_displacement = 0.0
    episode_max_drop = 0.0

    final_info = None

    episode_first_action = None


    for step in range(
        eval_env.max_steps
    ):

        action, _ = (
            trained_model.predict(
                obs,
                deterministic=True,
            )
        )

        if step == 0:

            episode_first_action = (
                float(action[0])
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            final_info,
        ) = eval_env.step(
            action
        )

        episode_max_displacement = max(
            episode_max_displacement,
            final_info[
                "displacement_mm"
            ],
        )

        episode_max_drop = max(
            episode_max_drop,
            final_info[
                "vertical_drop_mm"
            ],
        )

        if (
            terminated
            or truncated
        ):

            break


    episode_lengths.append(
        step + 1
    )

    maximum_displacements.append(
        episode_max_displacement
    )

    maximum_vertical_drops.append(
        episode_max_drop
    )

    final_commands.append(
        final_info[
            "command_deg"
        ]
    )

    final_hinges.append(
        final_info[
            "hinge_deg"
        ]
    )

    final_errors.append(
        final_info[
            "command_error_deg"
        ]
    )

    first_actions.append(
        episode_first_action
    )


    if final_info[
        "success"
    ]:

        successes += 1


    if final_info[
        "dropped"
    ]:

        drops += 1


    if episode < 10:

        print()

        print(
            f"Episode {episode + 1}"
        )

        print(
            "Initial command disturbance:",
            f"{reset_info['command_disturbance_deg']:+.3f} deg",
        )

        print(
            "First PPO action:",
            f"{episode_first_action:+.3f}",
        )

        print(
            "First command correction:",
            f"{episode_first_action * eval_env.max_delta_deg:+.3f} deg",
        )

        print(
            "Final command:",
            f"{final_info['command_deg']:.3f} deg",
        )

        print(
            "Final command error:",
            f"{final_info['command_error_deg']:.4f} deg",
        )

        print(
            "Maximum displacement:",
            f"{episode_max_displacement:.3f} mm",
        )

        print(
            "Maximum vertical drop:",
            f"{episode_max_drop:.3f} mm",
        )

        print(
            "Steps:",
            step + 1,
        )

        print(
            "Success:",
            final_info[
                "success"
            ],
        )

        print(
            "Dropped:",
            final_info[
                "dropped"
            ],
        )


success_rate = (
    successes
    / NUM_FINAL_EVAL_EPISODES
    * 100.0
)

improvement = (
    success_rate
    - ZERO_ACTION_BASELINE
)


print()
print(
    "===================================="
)
print(
    "FINAL SKILL 3 V4 RESULTS"
)
print(
    "===================================="
)

print(
    f"Successes: "
    f"{successes}/{NUM_FINAL_EVAL_EPISODES}"
)

print(
    f"Success rate: "
    f"{success_rate:.1f}%"
)

print(
    f"Zero-action baseline: "
    f"{ZERO_ACTION_BASELINE:.1f}%"
)

print(
    f"PPO improvement: "
    f"{improvement:+.1f} percentage points"
)

print(
    "Classical controller benchmark: "
    "100.0%"
)

print(
    f"Drops: "
    f"{drops}/{NUM_FINAL_EVAL_EPISODES}"
)

print(
    "Average episode length:",
    f"{np.mean(episode_lengths):.1f}",
)

print(
    "Average max displacement:",
    f"{np.mean(maximum_displacements):.3f} mm",
)

print(
    "Average max vertical drop:",
    f"{np.mean(maximum_vertical_drops):.3f} mm",
)

print(
    "Average final hinge:",
    f"{np.mean(final_hinges):.3f} deg",
)

print(
    "Average final command:",
    f"{np.mean(final_commands):.3f} deg",
)

print(
    "Average final command error:",
    f"{np.mean(final_errors):.4f} deg",
)

print(
    "Average first PPO action:",
    f"{np.mean(first_actions):+.3f}",
)


print()
print(
    "===================================="
)
print(
    "VERDICT"
)
print(
    "===================================="
)


if success_rate >= 90.0:

    print(
        "PASS: Skill 3 learned reliable "
        "dynamic grasp recovery."
    )

    print(
        "Ready to proceed toward Skill 4."
    )


elif success_rate >= 70.0:

    print(
        "PARTIAL PASS: PPO learned strong "
        "grasp recovery."
    )

    print(
        "Improve reliability before Skill 4."
    )


elif success_rate > ZERO_ACTION_BASELINE:

    print(
        "LEARNING DETECTED: PPO outperformed "
        "the zero-action baseline."
    )

    print(
        "More tuning is required."
    )


else:

    print(
        "FAIL: PPO did not outperform "
        "the zero-action baseline."
    )


eval_env.close()