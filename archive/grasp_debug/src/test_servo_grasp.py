import numpy as np

from chopstick_servo_env import ChopstickServoEnv


env = ChopstickServoEnv()


TARGETS = [
    10.75,
    10.25,
    10.00,
    9.75,
    9.50,
    9.25,
    9.00,
    8.75,
    8.50,
    8.25,
    8.00
]


def target_to_action(target_deg):

    minimum = env.min_target_deg
    maximum = env.max_target_deg

    normalized = (
        2.0
        * (
            target_deg
            - minimum
        )
        / (
            maximum
            - minimum
        )
        - 1.0
    )

    return np.array(
        [normalized],
        dtype=np.float32
    )


print("\n====================================")
print("DYNAMIC SERVO GRASP TEST")
print("====================================")


for target in TARGETS:

    obs, info = env.reset()

    action = target_to_action(
        target
    )


    lower_ever = False
    upper_ever = False
    both_ever = False

    final_info = None


    for step in range(100):

        (
            obs,
            reward,
            terminated,
            truncated,
            final_info
        ) = env.step(
            action
        )


        lower_ever |= (
            final_info["lower_contact"]
        )

        upper_ever |= (
            final_info["upper_contact"]
        )

        both_ever |= (
            final_info["both_contact"]
        )


        if terminated:
            break


    print(
        f"\nTarget: {target:.2f} deg"
    )

    print(
        "Actual hinge:",
        f"{final_info['hinge_deg']:.3f} deg"
    )

    print(
        "Lower distance:",
        f"{final_info['lower_distance_mm']:.3f} mm"
    )

    print(
        "Upper distance:",
        f"{final_info['upper_distance_mm']:.3f} mm"
    )

    print(
        "Lower contact:",
        lower_ever
    )

    print(
        "Upper contact:",
        upper_ever
    )

    print(
        "Both contact:",
        both_ever
    )

    print(
        "Success:",
        final_info["success"]
    )


env.close()