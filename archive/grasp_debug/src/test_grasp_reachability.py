import numpy as np

from chopstick_grasp_env import ChopstickGraspEnv


# ==================================================
# SETTINGS
# ==================================================

ACTION_VALUES = np.linspace(
    -1.0,
    1.0,
    17
)

STEPS_PER_TEST = 150


# ==================================================
# CREATE ENVIRONMENT ONCE
# ==================================================

env = ChopstickGraspEnv()


# ==================================================
# RESULTS STORAGE
# ==================================================

results = []

any_upper_contact = False
any_both_contact = False


print("\n====================================")
print("DETAILED GRASP REACHABILITY TEST")
print("====================================")

print(
    f"\nTesting {len(ACTION_VALUES) ** 2} action combinations..."
)


# ==================================================
# TEST ACTION GRID
# ==================================================

for action_0 in ACTION_VALUES:

    for action_1 in ACTION_VALUES:

        obs, reset_info = env.reset()

        action = np.array(
            [
                action_0,
                action_1
            ],
            dtype=np.float32
        )


        # --------------------------------------------------
        # TRACK THIS TRIAL
        # --------------------------------------------------

        lower_contact_ever = False
        upper_contact_ever = False
        both_contact_ever = False

        consecutive_both = 0
        max_consecutive_both = 0

        min_upper_distance = float("inf")
        min_lower_distance = float("inf")

        # Most important metric:
        # upper distance WHILE lower contact is maintained
        min_upper_while_lower_contact = float("inf")

        hinge_angles = []


        # --------------------------------------------------
        # RUN CONSTANT ACTION
        # --------------------------------------------------

        for step in range(STEPS_PER_TEST):

            (
                obs,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)


            lower_distance = info[
                "lower_distance_mm"
            ]

            upper_distance = info[
                "upper_distance_mm"
            ]


            min_lower_distance = min(
                min_lower_distance,
                lower_distance
            )

            min_upper_distance = min(
                min_upper_distance,
                upper_distance
            )


            # ----------------------------------------------
            # CONTACTS
            # ----------------------------------------------

            if info["lower_contact"]:

                lower_contact_ever = True

                min_upper_while_lower_contact = min(
                    min_upper_while_lower_contact,
                    upper_distance
                )


            if info["upper_contact"]:

                upper_contact_ever = True

                any_upper_contact = True


            if info["both_contact"]:

                both_contact_ever = True

                any_both_contact = True

                consecutive_both += 1

                max_consecutive_both = max(
                    max_consecutive_both,
                    consecutive_both
                )

            else:

                consecutive_both = 0


            # ----------------------------------------------
            # HINGE MOVEMENT
            # ----------------------------------------------

            hinge_angle = env.data.qpos[
                env.hinge_qpos
            ]

            hinge_angles.append(
                np.degrees(
                    hinge_angle
                )
            )


            if terminated or truncated:
                break


        # --------------------------------------------------
        # IF LOWER CONTACT NEVER OCCURRED
        # --------------------------------------------------

        if np.isinf(
            min_upper_while_lower_contact
        ):

            min_upper_while_lower_contact = (
                float("inf")
            )


        hinge_min = min(
            hinge_angles
        )

        hinge_max = max(
            hinge_angles
        )

        hinge_range = (
            hinge_max
            - hinge_min
        )


        results.append(
            {
                "action_0": float(action_0),
                "action_1": float(action_1),

                "lower_contact":
                    lower_contact_ever,

                "upper_contact":
                    upper_contact_ever,

                "both_contact":
                    both_contact_ever,

                "max_consecutive_both":
                    max_consecutive_both,

                "min_lower_distance":
                    min_lower_distance,

                "min_upper_distance":
                    min_upper_distance,

                "min_upper_while_lower":
                    min_upper_while_lower_contact,

                "hinge_min":
                    hinge_min,

                "hinge_max":
                    hinge_max,

                "hinge_range":
                    hinge_range
            }
        )


# ==================================================
# SORT RESULTS
#
# Priority:
#
# 1. Both contact
# 2. Upper contact
# 3. Maintain lower contact
# 4. Minimize upper distance
# ==================================================

def ranking(result):

    return (
        not result["both_contact"],
        not result["upper_contact"],
        not result["lower_contact"],
        result["min_upper_while_lower"]
    )


results.sort(
    key=ranking
)


# ==================================================
# PRINT TOP 10
# ==================================================

print("\n====================================")
print("TOP 10 ACTIONS")
print("====================================")


for i, result in enumerate(
    results[:10],
    start=1
):

    print(
        f"\n#{i}"
    )

    print(
        "Action:",
        (
            round(result["action_0"], 3),
            round(result["action_1"], 3)
        )
    )

    print(
        "Lower contact:",
        result["lower_contact"]
    )

    print(
        "Upper contact:",
        result["upper_contact"]
    )

    print(
        "Both contact:",
        result["both_contact"]
    )

    print(
        "Max consecutive both:",
        result["max_consecutive_both"]
    )

    print(
        "Min lower distance:",
        f"{result['min_lower_distance']:.3f} mm"
    )

    print(
        "Min upper distance:",
        f"{result['min_upper_distance']:.3f} mm"
    )

    print(
        "Min upper while lower touching:",
        (
            f"{result['min_upper_while_lower']:.3f} mm"
            if np.isfinite(
                result["min_upper_while_lower"]
            )
            else "N/A"
        )
    )

    print(
        "Upper hinge range:",
        f"{result['hinge_range']:.3f} deg"
    )

    print(
        "Upper hinge min/max:",
        f"{result['hinge_min']:.3f} / "
        f"{result['hinge_max']:.3f} deg"
    )


# ==================================================
# OVERALL RESULT
# ==================================================

best = results[0]


print("\n====================================")
print("OVERALL DIAGNOSIS")
print("====================================")

print(
    "Any upper contact found:",
    any_upper_contact
)

print(
    "Any simultaneous contact found:",
    any_both_contact
)

print(
    "\nBest action:",
    (
        best["action_0"],
        best["action_1"]
    )
)

print(
    "Best upper distance while lower touching:",
    (
        f"{best['min_upper_while_lower']:.3f} mm"
        if np.isfinite(
            best["min_upper_while_lower"]
        )
        else "N/A"
    )
)

print(
    "Upper hinge motion for best action:",
    f"{best['hinge_range']:.3f} deg"
)


# ==================================================
# INTERPRETATION
# ==================================================

print("\n====================================")
print("INTERPRETATION")
print("====================================")


if any_both_contact:

    print(
        "SUCCESS: the current 2D action space can "
        "physically create a two-chopstick pinch."
    )

    print(
        "We can move back to PPO training."
    )


elif any_upper_contact:

    print(
        "PARTIAL SUCCESS: the upper chopstick can "
        "reach the object, but simultaneous contact "
        "is not yet being maintained."
    )

    print(
        "The next step is reward/control tuning, "
        "not geometry redesign."
    )


else:

    print(
        "NO UPPER CONTACT FOUND."
    )

    print(
        "Use the hinge-motion result to determine "
        "the next change."
    )

    print(
        "If hinge motion is substantial, the issue "
        "is object/chopstick geometry."
    )

    print(
        "If hinge motion is tiny, FFJ3 + FFJ0 are "
        "not providing enough control and we should "
        "add another finger actuator."
    )