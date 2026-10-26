import mujoco
import numpy as np

from chopstick_grasp_env import ChopstickGraspEnv


# ==================================================
# SETTINGS
# ==================================================

# Test upper-chopstick hinge positions from
# -20 degrees to +20 degrees.
HINGE_ANGLES_DEG = np.linspace(
    -20.0,
    20.0,
    161
)


# ==================================================
# ENVIRONMENT
# ==================================================

env = ChopstickGraspEnv()


# ==================================================
# HELPERS
# ==================================================

def contact_flags():

    lower_contact = False
    upper_contact = False

    for i in range(env.data.ncon):

        contact = env.data.contact[i]

        body1_id = env.model.geom_bodyid[
            contact.geom1
        ]

        body2_id = env.model.geom_bodyid[
            contact.geom2
        ]

        body1 = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body1_id
        )

        body2 = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body2_id
        )

        bodies = {
            body1,
            body2
        }

        if (
            "lower_chopstick" in bodies
            and
            "grasp_object" in bodies
        ):
            lower_contact = True

        if (
            "upper_chopstick" in bodies
            and
            "grasp_object" in bodies
        ):
            upper_contact = True

    return (
        lower_contact,
        upper_contact
    )


# ==================================================
# INITIAL RESET
# ==================================================

obs, reset_info = env.reset()


print("\n====================================")
print("DIRECT HINGE REACHABILITY TEST")
print("====================================")


print(
    "\nInitial lower distance:",
    f"{reset_info['lower_distance_mm']:.3f} mm"
)

print(
    "Initial upper distance:",
    f"{reset_info['upper_distance_mm']:.3f} mm"
)


# ==================================================
# SAVE CLEAN RESET STATE
# ==================================================

initial_qpos = env.data.qpos.copy()
initial_qvel = env.data.qvel.copy()
initial_ctrl = env.data.ctrl.copy()


# ==================================================
# RESULTS
# ==================================================

results = []

contact_angles = []


# ==================================================
# SWEEP DIRECT HINGE POSITION
# ==================================================

for angle_deg in HINGE_ANGLES_DEG:

    # Restore exact same starting state
    env.data.qpos[:] = initial_qpos
    env.data.qvel[:] = initial_qvel
    env.data.ctrl[:] = initial_ctrl


    # Convert degrees to radians
    angle_rad = np.radians(
        angle_deg
    )


    # Directly set upper chopstick hinge
    env.data.qpos[
        env.hinge_qpos
    ] = angle_rad

    env.data.qvel[
        env.hinge_qvel
    ] = 0.0


    # Recalculate geometry only.
    # No dynamics yet.
    mujoco.mj_forward(
        env.model,
        env.data
    )


    # ==================================================
    # POSITIONS
    # ==================================================

    lower_tip = env.data.site_xpos[
        env.lower_tip_id
    ].copy()

    upper_tip = env.data.site_xpos[
        env.upper_tip_id
    ].copy()

    object_pos = env.data.xpos[
        env.object_id
    ].copy()


    lower_distance = np.linalg.norm(
        lower_tip
        - object_pos
    )

    upper_distance = np.linalg.norm(
        upper_tip
        - object_pos
    )


    # ==================================================
    # CONTACT
    # ==================================================

    (
        lower_contact,
        upper_contact
    ) = contact_flags()


    both_contact = (
        lower_contact
        and upper_contact
    )


    result = {
        "angle_deg":
            float(angle_deg),

        "lower_distance_mm":
            float(
                lower_distance
                * 1000
            ),

        "upper_distance_mm":
            float(
                upper_distance
                * 1000
            ),

        "lower_contact":
            bool(lower_contact),

        "upper_contact":
            bool(upper_contact),

        "both_contact":
            bool(both_contact)
    }

    results.append(
        result
    )


    if upper_contact:

        contact_angles.append(
            result
        )


# ==================================================
# BEST GEOMETRIC RESULT
# ==================================================

best = min(
    results,
    key=lambda x:
        x["upper_distance_mm"]
)


print("\n====================================")
print("CLOSEST UPPER CHOPSTICK POSITION")
print("====================================")


print(
    "Hinge angle:",
    f"{best['angle_deg']:.3f} deg"
)

print(
    "Lower distance:",
    f"{best['lower_distance_mm']:.3f} mm"
)

print(
    "Upper distance:",
    f"{best['upper_distance_mm']:.3f} mm"
)

print(
    "Lower contact:",
    best["lower_contact"]
)

print(
    "Upper contact:",
    best["upper_contact"]
)

print(
    "Both contact:",
    best["both_contact"]
)


# ==================================================
# CONTACT RANGE
# ==================================================

print("\n====================================")
print("UPPER CONTACT RANGE")
print("====================================")


if len(contact_angles) == 0:

    print(
        "Upper chopstick NEVER contacted "
        "the object anywhere from -20 to +20 deg."
    )

else:

    first = contact_angles[0]

    last = contact_angles[-1]


    print(
        "Upper contact first appears at:",
        f"{first['angle_deg']:.3f} deg"
    )

    print(
        "Upper contact last appears at:",
        f"{last['angle_deg']:.3f} deg"
    )


    simultaneous = [
        result
        for result in contact_angles
        if result["both_contact"]
    ]


    if len(simultaneous) > 0:

        print(
            "\nSIMULTANEOUS CONTACT IS "
            "GEOMETRICALLY POSSIBLE."
        )

        print(
            "First simultaneous-contact angle:",
            f"{simultaneous[0]['angle_deg']:.3f} deg"
        )

        print(
            "Last simultaneous-contact angle:",
            f"{simultaneous[-1]['angle_deg']:.3f} deg"
        )

    else:

        print(
            "\nUpper contact occurs, but "
            "simultaneous lower + upper "
            "contact was not found."
        )


# ==================================================
# DIAGNOSIS
# ==================================================

print("\n====================================")
print("DIAGNOSIS")
print("====================================")


simultaneous = [
    result
    for result in results
    if result["both_contact"]
]


if len(simultaneous) > 0:

    print(
        "GOOD: The current chopstick/object geometry "
        "CAN produce a pinch."
    )

    print(
        "Therefore the remaining problem is the "
        "finger-to-chopstick mechanical coupling."
    )

    print(
        "Do NOT move the ball again."
    )

else:

    print(
        "The direct hinge sweep could not produce "
        "simultaneous contact."
    )

    print(
        "Therefore the geometry still needs adjustment "
        "before working on finger control."
    )