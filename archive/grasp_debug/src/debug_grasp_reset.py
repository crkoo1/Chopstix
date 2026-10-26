import numpy as np

from chopstick_grasp_env import ChopstickGraspEnv


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

env = ChopstickGraspEnv()


# ==================================================
# RESET
# ==================================================

obs, info = env.reset()


# ==================================================
# GET POSITIONS IMMEDIATELY AFTER RESET
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


midpoint = (
    lower_tip
    + upper_tip
) / 2.0


# ==================================================
# DISTANCES
# ==================================================

lower_distance = np.linalg.norm(
    lower_tip
    - object_pos
)

upper_distance = np.linalg.norm(
    upper_tip
    - object_pos
)

midpoint_distance = np.linalg.norm(
    midpoint
    - object_pos
)

tip_separation = np.linalg.norm(
    lower_tip
    - upper_tip
)


# ==================================================
# CONTACTS
# ==================================================

lower_contact, upper_contact = (
    env._contact_flags()
)


# ==================================================
# PRINT RESET STATE
# ==================================================

print("\n====================================")
print("GRASP ENV RESET DEBUG")
print("====================================")


print("\nLOWER TIP")
print(lower_tip)


print("\nUPPER TIP")
print(upper_tip)


print("\nMIDPOINT")
print(midpoint)


print("\nOBJECT")
print(object_pos)


print("\n====================================")
print("DISTANCES")
print("====================================")


print(
    "Tip separation:",
    f"{tip_separation * 1000:.2f} mm"
)

print(
    "Lower tip -> object:",
    f"{lower_distance * 1000:.2f} mm"
)

print(
    "Upper tip -> object:",
    f"{upper_distance * 1000:.2f} mm"
)

print(
    "Midpoint -> object:",
    f"{midpoint_distance * 1000:.2f} mm"
)


print("\n====================================")
print("OBJECT JOINT")
print("====================================")


print(
    "Object qpos:",
    env.data.qpos[
        env.object_qpos
    ]
)

print(
    "Object qvel:",
    env.data.qvel[
        env.object_qvel
    ]
)


print("\n====================================")
print("CONTACTS AT RESET")
print("====================================")


print(
    "Lower contact:",
    lower_contact
)

print(
    "Upper contact:",
    upper_contact
)


# ==================================================
# NOW TAKE ONE STEP WITH NEUTRAL ACTION
# ==================================================

action = np.array(
    [0.0, 0.0],
    dtype=np.float32
)

obs, reward, terminated, truncated, step_info = (
    env.step(action)
)


lower_tip_after = env.data.site_xpos[
    env.lower_tip_id
].copy()

upper_tip_after = env.data.site_xpos[
    env.upper_tip_id
].copy()

object_after = env.data.xpos[
    env.object_id
].copy()


print("\n====================================")
print("AFTER ONE RL STEP")
print("====================================")


print(
    "Lower tip:",
    lower_tip_after
)

print(
    "Upper tip:",
    upper_tip_after
)

print(
    "Object:",
    object_after
)


print(
    "\nLower distance:",
    f"{step_info['lower_distance_mm']:.2f} mm"
)

print(
    "Upper distance:",
    f"{step_info['upper_distance_mm']:.2f} mm"
)


print(
    "Lower contact:",
    step_info["lower_contact"]
)

print(
    "Upper contact:",
    step_info["upper_contact"]
)


print(
    "FFJ3 command:",
    f"{step_info['ffj3_command']:.3f}"
)

print(
    "FFJ0 command:",
    f"{step_info['ffj0_command']:.3f}"
)