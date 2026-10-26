import mujoco
import numpy as np


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==================================================
# HELPER
# ==================================================

def site_position(name):

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name
    )

    return data.site_xpos[site_id].copy()


# ==================================================
# LET SYSTEM SETTLE
# ==================================================

for _ in range(500):
    mujoco.mj_step(model, data)


# ==================================================
# GET ACTUAL CHOPSTICK MARKERS
# ==================================================

lower_actual_back = site_position(
    "lower_actual_back"
)

lower_actual_front = site_position(
    "lower_actual_front"
)

upper_actual_back = site_position(
    "upper_actual_back"
)

upper_actual_front = site_position(
    "upper_actual_front"
)


# ==================================================
# GET GUIDE CENTERS
# ==================================================

lower_guide_back = site_position(
    "lower_guide_back_center"
)

lower_guide_front = site_position(
    "lower_guide_front_center"
)

upper_guide_back = site_position(
    "upper_guide_back_center"
)

upper_guide_front = site_position(
    "upper_guide_front_center"
)


# ==================================================
# POSITION ERRORS
# ==================================================

lower_back_error = np.linalg.norm(
    lower_actual_back
    - lower_guide_back
)

lower_front_error = np.linalg.norm(
    lower_actual_front
    - lower_guide_front
)

upper_back_error = np.linalg.norm(
    upper_actual_back
    - upper_guide_back
)

upper_front_error = np.linalg.norm(
    upper_actual_front
    - upper_guide_front
)


# ==================================================
# ORIENTATION ERROR
# ==================================================

def angle_between(v1, v2):

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot = np.clip(
        np.dot(v1, v2),
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(dot)
    )


# Actual chopstick directions

lower_actual_direction = (
    lower_actual_front
    - lower_actual_back
)

upper_actual_direction = (
    upper_actual_front
    - upper_actual_back
)


# Desired guide directions

lower_guide_direction = (
    lower_guide_front
    - lower_guide_back
)

upper_guide_direction = (
    upper_guide_front
    - upper_guide_back
)


lower_angle_error = angle_between(
    lower_actual_direction,
    lower_guide_direction
)

upper_angle_error = angle_between(
    upper_actual_direction,
    upper_guide_direction
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("\n====================================")
print("CHOPSTICK GRIP ALIGNMENT")
print("====================================")


print("\nLOWER CHOPSTICK")

print(
    "Back position error:",
    f"{lower_back_error * 1000:.2f} mm"
)

print(
    "Front position error:",
    f"{lower_front_error * 1000:.2f} mm"
)

print(
    "Orientation error:",
    f"{lower_angle_error:.2f} deg"
)


print("\nUPPER CHOPSTICK")

print(
    "Back position error:",
    f"{upper_back_error * 1000:.2f} mm"
)

print(
    "Front position error:",
    f"{upper_front_error * 1000:.2f} mm"
)

print(
    "Orientation error:",
    f"{upper_angle_error:.2f} deg"
)


# ==================================================
# SIMPLE PASS / FAIL
# ==================================================

POSITION_TOLERANCE = 0.005
ANGLE_TOLERANCE = 5.0


lower_good = (
    lower_back_error < POSITION_TOLERANCE
    and
    lower_front_error < POSITION_TOLERANCE
    and
    lower_angle_error < ANGLE_TOLERANCE
)


upper_good = (
    upper_back_error < POSITION_TOLERANCE
    and
    upper_front_error < POSITION_TOLERANCE
    and
    upper_angle_error < ANGLE_TOLERANCE
)


print("\n====================================")
print("GRIP CHECK")
print("====================================")

print(
    "Lower chopstick aligned:",
    lower_good
)

print(
    "Upper chopstick aligned:",
    upper_good
)

print(
    "Overall grip aligned:",
    lower_good and upper_good
)