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
# HELPERS
# ==================================================

def actuator_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


def site_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        name
    )


def body_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )


def body_name_from_geom(geom_id):
    body = model.geom_bodyid[geom_id]

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body
    )


# ==================================================
# ACTUATORS
# ==================================================

# Thumb
THJ5 = actuator_id("rh_A_THJ5")
THJ4 = actuator_id("rh_A_THJ4")
THJ3 = actuator_id("rh_A_THJ3")
THJ2 = actuator_id("rh_A_THJ2")
THJ1 = actuator_id("rh_A_THJ1")

# Index
FFJ4 = actuator_id("rh_A_FFJ4")
FFJ3 = actuator_id("rh_A_FFJ3")
FFJ0 = actuator_id("rh_A_FFJ0")

# Middle
MFJ4 = actuator_id("rh_A_MFJ4")
MFJ3 = actuator_id("rh_A_MFJ3")
MFJ0 = actuator_id("rh_A_MFJ0")

# Ring
RFJ4 = actuator_id("rh_A_RFJ4")
RFJ3 = actuator_id("rh_A_RFJ3")
RFJ0 = actuator_id("rh_A_RFJ0")

# Little
LFJ5 = actuator_id("rh_A_LFJ5")
LFJ4 = actuator_id("rh_A_LFJ4")
LFJ3 = actuator_id("rh_A_LFJ3")
LFJ0 = actuator_id("rh_A_LFJ0")


# ==================================================
# CURRENT GRIP POSE
# ==================================================

def set_grip_pose():

    # Thumb
    data.ctrl[THJ5] = 0.40
    data.ctrl[THJ4] = 0.90
    data.ctrl[THJ3] = 0.08
    data.ctrl[THJ2] = 0.45
    data.ctrl[THJ1] = 0.95

    # Index
    data.ctrl[FFJ4] = 0.00
    data.ctrl[FFJ3] = 0.72
    data.ctrl[FFJ0] = 0.95

    # Middle
    data.ctrl[MFJ4] = 0.00
    data.ctrl[MFJ3] = 0.75
    data.ctrl[MFJ0] = 0.90

    # Ring
    data.ctrl[RFJ4] = 0.00
    data.ctrl[RFJ3] = 1.05
    data.ctrl[RFJ0] = 1.25

    # Little
    data.ctrl[LFJ5] = 0.15
    data.ctrl[LFJ4] = 0.00
    data.ctrl[LFJ3] = 1.05
    data.ctrl[LFJ0] = 1.25


# ==================================================
# MOVE HAND INTO GRIP POSE
# ==================================================

for _ in range(1500):

    set_grip_pose()

    mujoco.mj_step(
        model,
        data
    )


# ==================================================
# GET IMPORTANT POSITIONS
# ==================================================

lower_tip = data.site_xpos[
    site_id("lower_tip")
].copy()

upper_tip = data.site_xpos[
    site_id("upper_tip")
].copy()

object_pos = data.xpos[
    body_id("grasp_object")
].copy()

midpoint = (
    lower_tip
    + upper_tip
) / 2.0


# ==================================================
# DISTANCES
# ==================================================

distance_to_midpoint = np.linalg.norm(
    object_pos
    - midpoint
)

tip_separation = np.linalg.norm(
    lower_tip
    - upper_tip
)


# ==================================================
# FIND EVERYTHING TOUCHING THE OBJECT
# ==================================================

object_contacts = set()

for i in range(data.ncon):

    contact = data.contact[i]

    body1 = body_name_from_geom(
        contact.geom1
    )

    body2 = body_name_from_geom(
        contact.geom2
    )

    if body1 == "grasp_object":
        object_contacts.add(body2)

    elif body2 == "grasp_object":
        object_contacts.add(body1)


# ==================================================
# PRINT
# ==================================================

print("\n====================================")
print("CURRENT GRASP GEOMETRY")
print("====================================")

print("\nLower tip:")
print(lower_tip)

print("\nUpper tip:")
print(upper_tip)

print("\nMidpoint between tips:")
print(midpoint)

print("\nCurrent object position:")
print(object_pos)


print("\n====================================")
print("DISTANCES")
print("====================================")

print(
    "Tip separation:",
    f"{tip_separation * 1000:.2f} mm"
)

print(
    "Object distance from midpoint:",
    f"{distance_to_midpoint * 1000:.2f} mm"
)


print("\n====================================")
print("OBJECT CONTACTS")
print("====================================")

if len(object_contacts) == 0:

    print("Object is touching nothing.")

else:

    for body in sorted(object_contacts):
        print(" -", body)


print("\n====================================")
print("RECOMMENDED OBJECT POSITION")
print("====================================")

print(
    f'pos="{midpoint[0]:.5f} '
    f'{midpoint[1]:.5f} '
    f'{midpoint[2]:.5f}"'
)