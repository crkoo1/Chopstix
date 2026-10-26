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


def body_name_from_geom(geom_id):

    body_id = model.geom_bodyid[geom_id]

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
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
# OBJECT
# ==================================================

object_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "grasp_object"
)

initial_object_z = data.xpos[object_id][2]

max_object_z = initial_object_z


# ==================================================
# CONTACT FLAGS
# ==================================================

lower_contact = False
upper_contact = False

first_lower_step = None
first_upper_step = None


# ==================================================
# SIMULATE
# ==================================================

SIM_STEPS = 1500


for step in range(SIM_STEPS):

    set_grip_pose()

    mujoco.mj_step(
        model,
        data
    )


    # Track maximum object height
    current_z = data.xpos[object_id][2]

    if current_z > max_object_z:
        max_object_z = current_z


    # Check contacts
    for i in range(data.ncon):

        contact = data.contact[i]

        body1 = body_name_from_geom(
            contact.geom1
        )

        body2 = body_name_from_geom(
            contact.geom2
        )

        bodies = {
            body1,
            body2
        }


        # Lower chopstick ↔ object
        if (
            "lower_chopstick" in bodies
            and
            "grasp_object" in bodies
        ):

            if not lower_contact:

                lower_contact = True
                first_lower_step = step


        # Upper chopstick ↔ object
        if (
            "upper_chopstick" in bodies
            and
            "grasp_object" in bodies
        ):

            if not upper_contact:

                upper_contact = True
                first_upper_step = step


# ==================================================
# RESULTS
# ==================================================

final_object_z = data.xpos[object_id][2]

lift_amount = (
    max_object_z
    - initial_object_z
)


print("\n====================================")
print("GRASP SETUP TEST")
print("====================================")


print(
    "\nLower chopstick touched object:",
    lower_contact
)

print(
    "First lower contact step:",
    first_lower_step
)


print(
    "\nUpper chopstick touched object:",
    upper_contact
)

print(
    "First upper contact step:",
    first_upper_step
)


print("\nOBJECT HEIGHT")

print(
    "Initial Z:",
    f"{initial_object_z:.5f} m"
)

print(
    "Final Z:",
    f"{final_object_z:.5f} m"
)

print(
    "Maximum Z:",
    f"{max_object_z:.5f} m"
)

print(
    "Maximum lift:",
    f"{lift_amount * 1000:.2f} mm"
)


both_contact = (
    lower_contact
    and upper_contact
)

lifted = (
    lift_amount > 0.005
)


print("\n====================================")
print("SUMMARY")
print("====================================")

print(
    "Both chopsticks contacted object:",
    both_contact
)

print(
    "Object lifted > 5 mm:",
    lifted
)