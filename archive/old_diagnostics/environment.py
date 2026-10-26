import mujoco
import numpy as np


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)


# ==================================================
# GET IMPORTANT IDS
# ==================================================

upper_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "upper_chopstick"
)

hinge_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "upper_chopstick_hinge"
)

# Location of the hinge angle inside qpos
hinge_qpos_address = model.jnt_qposadr[hinge_id]


# ==================================================
# SETTINGS
# ==================================================

SETTLE_STEPS = 500
CONTROL_STEPS = 500

# Index finger actuators
FFJ3_ACTUATOR = 8
FFJ0_ACTUATOR = 9

FFJ3_COMMAND = 0.8
FFJ0_COMMAND = 1.2


# ==================================================
# HELPER FUNCTION
# ==================================================

def get_contact_bodies(data, contact):
    body1_id = model.geom_bodyid[contact.geom1]
    body2_id = model.geom_bodyid[contact.geom2]

    body1_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body1_id
    )

    body2_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body2_id
    )

    return body1_name, body2_name


# ==================================================
# CREATE SIMULATION
# ==================================================

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


print("\n========================================")
print("CHOPSTICK CONTACT TEST")
print("========================================")


# ==================================================
# PHASE 1: LET EVERYTHING SETTLE
# ==================================================

print("\nLetting system settle...")

for _ in range(SETTLE_STEPS):
    mujoco.mj_step(model, data)


settled_hinge_angle = data.qpos[hinge_qpos_address]

print(
    "Upper chopstick hinge after settling:",
    f"{np.degrees(settled_hinge_angle):.2f} deg"
)


# ==================================================
# PHASE 2: CURL INDEX FINGER
# ==================================================

print("\nCurling index finger...")

data.ctrl[FFJ3_ACTUATOR] = FFJ3_COMMAND
data.ctrl[FFJ0_ACTUATOR] = FFJ0_COMMAND


index_contact_detected = False
first_contact_step = None
first_contact_pair = None


for step in range(CONTROL_STEPS):

    mujoco.mj_step(model, data)

    # Look through every contact at this timestep
    for i in range(data.ncon):

        contact = data.contact[i]

        body1_name, body2_name = get_contact_bodies(
            data,
            contact
        )

        bodies = {body1_name, body2_name}

        # Detect contact between index finger
        # and upper chopstick
        if (
            "upper_chopstick" in bodies
            and any(
                finger_body in bodies
                for finger_body in [
                    "rh_ffproximal",
                    "rh_ffmiddle",
                    "rh_ffdistal"
                ]
            )
        ):

            if not index_contact_detected:

                index_contact_detected = True
                first_contact_step = step

                first_contact_pair = (
                    body1_name,
                    body2_name
                )


# ==================================================
# RESULTS
# ==================================================

final_hinge_angle = data.qpos[hinge_qpos_address]

hinge_change = (
    final_hinge_angle
    - settled_hinge_angle
)


print("\n========================================")
print("RESULTS")
print("========================================")

print(
    "Index contacted upper chopstick:",
    index_contact_detected
)

if index_contact_detected:

    print(
        "First contact step:",
        first_contact_step
    )

    print(
        "Contact pair:",
        first_contact_pair[0],
        "<->",
        first_contact_pair[1]
    )


print(
    "Hinge angle before finger motion:",
    f"{np.degrees(settled_hinge_angle):.2f} deg"
)

print(
    "Hinge angle after finger motion:",
    f"{np.degrees(final_hinge_angle):.2f} deg"
)

print(
    "Finger-induced chopstick rotation:",
    f"{np.degrees(hinge_change):.2f} deg"
)