import time

import mujoco
import mujoco.viewer


MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


def actuator_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


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


print("Grip pose viewer")
print("Current problem: chopstick orientation relative to hand")


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    # Stable camera for comparison
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.45
    viewer.cam.lookat[:] = [0.33, 0.0, 0.02]

    while viewer.is_running():

        set_grip_pose()

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )