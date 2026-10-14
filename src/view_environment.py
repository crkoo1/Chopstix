import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(
    "models/shadow_hand/chopstick_scene.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = -0.4
        data.ctrl[8] = 0.8
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)