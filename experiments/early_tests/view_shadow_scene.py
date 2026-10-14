import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(
    "models/shadow_hand/chopstick_scene.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)