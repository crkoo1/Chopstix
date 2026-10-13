import mujoco
import mujoco_menagerie as mm
import time
import mujoco.viewer

shadow = mm.get("shadow_hand")
model = shadow.model("right_hand")
data = mujoco.MjData(model)

print("Initial FFJ3 position:", data.qpos[3])

data.ctrl[8] = 0.8

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # First position
        data.ctrl[8] = 0.8

        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        # Return toward straight position
        data.ctrl[8] = 0.0

        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

print("Final FFJ3 position:", data.qpos[3])
print("Final FFJ3 velocity:", data.qvel[3])