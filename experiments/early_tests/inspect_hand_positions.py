import mujoco

model = mujoco.MjModel.from_xml_path(
    "models/shadow_hand/chopstick_scene.xml"
)

data = mujoco.MjData(model)

# Calculate initial body positions
mujoco.mj_forward(model, data)

bodies = [
    "rh_palm",
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_thdistal",
    "chopstick_1",
    "chopstick_2"
]

for name in bodies:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    print(name, "position:", data.xpos[body_id])
    