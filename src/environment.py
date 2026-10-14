import mujoco


MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# print("Environment loaded successfully")
# print("Bodies:", model.nbody)
# print("Joints:", model.njnt)
# print("Actuators:", model.nu)

bodies_to_check = [
    "rh_palm",
    "rh_ffproximal",
    "rh_ffmiddle",
    "rh_mfproximal",
    "rh_mfmiddle",
    "rh_rfproximal",
    "rh_thmiddle",
    "rh_thdistal",
    "lower_chopstick",
    "upper_chopstick"
]

print("\n--- Candidate grip locations ---")

for name in bodies_to_check:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name
    )

    print(name, data.xpos[body_id])

    # Test whether the lower chopstick follows the palm

palm_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "rh_palm"
)

lower_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "lower_chopstick"
)

upper_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "upper_chopstick"
)

print("\n--- Before wrist movement ---")
print("Palm:", data.xpos[palm_id].copy())
print("Lower chopstick:", data.xpos[lower_id].copy())
print("Upper chopstick:", data.xpos[upper_id].copy())

# Wrist actuator 0 = rh_A_WRJ2
data.ctrl[0] = -0.4

for _ in range(1000):
    mujoco.mj_step(model, data)

print("\n--- After wrist movement ---")
print("Palm:", data.xpos[palm_id])
print("Lower chopstick:", data.xpos[lower_id])
print("Upper chopstick:", data.xpos[upper_id])

print("\n--- Model dimensions ---")
print("Joints:", model.njnt)
print("qpos:", model.nq)
print("qvel:", model.nv)
print("Actuators:", model.nu)
