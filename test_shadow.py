import mujoco_menagerie as mm

shadow = mm.get("shadow_hand")

model = shadow.model("right_hand")

# CHECK 1: See the number of joints, actuators, qpos, and qvel
# print("Shadow right hand loaded successfully")
# print("Number of joints:", model.njnt)
# print("Number of actuators:", model.nu)
# print("Number of qpos:", model.nq)
# print("Number of qvel:", model.nv)

import mujoco

# CHECK 2: See the joints within the environment
# print("\n--- JOINTS ---")

# for i in range(model.njnt):
#     joint_name = mujoco.mj_id2name(
#         model,
#         mujoco.mjtObj.mjOBJ_JOINT,
#         i
#     )

#     print(i, joint_name)

# CHECK 3: See the actuators within the environment
# print("\n--- ACTUATORS ---")

# for i in range(model.nu):
#     actuator_name = mujoco.mj_id2name(
#         model,
#         mujoco.mjtObj.mjOBJ_ACTUATOR,
#         i
#     )

#     print(i, actuator_name)

print("\n--- ACTUATOR RANGES ---")

for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(
        i,
        actuator_name,
        model.actuator_ctrlrange[i]
    )