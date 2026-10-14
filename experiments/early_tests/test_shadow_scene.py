import mujoco_menagerie as mm
import mujoco

shadow = mm.get("shadow_hand")

model = shadow.model("scene_right")


# print("Shadow scene loaded")
# print("Bodies:", model.nbody)
# print("Joints:", model.njnt)
# print("Actuators:", model.nu)

# Checking for extra joints in scene
# print("\n--- EXTRA JOINTS IN SCENE ---")

# for i in range(model.njnt):
#     joint_name = mujoco.mj_id2name(
#         model,
#         mujoco.mjtObj.mjOBJ_JOINT,
#         i
#     )

#     print(i, joint_name)

# Checking for extra bodies in scene
print("\n--- BODIES ---")

for i in range(model.nbody):
    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )

    print(i, body_name)