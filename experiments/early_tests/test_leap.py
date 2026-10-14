import mujoco
import mujoco_menagerie as mm

model = mm.load("leap_hand")

# Initial check to see if the LEAP hand loaded successfully
# print("LEAP Hand loaded successfully")
# print("Number of joints:", model.njnt)
# print("Number of actuators:", model.nu)
# print("Number of qpos:", model.nq)
# print("Number of qvel:", model.nv)

# Create the changing simulation state
data = mujoco.MjData(model)

print("\nInitial qpos:")
print(data.qpos)

print("\nInitial qvel:")
print(data.qvel)

print("\nInitial actuator controls:")
print(data.ctrl)

print("\n--- JOINTS ---")

for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(i, joint_name)

print("\n--- ACTUATORS ---")

for i in range(model.nu):
    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(
        i,
        actuator_name,
        "range:",
        model.actuator_ctrlrange[i]
    )

print("\n--- ACTUATOR TO JOINT MAPPING ---")

for i in range(model.nu):

    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    joint_id = model.actuator_trnid[i, 0]

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id
    )

    print(i, actuator_name, "->", joint_name)