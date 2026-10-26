import mujoco


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==================================================
# LET THE SYSTEM SETTLE
# ==================================================

for _ in range(500):
    mujoco.mj_step(model, data)


# ==================================================
# GET SITE / BODY IDS
# ==================================================

lower_tip_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "lower_tip"
)

upper_tip_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "upper_tip"
)

object_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "grasp_object"
)


# ==================================================
# GET POSITIONS
# ==================================================

lower_tip = data.site_xpos[lower_tip_id].copy()

upper_tip = data.site_xpos[upper_tip_id].copy()

object_pos = data.xpos[object_id].copy()

midpoint = (
    lower_tip
    + upper_tip
) / 2.0


# ==================================================
# PRINT RESULTS
# ==================================================

print("\n====================================")
print("CHOPSTICK TIP POSITIONS")
print("====================================")

print("Lower tip:", lower_tip)
print("Upper tip:", upper_tip)
print("Object:", object_pos)

print("\nMidpoint between tips:", midpoint)

print("\nThis midpoint is the approximate")
print("target position for the sphere.")