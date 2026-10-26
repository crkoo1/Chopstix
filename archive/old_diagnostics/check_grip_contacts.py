import mujoco


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==================================================
# ACTUATOR HELPER
# ==================================================

def actuator_id(name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )


# ==================================================
# ACTUATOR IDS
# ==================================================

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


# ==================================================
# APPLY CURRENT GRIP POSE
# ==================================================

def set_grip_pose():

    # Thumb
    data.ctrl[THJ5] = 0.25
    data.ctrl[THJ4] = 0.65
    data.ctrl[THJ3] = 0.05
    data.ctrl[THJ2] = 0.30
    data.ctrl[THJ1] = 0.70

    # Index
    data.ctrl[FFJ4] = 0.00
    data.ctrl[FFJ3] = 0.70
    data.ctrl[FFJ0] = 0.90

    # Middle
    data.ctrl[MFJ4] = 0.00
    data.ctrl[MFJ3] = 0.85
    data.ctrl[MFJ0] = 1.00

    # Ring
    data.ctrl[RFJ4] = 0.00
    data.ctrl[RFJ3] = 1.00
    data.ctrl[RFJ0] = 1.20

    # Little
    data.ctrl[LFJ5] = 0.10
    data.ctrl[LFJ4] = 0.00
    data.ctrl[LFJ3] = 1.00
    data.ctrl[LFJ0] = 1.20


# ==================================================
# LET HAND MOVE INTO GRIP
# ==================================================

for step in range(1500):

    set_grip_pose()

    mujoco.mj_step(
        model,
        data
    )


# ==================================================
# BODY NAME HELPER
# ==================================================

def body_name_from_geom(geom_id):

    body_id = model.geom_bodyid[geom_id]

    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )


# ==================================================
# CHECK CONTACTS
# ==================================================

print("\n====================================")
print("CHOPSTICK GRIP CONTACTS")
print("====================================")


lower_contacts = set()
upper_contacts = set()


for i in range(data.ncon):

    contact = data.contact[i]

    body1 = body_name_from_geom(
        contact.geom1
    )

    body2 = body_name_from_geom(
        contact.geom2
    )

    bodies = {
        body1,
        body2
    }


    # ----------------------------------------------
    # Lower chopstick contact
    # ----------------------------------------------

    if "lower_chopstick" in bodies:

        other = (
            body2
            if body1 == "lower_chopstick"
            else body1
        )

        lower_contacts.add(other)


    # ----------------------------------------------
    # Upper chopstick contact
    # ----------------------------------------------

    if "upper_chopstick" in bodies:

        other = (
            body2
            if body1 == "upper_chopstick"
            else body1
        )

        upper_contacts.add(other)


# ==================================================
# PRINT LOWER CONTACTS
# ==================================================

print("\nLOWER CHOPSTICK TOUCHING:")

if len(lower_contacts) == 0:

    print("No bodies")

else:

    for body in sorted(lower_contacts):

        print(" -", body)


# ==================================================
# PRINT UPPER CONTACTS
# ==================================================

print("\nUPPER CHOPSTICK TOUCHING:")

if len(upper_contacts) == 0:

    print("No bodies")

else:

    for body in sorted(upper_contacts):

        print(" -", body)


# ==================================================
# INTERPRET FINGER GROUPS
# ==================================================

def finger_group(body_name):

    if body_name is None:
        return "Unknown"

    if body_name.startswith("rh_th"):
        return "Thumb"

    if body_name.startswith("rh_ff"):
        return "Index"

    if body_name.startswith("rh_mf"):
        return "Middle"

    if body_name.startswith("rh_rf"):
        return "Ring"

    if body_name.startswith("rh_lf"):
        return "Little"

    if body_name == "rh_palm":
        return "Palm"

    return body_name


print("\n====================================")
print("CONTACT SUMMARY")
print("====================================")


print("\nLower chopstick:")

for body in sorted(lower_contacts):

    print(
        finger_group(body),
        "->",
        body
    )


print("\nUpper chopstick:")

for body in sorted(upper_contacts):

    print(
        finger_group(body),
        "->",
        body
    )