import time
import mujoco
import mujoco.viewer


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "models/shadow_hand/chopstick_scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ==================================================
# SETTINGS
# ==================================================

SETTLE_STEPS = 500

# How long to hold each state
RELAX_STEPS = 500
CURL_STEPS = 500

# Index finger actuators
FFJ3 = 8
FFJ0 = 9

# Curled position
FFJ3_CURL = 0.8
FFJ0_CURL = 1.2

# Relaxed position
FFJ3_RELAX = 0.0
FFJ0_RELAX = 0.0


# ==================================================
# RUN SIMULATION
# ==================================================

step = 0
last_phase = None

print("Simulation starting...")
print("First letting the chopsticks settle.")
print("Then the index finger will curl and relax repeatedly.")


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        # --------------------------------------------------
        # INITIAL SETTLING
        # --------------------------------------------------

        if step < SETTLE_STEPS:

            phase = "SETTLING"

            data.ctrl[FFJ3] = FFJ3_RELAX
            data.ctrl[FFJ0] = FFJ0_RELAX

        else:

            # Remove initial settling period
            loop_step = step - SETTLE_STEPS

            # Total length of one curl/relax cycle
            cycle_length = CURL_STEPS + RELAX_STEPS

            # Position inside current cycle
            cycle_step = loop_step % cycle_length


            # ----------------------------------------------
            # CURL PHASE
            # ----------------------------------------------

            if cycle_step < CURL_STEPS:

                phase = "CURL"

                data.ctrl[FFJ3] = FFJ3_CURL
                data.ctrl[FFJ0] = FFJ0_CURL


            # ----------------------------------------------
            # RELAX PHASE
            # ----------------------------------------------

            else:

                phase = "RELAX"

                data.ctrl[FFJ3] = FFJ3_RELAX
                data.ctrl[FFJ0] = FFJ0_RELAX


        # Print only when phase changes
        if phase != last_phase:

            if phase == "CURL":
                print("\nCURL → pushing upper chopstick")

            elif phase == "RELAX":
                print("\nRELAX → releasing upper chopstick")

            elif phase == "SETTLING":
                print("\nSETTLING...")

            last_phase = phase


        # Advance physics
        mujoco.mj_step(model, data)

        # Update viewer
        viewer.sync()

        step += 1

        # Real-time playback
        time.sleep(model.opt.timestep)