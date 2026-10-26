import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class ChopstickEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # LOAD MUJOCO MODEL
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_scene.xml"
        )

        self.data = mujoco.MjData(self.model)


        # ==================================================
        # UPPER CHOPSTICK HINGE
        # ==================================================

        hinge_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "upper_chopstick_hinge"
        )

        self.hinge_qpos = self.model.jnt_qposadr[
            hinge_id
        ]

        self.hinge_qvel = self.model.jnt_dofadr[
            hinge_id
        ]


        # ==================================================
        # INDEX FINGER ACTUATORS
        # ==================================================

        self.ffj3_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "rh_A_FFJ3"
        )

        self.ffj0_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "rh_A_FFJ0"
        )

        self.actuator_ids = np.array([
            self.ffj3_id,
            self.ffj0_id
        ])


        # ==================================================
        # ACTION SPACE
        #
        # PPO outputs:
        #
        # action[0] -> FFJ3
        # action[1] -> FFJ0
        #
        # Both range from -1 to +1
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )


        # ==================================================
        # OBSERVATION SPACE
        #
        # For this first test:
        #
        # observation[0] = upper chopstick hinge angle
        # observation[1] = upper chopstick hinge velocity
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32
        )


        # ==================================================
        # RL TARGET
        # ==================================================

        # We already experimentally showed that around
        # 2 degrees of upper-chopstick rotation is reachable.
        self.target_angle = np.deg2rad(2.0)

        # Agent is considered successful if it gets
        # within 0.3 degrees of the target.
        self.tolerance = np.deg2rad(0.3)


        # ==================================================
        # EPISODE SETTINGS
        # ==================================================

        self.max_steps = 150

        self.frame_skip = 5

        self.step_count = 0


    # ======================================================
    # GET OBSERVATION
    # ======================================================

    def _get_obs(self):

        hinge_angle = self.data.qpos[
            self.hinge_qpos
        ]

        hinge_velocity = self.data.qvel[
            self.hinge_qvel
        ]

        observation = np.array(
            [
                hinge_angle,
                hinge_velocity
            ],
            dtype=np.float32
        )

        return observation


    # ======================================================
    # RESET ENVIRONMENT
    # ======================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)


        # Reset MuJoCo simulation
        mujoco.mj_resetData(
            self.model,
            self.data
        )


        # Set all actuator commands to zero
        self.data.ctrl[:] = 0.0


        # Recalculate positions
        mujoco.mj_forward(
            self.model,
            self.data
        )


        # --------------------------------------------------
        # Let the chopsticks settle before each episode
        # --------------------------------------------------

        for _ in range(500):

            mujoco.mj_step(
                self.model,
                self.data
            )


        self.step_count = 0


        observation = self._get_obs()


        hinge_angle = self.data.qpos[
            self.hinge_qpos
        ]


        info = {
            "hinge_angle_deg":
                float(np.degrees(hinge_angle))
        }


        return observation, info


    # ======================================================
    # STEP ENVIRONMENT
    # ======================================================

    def step(self, action):

        self.step_count += 1


        # --------------------------------------------------
        # Make sure PPO action remains in [-1, +1]
        # --------------------------------------------------

        action = np.clip(
            action,
            -1.0,
            1.0
        )


        # --------------------------------------------------
        # Convert normalized PPO actions into the actual
        # Shadow Hand actuator control ranges
        # --------------------------------------------------

        ctrl_range = self.model.actuator_ctrlrange[
            self.actuator_ids
        ]

        ctrl_low = ctrl_range[:, 0]

        ctrl_high = ctrl_range[:, 1]


        controls = (
            ctrl_low
            + (action + 1.0)
            * 0.5
            * (ctrl_high - ctrl_low)
        )


        # Send commands to FFJ3 and FFJ0
        self.data.ctrl[
            self.actuator_ids
        ] = controls


        # --------------------------------------------------
        # Advance MuJoCo physics
        # --------------------------------------------------

        for _ in range(self.frame_skip):

            mujoco.mj_step(
                self.model,
                self.data
            )


        # --------------------------------------------------
        # Measure current upper chopstick angle
        # --------------------------------------------------

        hinge_angle = self.data.qpos[
            self.hinge_qpos
        ]


        # Difference between current and desired angle
        error = abs(
            self.target_angle
            - hinge_angle
        )


        # ==================================================
        # REWARD
        # ==================================================

        # Smaller error = better reward
        reward = -20.0 * error


        # --------------------------------------------------
        # Success condition
        # --------------------------------------------------

        success = bool(
            error < self.tolerance
        )


        # Reward bonus when target is reached
        if success:
            reward += 2.0


        # ==================================================
        # EPISODE END CONDITIONS
        # ==================================================

        # End immediately if successful
        terminated = bool(
            success
        )


        # Otherwise stop if maximum episode length reached
        truncated = bool(
            self.step_count >= self.max_steps
        )


        # ==================================================
        # OBSERVATION
        # ==================================================

        observation = self._get_obs()


        # ==================================================
        # DEBUG / LOG INFORMATION
        # ==================================================

        info = {

            "hinge_angle_deg":
                float(
                    np.degrees(
                        hinge_angle
                    )
                ),

            "target_angle_deg":
                float(
                    np.degrees(
                        self.target_angle
                    )
                ),

            "error_deg":
                float(
                    np.degrees(
                        error
                    )
                ),

            "success":
                bool(success)
        }


        # Gymnasium requires:
        #
        # observation
        # reward
        # terminated
        # truncated
        # info

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )