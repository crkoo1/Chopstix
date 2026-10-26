import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickSkill1Env(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # MODEL
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_skill1_scene.xml"
        )

        self.data = mujoco.MjData(self.model)


        # ==================================================
        # HELPERS
        # ==================================================

        def actuator_id(name):
            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name
            )


        # ==================================================
        # HINGE
        # ==================================================

        self.hinge_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "upper_chopstick_hinge"
        )

        self.hinge_qpos = self.model.jnt_qposadr[
            self.hinge_id
        ]

        self.hinge_qvel = self.model.jnt_dofadr[
            self.hinge_id
        ]


        # ==================================================
        # SERVO
        # ==================================================

        self.servo_id = actuator_id(
            "upper_chopstick_servo"
        )


        # ==================================================
        # HAND ACTUATORS
        # ==================================================

        self.THJ5 = actuator_id("rh_A_THJ5")
        self.THJ4 = actuator_id("rh_A_THJ4")
        self.THJ3 = actuator_id("rh_A_THJ3")
        self.THJ2 = actuator_id("rh_A_THJ2")
        self.THJ1 = actuator_id("rh_A_THJ1")

        self.FFJ4 = actuator_id("rh_A_FFJ4")
        self.FFJ3 = actuator_id("rh_A_FFJ3")
        self.FFJ0 = actuator_id("rh_A_FFJ0")

        self.MFJ4 = actuator_id("rh_A_MFJ4")
        self.MFJ3 = actuator_id("rh_A_MFJ3")
        self.MFJ0 = actuator_id("rh_A_MFJ0")

        self.RFJ4 = actuator_id("rh_A_RFJ4")
        self.RFJ3 = actuator_id("rh_A_RFJ3")
        self.RFJ0 = actuator_id("rh_A_RFJ0")

        self.LFJ5 = actuator_id("rh_A_LFJ5")
        self.LFJ4 = actuator_id("rh_A_LFJ4")
        self.LFJ3 = actuator_id("rh_A_LFJ3")
        self.LFJ0 = actuator_id("rh_A_LFJ0")


        # ==================================================
        # ACTION
        #
        # -1 = move servo target toward CLOSED
        #  0 = keep current servo command
        # +1 = move servo target toward OPEN
        #
        # It does NOT directly give PPO the target angle.
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )


        # ==================================================
        # OBSERVATION
        #
        # 0 current angle / 20
        # 1 angular velocity
        # 2 target angle / 20
        # 3 error / 10
        # 4 current commanded angle / 20
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )


        # ==================================================
        # SETTINGS
        # ==================================================

        self.frame_skip = 5

        self.max_steps = 150


        self.min_angle_deg = 8.0
        self.max_angle_deg = 14.0


        # Maximum change in commanded angle
        # per RL step.
        self.max_delta_deg = 0.40


        # Success conditions
        self.tolerance_deg = 0.25

        self.required_hold_steps = 10


        self.step_count = 0

        self.hold_counter = 0


        self.target_deg = 0.0

        self.command_deg = 0.0


    # ======================================================
    # FIXED HAND POSE
    # ======================================================

    def _set_hand_pose(self):

        # Thumb
        self.data.ctrl[self.THJ5] = 0.40
        self.data.ctrl[self.THJ4] = 0.90
        self.data.ctrl[self.THJ3] = 0.08
        self.data.ctrl[self.THJ2] = 0.45
        self.data.ctrl[self.THJ1] = 0.95

        # Index
        self.data.ctrl[self.FFJ4] = 0.00
        self.data.ctrl[self.FFJ3] = 0.72
        self.data.ctrl[self.FFJ0] = 0.95

        # Middle
        self.data.ctrl[self.MFJ4] = 0.00
        self.data.ctrl[self.MFJ3] = 0.75
        self.data.ctrl[self.MFJ0] = 0.90

        # Ring
        self.data.ctrl[self.RFJ4] = 0.00
        self.data.ctrl[self.RFJ3] = 1.05
        self.data.ctrl[self.RFJ0] = 1.25

        # Little
        self.data.ctrl[self.LFJ5] = 0.15
        self.data.ctrl[self.LFJ4] = 0.00
        self.data.ctrl[self.LFJ3] = 1.05
        self.data.ctrl[self.LFJ0] = 1.25


    # ======================================================
    # STATE VALUES
    # ======================================================

    def _hinge_deg(self):

        return float(
            np.degrees(
                self.data.qpos[
                    self.hinge_qpos
                ]
            )
        )


    def _error_deg(self):

        return (
            self.target_deg
            - self._hinge_deg()
        )


    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_obs(self):

        current_deg = self._hinge_deg()

        velocity = float(
            self.data.qvel[
                self.hinge_qvel
            ]
        )

        error_deg = (
            self.target_deg
            - current_deg
        )


        return np.array(
            [
                current_deg / 20.0,

                velocity,

                self.target_deg / 20.0,

                error_deg / 10.0,

                self.command_deg / 20.0
            ],
            dtype=np.float32
        )


    # ======================================================
    # RESET
    # ======================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )


        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.data.ctrl[:] = 0.0


        # ==================================================
        # RANDOM START
        # ==================================================

        start_deg = float(
            self.np_random.uniform(
                self.min_angle_deg,
                self.max_angle_deg
            )
        )


        # ==================================================
        # RANDOM TARGET
        #
        # Force target to be at least 2 degrees away.
        # ==================================================

        while True:

            target_deg = float(
                self.np_random.uniform(
                    self.min_angle_deg,
                    self.max_angle_deg
                )
            )

            if abs(
                target_deg
                - start_deg
            ) >= 2.0:

                break


        self.target_deg = target_deg

        self.command_deg = start_deg


        # ==================================================
        # SETTLE AT STARTING ANGLE
        # ==================================================

        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            start_deg
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


        for _ in range(600):

            self._set_hand_pose()

            self.data.ctrl[
                self.servo_id
            ] = np.radians(
                start_deg
            )

            mujoco.mj_step(
                self.model,
                self.data
            )


        self.step_count = 0

        self.hold_counter = 0


        info = {

            "start_deg":
                self._hinge_deg(),

            "target_deg":
                self.target_deg,

            "error_deg":
                self._error_deg()
        }


        return (
            self._get_obs(),
            info
        )


    # ======================================================
    # STEP
    # ======================================================

    def step(
        self,
        action
    ):

        self.step_count += 1


        self._set_hand_pose()


        # ==================================================
        # RL CHANGES SERVO COMMAND INCREMENTALLY
        # ==================================================

        action_value = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )


        delta_deg = (
            action_value
            * self.max_delta_deg
        )


        self.command_deg += (
            delta_deg
        )


        self.command_deg = float(
            np.clip(
                self.command_deg,
                self.min_angle_deg,
                self.max_angle_deg
            )
        )


        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )


        # ==================================================
        # PHYSICS
        # ==================================================

        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )


        # ==================================================
        # CURRENT STATE
        # ==================================================

        current_deg = self._hinge_deg()

        error_deg = (
            self.target_deg
            - current_deg
        )

        absolute_error = abs(
            error_deg
        )


        # ==================================================
        # REWARD
        # ==================================================

        # Main reward:
        # closer to target = better.
        reward = (
            -absolute_error
        )


        # Bonus for getting reasonably close
        if absolute_error < 1.0:
            reward += 1.0


        if absolute_error < 0.5:
            reward += 2.0


        if absolute_error <= self.tolerance_deg:
            reward += 4.0


        # Small penalty discourages huge unnecessary motion.
        reward -= (
            0.01
            * abs(
                action_value
            )
        )


        # ==================================================
        # HOLD TARGET
        # ==================================================

        if absolute_error <= self.tolerance_deg:

            self.hold_counter += 1

        else:

            self.hold_counter = 0


        success = bool(
            self.hold_counter
            >= self.required_hold_steps
        )


        if success:
            reward += 15.0


        terminated = bool(
            success
        )

        truncated = bool(
            self.step_count
            >= self.max_steps
        )


        info = {

            "current_deg":
                float(current_deg),

            "target_deg":
                float(self.target_deg),

            "error_deg":
                float(error_deg),

            "command_deg":
                float(self.command_deg),

            "hold_steps":
                int(self.hold_counter),

            "success":
                bool(success)
        }


        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )