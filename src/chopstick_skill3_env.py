import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickSkill3Env(gym.Env):

    def __init__(self):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_skill3_scene.xml"
        )

        self.data = mujoco.MjData(self.model)

        def actuator_id(name):
            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )

        def site_id(name):
            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )

        def body_id(name):
            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )

        self.lower_tip_id = site_id("lower_tip")
        self.upper_tip_id = site_id("upper_tip")
        self.cube_body_id = body_id("grasp_cube")

        self.hinge_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "upper_chopstick_hinge",
        )

        self.hinge_qpos = self.model.jnt_qposadr[
            self.hinge_id
        ]

        self.hinge_qvel = self.model.jnt_dofadr[
            self.hinge_id
        ]

        self.cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_freejoint",
        )

        self.cube_qpos = self.model.jnt_qposadr[
            self.cube_joint_id
        ]

        self.cube_qvel = self.model.jnt_dofadr[
            self.cube_joint_id
        ]

        self.servo_id = actuator_id(
            "upper_chopstick_servo"
        )

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

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32,
        )

        self.frame_skip = 5

        self.max_steps = 150

        self.min_angle_deg = 8.0
        self.max_angle_deg = 14.0

        # V4:
        # Experimentally validated control authority.
        #
        # 0.10 deg/step -> 58% controller success
        # 0.15 deg/step -> 79%
        # 0.20 deg/step -> 98%
        # 0.25 deg/step -> 100%
        self.max_delta_deg = 0.25

        self.hold_time_seconds = 1.0

        self.control_dt = (
            self.model.opt.timestep
            * self.frame_skip
        )

        self.required_hold_steps = int(
            np.ceil(
                self.hold_time_seconds
                / self.control_dt
            )
        )

        self.drop_distance = 0.015
        self.max_displacement = 0.025

        self.stick_radius = 0.004
        self.cube_half_width = 0.005
        self.lower_preload = 0.0005

        self.search_start_deg = 14.0
        self.search_end_deg = 8.0
        self.search_step_deg = 0.05
        self.search_settle_steps = 25

        # V4 disturbance configuration
        self.command_disturbance_deg = 0.25

        self.horizontal_velocity_disturbance = 0.0

        self.vertical_velocity_down = 0.010
        self.vertical_velocity_up = 0.002

        self.command_deg = 14.0

        self.grasp_command_deg = None
        self.grasp_region_min_deg = None
        self.grasp_region_max_deg = None

        self.base_grasp_state = None

        self.initial_cube_position = np.zeros(
            3,
            dtype=np.float64,
        )

        self.command_disturbance_used = 0.0

        self.velocity_disturbance_used = np.zeros(
            3,
            dtype=np.float64,
        )

        self.step_count = 0
        self.hold_counter = 0


    def _set_hand_pose(self):

        self.data.ctrl[self.THJ5] = 0.40
        self.data.ctrl[self.THJ4] = 0.90
        self.data.ctrl[self.THJ3] = 0.08
        self.data.ctrl[self.THJ2] = 0.45
        self.data.ctrl[self.THJ1] = 0.95

        self.data.ctrl[self.FFJ4] = 0.00
        self.data.ctrl[self.FFJ3] = 0.72
        self.data.ctrl[self.FFJ0] = 0.95

        self.data.ctrl[self.MFJ4] = 0.00
        self.data.ctrl[self.MFJ3] = 0.75
        self.data.ctrl[self.MFJ0] = 0.90

        self.data.ctrl[self.RFJ4] = 0.00
        self.data.ctrl[self.RFJ3] = 1.05
        self.data.ctrl[self.RFJ0] = 1.25

        self.data.ctrl[self.LFJ5] = 0.15
        self.data.ctrl[self.LFJ4] = 0.00
        self.data.ctrl[self.LFJ3] = 1.05
        self.data.ctrl[self.LFJ0] = 1.25


    def _hinge_deg(self):

        return float(
            np.degrees(
                self.data.qpos[
                    self.hinge_qpos
                ]
            )
        )


    def _cube_position(self):

        return self.data.xpos[
            self.cube_body_id
        ].copy()


    def _cube_linear_velocity(self):

        return self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 3
        ].copy()


    def _set_cube_pose(
        self,
        position,
        quaternion,
    ):

        self.data.qpos[
            self.cube_qpos:
            self.cube_qpos + 3
        ] = position

        self.data.qpos[
            self.cube_qpos + 3:
            self.cube_qpos + 7
        ] = quaternion

        self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 6
        ] = 0.0


    @staticmethod
    def _matrix_to_quaternion(matrix):

        m = matrix

        trace = (
            m[0, 0]
            + m[1, 1]
            + m[2, 2]
        )

        if trace > 0.0:

            s = np.sqrt(
                trace + 1.0
            ) * 2.0

            qw = 0.25 * s

            qx = (
                m[2, 1]
                - m[1, 2]
            ) / s

            qy = (
                m[0, 2]
                - m[2, 0]
            ) / s

            qz = (
                m[1, 0]
                - m[0, 1]
            ) / s

        elif (
            m[0, 0] > m[1, 1]
            and
            m[0, 0] > m[2, 2]
        ):

            s = np.sqrt(
                1.0
                + m[0, 0]
                - m[1, 1]
                - m[2, 2]
            ) * 2.0

            qw = (
                m[2, 1]
                - m[1, 2]
            ) / s

            qx = 0.25 * s

            qy = (
                m[0, 1]
                + m[1, 0]
            ) / s

            qz = (
                m[0, 2]
                + m[2, 0]
            ) / s

        elif m[1, 1] > m[2, 2]:

            s = np.sqrt(
                1.0
                + m[1, 1]
                - m[0, 0]
                - m[2, 2]
            ) * 2.0

            qw = (
                m[0, 2]
                - m[2, 0]
            ) / s

            qx = (
                m[0, 1]
                + m[1, 0]
            ) / s

            qy = 0.25 * s

            qz = (
                m[1, 2]
                + m[2, 1]
            ) / s

        else:

            s = np.sqrt(
                1.0
                + m[2, 2]
                - m[0, 0]
                - m[1, 1]
            ) * 2.0

            qw = (
                m[1, 0]
                - m[0, 1]
            ) / s

            qx = (
                m[0, 2]
                + m[2, 0]
            ) / s

            qy = (
                m[1, 2]
                + m[2, 1]
            ) / s

            qz = 0.25 * s

        quat = np.array(
            [
                qw,
                qx,
                qy,
                qz,
            ],
            dtype=np.float64,
        )

        quat /= np.linalg.norm(quat)

        return quat


    def _compute_cube_pose(self):

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ].copy()

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ].copy()

        gap_axis = (
            upper_tip
            - lower_tip
        )

        gap_axis /= np.linalg.norm(
            gap_axis
        )

        world_up = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        local_x = np.cross(
            gap_axis,
            world_up,
        )

        local_x /= np.linalg.norm(
            local_x
        )

        local_z = np.cross(
            local_x,
            gap_axis,
        )

        local_z /= np.linalg.norm(
            local_z
        )

        rotation = np.column_stack(
            (
                local_x,
                gap_axis,
                local_z,
            )
        )

        quaternion = (
            self._matrix_to_quaternion(
                rotation
            )
        )

        distance = (
            self.stick_radius
            + self.cube_half_width
            - self.lower_preload
        )

        cube_position = (
            lower_tip
            + gap_axis
            * distance
        )

        return (
            cube_position,
            quaternion,
        )


    def _contact_flags(self):

        lower_contact = False
        upper_contact = False

        for i in range(
            self.data.ncon
        ):

            contact = (
                self.data.contact[i]
            )

            body1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[
                    contact.geom1
                ],
            )

            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[
                    contact.geom2
                ],
            )

            bodies = {
                body1,
                body2,
            }

            if (
                "lower_chopstick" in bodies
                and
                "grasp_cube" in bodies
            ):

                lower_contact = True

            if (
                "upper_chopstick" in bodies
                and
                "grasp_cube" in bodies
            ):

                upper_contact = True

        return (
            lower_contact,
            upper_contact,
        )


    def _save_state(self):

        return {
            "qpos":
                self.data.qpos.copy(),

            "qvel":
                self.data.qvel.copy(),

            "ctrl":
                self.data.ctrl.copy(),

            "command_deg":
                float(
                    self.command_deg
                ),
        }


    def _restore_state(
        self,
        state,
    ):

        self.data.qpos[:] = (
            state["qpos"]
        )

        self.data.qvel[:] = (
            state["qvel"]
        )

        self.data.ctrl[:] = (
            state["ctrl"]
        )

        self.command_deg = float(
            state["command_deg"]
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def _fixed_cube_step(
        self,
        cube_position,
        cube_quaternion,
    ):

        self._set_hand_pose()

        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )

        self._set_cube_pose(
            cube_position,
            cube_quaternion,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        mujoco.mj_step(
            self.model,
            self.data,
        )

        self._set_cube_pose(
            cube_position,
            cube_quaternion,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def _build_base_grasp(self):

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.ctrl[:] = 0.0

        far_position = np.array(
            [
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )

        identity_quaternion = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )

        self._set_cube_pose(
            far_position,
            identity_quaternion,
        )

        self.command_deg = 14.0

        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        for _ in range(600):

            self._set_hand_pose()

            self.data.ctrl[
                self.servo_id
            ] = np.radians(
                self.command_deg
            )

            self._set_cube_pose(
                far_position,
                identity_quaternion,
            )

            mujoco.mj_step(
                self.model,
                self.data,
            )

        (
            cube_position,
            cube_quaternion,
        ) = self._compute_cube_pose()

        self._set_cube_pose(
            cube_position,
            cube_quaternion,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        candidates = []

        search_angles = np.arange(
            self.search_start_deg,
            self.search_end_deg
            - self.search_step_deg,
            -self.search_step_deg,
        )

        for target_deg in search_angles:

            self.command_deg = float(
                target_deg
            )

            for _ in range(
                self.search_settle_steps
            ):

                self._fixed_cube_step(
                    cube_position,
                    cube_quaternion,
                )

            (
                lower_contact,
                upper_contact,
            ) = self._contact_flags()

            if (
                lower_contact
                and upper_contact
            ):

                candidates.append(
                    {
                        "command":
                            float(
                                target_deg
                            ),

                        "state":
                            self._save_state(),
                    }
                )

        if len(candidates) == 0:

            raise RuntimeError(
                "Skill 3 initialization failed: "
                "no two-sided grasp was found."
            )

        commands = np.array(
            [
                candidate["command"]
                for candidate
                in candidates
            ],
            dtype=np.float64,
        )

        self.grasp_region_min_deg = float(
            np.min(commands)
        )

        self.grasp_region_max_deg = float(
            np.max(commands)
        )

        center_command = float(
            np.median(commands)
        )

        selected_index = int(
            np.argmin(
                np.abs(
                    commands
                    - center_command
                )
            )
        )

        selected = candidates[
            selected_index
        ]

        self._restore_state(
            selected["state"]
        )

        self.grasp_command_deg = float(
            selected["command"]
        )

        (
            lower_contact,
            upper_contact,
        ) = self._contact_flags()

        if not (
            lower_contact
            and upper_contact
        ):

            raise RuntimeError(
                "Skill 3 initialization failed: "
                "saved grasp state lost contact."
            )

        self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 6
        ] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.base_grasp_state = (
            self._save_state()
        )


    def _apply_random_disturbance(self):

        self.command_disturbance_used = float(
            self.np_random.uniform(
                -self.command_disturbance_deg,
                self.command_disturbance_deg,
            )
        )

        self.command_deg = float(
            np.clip(
                self.grasp_command_deg
                + self.command_disturbance_used,
                self.min_angle_deg,
                self.max_angle_deg,
            )
        )

        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )

        vx = 0.0
        vy = 0.0

        vz = float(
            self.np_random.uniform(
                -self.vertical_velocity_down,
                self.vertical_velocity_up,
            )
        )

        self.velocity_disturbance_used = np.array(
            [
                vx,
                vy,
                vz,
            ],
            dtype=np.float64,
        )

        self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 3
        ] = self.velocity_disturbance_used

        self.data.qvel[
            self.cube_qvel + 3:
            self.cube_qvel + 6
        ] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )


    def _metrics(self):

        cube_pos = (
            self._cube_position()
        )

        cube_velocity = (
            self._cube_linear_velocity()
        )

        (
            lower_contact,
            upper_contact,
        ) = self._contact_flags()

        both_contact = bool(
            lower_contact
            and upper_contact
        )

        displacement = float(
            np.linalg.norm(
                cube_pos
                - self.initial_cube_position
            )
        )

        vertical_drop = float(
            self.initial_cube_position[2]
            - cube_pos[2]
        )

        cube_speed = float(
            np.linalg.norm(
                cube_velocity
            )
        )

        return {
            "cube_pos":
                cube_pos,

            "cube_velocity":
                cube_velocity,

            "lower_contact":
                lower_contact,

            "upper_contact":
                upper_contact,

            "both_contact":
                both_contact,

            "displacement":
                displacement,

            "vertical_drop":
                vertical_drop,

            "cube_speed":
                cube_speed,
        }


    def _get_obs(self):

        cube_pos = (
            self._cube_position()
        )

        cube_velocity = (
            self._cube_linear_velocity()
        )

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ].copy()

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ].copy()

        (
            lower_contact,
            upper_contact,
        ) = self._contact_flags()

        current_deg = (
            self._hinge_deg()
        )

        hinge_velocity = float(
            self.data.qvel[
                self.hinge_qvel
            ]
        )

        if self.grasp_command_deg is None:

            command_offset = 0.0

        else:

            command_offset = (
                self.command_deg
                - self.grasp_command_deg
            )

        return np.array(
            [
                current_deg / 20.0,

                hinge_velocity,

                self.command_deg / 20.0,

                command_offset,

                *(
                    cube_pos
                    - self.initial_cube_position
                ),

                *cube_velocity,

                *(
                    upper_tip
                    - cube_pos
                ),

                *(
                    lower_tip
                    - cube_pos
                ),

                float(
                    lower_contact
                ),

                float(
                    upper_contact
                ),
            ],
            dtype=np.float32,
        )


    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        if self.base_grasp_state is None:

            self._build_base_grasp()

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self._restore_state(
            self.base_grasp_state
        )

        self.initial_cube_position = (
            self._cube_position()
        )

        self._apply_random_disturbance()

        self.step_count = 0
        self.hold_counter = 0

        metrics = (
            self._metrics()
        )

        info = {
            "initial_grasp_found":
                True,

            "grasp_command_deg":
                float(
                    self.grasp_command_deg
                ),

            "grasp_region_min_deg":
                float(
                    self.grasp_region_min_deg
                ),

            "grasp_region_max_deg":
                float(
                    self.grasp_region_max_deg
                ),

            "command_disturbance_deg":
                float(
                    self.command_disturbance_used
                ),

            "velocity_disturbance":
                self.velocity_disturbance_used.copy(),

            "lower_contact":
                bool(
                    metrics[
                        "lower_contact"
                    ]
                ),

            "upper_contact":
                bool(
                    metrics[
                        "upper_contact"
                    ]
                ),
        }

        return (
            self._get_obs(),
            info,
        )


    def step(
        self,
        action,
    ):

        self.step_count += 1

        self._set_hand_pose()

        action_value = float(
            np.clip(
                action[0],
                -1.0,
                1.0,
            )
        )

        # ==================================================
        # V4:
        # Measure error BEFORE the action.
        # ==================================================

        previous_command_error = abs(
            self.command_deg
            - self.grasp_command_deg
        )

        # ==================================================
        # ACTION
        #
        # ±1 corresponds to ±0.25 degrees.
        # ==================================================

        self.command_deg += (
            action_value
            * self.max_delta_deg
        )

        self.command_deg = float(
            np.clip(
                self.command_deg,
                self.min_angle_deg,
                self.max_angle_deg,
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
                self.data,
            )

        # ==================================================
        # STATE AFTER ACTION
        # ==================================================

        metrics = (
            self._metrics()
        )

        lower_contact = bool(
            metrics[
                "lower_contact"
            ]
        )

        upper_contact = bool(
            metrics[
                "upper_contact"
            ]
        )

        both_contact = bool(
            metrics[
                "both_contact"
            ]
        )

        displacement = float(
            metrics[
                "displacement"
            ]
        )

        vertical_drop = float(
            metrics[
                "vertical_drop"
            ]
        )

        cube_speed = float(
            metrics[
                "cube_speed"
            ]
        )

        displacement_mm = (
            displacement
            * 1000.0
        )

        vertical_drop_mm = (
            vertical_drop
            * 1000.0
        )

        # ==================================================
        # COMMAND ERROR AFTER ACTION
        # ==================================================

        current_command_error = abs(
            self.command_deg
            - self.grasp_command_deg
        )

        command_error_improvement = (
            previous_command_error
            - current_command_error
        )

        # ==================================================
        # FAILURE
        # ==================================================

        dropped = bool(
            vertical_drop
            > self.drop_distance
            or
            displacement
            > self.max_displacement
        )

        # ==================================================
        # HOLD COUNTER
        # ==================================================

        if (
            both_contact
            and not dropped
        ):

            self.hold_counter += 1

        else:

            self.hold_counter = 0

        success = bool(
            self.hold_counter
            >= self.required_hold_steps
        )

        # ==================================================
        # SKILL 3 V4 REWARD
        #
        # Main objective:
        #
        # Learn to return the command toward the stable
        # grasp command as quickly as possible.
        # ==================================================

        reward = 0.0

        # ==================================================
        # 1. REWARD IMPROVEMENT IN COMMAND ERROR
        #
        # If the action moves closer to the stable grasp,
        # reward is positive.
        #
        # If the action moves farther away,
        # reward is negative.
        # ==================================================

        reward += (
            40.0
            * command_error_improvement
        )

        # ==================================================
        # 2. PENALIZE REMAINING COMMAND ERROR
        # ==================================================

        reward -= (
            5.0
            * current_command_error
        )

        # ==================================================
        # 3. STABLE COMMAND REGION BONUS
        # ==================================================

        if current_command_error <= 0.03:

            reward += 2.0

        elif current_command_error <= 0.06:

            reward += 1.0

        # ==================================================
        # 4. PHYSICAL CONTACT
        #
        # Command recovery alone is not enough.
        # The cube must still physically remain grasped.
        # ==================================================

        if both_contact:

            reward += 1.0

        elif (
            lower_contact
            or upper_contact
        ):

            reward -= 1.0

        else:

            reward -= 3.0

        # ==================================================
        # 5. SURVIVAL
        # ==================================================

        if not dropped:

            reward += 0.25

        # ==================================================
        # 6. SMALL ACTION PENALTY
        #
        # Prevent unnecessary oscillation once recovered.
        # ==================================================

        reward -= (
            0.002
            * abs(
                action_value
            )
        )

        # ==================================================
        # 7. FAILURE / SUCCESS
        # ==================================================

        if dropped:

            reward -= 30.0

        if success:

            reward += 50.0

        # ==================================================
        # TERMINATION
        # ==================================================

        terminated = bool(
            success
            or dropped
        )

        truncated = bool(
            self.step_count
            >= self.max_steps
        )

        info = {
            "hinge_deg":
                float(
                    self._hinge_deg()
                ),

            "command_deg":
                float(
                    self.command_deg
                ),

            "grasp_command_deg":
                float(
                    self.grasp_command_deg
                ),

            "command_error_deg":
                float(
                    current_command_error
                ),

            "previous_command_error_deg":
                float(
                    previous_command_error
                ),

            "command_error_improvement_deg":
                float(
                    command_error_improvement
                ),

            "vertical_drop_mm":
                float(
                    vertical_drop_mm
                ),

            "displacement_mm":
                float(
                    displacement_mm
                ),

            "cube_speed":
                float(
                    cube_speed
                ),

            "lower_contact":
                bool(
                    lower_contact
                ),

            "upper_contact":
                bool(
                    upper_contact
                ),

            "both_contact":
                bool(
                    both_contact
                ),

            "hold_steps":
                int(
                    self.hold_counter
                ),

            "dropped":
                bool(
                    dropped
                ),

            "success":
                bool(
                    success
                ),
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )


if __name__ == "__main__":

    NUM_EPISODES = 100

    env = ChopstickSkill3Env()

    successes = 0
    drops = 0

    episode_lengths = []
    max_displacements = []
    max_vertical_drops = []

    print()
    print("====================================")
    print("SKILL 3 V4 ZERO-ACTION BASELINE")
    print("====================================")

    for episode in range(
        NUM_EPISODES
    ):

        obs, reset_info = env.reset(
            seed=3000 + episode
        )

        episode_max_displacement = 0.0
        episode_max_drop = 0.0

        final_info = None

        for step in range(
            env.max_steps
        ):

            (
                obs,
                reward,
                terminated,
                truncated,
                final_info,
            ) = env.step(
                np.array(
                    [0.0],
                    dtype=np.float32,
                )
            )

            episode_max_displacement = max(
                episode_max_displacement,
                final_info[
                    "displacement_mm"
                ],
            )

            episode_max_drop = max(
                episode_max_drop,
                final_info[
                    "vertical_drop_mm"
                ],
            )

            if (
                terminated
                or truncated
            ):

                break

        episode_lengths.append(
            step + 1
        )

        max_displacements.append(
            episode_max_displacement
        )

        max_vertical_drops.append(
            episode_max_drop
        )

        if final_info["success"]:

            successes += 1

        if final_info["dropped"]:

            drops += 1

        if episode < 10:

            print()

            print(
                f"Episode {episode + 1}"
            )

            print(
                "Disturbance:",
                f"{reset_info['command_disturbance_deg']:+.3f} deg",
            )

            print(
                "Steps:",
                step + 1,
            )

            print(
                "Final command:",
                f"{final_info['command_deg']:.3f} deg",
            )

            print(
                "Command error:",
                f"{final_info['command_error_deg']:.3f} deg",
            )

            print(
                "Max displacement:",
                f"{episode_max_displacement:.3f} mm",
            )

            print(
                "Max vertical drop:",
                f"{episode_max_drop:.3f} mm",
            )

            print(
                "Success:",
                final_info["success"],
            )

    success_rate = (
        successes
        / NUM_EPISODES
        * 100.0
    )

    print()
    print("====================================")
    print("FINAL ZERO-ACTION BASELINE")
    print("====================================")

    print(
        f"Successes: "
        f"{successes}/{NUM_EPISODES}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.1f}%"
    )

    print(
        f"Drops: "
        f"{drops}/{NUM_EPISODES}"
    )

    print(
        "Average episode length:",
        f"{np.mean(episode_lengths):.1f}",
    )

    print(
        "Average max displacement:",
        f"{np.mean(max_displacements):.3f} mm",
    )

    print(
        "Average max vertical drop:",
        f"{np.mean(max_vertical_drops):.3f} mm",
    )

    print()

    print(
        "Base grasp command:",
        f"{env.grasp_command_deg:.3f} deg",
    )

    print(
        "Maximum action authority:",
        f"{env.max_delta_deg:.3f} deg/step",
    )

    print(
        "Cached grasp initialization: ON"
    )

    env.close()