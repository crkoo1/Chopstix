import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickSkill3Env(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # MODEL
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_skill3_scene.xml"
        )

        self.data = mujoco.MjData(
            self.model
        )


        # ==================================================
        # HELPERS
        # ==================================================

        def actuator_id(name):

            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name
            )


        def site_id(name):

            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name
            )


        def body_id(name):

            return mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name
            )


        # ==================================================
        # OBJECT IDS
        # ==================================================

        self.lower_tip_id = site_id(
            "lower_tip"
        )

        self.upper_tip_id = site_id(
            "upper_tip"
        )

        self.cube_body_id = body_id(
            "grasp_cube"
        )


        # ==================================================
        # CHOPSTICK HINGE
        # ==================================================

        self.hinge_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "upper_chopstick_hinge"
        )


        self.hinge_qpos = (
            self.model.jnt_qposadr[
                self.hinge_id
            ]
        )


        self.hinge_qvel = (
            self.model.jnt_dofadr[
                self.hinge_id
            ]
        )


        # ==================================================
        # CUBE FREE JOINT
        # ==================================================

        self.cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "cube_freejoint"
        )


        self.cube_qpos = (
            self.model.jnt_qposadr[
                self.cube_joint_id
            ]
        )


        self.cube_qvel = (
            self.model.jnt_dofadr[
                self.cube_joint_id
            ]
        )


        # ==================================================
        # CHOPSTICK SERVO
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
        # -1 = close
        #  0 = maintain
        # +1 = open
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
        # 0       hinge angle
        # 1       hinge velocity
        # 2       servo command
        #
        # 3-5     cube displacement
        # 6-8     cube linear velocity
        #
        # 9-11    upper tip relative to cube
        # 12-14   lower tip relative to cube
        #
        # 15      lower contact
        # 16      upper contact
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )


        # ==================================================
        # CONTROL
        # ==================================================

        self.frame_skip = 5

        self.max_steps = 150

        self.min_angle_deg = 8.0

        self.max_angle_deg = 14.0

        self.max_delta_deg = 0.15


        # ==================================================
        # HOLD TASK
        # ==================================================

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


        # Cube failure thresholds
        self.drop_distance = 0.015

        self.max_displacement = 0.025


        # ==================================================
        # GEOMETRY
        # ==================================================

        self.stick_radius = 0.004

        self.cube_half_width = 0.005

        self.lower_preload = 0.0005


        # ==================================================
        # GRASP SEARCH
        # ==================================================

        self.search_start_deg = 14.0

        self.search_end_deg = 8.0

        self.search_step_deg = 0.05

        self.search_settle_steps = 25


        # ==================================================
        # INTERNAL STATE
        # ==================================================

        self.command_deg = 14.0

        self.step_count = 0

        self.hold_counter = 0


        self.initial_cube_position = np.zeros(
            3,
            dtype=np.float64
        )


        self.grasp_command_deg = None

        self.grasp_region_min_deg = None

        self.grasp_region_max_deg = None

        self.initial_grasp_found = False


    # ======================================================
    # HAND POSE
    # ======================================================

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


    # ======================================================
    # HINGE
    # ======================================================

    def _hinge_deg(self):

        return float(
            np.degrees(
                self.data.qpos[
                    self.hinge_qpos
                ]
            )
        )


    # ======================================================
    # SET CUBE POSE
    # ======================================================

    def _set_cube_pose(
        self,
        position,
        quaternion
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


    # ======================================================
    # MATRIX -> QUATERNION
    # ======================================================

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
                qz
            ],
            dtype=np.float64
        )


        quat /= np.linalg.norm(
            quat
        )


        return quat


    # ======================================================
    # COMPUTE CUBE POSE
    # ======================================================

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
                1.0
            ],
            dtype=np.float64
        )


        local_x = np.cross(
            gap_axis,
            world_up
        )


        local_x /= np.linalg.norm(
            local_x
        )


        local_z = np.cross(
            local_x,
            gap_axis
        )


        local_z /= np.linalg.norm(
            local_z
        )


        rotation = np.column_stack(
            (
                local_x,
                gap_axis,
                local_z
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
            quaternion
        )


    # ======================================================
    # CONTACT DETECTION
    # ======================================================

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
                ]
            )


            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.geom_bodyid[
                    contact.geom2
                ]
            )


            bodies = {
                body1,
                body2
            }


            if (
                "lower_chopstick"
                in bodies
                and
                "grasp_cube"
                in bodies
            ):

                lower_contact = True


            if (
                "upper_chopstick"
                in bodies
                and
                "grasp_cube"
                in bodies
            ):

                upper_contact = True


        return (
            lower_contact,
            upper_contact
        )


    # ======================================================
    # CUBE STATE
    # ======================================================

    def _cube_position(self):

        return self.data.xpos[
            self.cube_body_id
        ].copy()


    def _cube_linear_velocity(self):

        return self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 3
        ].copy()


    # ======================================================
    # FIXED-CUBE PHYSICS STEP
    #
    # Used ONLY during initialization/search.
    # ======================================================

    def _fixed_cube_step(
        self,
        cube_position,
        cube_quaternion
    ):

        self._set_hand_pose()


        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )


        self._set_cube_pose(
            cube_position,
            cube_quaternion
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


        mujoco.mj_step(
            self.model,
            self.data
        )


        self._set_cube_pose(
            cube_position,
            cube_quaternion
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


    # ======================================================
    # SAVE CURRENT MUJOCO STATE
    # ======================================================

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

            "hinge_deg":
                float(
                    self._hinge_deg()
                )
        }


    # ======================================================
    # RESTORE SAVED MUJOCO STATE
    # ======================================================

    def _restore_state(
        self,
        state
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
            state[
                "command_deg"
            ]
        )


        # Cube begins release with zero velocity.
        self.data.qvel[
            self.cube_qvel:
            self.cube_qvel + 6
        ] = 0.0


        mujoco.mj_forward(
            self.model,
            self.data
        )


    # ======================================================
    # FIND AND SAVE REAL GRASP STATE
    # ======================================================

    def _find_grasp_state(
        self,
        cube_position,
        cube_quaternion
    ):

        candidates = []


        search_angles = np.arange(
            self.search_start_deg,
            self.search_end_deg
            - self.search_step_deg,
            -self.search_step_deg
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
                    cube_quaternion
                )


            (
                lower_contact,
                upper_contact
            ) = self._contact_flags()


            both_contact = bool(
                lower_contact
                and upper_contact
            )


            if both_contact:

                candidates.append(
                    {
                        "command":
                            float(
                                target_deg
                            ),

                        "state":
                            self._save_state()
                    }
                )


        # ==================================================
        # NO GRASP FOUND
        # ==================================================

        if len(
            candidates
        ) == 0:

            self.initial_grasp_found = False

            self.grasp_command_deg = None

            self.grasp_region_min_deg = None

            self.grasp_region_max_deg = None

            return False


        # ==================================================
        # CONTACT REGION
        # ==================================================

        commands = np.array(
            [
                candidate[
                    "command"
                ]
                for candidate
                in candidates
            ],
            dtype=np.float64
        )


        self.grasp_region_min_deg = float(
            np.min(
                commands
            )
        )


        self.grasp_region_max_deg = float(
            np.max(
                commands
            )
        )


        # ==================================================
        # SELECT CENTRAL SUCCESSFUL CANDIDATE
        #
        # Key change:
        #
        # We DO NOT recreate this configuration.
        #
        # We restore the exact qpos/qvel/ctrl snapshot
        # from a state that was already verified to have
        # two physical contacts.
        # ==================================================

        center_command = float(
            np.median(
                commands
            )
        )


        selected_index = int(
            np.argmin(
                np.abs(
                    commands
                    - center_command
                )
            )
        )


        selected_candidate = (
            candidates[
                selected_index
            ]
        )


        self._restore_state(
            selected_candidate[
                "state"
            ]
        )


        self.grasp_command_deg = float(
            selected_candidate[
                "command"
            ]
        )


        # ==================================================
        # VERIFY RESTORED STATE
        # ==================================================

        (
            lower_contact,
            upper_contact
        ) = self._contact_flags()


        self.initial_grasp_found = bool(
            lower_contact
            and upper_contact
        )


        # ==================================================
        # FALLBACK
        #
        # If numerical recomputation somehow destroys the
        # selected contact state, test every saved candidate
        # until one reproduces correctly.
        # ==================================================

        if not self.initial_grasp_found:

            for candidate in candidates:

                self._restore_state(
                    candidate[
                        "state"
                    ]
                )


                (
                    lower_contact,
                    upper_contact
                ) = self._contact_flags()


                if (
                    lower_contact
                    and upper_contact
                ):

                    self.grasp_command_deg = float(
                        candidate[
                            "command"
                        ]
                    )

                    self.initial_grasp_found = True

                    break


        return self.initial_grasp_found


    # ======================================================
    # OBSERVATION
    # ======================================================

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
            upper_contact
        ) = self._contact_flags()


        current_deg = (
            self._hinge_deg()
        )


        hinge_velocity = float(
            self.data.qvel[
                self.hinge_qvel
            ]
        )


        return np.array(
            [
                current_deg / 20.0,

                hinge_velocity,

                self.command_deg / 20.0,

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
                )
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


        self.initial_grasp_found = False

        self.grasp_command_deg = None

        self.grasp_region_min_deg = None

        self.grasp_region_max_deg = None


        # ==================================================
        # MOVE CUBE OUT OF THE WAY
        # ==================================================

        far_position = np.array(
            [
                0.0,
                0.0,
                1.0
            ],
            dtype=np.float64
        )


        identity_quaternion = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            dtype=np.float64
        )


        self._set_cube_pose(
            far_position,
            identity_quaternion
        )


        # ==================================================
        # SETTLE HAND OPEN
        # ==================================================

        self.command_deg = 14.0


        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
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
                self.command_deg
            )


            self._set_cube_pose(
                far_position,
                identity_quaternion
            )


            mujoco.mj_step(
                self.model,
                self.data
            )


        # ==================================================
        # DETERMINE CUBE GRASP POSITION
        # ==================================================

        (
            cube_position,
            cube_quaternion
        ) = self._compute_cube_pose()


        self._set_cube_pose(
            cube_position,
            cube_quaternion
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


        # ==================================================
        # SEARCH FOR AND RESTORE EXACT GRASP STATE
        # ==================================================

        grasp_found = (
            self._find_grasp_state(
                cube_position,
                cube_quaternion
            )
        )


        # ==================================================
        # THIS IS NOW THE RELEASE POSITION
        #
        # From here onward the cube is NOT pinned.
        # ==================================================

        self.initial_cube_position = (
            self._cube_position()
        )


        self.step_count = 0

        self.hold_counter = 0


        (
            lower_contact,
            upper_contact
        ) = self._contact_flags()


        info = {

            "initial_grasp_found":
                bool(
                    grasp_found
                ),

            "grasp_command_deg":
                self.grasp_command_deg,

            "grasp_region_min_deg":
                self.grasp_region_min_deg,

            "grasp_region_max_deg":
                self.grasp_region_max_deg,

            "lower_contact":
                bool(
                    lower_contact
                ),

            "upper_contact":
                bool(
                    upper_contact
                )
        }


        return (
            self._get_obs(),
            info
        )


    # ======================================================
    # DYNAMIC STEP
    # ======================================================

    def step(
        self,
        action
    ):

        self.step_count += 1


        self._set_hand_pose()


        action_value = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )


        self.command_deg += (
            action_value
            * self.max_delta_deg
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
        # FREE-DYNAMIC CUBE
        # ==================================================

        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )


        # ==================================================
        # STATE
        # ==================================================

        cube_pos = (
            self._cube_position()
        )


        cube_velocity = (
            self._cube_linear_velocity()
        )


        (
            lower_contact,
            upper_contact
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


        # ==================================================
        # DROP
        # ==================================================

        dropped = bool(
            vertical_drop
            > self.drop_distance
            or
            displacement
            > self.max_displacement
        )


        # ==================================================
        # HOLD
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
        # REWARD
        # ==================================================

        reward = 0.0


        if both_contact:

            reward += 2.0

        else:

            reward -= 1.0


        reward -= (
            100.0
            * max(
                vertical_drop,
                0.0
            )
        )


        reward -= (
            25.0
            * displacement
        )


        reward -= (
            0.1
            * cube_speed
        )


        reward -= (
            0.005
            * abs(
                action_value
            )
        )


        if dropped:

            reward -= 25.0


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

            "vertical_drop_mm":
                float(
                    vertical_drop
                    * 1000.0
                ),

            "displacement_mm":
                float(
                    displacement
                    * 1000.0
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
                )
        }


        return (
            self._get_obs(),
            float(
                reward
            ),
            terminated,
            truncated,
            info
        )


# ==========================================================
# BUILT-IN DYNAMIC HOLD TEST
# ==========================================================

if __name__ == "__main__":

    env = ChopstickSkill3Env()


    print(
        "\n===================================="
    )

    print(
        "SKILL 3 DYNAMIC HOLD TEST"
    )

    print(
        "===================================="
    )


    obs, reset_info = env.reset()


    print(
        "\nAutomatic grasp search"
    )


    print(
        "Grasp found:",
        reset_info[
            "initial_grasp_found"
        ]
    )


    print(
        "Contact region:",
        reset_info[
            "grasp_region_min_deg"
        ],
        "to",
        reset_info[
            "grasp_region_max_deg"
        ],
        "deg"
    )


    print(
        "Selected command:",
        reset_info[
            "grasp_command_deg"
        ],
        "deg"
    )


    print(
        "\nInitial release state"
    )


    print(
        "Hinge:",
        f"{env._hinge_deg():.3f} deg"
    )


    print(
        "Command:",
        f"{env.command_deg:.3f} deg"
    )


    print(
        "Lower contact:",
        reset_info[
            "lower_contact"
        ]
    )


    print(
        "Upper contact:",
        reset_info[
            "upper_contact"
        ]
    )


    print(
        "Required hold:",
        env.required_hold_steps,
        "steps"
    )


    print(
        "Required hold time:",
        f"{env.required_hold_steps * env.control_dt:.3f} s"
    )


    # ======================================================
    # STOP IF INITIAL GRASP DOES NOT EXIST
    # ======================================================

    if not reset_info[
        "initial_grasp_found"
    ]:

        print(
            "\nFAIL: exact two-sided initial "
            "contact could not be restored."
        )

        env.close()

        raise SystemExit


    # ======================================================
    # ZERO ACTION
    #
    # Maintain selected servo command and see whether
    # friction/contact actually supports the cube.
    # ======================================================

    final_info = None


    for step in range(
        env.max_steps
    ):

        (
            obs,
            reward,
            terminated,
            truncated,
            final_info
        ) = env.step(
            np.array(
                [0.0],
                dtype=np.float32
            )
        )


        if (
            step < 10
            or
            (step + 1) % 10 == 0
            or
            terminated
        ):

            print(
                f"Step {step + 1:3d} | "
                f"hinge {final_info['hinge_deg']:6.3f} | "
                f"drop {final_info['vertical_drop_mm']:7.3f} mm | "
                f"move {final_info['displacement_mm']:7.3f} mm | "
                f"lower {str(final_info['lower_contact']):5s} | "
                f"upper {str(final_info['upper_contact']):5s} | "
                f"hold {final_info['hold_steps']:3d}"
            )


        if (
            terminated
            or truncated
        ):

            break


    # ======================================================
    # FINAL RESULT
    # ======================================================

    print(
        "\n===================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "===================================="
    )


    print(
        "Steps:",
        step + 1
    )


    print(
        "Vertical drop:",
        f"{final_info['vertical_drop_mm']:.3f} mm"
    )


    print(
        "Total displacement:",
        f"{final_info['displacement_mm']:.3f} mm"
    )


    print(
        "Lower contact:",
        final_info[
            "lower_contact"
        ]
    )


    print(
        "Upper contact:",
        final_info[
            "upper_contact"
        ]
    )


    print(
        "Hold:",
        final_info[
            "hold_steps"
        ]
    )


    print(
        "Dropped:",
        final_info[
            "dropped"
        ]
    )


    print(
        "Success:",
        final_info[
            "success"
        ]
    )


    if final_info[
        "success"
    ]:

        print(
            "\nPASS: free cube can be held "
            "under gravity."
        )

        print(
            "Skill 3 mechanics are valid."
        )


    else:

        print(
            "\nInitial two-sided contact was valid, "
            "but the cube slipped after release."
        )

        print(
            "Next issue: clamping force / friction."
        )


    env.close()