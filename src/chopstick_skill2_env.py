import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickSkill2Env(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # MODEL
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_skill2_scene.xml"
        )

        self.data = mujoco.MjData(self.model)


        # ==================================================
        # HELPER FUNCTIONS
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
        # CHOPSTICK / CUBE IDS
        # ==================================================

        self.lower_tip_id = site_id(
            "lower_tip"
        )

        self.upper_tip_id = site_id(
            "upper_tip"
        )

        self.cube_id = body_id(
            "grasp_cube"
        )


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
        # ACTION SPACE
        #
        # -1 = close upper chopstick
        #  0 = hold
        # +1 = open upper chopstick
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )


        # ==================================================
        # OBSERVATION SPACE
        #
        # 0       hinge angle
        # 1       hinge velocity
        # 2       commanded angle
        # 3-5     upper tip relative to cube
        # 6-8     lower tip relative to cube
        # 9       lower contact
        # 10      upper contact
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )


        # ==================================================
        # ENVIRONMENT SETTINGS
        # ==================================================

        self.frame_skip = 5

        self.max_steps = 150

        self.required_hold_steps = 10


        # Servo command limits
        self.min_angle_deg = 8.0
        self.max_angle_deg = 14.0


        # Start fully open
        self.start_angle_deg = 14.0


        # Maximum servo command change per RL step
        self.max_delta_deg = 0.30


        # ==================================================
        # KNOWN MECHANICAL GRASP REGION
        #
        # Sanity test showed:
        #
        # 13.50 deg -> valid grasp
        # 13.25 deg -> valid grasp
        #
        # Center:
        # 13.375 deg
        #
        # This is ONLY used for reward shaping.
        #
        # True success still requires physical contact.
        # ==================================================

        self.grasp_center_deg = 13.375


        # ==================================================
        # GEOMETRY
        # ==================================================

        self.stick_radius = 0.004

        self.cube_half_width = 0.005

        self.lower_preload = 0.0002


        # ==================================================
        # INTERNAL STATE
        # ==================================================

        self.command_deg = (
            self.start_angle_deg
        )

        self.step_count = 0

        self.hold_counter = 0


    # ======================================================
    # FIXED HAND GRIP POSE
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

        # Little finger
        self.data.ctrl[self.LFJ5] = 0.15
        self.data.ctrl[self.LFJ4] = 0.00
        self.data.ctrl[self.LFJ3] = 1.05
        self.data.ctrl[self.LFJ0] = 1.25


    # ======================================================
    # CURRENT HINGE ANGLE
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
    # CONTACT DETECTION
    # ======================================================

    def _contact_flags(self):

        lower_contact = False
        upper_contact = False


        for i in range(
            self.data.ncon
        ):

            contact = self.data.contact[i]


            body1_id = (
                self.model.geom_bodyid[
                    contact.geom1
                ]
            )

            body2_id = (
                self.model.geom_bodyid[
                    contact.geom2
                ]
            )


            body1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body1_id
            )

            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body2_id
            )


            bodies = {
                body1,
                body2
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
            upper_contact
        )


    # ======================================================
    # MATRIX TO QUATERNION
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


        quaternion = np.array(
            [
                qw,
                qx,
                qy,
                qz
            ],
            dtype=np.float64
        )


        quaternion /= np.linalg.norm(
            quaternion
        )


        return quaternion


    # ======================================================
    # AUTOMATIC CUBE PLACEMENT
    # ======================================================

    def _place_cube(self):

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ].copy()

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ].copy()


        # Direction from lower chopstick
        # toward upper chopstick.
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
            ]
        )


        # Construct cube orientation.
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


        # Place one cube face just against
        # the lower chopstick.
        lower_to_cube_center = (
            self.stick_radius
            + self.cube_half_width
            - self.lower_preload
        )


        cube_position = (
            lower_tip
            + gap_axis
            * lower_to_cube_center
        )


        self.model.body_pos[
            self.cube_id
        ] = cube_position


        self.model.body_quat[
            self.cube_id
        ] = quaternion


        mujoco.mj_forward(
            self.model,
            self.data
        )


    # ======================================================
    # DISTANCES
    # ======================================================

    def _distances(self):

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ]

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ]

        cube_pos = self.data.xpos[
            self.cube_id
        ]


        lower_distance = float(
            np.linalg.norm(
                lower_tip
                - cube_pos
            )
        )


        upper_distance = float(
            np.linalg.norm(
                upper_tip
                - cube_pos
            )
        )


        return (
            lower_distance,
            upper_distance
        )


    # ======================================================
    # OBSERVATION
    # ======================================================

    def _get_obs(self):

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ].copy()

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ].copy()

        cube_pos = self.data.xpos[
            self.cube_id
        ].copy()


        (
            lower_contact,
            upper_contact
        ) = self._contact_flags()


        current_deg = self._hinge_deg()


        velocity = float(
            self.data.qvel[
                self.hinge_qvel
            ]
        )


        return np.array(
            [
                current_deg / 20.0,

                velocity,

                self.command_deg / 20.0,

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


        # ----------------------------------------------
        # Temporarily move cube away.
        # ----------------------------------------------

        self.model.body_pos[
            self.cube_id
        ] = np.array(
            [
                0.0,
                0.0,
                1.0
            ]
        )


        # Start chopsticks open.
        self.command_deg = (
            self.start_angle_deg
        )


        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            self.command_deg
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


        # ----------------------------------------------
        # Allow hand + chopsticks to settle.
        # ----------------------------------------------

        for _ in range(600):

            self._set_hand_pose()


            self.data.ctrl[
                self.servo_id
            ] = np.radians(
                self.command_deg
            )


            mujoco.mj_step(
                self.model,
                self.data
            )


        # ----------------------------------------------
        # Place cube based on CURRENT chopstick
        # geometry.
        # ----------------------------------------------

        self._place_cube()


        self.step_count = 0

        self.hold_counter = 0


        return (
            self._get_obs(),
            {}
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
        # ACTION
        # ==================================================

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
        # SIMULATION
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


        (
            lower_contact,
            upper_contact
        ) = self._contact_flags()


        both_contact = bool(
            lower_contact
            and upper_contact
        )


        (
            lower_distance,
            upper_distance
        ) = self._distances()


        desired_distance = (
            self.stick_radius
            + self.cube_half_width
        )


        lower_gap = max(
            0.0,
            lower_distance
            - desired_distance
        )


        upper_gap = max(
            0.0,
            upper_distance
            - desired_distance
        )


        # ==================================================
        # HOLD COUNTER
        #
        # Success requires BOTH physical contacts
        # continuously for 10 steps.
        # ==================================================

        if both_contact:

            self.hold_counter += 1

        else:

            self.hold_counter = 0


        success = bool(
            self.hold_counter
            >= self.required_hold_steps
        )


        # ==================================================
        # REWARD V3
        # ==================================================
        #
        # Previous problem:
        #
        # PPO had almost no information about which
        # direction it should move from 14 degrees.
        #
        # We now provide a smooth shaping reward toward
        # the known grasp region.
        #
        # IMPORTANT:
        #
        # The angle DOES NOT define success.
        #
        # Success still requires actual MuJoCo contacts.
        # ==================================================

        grasp_error = abs(
            current_deg
            - self.grasp_center_deg
        )


        # --------------------------------------------------
        # ANGLE SHAPING
        #
        # Far from grasp center = larger penalty.
        #
        # 14.0 deg -> penalty
        # 13.8 deg -> smaller penalty
        # 13.4 deg -> almost zero penalty
        # --------------------------------------------------

        reward = (
            -2.0
            * grasp_error
        )


        # --------------------------------------------------
        # TIME PENALTY
        #
        # Prevent waiting forever.
        # --------------------------------------------------

        reward -= 0.05


        # --------------------------------------------------
        # ACTION PENALTY
        #
        # Encourage smooth control.
        # --------------------------------------------------

        reward -= (
            0.005
            * abs(
                action_value
            )
        )


        # --------------------------------------------------
        # LOWER CONTACT
        #
        # Lower support is useful.
        # --------------------------------------------------

        if lower_contact:

            reward += 1.0


        # --------------------------------------------------
        # UPPER CONTACT ALONE
        #
        # This was the exploit in Reward V1.
        #
        # Upper-only contact is therefore slightly
        # penalized.
        # --------------------------------------------------

        if (
            upper_contact
            and not lower_contact
        ):

            reward -= 0.25


        # --------------------------------------------------
        # TWO-SIDED PHYSICAL CONTACT
        # --------------------------------------------------

        if both_contact:

            reward += 5.0


        # --------------------------------------------------
        # SUCCESS
        #
        # Both contacts maintained for 10 steps.
        # --------------------------------------------------

        if success:

            reward += 50.0


        # ==================================================
        # TERMINATION
        # ==================================================

        terminated = bool(
            success
        )


        truncated = bool(
            self.step_count
            >= self.max_steps
        )


        # ==================================================
        # INFO
        # ==================================================

        info = {

            "hinge_deg":
                float(
                    current_deg
                ),

            "command_deg":
                float(
                    self.command_deg
                ),

            "grasp_error_deg":
                float(
                    grasp_error
                ),

            "lower_distance_mm":
                float(
                    lower_distance
                    * 1000.0
                ),

            "upper_distance_mm":
                float(
                    upper_distance
                    * 1000.0
                ),

            "lower_gap_mm":
                float(
                    lower_gap
                    * 1000.0
                ),

            "upper_gap_mm":
                float(
                    upper_gap
                    * 1000.0
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
# BUILT-IN MECHANICAL SANITY TEST
# ==========================================================

if __name__ == "__main__":

    env = ChopstickSkill2Env()


    print(
        "\n===================================="
    )

    print(
        "SKILL 2 CUBE SANITY TEST"
    )

    print(
        "===================================="
    )


    targets = np.arange(
        14.0,
        7.9,
        -0.25
    )


    any_both = False


    for target_deg in targets:

        obs, info = env.reset()


        for step in range(
            env.max_steps
        ):

            error = (
                target_deg
                - env.command_deg
            )


            action_value = np.clip(
                error
                / env.max_delta_deg,
                -1.0,
                1.0
            )


            (
                obs,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                np.array(
                    [
                        action_value
                    ],
                    dtype=np.float32
                )
            )


            if info[
                "both_contact"
            ]:

                any_both = True


            if (
                terminated
                or truncated
            ):

                break


        print(
            f"{target_deg:5.2f} deg | "
            f"actual {info['hinge_deg']:5.2f} | "
            f"lower {str(info['lower_contact']):5s} | "
            f"upper {str(info['upper_contact']):5s} | "
            f"both {str(info['both_contact']):5s} | "
            f"success {info['success']}"
        )


    print(
        "\n===================================="
    )


    print(
        "Any two-sided cube contact:",
        any_both
    )


    if any_both:

        print(
            "PASS: cube grasp is mechanically reachable."
        )

        print(
            "Reward V3 provides directional shaping."
        )


    else:

        print(
            "FAIL: cube grasp is not mechanically reachable."
        )


    env.close()