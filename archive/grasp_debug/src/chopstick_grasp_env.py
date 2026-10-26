import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickGraspEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # ==================================================
        # LOAD MODEL
        # ==================================================

        self.model = mujoco.MjModel.from_xml_path(
            "models/shadow_hand/chopstick_scene.xml"
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
        # CHOPSTICK + OBJECT IDS
        # ==================================================

        self.lower_tip_id = site_id(
            "lower_tip"
        )

        self.upper_tip_id = site_id(
            "upper_tip"
        )

        self.object_id = body_id(
            "grasp_object"
        )


        # ==================================================
        # UPPER CHOPSTICK HINGE
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
        # OBJECT VERTICAL JOINT
        # ==================================================

        self.object_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "object_vertical_joint"
        )

        self.object_qpos = self.model.jnt_qposadr[
            self.object_joint_id
        ]

        self.object_qvel = self.model.jnt_dofadr[
            self.object_joint_id
        ]


        # ==================================================
        # ACTUATORS
        # ==================================================

        # Thumb
        self.THJ5 = actuator_id("rh_A_THJ5")
        self.THJ4 = actuator_id("rh_A_THJ4")
        self.THJ3 = actuator_id("rh_A_THJ3")
        self.THJ2 = actuator_id("rh_A_THJ2")
        self.THJ1 = actuator_id("rh_A_THJ1")

        # Index
        self.FFJ4 = actuator_id("rh_A_FFJ4")
        self.FFJ3 = actuator_id("rh_A_FFJ3")
        self.FFJ0 = actuator_id("rh_A_FFJ0")

        # Middle
        self.MFJ4 = actuator_id("rh_A_MFJ4")
        self.MFJ3 = actuator_id("rh_A_MFJ3")
        self.MFJ0 = actuator_id("rh_A_MFJ0")

        # Ring
        self.RFJ4 = actuator_id("rh_A_RFJ4")
        self.RFJ3 = actuator_id("rh_A_RFJ3")
        self.RFJ0 = actuator_id("rh_A_RFJ0")

        # Little
        self.LFJ5 = actuator_id("rh_A_LFJ5")
        self.LFJ4 = actuator_id("rh_A_LFJ4")
        self.LFJ3 = actuator_id("rh_A_LFJ3")
        self.LFJ0 = actuator_id("rh_A_LFJ0")


        # ==================================================
        # ACTION SPACE
        #
        # action[0] -> FFJ3
        # action[1] -> MFJ3
        #
        # FFJ0 and MFJ0 stay fixed.
        # ==================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )


        # ==================================================
        # OBSERVATION SPACE
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )


        # ==================================================
        # SETTINGS
        # ==================================================

        self.frame_skip = 5

        self.max_steps = 150

        self.required_contact_steps = 5

        self.step_count = 0

        self.both_contact_counter = 0

        self.initial_object_z = 0.0


        # Ball is placed slightly toward lower chopstick
        self.object_preload_toward_lower = 0.0005


    # ======================================================
    # FIXED GRIP POSE
    # ======================================================

    def _set_fixed_grip(self):

        # Thumb
        self.data.ctrl[self.THJ5] = 0.40
        self.data.ctrl[self.THJ4] = 0.90
        self.data.ctrl[self.THJ3] = 0.08
        self.data.ctrl[self.THJ2] = 0.45
        self.data.ctrl[self.THJ1] = 0.95


        # Index
        self.data.ctrl[self.FFJ4] = 0.00

        # FFJ0 stays fixed
        self.data.ctrl[self.FFJ0] = 0.95


        # Middle
        self.data.ctrl[self.MFJ4] = 0.00

        # MFJ0 stays fixed
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
    # CONTACT DETECTION
    # ======================================================

    def _contact_flags(self):

        lower_contact = False
        upper_contact = False


        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            body1_id = self.model.geom_bodyid[
                contact.geom1
            ]

            body2_id = self.model.geom_bodyid[
                contact.geom2
            ]


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
                "grasp_object" in bodies
            ):
                lower_contact = True


            if (
                "upper_chopstick" in bodies
                and
                "grasp_object" in bodies
            ):
                upper_contact = True


        return (
            lower_contact,
            upper_contact
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

        object_pos = self.data.xpos[
            self.object_id
        ].copy()


        upper_relative = (
            upper_tip
            - object_pos
        )

        lower_relative = (
            lower_tip
            - object_pos
        )


        hinge_angle = self.data.qpos[
            self.hinge_qpos
        ]

        hinge_velocity = self.data.qvel[
            self.hinge_qvel
        ]


        object_z_displacement = (
            object_pos[2]
            - self.initial_object_z
        )


        object_vertical_velocity = (
            self.data.qvel[
                self.object_qvel
            ]
        )


        return np.array(
            [
                hinge_angle,
                hinge_velocity,

                *upper_relative,
                *lower_relative,

                object_z_displacement,
                object_vertical_velocity
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


        # ==================================================
        # RESET MUJOCO
        # ==================================================

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.data.ctrl[:] = 0.0


        # ==================================================
        # MOVE OBJECT AWAY WHILE HAND SETTLES
        # ==================================================

        self.model.body_pos[
            self.object_id
        ] = np.array(
            [0.0, 0.0, 1.0]
        )


        self.data.qpos[
            self.object_qpos
        ] = 0.0

        self.data.qvel[
            self.object_qvel
        ] = 0.0


        mujoco.mj_forward(
            self.model,
            self.data
        )


        # ==================================================
        # SETTLE HAND
        # ==================================================

        for _ in range(1000):

            self._set_fixed_grip()


            # Nominal index posture

            self.data.ctrl[
                self.FFJ3
            ] = 0.72


            # Nominal middle posture

            self.data.ctrl[
                self.MFJ3
            ] = 0.75


            mujoco.mj_step(
                self.model,
                self.data
            )


        # ==================================================
        # MEASURE CURRENT TIPS
        # ==================================================

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ].copy()

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ].copy()


        midpoint = (
            lower_tip
            + upper_tip
        ) / 2.0


        # ==================================================
        # DIRECTION TOWARD LOWER CHOPSTICK
        # ==================================================

        toward_lower = (
            lower_tip
            - upper_tip
        )

        tip_separation = np.linalg.norm(
            toward_lower
        )


        toward_lower = (
            toward_lower
            / tip_separation
        )


        # ==================================================
        # PLACE BALL
        # ==================================================

        object_target = (
            midpoint
            +
            self.object_preload_toward_lower
            * toward_lower
        )


        self.model.body_pos[
            self.object_id
        ] = object_target


        self.data.qpos[
            self.object_qpos
        ] = 0.0

        self.data.qvel[
            self.object_qvel
        ] = 0.0


        mujoco.mj_forward(
            self.model,
            self.data
        )


        # ==================================================
        # RESET COUNTERS
        # ==================================================

        self.step_count = 0

        self.both_contact_counter = 0


        self.initial_object_z = float(
            self.data.xpos[
                self.object_id
            ][2]
        )


        # ==================================================
        # INITIAL INFO
        # ==================================================

        object_pos = self.data.xpos[
            self.object_id
        ].copy()


        lower_distance = np.linalg.norm(
            lower_tip
            - object_pos
        )

        upper_distance = np.linalg.norm(
            upper_tip
            - object_pos
        )


        observation = self._get_obs()


        info = {

            "lower_distance_mm":
                float(
                    lower_distance * 1000
                ),

            "upper_distance_mm":
                float(
                    upper_distance * 1000
                ),

            "tip_separation_mm":
                float(
                    tip_separation * 1000
                )
        }


        return (
            observation,
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


        # Fixed parts of grip
        self._set_fixed_grip()


        action = np.clip(
            action,
            -1.0,
            1.0
        )


        # ==================================================
        # INDEX CONTROL
        # ==================================================

        ffj3_min = 0.40
        ffj3_max = 1.30


        ffj3_command = (
            ffj3_min
            +
            (action[0] + 1.0)
            * 0.5
            * (
                ffj3_max
                - ffj3_min
            )
        )


        self.data.ctrl[
            self.FFJ3
        ] = ffj3_command


        # ==================================================
        # MIDDLE FINGER CONTROL
        # ==================================================

        mfj3_min = 0.30
        mfj3_max = 1.30


        mfj3_command = (
            mfj3_min
            +
            (action[1] + 1.0)
            * 0.5
            * (
                mfj3_max
                - mfj3_min
            )
        )


        self.data.ctrl[
            self.MFJ3
        ] = mfj3_command


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
        # CONTACTS
        # ==================================================

        (
            lower_contact,
            upper_contact
        ) = self._contact_flags()


        both_contact = bool(
            lower_contact
            and upper_contact
        )


        if both_contact:

            self.both_contact_counter += 1

        else:

            self.both_contact_counter = 0


        # ==================================================
        # DISTANCES
        # ==================================================

        lower_tip = self.data.site_xpos[
            self.lower_tip_id
        ]

        upper_tip = self.data.site_xpos[
            self.upper_tip_id
        ]

        object_pos = self.data.xpos[
            self.object_id
        ]


        lower_distance = np.linalg.norm(
            lower_tip
            - object_pos
        )

        upper_distance = np.linalg.norm(
            upper_tip
            - object_pos
        )


        contact_distance = 0.006


        lower_gap = max(
            0.0,
            lower_distance
            - contact_distance
        )

        upper_gap = max(
            0.0,
            upper_distance
            - contact_distance
        )


        # ==================================================
        # REWARD
        # ==================================================

        reward = 0.0


        reward -= 50.0 * (
            lower_gap
            + upper_gap
        )


        if lower_contact:
            reward += 2.0


        if upper_contact:
            reward += 2.0


        if both_contact:
            reward += 8.0


        reward -= (
            0.001
            * float(
                np.sum(
                    np.square(
                        action
                    )
                )
            )
        )


        # ==================================================
        # SUCCESS
        # ==================================================

        success = bool(
            self.both_contact_counter
            >= self.required_contact_steps
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


        observation = self._get_obs()


        info = {

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

            "lower_distance_mm":
                float(
                    lower_distance * 1000
                ),

            "upper_distance_mm":
                float(
                    upper_distance * 1000
                ),

            "ffj3_command":
                float(
                    ffj3_command
                ),

            # Keep this for compatibility with
            # the older debug script.
            "ffj0_command":
                0.95,

            "mfj3_command":
                float(
                    mfj3_command
                ),

            "success":
                bool(
                    success
                )
        }


        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )