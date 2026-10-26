import gymnasium as gym
from gymnasium import spaces

import mujoco
import numpy as np


class ChopstickServoEnv(gym.Env):

    def __init__(self):
        super().__init__()

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
        # SITES / BODY
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
        # UPPER HINGE
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
        # OBJECT JOINT
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
        # One normalized action:
        #
        # -1 -> 7.5 degrees
        # +1 -> 11.5 degrees
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
        # hinge angle
        # hinge velocity
        # upper relative xyz
        # lower relative xyz
        # object vertical displacement
        # object vertical velocity
        # ==================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )


        self.frame_skip = 5

        self.max_steps = 150

        self.required_contact_steps = 5

        self.step_count = 0

        self.both_contact_counter = 0

        self.initial_object_z = 0.0

        self.object_preload_toward_lower = 0.0005


        # Servo range used by RL
        self.min_target_deg = 7.5
        self.max_target_deg = 11.5


    # ======================================================
    # FIXED HUMAN-LIKE GRIP
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
    # CONTACT
    # ======================================================

    def _contact_flags(self):

        lower_contact = False
        upper_contact = False

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

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


        return np.array(
            [
                self.data.qpos[
                    self.hinge_qpos
                ],

                self.data.qvel[
                    self.hinge_qvel
                ],

                *(
                    upper_tip
                    - object_pos
                ),

                *(
                    lower_tip
                    - object_pos
                ),

                object_pos[2]
                - self.initial_object_z,

                self.data.qvel[
                    self.object_qvel
                ]
            ],
            dtype=np.float32
        )


    # ======================================================
    # ACTION -> TARGET ANGLE
    # ======================================================

    def _action_to_target(self, action):

        action_value = float(
            np.clip(
                action[0],
                -1.0,
                1.0
            )
        )

        target_deg = (
            self.min_target_deg
            +
            (action_value + 1.0)
            * 0.5
            * (
                self.max_target_deg
                - self.min_target_deg
            )
        )

        return target_deg


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
        # Move object away while hand settles
        # ----------------------------------------------

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


        # Hold chopstick initially open
        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            10.75
        )


        mujoco.mj_forward(
            self.model,
            self.data
        )


        # ----------------------------------------------
        # Settle hand
        # ----------------------------------------------

        for _ in range(1000):

            self._set_hand_pose()

            self.data.ctrl[
                self.servo_id
            ] = np.radians(
                10.75
            )

            mujoco.mj_step(
                self.model,
                self.data
            )


        # ----------------------------------------------
        # Measure actual tips
        # ----------------------------------------------

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


        toward_lower = (
            lower_tip
            - upper_tip
        )

        tip_separation = np.linalg.norm(
            toward_lower
        )

        toward_lower /= tip_separation


        # ----------------------------------------------
        # Put object into staged grasp location
        # ----------------------------------------------

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


        self.step_count = 0

        self.both_contact_counter = 0

        self.initial_object_z = float(
            self.data.xpos[
                self.object_id
            ][2]
        )


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


        # ----------------------------------------------
        # RL controls target hinge angle
        # ----------------------------------------------

        target_deg = self._action_to_target(
            action
        )

        self.data.ctrl[
            self.servo_id
        ] = np.radians(
            target_deg
        )


        # ----------------------------------------------
        # Physics
        # ----------------------------------------------

        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data
            )


        # ----------------------------------------------
        # Contacts
        # ----------------------------------------------

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


        # ----------------------------------------------
        # Geometry
        # ----------------------------------------------

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


        # ----------------------------------------------
        # Reward
        # ----------------------------------------------

        reward = 0.0


        # Upper chopstick should approach object
        upper_gap = max(
            0.0,
            upper_distance
            - 0.006
        )

        reward -= (
            100.0
            * upper_gap
        )


        # Preserve lower support
        if lower_contact:
            reward += 1.0


        # Upper reaches object
        if upper_contact:
            reward += 4.0


        # True pinch
        if both_contact:
            reward += 8.0


        success = bool(
            self.both_contact_counter
            >= self.required_contact_steps
        )


        if success:
            reward += 15.0


        terminated = success

        truncated = bool(
            self.step_count
            >= self.max_steps
        )


        info = {
            "target_deg":
                float(target_deg),

            "hinge_deg":
                float(
                    np.degrees(
                        self.data.qpos[
                            self.hinge_qpos
                        ]
                    )
                ),

            "lower_distance_mm":
                float(
                    lower_distance * 1000
                ),

            "upper_distance_mm":
                float(
                    upper_distance * 1000
                ),

            "lower_contact":
                bool(lower_contact),

            "upper_contact":
                bool(upper_contact),

            "both_contact":
                bool(both_contact),

            "success":
                bool(success)
        }


        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info
        )