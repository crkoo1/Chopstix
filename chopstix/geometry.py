"""Rotation helpers and the fixed geometry shared with Makis' SuperDex framework (chops)."""

import numpy as np

# sr_description joint order, as in chops.dexgraspnet.loader. Grip files and actions use this order.
SR_JOINT_NAMES = [
    "rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1",
    "rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1",
    "rh_RFJ4", "rh_RFJ3", "rh_RFJ2", "rh_RFJ1",
    "rh_LFJ5", "rh_LFJ4", "rh_LFJ3", "rh_LFJ2", "rh_LFJ1",
    "rh_THJ5", "rh_THJ4", "rh_THJ3", "rh_THJ2", "rh_THJ1",
]  # fmt: skip
NUM_HAND_JOINTS = len(SR_JOINT_NAMES)

# Chopstick: tapered rod along +x, origin at its centre, thin (tip) end at +x (chops.assets.props).
CHOPSTICK_LENGTH = 0.23
CHOPSTICK_BASE_RADIUS = 0.0035
CHOPSTICK_TIP_RADIUS = 0.002
CHOPSTICK_MASS = 0.01

# Palm frame (rh_palm): fingers along +z, palm faces -y, +x is the thumb side.
# Palm down, fingers along world +x: palm x->world y, palm y->world z, palm z->world x.
PALM_DOWN = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def tip_offset() -> np.ndarray:
    return np.array([CHOPSTICK_LENGTH / 2, 0.0, 0.0])


def rotvec_to_mat(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def mat_to_rotvec(r: np.ndarray) -> np.ndarray:
    cos = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        b = (r + np.eye(3)) / 2
        axis = b[np.argmax(np.diag(b))]
        return axis / np.linalg.norm(axis) * angle
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * np.sin(angle))
    return axis * angle


def mat_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(r, dtype=float).ravel())
    return q


def quat_to_mat(q) -> np.ndarray:
    import mujoco

    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def rot6d(r: np.ndarray) -> np.ndarray:
    return r[:, :2].T.ravel()


def rot_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def chopstick_vertices(sections: int = 24) -> np.ndarray:
    """Vertices of the tapered chopstick (a truncated cone, so its convex hull is exact)."""
    half = CHOPSTICK_LENGTH / 2
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.stack([np.zeros_like(a), np.cos(a), np.sin(a)], axis=1)
    return np.vstack([ring * CHOPSTICK_BASE_RADIUS + [-half, 0, 0], ring * CHOPSTICK_TIP_RADIUS + [half, 0, 0]])
