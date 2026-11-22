"""Step 2: re-form a chops tripod grip natively in MuJoCo (the MuJoCo counterpart of `chops.stage1.grip search`).

A SuperDex grip loaded as-is is not a valid MuJoCo state: SuperDex's contacts let the sticks rest 3-6 mm inside the
fingers and its joint gains are in different units, so in MuJoCo the fingers squeeze with ~6 N and the sticks creep.
This keeps the source grip's hand posture and stick placement and lets MuJoCo settle it:

  1. Press directions. At the source pose (sticks overlapping the fingers), for every (finger, stick) contact pair,
     the joint-space direction that drives that finger's contact points into that stick: sum of J^T n over the pair's
     contacts (J = the finger's positional Jacobian at the contact, n = the contact normal into the stick).
     The tripod: thumb pad / index pad / middle side on the upper stick; on the lower stick, the thumb's middle
     phalanx presses it towards the palm, with the ring fingertip near its middle.
  2. Relax (gravity off): sticks free but damped and softly clamped at the source poses; finger targets follow the
     joints. Overlaps push sticks and fingers apart until everything just touches.
  3. Pinch: each finger's target moves along sum over its pairs of (--pinch amount x unit direction), in rad.
  4. Release the clamps, settle (damped, then free), save joints, targets and stick poses (palm frame).

    .venv/Scripts/python.exe -m chopstix.tools.form_grip --source closed_v5 --save-as closed_v5_mj --test
    .venv/Scripts/python.exe -m chopstix.tools.form_grip --source closed_v5 --save-as try \
        --pinch th-upper=0.02 th-lower=0.06 ff-upper=0.02 mf-upper=0.02 rf-lower=0.05 --set noslip_iterations=10
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from chopstix.geometry import SR_JOINT_NAMES, mat_to_rotvec, rot_angle_deg
from chopstix.sim import STICKS, Sim, SimCfg, load_grip
from chopstix.tools.common import parse_overrides
from chopstix.tools.hold_test import PALM_POS

GRIPS = Path(__file__).resolve().parents[1] / "grips"
FINGERS = ("th", "ff", "mf", "rf", "lf")
FINGER_JOINTS = {f: np.array([i for i, j in enumerate(SR_JOINT_NAMES) if j.startswith(f"rh_{f.upper()}J")]) for f in FINGERS}
DEFAULT_PINCH = ["th-upper=0.02", "th-lower=0.05", "ff-upper=0.02", "mf-upper=0.02", "rf-lower=0.05"]


def press_directions(sim: Sim) -> dict[tuple[str, str], np.ndarray]:
    """(finger, stick) -> unit vector over the 22 hand joints (SR order) pressing the finger into the stick."""
    m, d = sim.model, sim.data
    jacp = np.zeros((3, m.nv))
    acc: dict = {}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for fb, sb, sign in ((b1, b2, 1.0), (b2, b1, -1.0)):  # normal points geom1 -> geom2
            finger, stick = sim._finger_of_body.get(fb), sim._stick_of_body.get(sb)
            if finger in FINGER_JOINTS and stick:
                mujoco.mj_jac(m, d, jacp, None, c.pos, fb)
                g = (jacp.T @ (sign * c.frame[:3]))[sim._jv] * max(-c.dist, 1e-4)
                acc[(finger, stick)] = acc.get((finger, stick), 0.0) + g
    out = {}
    for key, g in acc.items():
        mask = np.zeros(len(SR_JOINT_NAMES))
        mask[FINGER_JOINTS[key[0]]] = 1.0
        g = g * mask
        if np.linalg.norm(g) > 1e-9:
            out[key] = g / np.linalg.norm(g)
    return out


def _forces(sim):
    ff = sim.finger_forces()
    return "  ".join(f"{f} u{ff[f]['upper']:.2f}/l{ff[f]['lower']:.2f}" for f in ("th", "ff", "mf", "rf", "lf", "palm"))


def form(sim: Sim, src: dict, args, log=print) -> dict:
    dt = sim.model.opt.timestep
    sim.reset()
    sim.set_gravity(False)
    sim.set_palm(PALM_POS, src["palm_rot"])
    joints = src["joints"].copy()
    sim.set_joints(joints)
    for n in STICKS:
        sim.place_stick(n, *src["sticks"][n])
    sim.forward()
    pen = {f"{a}-{b}": round(v * 1000, 1) for (a, b), v in sim.penetration().items() if {"upper", "lower"} & {a, b}}
    log(f"  overlap at the source pose (mm): {pen}")
    dirs = press_directions(sim)
    pinch = {tuple(k.split("-")): float(v) for k, v in (item.split("=") for item in args.pinch)}
    for key in pinch:
        if key not in dirs:
            log(f"  warning: no {key[0]}-{key[1]} contact at the source pose; that pinch is skipped")
    push = np.zeros(len(SR_JOINT_NAMES))
    for key, amount in pinch.items():
        if key in dirs:
            push += amount * dirs[key]

    # Relax: soft clamps keep the sticks near the source poses; fingers yield.
    sim.set_stick_damping(*args.stick_damping)
    for n in STICKS:
        sim.model.eq_solref[sim._clamp_eq[n]] = args.clamp_solref
        sim.clamp(n, True)
    for _ in range(round(args.relax / dt)):
        sim.set_targets(sim.joints())
        sim.step()
    q_touch = sim.joints()
    pen = {f"{a}-{b}": round(v * 1000, 2) for (a, b), v in sim.penetration().items() if {"upper", "lower"} & {a, b}}
    log(f"  relaxed: joints moved {np.abs(q_touch - joints).max():.3f} rad max; overlap (mm) {pen}")
    log(f"  relaxed forces (N): {_forces(sim)}")

    # Pinch, ramped over 0.3 s, then hold.
    ramp = round(0.3 / dt)
    for k in range(ramp):
        sim.set_targets(q_touch + push * (k + 1) / ramp)
        sim.step()
    sim.step(round(0.3 / dt))
    log(f"  pinched (clamped) forces (N): {_forces(sim)}")
    for n in STICKS:
        sim.clamp(n, False)
    sim.step(round(0.5 / dt))
    sim.set_stick_damping(0.0, 0.0)
    sim.step(round(0.3 / dt))

    moved = {}
    for n in STICKS:
        pos, rot = sim.stick_pose(n, "palm")
        p0, r0 = src["sticks"][n]
        moved[n] = (float(np.linalg.norm(pos - p0)), rot_angle_deg(r0.T @ rot))
    q, tg = sim.joints(), sim.targets()
    ff = sim.finger_forces()
    log("  sticks vs source pose: " + "  ".join(f"{n} {m[0] * 1000:.1f} mm {m[1]:.1f} deg" for n, m in moved.items()))
    log(f"  final forces (N): {_forces(sim)}")
    log(f"  joints vs source: {np.abs(q - joints).max():.3f} rad max, {np.sqrt(np.mean((q - joints) ** 2)):.3f} rms")
    sticks = {}
    for n in STICKS:
        pos, rot = sim.stick_pose(n, "palm")
        sticks[n] = {"pos": pos.tolist(), "rotvec": mat_to_rotvec(rot).tolist()}
    return {
        "palm_rotvec": mat_to_rotvec(src["palm_rot"]).tolist(),
        "joint_targets": dict(zip(SR_JOINT_NAMES, map(float, tg))),
        "joints": dict(zip(SR_JOINT_NAMES, map(float, q))),
        "sticks": sticks,
        "fingers_in_contact": [f for f in FINGERS if max(ff[f].values()) > 0.01],
        "formed": {"final_forces": ff, "moved_mm_deg": moved, "pinch": {f"{a}-{b}": v for (a, b), v in pinch.items()},
                   "settings": {k: v for k, v in vars(args).items() if k != "set"}},  # fmt: skip
        "env": {},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="closed_v5")
    p.add_argument("--save-as", required=True)
    p.add_argument("--pinch", nargs="*", default=DEFAULT_PINCH, metavar="FINGER-STICK=RAD")
    p.add_argument("--relax", type=float, default=1.0, help="s of relaxing (gravity off, damped sticks, yielding fingers)")
    p.add_argument("--clamp-solref", type=float, nargs=2, default=[0.05, 1.0], help="soft clamp while relaxing/pinching")
    p.add_argument("--stick-damping", type=float, nargs=2, default=[1.0, 0.01], metavar=("LIN", "ANG"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="SimCfg overrides (saved in the grip)")
    p.add_argument("--test", action="store_true", help="run the hold test on the new grip")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    log = (lambda *a: None) if args.quiet else print

    overrides = parse_overrides(args.set)
    sim = Sim(SimCfg.from_overrides(overrides))
    src = load_grip(args.source)
    log(f"forming {args.save_as} from {args.source} (pinch {' '.join(args.pinch)}, sim {overrides or 'defaults'})")
    grip = form(sim, src, args, log)
    grip["made_by"] = f"chopstix.tools.form_grip from {args.source} (MuJoCo)"
    grip["sim"] = overrides
    path = GRIPS / f"{args.save_as}.json"
    path.write_text(json.dumps(grip, indent=1, default=str))
    log(f"  wrote {path}")
    if args.test:
        from chopstix.tools import hold_test

        return hold_test.main(["--grip", args.save_as, "--seeds", "5", "--quiet", "--set", *args.set])
    return grip


if __name__ == "__main__":
    main()
