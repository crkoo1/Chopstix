"""Pose-by-pose summary for the results page (site/data/poses.json), user 2026-11-24.

The page opens on a pose toggle. For each hand pose: what the hold is, the best training set (best run of each stage:
hold under noise -> open/close on command -> pick and place) with its training settings, and every run of each stage
in order, with its eval curve and how it ended. Read from runs/train/*/config.json, train_log.txt and DONE.

    .venv/Scripts/python.exe -m chopstix.tools.poses [--render]      # --render also redraws site/data/pose*.jpg
"""

import argparse
import json
import re
import time
from pathlib import Path

from chopstix.tools.overnight_next import ROOT

TRAIN = ROOT / "runs" / "train"
OUT = ROOT / "site" / "data"
STAGES = [("hold", "Hold under disturbances", "The grip holds both sticks while the palm is pushed, wobbled and shaken."),
          ("openclose", "Open and close", "The sticks open and close on command and come back to the grip each time."),
          ("pick", "Pick and drop", "Come down, pinch a 25 mm cube, lift it, carry it 10 cm, lower it and let go.")]  # fmt: skip

POSES = [  # four ways the hand has held the sticks; "best" fixes the best set by hand (None = the chain that picked best)
    {"id": "pose1", "name": "Original grip, sticks at the grip's angle", "short": "Pose 1",
     "desc": "The first MuJoCo grip (closed_v5_mj3 family, open/close pair from Makis' motion): palm facing the table, the "
             "sticks at the angle the grip lab made them. First pick task: a 30 mm cube. Stages 1-2 are shared with pose 2.",
     "best": ["s1_hold95", "s2_openclose_v2", "s3_pick"], "image": "data/pose1.jpg"},
    {"id": "pose2", "name": "Original grip, sticks 15° down", "short": "Pose 2",
     "desc": "Same grip and same stages 1-2 as pose 1, but the hand tilts so the sticks point 15° below horizontal when "
             "picking. Trained on a 30 mm cube, a harder version, then the 25 mm cube.",
     "best": ["s1_hold95", "s2_openclose_v2", "s3_pick25b"], "image": "data/pose2.jpg"},
    {"id": "pose3", "name": "Human-style grip, back of the hand up", "short": "Pose 3",
     "desc": "Grip hg_1: thumb pad, index and middle on the upper stick; thumb base, index base and ring finger on the "
             "lower. The hand is turned over (back up), sticks 20° down, so the tips open and close up and down.",
     "best": [None, None, "s4_human"], "image": "data/pose3.jpg"},
    {"id": "pose4", "name": "Human-style grip, sideways closing", "short": "Pose 4",
     "desc": "Grip hg_1 (re-settled as hg_1s125), back of the hand up, sticks 20° down, hand turned so the tips open and "
             "close sideways. Stiffer hand-stick contacts so the sticks don't creep when the hand shakes.",
     "best": None, "image": "data/pose4.jpg"},
]  # fmt: skip

NOTES = {  # one line on runs that were replaced or are side experiments
    "s1_hold": "First hold run; continued as s1_hold95.",
    "s2_openclose": "First open/close run; continued as s2_openclose_v2.",
    "s2_openclose_b": "Open/close again from s1_hold95: worse (57%).",
    "s3_pick": "First pick, 30 mm ball-cube.",
    "s3_pick15": "Sticks 15° down, 30 mm cube: reached 100% on its checks.",
    "s3_pick_hard": "Harder pick: cube moved ±1.5 cm and turned ±17°, pushes on.",
    "s3_pick25": "First 25 mm cube run; continued as s3_pick25b.",
    "s3_pick25r": "Side test: finger range ±0.2 rad.",
    "s3_pick25s": "Side test: sticks must end within 15 mm of the grip. Worse than s3_pick25b.",
    "s4_human": "Back of the hand up, tips closing up and down: never pinched the cube.",
    "s4_level": "Sideways pose, straight to picking from scratch. Stopped: the learning rate ran away to 1e-2.",
    "s4l_hold": "Finger range ±0.1 rad, lr cap 1e-4: stuck at 80%. The sticks crept through the fingers under wobble.",
    "s4t_hold": "Finger range ±0.2 rad, lr cap 3e-4: still 80%. Fixed next by stiffer contacts (s4u).",
}


def stage_of(env: dict) -> str:
    if env.get("stage") == "pick":
        return "pick"
    return "openclose" if env.get("hold_grip_closed") else "hold"


def poses_of(env: dict) -> list[str]:
    """Which poses a run (or a test setup) belongs to, from its own settings."""
    human = str(env.get("hold_grip", "")).startswith("hg_")
    if env.get("stage") != "pick":
        if human:
            return ["pose4"] if env.get("hold_palm_rot") else ["pose3", "pose4"]
        return ["pose1", "pose2"]  # the original grip's stages 1-2 hold the hand in the grip's own orientation
    if human:
        return ["pose4"] if env.get("pick_level_open") else ["pose3"]
    return ["pose2"] if env.get("pick_stick_pitch") else ["pose1"]


def read_run(d: Path) -> dict | None:
    try:
        cfg = json.loads((d / "config.json").read_text())
        log = (d / "train_log.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    env, ppo, ev = cfg["env"], cfg["ppo"], cfg["eval"]
    per_it = ppo["rollout_steps"] * cfg["num_envs"]
    evals = [[int(m.group(1)) * per_it, float(m.group(2))] for m in re.finditer(r"eval it\s+(\d+): success ([0-9.]+)", log)]
    last_it = re.findall(r"^it\s+(\d+)\s+steps\s+(\d+)", log, re.M)
    done = (d / "DONE").read_text().strip() if (d / "DONE").exists() else None
    t0, t1 = (d / "config.json").stat().st_mtime, (d / "train_log.txt").stat().st_mtime
    if done is not None:
        status = "target" if "consecutive" in done else "plateau" if "no eval improvement" in done else "finished"
    else:
        status = "running" if time.time() - t1 < 15 * 60 else "stopped"
    sim = env.get("sim") or {}
    pre = cfg.get("pretrained")
    return {
        "id": d.name, "stage": stage_of(env), "poses": poses_of(env), "status": status, "done": done,
        "best": max((p for _, p in evals), default=None), "last": evals[-1][1] if evals else None, "n_evals": len(evals),
        "evals": evals[-400:], "steps": int(last_it[-1][1]) if last_it else 0,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t0)), "end": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t1)),
        "hours": round((t1 - t0) / 3600, 2), "from": Path(pre).parent.name if pre else None, "note": NOTES.get(d.name, ""),
        "gate": 0.98 if d.name.startswith("s4v") else 0.9,
        "settings": {
            "lr": ppo["lr"], "lr_min": ppo.get("lr_min", 1e-5), "lr_max": ppo.get("lr_max", 1e-2), "desired_kl": ppo["desired_kl"],
            "rollout": ppo["rollout_steps"], "envs": cfg["num_envs"], "batch": per_it, "epochs": ppo["epochs"], "minibatches": ppo["minibatches"],
            "entropy": ppo["entropy_coef"], "gamma": ppo["gamma"], "clip": ppo["clip"], "actor_freeze": ppo.get("actor_freeze_iters", 0),
            "finger_range": env.get("hold_residual", 0.1), "solref": (sim.get("solref") or [0.02])[0],
            "grip": env.get("hold_grip"), "grip_closed": env.get("hold_grip_closed"),
            "eval_episodes": ev["rounds"] * cfg["num_envs"], "eval_every": ev["every"], "plateau": ev["plateau_patience"],
            "target": ev["target_success"], "pushes": env.get("hold_push"), "wobble": env.get("hold_wobble"), "shake": env.get("hold_shake"),
        },
    }  # fmt: skip


def best_set(runs: list[dict], pose: dict) -> list[str]:
    if pose["best"]:
        return [r if r in {x["id"] for x in runs} else None for r in pose["best"]]
    by_prefix = {}  # chains named <prefix>_<stage>[_n]: the chain whose pick did best
    for r in runs:
        pre = r["id"].split("_")[0]
        by_prefix.setdefault(pre, {}).setdefault(r["stage"], []).append(r)
    full = {p: s for p, s in by_prefix.items() if all(k in s for k, *_ in STAGES)}
    if not full:
        return []
    pick_best = lambda p: max((r["best"] or 0) for r in full[p]["pick"])  # noqa: E731
    p = max(sorted(full), key=pick_best)
    return [max(full[p][k], key=lambda r: (r["best"] or 0, r["end"]))["id"] for k, *_ in STAGES]


def render(pose_runs: dict):
    """One picture per pose: the grip as it closes on the cube (no policy), front and side views."""
    import mujoco
    import numpy as np
    from PIL import Image

    from chopstix.envs.chopsticks_env import make_env

    for pid, run in pose_runs.items():
        env = make_env(json.loads((TRAIN / run / "config.json").read_text())["env"])
        env.reset(seed=0)
        for _ in range(sum(env._pick_steps[:3])):
            env.step(np.zeros(env.action_space.shape))
        m, dd = env.sim.model, env.sim.data
        p, R = env.sim.palm_pose()
        r = mujoco.Renderer(m, 420, 560)
        tiles = []
        look = (p + env._stick_tips().mean(0)) / 2
        for az, el, dist in [(200, -22, 0.42), (130, -18, 0.40)]:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(dd, cam)
            tiles.append(r.render())
        r.close()
        Image.fromarray(np.concatenate(tiles, 1)).save(OUT / f"{pid}.jpg", quality=82)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()
    runs = [r for d in sorted(TRAIN.iterdir()) if d.is_dir() and (r := read_run(d))]
    runs.sort(key=lambda r: r["start"])
    lineup = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]
    poses = []
    for pose in POSES:
        mine = [r for r in runs if pose["id"] in r["poses"]]
        best = best_set(mine, pose)
        setups = sorted(sid for sid, st in lineup.items() if pose["id"] in poses_of(st["env"]))
        poses.append({**{k: v for k, v in pose.items() if k != "best"}, "best_set": best, "setups": setups,
                      "stages": [{"key": k, "name": n, "what": w, "runs": [r["id"] for r in mine if r["stage"] == k]} for k, n, w in STAGES]})  # fmt: skip
    now = next((r["id"] for r in reversed(runs) if r["status"] == "running"), None)
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "now": now, "poses": poses, "runs": {r["id"]: r for r in runs}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poses.json").write_text(json.dumps(out, separators=(",", ":")))
    if args.render:
        render({ps["id"]: ps["best_set"][-1] for ps in poses if ps["best_set"] and ps["best_set"][-1]})
    print(f"poses: {[(ps['id'], ps['best_set']) for ps in poses]}; now training: {now}")


if __name__ == "__main__":
    main()
