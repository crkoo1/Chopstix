"""Build the data for the results page (site/index.html) and add a dated snapshot to its history.

    .venv/Scripts/python.exe -m chopstix.tools.report --note "Hold run s0 finished: 20M steps"
    .venv/Scripts/python.exe -m chopstix.tools.report --no-history          # refresh without a new snapshot

Reads site/lineup.json (which policies, which test setups) and writes site/data/:
    results.json   per (entry, setup): episode outcomes, milestones, time series, Wilson intervals; grip checks;
                   training curves of entries that name a run directory
    history.json   one snapshot per report run (time, note, git commit, headline numbers)
    scene.json     Shadow Hand visual meshes (per body, in the body frame) for the 3-D replay
    clips/<entry>_<setup>_s<seed>.json   recorded episodes: every hand body, stick and cube pose per control step
Every episode is deterministic (the policy's mean action) on fixed seeds, so reports are comparable over time.
"""

import argparse
import base64
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
PHASES = ["settle", "descend", "close", "lift", "carry", "lower", "release", "retreat"]
PICK_LIFTED = 0.005  # m: the cube counts as pinched off the table above this (while both sticks touch it)

_ENV_CACHE: dict = {}
_ACTOR_CACHE: dict = {}


# ---------------------------------------------------------------------------------------------- rollouts (workers)
def _env(overrides: dict):
    from chopstix.envs.chopsticks_env import make_env

    if overrides.get("domain_rand") and overrides.get("domain_rand_seed", -1) < 0:
        overrides = {**overrides, "domain_rand_seed": 0}  # page tests: one fixed randomized world, the same every report
    key = json.dumps(overrides, sort_keys=True)
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = make_env(overrides)
    return _ENV_CACHE[key]


def _actor(path: str | None, policy_cfg: dict | None):
    if path is None:
        return None
    key = (path, json.dumps(policy_cfg, sort_keys=True))
    if key not in _ACTOR_CACHE:
        import torch

        from chopstix.envs.chopsticks_env import ACTION_DIM, PROPRIO_DIM, TASK_DIM
        from chopstix.rl.policy import ActorCritic, PolicyCfg

        torch.set_num_threads(1)
        model = ActorCritic(PROPRIO_DIM, TASK_DIM, ACTION_DIM, PolicyCfg(**(policy_cfg or {})))
        report = model.load_pretrained(path)
        if report["skipped"]:
            raise ValueError(f"{path}: tensors did not match the policy config: {report['skipped']}")
        model.eval()
        _ACTOR_CACHE[key] = model
    return _ACTOR_CACHE[key]


def _pose(sim, body: int) -> list[float]:
    p, q = sim.data.xpos[body], sim.data.xquat[body]  # MuJoCo quaternion w, x, y, z -> three.js x, y, z, w
    return [round(float(x), 5) for x in (*p, q[1], q[2], q[3], q[0])]


def rollout(job: dict) -> dict:
    """One deterministic episode. Returns per-step signals (+ frames when job['record'])."""
    import torch

    env = _env(job["env"])
    model = _actor(job["actor"], job.get("policy"))
    sim = env.sim
    bodies = [sim.model.body(n).id for n in job.get("bodies", [])]
    extra = [sim.stick_body["upper"], sim.stick_body["lower"]] + ([sim.object_body] if sim.has_object else [])
    obs, info = env.reset(seed=job["seed"])
    steps, frames = [], []
    if job["record"]:
        frames.append([_pose(sim, b) for b in bodies + extra])
    ret, term = 0.0, False
    for _ in range(env.cfg.steps_per_episode):
        if model is None:
            action = np.zeros(env.action_space.shape)
        else:
            with torch.no_grad():
                action = model.act(torch.as_tensor(obs[None], dtype=torch.float32), deterministic=True)[0][0].numpy()
        obs, r, term, trunc, info = env.step(action)
        ret += r
        if env.cfg.stage == "hold":
            s = {"drift": max(info["upper_drift"], info["lower_drift"]), "turn": max(info["upper_turn"], info["lower_turn"]),
                 "ok": float(info["is_success"])}  # fmt: skip
        else:
            s = {k: float(info[k]) for k in ("phase", "object_lift", "both_touch", "touch_upper", "touch_lower", "pinch",
                                               "tip_gap", "upper_drift", "lower_drift", "object_to_b", "open_cmd",
                                               "dropped", "object_to_spot", "is_success")}  # fmt: skip
        s["ret"] = ret
        steps.append({k: round(v, 5) for k, v in s.items()})
        if job["record"]:
            frames.append([_pose(sim, b) for b in bodies + extra])
        if term or trunc:
            break
    out = {"seed": job["seed"], "steps": steps, "terminated": bool(term), "success": bool(info["is_success"]),
           "return": float(ret), "length": len(steps)}  # fmt: skip
    if env.cfg.stage == "pick":
        out["b"] = [round(float(x), 5) for x in env._pick_b]
        out["phase_steps"] = [int(x) for x in env._pick_steps]
        out["cube"] = 2 * sim.cfg.object_size
        out["max_drift"], out["max_away"] = env.cfg.pick_max_drift, env.cfg.pick_max_away
    else:
        out["max_drift"], out["max_turn"] = env.cfg.hold_max_drift, env.cfg.hold_max_turn
    if job["record"]:
        out["frames"] = frames
    return out


# ---------------------------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _rate(k: int, n: int) -> dict:
    return {"k": int(k), "n": int(n), "p": round(k / n, 4) if n else 0.0, "ci": wilson(k, n)}


def summarize_hold(eps: list[dict], steps_per_episode: int, dt: float) -> dict:
    n = len(eps)
    held = lambda s: sum(e["length"] >= round(s / dt) for e in eps)  # noqa: E731
    full = sum(not e["terminated"] for e in eps)
    ok = sum(e["success"] for e in eps)
    milestones = [
        {"label": "Held 2.5 s", **_rate(held(2.5), n)},
        {"label": "Held 5 s", **_rate(held(5.0), n)},
        {"label": "Held the full 10 s", **_rate(full, n)},
        {"label": "Within 1 cm and 10Â° at the end", **_rate(ok, n)},
    ]
    outcomes = [
        {"key": "success", "label": "Success", **_rate(ok, n)},
        {"key": "shifted", "label": "Full 10 s, but sticks shifted past 1 cm / 10Â°", **_rate(full - ok, n)},
        {"key": "slipped", "label": "Ended early: a stick slipped (3 cm / 30Â°)", **_rate(n - full, n)},
    ]
    T = steps_per_episode
    running = np.zeros(T)
    within = np.zeros(T)
    drift_sum, drift_n = np.zeros(T), np.zeros(T)
    for e in eps:
        for t, s in enumerate(e["steps"]):
            running[t] += 1
            within[t] += s["ok"]
            drift_sum[t] += s["drift"]
            drift_n[t] += 1
    series = {
        "running": (running / n).round(4).tolist(),
        "within": (within / n).round(4).tolist(),
        "drift_mm": np.where(drift_n > 0, 1000 * drift_sum / np.maximum(drift_n, 1), np.nan).round(2).tolist(),
    }
    rows = [{"seed": e["seed"], "outcome": "success" if e["success"] else ("slipped" if e["terminated"] else "shifted"),
             "length": e["length"], "score": round(1000 * max(s["drift"] for s in e["steps"]), 1)} for e in eps]  # fmt: skip
    return {"n": n, "success": _rate(ok, n), "milestones": milestones, "outcomes": outcomes, "series": series,
            "episodes": rows, "mean_return": round(float(np.mean([e["return"] for e in eps])), 1)}  # fmt: skip


def summarize_pick(eps: list[dict], steps_per_episode: int, env_cfg) -> dict:
    n = len(eps)
    starts = np.concatenate([[0], np.cumsum(eps[0]["phase_steps"])])
    release_at = int(starts[6])
    reach = pinch = lift = carried = ok = 0
    buckets = {"success": 0, "slipped": 0, "knocked": 0, "never": 0, "dropped": 0, "not_placed": 0}
    for e in eps:
        st = e["steps"]
        touched = any(s["both_touch"] for s in st)
        max_lift = max((s["object_lift"] for s in st if s["both_touch"]), default=0.0)
        kept = len(st) > release_at and st[release_at - 1]["both_touch"] > 0
        e["max_lift"] = max_lift
        reach += touched
        pinch += max_lift > PICK_LIFTED
        lift += max_lift > env_cfg.pick_success_height
        carried += max_lift > env_cfg.pick_success_height and kept
        ok += e["success"]
        if e["success"]:
            e["outcome"] = "success"
        elif e["terminated"]:
            last = st[-1]
            e["outcome"] = "slipped" if max(last["upper_drift"], last["lower_drift"]) > e["max_drift"] else "knocked"
        elif max_lift <= PICK_LIFTED:
            e["outcome"] = "never"
        elif not kept:
            e["outcome"] = "dropped"
        else:
            e["outcome"] = "not_placed"
        buckets[e["outcome"]] += 1
    milestones = [
        {"label": "Both sticks touch the cube", **_rate(reach, n)},
        {"label": "Cube pinched off the table", **_rate(pinch, n)},
        {"label": "Lifted 40 mm", **_rate(lift, n)},
        {"label": "Still held when the release starts", **_rate(carried, n)},
        {"label": "Placed at B and let go (success)", **_rate(ok, n)},
    ]
    labels = {
        "success": "Success", "not_placed": "Carried, but not placed at B", "dropped": "Dropped before the release",
        "never": "Never pinched it", "knocked": "Ended early: cube knocked away", "slipped": "Ended early: a stick slipped",
    }  # fmt: skip
    outcomes = [{"key": k, "label": labels[k], **_rate(v, n)} for k, v in buckets.items()]
    T = steps_per_episode
    running, touch, h_sum, h_n = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for e in eps:
        for t, s in enumerate(e["steps"]):
            running[t] += 1
            touch[t] += s["both_touch"]
            h_sum[t] += s["object_lift"]
            h_n[t] += 1
    series = {
        "running": (running / n).round(4).tolist(),
        "touch": (touch / n).round(4).tolist(),
        "height_mm": np.where(h_n > 0, 1000 * h_sum / np.maximum(h_n, 1), np.nan).round(2).tolist(),
    }
    return {"n": n, "success": _rate(ok, n), "milestones": milestones, "outcomes": outcomes, "series": series,
            "episodes": [{"seed": e["seed"], "outcome": e["outcome"], "length": e["length"], "score": round(1000 * e["max_lift"], 1)} for e in eps],
            "phase_steps": eps[0]["phase_steps"], "mean_return": round(float(np.mean([e["return"] for e in eps])), 1),
            "median_max_lift_mm": round(1000 * float(np.median([e["max_lift"] for e in eps])), 1)}  # fmt: skip


# ---------------------------------------------------------------------------------------------- scene & curves
def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def scene_meshes() -> tuple[dict, list[str]]:
    """Visual meshes per hand body, in that body's frame. The cosmetic wrist and forearm ride on the palm."""
    from chopstix.sim import HAND_XML, SR_MODELS, _symlink_assets

    m = mujoco.MjModel.from_xml_path(str(HAND_XML), _symlink_assets(SR_MODELS))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    visual = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    palm = m.body("rh_palm").id

    def tf(body):
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = d.xmat[body].reshape(3, 3), d.xpos[body]
        return t

    out, names = {}, []
    for g in visual:
        b = m.geom_bodyid[g]
        name = m.body(b).name
        host = "rh_palm" if name in ("rh_forearm", "rh_wrist") else name
        # geom in its body frame, then (forearm/wrist) into the palm frame
        gt = np.eye(4)
        gt[:3, :3], gt[:3, 3] = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        local = np.linalg.inv(tf(palm if host == "rh_palm" else b)) @ gt
        mesh = m.geom_dataid[g]
        v0, nv, f0, nf = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh], m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
        verts = m.mesh_vert[v0 : v0 + nv] @ local[:3, :3].T + local[:3, 3]
        faces = m.mesh_face[f0 : f0 + nf].astype(np.uint16 if nv < 65536 else np.uint32)
        mat = m.material(m.geom_matid[g]).name if m.geom_matid[g] >= 0 else "plastic"
        mat = "arm" if name == "rh_forearm" else ("metal" if mat == "metal" else "plastic")
        out.setdefault(host, []).append({"mat": mat, "v": _b64(verts.astype(np.float32)), "f": _b64(faces), "f32": faces.dtype == np.uint32})
        if host not in names:
            names.append(host)
    return out, names


def training_curves(run_dir: Path) -> dict | None:
    tb = run_dir / "tb"
    if not tb.exists():
        return None
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tb), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    out = {}
    for tag in ("eval/success", "episode/is_success", "episode/return", "episode/length"):
        if tag in tags:
            ev = acc.Scalars(tag)
            stride = max(1, len(ev) // 400)
            out[tag] = [[int(e.step), round(float(e.value), 4)] for e in ev[::stride]] + ([[int(ev[-1].step), round(float(ev[-1].value), 4)]] if (len(ev) - 1) % stride else [])
    cfg = run_dir / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        out["total_steps"] = c.get("total_steps")
    done = run_dir / "DONE"
    out["finished"] = done.exists()
    return out


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


# ---------------------------------------------------------------------------------------------- main
def main(argv=None):
    from chopstix.envs.chopsticks_env import ChopsticksEnvCfg

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--note", default="", help="what changed since the last snapshot (shown in the history)")
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--episodes", type=int, help="override every setup's episode count (quick runs)")
    args = p.parse_args(argv)

    spec = json.loads((SITE / "lineup.json").read_text())
    data_dir = SITE / "data"
    (data_dir / "clips").mkdir(parents=True, exist_ok=True)
    for old in (data_dir / "clips").glob("*.json"):  # every report rewrites all clips
        old.unlink()
    meshes, bodies = scene_meshes()

    jobs, keys = [], []
    for entry in spec["entries"]:
        actor = None if entry["actor"] == "zero" else str((ROOT / entry["actor"]).resolve())
        policy = None
        if entry.get("run") and (ROOT / entry["run"] / "config.json").exists():
            policy = json.loads((ROOT / entry["run"] / "config.json").read_text()).get("policy")
        for setup_id in entry["setups"]:
            setup = spec["setups"][setup_id]
            n = args.episodes or setup["episodes"]
            for seed in range(n):
                record = seed in setup.get("clip_seeds", [])
                jobs.append({"env": setup["env"], "actor": actor, "policy": policy, "seed": seed, "record": record,
                             "bodies": bodies if record else []})  # fmt: skip
                keys.append((entry["id"], setup_id))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        eps = list(ex.map(rollout, jobs, chunksize=4))
    print(f"{len(eps)} episodes in {time.time() - t0:.0f} s", flush=True)

    results = {}
    clip_index = []
    for (eid, sid), job, ep in zip(keys, jobs, eps):
        results.setdefault(eid, {}).setdefault(sid, []).append(ep)
        if job["record"]:
            cid = f"{eid}_{sid}_s{job['seed']}"
            clip = {k: ep[k] for k in ("seed", "terminated", "success", "length", "frames") if k in ep}
            clip.update(id=cid, entry=eid, setup=sid, stage=spec["setups"][sid]["env"]["stage"],
                        info=ep["steps"], b=ep.get("b"), cube=ep.get("cube"), phase_steps=ep.get("phase_steps"))  # fmt: skip
            (data_dir / "clips" / f"{cid}.json").write_text(json.dumps(clip, separators=(",", ":")))
            clip_index.append({"id": cid, "entry": eid, "setup": sid, "seed": job["seed"], "success": ep["success"],
                               "terminated": ep["terminated"], "length": ep["length"]})  # fmt: skip

    summaries = {}
    for eid, by_setup in results.items():
        for sid, ep_list in by_setup.items():
            cfg = ChopsticksEnvCfg.from_overrides(spec["setups"][sid]["env"])
            dt = 1.0 / cfg.control_frequency
            if cfg.stage == "hold":
                s = summarize_hold(ep_list, cfg.steps_per_episode, dt)
            else:
                s = summarize_pick(ep_list, cfg.steps_per_episode, cfg)
            summaries.setdefault(eid, {})[sid] = s
            print(f"{eid:12s} {sid:6s} success {s['success']['k']}/{s['success']['n']}  "
                  + "  ".join(f"{o['key']} {o['k']}" for o in s["outcomes"]), flush=True)  # fmt: skip

    # Also replay each (entry, setup)'s best attempt: a success if any, else the highest pinched lift (pick) or the
    # smallest drift (hold).
    extra_jobs = []
    for eid, by in summaries.items():
        for sid, summ in by.items():
            rows = summ["episodes"]
            if summ["success"]["k"] and sid in spec["setups"] and spec["setups"][sid]["env"]["stage"] == "hold":
                best = max(rows, key=lambda r: (r["outcome"] == "success", -r["score"]))
            else:
                best = max(rows, key=lambda r: (r["outcome"] == "success", r["score"]))
            if any(c["entry"] == eid and c["setup"] == sid and c["seed"] == best["seed"] for c in clip_index):
                summ["best_clip"] = f"{eid}_{sid}_s{best['seed']}"
                continue
            job = next(j for (k, j) in zip(keys, jobs) if k == (eid, sid) and j["seed"] == best["seed"])
            extra_jobs.append(((eid, sid), {**job, "record": True, "bodies": bodies}))
    if extra_jobs:
        with ProcessPoolExecutor(min(args.workers, len(extra_jobs))) as ex:
            extra_eps = list(ex.map(rollout, [j for _, j in extra_jobs]))
        for ((eid, sid), job), ep in zip(extra_jobs, extra_eps):
            cid = f"{eid}_{sid}_s{job['seed']}"
            clip = {k: ep[k] for k in ("seed", "terminated", "success", "length", "frames")}
            clip.update(id=cid, entry=eid, setup=sid, stage=spec["setups"][sid]["env"]["stage"],
                        info=ep["steps"], b=ep.get("b"), cube=ep.get("cube"), phase_steps=ep.get("phase_steps"))  # fmt: skip
            (data_dir / "clips" / f"{cid}.json").write_text(json.dumps(clip, separators=(",", ":")))
            clip_index.append({"id": cid, "entry": eid, "setup": sid, "seed": job["seed"], "success": ep["success"],
                               "terminated": ep["terminated"], "length": ep["length"], "best": True})  # fmt: skip
            summaries[eid][sid]["best_clip"] = cid

    curves = {}
    for entry in spec["entries"]:
        if entry.get("run"):
            c = training_curves(ROOT / entry["run"])
            if c:
                curves[entry["id"]] = c
    grip_checks = []
    for g in spec.get("grip_checks", []):
        f = ROOT / g["file"]
        if f.exists():
            for row in json.loads(f.read_text()):
                grip_checks.append({"grip": g["grip"], "case": row["case"], "hold": row["hold"], "runs": g["runs"],
                                    "seconds": g["seconds"], "ci": wilson(round(row["hold"] * g["runs"]), g["runs"]),
                                    "worst_mm": round(max(row["up_mm"], row["lo_mm"]), 1),
                                    "worst_deg": round(max(row["up_deg"], row["lo_deg"]), 1)})  # fmt: skip
    seen, uniq = set(), []
    for g in grip_checks:
        if g["case"] not in seen:
            seen.add(g["case"])
            uniq.append(g)

    # Grip lab results (chopstix.tools.grip_lab), and their gravity clips for the replay.
    grip_labs = []
    for gl in spec.get("grip_labs", []):
        f = ROOT / gl["file"]
        if not f.exists():
            continue
        lab = json.loads(f.read_text())
        gid = f.stem
        grip_labs.append({"id": gid, "label": gl["label"], "desc": gl.get("desc", ""), "default": gl.get("default", False), "grip": lab["grip"], "sim": lab["sim"],
                          "seeds": lab["seeds"], "duration": lab["duration"], "score": lab["score"], "ladders": lab["ladders"],
                          "gravity": lab["gravity"]})  # fmt: skip
        for c in lab.get("clips", []):
            slug = "".join(ch if ch.isascii() and ch.isalnum() else "_" for ch in c["pose"]).strip("_")[:24]
            cid = f"griplab_{gid}_{slug}"
            info = [{"drift": max(r[0], r[1]) / 1000, "turn": max(r[2], r[3]), "ok": float(max(r[0], r[1]) < 10 and max(r[2], r[3]) < 10), "ret": 0.0}
                    for r in c["series"]]  # fmt: skip
            clip = {"id": cid, "entry": f"griplab:{gid}", "setup": "griplab", "stage": "hold", "seed": 0, "terminated": False,
                    "success": bool(c.get("holds", info[-1]["ok"])), "length": len(info), "frames": c["frames"], "info": info,
                    "label": f"{gl['label']} · {c.get('title', 'gravity only · ' + c['pose'])}"}  # fmt: skip
            (data_dir / "clips" / f"{cid}.json").write_text(json.dumps(clip, separators=(",", ":")))
            clip_index.append({"id": cid, "entry": f"griplab:{gid}", "setup": "griplab", "seed": 0, "success": clip["success"],
                               "terminated": False, "length": len(info), "label": clip["label"]})  # fmt: skip

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for entry in spec["entries"]:  # when each policy was saved, so the page can show the newest first
        actor = ROOT / entry["actor"] if entry["actor"] != "zero" else None
        if actor and actor.exists():
            entry["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(actor.stat().st_mtime))
    commit = git_commit()
    out = {"generated": now, "commit": commit, "dt": 0.05, "setups": {k: {kk: v[kk] for kk in ("label", "desc")} | {"stage": v["env"]["stage"]} for k, v in spec["setups"].items()},
           "entries": spec["entries"], "summaries": summaries, "curves": curves, "grip_checks": uniq, "clips": clip_index,
           "grip_labs": grip_labs}  # fmt: skip
    (data_dir / "results.json").write_text(json.dumps(out, separators=(",", ":")))
    from chopstix.geometry import CHOPSTICK_BASE_RADIUS, CHOPSTICK_LENGTH, CHOPSTICK_TIP_RADIUS

    scene = {"bodies": bodies, "meshes": meshes, "stick": {"length": CHOPSTICK_LENGTH, "r_base": CHOPSTICK_BASE_RADIUS, "r_tip": CHOPSTICK_TIP_RADIUS}}
    (data_dir / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))

    hist_path = data_dir / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    if not args.no_history:
        history.append({
            "time": now, "commit": commit, "note": args.note,
            "results": {eid: {sid: {"success": s["success"], "mean_return": s["mean_return"]} for sid, s in by.items()} for eid, by in summaries.items()},
            "grip_hold": next((g["hold"] for g in uniq if g["case"] == "noslip_iterations=10 timestep=0.005"), None),
            "grip_lab": (lambda g: g["score"] | {"grip": g["grip"]})(next((g for g in grip_labs if g["default"]), grip_labs[-1])) if grip_labs else None,
        })  # fmt: skip
        hist_path.write_text(json.dumps(history, indent=1))
    print(f"wrote {data_dir} ({len(clip_index)} clips, {len(history)} snapshots)")


if __name__ == "__main__":
    main()
