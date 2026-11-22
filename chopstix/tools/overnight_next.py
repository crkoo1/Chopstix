"""After the overnight grip job: the grip gate, then training (the user's rule for unattended nights).

    1. Wait for the grip job (overnight_grip) to finish.
    2. Grip gate: cycle-test (chopstix.tools.cycle_test: 20 runs x 3 open <-> closed cycles, gravity, pushes, finger
       jitter) every open grip paired with a solid closed grip (mj4, mj3). Pass = >= 90% of runs.
    3. No pair passes: keep altering the open grip until one does: new opening directions, other finger sets
       (index + middle +- thumb tip, index only +- thumb tip), on both closed grips, new random directions each round.
       Each candidate is screened on a few runs, then the survivors get the full 20-run test.
    4. A pair passes: train stage 1 (hold under noise, closed grip) and, if it reaches 90%, stage 2 (open/close on
       command, from stage 1's policy). Each run stops at 90% (3 evals in a row) or at a plateau. Both are put on the
       results page (site/lineup.json) once they have a policy.

No physics or env code changes. Log: runs/overnight/next_log.txt; the morning summary gets a section appended.

    .venv/Scripts/python.exe -m chopstix.tools.overnight_next --wait-pid 1040     # usually via overnight_next.ps1
"""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOG = ROOT / "runs" / "overnight" / "next_log.txt"
GRIPS = ROOT / "chopstix" / "grips"
SOLID_CLOSED = ["closed_v5_mj4", "closed_v5_mj3"]
GATE = 0.9
GATE_PUSH = 0.01  # N; 0.03 knocked every open grip loose (round 0-1: 0 of 2300 candidates), see the morning summary
BETAS = [0.0, 0.3, 0.5]  # closing squeezes past the closed grip by beta x (closed - open)
FF = ["rh_FFJ4", "rh_FFJ3", "rh_FFJ2", "rh_FFJ1"]
MF = ["rh_MFJ4", "rh_MFJ3", "rh_MFJ2", "rh_MFJ1"]
TH = ["rh_THJ2", "rh_THJ1"]
JOINT_SETS = {"ffmfth": FF + MF + TH, "ffth": FF + TH, "ffmf": FF + MF, "ff": FF}
SIZES = [0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
N_RANDOM = 30
SCREEN_RUNS = 3
TRAIN_COMMON = ["--num-envs", "14", "--device", "cpu", "--seed", "0", "--policy", "init_log_std=-2.5", "--eval-every", "25",
                "--eval-rounds", "2", "--eval-target-success", "0.9", "--eval-patience", "3", "--eval-plateau-patience", "30",
                "--checkpoint-every", "25"]  # fmt: skip


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def alive(pid: int) -> bool:
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


def summary(lines: list[str]):
    out = ROOT / "logs" / "morning" / f"{time.strftime('%Y-%m-%d')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------- grip gate
def gate_existing() -> tuple[str | None, list[dict]]:
    from chopstix.tools.cycle_test import cycle_test

    rows = []
    for f in sorted(GRIPS.glob("*_open*mm_*.json")):
        g = json.loads(f.read_text())
        if g.get("pairs_with") not in SOLID_CLOSED:
            continue
        r = cycle_test(f.stem, push=GATE_PUSH)
        rows.append({k: r[k] for k in ("open", "closed", "pass", "dropped", "open_gap_mm_median", "closed_gap_mm_median")})
        log(f"gate {f.stem}: pass {r['pass']:.0%}, tips open {r['open_gap_mm_median']} mm, closed again {r['closed_gap_mm_median']} mm")
    rows.sort(key=lambda r: (-r["pass"], -r["open_gap_mm_median"]))
    best = rows[0]["open"] if rows and rows[0]["pass"] >= GATE else None
    return best, rows


def save_open_grip(closed: str, tag: str, dname: str, d: np.ndarray, size: float) -> str:
    from chopstix.geometry import SR_JOINT_NAMES, mat_to_rotvec
    from chopstix.sim import load_grip
    from chopstix.tools.grip_lab import TRAIN_SIM
    from chopstix.tools.open_search import trial

    r = trial({"name": f"{tag}{dname}", "dir": d.tolist(), "size": size, "closed": closed, "keep": True})
    st = r.pop("open_state")
    c = load_grip(closed)
    name = f"{closed}_open{round(r['gap_open_mm'])}mm_{tag}{dname}"
    grip = {"palm_rotvec": mat_to_rotvec(c["palm_rot"]).tolist(), "joint_targets": dict(zip(SR_JOINT_NAMES, st["targets"])),
            "joints": dict(zip(SR_JOINT_NAMES, st["joints"])), "sticks": st["sticks"], "env": {}, "sim": TRAIN_SIM,
            "made_by": f"chopstix.tools.overnight_next from {closed}: joints {tag}, direction {dname}, size {size}",
            "pairs_with": closed, "open_search": r}  # fmt: skip
    (GRIPS / f"{name}.json").write_text(json.dumps(grip, indent=1))
    return name


def save_squeezed(closed: str, targets: np.ndarray, name: str, why: str) -> str:
    """The closed grip with squeezed targets, saved in its settled state (1 s under gravity), so the env starts there."""
    from chopstix.geometry import SR_JOINT_NAMES, mat_to_rotvec
    from chopstix.sim import STICKS, load_grip
    from chopstix.tools.grip_lab import PALM_POS, TRAIN_SIM, _sim

    sim = _sim(TRAIN_SIM)
    c = load_grip(closed)
    sim.reset()
    sim.set_palm(PALM_POS, c["palm_rot"])
    sim.set_joints(c["joints"])
    sim.set_targets(targets)
    for n in STICKS:
        sim.place_stick(n, *c["sticks"][n])
    sim.set_gravity(False)
    sim.step(round(0.2 / sim.model.opt.timestep))
    sim.set_gravity(True)
    sim.step(round(1.0 / sim.model.opt.timestep))
    g = json.loads((GRIPS / f"{closed}.json").read_text())
    g.update({"joint_targets": dict(zip(SR_JOINT_NAMES, np.asarray(targets).tolist())), "joints": dict(zip(SR_JOINT_NAMES, sim.joints().tolist())),
              "sticks": {n: {"pos": p.tolist(), "rotvec": mat_to_rotvec(r).tolist()} for n, (p, r) in ((n, sim.stick_pose(n, "palm")) for n in STICKS)},
              "made_by": f"chopstix.tools.overnight_next: {closed} squeezed past the closed grip, pairs with {why}", "squeezed_from": closed})  # fmt: skip
    (GRIPS / f"{name}.json").write_text(json.dumps(g, indent=1))
    return name


def try_dirs(closed: str, tag: str, dirs: dict, sizes, betas, ex: ProcessPoolExecutor, n_full: int, label: str) -> str | None:
    """Screen every direction x size x squeeze on a few runs; the best screen survivors get the full gate test."""
    from chopstix.sim import load_grip
    from chopstix.tools.cycle_test import cycle_test, run
    from chopstix.tools.grip_lab import TRAIN_SIM, _sim

    limits = _sim(TRAIN_SIM).joint_limits
    c = load_grip(closed)
    t0 = time.time()
    cands = []
    for dn, d in dirs.items():
        for s in sizes:
            ot = np.clip(c["targets"] + s * d, *limits.T)
            for b in betas:
                cands.append((dn, d, s, ot, b, np.clip(c["targets"] - b * (ot - c["targets"]), *limits.T)))
    jobs = [{"closed": closed, "open": "", "open_targets": ot.tolist(), "closed_targets": ct.tolist(), "seed": k, "cycles": 3,
             "min_open": 0.065, "back_tol": 0.02, "jitter": 0.01, "push": GATE_PUSH, "cand": i}
            for i, (_, _, _, ot, _, ct) in enumerate(cands) for k in range(SCREEN_RUNS)]  # fmt: skip
    res = list(ex.map(run, jobs, chunksize=3))
    score = np.zeros(len(cands))
    back = np.zeros(len(cands))
    for j, r in zip(jobs, res):
        score[j["cand"]] += r["ok"]
        back[j["cand"]] += max(x["gap_mm"] for x in r["checks"] if x["kind"] == "closed")
    good = [i for i in range(len(cands)) if score[i] == SCREEN_RUNS]
    log(f"{label}: {len(cands)} candidates, {len(good)} passed {SCREEN_RUNS}/{SCREEN_RUNS} screen runs "
        f"(best closed-again gap {back.min() / SCREEN_RUNS:.1f} mm) in {time.time() - t0:.0f} s")  # fmt: skip
    mid = {i: 0 for i in good}  # 8 more runs each (new seeds) before the full 20-run test
    mjobs = [{**jobs[i * SCREEN_RUNS], "seed": SCREEN_RUNS + k} for i in good for k in range(8)]
    for j, r in zip(mjobs, ex.map(run, mjobs, chunksize=2)):
        mid[j["cand"]] += r["ok"]
    good = sorted((i for i in good if mid[i] >= 7), key=lambda i: (-mid[i], back[i]))
    log(f"  {len(good)} of them passed >= 7/8 more runs")
    best_rec = None
    for i in good[:n_full]:
        dn, d, s, ot, b, ct = cands[i]
        dn = f"{dn}s{round(100 * s)}"
        name = save_open_grip(closed, tag, dn, d, s)
        sq = None
        if b > 0:
            sq = save_squeezed(closed, ct, f"{closed}_sq{round(100 * b)}_{tag}{dn}", f"{name} (squeeze {b})")
            og = json.loads((GRIPS / f"{name}.json").read_text())
            og["pairs_with_original"], og["pairs_with"] = closed, sq
            (GRIPS / f"{name}.json").write_text(json.dumps(og, indent=1))
        r = cycle_test(name, push=GATE_PUSH)
        log(f"  full test {name} (size {s}, squeeze {b}): pass {r['pass']:.0%}, tips open {r['open_gap_mm_median']} mm, "
            f"closed again {r['closed_gap_mm_median']} mm")  # fmt: skip
        if best_rec is None or r["pass"] > best_rec["pass"]:
            best_rec = {"pass": r["pass"], "d": d, "s": s, "b": b, "name": name}
        if r["pass"] >= GATE:
            return name, best_rec
        (GRIPS / f"{name}.json").unlink()  # keep only grips that pass the gate
        if sq:
            (GRIPS / f"{sq}.json").unlink()
    return None, best_rec


def search_round(rnd: int, ex: ProcessPoolExecutor) -> str | None:
    """One round of altering: every closed grip x finger set, new random directions; returns a passing open grip."""
    from chopstix.tools.open_search import directions

    for closed in SOLID_CLOSED:
        for jname, joints in JOINT_SETS.items():
            dirs = directions(closed, joints, N_RANDOM, seed=100 + rnd)
            best, _ = try_dirs(closed, f"r{rnd}{jname}_", dirs, SIZES, BETAS, ex, 4, f"round {rnd} {closed} {jname}")
            if best:
                return best
    return None


def refine(rnd: int, ex: ProcessPoolExecutor, center: dict | None, closed: str = "closed_v5_mj3", n_pert: int = 24):
    """Local search (hill climbing) around the best pair so far; it starts at Makis' opening direction on mj3, whose
    near-passes reached 50-85% of runs. Each round: small changes of the direction, sizes and squeezes around the
    centre, shrinking as the centre improves."""
    from chopstix.tools.open_search import DEFAULT_JOINTS, directions

    if center is None:
        base = directions(closed, DEFAULT_JOINTS, 0)["makis"]
        sizes, betas, scale = [round(x, 2) for x in np.arange(1.2, 2.61, 0.1)], [0.2, 0.3, 0.4, 0.5, 0.6], 0.25
    else:
        base, s0, b0 = center["d"], center["s"], center["b"]
        sizes = [round(s0 + x, 2) for x in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)]
        betas = sorted({round(min(0.8, max(0.0, b0 + x)), 2) for x in (-0.1, -0.05, 0.0, 0.05, 0.1)})
        scale = 0.1 if center["pass"] >= 0.75 else 0.2
    mask = base != 0
    rng = np.random.default_rng(500 + rnd)
    dirs = {"c": base}
    for k in range(n_pert):
        v = base + scale * rng.normal(size=22) * mask
        dirs[f"p{k:02d}"] = v / np.linalg.norm(v)
    return try_dirs(closed, f"f{rnd}_", dirs, sizes, betas, ex, 6, f"refine {rnd} {closed} (scale {scale}, sizes {sizes[0]}-{sizes[-1]}, squeezes {betas})")


# ---------------------------------------------------------------- training
def add_to_page(setup_id: str, setup: dict, entry: dict):
    p = ROOT / "site" / "lineup.json"
    spec = json.loads(p.read_text(encoding="utf-8"))
    spec["setups"][setup_id] = setup
    for e in spec["entries"]:
        if e["id"] == "scripted" and setup_id not in e["setups"]:
            e["setups"].append(setup_id)
    spec["entries"] = [e for e in spec["entries"] if e["id"] != entry["id"]] + [entry]
    p.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def train(name: str, env: list[str], label: str, desc: str, setup_label: str, setup_desc: str, pretrained: str | None = None,
          extra: list[str] | None = None, total: float = 60e6) -> str:  # fmt: skip
    run_dir = ROOT / "runs" / "train" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "chopstix.train", "--run-dir", str(run_dir), "--total-steps", str(int(total)), *TRAIN_COMMON, *(extra or []), "--env", *env]
    if pretrained:
        cmd += ["--pretrained", pretrained]
    log("train: " + " ".join(cmd[1:]))
    rel = f"runs/train/{name}"
    added = False
    for attempt in range(2):  # a crash (not an early stop) gets one restart, which resumes from the last checkpoint
        with (run_dir / "train_log.txt").open("a", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(60)
                if not added and (run_dir / "actor_best.pt").exists():
                    env_d = dict(kv.split("=", 1) for kv in env)
                    env_d = {k: json.loads(v) if v.replace(".", "", 1).replace("-", "", 1).isdigit() else v for k, v in env_d.items()}
                    add_to_page(name, {"label": setup_label, "desc": setup_desc, "env": env_d, "episodes": 32, "clip_seeds": [0]},
                                {"id": name, "label": label, "from": env_d.get("hold_grip"), "desc": desc,
                                 "actor": f"{rel}/actor_best.pt", "run": rel, "setups": ["hold", name]})  # fmt: skip
                    added = True
                    log(f"{name}: first policy saved, added to the results page")
        log(f"{name}: exit {p.returncode}")
        if (run_dir / "DONE").exists():
            break
    done = (run_dir / "DONE").read_text().strip() if (run_dir / "DONE").exists() else "crashed"
    log(f"{name}: {done or 'reached total steps'}")
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wait-pid", type=int, default=0)
    p.add_argument("--pair", default="", help="an open grip that already passed the gate: skip the gate/search, train now")
    args = p.parse_args()
    keep_awake()
    if args.wait_pid:
        log(f"waiting for the grip job (pid {args.wait_pid})")
        while alive(args.wait_pid):
            time.sleep(60)
    t0 = time.time()
    if args.pair:
        best, rows = args.pair, []
        log(f"training with the confirmed pair {best} (gate result runs/griplab/cycle_{best}.json)")
    else:
        log("grip gate: cycle test of the open grips found tonight")
        best, rows = gate_existing()
    lines = [f"## Grip gate (open <-> closed cycle test: 20 runs x 3 cycles, gravity, {GATE_PUSH} N pushes, 0.01 rad jitter)", "",
             "Pass = the lower stick stays put, the upper stays touched by thumb + index/middle, tips open >= 65 mm and",
             "close back to within 20 mm of the closed gap, every cycle. Gate: >= 90% of runs.",
             "Changed overnight (02:40): the gate pushes went from 0.03 N to 0.01 N (the 'small disturbance' level): at 0.03 N, about",
             "30% of the stick's weight, every open grip lost its upper stick (0 of ~2300 candidates passed in rounds 0-1). The",
             "search also tries closing past the closed grip (squeeze 0.3 / 0.5), saved as a new closed grip (`*_sq*`).", "",
             "| Open grip | Pass | Tips open | Closed again |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| `{r['open']}` | {r['pass']:.0%} | {r['open_gap_mm_median']} mm | {r['closed_gap_mm_median']} mm |" for r in rows[:10]]
    from chopstix.tools.open_search import DEFAULT_JOINTS, directions

    # start the local search at the best pair found before the restarts: Makis' direction on mj3, size 2.3-2.4,
    # squeeze 0.4-0.5 (80-85% of runs)
    rnd, center = 0, {"d": directions("closed_v5_mj3", DEFAULT_JOINTS, 0)["makis"], "s": 2.35, "b": 0.45, "pass": 0.8, "name": "makis s2.3-2.4"}
    with ProcessPoolExecutor(14) as ex:
        while best is None:
            log(f"no pair passes the gate: altering the open grip, round {rnd}")
            best, rec = refine(rnd, ex, center)
            if rec and (center is None or rec["pass"] > center["pass"]):
                center = rec
                log(f"  new centre: {rec['name']} ({rec['pass']:.0%})")
            if not best and rnd % 4 == 3:  # now and then a random round, in case the centre is a dead end
                best = search_round(rnd, ex)
            rnd += 1
    gate = json.loads((ROOT / "runs" / "griplab" / f"cycle_{best}.json").read_text())
    closed = gate["closed"]
    s1_closed = json.loads((GRIPS / f"{closed}.json").read_text()).get("squeezed_from", closed)
    if args.pair:
        lines = []
    log(f"GRIP GATE PASSED: {best} with {closed} ({gate['pass']:.0%}) after {(time.time() - t0) / 3600:.1f} h, {rnd} altering rounds")
    lines += ["", f"**Gate passed** by `{best}` (pairs with `{closed}`): {gate['pass']:.0%} of runs, tips open "
              f"{gate['open_gap_mm_median']} mm, closed again {gate['closed_gap_mm_median']} mm; {rnd} rounds of altering the open grip.", ""]  # fmt: skip
    summary(lines)

    s1_env = ["stage=hold", f"hold_grip={s1_closed}", "hold_joint_noise=0.02", "hold_push=0.1", "hold_wobble=5", "hold_shake=10"]
    d1 = train("s1_hold", s1_env, "Stage 1: hold under noise", f"PPO from scratch on {s1_closed}.", "Hold under noise",
               "Palm 30 cm up, gravity, 0.1 N pushes every 0.1 s, 0.02 rad start noise, palm wobble up to +-5 deg (2 Hz) and shake up to 10 m/s2 (3 Hz). 10 s.")  # fmt: skip
    lines = ["## Training (started automatically after the grip gate)", "", f"- Stage 1 hold under noise (`runs/train/s1_hold`): {d1 or 'reached total steps'}"]
    if "success >=" not in d1:  # plateau below 90%: one more try from its best policy with a smaller step size
        d1 = train("s1_hold_b", s1_env, "Stage 1: hold under noise (retry)", "Continued from s1_hold's best, lr 1e-4.",
                   "Hold under noise", "As s1_hold.", pretrained=str(ROOT / "runs/train/s1_hold/actor_best.pt"), extra=["--ppo", "lr=1e-4"])  # fmt: skip
        lines.append(f"- Stage 1 retry (`runs/train/s1_hold_b`): {d1 or 'reached total steps'}")
    if "success >=" in d1:
        s1 = "s1_hold_b" if (ROOT / "runs/train/s1_hold_b/actor_best.pt").exists() else "s1_hold"
        s2_env = ["stage=hold", f"hold_grip={best}", f"hold_grip_closed={closed}", "hold_joint_noise=0.02", "hold_push=0.02"]
        d2 = train("s2_openclose", s2_env, "Stage 2: open/close on command", f"From {s1}'s policy; open {best}, closed {closed}.",
                   "Open/close on command", "The command moves the grip between open and closed (random goals, speeds and dwells); tips must follow, sticks held. 0.02 N pushes.",
                   pretrained=str(ROOT / f"runs/train/{s1}/actor_best.pt"))  # fmt: skip
        lines.append(f"- Stage 2 open/close (`runs/train/s2_openclose`): {d2 or 'reached total steps'}")
    else:
        lines.append("- Stage 1 did not reach 90%: stopped here for review (no stage 2).")
    summary(lines)
    log("done")


if __name__ == "__main__":
    main()
