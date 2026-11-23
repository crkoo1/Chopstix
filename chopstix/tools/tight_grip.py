"""A tighter open/closed grip pair for a 20 mm cube (user, 2026-11-22), gated at 90% like the overnight search.

Why: with closed_v5_mj3_open68mm_g_makis240 -> closed_v5_mj3_sq35_makis240 the tips come back to ~27 mm after an
opening, so the sticks never reach a 20 mm cube (0/32 pick tests, runs/pick_test.json). This searches pairs on
closed_v5_mj3 along Makis' opening direction (plus small perturbations, hill climbing) with:
    open   = closed + s * d           (s: opening size along Makis' direction d)
    closed = closed + t * e           (t: how far, e: which way the fingers close; the first version squeezed back
                                       along -d, but that pushes the sticks apart: 22-28 mm tip gap even before
                                       any opening, see the 2026-11-22 17:40 diagnosis in the log)
A pair passes a run of the cycle test (3 open/close cycles, 0.01 N pushes, 0.01 rad jitter, sticks held as in the
gate) when the tips open to >= 45 mm (2x the cube + margin) and close back to <= 14 mm every time. Screen 3 runs ->
8 more (>= 7/8) -> full 20 runs (>= 90%). Then a pick check (15 deg sticks, 20 mm cube) with the grip alone and with
the stage 3 policy. Result: runs/tight_grip/RESULT.json; log: runs/overnight/next_log.txt.

    .venv/Scripts/python.exe -m chopstix.tools.tight_grip [--workers 6]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chopstix.tools.overnight_next import GATE, GATE_PUSH, GRIPS, ROOT, keep_awake, log, save_open_grip, save_squeezed

CLOSED = "closed_v5_mj3"
MIN_OPEN, MAX_CLOSED = 0.045, 0.014
OUT = ROOT / "runs" / "tight_grip"


def passes(r: dict) -> bool:
    closed = [x["gap_mm"] for x in r["checks"] if x["kind"] == "closed"]
    return bool(r["ok"]) and max(closed) <= 1000 * MAX_CLOSED


def worst_closed(r: dict) -> float:
    return max(x["gap_mm"] for x in r["checks"] if x["kind"] == "closed")


PALM = None  # the pick pose (15 deg sticks): set in main; v3's pass held only in the holding pose and slipped in the pick


def job(c, d, s, e, t, limits, seed, cand):
    ot = np.clip(c["targets"] + s * d, *limits.T)
    ct = np.clip(c["targets"] + t * e, *limits.T)
    return {"palm_rot": PALM,"closed": CLOSED, "open": "", "open_targets": ot.tolist(), "closed_targets": ct.tolist(), "seed": seed, "cycles": 3,
            "min_open": MIN_OPEN, "back_tol": 1.0, "jitter": 0.01, "push": GATE_PUSH, "cand": cand}  # fmt: skip


def save_open_state(open_name: str, closed_name: str, ot) -> str:
    """The open grip as the cycle test reaches it (from the closed grip, gravity on, opened over 1 s, held 1 s), in the
    pick pose. save_open_grip's own opening (3 s from mj3) left v3's pair 84 mm apart with loose sticks."""
    from chopstix.geometry import SR_JOINT_NAMES, mat_to_rotvec
    from chopstix.sim import STICKS, load_grip
    from chopstix.tools.grip_lab import PALM_POS, TRAIN_SIM, _sim
    from chopstix.tools.open_search import tip_gap

    sim = _sim(TRAIN_SIM)
    c = load_grip(closed_name)
    dt = sim.model.opt.timestep
    n = max(1, round(0.05 / dt))
    sim.reset()
    sim.set_palm(PALM_POS, np.array(PALM))
    sim.set_joints(c["joints"])
    sim.set_targets(c["targets"])
    for s in STICKS:
        sim.place_stick(s, *c["sticks"][s])
    sim.set_gravity(False)
    sim.step(round(0.2 / dt))
    sim.set_gravity(True)
    for k in range(40):
        sim.set_targets(c["targets"] + min(1.0, (k + 1) / 20) * (np.asarray(ot) - c["targets"]))
        sim.step(n)
    g = json.loads((GRIPS / f"{open_name}.json").read_text())
    g.update({"joints": dict(zip(SR_JOINT_NAMES, sim.joints().tolist())), "joint_targets": dict(zip(SR_JOINT_NAMES, np.asarray(ot).tolist())),
              "sticks": {s: {"pos": p.tolist(), "rotvec": mat_to_rotvec(r).tolist()} for s, (p, r) in ((s, sim.stick_pose(s, "palm")) for s in STICKS)},
              "made_by": g["made_by"] + f"; state = opened from {closed_name} in the pick pose ({1000 * tip_gap(sim):.0f} mm)"})  # fmt: skip
    (GRIPS / f"{open_name}.json").write_text(json.dumps(g, indent=1))
    return open_name


def pick_check(open_name: str, closed_name: str, ex) -> dict:
    from chopstix.envs.chopsticks_env import make_env
    from chopstix.tools.report import rollout, summarize_pick

    base = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["pick_oc"]["env"]
    env = {**base, "pick_stick_pitch": 15, "ball_radius": 0.010, "hold_grip": open_name, "hold_grip_closed": closed_name}
    pol = json.loads((ROOT / "runs/train/s3_pick15/config.json").read_text())["policy"]
    out = {}
    for label, actor in (("grip alone", None), ("s3_pick15 policy", str(ROOT / "runs/train/s3_pick15/actor_best.pt"))):
        eps = list(ex.map(rollout, [{"env": env, "actor": actor, "policy": pol if actor else None, "seed": s, "record": False,
                                     "bodies": []} for s in range(32)]))  # fmt: skip
        sm = summarize_pick(eps, env["steps_per_episode"], make_env(env).cfg)
        out[label] = {"milestones": {m["label"]: m["k"] for m in sm["milestones"]}, "outcomes": {o["key"]: o["k"] for o in sm["outcomes"]}}
    return out


def main():
    from chopstix.sim import load_grip
    from chopstix.tools.cycle_test import run
    from chopstix.tools.grip_lab import TRAIN_SIM, _sim
    from chopstix.tools.open_search import DEFAULT_JOINTS, directions

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-rounds", type=int, default=40)
    args = p.parse_args()
    keep_awake()
    OUT.mkdir(parents=True, exist_ok=True)
    limits = _sim(TRAIN_SIM).joint_limits
    c = load_grip(CLOSED)
    makis = directions(CLOSED, DEFAULT_JOINTS, 0)["makis"]
    mask = makis != 0
    global PALM
    from chopstix.envs.chopsticks_env import make_env

    env15 = json.loads((ROOT / "site" / "lineup.json").read_text(encoding="utf-8-sig"))["setups"]["s3_pick15"]["env"]
    PALM = make_env(env15)._pick_palm_rot.tolist()
    center = {"e": -makis, "s": 1.6, "t": 0.6, "score": -9.0}
    if (OUT / "closing_direction.npy").exists():  # v3's pass (good in the holding pose) as the starting point
        center = {"e": np.load(OUT / "closing_direction.npy"), "s": 1.7, "t": 0.45, "score": -9.0}
    rng = np.random.default_rng(11)

    def score(rs):  # share of passing runs, minus how far the closing misses 14 mm and the opening misses 45 mm (mean, /10 mm)
        # (v2 first scored the closing alone and climbed to pairs that jam shut: 7-13 mm "closed" but 4-9 mm "open")
        miss_c = np.mean([max(0.0, worst_closed(r) - 1000 * MAX_CLOSED) for r in rs])
        miss_o = np.mean([max(0.0, 1000 * MIN_OPEN - min(x["gap_mm"] for x in r["checks"] if x["kind"] == "open")) for r in rs])
        held = np.mean([r["ok"] for r in rs])  # sticks held and opened >= 45 mm (back_tol is off, so the closing gap is not in it)
        return float(np.mean([passes(r) for r in rs]) + 0.5 * held - (miss_c + miss_o) / 10)

    log(f"tight_grip v4: start (checked in the 15 deg pick pose; a pass must also pinch the 20 mm cube in >= 16/32 picks; open >= {1000 * MIN_OPEN:.0f} mm, closed again <= {1000 * MAX_CLOSED:.0f} mm, gate {GATE:.0%})")
    with ProcessPoolExecutor(args.workers) as ex:
        for rnd in range(args.max_rounds):
            t0 = time.time()
            if rnd == 0 and center["score"] == -9.0 and not (OUT / "closing_direction.npy").exists():
                es = {"minus_makis": -makis}
                for k in range(20):
                    v = rng.normal(size=22) * mask
                    es[f"r{k:02d}"] = v / np.linalg.norm(v)
                sizes, ts = [1.3, 1.6, 2.0], [0.3, 0.6, 1.0]
            else:
                scale = 0.15 if center["score"] > 0 else 0.3
                es = {"c": center["e"]}
                for k in range(12):
                    v = center["e"] + scale * rng.normal(size=22) * mask
                    es[f"p{k:02d}"] = v / np.linalg.norm(v)
                sizes = sorted({round(max(1.2, center["s"] + x), 2) for x in (-0.2, 0.0, 0.2)})
                ts = sorted({round(max(0.1, center["t"] + x), 2) for x in (-0.15, 0.0, 0.15)})
            cands = [(en, e, s, t) for en, e in es.items() for s in sizes for t in ts]
            jobs = [job(c, makis, s, e, t, limits, k, i) for i, (_, e, s, t) in enumerate(cands) for k in range(3)]
            res = list(ex.map(run, jobs, chunksize=2))
            by = [[] for _ in cands]
            for j, r in zip(jobs, res):
                by[j["cand"]].append(r)
            sc = [score(rs) for rs in by]
            good = [i for i in range(len(cands)) if all(passes(r) for r in by[i])]
            bi = int(np.argmax(sc))
            log(f"tight_grip v4 round {rnd}: {len(cands)} candidates, {len(good)} passed 3/3; best {cands[bi][0]} s {cands[bi][2]} t {cands[bi][3]}: "
                f"{sum(passes(r) for r in by[bi])}/3, closed again {np.mean([worst_closed(r) for r in by[bi]]):.1f} mm ({time.time() - t0:.0f} s)")  # fmt: skip
            if sc[bi] > center["score"]:
                center = {"e": cands[bi][1], "s": cands[bi][2], "t": cands[bi][3], "score": sc[bi]}
            mid = {}
            for i in good:
                mj = [job(c, makis, cands[i][2], cands[i][1], cands[i][3], limits, 3 + k, i) for k in range(8)]
                mid[i] = sum(passes(r) for r in ex.map(run, mj))
            good = sorted((i for i in good if mid[i] >= 7), key=lambda i: (-mid[i], -sc[i]))
            for i in good[:3]:
                en, e, s, t = cands[i]
                fr = list(ex.map(run, [job(c, makis, s, e, t, limits, 100 + k, i) for k in range(20)]))
                rate = sum(passes(r) for r in fr) / 20
                log(f"  full test {en} s {s} t {t}: {rate:.0%}, closed again median {np.median([worst_closed(r) for r in fr]):.1f} mm, "
                    f"open median {np.median([x['gap_mm'] for r in fr for x in r['checks'] if x['kind'] == 'open']):.1f} mm")  # fmt: skip
                if rate >= GATE:
                    tag = f"v{rnd}_"
                    open_name = save_open_grip(CLOSED, tag, f"makis{round(100 * s)}", makis, s)
                    ct = np.clip(c["targets"] + t * e, *limits.T)
                    closed_name = save_squeezed(CLOSED, ct, f"{CLOSED}_tight_{tag}{en}t{round(100 * t)}", f"{open_name} (closing {en} x {t})")
                    g = json.loads((GRIPS / f"{open_name}.json").read_text())
                    g["pairs_with_original"], g["pairs_with"] = CLOSED, closed_name
                    (GRIPS / f"{open_name}.json").write_text(json.dumps(g, indent=1))
                    save_open_state(open_name, closed_name, np.clip(c["targets"] + s * makis, *limits.T))
                    pc = pick_check(open_name, closed_name, ex)
                    pinched = pc["grip alone"]["milestones"]["Cube pinched off the table"]
                    log(f"  pick check {open_name} -> {closed_name}: grip alone pinched {pinched}/32, {json.dumps(pc)}")
                    if pinched < 16:
                        continue
                    np.save(OUT / "closing_direction.npy", e)
                    rec = {"open": open_name, "closed": closed_name, "size": s, "closing": en, "closing_t": t, "gate": rate, "round": rnd,
                           "pick_check": pc, "time": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
                    (OUT / "RESULT.json").write_text(json.dumps(rec, indent=1))
                    log(f"tight_grip: PASSED {open_name} -> {closed_name}; pick check (15 deg, 20 mm): {json.dumps(pc)}")
                    return
                center_sc = rate - 0.0
                if center_sc > center["score"]:
                    center = {"e": e, "s": s, "t": t, "score": center_sc}
    (OUT / "RESULT.json").write_text(json.dumps({"open": None, "rounds": args.max_rounds, "time": time.strftime("%Y-%m-%d %H:%M")}))
    log(f"tight_grip: no pair passed in {args.max_rounds} rounds")


if __name__ == "__main__":
    main()
