"""Compare candidate grips on a short grip-lab battery, in parallel. Each case forms a grip (form_grip arguments,
including its --set physics) and tests it under that same physics.

    .venv/Scripts/python.exe -m chopstix.tools.grip_search --cases \
        "--pinch th-upper=0.01 th-lower=0.15 ff-upper=0.01 mf-upper=0.01 rf-lower=0.1" \
        "--pinch th-upper=0.01 th-lower=0.3 ff-upper=0.01 mf-upper=0.01 rf-lower=0.2 --set actuator_timeconst=0.05"

Battery (8 seeds x 5 s per rung unless --seeds/--duration): gravity at the 9 grip-lab poses; finger jitter 0.005,
0.01, 0.02 rad; pushes 0.1, 0.2 N (grip pose) and 0.1 N (pick pose). Prints one row per case, best first.
"""

import argparse
import json
import shlex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chopstix.tools import form_grip
from chopstix.tools.grip_lab import TESTS, TRAIN_SIM, orientations, run, test_job

ROOT = Path(__file__).resolve().parents[2]
RUNGS = [("jitter", 0.005), ("jitter", 0.01), ("jitter", 0.02), ("pushes", 0.1), ("pushes", 0.2), ("pushes_pick", 0.1)]  # any grip_lab TESTS name


def _form(job):
    i, case, source = job
    argv = shlex.split(case)
    sets = argv[argv.index("--set") + 1 :] if "--set" in argv else []
    sim = {**TRAIN_SIM, **form_grip.parse_overrides([s for s in sets if "=" in s])}
    # form under the same physics it will be tested with
    rest = argv[: argv.index("--set")] if "--set" in argv else argv
    grip = form_grip.main(["--source", source, "--save-as", f"search_{i}", "--quiet", *rest, "--set", *[f"{k}={json.dumps(v)}" for k, v in sim.items()]])
    return sim, grip["formed"]["final_forces"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--source", default="closed_v5")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--rungs", default="", help='e.g. "jitter:0.02,jitter:0.05,pushes:0.3" (default: the battery above)')
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "griplab" / "search.json")
    args = p.parse_args()
    global RUNGS
    if args.rungs:
        RUNGS = [(t, float(v)) for t, v in (r.split(":") for r in args.rungs.split(","))]

    with ProcessPoolExecutor(args.workers) as ex:
        formed = list(ex.map(_form, [(i, c, args.source) for i, c in enumerate(args.cases)]))
    jobs, tags = [], []
    for i, (sim, _) in enumerate(formed):
        name = f"search_{i}"
        orient = orientations(name)
        for pose, rot in orient.items():
            jobs.append({"grip": name, "palm_rot": rot.tolist(), "push": 0, "jitter": 0, "start": 0, "seed": 0, "duration": args.duration, "sim": sim})
            tags.append((i, "gravity", pose))
        for test, lv in RUNGS:
            for s in range(args.seeds):
                jobs.append(test_job(test, lv, name, s, args.duration, sim, orient))
                tags.append((i, test, lv))
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(run, jobs, chunksize=2))
    rows = []
    for i, case in enumerate(args.cases):
        mine = [(t, r) for t, r in zip(tags, res) if t[0] == i]
        grav = sum(r["holds"] for t, r in mine if t[1] == "gravity")
        rung = {f"{test}@{lv:g}": sum(r["holds"] for t, r in mine if t[1:] == (test, lv)) / args.seeds for test, lv in RUNGS}
        jit = [r["drift_mm"]["lower"] for t, r in mine if t[1] == "jitter"]
        worst_lower = max(jit) if jit else None
        forces = {f: round(max(v.values()), 2) for f, v in formed[i][1].items() if max(v.values()) > 0.005}
        score = grav / 9 + sum(rung.values())  # 0 .. 1 + number of rungs
        rows.append({"i": i, "case": case, "score": round(score, 3), "gravity": f"{grav}/9", **rung, "lower_mm_worst_jitter": worst_lower, "forces": forces})
    rows.sort(key=lambda r: -r["score"])
    head = "  ".join(f"{k:>14s}" for k, _ in [(f"{t}@{lv:g}", 0) for t, lv in RUNGS])
    print(f"{'#':>3} {'score':>5} {'grav':>5} {head}  case")
    for r in rows:
        print(f"{r['i']:3d} {r['score']:5.2f} {r['gravity']:>5} " + "  ".join(f"{r[f'{t}@{lv:g}']:14.0%}" for t, lv in RUNGS)
              + f"  {r['case']}   F {r['forces']}  lower worst {r['lower_mm_worst_jitter']} mm")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
