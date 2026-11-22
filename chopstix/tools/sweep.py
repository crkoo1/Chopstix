"""Run hold_test for several SimCfg override sets in parallel and print a table (one process per set).

    .venv/Scripts/python.exe -m chopstix.tools.sweep --grip closed_v5 --seeds 4 \
        --cases "joint_kp=1" "joint_kp=2" "impratio=100" "condim=6 impratio=100"
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chopstix.tools import hold_test


def _run(job):
    case, common = job
    argv = common + ["--quiet"] + (["--set", *case.split()] if case.strip() else [])
    out = hold_test.main(argv)
    runs = out["runs"]
    worst = lambda k, n: max(r[k][n] for r in runs)  # noqa: E731
    return {
        "case": case or "(defaults)", "hold": out["hold_rate"],
        "up_mm": worst("drift_m", "upper") * 1000, "lo_mm": worst("drift_m", "lower") * 1000,
        "up_deg": worst("turn_deg", "upper"), "lo_deg": worst("turn_deg", "lower"),
        "fail_s": min((r["first_fail_s"] for r in runs if r["first_fail_s"] is not None), default=None),
        "max_force": max(max(v.values()) for v in out["initial"]["finger_forces_N"].values()),
        "rt": out["realtime_factor"],
    }  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", required=True, help='each a quoted "key=value key=value" set ("" = defaults)')
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path)
    args, rest = p.parse_known_args()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_run, [(c, rest) for c in args.cases]))
    print(f"{'case':48s} {'hold':>5s} {'up mm':>6s} {'lo mm':>6s} {'up deg':>6s} {'lo deg':>6s} {'fail s':>6s} {'F0 max':>6s} {'xRT':>4s}")
    for r in rows:
        print(f"{r['case'][:48]:48s} {r['hold']:5.0%} {r['up_mm']:6.1f} {r['lo_mm']:6.1f} {r['up_deg']:6.1f} {r['lo_deg']:6.1f} "
              f"{str(r['fail_s']):>6s} {r['max_force']:6.2f} {r['rt']:4.0f}")  # fmt: skip
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
