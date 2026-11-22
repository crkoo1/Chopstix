"""Form grips with several form_grip settings (and SimCfg overrides) in parallel, hold-test each, print a table.

    .venv/Scripts/python.exe -m chopstix.tools.form_sweep --source closed_v5 --seeds 5 \
        --cases "--press source --squeeze 0.3" "--press contact --squeeze 0.04 --set impratio=100"

Each case is the argument string for form_grip (without --source/--save-as); grips are saved as sweep_<i>.json.
"""

import argparse
import json
import shlex
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chopstix.tools import form_grip, hold_test


def _run(job):
    i, case, source, seeds, duration = job
    argv = shlex.split(case)
    name = f"sweep_{i}"
    grip = form_grip.main(["--source", source, "--save-as", name, "--quiet", *argv])
    sets = argv[argv.index("--set") + 1 :] if "--set" in argv else []
    sets = [s for s in sets if "=" in s]
    out = hold_test.main(["--grip", name, "--seeds", str(seeds), "--duration", str(duration), "--quiet", "--set", *sets])
    runs = out["runs"]
    ff = grip["formed"]["final_forces"]
    return {
        "i": i, "case": case, "hold": out["hold_rate"],
        "up_mm": max(r["drift_m"]["upper"] for r in runs) * 1000, "lo_mm": max(r["drift_m"]["lower"] for r in runs) * 1000,
        "up_deg": max(r["turn_deg"]["upper"] for r in runs), "lo_deg": max(r["turn_deg"]["lower"] for r in runs),
        "fail_s": min((r["first_fail_s"] for r in runs if r["first_fail_s"] is not None), default=None),
        "forces": {f: round(max(v.values()), 2) for f, v in ff.items() if max(v.values()) > 0.005},
        "moved": {n: [round(m[0] * 1000, 1), round(m[1], 1)] for n, m in grip["formed"]["moved_mm_deg"].items()},
    }  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="closed_v5")
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    jobs = [(i, c, args.source, args.seeds, args.duration) for i, c in enumerate(args.cases)]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_run, jobs))
    for r in rows:
        print(f"[{r['i']:2d}] {r['case'][:60]:60s} hold {r['hold']:4.0%}  up {r['up_mm']:5.1f}mm/{r['up_deg']:5.1f}deg  "
              f"lo {r['lo_mm']:5.1f}mm/{r['lo_deg']:5.1f}deg  fail {r['fail_s']}  F {r['forces']}  moved {r['moved']}")  # fmt: skip
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
