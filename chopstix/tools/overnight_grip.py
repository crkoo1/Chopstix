"""Unattended grip work (no training, no git): open-grip search, then the full grip lab on every open grip found, then
a refresh of the results page data and a morning summary.

    .venv/Scripts/python.exe -m chopstix.tools.overnight_grip            # usually started by overnight_grip.ps1

Keeps Windows awake while it runs (the screen may still turn off). Log: runs/overnight/grip_log.txt.
Summary: logs/morning/<date>.md.
"""

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
LOG = ROOT / "runs" / "overnight" / "grip_log.txt"
CLOSED = ["closed_v5_mj4", "closed_v5_mj3", "closed_v5_mj2"]
SIZES = ["0.3", "0.45", "0.6", "0.8", "1.0", "1.3", "1.6", "2.0", "2.5"]
LAB_TESTS = ["pushes", "pushes_pick", "jitter", "start", "path", "shake", "mass", "friction", "timestep"]


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake(on: bool):
    if sys.platform == "win32":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (system stays up; the display may sleep)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def run(args: list[str]) -> int:
    log("run: " + " ".join(args))
    with LOG.open("a", encoding="utf-8") as f:
        r = subprocess.run([PY, "-m", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f"exit {r.returncode}")
    return r.returncode


def main():
    keep_awake(True)
    t0 = time.time()
    try:
        saved = []
        for closed in CLOSED:
            run(["chopstix.tools.open_search", "--closed", closed, "--random", "80", "--save", "6", "--sizes", *SIZES])
            f = ROOT / "runs" / "griplab" / f"open_search_{closed}.json"
            if f.exists():
                saved += json.loads(f.read_text())["saved"]
        log(f"open grips saved: {saved}")
        for g in saved:
            run(["chopstix.tools.grip_lab", "--grip", g, "--workers", "14", "--record", "--tests", *LAB_TESTS])
        labs = []
        for g in saved:
            f = ROOT / "runs" / "griplab" / f"{g}.json"
            if f.exists():
                lab = json.loads(f.read_text())
                passed = sum(r["p"] >= 0.9 for rows in lab["ladders"].values() for r in rows)
                total = sum(len(rows) for rows in lab["ladders"].values())
                labs.append((g, lab, passed, total))
        labs.sort(key=lambda x: (-x[2], -json.loads((ROOT / "chopstix" / "grips" / f"{x[0]}.json").read_text())["open_search"]["gap_open_mm"]))
        # put the 3 best open grips on the results page's grip lab
        spec_p = ROOT / "site" / "lineup.json"
        spec = json.loads(spec_p.read_text(encoding="utf-8"))
        spec["grip_labs"] = [g for g in spec["grip_labs"] if not g.get("open")]
        for g, lab, passed, total in labs[:3]:
            info = json.loads((ROOT / "chopstix" / "grips" / f"{g}.json").read_text())["open_search"]
            spec["grip_labs"].append({"file": f"runs/griplab/{g}.json", "label": f"Open: {g.split('_open')[1]} (from {g.split('_open')[0]})",
                                      "desc": f"Tips {info['gap_closed_mm']} -> {info['gap_open_mm']} mm, back to {info['gap_back_mm']} mm when closed again.",
                                      "open": True})  # fmt: skip
        spec_p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        run(["chopstix.tools.report", "--workers", "8", "--no-history"])
        write_summary(labs, time.time() - t0)
    finally:
        keep_awake(False)
        log("done")


def write_summary(labs, seconds: float):
    day = time.strftime("%Y-%m-%d")
    out = ROOT / "logs" / "morning" / f"{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Morning summary {day}", "", f"Overnight grip job (no training), {seconds / 3600:.1f} h. Log: `runs/overnight/grip_log.txt`.", "",
             "## Open-grip search", ""]  # fmt: skip
    for closed in CLOSED:
        f = ROOT / "runs" / "griplab" / f"open_search_{closed}.json"
        if not f.exists():
            lines.append(f"- {closed}: search did not finish")
            continue
        d = json.loads(f.read_text())
        n_ok = sum(t["ok"] for t in d["trials"])
        widest = d["ranked"][0] if d["ranked"] else None
        lines.append(f"- **{closed}**: {n_ok}/{len(d['trials'])} openings held and closed again; "
                     + (f"widest {widest['gap_closed_mm']} -> {widest['gap_open_mm']} mm ({widest['name']}, size {widest['size']}); " if widest else "")
                     + f"saved: {', '.join(d['saved']) or 'none'}")  # fmt: skip
    lines += ["", "## Grip lab of the open grips (20 runs x 10 s per level)", "",
              "| Open grip | Tips closed -> open -> closed again | Levels held >= 90% | Pushes grip/pick | Jitter | Shake | Friction down to |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
    for g, lab, passed, total in labs:
        info = json.loads((ROOT / "chopstix" / "grips" / f"{g}.json").read_text())["open_search"]
        s = lab["score"]
        lines.append(f"| `{g}` | {info['gap_closed_mm']} -> {info['gap_open_mm']} -> {info['gap_back_mm']} mm | {passed}/{total} | "
                     f"{s.get('max_push_N')} / {s.get('max_push_pick_N')} N | {s.get('max_jitter_rad')} rad | {s.get('held_shake')} m/s2 | {s.get('held_friction')} |")  # fmt: skip
    lines += ["", "## Next", "", "- Pick the open grip (widest opening that passes the lab and closes back), pair it with its closed grip,",
              "  and grip-lab the open <-> closed motion itself before the open/close stage.", ""]  # fmt: skip
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
