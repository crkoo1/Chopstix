"""Render a grip (as loaded, or after some seconds of the hold test) from a few angles into one PNG.

    .venv/Scripts/python.exe -m chopstix.tools.render --grip closed_v5 --out runs/renders/closed_v5.png
    .venv/Scripts/python.exe -m chopstix.tools.render --grip closed_v5_mj --after 3 --out runs/renders/after3s.png
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from chopstix.sim import STICKS, Sim, SimCfg, load_grip
from chopstix.tools.common import parse_overrides
from chopstix.tools.hold_test import load_into

VIEWS = [("front", 180, -15), ("side", 90, -15), ("top", 180, -80), ("thumb", 270, -20)]


def render_views(sim: Sim, size=(480, 480), collision=False, label=None) -> np.ndarray:
    from PIL import Image, ImageDraw

    renderer = mujoco.Renderer(sim.model, size[1], size[0])
    opt = mujoco.MjvOption()
    if collision:
        opt.geomgroup[:] = 0
        opt.geomgroup[3] = 1
        opt.geomgroup[0] = 0
    opt.frame = mujoco.mjtFrame.mjFRAME_NONE
    centre = np.mean([sim.stick_pose(n)[0] for n in STICKS] + [sim.palm_pose()[0]], axis=0)
    tiles = []
    for name, az, el in VIEWS:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = centre
        cam.distance, cam.azimuth, cam.elevation = 0.38, az, el
        renderer.update_scene(sim.data, cam, opt)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((8, 8), name + (f"  {label}" if label else ""), fill=(255, 255, 255))
        tiles.append(np.asarray(img))
    renderer.close()
    return np.concatenate(tiles, axis=1)


def main(argv=None):
    from PIL import Image

    p = argparse.ArgumentParser()
    p.add_argument("--grip", default="closed_v5")
    p.add_argument("--after", type=float, default=0.0, help="s of gravity + pushes before rendering (0 = as loaded)")
    p.add_argument("--collision", action="store_true", help="draw the collision geoms instead of the visual meshes")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    sim = Sim(SimCfg.from_overrides(parse_overrides(args.set)))
    grip = load_grip(args.grip)
    load_into(sim, grip)
    rows = [render_views(sim, collision=args.collision, label=f"{grip['name']} t=0")]
    if args.after > 0:
        rng = np.random.default_rng(0)
        dt = sim.model.opt.timestep
        sim.set_gravity(False)
        sim.step(round(0.2 / dt))
        sim.set_gravity(True)
        for _ in range(round(args.after / 0.1)):
            sim.step(round(0.1 / dt), {n: rng.uniform(-0.03, 0.03, 3) for n in STICKS})
        rows.append(render_views(sim, collision=args.collision, label=f"after {args.after:g} s gravity+pushes"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
