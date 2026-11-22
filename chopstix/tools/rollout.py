"""Play one episode (zero action = the scripted grip, or a trained actor) and save it as an animated GIF.

    .venv/Scripts/python.exe -m chopstix.tools.rollout --env stage=hold --out runs/gifs/hold.gif
    .venv/Scripts/python.exe -m chopstix.tools.rollout --env stage=pick pick_place=true --actor runs/x/actor_best.pt --out x.gif
"""

import argparse
from pathlib import Path

import numpy as np

from chopstix.envs.chopsticks_env import ACTION_DIM, PROPRIO_DIM, TASK_DIM, make_env
from chopstix.tools.common import parse_overrides


def main(argv=None):
    from PIL import Image, ImageDraw

    p = argparse.ArgumentParser()
    p.add_argument("--env", nargs="*", default=["stage=hold"], metavar="KEY=VALUE")
    p.add_argument("--actor", type=Path, help="chops-actor-v1 checkpoint (default: zero action)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--every", type=int, default=2, help="keep every Nth control step")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    env = make_env(parse_overrides(args.env))
    model = None
    if args.actor:
        import torch

        from chopstix.rl.policy import ActorCritic

        model = ActorCritic(PROPRIO_DIM, TASK_DIM, ACTION_DIM)
        model.load_pretrained(str(args.actor))
        model.eval()
    obs, info = env.reset(seed=args.seed)
    frames = []
    for t in range(env.cfg.steps_per_episode):
        if model is None:
            action = np.zeros(ACTION_DIM)
        else:
            with torch.no_grad():
                action = model.act(torch.as_tensor(obs[None], dtype=torch.float32), deterministic=True)[0][0].numpy()
        obs, r, term, trunc, info = env.step(action)
        if t % args.every == 0 or term or trunc:
            img = Image.fromarray(env.render()).resize((400, 300))
            text = f"t={(t + 1) / env.cfg.control_frequency:4.1f}s"
            if env.cfg.stage == "pick":
                text += f"  {env.PICK_PHASES[int(info['phase'])]}  lift {info['object_lift'] * 1000:3.0f} mm"
            else:
                text += f"  drift {max(info['upper_drift'], info['lower_drift']) * 1000:4.1f} mm"
            ImageDraw.Draw(img).text((6, 6), text, fill=(255, 255, 255))
            frames.append(img)
        if term or trunc:
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 * args.every / env.cfg.control_frequency), loop=0)
    print(f"wrote {args.out}: {len(frames)} frames, success {info['is_success']}, {'terminated' if term else 'full length'}")


if __name__ == "__main__":
    main()
