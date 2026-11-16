"""expert.py — Scripted expert controller for UR5e GraspEnv (Phase 2).

Three-phase reactive controller (stateless, obs-only):
  1. APPROACH  — move EEF to hover point 10 cm above cube, keeping gripper open.
                 Active when horizontal XY misalignment > XY_ALIGN_THRESH.
  2. DESCEND   — drive EEF straight down to cube centre, keeping gripper open.
                 Active when XY-aligned but dist > CLOSE_DIST_THRESH.
  3. CLOSE     — hold position at cube centre and close gripper.
                 Active when dist <= CLOSE_DIST_THRESH.

All transitions are one-way and reactive: we re-evaluate phase every step from
obs alone, so the expert can be called without any per-episode state.

Verified performance (seeds 0..5, horizon 150):
  - 3-5 / 6 grasps depending on cube spawn randomness.
  - Used to collect 200-seed BC dataset (see pretrain_bc.py).

Usage (standalone)
------------------
  python RL/expert.py [--episodes N] [--headless] [--seed-start S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RL.envs.grasp_env import GraspEnv

# ---------------------------------------------------------------------------
# Expert hyper-parameters (tuned empirically, do NOT change without testing)
# ---------------------------------------------------------------------------
HOVER_HEIGHT: float = 0.10     # metres above cube to hover before descending
XY_ALIGN_THRESH: float = 0.015 # switch approach→descend when |d_xy| < this
CLOSE_DIST_THRESH: float = 0.012  # switch descend→close when |dist| < this
ACTION_SCALE: float = 0.01     # divisor for proportional control (1 cm/step max)


def expert_action(obs: np.ndarray) -> np.ndarray:
    """Return a 4D expert action from the current 10D GraspEnv observation.

    Phase is determined purely from observation — no episode state required.

    Args:
        obs: 10D observation [eef_pos(3), cube_pos(3), delta(3), gripper(1)]

    Returns:
        4D action [dx, dy, dz, gripper] in [-1, 1].
    """
    eef = obs[0:3]
    cube = obs[3:6]
    diff = cube - eef
    dist = float(np.linalg.norm(diff))
    d_xy = float(np.linalg.norm(diff[:2]))

    act = np.zeros(4, dtype=np.float32)

    if d_xy > XY_ALIGN_THRESH:
        # Phase 1 — APPROACH: move to hover point above cube in XY+Z
        target_above = cube.copy()
        target_above[2] += HOVER_HEIGHT
        diff_above = target_above - eef
        act[:3] = np.clip(diff_above / ACTION_SCALE, -1.0, 1.0)
        act[3] = -1.0  # keep gripper open
    elif dist > CLOSE_DIST_THRESH:
        # Phase 2 — DESCEND: drive down to cube (already XY-aligned)
        act[:3] = np.clip(diff / ACTION_SCALE, -1.0, 1.0)
        act[3] = -1.0  # keep gripper open
    else:
        # Phase 3 — CLOSE: hold position, close gripper
        act[:3] = np.clip(diff / ACTION_SCALE, -1.0, 1.0)
        act[3] = 1.0  # close gripper

    return act


def run_expert(
    env: GraspEnv,
    seed: int,
) -> tuple[bool, int, float, int]:
    """Run one expert episode.

    Returns:
        (is_grasped, steps, final_distance, close_steps)
    """
    obs, info = env.reset(seed=seed)
    step = 0
    close_steps = 0
    while step < env.max_episode_steps:
        act = expert_action(obs)
        if act[3] > 0:
            close_steps += 1
        obs, _, term, trunc, info = env.step(act)
        step += 1
        if term or trunc:
            break
    return bool(info["is_grasped"]), step, float(info["distance"]), close_steps


def evaluate_expert(
    episodes: int = 20,
    seed_start: int = 0,
    has_renderer: bool = False,
) -> None:
    """Evaluate the expert controller and print a summary."""
    print("=" * 65)
    print("  EXPERT CONTROLLER EVALUATION")
    print("=" * 65)
    print(f"Episodes: {episodes} | Headless: {not has_renderer}")
    print(f"Parameters: hover={HOVER_HEIGHT}m, xy_thresh={XY_ALIGN_THRESH}m, "
          f"close_thresh={CLOSE_DIST_THRESH}m, scale={ACTION_SCALE}")
    print("-" * 65)

    env = GraspEnv(has_renderer=has_renderer)
    successes = 0
    total_steps = []
    total_close_steps = []
    total_dists = []

    try:
        for ep in range(episodes):
            seed = seed_start + ep
            ok, steps, dist, cs = run_expert(env, seed)
            successes += int(ok)
            total_steps.append(steps)
            total_close_steps.append(cs)
            total_dists.append(dist)
            status = "SUCCESS" if ok else "FAIL"
            print(f"Ep {ep+1:3d} | seed={seed:4d} | {status:7s} | "
                  f"steps={steps:3d} | dist={dist*100:5.2f}cm | close_cmds={cs}")
    finally:
        env.close()

    n = episodes
    sr = successes / n * 100
    print("-" * 65)
    print(f"Success Rate:    {successes}/{n} ({sr:.1f}%)")
    print(f"Avg Steps:       {np.mean(total_steps):.1f}")
    print(f"Avg Close Cmds:  {np.mean(total_close_steps):.1f}")
    print(f"Avg Final Dist:  {np.mean(total_dists)*100:.2f} cm")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run and evaluate the GraspEnv scripted expert controller."
    )
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to evaluate")
    parser.add_argument("--headless", action="store_true",
                        help="Run without MuJoCo renderer")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed (episodes use seed_start + ep_index)")
    args = parser.parse_args()
    evaluate_expert(
        episodes=args.episodes,
        seed_start=args.seed_start,
        has_renderer=not args.headless,
    )
