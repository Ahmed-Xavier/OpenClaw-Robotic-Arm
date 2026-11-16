"""evaluate.py — Evaluate trained PPO policy OR the scripted expert on GraspEnv.

Evaluation Pipeline:
  1. Loads the trained PPO model from RL/models/ OR runs the scripted expert.
  2. Optionally uses the MuJoCo renderer (--headless to suppress).
  3. Reports per-episode and aggregate metrics for grasp env:
       - Success (grasped), min distance, ever-any-contact, both-pad contact,
         number of steps with close command (act[3] > 0).
  4. Supports --baseline-expert to run the scripted controller as comparison.

Usage
-----
  # Evaluate trained PPO model
  python RL/evaluate.py --env grasp --headless --episodes 20

  # Evaluate expert baseline
  python RL/evaluate.py --env grasp --headless --episodes 20 --baseline-expert

  # Evaluate BC warm-start
  python RL/evaluate.py --env grasp --headless --model-path RL/models/bc_grasp_ppo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from RL.envs.reach_env import ReachEnv, SUCCESS_THRESHOLD
from RL.envs.object_reach_env import ObjectReachEnv
from RL.envs.grasp_env import GraspEnv


def evaluate_grasp(
    env: GraspEnv,
    predict_fn,                 # callable(obs) -> action
    episodes: int,
    has_renderer: bool,
    delay_s: float,
    label: str,
) -> dict:
    """Run grasp episodes and return aggregate statistics.

    Args:
        env:          Initialised GraspEnv.
        predict_fn:   Function (obs: ndarray) -> action: ndarray.
        episodes:     Number of episodes to run.
        has_renderer: Whether to call env.render() each step.
        delay_s:      Sleep delay between rendered frames.
        label:        Label string for console output (e.g., 'PPO' or 'Expert').

    Returns:
        Dict with aggregate stats.
    """
    print(f"\n[{label}] Running {episodes} evaluation episodes...")
    successes = 0
    all_min_dists: list[float] = []
    all_final_dists: list[float] = []
    all_close_steps: list[int] = []
    all_ever_contact: list[bool] = []
    all_both_contact: list[bool] = []
    all_ep_lengths: list[int] = []

    for ep in range(1, episodes + 1):
        seed = ep * 100
        obs, info = env.reset(seed=seed)
        step = 0
        min_dist = info["distance"]
        ever_contact = False
        ever_both = False
        n_close_steps = 0

        while True:
            action = predict_fn(obs)
            if action[3] > 0:
                n_close_steps += 1
            obs, _, term, trunc, info = env.step(action)
            step += 1
            if has_renderer:
                env.render()
                if delay_s > 0:
                    time.sleep(delay_s)
            min_dist = min(min_dist, info["distance"])
            if info.get("has_contact", False):
                ever_contact = True
            if info.get("has_both_contact", False):
                ever_both = True
            if term or trunc:
                break

        ok = bool(info.get("is_grasped", False))
        successes += int(ok)
        all_min_dists.append(min_dist)
        all_final_dists.append(info["distance"])
        all_close_steps.append(n_close_steps)
        all_ever_contact.append(ever_contact)
        all_both_contact.append(ever_both)
        all_ep_lengths.append(step)

        status = "SUCCESS" if ok else "FAIL"
        print(f"  [{label}] Ep {ep:3d} | {status:7s} | "
              f"steps={step:3d} | min_dist={min_dist*100:5.2f}cm | "
              f"close_cmds={n_close_steps:3d} | "
              f"any_contact={ever_contact} | both_pad={ever_both}")

    n = episodes
    sr = successes / n * 100
    result = {
        "label": label,
        "episodes": n,
        "successes": successes,
        "success_rate_pct": sr,
        "avg_min_dist_cm": float(np.mean(all_min_dists)) * 100,
        "best_min_dist_cm": float(np.min(all_min_dists)) * 100,
        "avg_close_steps": float(np.mean(all_close_steps)),
        "ever_contact_rate_pct": float(np.mean([int(x) for x in all_ever_contact])) * 100,
        "both_pad_contact_rate_pct": float(np.mean([int(x) for x in all_both_contact])) * 100,
        "avg_ep_length": float(np.mean(all_ep_lengths)),
    }
    return result


def print_summary(result: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  EVALUATION SUMMARY — {result['label']}")
    print("=" * 65)
    print(f"Total Episodes:              {result['episodes']}")
    print(f"Successful Grasps:           {result['successes']} / {result['episodes']} ({result['success_rate_pct']:.1f}%)")
    print(f"Avg Min Distance:            {result['avg_min_dist_cm']:.2f} cm")
    print(f"Best Min Distance:           {result['best_min_dist_cm']:.2f} cm")
    print(f"Avg Close-Command Steps:     {result['avg_close_steps']:.1f} / episode")
    print(f"Ever-Any-Contact Rate:       {result['ever_contact_rate_pct']:.1f}%")
    print(f"Both-Pad Contact Rate:       {result['both_pad_contact_rate_pct']:.1f}%")
    print(f"Avg Episode Length:          {result['avg_ep_length']:.1f} steps")
    print("=" * 65)


def evaluate(
    env_name: str = "reach",
    model_path: Optional[str] = None,
    episodes: int = 20,
    has_renderer: bool = True,
    delay_s: float = 0.02,
    deterministic: bool = True,
    baseline_expert: bool = False,
) -> Optional[dict]:
    """Run visual or headless evaluation of a trained PPO policy (or expert)."""

    # ---------- Non-grasp envs: simple reach evaluation ----------
    if env_name != "grasp":
        if model_path is None:
            model_path = f"RL/models/ur5e_{env_name}_ppo"
        full_model_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
        if not os.path.exists(full_model_path):
            print(f"[Error] Model not found at: {full_model_path}")
            sys.exit(1)

        print("=" * 65)
        print(f"  {'PHASE 1.5' if env_name == 'object_reach' else 'PHASE 1'}: UR5e {env_name} PPO EVALUATION")
        print("=" * 65)
        model = PPO.load(full_model_path)
        if env_name == "object_reach":
            env = ObjectReachEnv(has_renderer=has_renderer, action_dim=3, max_episode_steps=100)
        else:
            env = ReachEnv(has_renderer=has_renderer, action_dim=3, max_episode_steps=100)

        success_count = 0
        final_distances = []
        try:
            for ep in range(1, episodes + 1):
                obs, info = env.reset(seed=ep * 100)
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, _, term, trunc, info = env.step(action)
                    if has_renderer:
                        env.render()
                        if delay_s > 0:
                            time.sleep(delay_s)
                    done = term or trunc
                if info.get("success", False):
                    success_count += 1
                final_distances.append(info["distance"])
                status = "SUCCESS" if info.get("success", False) else "FAIL"
                print(f"  Ep {ep} | {status} | dist={info['distance']*100:.2f}cm")
        finally:
            env.close()
        sr = success_count / episodes * 100
        print(f"\nSuccess Rate: {success_count}/{episodes} ({sr:.1f}%) | "
              f"Avg Final Dist: {np.mean(final_distances)*100:.2f}cm")
        return None

    # ---------- Grasp env ----------
    print("=" * 65)
    print("  PHASE 2: UR5e GraspEnv EVALUATION")
    print("=" * 65)
    print(f"Headless: {not has_renderer} | Episodes: {episodes}")
    print(f"Seed pattern: ep*100 (ep=1..{episodes})")
    print("-" * 65)

    env = GraspEnv(has_renderer=has_renderer, max_episode_steps=150)

    results = []
    try:
        # --- Expert baseline ---
        if baseline_expert:
            from RL.expert import expert_action
            def expert_predict(obs):
                return expert_action(obs)
            r = evaluate_grasp(env, expert_predict, episodes, has_renderer, delay_s, "Expert")
            print_summary(r)
            results.append(r)

        # --- PPO policy ---
        if not baseline_expert or model_path is not None:
            if model_path is None:
                model_path = "RL/models/ur5e_grasp_ppo"
            full_model_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
            if os.path.exists(full_model_path):
                model = PPO.load(full_model_path)
                def ppo_predict(obs):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    return action
                r = evaluate_grasp(env, ppo_predict, episodes, has_renderer, delay_s, f"PPO ({Path(full_model_path).stem})")
                print_summary(r)
                results.append(r)
            else:
                print(f"[WARN] Model not found at {full_model_path}. Skipping PPO evaluation.")
    finally:
        env.close()

    return results[0] if len(results) == 1 else results if results else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PPO or expert on UR5e ReachEnv, ObjectReachEnv, or GraspEnv."
    )
    parser.add_argument("--env",             type=str,   default="grasp",
                        choices=["reach", "object_reach", "grasp"])
    parser.add_argument("--model-path",      type=str,   default=None,
                        help="Path to trained PPO model zip")
    parser.add_argument("--episodes",        type=int,   default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--headless",        action="store_true",
                        help="Run without MuJoCo renderer")
    parser.add_argument("--delay",           type=float, default=0.02,
                        help="Sleep delay between rendered frames (seconds)")
    parser.add_argument("--stochastic",      action="store_true",
                        help="Sample actions stochastically instead of deterministically")
    parser.add_argument("--baseline-expert", action="store_true",
                        help="Also run the scripted expert as a baseline")

    args = parser.parse_args()

    evaluate(
        env_name=args.env,
        model_path=args.model_path,
        episodes=args.episodes,
        has_renderer=not args.headless,
        delay_s=args.delay,
        deterministic=not args.stochastic,
        baseline_expert=args.baseline_expert,
    )
