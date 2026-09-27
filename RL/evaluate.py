"""evaluate.py — Evaluate trained PPO policy on UR5e ReachEnv, ObjectReachEnv, or GraspEnv with visualization.

Evaluation Pipeline:
1. Loads the trained PPO model from RL/models/ (ur5e_reach_ppo, ur5e_object_reach_ppo, or ur5e_grasp_ppo).
2. Instantiates environment with has_renderer=True (MuJoCo window).
3. Executes multiple test episodes with deterministic policy actions.
4. Renders each simulation step.
5. Displays clear per-episode and aggregate metrics:
   - Target / Cube position
   - Final EEF position
   - Final distance
   - Grasp / Contact status (for grasp env)
   - Success / Failure status
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO

from RL.envs.reach_env import ReachEnv, SUCCESS_THRESHOLD
from RL.envs.object_reach_env import ObjectReachEnv
from RL.envs.grasp_env import GraspEnv


def evaluate(
    env_name: str = "reach",
    model_path: Optional[str] = None,
    episodes: int = 5,
    has_renderer: bool = True,
    delay_s: float = 0.02,
    deterministic: bool = True,
):
    """Run visual evaluation of trained PPO policy."""
    if model_path is None:
        model_path = f"RL/models/ur5e_{env_name}_ppo"

    full_model_path = model_path if model_path.endswith(".zip") else f"{model_path}.zip"
    if not os.path.exists(full_model_path):
        print(f"[Error] Model weights not found at: {full_model_path}")
        print(f"Please train a model first using: python RL/train.py --env {env_name}")
        sys.exit(1)

    if env_name == "grasp":
        phase_label = "PHASE 2: UR5e GraspEnv"
    elif env_name == "object_reach":
        phase_label = "PHASE 1.5: UR5e ObjectReachEnv"
    else:
        phase_label = "PHASE 1: UR5e ReachEnv"
    print("=" * 75)
    print(f"  {phase_label} PPO POLICY EVALUATION")
    print("=" * 75)
    print(f"Environment:       {env_name}")
    print(f"Loading model:     {full_model_path}")
    print(f"Evaluation Mode:   {'Visual (Renderer ON)' if has_renderer else 'Headless'}")
    print(f"Episodes:          {episodes}")
    if env_name == "grasp":
        print("Success Criterion: robosuite _check_grasp (both fingerpads contact cube)")
    else:
        print(f"Success Threshold: {SUCCESS_THRESHOLD * 100:.1f} cm")
    print("-" * 75)

    # 1. Load trained policy
    model = PPO.load(full_model_path)

    # 2. Instantiate evaluation environment
    if env_name == "grasp":
        env = GraspEnv(
            has_renderer=has_renderer,
            max_episode_steps=150,
        )
    elif env_name == "object_reach":
        env = ObjectReachEnv(
            has_renderer=has_renderer,
            action_dim=3,
            max_episode_steps=100,
        )
    else:
        env = ReachEnv(
            has_renderer=has_renderer,
            action_dim=3,
            max_episode_steps=100,
        )

    success_count = 0
    final_distances = []
    initial_distances = []
    episode_lengths = []

    target_label = "Cube Position" if env_name in ("object_reach", "grasp") else "Target Position"

    try:
        for ep in range(1, episodes + 1):
            obs, info = env.reset(seed=ep * 100)
            target = np.array(info.get("cube_position", info.get("target")), dtype=np.float32)
            initial_eef = np.array(info["eef_position"], dtype=np.float32)
            initial_dist = info["distance"]
            initial_distances.append(initial_dist)

            print(f"\n[Episode {ep}/{episodes}]")
            print(f"  {target_label}:    [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}] m")
            print(f"  Initial EEF Pos:   [{initial_eef[0]:+.4f}, {initial_eef[1]:+.4f}, {initial_eef[2]:+.4f}] m")
            print(f"  Initial Distance:  {initial_dist * 100:.2f} cm")

            step_count = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Predict action
                action, _ = model.predict(obs, deterministic=deterministic)

                # Step simulation
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1

                # Render frame
                if has_renderer:
                    env.render()
                    if delay_s > 0:
                        time.sleep(delay_s)

            final_eef = np.array(info["eef_position"], dtype=np.float32)
            final_dist = info["distance"]
            success = info.get("success", False)

            final_distances.append(final_dist)
            episode_lengths.append(step_count)
            if success:
                success_count += 1
                status = ">>> SUCCESS (GRASPED) <<<" if env_name == "grasp" else ">>> SUCCESS <<<"
            else:
                status = "FAILED (Not Grasped)" if env_name == "grasp" else "FAILED (Horizon Reached)"

            print(f"  Steps Elapsed:     {step_count}")
            print(f"  Final EEF Pos:     [{final_eef[0]:+.4f}, {final_eef[1]:+.4f}, {final_eef[2]:+.4f}] m")
            print(f"  Final Distance:    {final_dist * 100:.2f} cm")
            if env_name == "grasp":
                print(f"  Contact Detected:  {info.get('has_contact', False)}")
                print(f"  Grasp Detected:    {info.get('is_grasped', False)}")
            print(f"  Outcome:           {status}")

    finally:
        env.close()

    # Summary
    success_rate = (success_count / episodes) * 100.0
    avg_final_dist = float(np.mean(final_distances)) * 100.0
    min_final_dist = float(np.min(final_distances)) * 100.0
    avg_ep_len = float(np.mean(episode_lengths))
    min_init_dist = float(np.min(initial_distances)) * 100.0
    max_init_dist = float(np.max(initial_distances)) * 100.0

    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Episodes:        {episodes}")
    print(f"Successful Episodes:   {success_count} / {episodes} ({success_rate:.1f}%)")
    print(f"Average Final Dist:    {avg_final_dist:.2f} cm")
    print(f"Best Final Dist:       {min_final_dist:.2f} cm")
    print(f"Average Episode Length:{avg_ep_len:.1f} steps")
    print(f"Initial Distance Range:[{min_init_dist:.2f} cm, {max_init_dist:.2f} cm]")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on UR5e ReachEnv, ObjectReachEnv, or GraspEnv.")
    parser.add_argument("--env", type=str, default="reach", choices=["reach", "object_reach", "grasp"], help="Environment to evaluate ('reach', 'object_reach', or 'grasp')")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained PPO model (defaults based on --env)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo render window")
    parser.add_argument("--delay", type=float, default=0.02, help="Sleep delay between frames in seconds")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of deterministically")

    args = parser.parse_args()

    evaluate(
        env_name=args.env,
        model_path=args.model_path,
        episodes=args.episodes,
        has_renderer=not args.headless,
        delay_s=args.delay,
        deterministic=not args.stochastic,
    )

