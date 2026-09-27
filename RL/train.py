"""train.py — Train PPO policy on UR5e ReachEnv using Stable-Baselines3.

Phase 1 Training Pipeline:
1. Instantiates ReachEnv (Gymnasium wrapper around robosuite UR5e Lift).
2. Wraps with Stable-Baselines3 Monitor for telemetry.
3. Configures PPO (MlpPolicy, n_steps=1024, batch_size=64, lr=3e-4).
4. Trains for specified timesteps (default 50,000).
5. Saves trained weights to RL/models/ur5e_reach_ppo.zip.
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

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from RL.envs.reach_env import ReachEnv, sanity_check as reach_sanity_check
from RL.envs.object_reach_env import ObjectReachEnv, sanity_check as object_reach_sanity_check



class TrainingProgressCallback(BaseCallback):
    """Custom callback to report episodic success rates and distances periodically."""

    def __init__(self, check_freq: int = 2048, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_count = 0
        self.success_count = 0
        self.recent_distances = []

    def _on_step(self) -> bool:
        # Check infos for episode completions
        for info in self.locals.get("infos", []):
            if "distance" in info:
                self.recent_distances.append(info["distance"])
            if info.get("success", False):
                self.success_count += 1

        if self.n_calls % self.check_freq == 0 and len(self.recent_distances) > 0:
            avg_dist = float(sum(self.recent_distances[-50:]) / max(1, len(self.recent_distances[-50:])))
            min_dist = float(min(self.recent_distances[-50:]))
            print(
                f"[PPO Step {self.n_calls:6d}] "
                f"Recent Avg Distance: {avg_dist:.4f} m | "
                f"Min Distance: {min_dist:.4f} m | "
                f"Total Successes: {self.success_count}"
            )
        return True


def train(
    env_name: str = "reach",
    timesteps: int = 50_000,
    save_path: Optional[str] = None,
    n_steps: int = 1024,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "auto",
):
    """Train PPO policy on UR5e ReachEnv or ObjectReachEnv."""
    if save_path is None:
        save_path = f"RL/models/ur5e_{env_name}_ppo"

    phase_label = "PHASE 1.5: UR5e ObjectReachEnv" if env_name == "object_reach" else "PHASE 1: UR5e ReachEnv"
    print("=" * 70)
    print(f"  {phase_label} PPO TRAINING PIPELINE")
    print("=" * 70)
    print(f"Environment:       {env_name}")
    print(f"Device:            {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"Total Timesteps:   {timesteps}")
    print(f"Save Path:         {save_path}")
    print(f"PPO Hyperparams:   n_steps={n_steps}, batch_size={batch_size}, lr={learning_rate}")
    print(f"Random Seed:       {seed}")
    print("-" * 70)

    # 1. Run environment sanity check
    print("[1/4] Running environment sanity check...")
    if env_name == "object_reach":
        object_reach_sanity_check()
    else:
        reach_sanity_check()

    # 2. Instantiate and wrap training environment
    print("[2/4] Instantiating training environment...")
    if env_name == "object_reach":
        raw_env = ObjectReachEnv(has_renderer=False, action_dim=3)
    else:
        raw_env = ReachEnv(has_renderer=False, action_dim=3)
    env = Monitor(raw_env)

    # 3. Create Stable-Baselines3 PPO agent
    print("[3/4] Initializing PPO agent...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=seed,
        device=device,
    )

    # Ensure output directory exists
    save_dir = Path(save_path).parent
    os.makedirs(save_dir, exist_ok=True)

    # 4. Train
    print("[4/4] Starting training loop...")
    callback = TrainingProgressCallback(check_freq=n_steps)
    start_time = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=False,
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes).")

    # 5. Save model
    model.save(save_path)
    print(f"Model saved successfully to: {save_path}.zip")

    # Clean up
    env.close()
    print("Environment closed.")
    print("=" * 70)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on UR5e ReachEnv or ObjectReachEnv.")
    parser.add_argument("--env", type=str, default="reach", choices=["reach", "object_reach"], help="Environment to train ('reach' or 'object_reach')")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total timesteps to train")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save trained weights (defaults based on --env)")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device ('auto', 'cpu', 'cuda')")

    args = parser.parse_args()

    train(
        env_name=args.env,
        timesteps=args.timesteps,
        save_path=args.save_path,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )

