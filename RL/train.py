"""train.py — Train PPO policy on UR5e ReachEnv, ObjectReachEnv, or GraspEnv using Stable-Baselines3.

RL Training Pipeline:
- Phase 1:   ReachEnv (reach sampled 3D Cartesian points)        -> RL/models/ur5e_reach_ppo.zip
- Phase 1.5: ObjectReachEnv (reach actual physical cube)         -> RL/models/ur5e_object_reach_ppo.zip
- Phase 2:   GraspEnv (grasp physical cube with gripper)         -> RL/models/ur5e_grasp_ppo.zip

New flags for Phase 2:
  --init-from PATH   Load an existing PPO .zip as starting weights (e.g. BC warm-start)
  --n-envs N         Number of parallel SubprocVecEnv workers (default 1, set >1 for speed)
  --timesteps N      Total training timesteps (default 50_000; max 20_000 per smoke test)
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

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from RL.envs.reach_env import ReachEnv, sanity_check as reach_sanity_check
from RL.envs.object_reach_env import ObjectReachEnv, sanity_check as object_reach_sanity_check
from RL.envs.grasp_env import GraspEnv, sanity_check as grasp_sanity_check


class GraspProgressCallback(BaseCallback):
    """Custom callback reporting episodic success, contact, and close-command stats."""

    def __init__(self, check_freq: int = 2048, verbose: int = 1, is_grasp: bool = False):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.is_grasp = is_grasp
        self.episode_count = 0
        self.success_count = 0
        self.recent_distances: list[float] = []
        # Grasp-specific accumulators
        self.recent_close_steps: list[int] = []
        self.recent_contact: list[bool] = []
        self.recent_both_contact: list[bool] = []
        self._ep_close = 0
        self._ep_contact = False
        self._ep_both = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Track close commands
            if self.is_grasp:
                actions = self.locals.get("actions", None)
                if actions is not None:
                    # actions shape: (n_envs, action_dim)
                    if hasattr(actions, "__len__") and len(actions.shape) > 1:
                        for a in actions:
                            if float(a[3]) > 0:
                                self._ep_close += 1
                    else:
                        if float(actions[3]) > 0:
                            self._ep_close += 1
                if info.get("has_contact", False):
                    self._ep_contact = True
                if info.get("has_both_contact", False):
                    self._ep_both = True

            # On episode end
            if "episode" in info:
                self.episode_count += 1
                if self.is_grasp:
                    self.recent_close_steps.append(self._ep_close)
                    self.recent_contact.append(self._ep_contact)
                    self.recent_both_contact.append(self._ep_both)
                    self._ep_close = 0
                    self._ep_contact = False
                    self._ep_both = False
            if info.get("success", False) or info.get("is_grasped", False):
                self.success_count += 1
            if "distance" in info:
                self.recent_distances.append(info["distance"])

        if self.n_calls % self.check_freq == 0 and len(self.recent_distances) > 0:
            w = min(100, len(self.recent_distances))
            avg_dist = float(sum(self.recent_distances[-w:]) / w)
            min_dist = float(min(self.recent_distances[-w:]))

            if self.is_grasp and len(self.recent_close_steps) > 0:
                wc = min(50, len(self.recent_close_steps))
                avg_close = float(sum(self.recent_close_steps[-wc:]) / wc)
                contact_rate = float(sum(int(x) for x in self.recent_contact[-wc:])) / wc
                both_rate = float(sum(int(x) for x in self.recent_both_contact[-wc:])) / wc
                print(
                    f"[PPO Step {self.n_calls:6d}] "
                    f"Avg Dist: {avg_dist:.4f}m | Min Dist: {min_dist:.4f}m | "
                    f"Successes: {self.success_count} | "
                    f"AvgCloseSteps: {avg_close:.1f} | "
                    f"ContactRate: {contact_rate*100:.1f}% | "
                    f"BothPadRate: {both_rate*100:.1f}%"
                )
            else:
                print(
                    f"[PPO Step {self.n_calls:6d}] "
                    f"Avg Dist: {avg_dist:.4f}m | Min Dist: {min_dist:.4f}m | "
                    f"Total Successes: {self.success_count}"
                )
        return True


def make_env_fn(env_name: str, seed: int, rank: int, action_dim: int):
    """Factory for SubprocVecEnv worker environments."""
    def _init():
        if env_name == "grasp":
            env = GraspEnv(has_renderer=False)
        elif env_name == "object_reach":
            env = ObjectReachEnv(has_renderer=False, action_dim=action_dim)
        else:
            env = ReachEnv(has_renderer=False, action_dim=action_dim)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    env_name: str = "reach",
    timesteps: int = 50_000,
    save_path: Optional[str] = None,
    n_steps: int = 2048,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 42,
    device: str = "auto",
    n_envs: int = 1,
    init_from: Optional[str] = None,
    action_dim: int = 3,
):
    """Train PPO policy on UR5e ReachEnv, ObjectReachEnv, or GraspEnv."""
    if save_path is None:
        save_path = f"RL/models/ur5e_{env_name}_ppo"

    is_grasp = (env_name == "grasp")
    if env_name == "grasp":
        phase_label = "PHASE 2: UR5e GraspEnv"
    elif env_name == "object_reach":
        phase_label = "PHASE 1.5: UR5e ObjectReachEnv"
    else:
        phase_label = "PHASE 1: UR5e ReachEnv"

    print("=" * 70)
    print(f"  {phase_label} PPO TRAINING PIPELINE")
    print("=" * 70)
    print(f"Environment:       {env_name}")
    print(f"Device:            {device} (CUDA available: {torch.cuda.is_available()})")
    print(f"Total Timesteps:   {timesteps}")
    print(f"Parallel Envs:     {n_envs}")
    print(f"Init From:         {init_from or 'scratch'}")
    print(f"Save Path:         {save_path}")
    print(f"PPO Hyperparams:   n_steps={n_steps}, batch_size={batch_size}, lr={learning_rate}")
    print(f"Random Seed:       {seed}")
    print("-" * 70)

    # 1. Sanity check
    print("[1/4] Running environment sanity check...")
    if env_name == "grasp":
        grasp_sanity_check()
    elif env_name == "object_reach":
        object_reach_sanity_check()
    else:
        reach_sanity_check()

    # 2. Build vectorised environment
    print(f"[2/4] Instantiating {n_envs} training environment(s)...")
    if n_envs == 1:
        if env_name == "grasp":
            raw_env = GraspEnv(has_renderer=False)
        elif env_name == "object_reach":
            raw_env = ObjectReachEnv(has_renderer=False, action_dim=action_dim)
        else:
            raw_env = ReachEnv(has_renderer=False, action_dim=action_dim)
        env = Monitor(raw_env)
    else:
        fns = [make_env_fn(env_name, seed, rank, action_dim) for rank in range(n_envs)]
        env = SubprocVecEnv(fns)

    # 3. Create or load PPO
    print("[3/4] Initializing PPO agent...")
    if init_from is not None:
        init_path = init_from if init_from.endswith(".zip") else f"{init_from}.zip"
        if not os.path.exists(init_path):
            print(f"[WARN] --init-from path not found: {init_path}. Starting from scratch.")
            init_from = None

    if init_from is not None:
        print(f"       Loading warm-start from: {init_path}")
        model = PPO.load(
            init_path,
            env=env,
            # Override key hyperparams that may differ from the saved file
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
    else:
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
    callback = GraspProgressCallback(check_freq=n_steps, is_grasp=is_grasp)
    start_time = time.time()

    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        progress_bar=False,
        reset_num_timesteps=(init_from is None),
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.2f}s ({elapsed/60:.2f} min).")
    print(f"Total successes during training: {callback.success_count}")

    # 5. Save
    model.save(save_path)
    print(f"Model saved to: {save_path}.zip")

    env.close()
    print("=" * 70)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on UR5e ReachEnv, ObjectReachEnv, or GraspEnv."
    )
    parser.add_argument("--env",        type=str,   default="reach",
                        choices=["reach", "object_reach", "grasp"],
                        help="Environment to train")
    parser.add_argument("--timesteps",  type=int,   default=50_000,
                        help="Total training timesteps")
    parser.add_argument("--save-path",  type=str,   default=None,
                        help="Path to save trained weights")
    parser.add_argument("--n-steps",    type=int,   default=2048,
                        help="PPO rollout buffer size per env")
    parser.add_argument("--batch-size", type=int,   default=256,
                        help="PPO mini-batch size")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--device",     type=str,   default="auto",
                        help="PyTorch device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--n-envs",     type=int,   default=1,
                        help="Number of parallel SubprocVecEnv workers (>1 for speed)")
    parser.add_argument("--init-from",  type=str,   default=None,
                        help="Path to a PPO .zip to warm-start from (e.g. BC policy)")

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
        n_envs=args.n_envs,
        init_from=args.init_from,
    )
